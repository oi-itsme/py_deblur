"""
多数据集自动调参脚本
====================

这个脚本的目标是把原来“每个数据集单独写一份调参脚本”的做法，整理成统一框架。
适用于你后面有十多个数据集的场景。

核心能力：
1. 自动扫描 datasets/ 下的候选数据集；
2. 自动寻找 event.mat 和 frame/；
3. 自动从帧文件名解析时间戳；
4. 自动生成 tau 候选（按相邻帧间隔比例）；
5. 自动运行代表性参数组合；
6. 为每个数据集输出 JSON、best 文本、对比图；
7. 最后汇总一个总表。
"""

import os
import re
import json
import glob
import logging
import math
import time
from itertools import product

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

from joint_reconstruction_cuda import joint_reconstruction_cuda

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'datasets_root': 'datasets',
    'output_root': 'results',
    'dataset_patterns': ['*', '*/*'],
    'frame_subdir_candidates': ['frame', 'frames'],
    'event_file_candidates': ['event.mat'],
    'default_search': {
        'tau_ratios': [0.5, 1.0, 1.5],
        'k_sizes': [13, 17, 21],
        'alphas': [0.12, 0.16, 0.24],
        'betas': [0.016, 0.032, 0.064],
        'omegas': [0.01, 0.02, 0.03],
        'outer_iters': [8, 10],
        'sigma': 2e-3,
        'contrast_threshold': 1.0,
    },
    'frame_selection': {
        'mode': 'explicit_or_auto',
        'explicit_frame_indices': [10],
        'auto_pick_count': 3,
    }
}


def load_config(path='dataset_config_example.json'):
    if os.path.exists(path):
        logger.info("加载配置文件: %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        for key in cfg:
            if key in user_cfg and key not in ('default_search', 'frame_selection'):
                cfg[key] = user_cfg[key]
        if 'default_search' in user_cfg:
            merged = DEFAULT_CONFIG['default_search'].copy()
            merged.update(user_cfg['default_search'])
            cfg['default_search'] = merged
        if 'frame_selection' in user_cfg:
            merged = DEFAULT_CONFIG['frame_selection'].copy()
            merged.update(user_cfg['frame_selection'])
            cfg['frame_selection'] = merged
        return cfg
    logger.info("未找到配置文件 %s，使用默认配置", path)
    return DEFAULT_CONFIG


def load_events(mat_path):
    mat = sio.loadmat(mat_path)
    events = np.asarray(mat['event']).astype(np.float32)
    # 自动检测事件格式: [x,y,t,p] 还是 [t,x,y,p]
    # 如果第 0 列的值范围远超图像尺寸（>10000），就是 [t,x,y,p] 格式
    if events[:, 0].max() > 10000:
        logger.info("    检测到事件格式 [t,x,y,p]，自动转换为 [x,y,t,p]")
        events = events[:, [1, 2, 0, 3]]
    events[:, 3] = np.where(events[:, 3] > 0, 1.0, -1.0)
    return events


def load_frame(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def parse_frame_info(path):
    name = os.path.basename(path)
    m = re.match(r'^(\d+)_(-?\d+)\.png$', name)
    if not m:
        raise ValueError(f'无法从文件名解析帧索引和时间戳: {name}')
    return int(m.group(1)), float(m.group(2))


def grad_metrics(img):
    gx = np.diff(img, axis=1, append=img[:, -1:])
    gy = np.diff(img, axis=0, append=img[-1:, :])
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    lap = (
        -4 * img
        + np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)
    )
    return {
        'mean_grad': float(np.mean(grad_mag)),
        'lap_var': float(np.var(lap)),
        'var': float(np.var(img)),
        'range': float(img.max() - img.min()),
    }


def score_metrics(m):
    return 0.3 * m['mean_grad'] + 0.4 * m['lap_var'] + 0.2 * m['var'] + 0.1 * m['range']


def discover_datasets(cfg):
    root = cfg['datasets_root']
    logger.info("扫描数据集目录: %s", os.path.abspath(root))
    candidates = []
    for pattern in cfg['dataset_patterns']:
        candidates.extend(glob.glob(os.path.join(root, pattern)))

    dataset_roots = []
    seen = set()
    for candidate in sorted(candidates):
        if not os.path.isdir(candidate):
            continue
        event_path = None
        frame_dir = None
        for event_name in cfg['event_file_candidates']:
            p = os.path.join(candidate, event_name)
            if os.path.exists(p):
                event_path = p
                break
        for subdir in cfg['frame_subdir_candidates']:
            p = os.path.join(candidate, subdir)
            if os.path.isdir(p):
                frame_dir = p
                break
        if event_path and frame_dir:
            norm = os.path.normpath(candidate)
            if norm not in seen:
                seen.add(norm)
                dataset_roots.append({'dataset_root': candidate, 'event_path': event_path, 'frame_dir': frame_dir})
    logger.info("发现 %d 个数据集", len(dataset_roots))
    for ds in dataset_roots:
        logger.info("  %s", os.path.basename(ds['dataset_root']))
    return dataset_roots


def choose_frames(frame_paths, cfg):
    parsed = []
    for p in frame_paths:
        try:
            idx, ts = parse_frame_info(p)
            parsed.append((idx, ts, p))
        except Exception:
            continue
    parsed.sort(key=lambda x: x[0])
    if not parsed:
        return []

    explicit = cfg['frame_selection'].get('explicit_frame_indices', [])
    selected = []
    index_map = {idx: (idx, ts, p) for idx, ts, p in parsed}
    for idx in explicit:
        if idx in index_map:
            selected.append(index_map[idx])

    if selected:
        return selected

    # 如果没有命中显式索引，就自动取头/中/尾附近的若干帧。
    auto_pick_count = cfg['frame_selection'].get('auto_pick_count', 3)
    positions = np.linspace(0, len(parsed) - 1, num=min(auto_pick_count, len(parsed)), dtype=int)
    uniq = sorted(set(positions.tolist()))
    return [parsed[i] for i in uniq]


def get_prev_timestamp(parsed_frames, current_index):
    current_pos = None
    for i, item in enumerate(parsed_frames):
        if item[0] == current_index:
            current_pos = i
            break
    if current_pos is None or current_pos == 0:
        return None
    return parsed_frames[current_pos - 1][1]


def select_window(events, start_ts, end_ts):
    mask = (events[:, 2] >= start_ts) & (events[:, 2] <= end_ts)
    return events[mask]


def save_triplet(blurred, restored, kernel, out_path, title=''):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(blurred, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Blurred')
    axes[1].imshow(restored, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Restored')
    axes[2].imshow(kernel, cmap='hot')
    axes[2].set_title('Kernel')
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_selected_combos(nominal_dt, search_cfg):
    tau_values = [ratio * nominal_dt for ratio in search_cfg['tau_ratios']]
    combos = []
    for tau, ks, alpha, beta, omega, outer_iters in product(
        tau_values,
        search_cfg['k_sizes'],
        search_cfg['alphas'],
        search_cfg['betas'],
        search_cfg['omegas'],
        search_cfg['outer_iters'],
    ):
        combos.append((tau, ks, alpha, beta, omega, outer_iters))

    # 为了控制计算量，这里采用“代表性子集”策略，而不是全量暴力组合。
    selected = []
    preferred = [
        (tau_values[0], search_cfg['k_sizes'][0], search_cfg['alphas'][0], search_cfg['betas'][0], search_cfg['omegas'][1], search_cfg['outer_iters'][0]),
        (tau_values[0], search_cfg['k_sizes'][1], search_cfg['alphas'][1], search_cfg['betas'][1], search_cfg['omegas'][1], search_cfg['outer_iters'][0]),
        (tau_values[1], search_cfg['k_sizes'][0], search_cfg['alphas'][0], search_cfg['betas'][0], search_cfg['omegas'][0], search_cfg['outer_iters'][0]),
        (tau_values[1], search_cfg['k_sizes'][1], search_cfg['alphas'][1], search_cfg['betas'][1], search_cfg['omegas'][1], search_cfg['outer_iters'][0]),
        (tau_values[1], search_cfg['k_sizes'][-1], search_cfg['alphas'][-1], search_cfg['betas'][-1], search_cfg['omegas'][-1], search_cfg['outer_iters'][-1]),
        (tau_values[2], search_cfg['k_sizes'][0], search_cfg['alphas'][0], search_cfg['betas'][0], search_cfg['omegas'][1], search_cfg['outer_iters'][0]),
        (tau_values[2], search_cfg['k_sizes'][1], search_cfg['alphas'][1], search_cfg['betas'][1], search_cfg['omegas'][-1], search_cfg['outer_iters'][0]),
        (tau_values[0], search_cfg['k_sizes'][-1], search_cfg['alphas'][-1], search_cfg['betas'][1], search_cfg['omegas'][-1], search_cfg['outer_iters'][-1]),
    ]
    for item in preferred:
        if item in combos:
            selected.append(item)
    return selected


def run_one_frame(dataset_name, frame_idx, end_ts, prev_ts, frame_path, events_all, out_dir, search_cfg):
    logger.info("  [frame %d] 开始处理，时间戳=%d", frame_idx, int(end_ts))
    t_start = time.time()
    blurry = load_frame(frame_path)
    baseline_metrics = grad_metrics(blurry)
    nominal_dt = end_ts - prev_ts
    selected = make_selected_combos(nominal_dt, search_cfg)
    logger.info("  [frame %d] 模糊基线评分=%.6f, Δt=%d, 参数组合数=%d",
                frame_idx, score_metrics(baseline_metrics), int(nominal_dt), len(selected))

    frame_results = []
    frame_dir = os.path.join(out_dir, f'frame_{frame_idx:04d}')
    os.makedirs(frame_dir, exist_ok=True)

    for run_id, (tau, ks, alpha, beta, omega, outer_iters) in enumerate(selected, start=1):
        t_run = time.time()
        logger.info("    [run %d/%d] tau=%d k=%d a=%.3f b=%.4f o=%.3f it=%d",
                    run_id, len(selected), int(tau), ks, alpha, beta, omega, outer_iters)
        start_ts = end_ts - tau
        events = select_window(events_all, start_ts, end_ts)
        out = joint_reconstruction_cuda(
            blurry_image=blurry,
            raw_events=events,
            k_size=(ks, ks),
            outer_iters=outer_iters,
            alpha=alpha,
            beta=beta,
            sigma=search_cfg['sigma'],
            omega=omega,
            contrast_threshold=search_cfg['contrast_threshold'],
        )
        restored = out['restored'].detach().cpu().numpy()
        kernel = out['kernel'].detach().cpu().numpy()
        metrics = grad_metrics(restored)
        score = score_metrics(metrics)
        tag = f'{dataset_name}_f{frame_idx:04d}_r{run_id:02d}_tau{int(tau)}_k{ks}_a{alpha}_b{beta}_o{omega}_it{outer_iters}'
        save_triplet(blurry, restored, kernel, os.path.join(frame_dir, tag + '.png'), title=tag)
        frame_results.append({
            'tag': tag,
            'frame_idx': frame_idx,
            'frame_path': frame_path,
            'tau': float(tau),
            'tau_ratio': float(tau / nominal_dt) if nominal_dt != 0 else None,
            'k_size': ks,
            'alpha': alpha,
            'beta': beta,
            'omega': omega,
            'outer_iters': outer_iters,
            'num_events': int(events.shape[0]),
            'metrics': metrics,
            'score': float(score),
            'device': out['device'],
            'nominal_dt': float(nominal_dt),
        })
        elapsed = time.time() - t_run
        logger.info("    [run %d/%d] 完成, score=%.6f, 事件数=%d, 耗时=%.1fs",
                    run_id, len(selected), score, int(events.shape[0]), elapsed)

    frame_results = sorted(frame_results, key=lambda x: x['score'], reverse=True)
    topn = min(4, len(frame_results))
    fig, axes = plt.subplots(1, topn + 1, figsize=(4 * (topn + 1), 4.2))
    axes[0].imshow(blurry, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Blurred baseline')
    axes[0].axis('off')
    for i in range(topn):
        img = np.array(Image.open(os.path.join(frame_dir, frame_results[i]['tag'] + '.png')).convert('RGB'))
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Top {i+1}\nscore={frame_results[i]['score']:.4f}")
        axes[i + 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(frame_dir, 'top_comparison.png'), dpi=160)
    plt.close()

    with open(os.path.join(frame_dir, 'tuning_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'baseline_metrics': baseline_metrics, 'results': frame_results, 'best': frame_results[0]}, f, indent=2)

    with open(os.path.join(frame_dir, 'best_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Best for {dataset_name} frame {frame_idx}\n')
        for k, v in frame_results[0].items():
            f.write(f'{k}: {v}\n')

    logger.info("  [frame %d] 最佳: tag=%s, score=%.6f, 总耗时=%.1fs",
                frame_idx, frame_results[0]['tag'], frame_results[0]['score'], time.time() - t_start)
    return frame_results[0], frame_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    t_total = time.time()
    cfg = load_config()
    os.makedirs(cfg['output_root'], exist_ok=True)

    datasets = discover_datasets(cfg)
    if not datasets:
        logger.error("未发现可用数据集。请检查 %s/ 下是否包含 event.mat 和 frame/。", cfg['datasets_root'])
        raise FileNotFoundError('未发现可用数据集。请检查 datasets/ 下是否包含 event.mat 和 frame/。')

    global_summary = []

    for ds_idx, ds in enumerate(datasets, start=1):
        dataset_root = ds['dataset_root']
        dataset_name = os.path.basename(os.path.normpath(dataset_root))
        logger.info("=" * 60)
        logger.info("[数据集 %d/%d] %s", ds_idx, len(datasets), dataset_name)
        logger.info("=" * 60)
        t_ds = time.time()
        out_dir = os.path.join(cfg['output_root'], dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info("  加载事件文件: %s", ds['event_path'])
        events_all = load_events(ds['event_path'])
        logger.info("  事件总数: %d", events_all.shape[0])

        frame_paths = sorted(glob.glob(os.path.join(ds['frame_dir'], '*.png')))
        logger.info("  帧目录: %s, 帧文件数: %d", ds['frame_dir'], len(frame_paths))

        parsed_all = []
        for p in frame_paths:
            try:
                parsed_all.append((*parse_frame_info(p), p))
            except Exception:
                continue
        parsed_all.sort(key=lambda x: x[0])
        selected_frames = choose_frames(frame_paths, cfg)
        logger.info("  选中帧: %s", [f[0] for f in selected_frames])

        dataset_report = {
            'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'event_path': ds['event_path'],
            'frame_dir': ds['frame_dir'],
            'frames': [],
        }

        for frame_idx, end_ts, frame_path in selected_frames:
            prev_ts = get_prev_timestamp(parsed_all, frame_idx)
            if prev_ts is None:
                logger.info("  [frame %d] 跳过（无前一帧，无法计算 Δt）", frame_idx)
                continue
            best, all_results = run_one_frame(
                dataset_name=dataset_name,
                frame_idx=frame_idx,
                end_ts=end_ts,
                prev_ts=prev_ts,
                frame_path=frame_path,
                events_all=events_all,
                out_dir=out_dir,
                search_cfg=cfg['default_search'],
            )
            dataset_report['frames'].append({
                'frame_idx': frame_idx,
                'best': best,
                'num_candidates': len(all_results),
            })
            global_summary.append({
                'dataset_name': dataset_name,
                'frame_idx': frame_idx,
                'best_tag': best['tag'],
                'score': best['score'],
                'tau': best['tau'],
                'tau_ratio': best['tau_ratio'],
                'k_size': best['k_size'],
                'alpha': best['alpha'],
                'beta': best['beta'],
                'omega': best['omega'],
                'outer_iters': best['outer_iters'],
            })

        with open(os.path.join(out_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_report, f, indent=2)
        logger.info("[数据集 %d/%d] %s 完成, 耗时=%.1fs", ds_idx, len(datasets), dataset_name, time.time() - t_ds)

    with open(os.path.join(cfg['output_root'], 'global_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2)

    with open(os.path.join(cfg['output_root'], 'global_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 多数据集自动调参汇总\n\n')
        for item in global_summary:
            f.write(
                f"- {item['dataset_name']} / frame {item['frame_idx']}: "
                f"score={item['score']:.6f}, tau={item['tau']:.1f}, tau_ratio={item['tau_ratio']:.3f}, "
                f"k={item['k_size']}, a={item['alpha']}, b={item['beta']}, o={item['omega']}, it={item['outer_iters']}\n"
            )

    logger.info("=" * 60)
    logger.info("全部完成！处理了 %d 个数据集, 总耗时=%.1fs", len(datasets), time.time() - t_total)
    logger.info("结果保存在 %s/", os.path.abspath(cfg['output_root']))

    # 打印汇总表
    if global_summary:
        logger.info("--- 全局汇总 ---")
        for item in global_summary:
            logger.info(
                "  %s / frame %d: score=%.6f tau_ratio=%.3f k=%d a=%.3f b=%.4f o=%.3f it=%d",
                item['dataset_name'], item['frame_idx'], item['score'], item['tau_ratio'],
                item['k_size'], item['alpha'], item['beta'], item['omega'], item['outer_iters'],
            )


if __name__ == '__main__':
    main()
