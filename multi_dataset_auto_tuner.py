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
import shutil
import time
from itertools import product

import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

from joint_reconstruction_cuda import joint_reconstruction_cuda

logger = logging.getLogger(__name__)


def _fmt(n):
    """1234567 → '1.2M', 127932 → '128k', 0.0123 → '0.012'."""
    if isinstance(n, float) and n < 1:
        return f"{n:.3f}"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(int(n))


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
}


def load_config(path='dataset_config_example.json'):
    if os.path.exists(path):
        logger.info("加载配置文件: %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        for key in cfg:
            if key in user_cfg and key != 'default_search':
                cfg[key] = user_cfg[key]
        if 'default_search' in user_cfg:
            merged = DEFAULT_CONFIG['default_search'].copy()
            merged.update(user_cfg['default_search'])
            cfg['default_search'] = merged
        return cfg
    logger.info("未找到配置文件 %s，使用默认配置", path)
    return DEFAULT_CONFIG


def load_events(mat_path):
    mat = sio.loadmat(mat_path)
    events = np.asarray(mat['event']).astype(np.float32)
    if events.shape[0] == 0:
        return events
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
    m = re.match(r'^(\d+)_(-?\d+(?:\.\d+)?)\.png$', name)
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
        (tau_values[0], search_cfg['k_sizes'][min(0, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(0, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(0, len(search_cfg['betas'])-1)], search_cfg['omegas'][min(1, len(search_cfg['omegas'])-1)], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[0], search_cfg['k_sizes'][min(1, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(1, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(1, len(search_cfg['betas'])-1)], search_cfg['omegas'][min(1, len(search_cfg['omegas'])-1)], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[1], search_cfg['k_sizes'][min(0, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(0, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(0, len(search_cfg['betas'])-1)], search_cfg['omegas'][min(0, len(search_cfg['omegas'])-1)], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[1], search_cfg['k_sizes'][min(1, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(1, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(1, len(search_cfg['betas'])-1)], search_cfg['omegas'][min(1, len(search_cfg['omegas'])-1)], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[1], search_cfg['k_sizes'][len(search_cfg['k_sizes'])-1], search_cfg['alphas'][len(search_cfg['alphas'])-1], search_cfg['betas'][len(search_cfg['betas'])-1], search_cfg['omegas'][len(search_cfg['omegas'])-1], search_cfg['outer_iters'][len(search_cfg['outer_iters'])-1]),
        (tau_values[2], search_cfg['k_sizes'][min(0, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(0, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(0, len(search_cfg['betas'])-1)], search_cfg['omegas'][min(1, len(search_cfg['omegas'])-1)], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[2], search_cfg['k_sizes'][min(1, len(search_cfg['k_sizes'])-1)], search_cfg['alphas'][min(1, len(search_cfg['alphas'])-1)], search_cfg['betas'][min(1, len(search_cfg['betas'])-1)], search_cfg['omegas'][len(search_cfg['omegas'])-1], search_cfg['outer_iters'][min(0, len(search_cfg['outer_iters'])-1)]),
        (tau_values[0], search_cfg['k_sizes'][len(search_cfg['k_sizes'])-1], search_cfg['alphas'][len(search_cfg['alphas'])-1], search_cfg['betas'][min(1, len(search_cfg['betas'])-1)], search_cfg['omegas'][len(search_cfg['omegas'])-1], search_cfg['outer_iters'][len(search_cfg['outer_iters'])-1]),
    ]
    for item in preferred:
        if item in combos:
            selected.append(item)
    return selected


def run_one_frame(dataset_name, frame_idx, end_ts, prev_ts, frame_path, events_all, out_dir, search_cfg):
    t_start = time.time()
    blurry = load_frame(frame_path)
    baseline_metrics = grad_metrics(blurry)
    baseline_score = score_metrics(baseline_metrics)
    nominal_dt = end_ts - prev_ts
    selected = make_selected_combos(nominal_dt, search_cfg)

    logger.info("  [f%d] ts=%s Δt=%s baseline=%s | %d combos",
                frame_idx, _fmt(end_ts), _fmt(nominal_dt), _fmt(baseline_score), len(selected))

    frame_results = []
    frame_dir = os.path.join(out_dir, f'frame_{frame_idx:04d}')
    os.makedirs(frame_dir, exist_ok=True)

    for run_id, (tau, ks, alpha, beta, omega, outer_iters) in enumerate(selected, start=1):
        t_run = time.time()
        start_ts = end_ts - tau
        events = events_all[(events_all[:, 2] >= start_ts) & (events_all[:, 2] <= end_ts)]
        tau_ratio = tau / nominal_dt if nominal_dt != 0 else 0
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
            'tau_ratio': float(tau_ratio),
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
        logger.info("    [%d/%d] τ=%s(%.1f) k=%d e=%s → %s  %.1fs",
                    run_id, len(selected), _fmt(tau), tau_ratio, ks, _fmt(events.shape[0]), _fmt(score), elapsed)

    frame_results = sorted(frame_results, key=lambda x: x['score'], reverse=True)
    best = frame_results[0]
    delta_pct = (best['score'] / baseline_score - 1) * 100 if baseline_score > 0 else 0
    sign = '↑' if delta_pct >= 0 else '↓'
    logger.info("    ✔ τ=%s(%.1f) k=%d → %s %s%.0f%%  %.1fs",
                _fmt(best['tau']), best['tau_ratio'], best['k_size'], _fmt(best['score']), sign, abs(delta_pct),
                time.time() - t_start)
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

    return frame_results[0], frame_results, baseline_score


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
        t_ds = time.time()
        out_dir = os.path.join(cfg['output_root'], dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        events_all = load_events(ds['event_path'])

        frame_paths = sorted(glob.glob(os.path.join(ds['frame_dir'], '*.png')))
        parsed_all = []
        for p in frame_paths:
            try:
                parsed_all.append((*parse_frame_info(p), p))
            except Exception:
                continue
        parsed_all.sort(key=lambda x: x[0])

        logger.info("===== %s (%d/%d) | %d frames | %s events =====",
                    dataset_name, ds_idx, len(datasets), len(parsed_all), _fmt(events_all.shape[0]))

        if not parsed_all:
            logger.warning("  无有效帧，跳过")
            continue

        dataset_report = {
            'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'event_path': ds['event_path'],
            'frame_dir': ds['frame_dir'],
            'frames': [],
        }

        if len(parsed_all) < 1:
            logger.warning("  无有效帧，跳过")
            continue

        # 逐帧独立处理（论文 Algorithm 1 的设计）
        frame_best_list = []
        frame_baselines = []
        events_min_ts = float(events_all[:, 2].min())

        for i, (frame_idx, end_ts, frame_path) in enumerate(parsed_all):
            # 首帧用事件流最早时间戳作为 prev_ts 兜底
            prev_ts = parsed_all[i - 1][1] if i > 0 else events_min_ts
            best, _, baseline = run_one_frame(
                dataset_name=dataset_name,
                frame_idx=frame_idx,
                end_ts=end_ts,
                prev_ts=prev_ts,
                frame_path=frame_path,
                events_all=events_all,
                out_dir=out_dir,
                search_cfg=cfg['default_search'],
            )
            frame_best_list.append(best)
            frame_baselines.append(baseline)

        dataset_report['frames'] = [
            {
                'frame_idx': b['frame_idx'],
                'baseline_score': frame_baselines[i],
                'best_score': b['score'],
            }
            for i, b in enumerate(frame_best_list)
        ]
        avg_baseline = sum(frame_baselines) / len(frame_baselines) if frame_baselines else 0
        avg_score = sum(b['score'] for b in frame_best_list) / len(frame_best_list) if frame_best_list else 0
        global_summary.append({
            'dataset_name': dataset_name,
            'num_frames': len(parsed_all),
            'avg_baseline': avg_baseline,
            'avg_score': avg_score,
            'tau_ratio': frame_best_list[0]['tau_ratio'] if frame_best_list else 0,
            'k_size': frame_best_list[0]['k_size'] if frame_best_list else 0,
            'alpha': frame_best_list[0]['alpha'] if frame_best_list else 0,
            'beta': frame_best_list[0]['beta'] if frame_best_list else 0,
            'omega': frame_best_list[0]['omega'] if frame_best_list else 0,
            'outer_iters': frame_best_list[0]['outer_iters'] if frame_best_list else 0,
        })

        with open(os.path.join(out_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_report, f, indent=2)

        ds_elapsed = time.time() - t_ds
        frames_done = len(dataset_report['frames'])
        if frames_done:
            ds_scores = [f['best_score'] for f in dataset_report['frames']]
            ds_bases = [f['baseline_score'] for f in dataset_report['frames']]
            avg_improve = sum(s / b - 1 for s, b in zip(ds_scores, ds_bases)) / frames_done * 100
            sign = '↑' if avg_improve >= 0 else '↓'
            logger.info("%s ✓ %d frames | %s | avg %s%.0f%%",
                        dataset_name, frames_done, _fmt(ds_elapsed) + 's', sign, abs(avg_improve))
        else:
            logger.info("%s ✓ 0 frames | %s",
                        dataset_name, _fmt(ds_elapsed) + 's')

    with open(os.path.join(cfg['output_root'], 'global_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2)

    with open(os.path.join(cfg['output_root'], 'global_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 多数据集全局优化汇总\n\n')
        f.write('| dataset | frames | avg_baseline | avg_score | τ_ratio | k | α | β | ω | it |\n')
        f.write('|---------|--------|-------------|-----------|---------|----|------|-------|------|----|\n')
        for item in global_summary:
            f.write(
                f"| {item['dataset_name']} | {item['num_frames']} | {item['avg_baseline']:.4f} | "
                f"{item['avg_score']:.4f} | {item['tau_ratio']:.2f} | {item['k_size']} | "
                f"{item['alpha']} | {item['beta']} | {item['omega']} | {item['outer_iters']} |\n"
            )

    logger.info("===== 全局汇总 | %d datasets | %s =====",
                len(global_summary), _fmt(time.time() - t_total) + 's')
    for item in global_summary:
        delta_pct = (item['avg_score'] / item['avg_baseline'] - 1) * 100 if item['avg_baseline'] > 0 else 0
        sign = '↑' if delta_pct >= 0 else '↓'
        logger.info("  %-20s  frames=%2d  avg=%.4f  base=%.4f  τr=%.2f  k=%d  %s%.0f%%",
                    item['dataset_name'], item['num_frames'], item['avg_score'],
                    item['avg_baseline'], item['tau_ratio'], item['k_size'], sign, abs(delta_pct))
    logger.info("结果保存在 %s/", os.path.abspath(cfg['output_root']))


if __name__ == '__main__':
    main()
