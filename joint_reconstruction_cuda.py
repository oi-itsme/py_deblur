"""
联合图像去模糊与事件去噪主流程（PyTorch / CUDA 版本）
====================================================
"""

import logging
import numpy as np
import torch
from deblur_cuda import (
    gradient_torch,
    gradient_torch_spatial,
    precompute_filters,
    precompute_blurry_fft,
    psf2otf_torch,
    update_auxiliary_gradients,
    update_blur_kernel,
    update_latent_image,
    divergence_from_gradients,
)

logger = logging.getLogger(__name__)


def normalize_image(img: torch.Tensor):
    """归一化到 [0, 1]。"""
    img = img - img.min()
    denom = img.max().clamp_min(1e-6)
    return img / denom


def accumulate_events_to_image(events: torch.Tensor, shape, contrast_threshold=1.0):
    """将事件累积为一张事件图 I_tau。"""
    H, W = shape
    device = events.device
    event_img = torch.zeros((H, W), dtype=torch.float32, device=device)
    if events.numel() == 0:
        return event_img

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 3].float()

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    event_img.index_put_((y[valid], x[valid]), contrast_threshold * p[valid], accumulate=True)
    return event_img


def generate_event_gradient_prior(events: torch.Tensor, shape, contrast_threshold=1.0):
    """先累积事件图，再取梯度，得到事件先验。"""
    event_img = accumulate_events_to_image(events, shape, contrast_threshold=contrast_threshold)
    prior_h, prior_v = gradient_torch_spatial(event_img)
    return prior_h, prior_v


def _compute_gradient_mode(grad_mag: torch.Tensor, bins: int = 100) -> float:
    """计算梯度幅值直方图的众数 q（Eq.25）。

    q 代表图像中"平坦背景"区域的典型梯度值，用于自适应区分
    真实边缘（梯度远离 q）与噪声（梯度接近 q）。
    """
    max_val = grad_mag.max().item()
    if max_val <= 0:
        return 0.0
    hist = torch.histc(grad_mag.flatten(), bins=bins, min=0, max=max_val)
    bin_width = max_val / bins
    q = (hist.argmax().float() + 0.5) * bin_width
    return q.item()


def filter_events_by_latent_gradient(events: torch.Tensor, S: torch.Tensor, omega=0.05):
    """根据当前恢复图像梯度，对事件进行筛选（Eq.25）。

    q = 梯度幅值众数（代表平坦背景）
    保留 |∇S| ∉ (q-ω, q+ω) 的事件，即只保留边缘处的事件。
    """
    if events.numel() == 0:
        return events

    H, W = S.shape
    grad_h, grad_v = gradient_torch_spatial(S)
    grad_mag = torch.sqrt(grad_h.pow(2) + grad_v.pow(2))
    q = _compute_gradient_mode(grad_mag)

    x = events[:, 0].long()
    y = events[:, 1].long()
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if valid.sum() == 0:
        return events[:0]

    sampled = grad_mag[y[valid], x[valid]]
    # Eq.25: keep if gradient is outside the background interval
    keep_valid = (sampled <= q - omega) | (sampled >= q + omega)
    valid_indices = torch.where(valid)[0]
    keep_indices = valid_indices[keep_valid]
    return events[keep_indices]


def spatiotemporal_compensation(events: torch.Tensor, filtered_events: torch.Tensor,
                                mu: int = 2, nu: float = 5000.0) -> torch.Tensor:
    """时空邻域补偿（Eq.26）：对梯度筛选后的事件，搜索其时空邻居并一并保留。

    对于每个被保留下来的事件 ė_i，在原始事件流 E 中寻找满足
    空间距离 ≤ μ 且时间距离 ≤ ν 的邻居 e_n，将其也视为真实信号。

    实现：批量将事件传到 CPU，用 numpy 向量化窗口搜索，避免 per-event GPU sync。
    """
    if filtered_events.numel() == 0:
        return filtered_events
    if events.numel() == 0:
        return events

    # 一次性传到 CPU
    events_np = events.cpu().numpy()
    kept_np = filtered_events.cpu().numpy()

    # 按时间排序
    sort_idx = np.argsort(events_np[:, 2])
    sorted_events = events_np[sort_idx]
    sorted_t = np.ascontiguousarray(sorted_events[:, 2])

    n = len(sorted_events)
    compensated = set()

    for i in range(len(kept_np)):
        t_i = kept_np[i, 2]
        x_i = kept_np[i, 0]
        y_i = kept_np[i, 1]

        left = np.searchsorted(sorted_t, t_i - nu)
        right = min(np.searchsorted(sorted_t, t_i + nu), n)
        if right <= left:
            continue

        window = sorted_events[left:right]
        dx = np.abs(window[:, 0] - x_i)
        dy = np.abs(window[:, 1] - y_i)
        matches = np.where((dx <= mu) & (dy <= mu))[0]
        for m in matches:
            compensated.add(int(sort_idx[left + m]))

    if not compensated:
        return filtered_events

    comp_indices = torch.tensor(sorted(compensated), dtype=torch.long, device=events.device)
    combined = torch.cat([filtered_events, events[comp_indices]], dim=0)
    return torch.unique(combined, dim=0)


def joint_reconstruction_cuda(
    blurry_image,
    raw_events,
    k_size=(25, 25),
    outer_iters=8,
    alpha=0.24,
    beta=0.064,
    sigma=2e-3,
    omega=0.05,
    mu=2,
    nu=5000.0,
    gamma_init=2.0,
    gamma_max=1e5,
    contrast_threshold=1.0,
    device=None,
):
    """联合重建主函数（单帧版本）。"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B = torch.as_tensor(blurry_image, dtype=torch.float32, device=device)
    if B.ndim != 2:
        raise ValueError('blurry_image must be 2D grayscale')
    B = normalize_image(B)

    E = torch.as_tensor(raw_events, dtype=torch.float32, device=device)
    if E.ndim != 2 or E.shape[1] != 4:
        raise ValueError('raw_events must have shape [N,4] = [x,y,t,p]')

    H, W = B.shape

    # ---- 预计算不变数据 ----
    filters = precompute_filters((H, W), device)
    blurry_fft = precompute_blurry_fft(B, filters)

    S = B.clone()
    kh, kw = k_size
    k = torch.zeros((kh, kw), dtype=torch.float32, device=device)
    k[kh // 2, kw // 2] = 1.0

    gamma = gamma_init
    history = []
    # Initialize auxiliary variables for L0 gradient prior (used in first iteration's S update)
    z_h = torch.zeros_like(S)
    z_v = torch.zeros_like(S)

    logger.debug("    联合重建开始: k_size=%s, outer_iters=%d, alpha=%.3f, beta=%.4f, omega=%.3f, device=%s",
                 k_size, outer_iters, alpha, beta, omega, device)

    for it in range(outer_iters):
        # Eq.25: 基于当前潜像梯度自适应阈值过滤事件
        filtered_events = filter_events_by_latent_gradient(E, S, omega=omega)
        # Eq.26: 时空邻域补偿
        filtered_events = spatiotemporal_compensation(E, filtered_events, mu=mu, nu=nu)
        E = filtered_events

        event_grad_h, event_grad_v = generate_event_gradient_prior(
            filtered_events, B.shape, contrast_threshold=contrast_threshold
        )
        event_div = divergence_from_gradients(event_grad_h, event_grad_v, filters=filters)
        # z_div uses z from previous iteration (or zero init on first iteration)
        z_div = divergence_from_gradients(z_h, z_v, filters=filters)

        F_k = psf2otf_torch(k, (H, W))

        S = update_latent_image(blurry_fft['F_B'], F_k, event_div, z_div, alpha, gamma, filters)
        S = torch.clamp(S, 0.0, 1.0)
        # Update z from NEW S (Algorithm 1: z after S) — will be used next iteration
        z_h, z_v = update_auxiliary_gradients(S, beta=beta, gamma=gamma)
        k = update_blur_kernel(blurry_fft['F_gBh'], blurry_fft['F_gBv'], S,
                               k_size=k_size, sigma=sigma, filters=filters)

        if logger.isEnabledFor(logging.DEBUG):
            history.append(
                {
                    'iter': it + 1,
                    'num_events': int(filtered_events.shape[0]),
                    'kernel_sum': float(k.sum().item()),
                    'image_min': float(S.min().item()),
                    'image_max': float(S.max().item()),
                }
            )
            logger.debug("    迭代 %d/%d: events=%d kernel_sum=%.4f img_range=[%.3f, %.3f] gamma=%.1f",
                         it + 1, outer_iters, int(filtered_events.shape[0]),
                         float(k.sum().item()), float(S.min().item()), float(S.max().item()), gamma)
        gamma = min(gamma * 2.0, gamma_max)

    return {
        'restored': S,
        'kernel': k,
        'filtered_events': filtered_events,
        'history': history,
        'device': device,
    }
