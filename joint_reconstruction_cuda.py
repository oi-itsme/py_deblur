"""
联合图像去模糊与事件去噪主流程（PyTorch / CUDA 版本）
====================================================
"""

import logging
import torch
from deblur_cuda import (
    gradient_torch,
    update_auxiliary_gradients,
    update_blur_kernel,
    update_latent_image,
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
    prior_h, prior_v = gradient_torch(event_img)
    return prior_h, prior_v, event_img


def filter_events_by_latent_gradient(events: torch.Tensor, S: torch.Tensor, omega=0.05):
    """根据当前恢复图像梯度，对事件进行筛选。"""
    if events.numel() == 0:
        return events

    H, W = S.shape
    grad_h, grad_v = gradient_torch(S)
    grad_mag = torch.sqrt(grad_h.pow(2) + grad_v.pow(2))

    x = events[:, 0].long()
    y = events[:, 1].long()
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if valid.sum() == 0:
        return events[:0]

    sampled = grad_mag[y[valid], x[valid]]
    keep_valid = sampled >= omega
    valid_indices = torch.where(valid)[0]
    keep_indices = valid_indices[keep_valid]
    return events[keep_indices]


def compute_global_gradient_activity(S_list):
    """用所有帧的潜像计算全局梯度活动图。

    逐像素取所有帧中梯度幅值的最大值，含义是：
    只要某个像素在任意一帧中有强梯度，就认为是真实边缘位置。
    """
    global_map = None
    for S in S_list:
        grad_h, grad_v = gradient_torch(S)
        grad_mag = torch.sqrt(grad_h.pow(2) + grad_v.pow(2))
        if global_map is None:
            global_map = grad_mag
        else:
            global_map = torch.maximum(global_map, grad_mag)
    return global_map


def filter_events_by_global_gradient(events: torch.Tensor, global_grad_map: torch.Tensor, omega=0.05):
    """使用预计算的全局梯度图对事件进行筛选。"""
    if events.numel() == 0:
        return events

    H, W = global_grad_map.shape
    x = events[:, 0].long()
    y = events[:, 1].long()
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if valid.sum() == 0:
        return events[:0]

    sampled = global_grad_map[y[valid], x[valid]]
    keep_valid = sampled >= omega
    valid_indices = torch.where(valid)[0]
    keep_indices = valid_indices[keep_valid]
    return events[keep_indices]


def global_joint_reconstruction(
    blurry_images,
    frame_timestamps,
    raw_events,
    tau_ratio=1.0,
    k_size=(25, 25),
    outer_iters=8,
    alpha=0.24,
    beta=0.064,
    sigma=2e-3,
    omega=0.05,
    gamma_init=2.0,
    gamma_max=1e5,
    contrast_threshold=1.0,
    device=None,
):
    """全局联合重建：用所有帧降噪事件流，再用事件流降噪所有帧。

    与单帧版本的区别：
    - 事件筛选使用所有帧梯度信息的 max（全局梯度活动图），而非仅看当前帧
    - 外迭代在所有帧的层面进行，每轮先全局筛选事件，再逐帧更新潜像和模糊核
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_frames = len(blurry_images)
    if num_frames == 0:
        return {'restored': [], 'kernels': [], 'history': [], 'device': device}

    H, W = blurry_images[0].shape

    B_list = []
    for b in blurry_images:
        B_t = torch.as_tensor(b, dtype=torch.float32, device=device)
        if B_t.ndim != 2:
            raise ValueError('blurry_image must be 2D grayscale')
        B_list.append(normalize_image(B_t))

    E = torch.as_tensor(raw_events, dtype=torch.float32, device=device)
    if E.ndim != 2 or E.shape[1] != 4:
        raise ValueError('raw_events must have shape [N,4] = [x,y,t,p]')

    S_list = [b.clone() for b in B_list]
    kh, kw = k_size
    k_list = []
    for _ in range(num_frames):
        k = torch.zeros((kh, kw), dtype=torch.float32, device=device)
        k[kh // 2, kw // 2] = 1.0
        k_list.append(k)

    # 按帧间隔计算每帧的 tau
    tau_list = []
    for i in range(num_frames):
        if i == 0:
            dt = frame_timestamps[1] - frame_timestamps[0] if num_frames > 1 else 0
        else:
            dt = frame_timestamps[i] - frame_timestamps[i - 1]
        tau_list.append(tau_ratio * dt)

    gamma = gamma_init
    history = []

    logger.debug("    全局联合重建开始: frames=%d, k_size=%s, outer_iters=%d, alpha=%.3f, beta=%.4f, omega=%.3f, device=%s",
                 num_frames, k_size, outer_iters, alpha, beta, omega, device)

    for it in range(outer_iters):
        # Phase 1: 用所有帧降噪事件流
        global_grad_map = None
        if it > 0:
            global_grad_map = compute_global_gradient_activity(S_list)

        frame_event_grads = []
        frame_num_events = []
        for i in range(num_frames):
            t_i = frame_timestamps[i]
            tau_i = tau_list[i]
            mask = (E[:, 2] >= t_i - tau_i) & (E[:, 2] <= t_i)
            events_i = E[mask]

            if global_grad_map is not None:
                events_i = filter_events_by_global_gradient(events_i, global_grad_map, omega=omega)

            event_grad_h, event_grad_v, _ = generate_event_gradient_prior(
                events_i, (H, W), contrast_threshold=contrast_threshold
            )
            frame_event_grads.append((event_grad_h, event_grad_v))
            frame_num_events.append(int(events_i.shape[0]))

        # Phase 2: 用事件流降噪所有帧
        for i in range(num_frames):
            event_grad_h, event_grad_v = frame_event_grads[i]
            z_h, z_v = update_auxiliary_gradients(S_list[i], beta=beta, gamma=gamma)
            S_list[i] = update_latent_image(
                B_list[i], k_list[i], event_grad_h, event_grad_v, z_h, z_v, alpha=alpha, gamma=gamma
            )
            S_list[i] = torch.clamp(S_list[i], 0.0, 1.0)
            k_list[i] = update_blur_kernel(B_list[i], S_list[i], k_size=k_size, sigma=sigma)

        history.append({
            'iter': it + 1,
            'num_events': frame_num_events,
            'gamma': float(gamma),
        })
        logger.debug("    迭代 %d/%d: events=%s gamma=%.1f",
                     it + 1, outer_iters, frame_num_events, gamma)
        gamma = min(gamma * 2.0, gamma_max)

    return {
        'restored': S_list,
        'kernels': k_list,
        'history': history,
        'device': device,
    }


def joint_reconstruction_cuda(
    blurry_image,
    raw_events,
    k_size=(25, 25),
    outer_iters=8,
    alpha=0.24,
    beta=0.064,
    sigma=2e-3,
    omega=0.05,
    gamma_init=2.0,
    gamma_max=1e5,
    contrast_threshold=1.0,
    device=None,
):
    """联合重建主函数。"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B = torch.as_tensor(blurry_image, dtype=torch.float32, device=device)
    if B.ndim != 2:
        raise ValueError('blurry_image must be 2D grayscale')
    B = normalize_image(B)

    E = torch.as_tensor(raw_events, dtype=torch.float32, device=device)
    if E.ndim != 2 or E.shape[1] != 4:
        raise ValueError('raw_events must have shape [N,4] = [x,y,t,p]')

    S = B.clone()
    kh, kw = k_size
    k = torch.zeros((kh, kw), dtype=torch.float32, device=device)
    k[kh // 2, kw // 2] = 1.0

    filtered_events = E
    gamma = gamma_init
    history = []

    logger.debug("    联合重建开始: k_size=%s, outer_iters=%d, alpha=%.3f, beta=%.4f, omega=%.3f, device=%s",
                 k_size, outer_iters, alpha, beta, omega, device)

    for it in range(outer_iters):
        if it > 0:
            filtered_events = filter_events_by_latent_gradient(E, S, omega=omega)

        event_grad_h, event_grad_v, event_img = generate_event_gradient_prior(
            filtered_events, B.shape, contrast_threshold=contrast_threshold
        )
        z_h, z_v = update_auxiliary_gradients(S, beta=beta, gamma=gamma)
        S = update_latent_image(B, k, event_grad_h, event_grad_v, z_h, z_v, alpha=alpha, gamma=gamma)
        S = torch.clamp(S, 0.0, 1.0)
        k = update_blur_kernel(B, S, k_size=k_size, sigma=sigma)

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
