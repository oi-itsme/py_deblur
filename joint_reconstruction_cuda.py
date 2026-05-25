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

    logger.info("    联合重建开始: k_size=%s, outer_iters=%d, alpha=%.3f, beta=%.4f, omega=%.3f, device=%s",
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
