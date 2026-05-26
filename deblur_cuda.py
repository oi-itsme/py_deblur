"""
去模糊核心模块（PyTorch / CUDA 版本）
=====================================

这个文件实现联合重建算法中"图像去模糊"部分的核心数值步骤。
整体思路与论文保持一致：
1. 在频域中更新潜在清晰图像 S；
2. 使用 L0 梯度先验对应的辅助变量 z 进行硬阈值稀疏约束；
3. 根据当前清晰图像估计模糊核 k；
4. 重复以上步骤，逐步把模糊图像 B 还原为更清晰的图像 S。

为了尽量贴近论文与历史修正版，这里特别保留了两个关键点：
- psf2otf_torch() 在 FFT 前做 MATLAB 风格的 circular shift；
- 事件先验最终以"梯度项"方式进入频域求解，而不是直接把事件图像当作强度真值。
"""

import logging
import torch

logger = logging.getLogger(__name__)


def psf2otf_torch(psf: torch.Tensor, shape):
    """将 PSF 转为 OTF，并保持 MATLAB 风格对齐。"""
    psf = psf.to(dtype=torch.float32)
    if torch.all(psf == 0):
        return torch.zeros(shape, dtype=torch.complex64, device=psf.device)

    ph, pw = psf.shape
    H, W = shape
    if ph > H or pw > W:
        raise ValueError(f"PSF shape {psf.shape} larger than target shape {shape}")

    out = torch.zeros((H, W), dtype=torch.float32, device=psf.device)
    out[:ph, :pw] = psf
    out = torch.roll(out, shifts=(-ph // 2, -pw // 2), dims=(0, 1))
    return torch.fft.fft2(out)


def get_gradient_filters(device):
    """返回水平/垂直差分滤波器。"""
    dx = torch.tensor([[1.0, -1.0]], device=device, dtype=torch.float32)
    dy = torch.tensor([[1.0], [-1.0]], device=device, dtype=torch.float32)
    return dx, dy


def precompute_filters(shape, device):
    """预计算梯度滤波器的频域表示，整个算法过程中不变。"""
    dx, dy = get_gradient_filters(device)
    F_dx = psf2otf_torch(dx, shape)
    F_dy = psf2otf_torch(dy, shape)
    return {
        'F_dx': F_dx,
        'F_dy': F_dy,
        'F_dx_abs2': torch.abs(F_dx) ** 2,
        'F_dy_abs2': torch.abs(F_dy) ** 2,
        'denom_base': torch.abs(F_dx) ** 2 + torch.abs(F_dy) ** 2,
        'shape': shape,
        'device': device,
    }


def gradient_torch(img: torch.Tensor, filters=None):
    """在频域中计算图像梯度。filters 为 precompute_filters() 的结果。"""
    if filters is not None:
        F_dx = filters['F_dx']
        F_dy = filters['F_dy']
    else:
        device = img.device
        dx, dy = get_gradient_filters(device)
        F_dx = psf2otf_torch(dx, img.shape)
        F_dy = psf2otf_torch(dy, img.shape)

    F_img = torch.fft.fft2(img)
    grad_h = torch.real(torch.fft.ifft2(F_img * F_dx))
    grad_v = torch.real(torch.fft.ifft2(F_img * F_dy))
    return grad_h, grad_v


def gradient_torch_spatial(img: torch.Tensor):
    """使用直接空间卷积计算图像梯度，比 FFT 快 5-10x（滤波器仅 2-tap）。"""
    img_4d = img.unsqueeze(0).unsqueeze(0)
    dx = torch.tensor([[[[1.0, -1.0]]]], device=img.device, dtype=img.dtype)
    dy = torch.tensor([[[[1.0], [-1.0]]]], device=img.device, dtype=img.dtype)
    grad_h = torch.nn.functional.conv2d(img_4d, dx, padding=(0, 1))[:, :, :, :-1].squeeze()
    grad_v = torch.nn.functional.conv2d(img_4d, dy, padding=(1, 0))[:, :, :-1, :].squeeze()
    return grad_h, grad_v


def divergence_from_gradients(theta_h: torch.Tensor, theta_v: torch.Tensor, filters=None):
    """把两个方向的梯度场转回散度项（频域形式）。"""
    if filters is not None:
        F_dx = filters['F_dx']
        F_dy = filters['F_dy']
    else:
        device = theta_h.device
        dx, dy = get_gradient_filters(device)
        F_dx = psf2otf_torch(dx, theta_h.shape)
        F_dy = psf2otf_torch(dy, theta_h.shape)

    return torch.conj(F_dx) * torch.fft.fft2(theta_h) + torch.conj(F_dy) * torch.fft.fft2(theta_v)


def update_latent_image(F_B, F_k, event_div, z_div, alpha, gamma, filters, eps=1e-6):
    """更新潜在清晰图像 S。

    F_B: 模糊图像的 FFT（预计算，不随迭代变化）
    filters: precompute_filters() 的结果
    """
    numerator = torch.conj(F_k) * F_B
    if event_div is not None:
        numerator = numerator + alpha * event_div
    numerator = numerator + gamma * z_div

    denominator = (
        torch.abs(F_k) ** 2
        + (alpha + gamma) * filters['denom_base']
        + eps
    )

    latent = torch.real(torch.fft.ifft2(numerator / denominator))
    return latent


def update_auxiliary_gradients(S, beta, gamma):
    """更新 L0 梯度先验中的辅助变量。"""
    grad_h, grad_v = gradient_torch_spatial(S)
    keep = (grad_h.pow(2) + grad_v.pow(2)) > (beta / max(gamma, 1e-8))
    z_h = torch.where(keep, grad_h, torch.zeros_like(grad_h))
    z_v = torch.where(keep, grad_v, torch.zeros_like(grad_v))
    return z_h, z_v


def update_blur_kernel(F_gBh, F_gBv, S, filters, k_size=(25, 25), sigma=2e-3, nonnegative=True):
    """根据当前 S 和预计算的 B 梯度 FFT 估计模糊核。

    F_gBh, F_gBv: 模糊图像梯度的 FFT（预计算，不随迭代变化）
    """
    H, W = S.shape
    kh, kw = k_size

    # Compute gradients directly in frequency domain to skip 2 IFFT + 2 FFT
    F_S = torch.fft.fft2(S)
    F_gSh = F_S * filters['F_dx']
    F_gSv = F_S * filters['F_dy']

    num = torch.conj(F_gSh) * F_gBh + torch.conj(F_gSv) * F_gBv
    den = torch.abs(F_gSh) ** 2 + torch.abs(F_gSv) ** 2 + sigma
    kernel_full = torch.real(torch.fft.ifft2(num / den))
    kernel_full = torch.fft.fftshift(kernel_full)

    cy, cx = H // 2, W // 2
    hy, hx = kh // 2, kw // 2
    kernel = kernel_full[cy - hy: cy + hy + 1, cx - hx: cx + hx + 1]

    if nonnegative:
        kernel = torch.clamp(kernel, min=0.0)
    s = kernel.sum()
    if s > 0:
        kernel = kernel / s
    else:
        kernel = torch.zeros_like(kernel)
        kernel[kh // 2, kw // 2] = 1.0
    return kernel


def precompute_blurry_fft(B, filters):
    """预计算模糊图像相关的 FFT，整个算法过程中不变。"""
    F_B = torch.fft.fft2(B)
    grad_B_h, grad_B_v = gradient_torch(B, filters=filters)
    F_gBh = torch.fft.fft2(grad_B_h)
    F_gBv = torch.fft.fft2(grad_B_v)
    return {'F_B': F_B, 'F_gBh': F_gBh, 'F_gBv': F_gBv}
