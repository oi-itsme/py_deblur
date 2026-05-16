import numpy as np
from numpy.fft import fft2, ifft2

def psf2otf(psf, shape):
    """
    将点扩散函数 (PSF) 转换为光学传递函数 (OTF)。
    这对齐了 MATLAB 的 psf2otf 函数，是在频域求解 S 和 k 时的标准操作。
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_shape = psf.shape
    # 将 PSF 填充到目标图像大小
    psf_padded = np.pad(psf, ((0, shape[0] - in_shape[0]), (0, shape[1] - in_shape[1])), mode='constant')

    # 循环移位，将中心移动到左上角 (0, 0)
    for axis, axis_size in enumerate(in_shape):
        psf_padded = np.roll(psf_padded, -int(axis_size / 2), axis=axis)

    otf = fft2(psf_padded)
    return otf

def update_S(B, k, I_tau_h, I_tau_v, z_h, z_v, alpha, gamma):
    """
    根据公式 (19) 更新清晰图像 S。
    B: 模糊图像
    k: 当前估计的模糊核
    I_tau_h, I_tau_v: 事件积分产生的梯度先验 (水平和垂直方向)
    z_h, z_v: L0 优化的辅助变量 (水平和垂直方向)
    alpha: 事件先验的权重
    gamma: 辅助变量的惩罚权重
    """
    h, w = B.shape

    # 图像梯度算子
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    F_B = fft2(B)
    F_k = psf2otf(k, (h, w))
    F_k_conj = np.conj(F_k)

    F_dx = psf2otf(dx, (h, w))
    F_dy = psf2otf(dy, (h, w))
    F_dx_conj = np.conj(F_dx)
    F_dy_conj = np.conj(F_dy)

    # 辅助变量 z 的频域项
    F_zh = fft2(z_h)
    F_zv = fft2(z_v)

    # 事件先验的频域项 (考虑到事件触发对应于边缘，可能被映射到 x 和 y 梯度)
    # 论文中：\hat{F}(\theta_0, \theta_1) = F(\partial_h)^* F(\theta_0) + F(\partial_v)^* F(\theta_1)
    F_I_tau_h = fft2(I_tau_h) if I_tau_h is not None else 0
    F_I_tau_v = fft2(I_tau_v) if I_tau_v is not None else 0

    # 计算分子 (Numerator)
    num1 = F_k_conj * F_B
    num2 = alpha * (F_dx_conj * F_I_tau_h + F_dy_conj * F_I_tau_v) if I_tau_h is not None else 0
    num3 = gamma * (F_dx_conj * F_zh + F_dy_conj * F_zv)
    numerator = num1 + num2 + num3

    # 计算分母 (Denominator)
    den1 = F_k_conj * F_k
    den2 = (alpha + gamma) * (F_dx_conj * F_dx + F_dy_conj * F_dy)
    denominator = den1 + den2

    # 防止分母为0
    F_S = numerator / (denominator + 1e-8)

    S = np.real(ifft2(F_S))
    return S

def update_z(S, beta, gamma):
    """
    根据公式 (21) 更新辅助变量 z (硬阈值)。
    """
    # 计算 S 的梯度
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])
    grad_h = np.real(ifft2(fft2(S) * psf2otf(dx, S.shape)))
    grad_v = np.real(ifft2(fft2(S) * psf2otf(dy, S.shape)))

    # 公式 21: |\nabla S|^2 <= beta / gamma 时 z = 0, 否则 z = \nabla S
    grad_mag_sq = grad_h**2 + grad_v**2
    threshold = beta / gamma

    mask = grad_mag_sq > threshold

    z_h = grad_h * mask
    z_v = grad_v * mask

    return z_h, z_v

def update_k(B, S, k_size, sigma):
    """
    根据公式 (23) 更新模糊核 k。
    """
    h, w = B.shape
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    # 计算 B 和 S 的梯度
    grad_B_h = np.real(ifft2(fft2(B) * psf2otf(dx, B.shape)))
    grad_B_v = np.real(ifft2(fft2(B) * psf2otf(dy, B.shape)))

    grad_S_h = np.real(ifft2(fft2(S) * psf2otf(dx, S.shape)))
    grad_S_v = np.real(ifft2(fft2(S) * psf2otf(dy, S.shape)))

    # 转到频域求解 k
    F_grad_S_h = fft2(grad_S_h)
    F_grad_S_v = fft2(grad_S_v)
    F_grad_B_h = fft2(grad_B_h)
    F_grad_B_v = fft2(grad_B_v)

    # \hat{F}(\nabla S) 和 \hat{F}(\nabla B) 的运算
    num = np.conj(F_grad_S_h) * F_grad_B_h + np.conj(F_grad_S_v) * F_grad_B_v
    den = np.conj(F_grad_S_h) * F_grad_S_h + np.conj(F_grad_S_v) * F_grad_S_v + sigma

    F_k = num / den
    k_full = np.real(ifft2(F_k))

    # 从完整尺寸的 k_full 截取所需大小的 kernel (居中)
    center_y, center_x = h // 2, w // 2
    half_y, half_x = k_size[0] // 2, k_size[1] // 2

    # fftshift 将低频移到中心
    k_full_shifted = np.fft.fftshift(k_full)
    k = k_full_shifted[center_y - half_y : center_y + half_y + 1, 
                       center_x - half_x : center_x + half_x + 1]

    # 负值置零并归一化
    k[k < 0] = 0
    k_sum = np.sum(k)
    if k_sum > 0:
        k = k / k_sum

    return k

def blind_deblurring_iter(B, I_tau_h, I_tau_v, k_size, max_iter=5, alpha=0.24, beta=0.064, sigma=2.0):
    """
    整合上述步骤的外层迭代框架。
    """
    # 初始化模糊核为冲击函数 (中心为1)
    k = np.zeros(k_size)
    k[k_size[0]//2, k_size[1]//2] = 1.0

    # 初始化 S 为模糊图像 B
    S = np.copy(B)

    # 参数调整
    gamma_max = 1e5
    gamma = 2.0  # 初始 gamma

    for iteration in range(max_iter):
        # 1. 更新辅助变量 z
        z_h, z_v = update_z(S, beta, gamma)

        # 2. 更新清晰图像 S
        S = update_S(B, k, I_tau_h, I_tau_v, z_h, z_v, alpha, gamma)

        # 3. 更新模糊核 k
        k = update_k(B, S, k_size, sigma)

        # 论文提到在迭代过程中可以渐进增大 gamma
        gamma = min(gamma * 2.0, gamma_max)

    return S, k
