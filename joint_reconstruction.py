import numpy as np
from numpy.fft import fft2, ifft2

def psf2otf(psf, shape):
    """
    将点扩散函数 (PSF) 转换为光学传递函数 (OTF)。
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_shape = psf.shape
    psf_padded = np.pad(psf, ((0, shape[0] - in_shape[0]), (0, shape[1] - in_shape[1])), mode='constant')

    for axis, axis_size in enumerate(in_shape):
        psf_padded = np.roll(psf_padded, -int(axis_size / 2), axis=axis)

    otf = fft2(psf_padded)
    return otf

def filter_events(events, S, omega=1.0):
    """
    根据公式 (24) 和 (25) 过滤事件，实现基于图像梯度的去噪。

    events: 形状为 (N, 4) 的 numpy 数组，列分别为 [x, y, t, p] (即 h, v, t, p)
    S: 当前估计的清晰图像
    omega: 梯度监督的控制水平，用于确定掩码带宽

    返回去噪后的事件列表
    """
    if len(events) == 0:
        return np.array([])

    h, w = S.shape
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    # 提取 S 的梯度
    grad_S_h = np.real(ifft2(fft2(S) * psf2otf(dx, S.shape)))
    grad_S_v = np.real(ifft2(fft2(S) * psf2otf(dy, S.shape)))

    # 获取梯度的绝对幅度
    grad_mag = np.sqrt(grad_S_h**2 + grad_S_v**2)

    # 找到 ∇xS 中出现次数最多的像素值 q (使用直方图统计非零梯度)
    # 为了避免被大面积的平坦区域(0)主导，我们只统计具有一定梯度的像素
    active_grads = grad_mag[grad_mag > 1e-4]
    if len(active_grads) == 0:
        # 如果图像完全平坦，则无信号可言，返回空
        return np.array([])

    hist, bin_edges = np.histogram(active_grads, bins=100)
    max_bin_idx = np.argmax(hist)
    # 取最常出现区间的中心作为 q
    q = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0

    # 构造掩码 g (对应公式 25)
    # 只有梯度在 [q - omega, q + omega] 范围内的区域被判定为有效边缘
    mask_g = (grad_mag >= (q - omega)) & (grad_mag <= (q + omega))

    # 执行事件过滤
    filtered_events = []

    # 提取坐标（这里假设 events 第一列是 x(宽)，第二列是 y(高)）
    x_coords = events[:, 0].astype(int)
    y_coords = events[:, 1].astype(int)

    # 边界保护
    valid_indices = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    valid_events = events[valid_indices]

    for ev in valid_events:
        x, y = int(ev[0]), int(ev[1])
        # 如果事件所在位置被图像梯度掩码选中，则保留
        if mask_g[y, x]:
            filtered_events.append(ev)

    return np.array(filtered_events)

def generate_event_prior(events, shape):
    """
    将离散事件累加为事件先验图像 I_tau (公式 5)。
    为了适应公式(19)，我们在这里生成水平和垂直方向上的事件计数或粗略梯度替代。

    events: 当前时间窗口内的事件 [x, y, t, p]
    shape: (h, w)
    """
    I_tau = np.zeros(shape)
    if len(events) > 0:
        x_coords = events[:, 0].astype(int)
        y_coords = events[:, 1].astype(int)
        polarities = events[:, 3]

        # 边界保护
        valid = (x_coords >= 0) & (x_coords < shape[1]) & (y_coords >= 0) & (y_coords < shape[0])

        np.add.at(I_tau, (y_coords[valid], x_coords[valid]), polarities[valid])

    # 为了简化，由于事件是受运动方向激发的，我们在实现中将 I_tau 在水平和垂直方向做一次差分
    # 从而近似得到 I_tau_h 和 I_tau_v
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    I_tau_h = np.real(ifft2(fft2(I_tau) * psf2otf(dx, shape)))
    I_tau_v = np.real(ifft2(fft2(I_tau) * psf2otf(dy, shape)))

    return I_tau_h, I_tau_v

def joint_reconstruction(B, raw_events, k_size, max_iter=5, alpha=0.24, beta=0.064, sigma=2.0, omega=1.0):
    """
    算法 1：图像去模糊与事件去噪的联合迭代重建 (Joint Reconstruction of the Blur-Free Image and Noise-Robust Events)

    B: 输入的模糊图像
    raw_events: 输入的原始（带噪声）事件数据 [N, 4] -> [x, y, t, p]
    k_size: 模糊核大小, e.g., (15, 15)

    返回:
    S: 恢复出的清晰图像
    k: 估计出的模糊核
    E_dot: 去噪后的事件
    """
    # 从 deblur_module 导入去模糊子函数
    from deblur_module import update_S, update_z, update_k

    # 1. 初始化
    S = np.copy(B)
    k = np.zeros(k_size)
    k[k_size[0]//2, k_size[1]//2] = 1.0

    # 初始化事件为原始事件
    E_dot = raw_events

    gamma_max = 1e5
    gamma = 2.0

    # 联合迭代
    for iteration in range(max_iter):

        # ----------- Event Denoising 阶段 -----------
        # 第0次迭代时，由于 S = B(模糊)，可能过滤效果不佳，但随着迭代进行会逐渐改善
        if iteration > 0:
            E_dot = filter_events(raw_events, S, omega)

        # ----------- Image Deblurring 阶段 -----------
        # 从 (可能已去噪的) 事件生成梯度的先验 I_tau
        I_tau_h, I_tau_v = generate_event_prior(E_dot, B.shape)

        # 1. 求解辅助变量 z (硬阈值去噪/稀疏化)
        z_h, z_v = update_z(S, beta, gamma)

        # 2. 在频域求解清晰图像 S
        S = update_S(B, k, I_tau_h, I_tau_v, z_h, z_v, alpha, gamma)

        # 3. 求解模糊核 k
        k = update_k(B, S, k_size, sigma)

        gamma = min(gamma * 2.0, gamma_max)

    return S, k, E_dot
