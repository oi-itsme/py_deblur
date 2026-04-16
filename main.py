import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==========================================
# 工具函数：计算图像的梯度（即找边缘）
# ==========================================
def grad_h(img):
    # 水平梯度：把图像向左平移一个像素相减
    return np.roll(img, -1, axis=1) - img

def grad_v(img):
    # 垂直梯度：把图像向上平移一个像素相减
    return np.roll(img, -1, axis=0) - img

# ==========================================
# 准备阶段：生成合成数据 (模拟相机的输入)
# ==========================================
print("正在生成模拟数据...")
# 1. 创建一个 64x64 的黑色画布，中间画一个白色方块（这是我们未知的完美清晰图 S）
S_true = np.zeros((64, 64))
S_true[20:44, 20:44] = 1.0 

# 2. 模拟手抖（水平向右滑动9个像素的运动模糊核 k）
k_true = np.zeros((64, 64))
k_true[0, 0:9] = 1/9  # 真实环境中这个核通常在中心，这里放在左上角是为了配合FFT的特性

# 3. 生成模糊图像 B (清晰图 * 模糊核)
B = np.fft.ifft2(np.fft.fft2(S_true) * np.fft.fft2(k_true)).real

# 4. 模拟事件相机捕获的累加图 I_tau (物体边缘 + 一些随机噪点)
# 真实的事件其实就是清晰图的梯度
I_h = grad_h(S_true) 
I_v = grad_v(S_true)
# 加上一些随机噪点（模拟相机的发热噪点）
I_h_noisy = I_h + np.random.normal(0, 0.2, I_h.shape)
I_v_noisy = I_v + np.random.normal(0, 0.2, I_v.shape)


# ==========================================
# 核心算法：Algorithm 1 (交替迭代去模糊)
# 假装我们不知道 S_true 和 k_true，我们只有 B 和 I_noisy
# ==========================================
print("开始执行 Algorithm 1 迭代...")

# 预先计算好一些在 FFT 里经常用到的常量 (比如梯度的频域表示)
# 相当于公式里的 F(partial_h) 和 F(partial_v)
D_h = np.fft.fft2(grad_h(np.zeros_like(B))) # 这是一个数学技巧，获取微分算子的频域
d_h_kernel = np.zeros_like(B); d_h_kernel[0,0]=-1; d_h_kernel[0,1]=1
D_h = np.fft.fft2(d_h_kernel)

d_v_kernel = np.zeros_like(B); d_v_kernel[0,0]=-1; d_v_kernel[1,0]=1
D_v = np.fft.fft2(d_v_kernel)

# 把 B 转到频率世界
F_B = np.fft.fft2(B)
# 把事件（含噪边缘）转到频率世界
F_Ih = np.fft.fft2(I_h_noisy)
F_Iv = np.fft.fft2(I_v_noisy)

# 初始化：瞎猜一个模糊核 k (假设就是一个点，没模糊)
k_est = np.zeros_like(B)
k_est[0, 0] = 1.0

# 算法参数
alpha = 0.5  # 事件边缘的权重 (告诉算法多大程度上相信事件数据)
sigma = 0.1  # 模糊核的平滑正则项
iters = 5    # 迭代次数 (论文里的 l_max)

for i in range(iters):
    # ---------------------------------------------------------
    # 步骤 1：固定 k，更新 S (对应论文公式 19 的超级简化版)
    # ---------------------------------------------------------
    F_k = np.fft.fft2(k_est)
    
    # 按照公式 19 计算分子和分母 (全都在频率世界做乘法和加法)
    # 分子： k的共轭 * F(B) + alpha * (水平梯度共轭 * F(水平事件) + 垂直梯度共轭 * F(垂直事件))
    numerator = np.conj(F_k) * F_B + alpha * (np.conj(D_h) * F_Ih + np.conj(D_v) * F_Iv)
    # 分母： |F(k)|^2 + alpha * (|F(水平梯度)|^2 + |F(垂直梯度)|^2)
    denominator = np.abs(F_k)**2 + alpha * (np.abs(D_h)**2 + np.abs(D_v)**2) + 1e-6 # 加1e-6防除零
    
    # 频率世界相除，然后 ifft2 送回现实世界！
    F_S = numerator / denominator
    S_est = np.fft.ifft2(F_S).real
    
    # --- 步骤 1.5：使用 S 去噪事件 ---
    # 实际论文中这里会用公式(24,25)，用算出来的 S 的边缘去过滤 I_h_noisy，
    # 为了精简代码让初学者跑通，本次测试跳过掩膜更新，只展示去模糊的核心威力。

    # ---------------------------------------------------------
    # 步骤 2：固定 S，更新 k (对应论文公式 23)
    # ---------------------------------------------------------
    # 求出当前估算的 S 的梯度 (频率世界)
    F_grad_S_h = D_h * F_S
    F_grad_S_v = D_v * F_S
    
    # 模糊图 B 的梯度 (频率世界)
    F_grad_B_h = D_h * F_B
    F_grad_B_v = D_v * F_B
    
    # 公式 23 分子
    num_k = np.conj(F_grad_S_h) * F_grad_B_h + np.conj(F_grad_S_v) * F_grad_B_v
    # 公式 23 分母
    den_k = np.abs(F_grad_S_h)**2 + np.abs(F_grad_S_v)**2 + sigma
    
    # 算出新的模糊核，并送回现实世界
    k_est = np.fft.ifft2(num_k / den_k).real
    
    # 物理限制：模糊核不能有负数，且总和为1 (去杂质)
    k_est[k_est < 0] = 0
    if np.sum(k_est) > 0:
        k_est = k_est / np.sum(k_est)
    
    print(f"完成第 {i+1}/{iters} 次迭代")

print("大功告成！正在出图...")

# ==========================================
# 绘图展示
# ==========================================
plt.figure(figsize=(15, 4))

plt.subplot(141)
plt.title("1. Blurred Image (Input B)")
plt.imshow(B, cmap='gray')
plt.axis('off')

plt.subplot(142)
plt.title("2. Event Prior (Input Events)")
plt.imshow(I_h_noisy, cmap='gray') # 这里只展示水平方向的事件
plt.axis('off')

plt.subplot(143)
plt.title("3. Recovered Sharp Image S")
plt.imshow(S_est, cmap='gray')
plt.axis('off')

plt.subplot(144)
plt.title("4. Estimated Blur Kernel k")
# 把核平移到中心方便查看
plt.imshow(np.fft.fftshift(k_est)[28:36, 28:36], cmap='hot') 
plt.axis('off')

plt.tight_layout()
plt.show()