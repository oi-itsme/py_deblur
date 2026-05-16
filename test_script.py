import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from joint_reconstruction import joint_reconstruction

# 1. 生成虚拟清晰图像 S_true (简单的几何形状)
size = 64
S_true = np.zeros((size, size))
S_true[20:44, 20:44] = 1.0  # 一个亮色正方形

# 2. 生成一维水平运动模糊核 k_true
k_size = 11
k_true = np.zeros((k_size, k_size))
k_true[k_size//2, :] = 1.0 / k_size  # 水平方向平均模糊

# 3. 生成模糊图像 B (带少量噪声)
B = convolve2d(S_true, k_true, mode='same', boundary='wrap')
B = B + np.random.normal(0, 0.01, B.shape)
B = np.clip(B, 0, 1)

# 4. 生成虚拟事件 raw_events (正方形在水平向右运动时触发的事件)
# 右边缘触发正极性(1)，左边缘触发负极性(-1)
events = []
# 假设运动距离为 10 个像素，在 t=0~10 的时间内
for t in range(10):
    shift_x = t
    # 右边缘: x = 44 + shift_x, y = 20 到 43
    for y in range(20, 44):
        events.append([44 + shift_x, y, t, 1])
    # 左边缘: x = 20 + shift_x, y = 20 到 43
    for y in range(20, 44):
        events.append([20 + shift_x, y, t, -1])

# 加入一些随机散粒噪声事件 (Background Activity noise)
for _ in range(100):
    rx = np.random.randint(0, size)
    ry = np.random.randint(0, size)
    rt = np.random.randint(0, 10)
    rp = np.random.choice([-1, 1])
    events.append([rx, ry, rt, rp])

raw_events = np.array(events)

# 5. 运行联合重建算法
print(f"原始事件数量: {len(raw_events)}")
S_est, k_est, E_denoised = joint_reconstruction(B, raw_events, k_size=(15, 15), max_iter=10)
print(f"去噪后事件数量: {len(E_denoised)}")

# 6. 可视化
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(S_true, cmap='gray')
axes[0].set_title('Ground Truth Image')

axes[1].imshow(B, cmap='gray')
axes[1].set_title('Blurry Image')

axes[2].imshow(S_est, cmap='gray')
axes[2].set_title('Recovered Image (S)')

axes[3].imshow(k_est, cmap='gray')
axes[3].set_title('Estimated Kernel (k)')

plt.tight_layout()
plt.savefig('test_result.png')
print("Test complete, image saved to test_result.png")
