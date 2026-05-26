# Py Deblur — 事件相机联合去模糊

基于事件相机与模糊图像联合重建的去模糊框架，PyTorch/CUDA 实现。

## 快速开始

```bash
# 对所有数据集的所有帧进行去模糊优化
uv run multi_dataset_auto_tuner.py
```

## 工作原理

对整个数据集的**所有帧**进行全局联合优化，每轮外迭代包含两个阶段：

1. **所有帧 → 事件流降噪**：计算所有帧潜像的梯度，逐像素取最大值得到全局梯度活动图（只要某像素在任意一帧中有强梯度，就认为是真实边缘）。用全局梯度图筛选事件流，滤除噪点。
2. **事件流 → 所有帧去模糊**：用筛选后的事件流为每帧生成梯度先验，引导潜像和模糊核的频域更新。

相比逐帧优化，全局筛选利用帧间共识判断边缘真伪，避免了单帧过于模糊导致边缘被误滤除的问题。每轮迭代结束后 gamma 翻倍（逐步加强 L0 稀疏约束），共运行 `outer_iters` 轮。

## 目录结构

```
datasets/<name>/event.mat          # 事件数据（[x,y,t,p] 或 [t,x,y,p]）
datasets/<name>/frame/*.png        # 模糊帧（命名: <idx>_<timestamp>.png）
results/<name>/global_results/     # 数据集级输出：每帧结果图、best_params.txt、global_tuning_results.json
results/global_summary.md          # 所有数据集的全局汇总
dataset_config_example.json        # 调参配置文件
```

## 配置要点

搜参空间默认值（`dataset_config_example.json`）：

```json
{
  "tau_ratios": [0.5, 1.0, 1.5],
  "k_sizes": [13, 17, 21],
  "alphas": [0.12, 0.16, 0.24],
  "betas": [0.016, 0.032, 0.064],
  "omegas": [0.01, 0.02, 0.03],
  "outer_iters": [8, 10]
}
```

tau 按相邻帧间隔比例生成：`tau = ratio × Δt`，比固定绝对值更适合多数据集。

无参考评分公式：`mean_grad × 0.3 + lap_var × 0.4 + var × 0.2 + range × 0.1`

## 结果分析

跑完后先看 `results/global_summary.md`，每数据集一行：

- tau_ratio 普遍偏 0.5 还是 1.0？
- k_size 是否普遍选最小值（13）？
- 哪些数据集恢复后评分反而低于模糊原图？（说明算法对该场景不适用或指标失效）
- 不同 tau/omega 组合之间评分是否有区分度？

每个数据集的详细结果在 `results/<name>/global_results/` 下：
- `global_tuning_results.json` — 所有参数组合的完整结果
- `best_params.txt` — 最优参数
- `<dataset>_fXXXX_best.png` — 每帧最优结果的去模糊对比图

## 架构说明

`global_joint_reconstruction()` 是全局优化入口（`joint_reconstruction_cuda.py`），它与单帧版 `joint_reconstruction_cuda()` 的区别：

- 事件筛选用 `compute_global_gradient_activity(S_list)` 聚合所有帧梯度，而非只看当前帧
- 外迭代在所有帧的层面进行：先全局筛选事件流，再逐帧更新潜像和核
- 每帧有独立的潜像 `S_i` 和模糊核 `k_i`，但共享同一个全局梯度活动图

## 下一步

1. **视觉验证** — 自动评分不能替代人眼，优先看 `top_comparison.png`
2. **局部精调** — 选 2~3 个代表数据集，在 top 结果附近收窄搜索范围
3. **人工标注** — 建立 `manual_review.csv` 记录主观评级（A/B/C）和备注
4. **沉淀经验** — 统计"数据特征 → 推荐参数"规则

## 核心文件

| 文件 | 用途 |
|------|------|
| `deblur_cuda.py` | 去模糊核心算子（频域求解、L0 先验、模糊核估计） |
| `joint_reconstruction_cuda.py` | 联合重建主流程：`global_joint_reconstruction()` 全局优化 + `joint_reconstruction_cuda()` 单帧版 |
| `multi_dataset_auto_tuner.py` | 多数据集统一调参入口：`run_all_frames_global()` 数据集级全局优化 |
| `dataset_config_example.json` | 搜参配置 |
