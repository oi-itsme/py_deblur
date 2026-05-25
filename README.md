# Py Deblur — 事件相机联合去模糊

基于事件相机与模糊图像联合重建的去模糊框架，PyTorch/CUDA 实现。

## 快速开始

```bash
# 对所有数据集的所有帧进行去模糊优化
uv run multi_dataset_auto_tuner.py
```

## 工作原理

对数据集的**每一帧**独立运行联合去模糊，帧之间通过事件流时间窗口切片自然关联：

- **图像 → 事件**：当前恢复的潜在清晰图像梯度用于筛选事件（`filter_events_by_latent_gradient`），滤除噪声
- **事件 → 图像**：筛选后的事件累积为梯度先验，引导潜在图像和模糊核的频域更新

每帧运行多轮迭代（`outer_iters`），整个事件流按帧时间窗口 `[t - tau, t]` 切分后完整参与双向迭代，不留死角。

## 目录结构

```
datasets/<name>/event.mat          # 事件数据（[x,y,t,p] 或 [t,x,y,p]）
datasets/<name>/frame/*.png        # 模糊帧（命名: <idx>_<timestamp>.png）
results/<name>/frame_XXXX/         # 每帧输出：结果图、best_params.txt、tuning_results.json
results/global_summary.md          # 所有数据集所有帧的全局汇总
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

跑完后先看 `results/global_summary.md`，关注：

- tau_ratio 普遍偏 0.5 还是 1.0？
- k_size 是否普遍选最小值（13）？
- 哪些数据集恢复后评分反而低于模糊原图？（说明算法对该场景不适用或指标失效）
- 不同 tau/omega 组合之间评分是否有区分度？

## 下一步

1. **视觉验证** — 自动评分不能替代人眼，优先看 `top_comparison.png`
2. **局部精调** — 选 2~3 个代表数据集，在 top 结果附近收窄搜索范围
3. **人工标注** — 建立 `manual_review.csv` 记录主观评级（A/B/C）和备注
4. **沉淀经验** — 统计"数据特征 → 推荐参数"规则

## 核心文件

| 文件 | 用途 |
|------|------|
| `deblur_cuda.py` | 去模糊核心算子（频域求解、L0 先验、模糊核估计） |
| `joint_reconstruction_cuda.py` | 联合重建主流程（事件筛选 + 去模糊迭代） |
| `multi_dataset_auto_tuner.py` | 多数据集统一调参入口 |
| `dataset_config_example.json` | 搜参配置 |
