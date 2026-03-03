# CLOP 诊断与下一步实施方案（2026-03-02）

## 一、结论先行（核心问题）

当前最核心、最限制上限的问题**不是数据数组格式错误**，而是：

1. **任务几何与监控口径错配**：
   - 你在做的是 dataset-level OOD 语义迁移（val group_id 与 train group_id 完全不重叠），
   - 但过去长期用 individual `val_acc` 来判断，容易误判“停滞”。

2. **过拟合与语义连续性冲突并存**：
   - v6.3 出现高训练拟合（train proto 很高）+ 有限泛化（val proto ~10%）。
   - whitening 后 val→train 最近邻语义相似度从 raw 的 ~0.99 降到 ~0.13（中位），
     说明语义邻域被显著打散，抑制了“跨 group 泛化桥梁”。

3. **评估拆分极难**：
   - `group_intersection = 0`（train/val 文本组完全不重叠），
   - 这本质上是“zero-overlap semantics transfer”，不是 class-ID 分类。

## 二、数据是否适配 CLOP？

### 2.1 数据格式与配对完整性：是健康的
来自 `results/clop_data_audit.json`：
- mapping 最大误差 `1.09e-06`，均值 `8.06e-09`。
- `text_group_ids` 范围与唯一数完整（0~1087，1088 组）。
- 无 singleton group（每组至少 12 cells）。

=> 结论：**格式和基础配对无明显工程错误**，不是“数组错位/错配”的问题。

### 2.2 数据任务定义：非常 OOD
- 80 datasets 中，验证 8 datasets。
- train_groups=969，val_groups=119，交集=0。

=> 结论：当前验证不是“同类不同样本”，而是“新文本组迁移”，难度本身很高。

## 三、训练动态为何会“停滞”

### 3.1 版本动态
- v6.3：best val_proto_acc ≈ 10.22%（epoch 32），强过拟合。
- v6.4：过度正则后欠拟合，best ≈ 5.65%。
- v6.4.1：恢复 capacity + 适中正则，best ≈ 10.45%（epoch 102），
  但 train/val proto gap 仍显著（约 3.5~5x 区间，取决于评估口径）。

### 3.2 温度动态
- 各版本 temperature 均快速触顶 20。
- 这说明模型持续需要更尖锐判别，常与“难负样本不足 + 过拟合驱动”共现。

### 3.3 关键机制矛盾
- raw 文本 embedding 中，val 文本与 train 文本语义近邻很高（maxcos 中位 ~0.99）；
- 但 whitened 后降到 ~0.13，语义平滑性被破坏。
- 你得到了一种“更去相关但更难迁移”的表示。

## 四、class-level 分类法是否足够？

**不够。**
在 `group_intersection=0` 条件下，纯 class-ID 分类评价不再是核心目标。应切到
“语义检索/排序 + 迁移稳健性”导向：

- 主指标：`val_proto_acc`, `val_proto_top5`, `val_proto_top10`
- 泛化差距：`train_proto_acc / val_proto_acc`
- 稳健性：跨 seed 的 best/mean/std（至少 3 seeds）
- 温度行为：触顶 epoch 与触顶后性能演化

## 五、你现在最该关注哪些指标（优先级）

1. **主北极星**：`val_proto_acc`（top-1）
2. **次北极星**：`val_proto_top5`, `val_proto_top10`（更稳定）
3. **健康度**：`gap = train_proto_acc / val_proto_acc`（目标 < 3）
4. **训练稳定性**：`val_loss` + `temperature`（是否早触顶）
5. **实验可信度**：3-seed 复现实验的 mean/std

不建议继续把 `val_acc`（individual）作为主判据。

## 六、下一步实施路径（可执行）

### Phase A（1-2 天，低风险高价值）
1. **双视角评估**：
   - 保留当前 dataset-level split（零重叠 OOD）；
   - 新增 semantic-level split（保证部分 group overlap）作为对照。
2. **Whitening 消融**（必须做）：
   - `none` / `zca_text_only` / `zca_both` 三组；
   - 固定 v6.4.1 其余超参。
3. **批次结构监控**：
   - 每 epoch 记录 groups/batch、hard negatives proxy。

### Phase B（2-4 天，结构优化）
1. **Group-aware sampler**：
   - 提高 batch 内近义 group 对比密度（而非随机抽样）。
2. **Hard-negative 采样**：
   - 基于 text embedding 近邻构建 in-batch hard negatives。
3. **温度策略**：
   - 限制触顶速度，或引入温度正则。

### Phase C（验证闭环）
1. 3 seeds 复现最优配置；
2. 输出 mean±std；
3. 再决定是否推进到 DiT 下游。

## 七、版本管理已实施（本次）

- 版本台账：`results/clop_experiment_registry.json`
- 归档目录：`models/checkpoints/archive/{v63,v64,v641}`
- 实验总览文档：`docs/CLOP_EXPERIMENTS.md`
- 数据配对审计：`results/clop_data_audit.json`

这套结构用于防止 checkpoint/history 被后续 run 覆盖，保证每次对比可追溯。
