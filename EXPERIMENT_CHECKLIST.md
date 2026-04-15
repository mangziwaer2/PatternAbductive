# 实验结果保存清单

## 1. 每次实验必须保存的内容

### 1.1 基本信息

- 实验名称
- 日期时间
- git commit hash
- 分支名
- 使用环境名称
- GPU 型号

### 1.2 数据信息

- 数据集名称
- 采样参数
  - `train_base_samples`
  - `valid_base_samples`
  - `test_base_samples`
  - `condition_samples_per_query`
  - `max_condition_arity`
  - `max_answer_size`
- 最终各 split 的样本数

### 1.3 模型与训练配置

- 模型名称
- 是否使用预训练文本权重
- `batch_size`
- `epochs`
- `lr`
- `warmup_steps`
- 是否使用约束生成
- 约束版本说明

### 1.4 训练曲线

至少保存：

- 每个 epoch 的 `train_loss`
- 每个 epoch 的 `valid_loss`

最好同时保存：

- loss 曲线图
- 最优 epoch

### 1.5 样例对比

每次实验至少保存：

- `train` 样例若干条
- `valid` 样例若干条
- `test` 样例若干条

每条样例建议包含：

- 输入 observation
- 输入 condition
- ground truth hypothesis
- 模型输出 hypothesis

### 1.6 最终评测指标

正式实验建议保存：

- Jaccard
- Dice
- Overlap
- Smatch
- validity
- condition adherence

如果做有约束生成，还应单独保存：

- 结构合法率
- 条件满足率
- EOS 合法率

## 2. 当前阶段最重要的日志

对当前项目来说，正式实验至少要保留两类日志：

1. 无约束实验日志
2. 有约束实验日志

每类日志都要包含：

- 完整配置
- loss 变化
- 若干 train/valid/test 对比样例

## 3. 当前推荐的目录组织

建议所有正式实验结果都放到：

- `results/experiments/<experiment_name>/`

每个实验目录下至少有：

- `config.json`
- `train_valid_loss.csv`
- `samples.txt`
- `summary.md`

## 4. 当前阶段的结论性结果最少要记录什么

如果现在还在做方法探索，最少要记录：

- loss 是否下降
- 训练集上是否能生成合理 hypothesis
- 验证集上是否出现高频关系塌缩
- 有约束是否比无约束更合法

## 5. 一句话总结

正式写论文时，最怕的不是模型效果差，而是没有留下足够的可复现实验记录；所以每次实验都必须留下配置、曲线、样例和最终指标。
