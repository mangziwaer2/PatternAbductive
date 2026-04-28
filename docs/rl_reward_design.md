# RL Reward 设计

## 核心原则

Stage 3 的目标不是让模型继续模仿数据集里的 silver ACTION，而是让模型在 KG 环境里学会：

```text
当前证据不够 -> 查一个有用 ACTION
证据足够 -> 输出可执行 DSL
DSL 执行结果能够覆盖 OBS，且不要过宽
```

因此 RL reward 分成两层：

```text
trajectory_reward = final_logic_reward + small_action_utility_reward - action_cost
```

最终 DSL 是主奖励；ACTION 是辅助奖励，只用于鼓励有效探索。

## 1. 最终 DSL 奖励

最终 DSL 的 reward 不能主要依赖 gold DSL，否则模型只是在复刻数据集答案。当前代码采用环境式打分：

```text
pred_dsl -> KG execute -> pred_answer_set
OBS -> observation_answer_set
answer_jaccard(pred_answer_set, observation_answer_set)
```

主要字段：

```text
parse_success          DSL 能否解析
execution_success      DSL 能否在 KG 上执行
answer_jaccard         预测答案集合与 OBS 的 Jaccard
answer_recall          OBS 被预测答案集合覆盖的比例
answer_precision       预测答案中有多少是真正 OBS
compactness            惩罚过宽答案集合
complexity_penalty     惩罚过复杂 DSL
```

当前实现里的主 reward：

```text
0.5 * parse_success
+ 0.5 * execution_success
+ 2.0 * answer_jaccard
+ 1.0 * answer_recall
+ 0.5 * answer_precision
+ 0.2 * compactness
- 0.1 * complexity_penalty
```

`smatch` 和 `gold_pattern_match` 只作为诊断指标保留，不进入主 reward。这样可以观察模型是否接近数据集结构，但不会强迫模型复制 gold DSL。

## 2. ACTION 奖励

中间 ACTION 需要打分，但不能按“是否等于数据集 action”打分。

如果 reward 是：

```text
action_type_match(gold_action)
target_f1(gold_action)
```

那 RL 就退化成 SFT，因为模型只是被奖励复刻 silver action。

当前 ACTION reward 改为环境式 utility：

```text
action_parse_success       ACTION 格式是否合法
action_execution_success   工具能否执行
target_grounding           ACTION targets 是否来自当前 frontier
target_diversity           target 是否重复
candidate_count_score      是否查到了候选
evidence_coverage          候选证据对 targets 的覆盖率
repeat_action_penalty      是否重复调用同一个 ACTION
```

frontier 的定义：

```text
没有 RESULT 时：frontier = OBS 实体
已有 RESULT 时：frontier = 上一步 RESULT 里的 CANDIDATE 实体
```

因此模型可以自主探索新的 ACTION，只要它基于当前证据状态、能查出有覆盖的候选、并且不重复无效调用，就会得到辅助奖励。

## 3. 为什么不能只看最终 Jaccard

如果只给最终 DSL 的 Jaccard 分数，模型可能学会：

```text
OBS -> 直接输出记住的 DSL
```

这在训练集上可能得高分，但没有学习工具使用。解决办法有四个：

1. RL 主 reward 不使用 gold action，也不使用 gold DSL 文本匹配。
2. RL 数据应使用重新采样或 held-out OBS，不能只在 SFT 原样本上训练。
3. 对没有任何有效 ACTION 就直接输出 DSL 的轨迹，可以降低证据使用分。
4. 评估必须使用 unseen split 的 rollout，而不是只看训练集 reward。

当前代码已经做到 1 和 3。第 2 点要求 RL 训练时只读取 `OBS / logic_dsl` 这类任务数据，不读取数据集里的 `stage2_trace` 作为目标轨迹。

## 4. 当前代码状态

已实现：

```text
utils/action_scoring.py
  环境式 ACTION utility reward

utils/rl_rewards.py
  target-free DSL reward
  target-free ACTION reward
  rollout trajectory reward

scripts/run_stage3_rollout_eval.py
  使用 trajectory reward 评估真实 tool-loop rollout

scripts/run_stage3_rollout_train.py
  从 OBS 出发做完整 rollout RL：
  OBS -> model ACTION -> KG RESULT -> model ACTION/DSL -> trajectory reward
```

正式 RL 入口应使用：

```bash
python scripts/run_stage3_rollout_train.py \
  --data_root ./sampled_data_abduction/ \
  --split train \
  --modelname GPT2_6_act_nt \
  --scale <stage2_scale> \
  --resume_epoch <stage2_epoch>
```

`training.py --mode optimizing` 保留为旧的 prefix-to-next-step GRPO 实验入口，不作为当前方法的主 RL 路线。当前方法的主 RL 路线是完整 rollout，不需要读取数据集里的 action trace。
