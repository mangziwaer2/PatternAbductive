# RL Reward 设计

## 原则

Stage 3 的目标不是复刻数据集里的 silver ACTION，而是让模型在 KG 环境中学会：

```text
证据不足 -> 继续查 ACTION
证据足够 -> 输出 DSL
ACTION 无效 -> 避免重复或换策略
```

因此 reward 分为两部分：

```text
trajectory_reward = final_logic_reward + small_action_utility_reward - action_cost
```

最终 DSL 是主奖励，ACTION 只是辅助奖励。

## 最终 DSL 奖励

最终 DSL 不主要按 gold DSL 字符串匹配打分，而是按执行结果打分：

```text
pred_dsl -> KG execute -> pred_answer_set
OBS -> observation_answer_set
score(pred_answer_set, observation_answer_set)
```

当前主要指标：

```text
parse_success       DSL 是否能解析
execution_success   DSL 是否能执行
answer_jaccard      预测答案集合和 OBS 的 Jaccard
answer_recall       OBS 被预测答案覆盖的比例
answer_precision    预测答案中有多少属于 OBS
compactness         惩罚过宽答案集合
complexity_penalty  惩罚过复杂 DSL
```

当前实现中的 DSL reward：

```text
0.5 * parse_success
+ 0.5 * execution_success
+ 2.0 * answer_jaccard
+ 1.0 * answer_recall
+ 0.5 * answer_precision
+ 0.2 * compactness
- 0.1 * complexity_penalty
```

`smatch` 或 gold pattern match 只作为诊断指标，不作为主 reward。

## ACTION 奖励

ACTION 不按“是否等于 silver action”打分。否则 RL 就会退化成 SFT。

当前 ACTION utility reward 看：

```text
action_parse_success       ACTION 格式是否合法
action_execution_success   KG 工具能否执行
target_grounding           target 是否来自当前 frontier
target_diversity           target 是否重复
candidate_count_score      是否查到候选
evidence_coverage          候选证据对 targets 的覆盖率
repeat_action_penalty      是否重复调用同一个 ACTION
```

frontier 定义：

```text
没有 RESULT 时: frontier = OBS 实体
已有 RESULT 时: frontier = 最近 RESULT 里的 CANDIDATE 实体
```

这样模型可以探索和 silver trace 不同的 ACTION，只要它能查到有用证据并提升最终 DSL。

## Rollout 训练入口

当前只保留一个主训练入口：

```bash
python training.py \
  --mode optimizing \
  --train_stage stage2_loop \
  --resume_epoch <stage2_epoch> \
  --rl_max_steps 100
```

`training.py --mode optimizing --train_stage stage2_loop` 会执行完整 rollout：

```text
OBS
model -> ACTION
KG -> RESULT
model -> ACTION or DSL
...
reward
```

数据集里的 `stage2_trace` 不作为 RL target。RL 只读取 `OBS`，gold `logic_dsl` 只用于诊断。

## 为什么不能只看最终 Jaccard

如果只奖励最终 Jaccard，模型可能直接记忆训练集：

```text
OBS -> memorized DSL
```

所以当前 reward 加了：

```text
no_action_penalty
action utility reward
每步 action cost
repeat action penalty
```

这会鼓励模型在需要时调用 KG，但不会强制它照抄数据集里的 ACTION。

## 后续可增强点

1. 加入 novelty reward：奖励 ACTION 查到历史中没出现过的新候选。
2. 加入 branch coverage reward：奖励 RESULT 同时支持多个 OBS 分支。
3. 加入 medical KG 约束：例如症状、疾病、检查、药物等类型约束。
4. 对 held-out OBS 做 rollout eval，避免只看训练集 reward。
