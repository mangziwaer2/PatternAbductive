# RL Reward Design

本文档记录当前项目中 Stage 3 RL 的打分方式。目标不是让模型复刻数据集里的 gold ACTION 或 gold DSL，而是让模型从 `OBS` 出发，自主生成 ACTION、调用 KG、得到 RESULT，并最终生成一个能够解释输入观测的 DSL。

## 1. RL 数据使用方式

RL 阶段只使用数据集中的 `observation_text`：

```text
OBS [a] [b] [c]
```

当前 RL 不使用：

- `stage2_trace` 中的 gold ACTION
- `logic_dsl` 中的 gold DSL
- `pattern_str` 中的 gold pattern

这样做的原因是：RL 的目的不是继续做 SFT，而是让模型探索合理 ACTION，并根据最终 DSL 的执行效果获得奖励。

一次 rollout 的流程是：

```text
OBS
-> 模型生成 <ACTION>...</ACTION>
-> 代码执行 KG 工具
-> 插入 <RESULT>...</RESULT>
-> 模型继续生成 ACTION 或 DSL
-> 生成 <DSL>...</DSL> 后停止并打分
```

如果达到 `--rl_max_action_steps` 仍未生成 DSL，会强制进入最后一次 DSL 生成尝试。

## 2. 总奖励公式

当前总奖励在 `utils/rl_rewards.py` 中计算：

```text
trajectory_reward
= logic_reward
  + 0.2 * action_reward_avg
  - 0.05 * num_actions
  - no_action_penalty
```

其中：

```text
no_action_penalty = 0.3, 如果模型没有生成任何可执行 ACTION
no_action_penalty = 0.0, 否则
```

解释：

- `logic_reward` 是最终 DSL 的主要奖励。
- `action_reward_avg` 是中间 ACTION 的平均合理性奖励，只作为辅助项。
- `num_actions` 惩罚过多工具调用，避免模型无限查 KG。
- `no_action_penalty` 惩罚完全不查 KG 就直接输出 DSL。

因此当前 RL 的重点仍然是最终 DSL 是否能解释 OBS，ACTION 只是辅助模型获得证据子图。

## 3. ACTION 奖励

ACTION 奖励在 `utils/action_scoring.py` 中计算：

```text
action_reward
= 0.2 * action_parse_success
 + 0.4 * action_execution_success
 + 0.2 * evidence_coverage
 + 0.4 * candidate_count_score
 + 0.4 * target_grounding
 + 0.1 * target_diversity
 - 0.5 * repeat_action_penalty
```

### 3.1 action_parse_success

```text
action_parse_success = 1, ACTION 能被解析
action_parse_success = 0, 否则
```

要求模型输出完整边界：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [a] [b] TOP_K 3
</ACTION>
```

这项奖励保证模型先学会合法工具调用格式。

### 3.2 action_execution_success

```text
action_execution_success = 1, ACTION 能成功执行 KG 查询
action_execution_success = 0, 否则
```

它不要求结果一定完美，只要求工具调用本身是 KG 可执行的。

### 3.3 evidence_coverage

```text
evidence_coverage = max_candidate(coverage_num / coverage_den)
```

它来自 KG 工具内部返回的候选统计，不会写进 `<RESULT>` 文本里。

注意：这项权重现在只有 `0.2`，是弱信号。原因是现实场景中 KG 子图通常不完整，不能强迫模型必须查到覆盖所有 OBS 的证据，否则模型会被误导去做无意义搜索。

### 3.4 candidate_count_score

```text
candidate_count_score = min(candidate_count, TOP_K) / TOP_K
```

作用是鼓励 ACTION 能查出一些候选子图，而不是空结果。

### 3.5 target_grounding

```text
target_grounding = |ACTION_TARGETS ∩ frontier_targets| / |ACTION_TARGETS|
```

`frontier_targets` 来自当前上下文：

- 初始阶段来自 OBS 实体；
- 后续阶段来自上一轮 RESULT 中的边起点。

这项奖励鼓励模型围绕当前证据前沿继续查，而不是突然查无关实体。

### 3.6 target_diversity

```text
target_diversity = unique(ACTION_TARGETS) / len(ACTION_TARGETS)
```

它防止模型重复写同一个 target。

### 3.7 repeat_action_penalty

```text
repeat_action_penalty = 1, 如果当前 ACTION 已在上下文中出现过
repeat_action_penalty = 0, 否则
```

这项用于减少重复工具调用。

## 4. DSL 奖励

DSL 奖励在 `utils/rl_rewards.py` 的 `score_logic_completion()` 中计算：

```text
logic_reward
= 0.5 * parse_success
 + 0.5 * execution_success
 + 2.0 * answer_jaccard
 + 1.0 * answer_recall
 + 0.5 * answer_precision
 + 0.2 * compactness
 - 0.1 * complexity_penalty
```

### 4.1 parse_success

```text
parse_success = 1, DSL 能被解析成内部 query
parse_success = 0, 否则
```

比如下面是合法 DSL：

```text
<DSL>
AND(PROJ([-symptom], ENT([fever])), PROJ([-symptom], ENT([cough])))
</DSL>
```

### 4.2 execution_success

```text
execution_success = 1, DSL 能在 KG 上执行
execution_success = 0, 否则
```

这项保证 DSL 不只是语法合法，还必须能落到 KG 查询。

### 4.3 answer_jaccard

设：

```text
P = DSL 执行得到的答案集合
G = 输入 OBS 对应的实体集合
```

则：

```text
answer_jaccard = |P ∩ G| / |P ∪ G|
```

这是当前最重要的结果奖励，权重为 `2.0`。

### 4.4 answer_recall

```text
answer_recall = |P ∩ G| / |G|
```

它鼓励 DSL 尽量覆盖输入 OBS。

### 4.5 answer_precision

```text
answer_precision = |P ∩ G| / |P|
```

它防止 DSL 返回过多无关实体。

### 4.6 compactness

```text
compactness
= execution_success
  * 1 / (1 + max(answer_count - len(G), 0) / max(len(G), 1))
```

如果 DSL 返回的答案比 OBS 多很多，compactness 会下降。

这项和 precision 类似，但更直接惩罚“答案集合膨胀”。

### 4.7 complexity_penalty

```text
complexity_penalty = min(query_complexity / 50, 1)
```

`query_complexity` 是 DSL 中逻辑节点数量，例如：

- `ENT`
- `PROJ`
- `AND`
- `OR`
- `NOT`

这项用于轻微惩罚过长、过复杂的 DSL。

## 5. 当前不使用 gold target 对比

当前 RL 不再使用模型输出和数据集 target 的直接比较。

也就是说，不做：

```text
generated DSL vs gold DSL
generated ACTION vs gold ACTION
generated pattern vs gold pattern
```

原因：

1. gold ACTION 只是 silver trace，不应限制模型探索。
2. gold DSL 只是某一种解释，真实溯因任务可能存在多个合理解释。
3. 直接比较 gold target 会把 RL 退化成另一种 SFT。

当前 target 只适合在 Stage1/Stage2 SFT 中使用，不适合进入 RL 主 reward。

## 6. pattern 与 DSL 结构一致性

当前 RL 不包含 pattern-DSL 一致性打分。

原因是 Stage2/RL 的最终输出设计是：

```text
<DSL>
...
</DSL>
```

模型在 RL 阶段并不输出 `PATTERN`，所以没有“模型自己输出的 pattern”和 DSL 可比。

如果未来把最终输出改成：

```text
PATTERN AND(PROJ(ENT), PROJ(ENT))
<DSL>
AND(PROJ([-symptom], ENT([fever])), PROJ([-symptom], ENT([cough])))
</DSL>
```

那么可以加入一致性奖励：

```text
pattern_dsl_consistency = 1, 如果 generated_pattern == skeleton(generated_dsl)
pattern_dsl_consistency = 0, 否则
```

其中 `skeleton(generated_dsl)` 是把 DSL 去掉具体关系和实体后的逻辑骨架：

```text
AND(PROJ([-symptom], ENT([fever])), PROJ([-symptom], ENT([cough])))
-> AND(PROJ(ENT), PROJ(ENT))
```

推荐公式可以是：

```text
logic_reward_new
= logic_reward
 + 0.3 * pattern_dsl_consistency
```

是否有必要：

- 如果最终输出只要 DSL：没有必要，因为 DSL 本身已经包含结构。
- 如果最终输出要求同时给 PATTERN 和 DSL：有必要，否则模型可能输出一个 pattern，却给出另一个结构的 DSL，内部逻辑不一致。

当前项目建议暂时不加。原因是现在三步走里 Stage2/RL 的目标是 ACTION + DSL，而不是 ACTION + PATTERN + DSL。先保证 ACTION 可执行、DSL 能覆盖 OBS，比额外输出 pattern 更重要。

## 7. smatch 是什么

`smatch` 原本常用于 AMR 图比较。它把两个结构图拆成三元组，然后找一个变量对齐，使三元组匹配 F1 尽可能高。

直观理解：

```text
smatch = 结构图之间的软匹配 F1
```

它适合比较：

- 两个语义图是否相似；
- 两个逻辑结构是否接近；
- 生成结构和 gold 结构的相似度。

## 8. smatch 还有没有必要

当前 RL 主 reward 不建议使用 smatch。

原因：

1. smatch 需要 gold target，对当前“自主探索 ACTION + 根据 OBS 反推 DSL”的目标不合适。
2. smatch 会奖励模型接近数据集答案，而不是奖励它找到其他合理解释。
3. 对真实医学 KG 场景，gold 解释可能不唯一，smatch 会过度约束模型。

建议：

- Stage1/Stage2 SFT 评估中可以保留 smatch 作为诊断指标。
- RL 主 reward 不使用 smatch。
- 论文实验可以报告 smatch，但不要让它决定 RL 更新方向。

## 9. 当前最需要观察的 RL 日志指标

训练时优先看：

```text
stopped_by
num_actions
action_parse_rate
action_execution_rate
action_reward_avg
jaccard
answer_recall
answer_precision
validity
compactness
complexity_penalty
stage3_reward
```

判断方式：

- `stopped_by=unparseable_generation` 很多：模型还没学会 ACTION/DSL 边界，Stage2 不足。
- `num_actions=0` 很多：模型倾向跳过工具调用。
- `action_parse_rate` 低：ACTION 格式没学好。
- `action_execution_rate` 低：ACTION 参数不合法或实体不在 KG。
- `jaccard/recall` 长期为 0：最终 DSL 没有覆盖 OBS。
- `precision` 低但 recall 高：DSL 返回太多无关实体。
- `complexity_penalty` 高：DSL 太复杂。

## 10. 小结

当前 RL reward 的核心是：

```text
合理 ACTION
+ 可解析、可执行 DSL
+ DSL 执行结果覆盖 OBS
- 无工具调用、重复工具调用、过长 DSL
```

它不是：

```text
生成结果和 gold target 做字符串/结构匹配
```

这更符合项目目标：让模型学会在不完全 KG 中主动查询证据，并用 DSL 表达可执行的溯因解释。
