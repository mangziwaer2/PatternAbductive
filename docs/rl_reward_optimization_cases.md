# RL Reward 优化方案与代表性样例

当前测试权重：

```text
ckpt/Qwen2.5-0.5B/DBpedia50-default-8-11000-rl-text2text.pth
ckpt/Qwen2.5-0.5B/DBpedia50-default-8-11000-rl-text2text.pth.adapter
```

测试方式：

```bash
conda run -n patternabductive python testing.py \
  --checkpoint-path "ckpt\Qwen2.5-0.5B\DBpedia50-default-8-11000-rl-text2text.pth" \
  --checkpoint_contents rlmodel \
  --modelname Qwen2.5-0.5B \
  --stage stage2 \
  --obs "OBS [...]" \
  --max_action_steps 3 \
  --max_new_tokens 96 \
  --do_sample false \
  --device cpu \
  --use_peft \
  --disable_text_extra_tokens
```

## 0. Groundtruth 与 EM 说明

本文档里的样例不是都和数据集 groundtruth 做了 exact match。

当前 Stage3 RL 的主 reward 不使用 gold ACTION / gold DSL，而是使用：

```text
生成 DSL 在 KG 上执行得到的答案集合
vs
输入 OBS 对应的实体集合
```

也就是说，文档中“成功样例”的含义是：

```text
模型生成的 ACTION 能执行；
模型生成的 DSL 能执行；
DSL 执行结果能覆盖输入 OBS；
```

而不是：

```text
generated DSL == dataset gold DSL
```

这点很重要，因为同一个 OBS 可能存在多个合理解释，尤其是当前数据集会对同一组 OBS 采到多条不同 pattern / target。例如：

```text
OBS [BBC]
```

在数据集中能找到多条 target：

```text
AND(AND(PROJ([+company], ENT([Cranford (TV series)])), PROJ([+network], ENT([Bob the Builder]))), PROJ([+employer], ENT([Nick Robinson])))

AND(NOT(PROJ([+board], ENT([Peaches Golding]))), PROJ([+employer], ENT([Wynford Vaughan-Thomas])))
```

而模型测试时生成的是另一条可以解释 `BBC` 的 DSL：

```text
AND(PROJ([+company], ENT([Match of the Day])), PROJ([+company], ENT([The Trials of Oz])))
```

它不是 gold EM，但如果执行后答案是 `BBC`，在 RL reward 里仍然能得到高覆盖分。

再比如：

```text
OBS [Mobsters] [Slipstream (1989 film)] [The Hire]
```

数据集中至少存在一条 gold target：

```text
PROJ([-starring], ENT([F. Murray Abraham]))
```

模型生成：

```text
PROJ([-starring], ENT([F. Murray Abraham]))
```

这个样例可以认为是 gold EM 或近似 gold EM。

因此本文档里的案例分两类：

1. **EM/近似 EM 样例**：模型输出和某条 gold target 一致或非常接近。
2. **coverage 样例**：模型输出不一定等于 gold target，但能通过 KG 执行覆盖 OBS。

```text
这些样例展示了当前模型的 rollout 行为，其中部分样例与 gold target 一致；更多样例用于说明 RL reward 下的可执行性、OBS 覆盖能力和 ACTION 策略问题。
```

## 1. 当前训练状态

Stage2 SFT loss 已经明显下降：

```text
epoch 2: train loss 0.168483, valid loss 0.169873
epoch 3: train loss 0.150663, valid loss 0.161881
```

但 Stage2 自由生成日志仍然有重复标签问题：

```text
<ACTION> ACTION EXPAND ... </ACTION> 3 </ACTION> 3 </ACTION> ...
```

这说明 token-level loss 降低不等于完整生成稳定。模型已经学会了大部分局部格式，但停止、标签边界和多步策略还不稳定。

Stage3 RL 当前日志统计：

| 范围 | reward 平均 | jaccard 平均 | jaccard=0 | stopped_by=dsl | action_execution_error | unparseable |
|---|---:|---:|---:|---:|---:|---:|
| all 11209 | 3.7112 | 0.6685 | 2386/11209 | 11049 | 59 | 101 |
| last1000 | 4.1236 | 0.7641 | 101/1000 | 980 | 5 | 15 |
| last200 | 4.0873 | 0.7590 | 21/200 | 191 | 1 | 8 |

RL 后期确实比早期稳定，说明 reward 对最终 DSL 覆盖有作用。

但是 action 类型明显塌缩：

| 范围 | FIND_COMMON | FIND_ALTERNATIVE | FIND_EXCLUSION | EXPAND |
|---|---:|---:|---:|---:|
| all | 9440 | 794 | 973 | 938 |
| last1000 | 947 | 42 | 11 | 7 |
| last200 | 186 | 13 | 1 | 3 |

主要问题：模型发现 `FIND_COMMON -> DSL` 最容易拿高分，于是逐渐减少 `FIND_EXCLUSION / FIND_ALTERNATIVE / EXPAND`。

PPT 中可以简化表述为：

```text
当前 RL 已经让模型基本学会了 ACTION -> RESULT -> DSL 的闭环，但动作策略还比较单一。

从日志看，FIND_COMMON 占比越来越高，FIND_EXCLUSION、FIND_ALTERNATIVE 和 EXPAND 的使用明显减少；同时大多数 rollout 只执行 1 到 2 步 ACTION 就输出 DSL。这说明模型倾向于选择最容易获得高覆盖分的短路径，而不是主动进行多步探索。

当前 reward 由两部分构成：
1. DSL reward：主要看 DSL 是否可解析、可执行，以及执行结果和 OBS 的 jaccard/recall/precision。
2. ACTION reward：主要看 ACTION 是否可解析、是否能执行、是否围绕当前 OBS/RESULT 中的实体查询，以及是否查到了有效子图。

问题在于 DSL 覆盖分权重较高，而 ACTION 只作为弱辅助项；另外当前每多一次 ACTION 都有轻微惩罚，所以模型会更偏向“少查一步、尽快输出 DSL”。因此下一步需要提高 ACTION 质量、ACTION-DSL 一致性和 RESULT 证据使用率在 reward 中的权重。
```

## 2. 当前 reward 公式

当前总 reward 在 `utils/rl_rewards.py` 中计算：

```text
R_total
= R_logic
 + 0.2 * R_action_avg
 - 0.05 * num_actions
 - no_action_penalty
```

其中 DSL 主 reward 为：

```text
R_logic
= 0.5 * parse_success
 + 0.5 * execution_success
 + 2.0 * answer_jaccard
 + 1.0 * answer_recall
 + 0.5 * answer_precision
 + 0.2 * compactness
 - 0.1 * complexity_penalty
```

ACTION reward 为：

```text
R_action
= 0.2 * action_parse_success
 + 0.4 * action_execution_success
 + 0.2 * evidence_coverage
 + 0.4 * candidate_count_score
 + 0.4 * target_grounding
 + 0.1 * target_diversity
 - 0.5 * repeat_action_penalty
```

当前设计的优点：

- 不直接比较 gold ACTION，避免 RL 退化成 SFT。
- 不直接比较 gold DSL，允许多种合理解释。
- 主要奖励最终 DSL 是否能反推出 OBS。

当前设计的问题：

- `R_action_avg` 权重只有 0.2，动作策略影响太弱。
- `num_actions` 每一步固定扣 0.05，会鼓励尽快结束，削弱多步探索。
- `FIND_COMMON` 往往能快速得到较高 jaccard，导致 action collapse。
- 对 malformed action 的惩罚不够强，例如 `FIND_ALTERNATIVE TARGET TARGET EXPAND ...` 仍然可能被宽松 parser 执行。
- 对最终 DSL 是否真的利用了 RESULT 子图没有明确奖励。

## 3. 代表性样例

### 3.1 成功样例：common 后直接得到正确 DSL

输入：

```text
OBS [Mobsters] [Slipstream (1989 film)] [The Hire]
```

模型输出：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [Mobsters] [Slipstream (1989 film)] [The Hire] TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[F. Murray Abraham] --[-starring]--> [Mobsters]
[F. Murray Abraham] --[-starring]--> [Slipstream (1989 film)]
[F. Murray Abraham] --[-starring]--> [The Hire]
</RESULT>
```

最终 DSL：

```text
PROJ([-starring], ENT([F. Murray Abraham]))
```

说明：

- 这是当前模型最擅长的情况。
- `FIND_COMMON` 找到共同入边实体。
- DSL 能直接反推出全部 OBS。

### 3.2 成功但偏简单：单实体 common

输入：

```text
OBS [BBC]
```

模型输出：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [BBC] TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[Match of the Day] --[+company]--> [BBC]
[The Trials of Oz] --[+company]--> [BBC]
[The Death of Yugoslavia] --[+distributor]--> [BBC]
</RESULT>
```

最终 DSL：

```text
AND(PROJ([+company], ENT([Match of the Day])), PROJ([+company], ENT([The Trials of Oz])))
```

说明：

- 结果可执行，格式正确。
- 但这类单实体 OBS 很容易让模型学成“查一跳然后拼 DSL”。
- 对真实医学场景，单次 common 不一定足够，需要 reward 鼓励“证据不足时继续探索”。

### 3.3 多实体 common 成功

输入：

```text
OBS [Dartmouth College] [Royal Military College Saint-Jean] [University of Winnipeg]
```

模型输出：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [Dartmouth College] [Royal Military College Saint-Jean] [University of Winnipeg] TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[University of the Arctic] --[-affiliation]--> [Dartmouth College]
[University of the Arctic] --[-affiliation]--> [Royal Military College Saint-Jean]
[University of the Arctic] --[-affiliation]--> [University of Winnipeg]
</RESULT>
```

最终 DSL：

```text
PROJ([-affiliation], ENT([University of the Arctic]))
```

说明：

- 这是理想的 common 场景。
- 但日志里也出现过同一个 OBS 在采样 rollout 下生成错误标签：

```text
<ACTION>
PROJ([+almaMater], ENT([Cory Parker Sarnoff]))
</ACTION>
```

这说明模型的标签控制还不稳定，尤其是采样式 RL rollout 中更容易出错。

### 3.4 问题样例：exclusion/alternative 被 common 覆盖

输入：

```text
OBS [Columbus Blue Jackets] [Dizel Penza] [HC MVD] [HC TPS] [Montreal Canadiens] [Salavat Yulaev Ufa]
```

模型输出第 1 步：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [Columbus Blue Jackets] [Dizel Penza] [HC MVD] [HC TPS] [Montreal Canadiens] [Salavat Yulaev Ufa] TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[Pete Zingoni] --[+draftTeam]--> [Columbus Blue Jackets]
[Pete Zingoni] --[+formerTeam]--> [HC TPS]
[Roman Derlyuk] --[+formerTeam]--> [HC MVD]
[Roman Derlyuk] --[+team]--> [Salavat Yulaev Ufa]
[Andrew Cassels] --[+formerTeam]--> [Columbus Blue Jackets]
[Andrew Cassels] --[+draftTeam]--> [Montreal Canadiens]
[Andrew Cassels] --[+formerTeam]--> [Montreal Canadiens]
</RESULT>
```

模型输出第 2 步：

```text
<ACTION>
ACTION EXPAND TARGETS [Pete Zingoni] [Roman Derlyuk] DIRECTION backward TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[HC TPS] --[-formerTeam]--> [Pete Zingoni]
[Salavat Yulaev Ufa] --[-team]--> [Roman Derlyuk]
[Defenceman] --[-position]--> [Roman Derlyuk]
</RESULT>
```

最终 DSL：

```text
PROJ([+formerTeam], PROJ([-birthPlace], ENT([Soviet Union])))
```

说明：

- 模型没有优先选择 `FIND_EXCLUSION`，而是继续走 common/expand。
- DSL 结构是可解析的，但解释路径和输入 OBS 的整体关系不够稳定。
- 当前 reward 如果最终答案覆盖还不错，就不会强烈区分“动作是否更适合该逻辑结构”。

### 3.5 问题样例：alternative action 语法脏，但 parser 仍然执行

输入：

```text
OBS [Cherwell Valley Line] [Princes Risborough to Aylesbury Line] [Sylvia Morales] [Waterloo to Reading Line]
```

模型输出：

```text
<ACTION>
ACTION FIND_ALTERNATIVE TARGET TARGET EXPAND TARGET TARGET EXPAND TARGET TARGETS [Cherwell Valley Line] [Princes Risborough to Aylesbury Line] [Sylvia Morales] [Waterloo to Reading Line] TOP_K 3
</ACTION>
```

KG 返回：

```text
<RESULT>
[Phoenix, Arizona] --[-birthPlace]--> [Josh McDermitt]
[Buckinghamshire] --[-location]--> [Marlow Branch Line]
[Buckinghamshire] --[-deathPlace]--> [William Ramsay]
</RESULT>
```

最终 DSL：

```text
OR(PROJ([-location], ENT([Buckinghamshire])), PROJ([+starring], ENT([The Big Red One])))
```

说明：

- `TARGET TARGET EXPAND ...` 是明显噪声。
- 当前 parser 过于宽松，仍然从文本中提取到了 `TARGETS [...]` 并执行。
- reward 应该增加 strict action syntax 分数，否则模型会学到“只要大概能执行就行”。

### 3.6 问题样例：实体拼写幻觉导致 KG 调用失败

RL 日志 step 11141：

```text
OBS [Anomalepididae] [Leptotyphlops koppesi] [Men with Guns] [Silver City (2004 film)] [Typhlops granti] [Typhlops lehneri] [Xenotyphlops grandidieri]
```

模型输出：

```text
<ACTION>
ACTION FIND_ALTERNATIVE TARGETS [Anomalepididae] [Leptotyphlops koppesi] [Men with Guns] [Silver City (2004 film)] [Typhlops granti] [Typhlops lehneri] [Xenhotyphlops grandidieri] TOP_K 3
</ACTION>
```

错误：

```text
'Xenhotyphlops grandidieri'
```

说明：

- 输入实体是 `Xenotyphlops grandidieri`。
- 模型生成成了 `Xenhotyphlops grandidieri`。
- 一个字符的实体幻觉会导致 KG action 失败。
- reward 需要明确惩罚 action target 不在 OBS、RESULT frontier 或 KG 中。

### 3.7 问题样例：DSL 被错误包进 ACTION 标签

RL 日志 step 11208：

```text
OBS [Bibbiena] [Marijampolė] [Soviet Union]
```

第 1 步：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [Bibbiena] [Marijampolė] [Soviet Union] TOP_K 3
</ACTION>
```

第 2 步错误输出：

```text
<ACTION>
AND(PROJ([+birthPlace], ENT([Aleksandr Aleksashenko])), PROJ([+birthPlace], ENT([Bilal Velija])))
</ACTION>
```

说明：

- 内容明显是 DSL，但标签是 `<ACTION>`。
- 当前 reward 对这类 tag-role mismatch 惩罚不足。
- 这也是 Stage2 日志中重复 `</ACTION>`、`</DSL>` 的同源问题：边界和角色还没学稳。

## 4. Reward 优化目标

下一版 reward 不应该把模型重新拉回 gold ACTION/GOLD DSL imitation，而是要强化三件事：

1. 输出必须是严格合法的工具调用或 DSL。
2. ACTION 类型要和最终 DSL 结构、RESULT 证据之间一致。
3. 最终 DSL 必须能覆盖 OBS，但不能只靠 `FIND_COMMON` 投机。

## 5. 建议的新 reward 公式

建议改为：

```text
R_total
= R_logic
 + 0.5 * R_action_valid
 + 0.4 * R_trace_dsl_consistency
 + 0.3 * R_evidence_usage
 + 0.2 * R_step_control
 - P_bad_generation
```

### 5.1 R_logic：保留最终结果主导，但略调权重

```text
R_logic
= 0.7 * dsl_parse_success
 + 0.7 * dsl_execution_success
 + 2.0 * answer_jaccard
 + 0.8 * answer_recall
 + 0.6 * answer_precision
 + 0.2 * compactness
 - 0.1 * complexity_penalty
```

变化：

- 保留 `answer_jaccard` 为核心。
- 稍微降低 recall 权重，避免模型为了覆盖 OBS 返回过宽答案。
- 提高 parse/execution 权重，强化 DSL 合法性。

### 5.2 R_action_valid：严格动作合法性

```text
R_action_valid
= mean(
    0.4 * strict_action_parse
  + 0.4 * action_execution_success
  + 0.2 * target_grounding
)
```

新增 `strict_action_parse`：

```text
strict_action_parse = 1
当且仅当 ACTION 完全符合以下模板之一：

ACTION FIND_COMMON TARGETS [x] ... TOP_K k
ACTION FIND_EXCLUSION TARGETS [x] ... TOP_K k
ACTION FIND_ALTERNATIVE TARGETS [x] ... TOP_K k
ACTION EXPAND TARGETS [x] ... DIRECTION forward/backward TOP_K k
```

如果出现：

```text
ACTION FIND_ALTERNATIVE TARGET TARGET EXPAND ...
```

宽松 parser 可以继续用于执行，但 strict parse 必须给 0。

### 5.3 R_trace_dsl_consistency：动作类型和 DSL 结构一致

这项不使用 gold pattern，只比较模型自己的 ACTION 轨迹和最终 DSL。

```text
R_trace_dsl_consistency
= 0.4 * not_consistency
 + 0.3 * or_consistency
 + 0.3 * expand_consistency
```

规则：

```text
not_consistency = 1
如果 DSL 包含 NOT，且 rollout 中使用过 FIND_EXCLUSION。

not_consistency = 0.5
如果使用过 FIND_EXCLUSION，但 DSL 没有 NOT。

not_consistency = 0
如果 DSL 包含 NOT，但没有使用 FIND_EXCLUSION。
```

```text
or_consistency = 1
如果 DSL 包含 OR，且 rollout 中使用过 FIND_ALTERNATIVE。

or_consistency = 0.5
如果使用过 FIND_ALTERNATIVE，但 DSL 没有 OR。

or_consistency = 0
如果 DSL 包含 OR，但没有使用 FIND_ALTERNATIVE。
```

```text
expand_consistency = 1
如果使用 EXPAND 后，最终 DSL 中出现了 EXPAND RESULT 里的实体或关系。

expand_consistency = 0
如果 EXPAND 后最终 DSL 完全没有使用扩展得到的证据。
```

作用：

- 避免模型一律 `FIND_COMMON`。
- 鼓励 `NOT` 结构真的来自 exclusion evidence。
- 鼓励 `OR` 结构真的来自 alternative evidence。
- 鼓励 EXPAND 是有目的的，而不是无意义多查一步。

### 5.4 R_evidence_usage：最终 DSL 是否使用 RESULT 子图

定义：

```text
E = RESULT 中出现的实体和关系集合
D = DSL 中出现的实体和关系集合

evidence_usage = |E ∩ D| / max(|D|, 1)
```

如果 DSL 中大量实体/关系没有出现在 OBS 或 RESULT 中，说明模型在“凭空猜”：

```text
hallucinated_dsl_token_rate
= |D - E - OBS_entities| / max(|D|, 1)
```

建议：

```text
R_evidence_usage
= evidence_usage
 - 0.5 * hallucinated_dsl_token_rate
```

作用：

- 不要求 RESULT 完整解释 OBS。
- 但要求最终 DSL 至少和查到的子图有联系。
- 这符合真实场景：KG 子图可以不完整，但应该辅助推理。

### 5.5 R_step_control：控制动作步数，但不抑制必要探索

当前 `-0.05 * num_actions` 会让模型偏向一步结束。建议改成：

```text
R_step_control
= 0.2 * has_action
 + 0.2 * has_final_dsl
 - 0.05 * max(num_actions - 2, 0)
 - 0.2 * repeat_action_count
```

含义：

- 1 到 2 步 ACTION 不惩罚。
- 超过 2 步才轻微惩罚。
- 重复同一个 ACTION 明确扣分。

这样模型有空间做：

```text
FIND_COMMON -> EXPAND -> DSL
FIND_EXCLUSION -> DSL
FIND_ALTERNATIVE -> EXPAND -> DSL
```

### 5.6 P_bad_generation：明显坏输出强惩罚

建议加入：

```text
P_bad_generation
= 1.0 * malformed_action
 + 1.0 * action_execution_error
 + 0.8 * tag_role_mismatch
 + 0.5 * unknown_entity_count
 + 0.3 * repeated_close_tag_count
```

其中：

```text
malformed_action:
ACTION 不符合严格模板。

tag_role_mismatch:
<ACTION> 中包了 DSL，例如 <ACTION> AND(...) </ACTION>。
或 <DSL> 中包了 ACTION。

unknown_entity_count:
ACTION TARGETS 中出现不在 OBS、RESULT frontier、KG entity vocab 中的实体。

repeated_close_tag_count:
出现 </ACTION> 3 </ACTION> 3 这类重复。
```

## 6. 代码落点

建议新增：

```text
utils/trace_scoring.py
```

包含：

```python
def strict_parse_action(action_text) -> dict:
    ...

def extract_trace_entities_and_relations(history, dsl_text) -> dict:
    ...

def score_trace_dsl_consistency(history, dsl_text) -> dict:
    ...

def score_evidence_usage(history, dsl_text, observation_text, kg) -> dict:
    ...

def score_bad_generation(raw_generations, history, dsl_text, kg) -> dict:
    ...
```

然后修改：

```text
utils/rl_rewards.py
```

把 `score_rollout_trajectory()` 从当前：

```text
logic + 0.2 * action_avg - 0.05 * num_actions
```

改为：

```text
logic
+ 0.5 * action_valid
+ 0.4 * trace_dsl_consistency
+ 0.3 * evidence_usage
+ 0.2 * step_control
- bad_generation_penalty
```

同时在 RL 日志里增加字段：

```text
strict_action_parse_rate
malformed_action_rate
trace_dsl_consistency
evidence_usage
hallucinated_dsl_token_rate
unknown_entity_count
tag_role_mismatch
action_type_sequence
```

这些字段对 PPT 和调参都很有用。

## 7. 是否需要 Stage2 ckpt

当前文档里的测试使用的是 RL ckpt，可以说明“RL 后模型已经能跑完整 ACTION -> RESULT -> DSL 链路，但策略塌缩到 FIND_COMMON”。

如果要在 PPT 里展示“Stage2 SFT 到 Stage3 RL 的改进”，最好再下载 Stage2 ckpt 跑同一组 OBS，对比：

```text
同一个 OBS
Stage2 ckpt 输出什么
Stage3 RL ckpt 输出什么
```

预计对比点：

- Stage2 更容易重复 `</ACTION>`、`</DSL>`。
- Stage2 更容易只学 target 格式，不一定能在 rollout 中稳定停住。
- Stage3 RL 的 `stopped_by=dsl` 比例更高。
- Stage3 RL 仍有 action collapse，所以 reward 还需要继续优化。

## 8. PPT 可用结论

可以这样概括当前阶段：

```text
目前模型已经初步具备工具调用链路能力：能够从 OBS 生成 ACTION，调用 KG 得到 RESULT，并继续生成 DSL。

但由于当前 reward 主要奖励最终 DSL 的覆盖率，中间 ACTION 只作为弱辅助项，并且动作步数存在轻微惩罚，因此模型倾向于选择最容易获得高分的短路径：通常是 FIND_COMMON 后直接输出 DSL。这导致 FIND_EXCLUSION、FIND_ALTERNATIVE 和多步 EXPAND 使用不足。

下一步将把 reward 从“最终答案覆盖”扩展为“答案覆盖 + 工具调用合法性 + ACTION-DSL 一致性 + RESULT 证据使用率 + 合理步数控制”，以鼓励模型进行更合理的多步探索，而不是只用单一动作快速猜答案。
```
