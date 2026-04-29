# 贡献与方法论

## 目标

当前项目只保留一条主线：让模型从 `OBS` 出发，主动调用 KG 工具获取证据，最后输出可解析、可执行的 DSL。

推理流程是：

```text
OBS ...
-> ACTION
-> KG TOOL fills RESULT
-> ACTION
-> KG TOOL fills RESULT
-> DSL
```

模型只生成 `ACTION` 和最终 `DSL`，不生成 `RESULT`。`RESULT` 由外部 KG 工具根据刚生成的 `ACTION` 实时执行并插入上下文。

## 核心贡献

1. 去掉 oracle condition 输入。

   真实溯因任务里，用户通常只能给出观测，例如症状、现象、异常事件，不能提前给出 pattern 或控制条件。因此训练和推理输入都应以 `OBS` 为核心，pattern 只作为采样和监督构造的中间信息。

2. 用工具调用轨迹替代静态 KG hints。

   旧的 `kg_hints` 是一次性给定事实，模型不能决定查什么。现在改成：

   ```text
   ACTION ...
   RESULT ...
   ```

   Stage 2 SFT 学的是“当前证据状态下下一步该查什么，或者是否已经能输出 DSL”。

3. DSL 是最终假设形式。

   DSL 不是自然语言解释，而是可执行逻辑表达式。它可以直接在 KG 上执行，得到答案集合，并用于 RL reward。

## 三阶段训练

```text
Stage 1: OBS -> PATTERN + DSL
Stage 2: OBS + ACTION/RESULT history -> next ACTION or final DSL
Stage 3: rollout RL from OBS with KG tool calls
```

Stage 1 是逻辑格式预热，不是最终任务。Stage 2 是主训练阶段。Stage 3 不读取数据集里的 silver ACTION 作为目标，而是让模型像推理时一样自己 rollout。

## ACTION 设计

当前 ACTION 不再暴露 oracle hop depth，而是用可迁移的一跳图搜索动作：

```text
ACTION FIND_COMMON TARGETS [...] TOP_K k
ACTION FIND_ALTERNATIVE TARGETS [...] TOP_K k
ACTION FIND_EXCLUSION TARGETS [...] TOP_K k
ACTION EXPAND TARGETS [...] DIRECTION backward|forward TOP_K k
ACTION CHECK_COVERAGE CANDIDATES [...] OBS [...] TOP_K k
```

含义：

```text
FIND_COMMON       从多个 OBS 反向找共同候选原因
FIND_ALTERNATIVE  找能解释不同 OBS 子集的替代候选
FIND_EXCLUSION    找部分覆盖或缺失覆盖的排除性证据
EXPAND            从已有候选继续扩展一跳
CHECK_COVERAGE    检查候选对原始 OBS 的覆盖情况
```

多跳不再写成 `MAX_HOPS 2`。两跳证据用两次一跳动作表达：

```text
ACTION FIND_COMMON TARGETS [obs1] [obs2] TOP_K 3
RESULT ...

ACTION EXPAND TARGETS [candidate1] [candidate2] DIRECTION backward TOP_K 3
RESULT ...
```

这样模型以后迁移到医学 KG 时，可以自己决定继续查、换方向查、做排除，或者停止输出 DSL。

## 采样方法

采样仍然从 KG 中的可执行 query 开始：

```text
pattern -> sampled query -> execute query -> OBS
```

然后把 sampled query 转成 gold `logic_dsl`，并从 `OBS` 自动生成 silver `stage2_trace`。

一条 JSONL 只保留核心字段：

```text
pattern_str
observation_text
logic_dsl
stage2_trace
```

Stage 2 dataloader 会把 `stage2_trace` 切成 prefix-to-next-step 训练样本：

```text
OBS -> ACTION1
OBS + ACTION1 + RESULT1 -> ACTION2
OBS + ACTION1 + RESULT1 + ACTION2 + RESULT2 -> DSL
```

## 例子

对于不均匀分支：

```text
(i,(p,(e)),(p,(p,(e))))
```

它包含一跳分支和两跳分支。新的轨迹不是简单写 `MAX_HOPS 2`，而是：

```text
OBS [o1] [o2]

ACTION FIND_COMMON TARGETS [o1] [o2] TOP_K 3
RESULT
CANDIDATE [a]
SUPPORT [a] --[r1]--> [o1]

ACTION EXPAND TARGETS [a] DIRECTION backward TOP_K 3
RESULT
CANDIDATE [b]
SUPPORT [b] --[r2]--> [a]

ACTION CHECK_COVERAGE CANDIDATES [a] [b] OBS [o1] [o2] TOP_K 3
RESULT ...

DSL AND(PROJ([r1], ENT([a])), PROJ([r1], PROJ([r2], ENT([b]))))
```

`AND` 本身表示交集，工具调用负责提供证据子图，模型负责把证据组织成 DSL。

## RL 路线

当前 Stage 3 统一从 `training.py` 进入：

```bash
python training.py \
  --mode optimizing \
  --train_stage stage2_loop \
  --resume_epoch <stage2_epoch> \
  --rl_max_steps 100
```

Rollout RL 的流程：

```text
输入 OBS
模型生成 ACTION 或 DSL
如果是 ACTION：执行 KG，插入 RESULT，继续生成
如果是 DSL：停止并打分
```

Reward 主要看：

```text
DSL 是否可解析、可执行
DSL 执行结果是否覆盖 OBS
ACTION 是否能执行、是否查到新证据、是否重复
轨迹是否过长
```

RL 不奖励“和数据集 action 一样”，否则会退化成 SFT。
