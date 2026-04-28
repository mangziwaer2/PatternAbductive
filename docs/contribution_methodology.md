# 贡献与方法论：面向 KG 溯因的三阶段 DSL 训练

## 1. 项目目标

当前项目只保留一个核心任务：让模型从 `OBS` 出发，主动调用 KG 工具取证，最后输出可执行 DSL。

推理时的目标流程是：

```text
OBS ...
-> ACTION_1
-> KG TOOL fills RESULT_1
-> ACTION_2
-> KG TOOL fills RESULT_2
-> ...
-> DSL
```

模型只生成 `ACTION` 和最终 `DSL`，不生成 `RESULT`。`RESULT` 由外部 KG 工具根据刚生成的 `ACTION` 实时填入。

最终输出不再附带自然语言解释，因为 DSL 本身就是解释载体：它可读、可解析、可执行，也能用于后续 RL reward。

## 2. 核心贡献

### 2.1 去掉 oracle condition

旧数据中的 condition / pattern 不适合作为真实输入，因为真实医学或开放 KG 溯因中，用户通常只能给出 observation，无法提前给出控制条件。

新设定中：

```text
输入只包含 OBS。
pattern 只作为训练监督，不作为模型输入。
```

这会逼迫模型从 observation 自己判断逻辑结构，而不是依赖人工给定的 pattern。

### 2.2 用 ACTION/RESULT 替代静态 KG hints

旧的 KG hints 是预先拼好的事实列表，模型无法决定自己要查什么。现在改为工具调用：

```text
ACTION FIND_COMMON_CAUSE TARGETS [...] TOP_K ...
RESULT ...
```

训练数据里的 `RESULT` 是采样阶段预先用 KG 工具填好的；推理时的 `RESULT` 是模型输出 `ACTION` 后由工具实时执行得到的。模型学习的是“当前证据历史下下一步该查什么，或者是否已经可以输出 DSL”。

### 2.3 三阶段训练路线

```text
Stage 1: OBS -> PATTERN + DSL
Stage 2: OBS + ACTION/RESULT history -> next ACTION or final DSL
Stage 3: RL with action reward + DSL execution reward
```

当前代码已经包含 Stage 3 的完整 rollout 入口：`scripts/run_stage3_rollout_train.py` 只读取 `OBS`，让模型自己输出 ACTION，系统调用 KG 填入 RESULT，循环后输出 DSL，再按完整轨迹打分更新模型。RL 不读取数据集中的 silver action trace 作为目标。

## 3. 数据集格式

采样脚本现在直接输出当前需要的 JSONL，不再需要二次转换脚本。

一条 JSONL 样本包含这些关键字段：

```text
pattern_str           原始结构 pattern
observation_text      原始 OBS 输入
logic_dsl             最终逻辑 DSL
stage2_trace          Stage 2 的压缩 ACTION/RESULT 轨迹
```

pattern 目标、Stage 1 目标和展开后的 prefix 样本不写入原始 JSONL，而是在 dataloader 中按需派生。这样可以避免正式数据集把同一段 OBS、ACTION、RESULT 重复存很多次。

小数据集采样命令示例：

```bash
conda run -n patternabductive python sampling.py ^
  --data_root ./sampled_data_preview/ ^
  --splits train ^
  --max-answer-size 8 ^
  --train-samples-per-pattern 1 ^
  --max-patterns-per-split 3 ^
  --condition-samples-per-query 0 ^
  --result-top-k 3 ^
  --flush-size 1 ^
  --checkpoint-frequency 1 ^
  --restart
```

正式采样时改大 `--train-samples-per-pattern`、`--valid-samples-per-pattern`、`--test-samples-per-pattern` 或去掉这些限制即可。

`--max-answer-size` 建议保持较小，例如 8。真实溯因输入通常是少量 observation，过多 observation 会让 ACTION 和 RESULT 历史迅速超过模型上下文。
`--result-top-k` 建议从 3 开始；如果模型上下文更长，再考虑调大。

如果已经有旧的 surface 数据集，可以不重新采样，直接转换：

```bash
conda run -n patternabductive python scripts/convert_surface_to_abduction.py ^
  --input-root ./sampled_data_surface/ ^
  --output-root ./sampled_data_abduction/ ^
  --splits train,valid,test ^
  --result-top-k 3 ^
  --max-observation-entities 8 ^
  --trace-mode lazy ^
  --overwrite
```

转换脚本会丢弃 `condition_text / condition_signature / kg_hints_text / hypothesis_text`，并默认按 `(pattern_str, observation_text, hypothesis_text)` 去重。`--max-observation-entities` 用于把旧 surface 数据里过长的 OBS 截成更适合工具调用训练的小 observation。`--trace-mode lazy` 只保存 compact row，适合先把完整 surface 数据集快速转成目标格式。

正式做 Stage 2 SFT 前，建议把 `stage2_trace` 预先补齐到一个 traced 数据目录：

```bash
conda run -n patternabductive python scripts/hydrate_stage2_traces.py ^
  --input-root ./sampled_data_abduction/ ^
  --output-root ./sampled_data_abduction_traced/ ^
  --splits train,valid,test ^
  --result-top-k 3 ^
  --trace-cache-size 100000 ^
  --overwrite
```

这样原始 JSONL 仍然只保存压缩轨迹，不保存展开后的 prefix 样本；但 Stage 2 dataloader 不需要在每次训练预处理时调用 KG，速度会明显更稳定。当前完整 DBpedia50 转换结果已经落在 `sampled_data_abduction_traced/DBpedia50`，三个 split 都有非空 `stage2_trace`。

为了区分“KG 证据子图不完整”和“gold DSL 与当前 KG split 不一致”，新增一个数据校验脚本：

```bash
conda run -n patternabductive python scripts/validate_abduction_dataset.py ^
  --data-root ./sampled_data_abduction_traced/ ^
  --splits train,valid,test ^
  --min-obs-recall 1.0 ^
  --diagnose-splits train,valid,test ^
  --overwrite
```

这里的 `OBS ⊆ execute(logic_dsl, split_KG)` 只检查 gold DSL 标签是否能在对应 KG split 上解释 observation。它不要求 `ACTION/RESULT` 能查出完整 hypothesis；ACTION 证据可以是不完整子图，模型仍然需要做溯因推断。

如果要给 Stage 3 RL 使用更干净的数据，可以写出过滤后的目录：

```bash
conda run -n patternabductive python scripts/validate_abduction_dataset.py ^
  --data-root ./sampled_data_abduction_traced/ ^
  --output-root ./sampled_data_abduction_checked/ ^
  --splits train,valid,test ^
  --min-obs-recall 1.0 ^
  --overwrite
```

## 4. Stage 1：逻辑能力 SFT

目标是让模型先学会 DSL 语法和 pattern 到 DSL 的关系。

训练样本：

```text
Source:
OBS [obs1] [obs2]

Target:
PATTERN AND(PROJ(ENT), PROJ(ENT)) DSL AND(PROJ([+r1], ENT([h1])), PROJ([+r2], ENT([h2])))
```

DSL 的方向约定：

- `PROJ([+r], ENT([h]))` 表示从解释实体 `h` 出发，沿 KG 中的正向关系 `+r` 投影到答案集合。
- `PROJ([-r], ENT([obs]))` 表示从观测实体 `obs` 出发，沿反向关系 `-r` 反查可能来源。
- 因此医学例子里如果 KG 边是 `流感 --+症状--> 发烧`，那么“流感解释发烧/咳嗽/乏力”的 DSL 应写成 `PROJ([+症状], ENT([流感]))`，而不是 `PROJ([+症状], ENT([发烧]))`。

训练命令：

```bash
conda run -n patternabductive python training.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --train_stage logic ^
  --dataset_num_proc 8 ^
  --dataloader_num_workers 4 ^
  --dataloader_pin_memory true
```

为什么要先做 Stage 1：

- 先让模型掌握 `AND / OR / NOT / PROJ / ENT` 这些结构。
- 降低 Stage 2 同时学习 action、result 上下文和 DSL 的难度。
- 让模型形成“OBS 对应某种逻辑结构”的先验。

## 5. Stage 2：工具调用轨迹 SFT

Stage 2 不是让模型一次性输出完整轨迹，而是把完整轨迹拆成多条 SFT 样本。

完整轨迹长这样：

```text
OBS ...
ACTION_1
RESULT_1
ACTION_2
RESULT_2
DSL
```

拆成训练样本后是：

```text
Source:
OBS ...

Target:
ACTION_1
```

```text
Source:
OBS ...
ACTION_1
RESULT_1

Target:
ACTION_2
```

```text
Source:
OBS ...
ACTION_1
RESULT_1
ACTION_2
RESULT_2

Target:
DSL
```

这样仍然是一次 Stage 2 SFT，只是数据集内部把一条轨迹展开成多个 prefix-to-next-step 样本。

原始 JSONL 中不直接保存这些展开样本，而是保存更小的：

```text
stage2_trace:
  - action: ACTION_1
    result: RESULT_1
  - action: ACTION_2
    result: RESULT_2
```

dataloader 读取时再展开为上面的训练样本。

为什么不直接训练：

```text
Input:
OBS ...

Target:
ACTION_1
RESULT_1
ACTION_2
RESULT_2
DSL
```

主要原因有四个。

第一，推理时 `RESULT` 不是模型生成的，而是 KG 工具生成的。如果训练 target 里包含 `RESULT`，模型会学到“自己编 RESULT”，这和真实工具调用流程冲突。

第二，工具调用是一个闭环决策过程。模型在生成 `ACTION_2` 之前必须已经看到 `RESULT_1`。一次性输出整段轨迹时，`ACTION_2` 在 teacher forcing 中可以依赖目标序列里前面的 gold `RESULT_1`，但推理时模型并不会生成这个 gold `RESULT_1`，系统会暂停并插入工具返回的真实 `RESULT_1`。prefix-to-next-step 正好模拟这个过程。

第三，prefix 样本让每一步都有明确监督：

```text
当前上下文 -> 下一步 ACTION
当前上下文 -> 最终 DSL
```

模型学到的是“何时继续查、查什么、何时停止并输出 DSL”，而不是背完整轨迹。

第四，展开后的样本可以复用同一条轨迹的中间状态，训练信号更多；原始 JSONL 又只保存压缩的 `stage2_trace`，不会把数据集体积放大。

可以做一个消融实验：

```text
A. Stage 2 final target = DSL
B. Stage 2 final target = PATTERN + DSL
C. Stage 1 预热后，再做 B
```

其中 B 可以在最终 prefix 样本里补上 pattern，用于辅助 DSL 结构学习；C 是更稳的 curriculum，但不是理论上唯一可行的路线。

训练命令：

```bash
conda run -n patternabductive python training.py ^
  --data_root ./sampled_data_abduction_traced/ ^
  --resume_epoch <stage1_epoch> ^
  --train_stage stage2_loop ^
  --result_top_k 3 ^
  --dataset_num_proc 8 ^
  --dataloader_num_workers 4 ^
  --dataloader_pin_memory true
```

推理时，模型每次只输出一个 `ACTION` 或最终 `DSL`。当脚本检测到完整 `ACTION` 后，暂停生成，调用 KG，插入 `RESULT`，再继续下一轮。

## 6. ACTION 设计

当前 action 格式：

```text
ACTION <TYPE> TARGETS <ENTITIES> TOP_K <K>
```

例子：

```text
ACTION FIND_COMMON_CAUSE TARGETS [fever] [cough] TOP_K 10
```

字段作用：

- `TYPE`：当前要做的 KG 检索类型。
- `TARGETS`：当前要检索的一组实体，第一步来自 observation，后续步骤来自上一步 `RESULT` 的候选实体。
- `TOP_K`：返回候选数量。

每个 `ACTION` 只执行一跳检索。多跳不是通过 action 里暴露深度字段实现，而是通过多次一跳 `ACTION -> RESULT` 循环实现。

当前 action 类型：

```text
FIND_COMMON_CAUSE
FIND_ALTERNATIVE_CAUSES
FIND_NEGATIVE_EVIDENCE
```

## 7. Silver ACTION 构造

现有数据没有人工 action 标注，所以从 `pattern_str + observation_text` 自动构造 silver action。

规则：

```text
包含 n       -> FIND_NEGATIVE_EVIDENCE
包含 u       -> FIND_ALTERNATIVE_CAUSES
其他情况     -> FIND_COMMON_CAUSE
```

pattern 不再写入 action，但仍用于构造 silver 轨迹长度。例如：

```text
pattern: (i,(p,(e)),(p,(p,(e))))
step 1 : ACTION FIND_COMMON_CAUSE TARGETS [obs] TOP_K 10
step 2 : ACTION FIND_COMMON_CAUSE TARGETS [candidate_1] [candidate_2] ... TOP_K 10
```

这样模型学到的是“每次只查一跳，根据新证据决定下一步”，而不是直接从 action 中读到 oracle 深度。

## 8. RESULT 设计

`RESULT` 是结构化证据，不是新的 observation，也不是模型要生成的目标。

一跳证据：

```text
RESULT
CANDIDATE [influenza]
SUPPORT [influenza] --[+has symptom]--> [fever]
SUPPORT [influenza] --[+has symptom]--> [cough]
COVERAGE 2/2
```

多跳证据：

```text
RESULT
CANDIDATE [x]
DEPTHS 2
PATH depth=2 [x] --[r1]--> [m] --[r2]--> [a]
COVERAGE 1/1
```

这些字段让模型看到候选解释、支持路径、缺失覆盖和证据深度，而不是一段松散自然语言。

## 9. Stage 3：RL 增强

Stage 3 的 rollout：

```text
OBS
-> ACTION
-> RESULT
-> ACTION or DSL
-> KG executes DSL
-> reward
```

reward 分两类：

```text
action_reward:
  action_parse_success
  action_execution_success
  evidence_coverage
  target_recall

logic_reward:
  dsl_parse_success
  dsl_execution_success
  answer_jaccard
  answer_overlap
  query_complexity_penalty
```

当前已把 action scoring、DSL execution/scoring 和完整 rollout 训练拆成独立工具。Stage 3 的主入口是 `scripts/run_stage3_rollout_train.py`，不是读取 Stage 2 silver trace 的 next-step GRPO。

当前代码中有两个层级：

```text
1. Rollout RL
   从 OBS 开始，不读取数据集 action trace。
   模型输出 ACTION 后由 KG 工具填 RESULT。
   模型继续输出 ACTION 或 DSL。
   最后用完整轨迹 reward 更新模型。

2. Rollout 评估
   从 OBS 开始，让模型输出 ACTION，系统调用 KG 填 RESULT，多步循环后得到 DSL，再计算最终 DSL 执行分数。
```

rollout RL 入口：

```bash
conda run -n patternabductive python scripts/run_stage3_rollout_train.py ^
  --data_root ./sampled_data_abduction/ ^
  --split train ^
  --scale <stage2_scale> ^
  --resume_epoch <stage2_epoch> ^
  --max-steps 100
```

rollout 评估入口：

```bash
conda run -n patternabductive python scripts/run_stage3_rollout_eval.py ^
  --data_root ./sampled_data_abduction/ ^
  --resume_epoch <stage2_or_rl_epoch> ^
  --scale <scale_name> ^
  --max-rows 20
```

`training.py --mode optimizing` 保留为旧的 next-step GRPO 实验入口，不作为当前主路线。当前主路线的 RL 是完整 rollout，因此不会把数据集中的 action 轨迹当作 target 去复刻。

## 10. 训练加速策略

当前项目的训练加速主要来自三层：

```text
数据层：预先 hydrate stage2_trace，避免 Stage 2 SFT 每次预处理都调用 KG。
预处理层：utils/dataloader.py 会把展开后的 HuggingFace dataset 保存到 dataset_cache/，同一 data_root、train_stage、top_k、max_rows 再跑会直接复用。
加载层：training.py 支持 dataset_num_proc、dataloader_num_workers、pin_memory、persistent_workers、prefetch_factor。
图加载层：当训练数据已经包含 logic_dsl/stage2_trace 时，training.py 会跳过 KG 加载；需要强制加载时使用 --force_load_kg。
```

云端正式训练时建议：

```bash
MODELNAME=Qwen2.5-0.5B \
DATA_ROOT=/path/to/sampled_data_abduction_traced/ \
BATCH_SIZE=4 \
USE_PEFT=1 \
LR=1e-4 \
MIXED_PRECISION=fp16 \
MAX_STAGE1_BATCHES=2000 \
MAX_STAGE2_BATCHES=3000 \
bash training_sft.sh
```

`stage2_trace` 只服务于 SFT 的 next-step 监督；Stage 3 RL rollout 不读取它作为目标。RL 仍然从 `OBS` 出发，让模型自己生成 ACTION，工具实时插入 RESULT，再根据完整轨迹打分。

LoRA 训练时脚本默认启用 `--disable_text_extra_tokens` 和 `--lora_modules_to_save none`，让 Qwen 直接用原 tokenizer 的普通文本 token 表示 `ACTION / DSL / PATTERN`，避免为了少量新增 token 训练整块 embedding/lm_head。

## 11. 当前代码入口

核心文件：

```text
sampling.py                     直接采样当前 JSONL 数据格式
training.py                     SFT 训练入口
scripts/hydrate_stage2_traces.py 预填 stage2_trace，加快 Stage 2 SFT
utils/text_dataset.py           构造 stage1 / stage2 字段
utils/dataloader.py             按 train_stage 展开训练样本
utils/logic_dsl.py              DSL 与旧 query 格式互转
utils/action_supervision.py     从 pattern 自动生成 silver ACTION
utils/evidence.py               从 KG 生成结构化 RESULT
utils/kg_actions.py             解析和执行 ACTION
utils/tool_loop.py              推理时 ACTION -> RESULT 接口
utils/execution.py              DSL 执行与评估
utils/action_scoring.py         ACTION 质量评分
scripts/run_stage2_tool_loop.py Stage 2 模型多步工具调用推理
scripts/run_stage3_rollout_train.py Stage 3 完整 rollout RL
scripts/run_stage3_rollout_eval.py  Stage 3 rollout 评估
```

训练参数只需要关心：

```text
--train_stage logic
--train_stage stage2_loop
--result_top_k
--data_root
--modelname
--config-model
--dataset_num_proc
--dataloader_num_workers
--dataloader_pin_memory
--mixed_precision
```

模型配置在：

```text
configs/config-model.yml
```

本地小实验默认使用：

```text
--modelname GPT2_6_act_nt
```

云端正式训练可以切换到配置文件里的长上下文/预训练 causal LM，例如：

```bash
conda run -n patternabductive python training.py ^
  --modelname Qwen2.5-0.5B ^
  --config-model configs/config-model.yml ^
  --data_root ./sampled_data_abduction_traced/ ^
  --train_stage stage2_loop ^
  --accelerate ^
  --mixed_precision bf16
```

切换模型时同时检查：

```text
configs/config-dataloader.yml 中的 text_obs_len / text_hyp_len
result_top_k
max-answer-size
显存是否能承受 batch_size 和上下文长度
```

本地 GPT2 适合验证链路，不适合最终多步 ACTION/RESULT 训练。正式训练更适合使用 4k/8k context 的预训练 causal LM，同时继续压缩 RESULT。

## 12. 实验对照

推荐至少保留这些实验：

```text
A. OBS -> DSL
B. OBS -> PATTERN + DSL
C. OBS + ACTION/RESULT history -> next ACTION or DSL
D. C + RL
E. OBS + oracle pattern/condition -> DSL
```

其中 E 只作为 upper bound，不作为真实任务设定。

核心指标：

```text
pattern_accuracy
action_parse_success
action_execution_success
evidence_coverage
dsl_parse_success
dsl_execution_success
answer_jaccard
answer_overlap
query_complexity
```

## 13. 与已有工作的边界

不能把这些单点作为创新：

```text
LLM + KG
工具调用
KG 执行反馈
逻辑查询生成
```

更准确的贡献表述是：

```text
本文将 KG 溯因中的逻辑假设生成建模为三阶段可执行 DSL 学习问题：
首先去掉 oracle condition，让模型从 observation 学习 pattern 和 DSL；
随后把 KG 取证建模为 ACTION/RESULT 工具调用轨迹，训练模型在 OBS 和已有证据历史下生成下一步 ACTION 或最终 DSL；
最后利用 KG 执行结果对 ACTION 和 DSL 进行强化优化。
```
