# ACTION Design

本文档说明当前项目中的 ACTION 设计、作用、执行细节和伪代码。ACTION 的目标不是直接给最终答案，而是让模型在输出 DSL 前主动查询 KG 子图，把外部图证据插入上下文。

## 1. 基本格式

ACTION 使用显式边界：

```text
<ACTION>
ACTION FIND_COMMON TARGETS [a] [b] TOP_K 3
</ACTION>
```

代码只有在检测到完整 `</ACTION>` 后才会执行工具。执行结果以 compact RESULT 插入：

```text
<RESULT>
[x] --[+r]--> [a]
[x] --[+r]--> [b]
</RESULT>
```

当前支持四种 ACTION：

```text
FIND_COMMON
FIND_ALTERNATIVE
FIND_EXCLUSION
EXPAND
```

字段含义：

- `TARGETS`：要查询的实体，必须是 `[entity]` 格式。
- `TOP_K`：最多随机返回多少条候选证据边。这里的 `K` 是预算，不表示语义排序。
- `DIRECTION`：仅 `EXPAND` 使用，当前主要使用 `backward`。

## 2. 执行入口

相关代码：

- `utils/action_supervision.py`：生成 silver ACTION、定义 ACTION 类型。
- `utils/kg_actions.py`：解析 ACTION 并执行。
- `utils/evidence.py`：在 KG 中构造证据子图并渲染 RESULT。
- `utils/text_dataset.py`：采样时构造 Stage2 trace。
- `training.py`：RL rollout 中检测 ACTION、调用 KG、插入 RESULT。

整体流程：

```text
模型生成 <ACTION>...</ACTION>
-> parse_action_text()
-> execute_action()
-> build_*_evidence()
-> render_evidence_package()
-> 插入 <RESULT>...</RESULT>
```

## 3. ACTION 解析

输入：

```text
<ACTION>
ACTION EXPAND TARGETS [x] [y] DIRECTION backward TOP_K 3
</ACTION>
```

解析结果：

```python
{
    "action_type": "EXPAND",
    "targets": ["[x]", "[y]"],
    "direction": "backward",
    "top_k": 3,
}
```

伪代码：

```python
def parse_action_text(text):
    body = strip_between_action_tags(text)
    tokens = tokenize_surface_text(body)

    assert tokens[0] == "ACTION"
    action_type = normalize_action_type(tokens[1])

    action = {
        "action_type": action_type,
        "targets": [],
        "direction": "backward",
        "top_k": 10,
    }

    scan tokens:
        if token == "TARGETS":
            read entity tokens until DIRECTION or TOP_K
        if token == "DIRECTION":
            read next token
        if token == "TOP_K":
            read next integer

    return action
```

## 4. FIND_COMMON

### 4.1 语义

`FIND_COMMON` 用来找能够共同解释多个 OBS 的候选实体。

例子：

```text
OBS [fever] [cough]
```

如果 KG 中有：

```text
[flu] --[symptom]--> [fever]
[flu] --[symptom]--> [cough]
```

那么：

```text
ACTION FIND_COMMON TARGETS [fever] [cough] TOP_K 3
```

可能得到：

```text
<RESULT>
[flu] --[symptom]--> [fever]
[flu] --[symptom]--> [cough]
</RESULT>
```

模型之后可以据此输出：

```text
<DSL>
AND(PROJ([-symptom], ENT([flu])), PROJ([-symptom], ENT([flu])))
</DSL>
```

### 4.2 当前实现

当前 `FIND_COMMON` 查每个 target 的入边：

```text
candidate --relation--> target
```

然后把相同 candidate 聚合起来。若存在覆盖多个 target 的 candidate，会优先在这些 candidate 中随机抽样；否则在全部 candidate 中随机抽样。这里不按实体 id 或固定顺序取 top k，因为 KG 边本身没有天然排名。

伪代码：

```python
def build_common_cause_evidence(observation_text, kg, top_k):
    obs_ids = observation_text_to_answer_ids(observation_text)
    support_by_candidate = {}

    for obs_id in obs_ids:
        for source_id, _, relation_id in sampler.in_edges(obs_id):
            if source_id == obs_id:
                continue
            support_by_candidate[source_id].append(
                edge(source_id, relation_id, obs_id)
            )

    candidates = []
    for candidate_id, supports in support_by_candidate.items():
        supports = dedupe_and_drop_self_loops(supports)
        covered_obs = {edge.object_id for edge in supports}
        candidates.append({
            "entity_id": candidate_id,
            "support": supports,
            "coverage_num": len(covered_obs & obs_ids),
            "coverage_den": len(obs_ids),
        })

    if any candidate covers multiple targets:
        candidate_pool = multi_target_candidates
    else:
        candidate_pool = candidates

    return random_sample(candidate_pool, k)
```

### 4.3 局限

当前 `FIND_COMMON` 是一跳 common search。它不能直接发现：

```text
a <- x <- z -> b
```

这种多跳共同解释。多跳由后续 `EXPAND` 逐步完成。

## 5. FIND_ALTERNATIVE

### 5.1 语义

`FIND_ALTERNATIVE` 对应 `OR(...)` 类型结构。它用于寻找多个可能解释分支。

例子：

```text
OBS [a] [b] [c]
```

如果某些 candidate 覆盖 `[a, b]`，另一些 candidate 覆盖 `[c]`，模型可能最后输出：

```text
<DSL>
OR(PROJ(...), PROJ(...))
</DSL>
```

### 5.2 当前实现

当前 `FIND_ALTERNATIVE` 不再复用 `FIND_COMMON`。它先找 target 的所有入边源实体，随机抽一部分源实体，再从这些源实体出发做一跳 forward 展开，最后随机返回若干真实边。

返回边可以包含 `source -> target` 这种 grounding 边，也可以包含 `source -> other` 这种扩展边。这样做的意图是：alternative 不是继续找“共同解释 target 的实体”，而是补充这些候选解释实体的周边子图，让模型看到多个可能分支。

伪代码：

```python
def execute_find_alternative(targets, top_k):
    source_set = set()

    for target in targets:
        for edge in in_edges(target):
            # source --r--> target
            source_set.add(edge.source)

    sampled_sources = random_sample(source_set, top_k)

    edge_pool = []
    for source in sampled_sources:
        for edge in out_edges(source):
            # both source --r--> target and source --r--> other can be returned
            edge_pool.append(edge)

    return random_sample(edge_pool, top_k)
```

### 5.3 设计意图

它给模型更多“可能解释分支”，让模型在输出 DSL 时自行决定是否用 `OR` 组合。

### 5.4 局限

当前 RESULT 里没有显式标注“这是 alternative 分支”。模型通过 ACTION 类型和子图内容理解它。

## 6. FIND_EXCLUSION

### 6.1 语义

`FIND_EXCLUSION` 对应含 `NOT(...)` 的逻辑结构。当前它可以理解为：

```text
带有排除意图的子图探索动作
```

它不是严格的逻辑反证工具。严格 exclusion 需要正例、反例、对比集合，而当前项目还没有引入这些结构。

### 6.2 当前实现

当前 `FIND_EXCLUSION` 使用同关系对比补集。它不会把 `source --r--> target` 这种能直接解释 OBS 的边返回给模型，因为这类边放进 `NOT(...)` 会和否定语义冲突。

它的做法是：

```text
1. 先看 target 有哪些入边关系 r
2. 在 KG 中找同一关系 r 的其他边
3. 丢掉那些 source 也能通过 r 指向当前 target 的边
4. 在剩余边中随机返回 K 条
```

因此 RESULT 代表的是：

```text
这些关系事实成立
但这些 source 在同一关系下不指向当前 OBS
```

这更接近 `NOT(PROJ(...))` 的需求：给模型提供“同关系下的排除/对比证据”。

伪代码：

```python
def execute_find_exclusion(targets, top_k):
    target_set = set(targets)

    relation_set = set()
    for target in targets:
        for edge in in_edges(target):
            # source --r--> target
            relation_set.add(edge.relation)

    edge_pool = []
    for relation in relation_set:
        source_to_objects = collect_all_sources_for_relation(relation)
        excluded_sources = {
            source
            for source, objects in source_to_objects.items()
            if objects intersects target_set
        }

        for source, objects in source_to_objects.items():
            if source in excluded_sources:
                continue
            for obj in objects:
                if obj not in target_set:
                    edge_pool.append(source --relation--> obj)

    return random_sample(edge_pool, top_k)
```

### 6.3 怎么理解 exclusion

例如：

```text
<ACTION>
ACTION FIND_EXCLUSION TARGETS [Martine Carol] TOP_K 3
</ACTION>
<RESULT>
[Some Film] --[+starring]--> [Another Actor]
</RESULT>
```

这条 RESULT 本身仍然只是普通子图边。它是否进入 `NOT(...)`，由模型在生成 DSL 时结合 ACTION 类型判断。

换句话说：

```text
ACTION 类型表达查询意图
RESULT 只表达查到的子图事实
DSL 决定这些事实如何被组合为 AND/OR/NOT
```

### 6.4 局限

当前 exclusion 仍然不显式输出“缺失边”，因为 KG 的缺失不等于事实否定。它输出的是同关系对比子图：这些边成立，并且它们的 source 在同一关系下没有指向当前 OBS。

未来如果要严格化，可以改成：

```text
FIND_CONTRAST
```

并返回正负对比证据：

```text
<RESULT>
POS [x] --[r]--> [a]
NEG [y] --[r]--> [b]
</RESULT>
```

但这会增加 RESULT 格式复杂度和 token 成本。当前阶段先保持 compact 子图更合适。

## 7. EXPAND

### 7.1 语义

`EXPAND` 用来从上一轮 RESULT 中得到的 candidate 继续做一跳探索。

例如第一轮：

```text
<RESULT>
[flu] --[symptom]--> [fever]
[flu] --[symptom]--> [cough]
</RESULT>
```

提取 candidate：

```text
[flu]
```

第二轮：

```text
<ACTION>
ACTION EXPAND TARGETS [flu] DIRECTION backward TOP_K 3
</ACTION>
```

可以继续找：

```text
<RESULT>
[virus] --[cause]--> [flu]
[winter] --[season]--> [flu]
</RESULT>
```

### 7.2 当前实现

`EXPAND` 支持方向：

```text
DIRECTION backward
DIRECTION forward
```

当前采样主要使用 `backward`。

如果是 `backward`：

```text
source --relation--> target
```

把 `source` 当作 candidate。

如果是 `forward`：

```text
target --relation--> object
```

把 `object` 当作 candidate。

伪代码：

```python
def build_expand_evidence(target_text, direction, top_k):
    target_ids = observation_text_to_answer_ids(target_text)
    support_by_candidate = {}

    for target_id in target_ids:
        if direction == "forward":
            edges = sampler.out_edges(target_id)
            candidate = edge.object_id
        else:
            edges = sampler.in_edges(target_id)
            candidate = edge.subject_id

        for edge in edges:
            if edge.subject_id == edge.object_id:
                continue
            support_by_candidate[candidate].append(edge)

    candidates = []
    for candidate_id, supports in support_by_candidate.items():
        supports = dedupe_and_drop_self_loops(supports)
        candidates.append({
            "entity_id": candidate_id,
            "support": supports,
            "coverage_num": number_of_targets_touched,
            "coverage_den": len(target_ids),
        })

    sort by coverage
    return top_k candidates
```

## 8. RESULT 渲染

当前 RESULT 非常精简，只输出边：

```text
<RESULT>
[a] --[+r]--> [b]
[c] --[-s]--> [d]
</RESULT>
```

不会输出：

```text
MODE
CANDIDATE
SUPPORT
COVERAGE
MISSING
DEPTHS
```

这些统计信息只在代码内部用于排序和 reward，不暴露给模型。这样做是为了减少 token 长度，并避免模型被 coverage 字段误导。

伪代码：

```python
def render_evidence_package(evidence):
    edges = []
    for candidate in evidence.candidates:
        edges.extend(candidate.support)

    edges = dedupe_and_drop_self_loops(edges)

    lines = []
    for edge in edges:
        lines.append(
            f"{subject} --{relation}--> {object}"
        )

    return tag_result_text("\n".join(lines))
```

## 9. Stage2 trace 如何生成

采样阶段会把 `pattern_str + OBS` 转成 silver ACTION trace。

规则：

```text
如果 pattern 中有 n/NOT -> FIND_EXCLUSION
否则如果 pattern 中有 u/OR -> FIND_ALTERNATIVE
否则 -> FIND_COMMON
```

然后根据 pattern projection depth 决定最多做几步：

```text
(p,(e)) -> 1 步
(p,(p,(e))) -> 2 步
(i,(p,(e)),(p,(p,(e)))) -> 2 步
```

第一步总是对 OBS 做：

```text
FIND_COMMON / FIND_ALTERNATIVE / FIND_EXCLUSION
```

如果需要多步，就从 RESULT 边的 subject 中抽 candidate，再做：

```text
EXPAND
```

伪代码：

```python
def build_stage2_trace(pattern_str, observation_text):
    action_type = infer_action_type(pattern_str)
    targets = extract_entities(observation_text)
    steps = infer_max_projection_depth(pattern_str)

    trace = []

    action = render_action(action_type, targets)
    result = execute_action(action)
    if result has no edges:
        return []
    trace.append((action, result))

    targets = extract_candidate_targets(result)

    for step in range(2, steps + 1):
        if not targets:
            break
        action = render_action("EXPAND", targets, direction="backward")
        result = execute_action(action)
        if result has no edges:
            break
        trace.append((action, result))
        targets = extract_candidate_targets(result)

    return trace
```

## 10. RL 中 ACTION 如何工作

RL 不读取数据集里的 silver ACTION。它从 OBS 开始，让模型自己生成。

伪代码：

```python
history = []

for step in range(max_action_steps):
    prompt = OBS + history
    generated = stream_generate_until_tags(prompt)

    if contains complete ACTION:
        action = parse_action_text(generated)
        result = execute_action(action)
        history.append(action)
        history.append(result)
        continue

    if contains complete DSL:
        stop and score DSL

    else:
        stop as unparseable_generation

final attempt:
    ask model to generate DSL
```

底层是逐 token 生成；上层在完整 `</ACTION>` 或 `</DSL>` 出现后按段处理。

## 11. 当前设计取舍

当前 ACTION 设计故意保持简单：

- RESULT 只给子图，不给解释文字；
- 不暴露 coverage；
- 不要求 KG 查到完整解释；
- 多跳通过多次一跳 ACTION 完成；
- exclusion 返回同关系对比子图，不直接返回能解释 OBS 的正向入边。

这样更接近真实场景：KG 往往不完整，工具查询只提供辅助证据，最终仍然需要模型做溯因组合并输出 DSL。
