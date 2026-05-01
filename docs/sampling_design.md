# Sampling Design

本文档说明当前项目如何按照 `pattern_str` 采样 observation-hypothesis pair，重点解释 `NOT` 是怎么来的。

## 1. 采样目标

采样不是直接从一条自然语言问题开始，而是从一个逻辑结构模板开始：

```text
pattern_str: (i,(n,(p,(e))),(p,(e)))
```

采样代码要把这个抽象结构实例化为一条具体 KG 查询：

```text
query: (i,
          (n,(p,([-r1]),(e,([a])))),
          (p,([-r2]),(e,([b]))))
```

然后执行这条查询，得到一组答案实体：

```text
answers: [x1, x2, x3]
```

最终写入数据集时：

```text
OBS [x1] [x2] [x3]
PATTERN ...
<DSL>
...
</DSL>
```

所以严格来说：

- `query / DSL` 是假设结构。
- `answers` 是该假设推出的一组观测实体。
- `observation_text` 来自 `answers`，不是直接来自最初随机选中的实体。

## 2. 整体流程

当前流程在 `sampling.py` 和 `utils/kgclass.py` 中实现：

```text
选择 pattern_str
-> sample_valid_query_given_pattern(pattern_str)
-> recur_sample_query_given_pattern_answer(pattern, tail_node)
-> 得到具体 query
-> search_answers_to_query(query)
-> 根据答案数量和 train/valid/test 约束筛掉不合格样本
-> answers 转 OBS
-> query 转 DSL
-> pattern + OBS 生成 stage2_trace
```

伪代码：

```python
while True:
    query = sample_query_by_pattern(pattern_str)
    train_answers = execute_query_on_train_graph(query)
    valid_answers = execute_query_on_valid_graph(query)
    test_answers = execute_query_on_test_graph(query)

    if answer_size_is_valid and split_constraint_is_valid:
        break

obs = render_answers_as_observation(answers)
dsl = render_query_as_dsl(query)
trace = build_stage2_trace(pattern_str, obs)
```

## 3. 正向结构怎么采

### ENTITY: `(e)`

`e` 是查询里的锚点实体。采样时会把当前节点写进去：

```text
(e) -> ENT([a])
```

### PROJECTION: `(p,(sub))`

`p` 表示沿关系投影。采样时不是随便选边，而是以当前目标答案节点 `tail_node` 为终点，找一条入边：

```text
head --[r]--> tail_node
```

然后递归采 `head` 对应的子查询，最终得到：

```text
PROJ([-r], sub_query)
```

这里关系会写成负号，是因为查询执行时从锚点实体沿正向关系找答案，而采样时是从答案反向找回锚点，所以存储时需要记录“从锚点到答案”的方向。

例子：

```text
[flu] --[symptom]--> [fever]
```

如果采样时当前答案是 `[fever]`，入边是 `[flu] -> [fever]`，那么 query 里会形成：

```text
PROJ([-symptom], ENT([flu]))
```

执行这条 query 时，可以从 `[flu]` 找回 `[fever]`。

### INTERSECTION: `(i, branch1, branch2, ...)`

`i` 是交集。采样时所有分支都围绕同一个 `tail_node` 构造，使这个节点倾向于同时满足多个分支：

```text
branch1 answers includes x
branch2 answers includes x
final answers = branch1 answers ∩ branch2 answers
```

这就是多条证据共同解释同一组 OBS 的来源。

### UNION: `(u, branch1, branch2, ...)`

`u` 是并集。采样时只要求随机一个分支围绕当前 `tail_node` 构造，其他分支可以从随机节点采：

```text
final answers = branch1 answers ∪ branch2 answers
```

所以 OR 样本中的 OBS 不一定都来自同一条分支。

## 4. NOT 是怎么来的

`NOT` 不是从 KG 中采一条“负边”，KG 里也没有专门的负关系。

当前代码对 `n` 的处理是：

```python
sub_query = sample_query_by_pattern(inner_pattern, tail_node)
query = NOT(sub_query)
```

真正执行 query 时：

```python
answers(NOT(Q)) = all_nodes - answers(Q)
```

也就是说，`NOT` 是集合运算，不是图上存在一条负向边。

## 5. NOT 样本例子

以 pattern 为例：

```text
(i,(n,(p,(e))),(p,(e)))
```

它对应的语义是：

```text
final_answers = answers(POSITIVE_BRANCH) ∩ (ALL_NODES - answers(NEGATED_BRANCH))
```

假设采到两个分支：

```text
NEGATED_BRANCH:
PROJ([-team], ENT([Club A]))

POSITIVE_BRANCH:
PROJ([-team], ENT([Club B]))
```

执行后得到：

```text
S_neg = 所有属于 Club A team 关系答案的实体
S_pos = 所有属于 Club B team 关系答案的实体

answers = S_pos - S_neg
```

最终 OBS 是：

```text
OBS [x1] [x2] ...
```

这些 OBS 满足：

```text
它们在 Club B 分支里
并且不在 Club A 分支里
```

所以 `NOT` 不是“从一个实体走出一条否定边”，而是“用一个正向子查询定义排除集合，然后从候选答案中排除掉这部分实体”。

### 5.1 按代码顺序走一遍

继续用这个 pattern：

```text
(i,(n,(p,(e))),(p,(e)))
```

先给一个小 KG：

```text
[Flu]  --[symptom]--> [fever]
[Flu]  --[symptom]--> [cough]
[Flu]  --[symptom]--> [fatigue]
[Allergy] --[symptom]--> [fever]
[Allergy] --[symptom]--> [rash]
[Cold] --[symptom]--> [cough]
[Cold] --[symptom]--> [runny nose]
```

这个 pattern 有两个分支：

```text
branch_1 = (n,(p,(e)))   # negative branch
branch_2 = (p,(e))       # positive branch
```

所谓 negative branch，就是被 `n` 包起来的那个分支。它内部仍然是普通正向查询，只是最终执行时会取补集：

```text
answers(n,Q) = all_nodes - answers(Q)
```

下面按代码采样流程走。

第一步，随机选一个候选答案实体：

```text
tail_node = [fever]
```

这里 `[fever]` 是采样器临时选中的“希望被 query 命中的答案”。它不一定最终留在答案集中；在这个例子里，后面的 `NOT` 会把 `[fever]` 排除掉，最终留下的是同一个正向候选集合里的 `[rash]`。

第二步，采 `i` 的第一个分支，也就是 negative branch：

```text
(n,(p,(e)))
```

代码先进入 `n`，但 `n` 本身不选边，它只是继续采里面的子查询：

```text
(p,(e))
```

采 `p` 时，会找一条指向 `[fever]` 的入边：

```text
[Flu] --[symptom]--> [fever]
```

于是内部正向子查询被采成：

```text
PROJ([-symptom], ENT([Flu]))
```

再被 `n` 包起来：

```text
NOT(PROJ([-symptom], ENT([Flu])))
```

这个分支执行时表示：

```text
不是 Flu 的 symptom 的所有实体
```

也就是：

```text
ALL_NODES - {fever, cough, fatigue}
```

第三步，采 `i` 的第二个分支，也就是 positive branch：

```text
(p,(e))
```

同样围绕 `[fever]` 找一条入边。为了让两个分支不同，代码会避免重复使用完全相同的子查询或来源节点。假设这次找到：

```text
[Allergy] --[symptom]--> [fever]
```

那么第二个分支可以采成：

```text
PROJ([-symptom], ENT([Allergy]))
```

第四步，得到完整 DSL：

```text
AND(
  NOT(PROJ([-symptom], ENT([Flu]))),
  PROJ([-symptom], ENT([Allergy]))
)
```

第五步，执行完整 query：

```text
S_neg_inner = answers(PROJ([-symptom], ENT([Flu])))
            = {fever, cough, fatigue}

S_neg = answers(NOT(PROJ([-symptom], ENT([Flu]))))
      = ALL_NODES - {fever, cough, fatigue}

S_pos = answers(PROJ([-symptom], ENT([Allergy])))
      = {fever, rash}

final_answers = S_neg ∩ S_pos
              = {rash}
```

所以最终写入数据集的 OBS 是：

```text
OBS [rash]
```

对应的监督目标是：

```text
PATTERN AND(NOT(PROJ(ENT)), PROJ(ENT))
<DSL>
AND(NOT(PROJ([-symptom], ENT([Flu]))), PROJ([-symptom], ENT([Allergy])))
</DSL>
```

这个例子说明了一个关键点：negative branch 不是“从 OBS 走出一条负边”，而是定义一个要排除的集合。最终 OBS 来自：

```text
正向候选集合 - 被 negative branch 排除的集合
```

### 5.2 从空集合开始看数据怎么变化

可以把这个 pattern 看成一个集合表达式：

```text
(i,(n,(p,(e_m))),(p,(e_a)))

= answers(p(e_a)) ∩ answers(n(p(e_m)))
= answers(p(e_a)) ∩ (ALL_NODES - answers(p(e_m)))
= answers(p(e_a)) - answers(p(e_m))
```

也就是说，它最终表达的是：

```text
属于 a 的一跳结果
但不属于 m 的一跳结果
```

给一个更贴近你描述的例子。假设 KG 里有：

```text
[a] --[r]--> [b]
[a] --[r]--> [c]
[a] --[r]--> [d]

[m] --[r]--> [b]
[m] --[r]--> [j]
[m] --[r]--> [k]
```

全集记为：

```text
V = {a, m, b, c, d, j, k, ...}
```

采样和执行可以按下面这样理解。

第一步，选择 pattern：

```text
(i,(n,(p,(e))),(p,(e)))
```

第二步，采样器随机选一个候选答案实体：

```text
tail_node = [b]
```

第三步，采 negative branch：

```text
(n,(p,(e)))
```

先采它内部的正向子查询 `(p,(e))`。因为当前 `tail_node` 是 `[b]`，所以代码会找一条指向 `[b]` 的入边：

```text
[m] --[r]--> [b]
```

于是填入：

```text
e = [m]
```

得到内部子查询：

```text
p(e_m) = PROJ([-r], ENT([m]))
```

执行内部子查询：

```text
answers(p(e_m)) = {b, j, k}
```

再套上 `n`：

```text
answers(n(p(e_m))) = V - {b, j, k}
```

所以当前 negative branch 的答案集合是：

```text
V - {b, j, k}
```

第四步，采 positive branch：

```text
(p,(e))
```

同样围绕 `[b]` 找另一条指向 `[b]` 的入边：

```text
[a] --[r]--> [b]
```

于是填入：

```text
e = [a]
```

得到 positive branch：

```text
p(e_a) = PROJ([-r], ENT([a]))
```

执行 positive branch：

```text
answers(p(e_a)) = {b, c, d}
```

第五步，执行最外层交集 `i`：

```text
answers(i(...))
= answers(n(p(e_m))) ∩ answers(p(e_a))
= (V - {b, j, k}) ∩ {b, c, d}
= {c, d}
```

所以最终得到的 OBS 是：

```text
OBS [c] [d]
```

完整 DSL 可以写成：

```text
AND(
  NOT(PROJ([-r], ENT([m]))),
  PROJ([-r], ENT([a]))
)
```

这个 DSL 对 `[c]` 和 `[d]` 成立，因为：

```text
[a] --[r]--> [c]
[a] --[r]--> [d]
```

并且：

```text
[m] 没有 --[r]--> [c]
[m] 没有 --[r]--> [d]
```

所以这里交集交的不是两个子图，而是两个分支各自执行后的“答案实体集合”：

```text
positive branch answer set = {b, c, d}
negative branch answer set = V - {b, j, k}

intersection = {c, d}
```

你说的这种情况：

```text
a 走一跳得到 {b, c, d}
m 走一跳得到 {j, k}
```

如果单看查询语义，那么：

```text
answers = {b, c, d} ∩ (V - {j, k})
        = {b, c, d}
```

这也是合法的逻辑结果。但当前采样代码在 `i` 下面会让两个分支围绕同一个 `tail_node` 构造，所以 negative branch 往往也会包含这个临时答案实体。这样做的目的，是让 `NOT` 分支真的对最终答案产生筛选作用，而不是变成一个几乎不起作用的补集条件。

实现上需要注意一个细节：如果两个 `i` 分支随机到了同一条入边，得到完全相同的子查询，或者来源实体相同，那么这次分支组合是无效的。当前代码不会立刻丢掉整个 `tail_node`，而是会在同一个分支上多试几次其他入边；只有多次尝试后仍然无法得到非重复分支，才放弃这次采样。

另外，这里没有“从补集中选 top k 个实体”这一步。采样阶段只有最终答案数量限制：

```text
max_answer_size
```

如果最终答案太多，才会从最终答案里随机截取一部分作为 OBS。`TOP_K` 是 Stage2 ACTION 返回 KG 子图证据时用的，和这里的 query 采样不是同一件事。

## 6. 和“从实体出发再反推 OBS”的关系

你的理解对纯正向 pattern 基本成立：

```text
随机选一个答案实体 x
-> 反向沿 KG 采几跳，得到锚点和关系
-> 形成 query / DSL
-> 再执行 query 得到多个 answers
-> answers 成为 OBS
```

但包含 `NOT` 时要补充一点：

```text
随机选中的 tail_node 主要用于构造一个有意义的子查询
最终 OBS 仍然要重新执行完整 query 才能得到
```

原因是 `NOT` 会把某个分支变成补集：

```text
NOT(Q) = 全部实体 - Q 的答案
```

所以最初用于递归采样的 `tail_node` 不一定会保留在最终答案里。代码会通过 `judge()` 和答案数量约束不断重采，直到完整 query 在当前 split 上有可用答案。

## 7. Stage2 Trace 与真实 Query 的关系

采样得到的 `query / DSL` 是监督目标；`stage2_trace` 是根据 `pattern_str + OBS` 自动生成的 silver 工具调用轨迹。

当前规则是：

```text
pattern 含 n -> 第一类 ACTION 用 FIND_EXCLUSION
pattern 含 u -> 第一类 ACTION 用 FIND_ALTERNATIVE
否则       -> 第一类 ACTION 用 FIND_COMMON
多跳深度   -> 后续用 EXPAND 继续扩展候选子图
```

注意：`stage2_trace` 不要求完全复原 gold query 的每一条边。它的目标是让模型学会：

```text
看到 OBS
-> 主动查 KG 子图
-> 根据子图辅助生成 DSL
```

这也符合真实医学 KG 场景：KG 子图通常不完整，工具调用提供的是证据，不是完整答案。

## 8. 当前采样的局限

当前 `NOT` 采样有几个需要注意的点：

- `NOT` 是全图补集，集合会很大，所以必须依赖 `max_answer_size` 和 split 约束过滤。
- 数据集中不会显式保存“负例集合”，只保存最终 OBS 和 DSL。
- `FIND_EXCLUSION` 生成的 RESULT 只是与 OBS 相关的子图，不等价于严格证明 `NOT`。
- 如果后续要做更强的否定推理，可以额外保存：
  - positive candidate set
  - excluded candidate set
  - final answer set
  - exclusion evidence edges

当前版本先保持最小可运行：用 pattern 生成可执行 DSL，用 DSL 产生 OBS，用 silver ACTION 训练模型查询辅助子图。
