# 创新路线与任务梳理

## 当前项目完成后的TODO
- 输出改为树结构
- 结合树形COT增加[ABDUCTIVE][\ABDUCTIVE]部分

## 一、当前项目基线

当前项目已经完成了 Gao 等论文中“可控知识图谱溯因假设生成”的核心闭环：

1. 从预定义逻辑 pattern 中采样 hypothesis。
2. 在 KG 上执行 hypothesis，得到 observation。
3. 构造带条件的训练样本：
   - `observation`
   - `condition_set`
   - `hypothesis`
4. 用生成模型学习从 observation 和 condition 预测 hypothesis。
5. 用执行结果和结构相似度做评测，并为后续 RL 留出接口。

当前代码里已经具备这些基础模块：

- 条件化采样：`sampling.py`、`utils/condition.py`
- 数据读取：`utils/load.py`、`utils/dataloader.py`
- 训练入口：`training.py`
- KG 查询执行：`kgclass.py`
- 样本预览：`preview_dataset.py`

## 二、当前阶段的明确目标

当前最应该做的，不是直接做开放关系生成，而是先把系统从：

- `ID -> ID`

升级成：

- `闭集 text -> text`

也就是：

- 输入不再是实体 ID 序列，而是实体名称文本
- 输出不再是关系/实体 ID 组成的逻辑式，而是带实体名和关系名的文本逻辑式
- 但底层关系和实体仍然来自当前 KG 的闭集词表
- 需要时仍然可以映射回 ID 执行和评测

这条路线更稳，也更适合在当前仓库基础上逐步落地。

## 三、当前已经形成的三点贡献

这里的“贡献”指当前项目在现阶段已经明确落地、可以对外表述的工作，而不是后续计划中的执行引导和冲突检测。

### 贡献 1：统一的多条件条件化数据构造框架

项目已经从原始的 `observation -> hypothesis` 样本，扩展为：

- `observation`
- `condition_set`
- `hypothesis`

并且支持统一的 condition schema，能够从一个采样得到的 hypothesis 中自动提取并组合以下条件：

- `pattern`
- `entity-number`
- `relation-number`
- `specific-entity`
- `specific-relation`

这意味着当前系统已经从“单一逻辑假设采样”升级为“多条件可控假设生成数据构造”。

### 贡献 2：闭集场景下的 text-to-text 假设生成表示

项目已经不再局限于纯 ID 输入输出，而是建立了闭集 text-to-text 表示：

- `observation_text`
- `hypothesis_text`
- `condition_text_textual`

其特点是：

- 模型训练时可以直接使用文本表示
- 文本中的实体和关系仍然来自当前 KG 的闭集
- 必要时仍可映射回 ID，用于执行、评测和后续 reward

这一步把项目从 `ID -> ID` 推进到了 `closed-set text -> text`，为后续更强的泛化与可解释性打下基础。

### 贡献 3：可运行的端到端实验基础设施

项目目前已经具备一条可运行的实验主链路：

- 条件化采样
- 文本化字段生成
- 数据预览工具
- 文本模式训练入口
- smoke test 验证

也就是说，当前不仅有方法设想，还已经有一条“可采样、可检查、可训练”的工程基础链路。

这使得后续两项更强的方法创新：

- 执行引导生成
- 冲突检测

能够直接建立在现有系统之上，而不是从零开始重写。

## 四、为什么先做闭集 text-to-text

### 1. 提升可读性

当前样本本质上是：

- observation：一组实体 ID
- hypothesis：一个带结构的 ID 逻辑式

虽然模型可以学，但人很难直接理解，调试和分析都不方便。

改成文本后，可以更直观看到：

- observation 对应哪些实体
- hypothesis 用了哪些关系
- 条件到底限制了什么

### 2. 提升泛化潜力

纯 ID 表示很容易让模型记住某个 KG 的编号分布。

文本表示至少让模型接触：

- 实体名字
- 关系名字
- 条件文本

这比纯数字表示更容易向跨图泛化、schema-aware 建模和开放关系扩展过渡。

### 3. 保留执行能力

闭集 text-to-text 的核心优势是：

- 表面上是文本生成
- 底层依然能映射回 KG 中已有的实体与关系

因此仍然可以：

- 执行 hypothesis
- 计算 observation 对齐 reward
- 做条件满足度评测

## 五、当前推荐的创新主线

中期主线仍然是：

1. 统一多条件控制
2. 执行引导生成
3. 冲突检测

但短期实现顺序应该改成：

1. 闭集 text-to-text 采样
2. 闭集 text-to-text SFT
3. 闭集 text-to-text 的执行映射
4. 在文本管线上接入执行引导
5. 在文本管线上接入冲突检测

## 六、执行引导生成

执行引导生成的核心思想是：

- 不是让模型先完整生成 hypothesis，再去 KG 上检查
- 而是在生成过程中，KG 执行器就参与进来

### 目标

- 保证生成前缀始终有机会满足条件
- 尽早剪掉已经不可能成立的分支
- 降低长逻辑 hypothesis 的无效率

### 可以分三层实现

#### 1. 语法约束

- 保证括号、操作符、投影结构合法

#### 2. 条件预算约束

- 关系数不能超预算
- 实体锚点数不能超预算
- 指定关系/实体如果还没出现，后续必须保留出现机会

#### 3. KG 执行约束

- 对已经闭合的子查询做局部执行
- 子查询为空或明显不合理时提前剪枝

### 最小可行版本

先做：

- 语法约束
- relation/entity budget 约束
- specific-relation / specific-entity 覆盖约束

暂时不把复杂的局部执行搜索做得太重。

## 七、冲突检测

冲突检测的核心作用是：

- 在生成前判断一组条件是否可能同时满足

### 典型冲突

- `pattern=(p,(e))` 但 `relation-number=3`
- 某个 pattern 至少需要 2 个 anchor entity，但条件要求 `entity-number=1`

### 输出可以分三类

- `feasible`
- `infeasible`
- `low-feasibility`

### 最小可行版本

先做规则型检查：

- pattern 和 relation-number 是否一致
- pattern 和 entity-number 是否一致
- 某些固定条件组合是否天然矛盾

后续再做 KG-aware feasibility：

- 指定 relation 在指定 entity 周围是否有局部可行结构
- 当前条件组合在 KG 上是否至少存在候选 hypothesis

## 八、开放关系生成不是当前第一阶段目标

需要明确：

当前阶段不是“真正 KG 外新关系生成”。

因为当前系统即使改成 text-to-text，仍然是闭集的：

- 关系名来自当前 KG
- 实体名来自当前 KG
- hypothesis 仍然要能映射回 ID 执行

所以当前阶段做的是：

- `闭集文本化`

而不是：

- `开放关系生成`

开放关系更适合作为下一阶段：

- 生成一个新的关系文本描述
- 同时给出由现有 KG 关系组成的可执行逻辑定义

## 九、当前代码方向

### 1. 采样层

采样阶段除了输出原始字段，还要输出文本字段：

- `observation_text`
- `hypothesis_text`
- `condition_text_textual`

其中：

- `observation_text`：实体名称构成的 observation 文本
- `hypothesis_text`：实体名/关系名构成的逻辑式文本
- `condition_text_textual`：文本化条件

### 2. 训练层

训练入口要支持两种表示：

- `id`
- `text`

当前阶段重点跑：

- `representation=text`

### 3. tokenizer 层

文本模式要优先用预训练 tokenizer，而不是自定义 ID 词表 tokenizer。

推荐做法：

- observation 和 hypothesis 用 GPT-2 tokenizer
- 额外加入条件相关 special tokens
- 结构符号继续保留文本形式

### 4. 评测层

虽然训练目标是 text-to-text，但闭集场景下后续仍然要支持：

- 文本 hypothesis -> ID hypothesis
- 在 KG 上执行
- 计算语义对齐和条件满足率

## 十、当前阶段的任务拆解

### 已完成或正在推进

- 条件化采样管线
- 样本预览脚本
- 条件化 SFT 基线可运行
- 训练入口初步支持切换表示模式

### 当前最重要的实现任务

1. 完成闭集文本化采样字段输出
2. 完成文本模式 dataloader
3. 完成文本 tokenizer 接入
4. 完成 text-to-text SFT 训练入口
5. 跑文本模式 smoke test

### 接下来再做

6. 文本 hypothesis 回映射到 ID
7. 文本模式下的执行评测
8. 文本模式下的执行引导生成
9. 文本模式下的冲突检测

## 十一、当前一句话总结

当前最合理的路线，是先把项目从“闭集 ID 到 ID 的 hypothesis 生成”升级成“闭集 text 到 text 的 hypothesis 生成”，在此基础上再继续做执行引导和冲突检测，而不是直接跳到开放关系生成。
