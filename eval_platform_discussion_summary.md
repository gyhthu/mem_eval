# 群聊 Memory 评测平台：讨论与实现总结

> 更新时间：2026-08-18  
> 仓库：<https://github.com/gyhthu/mem_eval>

## 1. 平台目标

搭建一个可以在本机和公司内网运行的群聊 Memory 评测平台，用同一批问题横向比较不同
Memory 系统。平台覆盖完整端到端链路：

```text
历史群聊
  -> Memory 构建
  -> 按问题检索 TopK Memory
  -> Reader 根据 Memory 回答
  -> Judge 对照标准答案判分
  -> 汇总指标、Badcase 和 Leaderboard
```

当前页面包含：

- **题库**：查看问题、标准答案和 Oracle 证据消息。
- **启动评测**：选择 Memory 系统、TopK、Reader/Judge 和并发数。
- **结果与 Badcase**：查看某次运行的分类指标、检索结果和答案错误。
- **Leaderboard**：v1 / v2 题集分榜；各自比较完整 172 题、无运行错误的正式结果。

平台采用 Streamlit，默认地址为 <http://127.0.0.1:8501>。

## 2. 当前数据

正式评测默认读第二版题目：

```text
results/feishu_generator/v2/questions_aggregate.json
results/feishu_generator/v2/episodes/
```

第一版流水线产物已冻结在 `results/feishu_generator/v1/`。Leaderboard 按
`dataset.questions_path` 分成 **v1 题集（原榜）** 和 **v2 题集**，同一方法配置不会互相覆盖。
v2 复用 v1 对话，只重写出题口径和 Judge。

| 项目 | 数量 | 含义 |
|---|---:|---|
| Questions | 172 | 评测问题总数 |
| Episodes | 18 | 18 段独立群聊历史；演示时可简化称为“18 个群聊” |
| Messages | 284 | 18 个 episode 中的原始消息总数 |

技术上 `episode` 是一段完整、相互隔离的群聊轨迹，不一定等同真实飞书中的永久群。
当前数据没有显式 `group_id`，评测时暂时用 `episode_id` 作为群聊/记忆隔离边界。

问题文件：

```text
results/feishu_generator/v1/06_eval_batch_150_2026-08-16/questions_aggregate.json
```

Episode 文件：

```text
results/feishu_generator/v1/06_eval_batch_150_2026-08-16/episodes/
results/feishu_generator/v1/05_pilot_0004/episode_from_prototype_0004.json
```

### 2.1 八类问题

| 类型 | 数量 | 主要考察内容 |
|---|---:|---|
| `state_update` | 82 | 能否识别状态变化、纠错和最终决定，而不是回答旧状态 |
| `multi_hop` | 31 | 能否组合多条消息或多个事件得到完整答案 |
| `knowledge_absence` | 22 | 记录没有答案时能否拒答，而不是编造 |
| `time_commitment` | 16 | 截止时间、承诺人、延期和时间先后关系 |
| `source_credibility` | 14 | 用户确认、Bot 声称、工具证据冲突时相信哪个来源 |
| `bot_writeback_pollution` | 5 | 未确认的 Bot 输出写入 Memory 后是否造成自我强化 |
| `implicit_user_reference` | 1 | “我之前说的”能否绑定到提问用户 |
| `terminology_ambiguity` | 1 | 同一个“状态”等词在不同上下文中的真实含义 |

问题不是让 LLM 自由生成答案。标准答案、`oracle_paths` 和
`evidence_message_ids` 来自 episode 的结构化真值。

## 3. 统一评测口径

### 3.1 检索范围

每道问题只允许检索：

1. 与问题具有相同 `episode_id` 的消息或记忆；
2. 提问时间之前已经出现的信息；
3. 当前配置规定的 TopK 个证据单元。

因此 TopK 表示同 episode、时间可见候选集中的返回数量，不是从所有 284 条消息中直接取。
Top10 对消息较少的 episode 会比较宽松，所以 Top3 和 Top10 必须作为两个不同配置分榜比较。

当前没有显式 `group_id`。后续如果加入真实 `group_id`，应将检索过滤条件扩展为：

```text
group_id + episode/thread scope + query_time + 可选 user_id
```

这可能改善跨群污染、同名用户、时间更新类 Badcase，但不能保证所有题型都提升。

### 3.2 Reader 和 Judge

正式结果统一使用：

```text
Reader model: deepseek-v4-flash
Judge model:  deepseek-v4-flash
temperature:  0
thinking:     disabled
```

Reader 只能读取 Memory 上下文；TeamAgent 的 Reader 同时读取 L2 摘要和 TopK 原始消息。
Judge 比较 Reader 答案与标准答案的语义，最终输出 `Correct` 或 `Incorrect`。

### 3.3 Leaderboard 指标

| 指标 | 含义 |
|---|---|
| **Top K** | Memory 系统最多返回的证据单元数量；Top3、Top10 分开比较 |
| **Hit@K** | TopK 中是否至少包含一条 Oracle 证据，再对所有问题取平均 |
| **Recall@K** | 找回的 Oracle 证据数除以该题全部 Oracle 证据数，再取平均 |
| **MRR** | 第一条 Oracle 证据排名的倒数；正确证据越靠前越高 |
| **Answer Accuracy** | Judge 判为正确的答案数除以问题总数，是主要端到端指标 |

`Hit@K` 高不代表答案一定正确。多跳题可能只命中部分证据，Reader 也可能没有正确处理
否定、时间和最终状态。

Leaderboard 只保留适合横向比较的指标。`Retrieval Badcase`、`Answer Badcase` 和
`LLM Errors` 放在结果详情页，不作为榜单列。榜单增加独立的模型列和版本列，避免只看运行名
无法判断配置差异。

## 4. 已接入方法

### 4.1 BM25 RAG

- 直接对同 episode 的原始消息做中文字符 unigram/bigram BM25 检索。
- 查询包含 `query_user_id + question`，并过滤未来消息。
- 不使用 embedding，也不进行 Memory 抽取。
- 优点是实体名、日期、术语与原文重合时命中率高，Oracle 原消息 ID 也能直接对齐。

### 4.2 Mem0

- 按 episode 建立共享作用域，将消息提交给 Mem0。
- `user_id`、`run_id` 和 metadata 主要承担存储隔离、运行隔离和结果过滤作用。
- LLM 实际读取消息文本、同作用域最近消息和已有记忆；metadata 通常不会直接交给 LLM 理解。
- Mem0 返回的是 LLM 抽取/改写后的记忆，不一定是原始消息，因此需要映射回源消息计算 Oracle 指标。
- 实现比 BM25/TeamAgent 复杂，主要因为包含并发写入、LLM 抽取、向量库、缓存、恢复和来源追踪。

### 4.3 TeamAgent Memory（L2 + BGE-M3）

每道问题对应的提问时间之前，TeamAgent 使用 Distiller 把同 episode 历史蒸馏为 L2 群共享摘要：

```text
同 episode 历史消息 -> TeamAgent Distiller -> L2 摘要
                                       + BGE-M3 TopK 原始消息
                                       -> Reader
```

- L2 摘要按 `episode_id + checkpoint_time` 缓存。
- BGE-M3 只在同 episode、提问时间前的原始消息中检索。
- Reader 同时读取 L2 摘要和 TopK 原始消息。
- `run_teamagent_retrieval()` 只负责构建/读取 L2 并检索；回答发生在后续 Reader/Judge 阶段。

### 4.4 TeamAgent + BM25

为了区分 TeamAgent 的收益来自 L2 摘要还是检索模型，新增独立实验配置，不覆盖 BGE-M3 版本：

```text
相同 L2 摘要 + 相同候选消息 + 相同 Reader/Judge
只把原始消息排序器从 BGE-M3 替换为 BM25
```

该配置可以直接观察 `BGE-M3 vs BM25` 对 TeamAgent 的影响。

### 4.5 MindMemOS

- 通过独立 HTTP 服务构建和搜索 Memory。
- 当前使用 `vanilla/fast`，Memory LLM 为 `deepseek-v4-flash`。
- Qdrant 存储向量和记忆 payload；Neo4j 支撑图关系；Kafka 用于异步事件流，但当前本机
  `vanilla` 实验关闭了 Kafka。
- 本机 embedding 使用 Ollama `bge-m3`。
- 初版发现 MindMemOS 接收 `user_id/session_id`，但 vanilla 默认过滤器不会自动按 actor
  限定向量搜索，曾造成跨 episode 召回。正式实验已显式加入
  `user_id + session_id + app_id` filters，并使用新的 v3 namespace 隔离旧数据。
- 返回内容是 LLM 抽取/改写后的记忆，而不是原始消息；评测通过来源时间和本地 manifest
  映射回原消息。

### 4.6 EverOS（Hybrid + BGE-M3）

- 通过独立 HTTP 服务调用 `/add`、`/flush` 和 `/search`，Memory 抽取模型使用
  `deepseek-v4-flash`，embedding 使用 `bge-m3`，检索策略为 `hybrid`。
- 每个 `episode_id + query_time` 使用独立 project checkpoint，只写入提问时间前的消息，
  防止未来消息泄漏。
- checkpoint 内再按原始 `event_id` 分段 flush，避免一整段群聊被压缩成单一主题 Memory。
- 每个 checkpoint 使用一个虚拟群 owner；原作者、消息类型和 `message_id` 作为显式标记保留
  在消息文本中，使多用户群聊进入同一份共享 Memory。
- Reader 读取 EverOS 返回的 episode、summary、subject 和命中的 atomic facts。
- Oracle 来源优先通过 EverOS 返回的 `session_id` 映射到原始 event 消息，缺失时再尝试显式
  `message_id` 标记和保守文本对齐；
  因此 EverOS 的 Hit/Recall/MRR 属于归因后指标，Answer Accuracy 仍是主要比较指标。
- 本机推荐将 EverOS 服务启动在 `127.0.0.1:18001`，避免与常见的 8000 端口服务冲突；
  平台通过 `EVAL_EVEROS_BASE_URL` 指向该服务。

#### 4.6.1 实际适配流程

```text
问题的 episode_id + query_time
  -> 截取 query_time 之前的消息
  -> 创建独立 EverOS project checkpoint
  -> 按 event_id 分段 add + flush
  -> EverOS 抽取 event episode
  -> Hybrid（BM25 + BGE-M3）搜索 episode/atomic facts
  -> session_id 映射回 event 的原始 message_id
  -> 统一 Reader / Judge
```

使用独立 project 而不是只依赖搜索时间过滤，是因为 EverOS episode 的写入/抽取时间不等于
原消息可见时间；先写入完整群聊再回答早期问题会造成未来信息泄漏。完整数据最终生成约 88 个
时间 checkpoint，首次运行成本主要来自这些 checkpoint 的 Memory 抽取。

#### 4.6.2 群聊 owner 和消息角色

EverOS 的 user Memory 搜索要求指定一个 `user_id`。如果直接把各群成员作为不同 owner，群共享
信息会被拆散。因此 adapter 为每个 checkpoint 创建一个虚拟群 owner，所有消息以该 owner 写入，
同时在 `sender_name` 和正文中保留原作者、消息类型与原消息 ID：

```text
[message_id=m_0014][author_id=u_dev][message_kind=human] 原始消息正文
```

这样既保证 Memory 属于同一群作用域，又保留作者和 human/bot 区别供抽取模型判断。

#### 4.6.3 为什么必须按 event flush

初版把整个 checkpoint 一次 flush。EverOS 会将较长群聊压缩成少量主题，后半段状态更新可能
没有形成独立可检索 episode。正式版本按 `event_id` 分段 flush，使每个业务事件成为一个
Memory 单元。这个选择同时决定 TopK 的含义：EverOS 的一个 TopK 单元是一个 event episode，
不是一条原消息或单个 atomic fact。

#### 4.6.4 来源映射和检索指标

EverOS 抽取后的文本通常不会保留正文中的 `message_id` 标记，但搜索结果会返回生成该 Memory
的 `session_id`。adapter 在写入时保存：

```text
session_id -> event_id -> source_message_ids
```

因此正式结果使用 `session_id` 精确回溯原始 event 消息。只有 `session_id` 缺失时，才退化到
显式 ID 或保守文本对齐。Hit/Recall/MRR 的 rank 是 EverOS event episode 的排名；一个 episode
命中时，其关联 event 内的全部来源消息参与 Oracle 计算。

**该口径已确认存在计分泄漏**：由于证据通常整段落在单个 event 内，命中一条 Memory 即认领
整个 event 的消息，使 Hit/Recall 严重虚高，且与按单消息计分的其他方法不可比。详见 5.4.1。

#### 4.6.5 异步索引与缓存

- `/flush` 返回时 episode Markdown 已落盘，但 LanceDB 索引仍是异步更新。
- 不能看到第一条搜索结果就认为整个 checkpoint 已完成索引；初版因此漏掉最后写入的 event。
- 也不能长期等待全局 cascade `pending=0`：OME 会持续生成 atomic facts/profile，队列可能长期
  保持非零，即使 episode 已经可搜索。
- adapter 将 checkpoint 和每道题的 Top100 搜索结果缓存在
  `results/eval_platform/everos/`。Top3/Top10 共享候选缓存，避免重复抽取和搜索漂移。
- 正式运行建议关闭不参与评测的 foresight/profile/agent 类 OME strategy，只保留 episode 和
  必要的 atomic facts；否则会增加 LLM 成本和索引尾部延迟。本次结果文件已经冻结搜索缓存，
  后续展示不会受后台 OME 变化影响。

#### 4.6.6 模型与服务配置

```text
Memory LLM:     deepseek-v4-flash
Embedding:      Ollama bge-m3
Search:         hybrid
Rerank:         disabled
Reader/Judge:   deepseek-v4-flash
EverOS address: http://127.0.0.1:18001
```

所有 LLM 调用统一使用 `deepseek-v4-flash`；BGE-M3 只用于 embedding。API Key 只从本地
git-ignored 环境文件加载，不写入配置文档、结果或日志。

## 5. 当前正式结果

以下结果均为完整 172 题、Reader/Judge 使用 `deepseek-v4-flash`、LLM Errors 为 0。
Leaderboard 按 Answer Accuracy 排名。

### 5.1 Top3

| 方法 | Hit@3 | Recall@3 | MRR | Answer Accuracy |
|---|---:|---:|---:|---:|
| TeamAgent + BM25 | 83.14% | 41.95% | 0.656 | **68.60%** |
| TeamAgent Memory（BGE-M3） | 76.16% | 36.20% | 0.556 | **66.28%** |
| EverOS（Hybrid + BGE-M3） | 95.93% | 93.31% | 0.728 | **66.28%** |
| MindMemOS（vanilla/fast） | 73.84% | 30.05% | 0.553 | **53.49%** |
| BM25 RAG | 84.88% | 42.89% | 0.661 | **52.33%** |
| Mem0 | 73.26% | 29.57% | 0.584 | **43.60%** |

### 5.2 Top10

| 方法 | Hit@10 | Recall@10 | MRR | Answer Accuracy |
|---|---:|---:|---:|---:|
| TeamAgent + BM25 | 100.00% | 85.78% | 0.692 | **73.26%** |
| TeamAgent Memory（BGE-M3） | 98.84% | 81.84% | 0.602 | **73.26%** |
| BM25 RAG | 100.00% | 86.51% | 0.692 | **72.09%** |
| EverOS（Hybrid + BGE-M3） | 100.00% | 100.00% | 0.738 | **69.77%** |
| MindMemOS（vanilla/fast） | 81.40% | 42.26% | 0.569 | **63.37%** |
| Mem0 | 83.72% | 43.62% | 0.606 | **52.91%** |

### 5.3 结果解读

1. **BM25 的高 Hit 不是 embedding 波动。**BM25 不使用 embedding；当前问题与原文中的
   人名、日期、项目名和术语重合较多，因此词法检索很强。
2. **原文检索在 Oracle 指标上天然占优。**BM25/TeamAgent 返回原始消息，可以直接匹配
   `evidence_message_ids`；Mem0/MindMemOS 返回改写记忆，来源映射丢失会低估 Hit/Recall。
3. **TeamAgent 的 L2 摘要贡献明显。**Top3 中纯 BM25 的 Hit 更高，但 TeamAgent+BM25
   Answer Accuracy 高出 16.27 个百分点，说明 L2 为 Reader 提供了状态整合信息。
4. **BM25 优于 BGE-M3 的部分主要体现在检索排序。**TeamAgent+BM25 相比 BGE-M3：
   Top3 Answer Accuracy 提升 2.32 个百分点；Top10 持平。Top10 证据足够后，瓶颈转向
   L2 摘要、Reader 推理和题目口径。
5. **Top10 的 100% Hit 区分度有限。**部分 episode 本身消息不多，Top10 接近宽范围召回；
   应重点结合 Recall、MRR、Answer Accuracy 和 Top3 观察。
6. **EverOS 的检索指标不可直接横向比较。**Top10 Recall 达到 100.00% 是计分口径造成的，
   不代表检索能力；同时 Answer Accuracy 69.77% 低于 BM25 的 72.09% 也不是检索不足造成的。
   两个现象来自两个独立缺陷，见 5.4。

### 5.4 EverOS 指标异常的两个归因

EverOS 出现了「检索指标满分、答题指标中游」的矛盾表象。经核查这不是同一个原因的两面，
而是两个方向相反、互相独立的缺陷：一个让检索指标虚高，一个让答题指标偏低。

#### 5.4.1 检索指标泄漏：session 级归因把整个 event 记为命中

`Top10 Recall = 100.00%`、172 题无一例外，这种数字不可能来自真实检索系统。根因是
`everos_eval.py` 写入时把一个 `event_id` 的**全部** `message_id` 存进 `session_sources`，
计分时只要命中的 Memory 属于该 session，就把整个 event 的消息都记为「已检索」，且该分支
优先于文本对齐，直接短路掉了内容校验：

```text
session_id -> [event 内全部 message_id]  ->  命中任意一条 Memory 即认领全部
```

三个因素叠加后满分成为必然：

| 现象 | 实测值 |
|---|---|
| 归因全部走 `session_id` 分支 | Top10 685/685、Top3 464/464；`explicit_message_id` 命中 0 次 |
| 证据完整落在单个 `event_id` 内 | 148/172 题 |
| 每个 checkpoint 可见消息数 | 平均 12.6 条，切成约 4~5 个 event |
| EverOS 实际返回的 Memory 条数 | Top10 平均 3.98 条，取不满 10 条 |
| 被记为「已检索」的语料占比 | Top10 平均 100.0%（172/172 题全量）；Top3 平均 74.5%（55 题全量） |

也就是说 `top_k=10` 时 EverOS 每次都把全部 Memory 返回，经 event 扩散后覆盖整个语料库，
「检索集合」等于「全部候选」。Top3 才开始有约束，Recall 因此回落到 93.31%。

归因粒度也与其他方法不对等，同一个 TopK 下 EverOS 获得两倍以上的信用额度：

| 方法 | 每条 Memory 认领消息数 | 每题认领消息数 |
|---|---:|---:|
| BM25 RAG | 1.00 | 10.0 |
| MindMemOS | 1.00 | 6.0 |
| EverOS | 3.17 | 12.6 |

作为对照，若改用纯内容对齐归因（严格字面匹配，对摘要型 Memory 偏苛刻，仅作下界参考），
Top10 的 Hit 从 100.00% 降到 11.63%、Recall 从 100.00% 降到 3.52%。真实水平在两者之间，
但可以确认当前 93%~100% 完全没有反映检索能力。

#### 5.4.2 时区改写：Memory 用 UTC 叙述，问题用 +08:00 壁钟

Answer Accuracy 偏低的原因**不是**摘要压缩丢信息。两项实测都指向反方向：

| 指标 | 原始消息 | EverOS Memory |
|---|---:|---:|
| 每 checkpoint 字符数 | 499 | 2137（4.48 倍） |
| 标准答案关键词覆盖率 | 0.490 | 0.521 |

Memory 文本是原文的 4.5 倍长，标准答案关键词在 Memory 中出现得比原文更多，不存在信息缺失。

真正的失分来自时间表示被改写。`api_messages()` 只把时间放进 `timestamp` 字段（epoch 毫秒），
正文标记里不携带时间文本，EverOS 抽取模型只能还原出 UTC。结果是 172/172 题的 Memory 全部
用 UTC 叙述，而数据集和问题用的是北京时间壁钟，两者相差 8 小时：

```text
原始消息   2026-09-09T11:02:00+08:00
Memory     2026-09-09 04:02 UTC
问题       “2026-09-09 11:02 的更新完成后……”
```

68 道题在问题中点名了具体时刻，这 68 道题的该时刻在 Memory 文本里**全部查不到**。
该因素与准确率的相关性很干净：

| 题目 | 题数 | EverOS | BM25 Temporal | 差值 |
|---|---:|---:|---:|---:|
| 含具体时刻 | 68 | 69.1% | 82.4% | −13.2 |
| 不含具体时刻 | 104 | 70.2% | 65.4% | +4.8 |

不需要时刻锚点时 EverOS 的抽象化是赢的，一旦需要对齐壁钟就大幅落后。按题型拆分完全吻合：

| 类型 | EverOS | BM25 Temporal |
|---|---:|---:|
| `multi_hop` | 30/31 = 0.97 | 28/31 = 0.90 |
| `state_update` | 58/82 = 0.71 | 55/82 = 0.67 |
| `source_credibility` | 9/14 = 0.64 | 8/14 = 0.57 |
| `knowledge_absence` | 14/22 = 0.64 | 16/22 = 0.73 |
| `bot_writeback_pollution` | 2/5 = 0.40 | 4/5 = 0.80 |
| `time_commitment` | 6/16 = 0.38 | 11/16 = 0.69 |

逐题对比（n=172）：双方都对 103 题，仅 EverOS 对 17 题，仅 BM25 对 21 题，双方都错 31 题。

另有两个次要失分模式：

1. **过度具体化覆盖语义答案。**`q_000008` 标准答案是「立即生效」，原文为「直接移除」+
   「我今天就下掉」。Memory 完整保留了语义，但同时补充出精确时刻，Reader 于是回答
   「2026-09-02 01:31 UTC」，被 Judge 判错。抽取模型补充的精度反而挤掉了答案本身。
2. **event 合并拍平了消息粒度。**每条 Memory 平均合并 3.17 条消息，但 `format_passages()`
   递交给 Reader 的 passage 头部只有 `source_ids[-1]` 一条代表消息的
   `message_id`/`author_id`/`timestamp`，`memory_text()` 也不含正文中的
   `[message_id=][author_id=][message_kind=]` 标记。因此需要锚定某条具体 Bot 消息的
   `bot_writeback_pollution` 题目容易拒答，这 3 题 BM25 全对。

需要说明的是拒答不是主要失分项（EverOS 6/52 = 12%，BM25 5/48 = 10%，差异不显著），
主因仍是时区错位，其次是答案没有收敛到状态词（Reader 答案平均 130 字，标准答案仅 14 字）。

#### 5.4.3 待修项

以上两点当前仅记录，未改动代码。修复方向：

| 问题 | 修复方向 |
|---|---|
| 检索指标泄漏 | 收紧归因粒度：按单条消息建 session 使 `session_sources` 为 1:1；或把 session 归因与内容对齐取交集；或让抽取保留 `[message_id=]` 标记走显式分支 |
| TopK 语义不对等 | 按「认领的消息条数」而非「Memory 条数」截断，否则小语料下 Top10 对任何 event 级方法都等于全量召回 |
| 时区改写 | 把带 `+08:00` 的原始时间文本注入消息正文，或为 EverOS 配置数据集时区 |
| 粒度拍平 | passage 头部输出全部 `source_message_ids` 及各自作者与时间，而不是只给代表消息 |

修好时区对齐后 EverOS 有机会反超——它在不依赖时刻的 104 道题上本来就领先 4.8 个百分点。

## 6. 完整运行方式

统一入口：

```text
src/eval_platform/runner.py
```

完整命令示例：

```bash
PYTHONPATH=src .venv/bin/python -m eval_platform.runner \
  --method teamagent_bm25 \
  --top-k 3 \
  --run-name "TeamAgent BM25 Top3" \
  --with-llm \
  --reader-model deepseek-v4-flash \
  --judge-model deepseek-v4-flash \
  --concurrency 8 \
  --env-file .env
```

可选 `--method`：

```text
bm25_rag
mem0
teamagent
teamagent_bm25
mindmemos
everos
```

EverOS adapter 的平台侧配置：

```text
EVAL_EVEROS_BASE_URL=http://127.0.0.1:18001
EVAL_EVEROS_API_PREFIX=/api/v2
EVAL_EVEROS_MEMORY_MODEL=deepseek-v4-flash
EVAL_EVEROS_EMBEDDING_MODEL=bge-m3
EVAL_EVEROS_SEARCH_METHOD=hybrid
EVAL_EVEROS_INDEX_TIMEOUT=240
```

运行示例：

```bash
PYTHONPATH=src .venv/bin/python -m eval_platform.runner \
  --method everos \
  --top-k 3 \
  --run-name "EverOS BGE-M3 Top3" \
  --memory-concurrency 4 \
  --with-llm \
  --reader-model deepseek-v4-flash \
  --judge-model deepseek-v4-flash \
  --concurrency 8 \
  --env-file .env
```

首次运行需要启动 EverOS 和 Ollama BGE-M3；同一 adapter 版本和数据指纹下再次运行会复用
checkpoint 与查询缓存。不要把 smoke test 和完整运行写入同一 namespace，adapter 会为
`--limit` 运行加入独立数据指纹。

执行顺序：

```text
load_benchmark()
  -> 对应 Memory adapter 构建/读取 Memory 并检索
  -> 保存 retrieval JSON
  -> run_reader_judge()
  -> 保存带 Reader/Judge 结果的最终 JSON
  -> Streamlit list_runs() 自动读取并进入 Leaderboard
```

Runner 不要求命令行传数据路径，因为默认数据路径定义在
`src/eval_platform/data.py`。如需替换数据，可设置：

```text
GROUPMEMBENCH_QUESTIONS
GROUPMEMBENCH_EPISODES_DIR
GROUPMEMBENCH_LEGACY_EPISODE
```

## 7. 代码位置

| 文件 | 作用 |
|---|---|
| `src/eval_platform/app.py` | Streamlit 页面和 Leaderboard |
| `src/eval_platform/runner.py` | 统一评测入口、检索指标和结果保存 |
| `src/eval_platform/llm_eval.py` | Reader、Judge、并发和断点续跑 |
| `src/eval_platform/memory_systems.py` | BM25 及方法注册表 |
| `src/eval_platform/mem0_eval.py` | Mem0 写入、搜索、缓存与来源映射 |
| `src/eval_platform/teamagent_eval.py` | TeamAgent L2、BGE-M3/BM25 检索 |
| `src/eval_platform/mindmemos_eval.py` | MindMemOS HTTP 适配和 episode 隔离 |
| `src/eval_platform/everos_eval.py` | EverOS HTTP 适配、checkpoint 隔离和来源映射 |
| `results/eval_platform/runs/` | retrieval 和端到端结果 JSON |

## 8. 本机与公司内网演示

### 8.1 本机启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_local.sh
```

然后打开 <http://127.0.0.1:8501>。

### 8.2 内网迁移

- 只浏览题库、预置结果、Badcase 和 Leaderboard：不需要模型或 API Key。
- 现场运行 BM25 纯检索：不需要外部服务。
- 重新运行 Reader/Judge、Mem0 或 TeamAgent L2：需要接入内网 OpenAI-compatible 模型。
- TeamAgent+BGE-M3 需要 embedding 服务；TeamAgent+BM25 不需要 embedding 服务。
- 重新运行 MindMemOS 需要单独部署 MindMemOS、Qdrant、Neo4j 和模型服务。
- `.env` 必须保持 Git ignored，不能把 API Key 提交到仓库。

推荐演示顺序：

1. 题库：展示 172 题、八类问题、Oracle 答案和证据消息。
2. Leaderboard：分别切换 Top3、Top10，解释不能混榜。
3. 对比纯 BM25、TeamAgent+BGE-M3、TeamAgent+BM25，说明 L2 和检索器的贡献。
4. 结果与 Badcase：展示“命中但答错”和“未命中导致答错”的实例。
5. 启动评测：现场跑 BM25 纯检索，证明平台在内网可以独立运行。

## 9. 已知限制与下一步

1. **问题分布不均衡**：`state_update` 占 82 题，两个稀有类型各只有 1 题，分类结论不稳定。
2. **Top10 容易饱和**：应增加更长 episode、更多干扰消息，或补充 Top5 等中间配置。
3. **缺少显式 group_id**：后续应构造跨群同用户、同术语、同项目干扰，验证群组隔离。
4. **Memory 与原消息的计分单元不同**：需要补充 memory-level relevance 标注或人工审计，减少
   对抽取型系统的不公平。EverOS 的 event 级归因是这个问题的极端案例，已导致指标失效，
   详见 5.4.1。
5. **EverOS 检索指标当前不可用于横向比较**：待按 5.4.3 收紧归因粒度并重跑后才能进榜；
   在此之前 EverOS 一行应只看 Answer Accuracy。
6. **写入链路会改写时间表示**：适配器只传 epoch 时间戳，抽取模型据此还原为 UTC，与数据集
   的 `+08:00` 壁钟错开 8 小时，系统性拉低所有时刻相关题型，详见 5.4.2。其他抽取型方法
   （Mem0、MindMemOS）是否存在同类改写尚未核查。
7. **Judge 仍是 LLM**：正式报告可对关键 Badcase 做人工复核，并记录重复评测方差。
8. **检索器实验可继续扩展**：增加 TeamAgent Hybrid（BM25+BGE-M3）、metadata filter、reranker。
9. **当前结论**：在本数据上，TeamAgent 的 L2 摘要对端到端回答有明显价值；BM25 是比
   当前 BGE-M3 更强的原始消息排序器，但 Top10 已接近饱和，下一步需要更难的数据验证。
   EverOS 在语义聚合类题型（`multi_hop`、`state_update`）已优于 BM25，其落后完全集中在
   时刻对齐类题型，修复时区后需要重新评估。
