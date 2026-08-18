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
- **Leaderboard**：比较完整 172 题、无运行错误的正式结果。

平台采用 Streamlit，默认地址为 <http://127.0.0.1:8501>。

## 2. 当前数据

| 项目 | 数量 | 含义 |
|---|---:|---|
| Questions | 172 | 评测问题总数 |
| Episodes | 18 | 18 段独立群聊历史；演示时可简化称为“18 个群聊” |
| Messages | 284 | 18 个 episode 中的原始消息总数 |

技术上 `episode` 是一段完整、相互隔离的群聊轨迹，不一定等同真实飞书中的永久群。
当前数据没有显式 `group_id`，评测时暂时用 `episode_id` 作为群聊/记忆隔离边界。

问题文件：

```text
results/feishu_generator/batch_150_2026-08-16/questions_aggregate.json
```

Episode 文件：

```text
results/feishu_generator/batch_150_2026-08-16/episodes/
results/feishu_generator/episode_from_prototype_0004.json
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

## 5. 当前正式结果

以下结果均为完整 172 题、Reader/Judge 使用 `deepseek-v4-flash`、LLM Errors 为 0。
Leaderboard 按 Answer Accuracy 排名。

### 5.1 Top3

| 方法 | Hit@3 | Recall@3 | MRR | Answer Accuracy |
|---|---:|---:|---:|---:|
| TeamAgent + BM25 | 83.14% | 41.95% | 0.656 | **68.60%** |
| TeamAgent Memory（BGE-M3） | 76.16% | 36.20% | 0.556 | **66.28%** |
| MindMemOS（vanilla/fast） | 73.84% | 30.05% | 0.553 | **53.49%** |
| BM25 RAG | 84.88% | 42.89% | 0.661 | **52.33%** |
| Mem0 | 73.26% | 29.57% | 0.584 | **43.60%** |

### 5.2 Top10

| 方法 | Hit@10 | Recall@10 | MRR | Answer Accuracy |
|---|---:|---:|---:|---:|
| TeamAgent + BM25 | 100.00% | 85.78% | 0.692 | **73.26%** |
| TeamAgent Memory（BGE-M3） | 98.84% | 81.84% | 0.602 | **73.26%** |
| BM25 RAG | 100.00% | 86.51% | 0.692 | **72.09%** |
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
```

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
   对抽取型系统的不公平。
5. **Judge 仍是 LLM**：正式报告可对关键 Badcase 做人工复核，并记录重复评测方差。
6. **检索器实验可继续扩展**：增加 TeamAgent Hybrid（BM25+BGE-M3）、metadata filter、reranker。
7. **当前结论**：在本数据上，TeamAgent 的 L2 摘要对端到端回答有明显价值；BM25 是比
   当前 BGE-M3 更强的原始消息排序器，但 Top10 已接近饱和，下一步需要更难的数据验证。
