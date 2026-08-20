# 第一版产物（冻结）

真实飞书群聊只贡献因果骨架；人名、业务、对话和标准答案都在后半段合成。本目录按流水线阶段归档，不再写入。

| 子目录 | 阶段 | 脚本 |
|---|---|---|
| `01_cleaned_chat/` | 清洗、匿名化 | `clean_real_chat.py` |
| `02_task_chains/` | 窗口抽 task / fragment | `extract_task_chains.py` |
| `03_story_outlines/` | 压成因果大纲 | `build_story_outlines.py` |
| `04_story_prototypes/` | 合并为 8 个可复用原型 | `build_story_prototypes.py` |
| `05_pilot_0004/` | 首个打通的 seed / episode / 7 题 | `instantiate_story_prototype.py` + `generate.py` + `build_memory_questions.py` |
| `06_eval_batch_150_2026-08-16/` | 正式评测快照：18 episode、172 题 | `build_dataset_batch.py` |

`06_eval_batch_150_2026-08-16/` 内结构保持原样：`episodes/`、`seeds/`、`questions/`、`questions_aggregate.json`、`logs/`、`manifest.json`。
