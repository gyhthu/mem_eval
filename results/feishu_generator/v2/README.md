# 第二版产物

对话轨迹复用 v1 episode，不重跑飞书清洗和消息生成。相对 v1 只改出题与判分：

1. 题面不再用含糊的「当前状态」；补题带上 `截至 {as_of}`，问的是该阶段进展。
2. `truth.status` 的展示答案改成可对齐对话的自然语言；Judge 接受同义，但不允许拿另一字段（如基线 `failed`）顶替。

| 路径 | 内容 |
|---|---|
| `episodes/` | 从 v1 复制的对话，便于本目录自洽 |
| `questions/` | 按 episode 重生成的题目 |
| `questions_aggregate.json` | 汇总，评测默认入口 |
| `logs/` | 重生成日志 |

重生成：

```bash
python3 code/feishu_generator/build_v2_questions.py
```
