from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import streamlit as st

from eval_platform.data import answer_text, load_benchmark, message_lookup
from eval_platform.llm_eval import DEFAULT_ENV_FILE, resolve_llm_config, run_reader_judge
from eval_platform.memory_systems import MEMORY_SYSTEMS
from eval_platform.runner import list_runs, run_evaluation


st.set_page_config(
    page_title="Group Memory Eval",
    page_icon="🧠",
    layout="wide",
)

def inject_styles() -> None:
    css = """
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 2rem;}
    .eval-note {padding: .8rem 1rem; border-left: 4px solid #4f7cff;
      background: color-mix(in srgb, #4f7cff 10%, transparent); border-radius: 6px;}
    .message {padding: .7rem .9rem; border: 1px solid rgba(128,128,128,.25);
      border-radius: 8px; margin-bottom: .5rem;}
    .eval-box {padding: .8rem 1rem; border: 1px solid rgba(128,128,128,.25);
      border-radius: 8px; height: 100%;}
    .eval-box h4 {margin: 0 0 .5rem 0; font-size: 0.95rem;}
    .eval-box p {margin: .25rem 0;}
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_styles()


@st.cache_resource
def benchmark():
    return load_benchmark()


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def run_label(run: dict[str, Any]) -> str:
    summary = run.get("summary") or {}
    method = run.get("method") or {}
    answer_score = summary.get("answer_accuracy")
    answer_label = f" · Answer {percentage(float(answer_score))}" if answer_score is not None else ""
    return (
        f"{run.get('run_name')} · {method.get('display_name')} · "
        f"Hit {percentage(float(summary.get('hit_rate') or 0))}{answer_label} · {run.get('run_id')}"
    )


def llm_model_label(run: dict[str, Any]) -> str:
    method = run.get("method") or {}
    roles = [
        ("Memory", method.get("memory_llm_model")),
        ("Reader", method.get("reader_model")),
        ("Judge", method.get("judge_model")),
    ]
    configured = [(role, str(model)) for role, model in roles if model]
    unique_models = list(dict.fromkeys(model for _, model in configured))
    if len(unique_models) == 1:
        return unique_models[0]
    if configured:
        return " / ".join(f"{role}: {model}" for role, model in configured)
    return "—"


def questions_board(run: dict[str, Any]) -> str:
    path = str((run.get("dataset") or {}).get("questions_path") or "").replace("\\", "/")
    return "v2" if "/v2/" in path else "v1"


def leaderboard_configuration(run: dict[str, Any]) -> tuple[Any, ...]:
    method = run.get("method") or {}
    return (
        method.get("method_id"),
        method.get("version"),
        method.get("top_k"),
        method.get("memory_llm_model"),
        method.get("reader_model"),
        method.get("judge_model"),
        method.get("memory_algorithm"),
        method.get("search_strategy"),
        method.get("rerank"),
    )


def eligible_leaderboard_runs(
    runs: list[dict[str, Any]], *, board: str, question_count: int
) -> list[dict[str, Any]]:
    full_runs = [
        run
        for run in runs
        if questions_board(run) == board
        and int((run.get("summary") or {}).get("questions") or 0) == question_count
        and int((run.get("summary") or {}).get("llm_errors") or 0) == 0
        and (run.get("summary") or {}).get("answer_accuracy") is not None
        and (
            (run.get("summary") or {}).get("llm_completed") is None
            or int((run.get("summary") or {}).get("llm_completed") or 0) == question_count
        )
        and (run.get("method") or {}).get("method_id") in MEMORY_SYSTEMS
        and (run.get("method") or {}).get("version")
        == MEMORY_SYSTEMS[(run.get("method") or {}).get("method_id")].version
    ]
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for run in full_runs:
        key = leaderboard_configuration(run)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(run)
    return deduplicated


def render_leaderboard_table(
    runs: list[dict[str, Any]], *, board_label: str, question_count: int, widget_key: str
) -> None:
    if not runs:
        st.info(f"{board_label}还没有完整 {question_count} 题且无运行错误的评测记录。")
        return
    available_top_k = sorted(
        {
            int((run.get("method") or {}).get("top_k"))
            for run in runs
            if (run.get("method") or {}).get("top_k") is not None
        }
    )
    selected_top_k = st.radio(
        "评测配置",
        options=available_top_k,
        format_func=lambda value: f"Top K = {value}",
        horizontal=True,
        key=widget_key,
    )
    configured_runs = [
        run
        for run in runs
        if int((run.get("method") or {}).get("top_k") or 0) == selected_top_k
    ]
    rank_by_answer = any(
        (run.get("summary") or {}).get("answer_accuracy") is not None
        for run in configured_runs
    )
    leaderboard = sorted(
        [
            {
                "方法": (run.get("method") or {}).get("display_name"),
                "LLM 模型": llm_model_label(run),
                "版本": (run.get("method") or {}).get("version"),
                "Top K": (run.get("method") or {}).get("top_k"),
                "题数": (run.get("summary") or {}).get("questions"),
                "Hit@K": (run.get("summary") or {}).get("hit_rate"),
                "Recall@K": (run.get("summary") or {}).get("mean_evidence_recall"),
                "MRR": (run.get("summary") or {}).get("mrr"),
                "Answer Accuracy": (run.get("summary") or {}).get("answer_accuracy"),
                "时间": run.get("created_at"),
            }
            for run in configured_runs
        ],
        key=lambda row: (
            float(row["Answer Accuracy"] if row["Answer Accuracy"] is not None else -1)
            if rank_by_answer
            else float(row["Hit@K"] or 0),
            float(row["Recall@K"] or 0),
            float(row["MRR"] or 0),
        ),
        reverse=True,
    )
    ranking_metric = "Answer Accuracy" if rank_by_answer else "Hit@K"
    st.caption(
        f"{board_label}仅比较 Top K = {selected_top_k}、完整 {question_count} 题且无运行错误的结果；"
        f"按 {ranking_metric} 排名。v1 / v2 题集分榜，互不覆盖。"
    )
    for rank, row in enumerate(leaderboard, start=1):
        row["排名"] = rank
    st.dataframe(
        leaderboard,
        column_order=[
            "排名",
            "方法",
            "LLM 模型",
            "版本",
            "Top K",
            "题数",
            "Hit@K",
            "Recall@K",
            "MRR",
            "Answer Accuracy",
            "时间",
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Hit@K": st.column_config.NumberColumn("Hit@K", format="percent"),
            "Recall@K": st.column_config.NumberColumn("Recall@K", format="percent"),
            "MRR": st.column_config.NumberColumn("MRR", format="%.3f"),
            "Answer Accuracy": st.column_config.NumberColumn(
                "Answer Accuracy", format="percent"
            ),
        },
    )


def render_message(message: dict[str, Any], *, evidence: bool = False) -> None:
    badge = " · ORACLE EVIDENCE" if evidence else ""
    st.markdown(
        f"**{message.get('message_id')} · {message.get('author_id')} · "
        f"{message.get('timestamp')}**{badge}\n\n{message.get('content')}",
    )


def render_version_pair(v1_html: str, v2_html: str) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown(f'<div class="eval-box"><h4>v1</h4>{v1_html}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="eval-box"><h4>v2</h4>{v2_html}</div>', unsafe_allow_html=True)


def render_dataset_changelog() -> None:
    st.subheader("v2 题集相对 v1 改了什么")
    st.markdown(
        "对话轨迹不变：仍是 18 个 episode、284 条消息、172 道题。"
        "v2 只重写出题口径和 Judge，用来去掉「问的是 schema 槽、群聊里却是另一件事」这类伪失败。"
        "Leaderboard 因此分成 **v1 原榜** 和 **v2 题集**，互不覆盖。"
    )
    st.markdown(
        """
        | 改什么 | v1 的问题 | v2 怎么改 |
        |---|---|---|
        | 题面 | 37 题问含糊的「当前状态」，同一事件里结果/根因/进展共用一个槽 | 问具体字段，例如「根因排查进展到哪一步」 |
        | 时间 | 91 题的提问时间是 episode 结束，检索截止却是事件结束 | 题面写 `截至 {as_of}`，`query_time` 与 `as_of` 对齐 |
        | 标准答案 | Judge 看到的是 schema token，如 `identified` | 增加自然语言展示，如「根因已定位」 |
        | Judge | 容易把同一事件里另一个字段判成正确 | 同字段近义可判对；`failed` 不能顶替「根因已定位」 |
        """
    )

    st.markdown("#### 例子 1：同一事件里两个字段，不能互相顶替")
    st.caption(
        "预算规则基线回归 / `root_cause_reproduced`。"
        "群聊里基线判定是 failed，根因进展是 identified。"
        "v1 把后者问成「当前状态」，Reader 常拿 failed 去填。"
    )
    render_version_pair(
        "<p><b>q_000122</b> 基线结果是什么？→ <code>failed</code></p>"
        "<p><b>q_000123</b> 当前状态是什么？→ <code>identified</code></p>"
        "<p>提问时间 11:03，检索截止 09:28。</p>",
        "<p><b>q_000113</b> 基线结果是什么？→ <code>failed</code></p>"
        "<p><b>q_000114</b> 根因排查进展到哪一步？→ <code>identified</code> / 根因已定位</p>"
        "<p>两题都写「截至 09:28」，提问时间也是 09:28。</p>",
    )
    st.markdown(
        "Judge 现在接受「identified ≡ 根因已定位」，"
        "但把基线判定 failed 当成根因排查进展仍判错。"
    )

    st.markdown("#### 例子 2：中间态题要对齐时间切片")
    st.caption(
        "同一讨论的 `sample_provided` 阶段。v1 在 episode 结束时刻提问，"
        "却按 09:03 做检索，题面又没写截止时间。"
    )
    render_version_pair(
        "<p><b>q_000121</b> 在“预算规则基线回归”讨论中的“sample_provided”阶段，"
        "当前状态是什么？</p>"
        "<p>gold：<code>submitted</code></p>"
        "<p>提问时间 11:03，as_of 09:03。</p>",
        "<p><b>q_000112</b> 截至2026-09-21 09:03，在“预算规则基线回归”讨论中的"
        "“sample_provided”阶段，样本提交进展到哪一步？</p>"
        "<p>gold：<code>submitted</code> / 已提交</p>"
        "<p>提问时间与 as_of 都是 09:03。</p>",
    )

    st.markdown("#### 例子 3：缺信息题也改问具体进展，不再说「当前状态」")
    render_version_pair(
        "<p><b>q_000025</b> 截至2026-09-02 09:03，在“客户续约折扣审批”讨论中的"
        "“submit_review”阶段，关于<strong>当前状态</strong>是否已有确定信息？"
        "当前记录是什么？</p>"
        "<p>gold：待复核</p>",
        "<p><b>q_000016</b> 截至2026-09-02 09:03，在“客户续约折扣审批”讨论中的"
        "“submit_review”阶段，关于<strong>提交复核进展</strong>是否已有确定信息？"
        "当时的记录是什么？</p>"
        "<p>gold：待复核</p>",
    )

    st.caption(
        "当前题库页和默认评测入口读的是 v2。"
        "v1 仍可在 Leaderboard「v1 题集（原榜）」查看历史分数。"
    )


data = benchmark()

with st.sidebar:
    st.markdown("### Group Memory Eval")
    st.caption("本地 / 公司内网可运行的 Memory Benchmark")
    st.divider()
    st.metric("题目", len(data.questions))
    col_a, col_b = st.columns(2)
    col_a.metric("Episodes", len(data.episodes))
    col_b.metric("Messages", data.message_count)
    st.divider()
    st.caption("当前题集")
    st.markdown("**v2**（对话同 v1，重写出题与 Judge）")
    st.caption("评分口径")
    st.markdown("检索指标 + Reader 答案 + LLM Judge 正确率。")

st.title("飞书群聊 Memory 评测平台")
st.markdown(
    '<div class="eval-note"><strong>完整评测链路</strong>　Memory 检索 → Reader 生成答案 → '
    "Judge 对照标准答案。当前默认评测 v2 题集；v1 原榜单独保留。"
    "支持 BM25、Mem0、TeamAgent、MindMemOS、EverOS；"
    "Reader / Judge 可配置为 DeepSeek V4 Flash。</div>",
    unsafe_allow_html=True,
)

tab_questions, tab_dataset, tab_run, tab_results, tab_leaderboard = st.tabs(
    ["题库", "v2 题集说明", "启动评测", "结果与 Badcase", "Leaderboard"]
)

with tab_questions:
    st.subheader("v2 题库 · 172 道评测题")
    st.caption("对话复用 v1；题面、标准答案展示和提问时间按 v2 重写。对照说明见「v2 题集说明」。")
    types = ["全部", *data.type_counts.keys()]
    filter_col, search_col = st.columns([1, 2])
    selected_type = filter_col.selectbox("题型", types)
    search = search_col.text_input("搜索问题", placeholder="输入关键词")
    filtered = [
        question
        for question in data.questions
        if (selected_type == "全部" or question["primary_memory_type"] == selected_type)
        and (not search or search.lower() in str(question["question"]).lower())
    ]
    st.caption(f"匹配 {len(filtered)} 道题；页面最多展开前 40 道。")
    st.dataframe(
        [
            {
                "ID": question["question_id"],
                "题型": question["primary_memory_type"],
                "Episode": question["episode_id"],
                "问题": question["question"],
            }
            for question in filtered
        ],
        width="stretch",
        hide_index=True,
        height=360,
    )
    for question in filtered[:40]:
        with st.expander(f"{question['question_id']} · {question['question']}"):
            st.markdown(f"**标准答案**：{answer_text(question)}")
            st.caption("Oracle paths: " + ", ".join(question.get("oracle_paths") or []))
            episode = data.episodes[str(question["episode_id"])]
            lookup = message_lookup(episode)
            st.markdown("**证据消息**")
            for message_id in question.get("evidence_message_ids") or []:
                if message_id in lookup:
                    render_message(lookup[message_id], evidence=True)

with tab_dataset:
    render_dataset_changelog()

with tab_run:
    st.subheader("启动一次评测")
    with st.form("new-evaluation"):
        run_name = st.text_input("运行名称", value="BM25 baseline")
        method_id = st.selectbox(
            "Memory 系统",
            options=list(MEMORY_SYSTEMS),
            format_func=lambda value: MEMORY_SYSTEMS[value].display_name,
        )
        if method_id == "mem0":
            st.caption(
                "Mem0 首次运行会用 deepseek-v4-flash 抽取 284 条消息的记忆；"
                "后续 Top3/Top10 共用本地缓存。"
            )
        elif method_id in {"teamagent", "teamagent_bm25"}:
            st.caption(
                "TeamAgent 会按每个提问时间生成 L2 群共享滚动摘要，并用 "
                f"{'BM25' if method_id == 'teamagent_bm25' else 'bge-m3'} 检索原始消息；"
                "Reader 同时读取 L2 与 TopK。首次运行会生成并缓存约 88 个时间 checkpoint。"
            )
        elif method_id == "mindmemos":
            st.caption(
                "MindMemOS 通过独立部署的 HTTP 服务构建和检索记忆；首次运行写入 284 条消息，"
                "后续 Top3/Top10 共用本地 manifest。需要在 .env 配置服务地址和 API key。"
            )
        elif method_id == "everos":
            st.caption(
                "EverOS 按 episode + 提问时间建立隔离 checkpoint，通过 Hybrid + BGE-M3 "
                "检索 episode/atomic facts；首次运行需要等待 EverOS 抽取与索引，后续复用本地缓存。"
            )
        top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
        with_llm = st.checkbox("运行 Reader + Judge（端到端评测）", value=False)
        st.caption(
            "浏览预置结果和运行 BM25 不需要 API Key；Mem0、TeamAgent、MindMemOS、EverOS "
            "或端到端 Reader/Judge 需要相应的本地/内网模型与服务配置。"
        )
        llm_col_a, llm_col_b, llm_col_c = st.columns([2, 2, 1])
        reader_model = llm_col_a.text_input(
            "Reader model", value=os.environ.get("EVAL_READER_MODEL", "deepseek-v4-flash")
        )
        judge_model = llm_col_b.text_input(
            "Judge model", value=os.environ.get("EVAL_JUDGE_MODEL", "deepseek-v4-flash")
        )
        concurrency = llm_col_c.number_input("并发", min_value=1, max_value=16, value=4)
        submitted = st.form_submit_button("开始评测", type="primary", width="stretch")
    if submitted:
        if with_llm:
            try:
                resolve_llm_config(
                    env_file=DEFAULT_ENV_FILE,
                    reader_model=reader_model,
                    judge_model=judge_model,
                )
            except ValueError as exc:
                st.error(f"LLM 配置不可用：{exc}。请先按照 .env.example 配置内网模型。")
                st.stop()
        progress_bar = st.progress(0.0, text="准备评测")
        status = st.empty()

        def update_progress(current: int, total: int, row: dict[str, Any]) -> None:
            progress_bar.progress(current / total, text=f"{current}/{total}")
            status.caption(
                f"{row['question_id']} · {row['primary_memory_type']} · "
                f"hit={row['hit_at_k']}"
            )

        try:
            output, result = run_evaluation(
                data=data,
                method_id=method_id,
                top_k=top_k,
                run_name=run_name,
                progress=update_progress,
                memory_concurrency=int(concurrency),
                env_file=DEFAULT_ENV_FILE,
            )
        except (RuntimeError, ValueError) as exc:
            st.error(f"评测启动失败：{exc}")
            st.stop()
        if with_llm:
            progress_bar.progress(0.0, text="Reader + Judge 准备中")

            def update_llm_progress(current: int, total: int, row: dict[str, Any]) -> None:
                progress_bar.progress(current / total, text=f"Reader/Judge {current}/{total}")
                status.caption(
                    f"{row['question_id']} · verdict={row.get('judge_verdict')} · "
                    f"status={row.get('llm_status')}"
                )

            output, result = run_reader_judge(
                retrieval_result=result,
                run_name=run_name,
                concurrency=int(concurrency),
                reader_model=reader_model,
                judge_model=judge_model,
                env_file=DEFAULT_ENV_FILE,
                progress=update_llm_progress,
            )
        st.session_state["selected_run_id"] = result["run_id"]
        completion = (
            f"，Answer Accuracy {percentage(result['summary']['answer_accuracy'])}"
            if "answer_accuracy" in result["summary"]
            else ""
        )
        st.success(f"完成：Hit@K {percentage(result['summary']['hit_rate'])}{completion}。")
        st.caption(f"结果已保存：{output}")

with tab_results:
    runs = list_runs()
    if not runs:
        st.info("还没有评测记录，请先在“启动评测”中运行 BM25。")
    else:
        default_index = 0
        selected_id = st.session_state.get("selected_run_id")
        if selected_id:
            default_index = next(
                (index for index, item in enumerate(runs) if item.get("run_id") == selected_id),
                0,
            )
        run_by_id = {str(item["run_id"]): item for item in runs}
        selected_run_id = st.selectbox(
            "选择评测记录",
            options=list(run_by_id),
            index=default_index,
            format_func=lambda run_id: run_label(run_by_id[run_id]),
        )
        selected_run = run_by_id[selected_run_id]
        summary = selected_run["summary"]
        has_answers = "answer_accuracy" in summary
        metric_cols = st.columns(6 if has_answers else 4)
        metric_cols[0].metric("Hit@K", percentage(summary["hit_rate"]))
        metric_cols[1].metric("Evidence Recall", percentage(summary["mean_evidence_recall"]))
        metric_cols[2].metric("MRR", f"{summary['mrr']:.3f}")
        metric_cols[3].metric("Badcase", summary["badcases"])
        if has_answers:
            metric_cols[4].metric("Answer Accuracy", percentage(summary["answer_accuracy"]))
            metric_cols[5].metric("LLM Errors", summary.get("llm_errors", 0))

        st.markdown("**分题型表现**")
        type_rows = [
            {
                "题型": key,
                "题数": value["questions"],
                "Hit@K": value["hit_rate"],
                "Recall@K": value["mean_recall"],
                "MRR": value["mrr"],
                "Answer Accuracy": value.get("answer_accuracy"),
            }
            for key, value in summary["type_scores"].items()
        ]
        st.dataframe(type_rows, width="stretch", hide_index=True)
        st.bar_chart(type_rows, x="题型", y="Hit@K")

        st.markdown("**全部逐题结果**")
        st.dataframe(
            [
                {
                    "ID": row["question_id"],
                    "题型": row["primary_memory_type"],
                    "问题": row["question"],
                    "检索命中": row["hit_at_k"],
                    "标准答案": row["gold_answer"],
                    "Reader 答案": row.get("reader_answer"),
                    "Judge": row.get("judge_verdict"),
                    "Judge 理由": row.get("judge_reasoning"),
                }
                for row in selected_run["rows"]
            ],
            width="stretch",
            hide_index=True,
            height=420,
        )
        st.download_button(
            "下载完整结果 JSON",
            data=json.dumps(selected_run, ensure_ascii=False, indent=2),
            file_name=f"{selected_run['run_id']}.json",
            mime="application/json",
        )

        if has_answers:
            badcases = [
                row for row in selected_run["rows"] if row.get("judge_verdict") != "Correct"
            ]
            badcase_title = "端到端 Answer Badcase"
        else:
            badcases = [row for row in selected_run["rows"] if not row["hit_at_k"]]
            badcase_title = "检索 Badcase"
        st.markdown(f"**{badcase_title}（{len(badcases)}）**")
        if not badcases:
            st.success("本次评测没有检索 Badcase。")
        else:
            st.dataframe(
                [
                    {
                        "ID": row["question_id"],
                        "题型": row["primary_memory_type"],
                        "问题": row["question"],
                        "Recall": row["evidence_recall_at_k"],
                    }
                    for row in badcases
                ],
                width="stretch",
                hide_index=True,
                height=300,
            )
            badcase_by_id = {str(row["question_id"]): row for row in badcases}
            selected_badcase_id = st.selectbox(
                "查看 Badcase 详情",
                options=list(badcase_by_id),
                format_func=lambda question_id: (
                    f"{question_id} · {badcase_by_id[question_id]['question']}"
                ),
            )
            selected_badcase = badcase_by_id[selected_badcase_id]
            st.markdown(f"### {selected_badcase['question']}")
            st.markdown(f"**标准答案**：{selected_badcase['gold_answer']}")
            if has_answers:
                st.markdown(f"**Reader 答案**：{selected_badcase.get('reader_answer') or '—'}")
                st.markdown(
                    f"**Judge**：{selected_badcase.get('judge_verdict') or '—'} · "
                    f"{selected_badcase.get('judge_reasoning') or selected_badcase.get('llm_error') or '—'}"
                )
            episode = data.episodes[selected_badcase["episode_id"]]
            lookup = message_lookup(episode)
            evidence_col, retrieved_col = st.columns(2)
            with evidence_col:
                st.markdown("#### Oracle 证据")
                for message_id in selected_badcase["evidence_message_ids"]:
                    if message_id in lookup:
                        render_message(lookup[message_id], evidence=True)
            with retrieved_col:
                method_name = (selected_run.get("method") or {}).get("display_name") or "Memory"
                st.markdown(f"#### {method_name} 检索结果")
                if selected_badcase.get("memory_context"):
                    with st.expander("TeamAgent L2 群共享摘要"):
                        st.markdown(selected_badcase["memory_context"])
                for item in selected_badcase["retrieved"]:
                    st.markdown(
                        f"**#{item['rank']} · {item['message_id']} · score={item['score']:.3f}**\n\n"
                        f"{item['content']}"
                    )

with tab_leaderboard:
    st.subheader("Leaderboard")
    with st.expander("评测指标说明", expanded=True):
        st.markdown(
            """
            | 指标 | 含义 | 如何理解 |
            |---|---|---|
            | **Top K** | Memory 系统最多返回的证据单元数量 | BM25/TeamAgent 一条是一条原消息；EverOS 一条是整个 event 的记忆；MindMemOS 一条是抽取后的记忆。Top K 不同、计分单元不同都必须分开看。 |
            | **Hit@K** | Top K 证据单元映射回的源消息中，是否至少包含一条 Oracle 证据，再对全部题目取平均 | 只说明“来源对上了”，不保证记忆正文能回答问题。 |
            | **Recall@K** | Top K 证据单元关联的 Oracle 源消息数 ÷ 该题全部 Oracle 证据数，再取平均 | EverOS 命中一个 event 会认领该事件全部消息，Recall 会系统性偏高。 |
            | **MRR** | 第一条 Oracle 证据所在单元排名的倒数，再对全部题目取平均 | 第一名命中得 1，第二名得 0.5，未命中得 0。 |
            | **Answer Accuracy** | Reader 答案被 LLM Judge 判定为正确的题数 ÷ 总题数 | **横向比较的主指标。** 同时受检索、记忆改写、Reader 和 Judge 影响。 |

            **EverOS 的 Hit 为什么能到 97%，Acc 却没那么高**

            EverOS 按 `event_id` 把多条原始消息打成一条记忆，计分时用 `session_id` 把该事件里**全部** `message_id` 算作召回。
            证据往往整段落在同一个 event 里，所以召回一条记忆，Hit/Recall 几乎就算满分。
            这测的是“有没有找到对的事件组”，不是“记忆正文里有没有可答题的事实”。
            Reader 读到的是改写后的 summary / atomic facts，时区还可能被写成 UTC，所以 Acc 会明显低于 Hit。
            v2 Top3：Hit 97.1%，Acc 73.8%。**不要用 EverOS 的 Hit/Recall 和 BM25、TeamAgent 比检索能力。**

            **MindMemOS 的 Hit 为什么通常更低**

            MindMemOS 返回的也是抽取后的记忆，不是原消息。一条记忆往往对应若干原消息里被压缩过的事实，
            计分靠来源时间和写入 manifest 回映 `message_id`。漏映射、压缩丢字段都会让 Hit 低于按原消息检索的方法；
            即便 Hit 上了，Reader 仍可能答错。横向比较同样以 Acc 为准。

            Hit@K 高不代表答案一定正确。v1 / v2 题集分榜，互不覆盖。v2 改了题面、时间对齐和 Judge，详见「v2 题集说明」。
            """
        )
    st.markdown(
        '<div class="eval-note"><strong>怎么读榜：</strong>方法之间请看 '
        "<strong>Answer Accuracy</strong>。"
        "EverOS 的 Hit@K / Recall@K 按 event 整组认领源消息，会出现 Hit≈97%、Acc 只有七十多的情况，"
        "不能据此认为它检索远强于 BM25 / TeamAgent。"
        "MindMemOS 返回压缩记忆，Hit 是映射回原消息后的结果，通常低于原消息检索，同样以 Acc 为准。"
        "</div>",
        unsafe_allow_html=True,
    )
    runs = list_runs()
    if not runs:
        st.info("完成至少一次评测后显示排名。")
    else:
        question_count = len(data.questions)
        v1_tab, v2_tab = st.tabs(["v1 题集（原榜）", "v2 题集"])
        with v1_tab:
            render_leaderboard_table(
                eligible_leaderboard_runs(
                    runs, board="v1", question_count=question_count
                ),
                board_label="v1 题集",
                question_count=question_count,
                widget_key="leaderboard_top_k_v1",
            )
        with v2_tab:
            render_leaderboard_table(
                eligible_leaderboard_runs(
                    runs, board="v2", question_count=question_count
                ),
                board_label="v2 题集",
                question_count=question_count,
                widget_key="leaderboard_top_k_v2",
            )
