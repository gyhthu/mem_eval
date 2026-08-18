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


def render_message(message: dict[str, Any], *, evidence: bool = False) -> None:
    badge = " · ORACLE EVIDENCE" if evidence else ""
    st.markdown(
        f"**{message.get('message_id')} · {message.get('author_id')} · "
        f"{message.get('timestamp')}**{badge}\n\n{message.get('content')}",
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
    st.caption("评分口径")
    st.markdown("检索指标 + Reader 答案 + LLM Judge 正确率。")

st.title("飞书群聊 Memory 评测平台")
st.markdown(
    '<div class="eval-note"><strong>完整评测链路</strong>　Memory 检索 → Reader 生成答案 → '
    "Judge 对照标准答案。当前支持 BM25、Mem0、TeamAgent Memory 和 MindMemOS；"
    "Reader / Judge 均可配置为 DeepSeek V4 Flash。</div>",
    unsafe_allow_html=True,
)

tab_questions, tab_run, tab_results, tab_leaderboard = st.tabs(
    ["题库", "启动评测", "结果与 Badcase", "Leaderboard"]
)

with tab_questions:
    st.subheader("172 道评测题")
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
        elif method_id == "teamagent":
            st.caption(
                "TeamAgent 会按每个提问时间生成 L2 群共享滚动摘要，并用 bge-m3 检索原始消息；"
                "Reader 同时读取 L2 与 TopK。首次运行会生成并缓存约 88 个时间 checkpoint。"
            )
        elif method_id == "mindmemos":
            st.caption(
                "MindMemOS 通过独立部署的 HTTP 服务构建和检索记忆；首次运行写入 284 条消息，"
                "后续 Top3/Top10 共用本地 manifest。需要在 .env 配置服务地址和 API key。"
            )
        top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
        with_llm = st.checkbox("运行 Reader + Judge（端到端评测）", value=False)
        st.caption(
            "浏览预置结果和运行 BM25 不需要 API Key；Mem0、TeamAgent、MindMemOS "
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
    with st.expander("评测指标说明", expanded=False):
        st.markdown(
            """
            | 指标 | 含义 | 如何理解 |
            |---|---|---|
            | **Top K** | Memory 系统最多返回的证据单元数量 | BM25/TeamAgent 返回原消息，Mem0/MindMemOS 返回抽取后的记忆。Top K 不同必须分开比较。 |
            | **Hit@K** | Top K 证据单元的源消息中是否至少包含一条 Oracle 证据，再对全部题目取平均 | 衡量“有没有找到证据”；命中一条即记为 1，否则为 0。 |
            | **Recall@K** | Top K 证据单元关联的 Oracle 源消息数 ÷ 该题全部 Oracle 证据数，再取平均 | 衡量“证据找得全不全”；多跳问题通常需要较高 Recall。 |
            | **MRR** | 第一条 Oracle 证据排名的倒数，再对全部题目取平均 | 第一名命中得 1，第二名得 0.5，未命中得 0；越高说明正确证据越靠前。 |
            | **Answer Accuracy** | Reader 答案被 LLM Judge 判定为正确的题数 ÷ 总题数 | 端到端指标，同时受检索质量、Reader 推理和 Judge 判定影响。 |

            **注意：** Hit@K 高不代表答案一定正确。只命中一条证据时，可能仍缺少多跳问题的其他证据；Reader 也可能在证据充分时推理错误。
            """
        )
    runs = list_runs()
    if not runs:
        st.info("完成至少一次评测后显示排名。")
    else:
        full_runs = [
            run
            for run in runs
            if int((run.get("summary") or {}).get("questions") or 0) == len(data.questions)
            and int((run.get("summary") or {}).get("llm_errors") or 0) == 0
            and (run.get("summary") or {}).get("answer_accuracy") is not None
            and (run.get("method") or {}).get("method_id") in MEMORY_SYSTEMS
            and (run.get("method") or {}).get("version")
            == MEMORY_SYSTEMS[(run.get("method") or {}).get("method_id")].version
        ]
        deduplicated_runs: list[dict[str, Any]] = []
        seen_configurations: set[tuple[Any, ...]] = set()
        for run in full_runs:
            method = run.get("method") or {}
            configuration = (
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
            if configuration in seen_configurations:
                continue
            seen_configurations.add(configuration)
            deduplicated_runs.append(run)
        full_runs = deduplicated_runs
        available_top_k = sorted(
            {
                int((run.get("method") or {}).get("top_k"))
                for run in full_runs
                if (run.get("method") or {}).get("top_k") is not None
            }
        )
        if not available_top_k:
            st.info("还没有完成全部题目且运行无错误的评测记录。")
            st.stop()
        selected_top_k = st.radio(
            "评测配置",
            options=available_top_k,
            format_func=lambda value: f"Top K = {value}",
            horizontal=True,
        )
        configured_runs = [
            run
            for run in full_runs
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
            f"当前仅比较 Top K = {selected_top_k}、完整 {len(data.questions)} 题且无运行错误的结果；"
            f"按 {ranking_metric} 排名。"
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
