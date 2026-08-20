from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any, Callable

from llm_utils import chat_completion_text, create_chat_client

from eval_platform.data import load_json, runs_dir
from eval_platform.runner import atomic_write_json, safe_run_name, summarize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
LLMProgressCallback = Callable[[int, int, dict[str, Any]], None]
_thread_state = local()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def resolve_llm_config(
    *,
    env_file: Path | None = None,
    reader_model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    load_env_file(env_file or DEFAULT_ENV_FILE)
    provider = os.environ.get("EVAL_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "deepseek"
    api_key = os.environ.get("EVAL_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = (
        os.environ.get("EVAL_LLM_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    resolved_reader = (
        reader_model
        or os.environ.get("EVAL_READER_MODEL")
        or os.environ.get("AGENT_MODEL")
        or "deepseek-v4-flash"
    )
    resolved_judge = (
        judge_model
        or os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("JUDGE_MODEL")
        or "deepseek-v4-flash"
    )
    if not api_key:
        raise ValueError("Missing EVAL_LLM_API_KEY or DEEPSEEK_API_KEY")
    # Reader/judge are short classification-style calls. Thinking is disabled by
    # default to keep a 172-question evaluation fast and inexpensive.
    os.environ["DEEPSEEK_THINKING"] = os.environ.get(
        "EVAL_DEEPSEEK_THINKING", "disabled"
    )
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "reader_model": resolved_reader,
        "judge_model": resolved_judge,
        "thinking": os.environ["DEEPSEEK_THINKING"],
    }


def public_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "api_key"}


def _client(config: dict[str, Any]) -> Any:
    cache_key = (config["provider"], config["base_url"], config["api_key"])
    if getattr(_thread_state, "client_key", None) != cache_key:
        _thread_state.client = create_chat_client(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            azure_endpoint=config["base_url"],
        )
        _thread_state.client_key = cache_key
    return _thread_state.client


def completion_extra(config: dict[str, Any]) -> dict[str, Any]:
    if str(config.get("thinking")).lower() == "disabled":
        # This endpoint defaults to reasoning unless it is explicitly disabled.
        # Merely omitting the field can exhaust max_tokens before content starts.
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def format_passages(row: dict[str, Any]) -> str:
    passages: list[str] = []
    for item in row.get("retrieved") or []:
        passages.append(
            "[rank={rank} message_id={message_id} author_id={author_id} "
            "timestamp={timestamp} thread_id={thread_id}]\n{content}".format(**item)
        )
    return "\n\n".join(passages) or "（没有召回到消息）"


def reader_prompt(row: dict[str, Any]) -> tuple[str, str]:
    memory_context = str(row.get("memory_context") or "").strip()
    system = (
        "你是群聊记忆问答系统的 reader。只能根据给出的 Memory 上下文回答，不得使用外部知识。"
        "要处理消息中的更新、否定、时间顺序、说话人身份和来源可信度；较新的最终决定优先于旧提议。"
        "如果记录不足以回答，要明确说无法从 Memory 上下文确定。直接给出简洁答案，不要解释检索过程。"
    )
    summary_section = (
        f"群共享滚动摘要：\n{memory_context}\n\n" if memory_context else ""
    )
    user = (
        f"提问用户：{row.get('query_user_id') or 'unknown'}\n"
        f"提问时间：{row.get('query_time') or 'unknown'}\n"
        f"问题：{row['question']}\n\n"
        f"{summary_section}"
        f"召回消息：\n{format_passages(row)}"
    )
    return system, user


def judge_prompt(row: dict[str, Any], answer: str) -> tuple[str, str]:
    system = (
        "你是严格的群聊记忆问答裁判。比较模型答案和标准答案的语义，而不是逐字匹配。"
        "列表答案必须覆盖全部关键项；时间、人物、否定不能出错；额外但不冲突的信息可接受。"
        "阶段进展类答案：中英近义或自然语言复述同一进展（如 identified 与「根因已定位」、"
        "submitted 与「已提交」）判对。不得用同一事件里另一个字段顶替，例如把基线判定 failed "
        "当成根因排查进展。"
        "若模型说无法确定而标准答案可确定，应判错。先用一句中文说明理由，最后必须单独输出"
        " `Final: Correct` 或 `Final: Incorrect`。"
    )
    user = (
        f"问题：\n{row['question']}\n\n"
        f"标准答案：\n{row['gold_answer']}\n\n"
        f"模型答案：\n{answer}"
    )
    return system, user


def split_judgment(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    verdict = "Unclear"
    final_index: int | None = None
    for index in range(len(lines) - 1, -1, -1):
        value = lines[index].lower().replace("**", "")
        if value.startswith("final:"):
            final_index = index
            label = value.split(":", 1)[1].strip()
            if "incorrect" in label:
                verdict = "Incorrect"
            elif "correct" in label:
                verdict = "Correct"
            break
    reasoning_lines = lines if final_index is None else lines[:final_index]
    return "\n".join(reasoning_lines), verdict


def evaluate_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    reader_system, reader_user = reader_prompt(result)
    answer = ""
    for attempt in range(3):
        answer = chat_completion_text(
            _client(config),
            model=config["reader_model"],
            messages=[
                {"role": "system", "content": reader_system},
                {"role": "user", "content": reader_user},
            ],
            max_tokens=512,
            temperature=0.0,
            **completion_extra(config),
        ).strip()
        if answer:
            break
        time.sleep(attempt + 1)
    if not answer:
        raise RuntimeError("reader returned empty content after 3 attempts")
    judge_system, judge_user = judge_prompt(result, answer)
    judgment = ""
    judge_reasoning = ""
    verdict = "Unclear"
    for attempt in range(3):
        judgment = chat_completion_text(
            _client(config),
            model=config["judge_model"],
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user},
            ],
            max_tokens=512,
            temperature=0.0,
            **completion_extra(config),
        ).strip()
        judge_reasoning, verdict = split_judgment(judgment)
        if verdict != "Unclear":
            break
        time.sleep(attempt + 1)
    result.update(
        {
            "reader_answer": answer,
            "judge_raw": judgment,
            "judge_reasoning": judge_reasoning,
            "judge_verdict": verdict,
            "answer_correct": verdict == "Correct",
            "llm_status": "completed",
            "llm_error": None,
        }
    )
    return result


def summarize_end_to_end(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    completed = [row for row in rows if row.get("llm_status") == "completed"]
    correct = sum(row.get("judge_verdict") == "Correct" for row in completed)
    incorrect = sum(row.get("judge_verdict") == "Incorrect" for row in completed)
    unclear = sum(row.get("judge_verdict") == "Unclear" for row in completed)
    summary.update(
        {
            "answer_correct": correct,
            "answer_badcases": len(completed) - correct,
            "answer_accuracy": correct / len(completed) if completed else 0.0,
            "llm_completed": len(completed),
            "llm_errors": sum(bool(row.get("llm_error")) for row in rows),
            "judge_unclear": unclear,
        }
    )
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_type.setdefault(str(row["primary_memory_type"]), []).append(row)
    for question_type, items in by_type.items():
        type_summary = summary["type_scores"][question_type]
        type_summary["answer_accuracy"] = sum(
            item.get("judge_verdict") == "Correct" for item in items
        ) / len(items)
        type_summary["answer_correct"] = sum(
            item.get("judge_verdict") == "Correct" for item in items
        )
    return summary


def run_reader_judge(
    *,
    retrieval_result: dict[str, Any] | None = None,
    resume_path: Path | None = None,
    run_name: str | None = None,
    concurrency: int = 4,
    reader_model: str | None = None,
    judge_model: str | None = None,
    env_file: Path | None = None,
    progress: LLMProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    config = resolve_llm_config(
        env_file=env_file, reader_model=reader_model, judge_model=judge_model
    )
    if resume_path:
        result = load_json(resume_path)
        output_path = resume_path
    else:
        if retrieval_result is None:
            raise ValueError("retrieval_result is required for a new LLM run")
        result = deepcopy(retrieval_result)
        now = datetime.now(timezone.utc)
        resolved_name = run_name or f"{result.get('run_name', 'BM25')} + reader/judge"
        run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{safe_run_name(resolved_name)}"
        result.update(
            {
                "run_id": run_id,
                "run_name": resolved_name,
                "created_at": now.isoformat(),
            }
        )
        result["method"].update(
            {
                "evaluation_mode": "retrieval_reader_judge",
                "reader_model": config["reader_model"],
                "judge_model": config["judge_model"],
                "llm_provider": config["provider"],
                "llm_base_url": config["base_url"],
                "llm_thinking": config["thinking"],
            }
        )
        for row in result["rows"]:
            row.update({"llm_status": "pending", "llm_error": None})
        output_path = runs_dir() / f"{run_id}.json"
        result["summary"] = summarize_end_to_end(result["rows"])
        atomic_write_json(output_path, result)

    pending_indices = [
        index
        for index, row in enumerate(result["rows"])
        if row.get("llm_status") != "completed"
        or row.get("judge_verdict") not in {"Correct", "Incorrect"}
        or not str(row.get("reader_answer") or "").strip()
    ]
    total = len(result["rows"])
    already_completed = total - len(pending_indices)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(evaluate_row, result["rows"][index], config): index
            for index in pending_indices
        }
        completed_count = already_completed
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result["rows"][index] = future.result()
            except Exception as exc:
                failed = dict(result["rows"][index])
                failed.update(
                    {
                        "llm_status": "error",
                        "llm_error": f"{type(exc).__name__}: {exc}",
                        "answer_correct": False,
                    }
                )
                result["rows"][index] = failed
            completed_count += 1
            result["summary"] = summarize_end_to_end(result["rows"])
            atomic_write_json(output_path, result)
            if progress:
                progress(completed_count, total, result["rows"][index])
    return output_path, result
