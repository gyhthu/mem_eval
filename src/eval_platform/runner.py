from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval_platform.data import BenchmarkData, answer_text, load_benchmark, load_json, runs_dir
from eval_platform.memory_systems import MEMORY_SYSTEMS


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_run_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized[:48] or "bm25"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    hits = sum(bool(row["hit_at_k"]) for row in rows)
    recall = sum(float(row["evidence_recall_at_k"]) for row in rows)
    mrr = sum(float(row["reciprocal_rank"]) for row in rows)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["primary_memory_type"])].append(row)
    type_scores = {
        qtype: {
            "questions": len(items),
            "hit_rate": sum(bool(item["hit_at_k"]) for item in items) / len(items),
            "mean_recall": sum(float(item["evidence_recall_at_k"]) for item in items)
            / len(items),
            "mrr": sum(float(item["reciprocal_rank"]) for item in items) / len(items),
        }
        for qtype, items in sorted(by_type.items())
    }
    return {
        "questions": total,
        "hits": hits,
        "badcases": total - hits,
        "hit_rate": hits / total if total else 0.0,
        "mean_evidence_recall": recall / total if total else 0.0,
        "mrr": mrr / total if total else 0.0,
        "type_scores": type_scores,
    }


def run_evaluation(
    *,
    data: BenchmarkData,
    method_id: str,
    top_k: int,
    run_name: str,
    limit: int = 0,
    progress: ProgressCallback | None = None,
    memory_concurrency: int = 4,
    env_file: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if method_id not in MEMORY_SYSTEMS:
        raise ValueError(f"unknown memory system: {method_id}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if method_id == "mem0":
        from eval_platform.mem0_eval import run_mem0_retrieval

        return run_mem0_retrieval(
            data=data,
            top_k=top_k,
            run_name=run_name,
            limit=limit,
            ingest_workers=memory_concurrency,
            env_file=env_file,
            progress=progress,
        )
    if method_id in {"teamagent", "teamagent_bm25"}:
        from eval_platform.teamagent_eval import run_teamagent_retrieval

        return run_teamagent_retrieval(
            data=data,
            top_k=top_k,
            run_name=run_name,
            limit=limit,
            env_file=env_file,
            retrieval_backend="bm25" if method_id == "teamagent_bm25" else "dense",
            progress=progress,
        )
    if method_id == "mindmemos":
        from eval_platform.mindmemos_eval import run_mindmemos_retrieval

        return run_mindmemos_retrieval(
            data=data,
            top_k=top_k,
            run_name=run_name,
            limit=limit,
            ingest_workers=memory_concurrency,
            env_file=env_file,
            progress=progress,
        )
    questions = data.questions[:limit] if limit > 0 else data.questions
    system_class = MEMORY_SYSTEMS[method_id]
    indexes = {
        episode_id: system_class(list(episode.get("messages") or []))
        for episode_id, episode in data.episodes.items()
    }
    rows: list[dict[str, Any]] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        episode_id = str(question["episode_id"])
        retrieved = indexes[episode_id].retrieve(question, top_k)
        retrieved_ids = [str(item.message.get("message_id") or "") for item in retrieved]
        evidence_ids = [str(value) for value in question.get("evidence_message_ids") or []]
        evidence_set = set(evidence_ids)
        matched = [message_id for message_id in retrieved_ids if message_id in evidence_set]
        first_rank = next(
            (rank for rank, message_id in enumerate(retrieved_ids, start=1) if message_id in evidence_set),
            None,
        )
        row = {
            "question_id": question["question_id"],
            "episode_id": episode_id,
            "primary_memory_type": question["primary_memory_type"],
            "question": question["question"],
            "query_user_id": (question.get("query_context") or {}).get("query_user_id"),
            "query_time": (question.get("temporal_scope") or {}).get("as_of")
            or (question.get("query_context") or {}).get("query_time"),
            "gold_answer": answer_text(question),
            "oracle_paths": question.get("oracle_paths") or [],
            "evidence_message_ids": evidence_ids,
            "retrieved_message_ids": retrieved_ids,
            "matched_evidence_ids": matched,
            "hit_at_k": bool(matched),
            "evidence_recall_at_k": len(set(matched)) / len(evidence_set) if evidence_set else 0.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "retrieved": [
                {
                    "rank": item.rank,
                    "score": item.score,
                    "message_id": item.message.get("message_id"),
                    "author_id": item.message.get("author_id"),
                    "timestamp": item.message.get("timestamp"),
                    "thread_id": item.message.get("thread_id"),
                    "content": item.message.get("content"),
                }
                for item in retrieved
            ],
        }
        rows.append(row)
        if progress:
            progress(index, total, row)

    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{safe_run_name(run_name)}"
    result = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": now.isoformat(),
        "dataset": {
            "questions_path": str(data.questions_path),
            "question_count": len(questions),
            "episode_count": len({str(item["episode_id"]) for item in questions}),
        },
        "method": {
            "method_id": method_id,
            "display_name": system_class.display_name,
            "version": system_class.version,
            "top_k": top_k,
            "evaluation_mode": "oracle_evidence_retrieval",
            "protocol_version": "feishu_eval_v2_temporal",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output_path = runs_dir() / f"{run_id}.json"
    atomic_write_json(output_path, result)
    return output_path, result


def list_runs() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(runs_dir().glob("*.json"), reverse=True):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        payload["_path"] = str(path)
        results.append(payload)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local GroupMemBench evaluation")
    parser.add_argument("--method", default="bm25_rag", choices=sorted(MEMORY_SYSTEMS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", default="BM25 baseline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--reader-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--memory-concurrency", type=int, default=4)
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[2] / ".env"),
    )
    parser.add_argument("--resume", default=None, help="Resume an interrupted reader/judge run JSON")
    args = parser.parse_args()
    if args.resume:
        from eval_platform.llm_eval import run_reader_judge

        output, result = run_reader_judge(
            resume_path=Path(args.resume),
            concurrency=args.concurrency,
            reader_model=args.reader_model,
            judge_model=args.judge_model,
            env_file=Path(args.env_file),
            progress=lambda current, total, row: print(
                f"[{current}/{total}] {row['question_id']} judge={row.get('judge_verdict')} "
                f"status={row.get('llm_status')}"
            ),
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"saved: {output}")
        return 0
    data = load_benchmark()
    output, result = run_evaluation(
        data=data,
        method_id=args.method,
        top_k=args.top_k,
        run_name=args.run_name,
        limit=args.limit,
        progress=lambda current, total, row: print(
            f"[{current}/{total}] {row['question_id']} hit={row['hit_at_k']}"
        ),
        memory_concurrency=args.memory_concurrency,
        env_file=Path(args.env_file),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    if args.with_llm:
        from eval_platform.llm_eval import run_reader_judge

        output, result = run_reader_judge(
            retrieval_result=result,
            run_name=args.run_name,
            concurrency=args.concurrency,
            reader_model=args.reader_model,
            judge_model=args.judge_model,
            env_file=Path(args.env_file),
            progress=lambda current, total, row: print(
                f"[LLM {current}/{total}] {row['question_id']} "
                f"judge={row.get('judge_verdict')} status={row.get('llm_status')}"
            ),
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
