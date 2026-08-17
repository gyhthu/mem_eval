from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from llm_utils import chat_completion_text, create_chat_client

from eval_platform.data import BenchmarkData, answer_text, runs_dir
from eval_platform.llm_eval import DEFAULT_ENV_FILE, completion_extra, resolve_llm_config
from eval_platform.runner import atomic_write_json, safe_run_name, summarize


os.environ.setdefault("MEM0_TELEMETRY", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_ROOT = REPO_ROOT / "results/eval_platform/mem0"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_DIMS = 512
MEM0_ADAPTER_VERSION = "mem0_v2_deepseek_bge_zh_v1"
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def _mem0_imports() -> tuple[Any, Any]:
    try:
        import mem0
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            "Mem0 dependencies are missing. Install code/eval_platform/requirements.txt."
        ) from exc
    return mem0, Memory


class DeepSeekMem0LLM:
    """Mem0 extraction adapter using the same configured DeepSeek endpoint."""

    def __init__(self, config: dict[str, Any], max_tokens: int = 1024) -> None:
        self.config = config
        self.client = create_chat_client(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            azure_endpoint=config["base_url"],
        )
        self.max_tokens = max_tokens

    def generate_response(
        self,
        messages: Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> str:
        extra = dict(kwargs)
        if response_format is not None:
            extra["response_format"] = response_format
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = tool_choice
        extra.update(completion_extra(self.config))
        for attempt in range(3):
            text = chat_completion_text(
                self.client,
                model=self.config["reader_model"],
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
                **extra,
            ).strip()
            if text:
                return text
            time.sleep(attempt + 1)
        raise RuntimeError("Mem0 extraction LLM returned empty content after 3 attempts")


def dataset_fingerprint(data: BenchmarkData) -> str:
    digest = hashlib.sha256()
    for episode_id, episode in sorted(data.episodes.items()):
        digest.update(episode_id.encode())
        for message in episode.get("messages") or []:
            digest.update(str(message.get("message_id") or "").encode())
            digest.update(str(message.get("content") or "").encode())
    return digest.hexdigest()


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    mem0, _ = _mem0_imports()
    return {
        "adapter_version": MEM0_ADAPTER_VERSION,
        "mem0_version": getattr(mem0, "__version__", "unknown"),
        "llm_provider": config["provider"],
        "llm_model": config["reader_model"],
        "llm_thinking": config["thinking"],
        "embedding_provider": "fastembed",
        "embedding_model": os.environ.get(
            "EVAL_MEM0_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
        "embedding_dims": int(
            os.environ.get("EVAL_MEM0_EMBEDDING_DIMS", str(DEFAULT_EMBEDDING_DIMS))
        ),
        "vector_store": "qdrant_local",
        "infer": True,
    }


def default_store_dir(config: dict[str, Any]) -> Path:
    override = os.environ.get("EVAL_MEM0_STORE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    identity = json.dumps(public_config(config), sort_keys=True).encode()
    digest = hashlib.sha256(identity).hexdigest()[:10]
    return DEFAULT_STORE_ROOT / f"store_{digest}"


def build_memory(store_dir: Path, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    _, Memory = _mem0_imports()
    store_dir.mkdir(parents=True, exist_ok=True)
    info = public_config(config)
    raw_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": config["reader_model"],
                "api_key": config["api_key"],
                "openai_base_url": config["base_url"],
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {
                "model": info["embedding_model"],
                "embedding_dims": info["embedding_dims"],
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "groupmem_mem0",
                "path": str(store_dir / "qdrant"),
                "embedding_model_dims": info["embedding_dims"],
                "on_disk": True,
            },
        },
        "history_db_path": str(store_dir / "history.db"),
    }
    memory = Memory.from_config(raw_config)
    memory.llm = DeepSeekMem0LLM(config)
    return memory, info


def flattened_messages(data: BenchmarkData) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for episode_id, episode in sorted(data.episodes.items()):
        for message in episode.get("messages") or []:
            item = dict(message)
            item["episode_id"] = episode_id
            flattened.append(item)
    return flattened


def message_key(message: dict[str, Any]) -> str:
    return f"{message['episode_id']}::{message['message_id']}"


def metadata_for_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: message[key]
        for key in (
            "message_id",
            "event_id",
            "episode_id",
            "author_id",
            "timestamp",
            "thread_id",
            "message_kind",
            "correctness",
        )
        if message.get(key) not in (None, "")
    }


def ingest_one(memory: Any, message: dict[str, Any]) -> tuple[int, Counter[str]]:
    result = memory.add(
        [{"role": "user", "content": str(message.get("content") or "")}],
        user_id=str(message.get("author_id") or "unknown_user"),
        run_id=str(message["episode_id"]),
        metadata=metadata_for_message(message),
        infer=True,
    )
    affected = result.get("results") or []
    return len(affected), Counter(
        str(item.get("event") or "UNKNOWN").upper() for item in affected
    )


def load_or_create_manifest(
    store_dir: Path, data: BenchmarkData, info: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = store_dir / "manifest.json"
    expected = {
        "dataset_fingerprint": dataset_fingerprint(data),
        "config": info,
    }
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"Mem0 store manifest {key} differs; choose a new EVAL_MEM0_STORE_DIR"
                )
        return path, manifest
    manifest = {
        **expected,
        "status": "ingesting",
        "completed_message_keys": [],
        "extracted_memories": 0,
        "event_counts": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, manifest)
    return path, manifest


def ingest_dataset(
    *,
    memory: Any,
    data: BenchmarkData,
    store_dir: Path,
    info: dict[str, Any],
    workers: int,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    """并发摄取全部消息，并在同一 episode + author 内保持消息顺序。

    Mem0 的一次 add() 不只是保存原文：它会调用 LLM 抽取事实，并可能对同一用户已有
    memory 执行 ADD/UPDATE/DELETE。因此相同 episode、相同 author 的消息必须串行处理；
    不同分组之间则可以并发，以兼顾状态一致性和摄取速度。
    """
    # manifest 记录已经成功处理的 message key。任务中断后重新运行时，会跳过这些消息，
    # 不需要从 284 条消息的第一条重新开始。
    manifest_path, manifest = load_or_create_manifest(store_dir, data, info)
    messages = flattened_messages(data)
    completed_keys = set(manifest.get("completed_message_keys") or [])

    # 每个 deque 是一条必须按原始顺序消费的队列。分组键包含 episode_id，是为了避免
    # 不同 episode 中相同 author_id 的状态互相影响；同组消息不允许同时执行 memory.add()。
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for message in messages:
        if message_key(message) not in completed_keys:
            group_key = f"{message['episode_id']}::{message.get('author_id') or 'unknown'}"
            groups[group_key].append(message)
    if not groups:
        manifest["status"] = "ready"
        atomic_write_json(manifest_path, manifest)
        return manifest

    event_counts = Counter(manifest.get("event_counts") or {})
    extracted = int(manifest.get("extracted_memories") or 0)
    total = len(messages)
    completed_count = len(completed_keys)

    # ready 保存“当前可以提交下一条消息”的分组键。一个 group 只有在上一条消息完成后，
    # 才会重新放回 ready，所以不同 group 可并发、同一 group 始终串行。
    ready = deque(groups)

    # executor.submit() 会立即返回 Future，Future 可以理解为“后台任务结果的占位符”。
    # active 用 Future 反查它属于哪个 group、处理哪条 message，任务完成后才能正确记账。
    active: dict[Future[tuple[int, Counter[str]]], tuple[str, dict[str, Any]]] = {}
    first_error: tuple[dict[str, Any], Exception] | None = None

    def record(message: dict[str, Any], added: int, events: Counter[str]) -> None:
        # nonlocal 表示这里修改的是 ingest_dataset() 外层的两个局部变量，而不是创建
        # record() 自己的新局部变量。
        nonlocal extracted, completed_count
        key = message_key(message)
        completed_keys.add(key)
        completed_count += 1
        extracted += added
        event_counts.update(events)
        manifest.update(
            {
                "status": "ready" if completed_count == total else "ingesting",
                "completed_message_keys": sorted(completed_keys),
                "completed_messages": completed_count,
                "total_messages": total,
                "extracted_memories": extracted,
                "event_counts": dict(event_counts),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        if progress:
            progress(
                completed_count,
                total,
                {
                    "question_id": f"ingest:{message['message_id']}",
                    "primary_memory_type": "mem0_ingest",
                    "hit_at_k": False,
                    "llm_status": "ingesting",
                },
            )

    def submit_ready(executor: ThreadPoolExecutor) -> None:
        # 只要还有空闲 worker，就从不同的 ready group 各取一条消息提交。
        # executor.submit(...) 不会在这里等待 LLM 完成，而是把 Future 放进 active。
        while ready and len(active) < workers:
            group = ready.popleft()
            message = groups[group].popleft()
            active[executor.submit(ingest_one, memory, message)] = (group, message)

    # with 代码块退出时，ThreadPoolExecutor 会等待已提交任务结束并释放线程。
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        submit_ready(executor)
        while active:
            # 字典迭代默认遍历 key，因此 wait(active, ...) 实际等待的是 active 中的 Future。
            # FIRST_COMPLETED 表示任意一个任务完成就立即返回；done 是已完成的 Future 集合，
            # 第二个返回值是尚未完成的集合，这里用下划线表示不需要直接使用。
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                group, message = active.pop(future)
                try:
                    # result() 取回 ingest_one() 的返回值；如果后台线程抛异常，会在这里
                    # 重新抛出，从而进入下面的 except。
                    added, events = future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = (message, exc)
                    continue
                record(message, added, events)

                # 当前 group 的上一条消息已经完成，若队列里还有消息，才允许它重新竞争
                # worker。这个“完成后再入队”就是同组串行的关键。
                if first_error is None and groups[group]:
                    ready.append(group)
            if first_error is None:
                # 用刚刚释放的 worker 槽位继续提交其他 ready group。
                submit_ready(executor)
    if first_error:
        message, exc = first_error
        raise RuntimeError(
            f"Mem0 ingest failed at {message_key(message)}; rerun to resume"
        ) from exc
    return manifest


def result_metadata(result: dict[str, Any], key: str) -> Any:
    value = result.get(key)
    if value not in (None, ""):
        return value
    return (result.get("metadata") or {}).get(key)


def search_memories(
    memory: Any, question: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    episode_id = str(question["episode_id"])
    query_user = str((question.get("query_context") or {}).get("query_user_id") or "")
    query = f"{query_user} {question['question']}".strip()
    response = memory.search(
        query,
        filters={"run_id": episode_id},
        top_k=max(top_k * 5, 20),
        threshold=0.0,
    )
    candidates = response.get("results") or []
    cutoff = (question.get("temporal_scope") or {}).get("as_of")
    filtered: list[dict[str, Any]] = []
    for result in candidates:
        timestamp = result_metadata(result, "timestamp")
        if cutoff and timestamp and str(timestamp) > str(cutoff):
            continue
        filtered.append(result)
        if len(filtered) == top_k:
            break
    return filtered


def run_mem0_retrieval(
    *,
    data: BenchmarkData,
    top_k: int,
    run_name: str,
    limit: int = 0,
    ingest_workers: int = 4,
    env_file: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = resolve_llm_config(
        env_file=env_file or DEFAULT_ENV_FILE,
        reader_model="deepseek-v4-flash",
        judge_model="deepseek-v4-flash",
    )
    if config["reader_model"] != "deepseek-v4-flash":
        raise ValueError("Mem0 extraction model must be deepseek-v4-flash")
    store_dir = default_store_dir(config)
    memory, info = build_memory(store_dir, config)
    try:
        manifest = ingest_dataset(
            memory=memory,
            data=data,
            store_dir=store_dir,
            info=info,
            workers=ingest_workers,
            progress=progress,
        )
        questions = data.questions[:limit] if limit > 0 else data.questions
        rows: list[dict[str, Any]] = []
        total = len(questions)
        for index, question in enumerate(questions, start=1):
            memories = search_memories(memory, question, top_k)
            retrieved_ids = [str(result_metadata(item, "message_id") or "") for item in memories]
            evidence_ids = [str(value) for value in question.get("evidence_message_ids") or []]
            evidence_set = set(evidence_ids)
            matched = [value for value in retrieved_ids if value and value in evidence_set]
            first_rank = next(
                (
                    rank
                    for rank, message_id in enumerate(retrieved_ids, start=1)
                    if message_id in evidence_set
                ),
                None,
            )
            row = {
                "question_id": question["question_id"],
                "episode_id": str(question["episode_id"]),
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
                "evidence_recall_at_k": len(set(matched)) / len(evidence_set)
                if evidence_set
                else 0.0,
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                "retrieved": [
                    {
                        "rank": rank,
                        "score": float(item.get("score") or 0.0),
                        "memory_id": item.get("id"),
                        "message_id": result_metadata(item, "message_id"),
                        "author_id": result_metadata(item, "author_id")
                        or result_metadata(item, "user_id"),
                        "timestamp": result_metadata(item, "timestamp"),
                        "thread_id": result_metadata(item, "thread_id"),
                        "content": item.get("memory") or item.get("data") or "",
                    }
                    for rank, item in enumerate(memories, start=1)
                ],
            }
            rows.append(row)
            if progress:
                progress(index, total, row)
    finally:
        memory.close()

    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{safe_run_name(run_name)}"
    result = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": now.isoformat(),
        "dataset": {
            "questions_path": str(data.questions_path),
            "question_count": len(rows),
            "episode_count": len({row["episode_id"] for row in rows}),
        },
        "method": {
            "method_id": "mem0",
            "display_name": "Mem0",
            "version": MEM0_ADAPTER_VERSION,
            "top_k": top_k,
            "evaluation_mode": "oracle_evidence_retrieval",
            "protocol_version": "feishu_eval_v2_temporal",
            "memory_llm_model": config["reader_model"],
            "embedding_model": info["embedding_model"],
            "store_dir": str(store_dir),
            "ingested_messages": manifest.get("completed_messages"),
            "extracted_memories": manifest.get("extracted_memories"),
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output_path = runs_dir() / f"{run_id}.json"
    atomic_write_json(output_path, result)
    return output_path, result
