from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from eval_platform.data import BenchmarkData, answer_text, runs_dir
from eval_platform.llm_eval import DEFAULT_ENV_FILE, load_env_file
from eval_platform.runner import atomic_write_json, safe_run_name, summarize

MINDMEMOS_ADAPTER_VERSION = "mindmemos_http_v2_checkpoint_temporal"
CACHE_TOP_K = 100
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_mindmemos_config(env_file: Path | None = None) -> dict[str, Any]:
    load_env_file(env_file or DEFAULT_ENV_FILE)
    base_url = str(os.environ.get("EVAL_MINDMEMOS_BASE_URL") or "").strip().rstrip("/")
    api_key = str(os.environ.get("EVAL_MINDMEMOS_API_KEY") or "").strip()
    if not base_url:
        raise ValueError("Missing EVAL_MINDMEMOS_BASE_URL")
    if not api_key:
        raise ValueError("Missing EVAL_MINDMEMOS_API_KEY")
    return {
        "base_url": base_url,
        "api_key": api_key,
        "app_id": str(os.environ.get("EVAL_MINDMEMOS_APP_ID") or "groupmembench"),
        "namespace": str(os.environ.get("EVAL_MINDMEMOS_NAMESPACE") or "").strip(),
        "search_strategy": str(
            os.environ.get("EVAL_MINDMEMOS_SEARCH_STRATEGY") or "fast"
        ),
        "rerank": parse_bool(os.environ.get("EVAL_MINDMEMOS_RERANK"), False),
        "timeout": float(os.environ.get("EVAL_MINDMEMOS_TIMEOUT") or 1200),
        "memory_model": str(
            os.environ.get("EVAL_MINDMEMOS_MEMORY_MODEL")
            or "deepseek-v4-flash"
        ),
        "algorithm": str(
            os.environ.get("EVAL_MINDMEMOS_ALGORITHM") or "vanilla"
        ),
    }


class MindMemosHttpClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["base_url"])
        self.api_key = str(config["api_key"])
        self.timeout = float(config["timeout"])

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def verify(self) -> None:
        try:
            response = requests.get(
                f"{self.base_url}/openapi.json",
                timeout=min(self.timeout, 10),
            )
            response.raise_for_status()
            paths = (response.json() or {}).get("paths") or {}
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"MindMemOS service is unavailable at {self.base_url}: {exc}"
            ) from exc
        required = {"/v1/memory/add", "/v1/memory/search"}
        if not required.issubset(paths):
            raise RuntimeError(
                f"{self.base_url} is not a MindMemOS service; missing API paths: "
                f"{sorted(required - set(paths))}"
            )

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}{path}",
                    headers=self.headers,
                    json=body,
                    timeout=self.timeout,
                )
                payload = response.json()
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: "
                        f"{str(payload.get('message') or response.text)[:500]}"
                    )
                if not isinstance(payload, dict):
                    raise RuntimeError("MindMemOS returned a non-object response")
                if payload.get("code") not in (None, "ok", "queued"):
                    raise RuntimeError(
                        f"MindMemOS API error {payload.get('code')}: "
                        f"{payload.get('message') or ''}"
                    )
                return dict(payload.get("data") or {})
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"MindMemOS request failed: {last_error}") from last_error

    def add(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.post("/v1/memory/add", body).get("memories") or [])

    def search(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.post("/v1/memory/search", body).get("memories") or [])


def dataset_fingerprint(data: BenchmarkData) -> str:
    digest = hashlib.sha256()
    for episode_id, episode in sorted(data.episodes.items()):
        digest.update(episode_id.encode("utf-8"))
        for message in episode.get("messages") or []:
            for key in ("message_id", "author_id", "timestamp", "content"):
                digest.update(str(message.get(key) or "").encode("utf-8"))
                digest.update(b"\0")
    return digest.hexdigest()[:16]


def config_fingerprint(config: dict[str, Any]) -> str:
    # The digest invalidates a local manifest when credentials select another
    # MindMemOS project, without persisting the API key itself.
    value = "\0".join(
        [
            str(config["base_url"]),
            str(config["api_key"]),
            str(config["algorithm"]),
            str(config["app_id"]),
            str(config["search_strategy"]),
            str(config["rerank"]),
            str(config["memory_model"]),
            MINDMEMOS_ADAPTER_VERSION,
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def namespace_for(data_digest: str, config: dict[str, Any]) -> str:
    return str(config.get("namespace") or f"gmb-{data_digest}")


def scope_id(namespace: str, episode_id: str) -> str:
    return f"{namespace}:{episode_id}"


def default_store_dir(
    data_digest: str, config: dict[str, Any]
) -> Path:
    configured = os.environ.get("EVAL_MINDMEMOS_STORE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        runs_dir().parent
        / "mindmemos"
        / f"{data_digest}-{config_fingerprint(config)}"
    )


def message_key(message: dict[str, Any]) -> str:
    return f"{message['episode_id']}::{message['message_id']}"


def timestamp_millis(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def add_body(
    messages: list[dict[str, Any]], *, namespace: str, config: dict[str, Any]
) -> dict[str, Any]:
    if not messages:
        raise ValueError("MindMemOS add batch cannot be empty")
    episode_id = str(messages[0]["episode_id"])
    shared_scope = scope_id(namespace, episode_id)
    dialogue_messages: list[dict[str, Any]] = []
    for message in messages:
        dialogue: dict[str, Any] = {
            # MindMemOS explicitly supports arbitrary role names as named speakers.
            "role": str(message.get("author_id") or "unknown_speaker"),
            "content": str(message.get("content") or ""),
        }
        timestamp = timestamp_millis(message.get("timestamp"))
        if timestamp is not None:
            dialogue["timestamp"] = timestamp
        dialogue_messages.append(dialogue)
    source_ids = [str(message["message_id"]) for message in messages]
    metadata: dict[str, Any] = {
        "episode_id": episode_id,
        "source_message_ids": source_ids,
    }
    if len(source_ids) == 1:
        metadata["message_id"] = source_ids[0]
    return {
        "messages": dialogue_messages,
        "mode": "sync",
        # A group episode is one shared memory scope. Author identity remains in
        # the dialogue role and metadata instead of fragmenting retrieval by user.
        "user_id": shared_scope,
        "session_id": shared_scope,
        "app_id": config["app_id"],
        "metadata": metadata,
    }


def episode_state_path(store_dir: Path, episode_id: str) -> Path:
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12]
    return store_dir / "episodes" / f"{digest}.json"


def load_or_create_episode_state(
    store_dir: Path,
    *,
    episode_id: str,
    data_digest: str,
    config: dict[str, Any],
    namespace: str,
) -> tuple[Path, dict[str, Any]]:
    path = episode_state_path(store_dir, episode_id)
    expected = {
        "adapter_version": MINDMEMOS_ADAPTER_VERSION,
        "dataset_fingerprint": data_digest,
        "service_fingerprint": config_fingerprint(config),
        "base_url": config["base_url"],
        "algorithm": config["algorithm"],
        "namespace": namespace,
        "episode_id": episode_id,
    }
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        mismatched = [key for key, value in expected.items() if state.get(key) != value]
        if mismatched:
            raise RuntimeError(
                "MindMemOS episode state does not match current configuration: "
                f"{mismatched}"
            )
    else:
        state = {
            **expected,
            "completed_message_keys": [],
            "memory_sources": {},
            "event_counts": {},
            "query_cache": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(path, state)
    state.setdefault("completed_message_keys", [])
    state.setdefault("memory_sources", {})
    state.setdefault("event_counts", {})
    state.setdefault("query_cache", {})
    return path, state


def update_memory_sources(
    state: dict[str, Any], message_ids: list[str], events: list[dict[str, Any]]
) -> None:
    memory_sources: dict[str, list[str]] = state["memory_sources"]
    for event in events:
        memory_id = str(event.get("memory_id") or "")
        if not memory_id:
            continue
        sources: list[str] = []
        for related_id in event.get("related_memory_ids") or []:
            sources.extend(memory_sources.get(str(related_id), []))
        sources.extend(memory_sources.get(memory_id, []))
        sources.extend(message_ids)
        memory_sources[memory_id] = list(dict.fromkeys(sources))


def checkpoint_value(question: dict[str, Any]) -> Any:
    return (question.get("temporal_scope") or {}).get("as_of") or (
        question.get("query_context") or {}
    ).get("query_time")


def checkpoint_sort_key(value: Any) -> tuple[int, datetime, str]:
    parsed = parse_datetime(value)
    if value in (None, ""):
        return (1, datetime.max.replace(tzinfo=timezone.utc), "")
    return (
        0,
        parsed or datetime.max.replace(tzinfo=timezone.utc),
        str(value),
    )


def search_body(
    question: dict[str, Any], *, namespace: str, config: dict[str, Any]
) -> dict[str, Any]:
    episode_id = str(question["episode_id"])
    shared_scope = scope_id(namespace, episode_id)
    query_user = str((question.get("query_context") or {}).get("query_user_id") or "")
    return {
        "query": f"{query_user} {question['question']}".strip(),
        "top_k": CACHE_TOP_K,
        "user_id": shared_scope,
        "session_id": shared_scope,
        "app_id": config["app_id"],
        "search_strategy": config["search_strategy"],
        "rerank": bool(config["rerank"]),
    }


def process_episode(
    *,
    client: MindMemosHttpClient,
    episode_id: str,
    episode: dict[str, Any],
    questions: list[dict[str, Any]],
    store_dir: Path,
    data_digest: str,
    config: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    path, state = load_or_create_episode_state(
        store_dir,
        episode_id=episode_id,
        data_digest=data_digest,
        config=config,
        namespace=namespace,
    )
    messages: list[dict[str, Any]] = []
    for raw_message in episode.get("messages") or []:
        message = dict(raw_message)
        message["episode_id"] = episode_id
        messages.append(message)
    completed = set(str(value) for value in state["completed_message_keys"])
    event_counts = Counter(
        {str(key): int(value) for key, value in state["event_counts"].items()}
    )

    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    checkpoint_original: dict[str, Any] = {}
    for question in questions:
        value = checkpoint_value(question)
        key = str(value) if value not in (None, "") else "__END__"
        by_checkpoint.setdefault(key, []).append(question)
        checkpoint_original[key] = value

    ordered_checkpoints = sorted(
        by_checkpoint,
        key=lambda key: checkpoint_sort_key(checkpoint_original[key]),
    )
    for checkpoint_key in ordered_checkpoints:
        cutoff = checkpoint_original[checkpoint_key]
        pending = [
            message
            for message in messages
            if message_key(message) not in completed
            and eligible_at(message.get("timestamp"), cutoff)
        ]
        if pending:
            try:
                events = client.add(
                    add_body(pending, namespace=namespace, config=config)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MindMemOS ingest failed at {episode_id}/{checkpoint_key}; "
                    "rerun to resume"
                ) from exc
            source_ids = [str(message["message_id"]) for message in pending]
            update_memory_sources(state, source_ids, events)
            event_counts.update(
                str(event.get("operation") or "unknown").lower()
                for event in events
            )
            completed.update(message_key(message) for message in pending)
            state["completed_message_keys"] = sorted(completed)
            state["event_counts"] = dict(sorted(event_counts.items()))
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(path, state)

        for question in by_checkpoint[checkpoint_key]:
            question_id = str(question["question_id"])
            if question_id in state["query_cache"]:
                continue
            try:
                state["query_cache"][question_id] = client.search(
                    search_body(question, namespace=namespace, config=config)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MindMemOS search failed at {question_id}; rerun to resume"
                ) from exc
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(path, state)

    # Complete the dataset ingestion after all temporal checkpoints. This does
    # not affect already cached search results, but makes the persisted scope
    # reusable for later questions at the end of the episode.
    remaining = [
        message for message in messages if message_key(message) not in completed
    ]
    if remaining:
        try:
            events = client.add(
                add_body(remaining, namespace=namespace, config=config)
            )
        except Exception as exc:
            raise RuntimeError(
                f"MindMemOS final ingest failed at {episode_id}; rerun to resume"
            ) from exc
        source_ids = [str(message["message_id"]) for message in remaining]
        update_memory_sources(state, source_ids, events)
        event_counts.update(
            str(event.get("operation") or "unknown").lower() for event in events
        )
        completed.update(message_key(message) for message in remaining)

    state["completed_message_keys"] = sorted(completed)
    state["event_counts"] = dict(sorted(event_counts.items()))
    state["completed_messages"] = len(completed)
    state["extracted_memories"] = len(state["memory_sources"])
    state["cached_queries"] = len(state["query_cache"])
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, state)
    return state


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def eligible_at(value: Any, cutoff: Any) -> bool:
    if not cutoff or not value:
        return True
    candidate_time = parse_datetime(value)
    cutoff_time = parse_datetime(cutoff)
    if candidate_time is None or cutoff_time is None:
        return str(value) <= str(cutoff)
    return candidate_time <= cutoff_time


def build_episode_states(
    *,
    client: MindMemosHttpClient,
    data: BenchmarkData,
    questions: list[dict[str, Any]],
    store_dir: Path,
    data_digest: str,
    namespace: str,
    config: dict[str, Any],
    workers: int,
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, Any]]:
    questions_by_episode: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        questions_by_episode.setdefault(str(question["episode_id"]), []).append(
            question
        )
    states: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_episode,
                client=client,
                episode_id=episode_id,
                episode=data.episodes[episode_id],
                questions=episode_questions,
                store_dir=store_dir,
                data_digest=data_digest,
                config=config,
                namespace=namespace,
            ): episode_id
            for episode_id, episode_questions in questions_by_episode.items()
        }
        total_steps = len(futures) + len(questions)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            episode_id = futures[future]
            states[episode_id] = future.result()
            if progress:
                progress(
                    completed_count,
                    total_steps,
                    {
                        "question_id": f"构建 {episode_id}",
                        "primary_memory_type": "memory_build",
                        "hit_at_k": "pending",
                    },
                )
    return states


def normalized_timestamp(value: Any) -> int | None:
    parsed = parse_datetime(value)
    return int(parsed.timestamp()) if parsed is not None else None


def source_ids_for_memory(
    memory: dict[str, Any],
    *,
    state: dict[str, Any],
    episode_messages: dict[str, dict[str, Any]],
    cutoff: Any,
) -> list[str]:
    source_time = normalized_timestamp(
        memory.get("source_timestamp") or memory.get("event_time")
    )
    if source_time is not None:
        exact = [
            message_id
            for message_id, message in episode_messages.items()
            if normalized_timestamp(message.get("timestamp")) == source_time
            and eligible_at(message.get("timestamp"), cutoff)
        ]
        if exact:
            return exact
    return [
        source_id
        for source_id in (state.get("memory_sources") or {}).get(
            str(memory.get("id") or ""), []
        )
        if source_id in episode_messages
        and eligible_at(episode_messages[source_id].get("timestamp"), cutoff)
    ]


def run_mindmemos_retrieval(
    *,
    data: BenchmarkData,
    top_k: int,
    run_name: str,
    limit: int = 0,
    ingest_workers: int = 4,
    env_file: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = resolve_mindmemos_config(env_file)
    client = MindMemosHttpClient(config)
    client.verify()
    questions = data.questions[:limit] if limit > 0 else data.questions
    base_data_digest = dataset_fingerprint(data)
    # A partial smoke run must never populate the same temporal memory scopes
    # as the full benchmark, because it may ingest past an omitted checkpoint.
    data_digest = (
        base_data_digest if limit <= 0 else f"{base_data_digest}-limit-{limit}"
    )
    namespace = namespace_for(data_digest, config)
    store_dir = default_store_dir(data_digest, config)
    states = build_episode_states(
        client=client,
        data=data,
        questions=questions,
        store_dir=store_dir,
        data_digest=data_digest,
        config=config,
        namespace=namespace,
        workers=ingest_workers,
        progress=progress,
    )
    aggregate_events: Counter[str] = Counter()
    for state in states.values():
        aggregate_events.update(
            {
                str(key): int(value)
                for key, value in (state.get("event_counts") or {}).items()
            }
        )
    manifest = {
        "adapter_version": MINDMEMOS_ADAPTER_VERSION,
        "dataset_fingerprint": data_digest,
        "base_url": config["base_url"],
        "algorithm": config["algorithm"],
        "namespace": namespace,
        "completed_messages": sum(
            int(state.get("completed_messages") or 0) for state in states.values()
        ),
        "extracted_memories": sum(
            int(state.get("extracted_memories") or 0) for state in states.values()
        ),
        "cached_queries": sum(
            int(state.get("cached_queries") or 0) for state in states.values()
        ),
        "event_counts": dict(sorted(aggregate_events.items())),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(store_dir / "manifest.json", manifest)
    message_lookup_by_episode = {
        episode_id: {
            str(message.get("message_id")): message
            for message in episode.get("messages") or []
        }
        for episode_id, episode in data.episodes.items()
    }
    rows: list[dict[str, Any]] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        episode_id = str(question["episode_id"])
        state = states[episode_id]
        memories = list(
            (state.get("query_cache") or {}).get(str(question["question_id"])) or []
        )[:top_k]
        cutoff = checkpoint_value(question)
        episode_messages = message_lookup_by_episode[episode_id]
        sources_by_rank = [
            source_ids_for_memory(
                memory,
                state=state,
                episode_messages=episode_messages,
                cutoff=cutoff,
            )
            for memory in memories
        ]
        retrieved_ids = list(
            dict.fromkeys(
                source_id for source_ids in sources_by_rank for source_id in source_ids
            )
        )
        evidence_ids = [
            str(value) for value in question.get("evidence_message_ids") or []
        ]
        evidence_set = set(evidence_ids)
        matched = [value for value in retrieved_ids if value in evidence_set]
        first_rank = next(
            (
                rank
                for rank, source_ids in enumerate(sources_by_rank, start=1)
                if evidence_set.intersection(source_ids)
            ),
            None,
        )
        retrieved_rows: list[dict[str, Any]] = []
        for rank, (memory, source_ids) in enumerate(
            zip(memories, sources_by_rank), start=1
        ):
            source_message = episode_messages.get(source_ids[-1]) if source_ids else {}
            retrieved_rows.append(
                {
                    "rank": rank,
                    "score": 0.0,
                    "memory_id": memory.get("id"),
                    "message_id": source_ids[-1] if source_ids else None,
                    "source_message_ids": source_ids,
                    "author_id": source_message.get("author_id"),
                    "timestamp": memory.get("source_timestamp")
                    or memory.get("event_time")
                    or source_message.get("timestamp"),
                    "thread_id": source_message.get("thread_id"),
                    "content": memory.get("memory") or "",
                    "memory_type": memory.get("memory_type"),
                }
            )
        row = {
            "question_id": question["question_id"],
            "episode_id": episode_id,
            "primary_memory_type": question["primary_memory_type"],
            "question": question["question"],
            "query_user_id": (question.get("query_context") or {}).get("query_user_id"),
            "query_time": cutoff
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
            "retrieved": retrieved_rows,
        }
        rows.append(row)
        if progress:
            progress(len(states) + index, len(states) + total, row)

    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{safe_run_name(run_name)}"
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
            "method_id": "mindmemos",
            "display_name": (
                f"MindMemOS ({config['algorithm']}/{config['search_strategy']})"
            ),
            "version": MINDMEMOS_ADAPTER_VERSION,
            "top_k": top_k,
            "evaluation_mode": "checkpoint_memory_source_retrieval",
            "protocol_version": "feishu_eval_v2_temporal",
            "memory_llm_model": config["memory_model"],
            "memory_algorithm": config["algorithm"],
            "search_strategy": config["search_strategy"],
            "rerank": config["rerank"],
            "service_base_url": config["base_url"],
            "store_dir": str(store_dir),
            "ingested_messages": manifest.get("completed_messages"),
            "extracted_memories": manifest.get("extracted_memories"),
            "event_counts": manifest.get("event_counts"),
            "cached_queries": manifest.get("cached_queries"),
            "candidate_cache_top_k": CACHE_TOP_K,
            "source_attribution": "source_timestamp_then_add_event_manifest",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output_path = runs_dir() / f"{run_id}.json"
    atomic_write_json(output_path, result)
    return output_path, result
