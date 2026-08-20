from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from eval_platform.data import BenchmarkData, answer_text, runs_dir
from eval_platform.llm_eval import DEFAULT_ENV_FILE, load_env_file
from eval_platform.runner import atomic_write_json, safe_run_name, summarize

ADAPTER_VERSION = "everos_http_v3_event_checkpoint_bgem3"
CACHE_TOP_K = 100
MESSAGE_ID_RE = re.compile(r"\[message_id=([^\]\s]+)\]")
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def resolve_everos_config(env_file: Path | None = None) -> dict[str, Any]:
    load_env_file(env_file or DEFAULT_ENV_FILE)
    base_url = str(os.environ.get("EVAL_EVEROS_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Missing EVAL_EVEROS_BASE_URL")
    embedding_model = str(os.environ.get("EVAL_EVEROS_EMBEDDING_MODEL") or "bge-m3")
    if embedding_model.lower() not in {"bge-m3", "baai/bge-m3"}:
        raise ValueError("EverOS evaluation requires EVAL_EVEROS_EMBEDDING_MODEL=bge-m3")
    return {
        "base_url": base_url,
        "api_prefix": str(os.environ.get("EVAL_EVEROS_API_PREFIX") or "/api/v2").rstrip("/"),
        "app_id": str(os.environ.get("EVAL_EVEROS_APP_ID") or "groupmembench_eval"),
        "search_method": str(os.environ.get("EVAL_EVEROS_SEARCH_METHOD") or "hybrid"),
        "memory_model": str(
            os.environ.get("EVAL_EVEROS_MEMORY_MODEL") or "deepseek-v4-flash"
        ),
        "embedding_model": "bge-m3",
        "timeout": float(os.environ.get("EVAL_EVEROS_TIMEOUT") or 1200),
        "index_timeout": float(os.environ.get("EVAL_EVEROS_INDEX_TIMEOUT") or 180),
    }


class EverOSClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["base_url"])
        self.prefix = str(config["api_prefix"])
        self.timeout = float(config["timeout"])

    def verify(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            payload = response.json() or {}
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"EverOS service is unavailable at {self.base_url}: {exc}") from exc
        capabilities = payload.get("capabilities") or {}
        if payload.get("status") != "ok" or not capabilities.get("llm") or not capabilities.get("embed"):
            raise RuntimeError(
                f"{self.base_url} is not a ready EverOS service: "
                f"status={payload.get('status')} capabilities={capabilities}"
            )

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("EverOS health returned a non-object response")
        return payload

    def post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}{self.prefix}/memory/{endpoint}",
                    json=body,
                    timeout=self.timeout,
                )
                payload = response.json()
                if response.status_code >= 400:
                    error = payload.get("error") or {}
                    raise RuntimeError(
                        f"HTTP {response.status_code}: "
                        f"{str(error.get('message') or payload.get('message') or payload.get('detail') or response.text)[:500]}"
                    )
                if not isinstance(payload, dict):
                    raise RuntimeError("EverOS returned a non-object response")
                return dict(payload.get("data") or {})
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"EverOS request failed: {last_error}") from last_error


def timestamp_millis(value: Any) -> int:
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number >= 10**12 else number * 1000
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def cutoff_for(question: dict[str, Any]) -> str:
    return str(
        (question.get("temporal_scope") or {}).get("as_of")
        or (question.get("query_context") or {}).get("query_time")
        or ""
    )


def eligible(message: dict[str, Any], cutoff: str) -> bool:
    return not cutoff or not message.get("timestamp") or str(message["timestamp"]) <= cutoff


def dataset_fingerprint(data: BenchmarkData) -> str:
    digest = hashlib.sha256()
    for episode_id, episode in sorted(data.episodes.items()):
        digest.update(episode_id.encode())
        for message in episode.get("messages") or []:
            digest.update(json.dumps(message, ensure_ascii=False, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def checkpoint_key(episode_id: str, cutoff: str) -> str:
    return hashlib.sha256(f"{episode_id}\0{cutoff}".encode()).hexdigest()[:12]


def project_id(data_digest: str, episode_id: str, cutoff: str) -> str:
    episode_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", episode_id)[:32]
    return f"gmb3_{data_digest}_{episode_slug}_{checkpoint_key(episode_id, cutoff)}"[:128]


def store_dir(data_digest: str, config: dict[str, Any]) -> Path:
    configured = os.environ.get("EVAL_EVEROS_STORE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    digest = hashlib.sha256(
        "\0".join(
            [
                config["base_url"],
                config["api_prefix"],
                config["app_id"],
                config["search_method"],
                config["memory_model"],
                config["embedding_model"],
                ADAPTER_VERSION,
            ]
        ).encode()
    ).hexdigest()[:12]
    return runs_dir().parent / "everos" / f"{data_digest}-{digest}"


def api_messages(messages: list[dict[str, Any]], owner_id: str) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        author = str(message.get("author_id") or "unknown")
        kind = str(message.get("message_kind") or "unknown")
        content = (
            f"[message_id={message.get('message_id')}][author_id={author}]"
            f"[message_kind={kind}] {message.get('content') or ''}"
        )
        result.append(
            {
                # A shared synthetic owner models group memory. Original authorship
                # remains visible to extraction in sender_name and content markers.
                "sender_id": owner_id,
                "sender_name": author,
                "role": "user",
                "timestamp": timestamp_millis(message.get("timestamp")),
                "content": content,
            }
        )
    return result


def search_body(*, question: str, owner_id: str, project: str, config: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "user_id": owner_id,
        "app_id": config["app_id"],
        "project_id": project,
        "query": question,
        "method": config["search_method"],
        "top_k": top_k,
        "include_profile": False,
    }


def ingest_checkpoint(
    client: EverOSClient,
    *,
    episode_id: str,
    cutoff: str,
    messages: list[dict[str, Any]],
    data_digest: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    project = project_id(data_digest, episode_id, cutoff)
    owner_id = f"group_owner_{checkpoint_key(episode_id, cutoff)}"
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        event_id = str(message.get("event_id") or message.get("message_id") or "unknown")
        if not groups or str(groups[-1][0].get("event_id") or groups[-1][0].get("message_id")) != event_id:
            groups.append([])
        groups[-1].append(message)
    session_ids: list[str] = []
    session_sources: dict[str, list[str]] = {}
    for group_index, group in enumerate(groups, start=1):
        session_id = f"event_{checkpoint_key(episode_id, cutoff)}_{group_index:03d}"
        session_ids.append(session_id)
        session_sources[session_id] = [str(message["message_id"]) for message in group]
        body = {
            "session_id": session_id,
            "app_id": config["app_id"],
            "project_id": project,
            "messages": api_messages(group, owner_id),
        }
        client.post("add", body)
        client.post(
            "flush",
            {"session_id": session_id, "app_id": config["app_id"], "project_id": project},
        )
    return {
        "episode_id": episode_id,
        "cutoff": cutoff,
        "project_id": project,
        "owner_id": owner_id,
        "session_ids": session_ids,
        "session_sources": session_sources,
        "source_message_ids": [str(message["message_id"]) for message in messages],
        "queries": {},
    }


def memory_text(item: dict[str, Any]) -> str:
    facts = "\n".join(str(fact.get("content") or "") for fact in item.get("atomic_facts") or [])
    return "\n".join(
        value for value in [str(item.get("subject") or ""), str(item.get("summary") or ""), str(item.get("episode") or ""), facts] if value
    )


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def attribute_sources(text: str, messages: list[dict[str, Any]]) -> tuple[list[str], str]:
    valid = {str(message["message_id"]) for message in messages}
    marked = list(dict.fromkeys(match for match in MESSAGE_ID_RE.findall(text) if match in valid))
    if marked:
        return marked, "explicit_message_id"
    haystack = normalize(text)
    scored: list[tuple[float, str]] = []
    for message in messages:
        source = normalize(str(message.get("content") or ""))
        if not source:
            continue
        if source in haystack:
            score = 1.0
        else:
            score = SequenceMatcher(None, source, haystack).ratio()
        if score >= 0.58:
            scored.append((score, str(message["message_id"])))
    scored.sort(reverse=True)
    return [message_id for _, message_id in scored[:3]], "conservative_text_alignment" if scored else "unattributed"


def run_everos_retrieval(
    *,
    data: BenchmarkData,
    top_k: int,
    run_name: str,
    limit: int = 0,
    ingest_workers: int = 4,
    env_file: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = resolve_everos_config(env_file)
    client = EverOSClient(config)
    client.verify()
    questions = data.questions[:limit] if limit > 0 else data.questions
    base_digest = dataset_fingerprint(data)
    data_digest = base_digest if limit <= 0 else f"{base_digest}-limit-{limit}"
    root = store_dir(data_digest, config)
    root.mkdir(parents=True, exist_ok=True)
    states: dict[str, dict[str, Any]] = {}
    specs: dict[str, tuple[str, str, list[dict[str, Any]]]] = {}
    for question in questions:
        episode_id = str(question["episode_id"])
        cutoff = cutoff_for(question)
        key = checkpoint_key(episode_id, cutoff)
        messages = [
            message for message in data.episodes[episode_id].get("messages") or []
            if eligible(message, cutoff)
        ]
        specs[key] = (episode_id, cutoff, messages)
    for key in specs:
        path = root / f"checkpoint_{key}.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("adapter_version") == ADAPTER_VERSION:
                states[key] = state
    missing = [(key, *spec) for key, spec in specs.items() if key not in states]
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, ingest_workers)) as executor:
            futures = {
                executor.submit(
                    ingest_checkpoint,
                    EverOSClient(config),
                    episode_id=episode_id,
                    cutoff=cutoff,
                    messages=messages,
                    data_digest=data_digest,
                    config=config,
                ): key
                for key, episode_id, cutoff, messages in missing
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                state = future.result()
                state["adapter_version"] = ADAPTER_VERSION
                states[key] = state
                atomic_write_json(root / f"checkpoint_{key}.json", state)
                if progress:
                    progress(completed, len(missing) + len(questions), {"question_id": f"checkpoint:{key}", "primary_memory_type": "ingest", "hit_at_k": False})

    rows: list[dict[str, Any]] = []
    attribution_counts: Counter[str] = Counter()
    for index, question in enumerate(questions, start=1):
        episode_id = str(question["episode_id"])
        cutoff = cutoff_for(question)
        key = checkpoint_key(episode_id, cutoff)
        state = states[key]
        qid = str(question["question_id"])
        cached = (state.get("queries") or {}).get(qid)
        if cached is None:
            deadline = time.time() + config["index_timeout"]
            while True:
                data_result = client.post(
                    "search",
                    search_body(
                        question=str(question["question"]), owner_id=state["owner_id"],
                        project=state["project_id"], config=config, top_k=CACHE_TOP_K,
                    ),
                )
                cached = list(data_result.get("episodes") or [])
                if cached or time.time() >= deadline:
                    break
                time.sleep(2)
            state.setdefault("queries", {})[qid] = cached
            atomic_write_json(root / f"checkpoint_{key}.json", state)
        memories = list(cached)[:top_k]
        source_messages = specs[key][2]
        retrieved_rows = []
        sources_by_rank = []
        for rank, item in enumerate(memories, start=1):
            text = memory_text(item)
            session_sources = state.get("session_sources") or {}
            source_ids = list(session_sources.get(str(item.get("session_id") or "")) or [])
            if source_ids:
                attribution = "everos_session_id"
            else:
                source_ids, attribution = attribute_sources(text, source_messages)
            attribution_counts[attribution] += 1
            sources_by_rank.append(source_ids)
            source = next((m for m in reversed(source_messages) if str(m.get("message_id")) in source_ids), {})
            retrieved_rows.append(
                {
                    "rank": rank,
                    "score": float(item.get("score") or 0.0),
                    "memory_id": item.get("id"),
                    "message_id": source_ids[-1] if source_ids else None,
                    "source_message_ids": source_ids,
                    "source_attribution": attribution,
                    "author_id": source.get("author_id"),
                    "timestamp": item.get("timestamp") or source.get("timestamp"),
                    "thread_id": source.get("thread_id"),
                    "content": text,
                    "memory_type": "episode_with_atomic_facts",
                }
            )
        retrieved_ids = list(dict.fromkeys(mid for ids in sources_by_rank for mid in ids))
        evidence_ids = [str(value) for value in question.get("evidence_message_ids") or []]
        evidence_set = set(evidence_ids)
        matched = [mid for mid in retrieved_ids if mid in evidence_set]
        first_rank = next((rank for rank, ids in enumerate(sources_by_rank, 1) if evidence_set.intersection(ids)), None)
        row = {
            "question_id": question["question_id"], "episode_id": episode_id,
            "primary_memory_type": question["primary_memory_type"], "question": question["question"],
            "query_user_id": (question.get("query_context") or {}).get("query_user_id"),
            "query_time": cutoff, "gold_answer": answer_text(question),
            "oracle_paths": question.get("oracle_paths") or [], "evidence_message_ids": evidence_ids,
            "retrieved_message_ids": retrieved_ids, "matched_evidence_ids": matched,
            "hit_at_k": bool(matched),
            "evidence_recall_at_k": len(set(matched)) / len(evidence_set) if evidence_set else 0.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "retrieved": retrieved_rows,
        }
        rows.append(row)
        if progress:
            progress(len(missing) + index, len(missing) + len(questions), row)

    manifest = {
        "adapter_version": ADAPTER_VERSION, "dataset_fingerprint": data_digest,
        "checkpoints": len(states), "queries": len(rows),
        "source_attribution_counts": dict(attribution_counts),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(root / "manifest.json", manifest)
    now = datetime.now(timezone.utc)
    result = {
        "run_id": f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{safe_run_name(run_name)}",
        "run_name": run_name, "created_at": now.isoformat(),
        "dataset": {"questions_path": str(data.questions_path), "question_count": len(rows), "episode_count": len({r['episode_id'] for r in rows})},
        "method": {
            "method_id": "everos", "display_name": "EverOS (Hybrid + BGE-M3)",
            "version": ADAPTER_VERSION, "top_k": top_k,
            "evaluation_mode": "checkpoint_memory_source_retrieval",
            "protocol_version": "feishu_eval_v2_temporal",
            "memory_llm_model": config["memory_model"], "embedding_model": "bge-m3",
            "search_strategy": config["search_method"], "include_profile": False,
            "reflection": False, "store_dir": str(root),
            "source_attribution": (
                "everos_session_id_then_explicit_message_id_then_conservative_text_alignment"
            ),
            "source_attribution_counts": dict(attribution_counts),
        },
        "summary": summarize(rows), "rows": rows,
    }
    output = runs_dir() / f"{result['run_id']}.json"
    atomic_write_json(output, result)
    return output, result
