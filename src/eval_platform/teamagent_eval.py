from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import requests

from llm_utils import chat_completion_text, create_chat_client

from eval_platform.data import BenchmarkData, answer_text, runs_dir
from eval_platform.llm_eval import DEFAULT_ENV_FILE, resolve_llm_config
from eval_platform.runner import atomic_write_json, safe_run_name, summarize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_ROOT = REPO_ROOT / "results/eval_platform/teamagent"
DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_EMBED_MODEL = "bge-m3"
ADAPTER_VERSION = "teamagent_l2_bgem3_v1"
PROMPT_VERSION = "teamagent_distill_exact_2026_08_16"
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


# Kept in sync with TeamAgent memory/distiller.py. The benchmark deliberately
# evaluates the same L2 write policy without depending on a production memory
# process or its mutable Chroma database.
DISTILL_PROMPT = """\
你在维护一个**多 bot 群聊**的共享「群体记忆」。群里有多个 bot + 人，这块记忆所有
bot 共享，必须干净、可被任何 bot 直接复用。给你【已有摘要】和【新消息】，输出更新后
的完整摘要。

⬛ 先分清记忆三层，你只负责第二层：
  - 【L1·原则】长期稳定的规矩/偏好/约定 = 【已有摘要】里以 `## 📌` 开头的小节。
    **原样保留、不改写不删除**，放最前面；不要往这层塞新内容。
  - 【L2·群体记忆】你维护的正文：所有 bot 都该知道的**共享**事实、决策、纠偏、待办、分工。
  - 【L3·某 bot 的本地信息】只对某一个 bot 或某台机器成立的东西——本地目录/文件绝对
    路径、该 bot 的 systemd 单元名、本机端口绑定、它的工作目录/运行环境/会话状态。
    **这类一律不写进共享摘要**（它们属于各 bot 自己的 events service）。新消息里出现就
    识别并丢弃；已有摘要里若混进了，借这次更新清理出去。

维护原则：
- 做总结、不记流水账：在做什么、结论/决定（连同简短「为什么」）、关键事实、未决问题、分工。
- **留住纠偏/踩坑**：认知被推翻时别只删旧的，保留一条「曾以为 X，实际是 Y（因为…）」。
  这种「错→对」的修正信息量最高，最该沉淀。
- **抗自蒸馏污染**：分清「人确认的」与「bot 自己声称的」，冲突以**人**为准；bot 未经人
  确认的结论标「（待确认）」，别写死。
- **保留共享性钩子**：open_id / app_id / 权限 scope / commit / 专有术语 / 跨 bot 通用的
  仓库名。但**区分**：仓库名、git 地址、约定俗成的结构属共享可留；某 bot 的本地 clone
  绝对路径属 L3，丢。
- **跨 bot 教学要进 L2**：用户明确要求“教某个 bot 怎么做”、或复盘 bot 间协作失败（如
  bot @ bot 没响应、权限申请链接怎么发）时，把可复用的诊断步骤、权限 scope、链接格式、
  后台发布/审批顺序写进 L2。只丢本机路径、个人 token、单个会话状态；不要把这类教学误判成
  某 bot 的私有 L3。
- 把新消息融进已有摘要：相关归并、**真正过时**的更新掉，避免重复与自相矛盾。
- 丢弃寒暄和无信息量的消息（如「在吗 / 你好 / 收到 / 试一下」这类）。
- 用简洁的叙述段落或要点都行，怎么清楚怎么来，不必套固定模板。
- 控制在 800 字以内（不含 `## 📌` 置顶小节）；信息少就写短，不要硬凑。
- 只输出摘要正文（markdown），不要任何解释或前后缀。

【已有摘要】
{board}

【新消息】
{messages}
"""


def dataset_fingerprint(data: BenchmarkData) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode())
    for episode_id, episode in sorted(data.episodes.items()):
        digest.update(episode_id.encode())
        for message in episode.get("messages") or []:
            for key in ("message_id", "author_id", "timestamp", "content"):
                digest.update(str(message.get(key) or "").encode())
    return digest.hexdigest()


def store_dir() -> Path:
    configured = os.environ.get("EVAL_TEAMAGENT_STORE_DIR")
    path = Path(configured).expanduser().resolve() if configured else DEFAULT_STORE_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_time(question: dict[str, Any]) -> str:
    return str(
        (question.get("temporal_scope") or {}).get("as_of")
        or (question.get("query_context") or {}).get("query_time")
        or ""
    )


def eligible_messages(data: BenchmarkData, question: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(data.episodes[str(question["episode_id"])].get("messages") or [])
    cutoff = checkpoint_time(question)
    if cutoff:
        messages = [
            message
            for message in messages
            if not message.get("timestamp") or str(message["timestamp"]) <= cutoff
        ]
    return messages


def checkpoint_key(question: dict[str, Any]) -> str:
    raw = f"{question['episode_id']}\0{checkpoint_time(question)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def format_messages(messages: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        "[{role}:{author}] {text}".format(
            role="bot" if str(message.get("message_kind") or "").startswith("bot") else "user",
            author=message.get("author_id") or "unknown",
            text=message.get("content") or "",
        )
        for message in messages
    )


class TeamAgentDistiller:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = os.environ.get("EVAL_TEAMAGENT_MEMORY_MODEL", "deepseek-v4-flash")
        self.client = create_chat_client(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            azure_endpoint=config["base_url"],
        )

    def distill(self, messages: Sequence[dict[str, Any]]) -> str:
        prompt = DISTILL_PROMPT.format(
            board="（暂无摘要）",
            messages=format_messages(messages),
        )
        extra_body: dict[str, Any] = {}
        if self.model.lower().startswith("deepseek-"):
            extra_body = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            }
        for attempt in range(3):
            result = chat_completion_text(
                self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                extra_body=extra_body or None,
            ).strip()
            if result:
                return result
            time.sleep(attempt + 1)
        raise RuntimeError("TeamAgent L2 distiller returned empty content after 3 attempts")


class OpenAICompatibleEmbedder:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "EVAL_TEAMAGENT_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL
        ).rstrip("/")
        self.model = os.environ.get("EVAL_TEAMAGENT_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.api_key = os.environ.get("EVAL_TEAMAGENT_EMBED_API_KEY", "ollama")
        self.batch_size = max(1, int(os.environ.get("EVAL_TEAMAGENT_EMBED_BATCH", "32")))
        self.timeout = float(os.environ.get("EVAL_TEAMAGENT_EMBED_TIMEOUT", "300"))
        self.session = requests.Session()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            response = self.session.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": chunk},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
            if len(rows) != len(chunk):
                raise RuntimeError("embedding response count mismatch")
            vectors.extend(item["embedding"] for item in rows)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def flattened_messages(data: BenchmarkData) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for episode_id, episode in sorted(data.episodes.items()):
        for message in episode.get("messages") or []:
            result.append({**message, "episode_id": episode_id})
    return result


def load_or_build_embeddings(
    data: BenchmarkData,
    embedder: OpenAICompatibleEmbedder,
    root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    messages = flattened_messages(data)
    fingerprint = dataset_fingerprint(data)
    cache_id = hashlib.sha256(
        f"{fingerprint}\0{embedder.base_url}\0{embedder.model}".encode()
    ).hexdigest()[:16]
    manifest_path = root / f"embeddings_{cache_id}.json"
    matrix_path = root / f"embeddings_{cache_id}.npy"
    expected_ids = [
        f"{message['episode_id']}::{message.get('message_id') or ''}" for message in messages
    ]
    if manifest_path.exists() and matrix_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("message_ids") == expected_ids:
            return messages, np.load(matrix_path)

    matrix = embedder.embed([str(message.get("content") or "") for message in messages])
    temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, matrix)
    os.replace(temporary, matrix_path)
    atomic_write_json(
        manifest_path,
        {
            "dataset_fingerprint": fingerprint,
            "embedding_base_url": embedder.base_url,
            "embedding_model": embedder.model,
            "message_ids": expected_ids,
            "dimensions": int(matrix.shape[1]),
        },
    )
    return messages, matrix


def load_or_build_summaries(
    *,
    data: BenchmarkData,
    questions: Sequence[dict[str, Any]],
    distiller: TeamAgentDistiller,
    root: Path,
    progress: ProgressCallback | None,
) -> tuple[dict[str, str], Path]:
    fingerprint = dataset_fingerprint(data)
    cache_id = hashlib.sha256(
        f"{fingerprint}\0{distiller.model}\0{PROMPT_VERSION}".encode()
    ).hexdigest()[:16]
    path = root / f"l2_summaries_{cache_id}.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset_fingerprint": fingerprint,
            "prompt_version": PROMPT_VERSION,
            "memory_llm_model": distiller.model,
            "summaries": {},
        }
    summaries = dict(manifest.get("summaries") or {})
    representatives: dict[str, dict[str, Any]] = {}
    for question in questions:
        representatives.setdefault(checkpoint_key(question), question)
    missing = [(key, question) for key, question in representatives.items() if key not in summaries]
    total = len(missing)
    workers = max(1, int(os.environ.get("EVAL_TEAMAGENT_DISTILL_CONCURRENCY", "8")))
    with ThreadPoolExecutor(max_workers=min(workers, max(1, total))) as executor:
        future_to_item = {
            executor.submit(distiller.distill, eligible_messages(data, question)): (key, question)
            for key, question in missing
        }
        for index, future in enumerate(as_completed(future_to_item), start=1):
            key, _question = future_to_item[future]
            summaries[key] = future.result()
            manifest["summaries"] = summaries
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(path, manifest)
            if progress:
                progress(
                    index,
                    total,
                    {
                        "question_id": f"distill:{key}",
                        "primary_memory_type": "teamagent_l2_distill",
                        "hit_at_k": False,
                        "llm_status": "distilling",
                    },
                )
    return summaries, path


def run_teamagent_retrieval(
    *,
    data: BenchmarkData,
    top_k: int,
    run_name: str,
    limit: int = 0,
    env_file: Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = resolve_llm_config(
        env_file=env_file or DEFAULT_ENV_FILE,
        reader_model="deepseek-v4-flash",
        judge_model="deepseek-v4-flash",
    )
    questions = data.questions[:limit] if limit > 0 else data.questions
    root = store_dir()
    embedder = OpenAICompatibleEmbedder()
    all_messages, matrix = load_or_build_embeddings(data, embedder, root)
    key_to_index = {
        (str(message["episode_id"]), str(message.get("message_id") or "")): index
        for index, message in enumerate(all_messages)
    }
    distiller = TeamAgentDistiller(config)
    summaries, summary_path = load_or_build_summaries(
        data=data,
        questions=questions,
        distiller=distiller,
        root=root,
        progress=progress,
    )
    query_vectors = embedder.embed(
        [
            f"{(question.get('query_context') or {}).get('query_user_id') or ''} "
            f"{question['question']}".strip()
            for question in questions
        ]
    )

    rows: list[dict[str, Any]] = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        # 检索隔离发生在这里：eligible_messages() 先用 question["episode_id"]
        # 取出同一 episode 的消息，再去掉 checkpoint_time 之后的消息。
        # 因此 eligible 不是全数据集的 284 条消息，而只是“同 episode + 时间可见”的候选集。
        eligible = eligible_messages(data, question)

        # matrix 为了复用 Embedding 缓存，保存了全部 284 条消息的向量；但打分前会通过
        # (episode_id, message_id) 复合键，只取 eligible 对应的向量行。message_id 会在
        # 不同 episode 中重复，所以不能只用 message_id 定位。
        eligible_indices = [
            key_to_index[(str(question["episode_id"]), str(message.get("message_id") or ""))]
            for message in eligible
        ]
        query_vector = query_vectors[index - 1]

        # 相似度只在 matrix[eligible_indices] 这个切片上计算，不会与其他 episode 的
        # 消息比较。scores 和 eligible 顺序一一对应，随后从该候选集选择 Top K。
        scores = matrix[eligible_indices] @ query_vector
        order = np.argsort(-scores)[:top_k]
        retrieved_messages = [eligible[int(position)] for position in order]
        retrieved_scores = [float(scores[int(position)]) for position in order]
        retrieved_ids = [str(message.get("message_id") or "") for message in retrieved_messages]

        # 以下只负责检索评估：把 Top K 与该题标注的 Oracle evidence message IDs 对齐，
        # 计算 Hit@K、Recall@K 和第一条证据的倒数排名 MRR。
        evidence_ids = [str(value) for value in question.get("evidence_message_ids") or []]
        evidence_set = set(evidence_ids)
        matched = [value for value in retrieved_ids if value in evidence_set]
        first_rank = next(
            (rank for rank, value in enumerate(retrieved_ids, start=1) if value in evidence_set),
            None,
        )
        row = {
            "question_id": question["question_id"],
            "episode_id": str(question["episode_id"]),
            "primary_memory_type": question["primary_memory_type"],
            "question": question["question"],
            "query_user_id": (question.get("query_context") or {}).get("query_user_id"),
            "query_time": checkpoint_time(question) or None,
            "gold_answer": answer_text(question),
            "oracle_paths": question.get("oracle_paths") or [],
            "evidence_message_ids": evidence_ids,
            "retrieved_message_ids": retrieved_ids,
            "matched_evidence_ids": matched,
            "hit_at_k": bool(matched),
            "evidence_recall_at_k": len(set(matched)) / len(evidence_set) if evidence_set else 0.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            # L2 摘要的缓存键同样包含 episode_id + checkpoint_time，所以这里取到的
            # memory_context 也只由当前 episode 在该时间点之前的历史消息蒸馏而成。
            "memory_context": summaries[checkpoint_key(question)],
            # 保存同 episode 候选集中实际召回的原始消息，后续 Reader 会同时读取
            # memory_context 和这些 Top K 原文来回答问题。
            "retrieved": [
                {
                    "rank": rank,
                    "score": score,
                    "message_id": message.get("message_id"),
                    "author_id": message.get("author_id"),
                    "timestamp": message.get("timestamp"),
                    "thread_id": message.get("thread_id"),
                    "content": message.get("content"),
                }
                for rank, (message, score) in enumerate(
                    zip(retrieved_messages, retrieved_scores), start=1
                )
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
            "question_count": len(rows),
            "episode_count": len({row["episode_id"] for row in rows}),
        },
        "method": {
            "method_id": "teamagent",
            "display_name": "TeamAgent Memory",
            "version": ADAPTER_VERSION,
            "top_k": top_k,
            "evaluation_mode": "l2_summary_plus_dense_retrieval",
            "protocol_version": "feishu_eval_v2_temporal",
            "memory_llm_model": distiller.model,
            "embedding_model": embedder.model,
            "embedding_base_url": embedder.base_url,
            "l2_prompt_version": PROMPT_VERSION,
            "l2_checkpoint_count": len({checkpoint_key(question) for question in questions}),
            "l2_cache": str(summary_path),
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output_path = runs_dir() / f"{run_id}.json"
    atomic_write_json(output_path, result)
    return output_path, result
