from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi


ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """English words plus Chinese character unigrams/bigrams.

    The tokenizer is deterministic and has no external model dependency, so it
    can run unchanged inside an isolated corporate network.
    """
    normalized = (text or "").lower()
    tokens = ASCII_TOKEN_RE.findall(normalized)
    for span in CHINESE_RE.findall(normalized):
        tokens.extend(span)
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    return tokens


@dataclass(frozen=True)
class RetrievedMessage:
    rank: int
    score: float
    message: dict[str, Any]


class BM25MemorySystem:
    method_id = "bm25_rag"
    display_name = "BM25 RAG"
    version = "chinese_char_bigram_v2_temporal"

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        corpus = [tokenize(str(message.get("content") or "")) for message in messages]
        self._index = BM25Okapi(corpus)

    def retrieve(self, question: dict[str, Any], top_k: int) -> list[RetrievedMessage]:
        query_user = str((question.get("query_context") or {}).get("query_user_id") or "")
        query = f"{query_user} {question.get('question') or ''}".strip()
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        cutoff = (question.get("temporal_scope") or {}).get("as_of")
        eligible = [
            index
            for index, message in enumerate(self.messages)
            if not cutoff
            or not message.get("timestamp")
            or str(message["timestamp"]) <= str(cutoff)
        ]
        order = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)
        return [
            RetrievedMessage(
                rank=rank,
                score=float(scores[index]),
                message=self.messages[index],
            )
            for rank, index in enumerate(order[:top_k], start=1)
        ]


class Mem0MemorySystem:
    method_id = "mem0"
    display_name = "Mem0"
    version = "mem0_v2_deepseek_bge_zh_v1"


class TeamAgentMemorySystem:
    method_id = "teamagent"
    display_name = "TeamAgent Memory"
    version = "teamagent_l2_bgem3_v1"


class TeamAgentBM25MemorySystem:
    method_id = "teamagent_bm25"
    display_name = "TeamAgent + BM25"
    version = "teamagent_l2_bm25_v1"


class MindMemosMemorySystem:
    method_id = "mindmemos"
    display_name = "MindMemOS"
    version = "mindmemos_http_v3_checkpoint_temporal_scoped"


class EverOSMemorySystem:
    method_id = "everos"
    display_name = "EverOS (Hybrid + BGE-M3)"
    version = "everos_http_v3_event_checkpoint_bgem3"


MEMORY_SYSTEMS = {
    BM25MemorySystem.method_id: BM25MemorySystem,
    Mem0MemorySystem.method_id: Mem0MemorySystem,
    TeamAgentMemorySystem.method_id: TeamAgentMemorySystem,
    TeamAgentBM25MemorySystem.method_id: TeamAgentBM25MemorySystem,
    MindMemosMemorySystem.method_id: MindMemosMemorySystem,
    EverOSMemorySystem.method_id: EverOSMemorySystem,
}
