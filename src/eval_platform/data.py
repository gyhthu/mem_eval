from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
# v2 is the active question set. Episodes are copied from the frozen v1 batch.
DEFAULT_QUESTIONS = (
    REPO_ROOT / "results/feishu_generator/v2/questions_aggregate.json"
)
DEFAULT_EPISODES_DIR = REPO_ROOT / "results/feishu_generator/v2/episodes"
DEFAULT_LEGACY_EPISODE = (
    REPO_ROOT
    / "results/feishu_generator/v1/05_pilot_0004/episode_from_prototype_0004.json"
)
DEFAULT_RUNS_DIR = REPO_ROOT / "results/eval_platform/runs"


def configured_path(env_key: str, default: Path) -> Path:
    value = os.environ.get(env_key)
    return Path(value).expanduser().resolve() if value else default.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class BenchmarkData:
    questions: list[dict[str, Any]]
    episodes: dict[str, dict[str, Any]]
    questions_path: Path

    @property
    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions:
            key = str(question.get("primary_memory_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def message_count(self) -> int:
        return sum(len(episode.get("messages") or []) for episode in self.episodes.values())


def load_benchmark(
    *,
    questions_path: Path | None = None,
    episodes_dir: Path | None = None,
    legacy_episode_path: Path | None = None,
) -> BenchmarkData:
    questions_path = questions_path or configured_path(
        "GROUPMEMBENCH_QUESTIONS", DEFAULT_QUESTIONS
    )
    episodes_dir = episodes_dir or configured_path(
        "GROUPMEMBENCH_EPISODES_DIR", DEFAULT_EPISODES_DIR
    )
    legacy_episode_path = legacy_episode_path or configured_path(
        "GROUPMEMBENCH_LEGACY_EPISODE", DEFAULT_LEGACY_EPISODE
    )
    payload = load_json(questions_path)
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"question file contains no questions: {questions_path}")

    episode_paths = sorted(episodes_dir.glob("*.json"))
    if legacy_episode_path.exists():
        episode_paths.append(legacy_episode_path)
    episodes: dict[str, dict[str, Any]] = {}
    for path in episode_paths:
        episode = load_json(path)
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        if episode_id in episodes:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode["_source_path"] = str(path)
        episodes[episode_id] = episode

    missing = sorted(
        {
            str(question.get("episode_id") or "")
            for question in questions
            if str(question.get("episode_id") or "") not in episodes
        }
    )
    if missing:
        raise ValueError(f"missing episodes for questions: {missing}")
    return BenchmarkData(
        questions=[dict(item) for item in questions],
        episodes=episodes,
        questions_path=questions_path,
    )


def runs_dir() -> Path:
    path = configured_path("GROUPMEMBENCH_RUNS_DIR", DEFAULT_RUNS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def answer_text(question: dict[str, Any]) -> str:
    answer = (question.get("oracle") or {}).get("answer_display")
    if answer is None:
        answer = (question.get("oracle") or {}).get("answer")
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def message_lookup(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(message["message_id"]): message
        for message in episode.get("messages") or []
        if message.get("message_id")
    }

