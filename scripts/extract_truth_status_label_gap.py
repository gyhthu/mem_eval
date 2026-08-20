"""Extract TeamAgent cases where gold `truth.status` is a schema label, not chat text.

The benchmark copies `event.truth.status` into the gold answer. That field is an
internal episode-schema token (often English, often never spoken). This script
checks whether that token appears in:

  - any same-episode message visible at query time
  - the TeamAgent L2 summary
  - the retrieved TopK messages
  - the Reader answer

Default run (TeamAgent + BM25 Top10):

    PYTHONPATH=src .venv/bin/python scripts/extract_truth_status_label_gap.py

Writes:
    results/eval_platform/analysis/truth_status_label_gap.json
    results/eval_platform/analysis/truth_status_label_gap.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/eval_platform/runs"
QUESTIONS = (
    REPO / "results/feishu_generator/v1/06_eval_batch_150_2026-08-16/questions_aggregate.json"
)
EPISODES_DIR = REPO / "results/feishu_generator/v1/06_eval_batch_150_2026-08-16/episodes"
LEGACY_EPISODE = REPO / "results/feishu_generator/v1/05_pilot_0004/episode_from_prototype_0004.json"
OUT_DIR = REPO / "results/eval_platform/analysis"

DEFAULT_RUN = "20260818T023423781362Z_TeamAgent_BM25_Top10_DeepSeek_V4_Flash.json"


def truth_field(question: dict[str, Any]) -> str:
    path = (question.get("oracle_paths") or [""])[0]
    return ".".join(str(path).split(".")[1:]) or "(none)"


def gold_text(question: dict[str, Any]) -> str:
    return str((question.get("oracle") or {}).get("answer"))


def load_episodes() -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    for path in sorted(EPISODES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes[str(payload["episode_id"])] = payload
    legacy = json.loads(LEGACY_EPISODE.read_text(encoding="utf-8"))
    episodes.setdefault(str(legacy["episode_id"]), legacy)
    return episodes


def event_for(episode: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    for event in episode.get("events") or []:
        if str(event.get("event_id")) == str(event_id):
            return event
    return None


def visible_messages(episode: dict[str, Any], cutoff: str | None) -> list[dict[str, Any]]:
    messages = list(episode.get("messages") or [])
    if not cutoff:
        return messages
    return [m for m in messages if not m.get("timestamp") or str(m["timestamp"]) <= cutoff]


def occurrences(text: str, needle: str) -> bool:
    return bool(needle) and needle in (text or "")


def clip(text: str | None, limit: int = 280) -> str:
    value = (text or "").replace("\n", " / ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def classify(case: dict[str, Any]) -> str:
    in_chat = case["gold_in_visible_chat"]
    in_reader = case["gold_in_reader"]
    correct = case["answer_correct"]
    if not in_chat and not correct:
        return "A. gold-not-in-chat and judged wrong"
    if not in_chat and correct:
        return "B. gold-not-in-chat but judged correct (lucky paraphrase)"
    if in_chat and in_reader and not correct:
        return "C. gold WAS spoken; Reader said it; Judge still rejected"
    if in_chat and not in_reader and not correct:
        return "D. gold WAS spoken; Reader missed it"
    if in_chat and correct:
        return "E. gold WAS spoken and judged correct"
    return "F. other"


def build_case(
    *,
    question: dict[str, Any],
    row: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    gold = gold_text(question)
    cutoff = str(
        (question.get("temporal_scope") or {}).get("as_of")
        or (question.get("query_context") or {}).get("query_time")
        or ""
    )
    messages = visible_messages(episode, cutoff or None)
    evidence_ids = [str(x) for x in question.get("evidence_message_ids") or []]
    event_ids = [str(x) for x in question.get("evidence_event_ids") or []]
    events = [event_for(episode, eid) for eid in event_ids]
    events = [e for e in events if e is not None]

    chat_hits = [
        {
            "message_id": m.get("message_id"),
            "author_id": m.get("author_id"),
            "timestamp": m.get("timestamp"),
            "content": m.get("content"),
        }
        for m in messages
        if occurrences(str(m.get("content") or ""), gold)
    ]
    evidence_msgs = [
        {
            "message_id": m.get("message_id"),
            "author_id": m.get("author_id"),
            "timestamp": m.get("timestamp"),
            "content": m.get("content"),
        }
        for m in messages
        if str(m.get("message_id")) in set(evidence_ids)
    ]
    retrieved = list(row.get("retrieved") or [])
    topk_hits = [
        {
            "rank": item.get("rank"),
            "message_id": item.get("message_id"),
            "author_id": item.get("author_id"),
            "content": item.get("content"),
        }
        for item in retrieved
        if occurrences(str(item.get("content") or ""), gold)
    ]

    case = {
        "question_id": question["question_id"],
        "episode_id": question["episode_id"],
        "primary_memory_type": question["primary_memory_type"],
        "question": question["question"],
        "oracle_paths": question.get("oracle_paths") or [],
        "evidence_event_ids": event_ids,
        "evidence_message_ids": evidence_ids,
        "gold": gold,
        "event_truth": [e.get("truth") or {} for e in events],
        "event_type": [e.get("type") for e in events],
        "query_time": cutoff or None,
        "answer_correct": bool(row.get("answer_correct")),
        "hit_at_k": bool(row.get("hit_at_k")),
        "evidence_recall_at_k": row.get("evidence_recall_at_k"),
        "gold_in_visible_chat": bool(chat_hits),
        "gold_in_evidence_messages": any(
            occurrences(str(m.get("content") or ""), gold) for m in evidence_msgs
        ),
        "gold_in_l2": occurrences(str(row.get("memory_context") or ""), gold),
        "gold_in_topk": bool(topk_hits),
        "gold_in_reader": occurrences(str(row.get("reader_answer") or ""), gold),
        "chat_hit_message_ids": [m["message_id"] for m in chat_hits],
        "topk_hit_message_ids": [m["message_id"] for m in topk_hits],
        "reader_answer": row.get("reader_answer"),
        "judge_reasoning": row.get("judge_reasoning"),
        "l2_summary": row.get("memory_context"),
        "evidence_messages": evidence_msgs,
        "retrieved_preview": [
            {
                "rank": item.get("rank"),
                "message_id": item.get("message_id"),
                "author_id": item.get("author_id"),
                "content": item.get("content"),
            }
            for item in retrieved[:6]
        ],
    }
    case["bucket"] = classify(case)
    return case


def render_markdown(cases: list[dict[str, Any]], run_name: str) -> str:
    buckets = Counter(c["bucket"] for c in cases)
    golds = Counter(c["gold"] for c in cases)
    lines = [
        f"# `truth.status` gold-label gap  ({run_name})",
        "",
        "Gold answers for these questions are copied from `event.truth.status`. "
        "That field is an episode-schema token; it is **not** required to appear in chat.",
        "",
        "## Counts",
        "",
        f"- `truth.status` questions: **{len(cases)}**",
        f"- distinct gold labels: **{len(golds)}**",
        f"- gold literally absent from visible chat: "
        f"**{sum(1 for c in cases if not c['gold_in_visible_chat'])} / {len(cases)}**",
        f"- of those, judged wrong: "
        f"**{sum(1 for c in cases if not c['gold_in_visible_chat'] and not c['answer_correct'])}**",
        "",
        "### Bucket",
        "",
    ]
    for bucket, n in sorted(buckets.items()):
        lines.append(f"- {bucket}: {n}")
    lines += ["", "### Gold vocabulary", ""]
    for gold, n in golds.most_common():
        lines.append(f"- `{gold}` × {n}")

    focus = [c for c in cases if c["bucket"] == "A. gold-not-in-chat and judged wrong"]
    lines += ["", "## Cases: gold never spoken, Reader judged wrong", ""]
    for case in focus:
        lines += [
            f"### {case['question_id']}  ·  gold=`{case['gold']}`  ·  {case['primary_memory_type']}",
            "",
            f"- episode: `{case['episode_id']}`",
            f"- oracle: `{', '.join(case['oracle_paths'])}`",
            f"- event type: `{', '.join(str(x) for x in case['event_type'])}`",
            f"- event.truth: `{json.dumps(case['event_truth'], ensure_ascii=False)}`",
            f"- recall@k: {case['evidence_recall_at_k']}",
            f"- gold in L2 / TopK / Reader: "
            f"{case['gold_in_l2']} / {case['gold_in_topk']} / {case['gold_in_reader']}",
            "",
            f"**Q.** {case['question']}",
            "",
            "**Evidence messages (what people actually said)**",
            "",
        ]
        for msg in case["evidence_messages"]:
            lines.append(
                f"- `{msg['message_id']}` {msg['author_id']}: {clip(str(msg.get('content') or ''), 220)}"
            )
        if not case["evidence_messages"]:
            lines.append("- (no evidence messages in episode)")
        lines += [
            "",
            f"**L2.** {clip(case['l2_summary'], 360)}",
            "",
            f"**Reader.** {clip(case['reader_answer'], 360)}",
            "",
            f"**Judge.** {clip(case['judge_reasoning'], 280)}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="run JSON filename under results/eval_platform/runs/")
    args = parser.parse_args()

    questions = {
        q["question_id"]: q
        for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    }
    rows = {
        r["question_id"]: r
        for r in json.loads((RUNS / args.run).read_text(encoding="utf-8"))["rows"]
    }
    episodes = load_episodes()

    cases = []
    for qid, question in sorted(questions.items()):
        if truth_field(question) != "truth.status":
            continue
        if qid not in rows:
            continue
        episode = episodes.get(str(question["episode_id"]))
        if episode is None:
            raise KeyError(f"missing episode {question['episode_id']} for {qid}")
        cases.append(build_case(question=question, row=rows[qid], episode=episode))

    buckets = Counter(c["bucket"] for c in cases)
    golds = Counter(c["gold"] for c in cases)
    absent = [c for c in cases if not c["gold_in_visible_chat"]]
    absent_wrong = [c for c in absent if not c["answer_correct"]]
    absent_right = [c for c in absent if c["answer_correct"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": args.run,
        "n_truth_status": len(cases),
        "n_distinct_gold_labels": len(golds),
        "n_gold_absent_from_chat": len(absent),
        "n_gold_absent_and_wrong": len(absent_wrong),
        "n_gold_absent_and_correct": len(absent_right),
        "buckets": dict(buckets),
        "gold_vocabulary": dict(golds.most_common()),
        "cases": cases,
    }
    json_path = OUT_DIR / "truth_status_label_gap.json"
    md_path = OUT_DIR / "truth_status_label_gap.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(cases, args.run), encoding="utf-8")

    print(f"run: {args.run}")
    print(f"truth.status questions: {len(cases)}")
    print(f"distinct gold labels:   {len(golds)}")
    print(
        f"gold NEVER spoken in visible chat: {len(absent)}/{len(cases)}  "
        f"(wrong {len(absent_wrong)}, lucky-correct {len(absent_right)})"
    )
    print()
    print("buckets:")
    for bucket, n in sorted(buckets.items()):
        print(f"  {n:3d}  {bucket}")
    print()
    print("gold vocabulary:")
    for gold, n in golds.most_common():
        print(f"  {n:2d}  {gold}")
    print()
    print("A. gold-not-in-chat and judged wrong:")
    for case in absent_wrong:
        print(
            f"  {case['question_id']}  gold={case['gold']!r:16s}  "
            f"type={case['primary_memory_type']:22s}  "
            f"recall={case['evidence_recall_at_k']:.2f}  "
            f"in_l2={int(case['gold_in_l2'])} in_topk={int(case['gold_in_topk'])} "
            f"in_reader={int(case['gold_in_reader'])}"
        )
    print()
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
