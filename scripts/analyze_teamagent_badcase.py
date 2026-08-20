"""Badcase breakdown for the TeamAgent + BM25 / BGE-M3 runs.

Answers two questions:
  1. Where are the badcases concentrated (retrieval vs reader, which question type,
     which oracle truth field)?
  2. How much of the loss is a genuine memory failure vs an answer-protocol artifact?

Usage: PYTHONPATH=src .venv/bin/python scripts/analyze_teamagent_badcase.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path("results/eval_platform/runs")
QUESTIONS = Path(
    "results/feishu_generator/v1/06_eval_batch_150_2026-08-16/questions_aggregate.json"
)

FILES = {
    "ta_bm25_top3": "20260818T023258869953Z_TeamAgent_BM25_Top3_DeepSeek_V4_Flash.json",
    "ta_bm25_top10": "20260818T023423781362Z_TeamAgent_BM25_Top10_DeepSeek_V4_Flash.json",
    "ta_bge_top3": "20260816T083154651171Z_TeamAgent_Top3_DeepSeek_V4_Flash.json",
    "ta_bge_top10": "20260816T083426220533Z_TeamAgent_Top10_DeepSeek_V4_Flash.json",
}
BASELINES = {
    "bm25_top3": "20260816T063503976919Z_BM25_Temporal_Top3_DeepSeek_V4_Flash.json",
    "bm25_top10": "20260816T063552931028Z_BM25_Temporal_Top10_DeepSeek_V4_Flash.json",
}

TEAMAGENT = list(FILES)


def load(filename: str) -> dict[str, dict]:
    payload = json.loads((RUNS / filename).read_text(encoding="utf-8"))
    return {row["question_id"]: row for row in payload["rows"]}


runs = {name: load(f) for name, f in {**FILES, **BASELINES}.items()}
meta = {
    q["question_id"]: q
    for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
}
qids = sorted(runs["ta_bm25_top10"])


def truth_field(qid: str) -> str:
    path = (meta[qid].get("oracle_paths") or [""])[0]
    return ".".join(str(path).split(".")[1:]) or "(none)"


def answer_type(qid: str) -> str:
    return (meta[qid].get("oracle") or {}).get("answer_type") or "?"


def gold(qid: str):
    return (meta[qid].get("oracle") or {}).get("answer")


def table(rows_by_label: dict[str, list[str]], configs: list[str], width: int = 40) -> None:
    print(f"{'':{width}}{'n':>4}" + "".join(f"{c:>15}" for c in configs))
    for label, ids in rows_by_label.items():
        line = f"{label:{width}}{len(ids):>4d}"
        for c in configs:
            k = sum(1 for q in ids if runs[c][q]["answer_correct"])
            line += f"{k:>7d}/{len(ids):<3d}{k / len(ids):>4.2f}"
        print(line)


print("=" * 100)
print("1. Overall: accuracy is NOT retrieval-bound at Top10")
for name in TEAMAGENT + list(BASELINES):
    rows = runs[name]
    hit = sum(r["hit_at_k"] for r in rows.values()) / len(rows)
    acc = sum(r["answer_correct"] for r in rows.values()) / len(rows)
    print(f"  {name:16s} hit@k={hit:.3f}  answer_accuracy={acc:.3f}")

print()
print("=" * 100)
print("2. Wrong answers split by whether retrieval actually failed")
for name in TEAMAGENT:
    wrong = [r for r in runs[name].values() if not r["answer_correct"]]
    no_hit = sum(1 for r in wrong if not r["hit_at_k"])
    partial = sum(1 for r in wrong if r["hit_at_k"] and r["evidence_recall_at_k"] < 1.0)
    full = sum(1 for r in wrong if r["evidence_recall_at_k"] >= 1.0)
    print(
        f"  {name:16s} wrong={len(wrong):3d} | retrieval missed={no_hit:3d} "
        f"| partial evidence={partial:3d} | ALL evidence present, still wrong={full:3d}"
    )

print()
print("=" * 100)
print("3. Accuracy by question type")
by_type = defaultdict(list)
for q in qids:
    by_type[meta[q]["primary_memory_type"]].append(q)
table(dict(sorted(by_type.items(), key=lambda kv: -len(kv[1]))), TEAMAGENT, width=28)

print()
print("=" * 100)
print("4. Accuracy by oracle answer_type")
by_atype = defaultdict(list)
for q in qids:
    by_atype[answer_type(q)].append(q)
table(dict(sorted(by_atype.items(), key=lambda kv: -len(kv[1]))), TEAMAGENT, width=28)

print()
print("=" * 100)
print("5. Accuracy by oracle truth-field family  <-- the real concentration")
STATUSY = re.compile(r"status$|_result$|confirmation$|correctness$")


def family(qid: str) -> str:
    field = truth_field(qid).replace("truth.", "")
    if answer_type(qid) == "participant":
        return "B. WHO / participant attribution"
    if field == "status":
        return "A. truth.status (generic stage status)"
    if STATUSY.search(field):
        return "C. other *_status / *_result enum"
    return "D. everything else"


by_family = defaultdict(list)
for q in qids:
    by_family[family(q)].append(q)
table(dict(sorted(by_family.items())), TEAMAGENT)

print()
print("   Share of each config's badcases:")
for name in TEAMAGENT:
    wrong = [q for q in qids if not runs[name][q]["answer_correct"]]
    counts = Counter(family(q) for q in wrong)
    print(f"     {name} (wrong={len(wrong)}):")
    for fam, n in sorted(counts.items()):
        print(f"        {fam:40s} {n:3d}  ({n / len(wrong):5.1%})")

print()
print("=" * 100)
print("6. Are the gold status labels even present in what the Reader saw?")
status_ids = [q for q in qids if truth_field(q) == "truth.status"]
rows = runs["ta_bm25_top10"]
wrong_status = [q for q in status_ids if not rows[q]["answer_correct"]]
in_l2 = sum(1 for q in wrong_status if str(gold(q)) in (rows[q]["memory_context"] or ""))
in_topk = sum(
    1
    for q in wrong_status
    if any(str(gold(q)) in (m.get("content") or "") for m in rows[q]["retrieved"])
)
either = sum(
    1
    for q in wrong_status
    if str(gold(q)) in (rows[q]["memory_context"] or "")
    or any(str(gold(q)) in (m.get("content") or "") for m in rows[q]["retrieved"])
)
print(f"  truth.status questions={len(status_ids)}, wrong={len(wrong_status)}")
print(f"    gold label literally in L2 summary : {in_l2}/{len(wrong_status)}")
print(f"    gold label literally in TopK msgs  : {in_topk}/{len(wrong_status)}")
print(f"    gold label in EITHER               : {either}/{len(wrong_status)}")
vocab = Counter(str(gold(q)) for q in status_ids)
print(f"  distinct gold labels for {len(status_ids)} questions: {len(vocab)} (open vocabulary)")
print(f"    mixed-language near-duplicates, e.g. "
      f"{[v for v in vocab if v in {'submitted', '已提交', 'confirmed', '已确认', 'pending'}]}")

print()
print("=" * 100)
print("7. Judge verdict language: 'not stated' vs real contradiction")
MISSING = re.compile(r"未提及|未明确|未直接|缺失|未覆盖|未准确给出|未给出|语义上未|不等同")
CONTRADICT = re.compile(r"相反|矛盾|人物身份错误|完全相反")
for name in TEAMAGENT:
    wrong = [q for q in qids if not runs[name][q]["answer_correct"]]
    miss = sum(1 for q in wrong if MISSING.search(runs[name][q]["judge_reasoning"] or ""))
    contra = sum(1 for q in wrong if CONTRADICT.search(runs[name][q]["judge_reasoning"] or ""))
    print(
        f"  {name:16s} wrong={len(wrong):3d} | judge cites missing/not-stated={miss:3d} "
        f"| judge cites contradiction={contra:3d}"
    )

print()
print("=" * 100)
print("8. Hard core: wrong in ALL four TeamAgent configs")
hard = [q for q in qids if not any(runs[n][q]["answer_correct"] for n in TEAMAGENT)]
print(f"  count={len(hard)} ({len(hard) / len(qids):.1%} of benchmark)")
print(f"  types    : {dict(Counter(meta[q]['primary_memory_type'] for q in hard))}")
print(f"  families : {dict(Counter(family(q) for q in hard))}")
print(
    f"  Top10 retrieved gold evidence for {sum(runs['ta_bm25_top10'][q]['hit_at_k'] for q in hard)}"
    f"/{len(hard)}; full recall for "
    f"{sum(runs['ta_bm25_top10'][q]['evidence_recall_at_k'] >= 1.0 for q in hard)}/{len(hard)}"
)

print()
print("=" * 100)
print("9. BM25 vs BGE-M3 pairwise (same L2, same candidates, only the ranker differs)")
for k in ("top3", "top10"):
    a, b = runs[f"ta_bm25_{k}"], runs[f"ta_bge_{k}"]
    only_a = [q for q in qids if a[q]["answer_correct"] and not b[q]["answer_correct"]]
    only_b = [q for q in qids if b[q]["answer_correct"] and not a[q]["answer_correct"]]
    both_w = [q for q in qids if not a[q]["answer_correct"] and not b[q]["answer_correct"]]
    both_c = len(qids) - len(only_a) - len(only_b) - len(both_w)
    print(
        f"  {k}: both correct={both_c}, only BM25={len(only_a)}, "
        f"only BGE={len(only_b)}, both wrong={len(both_w)}"
    )
