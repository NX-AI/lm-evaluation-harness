"""BFCL-v3 overall accuracy from the per-category scores of the official ``bfcl evaluate``.

    python scripts/bfcl_v3/bfcl_v3_overall.py SCORE_DIR [--out summary.json]

``SCORE_DIR`` is BFCL's ``score`` directory (``$BFCL_PROJECT_ROOT/score``) holding one model's
``BFCL_v4_<category>_score.json`` files (first line: accuracy, correct_count, total_count).

The current bfcl-eval package (v4) weights agentic categories into its "Overall Acc". This
script applies the v3 formula instead, as in ``eval_runner_helper.py`` of
ShishirPatil/gorilla at 65a1c94 (the last commit before the v4 formula):

* Non-Live = unweighted mean of [simple, multiple, parallel, parallel_multiple, irrelevance],
  where simple is the unweighted mean of simple_python, simple_java and simple_javascript;
* Live = mean of [live_simple, live_multiple, live_parallel, live_parallel_multiple,
  live_irrelevance, live_relevance] weighted by entry counts;
* Multi-Turn = unweighted mean of [base, miss_func, miss_param, long_context];
* Overall = unweighted mean of [Live, Non-Live, Multi-Turn].

Every category must be present: v3 counts a missing category as 0, which would hide an
incomplete run.
"""

import argparse
import json
from pathlib import Path

NON_LIVE_SIMPLE = ("simple_python", "simple_java", "simple_javascript")
NON_LIVE_OTHER = ("multiple", "parallel", "parallel_multiple", "irrelevance")
LIVE = ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance",
        "live_relevance")
MULTI_TURN = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context")
V3_CATEGORIES = NON_LIVE_SIMPLE + NON_LIVE_OTHER + LIVE + MULTI_TURN


def load_scores(score_dir):
    scores = {}
    for path in Path(score_dir).rglob("*_score.json"):
        category = path.name.split("_", 2)[2].removesuffix("_score.json")  # BFCL_v4_<category>_score.json
        with open(path) as f:
            header = json.loads(f.readline())
        if category in scores:
            raise SystemExit(f"{category}: several score files under {score_dir} (one model per directory)")
        scores[category] = header
    return scores


def unweighted(values):
    return sum(values) / len(values)


def v3_summary(scores):
    missing = [c for c in V3_CATEGORIES if c not in scores]
    if missing:
        raise SystemExit(f"missing v3 categories: {missing}")
    acc = {c: scores[c]["accuracy"] for c in V3_CATEGORIES}
    simple = unweighted([acc[c] for c in NON_LIVE_SIMPLE])
    non_live = unweighted([simple] + [acc[c] for c in NON_LIVE_OTHER])
    non_live_ast = unweighted([simple, acc["multiple"], acc["parallel"], acc["parallel_multiple"]])
    counts = {c: scores[c]["total_count"] for c in LIVE}
    live = sum(acc[c] * counts[c] for c in LIVE) / sum(counts.values())
    live_ast_cats = LIVE[:4]
    live_ast = sum(acc[c] * counts[c] for c in live_ast_cats) / sum(counts[c] for c in live_ast_cats)
    multi_turn = unweighted([acc[c] for c in MULTI_TURN])
    overall = unweighted([live, non_live, multi_turn])
    return {
        "overall_v3": overall,
        "non_live": non_live,
        "non_live_ast": non_live_ast,
        "live": live,
        "live_ast": live_ast,
        "multi_turn": multi_turn,
        "irrelevance": unweighted([acc["irrelevance"], acc["live_irrelevance"]]),
        "relevance": acc["live_relevance"],
        "categories": {c: dict(scores[c]) for c in V3_CATEGORIES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("score_dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    summary = v3_summary(load_scores(args.score_dir))
    for key in ("overall_v3", "non_live", "non_live_ast", "live", "live_ast", "multi_turn", "irrelevance", "relevance"):
        print(f"{key:14s} {100 * summary[key]:6.2f}")
    for category, header in summary["categories"].items():
        print(f"  {category:26s} {100 * header['accuracy']:6.2f}  ({header['correct_count']}/{header['total_count']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
