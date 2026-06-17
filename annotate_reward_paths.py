#!/usr/bin/env python3
"""
annotate_reward_paths.py — add the A1b `reward_path` field to every dataset row, and convert
soft-family rows to a GRADED rubric checker (so RL never routes them through a fake binary check).

  verifiable rows : unchanged checker (exact/contains_all), reward_path="verifiable"
  rubric  rows    : checker_kind -> "rubric", checker_args.items <- RUBRIC_ITEMS[category],
                    reward_path="rubric"  (graded fraction-of-items reward)

Idempotent. Usage:  python annotate_reward_paths.py dataset/sft_100.jsonl dataset/sft_flat.jsonl
"""
import json, sys
from reward_paths import reward_path_for_row, RUBRIC_FAMILIES, RUBRIC_ITEMS


def annotate(path):
    rows = [json.loads(l) for l in open(path)]
    n_rub = 0
    for r in rows:
        path_tag = "rubric" if r.get("category") in RUBRIC_FAMILIES else "verifiable"
        r["reward_path"] = path_tag
        if path_tag == "rubric":
            items = RUBRIC_ITEMS.get(r["category"], [])
            r["checker_kind"] = "rubric"
            r["checker_args"] = {"items": items}
            n_rub += 1
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} rows, {n_rub} routed to rubric (graded), "
          f"{len(rows)-n_rub} verifiable")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["dataset/sft_100.jsonl", "dataset/sft_flat.jsonl"]
    for p in paths:
        annotate(p)
