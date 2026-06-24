#!/usr/bin/env python3
"""
hard_bench_selftest.py — calibration guard for hard_bench.jsonl (NO model needed).

The user's goal is a benchmark a frozen 1.5B scores ~0% on. Two failure modes break that:
  (1) lenient grading -> a wrong/guessing answer scores 1.0 (false positive);
  (2) a small answer space -> a non-reasoning model that BLURTS a prior guess hits some rows.

This script checks both WITHOUT running the model:
  - canonical answer must score 100% under the STRICT grader (sanity);
  - no realistic BLIND GUESS may score above ~0% per category (the guess floor).

The STRICT grader is what hard_bench_run.py uses: a numeric answer is correct iff the committed
span contains exactly ONE distinct number equal to the canonical (kills list/range gaming); a string
answer iff the normalized committed span EQUALS the canonical (kills substring/echo gaming).

The realistic blind-guess set = small integers (0-15) + salient round numbers (what a model actually
blurts when it can't compute) + common one-word string guesses. If every answer sits ABOVE that set,
guessing scores 0 and only genuine computation can score — which a 1.5B can't do on these.
"""
import json, re, collections, sys, os

def _norm(s):  return re.sub(r"[^a-z0-9]", "", str(s).lower())
def _nums(s):  return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", str(s))]

def strict_grade(committed, row):
    gold = row["answer"]
    if row["checker_kind"] == "numeric":
        ns = set(_nums(committed))
        return 1.0 if len(ns) == 1 and abs(next(iter(ns)) - float(gold)) < 1e-9 else 0.0
    return 1.0 if _norm(committed) == _norm(gold) else 0.0

NUM_GUESSES = [str(x) for x in range(0, 16)] + ["20", "25", "30", "50", "100", "1000"]
STR_GUESSES = ["", "0", "abc", "monday", "the", "a", "cat", "no", "yes", "none", "true", "false"]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "hard_bench.jsonl"
    rows = [json.loads(l) for l in open(path)]
    bycat = collections.defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)

    canon = sum(strict_grade(r["answer"], r) for r in rows) / len(rows)
    print(f"rows={len(rows)}  categories={len(bycat)}  (want 20 x 10)")
    print(f"canonical under STRICT grader: {canon:.0%}  (must be 100%)\n")

    bad = []
    print(f"{'category':28} {'best blind guess':>16}")
    for cat in sorted(bycat):
        cr = bycat[cat]
        guesses = NUM_GUESSES if cr[0]["checker_kind"] == "numeric" else STR_GUESSES
        best = max(sum(strict_grade(g, r) for r in cr) / len(cr) for g in guesses)
        flag = "  <-- GUESSABLE (harden)" if best > 0.05 else ""
        if best > 0.05:
            bad.append(cat)
        print(f"{cat:28} {best:>15.0%}{flag}")

    ok = (canon == 1.0 and len(bycat) == 20 and all(len(v) == 10 for v in bycat.values()) and not bad)
    print("\n" + ("PASS: canonical 100%, every category guess-floor ~0%, exactly 20x10."
                  if ok else f"FAIL: {('guessable: '+', '.join(bad)) if bad else 'structure/canonical issue'}"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
