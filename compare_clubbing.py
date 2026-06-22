#!/usr/bin/env python3
"""compare_clubbing.py — the COST OF CLUBBING.

Compares grounding_out_c1/results.json (1 op/block) vs grounding_out_c2/results.json (2 clubbed ops/block).
The honest metric is the CONDITIONAL one: among rows the FROZEN model fails UNAIDED (acc_noplan==0), does it
(a) FOLLOW a wrong plan to the wrong answer (neg_follow, HIGH=grounding) and (b) avoid ignoring the plan and
solving anyway (neg_to_gold, LOW=grounding). We print overall + per-topic for c1 and c2 side by side, with
delta = c2 - c1 (negative neg_follow delta and/or positive neg_to_gold delta = clubbing HURTS grounding).

FAIRNESS: acc_noplan is a NO-PLAN decode (build_prompt(...turns=None...)) -> plan-independent. c1 and c2 share
identical problems/topics (clubbing rebundles turns, not the problem text), so the acc_noplan==0 conditional
subset is the SAME population in both runs. grounding_test.py already restricts its 'conditional' aggregate to
acc_noplan==0 rows, so we read those numbers directly -> apples-to-apples.
"""
import json, os, sys

def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run grounding_test.py for that club level first")
    return json.load(open(path))

c1 = load("grounding_out_c1/results.json")
c2 = load("grounding_out_c2/results.json")

KEYS = ("neg_follow", "neg_to_gold")

def row(label, m1, m2):
    out = [f"  {label:<20}"]
    for k in KEYS:
        v1 = m1.get(k, float('nan')); v2 = m2.get(k, float('nan'))
        d = v2 - v1
        out.append(f"{k}: c1={v1:6.2%} c2={v2:6.2%} d={d:+6.2%}")
    n1 = m1.get('n', '?'); n2 = m2.get('n', '?')
    out.append(f"[n: c1={n1} c2={n2}]")
    return "  ".join(out)

# sanity: marker_rate must be healthy in BOTH, else metrics are unreliable (raise --max_new_tokens & rerun)
for tag, d in (("c1", c1), ("c2", c2)):
    mr = d.get("marker_rate", 0.0)
    flag = "" if mr >= 0.9 else "  <-- LOW: raise --max_new_tokens and rerun; metrics unreliable"
    print(f"[{tag}] marker_rate={mr:.0%}  rows={d.get('n_rows','?')}  base={d.get('base','?')}{flag}")

# ---- OVERALL CONDITIONAL (the headline cost of clubbing) ----
print("\n==== CONDITIONAL (rows the model FAILS unaided; acc_noplan==0) — the honest test ====")
print("  HIGH neg_follow + LOW neg_to_gold = grounding. delta = c2 - c1 = COST OF CLUBBING.")
cc1, cc2 = c1.get("conditional", {}), c2.get("conditional", {})
print(row("OVERALL", cc1, cc2))
print(f"    (conditional_n: c1={c1.get('conditional_n','?')}  c2={c2.get('conditional_n','?')}; "
      f"need >=10 for a trustworthy verdict)")

# ---- PER-TOPIC CONDITIONAL ----
# results.json 'per_topic' is over ALL rows; for an honest per-topic conditional we'd need per-topic acc_noplan==0
# subsets. The probe only stores conditional in aggregate, so per-topic here is the ALL-ROWS view (still a valid
# club1-vs-club2 comparison on an identical row population, just not restricted to fails-unaided). We print it as
# the per-topic delta and ALSO surface acc_noplan per topic so you can see where the plan is actually load-bearing.
print("\n==== PER-TOPIC (all rows; same 8 topics, same problems across c1/c2) ====")
pt1, pt2 = c1.get("per_topic", {}), c2.get("per_topic", {})
for t in sorted(set(pt1) | set(pt2)):
    m1, m2 = pt1.get(t, {}), pt2.get(t, {})
    line = row(t, m1, m2)
    # annotate how often the plan is needed in this topic (low acc_noplan => clubbing delta is meaningful here)
    np1 = m1.get('acc_noplan', float('nan')); np2 = m2.get('acc_noplan', float('nan'))
    print(line + f"  acc_noplan: c1={np1:.0%} c2={np2:.0%}")

# ---- VERDICTS side by side ----
print("\n==== VERDICTS ====")
print(f"  c1 conditional verdict: {c1.get('verdict','?')}   (raw: {c1.get('raw_verdict','?')})")
print(f"  c2 conditional verdict: {c2.get('verdict','?')}   (raw: {c2.get('raw_verdict','?')})")
df = cc2.get('neg_follow', float('nan')) - cc1.get('neg_follow', float('nan'))
dg = cc2.get('neg_to_gold', float('nan')) - cc1.get('neg_to_gold', float('nan'))
print(f"\n  COST OF CLUBBING (conditional): neg_follow delta={df:+.2%} (want ~0; negative=clubbing hurts following), "
      f"neg_to_gold delta={dg:+.2%} (want ~0; positive=clubbing makes it ignore the plan).")
print("  Interpretation: if c2 neg_follow stays >=0.50 and neg_to_gold stays <=0.30, the frozen model handles "
      "2 clubbed ops per block about as well as 1 -> clubbing is ~free. Large negative neg_follow delta or large "
      "positive neg_to_gold delta = clubbing degrades grounding (the executor loses the second bundled op).")
