#!/usr/bin/env python3
"""compare_clubbing.py — granular club=1 (1 primitive/block) vs club=2 (2 primitives clubbed/block).

Reads grounding_out_c1/results.json and grounding_out_c2/results.json (produced by grounding_test.py)
and prints, PER CATEGORY and overall, the four probe metrics side by side with the club delta:

  acc_plan    gold plan  -> gold answer   (the plan rescues the problem)
  neg_follow  wrong plan -> wrong answer  HIGH = the model FOLLOWS the plan   <- grounding
  neg_to_gold wrong plan -> gold answer   LOW  = the model does NOT ignore it <- grounding
  acc_noplan  no plan    -> gold answer   (how often the plan is even NEEDED)

delta = c2 - c1. A big negative neg_follow delta or big positive neg_to_gold delta = clubbing two
primitives into one block HURTS following.

FAIRNESS: c1 and c2 are the SAME 100 problems/answers (clubbing only rebundles blocks), and acc_noplan
is a no-plan decode (plan-independent), so per-category and conditional comparisons are apples-to-apples.
per_topic is over ALL rows; the OVERALL CONDITIONAL row restricts to rows the model fails unaided
(acc_noplan==0) — the honest grounding number.
"""
import json, os, sys

def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run grounding_test.py for that club level first")
    return json.load(open(path))

c1 = load("grounding_out_c1/results.json")
c2 = load("grounding_out_c2/results.json")

def pct(x):
    try:    return f"{x:5.0%}"
    except: return "  n/a"

def cell(m1, m2, key, delta=True):
    v1, v2 = m1.get(key), m2.get(key)
    s = f"{pct(v1)} {pct(v2)}"
    if delta and v1 is not None and v2 is not None:
        s += f" {v2 - v1:+5.0%}"
    elif delta:
        s += "      "
    return s

# ---- marker_rate sanity (truncation guard) ----
print("="*100)
for tag, d in (("club1", c1), ("club2", c2)):
    mr = d.get("marker_rate", 0.0)
    flag = "" if mr >= 0.9 else "  <-- LOW: raise --max_new_tokens & rerun; metrics unreliable"
    print(f"[{tag}] marker_rate={mr:.0%}  rows={d.get('n_rows','?')}  dtype={d.get('dtype','?')}{flag}")

# ---- per-category table ----
hdr = f"{'category':<19}{'n':>3}  | {'neg_follow (HIGH=good)':^20} | {'neg_to_gold (LOW=good)':^20} | {'acc_plan':^12} | {'acc_noplan':^12}"
print("\nGRANULAR PER-CATEGORY  (each metric: club1  club2  Δ)")
print(hdr)
print(f"{'':<22}  | {'c1':>5} {'c2':>5} {'Δ':>5} | {'c1':>5} {'c2':>5} {'Δ':>5} | {'c1':>5} {'c2':>5} | {'c1':>5} {'c2':>5}")
print("-"*100)
pt1, pt2 = c1.get("per_topic", {}), c2.get("per_topic", {})
for t in sorted(set(pt1) | set(pt2)):
    m1, m2 = pt1.get(t, {}), pt2.get(t, {})
    n = m1.get("n", m2.get("n", "?"))
    print(f"{t:<19}{n:>3}  | {cell(m1,m2,'neg_follow')} | {cell(m1,m2,'neg_to_gold')} | "
          f"{cell(m1,m2,'acc_plan',delta=False)} | {cell(m1,m2,'acc_noplan',delta=False)}")

# ---- overall rows ----
print("-"*100)
o1, o2 = c1.get("overall", {}), c2.get("overall", {})
print(f"{'OVERALL (all rows)':<19}{o1.get('n','?'):>3}  | {cell(o1,o2,'neg_follow')} | "
      f"{cell(o1,o2,'neg_to_gold')} | {cell(o1,o2,'acc_plan',delta=False)} | {cell(o1,o2,'acc_noplan',delta=False)}")
cc1, cc2 = c1.get("conditional", {}), c2.get("conditional", {})
cn = c1.get("conditional_n", c2.get("conditional_n", "?"))   # same fails-unaided subset in both
print(f"{'OVERALL (cond.)':<19}{cn:>3}  | {cell(cc1,cc2,'neg_follow')} | "
      f"{cell(cc1,cc2,'neg_to_gold')} | {cell(cc1,cc2,'acc_plan',delta=False)} |  fails-unaided")
print("  ^ CONDITIONAL = only the rows the model fails unaided (acc_noplan==0); this is the honest grounding number.")

# ---- verdict + takeaway ----
print("="*100)
print(f"club1 verdict: {c1.get('verdict','?')}    club2 verdict: {c2.get('verdict','?')}")
df = (cc2.get('neg_follow') or 0) - (cc1.get('neg_follow') or 0)
dg = (cc2.get('neg_to_gold') or 0) - (cc1.get('neg_to_gold') or 0)
print(f"\nCOST OF CLUBBING (conditional):  neg_follow Δ={df:+.0%}   neg_to_gold Δ={dg:+.0%}")
if df > -0.10 and dg < 0.10:
    print("  -> clubbing 2 primitives per block costs little: the frozen model handles bundled blocks ~as well as single ops.")
else:
    print("  -> clubbing HURTS: the model loses the 2nd bundled primitive. Keep primitives one-per-block.")
print("Read per-category: a category only tells you about grounding where acc_noplan is LOW (the plan is needed).")
