#!/usr/bin/env python3
"""
plot_grounding_results.py — visualize the grounding-probe findings.

Two figures (headless, saved to assets/):
  1) assets/grounding_quadrant.png   — FOLLOWING (y) x CAPABILITY (x), each problem class placed.
     Hypothesis encoded in the axes: FOLLOWING is governed by ATTENTION (does the model weight the
     plan?), CAPABILITY by the MLP (can it compute the operation?). Three clusters emerge.
  2) assets/grounding_trajectory.png — what happens as instructions go GENERIC -> CONCRETE
     ("heavily contextual"): per-family neg_follow, generic vs concrete, + the overall conditional.

Numbers are the MEASURED frozen-Qwen2.5-1.5B-Instruct results (no training), from the runs in this
repo's history:
  GENERIC  = v2.1 clear generic blocks  (grounding_blocks_c1.jsonl)
  CONCRETE = v1 concrete plans          (grounding_test_data.jsonl)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- measured data: GENERIC (clear v2.1, club1): (acc_noplan, neg_follow, neg_to_gold) in % -----
GENERIC = {
    "multi_hop_lookup":  (100, 54,  0),
    "set_ops":           ( 85, 15, 77),
    "categorize_rule":   ( 85, 15, 69),
    "conditional_reco":  ( 85, 15, 38),
    "scheduling":        ( 69, 23, 46),
    "comparison_order":  ( 62,  0,  0),
    "constraint_select": ( 33, 11, 56),
    "transitive_logic":  (  8,  8, 38),
}
# ----- measured data: CONCRETE (v1) neg_follow %, for the contextuality trajectory -----
CONCRETE_FOLLOW = {
    "scheduling": 92, "set_ops": 85, "transitive_logic": 62, "constraint_select": 54,
    "conditional_reco": 50, "comparison_order": 31, "categorize_rule": 0, "multi_hop_lookup": 0,
}
CAP_TH, FOL_TH = 50, 35    # capability / following thresholds (quadrant dividers)

def classify(noplan, follow):
    if follow >= FOL_TH and noplan >= CAP_TH: return "follows+capable", "tab:green"
    if follow <  FOL_TH and noplan >= CAP_TH: return "override (attention)", "tab:orange"
    if follow <  FOL_TH and noplan <  CAP_TH: return "capability wall (MLP)", "tab:red"
    return "follows, can't execute", "tab:blue"

# =========================================================================== FIGURE 1: quadrant
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.axvline(CAP_TH, color="gray", lw=1, ls="--"); ax.axhline(FOL_TH, color="gray", lw=1, ls="--")
ax.axvspan(CAP_TH, 105, FOL_TH/100, 1.0, color="tab:green",  alpha=0.06)
ax.axvspan(CAP_TH, 105, 0, FOL_TH/100, color="tab:orange", alpha=0.06)
ax.axvspan(-5, CAP_TH,  0, FOL_TH/100, color="tab:red",    alpha=0.06)
ax.axvspan(-5, CAP_TH,  FOL_TH/100, 1.0, color="tab:blue",  alpha=0.06)

# small display jitter + per-label offset so coincident points (the 85,15 override cluster) are legible
JIT = {"set_ops": (0, 2.5), "categorize_rule": (0, -2.5), "conditional_reco": (-4, 0)}
LOFF = {"set_ops": (6, 4), "categorize_rule": (6, -10), "conditional_reco": (-78, -2),
        "multi_hop_lookup": (-90, 2), "scheduling": (8, 0), "comparison_order": (6, -2),
        "constraint_select": (8, 4), "transitive_logic": (-70, 8)}
for name, (noplan, follow, _) in GENERIC.items():
    jx, jy = JIT.get(name, (0, 0))
    _, c = classify(noplan, follow)
    ax.scatter(noplan + jx, follow + jy, s=160, color=c, edgecolor="black", zorder=3)
    ax.annotate(name, (noplan + jx, follow + jy), xytext=LOFF.get(name, (6, 6)),
                textcoords="offset points", fontsize=9, zorder=4)

bb = dict(boxstyle="round", fc="white", ec="none", alpha=0.65)
ax.text(78, 95, "FOLLOWS  +  CAPABLE\nplan works, model can do it",
        ha="center", color="tab:green", fontsize=10, weight="bold", bbox=bb)
ax.text(98, 33, "OVERRIDE\ncapable but IGNORES the plan\n→ ATTENTION (trainable)",
        ha="right", color="tab:orange", fontsize=10, weight="bold", bbox=bb)
ax.text(2, 33, "CAPABILITY WALL\ncan't compute, can't follow\n→ MLP / skill (bigger model,\nscratchpad)",
        ha="left", color="tab:red", fontsize=10, weight="bold", bbox=bb)
ax.text(2, 95, "follows direction\nbut botches execution\n(rare / empty)",
        ha="left", color="tab:blue", fontsize=9, bbox=bb)

ax.set_xlim(-5, 108); ax.set_ylim(-5, 100)
ax.set_xlabel("CAPABILITY  →  can the model do it unaided? (acc_noplan %)   [governed by the MLP]", fontsize=11)
ax.set_ylabel("FOLLOWING  →  does it follow the plan? (neg_follow %)   [governed by ATTENTION]", fontsize=11)
ax.set_title("Grounding of GENERIC abstract plans, frozen Qwen2.5-1.5B-Instruct\n"
             "Following (attention) and Capability (MLP) are independent axes", fontsize=12)
ax.grid(True, alpha=0.2)
fig.tight_layout(); fig.savefig("assets/grounding_quadrant.png", dpi=140); plt.close(fig)
print("wrote assets/grounding_quadrant.png")

# =========================================================================== FIGURE 2: trajectory
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]})

# (A) per-family slope: GENERIC -> CONCRETE neg_follow
fams = sorted(GENERIC, key=lambda f: CONCRETE_FOLLOW[f])
for f in fams:
    g, c = GENERIC[f][1], CONCRETE_FOLLOW[f]
    col = "tab:green" if c > g else ("tab:red" if c < g else "gray")
    axA.plot([0, 1], [g, c], "-o", color=col, lw=2, ms=7)
    axA.annotate(f, (1, c), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
axA.set_xticks([0, 1]); axA.set_xticklabels(["GENERIC\n(abstract,\nno context words)", "CONCRETE\n(heavily contextual,\nnames the thing)"])
axA.set_ylabel("neg_follow  (does the model follow the plan?) %")
axA.set_ylim(-5, 100); axA.grid(True, alpha=0.2)
axA.set_title("(A) Per family: following rises sharply\nwhen instructions are made heavily contextual")

# (B) overall conditional trajectory across the abstraction spectrum
levels = ["concrete\n(v1)", "terse\ngeneric (v2)", "clear\ngeneric (v2.1)"]
cond_follow = [59, 9, 10]   # conditional neg_follow %, measured
axB.plot(range(3), cond_follow, "-o", color="tab:purple", lw=2.5, ms=10)
for i, v in enumerate(cond_follow):
    axB.annotate(f"{v}%", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=11)
axB.axhline(50, color="gray", ls="--", lw=1); axB.text(2, 52, "grounding threshold", fontsize=8, ha="right")
axB.set_xticks(range(3)); axB.set_xticklabels(levels)
axB.set_ylim(0, 70); axB.set_ylabel("CONDITIONAL neg_follow %  (honest test)")
axB.grid(True, alpha=0.2)
axB.set_title("(B) Overall: decontextualizing the plan\ncollapses grounding (59% → ~10%)")
axB.invert_xaxis()   # left = more contextual

fig.tight_layout(); fig.savefig("assets/grounding_trajectory.png", dpi=140); plt.close(fig)
print("wrote assets/grounding_trajectory.png")
