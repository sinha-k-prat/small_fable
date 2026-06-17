#!/usr/bin/env python3
"""
reward_paths.py — A1b routing. Every prompt is tagged with a reward_path so RL uses the RIGHT
reward + the RIGHT prompt-weighting, and so we NEVER force a soft task through a fake binary checker
(which would reward keyword-stuffing and MGPO would faithfully amplify it).

  verifiable : exact-match / unit-test / constraint-check  -> binary r_i in {0,1}.  weight = mgpo_weight(p_q)
  rubric     : fraction of rubric items satisfied           -> graded r_i in [0,1]. weight = variance_weight
  judge      : LLM-as-judge normalized to [0,1] (noisy, gameable -> last resort, average several).

Routing is by task family (category). The 4 soft families are routed to `rubric`; everything else
is verifiable. Use `--exclude_rubric` in RL to HOLD soft families out entirely and rely on SFT for
them (also spec-sanctioned) if you don't trust the rubric.
"""

# Soft / open-ended families: no crisp single correct answer.
RUBRIC_FAMILIES = {"research_synthesis", "planning", "ambiguity", "adversarial_constraint"}
# (proof, counterfactual carry exact checkers in this dataset -> verifiable.)

# A per-family rubric: distinctive content tokens an acceptable answer should touch. For the
# bundled SYNTHETIC data these reduce to a small checklist; replace with real per-family rubrics
# (3-5 items) when you swap in real tasks. Graded reward = fraction of items present.
RUBRIC_ITEMS = {
    "research_synthesis": ["evidence", "compare", "conclusion"],
    "planning":           ["step", "order", "goal"],
    "ambiguity":          ["interpretation", "depends", "clarify"],
    "adversarial_constraint": ["constraint", "satisfy", "conflict"],
}


def reward_path_for_row(row):
    """Return 'verifiable' | 'rubric' | 'judge' for a dataset row.
    Honors an explicit row['reward_path'] if present, else routes by category."""
    if row.get("reward_path"):
        return row["reward_path"]
    return "rubric" if row.get("category") in RUBRIC_FAMILIES else "verifiable"


if __name__ == "__main__":
    assert reward_path_for_row({"category": "arithmetic"}) == "verifiable"
    assert reward_path_for_row({"category": "planning"}) == "rubric"
    assert reward_path_for_row({"category": "x", "reward_path": "judge"}) == "judge"
    print("reward_paths: OK")
