#!/usr/bin/env python3
"""
analyze_formats.py — which PLAN FORMAT does a FROZEN model react to, per TASK?

This is step (2)+(3) of the experiment. grounding_test.py runs the SAME frozen model on the SAME
problems under SEVERAL plan FORMATS (one --out dir per format), each dropping a results.json with a
per-row `records` list: [{id, topic, n_turns, acc_noplan, acc_plan, neg_follow, neg_to_gold, neg_other}].

We:
  (1) RESTRICT to the rows the model FAILS UNAIDED (acc_noplan == 0) — the rows where the plan is
      LOAD-BEARING (it must matter, because the model can't get the answer from the setup alone).
      acc_noplan is a property of the (problem, no-plan) decode and does NOT depend on the plan
      format, so this fails-unaided subset is IDENTICAL across formats; we assert that by row id.
  (2) print a TASK x FORMAT table of FOLLOWING (neg_follow on the fails-unaided subset): for each
      task, which format does the model actually follow? `concrete` (when supplied) is the
      upper-bound reference column — the established 59% ceiling from concrete English plans.
  (3) for the BEST generic format (highest mean following over tasks), print the per-task 3-way
      wrong-plan split — follow (neg_follow) / ignore (neg_to_gold) / botch (neg_other) — to
      attribute each non-follow to ATTENTION failure (->gold, the override class) vs EXECUTION
      failure (->other, the capability wall).
  (4) save a headless Agg heatmap assets/plan_format_following.png (rows=tasks, cols=formats,
      cell=neg_follow on fails-unaided).

Usage:
  python tools/analyze_formats.py \
      --runs terse=out_terse clear=out_clear club2=out_club2 concrete=out_concrete \
      --out assets/plan_format_following.png

  # or positional dir=label-free form (label inferred from dir basename):
  python tools/analyze_formats.py --runs out_terse out_clear out_club2

`concrete` is treated as the upper-bound reference column (sorted last) when present; it is excluded
from the "best generic format" pick. Pure inference-only analysis: reads JSON, no model, no judge.
"""
import argparse, json, os, sys


# Canonical task order (the 8 grounding families). Rows with an unseen topic are appended after.
TASK_ORDER = ["multi_hop_lookup", "conditional_reco", "categorize_rule", "set_ops",
              "scheduling", "comparison_order", "constraint_select", "transitive_logic"]

REFERENCE_LABEL = "concrete"   # upper-bound column; excluded from the best-generic pick


def _parse_runs(run_args):
    """Each entry is 'label=dir' or just 'dir' (label = basename, with a leading 'out_' stripped).
    Returns an ORDERED list of (label, dir) preserving CLI order."""
    runs = []
    for a in run_args:
        if "=" in a:
            label, d = a.split("=", 1)
        else:
            d = a
            label = os.path.basename(os.path.normpath(d))
            if label.startswith("out_"):
                label = label[len("out_"):]
        runs.append((label.strip(), d.strip()))
    return runs


def _load_records(run_dir):
    """Load results.json from a run dir -> {row_id: record}. Errors are fatal (a missing/old
    results.json means that format wasn't run with the patched runner)."""
    path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(path):
        sys.exit(f"[analyze] no results.json in {run_dir} (run grounding_test.py --out {run_dir})")
    data = json.load(open(path))
    recs = data.get("records")
    if not recs:
        sys.exit(f"[analyze] {path} has no 'records' (re-run with the patched grounding_test.py "
                 "that dumps per-row records)")
    by_id = {}
    for r in recs:
        by_id[str(r["id"])] = r
    return by_id


def _fails_unaided_ids(records_by_format):
    """The fails-unaided subset = rows with acc_noplan==0. acc_noplan is plan-format-independent, so
    the subset must be identical across formats; we take the intersection of ids present in every
    format, verify acc_noplan agrees, and keep those with acc_noplan==0. Warn on any disagreement."""
    labels = list(records_by_format)
    common = None
    for lab in labels:
        ids = set(records_by_format[lab])
        common = ids if common is None else (common & ids)
    common = common or set()

    fails, disagree = set(), 0
    for rid in common:
        vals = {lab: records_by_format[lab][rid]["acc_noplan"] for lab in labels}
        v0 = list(vals.values())[0]
        if any(v != v0 for v in vals.values()):
            disagree += 1
            # acc_noplan should be format-invariant; if a format disagrees (e.g. nondeterminism),
            # treat the row as fails-unaided only if ALL formats agree it's 0.
            if all(v == 0.0 for v in vals.values()):
                fails.add(rid)
            continue
        if v0 == 0.0:
            fails.add(rid)
    if disagree:
        print(f"[analyze] WARNING: acc_noplan disagreed across formats on {disagree} row(s) "
              "(expected format-invariant). Kept a row only if every format scored it 0.")
    return fails


def _task_table(records_by_format, fail_ids, metric):
    """{task: {format: mean(metric over fails-unaided rows of that task)}} plus per-cell n.
    Returns (table, counts, tasks_in_order, formats)."""
    formats = list(records_by_format)
    # discover tasks
    seen = set()
    for lab in formats:
        for rid in fail_ids:
            seen.add(records_by_format[lab][rid]["topic"])
    tasks = [t for t in TASK_ORDER if t in seen] + sorted(t for t in seen if t not in TASK_ORDER)

    table = {t: {} for t in tasks}
    counts = {t: {} for t in tasks}
    for lab in formats:
        recs = records_by_format[lab]
        for t in tasks:
            rows = [recs[rid] for rid in fail_ids if recs[rid]["topic"] == t]
            n = len(rows)
            counts[t][lab] = n
            table[t][lab] = (sum(r[metric] for r in rows) / n) if n else float("nan")
    return table, counts, tasks, formats


def _ordered_formats(formats):
    """Put the REFERENCE_LABEL ('concrete') column LAST so it reads as the right-hand upper bound."""
    gen = [f for f in formats if f != REFERENCE_LABEL]
    return gen + ([REFERENCE_LABEL] if REFERENCE_LABEL in formats else [])


def _fmt_cell(x):
    return "  -- " if x != x else f"{x:5.0%}"   # x!=x => NaN


def main():
    ap = argparse.ArgumentParser(description="Cross-format plan-following analysis (inference-only).")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="format runs as 'label=dir' (or bare 'dir'; label=basename). e.g. "
                         "terse=out_terse clear=out_clear club2=out_club2 concrete=out_concrete")
    ap.add_argument("--out", default="assets/plan_format_following.png",
                    help="heatmap PNG path (rows=tasks, cols=formats, cell=neg_follow on fails-unaided)")
    args = ap.parse_args()

    runs = _parse_runs(args.runs)
    records_by_format = {lab: _load_records(d) for lab, d in runs}
    # preserve CLI order, then move the reference column last
    ordered = _ordered_formats([lab for lab, _ in runs])
    records_by_format = {lab: records_by_format[lab] for lab in ordered}

    fail_ids = _fails_unaided_ids(records_by_format)
    n_total = len(next(iter(records_by_format.values())))
    print(f"[analyze] formats: {ordered}")
    print(f"[analyze] rows/format={n_total}  fails-unaided (acc_noplan==0, plan is load-bearing)="
          f"{len(fail_ids)}  (identical subset across formats)\n")
    if not fail_ids:
        sys.exit("[analyze] no fails-unaided rows — nothing to analyze (the model solves everything "
                 "unaided, so no plan is load-bearing).")

    # ---- (2) TASK x FORMAT following table (neg_follow on fails-unaided) -------------------------
    follow, counts, tasks, formats = _task_table(records_by_format, fail_ids, "neg_follow")
    w = max(len("TASK"), max(len(t) for t in tasks))
    head = f"{'TASK':<{w}}  " + "  ".join(f"{f:>7}" for f in formats) + "   n"
    print("=" * len(head))
    print("FOLLOWING (neg_follow) on fails-unaided rows  —  TASK x FORMAT")
    print("(higher = model FOLLOWED the wrong plan to its wrong answer = grounded; "
          f"'{REFERENCE_LABEL}' col = upper-bound reference)")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for t in tasks:
        n = max(counts[t].values()) if counts[t] else 0
        row = f"{t:<{w}}  " + "  ".join(f"{_fmt_cell(follow[t][f]):>7}" for f in formats) + f"  {n:>3}"
        print(row)
    # column means over tasks (macro-average; ignores empty-cell NaNs)
    def _col_mean(f):
        vals = [follow[t][f] for t in tasks if follow[t][f] == follow[t][f]]
        return sum(vals) / len(vals) if vals else float("nan")
    print("-" * len(head))
    print(f"{'MEAN/task':<{w}}  " + "  ".join(f"{_fmt_cell(_col_mean(f)):>7}" for f in formats))

    # ---- pick the BEST GENERIC format (highest mean following; reference excluded) ---------------
    generic = [f for f in formats if f != REFERENCE_LABEL]
    best = max(generic, key=lambda f: (_col_mean(f) if _col_mean(f) == _col_mean(f) else -1.0)) \
        if generic else formats[0]
    print(f"\n[analyze] best GENERIC format by mean following: '{best}' "
          f"(mean neg_follow over tasks = {_col_mean(best):.0%})")

    # ---- (3) per-task 3-way wrong-plan split for the best format ---------------------------------
    flw, _, _, _ = _task_table(records_by_format, fail_ids, "neg_follow")
    ign, _, _, _ = _task_table(records_by_format, fail_ids, "neg_to_gold")
    bot, _, _, _ = _task_table(records_by_format, fail_ids, "neg_other")
    print(f"\n3-WAY wrong-plan split for best format '{best}' (fails-unaided rows)")
    print("  follow = grounded   |   ignore (->gold) = ATTENTION failure (override class)   |   "
          "botch (->other) = EXECUTION failure (capability wall)")
    sub = f"{'TASK':<{w}}  {'follow':>7}  {'ignore':>7}  {'botch':>7}   n"
    print(sub); print("-" * len(sub))
    for t in tasks:
        n = counts[t].get(best, 0)
        print(f"{t:<{w}}  {_fmt_cell(flw[t][best]):>7}  {_fmt_cell(ign[t][best]):>7}  "
              f"{_fmt_cell(bot[t][best]):>7}  {n:>3}")

    # ---- (4) heatmap -----------------------------------------------------------------------------
    _heatmap(follow, tasks, formats, counts, args.out)


def _heatmap(follow, tasks, formats, counts, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")                       # headless backend BEFORE pyplot
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[analyze] matplotlib/numpy unavailable ({e}); skipping heatmap.")
        return

    M = np.array([[follow[t][f] for f in formats] for t in tasks], dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 * len(formats) + 3.5, 0.6 * len(tasks) + 2.2),
                           constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(formats)))
    xlabels = [f + ("\n(ref)" if f == REFERENCE_LABEL else "") for f in formats]
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    ax.set_title("Plan FOLLOWING (neg_follow) on fails-unaided rows\n"
                 "rows = task, cols = plan format  (brighter = model followed the plan)",
                 fontsize=10)
    for i, t in enumerate(tasks):
        for j, f in enumerate(formats):
            v = M[i, j]
            txt = "--" if v != v else f"{v:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (v == v and v < 0.55) else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("neg_follow", fontsize=9)
    if REFERENCE_LABEL in formats:
        jr = formats.index(REFERENCE_LABEL)
        ax.axvline(jr - 0.5, color="white", lw=2)   # divider before the reference column

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        fig.savefig(out_png, dpi=140)
        print(f"\n[analyze] wrote heatmap {out_png}")
    except Exception as e:
        print(f"[analyze] could not save {out_png}: {e}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()