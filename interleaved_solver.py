#!/usr/bin/env python3
"""
interleaved_solver.py — the FROZEN-model INTERLEAVED harness with a per-category generic-block
variant search + mutation loop, run until each category hits 100% accuracy or a budget is exhausted.

THE STORY (read hard_bench_run.py + GROUNDING_RESULTS.md + ARCHITECTURE_INTERLEAVED.md first)
--------------------------------------------------------------------------------------------
A FROZEN Qwen2.5-1.5B-Instruct scores ~0% on HARD_BENCH: the hard operations (counting, multi-digit
multiplication, sorting, reversal, modular exponentiation, calendar math, base conversion, ...) each
need a LOOP + a mutable register, which one transformer forward pass lacks. The fix is the SCRATCHPAD:
unroll the loop into generated tokens, ONE narrow step at a time.

INTERLEAVED execution does exactly this (ARCHITECTURE_INTERLEAVED.md section 5). For a problem and a
chosen block-variant (an ordered list of GENERIC blocks from tools/interleaved_blocks):

    scratchpad := PROBLEM
    for each generic block in the variant:
        prompt := PROBLEM + scratchpad-so-far + "Carry out ONLY this step and write its result:" + block
        chunk  := greedy-decode(prompt)           # frozen model EXECUTES just this one step
        scratchpad := scratchpad + chunk          # APPEND the result; the register grows
    # the final block instructs 'FINAL ANSWER:' -> the gradeable commitment lives in the last chunk

This is the run_interleaved loop of the canonical spec, executed on a FROZEN model with GENERIC
English blocks (the grounding probe showed a frozen 1.5B follows concrete English step-instructions).
Each block sees only the problem + what was written so far, so it is causally gated by the prior steps
exactly as the spec's `primitive_t = f(instruction, responses_<t)` requires.

INFERENCE ONLY: torch.no_grad + model.eval, NO LoRA, NO optimizer, NO gradients, NO training.

GRADING: judge-free, reusing hard_bench_run.grade() and _FINAL_RE VERBATIM (imported, not reimplemented)
the SAME strict grader hard_bench_run.py uses (numeric -> exactly one distinct number equal to gold;
exact -> normalized committed span EQUALS gold; a missing FINAL ANSWER: scores 0). No false positives.

PER-CATEGORY VARIANT SEARCH + GENERIC MUTATION LOOP
---------------------------------------------------
For each of the 20 categories we take its skill's generic block-variants (the variants of
tools.interleaved_blocks.SKILLS[CATEGORY_SKILL[cat]]) as the starting search space, evaluate each on the
category's rows with the strict grader, keep the best, and KEEP iterating that category — synthesizing
further GENERIC mutations of the best plan (add an explicit double-check/re-do block; a finer
decomposition that externalizes one item/step per line; a magnitude sanity-check) — until the category's
accuracy reaches 100% or a per-category budget (--rounds) is hit. EVERY mutation is GENERIC: it edits
only the abstract block wording, never injects problem-specific words; we ENFORCE that with the library's
generic_violations(block, problem) reuse gate (asserted empty for every inserted block). We report
per-category accuracy climbing across rounds.

At the end we DUMP, to results.json, the FINAL chosen generic interleaved block-plan per category (the
plan that solved it), the per-category accuracy, the overall accuracy, and a headless per-category
accuracy bar chart — argparse identical for SSH/Colab (the same conventions as hard_bench_run.py).

CLI (same on SSH and Colab — plain argparse):
  python interleaved_solver.py --data hard_bench.jsonl --base Qwen/Qwen2.5-1.5B-Instruct \
      --dtype bf16 --device cuda --limit 0 --out interleaved_out \
      --max_new_tokens 320 --rounds 6
"""
import argparse, json, os, re, sys, copy, collections

# Reuse hard_bench_run's STRICT judge-free grader + marker regex + chat-template + headless plot
# conventions VERBATIM (import, do not reimplement) so grading is byte-identical to the benchmark.
import hard_bench_run as HB
from hard_bench_run import grade, _FINAL_RE, _answer_span

# The GENERIC, reusable block library + category->skill map + per-category variants. A "block" here is
# a plain GENERIC English string (one interleaved turn); a "variant"/"plan" is an ordered list of such
# strings whose LAST string carries the 'FINAL ANSWER:' commit. interleaved_blocks also exposes
# generic_violations(block, problem) — the reusability GATE that returns the block's tokens overlapping a
# problem's content (empty for a truly generic block); we use it to assert our mutations stay generic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import interleaved_blocks as IB
from tools.interleaved_blocks import CATEGORY_SKILL, SKILLS, generic_violations
# FROZEN design-time leak audit -> green pool. The hunt SEEDS only from judge-approved (non-leaking)
# variants; a 'leak' variant (e.g. decompose_and_recombine v0's "Multiply ... Add") is dropped whenever
# a generic sibling exists. No judge runs here at hunt time — green_variants() reads frozen verdicts.
from tools.leak_audit import green_variants


def variants_for_category(category):
    """Ordered list of GENERIC block-variants (each a list of plain block strings) for a category's
    skill — the starting search space the per-category mutation loop explores. FILTERED to the frozen
    green pool (tools.leak_audit): leaking variants are excluded when a generic sibling covers the skill,
    so the hunt never seeds from a task-lifted block."""
    skill = CATEGORY_SKILL[category]
    all_variants = SKILLS[skill]["variants"]
    keep = green_variants(skill)
    return [list(all_variants[i]) for i in keep]


def skill_for_category(category):
    return CATEGORY_SKILL[category]


def is_final_block(block_text):
    """The committing block is the one that emits the literal FINAL ANSWER: marker."""
    return "final answer" in block_text.lower()


# ===========================================================================================
# PROMPTING — per-step interleaved prompt. The system message commits to the scratchpad discipline;
# the user message carries the ORIGINAL problem, the scratchpad written SO FAR, and exactly ONE generic
# block to execute now. We render through the Instruct chat template (HB conventions).
# ===========================================================================================
STEP_SYS = (
    "You are a careful, precise reasoner working ONE step at a time on a shared worksheet (a "
    "scratchpad). You are given the original PROBLEM, the WORKSHEET written so far, and exactly ONE "
    "step to carry out now. Carry out ONLY that one step on the data in the problem, do the "
    "character-level, digit-level, or calendar arithmetic faithfully and do not guess, and write just "
    "that step's result so it can be reused by later steps. Do not jump ahead to other steps and do not "
    "restate the whole solution."
)

# The committing (final) block additionally instructs the FINAL ANSWER: marker; the strict grader reads
# the span after the LAST marker, so we make the model emit it from the register it is already holding.
def step_user(problem, scratchpad, block_text, is_final):
    parts = [f"PROBLEM:\n{problem}\n"]
    if scratchpad.strip():
        parts.append(f"WORKSHEET SO FAR:\n{scratchpad}\n")
    if is_final:
        parts.append(
            "FINAL STEP — carry out ONLY this step and then commit:\n"
            f"{block_text}\n")
    else:
        parts.append(
            "Carry out ONLY this step and write its result (nothing else):\n"
            f"{block_text}\n")
    return "\n".join(parts)


def build_step_prompt(tok, problem, scratchpad, block_text, is_final, use_chat_template=True):
    user = step_user(problem, scratchpad, block_text, is_final)
    if use_chat_template and getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": STEP_SYS},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<<SYS>>\n{STEP_SYS}\n<</SYS>>\n\n{user}\n"


# ===========================================================================================
# INTERLEAVED EXECUTION — run ONE problem through ONE generic block-variant, growing a scratchpad.
# Returns (final_decoded_text_for_grading, full_scratchpad_for_logging).
# ===========================================================================================
def run_interleaved(decode, tok, problem, plan, has_ct, per_step_tokens, final_tokens, verbose=False):
    """plan = ordered list of GENERIC block strings (one interleaved turn each); the LAST/committing
    block carries 'FINAL ANSWER:'. We start the scratchpad at the problem, and for each block prompt the
    frozen model to carry out ONLY that step, APPENDING its decode to the scratchpad. The committing
    block's chunk (the one that emits FINAL ANSWER:) is what we grade."""
    scratchpad = ""             # worksheet written so far (problem is passed separately every step)
    final_chunk = ""
    for bi, block in enumerate(plan):
        is_final = is_final_block(block) or (bi == len(plan) - 1)
        prompt = build_step_prompt(tok, problem, scratchpad, block, is_final, has_ct)
        n_tok = final_tokens if is_final else per_step_tokens
        chunk = decode(prompt, n_tok)
        scratchpad += f"\n[step {bi+1}] {chunk.strip()}\n"
        if is_final:
            final_chunk = chunk
        if verbose:
            tag = "FINAL" if is_final else f"step{bi+1}"
            print(f"      [{tag}] {chunk.strip()[:120].replace(chr(10),' / ')}")
    # Grade the text that contains the committed FINAL ANSWER:. Prefer the final chunk; if the model
    # emitted the marker earlier, the whole scratchpad still contains it (grade reads the LAST marker).
    graded_text = final_chunk if _FINAL_RE.search(final_chunk) else scratchpad
    return graded_text, scratchpad


def eval_plan_on_rows(decode, tok, rows, plan, has_ct, per_step_tokens, final_tokens, verbose=False):
    """Run a plan over rows; return (accuracy, n_correct, per_row_scores, sample_logs)."""
    correct = 0.0
    scores = []
    logs = []
    for i, r in enumerate(rows):
        graded, scratch = run_interleaved(decode, tok, r["question"], plan, has_ct,
                                          per_step_tokens, final_tokens,
                                          verbose=verbose and i == 0)
        s = grade(graded, r)
        correct += s
        scores.append(s)
        if len(logs) < 3:
            committed = bool(_FINAL_RE.search(graded))
            ma = " ".join(_answer_span(graded).split())[:50] if committed else "(no FINAL ANSWER)"
            logs.append({"id": r.get("id"), "gold": r["answer"], "model_answer": ma,
                         "correct": s, "committed": committed})
    return correct / max(1, len(rows)), correct, scores, logs


# ===========================================================================================
# GENERIC MUTATIONS — synthesize NEW block-variants from a base plan WITHOUT any problem content.
# These are the moves the per-category loop applies after the seed variants. Each returns a NEW plan
# (list of block strings). All wording is abstract; nothing references a specific problem, and we ASSERT
# that with the library's generic_violations() reuse gate.
# ===========================================================================================
def _insert_before_final(plan, block_text):
    """Insert a generic block string just before the committing (FINAL ANSWER:) block."""
    out = list(plan)
    out.insert(len(out) - 1, block_text)
    return out

# A small bank of GENERIC re-do / verification block STRINGS the mutator can add (problem-agnostic):
# a double-check re-do, a line-by-line audit, and a magnitude sanity-check. None reference any problem.
_GENERIC_VERIFY_BLOCKS = [
    ("Now independently redo the very last step from scratch a second time without looking at your "
     "previous attempt; if the two results disagree, redo it once more carefully and keep the careful "
     "pass's value; state the reconciled running result"),
    ("Re-examine the worksheet line by line for any place you skipped an item, miscounted, dropped a "
     "carry, or mis-ordered a step; if you find a mistake, redo from that line and update the running "
     "result; state the corrected running result"),
    ("Sanity-check the running result against the size of the input (is its magnitude or length "
     "plausible given how many items or steps there were); if it looks off, redo the weakest step and "
     "update the running result"),
]

# A generic 'finer decomposition' block: force the next pass to externalize one item/step per line.
_GENERIC_FINER_BLOCK = (
    "Before continuing, rewrite whatever you are about to operate on as an explicit vertical list with "
    "one element or one step per line, in order, so the next step can be done line by line without "
    "skipping any")


def _assert_generic(block_text, problem_words):
    """The reusability GATE: a mutation block must NOT overlap the specific problem's content. We assert
    generic_violations == [] so the mutation loop can only ever add GENERIC (problem-agnostic) blocks."""
    viol = generic_violations(block_text, problem_words)
    assert not viol, f"non-generic mutation block leaks problem content {viol}: {block_text[:60]!r}"


def mutate(base_plan, round_idx, gate_problem_words=None):
    """Return a list of NEW GENERIC candidate plans (each a list of block strings) derived from base_plan
    for this mutation round. Generic-only: we add verification/re-do blocks and a finer-decomposition
    block. round_idx cycles which verify block we add so successive rounds explore different edits. If
    gate_problem_words is given, every inserted block is asserted GENERIC against it via the reuse gate."""
    vblock = _GENERIC_VERIFY_BLOCKS[round_idx % len(_GENERIC_VERIFY_BLOCKS)]
    if gate_problem_words is not None:
        for b in (vblock, _GENERIC_FINER_BLOCK, _GENERIC_VERIFY_BLOCKS[0], _GENERIC_VERIFY_BLOCKS[2]):
            _assert_generic(b, gate_problem_words)
    cands = []
    # 1) add the round's verify block before the commit
    cands.append(_insert_before_final(base_plan, vblock))
    # 2) prepend a finer-decomposition block, then add the round's verify block before the commit
    cands.append(_insert_before_final([_GENERIC_FINER_BLOCK] + list(base_plan), vblock))
    # 3) add TWO different verify blocks (re-do, then a magnitude sanity-check) before the commit
    two = _insert_before_final(base_plan, _GENERIC_VERIFY_BLOCKS[0])
    two = _insert_before_final(two, _GENERIC_VERIFY_BLOCKS[2])
    cands.append(two)
    return cands


# ===========================================================================================
# PER-CATEGORY SEARCH + MUTATION LOOP — try seed variants, keep the best, then KEEP mutating the best
# (generically) until accuracy == 1.0 or --rounds is exhausted. Records accuracy climbing per round.
# ===========================================================================================
def solve_category(decode, tok, category, rows, has_ct, args):
    seeds = variants_for_category(category)           # generic block-variants for this category's skill
    skill = skill_for_category(category)
    # one problem's words, used only as the genericness GATE for our mutations (asserts no leak)
    gate_words = rows[0]["question"] if rows else ""
    print(f"\n=== category {category}  (skill {skill}, n={len(rows)}, seeds={len(seeds)}) ===")

    history = []          # list of {round, kind, plan_len, acc}
    best = {"acc": -1.0, "plan": None, "where": ""}

    def consider(plan, kind, rnd):
        acc, corr, scores, logs = eval_plan_on_rows(
            decode, tok, rows, plan, has_ct, args.step_tokens, args.max_new_tokens,
            verbose=args.verbose)
        history.append({"round": rnd, "kind": kind, "plan_len": len(plan), "acc": acc})
        print(f"    round {rnd:<2} {kind:<22} blocks={len(plan):<2} acc={acc:.0%}"
              + ("   <-- new best" if acc > best["acc"] else ""))
        if acc > best["acc"]:
            best.update({"acc": acc, "plan": copy.deepcopy(plan), "where": kind, "logs": logs})
        return acc

    # round 0: evaluate every seed variant
    for vi, plan in enumerate(seeds):
        consider(plan, f"seed-variant-{vi}", 0)
        if best["acc"] >= 1.0:
            break

    # rounds 1..R: keep mutating the current best GENERICALLY until 100% or budget. Each round mutates
    # the CURRENT best plan, so improvements compound (a verify block that helped stays in the base).
    rnd = 1
    while best["acc"] < 1.0 and rnd <= args.rounds:
        for ci, plan in enumerate(mutate(best["plan"], rnd, gate_problem_words=gate_words)):
            consider(plan, f"mutate-r{rnd}-c{ci}", rnd)
            if best["acc"] >= 1.0:
                break
        rnd += 1

    print(f"  >>> {category}: FINAL acc={best['acc']:.0%} via {best['where']} "
          f"({len(best['plan'])} generic blocks)")
    return {
        "category": category,
        "skill": skill,
        "accuracy": best["acc"],
        "n": len(rows),
        "final_plan_blocks": list(best["plan"]),       # GENERIC block strings, in order
        "final_plan_n_blocks": len(best["plan"]),
        "chosen_via": best["where"],
        "round_history": history,
        "sample_decodes": best.get("logs", []),
    }


# ===========================================================================================
# DRIVER
# ===========================================================================================
def main():
    ap = argparse.ArgumentParser(
        description="FROZEN interleaved solver: per-category generic-block search + mutation until 100%.")
    ap.add_argument("--data", default="hard_bench.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows PER CATEGORY (0 = all 10). Use a small value for a quick smoke test.")
    ap.add_argument("--out", default="interleaved_out")
    ap.add_argument("--max_new_tokens", type=int, default=320,
                    help="tokens for the FINAL committing step (must reach FINAL ANSWER:).")
    ap.add_argument("--step_tokens", type=int, default=320,
                    help="tokens per intermediate scratchpad step (the unrolled-loop chunk).")
    ap.add_argument("--rounds", type=int, default=6,
                    help="per-category GENERIC mutation rounds after the seed variants (budget).")
    ap.add_argument("--only", default="",
                    help="comma-separated category subset to solve (default: all 20).")
    ap.add_argument("--verbose", action="store_true", help="print the first row's per-step decodes.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[interleaved] CUDA not available -> falling back to CPU")
        args.device = "cpu"
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu" and dtype is not torch.float32:
        dtype = torch.float32

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.data):
        sys.exit(f"[interleaved] data not found: {args.data} (run tools/gen_hard_bench.py first)")
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    by_cat = collections.OrderedDict()
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        by_cat = collections.OrderedDict((c, v) for c, v in by_cat.items() if c in want)
    if args.limit and args.limit > 0:
        by_cat = collections.OrderedDict((c, v[:args.limit]) for c, v in by_cat.items())
    print(f"[interleaved] {sum(len(v) for v in by_cat.values())} rows across {len(by_cat)} categories")

    print(f"[interleaved] loading FROZEN {args.base} ({args.dtype}) on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model.to(device)
    model.eval()                                  # INFERENCE ONLY
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)                   # belt-and-braces: no gradients anywhere
    has_ct = bool(getattr(tok, "chat_template", None))
    print(f"[interleaved] chat_template present: {has_ct}")

    # Decoder closure: greedy / frozen / no_grad, with a per-call max_new_tokens (HB.make_decoder fixes
    # one length; we need a different cap for intermediate vs final steps, so we wrap generate directly).
    @torch.no_grad()
    def decode(prompt, n_new):
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        out = model.generate(
            **enc, max_new_tokens=n_new, do_sample=False, num_beams=1,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        )
        return tok.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)

    # ---- per-category solve loop ----
    cat_results = {}
    for cat, crows in by_cat.items():
        cat_results[cat] = solve_category(decode, tok, cat, crows, has_ct, args)

    per_cat_acc = {c: cat_results[c]["accuracy"] for c in cat_results}
    total_n = sum(cat_results[c]["n"] for c in cat_results)
    overall = sum(cat_results[c]["accuracy"] * cat_results[c]["n"] for c in cat_results) / max(1, total_n)

    # ---- DUMP the FINAL chosen GENERIC interleaved plan per category to results.json ----
    results = {
        "base": args.base, "data": args.data, "device": device, "dtype": args.dtype,
        "chat_template": has_ct, "rounds_budget": args.rounds,
        "max_new_tokens": args.max_new_tokens, "step_tokens": args.step_tokens,
        "overall_accuracy": overall,
        "per_category_accuracy": per_cat_acc,
        "skill_for_category": {c: cat_results[c]["skill"] for c in cat_results},
        "final_generic_plans": {            # THE deliverable: the generic plan that solved each category
            c: {
                "skill": cat_results[c]["skill"],
                "accuracy": cat_results[c]["accuracy"],
                "n_blocks": cat_results[c]["final_plan_n_blocks"],
                "chosen_via": cat_results[c]["chosen_via"],
                "blocks": cat_results[c]["final_plan_blocks"],   # GENERIC block texts, in order
            } for c in cat_results
        },
        "per_category_detail": cat_results,
        "note": ("Plans are GENERIC interleaved block-lists (no problem-specific content). Each category "
                 "was solved by running the frozen model interleaved over a growing scratchpad, searching "
                 "+ generically mutating the blocks until accuracy hit 100% or the round budget."),
    }
    res_path = os.path.join(args.out, "results.json")
    json.dump(results, open(res_path, "w"), indent=2)
    print(f"\n[interleaved] wrote {res_path}")

    # ---- headless per-category accuracy plot (climbing toward 100%) ----
    plot_results(per_cat_acc, overall, os.path.join(args.out, "interleaved.png"))

    # ---- summary ----
    print("\n==================== INTERLEAVED SOLVER SUMMARY ====================")
    print(f"  rows={total_n}  base={args.base}  ({args.dtype}/{device})  rounds_budget={args.rounds}")
    print("  per-category accuracy (after generic search + mutation):")
    for c in sorted(per_cat_acc):
        d = cat_results[c]
        flag = "  *100%*" if per_cat_acc[c] >= 1.0 else ""
        print(f"    {c:<28} acc={per_cat_acc[c]:.0%}  ({d['final_plan_n_blocks']} generic blocks, "
              f"skill {d['skill']}){flag}")
    print("  ------------------------------------------------------------------")
    solved = sum(1 for c in per_cat_acc if per_cat_acc[c] >= 1.0)
    print(f"  >>> OVERALL ACCURACY = {overall:.2%}   "
          f"({solved}/{len(per_cat_acc)} categories at 100%) <<<")
    print("  >>> FINAL chosen GENERIC plans dumped to results.json -> final_generic_plans <<<")
    print("===================================================================")


# ===========================================================================================
# PLOT — headless Agg. Per-category accuracy (target 100%) + overall line. (HB plot, recolored green.)
# ===========================================================================================
def plot_results(per_cat_acc, overall, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")                     # headless backend — BEFORE importing pyplot
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping PNG.")
        return
    cats = sorted(per_cat_acc)
    accs = [per_cat_acc[c] for c in cats]
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    bars = ax.bar(cats, accs, color="tab:green")
    ax.axhline(overall, color="black", linestyle="--", linewidth=1.2,
               label=f"overall = {overall:.1%}")
    ax.axhline(1.0, color="tab:blue", linestyle=":", linewidth=1.0, label="target = 100%")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy (correct fraction)")
    ax.set_title(f"INTERLEAVED (frozen Qwen2.5-1.5B-Instruct) per-category accuracy after generic "
                 f"block search + mutation (overall {overall:.1%}; GOAL 100%)")
    ax.tick_params(axis="x", rotation=55)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.0%}", ha="center", fontsize=7)
    ax.legend()
    try:
        fig.savefig(out_png, dpi=130)
        print(f"[plot] wrote {out_png}")
    except Exception as e:
        print(f"[plot] could not save {out_png}: {e}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()