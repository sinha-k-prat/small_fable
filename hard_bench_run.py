#!/usr/bin/env python3
"""
hard_bench_run.py — HARD_BENCH: FROZEN-model failure benchmark. ONE script, identical for
SSH-remote and Colab.

WHAT THIS MEASURES
------------------
HARD_BENCH is 20 categories x 10 examples (200 rows) of tasks with CANONICAL, MACHINE-VERIFIABLE
answers, reverse-engineered from documented small-LLM failure modes (the 'how many r in
strawberry' char-count meme, exact multi-digit multiplication, base conversion, caesar/rot13,
nested-bracket depth, modular arithmetic, day-of-week/date math, nth-term sequences,
needle-in-a-long-list counting, sorting+ranking, ...). The corpus is CALIBRATED so a FROZEN
Qwen2.5-1.5B-Instruct scores ~0% — the GOAL is to SEE the model get essentially everything
wrong. Every answer was computed by a deterministic Python solver (tools/gen_hard_bench.py), so
grading is JUDGE-FREE: we defer to checkers.reward_for_row on each row's {checker_kind,
checker_args}.

This is INFERENCE-ONLY on a frozen model: torch.no_grad + model.eval, NO LoRA, NO optimizer,
NO gradients, NO training. (Reuses grounding_test.py's chat-template + FINAL ANSWER: extraction
+ headless Agg plotting conventions verbatim.)

For each row we build a clean Instruct prompt that asks the model to solve the task and commit
its answer after a 'FINAL ANSWER:' marker, greedy-decode it, extract the committed span, and
grade with checkers.reward_for_row. We report PER-CATEGORY accuracy + OVERALL accuracy and a
marker_rate (fraction of decodes that actually emitted FINAL ANSWER:), and save results.json +
a headless per-category accuracy bar chart. Overall accuracy is EXPECTED to be ~0%.

ROW SCHEMA: {id, category, question, answer, checker_kind, checker_args, why_hard}

CLI (same on SSH and Colab — plain argparse, no Colab-only calls):
  python hard_bench_run.py --data hard_bench.jsonl --base Qwen/Qwen2.5-1.5B-Instruct \
      --dtype bf16 --device cuda --limit 0 --out hard_bench_out --max_new_tokens 256
"""
import argparse, json, os, re, sys, collections


# ===========================================================================================
# PROMPT — a clean instruct prompt that asks for a committed answer after FINAL ANSWER:.
# ===========================================================================================
SYS_MSG = (
    "You are a careful, precise reasoner. Solve the problem exactly. Do the character-level, "
    "digit-level, or calendar arithmetic faithfully and do not guess. When you are done, write the "
    "marker 'FINAL ANSWER:' on a new line followed by ONLY the single final answer (a number, a "
    "word, or a string) and nothing else after it."
)

def user_msg(question):
    return (
        f"PROBLEM:\n{question}\n\n"
        f"Solve it step by step, then on a new line write:\nFINAL ANSWER:"
    )

def build_prompt(tok, question, use_chat_template=True):
    """Render to a single prompt string. For an Instruct model we use
    tokenizer.apply_chat_template(..., add_generation_prompt=True); fall back to a plain template
    only if the tokenizer has no chat template (e.g. a base, non-instruct model)."""
    user = user_msg(question)
    if use_chat_template and getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": SYS_MSG},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<<SYS>>\n{SYS_MSG}\n<</SYS>>\n\n{user}\n"


# ===========================================================================================
# JUDGE-FREE GRADING — extract the committed answer (text after the LAST FINAL ANSWER: marker),
# then grade with the row's own checker (checkers.reward_for_row). Same contract as the rest of
# the repo; NO LLM judge. A decode that never commits a FINAL ANSWER: marker scores 0.
# ===========================================================================================
_FINAL_RE = re.compile(r"final\s*answer\s*:?", re.I)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def _answer_span(decoded):
    """Text after the LAST 'FINAL ANSWER:' marker (the committed answer)."""
    ms = list(_FINAL_RE.finditer(str(decoded)))
    return str(decoded)[ms[-1].end():] if ms else ""

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def grade(decoded, row):
    """STRICT, judge-free scoring in {0.0, 1.0}. Requires a committed 'FINAL ANSWER:' marker.
    We do NOT use checkers.reward_for_row here: its check_numeric matches ANY number in the span
    (so a list/range '4, 5, or 6' scores 1.0 for gold 5) and check_exact does substring containment
    (so echoing an input that contains the gold scores 1.0). Both manufacture false positives on a
    benchmark meant to read ~0%. Instead:
      numeric -> the committed span must contain EXACTLY ONE distinct number, equal to the canonical;
      exact   -> the normalized committed span must EQUAL the canonical (no substring/echo gaming)."""
    if not _FINAL_RE.search(str(decoded)):
        return 0.0
    span = _answer_span(decoded)
    gold = row["answer"]
    if row.get("checker_kind") == "numeric":
        nset = {float(x.replace(",", "")) for x in _NUM_RE.findall(span)}
        return 1.0 if len(nset) == 1 and abs(next(iter(nset)) - float(gold)) < 1e-9 else 0.0
    return 1.0 if _norm(span) == _norm(gold) else 0.0


# ===========================================================================================
# DECODE — greedy, frozen, no_grad. (Identical to grounding_test.make_decoder.)
# ===========================================================================================
def make_decoder(model, tok, device, max_new_tokens):
    import torch

    @torch.no_grad()
    def decode(prompt):
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # greedy / deterministic
            num_beams=1,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        )
        return tok.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)

    return decode


# ===========================================================================================
# PLOT — headless Agg. Per-category accuracy bar + an OVERALL bar (expected ~0%).
# ===========================================================================================
def plot_results(per_cat_acc, overall, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")                # headless backend — BEFORE importing pyplot
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping PNG.")
        return
    cats = sorted(per_cat_acc)
    accs = [per_cat_acc[c] for c in cats]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    bars = ax.bar(cats, accs, color="tab:red")
    ax.axhline(overall, color="black", linestyle="--", linewidth=1.2,
               label=f"overall = {overall:.1%}")
    ax.set_ylim(0, max(0.1, max(accs) * 1.25 if accs else 0.1))
    ax.set_ylabel("accuracy (correct fraction)")
    ax.set_title(f"HARD_BENCH — frozen Qwen2.5-1.5B-Instruct per-category accuracy "
                 f"(overall {overall:.1%}; GOAL ~0%)")
    ax.tick_params(axis="x", rotation=55)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.002, f"{a:.0%}", ha="center", fontsize=7)
    ax.legend()
    try:
        fig.savefig(out_png, dpi=130)
        print(f"[plot] wrote {out_png}")
    except Exception as e:
        print(f"[plot] could not save {out_png}: {e}")
    finally:
        plt.close(fig)


# ===========================================================================================
# DRIVER
# ===========================================================================================
def main():
    ap = argparse.ArgumentParser(
        description="HARD_BENCH: frozen-model failure benchmark (inference only, judge-free grading).")
    ap.add_argument("--data", default="hard_bench.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--out", default="hard_bench_out")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="must be large enough for the CoT to reach FINAL ANSWER: — watch "
                         "marker_rate in the summary (want >=0.9); raise this if it's low.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # device resolution (works on SSH A100 and on Colab T4/A100; cpu fallback for smoke tests)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[hard_bench] CUDA not available -> falling back to CPU")
        args.device = "cpu"
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu" and dtype is not torch.float32:
        dtype = torch.float32  # half on CPU is slow/unsupported

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.data):
        sys.exit(f"[hard_bench] data file not found: {args.data} "
                 f"(run tools/gen_hard_bench.py first)")
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    print(f"[hard_bench] loaded {len(rows)} rows from {args.data}")

    print(f"[hard_bench] loading FROZEN {args.base} ({args.dtype}) on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model.to(device)
    model.eval()                                # INFERENCE ONLY
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)                 # belt-and-braces: no gradients anywhere
    has_ct = bool(getattr(tok, "chat_template", None))
    print(f"[hard_bench] chat_template present: {has_ct} "
          f"(Instruct templating {'ON' if has_ct else 'OFF (fallback)'})")

    decode = make_decoder(model, tok, device, args.max_new_tokens)

    per_cat = collections.defaultdict(lambda: {"correct": 0.0, "n": 0})
    total_correct = 0.0
    total_n = 0
    marker_hits = marker_total = 0
    records = []
    examples = []

    for i, r in enumerate(rows):
        cat = r["category"]
        decoded = decode(build_prompt(tok, r["question"], has_ct))
        score = grade(decoded, r)               # judge-free, {0,1}

        marker_total += 1
        committed = bool(_FINAL_RE.search(decoded))
        marker_hits += 1 if committed else 0

        per_cat[cat]["correct"] += score
        per_cat[cat]["n"] += 1
        total_correct += score
        total_n += 1

        records.append({"id": r.get("id", i), "category": cat, "correct": score,
                        "committed": committed, "gold": r["answer"]})
        if len(examples) < 20:
            examples.append({"id": r.get("id", i), "category": cat, "gold": r["answer"],
                             "why_hard": r.get("why_hard", ""),
                             "decode_tail": decoded[-160:], "correct": score,
                             "committed": committed})

        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            acc = total_correct / max(1, total_n)
            print(f"  [{i+1}/{len(rows)}] running overall accuracy = {acc:.2%}")

    per_cat_acc = {c: v["correct"] / max(1, v["n"]) for c, v in per_cat.items()}
    overall = total_correct / max(1, total_n)
    marker_rate = marker_hits / max(1, marker_total)

    results = {
        "base": args.base, "data": args.data, "n_rows": total_n,
        "dtype": args.dtype, "device": device, "max_new_tokens": args.max_new_tokens,
        "chat_template": has_ct, "marker_rate": marker_rate,
        "overall_accuracy": overall,
        "per_category_accuracy": per_cat_acc,
        "per_category_n": {c: v["n"] for c, v in per_cat.items()},
        "goal": "frozen model is CALIBRATED to score ~0%; high accuracy means the bench is too easy",
        "records": records,
        "examples": examples,
    }
    res_path = os.path.join(args.out, "results.json")
    json.dump(results, open(res_path, "w"), indent=2)
    print(f"[hard_bench] wrote {res_path}")

    plot_results(per_cat_acc, overall, os.path.join(args.out, "hard_bench.png"))

    print("\n==================== HARD_BENCH SUMMARY ====================")
    print(f"  rows={total_n}  base={args.base}  ({args.dtype}/{device})")
    flag = "" if marker_rate >= 0.9 else "  <-- LOW: raise --max_new_tokens; accuracy may be understated"
    print(f"  marker_rate (decodes reaching FINAL ANSWER:) {marker_rate:.0%}{flag}")
    print("  per-category accuracy:")
    for c in sorted(per_cat_acc):
        m = per_cat_acc[c]; n = per_cat[c]["n"]
        print(f"    {c:<18} n={n:<3} acc={m:.0%}")
    print("  ----------------------------------------------------------")
    print(f"  >>> OVERALL ACCURACY = {overall:.2%}   (GOAL: ~0% — frozen model should fail) <<<")
    print(f"  >>> OVERALL WRONG    = {1.0 - overall:.2%} <<<")
    print("===========================================================")


if __name__ == "__main__":
    main()