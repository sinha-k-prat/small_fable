#!/usr/bin/env python3
"""
grounding_test.py — FROZEN grounding probe. ONE script, identical for SSH-remote and Colab.

Question: can a FROZEN base instruct model (default Qwen2.5-1.5B-Instruct) GROUND and FOLLOW an
ABSTRACT MULTI-TURN plan written in plain ENGLISH? Plans are English on purpose: an untrained
custom primitive token would be a meaningless embedding to a frozen model, so the probe speaks the
language the model already understands and measures whether the model *uses* the plan.

This is INFERENCE-ONLY: torch.no_grad + model.eval, NO LoRA, NO optimizer, NO gradients, NO training.

For each row of grounding_test_data.jsonl we greedy-decode THREE ways with proper Instruct chat
templating and grade with a judge-free extractor (text after the LAST 'FINAL ANSWER:'):
  (i)   GOLD multi-turn plan   -> acc_plan    : decode == gold answer  (grounding produces the answer)
  (ii)  NO plan                -> acc_noplan  : decode == gold answer  (answer derivable w/o the plan)
  (iii) NEGATIVE multi-turn plan (a different but equally-valid English plan whose deterministic,
        known-by-construction result is neg_answer != gold):
            neg_follow  : decode == neg_answer (faithfully FOLLOWED the wrong plan -> grounding)
            neg_to_gold : decode == gold       (IGNORED the plan, solved from the prompt anyway)

VERDICT: grounding works iff neg_follow is HIGH and neg_to_gold is LOW. acc_plan high alone is weak
evidence (the answer may be derivable from the setup); the NEG-plan decode is what isolates genuine
plan-causation. Reports overall + per-topic + per-turn-count, saves an Agg PNG + results.json.

CLI (same on SSH and Colab — plain argparse, no Colab-only calls):
  python grounding_test.py --data grounding_test_data.jsonl --base Qwen/Qwen2.5-1.5B-Instruct \
      --dtype bf16 --device cuda --limit 0 --out grounding_out --max_new_tokens 24
"""
import argparse, json, os, re, sys, collections


# ===========================================================================================
# PROMPT FORMAT  — multi-turn plan, no-plan, neg-plan. Rendered through the Instruct chat template.
# ===========================================================================================
# A row carries:
#   setup        : str   — the English problem/context (topic-varied: arithmetic, logic, ordering,
#                          schedule, units, text-ops, ... ), answer is known-by-construction.
#   plan_turns   : [str] — 2..5 ENGLISH turns; the gold abstract plan (one step per turn).
#   neg_plan_turns:[str] — 2..5 ENGLISH turns; a DIFFERENT valid plan whose deterministic result
#                          (oracle-computed at gen time) is neg_answer != answer.
#   answer       : str   — gold final answer (unambiguous).
#   neg_answer   : str   — the final answer the NEG plan deterministically yields.
#   topic, n_turns       — for breakdowns.
#   checker_kind/checker_args (optional) — when present, grading defers to checkers.reward_for_row;
#                          else the built-in normalized comparator is used. Either way: NO LLM judge.
#
# We present the multi-turn plan as an explicit numbered list of turns inside ONE user message, then
# ask the model to work through the turns IN ORDER and commit. The no-plan prompt asks it to solve the
# setup directly. The neg-plan prompt is byte-identical to the gold-plan prompt except the turns.

SYS_MSG = (
    "You are a careful reasoner. You are given a problem and, when provided, an explicit step-by-step "
    "PLAN written as a numbered list of turns. Carry out the plan turn by turn IN ORDER, using only the "
    "plan to decide what to do at each step. Do the arithmetic/logic faithfully. When you are done, "
    "write the marker 'FINAL ANSWER:' followed by the single final answer and nothing else after it."
)

# Per-topic answer cue: commit the ENTITY the question asks for, not a value computed en route
# (fixes the 'cheapest is Onyx ... FINAL ANSWER: $35' artifact where grounding succeeded but the
# wrong field was committed). Keyed by the dataset's `topic`.
TOPIC_CUE = {
    "constraint_select": "the NAME of the single chosen item (a word, not its price or any number)",
    "comparison_order":  "the full ordering of names using '<', e.g. 'A < B < C'",
    "set_ops":           "the single person's NAME (one word)",
    "transitive_logic":  "the single player's NAME (one word)",
    "scheduling":        "the single step word (one word)",
    "categorize_rule":   "the single category word (one word)",
    "multi_hop_lookup":  "the single final value (one word)",
    "conditional_reco":  "the single recommended item's NAME (one word)",
}

def _format_turns(turns):
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(turns))

def _cue(cue):
    return f" Your final answer must be {cue}." if cue else ""

def user_with_plan(setup, turns, cue=""):
    return (
        f"PROBLEM:\n{setup}\n\n"
        f"PLAN (follow these turns in order, exactly):\n{_format_turns(turns)}\n\n"
        f"Work through the plan one turn at a time.{_cue(cue)}\n"
        f"Then on a new line write:\nFINAL ANSWER:"
    )

def user_no_plan(setup, cue=""):
    return (
        f"PROBLEM:\n{setup}\n\n"
        f"Solve it step by step.{_cue(cue)}\n"
        f"Then on a new line write:\nFINAL ANSWER:"
    )

def build_prompt(tok, setup, turns, cue="", use_chat_template=True):
    """Render to a single prompt string. For an Instruct model we use
    tokenizer.apply_chat_template(..., add_generation_prompt=True); fall back to a plain template
    only if the tokenizer has no chat template (e.g. a base, non-instruct model)."""
    user = user_no_plan(setup, cue) if turns is None else user_with_plan(setup, turns, cue)
    if use_chat_template and getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": SYS_MSG},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # non-instruct fallback (kept judge-free / deterministic)
    return f"<<SYS>>\n{SYS_MSG}\n<</SYS>>\n\n{user}\n"


# ===========================================================================================
# JUDGE-FREE GRADING  — robust to extra reasoning text; NOT gameable by mentioning numbers early.
# ===========================================================================================
# Extract the committed answer = text AFTER the LAST 'FINAL ANSWER:' marker (the model is instructed
# to emit exactly one trailing token there). Normalize, then compare to the *relevant* reference
# (gold for acc_plan/acc_noplan/neg_to_gold ; neg_answer for neg_follow). Garden-path traces that
# mention a wrong number early can't fool this because we only read the last committed span.

_FINAL_RE = re.compile(r"final\s*answer\s*:?", re.I)
_NUM_RE   = re.compile(r"-?\d[\d,]*\.?\d*")

def answer_span(text):
    """Text after the LAST 'FINAL ANSWER:' marker; fall back to the last line / tail if absent."""
    s = str(text)
    ms = list(_FINAL_RE.finditer(s))
    if ms:
        return s[ms[-1].end():]
    # no marker: take the last non-empty line, else the tail
    lines = [ln for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s[-160:]

def _norm_text(s):
    """lowercase, strip punctuation/whitespace, drop currency/percent so '$3,450.' ~ '3450'."""
    s = str(s).lower().replace("_", " ")
    s = s.replace("$", "").replace("%", "").replace(",", "")
    s = re.sub(r"[^a-z0-9\.\- ]", " ", s)
    return " ".join(s.split()).rstrip(".")

def _last_number(s):
    m = _NUM_RE.findall(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None

def _is_numeric(ref):
    return _last_number(ref) is not None and re.fullmatch(r"\s*-?\$?[\d,]*\.?\d+%?\s*\.?",
                                                          str(ref).strip()) is not None

def match_answer(decoded, reference, tol=0.01):
    """Judge-free: does the committed answer in `decoded` match `reference`?
    Numeric refs -> compare the LAST number in the span within tol (handles '3450' vs '$3,450.00').
    Word/name/ordering refs -> normalized substring / word-set containment in the committed span.
    Returns 1.0 / 0.0."""
    span = answer_span(decoded)
    ref = str(reference).strip()
    if _is_numeric(ref):
        a, b = _last_number(span), _last_number(ref)
        if a is None or b is None:
            return 0.0
        return 1.0 if abs(a - b) <= tol else 0.0
    # textual answer (yes/no, a name, an ordering like 'B A C', a single word)
    nref, nspan = _norm_text(ref), _norm_text(span)
    if not nref:
        return 0.0
    # exact normalized equality, OR ordered token-subsequence containment (orderings/multi-word names)
    if nref == nspan or nref in nspan:
        return 1.0
    rtoks, stoks = nref.split(), nspan.split()
    if rtoks and all(t in stoks for t in rtoks):
        # require the reference tokens to appear IN ORDER (so 'A B C' != 'C B A')
        idx = -1
        for t in rtoks:
            try:
                j = stoks.index(t, idx + 1)
            except ValueError:
                return 0.0
            idx = j
        return 1.0
    return 0.0

def grade(decoded, reference, row, want="gold"):
    """Prefer the row's own checker (checkers.reward_for_row) when present so grading is identical
    to the rest of the repo; else use the built-in judge-free comparator. For neg_follow we must
    grade against neg_answer, so we swap the checker's canonical/gold to the neg reference.
    REQUIRES a committed 'FINAL ANSWER:' marker — an uncommitted (e.g. truncated) decode scores 0,
    never a tail/last-line match (that fallback manufactures false positives on reasoning text)."""
    if not _FINAL_RE.search(str(decoded)):
        return 0.0
    ck = row.get("checker_kind")
    if ck:
        try:
            from checkers import reward_for_row
            import copy
            r = copy.deepcopy(row)
            # point the checker at the relevant reference
            if want == "neg":
                args = r.get("checker_args", {})
                if isinstance(args.get("canonical"), dict) and "value" in args["canonical"]:
                    try:
                        args["canonical"]["value"] = float(reference)
                    except (TypeError, ValueError):
                        args["canonical"]["value"] = reference
                elif isinstance(args.get("match"), dict) and "accept" in args["match"]:
                    args["match"]["accept"] = [reference]
                elif isinstance(args.get("match"), dict) and "key_phrase" in args["match"]:
                    args["match"]["key_phrase"] = reference
                elif "gold" in args:
                    args["gold"] = reference
            txt = decoded if _FINAL_RE.search(decoded) else f"FINAL ANSWER: {decoded}"
            return float(reward_for_row(r, txt))
        except Exception:
            pass  # any checker hiccup -> fall back to the built-in comparator (never crash)
    tol = 0.01
    try:
        tol = float(row.get("checker_args", {}).get("match", {}).get("tolerance", 0.01))
    except Exception:
        pass
    return match_answer(decoded, reference, tol)


# ===========================================================================================
# DECODE  — greedy, frozen, no_grad.
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
# PLOT  — headless Agg.
# ===========================================================================================
def plot_results(metrics, per_topic, per_turns, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")                # headless backend — BEFORE importing pyplot
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping PNG.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    keys = ["acc_plan", "neg_follow", "neg_to_gold", "acc_noplan"]
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:gray"]
    axes[0].bar(keys, [metrics[k] for k in keys], color=colors)
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("rate")
    axes[0].set_title("(A) OVERALL — grounding iff neg_follow HIGH & neg_to_gold LOW")
    for i, k in enumerate(keys):
        axes[0].text(i, metrics[k] + 0.02, f"{metrics[k]:.2f}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=20)

    if per_topic:
        topics = sorted(per_topic)
        import numpy as np
        x = np.arange(len(topics)); w = 0.2
        for j, (k, c) in enumerate(zip(keys, colors)):
            axes[1].bar(x + (j - 1.5) * w, [per_topic[t][k] for t in topics], w, label=k, color=c)
        axes[1].set_xticks(x); axes[1].set_xticklabels(topics, rotation=40, ha="right", fontsize=8)
        axes[1].set_ylim(0, 1.02); axes[1].set_title("(B) per-topic"); axes[1].legend(fontsize=7)

    if per_turns:
        tcs = sorted(per_turns)
        for k, c in zip(keys, colors):
            axes[2].plot(tcs, [per_turns[t][k] for t in tcs], "-o", color=c, label=k)
        axes[2].set_ylim(0, 1.02); axes[2].set_xlabel("# plan turns")
        axes[2].set_title("(C) per-turn-count"); axes[2].legend(fontsize=7)
        axes[2].set_xticks(tcs)

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
def _agg():
    return {"acc_plan": 0.0, "neg_follow": 0.0, "neg_to_gold": 0.0, "neg_other": 0.0,
            "acc_noplan": 0.0, "n": 0}

def _finalize(d):
    n = max(1, d["n"])
    return {k: d[k] / n for k in ("acc_plan", "neg_follow", "neg_to_gold", "neg_other",
                                  "acc_noplan")} | {"n": d["n"]}

def main():
    ap = argparse.ArgumentParser(description="Frozen-model English-plan grounding probe (inference only).")
    ap.add_argument("--data", default="grounding_test_data.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--out", default="grounding_out")
    ap.add_argument("--max_new_tokens", type=int, default=320,
                    help="must be large enough for the verbose CoT to reach FINAL ANSWER: — watch "
                         "marker_rate in the summary (want >=0.9); raise this if it's low.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # device resolution (works on SSH A100 and on Colab T4/A100; cpu fallback for smoke tests)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[probe] CUDA not available -> falling back to CPU")
        args.device = "cpu"
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu" and dtype is not torch.float32:
        dtype = torch.float32  # half on CPU is slow/unsupported

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.data):
        sys.exit(f"[probe] data file not found: {args.data} (run tools/gen_grounding_data.py first)")
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    print(f"[probe] loaded {len(rows)} rows from {args.data}")

    print(f"[probe] loading FROZEN {args.base} ({args.dtype}) on {device} ...")
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
    print(f"[probe] chat_template present: {has_ct}  (Instruct templating {'ON' if has_ct else 'OFF (fallback)'})")

    decode = make_decoder(model, tok, device, args.max_new_tokens)

    overall = _agg()
    per_topic = collections.defaultdict(_agg)
    per_turns = collections.defaultdict(_agg)
    cond = _agg()                           # CONDITIONAL: only rows the model FAILS unaided (acc_noplan==0)
    examples = []
    records = []                            # per-row dump for cross-format analysis (analyze_formats.py)
    marker_hits = marker_total = 0          # fraction of decodes that reached FINAL ANSWER: (truncation check)

    for i, r in enumerate(rows):
        setup   = r["problem"]                      # dataset field names (gen_grounding_data.py)
        gold    = r["gold_answer"]
        neg     = r.get("neg_answer", None)
        gturns  = r["gold_plan"]
        nturns  = r.get("neg_plan", None)
        topic   = r.get("topic", r.get("category", "unknown"))
        n_turns = int(r.get("n_turns", len(gturns)))
        cue     = TOPIC_CUE.get(topic, "")          # commit the asked-for entity, not a derived value

        d_plan   = decode(build_prompt(tok, setup, gturns, cue, has_ct))
        d_noplan = decode(build_prompt(tok, setup, None,   cue, has_ct))

        acc_plan   = grade(d_plan,   gold, r, want="gold")
        acc_noplan = grade(d_noplan, gold, r, want="gold")

        if nturns is not None and neg is not None:
            d_neg = decode(build_prompt(tok, setup, nturns, cue, has_ct))
            neg_follow  = grade(d_neg, neg,  r, want="neg")
            neg_to_gold = grade(d_neg, gold, r, want="gold")
            # 3rd wrong-plan outcome: the neg-plan decode committed an answer that matched NEITHER
            # the neg answer (followed) NOR gold (ignored/overrode) -> botched execution. Requires a
            # committed FINAL ANSWER: marker (an uncommitted/truncated decode is not 'other').
            neg_other = 1.0 if (_FINAL_RE.search(d_neg) and neg_follow == 0.0
                                and neg_to_gold == 0.0) else 0.0
        else:
            d_neg, neg_follow, neg_to_gold, neg_other = "", 0.0, 0.0, 0.0

        for dtxt in (d_plan, d_noplan, d_neg):     # marker_rate: did the decode actually commit?
            if dtxt != "":
                marker_total += 1
                marker_hits += 1 if _FINAL_RE.search(dtxt) else 0

        aggs = [overall, per_topic[topic], per_turns[n_turns]]
        if acc_noplan == 0.0:                      # plan-causation is only testable when it's NEEDED
            aggs.append(cond)
        for agg in aggs:
            agg["acc_plan"]    += acc_plan
            agg["neg_follow"]  += neg_follow
            agg["neg_to_gold"] += neg_to_gold
            agg["neg_other"]   += neg_other
            agg["acc_noplan"]  += acc_noplan
            agg["n"] += 1

        # per-row record (lean; one per row) — lets analyze_formats.py restrict to fails-unaided
        # (acc_noplan==0) per task and read the 3-way wrong-plan split, identically across formats.
        records.append({"id": r.get("id", i), "topic": topic, "n_turns": n_turns,
                        "acc_noplan": acc_noplan, "acc_plan": acc_plan,
                        "neg_follow": neg_follow, "neg_to_gold": neg_to_gold,
                        "neg_other": neg_other})

        if len(examples) < 12:
            examples.append({"id": r.get("id", i), "topic": topic, "n_turns": n_turns,
                             "gold": gold, "neg_answer": neg,
                             "plan_decode": d_plan[-120:], "neg_decode": d_neg[-120:],
                             "noplan_decode": d_noplan[-120:],
                             "acc_plan": acc_plan, "neg_follow": neg_follow,
                             "neg_to_gold": neg_to_gold, "neg_other": neg_other,
                             "acc_noplan": acc_noplan})

        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            m = _finalize(overall)
            print(f"  [{i+1}/{len(rows)}] acc_plan={m['acc_plan']:.2%} neg_follow={m['neg_follow']:.2%} "
                  f"neg_to_gold={m['neg_to_gold']:.2%} acc_noplan={m['acc_noplan']:.2%}")

    metrics   = _finalize(overall)
    pt        = {k: _finalize(v) for k, v in per_topic.items()}
    ptc       = {int(k): _finalize(v) for k, v in per_turns.items()}
    condm     = _finalize(cond)             # metrics over rows the model FAILS unaided

    # VERDICT — judge grounding ONLY on rows where the plan is load-bearing (the model can't solve
    # unaided). On rows it solves anyway, a wrong plan is correctly overridden, which looks like
    # 'ignoring' but isn't a grounding failure. So the honest test is the CONDITIONAL one.
    def _grounds(m):
        return (m["neg_follow"] >= 0.50) and (m["neg_to_gold"] <= 0.30)
    grounds_cond = cond["n"] >= 10 and _grounds(condm)
    verdict = ("GROUNDING WORKS" if grounds_cond else "GROUNDING WEAK/ABSENT")
    raw_verdict = ("GROUNDING WORKS" if _grounds(metrics) else "GROUNDING WEAK/ABSENT")

    marker_rate = marker_hits / max(1, marker_total)
    results = {
        "base": args.base, "data": args.data, "n_rows": overall["n"],
        "dtype": args.dtype, "device": device, "max_new_tokens": args.max_new_tokens,
        "chat_template": has_ct, "marker_rate": marker_rate,
        "overall": metrics, "per_topic": pt, "per_turn_count": ptc,
        "conditional": condm, "conditional_n": cond["n"],
        "verdict": verdict, "raw_verdict": raw_verdict,
        "verdict_rule": "CONDITIONAL: among rows the model fails unaided, grounding iff "
                        "neg_follow>=0.50 AND neg_to_gold<=0.30 (n>=10)",
        "records": records,
        "examples": examples,
    }
    res_path = os.path.join(args.out, "results.json")
    json.dump(results, open(res_path, "w"), indent=2)
    print(f"[probe] wrote {res_path}")

    plot_results(metrics, pt, ptc, os.path.join(args.out, "grounding.png"))

    print("\n==================== GROUNDING PROBE SUMMARY ====================")
    print(f"  rows={overall['n']}  base={args.base}  ({args.dtype}/{device})")
    flag = "" if marker_rate >= 0.9 else "  <-- LOW: raise --max_new_tokens; metrics below are unreliable"
    print(f"  marker_rate (decodes reaching FINAL ANSWER:) {marker_rate:.0%}{flag}")
    print(f"  RAW (all rows):")
    print(f"    acc_plan   (gold-plan -> gold) ........ {metrics['acc_plan']:.2%}")
    print(f"    neg_follow (neg-plan  -> neg)  HIGH=good {metrics['neg_follow']:.2%}")
    print(f"    neg_to_gold(neg-plan  -> gold) LOW=good  {metrics['neg_to_gold']:.2%}")
    print(f"    acc_noplan (no-plan   -> gold) ........ {metrics['acc_noplan']:.2%}")
    print(f"  CONDITIONAL (only the {cond['n']} rows the model FAILS unaided — the honest test):")
    print(f"    acc_plan   (plan rescues it) ......... {condm['acc_plan']:.2%}")
    print(f"    neg_follow (follows wrong plan) HIGH=good {condm['neg_follow']:.2%}")
    print(f"    neg_to_gold(ignores plan)       LOW=good  {condm['neg_to_gold']:.2%}")
    print("  per-topic:")
    for t in sorted(pt):
        m = pt[t]
        print(f"    {t:<18} n={m['n']:<3} plan={m['acc_plan']:.0%} "
              f"follow={m['neg_follow']:.0%} to_gold={m['neg_to_gold']:.0%} noplan={m['acc_noplan']:.0%}")
    print("  per-turn-count:")
    for t in sorted(ptc):
        m = ptc[t]
        print(f"    {t}-turn  n={m['n']:<3} plan={m['acc_plan']:.0%} "
              f"follow={m['neg_follow']:.0%} to_gold={m['neg_to_gold']:.0%} noplan={m['acc_noplan']:.0%}")
    print(f"\n  VERDICT (conditional, honest): {verdict}")
    print(f"  raw verdict (all rows, confounded by easy problems): {raw_verdict}")
    print(f"  rule: {results['verdict_rule']}")
    print("================================================================")


if __name__ == "__main__":
    main()