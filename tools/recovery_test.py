#!/usr/bin/env python3
"""
recovery_test.py — does COURSE-CORRECTION (wrong mid-stack -> right final) grow with scale, while
OVERWRITE (right mid-stack -> wrong final) shrinks?

For each model and each example we read the gold MARGIN out at every layer via the model's OWN
norm+lm_head (the normalized lens, quantization-safe): g_l = lens_logit[gold] - lens_logit[other].
g_l>0 = correct at layer l; |g_l| = confidence. Then, margin-gated (threshold delta) to ignore noise:

  OVERWRITE event : some non-final layer is CONFIDENTLY RIGHT (max_l g_l >  delta)  AND final wrong.
  RECOVERY  event : some non-final layer is CONFIDENTLY WRONG (min_l g_l < -delta)  AND final right.

  overwrite_rate = #overwrite / #(ever confidently right)     <- shrinks with scale (hypothesis)
  recovery_rate  = #recovery  / #(ever confidently wrong)      <- grows  with scale (hypothesis)

Self-contained (does NOT use residual_orrery's offline loader); loads from HF, 4-bit for 3B/7B so they
fit a 16GB GPU, frees between models. Binary yes/no tasks avoid multi-token contamination. 'prime' is
capacity-hard (recovery only measurable once the model is capable); 'compare' is doable by all sizes.

Colab:
  pip install -q bitsandbytes accelerate
  python tools/recovery_test.py --models 0.5B 1.5B 3B 7B --n 60 --delta 2.0 --out recovery
Produces recovery.json + recovery.png (recovery_rate & overwrite_rate vs model size).
"""
import argparse, gc, json, os
import numpy as np


def _isprime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _dataset(task, n_per_class, rng):
    yes, no = [], []
    if task == "prime":
        for N in range(2, 100000):
            if len(yes) >= n_per_class and len(no) >= n_per_class:
                break
            g = _isprime(N); q = f"Is {N} a prime number? Answer yes or no."
            (yes if g else no).append((q, g)) if (len(yes) < n_per_class if g else len(no) < n_per_class) else None
    elif task == "compare":
        while len(yes) < n_per_class or len(no) < n_per_class:
            a, b = int(rng.integers(1, 99)), int(rng.integers(1, 99))
            if a == b:
                continue
            g = a > b; q = f"Is {a} greater than {b}? Answer yes or no."
            (yes if g else no).append((q, g)) if (len(yes) < n_per_class if g else len(no) < n_per_class) else None
    return yes[:n_per_class] + no[:n_per_class]


def _load(name, want_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if want_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return tok, model


def _chatml(q):
    return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def _answer_ids(tok):
    def fid(s):
        ids = tok(s, add_special_tokens=False).input_ids
        return ids[0] if ids else None
    yes = sorted({i for i in (fid(x) for x in ["Yes", " Yes", "yes", " yes"]) if i is not None})
    no = sorted({i for i in (fid(x) for x in ["No", " No", "no", " no"]) if i is not None})
    return yes, no


def gold_margins(tok, model, data, yes_ids, no_ids):
    """Return [n_examples, n_layers+1] signed gold margins g_l (lens via the model's own norm+lm_head)."""
    import torch
    dev = next(model.parameters()).device
    base = model.model            # Qwen2Model; final norm = base.norm; head = model.lm_head
    yi = torch.tensor(yes_ids, device=dev); ni = torch.tensor(no_ids, device=dev)
    rows = []
    with torch.no_grad():
        for q, gold_yes in data:
            ids = tok(_chatml(q), return_tensors="pt").input_ids.to(dev)
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
            g = []
            for h in hs:
                v = base.norm(h[:, -1, :])                  # normalized lens
                logits = model.lm_head(v).float()[0]
                ly = logits[yi].max().item(); ln = logits[ni].max().item()
                margin = (ly - ln) if gold_yes else (ln - ly)   # signed: >0 = correct
                g.append(margin)
            rows.append(g)
    return np.asarray(rows)                                  # [N, L+1]


def rates(margins, delta):
    """overwrite_rate, recovery_rate, final_acc from the per-layer margin matrix."""
    mid = margins[:, :-1]                    # non-final layers (incl. embeddings)
    final = margins[:, -1]
    ever_right = mid.max(axis=1) > delta     # confidently right somewhere mid-stack
    ever_wrong = mid.min(axis=1) < -delta    # confidently wrong somewhere mid-stack
    final_right = final > 0
    overwrite = ever_right & (~final_right)
    recovery = ever_wrong & final_right
    o_rate = overwrite.sum() / max(ever_right.sum(), 1)
    r_rate = recovery.sum() / max(ever_wrong.sum(), 1)
    return float(o_rate), float(r_rate), float(final_right.mean()), int(ever_right.sum()), int(ever_wrong.sum())


def run(models, n_per_class, delta, out, force_4bit):
    import torch
    rng = np.random.default_rng(0)
    tasks = ["prime", "compare"]
    results = {}
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== loading {name}  (4bit={want_4bit}) ===")
        tok, model = _load(name, want_4bit)
        yes_ids, no_ids = _answer_ids(tok)
        for task in tasks:
            data = _dataset(task, n_per_class, rng)
            M = gold_margins(tok, model, data, yes_ids, no_ids)
            o, r, acc, nr, nw = rates(M, delta)
            results[f"{tag}/{task}"] = {
                "n": len(data), "final_acc": round(acc, 3),
                "overwrite_rate": round(o, 3), "recovery_rate": round(r, 3),
                "n_ever_right": nr, "n_ever_wrong": nw}
            print(f"  [{tag} {task:7s}] acc={acc:.2f}  overwrite={o:.2f} (of {nr})  "
                  f"recovery={r:.2f} (of {nw})")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(results, open(out + ".json", "w"), indent=2)
    print(f"\nwrote {out}.json")
    _plot(results, models, out + ".png")


def _plot(results, models, png):
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["prime", "compare"]):
        xs = list(range(len(models)))
        rec = [results.get(f"{m}/{task}", {}).get("recovery_rate", np.nan) for m in models]
        ovr = [results.get(f"{m}/{task}", {}).get("overwrite_rate", np.nan) for m in models]
        acc = [results.get(f"{m}/{task}", {}).get("final_acc", np.nan) for m in models]
        ax.plot(xs, rec, "-o", color="tab:green", label="recovery rate (wrong->right)")
        ax.plot(xs, ovr, "-o", color="tab:red", label="overwrite rate (right->wrong)")
        ax.plot(xs, acc, "--", color="gray", label="final accuracy (control)")
        ax.set_xticks(xs); ax.set_xticklabels(models)
        ax.set_title(task); ax.set_xlabel("model size"); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Does recovery grow and overwrite shrink with scale?  "
                 "(check recovery rises FASTER than accuracy = not just 'bigger is better')")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--n", type=int, default=60, help="examples PER CLASS per task")
    ap.add_argument("--delta", type=float, default=2.0, help="confidence margin (logits) to gate noise")
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="recovery")
    a = ap.parse_args()
    run(a.models, a.n, a.delta, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
