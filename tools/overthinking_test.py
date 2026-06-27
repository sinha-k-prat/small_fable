#!/usr/bin/env python3
"""
overthinking_test.py — the GO/NO-GO causal test for "late layers misroute".

Question: does the model COMPUTE the correct answer at a middle layer and then its OWN later
layers DEGRADE it? We measure, over MANY examples, the per-layer EARLY-EXIT accuracy: read the
answer out at every layer via the NORMALIZED lens (apply the final RMSNorm, then the tied
unembedding -> this is literally "what if we exited here"), predict, and score vs the known gold.

If best-middle-layer accuracy > final-layer accuracy (a positive "overthinking gap") across a large
sample, the late layers SYSTEMATICALLY misroute -> real effect, regularizable, target = the layers
after the peak. If the final layer is always best, there is no misroute (the single prime example was
a fluke). A binary yes/no task avoids multi-token contamination. 'prime' is the hard task; 'even' is a
trivially-computable CONTROL — a gap on prime but ~0 on even proves it's a real task-dependent misroute,
not a lens artifact. Run on both sizes to see if the small model overthinks MORE (the scale prediction).

Colab:
  python tools/overthinking_test.py --models 0.5B 1.5B --n 30 --device cuda --out overthinking
Produces overthinking.json + overthinking.png (per-layer accuracy curves).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_orrery.models import load_model
from residual_orrery.examples import build_chatml


def _isprime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _dataset(task, n_per_class):
    """Balanced yes/no set. Returns list of (prompt, gold_is_yes)."""
    yes, no = [], []
    for N in range(2, 100000):
        if len(yes) >= n_per_class and len(no) >= n_per_class:
            break
        if task == "prime":
            g = _isprime(N); q = f"Is {N} a prime number? Answer yes or no."
        elif task == "even":
            g = (N % 2 == 0); q = f"Is {N} an even number? Answer yes or no."
        else:
            raise ValueError(task)
        (yes if g else no).append((q, g)) if (len(yes) < n_per_class if g else len(no) < n_per_class) else None
    return yes[:n_per_class] + no[:n_per_class]


def _answer_ids(tok):
    """Token-id groups for yes / no (handle case + leading space)."""
    def fid(s):
        ids = tok(s, add_special_tokens=False).input_ids
        return ids[0] if ids else None
    yes = {i for i in (fid(x) for x in ["Yes", " Yes", "yes", " yes"]) if i is not None}
    no = {i for i in (fid(x) for x in ["No", " No", "no", " no"]) if i is not None}
    return sorted(yes), sorted(no)


def per_layer_accuracy(bundle, data, yes_ids, no_ids):
    """For each layer (incl. embeddings) -> early-exit accuracy + fraction-predicted-yes, over data."""
    import torch
    tok = bundle.tokenizer
    W = bundle.embed().weight.detach().to(torch.float32)     # [V, H] tied unembed
    norm = bundle.final_norm()
    L = bundle.n_layers
    correct = np.zeros(L + 1); pred_yes = np.zeros(L + 1); total = 0
    yi = torch.tensor(yes_ids); ni = torch.tensor(no_ids)
    with torch.no_grad():
        for q, gold_yes in data:
            ids = tok(build_chatml(q), return_tensors="pt").input_ids.to(bundle.device)
            hs = bundle.model(ids, output_hidden_states=True, use_cache=False).hidden_states
            total += 1
            for li, h in enumerate(hs):              # hs: [embeddings, after L0, ..., after L{N-1}]
                v = norm(h[0, -1, :])                # normalized lens (apply final RMSNorm)
                logits = (v @ W.t()).float()
                ly = logits[yi].max().item(); ln = logits[ni].max().item()
                pyes = ly > ln
                pred_yes[li] += 1.0 if pyes else 0.0
                correct[li] += 1.0 if (pyes == gold_yes) else 0.0
    return correct / max(total, 1), pred_yes / max(total, 1)


def run(models, n_per_class, device, out):
    tasks = ["prime", "even"]
    results = {}
    for tag in models:
        bundle = load_model(tag, device=device)
        yes_ids, no_ids = _answer_ids(bundle.tokenizer)
        for task in tasks:
            data = _dataset(task, n_per_class)
            acc, pyes = per_layer_accuracy(bundle, data, yes_ids, no_ids)
            fracs = (np.arange(len(acc)) / (len(acc) - 1)).tolist()
            final = float(acc[-1])
            # the REAL misroute metric: best genuine HIDDEN layer (exclude embeddings L0 and the
            # final layer) vs the model's output. A degenerate "best" at L0==chance is NOT misroute.
            mid = acc[1:-1]
            best_mid = float(mid.max()) if len(mid) else final
            bestL = int(np.argmax(mid)) + 1 if len(mid) else len(acc) - 1
            gap = best_mid - final
            misroute = gap > 0.05 and best_mid > 0.55      # clearly above final AND above chance
            results[f"{tag}/{task}"] = {
                "n": len(data), "n_layers": bundle.n_layers,
                "acc_by_layer": acc.round(3).tolist(),
                "predyes_by_layer": pyes.round(3).tolist(),
                "layer_frac": [round(f, 3) for f in fracs],
                "final_acc": round(final, 3),
                "best_mid_acc": round(best_mid, 3), "best_mid_layer": bestL,
                "overthinking_gap": round(gap, 3), "misroute": bool(misroute),
            }
            print(f"[{tag} {task:6s} n={len(data)}] final={final:.2f}  "
                  f"best_hidden={best_mid:.2f}@L{bestL}/{bundle.n_layers}  "
                  f"GAP={gap:+.2f}  misroute={'YES' if misroute else 'no'}  "
                  f"predyes(final)={pyes[-1]:.2f}")
    json.dump(results, open(out + ".json", "w"), indent=2)
    print(f"\nwrote {out}.json")
    _plot(results, out + ".png")


def _plot(results, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})"); return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"prime": "-", "even": "--"}
    for key, r in results.items():
        tag, task = key.split("/")
        ax.plot(r["layer_frac"], r["acc_by_layer"], styles[task], marker="o", ms=3,
                label=f"{tag} {task} (gap {r['overthinking_gap']:+.2f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xlabel("depth (layer fraction)"); ax.set_ylabel("early-exit accuracy")
    ax.set_title("Per-layer early-exit accuracy — does mid-stack beat the final layer?\n"
                 "(positive gap on prime but ~0 on even = real late-layer misroute)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description="per-layer early-exit accuracy: do late layers misroute?")
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B"])
    ap.add_argument("--n", type=int, default=30, help="examples PER CLASS per task (yes & no)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="overthinking")
    a = ap.parse_args()
    run(a.models, a.n, a.device, a.out)


if __name__ == "__main__":
    main()
