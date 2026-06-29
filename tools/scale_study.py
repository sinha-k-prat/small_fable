#!/usr/bin/env python3
"""
scale_study.py — ONE end-to-end study: across Qwen2.5 sizes, separate SOLVING skill from REPAIR skill,
and print a solid verdict. Forward-only, quantization-safe (uses each model's OWN norm+lm_head as the
normalized lens), defensive per-model (one model failing never kills the run).

Per model, per task, at the answer position we read the gold MARGIN at every layer:
    g_l = lens_logit[gold] - lens_logit[other]        (>0 correct, |g_l| = confidence)

From the margin trajectory we compute, with margin gate `delta` to ignore noise:
  SOLVING  = snap_acc   : correctness of the model's FIRST CONFIDENT answer (its instinct/direct solve)
  REPAIR   = final_acc - snap_acc : how much the LATER layers improve on the instinct (fixes - sabotage)
  overwrite_rate : of examples ever-confidently-RIGHT mid-stack, fraction that end WRONG (self-sabotage)
  recovery_rate  : of examples ever-confidently-WRONG  mid-stack, fraction that end RIGHT (course-correct)
  acc@25/50/75/final : the accuracy curve by depth (shape: ramp vs late cliff)

Two tasks: 'compare' (all sizes can do it -> solving & repair both measurable) and 'prime' (capacity-hard).

VERDICT: prints whether SOLVING rises with scale, whether REPAIR rises with scale INDEPENDENTLY, and
whether recovery rises FASTER than accuracy (the 'not just bigger-is-better' check).

Colab:
  pip install -q bitsandbytes accelerate
  python tools/scale_study.py --models 0.5B 1.5B 3B 7B --n 60 --delta 2.0 --out scale_study
"""
import argparse, gc, json, traceback
import numpy as np


def _isprime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _dataset(task, npc, rng):
    yes, no = [], []
    def add(q, g):
        if g and len(yes) < npc:
            yes.append((q, g))
        elif (not g) and len(no) < npc:
            no.append((q, g))
    if task == "prime":
        N = 2
        while (len(yes) < npc or len(no) < npc) and N < 100000:
            add(f"Is {N} a prime number? Answer yes or no.", _isprime(N)); N += 1
    else:  # compare
        while len(yes) < npc or len(no) < npc:
            a, b = int(rng.integers(1, 99)), int(rng.integers(1, 99))
            if a != b:
                add(f"Is {a} greater than {b}? Answer yes or no.", a > b)
    return yes[:npc] + no[:npc]


def _load(name, want_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if want_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return tok, AutoModelForCausalLM.from_pretrained(name, **kw).eval()


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


def _margins(tok, model, data, yes_ids, no_ids):
    import torch
    dev = next(model.parameters()).device
    base = model.model
    yi = torch.tensor(yes_ids, device=dev); ni = torch.tensor(no_ids, device=dev)
    rows = []
    with torch.no_grad():
        for q, gold_yes in data:
            ids = tok(_chatml(q), return_tensors="pt").input_ids.to(dev)
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
            g = []
            for h in hs:
                lo = model.lm_head(base.norm(h[:, -1, :])).float()[0]
                ly = lo[yi].max().item(); ln = lo[ni].max().item()
                g.append((ly - ln) if gold_yes else (ln - ly))
            rows.append(g)
    return np.asarray(rows)            # [N, L+1]


def _metrics(M, delta):
    N, Lp1 = M.shape
    L = Lp1 - 1
    final = M[:, -1]
    final_right = final > 0
    # snap = first CONFIDENT layer (over layers 1..L); fall back to final if never confident
    snap_right = np.zeros(N, bool)
    for i in range(N):
        conf = np.where(np.abs(M[i, 1:]) > delta)[0]
        snap_right[i] = (M[i, 1 + conf[0]] > 0) if len(conf) else (final[i] > 0)
    mid = M[:, :-1]
    ever_right = mid.max(1) > delta
    ever_wrong = mid.min(1) < -delta
    overwrite = ever_right & (~final_right)
    recovery = ever_wrong & final_right
    def acc_at(frac):
        return float((M[:, int(round(frac * L))] > 0).mean())
    return {
        "final_acc": round(float(final_right.mean()), 3),
        "snap_acc": round(float(snap_right.mean()), 3),
        "repair": round(float(final_right.mean() - snap_right.mean()), 3),
        "acc@25": round(acc_at(.25), 3), "acc@50": round(acc_at(.50), 3),
        "acc@75": round(acc_at(.75), 3),
        "overwrite_rate": round(float(overwrite.sum() / max(ever_right.sum(), 1)), 3),
        "recovery_rate": round(float(recovery.sum() / max(ever_wrong.sum(), 1)), 3),
        "n_ever_right": int(ever_right.sum()), "n_ever_wrong": int(ever_wrong.sum()),
    }


def run(models, npc, delta, out, force_4bit):
    import torch
    rng = np.random.default_rng(0)
    tasks = ["compare", "prime"]
    res, ok_models = {}, []
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== {name}  (4bit={want_4bit}) ===")
        try:
            tok, model = _load(name, want_4bit)
            yes_ids, no_ids = _answer_ids(tok)
            for task in tasks:
                data = _dataset(task, npc, rng)
                M = _margins(tok, model, data, yes_ids, no_ids)
                m = _metrics(M, delta); res[f"{tag}/{task}"] = m
                print(f"  [{task:7s}] solve(snap)={m['snap_acc']:.2f} final={m['final_acc']:.2f} "
                      f"REPAIR={m['repair']:+.2f} | recover={m['recovery_rate']:.2f} "
                      f"overwrite={m['overwrite_rate']:.2f} | acc@50={m['acc@50']:.2f}")
            ok_models.append(tag)
            del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            print(f"  !! {tag} FAILED, skipping:\n" + traceback.format_exc().splitlines()[-1])
    json.dump(res, open(out + ".json", "w"), indent=2)
    _plot(res, ok_models, out + ".png")
    _verdict(res, ok_models)
    return res


def _trend(xs):
    """+1 rising, -1 falling, 0 flat (by first-vs-last with a small deadband)."""
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if len(xs) < 2:
        return 0, 0.0
    d = xs[-1] - xs[0]
    return (1 if d > 0.03 else -1 if d < -0.03 else 0), d


def _verdict(res, models):
    print("\n" + "=" * 64 + "\nVERDICT (task = compare, the doable one)\n" + "=" * 64)
    snap = [res.get(f"{m}/compare", {}).get("snap_acc") for m in models]
    fin = [res.get(f"{m}/compare", {}).get("final_acc") for m in models]
    rep = [res.get(f"{m}/compare", {}).get("repair") for m in models]
    rec = [res.get(f"{m}/compare", {}).get("recovery_rate") for m in models]
    print(f"  sizes      : {models}")
    print(f"  SOLVING    : snap_acc  = {snap}")
    print(f"  final_acc  :            = {fin}")
    print(f"  REPAIR     : final-snap = {rep}")
    print(f"  recovery   :            = {rec}")
    ts, _ = _trend(snap); tr, dr = _trend(rep); trc, _ = _trend(rec); tf, df = _trend(fin)
    print("\n  ---- conclusions ----")
    print(f"  Solving improves with scale?  {'YES' if ts > 0 else 'no'}  (snap_acc {snap[0]}->{snap[-1]})")
    print(f"  Repair  improves with scale?  {'YES' if tr > 0 else 'no'}  (final-snap {rep[0]}->{rep[-1]})")
    faster = (df is not None and dr is not None and dr > 0 and (rec[-1] or 0) - (rec[0] or 0) > df)
    print(f"  Recovery rises FASTER than accuracy (not just bigger-is-better)?  {'YES' if faster else 'no'}")
    if tr > 0:
        print("  => REPAIR is a separable skill that grows with scale: the later layers add more\n"
              "     correction in bigger models, beyond the instinct (snap) accuracy.")
    elif ts > 0:
        print("  => Gains look like SOLVING (better instinct), not extra repair: snap accuracy carries it.")
    else:
        print("  => No clear scale trend on this task at this n/delta; increase --n or pick a harder task.")


def _plot(res, models, png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})"); return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = list(range(len(models)))
    for ax, task in zip(axes, ["compare", "prime"]):
        snap = [res.get(f"{m}/{task}", {}).get("snap_acc", np.nan) for m in models]
        fin = [res.get(f"{m}/{task}", {}).get("final_acc", np.nan) for m in models]
        rep = [res.get(f"{m}/{task}", {}).get("repair", np.nan) for m in models]
        rec = [res.get(f"{m}/{task}", {}).get("recovery_rate", np.nan) for m in models]
        ax.plot(xs, snap, "-o", color="tab:blue", label="solving (snap acc)")
        ax.plot(xs, fin, "-o", color="black", label="final acc")
        ax.plot(xs, rep, "-o", color="tab:green", label="repair (final-snap)")
        ax.plot(xs, rec, "--o", color="tab:orange", label="recovery rate")
        ax.set_xticks(xs); ax.set_xticklabels(models); ax.set_title(task)
        ax.set_xlabel("model size"); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.axhline(0, color="gray", lw=.6)
    fig.suptitle("Solving vs Repair across scale  (repair rising independently of solving = the finding)")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png} and {png.replace('.png','.json' if png.endswith('.png') else '.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--n", type=int, default=60, help="examples PER CLASS per task")
    ap.add_argument("--delta", type=float, default=2.0, help="confidence margin (logits)")
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="scale_study")
    a = ap.parse_args()
    run(a.models, a.n, a.delta, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
