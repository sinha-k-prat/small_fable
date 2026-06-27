#!/usr/bin/env python3
"""
export_metrics.py — dump FULL-HIDDEN-SPACE metrics per (model, variant, task) to a compact JSON,
so analysis is on the real vectors, not the ~12% PCA-shadow GIF.

For each run it records:
  pred_token, answer_text (generated to EOS), is_correct, gold
  turn_curve     : per-node direction-change angle (deg) in FULL hidden space  -> WHERE it steers
  max_turn       : (kind, layer, deg) of the sharpest turn (the candidate steering layer)
  node_dist      : per-node distance-from-mean-direction (which residuals are the far landmarks)
  goldrank_curve : NORMALIZED logit-lens rank of the gold answer's first token at each layer
                   (apply final RMSNorm before the tied unembedding -> avoids the intermediate-basis
                   artifact). A ramp = answer forms gradually; a last-layer cliff = deferral.
  commit_layer   : first layer where the gold token enters top-5 (depth-of-commit; -1 if never)

Run on Colab (1.5B is fast there):
  python tools/export_metrics.py --models 0.5B 1.5B --variants plain simple detailed \
      --device cuda --out metrics.json
Then download metrics.json (tiny) for quantitative 0.5B-vs-1.5B and success-vs-failure analysis.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residual_orrery.models import load_model
from residual_orrery.collect import collect_run
from residual_orrery.examples import VARIANTS


def _ang(u, v, eps=1e-9):
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def _turn_curve(nodes):
    """Direction-change angle at each interior node, on the FULL-space DIRECTION sphere."""
    Hn = np.stack([n.h for n in nodes]).astype(np.float64)
    Hn = Hn / (np.linalg.norm(Hn, axis=1, keepdims=True) + 1e-9)
    out = []
    for i in range(1, len(Hn) - 1):
        d0, d1 = Hn[i] - Hn[i - 1], Hn[i + 1] - Hn[i]
        out.append({"kind": nodes[i].kind.value, "layer": int(nodes[i].layer),
                    "deg": round(_ang(d0, d1), 1)})
    cen = Hn.mean(0); cen /= np.linalg.norm(cen) + 1e-9
    dist = [{"kind": nodes[i].kind.value, "layer": int(nodes[i].layer),
             "d": round(float(np.linalg.norm(Hn[i] - cen)), 3)} for i in range(len(Hn))]
    return out, dist


def _normalized_logit_lens(bundle, nodes, gold):
    """Gold-first-token RANK at each layer via a NORMALIZED logit lens (final RMSNorm THEN the
    tied unembedding) — the per-layer 'is the answer formed yet?' curve. Returns (curve, commit)."""
    import torch
    tok = bundle.tokenizer
    gold_ids = tok(str(gold), add_special_tokens=False).input_ids
    if not gold_ids:
        return [], -1
    g0 = int(gold_ids[0])
    W = bundle.embed().weight.detach().to(torch.float32)      # [V, H] (tied unembed)
    norm = bundle.final_norm()
    curve, commit = [], -1
    with torch.no_grad():
        for nd in nodes:
            if nd.kind.value != "mlp":            # post-layer-L residual stream
                continue
            h = torch.tensor(np.asarray(nd.h), dtype=torch.float32, device=W.device)
            hn = norm(h)                          # normalized lens: apply final RMSNorm
            logits = hn @ W.t()                   # [V]
            rank = int((logits > logits[g0]).sum().item()) + 1   # 1 = top
            curve.append({"layer": int(nd.layer), "gold_rank": rank})
            if commit < 0 and rank <= 5:
                commit = int(nd.layer)
    return curve, commit


def run(models, variants, device, gen_tokens, out_path):
    results = []
    for tag in models:
        bundle = load_model(tag, device=device)
        for vname in variants:
            for t in VARIANTS[vname]:
                gt = t.gen_tokens if gen_tokens <= 0 else gen_tokens
                rc = collect_run(bundle, t, generate=True, gen_tokens=gt,
                                 gold=t.gold, grade_mode=t.grade, self_test=False)
                turn, dist = _turn_curve(rc.nodes)
                turn_sorted = sorted(turn, key=lambda d: d["deg"], reverse=True)
                gold_curve, commit = _normalized_logit_lens(bundle, rc.nodes, t.gold)
                results.append({
                    "model": tag, "variant": vname, "task": t.key,
                    "pred_token": rc.pred_token_str, "answer_text": rc.answer_text,
                    "gold": t.gold, "is_correct": rc.is_correct,
                    "n_layers": rc.N,
                    "max_turn": turn_sorted[0] if turn_sorted else None,
                    "top_turns": turn_sorted[:5],
                    "far_nodes": sorted(dist, key=lambda d: d["d"], reverse=True)[:4],
                    "goldrank_curve": gold_curve,
                    "commit_layer": commit,
                })
                mt = results[-1]["max_turn"]
                print(f"[{tag} {vname:8s} {t.key:8s}] correct={rc.is_correct} "
                      f"commit_L={commit} maxturn={mt['kind'] if mt else '-'}"
                      f"{mt['layer'] if mt else ''}:{mt['deg'] if mt else ''} "
                      f"ans={rc.answer_text[:40]!r}")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}  ({len(results)} runs)")


def main():
    ap = argparse.ArgumentParser(description="export full-space orrery metrics to JSON")
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B"])
    ap.add_argument("--variants", nargs="+", default=["plain", "simple", "detailed"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gen_tokens", type=int, default=0, help="0 = per-task cap (EOS-stopped)")
    ap.add_argument("--out", default="metrics.json")
    a = ap.parse_args()
    run(a.models, a.variants, a.device, a.gen_tokens, a.out)


if __name__ == "__main__":
    main()
