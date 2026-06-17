#!/usr/bin/env python3
"""
self_distill.py — A7 (OPTIONAL): consolidate multi-stage RL gains into one clean model via offline
self-distillation.

Pipeline:
  1. REJECTION-SAMPLE correct trajectories from the RL checkpoint (sample N/prompt, keep reward>=thr).
  2. RANK traces by LEARNING POTENTIAL = length-normalized NLL under the CURRENT student
     (high NLL = verified-correct but NOT yet well modeled => most worth distilling).
  3. Rank WITHIN length buckets (so long traces don't dominate purely by length); DROP extremely
     short traces and extreme-high-NLL outliers (format noise / degenerate).
  4. EMIT distill_data.jsonl (the mid-to-high band) in the SFT schema, ready to SFT back in.

This is optional for v1: it produces the curated dataset (and can launch the SFT pass with --do_sft).

CLI:
  python self_distill.py --rl_ckpt rl_ckpt --data dataset/sft_flat.jsonl --train 800 \
      --n 8 --keep_frac 0.6 --out distill_data.jsonl --device cuda [--do_sft]
"""
import argparse, json, math
import torch, torch.nn.functional as F

from model_joint import JointModel, encode_plan, decode_plan, PAD_ID
from checkers import graded_reward_for_row


@torch.no_grad()
def trace_nll(model, instruction, plan_list, answer, max_resp):
    """Length-normalized NLL of (plan, answer) under the current student. Higher = less well modeled."""
    p_ids, p_attn = model.batch_prompts([instruction])
    plan_ids = encode_plan(plan_list, model.plan_max_len).unsqueeze(0).to(model.device)
    r_ids, r_attn = (lambda e: (e["input_ids"].to(model.device), e["attention_mask"].to(model.device)))(
        model.tok([answer], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_resp, add_special_tokens=False))
    if r_ids.numel() == 0:
        return 0.0
    logp, mask = model.resp_logp_tf(p_ids, p_attn, plan_ids, r_ids, r_attn, temp=1.0)
    return float(-(logp * mask).sum() / mask.sum().clamp_min(1))


@torch.no_grad()
def rejection_sample(model, rows, n, thr, max_resp, temp):
    traces = []
    for r in rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]] * n)
        plans = model.sample_plan(p_ids, p_attn, temp=temp, sample=True)
        gen = model.generate_answer(p_ids, p_attn, plans, sample=True, temp=temp, max_new_tokens=max_resp)
        for i in range(n):
            ans = model.tok.decode(gen[i], skip_special_tokens=True)
            if graded_reward_for_row(r, ans) >= thr:
                plan_list = decode_plan(plans[i])
                traces.append({"id": r["id"], "instruction": r["instruction"],
                               "plan": plan_list or ["TERMINATE"], "answer": ans,
                               "checker_kind": r["checker_kind"], "checker_args": r["checker_args"],
                               "category": r.get("category", ""), "reward_path": r.get("reward_path"),
                               "len": len(ans.split())})
    return traces


def select_band(traces, model, max_resp, keep_frac):
    """Rank by learning-potential within length buckets; drop short traces + extreme-NLL outliers."""
    for t in traces:
        t["nll"] = trace_nll(model, t["instruction"], t["plan"], t["answer"], max_resp)
    # length buckets (quartiles); drop the shortest bucket as 'too short / format noise'
    lens = sorted(t["len"] for t in traces) or [0]
    q1 = lens[len(lens)//4]
    buckets = {}
    for t in traces:
        if t["len"] <= max(1, q1):       # drop extremely short
            continue
        buckets.setdefault(t["len"] // 8, []).append(t)
    kept = []
    for _, b in buckets.items():
        b.sort(key=lambda x: x["nll"])               # ascending NLL
        lo = int(0.1 * len(b)); hi = int((0.1 + keep_frac) * len(b))  # drop extreme-high-NLL tail too
        kept.extend(b[lo:max(lo+1, hi)])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_ckpt", default="rl_ckpt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="dataset/sft_flat.jsonl")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--thr", type=float, default=1.0, help="reward threshold to accept a trace")
    ap.add_argument("--keep_frac", type=float, default=0.6)
    ap.add_argument("--max_resp", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="distill_data.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--do_sft", action="store_true", help="launch a short SFT pass on the curated data")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)][:args.train]
    model = JointModel.from_checkpoint(args.base, args.rl_ckpt, device=args.device, is_trainable=False)
    print(f"[distill] rejection-sampling {args.n}/prompt from {args.rl_ckpt} over {len(rows)} prompts ...")
    traces = rejection_sample(model, rows, args.n, args.thr, args.max_resp, args.temp)
    print(f"[distill] correct traces: {len(traces)}")
    kept = select_band(traces, model, args.max_resp, args.keep_frac)
    with open(args.out, "w") as f:
        for t in kept:
            f.write(json.dumps({k: t[k] for k in
                    ("id", "instruction", "plan", "answer", "checker_kind", "checker_args",
                     "category", "reward_path")}, ensure_ascii=False) + "\n")
    print(f"[distill] kept mid-to-high learning-potential band: {len(kept)} -> {args.out}")
    if args.do_sft:
        import subprocess, sys
        cmd = [sys.executable, "train_sft.py", "--data", args.out, "--train", str(len(kept)),
               "--held", "0", "--epochs", "2", "--out", "distill_ckpt", "--device", args.device]
        print("[distill] launching:", " ".join(cmd)); subprocess.run(cmd, check=True)
    else:
        print(f"[distill] to consolidate: python train_sft.py --data {args.out} "
              f"--train {len(kept)} --held 0 --epochs 2 --out distill_ckpt --device {args.device}")


if __name__ == "__main__":
    main()
