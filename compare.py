#!/usr/bin/env python3
"""
compare.py — 3-way evaluation: base vs SFT vs SFT+RL.

Reports, per model:
  - correctness (checker) on held-out instructions, decoded with a SAMPLED plan (deployment mode)
  - plan-vs-no-plan ABLATION GAP (gold plan vs no plan) — is the plan load-bearing?
  - mean reward + zero-variance-group fraction over G samples/instruction (the RL signal source)
And across SFT vs SFT+RL:
  - adapter L2 difference (must be > 0 to prove RL actually moved the backbone, RL ≠ SFT)
  - plan-token distribution diff (did RL find DIFFERENT plans, or only reweight the executor?)

IMPORTANT: pass --sample (temp 0.7). Greedy decoding (do_sample=False) masks small RL effects.

CLI:
  python compare.py --sft_ckpt joint_ckpt --rl_ckpt rl_ckpt --sample --device cuda
"""
import argparse, json, re
from collections import Counter
import torch

from model_joint import JointModel, encode_plan
from checkers import reward_for_row, _norm, _last_number

OOD_PROMPTS = [
    "A train leaves at 9:00 going 60 km/h. Another leaves at 10:00 going 90 km/h on the same "
    "track behind it. At what time does the second catch the first?",
    "Sort these by size then sum the two largest: [4, 19, 7, 22, 3, 15]. Give only the number.",
    "Ada is older than Bo. Cy is younger than Bo. Who is the youngest?",
]


def load_data(path, train, held):
    rows = [json.loads(l) for l in open(path)]
    return rows[train:train+held]


@torch.no_grad()
def eval_joint(model, held, max_resp, sample, temp, group):
    model.eval()
    corr_sampled = corr_plan = corr_noplan = 0
    rew_means, zero_var = [], 0
    plan_counter = Counter()
    for r in held:
        p_ids, p_attn = model.batch_prompts([r["instruction"]])
        # deployment mode: sample a plan, then answer
        plan = model.sample_plan(p_ids, p_attn, temp=(temp if sample else 1.0), sample=sample)
        plan_counter.update(model_decode_plan(plan))
        gen = model.generate_answer(p_ids, p_attn, plan, sample=sample, temp=temp, max_new_tokens=max_resp)
        corr_sampled += reward_for_row(r, model.tok.decode(gen[0], skip_special_tokens=True))
        # ablation: gold plan vs no plan
        gold_plan = encode_plan(r["plan"], model.plan_max_len).unsqueeze(0).to(model.device)
        g_plan = model.generate_answer(p_ids, p_attn, gold_plan, sample=sample, temp=temp, max_new_tokens=max_resp)
        g_none = model.generate_answer(p_ids, p_attn, None, sample=sample, temp=temp, max_new_tokens=max_resp)
        corr_plan += reward_for_row(r, model.tok.decode(g_plan[0], skip_special_tokens=True))
        corr_noplan += reward_for_row(r, model.tok.decode(g_none[0], skip_special_tokens=True))
        # G-sample group for reward variance
        pe_ids, pe_attn = model.batch_prompts([r["instruction"]] * group)
        gp = model.sample_plan(pe_ids, pe_attn, temp=max(temp, 1.3), sample=True)
        gg = model.generate_answer(pe_ids, pe_attn, gp, sample=True, temp=max(temp, 1.3), max_new_tokens=max_resp)
        rews = torch.tensor([reward_for_row(r, model.tok.decode(gg[i], skip_special_tokens=True))
                             for i in range(group)], dtype=torch.float)
        rew_means.append(float(rews.mean()))
        if float(rews.std(unbiased=False)) < 1e-6:
            zero_var += 1
    n = len(held)
    return {
        "acc_sampled": corr_sampled/n,
        "acc_gold_plan": corr_plan/n,
        "acc_no_plan": corr_noplan/n,
        "ablation_gap": (corr_plan - corr_noplan)/n,
        "mean_reward": sum(rew_means)/n,
        "zero_var_frac": zero_var/n,
        "plan_dist": dict(plan_counter.most_common(10)),
    }


def model_decode_plan(plan_ids):
    from model_joint import decode_plan
    return decode_plan(plan_ids[0])


@torch.no_grad()
def eval_base(base_name, held, max_resp, sample, temp, device):
    """Plain base LM: prompt -> answer, no planner. correctness only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype="auto").to(device).eval()
    corr = 0
    def _tmpl(instr):
        try:
            return tok.apply_chat_template([{"role": "user", "content": instr}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n"
    for r in held:
        text = _tmpl(r["instruction"])
        ids = tok(text, return_tensors="pt").to(device)
        gen = m.generate(**ids, do_sample=sample, temperature=(temp if sample else None),
                         max_new_tokens=max_resp, pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        corr += reward_for_row(r, ans)
    del m
    return {"acc_sampled": corr/len(held), "ablation_gap": None, "note": "base has no planner head"}


def lora_param_diff(base_name, ckpt_a, ckpt_b, device):
    """L2 of (adapter_b - adapter_a) over matched LoRA tensors."""
    ma = JointModel.from_checkpoint(base_name, ckpt_a, device=device, is_trainable=False)
    a = {n: p.detach().float().clone() for n, p in ma.backbone.named_parameters() if "lora_" in n}
    del ma
    mb = JointModel.from_checkpoint(base_name, ckpt_b, device=device, is_trainable=False)
    b = {n: p.detach().float().clone() for n, p in mb.backbone.named_parameters() if "lora_" in n}
    del mb
    diff = sum(float(((b[n] - a[n])**2).sum()) for n in a if n in b)
    return diff ** 0.5


def _claims(answer):
    """Split an answer into decision-relevant claims + the final salient token (number or last word)."""
    parts = [c.strip() for c in re.split(r"[.\n;]", str(answer)) if len(c.strip()) > 3]
    final = _last_number(answer) or (str(answer).split()[-1] if str(answer).split() else "")
    return parts[:4], final


def _answer_key(answer):
    n = _last_number(answer)
    return _norm(n) if n is not None else _norm(answer)


@torch.no_grad()
def clr_eval(model, held, K, M, max_resp, temp):
    """A8 CLR (eval-only, optional): sample K trajectories, extract claims + final answer, SELF-VERIFY
    each claim to a binary verdict, score a trajectory by (mean_verdict)^M (nonlinear -> any flawed
    claim hurts), cluster trajectories by answer-equivalence, and pick the answer maximizing summed
    reliability. Reports correctness of that selected answer. Bounded: K trajectories * <=4 claims."""
    model.eval()
    correct = 0
    for r in held:
        p_ids, p_attn = model.batch_prompts([r["instruction"]] * K)
        plans = model.sample_plan(p_ids, p_attn, temp=temp, sample=True)
        gen = model.generate_answer(p_ids, p_attn, plans, sample=True, temp=temp, max_new_tokens=max_resp)
        clusters = {}  # answer_key -> [reliability,...], plus a representative answer
        for i in range(K):
            ans = model.tok.decode(gen[i], skip_special_tokens=True)
            claims, _ = _claims(ans)
            if claims:
                vq = [f"Claim: {c}\nReply 'yes' if correct, else 'no'." for c in claims]
                vp_ids, vp_attn = model.batch_prompts(vq)
                vg = model.generate_answer(vp_ids, vp_attn, None, sample=False, max_new_tokens=4)
                verdicts = [1.0 if "yes" in model.tok.decode(vg[j], skip_special_tokens=True).lower()
                            else 0.0 for j in range(len(claims))]
                reliability = (sum(verdicts) / len(verdicts)) ** M
            else:
                reliability = 0.5 ** M
            key = _answer_key(ans)
            clusters.setdefault(key, {"rel": 0.0, "ans": ans})
            clusters[key]["rel"] += reliability
        best = max(clusters.values(), key=lambda c: c["rel"]) if clusters else {"ans": ""}
        correct += reward_for_row(r, best["ans"])
    return {"acc_clr": correct / len(held)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft_ckpt", default="joint_ckpt")
    ap.add_argument("--rl_ckpt", default="rl_ckpt")
    ap.add_argument("--data", default="dataset/sft_100.jsonl")
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--held", type=int, default=20)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max_resp", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--sample", action="store_true", help="REQUIRED to see small RL effects (temp 0.7)")
    ap.add_argument("--clr", action="store_true", help="A8: claim-level test-time scaling (eval-only)")
    ap.add_argument("--clr_k", type=int, default=4, help="A8 trajectories sampled per prompt")
    ap.add_argument("--clr_m", type=int, default=2, help="A8 nonlinearity: (mean_verdict)^M")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.sample:
        print("[compare] WARNING: --sample not set. Greedy decoding hides small RL effects.")

    held = load_data(args.data, args.train, args.held)
    print(f"[compare] held instructions: {len(held)}  sample={args.sample} temp={args.temp}\n")

    report = {}
    print("=== BASE (no planner) ===")
    report["base"] = eval_base(args.base, held, args.max_resp, args.sample, args.temp, args.device)
    print(json.dumps(report["base"], indent=2), "\n")

    print("=== SFT ===")
    msft = JointModel.from_checkpoint(args.base, args.sft_ckpt, device=args.device, is_trainable=False)
    report["sft"] = eval_joint(msft, held, args.max_resp, args.sample, args.temp, args.group)
    sft_plan_dist = report["sft"]["plan_dist"]
    del msft
    print(json.dumps(report["sft"], indent=2), "\n")

    print("=== SFT+RL ===")
    mrl = JointModel.from_checkpoint(args.base, args.rl_ckpt, device=args.device, is_trainable=False)
    report["rl"] = eval_joint(mrl, held, args.max_resp, args.sample, args.temp, args.group)
    rl_plan_dist = report["rl"]["plan_dist"]
    # OOD generations (qualitative)
    print("  OOD samples (SFT+RL):")
    for q in OOD_PROMPTS:
        p_ids, p_attn = mrl.batch_prompts([q])
        plan = mrl.sample_plan(p_ids, p_attn, temp=args.temp, sample=args.sample)
        gen = mrl.generate_answer(p_ids, p_attn, plan, sample=args.sample, temp=args.temp, max_new_tokens=args.max_resp)
        print(f"    Q: {q[:70]}...\n      plan={model_decode_plan(plan)}\n"
              f"      A: {mrl.tok.decode(gen[0], skip_special_tokens=True)[:120]}")
    if args.clr:
        clr = clr_eval(mrl, held, args.clr_k, args.clr_m, args.max_resp, args.temp)
        report["rl"]["acc_clr"] = clr["acc_clr"]
        print(f"  +CLR (A8): acc_clr={clr['acc_clr']:.3f} "
              f"(vs sampled acc={report['rl']['acc_sampled']:.3f})")
    del mrl
    print(json.dumps(report["rl"], indent=2), "\n")

    print("=== SFT vs SFT+RL deltas ===")
    l2 = lora_param_diff(args.base, args.sft_ckpt, args.rl_ckpt, args.device)
    print(f"adapter L2 diff |RL - SFT| = {l2:.5f}  (>0 proves RL moved the backbone)")
    keys = set(sft_plan_dist) | set(rl_plan_dist)
    print("plan-token distribution diff (SFT -> RL):")
    for k in sorted(keys, key=lambda x: -(rl_plan_dist.get(x, 0))):
        print(f"  {k:<22} SFT={sft_plan_dist.get(k,0):>3}  RL={rl_plan_dist.get(k,0):>3}")
    print(f"\nacc: base={report['base']['acc_sampled']:.3f} "
          f"sft={report['sft']['acc_sampled']:.3f} rl={report['rl']['acc_sampled']:.3f}")
    print(f"ablation_gap: sft={report['sft']['ablation_gap']:+.3f} rl={report['rl']['ablation_gap']:+.3f}")
    print(f"mean_reward: sft={report['sft']['mean_reward']:.3f} rl={report['rl']['mean_reward']:.3f}")
    print("\nACCEPTANCE: SFT+RL ≠ SFT under sampling, adapter L2 diff > 0, ablation_gap positive.")


if __name__ == "__main__":
    main()
