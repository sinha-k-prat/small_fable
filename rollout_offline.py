#!/usr/bin/env python3
"""
rollout_offline.py — Stage 2a: generate the offline RL dataset ONCE.

Generation is the slow part on a weak GPU, so we sample all rollouts up front and cache them
to disk. train_grpo_offline.py then trains from this file with NO generation (fast), reusing
each batch for 2-3 gradient passes (off-policy, corrected by the clipped IS ratio).

For each training instruction we draw a GROUP of G rollouts at a HOT temperature so the group
genuinely differs (this is what creates reward variance and avoids zero-variance groups):
  1. sample a plan (planner policy, hot)
  2. sample an answer conditioned on that plan (executor policy, hot)
  3. score the answer with its programmatic checker -> reward in {0,1}
  4. re-score plan and response at temp=1 to cache the BEHAVIOR-POLICY logprobs
     (plan_logp_old, resp_logp_old). temp=1 logprobs are the actual policy probs used by the
     PPO ratio; on the first training pass ratio == exp(logp_new - logp_old) == 1 exactly.

The G rollouts of one instruction are written as a CONTIGUOUS block of G lines, because
grpo_offpolicy.group_advantages normalizes contiguous blocks of group_size.

A1b: rewards are GRADED via reward_path (binary for verifiable, fraction-of-rubric for soft tasks).
A2 (pre-RL filter): each prompt's group is marked keep=False if its rewards have no spread (covers
the literal p_q==0 / p_q==1 case for binary); a pre_rl_filter_report.csv is written. GRPO trains on
keep=True groups. A6: logp_old is recomputed with the trainer's own HF teacher-forced path, never
the sampler — this is what keeps the offline importance ratio valid at step 0.

Output rl_rollouts.jsonl row:
  {id, instruction, group_size, reward_path, reward, p_q, keep,
   plan_tokens:[ids], plan_logp_old:[per-token logp],
   resp_tokens:[ids], resp_logp_old:[per-token logp], resp_len, answer_text, plan_str}

CLI:
  python rollout_offline.py --sft_ckpt joint_ckpt --data dataset/sft_100.jsonl \
      --train 80 --group 8 --temp 1.5 --max_resp 64 --out rl_rollouts.jsonl --device cuda
"""
import argparse, json, time, csv
import torch

from model_joint import JointModel, PAD_ID, decode_plan
from checkers import graded_reward_for_row
from reward_paths import reward_path_for_row


def resp_mask_from_ids(ids, eos_id, pad_id):
    """Build a 1/0 mask over generated ids: 1 up to and including the first eos, 0 after.
    Returns (mask, trim_len) so callers can trim trailing all-pad columns."""
    B, T = ids.shape
    mask = torch.ones_like(ids, dtype=torch.float)
    trim = 0
    for b in range(B):
        seen_eos = False
        for t in range(T):
            tok = int(ids[b, t])
            if seen_eos:
                mask[b, t] = 0.0
            elif tok == eos_id:
                seen_eos = True  # keep the eos itself as a real token
            trim = max(trim, t + 1 if mask[b, t] > 0 else trim)
    return mask, max(trim, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", default="joint_ckpt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="dataset/sft_100.jsonl")
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.5)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--max_resp", type=int, default=64)
    ap.add_argument("--out", default="rl_rollouts.jsonl")
    ap.add_argument("--report", default="pre_rl_filter_report.csv", help="A2 per-prompt keep report")
    ap.add_argument("--filter", action="store_true", default=True,
                    help="A2: mark zero-spread groups keep=False (no GRPO gradient)")
    ap.add_argument("--no-filter", dest="filter", action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260616)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(args.data)][:args.train]
    print(f"[rollout] {len(rows)} instructions x G={args.group} @ temp={args.temp}")

    # Load the SFT model. (Generation only -> trainability irrelevant here, but we keep the
    # frozen-backbone guard live downstream in train_grpo_offline.py.)
    model = JointModel.from_checkpoint(args.base, args.sft_ckpt, device=args.device,
                                       is_trainable=False)
    model.eval()
    eos_id = model.tok.eos_token_id
    pad_id = model.tok.pad_token_id

    n_written = 0
    zero_var = 0
    dropped = 0
    rew_sum = 0.0
    t0 = time.time()
    report_rows = []
    with open(args.out, "w") as fout:
        for ri, r in enumerate(rows):
            G = args.group
            path = reward_path_for_row(r)
            p_ids, p_attn = model.batch_prompts([r["instruction"]] * G)   # (G,Tp)
            with torch.no_grad():
                plan_ids = model.sample_plan(p_ids, p_attn, temp=args.temp, sample=True)  # (G,L)
                gen = model.generate_answer(p_ids, p_attn, plan_ids, temp=args.temp,
                                            sample=True, max_new_tokens=args.max_resp,
                                            top_p=args.top_p)                              # (G,Rg)
            rmask, trim = resp_mask_from_ids(gen, eos_id, pad_id)
            resp_ids = gen[:, :trim].contiguous()
            resp_attn = rmask[:, :trim].contiguous()
            # A6: behavior-policy logprobs are recomputed with the SAME HF teacher-forced path the
            # trainer uses (temp=1), NOT taken from the sampler. This is what makes logp_mismatch_t0
            # ~0 and the offline importance ratio valid at step 0.
            with torch.no_grad():
                plan_logp, _ = model.plan_logp_tf(p_ids, p_attn, plan_ids, temp=1.0)        # (G,L)
                resp_logp, _ = model.resp_logp_tf(p_ids, p_attn, plan_ids, resp_ids,
                                                  resp_attn, temp=1.0)                       # (G,R)
            # graded rewards (binary for verifiable, [0,1] for rubric — A1b)
            rewards, group_recs = [], []
            for g in range(G):
                ans = model.tok.decode(resp_ids[g][resp_attn[g].bool()], skip_special_tokens=True)
                rew = graded_reward_for_row(r, ans)
                rewards.append(rew)
                pmask = (plan_ids[g] != PAD_ID); amask = resp_attn[g].bool()
                group_recs.append({
                    "id": r["id"], "instruction": r["instruction"], "group_size": G,
                    "reward_path": path, "reward": rew,
                    "plan_tokens": plan_ids[g][pmask].tolist(),
                    "plan_logp_old": plan_logp[g][pmask].tolist(),
                    "resp_tokens": resp_ids[g][amask].tolist(),
                    "resp_logp_old": resp_logp[g][amask].tolist(),
                    "resp_len": int(amask.sum()),
                    "answer_text": ans,
                    "plan_str": decode_plan(plan_ids[g]),
                })
            rew_t = torch.tensor(rewards)
            p_q = float(rew_t.mean()); std = float(rew_t.std(unbiased=False))
            # A2: a group with no within-group spread yields no GRPO gradient (covers the literal
            # "p_q exactly 0.0 or 1.0" case for binary, and zero-spread graded groups). Mark keep.
            keep = std > 1e-6 if args.filter else True
            for rec in group_recs:
                rec["p_q"] = p_q
                rec["keep"] = bool(keep)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            report_rows.append((r["id"], path, p_q, std, int(keep)))
            rew_sum += p_q
            if std < 1e-6:
                zero_var += 1
            if not keep:
                dropped += 1
            if (ri + 1) % 10 == 0:
                print(f"[rollout] {ri+1}/{len(rows)} mean_reward={rew_sum/(ri+1):.3f} "
                      f"zero_var={zero_var}/{ri+1} dropped={dropped} ({time.time()-t0:.0f}s)")

    with open(args.report, "w", newline="") as rf:
        w = csv.writer(rf); w.writerow(["id", "reward_path", "p_q", "std", "kept"])
        w.writerows(report_rows)

    print(f"[rollout] wrote {n_written} rollouts -> {args.out}")
    print(f"[rollout] mean reward={rew_sum/len(rows):.3f} | zero-variance groups="
          f"{zero_var}/{len(rows)} ({zero_var/len(rows)*100:.1f}%) | A2-dropped={dropped} "
          f"(kept groups go to GRPO). Report -> {args.report}")


if __name__ == "__main__":
    main()
