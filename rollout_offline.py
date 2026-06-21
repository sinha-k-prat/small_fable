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

def _gpu_mem() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"gpu: alloc={alloc:.1f}GB peak={peak:.1f}GB/{total:.0f}GB ({100*peak/total:.0f}%)"

from model_joint import JointModel, PAD_ID, decode_plan
from checkers import graded_reward_for_row
from reward_paths import reward_path_for_row


def resp_mask_from_ids(ids, eos_id, pad_id):
    """Build a 1/0 mask over generated ids: 1 up to and including the first EOS, 0 after.
    Vectorised: no Python loop over B*T. Returns (mask, trim_len)."""
    eos_flags = (ids == eos_id).float()               # 1 exactly at EOS positions
    # cumsum > 1 means we are strictly AFTER the first EOS token
    after_eos = eos_flags.cumsum(dim=1) > 1.0
    mask = (~after_eos).float()                        # 1 up to and including the first EOS
    # trim: last column index where any row is still active, plus 1
    active_cols = mask.any(dim=0).nonzero(as_tuple=False)
    trim = int(active_cols[-1]) + 1 if len(active_cols) else 1
    return mask, max(trim, 1)


def _rec_to_turns(model, rec):
    """run_interleaved record {turns:[{plan:[ids],resp:[ids]}]} -> the turns_batch schema
    interleaved_tf/_build_interleaved_row consume: [{'plan':[str], 'response':str}]. Plan ids are
    decoded to primitive names; the EOP/FINALIZE markers are stripped (the assembler re-inserts them);
    the response ids are decoded to text (RESP_EOS stripped, re-inserted by the assembler)."""
    from model_joint import ID2PLAN, EOP_ID, FINALL_ID, BOP_ID
    turns = []
    for tn in rec["turns"]:
        plan_names = [ID2PLAN[int(i)] for i in tn["plan"]
                      if int(i) not in (EOP_ID, FINALL_ID, BOP_ID) and int(i) in ID2PLAN]
        resp_ids = [i for i in tn["resp"] if i != model.resp_eos_id]
        if not resp_ids:                       # a finalize-only turn carries no response
            continue
        text = model.tok.decode(resp_ids, skip_special_tokens=True)
        turns.append({"plan": plan_names, "response": text})
    if not turns:                              # guarantee at least one gradeable turn
        turns = [{"plan": [], "response": ""}]
    return turns


def _main_interleaved(args):
    """Agentic closed-loop rollout. For each instruction draw G full trajectories with run_interleaved,
    grade the last turn's prose tail, then recompute temp=1 teacher-forced logp via interleaved_logp_tf
    and CONCATENATE all turns' plan tokens into one plan stream and all turns' resp tokens into one resp
    stream — exactly the PLAN/RESP position sets joint_grpo_loss consumes (A6 ratio==1 at pass 0)."""
    from model_joint import JointModel as _JM
    rows = [json.loads(l) for l in open(args.data)][:args.train]
    print(f"[rollout:interleaved] {len(rows)} instructions x G={args.group} @ temp={args.temp}")
    model = _JM.from_checkpoint(args.base, args.sft_ckpt, device=args.device, is_trainable=False)
    model.eval()
    model._assert_interleaved()

    n_written = zero_var = dropped = 0
    rew_sum = 0.0
    t0 = time.time()
    report_rows = []
    with open(args.out, "w") as fout:
        for ri, r in enumerate(rows):
            G = args.group
            path = reward_path_for_row(r)
            p_ids, p_attn = model.batch_prompts([r["instruction"]])
            rewards, group_recs = [], []
            for g in range(G):
                rec = model.run_interleaved(p_ids[0], p_attn[0], temp=args.temp, sample=True,
                                            max_turns=args.max_turns, max_plan=args.max_plan,
                                            max_resp=args.max_resp)
                turns = _rec_to_turns(model, rec)
                ans = model.interleaved_answer_text(rec)
                rew = graded_reward_for_row(r, ans)
                rewards.append(rew)
                # A6: recompute temp=1 logp via the SAME teacher-forced path the trainer uses, then
                # CONCATENATE across turns into one plan stream + one resp stream.
                with torch.no_grad():
                    pl, pm, rl, rm = model.interleaved_logp_tf(p_ids, p_attn, [turns], temp=1.0)
                pm0 = pm[0].bool(); rm0 = rm[0].bool()
                # the recorded token id at each masked position == the teacher-forced target there.
                (_h, _lm, _pm, ptgt, _rm, rtgt, _a, raw, _e) = model.interleaved_tf(
                    p_ids, p_attn, [turns])
                plan_tokens = raw[0][pm0].tolist()
                resp_tokens = raw[0][rm0].tolist()
                group_recs.append({
                    "id": r["id"], "instruction": r["instruction"], "group_size": G,
                    "reward_path": path, "reward": rew, "interleaved": True,
                    "turns": turns,
                    "plan_tokens": plan_tokens, "plan_logp_old": pl[0][pm0].tolist(),
                    "resp_tokens": resp_tokens, "resp_logp_old": rl[0][rm0].tolist(),
                    "resp_len": int(rm0.sum()), "answer_text": ans,
                    "plan_str": [ID for tn in turns for ID in tn["plan"]],
                })
            rew_t = torch.tensor(rewards)
            p_q = float(rew_t.mean()); std = float(rew_t.std(unbiased=False))
            keep = std > 1e-6 if args.filter else True
            for rec in group_recs:
                rec["p_q"] = p_q; rec["keep"] = bool(keep)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            report_rows.append((r["id"], path, p_q, std, int(keep)))
            rew_sum += p_q
            if std < 1e-6:
                zero_var += 1
            if not keep:
                dropped += 1
            if (ri + 1) % 10 == 0:
                print(f"[rollout:interleaved] {ri+1}/{len(rows)} mean_reward={rew_sum/(ri+1):.3f} "
                      f"zero_var={zero_var}/{ri+1} dropped={dropped} ({time.time()-t0:.0f}s)")
    with open(args.report, "w", newline="") as rf:
        w = csv.writer(rf); w.writerow(["id", "reward_path", "p_q", "std", "kept"])
        w.writerows(report_rows)
    print(f"[rollout:interleaved] wrote {n_written} rollouts -> {args.out}")
    print(f"[rollout:interleaved] mean reward={rew_sum/max(1,len(rows)):.3f} | zero-variance groups="
          f"{zero_var}/{len(rows)} | A2-dropped={dropped}. Report -> {args.report}")


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
    ap.add_argument("--dtype", default=None,
                    help="model dtype: float32 | bfloat16 | auto (default: bfloat16 on CUDA, float32 on CPU)")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--interleaved", action="store_true",
                    help="agentic closed-loop rollout: run_interleaved + ONE concatenated plan/resp "
                         "stream per trajectory (fed UNCHANGED to joint_grpo_loss).")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_plan", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    _dtype = None
    if args.dtype and args.dtype != "auto":
        _dtype = getattr(torch, args.dtype)

    rows = [json.loads(l) for l in open(args.data)][:args.train]
    print(f"[rollout] {len(rows)} instructions x G={args.group} @ temp={args.temp}")

    # Load the SFT model. (Generation only -> trainability irrelevant here, but we keep the
    # frozen-backbone guard live downstream in train_grpo_offline.py.)
    model = JointModel.from_checkpoint(args.base, args.sft_ckpt, device=args.device,
                                       dtype=_dtype, is_trainable=False)
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
                      f"zero_var={zero_var}/{ri+1} dropped={dropped} ({time.time()-t0:.0f}s) "
                      f"| {_gpu_mem()}")

    with open(args.report, "w", newline="") as rf:
        w = csv.writer(rf); w.writerow(["id", "reward_path", "p_q", "std", "kept"])
        w.writerows(report_rows)

    print(f"[rollout] wrote {n_written} rollouts -> {args.out}")
    print(f"[rollout] mean reward={rew_sum/len(rows):.3f} | zero-variance groups="
          f"{zero_var}/{len(rows)} ({zero_var/len(rows)*100:.1f}%) | A2-dropped={dropped} "
          f"(kept groups go to GRPO). Report -> {args.report}")


if __name__ == "__main__":
    main()
