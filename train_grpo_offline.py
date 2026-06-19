#!/usr/bin/env python3
"""
train_grpo_offline.py — Stage 2b: off-policy GRPO from the cached rollouts (no generation).

Reads rl_rollouts.jsonl (contiguous blocks of G per instruction), recomputes the CURRENT
policy's per-token logprobs for the saved plan/response tokens, and minimizes the off-policy
clipped objective from grpo_offpolicy.joint_grpo_loss:

    L = L_exec_clip + beta_plan * L_plan_clip + beta_ce * CE_plan_anchor
        [+ kl_resp * KL(exec resp_new || SFT resp_old)   when --kl_resp > 0]

The optional --kl_resp term anchors the FINAL ANSWER distribution to SFT. With --freeze_executor the
LoRA is frozen, so the executor's SFT vs RL responses differ ONLY because the planner changed the plan
prefix; this KL (k3, non-negative) penalizes that drift and is differentiable w.r.t. plan_emb (the soft
plan prefix the frozen executor attends over), so it trains the planner to steer the executor without
dragging its answers away from SFT. Reference = the cached behavior logp, which IS the SFT executor
because rollouts are generated once from the SFT checkpoint.

The crux is OFF-POLICY CORRECTION. We reuse each group for `inner_epochs` gradient steps; after
step 1 the policy drifts from the saved behavior policy, so each token's gradient is scaled by
    ratio = exp(logp_new - logp_old)   (per token, clipped to [1-eps, 1+eps]).
The ratio CORRECTS sampling staleness; it is not a reward bonus. On the first pass ratio==1, so
this reduces to plain reinforce. Watch approx_kl per inner epoch — if it climbs past ~0.10-0.15
the rollouts are stale: cut inner epochs (before cutting lr) and regenerate.

Architecture-specific correctness (handled by grpo_offpolicy, fed correctly here):
  (1) executor ratio is computed over RESPONSE tokens only — the resp logp tensor never contains
      prompt or plan tokens, so plan logprobs cannot contaminate the executor ratio.
  (2) planner and executor are SEPARATE action spaces -> two independent ratios/advantages.

Addenda wired here:
  A1/A1b  MaxEnt prompt weighting via adv_weights — verifiable prompts use mgpo_weight(p_q) (bell on
          group accuracy), rubric/judge prompts use variance_weight (reward spread). Replaces the
          "delete zero-variance groups" hack; logs the p_q histogram and mean weight per path.
  A2      train only on keep=True groups (zero-spread groups dropped upstream in rollout_offline).
  A5      --long2short: zero-sum brevity reward shaping among CORRECT trajectories, BEFORE advantages.
  A6      logp_mismatch_t0 assertion before any update — the saved logp_old must match the trainer's
          recomputed logp on the first pass, or the importance ratio is corrupted at step 0.

Startup ASSERTS >0 trainable backbone tensors (guards the frozen-backbone no-op bug).

CLI:
  python train_grpo_offline.py --rollouts rl_rollouts.jsonl --sft_ckpt joint_ckpt \
      --out rl_ckpt --inner_epochs 3 --lr 1e-4 --clip_eps 0.2 --beta_plan 1.0 --beta_ce 0.1 \
      --device cuda
"""
import argparse, json, copy, os, time
from collections import Counter
import torch, torch.nn.functional as F

from model_joint import JointModel, encode_plan, PAD_ID, decode_plan, N_PLAN
from grpo_offpolicy import (joint_grpo_loss, mgpo_weight, variance_weight, group_pq,
                            long2short_shape)
from checkers import graded_reward_for_row
from checkpointing import (Checkpointer, load_train_state, restore_optimizer, restore_rng,
                           scalar_args)


def _grpo_state(inner_epoch, group_idx, global_step, opt, args):
    """Resume payload: position = next (inner_epoch, group_idx) to run."""
    import random
    return {"kind": "grpo", "inner_epoch": inner_epoch, "group_idx": group_idx,
            "global_step": global_step, "n_plan": N_PLAN, "optimizer": opt.state_dict(),
            "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(),
            "args": scalar_args(args)}


def load_groups(path, use_filter=True, exclude_rubric=False):
    """Chunk the rollout file into contiguous groups of size group_size.
    A2: drop groups marked keep=False (no within-group spread -> no GRPO gradient).
    --exclude_rubric: hold soft (rubric) families OUT of RL entirely (rely on SFT)."""
    recs = [json.loads(l) for l in open(path)]
    groups, i, n_drop_filter, n_drop_rubric = [], 0, 0, 0
    while i < len(recs):
        G = recs[i]["group_size"]
        grp = recs[i:i+G]; i += G
        if use_filter and not grp[0].get("keep", True):
            n_drop_filter += 1; continue
        if exclude_rubric and grp[0].get("reward_path") == "rubric":
            n_drop_rubric += 1; continue
        groups.append(grp)
    print(f"[grpo] groups kept={len(groups)} dropped(A2 zero-spread)={n_drop_filter} "
          f"dropped(rubric held-out)={n_drop_rubric}")
    return groups


def pad_to(seqs, fill, dtype, device):
    L = max((len(s) for s in seqs), default=1)
    L = max(L, 1)
    out = torch.full((len(seqs), L), fill, dtype=dtype, device=device)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, :len(s)] = torch.tensor(s, dtype=dtype, device=device)
    return out


def _trim_width(old, like):
    """Pad/truncate a saved (B,L_old) compacted logp tensor to the compacted (B,L_new) width."""
    B, Ln = like.shape
    out = torch.zeros(B, Ln, dtype=old.dtype, device=old.device)
    L = min(old.shape[1], Ln)
    out[:, :L] = old[:, :L]
    return out


def _compact(logp_full, mask_full):
    """COMPACT a scattered (B,S) per-position logp + its (B,S) mask down to (B,Lmax) where each row
    holds only its masked values left-aligned (the same compacted layout rollout_offline saved). Both
    the rollout (logp_old) and the trainer (logp_new) compact by iterating the mask in sequence order,
    so the two compacted tensors align position-for-position. Returns (compact_logp, compact_mask)."""
    B = logp_full.size(0)
    rows = [logp_full[b][mask_full[b].bool()] for b in range(B)]
    Lmax = max((r.numel() for r in rows), default=1)
    Lmax = max(Lmax, 1)
    out = torch.zeros(B, Lmax, dtype=logp_full.dtype, device=logp_full.device)
    cmask = torch.zeros(B, Lmax, dtype=torch.float, device=logp_full.device)
    for b, r in enumerate(rows):
        out[b, :r.numel()] = r
        cmask[b, :r.numel()] = 1.0
    return out, cmask


def build_group_tensors_interleaved(model, group):
    """Interleaved variant: recompute temp=1 logp over the WHOLE multi-turn trajectory via
    interleaved_logp_tf and return ONE concatenated plan stream + ONE concatenated resp stream per
    trajectory. These ARE the PLAN/RESP position sets joint_grpo_loss consumes UNCHANGED (exec_mask =
    response tokens only; plan ratio over plan-vocab steps only). resp_logp_new doubles as both the
    teacher-forced new logp AND, at pass 0, matches resp_logp_old (A6)."""
    instrs = [g["instruction"] for g in group]
    p_ids, p_attn = model.batch_prompts(instrs)
    turns_batch = [g["turns"] for g in group]
    plan_logp_old = pad_to([g["plan_logp_old"] for g in group], 0.0, torch.float, model.device)
    resp_logp_old = pad_to([g["resp_logp_old"] for g in group], 0.0, torch.float, model.device)
    rewards = torch.tensor([g["reward"] for g in group], dtype=torch.float, device=model.device)
    lengths = torch.tensor([g.get("resp_len", len(g["resp_tokens"])) for g in group],
                           dtype=torch.float, device=model.device)
    return p_ids, p_attn, turns_batch, plan_logp_old, resp_logp_old, rewards, lengths


def build_group_tensors(model, group):
    """From G rollout records -> padded tensors for one group (B=G)."""
    instrs = [g["instruction"] for g in group]
    p_ids, p_attn = model.batch_prompts(instrs)
    plan_ids = pad_to([g["plan_tokens"] for g in group], PAD_ID, torch.long, model.device)
    resp_ids = pad_to([g["resp_tokens"] for g in group], model.tok.pad_token_id, torch.long, model.device)
    resp_attn = pad_to([[1]*len(g["resp_tokens"]) for g in group], 0, torch.long, model.device).float()
    plan_logp_old = pad_to([g["plan_logp_old"] for g in group], 0.0, torch.float, model.device)
    resp_logp_old = pad_to([g["resp_logp_old"] for g in group], 0.0, torch.float, model.device)
    rewards = torch.tensor([g["reward"] for g in group], dtype=torch.float, device=model.device)
    lengths = torch.tensor([g.get("resp_len", len(g["resp_tokens"])) for g in group],
                           dtype=torch.float, device=model.device)
    return (p_ids, p_attn, plan_ids, resp_ids, resp_attn,
            plan_logp_old, resp_logp_old, rewards, lengths)


@torch.no_grad()
def adapter_l2(model):
    return float(torch.sqrt(sum((p.detach()**2).sum()
                 for n, p in model.backbone.named_parameters() if "lora_" in n)))


@torch.no_grad()
def held_reward(model, held_rows, max_resp, temp=0.7):
    """Generate (sampled, with a sampled plan) on a held set and score -> shows RL moving."""
    if not held_rows:
        return None
    model.eval()
    corr = 0
    for r in held_rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]])
        plan = model.sample_plan(p_ids, p_attn, temp=temp, sample=True)
        gen = model.generate_answer(p_ids, p_attn, plan, temp=temp, sample=True,
                                    max_new_tokens=max_resp)
        ans = model.tok.decode(gen[0], skip_special_tokens=True)
        corr += graded_reward_for_row(r, ans)
    model.train()
    return corr / len(held_rows)


@torch.no_grad()
def logp_mismatch_t0(model, group):
    """A6: BEFORE any update, the trainer-recomputed logp must match the saved logp_old (same
    numerical path). If this isn't ~0 the importance ratio is corrupted at step 0 and no clipping
    saves it. Returns mean |logp_train - logp_saved| over masked plan+resp tokens."""
    (p_ids, p_attn, plan_ids, resp_ids, resp_attn,
     plan_logp_old, resp_logp_old, _, _) = build_group_tensors(model, group)
    plan_new, pmask = model.plan_logp_tf(p_ids, p_attn, plan_ids, temp=1.0)
    resp_new, rmask = model.resp_logp_tf(p_ids, p_attn, plan_ids, resp_ids, resp_attn, temp=1.0)
    dp = ((plan_new - plan_logp_old).abs() * pmask).sum() / pmask.sum().clamp_min(1)
    dr = ((resp_new - resp_logp_old).abs() * rmask).sum() / rmask.sum().clamp_min(1)
    return float((dp + dr) / 2)


@torch.no_grad()
def logp_mismatch_t0_interleaved(model, group):
    """A6 for the interleaved path: trainer-recomputed concatenated plan+resp logp must match the saved
    logp_old over masked positions (exec_ratio~1 at pass 0)."""
    (p_ids, p_attn, turns_batch, plan_logp_old, resp_logp_old, _, _) = \
        build_group_tensors_interleaved(model, group)
    pl_full, pm_full, rl_full, rm_full = model.interleaved_logp_tf(p_ids, p_attn, turns_batch, temp=1.0)
    plan_new, pmask = _compact(pl_full, pm_full); resp_new, rmask = _compact(rl_full, rm_full)
    plan_old = _trim_width(plan_logp_old, pmask); resp_old = _trim_width(resp_logp_old, rmask)
    dp = ((plan_new - plan_old).abs() * pmask).sum() / pmask.sum().clamp_min(1)
    dr = ((resp_new - resp_old).abs() * rmask).sum() / rmask.sum().clamp_min(1)
    return float((dp + dr) / 2)


@torch.no_grad()
def held_reward_interleaved(model, held_rows, max_turns, max_resp, temp=0.7):
    """Interleaved held reward: run_interleaved (sampled), grade the last turn's prose tail."""
    if not held_rows:
        return None
    model.eval()
    corr = 0
    for r in held_rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]])
        rec = model.run_interleaved(p_ids[0], p_attn[0], temp=temp, sample=True,
                                    max_turns=max_turns, max_resp=max_resp)
        corr += graded_reward_for_row(r, model.interleaved_answer_text(rec))
    model.train()
    return corr / len(held_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="rl_rollouts.jsonl")
    ap.add_argument("--sft_ckpt", default="joint_ckpt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="dataset/sft_100.jsonl", help="for CE anchor gold + held eval")
    ap.add_argument("--out", default="rl_ckpt")
    ap.add_argument("--inner_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--beta_plan", type=float, default=1.0)
    ap.add_argument("--beta_ce", type=float, default=0.1)
    ap.add_argument("--kl_resp", type=float, default=0.0,
                    help="KL-to-SFT anchor on RESPONSE tokens: kl_resp * KL(exec resp_new || SFT "
                         "resp_old), k3 estimator (non-negative). Penalizes the planner for steering "
                         "the (frozen) executor's answers away from the SFT response distribution. "
                         "0 = off. Reference = cached behavior logp == SFT executor (rollouts come "
                         "from the SFT ckpt). Differentiable w.r.t. plan_emb.")
    ap.add_argument("--lam_resp", type=float, default=1.0,
                    help="executor clipped-objective weight (forced to 0 by --freeze_executor)")
    ap.add_argument("--max_resp", type=int, default=64)
    ap.add_argument("--kl_stop", type=float, default=0.15, help="cut inner epochs if approx_kl exceeds this")
    ap.add_argument("--held", type=int, default=16, help="held instructions for held_reward (0=skip)")
    # A1/A1b MaxEnt weighting
    ap.add_argument("--maxent", action="store_true", default=True,
                    help="A1/A1b: weight prompts by signal carried (replaces zero-var deletion)")
    ap.add_argument("--no-maxent", dest="maxent", action="store_false")
    ap.add_argument("--gamma", type=float, default=2.0, help="A1 MGPO bell width; gamma=1/(2*delta^2)")
    ap.add_argument("--std_ref", type=float, default=0.5, help="A1b variance-weight normalizer")
    # A2 / routing
    ap.add_argument("--filter", action="store_true", default=True, help="A2: drop keep=False groups")
    ap.add_argument("--no-filter", dest="filter", action="store_false")
    ap.add_argument("--exclude_rubric", action="store_true", help="hold soft (rubric) families out of RL")
    # A5 long2short
    ap.add_argument("--long2short", action="store_true", help="A5: brevity reward shaping among correct")
    ap.add_argument("--l2s_lam", type=float, default=0.2)
    ap.add_argument("--mismatch_tol", type=float, default=1e-3, help="A6 logp_mismatch_t0 tolerance")
    # checkpoint / resume
    ap.add_argument("--ckpt_every_min", type=float, default=0.0,
                    help="periodic checkpoint interval in minutes (0 = only at inner-epoch boundaries)")
    ap.add_argument("--ckpt_every_steps", type=int, default=0,
                    help="periodic checkpoint every N group steps (0 = off)")
    ap.add_argument("--hf_repo", default=None,
                    help="push each checkpoint to this HF model repo (e.g. user/small_fable-planner)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --out if it contains a train_state.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--interleaved", action="store_true",
                    help="agentic GRPO: concatenated plan/resp streams via interleaved_logp_tf, fed "
                         "to the UNCHANGED joint_grpo_loss.")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--freeze_executor", action="store_true",
                    help="PLANNER-ONLY RL: freeze LoRA adapters + set lam_resp=0 so ONLY the planner "
                         "head + plan_emb train (clipped plan-policy term + beta_ce CE anchor).")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # --- load model FOR TRAINING. is_trainable=True is mandatory (frozen-backbone guard).
    #     On --resume, continue from the RL checkpoint (--out); else start from the SFT checkpoint. ---
    resume_state = load_train_state(args.out) if (args.resume and os.path.isdir(args.out)) else None
    if resume_state is not None:   # don't resume a checkpoint from a DIFFERENT data/rollouts/vocab
        prev = resume_state.get("args", {})
        if (prev.get("data") != args.data or prev.get("rollouts") != args.rollouts
                or resume_state.get("n_plan") != N_PLAN):
            print(f"[grpo] checkpoint config differs (data={prev.get('data')}, rollouts="
                  f"{prev.get('rollouts')}, n_plan={resume_state.get('n_plan')}) — ignoring --resume, "
                  f"starting fresh from {args.sft_ckpt}.")
            resume_state = None
    load_from = args.out if resume_state is not None else args.sft_ckpt
    if resume_state is not None:
        print(f"[grpo] RESUMING from {args.out}: inner_epoch={resume_state['inner_epoch']} "
              f"group={resume_state['group_idx']} step={resume_state['global_step']}")
    model = JointModel.from_checkpoint(args.base, load_from, device=args.device,
                                       is_trainable=True)
    if args.interleaved:
        model.interleaved = True
        model._assert_interleaved()
    # PLANNER-ONLY RL: freeze the LoRA executor adapters + zero lam_resp so ONLY the planner head and
    # plan embeddings train (via the clipped plan-policy term + the beta_ce CE anchor).
    if args.freeze_executor:
        for n, p in model.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
        args.lam_resp = 0.0
        n_planner = sum(p.numel() for p in model.planner.parameters() if p.requires_grad) \
                    + sum(p.numel() for p in model.plan_emb.parameters() if p.requires_grad)
        assert n_planner > 0, ("FROZEN-PLANNER BUG: --freeze_executor but 0 trainable PLANNER params "
                               "(planner head + plan_emb). Nothing would train.")
        print(f"[grpo] PLANNER-ONLY RL: executor (LoRA) FROZEN; trainable planner params={n_planner} "
              f"(planner head + plan_emb). lam_resp forced to 0.")
    else:
        args.lam_resp = getattr(args, "lam_resp", 1.0)
        n_bb = model.n_trainable_backbone()
        assert n_bb > 0, "FROZEN-BACKBONE BUG: 0 trainable backbone tensors -> RL is a no-op."
        print(f"[grpo] >0 trainable backbone tensors: {n_bb} (≈336 expected) -- RL will actually move.")

    # T4 memory: the fp32 1.5B + three forward graphs over a group of G don't fit at full activation
    # footprint. Gradient checkpointing recomputes activations in backward instead of storing them.
    try:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()   # needed for checkpointing with inputs_embeds
        print("[grpo] gradient checkpointing ON (lower activation memory).")
    except Exception as e:
        print(f"[grpo] gradient checkpointing unavailable ({e})")

    groups = load_groups(args.rollouts, use_filter=args.filter, exclude_rubric=args.exclude_rubric)
    assert groups, "no groups left after filtering — loosen --no-filter or regenerate hotter rollouts"

    # gold plans/answers for the CE anchor + held set
    data = {r["id"]: r for r in (json.loads(l) for l in open(args.data))}
    train_ids = {g[0]["id"] for g in groups}
    held_rows = [r for rid, r in data.items() if rid not in train_ids][:args.held]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # checkpoint + resume wiring
    ckpt = Checkpointer(args.out, args.base, every_min=args.ckpt_every_min,
                        every_steps=args.ckpt_every_steps, hf_repo=args.hf_repo)
    start_ie = start_gi = global_step = 0
    if resume_state is not None:
        restore_optimizer(opt, resume_state, model.device)
        restore_rng(resume_state)
        start_ie, start_gi, global_step = (resume_state["inner_epoch"], resume_state["group_idx"],
                                           resume_state["global_step"])

    l2_start = adapter_l2(model)
    plan_dist_before = Counter(p for g in groups for rec in g for p in rec["plan_str"])

    rollout_mean_reward = sum(rec["reward"] for g in groups for rec in g) / max(
        1, sum(len(g) for g in groups))
    print(f"[grpo] rollout mean reward (static) = {rollout_mean_reward:.3f}")

    # A6: validate the offline importance ratio is sane at step 0 (BEFORE any update). Skip on resume:
    # after some training the policy has drifted, so logp_old no longer matches by design.
    if resume_state is None:
        mm = (logp_mismatch_t0_interleaved(model, groups[0]) if args.interleaved
              else logp_mismatch_t0(model, groups[0]))
        print(f"[grpo] logp_mismatch_t0 = {mm:.3e} (must be ~0; tol={args.mismatch_tol:g})")
        if mm > args.mismatch_tol:
            print("[grpo] WARNING: logp_mismatch_t0 exceeds tol — saved logp_old was recorded with a "
                  "DIFFERENT numerical path than training. Re-record logp_old via the HF teacher-forced "
                  "forward (rollout_offline already does this). The IS ratio is invalid until this is ~0.")
        if args.held:
            hr0 = (held_reward_interleaved(model, held_rows, args.max_turns, args.max_resp)
                   if args.interleaved else held_reward(model, held_rows, args.max_resp))
            print(f"[grpo] held_reward(before RL) = {hr0:.3f}")

    # A1b: per-prompt p_q histogram (logged each inner epoch so the operator sees the bell working)
    def pq_hist(vals):
        bins = [0, 0, 0, 0, 0]  # [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]
        for v in vals:
            bins[min(int(v * 5), 4)] += 1
        return bins

    for ie in range(start_ie, args.inner_epochs):
        model.train(); t0 = time.time()
        agg = {"total_loss": 0.0, "l_exec": 0.0, "exec_approx_kl": 0.0, "exec_clip_frac": 0.0,
               "plan_approx_kl": 0.0, "zero_var_frac": 0.0, "kl_resp": 0.0, "n": 0}
        w_by_path = {"verifiable": [], "rubric": [], "judge": []}
        std_by_path = {"verifiable": [], "rubric": [], "judge": []}
        pqs = []
        sgi = start_gi if ie == start_ie else 0   # mid-inner-epoch resume only on the resumed epoch
        for gi in range(len(groups)):
            if gi < sgi:
                continue
            group = groups[gi]
            if args.interleaved:
                (p_ids, p_attn, turns_batch, plan_logp_old, resp_logp_old,
                 rewards, lengths) = build_group_tensors_interleaved(model, group)
            else:
                (p_ids, p_attn, plan_ids, resp_ids, resp_attn,
                 plan_logp_old, resp_logp_old, rewards, lengths) = build_group_tensors(model, group)
            G = len(group)
            path = group[0].get("reward_path", "verifiable")

            # A5: brevity reward shaping among CORRECT trajectories, BEFORE advantages (zero-sum).
            if args.long2short:
                rewards = long2short_shape(rewards, lengths, G, lam=args.l2s_lam)

            # A1/A1b: prompt weight = how much signal the group carries. verifiable -> MGPO bell on
            # group accuracy; rubric/judge -> direct reward-spread (variance) weight.
            pq = group_pq(rewards, G)
            if args.maxent:
                if path == "verifiable":
                    adv_weights = mgpo_weight(pq, gamma=args.gamma)
                else:
                    adv_weights = variance_weight(rewards, G, std_ref=args.std_ref)
            else:
                adv_weights = None
            pqs.append(float(pq[0]))
            if adv_weights is not None:
                w_by_path[path].append(float(adv_weights.mean()))
            std_by_path[path].append(float(rewards.view(-1, G).std(unbiased=False).mean()))

            # current-policy logprobs for the SAME saved tokens (temp=1 -> the policy). The interleaved
            # path recomputes ONE concatenated plan stream + ONE resp stream over the whole trajectory;
            # the flat path scores plan and resp segments separately. BOTH feed the UNCHANGED loss.
            if args.interleaved:
                pl_full, pm_full, rl_full, rm_full = model.interleaved_logp_tf(
                    p_ids, p_attn, turns_batch, temp=1.0)
                # COMPACT the scattered (B,S) recomputed logp to the left-aligned layout rollout_offline
                # saved logp_old in, so logp_new and logp_old align position-for-position. Then trim the
                # saved (already-compacted) logp_old to the same width.
                plan_logp_new, plan_mask = _compact(pl_full, pm_full)
                resp_logp_new, resp_mask = _compact(rl_full, rm_full)
                plan_logp_old = _trim_width(plan_logp_old, plan_mask)
                resp_logp_old = _trim_width(resp_logp_old, resp_mask)
            else:
                plan_logp_new, plan_mask = model.plan_logp_tf(p_ids, p_attn, plan_ids, temp=1.0)
                resp_logp_new, resp_mask = model.resp_logp_tf(p_ids, p_attn, plan_ids,
                                                              resp_ids, resp_attn, temp=1.0)

            # small CE anchor toward the gold plan (keeps planner from wandering off-vocab)
            gold = data[group[0]["id"]]
            gold_plan = encode_plan(gold["plan"], model.plan_max_len).unsqueeze(0).to(model.device)
            gp_logits = model.planner_logits_tf(p_ids[:1], p_attn[:1], gold_plan)
            gp_mask = (gold_plan != PAD_ID)
            ce_plan = (F.cross_entropy(gp_logits.reshape(-1, gp_logits.size(-1)),
                                       gold_plan.reshape(-1), reduction="none").reshape(gold_plan.shape)
                       * gp_mask).sum() / gp_mask.sum().clamp_min(1)

            loss, logs = joint_grpo_loss(
                rewards=rewards, group_size=G,
                exec_logp_new=resp_logp_new, exec_logp_old=resp_logp_old, exec_mask=resp_mask,
                plan_logp_new=plan_logp_new, plan_logp_old=plan_logp_old, plan_mask=plan_mask,
                beta_plan=args.beta_plan, clip_eps=args.clip_eps, lam_resp=args.lam_resp,
                ce_plan=ce_plan, beta_ce=args.beta_ce, adv_weights=adv_weights)

            # KL-to-SFT anchor on response tokens (k3, non-negative). delta = resp_logp_new (grad,
            # via plan_emb) - resp_logp_old (the cached SFT reference, constant). k3 pulls new->old
            # from both sides, so the planner can't drag the frozen executor's answers off the SFT
            # distribution. Clamp delta for exp() numerical safety.
            if args.kl_resp > 0:
                delta = (resp_logp_new - resp_logp_old).clamp(-10.0, 10.0)
                k3 = torch.exp(-delta) - 1.0 + delta
                kl_resp = (k3 * resp_mask).sum() / resp_mask.sum().clamp_min(1)
                loss = loss + args.kl_resp * kl_resp
                logs["kl_resp"] = float(kl_resp)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            global_step += 1
            for k in ("total_loss", "l_exec", "exec_approx_kl", "exec_clip_frac",
                      "plan_approx_kl", "zero_var_frac", "kl_resp"):
                agg[k] += logs.get(k, 0.0)
            agg["n"] += 1
            if ckpt.due(global_step):
                ckpt.save(model, _grpo_state(ie, gi + 1, global_step, opt, args),
                          reason=f"grpo-ie{ie+1}-g{gi+1}")

        n = max(agg["n"], 1)
        if args.held:
            hr = (held_reward_interleaved(model, held_rows, args.max_turns, args.max_resp)
                  if args.interleaved else held_reward(model, held_rows, args.max_resp))
        else:
            hr = None
        ekl = agg["exec_approx_kl"]/n
        wv = sum(w_by_path["verifiable"])/max(1, len(w_by_path["verifiable"]))
        wr = sum(w_by_path["rubric"])/max(1, len(w_by_path["rubric"]))
        print(f"[grpo] inner_epoch {ie+1}/{args.inner_epochs} ({time.time()-t0:.1f}s) "
              f"loss={agg['total_loss']/n:.4f} l_exec={agg['l_exec']/n:.4f} "
              f"exec_approx_kl={ekl:.4f} exec_clip_frac={agg['exec_clip_frac']/n:.3f} "
              f"plan_approx_kl={agg['plan_approx_kl']/n:.4f} "
              + (f"kl_resp(to-SFT)={agg['kl_resp']/n:.4f} " if args.kl_resp > 0 else "")
              + (f"held_reward={hr:.3f} " if hr is not None else "")
              + f"executor={'FROZEN' if args.freeze_executor else 'trainable'} "
              + f"| mean_w(verif)={wv:.2f} mean_w(rubric)={wr:.2f} "
              f"p_q_hist[0-1]={pq_hist(pqs)}")
        # end-of-inner-epoch checkpoint: next position = (ie+1, 0)
        ckpt.save(model, _grpo_state(ie + 1, 0, global_step, opt, args), reason=f"grpo-inner{ie+1}")
        if ekl > args.kl_stop:
            print(f"[grpo] approx_kl {ekl:.3f} > kl_stop {args.kl_stop}: rollouts are stale. "
                  "Stopping inner epochs; REGENERATE rollouts before continuing.")
            break

    # final checkpoint: position past the end (resume would be a no-op)
    ckpt.save(model, _grpo_state(args.inner_epochs, 0, global_step, opt, args), reason="final")
    l2_end = adapter_l2(model)
    print(f"[grpo] saved -> {args.out}")
    print(f"[grpo] adapter L2: start={l2_start:.4f} end={l2_end:.4f} "
          f"|Δ|={abs(l2_end-l2_start):.4f}  (must be > 0 to prove RL ≠ SFT)")
    print("[grpo] plan-token distribution (rollout/before):",
          dict(plan_dist_before.most_common(8)))
    print("[grpo] ACCEPTANCE: zero_var_frac LOW; held_reward MOVED across inner epochs; |ΔL2|>0.")


if __name__ == "__main__":
    main()
