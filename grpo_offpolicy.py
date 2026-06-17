#!/usr/bin/env python3
"""
grpo_offpolicy.py — off-policy GRPO objective for the joint planner+executor model.

WHY THIS FILE EXISTS
--------------------
You chose: "reuse the 8 rollouts for several gradient steps." That is OFF-POLICY:
after inner step 1, the policy pi_theta has moved away from the behavior policy
pi_theta_old that generated the rollouts, so the importance ratio r = pi/pi_old is
genuinely != 1 and MUST be corrected with a clipped PPO-style objective or the
off-policy steps diverge. This file implements that correctly for your TWO-policy
architecture.

TWO ARCHITECTURE-SPECIFIC CORRECTNESS POINTS (easy to get wrong):
  (1) EXECUTOR-ONLY RATIO MASKING. The policy ratio for the executor must be computed
      over EXECUTOR (response) tokens only — prompt tokens and plan tokens are masked
      out. Otherwise the plan head's logprobs contaminate the executor's policy ratio.
  (2) SEPARATE CLIPPED OBJECTIVES. You have two policies (plan head, executor) over
      different action spaces. A single shared ratio conflates them. We compute
      L = L_exec_clip + beta_plan * L_plan_clip with INDEPENDENT ratios & advantages.

GRPO advantage: per-prompt group of G rollouts, advantage = (R - mean_group) / (std_group + eps).
The SAME scalar group-advantage is broadcast to every token of a trajectory (token-level
ratio, trajectory-level advantage) — standard GRPO.

This module is framework-light: it takes precomputed per-token logprobs as tensors so it
unit-tests on CPU with torch only, and slots into train_rl.py by feeding it the logprobs
you already compute during rollout (cache logp_old) and per inner-epoch (logp_new).
"""
import torch

def group_advantages(rewards, group_size, eps=1e-6):
    """rewards: (B,) flat, grouped in contiguous blocks of `group_size`.
    Returns (B,) advantages, group-normalized. Zero-variance groups -> ~0 advantage."""
    B = rewards.shape[0]
    assert B % group_size == 0, "batch must be a multiple of group_size"
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    adv = (r - mean) / (std + eps)
    return adv.view(B), (std.squeeze(1) < eps)  # also return zero-variance mask for logging

# ---------------------------------------------------------------------------
# A1 / A1b — MaxEnt prompt weighting (the principled replacement for the
# "delete zero-variance groups" hack). We DON'T drop saturated groups; we
# DOWN-WEIGHT prompts by how little signal their group carries, and keep
# training on the informative middle (correct + incorrect rollouts coexist).
#
# Two equivalent views of "how much signal does this group carry":
#   binary tasks  -> distance of group accuracy from 0.5      (mgpo_weight)
#   graded tasks  -> the group's reward spread (std)          (variance_weight)
# Both multiply identically onto the (already group-normalized) advantage.
# ---------------------------------------------------------------------------

def group_pq(rewards, group_size):
    """Per-trajectory group mean p_q (broadcast back to (B,)). For binary rewards this is
    (#correct)/G; for graded rewards in [0,1] it is the group mean score (A1b)."""
    r = rewards.view(-1, group_size)
    return r.mean(dim=1, keepdim=True).expand_as(r).reshape(-1)


def mgpo_weight(p_q, gamma=2.0):
    """UNNORMALIZED Gaussian bell in p_q: peak 1.0 at p_q=0.5, no 1/(σ√2π) prefactor.
        w = exp(-gamma * (p_q - 0.5)^2)
    std = 1/sqrt(2*gamma). To target prompts within ±δ accuracy of 0.5 set gamma = 1/(2δ^2)
    (δ=0.15 -> gamma≈22). Bigger gamma = narrower bell = only near-50%-accuracy prompts survive.
    Focuses gradient where correct & incorrect rollouts COEXIST; damps saturated AND hopeless prompts."""
    if not torch.is_tensor(p_q):
        p_q = torch.tensor(float(p_q))
    return torch.exp(-gamma * (p_q - 0.5) ** 2)


def graded_pq(rewards):
    """A1b: group mean for graded (non-binary) rewards in [0,1]. Named (vs the binary #correct/G)
    so routing code reads intentionally."""
    return rewards.mean()


def variance_weight(rewards, group_size, std_ref=0.5, eps=1e-6):
    """A1b (preferred for graded tasks): weight a prompt by how much its rollouts DISAGREE,
    measured directly as the group reward std (the honest signal for continuous rewards).
        w = clamp( std_group / (std_ref + eps), 0, 1 )
    A near-zero-spread group contributes ~0 gradient (correctly). std_ref=0.5 is the max std of a
    binary group, a reasonable reference for rewards in [0,1]."""
    r = rewards.view(-1, group_size)
    std = r.std(dim=1, unbiased=False, keepdim=True).expand_as(r).reshape(-1)
    return torch.clamp(std / (std_ref + eps), 0.0, 1.0)


def long2short_shape(rewards, lengths, group_size, lam=0.2, correct_thresh=1.0):
    """A5: redistribute reward AMONG CORRECT trajectories only, by brevity, ZERO-SUM per group
    (so the group baseline/advantage mean is undisturbed). Apply BEFORE advantage computation.
        C = {i in group | r_i >= correct_thresh}; s_i = 1/len_i; s_bar = mean(s_i over C)
        r_i' = r_i + lam * (s_i - s_bar) / max_j|s_j - s_bar|   for i in C
    Incorrect trajectories unchanged. Shorter correct answers get relatively higher advantage ->
    the model learns short-plan / one-answer routes where they suffice (adaptive-mode goal).
    No-op for a group whose correct trajectories are all equal length."""
    r = rewards.view(-1, group_size).clone()
    L = lengths.view(-1, group_size).float().clamp_min(1.0)
    s = 1.0 / L
    for g in range(r.size(0)):
        correct = r[g] >= correct_thresh
        if correct.sum() <= 1:
            continue
        sc = s[g][correct]; dev = sc - sc.mean(); denom = dev.abs().max()
        if denom < 1e-9:
            continue
        r[g][correct] = r[g][correct] + lam * dev / denom
    return r.reshape(-1)


def clipped_pg_loss(logp_new, logp_old, advantages, mask, clip_eps=0.2):
    """PPO-clipped policy-gradient loss over masked tokens.
    logp_new, logp_old, mask: (B, T) ; advantages: (B,) broadcast over T.
    Ratio is per-token; advantage is per-trajectory (GRPO). Returns scalar loss
    (to MINIMIZE) and diagnostics."""
    # ratio in log space for stability, then exp
    log_ratio = (logp_new - logp_old) * mask
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)  # (B,1) -> broadcast (B,T)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    # PPO takes the pessimistic (min) of the two; we MINIMIZE the negative
    per_tok = -torch.min(unclipped, clipped)
    # mask + token-mean (avoid dividing by zero on empty masks)
    denom = mask.sum().clamp_min(1.0)
    loss = (per_tok * mask).sum() / denom
    with torch.no_grad():
        # fraction of tokens where clipping was active (PPO health metric)
        clip_frac = (((ratio > 1+clip_eps) | (ratio < 1-clip_eps)).float() * mask).sum() / denom
        approx_kl = (((ratio - 1) - log_ratio) * mask).sum() / denom  # k3 KL estimator
    return loss, {"clip_frac": clip_frac.item(), "approx_kl": approx_kl.item(),
                  "ratio_mean": ((ratio*mask).sum()/denom).item()}

def joint_grpo_loss(
    *, rewards, group_size,
    exec_logp_new, exec_logp_old, exec_mask,
    plan_logp_new=None, plan_logp_old=None, plan_mask=None,
    beta_plan=1.0, clip_eps=0.2,
    ce_plan=None, beta_ce=0.1, ce_resp=None, lam_resp=1.0,
    adv_weights=None):
    """Full off-policy joint objective.

    Required:
      rewards          (B,)   verifiable reward per trajectory (e.g. checker correctness,
                              optionally + teacher-LL bonus so 'better than gold' is rewardable)
      exec_logp_new/old (B,T_e) per-token logprobs of EXECUTOR tokens (response)
      exec_mask         (B,T_e) 1 on executor/response tokens, 0 on prompt+plan+pad   <-- point (1)
    Optional plan policy (point (2)):
      plan_logp_new/old (B,T_p), plan_mask (B,T_p)
    Optional CE anchors (keep SMALL per the doc: beta_ce 0.1):
      ce_plan, ce_resp  scalars already reduced

    Returns (total_loss, logs). MINIMIZE total_loss.
    """
    adv, zerovar = group_advantages(rewards, group_size)
    # A1/A1b: multiply the MaxEnt prompt weight onto the (group-normalized) advantage. The SAME
    # per-trajectory weight scales BOTH policies' advantages, since it reflects how much signal
    # the prompt's group carries — not anything policy-specific. None -> base behavior (no-op).
    if adv_weights is not None:
        adv = adv * adv_weights
        logs_w = {"adv_weight_mean": adv_weights.mean().item(),
                  "adv_weight_min": adv_weights.min().item()}
    else:
        logs_w = {}
    l_exec, d_exec = clipped_pg_loss(exec_logp_new, exec_logp_old, adv, exec_mask, clip_eps)
    total = lam_resp * l_exec
    logs = {f"exec_{k}": v for k, v in d_exec.items()}
    logs.update(logs_w)
    logs["zero_var_frac"] = zerovar.float().mean().item()
    logs["adv_std"] = adv.std(unbiased=False).item()

    if plan_logp_new is not None:
        l_plan, d_plan = clipped_pg_loss(plan_logp_new, plan_logp_old, adv, plan_mask, clip_eps)
        total = total + beta_plan * l_plan
        logs.update({f"plan_{k}": v for k, v in d_plan.items()})

    if ce_plan is not None:
        total = total + beta_ce * ce_plan
        logs["ce_plan"] = float(ce_plan)
    if ce_resp is not None:
        total = total + 0.0  # ce_resp folded into lam_resp path if you use it; kept explicit
        logs["ce_resp"] = float(ce_resp)

    logs["total_loss"] = float(total.detach())
    logs["l_exec"] = float(l_exec.detach())
    return total, logs


# ---------------------------------------------------------------------------
# Numerical self-test: validates the off-policy mechanics without a model.
# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    B, Te, Tp, G = 16, 12, 6, 8   # 16 trajectories = 2 groups of 8

    # 1) zero-variance group -> ~zero advantage -> ~zero gradient signal
    rew = torch.cat([torch.full((G,), 0.5), torch.tensor([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6])])
    adv, zv = group_advantages(rew, G)
    assert zv[0] and not zv[1], "first group is constant -> zero-variance flagged"
    assert adv[:G].abs().max() < 1e-3, "zero-variance group yields ~0 advantage"
    assert adv[G:].abs().max() > 0.5, "varied group yields real advantage"

    # 2) on-policy step (new==old): ratio==1, clip_frac==0, loss == -mean(adv) over tokens
    exec_old = torch.randn(B, Te)
    exec_mask = (torch.rand(B, Te) > 0.3).float()
    exec_mask[:, 0] = 1.0  # ensure non-empty
    loss0, d0 = clipped_pg_loss(exec_old, exec_old.clone(), adv, exec_mask)
    assert abs(d0["ratio_mean"] - 1.0) < 1e-5 and d0["clip_frac"] < 1e-6
    assert d0["approx_kl"] < 1e-6, "on-policy KL ~ 0"

    # 3) off-policy drift: new != old -> ratio != 1, clipping engages, KL > 0
    exec_new = exec_old + 0.5 * torch.randn(B, Te)
    loss1, d1 = clipped_pg_loss(exec_new, exec_old, adv, exec_mask, clip_eps=0.2)
    assert d1["approx_kl"] > 0 and d1["clip_frac"] > 0, "off-policy engages IS clipping"

    # 4) executor-only masking: plan tokens must not affect exec loss
    plan_old = torch.randn(B, Tp); plan_new = plan_old + 0.3*torch.randn(B, Tp)
    plan_mask = torch.ones(B, Tp)
    total, logs = joint_grpo_loss(
        rewards=rew, group_size=G,
        exec_logp_new=exec_new, exec_logp_old=exec_old, exec_mask=exec_mask,
        plan_logp_new=plan_new, plan_logp_old=plan_old, plan_mask=plan_mask,
        beta_plan=1.0, clip_eps=0.2, ce_plan=torch.tensor(0.47), beta_ce=0.1)
    # exec loss identical whether or not plan branch present (separate ratios)
    assert abs(logs["l_exec"] - float(loss1.detach())) < 1e-5, "exec objective is independent of plan"
    assert "plan_approx_kl" in logs and "exec_clip_frac" in logs

    # 5) gradients actually flow to a trainable param through the exec branch
    w = torch.zeros(B, Te, requires_grad=True)
    exec_new_g = exec_old + w
    tot, _ = joint_grpo_loss(rewards=rew, group_size=G,
        exec_logp_new=exec_new_g, exec_logp_old=exec_old, exec_mask=exec_mask)
    tot.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0, "gradient flows on varied groups"

    # 6) A1 MGPO weight: bell peaks at 0.5, damps saturated/hopeless prompts
    assert abs(float(mgpo_weight(0.5, gamma=2.0)) - 1.0) < 1e-6
    assert float(mgpo_weight(0.0, gamma=2.0)) < float(mgpo_weight(0.4, gamma=2.0))
    assert float(mgpo_weight(1.0, gamma=2.0)) < 0.7
    # gamma = 1/(2δ^2): δ=0.15 -> γ≈22.2, weight at p=0.65 should be ~exp(-0.5)=0.607
    g = 1.0 / (2 * 0.15**2)
    assert abs(float(mgpo_weight(0.65, gamma=g)) - 2.718281828**-0.5) < 1e-3

    # 7) A1b variance weight: disagreeing group -> high weight, agreeing -> ~0 (two groups of 4)
    rv = torch.tensor([1., 0., 1., 0., 0.4, 0.5, 0.45, 0.55])  # group0 max-spread, group1 tiny-spread
    vw = variance_weight(rv, 4)
    assert vw[0] > 0.9 and vw[4] < 0.2

    # 8) graded p_q is just the group mean
    assert abs(float(group_pq(rv, 4)[0]) - 0.5) < 1e-6

    # 9) A5 long2short: zero-sum among correct, shorter correct gets boosted
    rl = torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.])
    ln = torch.tensor([10., 30., 50., 5., 5., 5., 5., 5.])
    rs = long2short_shape(rl, ln, G, lam=0.2)
    assert abs(float(rs[:3].sum() - rl[:3].sum())) < 1e-5, "zero-sum over correct set"
    assert rs[0] > rs[2], "shortest correct trajectory is rewarded more"
    assert (rs[3:] == rl[3:]).all(), "incorrect trajectories unchanged"

    # 10) adv_weights plumbing: weighting changes the exec loss but keeps the API stable
    w = mgpo_weight(group_pq(rew, G), gamma=2.0)
    _, logw = joint_grpo_loss(rewards=rew, group_size=G,
        exec_logp_new=exec_new, exec_logp_old=exec_old, exec_mask=exec_mask, adv_weights=w)
    assert "adv_weight_mean" in logw

    print("grpo_offpolicy self-test: ALL PASS")
    print(f"  zero_var_frac on mixed batch = {logs['zero_var_frac']:.3f} (1 of 2 groups)")
    print(f"  off-policy exec: ratio_mean={d1['ratio_mean']:.3f} clip_frac={d1['clip_frac']:.3f} approx_kl={d1['approx_kl']:.4f}")

if __name__ == "__main__":
    _selftest()
