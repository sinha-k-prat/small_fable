#!/usr/bin/env python3
"""Unit tests for the off-policy GRPO objective (grpo_offpolicy.py).

Covers the four properties named in the spec's acceptance check:
  - on-policy step: ratio ~= 1, clipping inactive, KL ~= 0
  - off-policy drift: ratio != 1, clipping engages, KL > 0
  - zero-variance group -> ~0 advantage (no signal)
  - executor objective is independent of the plan branch (separate action spaces)

Runnable with `pytest tests/` or directly `python tests/test_grpo_offpolicy.py`.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grpo_offpolicy import group_advantages, clipped_pg_loss, joint_grpo_loss

G = 8


def _adv():
    rew = torch.cat([torch.full((G,), 0.5),
                     torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])])
    return rew, *group_advantages(rew, G)


def test_zero_variance_group_zero_advantage():
    rew, adv, zv = _adv()
    assert zv[0] and not zv[1]
    assert adv[:G].abs().max() < 1e-3, "constant group -> ~0 advantage"
    assert adv[G:].abs().max() > 0.5, "varied group -> real advantage"


def test_on_policy_ratio_one_no_clip():
    torch.manual_seed(0)
    rew, adv, _ = _adv()
    old = torch.randn(2 * G, 12)
    mask = (torch.rand(2 * G, 12) > 0.3).float(); mask[:, 0] = 1.0
    _, d = clipped_pg_loss(old, old.clone(), adv, mask)
    assert abs(d["ratio_mean"] - 1.0) < 1e-5
    assert d["clip_frac"] < 1e-6
    assert d["approx_kl"] < 1e-6


def test_off_policy_engages_clipping():
    torch.manual_seed(1)
    rew, adv, _ = _adv()
    old = torch.randn(2 * G, 12)
    mask = torch.ones(2 * G, 12)
    new = old + 0.6 * torch.randn(2 * G, 12)
    _, d = clipped_pg_loss(new, old, adv, mask, clip_eps=0.2)
    assert d["approx_kl"] > 0
    assert d["clip_frac"] > 0


def test_exec_independent_of_plan_branch():
    torch.manual_seed(2)
    rew, _, _ = _adv()
    Te, Tp = 12, 6
    exec_old = torch.randn(2 * G, Te); exec_new = exec_old + 0.5 * torch.randn(2 * G, Te)
    exec_mask = (torch.rand(2 * G, Te) > 0.3).float(); exec_mask[:, 0] = 1.0
    l_exec_alone, _ = clipped_pg_loss(exec_new, exec_old,
                                      group_advantages(rew, G)[0], exec_mask, 0.2)
    plan_old = torch.randn(2 * G, Tp); plan_new = plan_old + 0.3 * torch.randn(2 * G, Tp)
    plan_mask = torch.ones(2 * G, Tp)
    _, logs = joint_grpo_loss(
        rewards=rew, group_size=G,
        exec_logp_new=exec_new, exec_logp_old=exec_old, exec_mask=exec_mask,
        plan_logp_new=plan_new, plan_logp_old=plan_old, plan_mask=plan_mask,
        beta_plan=1.0, clip_eps=0.2, ce_plan=torch.tensor(0.47), beta_ce=0.1)
    assert abs(logs["l_exec"] - float(l_exec_alone.detach())) < 1e-5
    assert "plan_approx_kl" in logs and "exec_clip_frac" in logs


def test_gradient_flows():
    torch.manual_seed(3)
    rew, _, _ = _adv()
    Te = 12
    exec_old = torch.randn(2 * G, Te)
    exec_mask = (torch.rand(2 * G, Te) > 0.3).float(); exec_mask[:, 0] = 1.0
    w = torch.zeros(2 * G, Te, requires_grad=True)
    tot, _ = joint_grpo_loss(rewards=rew, group_size=G,
                             exec_logp_new=exec_old + w, exec_logp_old=exec_old, exec_mask=exec_mask)
    tot.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("grpo_offpolicy unit tests: ALL PASS")
