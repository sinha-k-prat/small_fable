#!/usr/bin/env python3
"""Resume/checkpoint unit tests — no network, CPU, tiny model.

Validates the pieces train_sft.py / train_grpo_offline.py rely on for resuming after a Colab crash:
  - train_state round-trips (save/load) including position fields
  - restore_optimizer reproduces IDENTICAL continuation (the whole point of saving optimizer state)
  - the (epoch, batch_idx) / (inner_epoch, group_idx) skip arithmetic covers every item exactly once
  - due() periodic-trigger logic
"""
import os, sys, tempfile, copy
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checkpointing import (save_train_state, load_train_state, restore_optimizer,
                           scalar_args, Checkpointer)
from model_joint import encode_plan, PAD_ID, N_PLAN
from test_pipeline_smoke import tiny_model


# --------------------------------------------------------------- helpers
def _fixed_batch(m):
    p_ids, p_attn = m.batch_prompts(["sum the evens", "who is tallest"])
    plan_ids = torch.stack([encode_plan(["FILTER[even]", "EVAL", "TERMINATE"], 6),
                            encode_plan(["ORDER", "COMPARE", "TERMINATE"], 6)])
    ans = m.tok(["12", "Bo"], return_tensors="pt")
    return p_ids, p_attn, plan_ids, ans["input_ids"], ans["attention_mask"].float()


def _step(m, opt, batch):
    p_ids, p_attn, plan_ids, r_ids, r_attn = batch
    pl = m.planner_logits_tf(p_ids, p_attn, plan_ids)
    pmask = (plan_ids != PAD_ID)
    l_plan = (F.cross_entropy(pl.reshape(-1, N_PLAN), plan_ids.reshape(-1), reduction="none")
              .reshape(plan_ids.shape) * pmask).sum() / pmask.sum()
    rl = m.executor_logits_tf(p_ids, p_attn, plan_ids, r_ids, r_attn)
    l_resp = (F.cross_entropy(rl.reshape(-1, rl.size(-1)), r_ids.reshape(-1), reduction="none")
              .reshape(r_ids.shape) * r_attn).sum() / r_attn.sum()
    opt.zero_grad(); (l_plan + l_resp).backward(); opt.step()


def _trainable(m):
    return [p for p in m.parameters() if p.requires_grad]


def _snapshot(m):
    return [p.detach().clone() for p in _trainable(m)]


# --------------------------------------------------------------- tests
def test_train_state_roundtrip():
    d = tempfile.mkdtemp()
    state = {"kind": "grpo", "inner_epoch": 1, "group_idx": 7, "global_step": 42,
             "optimizer": {"foo": 1}, "torch_rng": torch.get_rng_state()}
    save_train_state(d, state)
    assert os.path.exists(os.path.join(d, "train_state.pt"))
    st = load_train_state(d)
    assert st["inner_epoch"] == 1 and st["group_idx"] == 7 and st["global_step"] == 42
    assert load_train_state(tempfile.mkdtemp()) is None   # no file -> None


def test_restore_optimizer_identical_continuation():
    """Save optimizer mid-run, reload into a FRESH optimizer, and confirm the next steps match a
    run that was never interrupted. This is the property resume depends on."""
    # uninterrupted reference: 4 steps
    torch.manual_seed(0); mA = tiny_model(); mA.train()
    optA = torch.optim.AdamW(_trainable(mA), lr=1e-2)
    batch = _fixed_batch(mA)
    for _ in range(4):
        _step(mA, optA, batch)
    ref = _snapshot(mA)

    # interrupted: 2 steps, save optimizer, reload into a new optimizer object, 2 more steps
    torch.manual_seed(0); mB = tiny_model(); mB.train()
    optB = torch.optim.AdamW(_trainable(mB), lr=1e-2)
    batchB = _fixed_batch(mB)
    for _ in range(2):
        _step(mB, optB, batchB)
    sd = copy.deepcopy(optB.state_dict())
    optB2 = torch.optim.AdamW(_trainable(mB), lr=1e-2)
    restore_optimizer(optB2, {"optimizer": sd}, device="cpu")
    for _ in range(2):
        _step(mB, optB2, batchB)
    got = _snapshot(mB)

    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-6), "resumed optimizer must continue identically"


def _processed(n_outer, n_inner, start_outer=0, start_inner=0):
    """Mirror the resume skip loop used in both trainers."""
    out = []
    for o in range(start_outer, n_outer):
        si = start_inner if o == start_outer else 0
        for i in range(n_inner):
            if i < si:
                continue
            out.append((o, i))
    return out


def test_resume_skip_covers_every_item_once():
    E, B = 3, 4
    full = _processed(E, B)
    assert len(full) == E * B and len(set(full)) == E * B
    # crash right after finishing item (1,2) -> saved next position is (1,3)
    done = _processed(E, B)[:full.index((1, 2)) + 1]
    resumed = _processed(E, B, start_outer=1, start_inner=3)
    assert done + resumed == full, "resume must process each item exactly once, in order"
    # crash at an epoch boundary -> next position (2,0)
    done2 = _processed(E, B)[:full.index((1, 3)) + 1]
    resumed2 = _processed(E, B, start_outer=2, start_inner=0)
    assert done2 + resumed2 == full
    # completed run -> position past the end -> resume is a no-op
    assert _processed(E, B, start_outer=E, start_inner=0) == []


def test_checkpointer_due_logic():
    ck = Checkpointer(tempfile.mkdtemp(), base="x", every_min=0.0, every_steps=5)
    assert not ck.due(0) and not ck.due(4)
    assert ck.due(5) and ck.due(10)
    ck2 = Checkpointer(tempfile.mkdtemp(), base="x", every_min=0.0, every_steps=0)
    assert not ck2.due(100)          # both triggers off -> never periodic (only boundary saves)


def test_scalar_args_filters_nonscalars():
    class A: pass
    a = A(); a.lr = 1e-4; a.out = "ckpt"; a.flag = True; a._held = [1, 2, 3]; a.obj = object()
    s = scalar_args(a)
    assert s == {"lr": 1e-4, "out": "ckpt", "flag": True}   # underscore + non-scalar dropped
