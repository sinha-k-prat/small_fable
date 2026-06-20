#!/usr/bin/env python3
"""Interleaved (agentic, closed-loop) architecture tests on a TINY Qwen2 (CPU, no network).

Mirrors the tiny-model harness in test_pipeline_smoke.py but exercises the interleaved path:
  - build-alignment: PLAN/RESP masks are disjoint, cover every non-ignore position, and own the
    plan->resp / resp->plan boundaries correctly.
  - interleaved_loss: shapes + gradient reaches LoRA AND planner head AND plan embeddings.
  - run_interleaved: terminates within caps, alternates plan/resp, force_plan ablations run.
  - logp roundtrip: |recompute - record| < 1e-4 over the concatenated trajectory.
  - GRPO step through the UNCHANGED joint_grpo_loss (exec_ratio~1 at pass 0; exec_mask excludes plan).
  - ablation branches (empty plan / shuffle) run.
  - flat path unchanged when the interleaved vocab is loaded but flat methods are used.
  - dual-write adapter (traces_to_sft --interleaved) keeps flat plan/answer AND adds turns.

The interleaved plan vocab (with BOP/FINALIZE_ALL markers) is written to a temp dir and pointed to
via PLAN_VOCAB_FILE BEFORE importing model_joint, so the module-level marker ids resolve.
"""
import os, sys, json, tempfile, importlib
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V = 261  # tiny char-level vocab; 0=pad, 1=eos, 2.. = printable chars


class FakeTok:
    """Char-level tokenizer with left padding, a no-op chat template, and __call__ supporting
    add_special_tokens (interleaved response tokenization passes add_special_tokens=False)."""
    pad_token = "<pad>"; eos_token = "<eos>"; pad_token_id = 0; eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "Q: " + msgs[-1]["content"] + " A:"

    def _enc(self, s):
        return [min(ord(c) + 2, V - 1) for c in s][:64] or [2]

    def __call__(self, texts, return_tensors=None, padding=True, truncation=False,
                 max_length=None, add_special_tokens=True):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        seqs = [self._enc(t) for t in texts]
        if max_length:
            seqs = [s[:max_length] for s in seqs]
        if return_tensors is None:
            # interleaved per-chunk tokenization: return plain python lists
            return {"input_ids": (seqs[0] if single else seqs)}
        L = max(len(s) for s in seqs)
        ids, attn = [], []
        for s in seqs:
            pad = L - len(s)
            ids.append([self.pad_token_id] * pad + s)
            attn.append([0] * pad + [1] * len(s))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        return "".join(chr(i - 2) for i in ids if i > 1)


def _write_interleaved_vocab(path):
    """A small interleaved plan_vocab.json: data primitives + END, then APPENDED BOP/FINALIZE_ALL."""
    vocab = ["PAD", "MODEL", "as=truth_table", "LINK", "guard=on", "VERIFY", "FINALIZE",
             "END", "BOP", "FINALIZE_ALL"]
    json.dump({"vocab": vocab, "terminator": "END", "interleaved": True,
               "markers": {"BOP": "BOP", "PLAN_EOS": "END", "FINALIZE_ALL": "FINALIZE_ALL"}},
              open(path, "w"))


# model_joint is reloaded with the interleaved vocab in setup_module() and RESTORED to its default
# vocab in teardown_module() so test files collected AFTER this one (which bind the default N_PLAN=41
# vocab at import time) are unaffected by the mutated module-level marker ids.
import model_joint as MJ  # noqa: E402
_TMPDIR = tempfile.mkdtemp()
_VOCAB_PATH = os.path.join(_TMPDIR, "plan_vocab.json")
_write_interleaved_vocab(_VOCAB_PATH)
_PREV_ENV = None


def setup_module(module):
    global _PREV_ENV
    _PREV_ENV = os.environ.get("PLAN_VOCAB_FILE")
    os.environ["PLAN_VOCAB_FILE"] = _VOCAB_PATH
    importlib.reload(MJ)   # re-resolve PLAN_VOCAB / BOP_ID / FINALL_ID from the interleaved vocab


def teardown_module(module):
    if _PREV_ENV is None:
        os.environ.pop("PLAN_VOCAB_FILE", None)
    else:
        os.environ["PLAN_VOCAB_FILE"] = _PREV_ENV
    importlib.reload(MJ)   # restore the default-vocab module state for subsequent test files


def _tiny_model(device="cpu"):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    base = Qwen2ForCausalLM(cfg)
    backbone = MJ.build_lora(base, is_trainable=True)
    return MJ.JointModel(backbone, FakeTok(), hidden=64, plan_max_len=6).to(device)


def _example_turns():
    return [{"plan": ["MODEL", "as=truth_table", "LINK", "guard=on"], "response": "from it rained"},
            {"plan": ["VERIFY", "FINALIZE"], "response": "so yes\nFINAL ANSWER: yes"}]


# ---------------------------------------------------------------------------------------------
def test_vocab_markers_resolved():
    assert MJ.INTERLEAVED_VOCAB is True
    assert MJ.BOP_ID is not None and MJ.FINALL_ID is not None
    assert MJ.EOP_ID == MJ.PLAN2ID["END"]
    assert MJ.BOP_ID != MJ.FINALL_ID != MJ.EOP_ID


def test_build_alignment_disjoint_cover_and_boundaries():
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    p_ids, p_attn = m.batch_prompts(["sum the evens"])
    turns = _example_turns()
    (h, lm_logits, plan_m, _, resp_m, _, attn, tgt, emb) = m.interleaved_tf(p_ids, p_attn, [turns])
    # disjoint
    assert not (plan_m & resp_m).any(), "PLAN and RESP masks must be disjoint"
    # cover: every position with a real target (tgt != -100 and attn==1) is owned by exactly one head
    real = (tgt != -100) & attn.bool()
    assert (real == (plan_m | resp_m)).all(), "every real-target position owned by exactly one head"
    # boundary ownership: the position predicting END(=EOP) is a PLAN target; the next plan token after
    # a RESP_EOS is a PLAN target; the first response token is a RESP target.
    raw = tgt[0]
    # END appears as a plan target somewhere
    assert (plan_m[0] & (raw == MJ.EOP_ID)).any(), "END handoff is a PLAN target"
    # FINALIZE_ALL is the terminal plan target (resp->plan boundary of the last turn)
    assert (plan_m[0] & (raw == MJ.FINALL_ID)).any(), "FINALIZE_ALL is a PLAN target"
    # at least one RESP target equals resp_eos (executor learns to yield)
    assert (resp_m[0] & (raw == m.resp_eos_id)).any(), "RESP_EOS is a RESP target"


def test_interleaved_loss_shapes_and_grads():
    torch.manual_seed(0)
    m = _tiny_model(); m.train()
    p_ids, p_attn = m.batch_prompts(["sum the evens", "who is tallest"])
    turns_batch = [_example_turns(), _example_turns()]
    loss, logs = m.interleaved_loss(p_ids, p_attn, turns_batch, lam_resp=1.0, lam_kl=0.1)
    assert loss.dim() == 0
    for k in ("ce_plan", "ce_resp", "kl"):
        assert k in logs
    loss.backward()
    g_lora = [p.grad for n, p in m.backbone.named_parameters()
              if "lora_" in n and p.grad is not None]
    assert any(g.abs().sum() > 0 for g in g_lora), "grad reaches LoRA (executor)"
    assert m.planner.proj.weight.grad is not None and m.planner.proj.weight.grad.abs().sum() > 0, \
        "grad reaches planner head"
    assert m.plan_emb.emb.weight.grad is not None and m.plan_emb.emb.weight.grad.abs().sum() > 0, \
        "grad reaches plan embeddings"


def test_run_interleaved_terminates_and_alternates():
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    p_ids, p_attn = m.batch_prompts(["sum the evens"])
    rec = m.run_interleaved(p_ids[0], p_attn[0], sample=True, temp=1.0,
                            max_turns=3, max_plan=4, max_resp=6)
    assert "turns" in rec and 1 <= len(rec["turns"]) <= 3, "terminates within max_turns"
    for tn in rec["turns"]:
        assert "plan" in tn and "resp" in tn
        # every turn's plan ends with a marker (EOP or FINALIZE_ALL)
        if tn["plan"]:
            assert tn["plan"][-1] in (MJ.EOP_ID, MJ.FINALL_ID, *[
                p for p in tn["plan"] if p == MJ.EOP_ID])
    # answer text decodes
    _ = m.interleaved_answer_text(rec)


def test_run_interleaved_ablation_branches():
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    p_ids, p_attn = m.batch_prompts(["sum the evens"])
    # empty-plan ablation
    rec_n = m.run_interleaved(p_ids[0], p_attn[0], sample=False, max_turns=2, max_plan=4, max_resp=4,
                              force_plan=lambda t: [])
    assert rec_n["turns"], "empty-plan ablation runs"
    for tn in rec_n["turns"]:
        # forced empty plan -> plan is just [EOP]
        assert tn["plan"] == [MJ.EOP_ID]
    # shuffle ablation with random in-vocab primitives
    pool = [i for i in range(MJ.N_PLAN)
            if i not in (MJ.PAD_ID, MJ.BOP_ID, MJ.FINALL_ID, MJ.EOP_ID)]
    import random
    rng = random.Random(0)
    rec_s = m.run_interleaved(p_ids[0], p_attn[0], sample=False, max_turns=2, max_plan=4, max_resp=4,
                              force_plan=lambda t: [rng.choice(pool), rng.choice(pool)])
    assert rec_s["turns"], "shuffle ablation runs"


def test_logp_determinism():
    """interleaved_logp_tf is deterministic + returns valid logprobs (<=0). NOTE: the real A6
    save-vs-recompute roundtrip (RL ratio==1 at step 0) is covered by
    test_grpo_rollout_compaction_roundtrip_a6 / test_grpo_step_unchanged_loss — this only checks the
    recompute path is stable and well-formed."""
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    p_ids, p_attn = m.batch_prompts(["sum the evens"])
    turns = _example_turns()
    with torch.no_grad():
        pl1, pm1, rl1, rm1 = m.interleaved_logp_tf(p_ids, p_attn, [turns], temp=1.0)
        pl2, pm2, rl2, rm2 = m.interleaved_logp_tf(p_ids, p_attn, [turns], temp=1.0)
    assert (pm1 == pm2).all() and (rm1 == rm2).all()
    assert ((pl1 - pl2).abs() * pm1).max() < 1e-4
    assert ((rl1 - rl2).abs() * rm1).max() < 1e-4
    assert (pl1 <= 1e-5).all() and (rl1 <= 1e-5).all(), "logprobs are <= 0"


def test_grpo_step_unchanged_loss():
    """GRPO step over concatenated interleaved streams via the UNCHANGED joint_grpo_loss.
    exec_ratio~1 at pass 0; exec_mask excludes plan (RESP positions only)."""
    from grpo_offpolicy import joint_grpo_loss
    torch.manual_seed(0)
    m = _tiny_model(); m.train()
    G = 4
    p_ids, p_attn = m.batch_prompts(["sum the evens"] * G)
    turns_batch = [_example_turns() for _ in range(G)]
    with torch.no_grad():
        plan_old, pmask, resp_old, rmask = m.interleaved_logp_tf(p_ids, p_attn, turns_batch, temp=1.0)
    plan_new, pmask2, resp_new, rmask2 = m.interleaved_logp_tf(p_ids, p_attn, turns_batch, temp=1.0)
    # exec_mask (resp) and plan_mask are disjoint position sets
    assert not (pmask.bool() & rmask.bool()).any(), "plan and exec masks are disjoint"
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss, logs = joint_grpo_loss(
        rewards=rewards, group_size=G,
        exec_logp_new=resp_new, exec_logp_old=resp_old, exec_mask=rmask2,
        plan_logp_new=plan_new, plan_logp_old=plan_old, plan_mask=pmask2,
        beta_plan=1.0, clip_eps=0.2)
    assert abs(logs["exec_ratio_mean"] - 1.0) < 1e-4, "pass-0 exec ratio ~ 1"
    assert logs["zero_var_frac"] == 0.0
    loss.backward()
    g = [p.grad for n, p in m.backbone.named_parameters() if "lora_" in n and p.grad is not None]
    assert any(x.abs().sum() > 0 for x in g), "grad flows to LoRA via interleaved GRPO"


def test_grpo_rollout_compaction_roundtrip_a6():
    """rollout-record (compacted logp_old) -> train_grpo compaction of recomputed logp -> A6 ~0 and a
    full joint_grpo_loss step. Exercises build_group_tensors_interleaved + _compact/_trim_width."""
    import train_grpo_offline as TG
    from grpo_offpolicy import joint_grpo_loss
    importlib.reload(TG)
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    turns = _example_turns()
    p_ids, p_attn = m.batch_prompts(["q"])
    pl, pm, rl, rm = m.interleaved_logp_tf(p_ids, p_attn, [turns], temp=1.0)
    pm0 = pm[0].bool(); rm0 = rm[0].bool()
    (_h, _lm, _pm, _pt, _rm, _rt, _a, raw, _e) = m.interleaved_tf(p_ids, p_attn, [turns])
    recs = []
    for g in range(2):
        recs.append({"id": "x", "instruction": "q", "group_size": 2, "reward_path": "verifiable",
                     "reward": float(g), "interleaved": True, "turns": turns,
                     "plan_tokens": raw[0][pm0].tolist(), "plan_logp_old": pl[0][pm0].tolist(),
                     "resp_tokens": raw[0][rm0].tolist(), "resp_logp_old": rl[0][rm0].tolist(),
                     "resp_len": int(rm0.sum()), "p_q": 0.5, "keep": True})
    mm = TG.logp_mismatch_t0_interleaved(m, recs)
    assert mm < 1e-4, f"A6 interleaved logp_mismatch_t0 too large: {mm}"
    p_ids, p_attn, tb, po, ro, rew, ln = TG.build_group_tensors_interleaved(m, recs)
    plf, pmf, rlf, rmf = m.interleaved_logp_tf(p_ids, p_attn, tb, temp=1.0)
    pn, pmask = TG._compact(plf, pmf); rn, rmask = TG._compact(rlf, rmf)
    po = TG._trim_width(po, pmask); ro = TG._trim_width(ro, rmask)
    loss, logs = joint_grpo_loss(rewards=rew, group_size=2, exec_logp_new=rn, exec_logp_old=ro,
                                 exec_mask=rmask, plan_logp_new=pn, plan_logp_old=po, plan_mask=pmask,
                                 beta_plan=1.0, clip_eps=0.2, lam_resp=1.0)
    assert abs(logs["exec_ratio_mean"] - 1.0) < 1e-4, "pass-0 exec ratio ~ 1 through the rollout path"
    loss.backward()


def test_freeze_executor_planner_only():
    """--freeze_executor semantics: freeze LoRA + lam_resp=0 -> only planner head + plan_emb train."""
    from grpo_offpolicy import joint_grpo_loss
    torch.manual_seed(0)
    m = _tiny_model(); m.train()
    for n, p in m.backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad = False
    n_planner = sum(p.numel() for p in m.planner.parameters() if p.requires_grad) \
        + sum(p.numel() for p in m.plan_emb.parameters() if p.requires_grad)
    assert n_planner > 0
    G = 4
    p_ids, p_attn = m.batch_prompts(["sum the evens"] * G)
    turns_batch = [_example_turns() for _ in range(G)]
    with torch.no_grad():
        plan_old, _, resp_old, _ = m.interleaved_logp_tf(p_ids, p_attn, turns_batch, temp=1.0)
    plan_new, pmask, resp_new, rmask = m.interleaved_logp_tf(p_ids, p_attn, turns_batch, temp=1.0)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss, _ = joint_grpo_loss(
        rewards=rewards, group_size=G,
        exec_logp_new=resp_new, exec_logp_old=resp_old, exec_mask=rmask,
        plan_logp_new=plan_new, plan_logp_old=plan_old, plan_mask=pmask,
        beta_plan=1.0, clip_eps=0.2, lam_resp=0.0)
    loss.backward()
    assert m.planner.proj.weight.grad is not None and m.planner.proj.weight.grad.abs().sum() > 0, \
        "planner head still trains"
    # frozen LoRA receives no grad
    for n, p in m.backbone.named_parameters():
        if "lora_" in n:
            assert p.grad is None or p.grad.abs().sum() == 0, "frozen executor gets no gradient"


def test_flat_methods_still_work_on_interleaved_vocab():
    """Loading the interleaved vocab must NOT break the flat methods (backward compat)."""
    torch.manual_seed(0)
    m = _tiny_model(); m.eval()
    p_ids, p_attn = m.batch_prompts(["sum the evens"])
    plan = m.sample_plan(p_ids, p_attn, temp=1.0, sample=True)
    assert plan.shape == (1, 6)
    logp, mask = m.plan_logp_tf(p_ids, p_attn, plan, temp=1.0)
    assert logp.shape == (1, 6)
    gen = m.generate_answer(p_ids, p_attn, plan, sample=False, max_new_tokens=4)
    assert gen.shape[0] == 1


def test_dual_write_adapter():
    """traces_to_sft --interleaved: keep flat plan/answer AND add turns:[{plan,response}], last
    turn's response ends with FINAL ANSWER: X. Vocab gets BOP/FINALIZE_ALL appended."""
    import traces_to_sft as TTS
    trace = ("TURN 1 [ MODEL[as=truth_table] ; LINK[guard=on] ]\n"
             "response: from it rained the ground is wet\n"
             "TURN 2 [ VERIFY[aspect=logic] ; FINALIZE[form=yes_no] ]\n"
             "response: so the answer holds\n")
    row = {"instruction": "did it rain?", "trace": trace, "family": "logic",
           "answer_form": "yes_no"}
    ak = {"match": {"type": "exact_choice", "accept": ["yes"]}, "canonical": "yes"}
    rec = TTS.convert_interleaved(row, ak, 0)
    # dual-write: flat fields preserved
    assert "plan" in rec and "answer" in rec and rec["plan"][-1] == "END"
    assert "FINAL ANSWER:" in rec["answer"]
    # turns added
    assert "turns" in rec and len(rec["turns"]) == 2
    assert rec["turns"][0]["plan"] == ["MODEL", "as=truth_table", "LINK", "guard=on"]
    assert "FINAL ANSWER:" in rec["turns"][-1]["response"], "terminal turn carries the commitment"
    # parse_turns factoring
    turns = TTS.parse_turns(trace)
    assert turns[1]["plan"] == ["VERIFY", "aspect=logic", "FINALIZE", "form=yes_no"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("interleaved tests: ALL PASS")
