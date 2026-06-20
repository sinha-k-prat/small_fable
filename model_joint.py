#!/usr/bin/env python3
"""
model_joint.py — the two-head adaptive agent (planner head + shared executor backbone).

ARCHITECTURE
------------
  backbone   : Qwen2.5-1.5B-Instruct (the "executor"), wrapped with LoRA on ALL 7 matrices.
  planner    : a small 2-layer MLP head (hidden -> hidden//4 -> n_plan, SiLU) over the backbone's
               last-layer hidden states that emits PLAN PRIMITIVE logits over a SEPARATE small
               vocabulary (PLAN_VOCAB), NOT the executor token vocab. The planner is autoregressive:
               it is run step by step, re-feeding each chosen primitive as a learned plan embedding.
  plan_emb   : embeddings for plan primitives. The chosen plan is embedded and prepended as a
               SOFT PREFIX (vectors in hidden space) so the executor is conditioned on the plan
               before it writes the answer.  -> "plan in planning mode, then answer".

TWO POLICIES, KEPT SEPARATE EVERYWHERE
  planner policy  -> action space = PLAN_VOCAB (factored primitives + key=value atoms + END)
  executor policy -> action space = token vocab
RL uses two INDEPENDENT clipped objectives over these (see grpo_offpolicy.joint_grpo_loss).

PADDING CONVENTION (important for the tensor math)
  Prompts are LEFT-padded. With left padding the last real prompt token is always at index -1,
  and the plan-prefix / response embeddings appended after it are contiguous. RoPE is relative,
  so a constant left shift of an example's positions is harmless. This lets every "predict the
  next k things" slice be a simple `[:, -k:, :]` with no per-example index gather.

DTYPE
  Pass dtype=torch.bfloat16 on CUDA (A100/H100) for 2x speed with full stability.
  Default is torch.float32 on CPU, torch.bfloat16 on CUDA. Override via --dtype or dtype=.
  The planner head and plan_emb are cast to the same dtype as the backbone so no mixed-precision
  matmul errors occur.

SELF-CONTAINED CHECKPOINTS
  model.save(out_dir, base) always writes plan_vocab.json into out_dir alongside heads.pt.
  from_checkpoint verifies that the loaded vocabulary matches the planner head's output dimension.
  This makes every checkpoint independently reloadable without relying on an ambient plan_vocab.json.
"""
import os, json
import torch, torch.nn as nn
import torch.nn.functional as F

LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # all 7

_VOCAB_FILE_NAME = "plan_vocab.json"

# Plan primitive vocabulary. Parameterized ops (FILTER[even], TOP_K[k=3]) collapse to their
# base primitive via split("[") so the planner head stays a fixed, separate action space.
_DEFAULT_PLAN_VOCAB = ["PAD","EXTRACT","DECOMPOSE","MODEL","IDENTIFY_UNKNOWN","ORDER","FIND",
    "GENERATE","GENERATE_ALT","EXPLORE","DIVERGE","LINK","SIMULATE","TRACE","CALCULATE",
    "PREDICT","COMPARE","WEIGH","VERIFY_LOGIC","VERIFY_CONSTRAINTS","VERIFY_COMPLETENESS",
    "VERIFY_CONSISTENCY","VERIFY_STEP","VERIFY_EVIDENCE","REFLECT","EVAL","REFINE","CORRECT",
    "REPAIR","EXPAND","SIMPLIFY","MERGE","COMBINE","GENERALIZE","RESOLVE_CONFLICT","PLAN",
    "PLAN_NEXT","SELECT","CLARIFY","ADAPT","TERMINATE"]
_DEFAULT_TERM_NAME = "TERMINATE"

# A run can override the planner's action space + terminator via plan_vocab.json (written by
# traces_to_sft.py). MUST be the same file at train and inference time (the planner head is sized to
# it). Absent -> the default vocab above, so existing tests/synthetic data are unaffected.
_VOCAB_FILE = os.environ.get("PLAN_VOCAB_FILE",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), _VOCAB_FILE_NAME))
if os.path.exists(_VOCAB_FILE):
    _vc = json.load(open(_VOCAB_FILE))
    PLAN_VOCAB = _vc["vocab"]; _TERM_NAME = _vc.get("terminator", _DEFAULT_TERM_NAME)
else:
    PLAN_VOCAB = _DEFAULT_PLAN_VOCAB; _TERM_NAME = _DEFAULT_TERM_NAME
PLAN2ID = {p:i for i,p in enumerate(PLAN_VOCAB)}
ID2PLAN = {i:p for p,i in PLAN2ID.items()}
PAD_ID  = PLAN2ID["PAD"]
# Terminator matched by BASE name so every parameterized variant (FINALIZE[form=yes_no],
# FINALIZE[form=number_with_units], …) ends a plan. TERM_ID is a single representative for legacy.
_TERM_BASE = _TERM_NAME.split("[")[0]
TERM_IDS = {i for p, i in PLAN2ID.items() if p.split("[")[0] == _TERM_BASE}
TERM_ID  = min(TERM_IDS) if TERM_IDS else len(PLAN_VOCAB) - 1
N_PLAN   = len(PLAN_VOCAB)

# Cached tensor for vectorised terminator membership test (filled on first GPU call).
_TERM_IDS_TENSOR: torch.Tensor | None = None


def _is_terminator(nxt: torch.Tensor) -> torch.Tensor:
    """(B,) bool — True where nxt ∈ TERM_IDS.  Vectorised; tensor is cached per device."""
    global _TERM_IDS_TENSOR
    if _TERM_IDS_TENSOR is None or _TERM_IDS_TENSOR.device != nxt.device:
        _TERM_IDS_TENSOR = torch.tensor(sorted(TERM_IDS), dtype=torch.long, device=nxt.device)
    return nxt.unsqueeze(1).eq(_TERM_IDS_TENSOR).any(1)


def save_vocab(out_dir: str) -> None:
    """Write the current module-level plan vocabulary into out_dir/plan_vocab.json.
    Called by JointModel.save() so every checkpoint is self-contained."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _VOCAB_FILE_NAME)
    with open(path, "w") as f:
        json.dump({"vocab": PLAN_VOCAB, "terminator": _TERM_NAME}, f, indent=2)


def _resolve_dtype(dtype, device: str) -> torch.dtype:
    """Return dtype, defaulting to bfloat16 on CUDA (A100/H100 native), float32 on CPU."""
    if dtype is not None:
        return dtype
    return torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32


def build_lora(base_model, r=16, alpha=32, dropout=0.05, is_trainable=True):
    """Wrap a base causal LM with LoRA on all 7 projection matrices.

    GUARD: PeftModel.from_pretrained loads adapters FROZEN. If is_trainable is not forced on,
    0 LoRA tensors require grad and RL becomes a no-op (SFT and SFT+RL produce byte-identical
    outputs). We force requires_grad and assert the count is > 0."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(base_model, cfg)
    _force_lora_trainable(model, is_trainable)
    return model


def _force_lora_trainable(model, is_trainable):
    if is_trainable:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    n_train = sum(1 for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    if is_trainable:
        assert n_train > 0, ("FROZEN-BACKBONE BUG: 0 trainable LoRA tensors. "
                             "Pass is_trainable=True to from_pretrained / build_lora.")
    print(f"[model_joint] trainable LoRA tensors: {n_train} "
          f"({'trainable' if is_trainable else 'inference'}; expect ~14 per layer)")
    return n_train


class PlannerHead(nn.Module):
    """2-layer MLP: hidden -> hidden//4 (SiLU) -> n_plan.

    The hidden//4 bottleneck gives the planner dedicated non-linear capacity to compose plan
    sequences without relying solely on the shared backbone's hidden states. The forward
    accepts (B,T,H) and returns (B,T,N_PLAN) for teacher-forced scoring over full plan seqs;
    use `[..., -1, :]` when you need only the next-primitive distribution at the last step.

    Backwards-compat loading: old checkpoints contain a single nn.Linear (keys proj.weight /
    proj.bias). from_checkpoint transplants those weights into proj[2] automatically."""
    def __init__(self, hidden, n_plan=N_PLAN):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, n_plan),
        )

    def forward(self, hidden_states):            # (B,T,H) -> (B,T,N_PLAN)
        return self.proj(hidden_states)


class PlanEmbedding(nn.Module):
    """Embeds plan primitive ids into soft-prefix vectors (hidden space) for the executor.
    PAD maps to the zero vector (padding_idx) so an empty plan == no conditioning."""
    def __init__(self, n_plan, hidden):
        super().__init__()
        self.emb = nn.Embedding(n_plan, hidden, padding_idx=PAD_ID)
    def forward(self, plan_ids):                 # (B,L) -> (B,L,H)
        return self.emb(plan_ids)


def encode_plan(plan_list, max_len=12):
    """Gold plan (list of primitive strings) -> padded id tensor (max_len,). Looks up the FULL
    (parameterized) token first so 'REFLECT[reason=naive_vs_correct]' keeps its strategy; falls back
    to the bare base name (so unparameterized/synthetic plans still resolve)."""
    ids = [PLAN2ID.get(p, PLAN2ID.get(p.split("[")[0], PAD_ID)) for p in plan_list][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode_plan(plan_ids):
    """id tensor/list -> list of primitive strings, stopping at the first PAD."""
    out = []
    for i in (plan_ids.tolist() if torch.is_tensor(plan_ids) else plan_ids):
        if i == PAD_ID:
            break
        out.append(ID2PLAN[int(i)])
    return out


class JointModel(nn.Module):
    """Holds the backbone (LoRA), planner head, plan embeddings, and all forward passes.

    Two construction paths:
      from_base(...)       fresh adapter + heads for SFT.
      from_checkpoint(...) load adapter + heads; pass is_trainable=True for RL.
    """
    def __init__(self, backbone, tokenizer, hidden, plan_max_len=12):
        super().__init__()
        self.backbone = backbone
        self.tok = tokenizer
        self.hidden = hidden
        self.plan_max_len = plan_max_len
        self.planner = PlannerHead(hidden)
        self.plan_emb = PlanEmbedding(N_PLAN, hidden)

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_base(cls, base_name, device="cpu", dtype=None, plan_max_len=12,
                  r=16, alpha=32, dropout=0.05):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        dtype = _resolve_dtype(dtype, device)
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        backbone = build_lora(base, r=r, alpha=alpha, dropout=dropout, is_trainable=True)
        hidden = base.config.hidden_size
        m = cls(backbone, tok, hidden, plan_max_len)
        return m.to(device=device, dtype=dtype)

    @classmethod
    def from_checkpoint(cls, base_name, ckpt_dir, device="cpu", dtype=None,
                        is_trainable=False, plan_max_len=None):
        """Load adapter + heads. For RL you MUST pass is_trainable=True or RL is a no-op.

        Verifies that the vocabulary in PLAN_VOCAB matches the planner head's output dimension.
        If the checkpoint contains its own plan_vocab.json (all new checkpoints do), it is compared
        to the module-level vocab and a clear error is raised on mismatch."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        cfg = {}
        cfg_path = os.path.join(ckpt_dir, "joint_config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
        base_name = cfg.get("base", base_name)
        plan_max_len = plan_max_len or cfg.get("plan_max_len", 12)
        tok = AutoTokenizer.from_pretrained(ckpt_dir if os.path.exists(
            os.path.join(ckpt_dir, "tokenizer_config.json")) else base_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        dtype = _resolve_dtype(dtype, device)
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        backbone = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=is_trainable)
        _force_lora_trainable(backbone, is_trainable)
        hidden = base.config.hidden_size
        m = cls(backbone, tok, hidden, plan_max_len)

        heads_path = os.path.join(ckpt_dir, "heads.pt")
        heads = torch.load(heads_path, map_location="cpu")

        # --- load planner head with backwards compat for old single-Linear checkpoints ---
        try:
            m.planner.load_state_dict(heads["planner"])
        except RuntimeError:
            if "proj.weight" in heads["planner"]:
                # Old checkpoint: single nn.Linear with keys proj.weight / proj.bias.
                # Transplant into the new MLP's final layer (proj[2]) so inference still works.
                m.planner.proj[2].weight.data.copy_(heads["planner"]["proj.weight"])
                m.planner.proj[2].bias.data.copy_(heads["planner"]["proj.bias"])
                print("[model_joint] compat: loaded old single-Linear planner head into MLP proj[2]")
            else:
                raise

        m.plan_emb.load_state_dict(heads["plan_emb"])

        # --- vocab size guard: checkpoint planner head must match current PLAN_VOCAB ---
        head_out = m.planner.proj[-1].out_features
        if head_out != N_PLAN:
            ckpt_vocab_path = os.path.join(ckpt_dir, _VOCAB_FILE_NAME)
            hint = (f"Set PLAN_VOCAB_FILE={ckpt_vocab_path} before importing model_joint, "
                    "or load via the checkpoint's plan_vocab.json.")
            raise RuntimeError(
                f"Plan vocab size mismatch: checkpoint planner head has {head_out} outputs "
                f"but current PLAN_VOCAB has {N_PLAN} tokens. {hint}")

        return m.to(device=device, dtype=dtype)

    def save(self, out_dir, base_name):
        os.makedirs(out_dir, exist_ok=True)
        self.backbone.save_pretrained(out_dir)            # LoRA adapter
        self.tok.save_pretrained(out_dir)
        torch.save({"planner": self.planner.state_dict(),
                    "plan_emb": self.plan_emb.state_dict()},
                   os.path.join(out_dir, "heads.pt"))
        json.dump({"base": base_name, "plan_max_len": self.plan_max_len, "hidden": self.hidden},
                  open(os.path.join(out_dir, "joint_config.json"), "w"), indent=2)
        # Always write the vocabulary so the checkpoint is self-contained.
        save_vocab(out_dir)

    # ---- low-level helpers ---------------------------------------------------
    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def embed_tokens(self, input_ids):
        return self.backbone.get_input_embeddings()(input_ids)

    def n_trainable_backbone(self):
        return sum(1 for n, p in self.backbone.named_parameters()
                   if p.requires_grad and "lora_" in n)

    def encode_prompt(self, instruction):
        """Chat-templated prompt text."""
        msgs = [{"role": "user", "content": instruction}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n")

    def batch_prompts(self, instructions):
        texts = [self.encode_prompt(x) for x in instructions]
        enc = self.tok(texts, return_tensors="pt", padding=True)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # ---- PLANNER policy ------------------------------------------------------
    def planner_logits_tf(self, prompt_ids, prompt_attn, plan_ids):
        """Teacher-forced planner logits for predicting plan_0..plan_{L-1}.

        Feeds [prompt_embeds, plan_emb(plan_0..plan_{L-2})] and reads the last L hidden states.
        Returns (B, L, N_PLAN). Left padding => the predict-next slice is exactly `[:, -L:]`."""
        B, L = plan_ids.shape
        p_emb = self.embed_tokens(prompt_ids)                       # (B,Tp,H)
        if L > 1:
            pl_in = self.plan_emb(plan_ids[:, :-1])                 # (B,L-1,H)
            inp = torch.cat([p_emb, pl_in], dim=1)
            attn = torch.cat([prompt_attn, (plan_ids[:, :-1] != PAD_ID).long()], dim=1)
        else:
            inp, attn = p_emb, prompt_attn
        out = self.backbone(inputs_embeds=inp, attention_mask=attn, output_hidden_states=True)
        h = out.hidden_states[-1][:, -L:, :]                        # (B,L,H)
        return self.planner(h)                                      # (B,L,N_PLAN)

    def plan_logp_tf(self, prompt_ids, prompt_attn, plan_ids, temp=1.0):
        """Per-token logprob of the given plan under the planner policy (temp=1 -> the policy).
        Returns logp (B,L) and mask (B,L) of non-PAD plan tokens."""
        logits = self.planner_logits_tf(prompt_ids, prompt_attn, plan_ids) / temp
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, plan_ids.unsqueeze(-1)).squeeze(-1)   # (B,L)
        mask = (plan_ids != PAD_ID).float()
        return logp * mask, mask

    @torch.no_grad()
    def sample_plan(self, prompt_ids, prompt_attn, temp=1.0, sample=True, max_len=None):
        """Autoregressively roll out a plan. Returns plan_ids (B, max_len) padded with PAD,
        stopping each sequence after it emits a terminator token. Uses vectorised terminator check."""
        max_len = max_len or self.plan_max_len
        B = prompt_ids.size(0)
        p_emb = self.embed_tokens(prompt_ids)
        cur_emb, cur_attn = p_emb, prompt_attn
        plan = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=self.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        for t in range(max_len):
            out = self.backbone(inputs_embeds=cur_emb, attention_mask=cur_attn,
                                output_hidden_states=True)
            logits = self.planner(out.hidden_states[-1][:, -1, :])      # (B,N_PLAN)
            logits[:, PAD_ID] = float("-inf")                           # never emit PAD
            if sample:
                probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            plan[:, t] = nxt
            done = done | _is_terminator(nxt)                           # vectorised check
            if done.all():
                break
            step_emb = self.plan_emb(nxt.clamp_min(0)).unsqueeze(1)     # (B,1,H)
            cur_emb = torch.cat([cur_emb, step_emb], dim=1)
            cur_attn = torch.cat([cur_attn, (~done).long().unsqueeze(1)], dim=1)
        return plan

    @torch.no_grad()
    def sample_random_plan(self, prompt_ids, max_len=None):
        """Sample a plan by drawing tokens uniformly from non-PAD plan vocab (no backbone call).
        Used for the random-plan ablation: proves the CONTENT of a plan matters, not just its
        presence as a soft prefix. Each sequence stops after the first terminator token."""
        import random as _random
        max_len = max_len or self.plan_max_len
        B = prompt_ids.size(0)
        non_pad = [i for i in range(N_PLAN) if i != PAD_ID]
        plan = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=self.device)
        for b in range(B):
            for t in range(max_len):
                tok = _random.choice(non_pad)
                plan[b, t] = tok
                if tok in TERM_IDS:
                    break
        return plan

    # ---- EXECUTOR policy -----------------------------------------------------
    def _plan_prefix(self, prompt_ids, prompt_attn, plan_ids):
        """[prompt_embeds, plan_prefix_embeds] and matching attention mask.
        plan_ids=None -> no prefix (the 'no-plan' ablation condition)."""
        p_emb = self.embed_tokens(prompt_ids)
        if plan_ids is None or plan_ids.numel() == 0:
            return p_emb, prompt_attn, 0
        pre = self.plan_emb(plan_ids)                                   # (B,L,H)
        plan_mask = (plan_ids != PAD_ID).long()
        inp = torch.cat([p_emb, pre], dim=1)
        attn = torch.cat([prompt_attn, plan_mask], dim=1)
        return inp, attn, plan_ids.size(1)

    def executor_logits_tf(self, prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn):
        """Teacher-forced executor logits for predicting resp_0..resp_{R-1}.
        Feeds [prompt, plan_prefix, resp_0..resp_{R-2}] and reads the last R LM-head logits."""
        R = resp_ids.size(1)
        pre_inp, pre_attn, _ = self._plan_prefix(prompt_ids, prompt_attn, plan_ids)
        if R > 1:
            r_in = self.embed_tokens(resp_ids[:, :-1])
            inp = torch.cat([pre_inp, r_in], dim=1)
            attn = torch.cat([pre_attn, resp_attn[:, :-1]], dim=1)
        else:
            inp, attn = pre_inp, pre_attn
        out = self.backbone(inputs_embeds=inp, attention_mask=attn)
        return out.logits[:, -R:, :]                                    # (B,R,V)

    def resp_logp_tf(self, prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn, temp=1.0):
        """Per-token logprob of the response under the executor policy. Returns logp (B,R),
        mask (B,R). EXECUTOR tokens only — prompt and plan are never in this tensor."""
        logits = self.executor_logits_tf(prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn) / temp
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
        mask = resp_attn.float()
        return logp * mask, mask

    @torch.no_grad()
    def generate_answer(self, prompt_ids, prompt_attn, plan_ids, temp=1.0, sample=True,
                        max_new_tokens=64, top_p=0.95):
        """Sample/greedy-decode an answer conditioned on the plan prefix (soft-prefix embeds).
        Returns generated token ids (B, gen_len) — new tokens only (inputs_embeds path,
        transformers>=4.51)."""
        inp, attn, _ = self._plan_prefix(prompt_ids, prompt_attn, plan_ids)
        gen = self.backbone.generate(
            inputs_embeds=inp, attention_mask=attn,
            do_sample=sample, temperature=(temp if sample else None),
            top_p=(top_p if sample else None),
            max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        return gen

    def base_executor_logits(self, prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn):
        """Executor logits with LoRA adapters DISABLED (the frozen base model). Used for the
        KL-to-base anchor in SFT. Returns (B,R,V) aligned like executor_logits_tf."""
        with self.backbone.disable_adapter():
            with torch.no_grad():
                return self.executor_logits_tf(prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn)


if __name__ == "__main__":
    # smoke test of the plan vocab + guard logic (no base model needed)
    assert encode_plan(["EXTRACT","EVAL","TERMINATE"]).shape[0] == 12
    assert PLAN2ID["EVAL"] > 0
    assert decode_plan(encode_plan(["EXTRACT","TERMINATE"])) == ["EXTRACT","TERMINATE"]
    assert PLAN2ID.get("FILTER", PAD_ID) == PAD_ID  # FILTER[even] base collapses to PAD if absent
    # terminator vectorised check
    nxt = torch.tensor([TERM_ID, 0, TERM_ID])
    expected = torch.tensor([True, False, True])
    assert (_is_terminator(nxt) == expected).all(), "_is_terminator failed"
    print("model_joint smoke: plan vocab OK, encode/decode OK, _is_terminator OK")
    print(f"plan vocab size: {N_PLAN} | LoRA targets: {LORA_TARGETS}")
