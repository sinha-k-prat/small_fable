# Architecture: Base Qwen vs the joint planner+executor (proposed)

A comparative reference for the two-head adaptive agent. Ground truth is
[`model_joint.py`](model_joint.py), [`train_sft.py`](train_sft.py),
[`train_grpo_offline.py`](train_grpo_offline.py), and [`grpo_offpolicy.py`](grpo_offpolicy.py).

## Intro

The proposed system is **one model with two separate heads on a shared decoder-only LoRA backbone**,
run in two phases: **plan, then execute**. The backbone is Qwen2.5-1.5B-Instruct (decoder-only causal
LM), LoRA-adapted. On top of its last-layer hidden states sit two *untied* new modules: a **planner
head** (`PlannerHead`, an `nn.Linear(hidden, N_PLAN)` emitting plan-token logits) and a **plan
embedding table** (`PlanEmbedding`, an `nn.Embedding(N_PLAN, hidden, padding_idx=PAD)` that turns
chosen plan tokens back into hidden-space vectors). First the planner autoregressively writes a short
plan over a small factored plan vocabulary; then that whole plan (including its terminator) is embedded
as a **soft prefix** and the same LoRA backbone — reading through the **frozen, tied `lm_head`** —
writes the prose answer (`… FINAL ANSWER: X`).

Markers used below: **❄ frozen** · **▣ LoRA (low-rank trained)** · **★ fully-trained (new module)**.

## What trains vs frozen

In the base model `tie_word_embeddings=true`, so `embed_tokens` and `lm_head` are the **same** frozen
tensor. In the proposed model the new `plan_emb` (plan input embedding) and `planner` (plan output
logits) are **separate and untied** — unlike the tied `embed_tokens ↔ lm_head` pair.

| Parameter group | Base Qwen | Proposed |
| --- | :---: | :---: |
| `embed_tokens` (token input embedding) | ❄ frozen | ❄ frozen |
| tied `lm_head` (= `embed_tokens`) | ❄ frozen | ❄ frozen |
| RMSNorm gains (per-layer input/post-attn + final) | ❄ frozen | ❄ frozen |
| q/k/v attention biases | ❄ frozen | ❄ frozen |
| 7 projection weights (q,k,v,o,gate,up,down) | ❄ frozen | ❄ base slice frozen + ▣ LoRA `A`/`B` |
| `plan_emb` (plan **input** embedding) | — (absent) | ★ fully-trained |
| `planner` head (plan **output** logits) | — (absent) | ★ fully-trained |

LoRA is `r=16, alpha=32, bias="none"` on exactly the 7 projection matrices per block
(`LORA_TARGETS = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`). Only the low-rank
`lora_A`/`lora_B` train; the base weight slices stay frozen. Everything in the ❄ rows is never
updated — `bias="none"` adds no biases and leaves any existing ones frozen.

The plan vocabulary is **factored**: primitives + `key=value` param-atoms + a terminator, all ids
living in the **one** `plan_emb` table (no separate "parameter embedding"); ~79 tokens for the traces
run, where the terminator is `END`. With no `plan_vocab.json` present, `model_joint` falls back to a
41-token bare vocab whose terminator is `TERMINATE`. The terminator is run-configured and matched by
**base name** via `TERM_IDS`, so every parameterized variant ends a plan.

## (a) BASE — plain Qwen forward

```
 input_ids
    │
    ▼
 embed_tokens ❄ ─────────────────────────────┐ (tied tensor)
    │                                         │
    ▼                                         │
 ┌──────────────────────────────────────┐    │
 │  Decoder blocks ×N                    │    │
 │   RMSNorm ❄                           │    │
 │   self-attn: q,k,v,o_proj ❄  (+bias ❄)│    │
 │   RMSNorm ❄                           │    │
 │   MLP: gate,up,down_proj ❄            │    │
 └──────────────────────────────────────┘    │
    │                                         │
    ▼                                         │
 final RMSNorm ❄                              │
    │                                         │
    ▼                                         │
 lm_head ❄  ◄───────── SAME tensor as ────────┘
    │
    ▼
 token logits → next token (prose)
```

## (b) PROPOSED — plan phase then execute phase (shared backbone)

The two heads interleave through the **SAME** LoRA backbone. The PLAN phase loops the planner head
autoregressively; the EXECUTE phase runs once over a soft-prefix sequence and reads the frozen `lm_head`.

```
                       ┌──────────────────────────────────────────────┐
                       │   SHARED BACKBONE                            │
                       │   embed_tokens ❄ (tied lm_head ❄)            │
                       │   decoder blocks ×N:                         │
                       │     RMSNorm ❄ · q,k,v,o ❄+▣ (bias ❄)        │
                       │     RMSNorm ❄ · gate,up,down ❄+▣            │
                       │   final RMSNorm ❄                            │
                       └──────────────────────────────────────────────┘
                                  ▲                       │
                                  │ inputs_embeds         │ last hidden
                                  │                       ▼

 ── PLAN phase (autoregressive loop, until terminator) ──────────────────────
    prompt embeds ❄ ──►│ backbone │──► last hidden ──► planner head ★
                                                          │  plan-token logits
                                                          ▼
                                                       sample plan token
                                                          │
                                          plan_emb ★  ◄───┘  (re-feed chosen token)
                                                          │  append as next input embed
                                                          └──────────► loop ▲
                                      stop when token ∈ TERM_IDS  ──► plan = [p0 … END]

 ── EXECUTE phase (single pass) ─────────────────────────────────────────────
    build sequence:
       [ prompt embeds ❄ | plan_emb(whole plan incl terminator) ★  SOFT PREFIX | answer embeds ❄ ]
                                   │
                                   ▼
                            │ SAME backbone ❄+▣ │
                                   │  last hidden
                                   ▼
                            lm_head ❄ (frozen)
                                   │
                                   ▼
                     answer-token logits → prose "… FINAL ANSWER: X"
```

Key points: the planner head ★ and `plan_emb` ★ are the only new modules; the backbone is shared and
LoRA-adapted (❄ base + ▣ low-rank). The plan is fed into the **same** backbone twice — once
token-by-token during PLAN (via `plan_emb`), and once as a whole soft prefix during EXECUTE.

## Gradient flow — which loss updates which params

`plan_emb` is updated by **both** heads; the planner head is updated by the planner loss only; the
executor's trainable footprint is **shared LoRA + `plan_emb`** (it has no head of its own — the frozen
`lm_head` means the executor learns only by reshaping hidden states via LoRA and by conditioning on the
plan prefix).

```
  PLANNER loss                                    EXECUTOR loss
  (plan-CE in SFT /                               (response-CE/KL in SFT /
   clipped plan-policy + beta_ce CE in RL)         clipped response-policy in RL)
        │                                               │
        ├──► planner head ★  (updates)                  │
        │                                               │
        ├──► plan_emb ★      ◄── BOTH heads update ──►   ├──► plan_emb ★
        │                                               │
        └──► LoRA A/B ▣      (updates)        (updates) ─┘──► LoRA A/B ▣

                          ┄ gradient-transparent ┄
   lm_head ❄ · base projection slices ❄ · embed_tokens ❄ · RMSNorm ❄ · q/k/v bias ❄
   loss flows THROUGH them to the trainable params — they carry gradient but never update.
```

The executor loss updates **LoRA + `plan_emb`** (NOT the planner head, NOT `lm_head`). The frozen
`lm_head` and frozen base weights are **gradient-transparent**: the loss passes through them to reach
the trainable parameters, they simply don't receive updates.

## Plan → execute boundary

The plan→execute boundary is the plan's **terminator token** (`END` in the traces run; `TERMINATE` in
the default vocab — matched by base name via `TERM_IDS`). There is **no separate "begin-execution"
token**: the planner stops when it emits the terminator, and execution begins automatically by
generating after the plan soft-prefix.

The terminator's embedding row, `plan_emb[<terminator>]`, is **gradient-carrying from BOTH heads**:
- **Executor side** — the terminator sits in the soft prefix the executor reads (`_plan_prefix` embeds
  all non-PAD plan tokens, including the terminator), so `resp_CE` back-props into it.
- **Planner side** — during teacher-forcing the terminator is fed back as an **input** plan-embedding
  when predicting the token after it (it is the last real token before PAD, inside `plan_ids[:, :-1]`),
  so the planner path updates it too. *(The signal for **emitting** the terminator trains the planner
  **head**, not `plan_emb`.)*

**Never freeze `plan_emb`** — it is the executor's only learnable input-side knob and the carrier of
the boundary representation; freezing it cuts the executor's gradient into the boundary.

## Both heads: trained in SFT, retrained in RL

Both heads are trained in SFT and **retrained** in offline GRPO. In SFT the planner is fit with plan
cross-entropy and the executor with response-CE/KL; the shared `plan_emb` accumulates gradient from
both. Offline GRPO then **reloads the SFT adapter plus `heads.pt`** (the planner head + `plan_emb`)
with `is_trainable=True` and continues training both heads under two independent clipped objectives — a
clipped plan-policy loss with a `beta_ce` CE anchor toward the gold plan for the planner, and a clipped
response-policy loss for the executor — so the planner head, `plan_emb`, and LoRA all keep updating
while the frozen `lm_head` and base weights stay gradient-transparent throughout.
