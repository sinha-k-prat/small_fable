# ARCHITECTURE_INTERLEAVED.md — Canonical interleaved (agentic, closed-loop) spec for small_fable

## 0. Decision

Synthesize Approach A's **backward-compatibility discipline** with Approach B's **clean per-turn MDP layout**. The canonical design is:

- **One interleaved hidden-space stream per trace**, built turn-by-turn (B's layout), living entirely on the existing `inputs_embeds` path (no new backbone inputs, no lm_head resize).
- **Two existing heads only** (`planner` over `PLAN_VOCAB`, `lm_head` over text vocab). **Target-type position routing**, not a learned switch (both A and B agree; we adopt it).
- **Markers are PLAN-vocab control ids** (so they are embedded by `plan_emb` and predicted by `planner`), EXCEPT the response terminator, which is the **existing text EOS / `<|im_end|>`** (no text-vocab growth). This is the one place A and B differ and we take the cheaper option: reuse `<|im_end|>` for `RESP_EOS`.
- Everything is behind **`--interleaved` (default False)**. The flat path (`planner_logits_tf`, `executor_logits_tf`, `sample_plan`, `generate_answer`, `_plan_prefix`) is untouched and byte-identical so all current tests pass.
- New plan-vocab control ids are **appended** so old ids are stable; the existing `n_plan` resume/load guard catches mismatches.

### Why this makes the plan load-bearing (the whole point)

In the flat path the executor target was the *entire self-contained prose*, so the executor learned to reason without the plan → `ablation_gap ≤ 0`. Here each `response_t` is emitted *only after* and *causally downstream of* `primitive_t`, and the executor **cannot see future turns' reasoning** — it emits exactly one chunk per primitive set, then yields. `primitive_t = f(instruction, responses_<t)` is enforced by causal position order. Corrupting/dropping `primitive_t` removes the only conditioning that gates `response_t`, so accuracy must drop. Closed-loop corrective primitives (BACKTRACK/CORRECT/ITERATE/FINALIZE-vs-continue) only become *selectable* in this layout.

---

## 1. Plan-vocab control symbols (appended)

`traces_to_sft.py --interleaved` appends, after the data-derived tokens, these reserved PLAN-vocab ids:

```
... data primitives + param atoms ...
END         # ALREADY EXISTS (per-turn plan terminator == "stop planning this turn, hand off to executor")
BOP         # NEW: begin-of-plan marker (planner-mode entry for a turn); a real plan input embedding
FINALIZE_ALL# NEW: global stop — "no more turns" (planner predicts this at a resp->plan boundary)
```

So `N_PLAN` grows by **+2** (BOP, FINALIZE_ALL). `END` is reused as `PLAN_EOS` (the per-turn handoff). `RESP_EOS` = tokenizer `<|im_end|>` / `eos` (a TEXT id — no plan-vocab and no text-vocab growth). `PAD_ID`/`TERM_IDS` logic unchanged.

`joint_config.json` gains:
```json
{"interleaved": true, "n_plan": <N>, "markers": {"BOP": <id>, "PLAN_EOS": <END id>,
 "FINALIZE_ALL": <id>, "RESP_EOS_text": <tok.eos or im_end id>},
 "plan_max_len_per_turn": 12, "max_turns": 6}
```
Absent `interleaved` ⇒ flat (old checkpoints load + run flat unchanged).

---

## 2. Sequence layout (one training row → one hidden stream)

A `--interleaved` SFT row carries:
```json
{"instruction": "...", "turns": [{"plan": ["MODEL","as=truth_table","LINK","guard=on"],
                                   "response": "From it rained ..."}, ...],
 "plan": [...flat...], "answer": "...FINAL ANSWER: yes", ...}   // flat fields dual-written
```

The hidden stream assembled at train time (all in hidden space, left-padded prompt as today):

```
 [ prompt_embeds ]                                   embed_tokens(prompt_ids)         (LEFT-padded)
 for t in 1..T:
   [ plan_emb(BOP) ]                                 plan-space marker
   [ plan_emb(primitive_t, param atoms ...) ]        plan-space tokens
   [ plan_emb(END) ]                                 per-turn handoff (== PLAN_EOS)
   [ embed_tokens(response_t tokens) ]               text-space prose chunk
   [ embed_tokens(RESP_EOS=<|im_end|>) ]             text-space turn terminator
 [ plan_emb(FINALIZE_ALL) ]                          global stop (terminal plan target)
```
The terminal turn's `response_t` already ends with `FINAL ANSWER: X` so the answer-key grader has a span.

ASCII, target-type ownership (next-token per source position):

```
 source emb:  P P P | BOP m m m END | r r r EOR | BOP m m END | r r EOR | FIN
              embed   ----plan_emb----  -embed_tok-  --plan_emb--  embed   plan
 next token:  - - B | m   m m E  r  | r r E  B  | m  m  E  r  | r E  F  | (stop)
 owner head:  . . N | N   N N N  L  | L L L  N  | N  N  N  L  | L L  N  | -
              (N=planner over N_PLAN ; L=lm_head over text vocab ; .=masked/prompt)
```

Boundary semantics (the load-bearing handoffs):
- **plan→resp**: the position emitting `END` is a PLAN target (planner learns *when to stop planning this turn*). The first response token is an L/lm_head target read from the post-END hidden state.
- **resp→plan**: the position emitting `RESP_EOS` is an L target (executor learns *when a chunk ends*). The NEXT plan token (`BOP`/primitives, or terminally `FINALIZE_ALL`) is a PLAN target (planner learns *plan again vs finalize* — where BACKTRACK/CORRECT/FINALIZE is chosen, conditioned on the response just produced).

`attention_mask` = 1 on every non-PAD position. RoPE is relative; the whole stream is one causal context. `primitive_t` attends to prompt + responses_<t only — the structural `primitive_t = f(instruction, responses_<t)` guarantee.

> **Critical alignment rule:** the stream, the per-position `seg_type` (PLAN/RESP/IGNORE), and the per-position `target_id` are built in **one pass** so labels are next-token-shifted consistently. Marker *source* embeddings and PAD/prompt-interior positions are IGNORE for both heads. Every non-ignore position is owned by **exactly one** head (asserted).

---

## 3. Forward + heads

ONE backbone forward over the full `inputs_embeds` with `output_hidden_states=True`:
- `h = out.hidden_states[-1]` (B,S,H) → `planner(h[plan_target_mask])` → (·, N_PLAN).
- `lm_logits = out.logits` (B,S,V) → `lm_logits[resp_target_mask]` → (·, V) (frozen+tied lm_head).

Both from the **same** call. Routing is by `seg_type` of the NEXT target, computed at assembly time, not learned. New method `interleaved_tf(prompt_ids, prompt_attn, turns_batch)` returns `h, lm_logits, plan_target_mask, plan_targets, resp_target_mask, resp_targets, attn, raw_targets`, reused by SFT, KL anchor, and RL logp recompute.

---

## 4. Loss + masking

Two cross-entropies over the SAME forward, disjoint masks, plus KL on RESP only:
```
L = CE_plan(planner(h)[plan_target_mask], plan_targets) 
  + lam_resp * CE_resp(lm_logits[resp_target_mask], resp_targets)
  + lam_kl   * KL(executor || base) over resp_target_mask only
```
- `plan_target_mask`: next token is a primitive, param atom, `END`(=PLAN_EOS), or `FINALIZE_ALL`. The handoff `END` and global `FINALIZE_ALL` ARE plan labels (load-bearing: planner trained to stop + to finalize).
- `resp_target_mask`: next token is a response text token or `RESP_EOS`. `RESP_EOS` IS a text label (executor trained to yield).
- Masked from both: prompt interior, PAD, marker source positions.
- KL: `disable_adapter()` over the SAME interleaved embeds (cached from `interleaved_tf`), read lm_logits only at resp positions.

Same `L = plan_ce + lam_resp*resp_ce + lam_kl*KL` shape as today; per-position masks replace per-segment slices.

---

## 5. Generation loop (`run_interleaved`)

Autoregressive, marker-driven, growing one `inputs_embeds` buffer:
```
state=PLAN; turn=1; append prompt_embeds; append plan_emb(BOP)
loop:
  PLAN: forward last pos -> planner logits (PAD=-inf); sample next plan id; append plan_emb(next)
        next==END         -> state=RESP
        next==FINALIZE_ALL -> break (finalize)
        (cap plan tokens -> force END)
  RESP: forward last pos -> lm_logits (lm_head); sample next text token; append embed_tokens(next)
        next==RESP_EOS(<|im_end|>) -> state=PLAN; append plan_emb(BOP); turn+=1
        (hard max_resp cap -> force RESP_EOS)
  turn>max_turns -> force terminal FINALIZE turn then break
```
The executor literally cannot decode past `RESP_EOS` without the next turn's primitives chosen first ⇒ plan load-bearing by construction. To guarantee a gradeable commitment, force a terminal FINALIZE turn (response ends `FINAL ANSWER: X`) before `FINALIZE_ALL`, OR grade the last turn's prose tail. `sample_plan`/`generate_answer` stay for the flat path.

`force_plan(turn_idx) -> list[plan ids]` overrides the planner per turn (drives the ablations: empty plan or shuffled primitives) while keeping the handoff structure intact.

---

## 6. RL (GRPO) — `joint_grpo_loss` UNCHANGED

Trajectory = sequence of (plan-seg, resp-seg) pairs. The two-policy contract holds with longer tensors:
- `rollout_offline --interleaved`: replace `sample_plan`+`generate_answer` with `run_interleaved`. Record per position: token id, head type, temp=1 teacher-forced `logp_old`. **Concatenate ALL turns' plan tokens** into one `plan_logp_old`/`plan_mask`, **ALL turns' resp tokens** into one `resp_logp_old`/`resp_mask`. These are exactly the PLAN/RESP position sets, so `exec_mask` = response tokens only (point 1) and the plan ratio is over plan-vocab steps only (point 2).
- `logp_new` per inner epoch: recompute via `interleaved_logp_tf` teacher-forcing the recorded trajectory. ratio==1 at pass 0 holds (A6 `logp_mismatch_t0 ~ 0`).
- **Advantage: one scalar per trajectory** (group-normalized checker reward), broadcast over ALL tokens of BOTH heads across ALL turns. `group_advantages` + `adv_weights` (A1/A1b) + A2 zero-spread + A5 long2short (`lengths` = total resp tokens across turns) all transfer unchanged.
- `grpo_offpolicy.py`: **no changes**. `train_grpo_offline.build_group_tensors` gains an interleaved branch returning the same shapes/contract.

Per-turn/process credit is a flagged future extension (does NOT fit the drop-in objective).

---

## 7. Ablation redefined (`eval_held --interleaved`)

Two closed-loop tests (no single prefix exists here):
1. **NO-PLAN (drop)** = headline `ablation_gap`: `run_interleaved` with `force_plan=lambda t: []` (empty `BOP→END`), each chunk written with no primitives. `acc(chosen) - acc(empty)`.
2. **PLAN-CORRUPTION (shuffle/replace)**: force WRONG/SHUFFLED in-vocab primitives each turn. `acc(chosen) - acc(shuffled)` isolates *right* primitives from *any* primitives (detects per-family template memorization).

Both should be **> 0** now because the executor is gated turn-by-turn. The flat path keeps the existing two-decode gap; `eval_held` branches on `args.interleaved`.

---

## 8. Backward compatibility (everything behind `--interleaved`, default False)

- `traces_to_sft.py`: default flatten unchanged; `--interleaved` ALSO writes `turns` while STILL writing `plan`/`answer` (dual-write). Appends BOP/FINALIZE_ALL to `plan_vocab.json`; END already present.
- `model_joint.py`: ADD `interleaved_tf`, `interleaved_loss`, `interleaved_logp_tf`, `run_interleaved`, marker id resolution. KEEP all flat methods verbatim. `save()` writes `interleaved`/`markers` only when set; `from_checkpoint` reads them. heads.pt sized to the new `N_PLAN`. Existing `n_plan` guard rejects cross-size loads.
- `train_sft.py` / `rollout_offline.py` / `eval_held`: branch on `args.interleaved`. Off ⇒ current code verbatim. `grpo_offpolicy.py`: unchanged.
- Requesting `--interleaved` with an OLD flat `plan_vocab.json` (no BOP/FINALIZE_ALL) raises a clear error.

---

## 9. Memory note (T4, fp32 1.5B)

Stream length = prompt + Σ_t(plan_t + resp_t). Mitigations: gradient checkpointing ON; `max_turns` cap (≤6); `plan_max_len_per_turn` and `max_resp_tokens` caps; single forward with a global length cap; the KL base forward reuses the SAME cached embeds (one extra `disable_adapter()` pass, resp positions only).
