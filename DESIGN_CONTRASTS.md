# Design contrasts — what was rejected, and what was chosen instead

Every non-obvious decision in this repo as a **contrast**: the obvious/standard alternative (✗) vs.
the choice actually made (✓), with the one reason it went that way. Read alongside
[`important_design_choices.md`](important_design_choices.md) and the inline comments.

```
Legend:   ✗ = the alternative that was rejected      ✓ = what was chosen instead
```

---

## A · Architecture

### 1. The plan's representation
```
        ✗ plan = special/text tokens in the LM's own vocabulary  (the "planning tokens" route)
          planning blurs into prose — not a thing you can steer or reward on its own
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ plan = a SEPARATE planner vocabulary with its own head + embeddings
          (synthetic fallback: 41 tokens; the reasoning-traces pipeline uses a FACTORED vocab =
          primitives + key=value param-atoms + END — see §E)
          a discrete action space you can reinforce as an independent policy
```
**Why:** planning becomes first-class and RL-able, not more text.

### 2. Where the planner lives
```
        ✗ a second, separate planner transformer alongside the executor
          two models to host, train, and keep in sync
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ an autoregressive planner HEAD reading the SHARED backbone's hidden states
          one backbone, TWO SEPARATE HEADS (planner linear + executor LM path) + a shared plan_emb
```
**Why:** the plan is conditioned on the same prompt representation the answer is — true "one model."
**Both heads are trained in SFT and BOTH are retrained in RL** from the same SFT checkpoint (GRPO
reloads the SFT adapter + planner head + plan_emb): in SFT the planner is trained by plan-CE and the
executor adapter by response-CE/KL; in GRPO the planner is updated by the clipped plan-policy term + a
small CE anchor and the executor by the clipped response term. The frozen base model — including the
tied LM head — is never trained in either stage; only the LoRA adapter, planner head, and plan_emb move.

### 3. How the plan reaches the answer
```
        ✗ paste the plan as a text string into the prompt (or never feed it back at all)
          the model can ignore it; nothing forces conditioning
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ prepend the plan's EMBEDDINGS as a soft prefix (hidden-space vectors), then answer
          the executor physically attends to the plan before it writes
```
**Why:** this is the *mechanism* that makes the plan load-bearing — and it's directly testable.

### 4. Proving the plan matters
```
        ✗ assume the plan helps; report only final accuracy
          a decorative plan looks identical to a real one
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ decode WITH the plan vs WITH NO plan; track the accuracy gap as a first-class metric
          gap > 0  ⇒  the plan is load-bearing
```
**Why:** turns "does the plan do anything?" into a number you watch every epoch.

### 5. Padding convention
```
        ✗ right-padding + per-example index gather to find each "predict next-k" slice
          fiddly, error-prone indexing
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ LEFT-padding → last real prompt token always at index −1 → clean  [:, -k:]  slices
          RoPE is relative, so a constant left shift leaves attention identical
```
**Why:** the tensor math stays simple and exact.

### 6. The reward signal
```
        ✗ reward = NLL or embedding similarity to a reference answer
          saturates on easy items; hides whether the plan helped
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ reward = a programmatic VERIFIABLE checker  (correctness in {0,1}, or graded)
          clean signal, genuine reward variance
```
**Why:** RL needs honest, non-saturating variance — not a similarity score.

---

## B · SFT (Stage 1)

### 7. Stabilizing early SFT
```
        ✗ train the heads with no anchor, or hard-freeze the backbone for safety
          either drifts, or can't move at all
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ CE(plan) + λ·CE(answer) + λ_kl·KL(executor ‖ base)   (KL via PEFT disable_adapter)
          a light leash to the base model, adapter still free to move
```
**Why:** stability without freezing the very capacity you're training.

### 8. Picking the "hard" curriculum
```
        ✗ train uniformly on everything, or hand-label which prompts are "hard"
          capacity spread thin; "hard" is subjective
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ broad pass first, then the STAGE-1 MODEL ITSELF keeps prompts it solves ≤ 2 of 8
          difficulty defined by the current model, not a human
```
**Why:** capacity concentrates where plans matter — the ablation gap should *grow* in stage 2.

### 9. Batch ordering
```
        ✗ fully shuffled batches  (no progression)   —or—   strict global sort  (kills coverage)
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ ascending difficulty BANDS with intra-band shuffle
          genuine easy→hard, categories still interleave
```
**Why:** a real curriculum without collapsing category coverage or stochasticity.

### 10. Guaranteeing RL has variance
```
        ✗ one (plan, answer) per row; discover that RL groups are all zero-variance... in RL
          you find out when training has already stalled
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ train on (plan, answer) ALTERNATIVES, then probe distinct-plans/prompt + Pass@k after SFT
          low diversity is surfaced in SFT, where it's fixable
```
**Why:** moves a fatal RL failure mode upstream to where you can act on it.

---

## C · Off-policy GRPO (Stage 2) — the core

### 11. On-policy vs offline
```
        ✗ on-policy: regenerate rollouts every gradient step
          generation is the slow part on a weak GPU — brutal
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ generate the group ONCE, reuse it for 2–3 gradient passes
          amortize the expensive step (but now the batch is off-policy → see #12)
```
**Why:** makes RL feasible on a free T4.

### 12. Handling the staleness reuse creates
```
        ✗ reuse the stale rollouts as if they were on-policy
          the policy has drifted; uncorrected updates diverge
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ PPO-CLIPPED importance ratio  exp(logp_new − logp_old),  auto-cut past kl_stop≈0.15
          the ratio CORRECTS staleness — it is not a reward bonus
```
**Why:** off-policy reuse is only safe with the importance correction.

### 13. Crediting two policies from one outcome
```
        ✗ a single shared ratio over plan+response tokens  —or—  invent a separate plan reward
          conflates two action spaces / fabricates a signal that doesn't exist
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ TWO independent ratios (own masks),  the SAME group advantage (r−mean)/std on both
          planner + executor share the single terminal reward, hierarchically
                 reward (answer only)
                        │
              advantage = (r − mean)/std        ← computed ONCE
                   ┌────┴────┐
              plan ratio   exec ratio           ← separate, clipped separately
                   └────┬────┘
                shared LoRA trunk                ← gradients from BOTH pour in here
```
**Why:** the plan earns credit purely by leading the executor to better answers — clean options-style RL.

### 14. Keeping the executor ratio honest
```
        ✗ compute the executor ratio over prompt + plan + response tokens
          the plan head's logprobs leak into the executor's policy ratio
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ the executor logp tensor contains RESPONSE TOKENS ONLY
          two action spaces stay cleanly separated
```
**Why:** prevents silent cross-contamination of the importance ratio.

### 15. Trusting logp_old
```
        ✗ take logp_old from the sampler/rollout engine
          if sampler ≠ trainer numerics, the ratio is corrupt at step 0 and no clip saves it
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ recompute logp_old via the TRAINER's own teacher-forced path @ temp=1; assert |Δ|≈0
          the offline ratio is provably valid before the first update
```
**Why:** kills the train/infer-mismatch failure (the VibeThinker trap) before it starts.

### 16. Did RL actually train anything?
```
        ✗ trust PeftModel.from_pretrained  (it loads adapters FROZEN by default)
          RL silently becomes a no-op; SFT == SFT+RL byte-for-byte
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ load with is_trainable=True AND assert > 0 trainable backbone tensors at startup
          plus |ΔL2| > 0 at the end to prove the adapter moved
```
**Why:** a guard against a failure that already bit a prior run.

### 17. Dealing with zero-variance groups
```
        ✗ DELETE every group whose rollouts all agree  (hard filter)
          throws away prompts and is brittle at the 0/1 edges
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ MaxEnt weight  w = exp(−γ(p_q−0.5)²):  down-weight saturated AND hopeless, keep the middle
          a smooth bell peaking where correct & incorrect rollouts coexist
```
**Why:** the principled, soft version of the deletion hack — keeps the informative signal.

### 18. Scoring soft (non-verifiable) tasks
```
        ✗ force every task through one binary keyword checker
          soft tasks reward keyword-stuffing; MGPO would faithfully AMPLIFY that
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ route by reward_path: verifiable→binary+bell, rubric→graded+variance-weight, judge→avg
          (or --exclude_rubric: hold soft families out of RL, rely on SFT)
```
**Why:** the weighting scheme is only as honest as the reward it amplifies.

### 19. The pre-RL filter
```
        ✗ filter only the literal p_q ∈ {0, 1} case
          misses zero-spread GRADED groups that also yield no gradient
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ drop groups with reward std ≈ 0  (binary AND graded), log to pre_rl_filter_report.csv
```
**Why:** "no spread" — not "exactly 0 or 1" — is what actually means "no gradient."

### 20. Encouraging short plans
```
        ✗ add a brevity BONUS to short answers
          inflates the group mean → biases the advantage baseline
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ ZERO-SUM brevity redistribution among CORRECT rollouts, applied BEFORE advantages
          shorter-correct gets relatively more; group baseline undisturbed
```
**Why:** teaches "short plan when it suffices" without secretly moving the goalposts.

### 21. Keeping the planner on-vocab
```
        ✗ heavy CE pull toward the gold plan
          pins the policy back to SFT and erases the RL movement you're measuring
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ a SMALL CE anchor (beta_ce 0.1) only
          enough to prevent off-vocab drift, light enough to let RL move
```
**Why:** a leash, not a cage.

---

## D · Eval & infrastructure

### 22. Ranking traces for distillation
```
        ✗ rank distillation traces by raw (length-normalized) NLL across all lengths
          long traces dominate; "missing" and "merely long" get confused
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ rank by length-normalized NLL WITHIN length buckets, drop shortest + extreme-NLL tail
```
**Why:** isolates what the student is actually missing from what's just verbose.

### 23. Test-time answer selection
```
        ✗ majority vote / mean self-verdict over sampled trajectories
          a mostly-right trajectory with one broken claim still passes
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ score (mean_verdict)^M  (nonlinear) + cluster by answer-equivalence, pick max reliability
          one bad claim sinks the trajectory
```
**Why:** rewards internally-consistent reasoning, not just popular answers.

### 24. Surviving Colab
```
        ✗ save only the FINAL checkpoint
          Colab kills the runtime at ~4h → the long RL run restarts from scratch
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ checkpoint model + optimizer + EXACT position every ~10 min, pushed to HF; --resume
          SFT resumes at (epoch, batch); GRPO resumes at (inner_epoch, group_idx)
```
**Why:** the run that dies midway continues from the exact group, not the beginning.

---

## ★ The five that matter most

If you remember only five — these are the ones that signal the failure modes were actually
thought through, not just a stock GRPO wiring:

| # | Choice | The rejected default | Why it's the original bit |
|---|--------|----------------------|---------------------------|
| **1** | Separate-vocabulary planner head | plan = tokens in the LM vocab | makes "planning" a discrete, independently-reinforceable policy |
| **13** | Two policies, **one** shared reward, on a shared trunk | one ratio, or a fake separate plan reward | hierarchical credit assignment done correctly — the conceptual core |
| **4** | Load-bearing ablation as a first-class metric | report only final accuracy | converts "does the plan help?" into a number you can watch |
| **15** | `logp_old` recomputed by the trainer, asserted ≈ 0 | trust the sampler's logprobs | pre-empts the train/infer-mismatch that silently corrupts off-policy RL |
| **17** | MaxEnt weighting over zero-variance deletion | delete saturated groups | keeps the informative middle instead of throwing prompts away |

```
                              ┌─────────────────────────────────────────┐
   prompt ──▶ [ SHARED Qwen-1.5 backbone + LoRA ] ──┬──▶ planner head ──▶ PLAN   ★1
                              └─────────────────────┘  (factored vocab:    │
                                                         primitives+params+END)
                                        ▲                                 ▼ soft prefix
                                        │                         [ executor ] ──▶ ANSWER
                                        │                                 │
                                        │                                 ▼
                              gradients from BOTH      ◀── checker reward ─┘
                              policies, ONE advantage  ★13          │
                                                                    ▼
                                                   ablation: with-plan vs no-plan  ★4
```

---

## E · Reasoning-traces era (real data + compositional plans)

These supersede the synthetic-data choices above for the current pipeline.

### 25. The training data
```
        ✗ template-generated synthetic tasks (toy; plan often redundant)
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ hard reasoning traces (3×1000) — designed so a model that MEMORIZED one set
          must REASON to solve the deeper (2000) and flipped-answer (3000) variations
```
**Why:** real traps (off-by-one, non-transitive chaining, knaves) where the plan genuinely matters.

### 26. The plan's parameters
```
        ✗ strip parameters: REFLECT[reason=naive_vs_correct] -> bare REFLECT
          1000 problems collapse to 18 generic plans -> plan carries no answer info -> redundant
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ KEEP the parameters — they ARE the strategy (reason=off_by_one, watch=escape_midcycle,
          prop=transitive). With them the executor dodges the trap; without them it falls in.
```
**Why:** the parameter is the load-bearing decision; stripping it is what made the gap non-positive.

### 27. How parameters enter the vocab
```
        ✗ atomic compound tokens MODEL[as=rate] (welds primitive to params; no composition;
          a primitive+param combo unseen in training is unrepresentable)   —or—   two planner heads
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ ONE autoregressive head over a FACTORED vocab: primitive and key=value are separate tokens
          MODEL as=rate VERIFY aspect=units ... FINALIZE form=number_with_units END
```
**Why:** a novel primitive+param pairing is just a novel sequence of known tokens — compositional,
single head, no extra machinery.

### 28. The executor target
```
        ✗ terse final answer ("9 days")  -> can't solve hard problems, output incoherent
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ full reasoning prose + "FINAL ANSWER: <canonical>"  (GSM8K-style commit)
```
**Why:** the model learns to reason AND commits a clean answer the checker can grade precisely.

### 29. The verifiable reward
```
        ✗ one binary keyword checker for everything (false-matches: "no" inside "cannot")
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ per-example answer-key graders (exact_choice / numeric / numeric_or_word / exact_term /
          role_map / string_contains / plan_rubric), grading only the FINAL ANSWER span
```
**Why:** garden-path prose names wrong answers mid-reasoning; grade the commitment, not the prose.
Calibration: gold passes its own checker 100%, wrong answers score 0.

### 30. The held-out split
```
        ✗ rows[train:train+held] — a fixed positional slice (ordering-biased)
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ seeded shuffle of the corpus -> representative random held slice (consistent across stages)
```

### 31. The generalization test
```
        ✗ pool all 3 sets, 90/10 random split  (model SEES flipped variations -> can memorize them)
   ───────────────────────────────────── instead ─────────────────────────────────────
        ✓ train on sets 1000+2000, hold out set 3000 (flipped-answer) ENTIRELY as the unseen test
```
**Why:** keeps the controlled memorize-vs-reason experiment — holding on the flipped set is the
claim that can't be faked by pattern-matching.
