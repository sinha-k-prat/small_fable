# Important design choices (and why)

Every non-obvious decision in this repo, with the reason it was made that way. Read alongside the
inline comments in each file.

## Architecture

1. **Two policies, separate action spaces, separate logprobs.**
   The planner emits over `PLAN_VOCAB` — the factored, parameterized plan vocabulary loaded from
   `plan_vocab.json` (primitives + `key=value` param-atoms + `END`; a bare default, 41 tokens incl. PAD,
   only when that file is absent); the executor emits over the token vocab.
   They are different distributions, so their logprobs, ratios, and clipped objectives are kept
   independent everywhere (`joint_grpo_loss` = `L_exec_clip + beta_plan·L_plan_clip`). *Why:* a single
   shared ratio would conflate two unrelated probability spaces and corrupt both gradients.

2. **Planner is autoregressive over the SHARED backbone, not a separate transformer.**
   At each plan step we run the backbone, read its last-layer hidden state, project with the linear
   planner head to plan-primitive logits, sample, and re-feed the chosen primitive as a learned
   **plan embedding**. *Why:* keeps the model "one backbone, two heads" (the brief), lets LoRA serve
   both paths, and makes the plan genuinely conditioned on the prompt representation.

3. **Plan fed back as a SOFT PREFIX (embeddings), then the answer.**
   The chosen plan's embeddings are prepended to the executor's input as hidden-space vectors before
   answer generation. *Why:* this is the mechanism that makes the plan **load-bearing** — the executor
   physically attends to the plan. We verify it with the plan-vs-no-plan ablation gap.

4. **Left-padding convention.**
   Prompts are left-padded so the last real prompt token is always at index −1 and the
   plan/response embeddings appended after it are contiguous. *Why:* every "predict the next k things"
   becomes a clean `[:, -k:, :]` slice with no per-example index gather. RoPE is relative, so a
   constant left shift of an example's positions leaves attention identical — left padding is safe.

5. **`reward = programmatic checker`, never NLL or embedding-similarity.**
   *Why:* NLL/embedding-sim saturate on easy tasks and hide whether the plan helped. A verifiable
   checker gives a clean signal and real reward variance.

## SFT (Stage 1)

6. **Joint loss `CE(plan) + lam_resp·CE(answer) + lam_kl·KL(executor‖base)`.**
   The KL anchor uses PEFT `disable_adapter()` to get the frozen base distribution. *Why:* small KL
   stabilizes early SFT without freezing the adapter's ability to move.

7. **A3 two-stage curriculum (broad → hard), cosine LR + warmup.**
   Stage 1 trains the full set for coverage; stage 2 continues from stage 1 on a **hard subset the
   stage-1 model itself selects** (8 rollouts/prompt, keep error-rate ≥ 0.75 i.e. solved ≤ 2/8).
   *Why:* broad-first builds plan diversity; hard-second concentrates capacity where plans matter,
   which should make the ablation gap **grow** in stage 2 (logged for exactly this check).

8. **Curriculum batching (easy → hard with intra-band shuffle).**
   Difficulty proxy = plan length (+ answer length tiebreak); batches are ordered ascending, but rows
   within an equal-difficulty band are shuffled. *Why:* gives a genuine easy→hard progression without
   collapsing category coverage or killing stochasticity.

9. **A4 spectrum-to-signal: multi-path SFT + diversity probe.**
   Rows may carry an `alternatives` list of `(plan, answer)`; we train on all of them. After SFT we
   measure **distinct plans per prompt** and **Pass@k** on a probe set. *Why:* RL needs the post-SFT
   model to sample diverse rollouts or every group is zero-variance. Low probe diversity *predicts*
   flat RL — so we surface it in SFT, where it is fixable, not in RL where it just stalls.

## Offline GRPO (Stage 2)

10. **Offline by design: generate once, reuse 2–3 passes.**
    Generation is the slow part on a weak GPU. *Why:* amortize it. But reuse makes the batch
    off-policy, which forces the next two choices.

11. **PPO-clipped importance ratio corrects staleness (not a reward bonus).**
    `ratio = exp(logp_new − logp_old)`, clipped to `[1−ε, 1+ε]`, pessimistic `min`. On pass 1
    `ratio≈1` (plain reinforce); later passes the ratio brakes over-represented tokens. We watch
    `approx_kl` and auto-cut inner epochs past `kl_stop≈0.15`. *Why:* without the correction the
    off-policy steps diverge.

12. **Executor-only ratio masking.**
    The executor logp tensor contains **response tokens only** — prompt and plan tokens are never in
    it. *Why:* otherwise plan logprobs leak into the executor policy ratio and contaminate it.

13. **A6 `logp_mismatch_t0` guard.**
    `logp_old` is recomputed at rollout time with the **trainer's own HF teacher-forced path at
    temp=1**, never taken from the sampler. Before any update the trainer re-derives logp on a batch
    and asserts the mean abs diff is ~0. *Why:* if the rollout-engine and training forward disagree,
    the IS ratio is corrupted at step 0 and no clipping saves it (the VibeThinker train/infer-mismatch
    failure). temp=1 logprobs are the actual policy probs the ratio needs.

14. **Frozen-backbone guard — and RL retrains BOTH heads, not just the adapter.**
    The planner head and the LoRA-adapted executor are **separate modules**, and BOTH are trained in
    SFT (planner by plan-CE, executor adapter by response-CE/KL, shared `plan_emb` by both). RL is a
    *continuation* of both: GRPO reloads the SFT'd adapter **and** `heads.pt` (planner + plan_emb), puts
    every trainable param in one optimizer, and keeps updating both — the executor via the clipped
    response term `L_exec`, the planner via the clipped plan term `beta_plan·L_plan` + the small
    `beta_ce` gold-plan CE anchor. It loads the adapter with `is_trainable=True` and asserts `>0`
    trainable backbone tensors at startup. *Why:* `PeftModel.from_pretrained` loads adapters frozen by
    default; a silent freeze makes RL a no-op and SFT==SFT+RL byte-for-byte. This bit a previous run.
    (Only the LoRA adapters + the two heads ever train; the base weights, incl. the tied LM head, stay
    frozen in both stages.)

15. **A1 MGPO MaxEnt weighting replaces "delete zero-variance groups."**
    `w_q = exp(−γ·(p_q−0.5)²)` — an **unnormalized** Gaussian in group accuracy: peak 1.0 at 0.5, no
    `1/(σ√2π)` prefactor, std `= 1/√(2γ)`. To target prompts within ±δ of 0.5 accuracy set
    `γ = 1/(2δ²)` (δ=0.15 → γ≈22). *Why:* down-weighting (not deleting) keeps the informative middle
    where correct and incorrect rollouts coexist, and damps both saturated and hopeless prompts —
    the soft, principled version of the zero-variance filter. The weight multiplies the
    already-group-normalized advantage; the **same** per-prompt weight scales both policies.

16. **A1b graded routing: `reward_path` decides reward AND weighting.**
    `verifiable` → binary reward, weight = `mgpo_weight(p_q)`.
    `rubric` → graded fraction-of-rubric reward in [0,1], weight = `variance_weight` (group reward
    std — the honest "do the rollouts disagree" signal for continuous rewards).
    `judge` → normalized [0,1], last resort, average several calls.
    *Why:* forcing a soft task through a fake binary keyword checker rewards keyword-stuffing and MGPO
    would faithfully amplify it. Soft families (`research_synthesis, planning, ambiguity,
    adversarial_constraint`) route to graded rubric — or `--exclude_rubric` holds them out of RL
    entirely and relies on SFT.

17. **A2 pre-RL filter = zero-spread, not literal 0/1.**
    A group is dropped if its reward std is ~0. *Why:* this covers the literal "p_q exactly 0 or 1"
    case for binary rewards AND zero-spread graded groups (both yield no GRPO gradient), and is logged
    to `pre_rl_filter_report.csv`. It is the cheap per-phase cousin of the planning-sensitivity filter.

18. **A5 long2short: zero-sum brevity shaping among CORRECT trajectories, BEFORE advantages.**
    Within a group, redistribute reward among correct rollouts by `1/length`, zero-sum so the group
    baseline is undisturbed; incorrect rollouts unchanged. *Why:* teaches the adaptive agent to prefer
    the SHORT plan when it suffices (the mode-switching goal) without changing what counts as correct.
    Zero-sum is essential — otherwise it would inflate the group mean and bias the advantage.

19. **Small CE anchors only (`beta_ce 0.1`).**
    *Why:* a light pull toward the gold plan keeps the planner from drifting off-vocab, but a heavy CE
    would pin the policy back to SFT and kill the RL movement we are trying to measure.

## Optional consolidation / eval

20. **A7 learning-potential ranking within length buckets.**
    Distill traces ranked by length-normalized NLL under the current student (high = correct but not
    yet modeled), bucketed by length, dropping the shortest traces and the extreme-NLL tail. *Why:*
    ranking across raw length lets long traces dominate; bucketing isolates "what the student is
    missing" from "what is merely long," and the drops remove format noise / degenerate outliers.

21. **A8 CLR `(mean_verdict)^M` + answer clustering.**
    Sample K trajectories, self-verify each extracted claim to a binary verdict, score a trajectory
    by `(mean_verdict)^M` (nonlinear so any flawed claim hurts), cluster by answer-equivalence, pick
    the answer with max summed reliability. *Why:* eval-only test-time scaling that rewards
    internally-consistent reasoning; the `^M` makes a single bad claim sink a trajectory.

## Data

22. **Bundled data is synthetic; soft families get keyword rubrics.**
    The 1000 rows are template-generated — correct for proving the **pipeline** and the ablation gap,
    not for quality claims. Soft-family rows are converted to graded rubric checkers via
    `annotate_reward_paths.py`. *Why:* exercises the full graded/variance-weighted path honestly; for
    real conclusions, swap in real tasks (same schema) or expand with `build_sensitivity_corpus.py`,
    which filters tasks for `P(correct|plan) ≫ P(correct|no plan)` so RL has guaranteed variance.

## Reasoning-traces pipeline (supersedes the synthetic path)

23. **Train on real hard-reasoning traces, not synthetic templates.** `traces/` holds 3×1000 traces
    + machine-readable answer keys. The three sets are a *controlled* experiment: 2000 is
    "one-step-deeper" and 3000 is "flipped-answer" variations of 1000, so a model that memorized 1000
    must REASON to solve them. *Why:* the synthetic set proved the pipeline but the plan wasn't
    load-bearing on it; these traps (off-by-one, non-transitive chaining, knaves) need the plan.

24. **Factored, parameterized plans from a single autoregressive head.** A plan is a flat token
    sequence of primitives + `key=value` parameter-atoms + `END`
    (`MODEL as=truth_table ... FINALIZE form=yes_no END`), emitted by one head. The parameters are the
    load-bearing strategy (`reason=naive_vs_correct`, `watch=escape_midcycle`, `prop=transitive`).
    *Why:* keeping parameters (not stripping them) is what makes the plan carry answer-determining
    info; the factored sub-token form keeps the head a single fixed action space while staying
    compositional — a novel primitive+param pairing is a novel sequence of known tokens. The vocab is
    written to `plan_vocab.json` by `traces_to_sft.py` and loaded by `model_joint.py` (default 41-token
    bare vocab if absent). The traces run sets `--plan_max_len 24` in the notebook (factored sequences
    reach ~17 tokens); the code default is 12.

25. **Executor target = reasoning prose + `FINAL ANSWER: <canonical>`.** The executor learns to reason
    in text and then commit a canonical answer (GSM8K-style). *Why:* coherent reasoning AND precise
    grading — the checker reads only the committed span, so garden-path prose that names wrong answers
    mid-reasoning doesn't fool it.

26. **Per-example answer-key graders.** `checkers.py` implements the answer-key `match.type`s:
    `exact_choice, numeric, numeric_or_word, exact_term, role_map, string_contains, plan_rubric`
    (with contraction + number-word handling). Verifiable types route `reward_path=verifiable`;
    `role_map`/`plan_rubric` route `rubric`. *Why:* one binary keyword checker false-matches ("no"
    inside "cannot"); per-type grading of the FINAL ANSWER span calibrates to gold-passes-100% /
    wrong-scores-0.

27. **Split: train 1000+2000, hold out 3000 (flipped) entirely.** Plus a seeded shuffle so the
    in-distribution held slice is representative. *Why:* a 90/10 pool would let the model see flipped
    variations and memorize them; holding the flipped set out keeps the memorize-vs-reason test honest.

28. **Resume guard also checks `n_plan` (vocab size).** *Why:* the planner head is sized to the vocab;
    a checkpoint from a different vocab must not be loaded into a differently-sized model.
