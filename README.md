# Adaptive planner-head small agent (SFT → offline GRPO)

<p align="center">
  <a href="https://sinha-k-prat.github.io/small_fable/">
    <img src="assets/showcase.gif" alt="Animated showcase of the core ideas — click to explore the interactive version" width="760">
  </a>
</p>

> 🎬 **[Core ideas, animated (interactive) →](https://sinha-k-prat.github.io/small_fable/)** &nbsp;·&nbsp;
> ▶️ **[Train on Colab (GPU) → Hugging Face →](https://colab.research.google.com/github/sinha-k-prat/small_fable/blob/main/train_all_colab.ipynb)** &nbsp;·&nbsp;
> 🔎 **[Inference-only Colab →](https://colab.research.google.com/github/sinha-k-prat/small_fable/blob/main/inference_colab.ipynb)** &nbsp;·&nbsp;
> 🧩 **[Planning primitives →](PRIMITIVES.md)** &nbsp;·&nbsp;
> ⚖️ **[Design contrasts (rejected vs chosen) →](DESIGN_CONTRASTS.md)**

A small model trained to behave like an **adaptive agent that switches mode from response to
response**. A **planner head** emits a short plan (a sequence of primitives) per input; the shared
backbone (**executor**) then produces the answer **conditioned on that plan**. Two training stages:

1. **SFT** on gold `(instruction → plan, answer)` pairs.
2. **Offline GRPO** that improves executor + planner from the model's *own sampled* rollouts, scored
   by a verifiable programmatic checker (not NLL, not embedding-similarity).

The headline property is that the **plan is load-bearing**: the answer depends on the plan chosen.
We measure this directly with the **plan-vs-no-plan ablation gap**.

## Architecture (one model, two heads, LoRA everywhere)
- **Executor** = `Qwen/Qwen2.5-1.5B-Instruct` causal LM + **LoRA on all 7 projection matrices**
  (`q,k,v,o,gate,up,down`, r=16, α=32, dropout=0.05). One adapter shared by both paths.
- **Planner head** = a linear head over the backbone's last-layer hidden states emitting **plan
  primitive** logits over a **separate** vocabulary (`PLAN_VOCAB`, 41 primitives). It is
  autoregressive: each chosen primitive is re-fed as a learned **plan embedding**.
- **Plan embeddings** = the chosen plan is embedded and prepended as a **soft prefix** (vectors in
  hidden space) so the executor is conditioned on the plan before it writes the answer.
- **Two policies, kept separate everywhere**: planner (plan vocab) and executor (token vocab) have
  **independent logprobs and independent clipped RL objectives**. They are never mixed.

Padding convention: prompts are **left-padded** so every "predict the next k things" is a clean
`[:, -k:]` slice (RoPE is relative, so the constant left shift is harmless). See `model_joint.py`.

## File map
| File | Role |
|---|---|
| `model_joint.py` | Backbone+LoRA, planner head, plan embeddings, all forward/rollout/scoring passes, frozen-backbone guard. |
| `train_sft.py` | Stage 1: joint `CE(plan) + lam_resp·CE(answer) + lam_kl·KL(executor‖base)`. Logs the **ablation gap**. |
| `rollout_offline.py` | Stage 2a: sample G=8 hot rollouts/instruction, score with checker, cache `*_logp_old`. Generate ONCE. |
| `train_grpo_offline.py` | Stage 2b: off-policy clipped GRPO from the cached file (no generation). |
| `grpo_offpolicy.py` | Tested off-policy objective: `group_advantages`, `clipped_pg_loss`, `joint_grpo_loss`. |
| `checkers.py` | Verifiable reward functions (`make_checker`, `reward_for_row`). |
| `compare.py` | 3-way base vs SFT vs SFT+RL: correctness, ablation gap, reward, zero-var fraction, adapter L2, plan-dist diff. |
| `reward_paths.py` | A1b routing: family → `verifiable`/`rubric`/`judge`; per-family rubric checklists. |
| `annotate_reward_paths.py` | Adds `reward_path` to dataset rows; converts soft families to graded rubric checkers. |
| `self_distill.py` | A7 (optional): rejection-sample correct traces, rank by learning-potential, emit a distill set. |
| `train_all.ipynb` | **All-in-one notebook**: SFT → rollouts → GRPO → compare SFT vs SFT+RL on 1000 instructions. |
| `important_design_choices.md` | Every non-obvious decision with its reason (read this). |
| `flatten_to_sft.py` | Rebuild `sft_flat.jsonl` from rich trajectory files. |
| `build_sensitivity_corpus.py` | Optional: build a **planning-sensitive** corpus (`q(plan) − q(no-plan) ≥ τ`). |
| `dataset/` | `sft_100.jsonl` (100 rows, default), `sft_flat.jsonl` (1000 rows, to scale up). Both carry `reward_path`. |
| `configs/` | Reference YAML mirroring each script's CLI defaults. |
| `tests/` | `test_grpo_offpolicy.py`, `test_checkers.py`, `test_curriculum.py`, `test_pipeline_smoke.py` (tiny-Qwen2 end-to-end, CPU/no-network). |

## Install
```bash
pip install -r requirements.txt
```

## Run order
```bash
# Stage 1: SFT
python train_sft.py --data dataset/sft_100.jsonl --train 70 --held 30 --epochs 6 --device cuda
# -> joint_ckpt/  (LoRA adapter + planner head + plan embeddings + tokenizer)

# Stage 2a: generate offline rollouts ONCE (slow part)
python rollout_offline.py --sft_ckpt joint_ckpt --data dataset/sft_100.jsonl \
  --train 80 --group 8 --temp 1.5 --max_resp 64 --out rl_rollouts.jsonl --device cuda
# -> rl_rollouts.jsonl  (THE offline RL dataset: sampled answers + checker rewards + logp_old)

# Stage 2b: GRPO from disk (fast; 2-3 passes then REGENERATE)
python train_grpo_offline.py --rollouts rl_rollouts.jsonl --sft_ckpt joint_ckpt \
  --out rl_ckpt --inner_epochs 3 --lr 1e-4 --clip_eps 0.2 --beta_plan 1.0 --beta_ce 0.1 --device cuda
# expect at startup: "[grpo] >0 trainable backbone tensors: NNN (≈336 expected)"

# Compare (ALWAYS pass --sample; greedy hides RL effects)
python compare.py --sft_ckpt joint_ckpt --rl_ckpt rl_ckpt --sample --device cuda
```
Swap the base model anytime with `--base Qwen/Qwen3-1.7B` (all scripts honor `--base`).

## How RL stays correct offline (the crux)
We generate rollouts once and reuse them for `inner_epochs` gradient steps. After step 1 the policy
`π_new` drifts from the saved behavior policy `π_old`, so each token's gradient is corrected by a
**PPO-clipped importance ratio**:
```
ratio  = exp(logp_new − logp_old)              # per token
L_clip = − mean( min( ratio·adv, clip(ratio, 1−ε, 1+ε)·adv ) )   # ε = 0.2
```
The ratio **corrects sampling staleness** — it is not a reward bonus. `ratio>1` (new policy already
favors this, over-represented in the stale batch) is braked by the clip; `ratio<1` fades the push.
On the **first pass `ratio≈1`**, so it reduces to plain reinforce. Two architecture-specific points,
both handled in `grpo_offpolicy.joint_grpo_loss`:
1. **Executor-only ratio masking** — the executor ratio is over **response tokens only**; the
   response logp tensor never contains prompt or plan tokens, so plan logprobs can't contaminate it.
2. **Separate clipped objectives** — planner and executor are different action spaces:
   `L = L_exec_clip + beta_plan·L_plan_clip` with **independent** ratios and advantages.

**Reuse rule:** 2–3 passes, then STOP and regenerate. Watch `exec_approx_kl` per inner epoch; if it
climbs past ~0.10–0.15 the rollouts are stale — `train_grpo_offline.py` auto-cuts inner epochs
(`--kl_stop`). Cut inner epochs before cutting lr.

## All-in-one notebook
`train_all.ipynb` runs every stage and the final comparison in sequence. Set `SMOKE=True` in the
config cell to dry-run on a tiny slice first; `SMOKE=False` runs the full 1000-instruction pipeline.

## Addenda (A1–A8) — what changed beyond the base spec
- **A1 MGPO MaxEnt weighting** (`grpo_offpolicy.mgpo_weight`): replaces deleting zero-variance groups
  with a soft bell `exp(−γ·(p_q−0.5)²)` on group accuracy. `--gamma` (= `1/(2δ²)`) sets the width.
- **A1b graded routing** (`reward_paths.py`, `--exclude_rubric`): `verifiable`→binary+MGPO,
  `rubric`→graded fraction + `variance_weight`, `judge`→last resort. Soft families are NOT forced
  through a fake binary checker. `graded_pq`, `variance_weight`, `group_pq` in `grpo_offpolicy.py`.
- **A2 pre-RL filter** (`rollout_offline.py`): zero-spread groups marked `keep=False` →
  `pre_rl_filter_report.csv`; GRPO trains only on kept groups.
- **A3 two-stage curriculum SFT** (`train_sft.py --curriculum`): cosine LR + warmup, broad stage 1 →
  hard-subset stage 2 (selected by the stage-1 model, error-rate ≥ 0.75). Logs ablation gap per stage.
  Batches are ordered **easy → hard** (`curriculum_batches`).
- **A4 spectrum-to-signal**: multi-path SFT via `alternatives`; post-SFT **probe diversity + Pass@k**.
- **A5 long2short** (`train_grpo_offline.py --long2short`): zero-sum brevity shaping among correct.
- **A6 logp-mismatch guard**: `logp_old` recomputed with the trainer's own teacher-forced path;
  GRPO asserts `logp_mismatch_t0 ≈ 0` before any update.
- **A7 self-distillation** (`self_distill.py`, optional) and **A8 CLR** (`compare.py --clr`, optional).

See `important_design_choices.md` for the reasoning behind each.

## Guarded pitfalls (these broke previous runs)
1. **Frozen-backbone no-op.** `PeftModel.from_pretrained` loads adapters frozen. RL loads with
   `is_trainable=True` and **asserts >0 trainable backbone tensors** at startup (prints the count).
   If it were 0, SFT and SFT+RL would be byte-identical.
2. **Flat reward / zero-variance groups.** Mitigated with G=8, hot temp (1.5), and (optionally) a
   planning-sensitive corpus. Both rollout and GRPO log the **zero-variance-group fraction**; keep it low.
3. **Greedy hides changes.** `compare.py` requires `--sample` (temp 0.7).
4. **Prove RL ≠ SFT.** `compare.py` reports the **adapter L2 diff** (`|RL − SFT|`, must be >0) and a
   **before/after plan-token distribution** so you can see whether RL found *different plans* or only
   reweighted the executor.

## Acceptance checks
- **SFT**: held `plan_ce` drops clearly; `ablation_gap` (plan vs no-plan) is **positive**.
- **RL startup**: asserts >0 trainable backbone tensors.
- **RL**: `zero_var_frac` low and `held_reward` **moves** across inner epochs (not flat within noise).
- **compare**: SFT+RL ≠ SFT under sampling; adapter L2 diff > 0.
- **Unit test**: `python tests/test_grpo_offpolicy.py` passes (on-policy ratio≈1/no clip; off-policy
  engages clipping; zero-variance → ~0 advantage; exec objective independent of plan branch).
- **A6**: `logp_mismatch_t0 ≈ 0` at GRPO startup (offline IS ratio valid at step 0).
- **A1**: `p_q` histogram logged; mean MaxEnt weight on near-saturated prompts < near-0.5 prompts.
- **A3**: stage-2 SFT ablation gap ≥ stage-1 gap (curriculum made plans more load-bearing).
- **A4**: post-SFT probe shows > 1 distinct plan per prompt (predicts RL variance).
- **A5**: with `--long2short`, correct-answer length drops while accuracy holds (within noise).

## Tests
```bash
python tests/test_grpo_offpolicy.py     # the GRPO math
python tests/test_checkers.py           # verifiable rewards
python tests/test_pipeline_smoke.py     # full two-head wiring on a tiny Qwen2 (CPU, no download)
# or: pytest tests/
```

## Data
`dataset/sft_100.jsonl` (default, 100 rows) and `dataset/sft_flat.jsonl` (1000 rows, to scale up).
Schema:
```json
{"id":"ex_0000","category":"arithmetic","instruction":"...",
 "plan":["GENERATE_ALT","EVAL","SELECT","VERIFY_LOGIC","SIMULATE","CORRECT","TERMINATE"],
 "answer":"$3450.","checker_kind":"exact","checker_args":{"gold":"3450."}}
```
**The bundled data is synthetic** (template-generated) — correct for proving the pipeline and the
ablation gap end-to-end. Before drawing quality conclusions, replace with real tasks in the same
schema or expand via `build_sensitivity_corpus.py --base Qwen/Qwen2.5-1.5B-Instruct` so tasks are
filtered for planning sensitivity (`P(correct|plan) ≫ P(correct|no plan)`), guaranteeing SFT
headroom and GRPO group variance.

## Notes on this build environment
This repo was scaffolded and verified on a CPU-only box (no CUDA/MPS). The offline unit tests and a
**full tiny-model end-to-end smoke test** (`tests/test_pipeline_smoke.py`, randomly-initialized
Qwen2, no network) all pass, validating the exact tensor wiring used by the real run. Stages 1–2
themselves need a GPU and will download `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face on first run.
```bash
pip install -r requirements.txt   # transformers>=4.51 recommended
```
