# HUNT_CHECKPOINT.md — resume point for the HARD_BENCH plan-block hunt

**Status as of this commit: green pool SEEDED + FROZEN. The hunt itself has NOT been run yet.**
Pick up here next session.

## Reminder phrase (say this to resume)
> **"Resume the HARD_BENCH plan-block hunt — read HUNT_CHECKPOINT.md."**
That's enough. I'll read this file and continue from "NEXT STEP" below.

---

## The task (unchanged)
Find GENERIC plan-blocks that solve each of the 20 HARD_BENCH categories on a FROZEN
Qwen2.5-1.5B-Instruct, in the format **meta → user → (plan → response)\* → finish**, and don't stop
until each category succeeds under the STRICT grader. Plan-blocks must be operator-generic: the
operation is deferred to the problem, never named in the plan (so the block "could work on another
instruction").

## What is DONE (this session)
1. **Scorecard.** Design/format succeeded; demonstration did NOT — 0/20 categories verified at 100%.
   Only recorded empirical result: `multidigit_multiplication` = 0.0 (partial products correct, final
   3-addend addition wrong). See `DISCOVERED_PROMPTS.md`.
2. **Leak critique (user).** The old genericness gate `generic_violations()` only checks PROBLEM-INSTANCE
   token overlap, so a task-lift like "Multiply the first operand by each part" passed it. A string/verb
   blocklist is too brittle (false pos/neg). Robust test = **LLM-as-judge discriminator, design-time
   only** (judge NOT needed at hunt/inference time once the pool is frozen).
3. **Audit (judge = Claude, each block read in isolation).** Recorded in `tools/leak_audit.py::AUDIT`:
   - `decompose_and_recombine` **v0** = LEAK ("Multiply"/"Add", spells long-mult; also WRONG for modexp).
     Its siblings v1/v2 are already operator-neutral -> v0 EXCLUDED from the green pool.
   - `digit_by_digit`, `calendar_step`, `track_running_extreme`, `first_token_collect` = DOMAIN-bound
     (single-task algorithms, no generic sibling). Kept as sole seeds but flagged for neutral rewrite.
   - All other skills = GENERIC (operation deferred to "the rule"/"match test"/"criterion").
4. **Green pool SEEDED + WIRED + FROZEN.** `tools/leak_audit.py::green_variants()` returns the allowed
   seed-variant indices. `interleaved_solver.variants_for_category()` now seeds ONLY from the green pool.
   Verified: every category has >=1 seed; the leaky Multiply block is gone from multiplication seeds.

## NEXT STEP (do this when resumed — needs GPU; user has no local compute right now)
1. **(design, can do locally) Operator-neutral REWRITES** for the 4 domain-bound skills — see
   `tools/leak_audit.py::REWRITE_TODO`. Add them as new green variants, re-audit (judge in isolation),
   mark `generic`. This is the only remaining design-time work; do it before the GPU run if time allows.
2. **(compute) RUN THE HUNT on GPU** (Colab/A100 — see `RUN_ON_A100.md`). Seeded from the green pool,
   strict grader, mutate-until-100%-or-budget:
   ```
   python interleaved_solver.py --data hard_bench.jsonl --base Qwen/Qwen2.5-1.5B-Instruct \
       --dtype bf16 --device cuda --limit 0 --out interleaved_out \
       --max_new_tokens 320 --step_tokens 320 --rounds 6
   ```
   Smoke first: add `--only multidigit_multiplication --limit 2` to confirm the loop emits a real graded
   number, then drop `--only/--limit` for the full sweep.
3. **Multiplication-specific fix** (known failure): also decompose the FINAL ADDITION (two addends at a
   time, explicit carries) — the one step that sank the near-miss. This is a generic mutation, already
   expressible via `_GENERIC_VERIFY_BLOCKS` / a finer-decomposition block.
4. **Record results** to `interleaved_out/results.json` + update `DISCOVERED_PROMPTS.md` with real
   per-category accuracy. THIS is the deliverable we still owe (currently zero verified categories).

## Key files
- `tools/leak_audit.py` — FROZEN audit verdicts + `green_variants()` (the green pool). NEW this session.
- `interleaved_solver.py` — the hunt (`solve_category` = search + generic mutate until 100%/budget); now
  seeds from the green pool.
- `tools/interleaved_blocks.py` — block library (SKILLS, CATEGORY_SKILL, `generic_violations`).
- `hard_bench.jsonl` — 20 categories x 10 rows, strict machine-verifiable.
- `hard_bench_run.py` — strict judge-free grader (reused verbatim by the solver).
- `DISCOVERED_PROMPTS.md` — only recorded result so far (multiplication near-miss).
