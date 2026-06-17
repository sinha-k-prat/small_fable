# Canonical Answer Keys for RL Reward

Three answer-key files, one per trace set, with **one canonical machine-readable
answer per instruction**. Keyed by the exact `instruction` string so you can join
to the trace files (1:1, verified).

- `answers_1000.jsonl`  ↔  `hard_reasoning_traces_1000.jsonl`
- `answers_2000.jsonl`  ↔  `hard_reasoning_traces_2000.jsonl`
- `answers_3000.jsonl`  ↔  `hard_reasoning_traces_3000.jsonl`

## Line schema

```json
{
  "instruction": "<exact instruction, join key>",
  "answer_form": "yes_no|cannot_conclude|number_with_units|value|next_term|plan",
  "canonical":   <the ground-truth answer, shape depends on form/match>,
  "match":       {"type": "<grader>", ...},
  "family":      "<problem family>"
}
```

## How RL should grade (match.type)

Parse the model's FINALIZE output, then grade against `canonical` using `match.type`:

- **exact_choice** — normalize model answer to a label; reward 1 if it is in
  `match.accept`, else 0. Used for `yes_no` (`"yes"`/`"no"`) and
  `cannot_conclude` (accepts `cannot_conclude`, `unknown`, `indeterminate`,
  `underdetermined`, …).

- **numeric** — extract a number from the model output; reward 1 if
  `abs(model - canonical.value) <= match.tolerance` (tolerance 0.0 for integer
  answers, 0.01 for money/hours). If `match.accept_units` is present, optionally
  require the unit to match for full reward.

- **numeric_or_word** — same as numeric, but `match.accept` also lists the word
  form (e.g. `["12", 12]`), so "twelve" and "12" both score.

- **exact_term** — for `next_term`; compare the model's next term to
  `match.accept` (string compare, case-insensitive for letters).

- **role_map** — for knights-and-knaves; `canonical.roles` maps each named person
  to `"truth_teller"`/`"liar"`. Reward = fraction of names assigned correctly
  (or all-or-nothing if you prefer strict grading).

- **string_contains** — for inherently verbal answers (some false-premise traps).
  `match.key_phrase` holds the essential concept; reward 1 if the model's answer
  conveys it (substring or, better, a small entailment/keyword check). These are
  the minority (~7–9% per set) where no clean numeric/label answer exists.

- **plan_rubric** — for `plan` answers; `canonical.gold_summary` is the intended
  resolution. Grade on whether the model **resolves or correctly avoids the
  conflict** (for set 1000/2000) or **recognizes there is no conflict** (set
  3000), not on exact wording. Best graded by a lightweight judge or keyword
  rubric rather than string equality.

## Special canonical values

- `number_with_units` may be `"no_finite_answer"` (snail that never escapes,
  opposing-worker net ≤ 0). `match.accept` then includes
  `["no_finite_answer","never","infinite","cannot_conclude"]`.

## Integrity (verified at build time)

- 1:1 instruction alignment between each answer file and its trace file.
- Numeric answers for snail, bat-and-ball, work-rate, set-overlap, doubling,
  tripling, and river-crossing are **independently recomputed from the
  instruction text**, not parsed from prose, so they are trustworthy ground truth.
- Verbal-trap numbers (e.g. "twelve, not one") are extracted with negation-aware
  rules and spot-checked against the trace conclusion.

## Reward suggestion

```
reward = base_match_score
       - step_penalty * num_plan_steps          # discourage ceremony
       + form_bonus  if model FINALIZE form == answer_form
       + calibration_bonus  if confidence high and correct (penalize confident-wrong)
```
The `answer_form` field lets you additionally reward the model for committing the
answer in the **correct form** (e.g. `cannot_conclude` rather than a fabricated
yes/no on a non-transitive chain) — often the hardest behavior to learn.
```
