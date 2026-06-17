# Planning primitives

The **planner head** emits a short program — a sequence of *primitives* — drawn from a fixed,
separate action space called `PLAN_VOCAB`. This is **not** the executor's token vocabulary; it is a
small symbolic vocabulary of **41 primitives** (40 operations + `PAD`). The executor then writes its
answer *conditioned on the emitted plan*.

> **Source of truth:** [`model_joint.py`](model_joint.py) → `PLAN_VOCAB` (the list below is generated
> from it). To print it yourself: `python -c "from model_joint import PLAN_VOCAB; print(PLAN_VOCAB)"`.

**Conventions**
- **Parameterized ops collapse to their base primitive.** A plan step like `FILTER[even]` or
  `TOP_K[k=3]` is mapped to its base token via `split("[")`, so the planner head stays a fixed-size
  action space. (If a base token isn't in the vocab, it collapses to `PAD`.)
- **`PAD`** is the padding / no-op token. An all-`PAD` plan means *no conditioning* — this is exactly
  what the **plan-vs-no-plan ablation** decodes against.
- **`TERMINATE`** ends a plan; the autoregressive planner stops a sequence once it emits it.
- Plans are capped at `plan_max_len` steps (default **12**).

---

## The 41 primitives, by role

### Special tokens
| Primitive | Role |
|---|---|
| `PAD` | Padding / empty-plan no-op (index 0). Empty plan ⇒ no conditioning. |
| `TERMINATE` | End of plan. |

### Understand / frame the problem
| Primitive | Intended role |
|---|---|
| `EXTRACT` | Pull out the given quantities / facts. |
| `DECOMPOSE` | Break the problem into sub-parts. |
| `MODEL` | Set up a formal model (equations, relations). |
| `IDENTIFY_UNKNOWN` | Name what is being solved for. |
| `ORDER` | Sort / sequence the inputs or steps. |
| `FIND` | Locate a specific value or item. |
| `CLARIFY` | Resolve ambiguity in the instruction. |

### Generate candidates / explore
| Primitive | Intended role |
|---|---|
| `GENERATE` | Produce a candidate answer. |
| `GENERATE_ALT` | Produce **alternative** candidates (drives rollout diversity → RL variance). |
| `EXPLORE` | Search the solution space. |
| `DIVERGE` | Deliberately branch into different approaches. |
| `EXPAND` | Elaborate / expand a candidate. |

### Reason / compute
| Primitive | Intended role |
|---|---|
| `LINK` | Connect facts / relate sub-results. |
| `SIMULATE` | Step through the process / run it mentally. |
| `TRACE` | Walk through a chain of reasoning step by step. |
| `CALCULATE` | Do the arithmetic. |
| `PREDICT` | Infer a downstream value or outcome. |
| `COMPARE` | Compare candidates / quantities. |
| `WEIGH` | Weigh trade-offs across options. |

### Verify / check
| Primitive | Intended role |
|---|---|
| `VERIFY_LOGIC` | Check the reasoning is logically valid. |
| `VERIFY_CONSTRAINTS` | Check all stated constraints are satisfied. |
| `VERIFY_COMPLETENESS` | Check nothing required is missing. |
| `VERIFY_CONSISTENCY` | Check parts don't contradict each other. |
| `VERIFY_STEP` | Check an individual step. |
| `VERIFY_EVIDENCE` | Check the answer is supported by the givens. |
| `REFLECT` | Re-examine the approach. |
| `EVAL` | Score / evaluate candidate(s) before selecting. |

### Revise / fix
| Primitive | Intended role |
|---|---|
| `REFINE` | Improve a candidate. |
| `CORRECT` | Fix an identified error. |
| `REPAIR` | Repair a broken / invalid solution. |
| `SIMPLIFY` | Reduce to a simpler form. |
| `RESOLVE_CONFLICT` | Reconcile conflicting sub-results. |

### Combine / synthesize
| Primitive | Intended role |
|---|---|
| `MERGE` | Merge partial results. |
| `COMBINE` | Combine candidates into one answer. |
| `GENERALIZE` | Abstract to a general rule. |

### Control / meta
| Primitive | Intended role |
|---|---|
| `PLAN` | Lay out a plan of attack. |
| `PLAN_NEXT` | Decide the next step. |
| `SELECT` | Pick the best candidate (typically after `EVAL`). |
| `ADAPT` | Switch strategy mid-solution. |

---

## What the seed data actually exercises

The vocabulary is the planner's **full action space**; the seed SFT corpus exercises a subset (the
rest are reachable by the planner and can surface under RL exploration). Counts over
`dataset/sft_100.jsonl` + `dataset/sft_flat.jsonl`:

| Primitive | Count |
|---|---|
| `GENERATE_ALT` | 1363 |
| `EVAL` | 1363 |
| `SELECT` | 1100 |
| `TERMINATE` | 1100 |
| `CORRECT` | 970 |
| `VERIFY_LOGIC` | 930 |
| `SIMULATE` | 930 |
| `EXPAND` | 100 |
| `MERGE` | 100 |
| `VERIFY_CONSISTENCY` | 40 |
| `TRACE` | 40 |
| `VERIFY_CONSTRAINTS` | 30 |
| `RESOLVE_CONFLICT` | 30 |

A canonical "verify-and-correct" plan in the data looks like:

```
GENERATE_ALT → EVAL → SELECT → VERIFY_LOGIC → SIMULATE → CORRECT → TERMINATE
```
