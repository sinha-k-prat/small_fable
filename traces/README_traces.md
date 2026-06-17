# Hard Reasoning Traces - 3000 examples (3 x 1000)

Training data for a small LLM acting as a primitive-modular reasoning engine.
Three sets of 1000 traces each. Every example is hard for a non-thinking model,
and the three sets are deliberately related so that a model which MEMORIZED one
set must still REASON to solve the others.

## The three sets

- **hard_reasoning_traces_1000.jsonl** - base set. Classic traps: non-transitive
  chaining, off-by-one simulation, garden-path arithmetic, false premises,
  knights-and-knaves, logical fallacies, set-overlap, correlation-vs-cause,
  hidden constraint conflicts, ambiguity.
- **hard_reasoning_traces_2000.jsonl** - 'one-step-deeper' variations. The surface
  looks familiar but one reasoning hinge is perturbed so an EXTRA check or reversed
  direction is needed: mixed transitive+non-transitive chains, snail that never
  escapes (slip >= climb), bat-and-ball asking for the expensive item, 'all but N
  survive' (inverted phrasing), opposing-worker rates, quarter-coverage doubling,
  'how many like neither', three-speaker knaves, genuine-but-unsuitable financial
  options, contrapositive recognition.
- **hard_reasoning_traces_3000.jsonl** - 'flipped-answer' variations. A small change
  flips the conclusion: genuinely transitive chains (answer YES), syllogisms flipped
  valid<->invalid by all-vs-some, conditionals flipped ponens<->affirming-consequent,
  constraint problems with NO hidden conflict (no backtrack), three-worker rates,
  tripling instead of doubling, knaves that are underdetermined (cannot_conclude),
  questions that look like tricks but are literal, arithmetic-vs-geometric pattern
  discrimination, and cases where causation IS justified (mechanism/RCT).

Why this matters: in the relation family, set 1 answers 'cannot_conclude' (relation
not transitive), set 2 also 'cannot_conclude' but for a different reason (mixed
relations), and set 3 flips to 'yes' (relation genuinely transitive). Same chain
shape, three different required reasonings and two different answers.

## Schema (JSONL, one object per line)

    {
      "instruction": "<the user query>",
      "trace":       "<full reasoning trace, ending in: accepted>",
      "family":      "<problem family tag>",
      "answer_form": "<value|yes_no|cannot_conclude|number_with_units|next_term|plan>"
    }

## Trace format

- Turns of 1-3 parameterized primitives, bundled only when non-entangling.
- Parameters use commas inside [...]; ' ; ' separates primitives within a turn.
- Responses are plain prose executing the turn's primitives in order (no indexing).
- FINALIZE commits the answer in the matching form; the trace ends with: accepted

## Integrity (verified)

- 3000 examples total; 1000 unique instructions per set; ~0 cross-set instruction overlap.
- All traces end with 'accepted'; no turn exceeds 3 primitives.
- Numeric traces are instance-correct (each snail/rate/price computed for that item).
- Variation sets share family structure with set 1 but require different reasoning
  and frequently a different answer, so they test reasoning rather than recall.

## Set 1000 - family distribution

-  104  sim_offbyone
-   86  relation_nontransitive
-   85  garden_path_arith
-   78  false_premise
-   64  relation_transitive
-   62  constraint_conflict
-   61  knaves
-   60  causal_confound
-   59  sensitive_domain
-   54  work_rate
-   52  ambiguity_clarify
-   49  sim_doubling
-   37  set_overlap
-   25  garden_path_rate
-   21  river_crossing
-   20  pattern_initials
-   18  knaves_paradox
-   17  pattern_lookandsay
-    8  modus_ponens
-    8  affirm_consequent
-    8  deny_antecedent
-    8  modus_tollens
-    6  syllogism_invalid
-    6  syllogism_valid
-    4  pattern_quadratic

Set 1000 answer forms: number_with_units=338, cannot_conclude=269, value=176, yes_no=114, plan=62, next_term=41

## Set 2000 - family distribution

-  120  contrapositive_valid
-  110  relation_mixed_chain
-   95  garden_path_expensive
-   95  sim_no_escape
-   95  work_opposing
-   90  phrasing_inverted
-   85  false_premise_v2
-   80  set_neither
-   80  knaves_three
-   75  sensitive_suitability
-   75  doubling_quarter

Set 2000 answer forms: number_with_units=324, cannot_conclude=306, value=250, yes_no=120

## Set 3000 - family distribution

-  110  relation_transitive_yes
-   95  sim_offbyone_v
-   84  work_three
-   80  trick_but_literal
-   80  knaves_underdetermined
-   80  tripling
-   75  causal_genuine
-   68  conditional_flip_ponens
-   67  conditional_flip_consequent
-   58  syllogism_flip_valid
-   57  constraint_no_conflict
-   56  syllogism_flip_invalid
-   45  pattern_arithmetic
-   45  pattern_geometric

Set 3000 answer forms: yes_no=311, number_with_units=259, cannot_conclude=203, next_term=90, value=80, plan=57
