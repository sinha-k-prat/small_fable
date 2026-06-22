#!/usr/bin/env python3
"""
gen_grounding_data.py — procedural data for the GROUNDING PROBE (grounding_test.py).

WHAT THIS PROBES (and why the plans are ENGLISH, not primitive tokens)
----------------------------------------------------------------------
The probe asks a single question of a FROZEN base instruct model (Qwen2.5-1.5B-Instruct,
NO training, NO LoRA, NO gradients): can it *ground* and *follow* an abstract MULTI-TURN plan
written in plain English? A custom primitive vocabulary (GENERATE_ALT, MUL, ...) would carry
untrained embeddings that mean nothing to a frozen model, so this corpus deliberately writes
every plan turn as an English imperative the model already understands ("Keep only the cafes
that are open on Sunday", "Among the survivors pick the cheapest", ...).

Mirrors the gen_synth_v4.py contract so the same checkers.reward_for_row grades everything:
each row is solved BY CONSTRUCTION (a tiny deterministic per-family solver consumes the sampled
problem + a plan and returns the answer), so BOTH the gold answer and the perturbed-NEGATIVE
answer are oracle-known and NO LLM judge is needed.

Per row we ship:
  * gold_plan  : a 2-5 turn ENGLISH plan; each turn is one reasoning step the executor performs.
  * gold_answer: solver(problem, gold_plan)            (short, unambiguous)
  * neg_plan   : the gold plan with EXACTLY ONE turn swapped for a plausible-but-wrong step.
  * neg_answer : solver(problem, neg_plan)             — asserted != gold_answer by construction.

grounding_test.py then decodes the frozen model THREE ways per row (gold plan / no plan / neg
plan) and checks: does the gold plan yield gold_answer (grounding works), and does the WRONG
plan drag the model to neg_answer (faithful FOLLOWING)?

Each family's solver is a small interpreter over a SHARED step language: a plan turn is a
(verb, arg) pair the solver applies to a running candidate set / value. The gold plan and the
neg plan are the SAME list of turns with one (verb,arg) replaced, so the neg is a true
single-turn perturbation that re-executes to a different answer — the language analogue of
gen_synth_v4's "perturb one step, re-execute, keep only if the answer changed".

TOPIC FAMILIES (all non-math-heavy; >=6 required, 8 implemented):
  constraint_select  which-to-buy: filter a catalog by feature constraints, then pick by a rule
  comparison_order   sort named items by an attribute, emit the full ordering
  set_ops            apply union/intersection/difference over named member lists
  transitive_logic   chain "X beat Y" / "X older than Y" facts to a single winner/extreme
  scheduling         order tasks under precedence ("X before Y"), emit first/last/the sequence
  categorize_rule    route an item to a bucket by a stated if/else categorization rule
  multi_hop_lookup   follow a chain of key->value maps to resolve a final value
  conditional_reco   pick a recommendation by walking an if/elif/else decision tree

Output row schema (consumed by grounding_test.py + checkers.reward_for_row):
  {id, topic, n_turns, problem,
   gold_plan:[str,...], gold_answer,
   neg_plan:[str,...],  neg_answer,
   checker_kind, checker_args}

stdlib-only, fully seeded.
"""
import argparse, collections, json, os, random


# =========================================================================== checker helpers
# Mirror gen_synth_v4: every row carries a checker_kind + checker_args understood by checkers.py.
# Final answers are SHORT and unambiguous so grading needs no judge:
#   * a single name / letter / yes-no / category word  -> "exact_choice" (word-boundary match in
#     the FINAL ANSWER span; underscores treated as spaces)
#   * a full ordering of a fixed item set ("c < a < b") -> "string_contains" (whitespace-insensitive
#     substring of the FINAL ANSWER span). Safe ONLY because every alternative answer is a full
#     permutation of the SAME item set, so no answer is a substring of another (asserted below).
def choice_checker(answer):
    return "exact_choice", {"match": {"accept": [str(answer)]}}

def order_checker(answer):
    return "string_contains", {"match": {"key_phrase": str(answer)}}


def _norm_ws(s):
    return "".join(str(s).split())


# =========================================================================== shared vocab
NAMES = ["Ava", "Ben", "Cleo", "Dane", "Esme", "Finn", "Gus", "Hana",
         "Ivo", "Jade", "Kai", "Lena", "Milo", "Nia", "Omar", "Priya"]
PLACES = ["Maple Cafe", "Birch Diner", "Cedar Bistro", "Aspen Grill",
          "Willow Bar", "Olive Deli", "Rowan Pub", "Hazel Bakery"]
PRODUCTS = ["Falcon", "Comet", "Nimbus", "Quartz", "Vega", "Onyx", "Coral", "Drift"]
CITIES = ["Aralu", "Borvik", "Calmar", "Dunfel", "Esport", "Fernby"]


def pick(rng, pool, k):
    return rng.sample(pool, k)


# =========================================================================== family solvers
# Each family returns: (topic, problem, gold_plan, gold_answer, neg_plan, neg_answer, checker).
# The solver is embedded: we build a plan as a list of (turn_text, op) where op is a closure that
# transforms the running state, then run gold ops -> gold_answer and neg ops -> neg_answer. The
# gold and neg plans differ in EXACTLY ONE turn (text + op). This guarantees the negative is a
# single-turn perturbation whose answer is computed, not typed.


def f_constraint_select(rng):
    """Catalog of products with boolean features + a price; gold filters then picks by a rule;
    neg flips one filter (or the pick rule) so a different product survives/wins."""
    prods = pick(rng, PRODUCTS, rng.randint(4, 5))
    feats = {p: {"waterproof": rng.random() < 0.5,
                 "wireless": rng.random() < 0.5,
                 "price": rng.randrange(20, 80, 5)} for p in prods}

    def solve(require_wp, require_wl, pick_cheapest):
        keep = [p for p in prods
                if (feats[p]["waterproof"] or not require_wp)
                and (feats[p]["wireless"] or not require_wl)]
        if not keep:
            return "none"
        keep.sort(key=lambda p: (feats[p]["price"], p))
        return keep[0] if pick_cheapest else keep[-1]

    lines = [f"- {p}: {'waterproof' if feats[p]['waterproof'] else 'not waterproof'}, "
             f"{'wireless' if feats[p]['wireless'] else 'wired'}, ${feats[p]['price']}"
             for p in prods]
    problem = ("You are choosing one gadget from this catalog:\n" + "\n".join(lines))

    gold_plan = ["Keep only the gadgets that are waterproof.",
                 "From those, keep only the ones that are wireless.",
                 "Among the remaining gadgets, choose the cheapest one."]
    gold = solve(True, True, True)

    # perturb ONE turn: choose the priciest instead of the cheapest.
    neg_plan = list(gold_plan)
    neg_plan[2] = "Among the remaining gadgets, choose the most expensive one."
    neg = solve(True, True, False)
    if neg == gold:                       # fall back: drop the wireless requirement instead
        neg_plan = list(gold_plan)
        neg_plan[1] = "From those, keep only the ones that are wired."
        neg = solve(True, False, True)
    return ("constraint_select", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold)) \
        if neg != gold else None


def f_comparison_order(rng):
    """Rank named items by a numeric attribute; gold sorts one way, neg reverses the direction."""
    items = pick(rng, PRODUCTS, rng.randint(3, 4))
    score = {it: rng.randint(1, 99) for it in items}
    while len(set(score.values())) < len(items):          # distinct -> unambiguous order
        score = {it: rng.randint(1, 99) for it in items}
    attr = rng.choice(["battery life (hours)", "weight (grams)", "review score"])
    problem = (f"Each model's {attr}:\n"
               + "\n".join(f"- {it}: {score[it]}" for it in items))

    def order(ascending):
        s = sorted(items, key=lambda it: score[it], reverse=not ascending)
        return " < ".join(s)

    gold_plan = [f"List the models from lowest {attr} to highest.",
                 "Write them in that order separated by '<'."]
    gold = order(ascending=True)
    neg_plan = list(gold_plan)
    neg_plan[0] = f"List the models from highest {attr} to lowest."
    neg = order(ascending=False)
    return ("comparison_order", problem, gold_plan, gold, neg_plan, neg, order_checker(gold))


def f_set_ops(rng):
    """Two named club rosters built so each of {in BOTH, only A, only B} has EXACTLY ONE distinct
    member; gold applies one of those three set ops, neg applies a different one, and the answer is
    the single resulting name. Sizes are forced by construction so the answer is always one name."""
    shared, onlyA, onlyB = pick(rng, NAMES, 3)        # one member per region, all distinct
    A = sorted([shared, onlyA])
    B = sorted([shared, onlyB])
    problem = (f"Club A members: {', '.join(A)}.\n"
               f"Club B members: {', '.join(B)}.")

    region = {"both": shared, "onlyA": onlyA, "onlyB": onlyB}
    phrase = {"both": "who are in BOTH Club A and Club B",
              "onlyA": "who are in Club A but NOT Club B",
              "onlyB": "who are in Club B but NOT Club A"}
    g_kind, n_kind = rng.sample(list(region), 2)      # gold op + a DIFFERENT neg op
    gold_plan = [f"Find the people {phrase[g_kind]}.",
                 "Report the single person you found."]
    gold = region[g_kind]
    neg_plan = list(gold_plan)
    neg_plan[0] = f"Find the people {phrase[n_kind]}."
    neg = region[n_kind]
    return ("set_ops", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold)) \
        if neg != gold else None


def f_transitive_logic(rng):
    """A chain of pairwise 'X beat Y' facts gives a total order; gold follows the chain to the
    overall winner, neg follows it to the overall loser."""
    players = pick(rng, NAMES, rng.randint(4, 5))
    rng.shuffle(players)                               # players[0] strongest ... [-1] weakest
    facts = [f"{players[i]} beat {players[i+1]}" for i in range(len(players) - 1)]
    rng.shuffle(facts)
    problem = ("In a tournament (beating is transitive):\n- " + "\n- ".join(facts))

    gold_plan = ["Chain the results to put everyone in order of strength.",
                 "Name the player who beats everyone (the overall winner)."]
    gold = players[0]
    neg_plan = list(gold_plan)
    neg_plan[1] = "Name the player who loses to everyone (the overall loser)."
    neg = players[-1]
    return ("transitive_logic", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold))


def f_scheduling(rng):
    """Tasks with 'X must come before Y' precedence forming a chain; gold reports the task that
    must be done FIRST, neg reports the one done LAST."""
    tasks = pick(rng, ["prep", "mix", "bake", "cool", "frost", "box"], rng.randint(3, 5))
    rng.shuffle(tasks)                                 # tasks[0] first ... tasks[-1] last
    rules = [f"'{tasks[i]}' must come before '{tasks[i+1]}'" for i in range(len(tasks) - 1)]
    rng.shuffle(rules)
    problem = ("Steps with ordering rules:\n- " + "\n- ".join(rules))

    gold_plan = ["Use the rules to put all the steps in a single valid order.",
                 "Report the step that must be done FIRST."]
    gold = tasks[0]
    neg_plan = list(gold_plan)
    neg_plan[1] = "Report the step that must be done LAST."
    neg = tasks[-1]
    return ("scheduling", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold))


_CAT_THEMES = [
    ("Air-quality", "reading", ("good", "fair", "poor")),
    ("Loan-risk", "score", ("low", "medium", "high")),
    ("Lake-level", "depth", ("shallow", "normal", "flooded")),
    ("Battery-health", "percent", ("worn", "okay", "fresh")),
    ("Spice-heat", "rating", ("mild", "medium", "hot")),
    ("Crowd-size", "count", ("quiet", "busy", "packed")),
]

def f_categorize_rule(rng):
    """A two-threshold if/elif/else rule routes a numeric reading to three named buckets; gold
    applies the stated rule, neg swaps the two thresholds so the reading routes to a different
    bucket. Theme + bucket words are randomized so distinct problems are plentiful."""
    metric, noun, (b_lo, b_mid, b_hi) = rng.choice(_CAT_THEMES)
    lo, hi = sorted(rng.sample(range(20, 80), 2))
    val = rng.randint(0, 100)
    while val == lo or val == hi:
        val = rng.randint(0, 100)
    problem = (f"{metric} rule: below {lo} is '{b_lo}', from {lo} to {hi} is '{b_mid}', "
               f"above {hi} is '{b_hi}'.\nToday's {noun} is {val}.")

    def classify(flip):
        # flip swaps the two boundaries used by the rule -> a perturbed categorization
        a, b = (hi, lo) if flip else (lo, hi)
        if val < a:   return b_lo
        if val <= b:  return b_mid
        return b_hi

    gold_plan = ["Compare the reading to the two thresholds in the rule.",
                 f"Report the matching category word ({b_lo}, {b_mid}, or {b_hi})."]
    gold = classify(flip=False)
    neg_plan = list(gold_plan)
    neg_plan[0] = "Compare the reading to the two thresholds, but swap which threshold is which."
    neg = classify(flip=True)
    if neg == gold:                                   # ensure the perturbation changes the bucket
        return None
    return ("categorize_rule", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold))


def f_multi_hop_lookup(rng):
    """Chained key->value maps (person->city, city->mascot); gold follows person->city->mascot,
    neg stops one hop short (or hops the wrong map) and reports the intermediate value."""
    ppl = pick(rng, NAMES, 4)
    cities = pick(rng, CITIES, 4)
    mascots = pick(rng, ["Owls", "Bears", "Hawks", "Foxes", "Wolves", "Rams"], 4)
    p2c = dict(zip(ppl, cities))
    c2m = dict(zip(cities, mascots))
    who = rng.choice(ppl)
    problem = ("Lives-in:\n- " + "\n- ".join(f"{p} lives in {p2c[p]}" for p in ppl)
               + "\nTeam mascots:\n- " + "\n- ".join(f"{c}'s team is the {c2m[c]}" for c in cities)
               + f"\nQuestion: what is {who}'s team mascot?")

    gold_plan = [f"Look up which city {who} lives in.",
                 "Look up that city's team mascot.",
                 "Report that mascot."]
    gold = c2m[p2c[who]]
    # perturb the middle hop: report the CITY instead of doing the mascot hop.
    neg_plan = list(gold_plan)
    neg_plan[1] = "Skip the mascot table and just use the city name itself."
    neg = p2c[who]
    return ("multi_hop_lookup", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold))


def f_conditional_reco(rng):
    """A 2x2 if/elif/else decision tree over two binary attributes recommends one item; gold walks
    the true branch, neg negates the FIRST condition and lands on a sibling item. The two axes and
    the four leaf items are randomized so distinct problems are plentiful."""
    # (axisA name, optionsA), (axisB name, optionsB), and four DISTINCT leaf labels for the grid.
    axisA, optsA = rng.choice([("temperature", ("hot", "cold")),
                               ("budget", ("cheap", "premium")),
                               ("season", ("summer", "winter"))])
    axisB, optsB = rng.choice([("taste", ("sweet", "plain")),
                               ("size", ("small", "large")),
                               ("mood", ("calm", "lively"))])
    leaves = pick(rng, ["Cocoa", "Tea", "Lemonade", "Water", "Cider", "Juice",
                        "Mocha", "Soda", "Punch", "Latte"], 4)
    a = rng.choice(optsA); b = rng.choice(optsB)
    grid = {(optsA[0], optsB[0]): leaves[0], (optsA[0], optsB[1]): leaves[1],
            (optsA[1], optsB[0]): leaves[2], (optsA[1], optsB[1]): leaves[3]}
    rules = "\n".join(f"- If {x} and {y} -> {grid[(x, y)]}" for x in optsA for y in optsB)
    problem = (f"Picker (by {axisA} and {axisB}):\n{rules}\n"
               f"The guest is {a} and {b}.")

    gold_plan = [f"Check the {axisA}: is it {optsA[0]} or {optsA[1]}?",
                 f"Check the {axisB}: is it {optsB[0]} or {optsB[1]}?",
                 "Follow the matching branch and report the item."]
    gold = grid[(a, b)]
    # perturb the FIRST check: read axisA as the opposite option.
    flip_a = optsA[0] if a == optsA[1] else optsA[1]
    neg_plan = list(gold_plan)
    neg_plan[0] = f"Check the {axisA}, but read it as the opposite of what the guest is."
    neg = grid[(flip_a, b)]
    return ("conditional_reco", problem, gold_plan, gold, neg_plan, neg, choice_checker(gold))


FAMILIES = [f_constraint_select, f_comparison_order, f_set_ops, f_transitive_logic,
            f_scheduling, f_categorize_rule, f_multi_hop_lookup, f_conditional_reco]


# =========================================================================== row assembly
# Benign, answer-NEUTRAL setup turns used only to pad a short natural plan up to the target
# turn-count. They read as ordinary "orient yourself" steps and, crucially, are IDENTICAL in the
# gold and neg plans, so padding never moves or duplicates the single perturbed turn nor changes
# any answer. They are prepended (before the perturbed turn) in a fixed order for determinism.
_NOOPS = ["Read the problem statement carefully.",
          "List out the items the problem gives you.",
          "Note what the question is asking for."]

def trim_to_turns(gold_plan, neg_plan, n_turns):
    """Pad both plans to `n_turns` (in [2,5]) with identical benign setup turns, WITHOUT moving the
    single perturbed turn. Natural plan length is always 2 or 3, so we only ever pad (never trim)."""
    g, n = list(gold_plan), list(neg_plan)
    pad = 0
    while len(g) < n_turns:                            # prepend so the perturbed turn stays aligned
        g.insert(0, _NOOPS[pad % len(_NOOPS)]); n.insert(0, _NOOPS[pad % len(_NOOPS)]); pad += 1
    return g, n


def build_row(rng, idx, target_turns, fam):
    """Sample ONE valid row from family `fam` (retrying within the family up to a small budget,
    since some families reject a sample when the perturbation happens not to change the answer)."""
    res = None
    for _ in range(60):
        res = fam(rng)
        if res is not None:
            break
    if res is None:
        return None
    topic, problem, gold_plan, gold, neg_plan, neg, (ckind, cargs) = res
    if str(gold) == str(neg):                         # hard guarantee: negative changes the answer
        return None

    # the family MUST emit a single-turn perturbation (same length, exactly one differing turn)
    diffs = [i for i in range(len(gold_plan)) if gold_plan[i] != neg_plan[i]]
    if len(diffs) != 1 or len(gold_plan) != len(neg_plan):
        return None
    gold_plan, neg_plan = trim_to_turns(gold_plan, neg_plan, target_turns)
    if not (2 <= len(gold_plan) <= 5):
        return None

    return {
        "id": f"grnd_{idx:04d}",
        "topic": topic,
        "n_turns": len(gold_plan),
        "problem": problem,
        "gold_plan": gold_plan,
        "gold_answer": str(gold),
        "neg_plan": neg_plan,
        "neg_answer": str(neg),
        "checker_kind": ckind,
        "checker_args": cargs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="grounding_test_data.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows, per_topic, per_turns = [], collections.Counter(), collections.Counter()
    seen = set()
    idx = 0
    # ROUND-ROBIN both the family and the target turn-count so the 100 rows are balanced across all
    # 8 topics and across turn-counts 2..5 (rather than skewed by per-family rejection rates). Cycle
    # by `tick` (always advances, even on a rejected/duplicate sample) so a family that dups does not
    # trap the loop; turn-count (len 4) vs families (len 8) are co-prime-strided so each family sees
    # all of 2..5 across its appearances rather than a single fixed turn-count.
    turn_cycle = [2, 3, 4, 5]
    tick = 0
    while len(rows) < args.n and tick < args.n * 200:
        fam = FAMILIES[tick % len(FAMILIES)]
        target = turn_cycle[(tick + tick // len(FAMILIES)) % len(turn_cycle)]
        tick += 1
        row = build_row(rng, idx, target, fam)
        if row is None:
            continue
        key = row["problem"]
        if key in seen:                               # dedupe identical sampled problems
            continue

        # ---- ORACLE assertions (mirror gen_synth_v4's verify-by-construction discipline) ----
        assert row["gold_answer"] != row["neg_answer"], "negative did not change the answer"
        assert 2 <= row["n_turns"] <= 5
        assert len(row["gold_plan"]) == len(row["neg_plan"])
        # exactly one perturbed turn
        d = sum(1 for a, b in zip(row["gold_plan"], row["neg_plan"]) if a != b)
        assert d == 1, f"expected one perturbed turn, got {d}"
        # for orderings: gold/neg are full permutations of one item set, so neither is a
        # whitespace-insensitive substring of the other (else string_contains could misgrade).
        if row["checker_kind"] == "string_contains":
            g, nn = _norm_ws(row["gold_answer"]), _norm_ws(row["neg_answer"])
            assert g not in nn and nn not in g, "ordering answers substring-collide"

        seen.add(key)
        rows.append(row)
        per_topic[row["topic"]] += 1
        per_turns[row["n_turns"]] += 1
        idx += 1

    if len(rows) < args.n:
        raise SystemExit(f"only generated {len(rows)}/{args.n} rows; raise attempt budget or pools")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} rows -> {args.out}")
    print("per-topic :", dict(sorted(per_topic.items())))
    print("per-turns :", dict(sorted(per_turns.items())))


if __name__ == "__main__":
    main()