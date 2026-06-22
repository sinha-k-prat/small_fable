#!/usr/bin/env python3
"""
gen_grounding_blocks.py — v2 data for the GROUNDING PROBE (grounding_test.py).

v1 (tools/gen_grounding_data.py) proved EXECUTION grounding: a FROZEN Qwen2.5-1.5B-Instruct follows
CONCRETE English plans ("Keep only the gadgets that are waterproof"). But those plan turns are
MARRIED to the problem's own words, so they could never become a REUSABLE universal-primitive vocab.

v2 tests ABSTRACTION grounding. Two changes:
  (1) Problem-specific CRITERIA move OUT of the plan and INTO the PROBLEM as a NUMBERED named list
      (Requirements: 1) waterproof; 2) wireless; Preference: cheapest). Each plan block uses ONLY
      universal-primitive verbs + POSITIONAL references ("Keep only the items that satisfy the 1st
      stated requirement.") and NEVER names a domain content-word.
  (2) A CLUBBING axis (--club {1,2}): club=1 = one op per block; club=2 = two ops bundled per block.
      The clubbed plan executes to the SAME gold answer (asserted via the solver); the negative
      perturbs EXACTLY ONE op in whichever block it lands -> a different solver-computed answer, in
      BOTH club levels.

INFERENCE-ONLY on a FROZEN model. No training/LoRA/gradients. Plans stay ENGLISH. Judge-free: we
REUSE the v1 per-family solvers so gold + neg answers are known BY CONSTRUCTION. Emits
grounding_blocks_c1.jsonl and grounding_blocks_c2.jsonl from the SAME seed with the SAME
problems/answers, differing ONLY in block phrasing/clubbing.

Row schema (consumed UNCHANGED by grounding_test.py + checkers.reward_for_row), plus 'club':
  {id, topic, n_turns, problem, gold_plan:[block strings], gold_answer,
   neg_plan:[block strings], neg_answer, checker_kind, checker_args, club}
  NB: n_turns == number of BLOCKS (club=2 rows report fewer turns than club=1).

stdlib-only, fully seeded.
"""
import argparse, collections, json, os, random, re


# =========================================================================== checker helpers (v1)
def choice_checker(answer):
    return "exact_choice", {"match": {"accept": [str(answer)]}}

def order_checker(answer):
    return "string_contains", {"match": {"key_phrase": str(answer)}}

def _norm_ws(s):
    return "".join(str(s).split())


# =========================================================================== shared vocab (v1)
NAMES = ["Ava", "Ben", "Cleo", "Dane", "Esme", "Finn", "Gus", "Hana",
         "Ivo", "Jade", "Kai", "Lena", "Milo", "Nia", "Omar", "Priya"]
PRODUCTS = ["Falcon", "Comet", "Nimbus", "Quartz", "Vega", "Onyx", "Coral", "Drift"]
CITIES = ["Aralu", "Borvik", "Calmar", "Dunfel", "Esport", "Fernby"]


def pick(rng, pool, k):
    return rng.sample(pool, k)


# =========================================================================== GENERIC PLAN VOCAB
# Every plan block is built ONLY from STRUCT_OK + positional/ordinal references. NO block may contain
# a domain content-word. The genericness assertion is enforced against STRUCT_OK (whitelist): any
# alphabetic token in a block not in STRUCT_OK (and not a bare ordinal/number) is a domain leak.
# Positional references ("the 1st stated requirement", "the stated preference", "the 2nd stated
# mapping", ...) name a SLOT in the problem, never its contents.
STRUCT_OK = {
    "the", "a", "an", "of", "to", "is", "are", "in", "into", "by", "and", "or", "then", "it",
    "that", "those", "them", "this", "these", "with", "as", "on", "at", "for", "from", "out",
    "their", "its", "each", "all", "both", "only", "single", "one", "two", "no", "not",
    "1st", "2nd", "3rd", "first", "second", "third", "next", "last", "final",
    "keep", "remove", "discard", "filter", "select", "choose", "pick", "report", "state",
    "name", "find", "identify", "apply", "follow", "use", "look", "up", "resolve", "chain",
    "order", "arrange", "sort", "list", "write", "place", "put", "build", "produce", "emit",
    "separate", "separated", "join", "joined", "combine", "take", "read", "begin", "start",
    "requirement", "requirements", "preference", "attribute", "relation", "rule", "rules",
    "mapping", "condition", "conditions", "value", "values", "item", "items", "member", "members",
    "step", "steps", "element", "elements", "entry", "entries", "people", "person", "match",
    "matching", "satisfy", "satisfies", "satisfying", "remaining", "result", "results", "set",
    "stated", "given", "asked", "question", "problem", "statement", "data", "above", "below",
    "ascending", "descending", "extreme", "overall", "between", "intersection", "leftover",
    "survivors", "survivor", "leads", "leads-to", "outcome", "branch", "branches", "table",
    "tables", "chain", "thresholds", "threshold", "range", "ranges", "category", "categories",
    "position", "positions", "ordering", "sequence", "after", "before", "among", "carefully",
    "what", "you", "your", "yourself", "orient", "noting", "note", "which", "whose", "where",
    "exactly", "way", "direction", "opposite", "swap", "swapped", "wrong", "different",
    "symbol", "symbols",
    "end", "ends", "target", "through", "lists", "found", "everyone", "valid", "key", "keys",
    "fail", "fails", "than", "across", "facts", "fact", "skip", "unchanged", "get", "falls",
    "written", "but", "asking", "than", "states",
}
STRUCT_OK |= {"4th", "5th", "fourth", "fifth"}


def _generic_violations(block):
    """tokens in a generic block that are NOT in STRUCT_OK (i.e. potential domain leaks).
    Possessives are stripped to the stem ("rule's" -> "rule") so apostrophe-s isn't a stray 's'."""
    text = re.sub(r"'s\b", "", str(block).lower())
    bad = []
    for tok in re.sub(r"[^a-z0-9 ]", " ", text).split():
        if tok in STRUCT_OK:
            continue
        if re.fullmatch(r"\d+(st|nd|rd|th)?", tok):
            continue
        bad.append(tok)
    return bad


# =========================================================================== family solvers (v2)
# Each family returns a SPEC dict: problem + an ORDERED list of OPERATIONS, each op =
# {gold: block_str, neg: block_str|None, apply: fn(state)->state, neg_apply: fn(state)->state}.
# Exactly ONE op has neg != None (the perturbed op). We compose gold ops -> gold answer, neg-ops
# (one op swapped) -> neg answer. The OP LIST is the single source of truth; clubbing only changes
# how ops are GROUPED into block strings (via club2_groups + club2_block), never the ops, so the
# answer is invariant to --club.
def _build_answer(spec, perturb):
    state = spec["init"]
    for op in spec["ops"]:
        fn = op["neg_apply"] if (perturb and op["neg"] is not None) else op["apply"]
        state = fn(state)
    return spec["finalize"](state)


def f_constraint_select(rng):
    prods = pick(rng, PRODUCTS, rng.randint(5, 6))
    feats = {p: {"waterproof": rng.random() < 0.5, "wireless": rng.random() < 0.5,
                 "price": rng.randrange(20, 80, 5)} for p in prods}
    lines = [f"- {p}: {'waterproof' if feats[p]['waterproof'] else 'not waterproof'}, "
             f"{'wireless' if feats[p]['wireless'] else 'wired'}, ${feats[p]['price']}" for p in prods]
    problem = ("You are choosing one gadget from this catalog:\n" + "\n".join(lines) +
               "\n\nRequirements (apply in this order):\n  1) waterproof\n  2) wireless\n"
               "Preference (tie-break / final pick): cheapest.")

    def keep_req1(s): return [p for p in s if feats[p]["waterproof"]]
    def keep_req1_neg(s): return [p for p in s if not feats[p]["waterproof"]]
    def keep_req2(s): return [p for p in s if feats[p]["wireless"]]
    def select_pref(s):
        return "none" if not s else sorted(s, key=lambda p: (feats[p]["price"], p))[0]

    ops = [
        {"gold": "Keep only the items that satisfy the 1st stated requirement.",
         "neg": "Keep only the items that fail the 1st stated requirement.",
         "apply": keep_req1, "neg_apply": keep_req1_neg},
        {"gold": "From those, keep only the items that satisfy the 2nd stated requirement.",
         "neg": None, "apply": keep_req2, "neg_apply": keep_req2},
        {"gold": "Among the remaining items, select the one matching the stated preference.",
         "neg": None, "apply": select_pref, "neg_apply": select_pref},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Keep only the items that FAIL the 1st stated requirement, then from those "
                        "keep only the items that satisfy the 2nd stated requirement.")
            return ("Keep only the items that satisfy the 1st stated requirement, then from those "
                    "keep only the items that satisfy the 2nd stated requirement.")
        return "Among the remaining items, select the one matching the stated preference."

    spec = {"topic": "constraint_select", "problem": problem, "ops": ops,
            "init": list(prods), "finalize": lambda x: str(x),
            "club2_groups": [(0, 1), (2,)], "club2_block": club2_block}
    spec["checker"] = choice_checker(_build_answer(spec, perturb=False))
    return spec


def f_comparison_order(rng):
    items = pick(rng, PRODUCTS, rng.randint(3, 4))
    score = {it: rng.randint(1, 99) for it in items}
    while len(set(score.values())) < len(items):
        score = {it: rng.randint(1, 99) for it in items}
    attr = rng.choice(["battery life (hours)", "weight (grams)", "review score"])
    problem = (f"Each model's stated attribute is its {attr}:\n"
               + "\n".join(f"- {it}: {score[it]}" for it in items)
               + "\n\nThe stated order to produce: ASCENDING (lowest attribute value first).")

    def sort_asc(s): return sorted(s, key=lambda it: score[it])
    def sort_desc(s): return sorted(s, key=lambda it: score[it], reverse=True)
    def emit(s): return " < ".join(s)

    ops = [
        {"gold": "Arrange the items by the stated attribute in the stated order.",
         "neg": "Arrange the items by the stated attribute in the opposite of the stated order.",
         "apply": sort_asc, "neg_apply": sort_desc},
        {"gold": "Write them in that order, separated by the '<' symbol.",
         "neg": None, "apply": emit, "neg_apply": emit},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Arrange the items by the stated attribute in the OPPOSITE of the stated "
                        "order, then write them in that order separated by the '<' symbol.")
            return ("Arrange the items by the stated attribute in the stated order, then write them "
                    "in that order separated by the '<' symbol.")
        raise AssertionError("unexpected group")

    spec = {"topic": "comparison_order", "problem": problem, "ops": ops,
            "init": list(items), "finalize": lambda x: str(x),
            "club2_groups": [(0, 1)], "club2_block": club2_block}
    spec["checker"] = order_checker(_build_answer(spec, perturb=False))
    return spec


def f_set_ops(rng):
    shared, onlyA, onlyB = pick(rng, NAMES, 3)
    A, B = sorted([shared, onlyA]), sorted([shared, onlyB])
    region = {"both": shared, "onlyA": onlyA, "onlyB": onlyB}
    desc = {"both": "members who are in BOTH Club A and Club B",
            "onlyA": "members who are in Club A but NOT Club B",
            "onlyB": "members who are in Club B but NOT Club A"}
    g_kind, n_kind = rng.sample(list(region), 2)
    problem = (f"Club A members: {', '.join(A)}.\nClub B members: {', '.join(B)}.\n\n"
               f"The stated relation to compute: {desc[g_kind]}.")

    def compute(s): return region[g_kind]
    def compute_neg(s): return region[n_kind]
    def report(s): return s

    ops = [
        {"gold": "Apply the stated relation between the two member lists to get the matching people.",
         "neg": "Apply a DIFFERENT relation than the stated one between the two member lists.",
         "apply": compute, "neg_apply": compute_neg},
        {"gold": "Report the single person you found.",
         "neg": None, "apply": report, "neg_apply": report},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Apply a DIFFERENT relation than the stated one between the two member "
                        "lists, then report the single person you found.")
            return ("Apply the stated relation between the two member lists, then report the single "
                    "person you found.")
        raise AssertionError

    spec = {"topic": "set_ops", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1)], "club2_block": club2_block}
    spec["checker"] = choice_checker(_build_answer(spec, perturb=False))
    return spec if region[g_kind] != region[n_kind] else None


def f_transitive_logic(rng):
    players = pick(rng, NAMES, 5)
    rng.shuffle(players)
    facts = [f"{players[i]} beat {players[i+1]}" for i in range(len(players) - 1)]
    rng.shuffle(facts)
    problem = ("In a tournament (beating is transitive):\n- " + "\n- ".join(facts)
               + "\n\nThe stated relation: 'beat'. The stated target: the overall WINNER "
                 "(the one who beats everyone).")
    order_hi, order_lo = players[0], players[-1]

    def chain(s): return ("hi",)
    def chain_neg(s): return ("lo",)
    def report(s): return order_hi if s == ("hi",) else order_lo

    ops = [
        {"gold": "Chain the stated relation across all the facts to order everyone, then take the "
                 "stated target end.",
         "neg": "Chain the stated relation across all the facts to order everyone, then take the "
                "OPPOSITE end from the stated target.",
         "apply": chain, "neg_apply": chain_neg},
        {"gold": "Report the single person at that end.",
         "neg": None, "apply": report, "neg_apply": report},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Chain the stated relation to order everyone, take the OPPOSITE end from "
                        "the stated target, then report the single person at that end.")
            return ("Chain the stated relation to order everyone, take the stated target end, then "
                    "report the single person at that end.")
        raise AssertionError

    spec = {"topic": "transitive_logic", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1)], "club2_block": club2_block}
    spec["checker"] = choice_checker(_build_answer(spec, perturb=False))
    return spec


def f_scheduling(rng):
    tasks = pick(rng, ["prep", "mix", "bake", "cool", "frost", "box"], rng.randint(3, 5))
    rng.shuffle(tasks)
    rules = [f"'{tasks[i]}' must come before '{tasks[i+1]}'" for i in range(len(tasks) - 1)]
    rng.shuffle(rules)
    problem = ("Steps with ordering rules:\n- " + "\n- ".join(rules)
               + "\n\nThe stated relation: 'must come before'. The stated target: the step that "
                 "must be done FIRST.")
    first, last = tasks[0], tasks[-1]

    def order(s): return ("first",)
    def order_neg(s): return ("last",)
    def report(s): return first if s == ("first",) else last

    ops = [
        {"gold": "Use the stated relation to put all the steps in one valid order, then take the "
                 "stated target end.",
         "neg": "Use the stated relation to put all the steps in one valid order, then take the "
                "OPPOSITE end from the stated target.",
         "apply": order, "neg_apply": order_neg},
        {"gold": "Report the single step at that end.",
         "neg": None, "apply": report, "neg_apply": report},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Use the stated relation to put all the steps in one valid order, take the "
                        "OPPOSITE end from the stated target, then report the single step at that end.")
            return ("Use the stated relation to put all the steps in one valid order, take the "
                    "stated target end, then report the single step at that end.")
        raise AssertionError

    spec = {"topic": "scheduling", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1)], "club2_block": club2_block}
    spec["checker"] = choice_checker(_build_answer(spec, perturb=False))
    return spec


_CAT_THEMES = [
    ("Air-quality", "reading", ("good", "fair", "poor")),
    ("Loan-risk", "score", ("low", "medium", "high")),
    ("Lake-level", "depth", ("shallow", "normal", "flooded")),
    ("Battery-health", "percent", ("worn", "okay", "fresh")),
    ("Spice-heat", "rating", ("mild", "medium", "hot")),
    ("Crowd-size", "count", ("quiet", "busy", "packed")),
]

def f_categorize_rule(rng):
    metric, noun, (b_lo, b_mid, b_hi) = rng.choice(_CAT_THEMES)
    lo, hi = sorted(rng.sample(range(20, 80), 2))
    val = rng.randint(0, 100)
    while val == lo or val == hi:
        val = rng.randint(0, 100)
    problem = (f"{metric} rule: below {lo} is '{b_lo}', from {lo} to {hi} is '{b_mid}', "
               f"above {hi} is '{b_hi}'.\nThe stated value to categorize: {val}.\n"
               f"The stated rule is the {metric} rule above (apply its thresholds as written).")

    def classify(flip):
        a, b = (hi, lo) if flip else (lo, hi)
        if val < a: return b_lo
        if val <= b: return b_mid
        return b_hi

    def step_compare(s): return ("ok",)
    def step_compare_neg(s): return ("swapped",)
    def step_state(s): return classify(flip=(s == ("swapped",)))

    ops = [
        {"gold": "Apply the stated rule's thresholds to the stated value as written.",
         "neg": "Apply the stated rule's thresholds to the stated value, but swap which threshold "
                "is which.",
         "apply": step_compare, "neg_apply": step_compare_neg},
        {"gold": "State the category whose range the value falls into.",
         "neg": None, "apply": step_state, "neg_apply": step_state},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Apply the stated rule's thresholds to the stated value but SWAP which "
                        "threshold is which, then state the category whose range the value falls into.")
            return ("Apply the stated rule's thresholds to the stated value as written, then state "
                    "the category whose range the value falls into.")
        raise AssertionError

    spec = {"topic": "categorize_rule", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1)], "club2_block": club2_block}
    gold = _build_answer(spec, perturb=False)
    neg = _build_answer(spec, perturb=True)
    if gold == neg:
        return None
    spec["checker"] = choice_checker(gold)
    return spec


def f_multi_hop_lookup(rng):
    ppl = pick(rng, NAMES, 4)
    cities = pick(rng, CITIES, 4)
    mascots = pick(rng, ["Owls", "Bears", "Hawks", "Foxes", "Wolves", "Rams"], 4)
    p2c = dict(zip(ppl, cities))
    c2m = dict(zip(cities, mascots))
    who = rng.choice(ppl)
    problem = ("Mapping 1 (person -> city):\n- " + "\n- ".join(f"{p} -> {p2c[p]}" for p in ppl)
               + "\nMapping 2 (city -> mascot):\n- " + "\n- ".join(f"{c} -> {c2m[c]}" for c in cities)
               + f"\n\nThe stated start key: {who}. Resolve it through mapping 1 then mapping 2 to "
                 "the final value.")

    def hop1(s): return p2c[who]
    def hop2(s): return c2m[s]
    def hop2_neg(s): return s
    def report(s): return s

    ops = [
        {"gold": "Resolve the stated start key through the 1st stated mapping.",
         "neg": None, "apply": hop1, "neg_apply": hop1},
        {"gold": "Resolve that result through the 2nd stated mapping.",
         "neg": "Skip the 2nd stated mapping and keep the result from the 1st mapping unchanged.",
         "apply": hop2, "neg_apply": hop2_neg},
        {"gold": "Report that final value.",
         "neg": None, "apply": report, "neg_apply": report},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Resolve the stated start key through the 1st stated mapping, then SKIP the "
                        "2nd stated mapping and keep that result unchanged.")
            return ("Resolve the stated start key through the 1st stated mapping, then resolve that "
                    "result through the 2nd stated mapping.")
        return "Report that final value."

    spec = {"topic": "multi_hop_lookup", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1), (2,)], "club2_block": club2_block}
    gold = _build_answer(spec, perturb=False)
    neg = _build_answer(spec, perturb=True)
    if gold == neg:
        return None
    spec["checker"] = choice_checker(gold)
    return spec


def f_conditional_reco(rng):
    axisA, optsA = rng.choice([("temperature", ("hot", "cold")),
                               ("budget", ("cheap", "premium")),
                               ("season", ("summer", "winter"))])
    axisB, optsB = rng.choice([("taste", ("sweet", "plain")),
                               ("size", ("small", "large")),
                               ("mood", ("calm", "lively"))])
    leaves = pick(rng, ["Cocoa", "Tea", "Lemonade", "Water", "Cider", "Juice",
                        "Mocha", "Soda", "Punch", "Latte"], 4)
    a, b = rng.choice(optsA), rng.choice(optsB)
    grid = {(optsA[0], optsB[0]): leaves[0], (optsA[0], optsB[1]): leaves[1],
            (optsA[1], optsB[0]): leaves[2], (optsA[1], optsB[1]): leaves[3]}
    rules = "\n".join(f"- If {x} and {y} -> {grid[(x, y)]}" for x in optsA for y in optsB)
    problem = (f"Decision table (by {axisA} and {axisB}):\n{rules}\n"
               f"\nThe guest's stated 1st condition ({axisA}): {a}.\n"
               f"The guest's stated 2nd condition ({axisB}): {b}.\n"
               "Match both stated conditions in the table to one item.")
    flip_a = optsA[0] if a == optsA[1] else optsA[1]

    def read1(s): return (a,)
    def read1_neg(s): return (flip_a,)
    def read2(s): return s + (b,)
    def lookup(s): return grid[(s[0], s[1])]

    ops = [
        {"gold": "Read the 1st stated condition as given.",
         "neg": "Read the 1st stated condition as the OPPOSITE of what is given.",
         "apply": read1, "neg_apply": read1_neg},
        {"gold": "Read the 2nd stated condition as given.",
         "neg": None, "apply": read2, "neg_apply": read2},
        {"gold": "Match both stated conditions in the table and report the single item.",
         "neg": None, "apply": lookup, "neg_apply": lookup},
    ]

    def club2_block(gi, idxs, which):
        if idxs == (0, 1):
            if which == "neg":
                return ("Read the 1st stated condition as the OPPOSITE of what is given, "
                        "then read the 2nd stated condition as given.")
            return ("Read the 1st stated condition as given, then read the 2nd "
                    "stated condition as given.")
        return "Match both stated conditions in the table and report the single item."

    spec = {"topic": "conditional_reco", "problem": problem, "ops": ops, "init": None,
            "finalize": lambda x: str(x), "club2_groups": [(0, 1), (2,)], "club2_block": club2_block}
    gold = _build_answer(spec, perturb=False)
    neg = _build_answer(spec, perturb=True)
    if gold == neg:
        return None
    spec["checker"] = choice_checker(gold)
    return spec


FAMILIES = [f_constraint_select, f_comparison_order, f_set_ops, f_transitive_logic,
            f_scheduling, f_categorize_rule, f_multi_hop_lookup, f_conditional_reco]


# =========================================================================== block rendering
def render_blocks(spec, club, which):
    """op list -> list of BLOCK strings. which in {'gold','neg'} (neg uses the single perturbed op's
    neg phrasing). club=1: one op -> one block. club=2: bundle per spec['club2_groups']."""
    if club == 1:
        return [op["neg"] if (which == "neg" and op["neg"] is not None) else op["gold"]
                for op in spec["ops"]]
    return [spec["club2_block"](gi, idxs, which) for gi, idxs in enumerate(spec["club2_groups"])]


_NOOPS = ["Read the problem statement carefully.",
          "List out the items the problem states.",
          "Note what the question is asking for."]


def pad_to_turns(gold_blocks, neg_blocks, n_turns):
    g, n = list(gold_blocks), list(neg_blocks)
    pad = 0
    while len(g) < n_turns:
        g.insert(0, _NOOPS[pad % len(_NOOPS)]); n.insert(0, _NOOPS[pad % len(_NOOPS)]); pad += 1
    return g, n


# =========================================================================== row assembly
def build_row(rng, idx, club, fam, target_turns):
    spec = None
    for _ in range(60):
        spec = fam(rng)
        if spec is not None:
            break
    if spec is None:
        return None
    perturbed = [i for i, op in enumerate(spec["ops"]) if op["neg"] is not None]
    if len(perturbed) != 1:
        return None
    gold = _build_answer(spec, perturb=False)
    neg = _build_answer(spec, perturb=True)
    if str(gold) == str(neg):
        return None
    if str(gold).strip().lower() == "none" or str(neg).strip().lower() == "none":
        return None                       # filter chain emptied out -> ambiguous answer, skip
        # (club-independent: dropped at the same tick in c1 and c2, so alignment is preserved)
    gold_blocks = render_blocks(spec, club, "gold")
    neg_blocks = render_blocks(spec, club, "neg")
    if len(gold_blocks) != len(neg_blocks):
        return None
    block_diffs = [i for i in range(len(gold_blocks)) if gold_blocks[i] != neg_blocks[i]]
    if len(block_diffs) != 1:
        return None
    for blk in gold_blocks + neg_blocks:
        if _generic_violations(blk):
            return None
    gold_blocks, neg_blocks = pad_to_turns(gold_blocks, neg_blocks, target_turns)
    if not (2 <= len(gold_blocks) <= 5):
        return None
    ckind, cargs = spec["checker"]
    return {
        "id": f"grnb_{idx:04d}", "topic": spec["topic"], "n_turns": len(gold_blocks),
        "problem": spec["problem"], "gold_plan": gold_blocks, "gold_answer": str(gold),
        "neg_plan": neg_blocks, "neg_answer": str(neg),
        "checker_kind": ckind, "checker_args": cargs, "club": club,
    }


def gen_dataset(seed, n, club):
    """Generate n rows at the given club level. Same (seed,n) across club levels yields the SAME
    sampled problems & answers (club affects rendering, not sampling), so c1 and c2 align row-for-row."""
    rng = random.Random(seed)
    rows, seen = [], set()
    per_topic, per_turns = collections.Counter(), collections.Counter()
    idx = 0
    tick = 0
    while len(rows) < n and tick < n * 300:
        fam = FAMILIES[tick % len(FAMILIES)]
        if club == 1:
            target = [2, 3, 4, 5][(tick + tick // len(FAMILIES)) % 4]
        else:
            target = [2, 3][(tick + tick // len(FAMILIES)) % 2]
        tick += 1
        row = build_row(rng, idx, club, fam, target)
        if row is None:
            continue
        key = row["problem"]
        if key in seen:
            continue
        assert row["gold_answer"] != row["neg_answer"], "negative did not change the answer"
        assert 2 <= row["n_turns"] <= 5
        assert len(row["gold_plan"]) == len(row["neg_plan"])
        d = sum(1 for x, y in zip(row["gold_plan"], row["neg_plan"]) if x != y)
        assert d == 1, f"expected one perturbed block, got {d}"
        for blk in row["gold_plan"] + row["neg_plan"]:
            assert not _generic_violations(blk), \
                f"non-generic block leaked domain word: {blk!r} -> {_generic_violations(blk)}"
        if row["checker_kind"] == "string_contains":
            g, nn = _norm_ws(row["gold_answer"]), _norm_ws(row["neg_answer"])
            assert g not in nn and nn not in g, "ordering answers substring-collide"
        seen.add(key)
        rows.append(row)
        per_topic[row["topic"]] += 1
        per_turns[row["n_turns"]] += 1
        idx += 1
    if len(rows) < n:
        raise SystemExit(f"only generated {len(rows)}/{n} rows at club={club}; raise budget/pools")
    return rows, per_topic, per_turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--club", type=int, default=0, choices=[0, 1, 2],
                    help="0 = emit BOTH c1 and c2 (default); 1 or 2 = emit only that level")
    ap.add_argument("--out", default=None,
                    help="if --club is 1 or 2 and --out given, write just that file; else write "
                         "grounding_blocks_c1.jsonl and grounding_blocks_c2.jsonl.")
    args = ap.parse_args()
    levels = [1, 2] if args.club == 0 else [args.club]

    built = {}
    for club in levels:
        rows, per_topic, per_turns = gen_dataset(args.seed, args.n, club)
        built[club] = rows
        out_path = args.out if (args.out and len(levels) == 1) else f"grounding_blocks_c{club}.jsonl"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows (club={club}) -> {out_path}")
        print(f"  per-topic : {dict(sorted(per_topic.items()))}")
        print(f"  per-turns : {dict(sorted(per_turns.items()))}")

    if 1 in built and 2 in built:
        c1, c2 = built[1], built[2]
        assert len(c1) == len(c2), "c1/c2 length mismatch"
        for r1, r2 in zip(c1, c2):
            assert r1["problem"] == r2["problem"], "c1/c2 problems diverged (same seed must align)"
            assert r1["gold_answer"] == r2["gold_answer"], "c1/c2 gold answers diverged"
            assert r1["neg_answer"] == r2["neg_answer"], "c1/c2 neg answers diverged"
            assert r1["topic"] == r2["topic"]
        print(f"[align] OK: c1 and c2 share identical problems + gold/neg answers across {len(c1)} "
              "rows (differ only in block phrasing/clubbing).")


if __name__ == "__main__":
    main()