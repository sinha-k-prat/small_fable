#!/usr/bin/env python3
"""
gen_plan_formats.py — PLAN-FORMAT sweep for the GROUNDING PROBE (grounding_test.py).

PRIOR FINDING (GROUNDING_RESULTS.md): a FROZEN Qwen2.5-1.5B-Instruct GROUNDS *concrete* English
plans (conditional neg_follow ~59%) but NOT *generic / positional* ones (~10%), and clearer
phrasing of the generic plan did not help. This script asks a sharper question: holding the SAME
100 problems and the SAME gold/neg answers fixed, does the *FORMAT* in which we phrase the generic
plan change whether the frozen model follows it? It renders each family's gold+neg plan in >=5
FORMATS, one row-aligned JSONL per format, so grounding_test.py can be run unchanged on each and the
resulting neg_follow / neg_to_gold compared FORMAT-vs-FORMAT on the very same items.

FORMATS (each its own file, identical id/problem/gold_answer/neg_answer, differing ONLY in plan
phrasing):
  (1) terse        — minimal one-clause imperative primitives.                        [GENERIC]
  (2) verbose      — fuller natural English (the current v2.1 gold/neg phrasing).     [GENERIC]
  (3) numbered     — the operation decomposed into explicit tiny numbered sub-steps.  [GENERIC]
  (4) definitional — first GENERICALLY DEFINE what the primitive does, then apply it.  [GENERIC]
  (5) concrete     — the v1-style plan that NAMES the domain thing (UPPER BOUND).     [NAMES THINGS]

The four GENERIC formats use ONLY positional/ordinal references + structural vocabulary and are
hard-asserted to leak ZERO domain content-words via gen_grounding_blocks._generic_violations (the
SAME STRUCT_OK gate the probe data already passes). The CONCRETE format is the only one allowed to
name domain words — it is the reference upper bound, mirroring v1 ("Keep only the gadgets that are
waterproof").

GROUND TRUTH IS SHARED, NOT RE-DERIVED. We reuse gen_grounding_blocks's per-family op/solver
infrastructure verbatim (FAMILIES, the per-op apply/neg_apply closures, _build_answer, render_blocks,
the STRUCT_OK genericness gate). We replay the EXACT club=1 sampling driver (same seed -> same RNG
draws -> same problems and same gold/neg answers as grounding_blocks_c1.jsonl). For each accepted
problem we render its single perturbed op in all 5 formats. Single-op-per-block (club=1) throughout.

Because every format file is produced from the SAME spec stream, every file is row-for-row aligned:
identical id / topic / problem / gold_answer / neg_answer; only gold_plan/neg_plan phrasing differs.
This is exactly what makes "which format does the model follow?" a fair comparison — and it means the
fails-unaided (acc_noplan==0) subset that grounding_test.py conditions on is IDENTICAL across formats
(acc_noplan depends only on the problem + the model, never on the plan).

stdlib-only, fully seeded. Hard asserts: per-format genericness (0 leaks for the 4 generic formats),
gold_answer != neg_answer, and cross-format alignment (all formats share problems + answers per id).
"""
import argparse, collections, json, os, random, re

import gen_grounding_blocks as G  # reuse FAMILIES / solvers / _build_answer / STRUCT_OK gate
# (tools/ is on sys.path when run as `python tools/gen_plan_formats.py`; the import below also works
#  when invoked as a module — see the sys.path shim in main().)


# STRUCT_OK EXTENSION (in-script, documented). The terse/numbered/definitional formats introduce a
# few extra *purely structural* English words that the verbose-vocab STRUCT_OK simply hadn't listed
# yet. EVERY token below is content-free (a connective, a generic verb, or a generic noun for a
# slot/step) — NONE names any domain entity, attribute, relation, or value. We extend the SAME gate
# the probe uses (G.STRUCT_OK) so genericness is judged by one shared whitelist, and the 4 generic
# formats still hard-assert to 0 leaks. (Audited against the leak set printed by a dry run; if a new
# word were a domain leak it would NOT be added here — it would be a bug to fix in the wording.)
_EXTRA_STRUCT_OK = {
    "means", "here", "obeying", "band", "meeting", "right", "separator", "total", "orders",
    "see", "landed", "categorize", "fell", "ended", "rest", "ones", "check", "whether", "so",
    "far", "comparing", "consult", "same",
}
G.STRUCT_OK |= _EXTRA_STRUCT_OK


GENERIC_FORMATS = ["terse", "verbose", "numbered", "definitional"]
ALL_FORMATS = GENERIC_FORMATS + ["concrete"]


# =========================================================================================
# PROBLEM PARSERS — pull the DISPLAY labels needed by the CONCRETE (naming) format straight out of
# spec["problem"] (the text the model already sees). No new randomness; purely deterministic parse of
# the rendered problem, so the concrete plan stays perfectly consistent with the shown problem.
# =========================================================================================
def _p_constraint(problem):
    req = re.findall(r"\d\)\s*(.+)", problem)
    pref = re.search(r"Preference[^:]*:\s*(.+?)\.", problem)
    return {"req1": req[0].strip(), "req2": req[1].strip(),
            "pref": (pref.group(1).strip() if pref else "cheapest")}

def _p_comparison(problem):
    attr = re.search(r"stated attribute is its (.+?):", problem).group(1).strip()
    order = "ASCENDING" if "ASCENDING" in problem else "DESCENDING"
    lowfirst = "lowest" in problem.split("order to produce")[-1]
    return {"attr": attr, "order": order, "lowfirst": lowfirst}

def _p_set_ops(problem):
    rel = re.search(r"stated relation to compute:\s*(.+?)\.", problem).group(1).strip()
    return {"rel": rel}

def _p_transitive(problem):
    rel = re.search(r"stated relation:\s*'(.+?)'", problem).group(1)
    tgt = re.search(r"stated target:\s*(.+?)\s*\(", problem).group(1).strip()
    return {"rel": rel, "target": tgt}

def _p_scheduling(problem):
    rel = re.search(r"stated relation:\s*'(.+?)'", problem).group(1)
    tgt = re.search(r"stated target:\s*(.+?)\.", problem).group(1).strip()
    return {"rel": rel, "target": tgt}

def _p_categorize(problem):
    lo = re.search(r"below (\d+) is '(.+?)'", problem)
    hi = re.search(r"above (\d+) is '(.+?)'", problem)
    val = re.search(r"value to categorize:\s*(\d+)", problem).group(1)
    return {"lo": lo.group(1), "hi": hi.group(1), "val": val}

def _p_multihop(problem):
    key = re.search(r"stated start key:\s*(.+?)\.", problem).group(1).strip()
    return {"key": key}

def _p_conditional(problem):
    a = re.search(r"1st condition \((.+?)\):\s*(.+?)\.", problem)
    b = re.search(r"2nd condition \((.+?)\):\s*(.+?)\.", problem)
    return {"axisA": a.group(1), "valA": a.group(2).strip(),
            "axisB": b.group(1), "valB": b.group(2).strip()}


# =========================================================================================
# FORMAT RENDERERS — one per topic. Each returns a list (one entry per op) of {fmt: block_str}
# for a given `which` ('gold' or 'neg'), with exactly the SAME number of ops as spec['ops']
# (club=1, one op per block). The op that carries the perturbation is the one whose 'gold' vs
# 'neg' wording differs; every other op is identical across gold/neg (matching gen_grounding_blocks).
# The 4 generic formats stay positional; only `concrete` names domain tokens (parsed from problem).
# =========================================================================================
def r_constraint_select(spec, which):
    c = _p_constraint(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": "Keep only the items meeting the 1st stated requirement; drop the rest.",
            "verbose": ("Go through the list of items one by one, and keep only the items that meet "
                        "the first requirement stated in the problem. Remove all the items that do "
                        "not meet it."),
            "numbered": ("1) Read the 1st stated requirement. 2) For each item, check whether it "
                         "meets that requirement. 3) Keep the items that do and remove the items "
                         "that do not."),
            "definitional": ("To FILTER by a stated requirement means to keep exactly the items that "
                             "satisfy it and remove the others. Apply that to the 1st stated "
                             "requirement: keep the items that meet it."),
            "concrete": f"Keep only the gadgets that are {c['req1']}; remove the rest.",
        }
    else:
        op0 = {
            "terse": "Keep only the items NOT meeting the 1st stated requirement; drop the rest.",
            "verbose": ("Go through the list of items one by one, and keep only the items that do "
                        "NOT meet the first requirement stated in the problem. Remove all the items "
                        "that do meet it."),
            "numbered": ("1) Read the 1st stated requirement. 2) For each item, check whether it "
                         "meets that requirement. 3) Keep the items that do NOT and remove the "
                         "items that do."),
            "definitional": ("To FILTER by a stated requirement means to keep exactly the items that "
                             "satisfy it and remove the others. Apply the OPPOSITE to the 1st stated "
                             "requirement: keep the items that do NOT meet it."),
            "concrete": f"Keep only the gadgets that are NOT {c['req1']}; remove the rest.",
        }
    op1 = {
        "terse": "From those, keep only the ones meeting the 2nd stated requirement.",
        "verbose": ("From the items you just kept, keep only the items that also meet the second "
                    "requirement stated in the problem, and remove the others."),
        "numbered": ("1) Take the items you just kept. 2) Read the 2nd stated requirement. "
                     "3) Keep only the items that also meet it."),
        "definitional": ("FILTER again: from the items kept so far, keep exactly the items that "
                         "satisfy the 2nd stated requirement and remove the others."),
        "concrete": f"From those, keep only the ones that are {c['req2']}.",
    }
    op2 = {
        "terse": "Among those left, pick the single one best matching the stated preference.",
        "verbose": ("Among the items that are still left, pick the single item that best matches the "
                    "preference stated in the problem."),
        "numbered": ("1) Take the items still left. 2) Read the stated preference. 3) Pick the "
                     "single item that best matches it."),
        "definitional": ("To SELECT-BY-PREFERENCE means to choose the one item that best matches a "
                         "stated preference. Apply it: among the items still left, pick the single "
                         "best match for the stated preference."),
        "concrete": f"Among the remaining gadgets, choose the {c['pref']} one.",
    }
    return [op0, op1, op2]


def r_comparison_order(spec, which):
    c = _p_comparison(spec["problem"])
    dir_word = "lowest value first" if c["lowfirst"] else "highest value first"
    cdir = "lowest to highest" if c["lowfirst"] else "highest to lowest"
    cdir_op = "highest to lowest" if c["lowfirst"] else "lowest to highest"
    if which == "gold":
        op0 = {
            "terse": "Sort the items by the stated attribute in the stated direction.",
            "verbose": ("Sort all of the items by the attribute named in the problem, going in the "
                        "direction the problem states (lowest value first if it says ascending)."),
            "numbered": ("1) Read the attribute named in the problem and the stated direction. "
                         "2) Compare the items by that attribute. 3) Order them in the stated "
                         "direction."),
            "definitional": ("To SORT means to arrange the items by an attribute in a stated "
                             "direction. Apply it using the stated attribute and the stated "
                             "direction."),
            "concrete": f"List the models from {cdir} {c['attr']} ({dir_word}).",
        }
    else:
        op0 = {
            "terse": "Sort the items by the stated attribute in the OPPOSITE of the stated direction.",
            "verbose": ("Sort all of the items by the attribute named in the problem, but going in "
                        "the OPPOSITE direction to the one the problem states."),
            "numbered": ("1) Read the attribute named in the problem and the stated direction. "
                         "2) Compare the items by that attribute. 3) Order them in the OPPOSITE of "
                         "the stated direction."),
            "definitional": ("To SORT means to arrange the items by an attribute in a stated "
                             "direction. Apply it using the stated attribute but the OPPOSITE of the "
                             "stated direction."),
            "concrete": f"List the models from {cdir_op} {c['attr']}.",
        }
    op1 = {
        "terse": "Write them in that order with '<' between each one.",
        "verbose": ("Now write the items out in that sorted order, putting the '<' symbol between "
                    "each one and the next."),
        "numbered": ("1) Take the sorted order. 2) Write the items left to right. 3) Put the '<' "
                     "symbol between each one and the next."),
        "definitional": ("To EMIT the ordering means to write the items in their sorted order joined "
                         "by a separator. Apply it, using '<' between each one and the next."),
        "concrete": "Write them out in that order with '<' between each model.",
    }
    return [op0, op1]


def r_set_ops(spec, which):
    c = _p_set_ops(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": "Work out exactly which people the stated relation selects.",
            "verbose": ("The problem names one relation between the two member lists. Work out "
                        "exactly which people that stated relation selects."),
            "numbered": ("1) Read the relation the problem states. 2) Apply it to the two member "
                         "lists. 3) Work out exactly which people it selects."),
            "definitional": ("To apply the STATED RELATION means to select the people that the "
                             "relation named in the problem picks out. Apply the stated relation to "
                             "the two member lists."),
            "concrete": f"Work out which people are {c['rel']}.",
        }
    else:
        op0 = {
            "terse": "Ignore the stated relation; work out which people a DIFFERENT relation selects.",
            "verbose": ("Ignore the stated relation. Instead, work out which people a DIFFERENT "
                        "relation between the two member lists selects."),
            "numbered": ("1) Ignore the relation the problem states. 2) Take a DIFFERENT relation "
                         "between the two lists. 3) Work out which people that one selects."),
            "definitional": ("To apply the STATED RELATION means to select the people the stated "
                             "relation picks out. Here do the OPPOSITE: ignore the stated relation "
                             "and apply a DIFFERENT relation between the two lists."),
            "concrete": "Ignore that; instead work out which people are in BOTH clubs.",
        }
    op1 = {
        "terse": "Report the single person selected.",
        "verbose": "Report the single person you selected.",
        "numbered": "1) Take the people you selected. 2) Report the single person.",
        "definitional": ("To REPORT means to state the selected result. Apply it: state the single "
                         "person you selected."),
        "concrete": "Report that single person's name.",
    }
    return [op0, op1]


def r_transitive_logic(spec, which):
    c = _p_transitive(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": ("Chain the facts into one ranking; take the person at the stated target end."),
            "verbose": ("Each fact links two people by the stated relation. Follow the links from "
                        "one end to the other to put everyone into a single ranking, and then take "
                        "the person at the stated target end of that ranking."),
            "numbered": ("1) Each fact links two people. 2) Follow the links end to end to build "
                         "one ranking of everyone. 3) Take the person at the stated target end."),
            "definitional": ("To CHAIN a relation means to follow its links into a single total "
                             "ranking. Apply it: build the ranking, then take the person at the "
                             "stated target end."),
            "concrete": f"Chain the '{c['rel']}' facts into one ranking and take the {c['target']}.",
        }
    else:
        op0 = {
            "terse": ("Chain the facts into one ranking; take the person at the OPPOSITE end from "
                      "the stated target."),
            "verbose": ("Each fact links two people by the stated relation. Follow the links to put "
                        "everyone into a single ranking, and then take the person at the OPPOSITE "
                        "end from the stated target."),
            "numbered": ("1) Each fact links two people. 2) Follow the links end to end to build "
                         "one ranking of everyone. 3) Take the person at the OPPOSITE end from the "
                         "stated target."),
            "definitional": ("To CHAIN a relation means to follow its links into a single total "
                             "ranking. Apply it: build the ranking, then take the person at the "
                             "OPPOSITE end from the stated target."),
            "concrete": (f"Chain the '{c['rel']}' facts into one ranking and take the LOSER (the one "
                         "who beats no one)."),
        }
    op1 = {
        "terse": "Report that single person.",
        "verbose": "Report that single person.",
        "numbered": "1) Take the person you picked. 2) Report that single person.",
        "definitional": "To REPORT means to state the result. Apply it: state that single person.",
        "concrete": "Report that single player's name.",
    }
    return [op0, op1]


def r_scheduling(spec, which):
    c = _p_scheduling(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": ("Order the steps to obey every rule; take the step at the stated target end."),
            "verbose": ("Each rule says one step must come before another. Arrange all of the steps "
                        "into a single sequence that obeys every rule, and then take the step at the "
                        "stated target end of that sequence."),
            "numbered": ("1) Each rule orders two steps. 2) Arrange all steps into one sequence "
                         "obeying every rule. 3) Take the step at the stated target end."),
            "definitional": ("To ORDER by the rules means to build one sequence obeying every stated "
                             "ordering rule. Apply it, then take the step at the stated target end."),
            "concrete": "Order the steps so each comes before the one it must, then take the FIRST.",
        }
    else:
        op0 = {
            "terse": ("Order the steps to obey every rule; take the step at the OPPOSITE end from "
                      "the stated target."),
            "verbose": ("Each rule says one step must come before another. Arrange all of the steps "
                        "into a single sequence that obeys every rule, and then take the step at the "
                        "OPPOSITE end from the stated target."),
            "numbered": ("1) Each rule orders two steps. 2) Arrange all steps into one sequence "
                         "obeying every rule. 3) Take the step at the OPPOSITE end from the stated "
                         "target."),
            "definitional": ("To ORDER by the rules means to build one sequence obeying every stated "
                             "ordering rule. Apply it, then take the step at the OPPOSITE end from "
                             "the stated target."),
            "concrete": "Order the steps so each comes before the one it must, then take the LAST.",
        }
    op1 = {
        "terse": "Report that single step.",
        "verbose": "Report that single step.",
        "numbered": "1) Take the step you picked. 2) Report that single step.",
        "definitional": "To REPORT means to state the result. Apply it: state that single step.",
        "concrete": "Report that single step.",
    }
    return [op0, op1]


def r_categorize_rule(spec, which):
    c = _p_categorize(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": "Compare the stated value against the two thresholds in the order written.",
            "verbose": ("The rule in the problem sorts a value into a category using two threshold "
                        "points. Compare the value given in the problem against those two threshold "
                        "points, in the order the rule writes them."),
            "numbered": ("1) Read the rule's two thresholds in the order written. 2) Read the "
                         "stated value. 3) Compare the value against the two thresholds in that "
                         "order."),
            "definitional": ("To CATEGORIZE-BY-THRESHOLDS means to compare a value against the "
                             "rule's thresholds in the order written. Apply it to the stated "
                             "value, using the thresholds in the order written."),
            "concrete": f"Compare {c['val']} against the thresholds {c['lo']} then {c['hi']} as written.",
        }
    else:
        op0 = {
            "terse": "Compare the stated value against the two thresholds but SWAP the two thresholds.",
            "verbose": ("The rule in the problem uses two thresholds. Compare the value given in "
                        "the problem against them, but SWAP the two thresholds before you "
                        "compare."),
            "numbered": ("1) Read the rule's two thresholds. 2) SWAP the two thresholds. "
                         "3) Compare the stated value against the swapped thresholds."),
            "definitional": ("To CATEGORIZE-BY-THRESHOLDS means to compare a value against the "
                             "rule's thresholds in the order written. Here do it WRONG: SWAP the "
                             "two thresholds before comparing the stated value."),
            "concrete": f"Compare {c['val']} against the thresholds but SWAPPED: {c['hi']} then {c['lo']}.",
        }
    op1 = {
        "terse": "State the category the value falls into.",
        "verbose": "State the name of the category that the value falls into.",
        "numbered": "1) See which band the value landed in. 2) State that category's name.",
        "definitional": ("To REPORT the category means to state the band the value fell into. Apply "
                         "it: state that category's name."),
        "concrete": "State the category that value falls into.",
    }
    return [op0, op1]


def r_multi_hop_lookup(spec, which):
    c = _p_multihop(spec["problem"])
    op0 = {
        "terse": "Look up the stated start key in the 1st mapping to get its value.",
        "verbose": ("Take the starting key given in the problem and look it up in the first mapping "
                    "to find its matching value."),
        "numbered": ("1) Read the stated start key. 2) Find its row in the 1st mapping. 3) Take "
                     "the matching value."),
        "definitional": ("To LOOK UP a key means to find its matching value in a mapping. Apply it "
                         "to the stated start key using the 1st mapping."),
        "concrete": f"Look up {c['key']} in mapping 1 to find its city.",
    }
    if which == "gold":
        op1 = {
            "terse": "Look that value up in the 2nd mapping to get the next value.",
            "verbose": ("Now take that value and look it up in the second mapping to find the next "
                        "value."),
            "numbered": ("1) Take the value from the 1st mapping. 2) Find its row in the 2nd "
                         "mapping. 3) Take the next matching value."),
            "definitional": ("To LOOK UP a key means to find its matching value in a mapping. Apply "
                             "it again: take the value just found and look it up in the 2nd mapping."),
            "concrete": "Look that city up in mapping 2 to find its mascot.",
        }
    else:
        op1 = {
            "terse": "Do NOT use the 2nd mapping; keep the value from the 1st mapping.",
            "verbose": ("Do not use the second mapping at all; simply keep the value you already "
                        "found from the first mapping."),
            "numbered": ("1) Take the value from the 1st mapping. 2) Do NOT consult the 2nd "
                         "mapping. 3) Keep that same value."),
            "definitional": ("To LOOK UP a key means to find its matching value in a mapping. Here "
                             "SKIP the second look-up: do not use the 2nd mapping, keep the value "
                             "from the 1st."),
            "concrete": "Do NOT use mapping 2; keep the city from mapping 1.",
        }
    op2 = {
        "terse": "Report that final value.",
        "verbose": "Report that final value.",
        "numbered": "1) Take the value you ended on. 2) Report that final value.",
        "definitional": "To REPORT means to state the result. Apply it: state that final value.",
        "concrete": "Report that final value.",
    }
    return [op0, op1, op2]


def r_conditional_reco(spec, which):
    c = _p_conditional(spec["problem"])
    if which == "gold":
        op0 = {
            "terse": "Read the 1st condition exactly as stated.",
            "verbose": "Read the first condition exactly as the problem states it.",
            "numbered": ("1) Find the 1st stated condition. 2) Read its value exactly as written."),
            "definitional": ("To READ a condition means to take its value exactly as the problem "
                             "states it. Apply it to the 1st stated condition."),
            "concrete": f"Read the 1st condition ({c['axisA']}) as stated: {c['valA']}.",
        }
    else:
        op0 = {
            "terse": "Read the 1st condition as the OPPOSITE of what is stated.",
            "verbose": "Read the first condition as the OPPOSITE of what the problem states.",
            "numbered": ("1) Find the 1st stated condition. 2) Read its value as the OPPOSITE of "
                         "what is written."),
            "definitional": ("To READ a condition means to take its value as stated. Here do it "
                             "WRONG: read the 1st condition as the OPPOSITE of what is stated."),
            "concrete": f"Read the 1st condition ({c['axisA']}) as the OPPOSITE of {c['valA']}.",
        }
    op1 = {
        "terse": "Read the 2nd condition exactly as stated.",
        "verbose": "Read the second condition exactly as the problem states it.",
        "numbered": "1) Find the 2nd stated condition. 2) Read its value exactly as written.",
        "definitional": ("To READ a condition means to take its value exactly as stated. Apply it to "
                         "the 2nd stated condition."),
        "concrete": f"Read the 2nd condition ({c['axisB']}) as stated: {c['valB']}.",
    }
    op2 = {
        "terse": "Match both conditions to the single table row and report that item.",
        "verbose": ("Using both conditions together, find the single row of the table they point to, "
                    "and report the item on that row."),
        "numbered": ("1) Take both condition values. 2) Find the single table row they point to. "
                     "3) Report the item on that row."),
        "definitional": ("To MATCH in the table means to find the single row both conditions point "
                         "to. Apply it and report the item on that row."),
        "concrete": "Find the table row for both conditions and report that item.",
    }
    return [op0, op1, op2]


RENDERERS = {
    "constraint_select": r_constraint_select,
    "comparison_order": r_comparison_order,
    "set_ops": r_set_ops,
    "transitive_logic": r_transitive_logic,
    "scheduling": r_scheduling,
    "categorize_rule": r_categorize_rule,
    "multi_hop_lookup": r_multi_hop_lookup,
    "conditional_reco": r_conditional_reco,
}


# =========================================================================================
# ROW BUILD — replay gen_grounding_blocks's club=1 driver to recover the SAME specs (same seed ->
# same RNG draws), recompute gold/neg answers via the shared solver, and render all 5 formats.
# =========================================================================================
def _spec_for(fam, rng):
    spec = None
    for _ in range(60):
        spec = fam(rng)
        if spec is not None:
            break
    return spec


def _format_blocks(spec, fmt):
    """-> (gold_blocks, neg_blocks) for one format, one op per block."""
    renderer = RENDERERS[spec["topic"]]
    gold_ops = renderer(spec, "gold")
    neg_ops = renderer(spec, "neg")
    assert len(gold_ops) == len(neg_ops) == len(spec["ops"]), \
        f"{spec['topic']}: renderer produced wrong block count"
    return [op[fmt] for op in gold_ops], [op[fmt] for op in neg_ops]


def _assert_one_perturbed(gold_blocks, neg_blocks, topic, fmt):
    diffs = [i for i in range(len(gold_blocks)) if gold_blocks[i] != neg_blocks[i]]
    assert len(diffs) == 1, (f"{topic}/{fmt}: expected exactly ONE perturbed block, got "
                             f"{len(diffs)} ({diffs})")


def build_rows(seed, n):
    """Replay the club=1 sampling loop of gen_grounding_blocks.gen_dataset, but for each accepted
    problem emit a dict of {format: row}. All formats share id/topic/problem/answers per row."""
    rng = random.Random(seed)
    out = {f: [] for f in ALL_FORMATS}     # format -> list of rows (row-aligned across formats)
    seen = set()
    per_topic = collections.Counter()
    idx = 0
    tick = 0
    while len(out["verbose"]) < n and tick < n * 300:
        fam = G.FAMILIES[tick % len(G.FAMILIES)]
        # club=1 target-turns schedule, IDENTICAL to gen_grounding_blocks.gen_dataset(club=1),
        # so the RNG-advancing accept/reject path matches exactly -> same problems & answers.
        target = [2, 3, 4, 5][(tick + tick // len(G.FAMILIES)) % 4]
        tick += 1
        spec = _spec_for(fam, rng)
        if spec is None:
            continue
        perturbed = [i for i, op in enumerate(spec["ops"]) if op["neg"] is not None]
        if len(perturbed) != 1:
            continue
        gold = G._build_answer(spec, perturb=False)
        neg = G._build_answer(spec, perturb=True)
        if str(gold) == str(neg):
            continue
        if str(gold).strip().lower() == "none" or str(neg).strip().lower() == "none":
            continue
        # gate the *verbose* generic blocks via the SAME path build_row uses, to mirror its accept/
        # reject exactly (so we keep the identical 100 problems as grounding_blocks_c1.jsonl).
        v_gold, v_neg = _format_blocks(spec, "verbose")
        if any(G._generic_violations(b) for b in v_gold + v_neg):
            continue
        vdiffs = [i for i in range(len(v_gold)) if v_gold[i] != v_neg[i]]
        if len(vdiffs) != 1:
            continue
        if not (2 <= len(v_gold) <= 5):
            continue
        key = spec["problem"]
        if key in seen:
            continue
        seen.add(key)

        rid = f"grnf_{idx:04d}"
        ckind, cargs = spec["checker"]
        # render every format for this accepted problem
        for fmt in ALL_FORMATS:
            g_blocks, n_blocks = _format_blocks(spec, fmt)
            _assert_one_perturbed(g_blocks, n_blocks, spec["topic"], fmt)
            if fmt in GENERIC_FORMATS:
                for b in g_blocks + n_blocks:
                    leaks = G._generic_violations(b)
                    assert not leaks, (f"{spec['topic']}/{fmt} domain leak in generic format: "
                                       f"{b!r} -> {leaks}")
            row = {
                "id": rid, "topic": spec["topic"], "format": fmt,
                "n_turns": len(g_blocks), "problem": spec["problem"],
                "gold_plan": g_blocks, "gold_answer": str(gold),
                "neg_plan": n_blocks, "neg_answer": str(neg),
                "checker_kind": ckind, "checker_args": cargs, "club": 1,
            }
            out[fmt].append(row)
        per_topic[spec["topic"]] += 1
        idx += 1

    if len(out["verbose"]) < n:
        raise SystemExit(f"only generated {len(out['verbose'])}/{n} rows; raise budget/pools")
    return out, per_topic


# =========================================================================================
# ALIGNMENT + GENERICNESS ASSERTIONS (hard) across all format files
# =========================================================================================
def assert_aligned(out):
    formats = list(out)
    ref = out[formats[0]]
    n = len(ref)
    for f in formats:
        assert len(out[f]) == n, f"format {f} has {len(out[f])} rows, expected {n}"
    for i in range(n):
        ids = {out[f][i]["id"] for f in formats}
        probs = {out[f][i]["problem"] for f in formats}
        golds = {out[f][i]["gold_answer"] for f in formats}
        negs = {out[f][i]["neg_answer"] for f in formats}
        topics = {out[f][i]["topic"] for f in formats}
        assert len(ids) == 1, f"row {i}: id diverged across formats: {ids}"
        assert len(probs) == 1, f"row {i}: problem diverged across formats (id {ids})"
        assert len(golds) == 1, f"row {i}: gold_answer diverged across formats (id {ids})"
        assert len(negs) == 1, f"row {i}: neg_answer diverged across formats (id {ids})"
        assert len(topics) == 1, f"row {i}: topic diverged across formats (id {ids})"
        g = next(iter(golds)); ne = next(iter(negs))
        assert g != ne, f"row {i}: gold_answer == neg_answer ({g!r})"
    # genericness: 0 domain leaks for the 4 generic formats; concrete is exempt (it NAMES things)
    for f in GENERIC_FORMATS:
        for r in out[f]:
            for b in r["gold_plan"] + r["neg_plan"]:
                assert not G._generic_violations(b), \
                    f"format {f} row {r['id']} leaked: {b!r} -> {G._generic_violations(b)}"


def main():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)             # so `import gen_grounding_blocks` works from any CWD

    ap = argparse.ArgumentParser(description="Render the SAME 100 grounding problems' plans in "
                                             ">=5 formats, one row-aligned JSONL per format.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default=".", help="directory for grounding_fmt_<format>.jsonl files")
    ap.add_argument("--prefix", default="grounding_fmt_", help="output filename prefix")
    args = ap.parse_args()

    out, per_topic = build_rows(args.seed, args.n)
    assert_aligned(out)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for fmt in ALL_FORMATS:
        path = os.path.join(args.out_dir, f"{args.prefix}{fmt}.jsonl")
        with open(path, "w") as fh:
            for r in out[fmt]:
                fh.write(json.dumps(r) + "\n")
        written.append(path)
        print(f"wrote {len(out[fmt])} rows (format={fmt}) -> {path}")

    print(f"  per-topic : {dict(sorted(per_topic.items()))}")
    print(f"[align] OK: {len(ALL_FORMATS)} formats x {len(out['verbose'])} rows share identical "
          "id/topic/problem/gold_answer/neg_answer; differ only in plan phrasing.")
    print(f"[generic] OK: 0 domain leaks across the {len(GENERIC_FORMATS)} generic formats "
          f"({', '.join(GENERIC_FORMATS)}); concrete is the naming reference upper bound.")
    print("[run] feed each file to the frozen probe, e.g.:")
    for p in written:
        print(f"    python grounding_test.py --data {p} --out out_{os.path.basename(p)[:-6]}")


if __name__ == "__main__":
    main()