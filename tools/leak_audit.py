#!/usr/bin/env python3
"""
leak_audit.py — FROZEN design-time leak audit + green-pool seeding for the plan-block hunt.

WHY THIS EXISTS
---------------
A plan-block must be GENERIC in the strong sense the task demands: its operator must be one that
"can at least work on another instruction", not a direct lift of the task it solves. The library's
existing gate `generic_violations(block, problem)` only checks token overlap with the PROBLEM INSTANCE
(so it never contains '9246'/'897'), which is too weak: "Multiply the first operand by each part"
passes that gate even though it names the multiplication OPERATION and spells the long-multiplication
algorithm. A string/verb blocklist is also wrong (false positives on "add a line", false negatives on
paraphrases like "combine by repeated incrementing").

The robust test is a DISCRIMINATOR (LLM-as-judge, design-time only): read a block IN ISOLATION (no
problem, no category/skill name) and ask "can the single task be recovered from this block alone?" If
yes -> it leaks. This audit records the judge's verdict per (skill, variant). It runs ONCE, offline;
once the green pool is frozen, NO judge runs at hunt time or inference time.

VERDICTS (judge = Claude, reading each block in isolation; see AUDIT below for the per-block reason):
  'generic'  : operation is deferred to "the rule"/"the match test"/"the criterion"; task NOT recoverable.
  'leak'     : names the operation or spells a single-task algorithm; task instantly recoverable. EXCLUDED
               from the green pool *when a generic sibling variant exists for the same skill*.
  'domain'   : intrinsically single-task algorithm with NO generic sibling (base conversion, calendar,
               bracket-depth). Kept as the only available seed, but FLAGGED for an operator-neutral
               rewrite (see REWRITE_TODO). Not a clean pass.

green_variants(skill) -> the variant indices the hunt is allowed to SEED from (leaks dropped iff a
generic sibling exists; domain/borderline kept so no category is left with zero seeds).
"""

# Per-skill, per-variant verdict + the one-line discriminator reason (what gave the task away).
# Index = variant index in tools.interleaved_blocks.SKILLS[skill]['variants'].
AUDIT = {
    "enumerate_and_tally": [
        ("generic", "running count under a deferred 'match test'; can't tell letter/word/vowel count"),
        ("generic", "'match condition' deferred; tally over numbered atomic units"),
        ("generic", "chunk subtotals + cumulative fold; match deferred"),
    ],
    "scan_to_position": [
        ("generic", "advance a pointer until a deferred 'stop condition' (sought pattern OR index)"),
        ("generic", "numbered layout + locate requested position/first match; operation deferred"),
    ],
    "decompose_and_recombine": [
        ("leak", "names 'Multiply' and 'Add' and spells long-multiplication -> multiplication; "
                 "also WRONG for modexp. Generic siblings v1/v2 exist -> EXCLUDED."),
        ("generic", "'repeated application of one binary combine to a running value, reduce by modulus "
                    "if given' -> can't tell multiplication from exponentiation; also CORRECT for modexp"),
        ("generic", "'repeated binary combine to a running value (+ optional modulus)'; operation deferred"),
    ],
    "step_the_recurrence": [
        ("generic", "'next term from recent term(s) using the stated rule' -> can't tell arithmetic vs fib"),
        ("generic", "'advance the state one step using the rule'; rule deferred to the problem"),
    ],
    "digit_by_digit": [
        ("domain", "'divide by the target base, collect remainders, read in reverse' -> base conversion; "
                   "single-task algorithm, NO generic sibling. Kept as sole seed; needs neutral rewrite."),
        ("domain", "same divide/mod/reverse base-conversion algorithm; domain-bound"),
    ],
    "calendar_step": [
        ("domain", "names dates/year-month-day and 'leap years'/'month lengths' -> calendar math; "
                   "domain-bound (defers add-vs-measure but names the whole domain)"),
        ("domain", "same calendar-walk in safe chunks then single days; domain-bound"),
    ],
    "char_by_char_transform": [
        ("generic", "'apply the single per-element rule, then concatenate' -> covers reversal AND both "
                    "Caesar directions without naming any operation"),
        ("generic", "'deterministic rule mapping one input element to one output element'; rule deferred"),
    ],
    "selection_extract_step": [
        ("generic", "'extract the extreme under the criterion, repeat to the target rank'; direction deferred"),
        ("generic", "single-sweep extreme extraction to a 1-based rank; criterion deferred"),
    ],
    "track_running_extreme": [
        ("domain", "names 'opener'/'closer'/'balanced' -> parenthesis depth; control structure "
                   "(running max of a +-1 walk) is generic but the naming pins the task -> needs rewrite"),
        ("domain", "same opener/closer counter + peak holder + balanced check; domain-bound"),
    ],
    "first_token_collect": [
        ("domain", "'take the leading element of each item and concatenate' IS the acronym operation, "
                   "unworded; borderline-domain, only task that uses it -> candidate for neutral rewrite"),
        ("domain", "'append each item's first element to a collector and join'; same acronym operation"),
    ],
}

# Skills where the FROZEN-judge found at least one clean generic variant. For these, any 'leak' variant
# is EXCLUDED from seeding (a generic sibling covers the skill). For 'domain'-only skills, all variants
# are kept (excluding would leave the category with no seed) but they are NOT clean passes.
def green_variants(skill):
    """Variant indices the hunt may SEED from. Drop a 'leak' variant ONLY if a 'generic' sibling exists
    in the same skill; otherwise keep everything (never leave a category seedless)."""
    verdicts = [v for v, _ in AUDIT[skill]]
    has_generic_sibling = "generic" in verdicts
    keep = []
    for i, v in enumerate(verdicts):
        if v == "leak" and has_generic_sibling:
            continue                       # excluded: a clean sibling covers this skill
        keep.append(i)
    return keep


def green_blocks(skill, variants):
    """Convenience: given the live SKILLS[skill]['variants'] list, return only the green-listed variants."""
    return [variants[i] for i in green_variants(skill)]


# Blocks the judge rejected ('leak') or flagged ('domain') that still need an operator-neutral rewrite
# before the hunt can claim a CLEAN generic plan for that category. This is the next design-time task.
REWRITE_TODO = {
    "decompose_and_recombine#v0": "DONE-by-exclusion: v1/v2 already neutral; v0 dropped from green pool.",
    "digit_by_digit": "Rewrite divide/mod/reverse as a deferred 'repeated extract-residue-then-shrink' "
                      "primitive so the base is read from the problem, not named in the plan.",
    "calendar_step": "Push 'leap year/month length' into the problem ('honor the calendar the problem "
                     "states'); keep only the generic 'advance running value in safe chunks then unit steps'.",
    "track_running_extreme": "Replace 'opener/closer/balanced' with 'apply the +1/-1 rule the problem "
                             "states to a running register; track its running maximum'.",
    "first_token_collect": "Replace 'leading element' with 'the one designated piece per item the problem "
                           "names'; defer which piece to the problem.",
}


def summary():
    """One-line-per-skill audit summary for logs/checkpoints."""
    lines = []
    for sk, rows in AUDIT.items():
        verds = [v for v, _ in rows]
        keep = green_variants(sk)
        lines.append(f"{sk:<24} verdicts={verds}  green_seed_variants={keep}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
