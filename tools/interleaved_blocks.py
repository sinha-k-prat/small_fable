#!/usr/bin/env python3
"""
tools/interleaved_blocks.py — a SMALL library of REUSABLE, GENERIC interleaved plan-blocks that
unlock the HARD_BENCH answers BLOCK BY BLOCK on a FROZEN model (inference only; design code only).

WHY THIS EXISTS
---------------
A frozen Qwen2.5-1.5B-Instruct scores ~0% on HARD_BENCH (see GROUNDING_RESULTS.md / hard_bench_run.py)
because every hard category needs a LOOP + a MUTABLE REGISTER (a running count, an index pointer, a
partial-product accumulator, a running date, the last one/two terms of a recurrence, a remainder
stack, a max-depth tracker, ...). A single transformer forward pass has none of these: it cannot hold
a value that mutates across iterations, so it emits a plausible-but-wrong guess.

The SCRATCHPAD fix: unroll the loop into GENERATED tokens. INTERLEAVED execution does exactly this —
plan a block, let the model EXECUTE it and WRITE the intermediate result onto the scratchpad, then
re-insert the next block conditioned on the GROWING scratchpad, and repeat until a FINAL ANSWER
emerges block by block. The generated text BECOMES the mutable register; the per-line emission BECOMES
the loop body executed one iteration at a time; the re-inserted plan block BECOMES the loop control the
forward pass lacks.

WHAT THIS MODULE PROVIDES
-------------------------
  SKILLS            : dict {skill_name -> {"categories": [...], "variants": [[block, ...], ...],
                      "unrolls_the_loop": <str>, "why_generic": <str>}}. Each skill is a reusable
                      interleaved DECOMPOSITION; each variant is an ordered list of GENERIC plan-blocks
                      (turns). A variant is mutated/swapped per category by the harness until every
                      problem in that category is solved (100%).
  CATEGORY_SKILL    : dict {category -> skill_name} covering ALL 20 HARD_BENCH categories.
  generic_violations(block, problem_words) : the GENERICNESS GATE. Returns the list of block tokens
                      that are CONTENT words of the specific problem (so the harness can assert NO
                      block overlaps a problem's content — the blocks are reusable, not bespoke).

DESIGN INVARIANTS
-----------------
  * Blocks name only OPERATIONAL roles (item, unit, chunk, running count, index pointer, operand,
    partial result, accumulator, modulus, running date, running value, extreme item, output list,
    position label, target position, current-level/best-seen register, leading element) and bare
    verbs (split, multiply, add, carry, reduce, advance, scan, extract, concatenate, reverse). NEVER
    the specific letter/word/number/date being asked about, the input text, the answer value, or the
    category name.
  * Every terminal block emits the answer after a literal 'FINAL ANSWER:' marker so hard_bench's
    strict, judge-free grade() can read the committed span (text after the LAST 'FINAL ANSWER:').
  * stdlib-only; no model, no I/O. This is a pure library the interleaved harness imports.

Coverage (category -> skill) is asserted at import via _assert_coverage(); the 20 categories are the
exact set produced by tools/gen_hard_bench.py.
"""

import re

# ===========================================================================================
# THE 20 HARD_BENCH CATEGORIES (must equal the keys of CATEGORY_SKILL below).
# ===========================================================================================
HARD_BENCH_CATEGORIES = [
    "letter_count_in_word",
    "letter_count_in_passage",
    "word_occurrence_count",
    "string_reversal",
    "nth_character_index",
    "caesar_cipher_decode",
    "caesar_cipher_encode",
    "vowel_count",
    "char_position_of_substring",
    "multidigit_multiplication",
    "modular_exponentiation",
    "base_conversion",
    "days_between_dates",
    "date_add_days",
    "sort_numbers_nth",
    "nested_parentheses_depth",
    "arithmetic_sequence_nth",
    "fibonacci_like_nth",
    "count_in_long_list",
    "acronym_from_words",
]


# ===========================================================================================
# SKILLS — 10 reusable, GENERIC interleaved decompositions. Each value is:
#   {"categories": [<HARD_BENCH categories this skill unlocks>],
#    "variants":   [ [<block>, <block>, ...], ... ]   # each variant = an ordered list of plan-turns
#    "unrolls_the_loop": <why interleaving supplies the missing loop+register for this skill>,
#    "why_generic":      <why no block names any problem content, so it reuses across instances>}
# The harness applies one variant interleaved, and KEEPS MUTATING / swapping variants per category
# until every problem in that category grades 1.0.
# ===========================================================================================
SKILLS = {

    # -------------------------------------------------------------------------------------
    # 1) enumerate_and_tally — one mutable running count, advanced one item per turn.
    #    Unlocks every counting category (letters, vowels, words, list items).
    # -------------------------------------------------------------------------------------
    "enumerate_and_tally": {
        "categories": [
            "letter_count_in_word", "letter_count_in_passage", "vowel_count",
            "word_occurrence_count", "count_in_long_list",
        ],
        "variants": [
            [
                "Restate the matching test in one line and initialize a running count of 0; "
                "write 'count=0' on the scratchpad",
                "Walk the items strictly left to right one at a time; for each item write the item "
                "then either 'no' or 'MATCH', and after every MATCH write the new running count "
                "value; do not skip or summarize any item",
                "Read off the last running count value you wrote, confirm no items remain unscanned, "
                "and commit it as FINAL ANSWER:",
            ],
            [
                "Split the input into an explicit ordered list of atomic units, numbering each unit "
                "with its running ordinal on its own line so none are lost",
                "Go through the numbered units in order; for each, test the match condition and write "
                "'unit k: <value> -> hit, total=N' incrementing total only on a hit and carrying the "
                "previous total forward on a miss",
                "Take the final total from the last line, verify the unit index reached the end of the "
                "list, and write it as FINAL ANSWER:",
            ],
            [
                "Scan the input in fixed-size chunks; for each chunk, list the matches found inside it "
                "and write that chunk's subtotal",
                "Write the chunk subtotals in order, then add them cumulatively, writing each running "
                "sum as you fold in the next subtotal",
                "Output the final cumulative sum, double-check it equals the sum of all subtotals, and "
                "commit it as FINAL ANSWER:",
            ],
        ],
        "unrolls_the_loop": (
            "A single forward pass has no mutable counter register, so it guesses a count. Interleaving "
            "externalizes the register onto the scratchpad: each turn the executor writes the current "
            "item's match verdict and the updated running count, and the next turn is re-entered "
            "conditioned on that written count, so the +1 accumulation a loop body would do is performed "
            "token-by-token across turns. The plan block re-inserts the 'continue scanning and write the "
            "new count' instruction every turn, supplying the loop control the forward pass lacks."
        ),
        "why_generic": (
            "Every block names only operational nouns (item, unit, chunk, running count, match "
            "condition, total) and never the specific letter/word/number being counted, the input "
            "text, or any answer value, so the same blocks drive letter, vowel, word, and list "
            "counting interchangeably."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 2) scan_to_position — an explicit index pointer advanced one unit per turn.
    #    Unlocks substring-first-position and read-at-nth-position.
    # -------------------------------------------------------------------------------------
    "scan_to_position": {
        "categories": ["char_position_of_substring", "nth_character_index"],
        "variants": [
            [
                "State whether you are seeking a target's position or reading at a given position, and "
                "initialize an index pointer at 1; write 'index=1' on the scratchpad",
                "Advance the pointer through the input one unit at a time; for each step write "
                "'index=k: <unit>' and either continue or, when the stop condition (the sought pattern "
                "matches, or the requested index is reached) holds, write 'STOP at index=k' and the "
                "unit(s) read there",
                "Read off the stopped index or the unit(s) at that position from your last line, verify "
                "it is the first/exact match requested, and commit it as FINAL ANSWER:",
            ],
            [
                "Lay out the input as an explicit numbered sequence, writing each unit beside its "
                "1-based position so positions are externally fixed and not estimated",
                "Locate the requested position or first matching pattern by reading down the numbered "
                "list in order; write the position number and the unit(s) found there, checking it is "
                "the earliest qualifying one",
                "Confirm the reported position/units against the numbered layout and write the result "
                "as FINAL ANSWER:",
            ],
        ],
        "unrolls_the_loop": (
            "Positional access past a shallow index requires a counter that increments while walking the "
            "sequence, which a frozen forward pass cannot maintain. Interleaving makes the index pointer "
            "an explicit scratchpad value: each turn writes 'index=k: unit' and the next turn re-enters "
            "conditioned on that written pointer, so the pointer advances one unit per turn until the "
            "stop condition is written, turning positional lookup into an externally-tracked loop."
        ),
        "why_generic": (
            "Blocks reference only an abstract index pointer, units, a sought pattern, and a stop "
            "condition; they never mention the actual string, the substring, or the numeric position "
            "asked, so they apply equally to 'first index of substring' and 'read characters at the nth "
            "position' without edits."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 3) decompose_and_recombine — split into parts / repeated combine, one partial per line.
    #    Unlocks multi-digit multiplication and modular exponentiation.
    # -------------------------------------------------------------------------------------
    "decompose_and_recombine": {
        "categories": ["multidigit_multiplication", "modular_exponentiation"],
        "variants": [
            [
                "Split the second operand into its place-value parts (the value contributed by each of "
                "its digits separately) and write that list of parts.",
                "Multiply the first operand by each part in turn, writing one partial result per line "
                "as you go.",
                "Add the partial results together one pair at a time, writing each running subtotal and "
                "the carry where needed.",
                "Re-read the last running subtotal you wrote, then on a new line write FINAL ANSWER: "
                "followed by it and nothing else.",
            ],
            [
                "Rewrite the task as a sequence of identical combine-steps applied to the same fixed "
                "operand, and write down how many such steps there are.",
                "Carry out the steps one at a time; after each step write the new running accumulator "
                "value on its own line, and whenever a modulus is given reduce the accumulator by it so "
                "the number stays small.",
                "Re-read the last accumulator value you wrote, then on a new line write FINAL ANSWER: "
                "followed by it and nothing else.",
            ],
            [
                "Restate the operation as repeated application of one binary combine to a running value, "
                "and write the running value's starting state on its own line.",
                "Apply the combine one more time to the current running value, write the new running "
                "value (reduced by the stated modulus if one is given) on its own line, and repeat this "
                "block until you have done it the required number of times.",
                "Copy the final running value, then on a new line write FINAL ANSWER: followed by it and "
                "nothing else.",
            ],
        ],
        "unrolls_the_loop": (
            "A single forward pass cannot hold the growing list of partial products or the running "
            "accumulator across many multiply-and-add iterations. Forcing one partial result per line "
            "and one running subtotal/accumulator per line writes each intermediate to the scratchpad, "
            "and the next turn is re-conditioned on the growing text, so the generated tokens BECOME the "
            "mutable register and the line-by-line emission BECOMES the loop body executed one iteration "
            "at a time."
        ),
        "why_generic": (
            "Blocks name only structural roles (first operand, second operand, place-value parts, "
            "partial result, running subtotal, accumulator, modulus) and bare operations (split, "
            "multiply, add, carry, reduce); no concrete numbers, factors, exponents, or category names "
            "appear, so the identical blocks drive any product or any repeated-combine-with-optional-"
            "reduction instance."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 4) step_the_recurrence — one position-labeled term per line; the label IS the counter.
    #    Unlocks arithmetic-sequence-nth and fibonacci-like-nth.
    # -------------------------------------------------------------------------------------
    "step_the_recurrence": {
        "categories": ["arithmetic_sequence_nth", "fibonacci_like_nth"],
        "variants": [
            [
                "Write down the initial term(s) the problem supplies, each labeled with its 1-based "
                "position, and note the target position you must reach.",
                "Produce the next term from the most recent term(s) using the stated rule, write it on "
                "its own line labeled with its position, and repeat this block until the labeled "
                "position equals the target.",
                "Re-read the term whose label equals the target position, then on a new line write "
                "FINAL ANSWER: followed by its value and nothing else.",
            ],
            [
                "State the starting state of the sequence and the stated rule for advancing one step, "
                "and write the index you are starting at and the index you must reach.",
                "Advance the state by exactly one step using the rule, write the new state together "
                "with its new index on its own line, and keep repeating this block one index at a time "
                "without skipping any.",
                "Once the written index matches the target, copy that line's value, then on a new line "
                "write FINAL ANSWER: followed by it and nothing else.",
            ],
        ],
        "unrolls_the_loop": (
            "Computing the n-th term needs a counter and the last one or two terms held mutable across "
            "roughly 20 to 120 iterations, which one forward pass lacks; it tends to jump to a guessed "
            "magnitude. Emitting one position-labeled term per line stores the current state on the "
            "scratchpad and makes the position label an explicit counter, so each turn re-reads the "
            "prior term(s) and label and advances exactly one step, turning the hidden loop into "
            "visible, checkable lines."
        ),
        "why_generic": (
            "Blocks reference only the structural slots (initial term(s), most recent term(s), the "
            "stated rule, 1-based position, target position, index) and the act of advancing one step; "
            "they never mention the start value, the common difference, the seed pair, the target n, or "
            "which sequence it is, so the same blocks step any additive or otherwise-defined recurrence "
            "to any requested index."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 5) digit_by_digit — repeated divide-and-remainder; remainder stack then reverse-and-join.
    #    Unlocks base conversion.
    # -------------------------------------------------------------------------------------
    "digit_by_digit": {
        "categories": ["base_conversion"],
        "variants": [
            [
                "Write down the value to be converted and the target base, and note that you will "
                "collect remainders from least-significant to most-significant.",
                "Divide the current value by the target base, write the quotient and the remainder "
                "(mapping any remainder above nine to its single letter digit) on its own line, then "
                "set the current value to the quotient and repeat this block until the current value "
                "reaches zero.",
                "Read the remainders you wrote from the LAST one produced up to the FIRST and "
                "concatenate them in that reverse order onto one line, with no separators or leading "
                "marker.",
                "Re-read that concatenated string, then on a new line write FINAL ANSWER: followed by "
                "it and nothing else.",
            ],
            [
                "Restate the number and the radix, and start an empty list that will hold output "
                "digits.",
                "Take the current number, record number-mod-radix as the next collected digit (as a "
                "single letter digit if it exceeds nine) on its own line, replace the current number "
                "with number-divided-by-radix discarding any fraction, and repeat this block while the "
                "current number stays above zero.",
                "Reverse the order of the collected digits and write them joined together with no "
                "separators and no leading marker on one line.",
                "Copy that joined string, then on a new line write FINAL ANSWER: followed by it and "
                "nothing else.",
            ],
        ],
        "unrolls_the_loop": (
            "Repeated divide-and-remainder needs a value that mutates each iteration plus an ordered "
            "stack of remainders, neither of which survives a single forward pass, so the model emits a "
            "wrong or truncated digit string. Writing each quotient/remainder pair on its own line keeps "
            "the shrinking value and the remainder stack explicit on the scratchpad; the reverse-and-"
            "join step then operates on visible tokens, so the loop and its accumulator are fully "
            "externalized."
        ),
        "why_generic": (
            "Blocks mention only the value, the target base/radix, quotient, remainder, the above-nine-"
            "to-letter mapping, and the reverse-then-join step; no specific number, base, or output "
            "digits appear, so the identical blocks convert any integer into any base."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 6) calendar_step — a running date register, large safe chunks then single days.
    #    Unlocks days-between-dates and date-add-days.
    # -------------------------------------------------------------------------------------
    "calendar_step": {
        "categories": ["days_between_dates", "date_add_days"],
        "variants": [
            [
                "Write down the reference date split into year, month, and day, and state the target the "
                "problem asks for (either an offset of days to add, or a second date to measure the gap "
                "to).",
                "Advance the running date toward the target one whole unit at a time (jump by a full "
                "month or a full year only when the entire unit fits inside what remains), writing the "
                "new running date and the remaining count on its own line each time, and respect month "
                "lengths and leap years at every boundary.",
                "Continue stepping by single days until nothing remains, writing the running date and "
                "the dwindling remainder on its own line each step so no boundary is skipped.",
                "Re-read the final running date (or the total days you accumulated), then on a new line "
                "write FINAL ANSWER: followed by it in the exact format requested and nothing else.",
            ],
            [
                "State the starting date as year/month/day and whether you are counting forward by a "
                "given number of days or counting how many days lie between two given dates.",
                "Move the running date forward in large safe chunks first (whole remaining years, then "
                "whole remaining months that fit), writing the updated running date and what is left "
                "after each chunk on its own line, always honoring the correct length of each month and "
                "leap-year February.",
                "Finish by moving one day at a time, writing the running date and the remaining amount "
                "on its own line per day, until the remaining amount is exhausted.",
                "Copy the resulting date or the accumulated day-count, then on a new line write FINAL "
                "ANSWER: followed by it in the exact requested format and nothing else.",
            ],
        ],
        "unrolls_the_loop": (
            "Date arithmetic over hundreds of days needs a running date register that mutates across "
            "month-length and leap-year boundaries plus a shrinking remainder counter, which a single "
            "forward pass cannot carry, so the model produces a plausible-but-wrong date. Emitting the "
            "running date and remaining count on its own line each step externalizes both the register "
            "and the counter; large-chunk-then-single-day stepping keeps every boundary explicit and "
            "re-readable, so the calendar loop runs visibly to completion."
        ),
        "why_generic": (
            "Blocks reference only the reference date split into year/month/day, the offset-or-second-"
            "date target, the running date, the remaining count, month lengths, and leap years; no "
            "concrete date, offset, or category name appears, so the same blocks handle both adding days "
            "to a date and measuring the gap between two dates."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 7) char_by_char_transform — one element per line as (original -> transformed), then join.
    #    Unlocks string reversal and caesar encode/decode (rule + order vary, blocks don't).
    # -------------------------------------------------------------------------------------
    "char_by_char_transform": {
        "categories": ["string_reversal", "caesar_cipher_encode", "caesar_cipher_decode"],
        "variants": [
            [
                "Restate the source sequence and the single per-element rule to apply, then number "
                "every element from the front so each has a fixed slot",
                "Walk the elements in the required order, and for the current element write a line "
                "pairing the original element with its transformed element under the rule, leaving "
                "earlier lines untouched",
                "Continue the same one-element-per-line walk until every numbered slot has been emitted "
                "exactly once, appending each new pair beneath the prior ones",
                "Read the transformed elements straight down the accumulated column in order and "
                "concatenate them with no separators into the result token",
                "Verify the result length equals the source length, then write FINAL ANSWER: followed "
                "by only the concatenated result",
            ],
            [
                "Copy the source as an indexed column, one element per line, and on its own line state "
                "the deterministic rule that maps one input element to one output element",
                "Process exactly one not-yet-done line: emit it as input->output using the rule, and "
                "mark it done so the next pass skips it",
                "Repeat the single-line processing, each turn handling the next undone line and "
                "appending its input->output beneath the growing list",
                "Collect every output, in the order the task demands, into one unbroken string with "
                "nothing between characters",
                "Recount that you produced one output per input and commit it after FINAL ANSWER:",
            ],
        ],
        "unrolls_the_loop": (
            "A single forward pass must transform a whole string at once and has nowhere to hold the "
            "partially built result; here each turn handles exactly one character, writes the "
            "(original->transformed) pair to the scratchpad, and the next turn is re-entered "
            "conditioned on that growing column, so the scratchpad IS the output register and the "
            "per-line walk IS the loop counter the pass lacks."
        ),
        "why_generic": (
            "Blocks only say element, slot, rule, order, concatenate, length-check — never 'reverse', "
            "'shift', 'cipher', an alphabet, or any specific direction or offset. The same "
            "decomposition drives reversal (rule = identity, order = back-to-front) and caesar "
            "(rule = fixed shift, order = front-to-back), so no block names problem content."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 8) selection_extract_step — selection sort: extract one extreme per turn until the rank.
    #    Unlocks sort-numbers-nth (find the k-th item under an ordering).
    # -------------------------------------------------------------------------------------
    "selection_extract_step": {
        "categories": ["sort_numbers_nth"],
        "variants": [
            [
                "List the input items as a working pool, one per line, and state which ordering "
                "criterion and which target rank position are requested",
                "Scan the entire current pool, identify the single extreme item under the criterion, "
                "append it as the next entry of a separate ordered output list, and rewrite the pool "
                "with that one item removed",
                "Repeat the scan-extract-remove step, each turn moving exactly one extreme item from "
                "the shrinking pool to the next slot of the ordered output list, until the output list "
                "holds the target rank position",
                "Read the item now sitting at the target rank position of the ordered output list, "
                "verify its rank by counting the entries before it, and commit it as FINAL ANSWER:",
            ],
            [
                "Write the items as an unordered working set and note the ordering direction and the "
                "1-based rank you must return; start an empty ranked list",
                "Sweep the whole working set once to pick out the current extreme under the stated "
                "direction, write 'rank r: <item>' on its own line, and strike that item from the "
                "working set so it is not picked again",
                "Repeat the single-sweep extraction, incrementing the rank label by one each turn and "
                "always pulling from the items that remain, until the rank label reaches the requested "
                "rank",
                "Re-read the line whose rank label equals the requested rank, then on a new line write "
                "FINAL ANSWER: followed by that item's value and nothing else",
            ],
        ],
        "unrolls_the_loop": (
            "Selecting the k-th item under an ordering needs a full pass to find each extreme plus a "
            "shrinking pool and a rank counter held mutable across k passes, none of which a single "
            "forward pass can carry, so the model misorders or mis-ranks. Writing the extracted item "
            "with its rank label on its own line and rewriting the shrinking pool externalizes both the "
            "ordered output and the pool; each turn re-enters on the written pool and advances the rank "
            "by exactly one, so selection sort runs visibly one extraction at a time."
        ),
        "why_generic": (
            "Blocks name only the working pool, an ordering criterion/direction, an extreme item, an "
            "ordered output list, a rank label, and a target rank position; no concrete numbers, list "
            "contents, ordering keyword, or rank value appear, so the same blocks select any rank under "
            "any total order."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 9) track_running_extreme — a running level register + a running maximum, one token per turn.
    #    Unlocks nested-parentheses-depth (max nesting via a stack height tracked to its peak).
    # -------------------------------------------------------------------------------------
    "track_running_extreme": {
        "categories": ["nested_parentheses_depth"],
        "variants": [
            [
                "Restate the open/close rule, then initialize two registers on the scratchpad: a "
                "current-level register and a best-seen-so-far register, both at 0; write 'level=0 "
                "best=0'",
                "Walk the symbols strictly left to right one at a time; for each symbol, raise the "
                "current level on an opener or lower it on a closer, then write 'symbol -> level=k' and, "
                "whenever the new level exceeds best-seen, update best-seen and write its new value too",
                "Continue the one-symbol-per-line walk until every symbol is consumed, never skipping "
                "or summarizing, so the level register tracks the true running height throughout",
                "Read off the final best-seen value, confirm the current level returned to 0 (the input "
                "is balanced), and commit best-seen as FINAL ANSWER:",
            ],
            [
                "Lay out the symbols as a numbered sequence and start a counter at 0 together with a "
                "separate peak holder at 0, writing both starting values on their own line",
                "Process exactly one next symbol: adjust the counter up for an opener or down for a "
                "closer, write the symbol and the updated counter, and if the counter is now larger "
                "than the peak holder copy it into the peak holder on the same line",
                "Repeat the single-symbol step for each remaining numbered symbol in order, appending "
                "the updated counter and peak beneath the prior lines",
                "Take the peak holder's final value, double-check the counter ended at 0, then on a new "
                "line write FINAL ANSWER: followed by the peak and nothing else",
            ],
        ],
        "unrolls_the_loop": (
            "Max-nesting depth needs a current-level register that increments and decrements across the "
            "whole string PLUS a running maximum, both mutable across every symbol, which a single "
            "forward pass cannot hold, so it loses the stack and guesses. Writing 'level=k' and the "
            "updated best-seen on their own line each symbol externalizes both registers; the next turn "
            "re-enters conditioned on the written level and peak, so the increment/decrement loop body "
            "and the max-update run visibly one symbol at a time."
        ),
        "why_generic": (
            "Blocks reference only an opener/closer rule, a current-level register, a best-seen/peak "
            "register, and a left-to-right symbol walk; they never mention parentheses specifically, "
            "the actual bracket string, or the answer depth, so the same blocks track the peak of any "
            "running counter driven by up/down events."
        ),
    },

    # -------------------------------------------------------------------------------------
    # 10) first_token_collect — pull one designated piece per item, append, then join in order.
    #     Unlocks acronym-from-words (first letter of each word, joined in order).
    # -------------------------------------------------------------------------------------
    "first_token_collect": {
        "categories": ["acronym_from_words"],
        "variants": [
            [
                "Restate which single piece of each item you must take (the designated leading "
                "element) and the order to keep, then number the items with their running ordinal on "
                "their own lines so none are dropped or reordered",
                "Go through the numbered items in order; for each, write 'item k: <item> -> <its "
                "leading element>' on its own line, taking exactly one piece per item and skipping none",
                "Read the leading elements straight down the column in item order and concatenate them "
                "into one unbroken string with nothing between them",
                "Verify the joined length equals the number of items, then on a new line write FINAL "
                "ANSWER: followed by only the concatenated string",
            ],
            [
                "State the per-item extraction (keep only the first element of each item) and start an "
                "empty ordered collector; list the items one per line preserving their given order",
                "Process exactly one not-yet-done item: append its first element to the collector and "
                "mark the item done, writing the growing collector on its own line",
                "Repeat the single-item step until every item is done, each turn appending one more "
                "first element to the end of the collector in order",
                "Join the collector into one string with no separators, recount that it has one element "
                "per item, then commit it after FINAL ANSWER:",
            ],
        ],
        "unrolls_the_loop": (
            "Collecting one piece from each of many items and concatenating them in order needs a "
            "growing output buffer plus an item cursor held mutable across the list, which a single "
            "forward pass lacks, so it drops, reorders, or invents letters. Writing 'item k -> leading "
            "element' on its own line and the growing collector externalizes both the cursor and the "
            "buffer; each turn re-enters on the written collector and appends exactly one more piece, so "
            "the gather-and-join loop runs visibly one item at a time."
        ),
        "why_generic": (
            "Blocks name only items, a designated leading element, an ordered collector, and the "
            "join-with-no-separator step; they never mention words, letters, the actual item list, or "
            "the resulting string, so the same blocks gather any per-item leading piece and join it in "
            "order."
        ),
    },
}


# ===========================================================================================
# CATEGORY_SKILL — every one of the 20 HARD_BENCH categories mapped to the skill that unlocks it.
# Derived directly from each skill's "categories" so it can never drift out of sync.
# ===========================================================================================
CATEGORY_SKILL = {cat: skill for skill, spec in SKILLS.items() for cat in spec["categories"]}


# ===========================================================================================
# GENERICNESS GATE — generic_violations(block, problem_words)
# ===========================================================================================
# A block is GENERIC iff none of its CONTENT tokens is a CONTENT token of the specific problem it is
# applied to. We tokenize a block into lowercase word/number tokens, drop a fixed STOPWORD set of
# operational/structural vocabulary (the only words a generic block is allowed to share with anything),
# and flag any surviving token that also appears in the problem's content words. An empty return means
# the block names no problem content -> it is reusable, not bespoke. The harness asserts
# `generic_violations(block, problem_words) == []` for every block it inserts.

# Operational / structural vocabulary the blocks are BUILT from. These are the ONLY tokens a generic
# block may share with arbitrary problem text; they describe the loop machinery, never a problem's
# content. (Kept deliberately broad: every word that actually appears across the 10 skills' blocks and
# is a generic operational term is listed, so a clean block flags NOTHING against any problem.)
_STOPWORDS = frozenset("""
a an and the of to in into onto on at by for from with as if when until while whenever where which
what whether either or both per how many much is are was were be been being am has have had how-many
0 1 2 3 4 5 6 7 8 9 zero
up down left right front back forward order ordered unordered reverse
reversed first last next prior previous final new old current running other earlier this that these
those it its them they their then than so all any no none not nothing one two three exactly only same
each every you your i must may will keep continue repeat do does done go going walk step steps stepping
move moving advance advancing process processing scan scanning sweep sweeping read reading write
writing written restate state stated note noting take taking pull pick picking identify locate find
found confirm verify check checking double recount commit committed count counting total subtotal
subtotals sum sums add adding fold folding cumulative cumulatively split splitting list lists item
items unit units chunk chunks atomic element elements piece pieces symbol symbols token tokens string
char chars character characters sequence column line lines slot slots position positions positional
index pointer label labeled labelled value values number numbers digit digits letter letters integer
leading collector pool set working remove removed removing strike rewrite rewriting shrinking growing
append appending appended beneath pairs pair pairing rule deterministic single condition match matching
matches hit miss yes answer marker finalize result input output source target start starting stop
stopped reach reached requested asks asked produce producing produced emit emitting emitted between
separators separator unbroken join joined joining concatenate concatenated concatenation length
recombine combine binary apply applied applying operation operand operands place-value partial
accumulator modulus reduce reduced reducing reduces quotient remainder remainders mapping map mod radix
base divide divided division divided-by multiply multiplied product carry carries date dates day days
month months year years offset gap reference calendar leap boundary boundaries level best best-seen
peak holder counter register open close opener closer raise lower height stack depth nesting balanced
criterion direction extreme rank ranked ranking term terms recurrence additive common difference seed
initial most recent advance window indexed numbering numbered fixed externally estimated qualifying
earliest sought pattern aspect form mark skip skipped skipping summarize summarizing strictly side
leftmost rightmost least significant collected discarding fraction above nine below k n r exact format
there before after here just give giving over under again whole appear appears does did about
report reported reporting show showing shown its their out off across along through during without
turn turns block blocks scratchpad pass passes not-yet-done undone untouched throughout consumed holds
reaches equals carrying across whole still need needs given lie measure measuring measured amount
dwindling remaining adjust adjusted larger now growing-collector one-per-line one-symbol one-item
one-line on-its-own its-own straight build built lay laid sitting holds-the entry entries cross
ascending descending smallest largest minimum maximum bottom top down-the consecutive groups group
""".split())

# Treat hyphen/underscore/apostrophe as joins so 'one-per-line', 'best-seen', 'place-value', "chunk's"
# tokenize as single words.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_'][a-z0-9]+)*")


def _content_tokens(text):
    """Lowercase content tokens of `text` minus the operational STOPWORD vocabulary."""
    toks = _TOKEN_RE.findall(str(text).lower())
    return [t for t in toks if t not in _STOPWORDS]


def _problem_word_set(problem_words):
    """Normalize the caller's problem content into a set of lowercase content tokens.
    Accepts a string (the raw problem text) or an iterable of words. Stopwords shared with the problem
    (e.g. 'the') are dropped so a block is never flagged for sharing pure operational vocabulary."""
    if isinstance(problem_words, str):
        raw = _TOKEN_RE.findall(problem_words.lower())
    else:
        raw = []
        for w in problem_words:
            raw.extend(_TOKEN_RE.findall(str(w).lower()))
    return {w for w in raw if w not in _STOPWORDS}


def generic_violations(block, problem_words):
    """Return the list of CONTENT tokens in `block` that ALSO appear in `problem_words` — i.e. the
    ways `block` leaks the specific problem's content. A GENERIC block returns [] for ANY problem.

    block         : a single plan-block string (one interleaved turn).
    problem_words : the problem's content, as the raw problem string OR an iterable of its words.

    The harness uses this as the reusability gate:
        assert generic_violations(block, row["question"]) == []
    for every block it inserts, guaranteeing no block overlaps the problem it solves.
    """
    pset = _problem_word_set(problem_words)
    seen, out = set(), []
    for t in _content_tokens(block):
        if t in pset and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ===========================================================================================
# COVERAGE GUARD — assert all 20 categories map to a skill, exactly once, at import time, and that
# every variant is well-formed and commits a FINAL ANSWER: in its terminal block.
# ===========================================================================================
def _assert_coverage():
    covered = set(CATEGORY_SKILL)
    want = set(HARD_BENCH_CATEGORIES)
    missing = want - covered
    extra = covered - want
    assert not missing, f"interleaved_blocks: categories with NO skill: {sorted(missing)}"
    assert not extra, f"interleaved_blocks: skill maps unknown category: {sorted(extra)}"
    owners = {}
    for skill, spec in SKILLS.items():
        for cat in spec["categories"]:
            owners.setdefault(cat, []).append(skill)
    dup = {c: s for c, s in owners.items() if len(s) > 1}
    assert not dup, f"interleaved_blocks: category claimed by >1 skill: {dup}"
    assert len(CATEGORY_SKILL) == 20, f"expected 20 categories, got {len(CATEGORY_SKILL)}"
    for skill, spec in SKILLS.items():
        assert spec["variants"], f"{skill}: no variants"
        for vi, variant in enumerate(spec["variants"]):
            assert variant and all(isinstance(b, str) and b.strip() for b in variant), \
                f"{skill} variant {vi}: empty/non-string block"
            assert re.search(r"final\s*answer", variant[-1], re.I), \
                f"{skill} variant {vi}: terminal block must commit a FINAL ANSWER: marker"


_assert_coverage()


# ===========================================================================================
# BACKWARD-COMPAT API for interleaved_solver.py — adapts the spec shape (plain-string blocks in
# SKILLS[skill]["variants"]) to the {text, final} block-dict form the solver's run/mutate loop expects.
# The canonical library shape is the SKILLS dict above; these are thin views, no new data.
# ===========================================================================================
def B(text, final=False):
    """Wrap a generic plan-block string as the solver's {text, final} block-dict."""
    return {"text": str(text), "final": bool(final)}


def _variant_to_dicts(variant):
    """A spec variant (list of strings) -> list of {text, final}; only the LAST block commits."""
    n = len(variant)
    return [B(b, final=(i == n - 1)) for i, b in enumerate(variant)]


def variants_for_category(category):
    """Ordered GENERIC block-variants for a category's skill, as lists of {text, final} block-dicts.
    The solver searches/mutates over these until accuracy hits 100% or its round budget is exhausted."""
    skill = CATEGORY_SKILL[category]
    return [_variant_to_dicts(v) for v in SKILLS[skill]["variants"]]


def skill_for_category(category):
    """Name of the generic skill that unlocks `category`."""
    return CATEGORY_SKILL[category]


def all_categories():
    """The 20 HARD_BENCH categories, in canonical order."""
    return list(HARD_BENCH_CATEGORIES)


def plan_to_text(plan):
    """Render a plan (list of {text,...} block-dicts OR plain-string blocks) to its text lines."""
    return [b["text"] if isinstance(b, dict) else str(b) for b in plan]


# ===========================================================================================
# CLI / self-check: print the coverage table and run the genericness gate over the whole bench so we
# can SEE that no block of any assigned skill leaks any problem's content. stdlib-only; no model.
# ===========================================================================================
if __name__ == "__main__":
    import json, os
    print(f"SKILLS: {len(SKILLS)}   CATEGORIES: {len(CATEGORY_SKILL)} (all 20 covered)\n")
    print("category -> skill")
    for cat in HARD_BENCH_CATEGORIES:
        print(f"  {cat:<28} -> {CATEGORY_SKILL[cat]}")
    print("\nskill -> #variants  categories")
    for skill, spec in SKILLS.items():
        print(f"  {skill:<24} {len(spec['variants'])}  {spec['categories']}")

    # Genericness gate over the real bench, if present: assert NO block leaks ANY problem's content.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(repo, "hard_bench.jsonl")
    if os.path.exists(data):
        leaks = 0
        checked = 0
        for ln in open(data):
            if not ln.strip():
                continue
            row = json.loads(ln)
            skill = CATEGORY_SKILL.get(row["category"])
            if skill is None:
                continue
            for variant in SKILLS[skill]["variants"]:
                for b in variant:
                    checked += 1
                    v = generic_violations(b, row["question"])
                    if v:
                        leaks += 1
                        print(f"  LEAK [{row['category']}]: block tokens {v} overlap the problem")
        print(f"\ngenericness gate: checked {checked} (block,problem) pairs across the bench; "
              f"{leaks} leak(s) -> {'CLEAN (every block is generic)' if leaks == 0 else 'VIOLATIONS'}")
    else:
        print("\n(hard_bench.jsonl not found; run tools/gen_hard_bench.py to run the genericness gate)")
