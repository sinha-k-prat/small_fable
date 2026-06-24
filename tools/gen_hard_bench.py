#!/usr/bin/env python3
"""
tools/gen_hard_bench.py — generate a HARD, judge-free benchmark calibrated so a FROZEN
Qwen2.5-1.5B-Instruct scores ~0% (the user wants to SEE 0% correct / 100% wrong).

20 categories x 10 examples = 200 rows. Every row carries a CANONICAL answer computed by a
deterministic Python SOLVER (no LLM judge), plus a checker consumable by checkers.reward_for_row.

Row schema EXACTLY:
  {id, category, question, answer, checker_kind, checker_args, why_hard}

checker_kind in {"exact","numeric"}:
  - "numeric": checker_args = {"canonical": {"value": <num>}, "match": {"tolerance": 0}}
               -> reward_for_row -> check_numeric (matches the last number in the FINAL ANSWER span)
  - "exact"  : checker_args = {"gold": "<canonical string>"}
               -> reward_for_row -> check_exact. NOTE on check_exact semantics: if gold matches the
                  numeric pattern it is compared as the LAST number in the answer; otherwise it is a
                  normalized substring test (lowercase, ALL whitespace stripped, trailing '.' removed).
                  We therefore emit string golds as CONTIGUOUS lowercase tokens (no internal spaces) so
                  the match is unambiguous & unique.

Difficulty is CRANKED (long strings, many digits, big lists, deep indices, multi-step transforms)
so the frozen 1.5B's pattern-guess is wrong. stdlib-only, fully seeded -> reproducible.

Writes hard_bench.jsonl (repo root by default). Run:
  python tools/gen_hard_bench.py
"""
import argparse, json, os, random, string, datetime, re

SEED = 20240621
N_PER = 10

ALNUM = string.ascii_lowercase + string.digits


def rand_alnum(rng, n):
    return "".join(rng.choice(ALNUM) for _ in range(n))


def numeric_row(idx, cat, q, value, why):
    """Canonical numeric answer; tolerance 0 -> must be exact."""
    return {
        "id": f"{cat}-{idx:02d}",
        "category": cat,
        "question": q,
        "answer": str(value),
        "checker_kind": "numeric",
        "checker_args": {"canonical": {"value": value}, "match": {"tolerance": 0}},
        "why_hard": why,
    }


def exact_row(idx, cat, q, gold, why):
    """Canonical string answer; gold is a contiguous lowercase token -> unambiguous containment."""
    return {
        "id": f"{cat}-{idx:02d}",
        "category": cat,
        "question": q,
        "answer": gold,
        "checker_kind": "exact",
        "checker_args": {"gold": gold},
        "why_hard": why,
    }


FINAL = "Reply with FINAL ANSWER: "

# ===========================================================================================
# 20 CATEGORY GENERATORS. Each returns a list of 10 rows; each row carries a deterministic SOLVER.
# ===========================================================================================

# 1) letter_count_in_word -------------------------------------------------------------------
def gen_letter_count_in_word(rng):
    pool = [
        ("possessiveness", "s"), ("preposterousness", "s"), ("senselessness", "s"),
        ("entertainment", "e"), ("interdependence", "e"), ("representativeness", "e"),
        ("inimitability", "i"), ("invisibility", "i"), ("antidisestablishmentarianism", "i"),
        ("noncommittally", "n"), ("unconventionalities", "n"), ("misunderstanding", "n"),
        ("reassessment", "s"), ("effervescence", "e"), ("indivisibility", "i"),
    ]
    rng.shuffle(pool)
    rows = []
    for i, (word, letter) in enumerate(pool[:N_PER]):
        ans = word.lower().count(letter.lower())                      # SOLVER
        q = (f"How many times does the letter '{letter}' appear in the word "
             f"'{word}'?\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "letter_count_in_word", q, ans,
            "char-level counting blindspot; high target-letter frequency makes the modal guess of 2/3 wrong"))
    return rows


# 2) letter_count_in_passage ----------------------------------------------------------------
def gen_letter_count_in_passage(rng):
    letters = "etaoinsrl"
    nouns = ["engineers", "envelopes", "sentences", "references", "experiments", "tendencies",
             "elements", "settlements", "tenements", "resentments", "sediments", "departments",
             "investments", "instruments", "treatments", "statements", "arguments", "movements"]
    rows = []
    for i in range(N_PER):
        letter = letters[i % len(letters)]
        words = rng.sample(nouns, 8)
        passage = "The " + " ".join(words) + " were carefully recorded and reviewed."
        ans = passage.lower().count(letter)                           # SOLVER
        q = (f'In the passage below, how many times does the letter "{letter}" appear '
             f'(case-insensitive)?\n"{passage}"\n{FINAL}<integer>.')
        rows.append(numeric_row(i, "letter_count_in_passage", q, ans,
            "running-tally over a long string; exact integer over ~90 chars is unreachable for a 1.5B"))
    return rows


# 3) word_occurrence_count ------------------------------------------------------------------
def gen_word_occurrence_count(rng):
    rows = []
    for i in range(N_PER):
        target = ["cat", "run", "car", "pin", "log"][i % 5]
        distractors = [target + "s", target + "e", "s" + target, target + "ty", target + "ch"]
        n_target = rng.randint(5, 9)
        tokens = ([target] * n_target
                  + rng.choices(distractors, k=rng.randint(8, 12))
                  + rng.choices(["the", "a", "near", "while", "saw", "and", "of", "by"], k=14))
        rng.shuffle(tokens)
        passage = " ".join(tokens)
        ans = sum(1 for w in re.findall(r"[a-z']+", passage.lower()) if w == target)   # SOLVER
        q = (f"How many times does the exact word '{target}' appear in this text "
             f"(count '{target}' only, not variants like '{target}s' or '{target}ty')?\n"
             f'"{passage}"\n{FINAL}<integer>.')
        rows.append(numeric_row(i, "word_occurrence_count", q, ans,
            "needle-in-list counting with prefix/suffix distractors defeats fuzzy matching"))
    return rows


# 4) string_reversal ------------------------------------------------------------------------
def gen_string_reversal(rng):
    rows = []
    for i in range(N_PER):
        n = rng.randint(12, 16)
        s = rand_alnum(rng, n)
        ans = s[::-1]                                                 # SOLVER
        q = (f"Reverse this string exactly, character by character: '{s}'.\n"
             f"{FINAL}<reversed string, no spaces>.")
        rows.append(exact_row(i, "string_reversal", q, ans,
            "random alphanumerics are the worst StringLLM case; exact 12-16 char reversal is unreachable"))
    return rows


# 5) nth_character_index --------------------------------------------------------------------
def gen_nth_character_index(rng):
    # Ask for the 3-character window starting at a deep 1-indexed position. A 3-char contiguous gold
    # (vs a single char) cannot be accidentally satisfied by the checker's substring containment, so a
    # right score requires genuinely locating the index AND reading 3 chars in order.
    rows = []
    for i in range(N_PER):
        n = rng.randint(24, 30)
        s = rand_alnum(rng, n)
        pos = rng.randint(15, n - 2)                                  # deep, 1-indexed; room for 3 chars
        ans = s[pos - 1:pos + 2]                                      # SOLVER (3-char window)
        q = (f"In the string '{s}', what are the 3 characters starting at the {pos}th position "
             f"(1-indexed), read left to right?\n{FINAL}<3 characters, no spaces>.")
        rows.append(exact_row(i, "nth_character_index", q, ans,
            "positional char access past index ~15 in a scrambled string; token boundaries hide it"))
    return rows


# 6) caesar_cipher_decode -------------------------------------------------------------------
def gen_caesar_cipher_decode(rng):
    plains = ["the silver fox", "an amber lantern", "the velvet curtain", "a frozen harbor",
              "the crimson banner", "an ivory tower", "the hollow oak", "a distant beacon",
              "the rusty anchor", "an ancient scroll", "the marble statue", "a golden ember"]
    rng.shuffle(plains)
    rows = []
    for i in range(N_PER):
        shift = rng.choice([5, 6, 7, 8, 9, 10, 11])                  # never 1/3/13
        plain = plains[i % len(plains)]
        cipher = "".join(chr((ord(c) - 97 + shift) % 26 + 97) if c.isalpha() else c for c in plain)
        decoded = "".join(chr((ord(c) - 97 - shift) % 26 + 97) if c.isalpha() else c for c in cipher)
        gold = decoded.replace(" ", "")                              # SOLVER; contiguous (norm strips spaces)
        q = (f"Decode this Caesar cipher (each letter was shifted forward by {shift}; shift it back): "
             f"'{cipher}'.\n{FINAL}<decoded text, lowercase, no spaces>.")
        rows.append(exact_row(i, "caesar_cipher_decode", q, gold,
            "Embers of Autoregression: non-1/3/13 shifts collapse; per-char mod-26 arithmetic fails"))
    return rows


# 7) caesar_cipher_encode -------------------------------------------------------------------
def gen_caesar_cipher_encode(rng):
    words = ["puzzle", "wizard", "oxygen", "rhythm", "jukebox", "vortex", "zephyr", "quartz",
             "syzygy", "crypts", "glyphs", "plywood", "jazzily", "buzzards"]
    rng.shuffle(words)
    rows = []
    for i in range(N_PER):
        shift = rng.choice([5, 6, 7, 8, 9, 10, 11])
        word = words[i % len(words)]
        ans = "".join(chr((ord(c) - 97 + shift) % 26 + 97) for c in word)   # SOLVER
        q = (f"Encode the word '{word}' with a Caesar shift of +{shift} (wrap z->a).\n"
             f"{FINAL}<encoded word>.")
        rows.append(exact_row(i, "caesar_cipher_encode", q, ans,
            "encoding has no familiar target to anchor; wrap-around letters force the mod-26 step"))
    return rows


# 8) vowel_count ----------------------------------------------------------------------------
def gen_vowel_count(rng):
    words = ["unconstitutionally", "incomprehensibilities", "counterrevolutionary",
             "disproportionately", "individualistically", "characteristically",
             "internationalization", "institutionalization", "compartmentalization",
             "interchangeability", "overcompensation", "misappropriation",
             "uncharacteristically", "environmentalists"]
    rng.shuffle(words)
    rows = []
    for i in range(N_PER):
        word = words[i % len(words)]
        ans = sum(1 for c in word.lower() if c in "aeiou")           # SOLVER
        q = (f"How many vowels (a, e, i, o, u) are in the word '{word}'?\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "vowel_count", q, ans,
            "per-character category counting on 15-21 letter words; token embeddings hide it"))
    return rows


# 9) char_position_of_substring -------------------------------------------------------------
def gen_char_position_of_substring(rng):
    rows = []
    for i in range(N_PER):
        sub = rand_alnum(rng, 2)
        prefix = rand_alnum(rng, rng.randint(8, 12))
        while sub in prefix:
            prefix = rand_alnum(rng, len(prefix))
        suffix = rand_alnum(rng, rng.randint(4, 7))
        s = prefix + sub + suffix
        ans = s.find(sub) + 1                                        # SOLVER (1-indexed)
        q = (f"At what 1-indexed character position does the substring '{sub}' first appear in "
             f"'{s}'?\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "char_position_of_substring", q, ans,
            "substring indexing at a deep position; off-by-one and token drift dominate"))
    return rows


# 10) multidigit_multiplication -------------------------------------------------------------
def gen_multidigit_multiplication(rng):
    rows = []
    seen = set()
    for i in range(N_PER):
        while True:
            a = rng.randint(1000, 9999)
            b = rng.randint(100, 999)
            if (a, b) not in seen:
                seen.add((a, b)); break
        ans = a * b                                                  # SOLVER
        q = (f"Compute the exact product: {a} x {b}.\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "multidigit_multiplication", q, ans,
            "4-digit x 3-digit exact product; LLMs approximate magnitude but miss exact digits"))
    return rows


# 11) modular_exponentiation ----------------------------------------------------------------
def gen_modular_exponentiation(rng):
    rows = []
    seen = set()
    for i in range(N_PER):
        while True:
            base = rng.randint(3, 12)
            exp = rng.randint(7, 15)
            mod = rng.choice([97, 101, 103, 89, 83, 79])
            if (base, exp, mod) not in seen:
                seen.add((base, exp, mod)); break
        ans = pow(base, exp, mod)                                    # SOLVER
        q = (f"Compute ({base}^{exp}) mod {mod}.\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "modular_exponentiation", q, ans,
            "modular exponentiation requires exact big-integer arithmetic then a mod; no shortcut"))
    return rows


# 12) base_conversion -----------------------------------------------------------------------
def _to_base(n, b):
    digs = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = ""
    while n:
        out = digs[n % b] + out
        n //= b
    return out


def gen_base_conversion(rng):
    rows = []
    names = {2: "binary", 3: "base-3", 8: "octal", 16: "hexadecimal"}
    seen = set()
    for i in range(N_PER):
        while True:
            n = rng.randint(2000, 60000)
            base = rng.choice([2, 3, 8, 16])
            if (n, base) not in seen:
                seen.add((n, base)); break
        gold = _to_base(n, base)                                     # SOLVER (lowercase digits)
        q = (f"Convert the decimal number {n} to {names[base]} (base {base}). "
             f"Use lowercase letters for digits above 9.\n"
             f"{FINAL}<the {names[base]} digits, no prefix>.")
        rows.append(exact_row(i, "base_conversion", q, gold,
            "multi-step repeated division/remainder; long output digit strings are error-prone"))
    return rows


# 13) day_of_week_date ----------------------------------------------------------------------
def gen_day_of_week_date(rng):
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    rows = []
    used = set()
    for i in range(N_PER):
        while True:
            y = rng.randint(1850, 2300)
            m = rng.randint(1, 12)
            d = rng.randint(1, 28)
            if (y, m, d) not in used:
                used.add((y, m, d)); break
        gold = weekdays[datetime.date(y, m, d).weekday()]            # SOLVER (0=Mon)
        q = (f"What day of the week was {y:04d}-{m:02d}-{d:02d} (ISO date)? "
             f"Answer with the full weekday name in lowercase.\n{FINAL}<weekday>.")
        rows.append(exact_row(i, "day_of_week_date", q, gold,
            "day-of-week of a far date needs Zeller/calendar arithmetic across leap years"))
    return rows


# 14) date_add_days -------------------------------------------------------------------------
def gen_date_add_days(rng):
    rows = []
    used = set()
    for i in range(N_PER):
        while True:
            y = rng.randint(1990, 2090)
            m = rng.randint(1, 12)
            d = rng.randint(1, 28)
            if (y, m, d) not in used:
                used.add((y, m, d)); break
        add = rng.randint(100, 900)
        res = datetime.date(y, m, d) + datetime.timedelta(days=add)  # SOLVER
        gold = res.isoformat()                                       # YYYY-MM-DD
        q = (f"Starting from {y:04d}-{m:02d}-{d:02d}, what is the date {add} days later? "
             f"Give the ISO date.\n{FINAL}<YYYY-MM-DD>.")
        rows.append(exact_row(i, "date_add_days", q, gold,
            "adding hundreds of days across month/leap-year boundaries; carries are error-prone"))
    return rows


# 15) sort_numbers_nth ----------------------------------------------------------------------
def gen_sort_numbers_nth(rng):
    rows = []
    for i in range(N_PER):
        nums = rng.sample(range(100, 1000), 12)
        k = rng.randint(3, 9)                                        # k-th smallest, 1-indexed
        ans = sorted(nums)[k - 1]                                    # SOLVER
        q = (f"Sort these numbers in ascending order and report the {k}th smallest: "
             f"{', '.join(map(str, nums))}.\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "sort_numbers_nth", q, ans,
            "exact ordering of 12 three-digit numbers then selecting a deep rank; no global sort"))
    return rows


# 16) nested_parentheses_depth --------------------------------------------------------------
def gen_nested_parentheses_depth(rng):
    rows = []
    for i in range(N_PER):
        depth = rng.randint(4, 7)                                    # guaranteed max depth
        s = ""
        cur = 0
        forced = depth
        target_len = rng.randint(20, 30)
        while len(s) < target_len or cur > 0 or forced > 0:
            if forced > 0 and cur < forced:
                s += "("; cur += 1
                if cur == forced:
                    forced = 0
            elif cur > 0 and (rng.random() < 0.5 or len(s) >= target_len):
                s += ")"; cur -= 1
            elif len(s) < target_len:
                s += "("; cur += 1
            else:
                s += ")"; cur -= 1
        mx = cur = 0                                                 # SOLVER: max nesting depth
        for c in s:
            if c == "(":
                cur += 1; mx = max(mx, cur)
            else:
                cur -= 1
        q = (f"What is the maximum nesting depth of this balanced parenthesis string?\n"
             f"'{s}'\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "nested_parentheses_depth", q, mx,
            "tracking a running depth counter to its max over 20-30 brackets; LLMs lose the stack"))
    return rows


# 17) arithmetic_sequence_nth ---------------------------------------------------------------
def gen_arithmetic_sequence_nth(rng):
    rows = []
    for i in range(N_PER):
        a0 = rng.randint(2, 40)
        d = rng.randint(3, 19)
        n = rng.randint(40, 120)                                     # nth term, 1-indexed
        ans = a0 + (n - 1) * d                                       # SOLVER
        q = (f"An arithmetic sequence starts at {a0} and increases by {d} each step. "
             f"What is the {n}th term (the 1st term is {a0})?\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "arithmetic_sequence_nth", q, ans,
            "closed-form a0+(n-1)d with large n; LLMs off-by-one on index or drop a multiply"))
    return rows


# 18) fibonacci_like_nth --------------------------------------------------------------------
def gen_fibonacci_like_nth(rng):
    rows = []
    for i in range(N_PER):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        n = rng.randint(14, 22)                                      # 1-indexed: term1=a, term2=b
        if n == 1:
            ans = a
        elif n == 2:
            ans = b
        else:
            x, y = a, b
            for _ in range(n - 2):                                   # SOLVER
                x, y = y, x + y
            ans = y
        q = (f"A sequence has term1={a}, term2={b}, and each later term is the sum of the two "
             f"before it. What is term{n}?\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "fibonacci_like_nth", q, ans,
            "iterating an additive recurrence ~20 steps; magnitude grows fast, exact value slips"))
    return rows


# 19) count_in_long_list --------------------------------------------------------------------
def gen_count_in_long_list(rng):
    rows = []
    for i in range(N_PER):
        target = rng.randint(1, 9)
        n_target = rng.randint(7, 14)
        others = [x for x in range(1, 10) if x != target]
        seq = [target] * n_target + rng.choices(others, k=rng.randint(28, 40))
        rng.shuffle(seq)
        ans = seq.count(target)                                      # SOLVER
        q = (f"How many times does the number {target} appear in this list?\n"
             f"{', '.join(map(str, seq))}\n{FINAL}<integer>.")
        rows.append(numeric_row(i, "count_in_long_list", q, ans,
            "counting a needle across ~40-50 items; running tally over a long list fails"))
    return rows


# 20) acronym_from_words --------------------------------------------------------------------
def gen_acronym_from_words(rng):
    words = ["quantum", "velvet", "oxide", "harbor", "jungle", "ember", "wizard", "cobalt",
             "ripple", "tundra", "nimbus", "fjord", "glacier", "kelp", "mango", "onyx",
             "prism", "quasar", "raven", "sienna", "topaz", "umber", "vortex", "willow",
             "xenon", "yarrow", "zephyr", "basil", "cedar", "dune"]
    rows = []
    for i in range(N_PER):
        k = rng.randint(8, 12)
        chosen = rng.sample(words, k)
        gold = "".join(w[0] for w in chosen)                         # SOLVER (first letters)
        q = (f"Take the FIRST letter of each of the following words, in order, and join them into "
             f"one lowercase string: {', '.join(chosen)}.\n{FINAL}<the joined letters>.")
        rows.append(exact_row(i, "acronym_from_words", q, gold,
            "first-char extraction across 8-12 multi-token words then exact concatenation"))
    return rows


# =========================================================================================
# HARDENED generators — replace the guessable (small-integer answer-space) categories with
# CONSTRUCTED examples whose answers are LARGE and DISTINCT, so a non-reasoning model that blurts
# a small prior guess ("5", "Monday") scores ~0%. Strict grading (see hard_bench_run.py) + large
# distinct answers together push the guess floor to 0. Verified by tools/_guess_selftest.
# =========================================================================================
_CONS = "bcdfghjklmnpqrstvwxz"
_VOW = "aeiou"
# answer values chosen ABOVE the blind-guess ceiling (>15) + non-round, so no prior guess hits
_COUNTS = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
_POS = [17, 19, 23, 26, 29, 32, 34, 38, 41, 44]
_DEPTHS = [16, 17, 19, 21, 23, 26, 27, 29, 31, 34]
_WORDCOUNTS = [17, 18, 19, 21, 22, 23, 24, 26, 27, 28]

def _sprinkle(rng, base, inserts):
    seq = list(base)
    for ch in inserts:
        seq.insert(rng.randint(0, len(seq)), ch)
    return "".join(seq)

def gen_letter_count_in_word_h(rng):
    rows = []; ks = _COUNTS[:]; rng.shuffle(ks)
    for i in range(N_PER):
        k = ks[i]; target = "aeioustnr"[i % 9]
        others = [c for c in string.ascii_lowercase if c != target]
        s = _sprinkle(rng, [rng.choice(others) for _ in range(rng.randint(30, 45))], [target] * k)
        rows.append(numeric_row(i, "letter_count_in_word",
            f"How many times does the letter '{target}' appear in this string?\n{s}\n{FINAL}<integer>.",
            s.count(target), "count a target letter in a long string; large distinct answer"))
    return rows

def gen_letter_count_in_passage_h(rng):
    rows = []; ks = _COUNTS[:]; rng.shuffle(ks)
    for i in range(N_PER):
        k = ks[i]; target = "etaoin"[i % 6]
        others = [c for c in string.ascii_lowercase if c != target]
        s = _sprinkle(rng, [rng.choice(others) for _ in range(rng.randint(40, 55))], [target] * k)
        words, j = [], 0
        while j < len(s):
            n = rng.randint(3, 6); words.append(s[j:j + n]); j += n
        passage = " ".join(words)
        rows.append(numeric_row(i, "letter_count_in_passage",
            f'In the passage below, how many times does the letter "{target}" appear?\n"{passage}"\n{FINAL}<integer>.',
            passage.count(target), "count a letter across a long passage; large distinct answer"))
    return rows

def gen_word_occurrence_count_h(rng):
    fillers = ["river","table","cloud","stone","plant","north","music","paper","green","light",
               "horse","bread","chair","glass","frame","whale","brick","sugar","crane","pearl"]
    targets = ["the","and","cat","dog","sun","map","key","box","red","ant"]
    rows = []; ks = _WORDCOUNTS[:]; rng.shuffle(ks)
    for i in range(N_PER):
        k = ks[i]; target = targets[i]
        fill = [w for w in fillers if w != target]
        words = [target] * k + [rng.choice(fill) for _ in range(rng.randint(30, 40))]
        rng.shuffle(words); passage = " ".join(words)
        rows.append(numeric_row(i, "word_occurrence_count",
            f'How many times does the whole word "{target}" appear in this passage?\n{passage}\n{FINAL}<integer>.',
            passage.split().count(target), "exact whole-word count over a long passage; large distinct answer"))
    return rows

def gen_vowel_count_h(rng):
    rows = []; ks = _COUNTS[:]; rng.shuffle(ks)
    for i in range(N_PER):
        k = ks[i]
        s = _sprinkle(rng, [rng.choice(_CONS) for _ in range(rng.randint(30, 45))],
                      [rng.choice(_VOW) for _ in range(k)])
        rows.append(numeric_row(i, "vowel_count",
            f"How many vowels (a, e, i, o, u) are in this string?\n{s}\n{FINAL}<integer>.",
            sum(s.count(v) for v in _VOW), "per-character vowel counting over a long string; large distinct answer"))
    return rows

def gen_char_position_of_substring_h(rng):
    rows = []; ps = _POS[:]; rng.shuffle(ps)
    for i in range(N_PER):
        p = ps[i]; sub = rng.choice(_CONS) + rng.choice(_CONS)
        L = p + rng.randint(3, 8)
        while True:
            chars = [rng.choice(_CONS) for _ in range(L)]
            chars[p - 1], chars[p] = sub[0], sub[1]
            s = "".join(chars)
            if s.find(sub) == p - 1:
                break
        rows.append(numeric_row(i, "char_position_of_substring",
            f"In the string below, what is the 1-indexed position where the substring '{sub}' first appears?\n{s}\n{FINAL}<integer>.",
            s.find(sub) + 1, "first-occurrence index deep in a long string; large distinct position"))
    return rows

_BIGPRIMES = [101, 127, 149, 163, 181, 199, 211, 233, 251, 271, 293, 311, 337, 353, 379, 397]
def gen_modular_exponentiation_h(rng):
    rows = []; used = set()
    for i in range(N_PER):
        while True:
            base = rng.randint(2, 9); exp = rng.randint(8, 16); pr = rng.choice(_BIGPRIMES)
            ans = pow(base, exp, pr)
            if ans >= 10 and ans not in used:
                break
        used.add(ans)
        rows.append(numeric_row(i, "modular_exponentiation",
            f"Compute ({base}^{exp}) mod {pr}.\n{FINAL}<integer>.",
            ans, "exact big-integer modular exponentiation; large answer space"))
    return rows

def gen_nested_parentheses_depth_h(rng):
    rows = []; ds = _DEPTHS[:]; rng.shuffle(ds)
    for i in range(N_PER):
        d = ds[i]
        pre = "".join(rng.choice(["()", "(())"]) for _ in range(rng.randint(1, 3)))   # balanced, depth<=2
        post = "".join(rng.choice(["()", "(())"]) for _ in range(rng.randint(1, 3)))  # balanced, depth<=2
        s = pre + "(" * d + ")" * d + post                                            # max depth == d
        depth = mx = 0
        for ch in s:
            if ch == "(":
                depth += 1; mx = max(mx, depth)
            elif ch == ")":
                depth -= 1
        rows.append(numeric_row(i, "nested_parentheses_depth",
            f"What is the maximum nesting depth of parentheses in this string?\n{s}\n{FINAL}<integer>.",
            mx, "max nesting depth via a running stack; deep (14-23) so small guesses miss"))
    return rows

def gen_count_in_long_list_h(rng):
    rows = []; ks = _COUNTS[:]; rng.shuffle(ks)
    for i in range(N_PER):
        k = ks[i]; target = rng.randint(1, 9)
        others = [x for x in range(0, 10) if x != target]
        seq = [target] * k + [rng.choice(others) for _ in range(rng.randint(40, 55))]
        rng.shuffle(seq)
        rows.append(numeric_row(i, "count_in_long_list",
            f"How many times does the number {target} appear in this list?\n{', '.join(map(str, seq))}\n{FINAL}<integer>.",
            seq.count(target), "count a needle across ~55 items; large distinct answer"))
    return rows

def gen_days_between_dates_h(rng):
    rows = []; used = set()
    for i in range(N_PER):
        while True:
            a = datetime.date(rng.randint(1700, 2400), rng.randint(1, 12), rng.randint(1, 28))
            delta = rng.randint(200, 3000)
            if delta not in used:
                break
        used.add(delta); b = a + datetime.timedelta(days=delta)
        rows.append(numeric_row(i, "days_between_dates",
            f"How many days are there from {a.isoformat()} to {b.isoformat()}? (give the difference in days)\n{FINAL}<integer>.",
            delta, "date subtraction across months and leap years; large distinct answer"))
    return rows

GENERATORS = [
    gen_letter_count_in_word_h,
    gen_letter_count_in_passage_h,
    gen_word_occurrence_count_h,
    gen_string_reversal,
    gen_nth_character_index,
    gen_caesar_cipher_decode,
    gen_caesar_cipher_encode,
    gen_vowel_count_h,
    gen_char_position_of_substring_h,
    gen_multidigit_multiplication,
    gen_modular_exponentiation_h,
    gen_base_conversion,
    gen_days_between_dates_h,
    gen_date_add_days,
    gen_sort_numbers_nth,
    gen_nested_parentheses_depth_h,
    gen_arithmetic_sequence_nth,
    gen_fibonacci_like_nth,
    gen_count_in_long_list_h,
    gen_acronym_from_words,
]


def main():
    ap = argparse.ArgumentParser(description="Generate the hard, judge-free 20x10 benchmark.")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(repo_root, "hard_bench.jsonl"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    for gen in GENERATORS:
        sub = random.Random(rng.randint(0, 2**31 - 1))             # independent, reproducible per category
        rows = gen(sub)
        assert len(rows) == N_PER, f"{gen.__name__} produced {len(rows)} rows, expected {N_PER}"
        all_rows.extend(rows)

    # ---- validate every row round-trips through checkers.reward_for_row on its own answer ----
    import sys
    sys.path.insert(0, repo_root)
    from checkers import reward_for_row
    seen_ids = set()
    for r in all_rows:
        assert r["id"] not in seen_ids, f"duplicate id {r['id']}"
        seen_ids.add(r["id"])
        assert set(r.keys()) == {"id", "category", "question", "answer", "checker_kind",
                                 "checker_args", "why_hard"}, f"bad schema {r['id']}: {set(r.keys())}"
        assert r["checker_kind"] in ("exact", "numeric"), f"bad checker_kind {r['id']}"
        probe = f"FINAL ANSWER: {r['answer']}"
        rw = reward_for_row(r, probe)
        assert rw == 1.0, (f"row {r['id']} canonical answer does not self-verify "
                           f"(reward={rw}); ans={r['answer']!r}")

    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    # ---- report ----
    from collections import Counter
    cnt = Counter(r["category"] for r in all_rows)
    print(f"[gen_hard_bench] wrote {len(all_rows)} rows -> {args.out}")
    print(f"[gen_hard_bench] {len(cnt)} categories; per-category counts:")
    for c in sorted(cnt):
        flag = "" if cnt[c] == N_PER else "  <-- WRONG COUNT"
        print(f"    {c:<28} {cnt[c]}{flag}")
    assert all(v == N_PER for v in cnt.values()), "every category must have exactly 10 rows"
    assert len(all_rows) == 200, f"expected 200 rows, got {len(all_rows)}"
    print(f"[gen_hard_bench] ALL rows self-verify via checkers.reward_for_row (canonical answer -> 1.0)")
    print("\n[gen_hard_bench] sample rows:")
    for r in (all_rows[0], all_rows[35], all_rows[125], all_rows[195]):
        print(f"  --- {r['id']} ({r['checker_kind']}) ---")
        print(f"    Q: {r['question'][:120].replace(chr(10), ' / ')} ...")
        print(f"    A: {r['answer']}")


if __name__ == "__main__":
    main()
