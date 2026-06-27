"""examples.py — Task records + three index-aligned variant sets + system-aware MANUAL ChatML.

CRITICAL (verified live on this stack): ``tokenizer.apply_chat_template(...)`` RAISES
``ImportError: apply_chat_template requires jinja2>=3.1.0 ... Your version is 2.11.2``.
So we build the Qwen2.5 ChatML prompt as a plain string and tokenize that. Do NOT call
``apply_chat_template`` anywhere in this package.

v2: the 5 legacy prompts are preserved verbatim (so ``EXAMPLES`` is byte-identical and v1
caches/notebook examples don't drift), and a 6th task (multi-digit multiplication, the
headline) is added. Three index-aligned variant sets share the SAME 6 tasks in the SAME
order: PLAIN (legacy), SIMPLE (terse, where the small model tends to FAIL), DETAILED
(step-by-step programmatic scaffold that walks the METHOD but NEVER states the answer).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    key: str            # stable id: "add","antonym","capital","prime","reverse","mult"
    system: object      # optional system message (str) or None -> default helpful-assistant
    user: str           # user prompt
    gold: str           # gold answer text
    grade: str          # "digits" | "substr" | "equal"  -> RunCollection.grade_mode
    gen_tokens: int     # per-task EOS-safety cap (generation stops at <|im_end|>; this is just the max)


TASK_KEYS = ["add", "antonym", "capital", "prime", "reverse", "mult"]


# ----------------------------------------------------------------------------
# system-aware MANUAL ChatML (jinja2-safe; default system byte-identical to v1)
# ----------------------------------------------------------------------------
_CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_DEFAULT_SYSTEM = "You are a helpful assistant."


def build_chatml(prompt, system=None):
    """Raw ChatML string. system=None -> default helpful-assistant (byte-identical to v1)."""
    return _CHATML.format(system=(system or _DEFAULT_SYSTEM), prompt=prompt)


def build_input_ids(bundle, prompt, system=None):
    """Qwen2.5 ChatML prompt -> input ids tensor of shape [1, T] on bundle.device.

    Built as a plain string (apply_chat_template RAISES on jinja2 2.11.2 here).
    ``prompt`` may be: a plain user str; a ``Task`` (system/user pulled from it); or a
    (system, user) tuple/list (its system overrides the kwarg). Back-compat:
    ``build_input_ids(bundle, str)`` is unchanged (default system).
    """
    if isinstance(prompt, Task):
        system, prompt = prompt.system, prompt.user
    elif isinstance(prompt, (tuple, list)):
        system, prompt = prompt[0], prompt[1]
    text = build_chatml(prompt, system=system)
    ids = bundle.tokenizer(text, return_tensors="pt").input_ids  # [1, T]
    assert ids.dim() == 2 and ids.shape[0] == 1, ids.shape
    return ids.to(bundle.device)


# ----------------------------------------------------------------------------
# the three index-aligned variant sets (SAME 6 tasks, SAME order = TASK_KEYS)
# ----------------------------------------------------------------------------
# PLAIN: the legacy 5 user prompts (verbatim) + mult. system=None for all.
EXAMPLES_PLAIN = [
    Task("add", None,
         "What is 17 plus 25? Reply with just the number.",
         "42", "digits", 512),
    Task("antonym", None,
         "Give one word that means the opposite of 'hot'.",
         "cold", "substr", 512),
    Task("capital", None,
         "Complete the sentence: The capital of France is",
         "Paris", "substr", 512),
    Task("prime", None,
         "Is 7 a prime number? Answer yes or no.",
         "yes", "substr", 512),
    Task("reverse", None,
         "Reverse the letters of the word 'cat'.",
         "tac", "substr", 512),
    Task("mult", None,
         "Compute the exact product: 9246 x 897. Reply with just the number.",
         "8293662", "digits", 768),
]

# SIMPLE: terse prompts where the small model tends to FAIL. system=None.
EXAMPLES_SIMPLE = [
    Task("add", None,
         "What is 17 plus 25? Reply with just the number.",
         "42", "digits", 512),
    Task("antonym", None,
         "Give one word that means the opposite of 'hot'. Reply with one word.",
         "cold", "substr", 512),
    Task("capital", None,
         "The capital of France is",
         "Paris", "substr", 512),
    Task("prime", None,
         "Is 7 a prime number? Answer yes or no.",
         "yes", "substr", 512),
    Task("reverse", None,
         "Reverse the letters of the word 'cat'. Reply with just the reversed word.",
         "tac", "substr", 512),
    Task("mult", None,
         "Compute the exact product: 9246 x 897. Reply with just the number.",
         "8293662", "digits", 768),
]

# DETAILED: every task shares ONE system message (so the notebook can quote it once).
# User prompts walk the METHOD and NEVER state the answer.
_DETAILED_SYSTEM = (
    "You are a meticulous step-by-step solver. Show ALL intermediate work explicitly on "
    "separate lines, never skip or summarize a step. End with a line 'FINAL ANSWER:' then "
    "only the answer."
)

EXAMPLES_DETAILED = [
    Task("add", _DETAILED_SYSTEM,
         "Add 17 and 25. Add column by column from the right: add the ones digits (7 + 5), "
         "write the ones digit of that sum and carry any tens; then add the tens digits plus "
         "the carry. Combine the digits.",
         "42", "digits", 512),
    Task("antonym", _DETAILED_SYSTEM,
         "Find one word that is the opposite of 'hot'. State the dimension the word varies "
         "along, then name the word at the far opposite end of that dimension from 'hot'.",
         "cold", "substr", 512),
    Task("capital", _DETAILED_SYSTEM,
         "Name the capital city of France. State the country, then recall the city that is its "
         "seat of national government.",
         "Paris", "substr", 512),
    Task("prime", _DETAILED_SYSTEM,
         "Determine whether 7 is prime. A prime has exactly two distinct positive divisors, 1 "
         "and itself. Test each integer d from 2 up to 6 and state whether d divides 7 evenly. "
         "If none divide it, it is prime. Answer yes or no.",
         "yes", "substr", 512),
    Task("reverse", _DETAILED_SYSTEM,
         "Reverse the letters of the word 'cat'. List its letters in order, numbering them. "
         "Then write the letters from the last numbered one to the first, concatenated.",
         "tac", "substr", 512),
    Task("mult", _DETAILED_SYSTEM,
         "Compute 9246 x 897. Multiply by long multiplication: multiply the top number by each "
         "digit of the bottom number (write each partial product, shifted by place), then add "
         "the partial products.",
         "8293662", "digits", 768),
]


VARIANTS = {
    "plain": EXAMPLES_PLAIN,
    "simple": EXAMPLES_SIMPLE,
    "detailed": EXAMPLES_DETAILED,
}

# Back-compat: existing code imports EXAMPLES as a list[str] of USER prompts.
# Unchanged behavior for single/compare/cli/cache.
EXAMPLES = [t.user for t in EXAMPLES_PLAIN]

# integrity: index-aligned, same keys, no variant set leaks the answer.
assert len(EXAMPLES_PLAIN) == len(EXAMPLES_SIMPLE) == len(EXAMPLES_DETAILED) == 6
for _s in (EXAMPLES_PLAIN, EXAMPLES_SIMPLE, EXAMPLES_DETAILED):
    assert [t.key for t in _s] == TASK_KEYS


def resolve_task(task, variant="plain"):
    """Resolve a ``--task`` value (int index OR key str like 'mult') to an index into
    TASK_KEYS. None stays None (= all tasks)."""
    if task is None:
        return None
    if isinstance(task, int):
        return task
    s = str(task).strip()
    if s.isdigit():
        return int(s)
    if s in TASK_KEYS:
        return TASK_KEYS.index(s)
    raise ValueError("unknown task %r (keys: %s)" % (task, TASK_KEYS))
