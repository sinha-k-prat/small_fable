"""examples.py — the 5 demo prompts + a MANUAL ChatML builder.

CRITICAL (verified live on this stack): ``tokenizer.apply_chat_template(...)`` RAISES
``ImportError: apply_chat_template requires jinja2>=3.1.0 ... Your version is 2.11.2``.
So we build the Qwen2.5 ChatML prompt as a plain string and tokenize that. Do NOT call
``apply_chat_template`` anywhere in this package.
"""

EXAMPLES = [
    "What is 17 plus 25? Reply with just the number.",
    "Give one word that means the opposite of 'hot'.",
    "Complete the sentence: The capital of France is",
    "Is 7 a prime number? Answer yes or no.",
    "Reverse the letters of the word 'cat'.",
]

# Qwen2.5 ChatML template, built by hand (jinja2 here is 2.11.2 < 3.1.0).
_CHATML = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_chatml(prompt):
    """Return the raw ChatML string for ``prompt`` (no tokenization)."""
    return _CHATML.format(prompt=prompt)


def build_input_ids(bundle, prompt):
    """Qwen2.5 ChatML prompt -> input ids tensor of shape [1, T] on bundle.device.

    Built as a plain string (apply_chat_template RAISES on jinja2 2.11.2 here).
    """
    text = _CHATML.format(prompt=prompt)
    ids = bundle.tokenizer(text, return_tensors="pt").input_ids  # [1, T]
    assert ids.dim() == 2 and ids.shape[0] == 1, ids.shape
    return ids.to(bundle.device)
