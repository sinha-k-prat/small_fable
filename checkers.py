#!/usr/bin/env python3
"""
checkers.py — verifiable correctness functions for RL rewards.

RL must be scored by a PROGRAMMATIC checker (exact-match / constraint-satisfaction),
NOT by NLL or embedding-similarity — those saturate on easy tasks and hide whether the
plan actually helped. Each dataset row carries {checker_kind, checker_args}; this module
turns that into a callable reward in {0.0, 1.0} (extend to graded if you like).
"""
import re

def _norm(s):
    return "".join(str(s).lower().split()).rstrip(".")

def _last_number(s):
    m=re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    return m[-1].replace(",","") if m else None

def check_exact(model_answer, gold):
    # accept if the model's salient token equals gold (number-aware)
    g=_norm(gold)
    if re.fullmatch(r"-?\d[\d,]*\.?\d*", str(gold).strip()):
        n=_last_number(model_answer)
        return 1.0 if (n is not None and _norm(n)==_norm(gold)) else 0.0
    return 1.0 if g in _norm(model_answer) else 0.0

def check_contains_all(model_answer, tokens):
    a=str(model_answer).lower()
    return 1.0 if all(t.lower() in a for t in tokens) else 0.0

def make_checker(kind, args):
    if kind=="exact":          return lambda ans: check_exact(ans, args["gold"])
    if kind=="contains_all":   return lambda ans: check_contains_all(ans, args["tokens"])
    raise ValueError(f"unknown checker_kind: {kind}")

def check_rubric(model_answer, items):
    """A1b GRADED reward: fraction of rubric items present in the answer -> [0,1].
    An HONEST graded reward (not a fake binary keyword gate): it yields a real spread across
    rollouts, which is exactly what variance_weight needs to find signal on soft tasks."""
    if not items:
        return 0.0
    a = str(model_answer).lower()
    return sum(1.0 for t in items if str(t).lower() in a) / len(items)

def reward_for_row(row, model_answer):
    """Score a model answer against a row's gold checker. {0.0,1.0} for exact/contains_all,
    graded [0,1] for rubric."""
    if row["checker_kind"] == "rubric":
        return check_rubric(model_answer, row["checker_args"]["items"])
    return make_checker(row["checker_kind"], row["checker_args"])(model_answer)

def graded_reward_for_row(row, model_answer):
    """Alias making the graded intent explicit at call sites (A1b)."""
    return float(reward_for_row(row, model_answer))

# ---- optional teacher-LL bonus so 'better than gold' is rewardable (stub) ----
def teacher_ll_bonus(teacher_model, tokenizer, prompt, answer, device, weight=0.2):
    """Length-normalized log-prob of `answer` under a frozen teacher. Add to the checker
    reward so the model can be rewarded for exceeding the SFT gold, not just matching it.
    Returns 0.0 if no teacher is provided."""
    if teacher_model is None: return 0.0
    import torch
    text=prompt+answer
    ids=tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out=teacher_model(**ids, labels=ids["input_ids"])
    # out.loss is mean NLL; convert to length-normalized LL bonus
    return weight*(-float(out.loss))

if __name__=="__main__":
    # self-test
    assert check_exact("the answer is 42.", "42")==1.0
    assert check_exact("I think 41", "42")==0.0
    assert check_exact("Ben", "ben")==1.0
    assert check_contains_all("validate then build mvp then launch", ["validate","mvp","launch"])==1.0
    assert check_contains_all("just launch it", ["validate","mvp","launch"])==0.0
    print("checkers self-test: ALL PASS")
