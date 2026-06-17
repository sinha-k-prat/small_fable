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

# --------------------------------------------------------------------------- trace answer-key graders
# These grade a MODEL's reasoning output against the canonical answer-key (answers_*.jsonl). The
# model's final answer is in the TAIL of its prose, and garden-path traces mention WRONG answers
# earlier, so label/term graders look at the last segment, and numeric uses the last number.
def _canon_value(canonical):
    return canonical.get("value") if isinstance(canonical, dict) else canonical

_CONTRACT = [("can't","cannot"),("won't","will not"),("n't"," not"),("'re"," are"),("'s"," is")]
def _decontract(s):
    s = str(s).lower()
    for a, b in _CONTRACT:
        s = s.replace(a, b)
    return s

_NUMWORD = {w:i for i,w in enumerate(
    ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve",
     "thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"])}
_NUMWORD.update({"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100})

def _numbers_in(text):
    """All numeric values mentioned, as floats — from digits AND spelled-out number words."""
    vals = [float(m.replace(",","")) for m in re.findall(r"-?\d[\d,]*\.?\d*", str(text))]
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if w in _NUMWORD:
            vals.append(float(_NUMWORD[w]))
    return vals

def _words(s):
    return re.findall(r"[a-z0-9_]+", str(s).lower())

def _tail(s, n=160):
    return str(s)[-n:]

def _answer_span(ans):
    """The committed answer = text after the last 'FINAL ANSWER:' marker (training appends it, and
    the model learns to emit it). Grading only this span avoids crediting numbers/labels that merely
    appear in the reasoning. Falls back to the tail if no marker is present."""
    m = list(re.finditer(r"final answer\s*:?", str(ans), re.I))
    span = str(ans)[m[-1].end():] if m else _tail(ans)
    return span.replace("_", " ")

def _has_word(text, label):
    lab = str(label).lower().strip().replace("_", " ")
    return re.search(r"\b" + re.escape(lab) + r"\b", text) is not None

def check_exact_choice(ans, accept):
    text = " ".join(_words(_decontract(_answer_span(ans))))   # grade only the committed answer
    return 1.0 if any(_has_word(text, a) for a in accept) else 0.0

def check_numeric(ans, value, tolerance=0.0):
    if not isinstance(value, (int, float)):             # e.g. "no_finite_answer"
        return check_exact_choice(ans, [value, "no finite", "never", "infinite"])
    cands = _numbers_in(_answer_span(ans))              # digits + number-words in the committed answer
    return 1.0 if any(abs(c - float(value)) <= float(tolerance) for c in cands) else 0.0

def check_numeric_or_word(ans, value, accept, tolerance=0.0):
    if isinstance(value, (int, float)) and check_numeric(ans, value, tolerance) == 1.0:
        return 1.0
    text = " ".join(_words(_decontract(_answer_span(ans))))
    return 1.0 if any(_has_word(text, str(a)) for a in accept) else 0.0

def check_exact_term(ans, accept):
    text = " ".join(_words(_answer_span(ans)))
    return 1.0 if any(_has_word(text, a) for a in accept) else 0.0

def check_role_map(ans, roles):
    """Knights-and-knaves: reward = fraction of names assigned the correct role. For each name, the
    role word that appears CLOSEST after the name in the text wins."""
    a = _answer_span(ans).lower()
    pos = {"truth_teller": ["truth teller", "truthful", "knight", "honest"],
           "liar": ["liar", "lies", "lying", "knave", "false"]}
    correct = 0
    for name, role in roles.items():
        i = a.rfind(name.lower())
        seg = a[i:i+60] if i >= 0 else a
        said = None
        best = 1e9
        for r, kws in pos.items():
            for kw in kws:
                j = seg.find(kw)
                if j >= 0 and j < best:
                    best, said = j, r
        correct += 1 if said == role else 0
    return correct / max(1, len(roles))

def check_string_contains(ans, key_phrase):
    return 1.0 if _norm(key_phrase) in _norm(ans) else 0.0

def check_plan_rubric(ans, gold_summary):
    """Graded: fraction of the gold summary's CONTENT words present in the answer (a light keyword
    rubric; a judge would be better but this gives an honest [0,1] spread)."""
    stop = {"the","a","an","it","is","to","and","of","so","still","but","one","keep","feels"}
    keys = [w for w in _words(gold_summary) if w not in stop and len(w) > 2]
    if not keys:
        return 0.0
    a = " ".join(_words(ans))
    return sum(1.0 for w in set(keys) if _has_word(a, w)) / len(set(keys))

def make_checker(kind, args):
    if kind=="exact":           return lambda ans: check_exact(ans, args["gold"])
    if kind=="contains_all":    return lambda ans: check_contains_all(ans, args["tokens"])
    # trace answer-key graders (checker_args carries the answer-key's canonical + match)
    if kind=="exact_choice":    return lambda ans: check_exact_choice(ans, args["match"]["accept"])
    if kind=="numeric":         return lambda ans: check_numeric(
        ans, _canon_value(args["canonical"]), args["match"].get("tolerance", 0.0))
    if kind=="numeric_or_word": return lambda ans: check_numeric_or_word(
        ans, _canon_value(args["canonical"]), args["match"]["accept"], args["match"].get("tolerance", 0.0))
    if kind=="exact_term":      return lambda ans: check_exact_term(ans, args["match"]["accept"])
    if kind=="role_map":        return lambda ans: check_role_map(ans, args["canonical"]["roles"])
    if kind=="string_contains": return lambda ans: check_string_contains(ans, args["match"]["key_phrase"])
    if kind=="plan_rubric":     return lambda ans: check_plan_rubric(ans, args["canonical"]["gold_summary"])
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
