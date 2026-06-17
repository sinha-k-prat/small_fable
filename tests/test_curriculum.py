#!/usr/bin/env python3
"""Unit tests for A3/A4 curriculum + spectrum helpers in train_sft.py (pure Python, no model)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_sft as S


ROWS = [
    {"instruction": "a", "plan": ["X", "TERMINATE"], "answer": "1"},
    {"instruction": "b", "plan": ["A", "B", "C", "D", "TERMINATE"], "answer": "a longer answer here"},
    {"instruction": "c", "plan": ["M", "TERMINATE"], "answer": "2",
     "alternatives": [{"plan": ["M", "TERMINATE"], "answer": "2"},
                      {"plan": ["N", "TERMINATE"], "answer": "two"}]},
]


def test_expand_alternatives():
    ex = S.expand_alternatives(ROWS)
    assert len(ex) == 4, "row c expands to 2 items, others pass through"
    plans = [tuple(r["plan"]) for r in ex if r["instruction"] == "c"]
    assert ("M", "TERMINATE") in plans and ("N", "TERMINATE") in plans


def test_difficulty_monotonic():
    assert S.difficulty(ROWS[0]) < S.difficulty(ROWS[1]), "longer plan => harder"


def test_curriculum_batches_easy_to_hard():
    batches = list(S.curriculum_batches(ROWS, 2, seed=0))
    order = [r["instruction"] for b in batches for r in b]
    assert order[-1] == "b", f"hardest (longest plan) must come last, got {order}"
    assert set(order) == {"a", "b", "c"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("curriculum unit tests: ALL PASS")
