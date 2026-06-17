#!/usr/bin/env python3
"""Unit tests for checkers.py (the verifiable RL reward functions)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checkers import check_exact, check_contains_all, make_checker, reward_for_row


def test_exact_numeric():
    assert check_exact("the answer is 42.", "42") == 1.0
    assert check_exact("I think 41", "42") == 0.0
    assert check_exact("total: 3,450 dollars", "3450") == 1.0   # comma-insensitive


def test_exact_text():
    assert check_exact("Ben", "ben") == 1.0
    assert check_exact("the shortest is Ana", "ana") == 1.0


def test_contains_all():
    assert check_contains_all("validate then build mvp then launch", ["validate", "mvp", "launch"]) == 1.0
    assert check_contains_all("just launch it", ["validate", "mvp", "launch"]) == 0.0


def test_make_and_row():
    row = {"checker_kind": "exact", "checker_args": {"gold": "3450."}}
    assert reward_for_row(row, "profit is $3450.") == 1.0
    assert reward_for_row(row, "profit is $3451.") == 0.0


def test_runs_on_real_dataset_row():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dataset", "sft_100.jsonl")
    row = json.loads(open(path).readline())
    # gold answer must satisfy its own checker
    assert reward_for_row(row, row["answer"]) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("checkers unit tests: ALL PASS")
