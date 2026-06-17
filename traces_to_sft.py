#!/usr/bin/env python3
"""
traces_to_sft.py — adapt hard_reasoning_traces_*.jsonl into the joint SFT schema.

Input row (yours):
    {instruction, trace, family, answer_form, gold_answer?}
Output row (what train_sft / rollout / grpo / compare expect):
    {id, instruction, plan:[primitives], answer:<reasoning prose>, checker_kind, checker_args,
     reward_path, family, answer_form}

- plan   = the primitives inside each "TURN n [ A[..] ; B[..] ]" header, base names in order.
- answer = the executor target = the concatenated `response:` prose (the model learns to REASON).
- checker = derived from answer_form + gold_answer (verifiable reward + ablation gap).
  answer_form 'plan' (or missing gold_answer) -> reward_path='rubric' (SFT-only, held out of RL).

CLI:
  python traces_to_sft.py --in hard_reasoning_traces_1000.jsonl --out dataset/traces_sft.jsonl
"""
import argparse, json, re

TURN_RE = re.compile(r'^\s*TURN\s+\d+\s*\[(.*)\]\s*$')

def parse_plan(trace):
    """Primitives from each TURN bracket, base name (strip [params]), in order."""
    plan = []
    for line in trace.splitlines():
        m = TURN_RE.match(line)
        if not m:
            continue
        for prim in m.group(1).split(';'):
            name = prim.strip().split('[')[0].strip()
            if name:
                plan.append(name)
    return plan

def parse_answer(trace):
    """Executor target = the reasoning prose (all `response:` lines joined)."""
    out = []
    for line in trace.splitlines():
        s = line.strip()
        if s.lower().startswith('response:'):
            out.append(s[len('response:'):].strip())
    return ' '.join(out).strip()

def checker_for(answer_form, gold):
    """Map (answer_form, gold_answer) -> (checker_kind, checker_args, reward_path)."""
    if gold is None or answer_form == 'plan':
        # no single verifiable answer -> SFT only; held out of RL (reward_path rubric / exclude)
        return 'rubric', {'items': []}, 'rubric'
    g = str(gold).strip()
    if answer_form == 'number_with_units':
        num = re.findall(r'-?\d[\d,]*\.?\d*', g)
        return 'numeric', {'gold': (num[-1].replace(',', '') if num else g)}, 'verifiable'
    if answer_form == 'yes_no':
        return 'yesno', {'gold': g.lower()}, 'verifiable'
    if answer_form == 'cannot_conclude':
        return 'label', {'gold': 'cannot_conclude'}, 'verifiable'
    # value / next_term -> exact (substring/number-aware)
    return 'exact', {'gold': g}, 'verifiable'

def convert(row, i):
    plan = parse_plan(row['trace'])
    answer = parse_answer(row['trace'])
    kind, cargs, rp = checker_for(row.get('answer_form'), row.get('gold_answer'))
    return {
        'id': f"trace_{i:05d}",
        'instruction': row['instruction'],
        'plan': plan,
        'answer': answer,
        'checker_kind': kind,
        'checker_args': cargs,
        'reward_path': rp,
        'family': row.get('family'),
        'answer_form': row.get('answer_form'),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, nargs='+', help='one or more trace jsonl files')
    ap.add_argument('--out', default='dataset/traces_sft.jsonl')
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n = 0
    with open(args.out, 'w') as f:
        for path in args.inp:
            for line in open(path):
                row = json.loads(line)
                f.write(json.dumps(convert(row, n), ensure_ascii=False) + '\n')
                n += 1
    print(f"wrote {n} rows -> {args.out}")

if __name__ == '__main__':
    main()
