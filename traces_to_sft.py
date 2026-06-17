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

# graded graders route to 'rubric' (variance-weighted / SFT-only); the rest are binary 'verifiable'.
RUBRIC_TYPES = {'role_map', 'plan_rubric'}

def canonical_str(ak):
    """A clean committed-answer string to append as 'FINAL ANSWER: ...' (so the model learns to
    reason THEN commit, and the checker grades only the commitment)."""
    c = ak.get('canonical')
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        if 'value' in c:
            return f"{c['value']} {c['unit']}".strip() if c.get('unit') else f"{c['value']}"
        if 'roles' in c:
            return ", ".join(f"{k}={v}" for k, v in c['roles'].items())
        if 'gold_summary' in c:
            return c['gold_summary']
    return str(c)

def checker_from_answerkey(ak):
    """Answer-key row -> (checker_kind, checker_args, reward_path)."""
    if ak is None:
        return 'rubric', {'items': []}, 'rubric'          # no key -> SFT-only
    mt = ak['match']['type']                               # checker_kind == match.type
    cargs = {'canonical': ak.get('canonical'), 'match': ak['match']}
    rp = 'rubric' if mt in RUBRIC_TYPES else 'verifiable'
    return mt, cargs, rp

def convert(trace_row, ak, i):
    kind, cargs, rp = checker_from_answerkey(ak)
    answer = parse_answer(trace_row['trace'])             # reasoning prose
    if ak is not None:                                    # reason THEN commit (GSM8K-style)
        answer = f"{answer}\nFINAL ANSWER: {canonical_str(ak)}"
    return {
        'id': f"trace_{i:05d}",
        'instruction': trace_row['instruction'],
        'plan': parse_plan(trace_row['trace']),
        'answer': answer,
        'checker_kind': kind,
        'checker_args': cargs,
        'reward_path': rp,
        'family': trace_row.get('family'),
        'answer_form': trace_row.get('answer_form'),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True, nargs='+', help='hard_reasoning_traces_*.jsonl')
    ap.add_argument('--answers', nargs='+', default=[], help='answers_*.jsonl (join key = instruction)')
    ap.add_argument('--out', default='dataset/traces_sft.jsonl')
    ap.add_argument('--vocab_out', default='plan_vocab.json',
                    help='write the planner vocab (refined primitives + FINALIZE terminator) here')
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    key = {}                                               # instruction -> answer-key row
    for path in args.answers:
        for line in open(path):
            a = json.loads(line); key[a['instruction']] = a

    n, matched = 0, 0
    vocab_order, seen = [], set()                          # collect refined primitives in first-seen order
    with open(args.out, 'w') as f:
        for path in args.traces:
            for line in open(path):
                row = json.loads(line)
                ak = key.get(row['instruction'])
                matched += ak is not None
                rec = convert(row, ak, n)
                for p in rec['plan']:
                    if p not in seen:
                        seen.add(p); vocab_order.append(p)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n += 1
    vocab = ['PAD'] + vocab_order
    term = 'FINALIZE' if 'FINALIZE' in seen else vocab_order[-1]
    json.dump({'vocab': vocab, 'terminator': term}, open(args.vocab_out, 'w'), indent=1)
    print(f"wrote {n} rows -> {args.out}  (answer-key matched {matched}/{n})")
    print(f"wrote {args.vocab_out}: {len(vocab)} primitives, terminator={term}")

if __name__ == '__main__':
    main()
