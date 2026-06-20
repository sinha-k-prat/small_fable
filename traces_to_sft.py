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
DROP_PARAMS = {'confidence'}   # noise — not part of the strategy
END = 'END'                    # explicit plan terminator (the last FINALIZE param is real content)

def parse_plan(trace):
    """FACTORED plan: a flat sequence of sub-tokens — each primitive followed by its parameter atoms
    (key=value), ending with END. So a single autoregressive head composes primitives with params,
    and a novel primitive+param pairing is just a novel sequence of already-seen tokens.
      MODEL as=truth_table LINK guard=on VERIFY aspect=logic FINALIZE form=yes_no END
    """
    seq = []
    for line in trace.splitlines():
        m = TURN_RE.match(line)
        if not m:
            continue
        for prim in m.group(1).split(';'):
            prim = prim.strip()
            if not prim:
                continue
            seq.append(prim.split('[')[0].strip())            # the primitive
            pm = re.search(r'\[(.*)\]', prim)
            if pm:
                for kv in pm.group(1).split(','):              # its parameter atoms
                    kv = kv.strip().replace(' ', '')
                    if kv and kv.split('=')[0] not in DROP_PARAMS:
                        seq.append(kv)
    seq.append(END)
    return seq

def _factor_primitive_group(inner):
    """'MODEL[as=truth_table] ; LINK[guard=on]' inner -> ['MODEL','as=truth_table','LINK','guard=on'].
    Same factoring rule parse_plan uses, but for a SINGLE turn header (one TURN's brackets)."""
    out = []
    for prim in inner.split(';'):
        prim = prim.strip()
        if not prim:
            continue
        out.append(prim.split('[')[0].strip())
        pm = re.search(r'\[(.*)\]', prim)
        if pm:
            for kv in pm.group(1).split(','):
                kv = kv.strip().replace(' ', '')
                if kv and kv.split('=')[0] not in DROP_PARAMS:
                    out.append(kv)
    return out


def parse_turns(trace):
    """Walk the raw trace turn-by-turn. Each 'TURN n [ ... ]' header starts a turn whose plan is the
    factored primitives+atoms; the following 'response:' lines (until the next TURN) are its prose.
    Returns [{'plan':[...], 'response':'...'}]. The per-turn plan does NOT include END/markers here;
    model_joint's assembler inserts BOP/END/RESP_EOS."""
    turns, cur = [], None
    for line in trace.splitlines():
        m = TURN_RE.match(line)
        if m:
            if cur is not None:
                cur['response'] = ' '.join(cur.pop('_resp')).strip(); turns.append(cur)
            cur = {'plan': _factor_primitive_group(m.group(1)), '_resp': []}
            continue
        s = line.strip()
        if cur is not None and s.lower().startswith('response:'):
            cur['_resp'].append(s[len('response:'):].strip())
    if cur is not None:
        cur['response'] = ' '.join(cur.pop('_resp')).strip(); turns.append(cur)
    return turns


def convert_interleaved(trace_row, ak, i):
    """Wrap the existing flat convert(): keep ALL flat fields (DUAL-WRITE), ADD 'turns'. The terminal
    turn's response gets the committed 'FINAL ANSWER: X' tail so the answer-key grader has a span."""
    rec = convert(trace_row, ak, i)               # flat convert() -> plan/answer/checker/...
    turns = parse_turns(trace_row['trace'])
    if not turns:                                  # degenerate trace with no TURN headers -> one turn
        turns = [{'plan': [], 'response': parse_answer(trace_row['trace'])}]
    if ak is not None and 'FINAL ANSWER:' in rec['answer']:
        tail = rec['answer'].split('FINAL ANSWER:', 1)[1]
        turns[-1]['response'] = turns[-1]['response'].rstrip() + '\nFINAL ANSWER:' + tail
    rec['turns'] = turns
    return rec


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
                    help='write the planner vocab (primitives + param-atoms + END) here')
    ap.add_argument('--shuffle', action='store_true', default=True,
                    help='shuffle the corpus (seeded) so the train/held split is representative')
    ap.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    ap.add_argument('--seed', type=int, default=20260616)
    ap.add_argument('--interleaved', action='store_true',
                    help='ALSO emit per-turn turns:[{plan,response}] (dual-write) and append BOP/'
                         'FINALIZE_ALL plan-vocab markers + {"interleaved":true,"markers":{...}}.')
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    key = {}                                               # instruction -> answer-key row
    for path in args.answers:
        for line in open(path):
            a = json.loads(line); key[a['instruction']] = a

    recs, matched = [], 0
    for path in args.traces:
        for line in open(path):
            row = json.loads(line)
            ak = key.get(row['instruction'])
            matched += ak is not None
            rec = (convert_interleaved(row, ak, len(recs)) if args.interleaved
                   else convert(row, ak, len(recs)))
            recs.append(rec)
    if args.shuffle:                                       # representative train/held split
        import random
        random.Random(args.seed).shuffle(recs)
    vocab_order, seen = [], set()
    with open(args.out, 'w') as f:
        for rec in recs:
            for p in rec['plan']:
                if p not in seen:
                    seen.add(p); vocab_order.append(p)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    vocab = ['PAD'] + vocab_order
    markers = {}
    if args.interleaved:
        # APPEND the interleaved control markers after the data tokens so existing ids stay stable.
        # END already exists (added by parse_plan) and is reused as the per-turn PLAN_EOS.
        for ctrl in ('BOP', 'FINALIZE_ALL'):
            if ctrl not in vocab:
                vocab.append(ctrl)
        markers = {'BOP': 'BOP', 'PLAN_EOS': END, 'FINALIZE_ALL': 'FINALIZE_ALL'}
    out_vocab = {'vocab': vocab, 'terminator': END}
    if args.interleaved:                                   # flat plan_vocab.json stays byte-identical
        out_vocab['interleaved'] = True
        out_vocab['markers'] = markers
    json.dump(out_vocab, open(args.vocab_out, 'w'), indent=1)
    print(f"wrote {len(recs)} rows -> {args.out}  (answer-key matched {matched}/{len(recs)}, "
          f"shuffled={args.shuffle}, interleaved={args.interleaved})")
    print(f"wrote {args.vocab_out}: {len(vocab)} tokens (primitives + param-atoms"
          f"{'+ BOP/FINALIZE_ALL markers' if args.interleaved else ''}), terminator={END}")

if __name__ == '__main__':
    main()
