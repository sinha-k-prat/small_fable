#!/usr/bin/env python3
"""
flatten_to_sft.py — convert the rich trajectory files (sft_trajectories_v3.jsonl) into the
FLAT schema the training pipeline consumes:

  {"id", "instruction", "plan": [primitives...], "answer", "checker_kind", "checker_args", "category"}

- plan   = concatenation of every planner-emitted program in the trajectory, in order
           (this is the gold plan-of-primitives the planner head learns to produce).
- answer = the final_answer content, stripped of markdown emphasis.
- checker = derived from the answer: if the gold answer is a number/short token we use
           "exact" matching; otherwise "contains_all" of the key tokens. This gives RL a
           programmatic, verifiable reward instead of NLL/embedding-sim.

Usage:
  python flatten_to_sft.py --in sft_trajectories_v3.jsonl --out sft_flat.jsonl
"""
import json, argparse, re

def extract_plan(turns):
    plan=[]
    for t in turns:
        if t.get("type")=="primitive_program":
            plan.extend(t.get("program",[]))
    return plan

def final_answer(turns):
    for t in reversed(turns):
        if t.get("type")=="final_answer":
            return t.get("content","")
    # fallback: last assistant turn
    for t in reversed(turns):
        if t.get("role")=="assistant":
            return t.get("content","")
    return ""

def clean(ans):
    a=ans.replace("**","").strip()
    return a

def derive_checker(ans):
    a=clean(ans)
    # numeric / short answer -> exact match on the salient token
    m=re.findall(r"-?\d[\d,]*\.?\d*", a)
    if m:
        gold=m[-1].replace(",","")
        return "exact", {"gold":gold}
    # very short text answer -> exact on normalized string
    if len(a.split())<=4:
        return "exact", {"gold":a}
    # longer answer -> require the key content tokens to all appear
    # pick distinctive tokens (len>3, not stopwords)
    stop={"the","and","for","with","that","this","from","each","need","needs","metric"}
    toks=[w.strip(".,:;").lower() for w in a.split() if len(w)>3 and w.lower() not in stop]
    key=toks[:4] if toks else [a.lower()]
    return "contains_all", {"tokens":key}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="sft_trajectories_v3.jsonl")
    ap.add_argument("--out", default="sft_flat.jsonl")
    a=ap.parse_args()
    n=0
    with open(a.inp) as f, open(a.out,"w") as g:
        for line in f:
            r=json.loads(line)
            turns=r["turns"]
            ans=clean(final_answer(turns))
            ck,ckargs=derive_checker(ans)
            row={"id":r["id"],"category":r.get("category",""),
                 "instruction":r["instruction"],
                 "plan":extract_plan(turns),
                 "answer":ans,
                 "checker_kind":ck,"checker_args":ckargs}
            g.write(json.dumps(row,ensure_ascii=False)+"\n"); n+=1
    print(f"wrote {n} rows to {a.out}")

if __name__=="__main__":
    main()
