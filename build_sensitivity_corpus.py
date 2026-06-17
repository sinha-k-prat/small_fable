#!/usr/bin/env python3
"""
build_sensitivity_corpus.py
============================
Implements the §3.5 principle: keep only tasks where decomposition is LOAD-BEARING.

    planning_sensitivity(task) = quality(gold_plan) - quality(no_plan)

measured WITH THE BASE MODEL and a VERIFIABLE correctness signal (exact-match /
constraint-satisfaction), NOT NLL or embedding-sim (those saturate on easy tasks and
hide whether the plan helped).

Pipeline
--------
1. GENERATE candidate tasks across 5 planning-sensitive families, each carrying:
     - a gold op-pipeline (parameterized ops: FILTER[even], TOP_K[k=3], ...),
     - a programmatic CHECKER (returns True/False on a model answer),
     - a deterministic gold answer.
2. MEASURE sensitivity per task with the base model:
     q_plan  = mean correctness over N samples WITH the gold plan in context
     q_naive = mean correctness over N samples with NO plan (naive execution)
     sensitivity = q_plan - q_naive
3. FILTER: keep tasks with sensitivity >= TAU (default 0.30) AND q_naive <= CEIL
   (so naive isn't already solving it) AND q_plan >= FLOOR (so the plan is achievable).
   This GUARANTEES SFT headroom + GRPO group-variance by construction.
4. EMIT plan_dataset_sensitive.jsonl in the same schema train_joint.py expects,
   plus sensitivity_report.csv for auditing the distribution.

Run with a real model:
    python build_sensitivity_corpus.py --base Qwen/Qwen2.5-1.5B-Instruct \
        --n_candidates 1500 --samples 4 --tau 0.30 --device cuda --out dataset/plan_dataset_sensitive.jsonl

Dry-run the pipeline logic without a GPU (uses a mock scorer that simulates a base
model which benefits from plans on hard items):
    python build_sensitivity_corpus.py --mock --n_candidates 400 --out /tmp/mock.jsonl
"""
import json, random, argparse, statistics as st
from collections import Counter

# ----------------------------------------------------------------------------
# Parameterized op vocabulary (finer plan space => more outcome-determining bits)
# ----------------------------------------------------------------------------
OPS = ["EXTRACT","MODEL","SETUP","COMPUTE","FILTER","MAP","SORT","TOP_K",
       "AGGREGATE","DEDUCE","ENUMERATE","CHECK_CONSTRAINT","VERIFY","REVISE","TERMINATE"]

# ----------------------------------------------------------------------------
# Task families. Each generator returns a dict with:
#   prompt, gold_plan (list of parameterized op strings), gold_answer (str),
#   checker_kind + checker_args (so we can rebuild a checker without pickling)
# ----------------------------------------------------------------------------
def _check_exact(ans, gold):
    return _norm(ans) == _norm(gold)
def _norm(s):
    return "".join(str(s).lower().split()).rstrip(".")

def fam_gsm(r):
    # multi-step word problem; order-sensitive (extract->setup->compute->verify)
    a=r.randint(3,12); b=r.randint(4,15); c=r.randint(1,b-1)   # c < b: can't remove more than a crate holds
    total=a*b; kept=total-c*a
    prompt=(f"There are {a} crates, each holding {b} apples. You remove {c} apples from "
            f"every crate. How many apples remain in total?")
    plan=["EXTRACT[crates,per_crate,removed_each]","SETUP[total=crates*per_crate]",
          "COMPUTE[removed=removed_each*crates]","COMPUTE[remain=total-removed]","VERIFY","TERMINATE"]
    return dict(prompt=prompt, gold_plan=plan, gold_answer=str(kept),
                checker_kind="exact", checker_args={"gold":str(kept)}, family="gsm")

def fam_pipeline(r):
    # order-sensitive pipeline: sort -> top_k -> aggregate. Wrong order => wrong answer.
    nums=[r.randint(1,99) for _ in range(r.randint(6,9))]
    k=r.randint(2,3)
    gold=sum(sorted(nums,reverse=True)[:k])
    prompt=(f"Given the list {nums}: take the {k} largest values and report their sum.")
    plan=[f"SORT[desc]",f"TOP_K[k={k}]","AGGREGATE[sum]","VERIFY","TERMINATE"]
    return dict(prompt=prompt, gold_plan=plan, gold_answer=str(gold),
                checker_kind="exact", checker_args={"gold":str(gold)}, family="pipeline")

def fam_filter(r):
    # filter-then-aggregate; naive often sums everything or mis-filters
    nums=[r.randint(1,50) for _ in range(r.randint(7,11))]
    gold=sum(x for x in nums if x%2==0)
    prompt=(f"From the list {nums}, sum only the even numbers.")
    plan=["FILTER[even]","AGGREGATE[sum]","VERIFY","TERMINATE"]
    return dict(prompt=prompt, gold_plan=plan, gold_answer=str(gold),
                checker_kind="exact", checker_args={"gold":str(gold)}, family="filter")

def fam_constraint(r):
    # multi-constraint satisfaction: enumerate -> filter-by-constraint -> check
    lo,hi=r.randint(10,30), r.randint(60,90)
    # find numbers in [lo,hi] divisible by 3 and by 4 (=>by 12); report count
    gold=sum(1 for x in range(lo,hi+1) if x%12==0)
    prompt=(f"How many integers from {lo} to {hi} inclusive are divisible by BOTH 3 and 4?")
    plan=["ENUMERATE[range]","CHECK_CONSTRAINT[div3]","CHECK_CONSTRAINT[div4]",
          "AGGREGATE[count]","VERIFY","TERMINATE"]
    return dict(prompt=prompt, gold_plan=plan, gold_answer=str(gold),
                checker_kind="exact", checker_args={"gold":str(gold)}, family="constraint")

def fam_multihop(r):
    # multi-hop deduction: decompose -> deduce -> check-logic
    ppl=r.sample(["Ana","Ben","Cara","Dan","Eli"],3); o=ppl[:]; r.shuffle(o)
    a,b,c=o  # a<b<c by some attribute (height)
    prompt=(f"{b} is taller than {a}. {c} is taller than {b}. Who is the shortest, "
            f"and who is in the middle?")
    gold=f"{a},{b}"
    plan=["EXTRACT[relations]","ORDER[by_height]","DEDUCE[shortest,middle]","VERIFY","TERMINATE"]
    return dict(prompt=prompt, gold_plan=plan, gold_answer=gold,
                checker_kind="exact", checker_args={"gold":gold}, family="multihop")

FAMILIES=[fam_gsm,fam_pipeline,fam_filter,fam_constraint,fam_multihop]

# ----------------------------------------------------------------------------
# Checker rebuild (no pickling): map kind -> callable
# ----------------------------------------------------------------------------
def make_checker(kind,args):
    if kind=="exact":
        gold=args["gold"]
        return lambda ans: _check_exact(ans,gold)
    raise ValueError(kind)

# ----------------------------------------------------------------------------
# Base-model scorer. Real path uses transformers; mock path simulates a model that
# (a) is decent at naive on easy items, (b) gets a real lift from the gold plan on
# hard items, (c) is noisy -> produces group variance. This lets you validate the
# FILTER LOGIC and schema without a GPU.
# ----------------------------------------------------------------------------
class MockScorer:
    """Simulates correctness prob; harder families benefit more from a plan."""
    LIFT={"gsm":0.45,"pipeline":0.55,"filter":0.25,"constraint":0.50,"multihop":0.40}
    NAIVE={"gsm":0.35,"pipeline":0.20,"filter":0.55,"constraint":0.25,"multihop":0.45}
    def __init__(self,seed=0): self.r=random.Random(seed)
    def correctness(self, task, with_plan, n):
        base=self.NAIVE[task["family"]]
        p=min(0.97, base + (self.LIFT[task["family"]] if with_plan else 0.0))
        return sum(1 for _ in range(n) if self.r.random()<p)/n

class HFScorer:
    """Real base-model scorer. Lazy-imports transformers so --mock needs no GPU."""
    def __init__(self, base, device, max_new=96, temp=1.3):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch=torch
        self.tok=AutoTokenizer.from_pretrained(base)
        self.model=AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto").to(device)
        self.model.eval(); self.device=device; self.max_new=max_new; self.temp=temp
    def _gen(self, prompt, n):
        msgs=[{"role":"user","content":prompt}]
        text=self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids=self.tok([text]*n, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out=self.model.generate(**ids, do_sample=True, temperature=self.temp,
                                     max_new_tokens=self.max_new, pad_token_id=self.tok.eos_token_id)
        gen=out[:, ids["input_ids"].shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]
    def correctness(self, task, with_plan, n):
        checker=make_checker(task["checker_kind"], task["checker_args"])
        if with_plan:
            plan_str=" ".join(task["gold_plan"])
            prompt=f"{task['prompt']}\n\nFollow this exact plan of operations: {plan_str}\nGive only the final answer."
        else:
            prompt=f"{task['prompt']}\nGive only the final answer."
        outs=self._gen(prompt, n)
        # extract a trailing number/token heuristically; checker does the norm-compare
        return sum(1 for o in outs if checker(_last_token(o)))/n

def _last_token(s):
    # grab last number if present, else last word
    import re
    nums=re.findall(r"-?\d+", s)
    if nums: return nums[-1]
    ws=s.strip().split()
    return ws[-1] if ws else ""

# ----------------------------------------------------------------------------
# Main: generate candidates, measure sensitivity, filter, emit
# ----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n_candidates", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=4, help="N samples per (task,condition)")
    ap.add_argument("--tau", type=float, default=0.30, help="min planning sensitivity to keep")
    ap.add_argument("--naive_ceil", type=float, default=0.60, help="drop if naive already solves it")
    ap.add_argument("--plan_floor", type=float, default=0.50, help="drop if even gold plan can't do it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--out", default="dataset/plan_dataset_sensitive.jsonl")
    ap.add_argument("--report", default="dataset/sensitivity_report.csv")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--max_new", type=int, default=64, help="answer is just the final token; keep short for speed")
    ap.add_argument("--progress_every", type=int, default=25, help="print progress + ETA every N candidates")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted build (within the same runtime)")
    args=ap.parse_args()

    import os, time
    rng=random.Random(args.seed)
    scorer = MockScorer(args.seed) if args.mock else HFScorer(
        args.base, args.device, max_new=args.max_new, temp=args.temp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prog_file = args.out + ".progress"

    # RESUME (within a runtime): replay the RNG to the saved index, append to the existing file.
    start, fmode, kept_n = 0, "w", 0
    if args.resume and os.path.exists(args.out) and os.path.exists(prog_file):
        start = int((open(prog_file).read().strip() or "0"))
        kept_n = sum(1 for _ in open(args.out))
        for i in range(start):
            FAMILIES[i % len(FAMILIES)](rng)        # advance RNG (task-gen is cheap, no model)
        fmode = "a"
        print(f"[build] RESUMING at candidate {start}/{args.n_candidates} (kept so far: {kept_n})", flush=True)

    rows_report=[]; fam_keep=Counter()
    fout = open(args.out, fmode)
    t0 = time.time()
    for i in range(start, args.n_candidates):
        task=FAMILIES[i%len(FAMILIES)](rng)
        q_plan = scorer.correctness(task, with_plan=True,  n=args.samples)
        q_naive= scorer.correctness(task, with_plan=False, n=args.samples)
        sens   = round(q_plan - q_naive, 3)
        keep = (sens>=args.tau and q_naive<=args.naive_ceil and q_plan>=args.plan_floor)
        rows_report.append((task["family"], q_plan, q_naive, sens, int(keep)))
        if keep:
            fam_keep[task["family"]]+=1; kept_n+=1
            fout.write(json.dumps({
                "id": f"task_{i:05d}", "family": task["family"], "prompt": task["prompt"],
                "plan": task["gold_plan"],            # gold op-pipeline (planner-head target)
                "answer": task["gold_answer"],        # checkable gold
                "checker_kind": task["checker_kind"], "checker_args": task["checker_args"],
                "sensitivity": sens, "q_plan": round(q_plan,3), "q_naive": round(q_naive,3),
            }, ensure_ascii=False)+"\n")
            fout.flush()                              # incremental: a crash never loses kept rows
        if (i+1) % args.progress_every == 0 or (i+1)==args.n_candidates:
            done=i+1-start; rate=done/max(1e-9, time.time()-t0); eta=(args.n_candidates-(i+1))/max(1e-9, rate)
            open(prog_file,"w").write(str(i+1))       # resume marker
            print(f"[build] {i+1}/{args.n_candidates}  kept={kept_n}  "
                  f"({rate:.1f} cand/s, ETA {eta/60:.0f} min)", flush=True)
    fout.close()

    with open(args.report,"w") as f:                  # this session's audit rows (partial after resume)
        f.write("family,q_plan,q_naive,sensitivity,kept\n")
        for fa,qp,qn,se,k in rows_report:
            f.write(f"{fa},{qp:.3f},{qn:.3f},{se:.3f},{k}\n")
    sens_all=[r[3] for r in rows_report]; sens_kept=[r[3] for r in rows_report if r[4]]
    print(f"candidates: {args.n_candidates} | total kept in {args.out}: {kept_n}")
    if sens_all:  print(f"sensitivity (this session) mean {st.mean(sens_all):+.3f}")
    if sens_kept: print(f"kept sensitivity mean {st.mean(sens_kept):+.3f}  min {min(sens_kept):+.3f}")
    print("kept by family (this session):", dict(fam_keep))
    if os.path.exists(prog_file): os.remove(prog_file)   # completed run leaves no progress marker
    print(f"wrote {args.out} and {args.report}")
    print("\nNOTE: kept tasks have P(correct|plan) >> P(correct|naive) BY CONSTRUCTION,")
    print("so SFT has headroom and GRPO groups will have reward variance.")

if __name__=="__main__":
    main()
