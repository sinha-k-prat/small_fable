#!/usr/bin/env python3
"""
train_sft.py — Stage 1: joint SFT of the planner head + executor backbone.

Loss (teacher-forced):
    L = CE(plan_head, gold_plan) + lam_resp * CE(executor, gold_answer) + lam_kl * KL(executor || base)

Diagnostic that matters most: the PLAN-VS-NO-PLAN ABLATION GAP — three conditions:
  - gold plan:   decode with the gold (teacher-forced) plan
  - random plan: decode with a plan sampled uniformly from the plan vocab (no backbone call)
  - no plan:     decode with no plan prefix at all
  gap_content = acc_gold  - acc_random  -> proves the plan CONTENT matters, not just its presence
  gap_presence = acc_random - acc_noplan -> proves the soft prefix itself helps
Both should be positive. If only gap_presence > 0, the plan acts as a warm-up token, not strategy.

ADDENDA
-------
A3  Two-stage curriculum (broad -> hard), enabled with --curriculum:
      Stage 1 (broad coverage): FULL set, cosine LR (lr -> lr_min), warmup, ~5 epochs.
      Stage 2 (hard reasoning): init from stage-1, train on a HARD SUBSET selected by the stage-1
              model itself (8 rollouts/prompt, keep error-rate >= 0.75 i.e. solved <= 2/8), ~2 epochs.
      Logs the held ablation gap after BOTH stages; expect it to GROW in stage 2.
      Supports --resume: stage-2 hard ids are saved to {out}/curriculum_hard_ids.json so a
      killed run can be resumed mid-stage-2 without re-running stage 1.
A4  Spectrum-to-signal: data rows may carry >1 (plan, answer) via an "alternatives" list; we train
    on all of them so the post-SFT model samples DIVERSE rollouts (=> non-zero GRPO variance). After
    SFT we measure Pass@k and rollout PLAN DIVERSITY on a probe set; low diversity here PREDICTS
    zero-variance RL groups — fix it in SFT, not RL.
Curriculum batching: batches are ordered EASY -> HARD by a difficulty proxy (plan length + answer
    length), with light intra-band shuffling to keep category coverage / stochasticity.

DATA SPLIT
  Rows are SHUFFLED with a seeded RNG before splitting into train/held so the held set is
  a representative random sample, not a positional slice (which can be biased if the data
  has ordering structure).

Saves joint_ckpt/ = LoRA adapter + planner head + plan embeddings + tokenizer + joint_config.json
                    + plan_vocab.json (self-contained; no ambient plan_vocab.json needed to reload).

CLI (single stage, base spec):
  python train_sft.py --data dataset/sft_100.jsonl --train 70 --held 30 --epochs 6 --device cuda
CLI (A3 two-stage curriculum):
  python train_sft.py --data dataset/sft_flat.jsonl --train 800 --held 100 --curriculum \
      --stage1_epochs 5 --stage2_epochs 2 --lr 5e-5 --lr_min 8e-8 --device cuda
CLI (A3 resume after kill):
  python train_sft.py --data dataset/sft_flat.jsonl --train 800 --held 100 --curriculum \
      --stage1_epochs 5 --stage2_epochs 2 --resume --device cuda
"""
import argparse, json, math, os, random, time
import torch, torch.nn.functional as F

import random as _random_mod

def _gpu_mem() -> str:
    """One-line GPU memory summary: alloc / peak / total in GB with % used."""
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct   = 100 * peak / total
    return f"gpu: alloc={alloc:.1f}GB peak={peak:.1f}GB/{total:.0f}GB ({pct:.0f}%)"
from model_joint import JointModel, encode_plan, decode_plan, PAD_ID, N_PLAN
from checkers import graded_reward_for_row
from checkpointing import (Checkpointer, load_train_state, restore_optimizer, restore_rng,
                           scalar_args)

_HARD_IDS_FILE = "curriculum_hard_ids.json"


# --------------------------------------------------------------------------- data
def load_rows(path):
    return [json.loads(l) for l in open(path)]


def expand_alternatives(rows):
    """A4: expand a row with multiple (plan, answer) pairs into multiple training items."""
    out = []
    for r in rows:
        alts = r.get("alternatives")
        if alts:
            for a in alts:
                rr = dict(r); rr["plan"] = a["plan"]; rr["answer"] = a["answer"]
                out.append(rr)
        else:
            out.append(r)
    return out


def difficulty(row):
    """Easy->hard proxy for curriculum ordering: plan length dominates, answer length breaks ties."""
    return len(row.get("plan", [])) + 0.01 * len(str(row.get("answer", "")).split())


def curriculum_batches(rows, bs, seed=0):
    """Yield batches ordered EASY -> HARD. Within equal-difficulty bands we shuffle so categories
    interleave (broad coverage) without destroying the global easy->hard progression."""
    rng = random.Random(seed)
    bands = {}
    for r in rows:
        bands.setdefault(round(difficulty(r)), []).append(r)
    ordered = []
    for d in sorted(bands):
        b = bands[d][:]; rng.shuffle(b); ordered.extend(b)
    for i in range(0, len(ordered), bs):
        yield ordered[i:i+bs]


# --------------------------------------------------------------------------- losses
def tok_answer(model, answers, max_len):
    enc = model.tok(answers, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    return enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)


def plan_ce(model, prompt_ids, prompt_attn, plan_ids):
    logits = model.planner_logits_tf(prompt_ids, prompt_attn, plan_ids)
    mask = (plan_ids != PAD_ID)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), plan_ids.reshape(-1),
                         reduction="none").reshape(plan_ids.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def resp_ce_and_kl(model, prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn, lam_kl):
    logits = model.executor_logits_tf(prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn)
    mask = resp_attn.bool()
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), resp_ids.reshape(-1),
                         reduction="none").reshape(resp_ids.shape)
    ce = (ce * mask).sum() / mask.sum().clamp_min(1)
    kl = torch.tensor(0.0, device=logits.device)
    if lam_kl > 0:
        base_logits = model.base_executor_logits(prompt_ids, prompt_attn, plan_ids, resp_ids, resp_attn)
        lp = F.log_softmax(logits, dim=-1); lq = F.log_softmax(base_logits, dim=-1)
        kl_tok = (lp.exp() * (lp - lq)).sum(-1)
        kl = (kl_tok * mask).sum() / mask.sum().clamp_min(1)
    return ce, kl


# --------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def eval_held(model, rows, max_resp, sample=False, temp=0.7):
    """Held-out plan CE, answer CE, and the three-way ablation gap.

    Three decoding conditions:
      gold plan   — the teacher-forced gold plan from the dataset
      random plan — uniformly sampled from plan vocab (no backbone; tests content vs presence)
      no plan     — zero soft prefix

    gap_content  = acc_gold - acc_random  (plan content is load-bearing)
    gap_presence = acc_random - acc_noplan (soft prefix itself helps)
    ablation_gap = acc_gold - acc_noplan   (overall: retained for backwards compat)
    """
    if not rows:
        return {}
    model.eval()
    pce_sum = rce_sum = 0.0
    corr_plan = corr_randplan = corr_noplan = 0.0
    for r in rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]])
        plan_ids = encode_plan(r["plan"], model.plan_max_len).unsqueeze(0).to(model.device)
        r_ids, r_attn = tok_answer(model, [r["answer"]], max_resp)
        pce_sum += float(plan_ce(model, p_ids, p_attn, plan_ids))
        rce, _ = resp_ce_and_kl(model, p_ids, p_attn, plan_ids, r_ids, r_attn, 0.0)
        rce_sum += float(rce)
        g_plan = model.generate_answer(p_ids, p_attn, plan_ids, sample=sample, temp=temp,
                                       max_new_tokens=max_resp)
        rand_plan = model.sample_random_plan(p_ids, max_len=model.plan_max_len)
        g_rand = model.generate_answer(p_ids, p_attn, rand_plan, sample=sample, temp=temp,
                                       max_new_tokens=max_resp)
        g_none = model.generate_answer(p_ids, p_attn, None, sample=sample, temp=temp,
                                       max_new_tokens=max_resp)
        corr_plan    += graded_reward_for_row(r, model.tok.decode(g_plan[0], skip_special_tokens=True))
        corr_randplan += graded_reward_for_row(r, model.tok.decode(g_rand[0], skip_special_tokens=True))
        corr_noplan  += graded_reward_for_row(r, model.tok.decode(g_none[0], skip_special_tokens=True))
    n = len(rows)
    acc_gold = corr_plan / n
    acc_rand = corr_randplan / n
    acc_none = corr_noplan / n
    return {"plan_ce": pce_sum/n, "resp_ce": rce_sum/n,
            "acc_gold_plan": acc_gold, "acc_random_plan": acc_rand, "acc_noplan": acc_none,
            "gap_content":  acc_gold - acc_rand,   # plan CONTENT matters (not just presence)
            "gap_presence": acc_rand - acc_none,   # soft prefix itself helps
            "ablation_gap": acc_gold - acc_none}   # overall (backwards compat)


@torch.no_grad()
def eval_held_interleaved(model, rows, max_turns, max_plan, max_resp, sample=False, temp=0.7,
                          seed=0):
    """Held-out interleaved diagnostics. Three closed-loop decodes per prompt:
      acc_chosen : run_interleaved with the model's own plans.
      acc_noplan : force EMPTY plans every turn (BOP->END) -> headline ablation_gap.
      acc_shuffle: force RANDOM in-vocab primitives every turn -> shuffle_gap.
    Grades the LAST turn's prose tail (ends 'FINAL ANSWER: X'). Reports both gaps."""
    if not rows:
        return {}
    model.eval()
    # in-vocab primitives to draw shuffled plans from (exclude PAD + markers/terminator)
    from model_joint import PLAN_VOCAB, BOP_ID, FINALL_ID, EOP_ID
    pool = [i for i in range(len(PLAN_VOCAB))
            if i not in (PAD_ID, BOP_ID, FINALL_ID, EOP_ID)]
    rng = _random_mod.Random(seed)
    pce_sum = 0.0
    corr_chosen = corr_noplan = corr_shuffle = 0.0
    for r in rows:
        turns = r.get("turns") or [{"plan": r.get("plan", []), "response": r.get("answer", "")}]
        p_ids, p_attn = model.batch_prompts([r["instruction"]])
        try:
            _, logs = model.interleaved_loss(p_ids, p_attn, [turns], lam_resp=1.0, lam_kl=0.0)
            pce_sum += logs["ce_plan"]
        except Exception:
            pass
        rec_c = model.run_interleaved(p_ids[0], p_attn[0], sample=sample, temp=temp,
                                      max_turns=max_turns, max_plan=max_plan, max_resp=max_resp)
        rec_n = model.run_interleaved(p_ids[0], p_attn[0], sample=sample, temp=temp,
                                      max_turns=max_turns, max_plan=max_plan, max_resp=max_resp,
                                      force_plan=lambda t: [])
        rec_s = model.run_interleaved(
            p_ids[0], p_attn[0], sample=sample, temp=temp, max_turns=max_turns,
            max_plan=max_plan, max_resp=max_resp,
            force_plan=lambda t: [rng.choice(pool) for _ in range(max(1, rng.randint(1, 3)))])
        corr_chosen += graded_reward_for_row(r, model.interleaved_answer_text(rec_c))
        corr_noplan += graded_reward_for_row(r, model.interleaved_answer_text(rec_n))
        corr_shuffle += graded_reward_for_row(r, model.interleaved_answer_text(rec_s))
    n = len(rows)
    return {"plan_ce": pce_sum / n, "acc_chosen": corr_chosen / n,
            "acc_noplan": corr_noplan / n, "acc_shuffle": corr_shuffle / n,
            "ablation_gap": (corr_chosen - corr_noplan) / n,
            "shuffle_gap": (corr_chosen - corr_shuffle) / n}


@torch.no_grad()
def probe_diversity(model, rows, group=8, max_resp=64, temp=1.0):
    """A4: post-SFT rollout diversity + Pass@k on a probe set."""
    model.eval()
    distinct, passk = [], 0
    for r in rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]] * group)
        plans = model.sample_plan(p_ids, p_attn, temp=max(temp, 1.3), sample=True)
        gen = model.generate_answer(p_ids, p_attn, plans, sample=True, temp=max(temp, 1.3),
                                    max_new_tokens=max_resp)
        plan_strs = {tuple(decode_plan(plans[i])) for i in range(group)}
        distinct.append(len(plan_strs))
        if any(graded_reward_for_row(r, model.tok.decode(gen[i], skip_special_tokens=True)) >= 0.5
               for i in range(group)):
            passk += 1
    n = len(rows)
    return {"distinct_plans_per_prompt": sum(distinct)/n, f"pass@{group}": passk/n}


@torch.no_grad()
def hard_subset(model, rows, samples=8, max_resp=64, temp=1.3, err_rate=0.75):
    """A3 stage-2 filter: keep prompts the CURRENT model rarely solves."""
    model.eval()
    keep_max = math.floor((1 - err_rate) * samples)
    hard = []
    for r in rows:
        p_ids, p_attn = model.batch_prompts([r["instruction"]] * samples)
        plans = model.sample_plan(p_ids, p_attn, temp=temp, sample=True)
        gen = model.generate_answer(p_ids, p_attn, plans, sample=True, temp=temp,
                                    max_new_tokens=max_resp)
        solved = sum(graded_reward_for_row(r, model.tok.decode(gen[i], skip_special_tokens=True)) >= 0.5
                     for i in range(samples))
        if solved <= keep_max:
            hard.append(r)
    return hard


# --------------------------------------------------------------------------- training
def _sft_state(epoch, batch_idx, global_step, opt, sched, args, stage=None):
    """Resume payload: position = next (epoch, batch_idx) to run."""
    return {"kind": "sft", "epoch": epoch, "batch_idx": batch_idx, "global_step": global_step,
            "stage": stage,   # None = single-stage; 1 or 2 for curriculum
            "n_plan": N_PLAN, "optimizer": opt.state_dict(),
            "scheduler": (sched.state_dict() if sched is not None else None),
            "torch_rng": torch.get_rng_state(), "py_rng": random.getstate(),
            "args": scalar_args(args)}


def run_stage(model, opt, sched, stage_rows, epochs, args, tag,
              ckpt=None, start_epoch=0, start_batch=0, global_step=0, stage=None):
    held = {"ablation_gap": 0.0}
    for ep in range(start_epoch, epochs):
        model.train(); t0 = time.time()
        run = {"plan_ce": 0.0, "resp_ce": 0.0, "kl": 0.0, "n": 0}
        batches = list(curriculum_batches(stage_rows, args.bs, seed=args.seed + ep))
        sb = start_batch if ep == start_epoch else 0
        for bi in range(len(batches)):
            if bi < sb:
                continue
            batch = batches[bi]
            instrs = [r["instruction"] for r in batch]
            p_ids, p_attn = model.batch_prompts(instrs)
            plan_ids = torch.stack([encode_plan(r["plan"], model.plan_max_len)
                                    for r in batch]).to(model.device)
            r_ids, r_attn = tok_answer(model, [r["answer"] for r in batch], args.max_resp)
            l_plan = plan_ce(model, p_ids, p_attn, plan_ids)
            l_resp, l_kl = resp_ce_and_kl(model, p_ids, p_attn, plan_ids, r_ids, r_attn,
                                           args.lam_kl)
            loss = l_plan + args.lam_resp * l_resp + args.lam_kl * l_kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            global_step += 1
            run["plan_ce"] += float(l_plan); run["resp_ce"] += float(l_resp)
            run["kl"] += float(l_kl); run["n"] += 1
            if ckpt is not None and ckpt.due(global_step):
                ckpt.save(model, _sft_state(ep, bi + 1, global_step, opt, sched, args, stage),
                          reason=f"{tag}-e{ep+1}-b{bi+1}")
        n = max(run["n"], 1)
        held = eval_held(model, args._held_rows, args.max_resp, sample=args.eval_sample)
        torch.cuda.reset_peak_memory_stats()   # reset so next epoch peak is fresh
        print(f"[sft:{tag}] epoch {ep+1}/{epochs} ({time.time()-t0:.1f}s) "
              f"plan_ce={run['plan_ce']/n:.4f} resp_ce={run['resp_ce']/n:.4f} kl={run['kl']/n:.4f} "
              f"| held {json.dumps(held)} | {_gpu_mem()}")
        if getattr(args, "metrics_out", None):
            rec = {"tag": tag, "epoch": ep + 1, "time_s": round(time.time() - t0, 1),
                   "train_plan_ce": round(run["plan_ce"]/n, 4),
                   "train_resp_ce": round(run["resp_ce"]/n, 4),
                   "kl": round(run["kl"]/n, 4),
                   **{f"held_{k}": round(v, 4) for k, v in held.items()}}
            with open(args.metrics_out, "a") as mf:
                mf.write(json.dumps(rec) + "\n")
        if ckpt is not None:
            ckpt.save(model, _sft_state(ep + 1, 0, global_step, opt, sched, args, stage),
                      reason=f"{tag}-epoch{ep+1}")
    return held or {"ablation_gap": 0.0}


def make_opt_sched(model, lr, lr_min, total_steps, warmup_frac):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    try:
        from transformers import get_cosine_schedule_with_warmup
        warmup = max(1, int(warmup_frac * total_steps))
        floor = lr_min / lr
        base = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
        orig = base.lr_lambdas[0]
        base.lr_lambdas[0] = (lambda step, _o=orig, _f=floor: max(_f, _o(step)))
        sched = base
    except Exception:
        sched = None
    return opt, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/sft_100.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", type=int, default=70)
    ap.add_argument("--held", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=6, help="single-stage epochs (no --curriculum)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_min", type=float, default=8e-8)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lam_resp", type=float, default=1.0)
    ap.add_argument("--lam_kl", type=float, default=0.1)
    ap.add_argument("--max_resp", type=int, default=64)
    ap.add_argument("--plan_max_len", type=int, default=12)
    ap.add_argument("--interleaved", action="store_true",
                    help="agentic closed-loop SFT over turns:[{plan,resp}] (model.interleaved_loss)")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_plan", type=int, default=12, help="interleaved per-turn plan token cap")
    ap.add_argument("--out", default="joint_ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None,
                    help="model dtype: float32 | bfloat16 | auto (default: bfloat16 on CUDA, float32 on CPU)")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--eval_sample", action="store_true")
    # A3 curriculum
    ap.add_argument("--curriculum", action="store_true", help="A3: two-stage broad->hard")
    ap.add_argument("--stage1_epochs", type=int, default=5)
    ap.add_argument("--stage2_epochs", type=int, default=2)
    ap.add_argument("--hard_err_rate", type=float, default=0.75)
    ap.add_argument("--hard_samples", type=int, default=8)
    ap.add_argument("--stage1_out", default=None, help="optional path to save the stage-1 checkpoint")
    # A4 probe
    ap.add_argument("--probe", type=int, default=16, help="probe set size for diversity/Pass@k (0=skip)")
    # checkpoint / resume
    ap.add_argument("--ckpt_every_min", type=float, default=0.0)
    ap.add_argument("--ckpt_every_steps", type=int, default=0)
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="resume from --out if it contains a train_state.pt")
    ap.add_argument("--metrics_out", default="sft_metrics.jsonl")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    # resolve dtype
    _dtype = None
    if args.dtype and args.dtype != "auto":
        _dtype = getattr(torch, args.dtype)

    rows = load_rows(args.data)
    # Seeded shuffle BEFORE split so held is a representative random sample, not a positional slice.
    random.Random(args.seed).shuffle(rows)
    assert len(rows) >= args.train + args.held, "not enough rows for the requested train/held split"
    train_rows = expand_alternatives(rows[:args.train])           # A4
    held_rows  = rows[args.train:args.train+args.held]
    probe_rows = held_rows[:args.probe] if args.probe else []
    args._held_rows = held_rows
    print(f"[sft] data={args.data} train={len(train_rows)} (expanded) held={len(held_rows)} base={args.base}")

    # ------------------------------------------------------------------
    # CURRICULUM RESUME PATH
    # ------------------------------------------------------------------
    if args.resume and args.curriculum:
        hard_ids_path = os.path.join(args.out, _HARD_IDS_FILE)
        resume_state = load_train_state(args.out) if os.path.isdir(args.out) else None

        if (resume_state is not None
                and resume_state.get("stage") == 2
                and os.path.exists(hard_ids_path)):
            # Resume was killed mid stage-2. Re-load hard rows from saved ids, skip stage 1.
            hard_ids = set(json.load(open(hard_ids_path)))
            hard = [r for r in train_rows if r.get("id") in hard_ids]
            print(f"[sft] curriculum RESUME mid-stage-2: {len(hard)} hard rows, skipping stage 1")
            model = JointModel.from_checkpoint(args.base, args.out, device=args.device,
                                               dtype=_dtype, is_trainable=True,
                                               plan_max_len=args.plan_max_len)
            model.backbone.gradient_checkpointing_enable()
            model.backbone.enable_input_require_grads()
            s2 = max(1, args.stage2_epochs * math.ceil(len(hard)/args.bs))
            opt, sched = make_opt_sched(model, args.lr, args.lr_min, s2, args.warmup_frac)
            restore_optimizer(opt, resume_state, model.device)
            if sched is not None and resume_state.get("scheduler"):
                try: sched.load_state_dict(resume_state["scheduler"])
                except Exception as e: print("[sft] scheduler restore skipped:", e)
            restore_rng(resume_state)
            s_ep = resume_state["epoch"]; s_bi = resume_state["batch_idx"]
            g_step = resume_state["global_step"]
            ckpt = Checkpointer(args.out, args.base, every_min=args.ckpt_every_min,
                                every_steps=args.ckpt_every_steps, hf_repo=args.hf_repo)
            gap2 = run_stage(model, opt, sched, hard, args.stage2_epochs, args, "stage2",
                             ckpt=ckpt, start_epoch=s_ep, start_batch=s_bi,
                             global_step=g_step, stage=2)
            print(f"[sft] curriculum resume complete. held ablation_gap={gap2['ablation_gap']:+.3f}")
            if probe_rows:
                print(f"[sft] A4 probe: {json.dumps(probe_diversity(model, probe_rows, max_resp=args.max_resp))}")
            model.save(args.out, args.base)
            return

        elif resume_state is not None and resume_state.get("stage") == 1:
            # Resume killed mid stage-1; continue normally (stage1_out not yet written).
            print("[sft] curriculum RESUME mid-stage-1: will re-run stage 1 from saved position, then stage 2")
        else:
            resume_state = None   # no valid state; start fresh
    else:
        # single-stage resume path (original behaviour)
        resume_state = None
        if args.resume and not args.curriculum and os.path.isdir(args.out):
            resume_state = load_train_state(args.out)
            if resume_state is not None:
                prev = resume_state.get("args", {})
                if (prev.get("data") != args.data or prev.get("train") != args.train
                        or resume_state.get("n_plan") != N_PLAN):
                    print(f"[sft] checkpoint config differs — ignoring --resume, starting fresh.")
                    resume_state = None

    # ------------------------------------------------------------------
    # MODEL CONSTRUCTION
    # ------------------------------------------------------------------
    if resume_state is not None:
        print(f"[sft] RESUMING from {args.out}: epoch={resume_state['epoch']} "
              f"batch={resume_state['batch_idx']} step={resume_state['global_step']}")
        model = JointModel.from_checkpoint(args.base, args.out, device=args.device,
                                           dtype=_dtype, is_trainable=True,
                                           plan_max_len=args.plan_max_len)
    else:
        model = JointModel.from_base(args.base, device=args.device, dtype=_dtype,
                                     plan_max_len=args.plan_max_len)
    print(f"[sft] trainable backbone (LoRA) tensors: {model.n_trainable_backbone()}")
    try:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
        print("[sft] gradient checkpointing ON.")
    except Exception as e:
        print(f"[sft] gradient checkpointing unavailable ({e})")
    if args.interleaved:
        print("[sft] ===== STAGE: interleaved (agentic, closed-loop) SFT =====")
    if resume_state is None:
        if args.interleaved:
            print("[sft] initial held:", json.dumps(eval_held_interleaved(
                model, held_rows, args.max_turns, args.max_plan, args.max_resp,
                sample=args.eval_sample, seed=args.seed)))
        else:
            print("[sft] initial held:",
                  json.dumps(eval_held(model, held_rows, args.max_resp, sample=args.eval_sample)))

    final_held = None

    # ------------------------------------------------------------------
    # SINGLE-STAGE PATH
    # ------------------------------------------------------------------
    if not args.curriculum:
        steps = max(1, args.epochs * math.ceil(len(train_rows)/args.bs))
        opt, sched = make_opt_sched(model, args.lr, args.lr_min, steps, args.warmup_frac)
        ckpt = Checkpointer(args.out, args.base, every_min=args.ckpt_every_min,
                            every_steps=args.ckpt_every_steps, hf_repo=args.hf_repo)
        s_ep = s_bi = g_step = 0
        if resume_state is not None:
            restore_optimizer(opt, resume_state, model.device)
            if sched is not None and resume_state.get("scheduler") is not None:
                try: sched.load_state_dict(resume_state["scheduler"])
                except Exception as e: print("[sft] scheduler restore skipped:", e)
            restore_rng(resume_state)
            s_ep, s_bi, g_step = (resume_state["epoch"], resume_state["batch_idx"],
                                  resume_state["global_step"])
        final_held = run_stage(model, opt, sched, train_rows, args.epochs, args, "single",
                               ckpt=ckpt, start_epoch=s_ep, start_batch=s_bi,
                               global_step=g_step, stage=None)

    # ------------------------------------------------------------------
    # CURRICULUM PATH (A3)
    # ------------------------------------------------------------------
    else:
        hard_ids_path = os.path.join(args.out, _HARD_IDS_FILE)

        # Stage 1: broad coverage
        s1 = max(1, args.stage1_epochs * math.ceil(len(train_rows)/args.bs))
        opt, sched = make_opt_sched(model, args.lr, args.lr_min, s1, args.warmup_frac)
        ckpt = Checkpointer(args.out, args.base, every_min=args.ckpt_every_min,
                            every_steps=args.ckpt_every_steps, hf_repo=args.hf_repo)

        # If resuming mid stage-1, restore position
        s_ep = s_bi = g_step = 0
        if resume_state is not None and resume_state.get("stage") == 1:
            restore_optimizer(opt, resume_state, model.device)
            restore_rng(resume_state)
            s_ep, s_bi, g_step = (resume_state["epoch"], resume_state["batch_idx"],
                                  resume_state["global_step"])

        gap1 = run_stage(model, opt, sched, train_rows, args.stage1_epochs, args, "stage1",
                         ckpt=ckpt, start_epoch=s_ep, start_batch=s_bi,
                         global_step=g_step, stage=1)
        if args.stage1_out:
            model.save(args.stage1_out, args.base)
            print(f"[sft] stage-1 saved -> {args.stage1_out}")

        # Stage 2: hard subset selected by the stage-1 model
        print("[sft] selecting hard subset with the stage-1 model "
              f"(keep error-rate >= {args.hard_err_rate}) ...")
        hard = hard_subset(model, train_rows, samples=args.hard_samples, max_resp=args.max_resp,
                           err_rate=args.hard_err_rate)
        print(f"[sft] hard subset: {len(hard)}/{len(train_rows)} prompts")

        if hard:
            # Persist hard ids so a killed stage-2 run can be resumed without re-running stage 1.
            json.dump([r.get("id", i) for i, r in enumerate(hard)], open(hard_ids_path, "w"))
            print(f"[sft] hard ids saved -> {hard_ids_path}")
            s2 = max(1, args.stage2_epochs * math.ceil(len(hard)/args.bs))
            opt, sched = make_opt_sched(model, args.lr, args.lr_min, s2, args.warmup_frac)
            gap2 = run_stage(model, opt, sched, hard, args.stage2_epochs, args, "stage2",
                             ckpt=ckpt, stage=2)
            print(f"[sft] ABLATION GAP: stage1={gap1['ablation_gap']:+.3f} "
                  f"stage2={gap2['ablation_gap']:+.3f} "
                  f"(curriculum should make plans MORE load-bearing: stage2 >= stage1)")
            final_held = gap2
        else:
            print("[sft] no hard prompts found — model already solves the set; skip stage 2.")
            final_held = gap1

    if probe_rows:
        div = probe_diversity(model, probe_rows, max_resp=args.max_resp)
        print(f"[sft] A4 probe: {json.dumps(div)}  (distinct_plans>1 predicts RL variance)")

    model.save(args.out, args.base)
    print(f"[sft] saved -> {args.out}")
    gap = (final_held or {}).get("ablation_gap")
    if gap is not None:
        if gap > 0:
            print(f"[sft] RESULT: held ablation_gap = {gap:+.3f}  ✓ plan is LOAD-BEARING.")
        else:
            print(f"[sft] RESULT: held ablation_gap = {gap:+.3f}  ✗ plan is NOT load-bearing on this "
                  "data. Use a plan-sensitive corpus (build_sensitivity_corpus.py).")
    gc = (final_held or {}).get("gap_content")
    gp = (final_held or {}).get("gap_presence")
    if gc is not None:
        print(f"[sft] gap_content={gc:+.3f}  gap_presence={gp:+.3f}  "
              f"(want both positive; gap_content>0 proves plan CONTENT matters)")
    print("[sft] acceptance targets: held plan_ce dropped; ablation_gap POSITIVE; "
          "gap_content POSITIVE; probe distinct_plans/prompt > 1"
          + ("; with --curriculum stage2 gap ≥ stage1." if args.curriculum else "."))


if __name__ == "__main__":
    main()
