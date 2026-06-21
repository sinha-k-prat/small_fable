#!/usr/bin/env python3
"""
train_sft_contrastive.py — plan-conditioned SFT with a contrastive (mutual-information) term.

Trains the EXECUTOR (Qwen2.5-1.5B + LoRA) to answer conditioned on a plan supplied as a TEXT
prefix, on the v4 executable-plan corpus (dataset/sft_synth_v4.jsonl), where answer == execute(plan)
and every row ships 3 oracle-verified HARD NEGATIVE plans (same problem, one step perturbed, answer
genuinely changed).

Per-example loss over {gold plan, K hard-neg plans}:

    L = CE(a* | instr, plan_gold)                               # (1) fit the answer under the right plan
      + beta * InfoNCE                                          # (2) MI: answer must be likelier under
      + lam_kl * KL(executor || base)  on answer tokens         #     the right plan than the wrong ones
                                                                # (3) anchor to base fluency

  InfoNCE = -log  exp(s_gold) / ( exp(s_gold) + Σ_k exp(s_neg_k) ),   s_x := -CE(a* | instr, plan_x)/tau

This is exactly the term derived in the design discussion: a lower bound on I(plan; answer | instr),
with the single-hinge as its K=1 special case. Because the negatives are oracle-verified true
negatives, the term has real gradient (unlike the v3 corpus, where it was structurally zero).

Outputs sft_contrastive_ckpt/ = LoRA adapter + tokenizer + config.json.

CLI:
  python train_sft_contrastive.py --data dataset/sft_synth_v4.jsonl --epochs 3 --device cuda
"""
import argparse, json, math, os, random, time
import torch, torch.nn.functional as F


# --------------------------------------------------------------------------- prompt formatting
def prompt_with_plan(instr, plan_str):
    return (f"Problem: {instr}\nPlan: {plan_str}\n"
            f"Execute the plan step by step, then commit.\nFINAL ANSWER:")

def prompt_no_plan(instr):
    return (f"Problem: {instr}\n"
            f"Solve step by step, then commit.\nFINAL ANSWER:")

def target_text(answer):
    return f" {answer}"


# --------------------------------------------------------------------------- tokenize -> (ids, labels)
def build_seq(tok, prompt, target, max_len):
    """Return (input_ids, labels) where labels mask the prompt (-100) and supervise only the
    answer tokens + EOS. Right-padded later by the collate."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    return ids, labels

def collate(seqs, pad_id, device):
    """seqs: list of (ids, labels). Right-pad to a batch; build attention mask."""
    m = max(len(ids) for ids, _ in seqs)
    input_ids, labels, attn = [], [], []
    for ids, lab in seqs:
        pad = m - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device),
            torch.tensor(attn, device=device))


def per_row_ce(logits, labels):
    """Token-mean cross-entropy per row over non-masked (answer) positions. Returns (B,) tensor."""
    sl = logits[:, :-1, :].contiguous()
    lb = labels[:, 1:].contiguous()
    ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), lb.reshape(-1),
                         ignore_index=-100, reduction="none").reshape(lb.shape)
    mask = (lb != -100).float()
    return (ce * mask).sum(1) / mask.sum(1).clamp_min(1.0)


# --------------------------------------------------------------------------- the three loss terms
def example_loss(model, tok, row, args, device):
    """CE(gold) + beta*InfoNCE(gold vs K negs) + lam_kl*KL(exec||base). One example.
    Returns (loss, logdict)."""
    instr = row["instruction"]
    variants = [(prompt_with_plan(instr, row["plan_str"]), target_text(row["answer"]))]   # gold = index 0
    for hn in row["hard_negatives"][:args.k]:
        variants.append((prompt_with_plan(instr, hn["plan_str"]), target_text(row["answer"])))
    seqs = [build_seq(tok, p, t, args.max_len) for p, t in variants]
    input_ids, labels, attn = collate(seqs, tok.pad_token_id, device)

    with torch.autocast(device_type="cuda", dtype=args.amp_dtype, enabled=args.amp):
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        ce = per_row_ce(logits, labels)                      # (1+K,)
        ce_pos = ce[0]
        ce_neg = ce[1:]

        # (2) InfoNCE over scores s = -CE/tau, positive at index 0  == cross-entropy with label 0
        scores = (-ce / args.tau).unsqueeze(0)               # (1, 1+K)
        infonce = F.cross_entropy(scores, torch.zeros(1, dtype=torch.long, device=device))

        # (3) KL(executor || base) on the GOLD answer tokens, base = adapter disabled
        kl = torch.zeros((), device=device)
        if args.lam_kl > 0:
            g_ids, g_lab, g_attn = input_ids[:1], labels[:1], attn[:1]
            with torch.no_grad(), model.disable_adapter():
                base_logits = model(input_ids=g_ids, attention_mask=g_attn).logits
            lp = F.log_softmax(logits[:1][:, :-1, :], -1)
            lq = F.log_softmax(base_logits[:, :-1, :], -1)
            kl_tok = (lp.exp() * (lp - lq)).sum(-1)           # (1, S-1)
            m = (g_lab[:, 1:] != -100).float()
            kl = (kl_tok * m).sum() / m.sum().clamp_min(1.0)

    loss = ce_pos + args.beta * infonce + args.lam_kl * kl
    gap = float(ce_neg.mean() - ce_pos)                       # MI estimate / training-time ablation gap
    return loss, {"ce": float(ce_pos), "infonce": float(infonce), "kl": float(kl),
                  "ce_neg": float(ce_neg.mean()), "gap": gap}


# --------------------------------------------------------------------------- decode eval (is the plan load-bearing?)
@torch.no_grad()
def eval_held(model, tok, rows, args, device):
    """Greedy-decode the answer WITH the gold plan vs WITH NO plan; grade both with checkers.py.
    The (with-plan − no-plan) accuracy gap is the headline: positive => plan is load-bearing."""
    from checkers import reward_for_row
    model.eval()
    acc_plan = acc_noplan = 0.0
    gap_ce = 0.0
    for r in rows:
        for tag, prompt in (("plan", prompt_with_plan(r["instruction"], r["plan_str"])),
                            ("noplan", prompt_no_plan(r["instruction"]))):
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=args.amp_dtype, enabled=args.amp):
                out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
            rew = reward_for_row(r, f"FINAL ANSWER: {text}")
            if tag == "plan":   acc_plan += rew
            else:               acc_noplan += rew
    model.train()
    n = max(1, len(rows))
    return {"acc_plan": acc_plan / n, "acc_noplan": acc_noplan / n,
            "ablation_gap": (acc_plan - acc_noplan) / n}


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/sft_synth_v4.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="sft_contrastive_ckpt")
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--held", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2, help="examples per optimizer step (each expands to 1+K seqs)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--k", type=int, default=3, help="hard negatives per example")
    ap.add_argument("--beta", type=float, default=1.0, help="InfoNCE (MI) weight")
    ap.add_argument("--tau", type=float, default=1.0, help="InfoNCE temperature on -CE scores")
    ap.add_argument("--lam_kl", type=float, default=0.1, help="KL-to-base anchor weight")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval_n", type=int, default=40)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    args.amp = (args.dtype != "fp32") and device.startswith("cuda")
    args.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    print(f"[ctr] loading {args.base} ({args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    load_dtype = torch.float32 if args.dtype == "fp32" else args.amp_dtype
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=load_dtype)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg).to(device)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            p.data = p.data.float()                  # keep LoRA params fp32 for stable AdamW
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    print(f"[ctr] trainable LoRA params: {n_lora:,}")

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    train_rows = rows[:args.train]
    held_rows = rows[args.train:args.train + args.held]
    eval_rows = held_rows[:args.eval_n]
    print(f"[ctr] train={len(train_rows)} held={len(held_rows)} eval_n={len(eval_rows)}")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "fp16"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    step = 0
    for ep in range(args.epochs):
        random.shuffle(train_rows)
        agg = {"ce": 0.0, "infonce": 0.0, "kl": 0.0, "gap": 0.0, "n": 0}
        for i in range(0, len(train_rows), args.bs):
            batch = train_rows[i:i + args.bs]
            opt.zero_grad()
            for row in batch:
                loss, logs = example_loss(model, tok, row, args, device)
                loss = loss / len(batch)
                scaler.scale(loss).backward()
                for kk in ("ce", "infonce", "kl", "gap"):
                    agg[kk] += logs[kk]
                agg["n"] += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            if step % 25 == 0:
                n = max(1, agg["n"])
                print(f"  ep{ep} step{step}  CE={agg['ce']/n:.3f}  InfoNCE={agg['infonce']/n:.3f}  "
                      f"KL={agg['kl']/n:.3f}  gap(CEneg-CEpos)={agg['gap']/n:+.3f}")
                agg = {"ce": 0.0, "infonce": 0.0, "kl": 0.0, "gap": 0.0, "n": 0}
        ev = eval_held(model, tok, eval_rows, args, device)
        print(f"[ctr] epoch {ep} HELD  acc_plan={ev['acc_plan']:.2%}  acc_noplan={ev['acc_noplan']:.2%}  "
              f"ablation_gap={ev['ablation_gap']:+.2%}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"base": args.base, "beta": args.beta, "tau": args.tau, "lam_kl": args.lam_kl,
               "k": args.k, "format": "Problem/Plan/FINAL ANSWER text prefix"},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"[ctr] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
