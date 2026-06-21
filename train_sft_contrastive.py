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
import argparse, copy, json, math, os, random, time
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
    p = p[:max(0, max_len - len(t))]          # truncate the PROMPT so the answer+EOS always survive
    ids = p + t                               # (else a long prompt could zero a row's supervised CE)
    labels = [-100] * len(p) + t
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
def example_loss(model, tok, row, args, device, ref_model=None):
    """CE(gold) + beta*InfoNCE(gold vs K negs) + lam_kl*KL(exec||base). One example.
    ref_model (if given) is a SEPARATE frozen original-Qwen used as the KL reference; else the KL
    reference is the LoRA model with adapters disabled. Returns (loss, logdict)."""
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

        # (3) KL(executor || base) on the GOLD answer tokens ONLY. Reference = ORIGINAL frozen Qwen.
        kl = torch.zeros((), device=device)
        if args.lam_kl > 0:
            g_ids, g_lab, g_attn = input_ids[:1], labels[:1], attn[:1]
            if ref_model is not None:
                # separate frozen base model: unambiguously the original Qwen (no LoRA at all)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.amp_dtype,
                                                     enabled=args.amp):
                    base_logits = ref_model(input_ids=g_ids, attention_mask=g_attn).logits
            else:
                # reuse the LoRA model with adapters disabled. On Qwen2.5 (attention_dropout=0) this
                # is numerically identical to original Qwen — the disabled forward bypasses lora_*.
                with torch.no_grad(), model.disable_adapter():
                    base_logits = model(input_ids=g_ids, attention_mask=g_attn).logits
            # compute the KL in fp32: under fp16 autocast the raw logits round badly in p*(lp-lq)
            lp = F.log_softmax(logits[:1][:, :-1, :].float(), -1)   # executor (p)
            lq = F.log_softmax(base_logits[:, :-1, :].float(), -1)  # original Qwen (q)
            kl_tok = (lp.exp() * (lp - lq)).sum(-1)                 # (1, S-1)  KL(exec||base) per pos
            m = (g_lab[:, 1:] != -100).float()                     # GOLD answer tokens only
            kl = (kl_tok * m).sum() / m.sum().clamp_min(1.0)

    loss = ce_pos + args.beta * infonce + args.lam_kl * kl
    gap = float(ce_neg.mean() - ce_pos)                       # MI estimate / training-time ablation gap
    return loss, {"ce": float(ce_pos), "infonce": float(infonce), "kl": float(kl),
                  "ce_neg": float(ce_neg.mean()), "gap": gap}


# --------------------------------------------------------------------------- decode eval (is the plan CAUSAL?)
@torch.no_grad()
def eval_held(model, tok, rows, args, device):
    """Causal eval of whether the plan is LOAD-BEARING, robust to the confound that the answer is
    computable from the instruction alone. For each held row, greedy-decode THREE ways and grade
    with checkers.reward_for_row:
      (i)  gold plan prefix   -> A acc_plan    (decode == gold answer)
      (ii) EACH hard-neg plan -> B neg_follow  (decode == THAT neg's answer: executed the wrong plan)
                              -> C neg_to_gold (decode == gold answer: ignored the plan)
      (iii) no plan           -> D acc_noplan  (decode == gold answer)
    DECISION: planning is causally load-bearing iff B is HIGH and C is LOW. A high + ablation_gap>0
    is only SUPPORTING evidence — if the answer is derivable from instr, A can be high with the plan
    ignored. The neg-plan decode is what isolates plan-causation from instruction-solving."""
    from checkers import reward_for_row
    model.eval()

    def decode(prompt):
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        with torch.autocast(device_type="cuda", dtype=args.amp_dtype, enabled=args.amp):
            out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)

    A = B = C = D = 0.0
    n = max(1, len(rows))
    for r in rows:
        A += reward_for_row(r, f"FINAL ANSWER: {decode(prompt_with_plan(r['instruction'], r['plan_str']))}")
        negs = r.get("hard_negatives", [])[:max(1, args.k)]
        if negs:
            b_sum = c_sum = 0.0
            for neg in negs:
                n_text = decode(prompt_with_plan(r["instruction"], neg["plan_str"]))
                neg_row = copy.deepcopy(r)                 # grade decode against the NEG's answer
                try:
                    neg_row["checker_args"]["canonical"]["value"] = float(neg["answer"])
                except (KeyError, TypeError, ValueError):
                    neg_row["checker_kind"] = "numeric"
                    neg_row["checker_args"] = {"canonical": {"value": float(neg["answer"])},
                                               "match": {"tolerance": 0.01}}
                b_sum += reward_for_row(neg_row, f"FINAL ANSWER: {n_text}")     # followed wrong plan
                c_sum += reward_for_row(r, f"FINAL ANSWER: {n_text}")           # ignored the plan
            B += b_sum / len(negs); C += c_sum / len(negs)
        D += reward_for_row(r, f"FINAL ANSWER: {decode(prompt_no_plan(r['instruction']))}")

    model.train()
    A, B, C, D = A / n, B / n, C / n, D / n
    return {"acc_plan": A, "neg_follow": B, "neg_to_gold": C, "acc_noplan": D, "ablation_gap": A - D}


# --------------------------------------------------------------------------- val error + plotting
@torch.no_grad()
def val_answer_ce(model, tok, rows, args, device):
    """Mean teacher-forced answer cross-entropy under the GOLD plan over `rows` — the val-error
    scalar plotted against the training CE curve. Builds the SAME sequence training uses for the
    gold variant (variants[0] in example_loss), so train CE and val CE are measured identically."""
    if not rows:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(rows), max(1, args.bs)):
        chunk = rows[i:i + max(1, args.bs)]
        seqs = [build_seq(tok, prompt_with_plan(r["instruction"], r["plan_str"]),
                          target_text(r["answer"]), args.max_len) for r in chunk]
        input_ids, labels, attn = collate(seqs, tok.pad_token_id, device)
        with torch.autocast(device_type="cuda", dtype=args.amp_dtype, enabled=args.amp):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        ce = per_row_ce(logits, labels)
        total += float(ce.sum()); count += ce.numel()
    model.train()
    return total / max(1, count)


def plot_curves(history, out_png):
    """Headless train-vs-val error curves. (A) train CE vs step + val CE markers; (B) causal metrics
    vs epoch. Always dumps history.json; matplotlib import is GUARDED so a missing install only
    warns (training never crashes)."""
    hist_path = os.path.join(os.path.dirname(out_png) or ".", "history.json")
    try:
        json.dump(history, open(hist_path, "w"), indent=2)
    except Exception as e:
        print(f"[plot] WARNING: could not write {hist_path}: {e}")
    try:
        import matplotlib
        matplotlib.use("Agg")                 # headless backend — set BEFORE importing pyplot
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] WARNING: matplotlib unavailable ({e}); wrote {hist_path}, skipping PNG.")
        return
    ts, ve = history.get("train_steps", []), history.get("val_epochs", [])
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)
    if ts:
        axA.plot([d["step"] for d in ts], [d["ce"] for d in ts], "-o", ms=3, lw=1.5,
                 color="tab:blue", label="train CE (gold plan)")
    if ve:
        axA.plot([d["step"] for d in ve], [d["val_ce"] for d in ve], "s--", ms=9, lw=1.0,
                 color="tab:red", alpha=0.8, label="val CE (gold plan, teacher-forced)")
    axA.set_xlabel("optimizer step"); axA.set_ylabel("cross-entropy (answer tokens)")
    axA.set_title("(A) ERROR — train vs validation answer-CE under the GOLD plan")
    axA.grid(True, alpha=0.3); axA.legend(loc="best")
    if ve:
        ep = [d["epoch"] for d in ve]
        for key, lab, c in [("acc_plan", "acc_plan (gold-plan correct)", "tab:green"),
                            ("neg_follow", "neg_follow (HIGH = plan causal)", "tab:orange"),
                            ("neg_to_gold", "neg_to_gold (LOW = plan causal)", "tab:purple"),
                            ("acc_noplan", "acc_noplan (no plan)", "tab:gray")]:
            axB.plot(ep, [d[key] for d in ve], "-o", ms=4, color=c, label=lab)
        axB.set_ylim(-0.02, 1.02); axB.set_xlabel("epoch"); axB.set_ylabel("rate")
        axB.set_title("(B) CAUSAL — is the plan load-bearing? (neg_follow HIGH & neg_to_gold LOW)")
        axB.grid(True, alpha=0.3); axB.legend(loc="best", fontsize=8)
    else:
        axB.text(0.5, 0.5, "no val_epochs recorded", ha="center", va="center")
    try:
        fig.savefig(out_png, dpi=130)
        print(f"[plot] wrote {out_png} and {hist_path}")
    except Exception as e:
        print(f"[plot] WARNING: could not save {out_png}: {e}")
    finally:
        plt.close(fig)


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
    ap.add_argument("--kl_ref", default="disable_adapter", choices=["disable_adapter", "frozen"],
                    help="KL reference: 'disable_adapter' reuses the LoRA model with adapters off "
                         "(numerically original Qwen on Qwen2.5; 0 extra memory). 'frozen' loads a "
                         "SEPARATE untouched base model (unambiguous, +~3.1GB).")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--val", default="dataset/sft_synth_v4_val.jsonl",
                    help="separate holdout VALIDATION file, disjoint from --data. When it exists, "
                         "ALL its rows are used for eval_held AND the val_ce error curve; when "
                         "absent, fall back to the internal held split (capped at --eval_n).")
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

    # KL reference = the ORIGINAL frozen Qwen. 'frozen' = a separate untouched copy (provably
    # independent); only ever run under no_grad, so no optimizer state / no retained activations.
    ref_model = None
    if args.lam_kl > 0 and args.kl_ref == "frozen":
        print(f"[ctr] loading SEPARATE frozen reference {args.base} for KL ...")
        ref_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=load_dtype).to(device)
        ref_model.eval()
        ref_model.config.use_cache = False
        for p in ref_model.parameters():
            p.requires_grad_(False)

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    train_rows = rows[:args.train]
    held_rows = rows[args.train:args.train + args.held]

    # validation set: prefer the SEPARATE --val holdout file (disjoint from --data); ALL its rows
    # feed BOTH eval_held (decode/causal metrics) and the val_ce error curve. Else fall back to the
    # internal held split (capped at --eval_n).
    if os.path.exists(args.val):
        val_rows = [json.loads(l) for l in open(args.val)]
        eval_rows = val_rows
        print(f"[ctr] train={len(train_rows)} held={len(held_rows)} "
              f"VAL(file={args.val})={len(val_rows)} -> eval on ALL {len(val_rows)} val rows")
    else:
        val_rows = held_rows[:args.eval_n]
        eval_rows = val_rows
        print(f"[ctr] train={len(train_rows)} held={len(held_rows)} "
              f"VAL(file MISSING -> internal held split)={len(val_rows)}")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "fp16"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    history = {"train_steps": [], "val_epochs": []}      # train CE per step + val metrics per epoch
    step = 0
    for ep in range(args.epochs):
        random.shuffle(train_rows)
        agg = {"ce": 0.0, "infonce": 0.0, "kl": 0.0, "gap": 0.0, "n": 0}
        for i in range(0, len(train_rows), args.bs):
            batch = train_rows[i:i + args.bs]
            opt.zero_grad()
            for row in batch:
                loss, logs = example_loss(model, tok, row, args, device, ref_model)
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
                ce_w, inf_w, kl_w, gap_w = agg["ce"]/n, agg["infonce"]/n, agg["kl"]/n, agg["gap"]/n
                print(f"  ep{ep} step{step}  CE={ce_w:.3f}  InfoNCE={inf_w:.3f}  "
                      f"KL={kl_w:.3f}  gap(CEneg-CEpos)={gap_w:+.3f}")
                history["train_steps"].append({"step": step, "epoch": ep, "ce": ce_w,
                                               "infonce": inf_w, "kl": kl_w, "gap": gap_w})
                agg = {"ce": 0.0, "infonce": 0.0, "kl": 0.0, "gap": 0.0, "n": 0}
        ev = eval_held(model, tok, eval_rows, args, device)
        print(f"[ctr] epoch {ep} HELD  acc_plan={ev['acc_plan']:.2%}  neg_follow={ev['neg_follow']:.2%}  "
              f"neg_to_gold={ev['neg_to_gold']:.2%}  acc_noplan={ev['acc_noplan']:.2%}  "
              f"ablation_gap={ev['ablation_gap']:+.2%}   "
              f"[plan causal iff neg_follow HIGH & neg_to_gold LOW]")
        val_ce = val_answer_ce(model, tok, val_rows, args, device)
        print(f"[ctr] epoch {ep} VAL   val_ce(gold-plan, teacher-forced)={val_ce:.4f}")
        history["val_epochs"].append({"epoch": ep, "step": step, "val_ce": val_ce,
                                      "acc_plan": ev["acc_plan"], "neg_follow": ev["neg_follow"],
                                      "neg_to_gold": ev["neg_to_gold"], "acc_noplan": ev["acc_noplan"],
                                      "ablation_gap": ev["ablation_gap"]})

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"base": args.base, "beta": args.beta, "tau": args.tau, "lam_kl": args.lam_kl,
               "kl_ref": args.kl_ref, "k": args.k, "format": "Problem/Plan/FINAL ANSWER text prefix"},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"[ctr] saved adapter -> {args.out}")
    plot_curves(history, os.path.join(args.out, "curves.png"))


if __name__ == "__main__":
    main()
