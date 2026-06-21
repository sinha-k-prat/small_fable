#!/usr/bin/env bash
# run_a100.sh — full small_fable pipeline on a single A100 80 GB, inside tmux.
#
# USAGE
#   tmux new -s fable
#   bash run_a100.sh          # full run from scratch
#   bash run_a100.sh --resume # resume any stage that was interrupted
#
# All stages write checkpoints and auto-resume; re-running with --resume is safe.
# Set HF_REPO to stream checkpoints off-box (optional but recommended for long runs).
#
# PREREQUISITES
#   pip install -r requirements.txt
#   export HF_TOKEN=hf_...   (write token, only needed when HF_REPO is set)
set -euo pipefail

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then RESUME="--resume"; fi

# ── tunables ─────────────────────────────────────────────────────────────────
BASE="Qwen/Qwen2.5-1.5B-Instruct"
TR=traces
DATA=dataset/traces_sft.jsonl
DATA_3K=dataset/traces_sft_3000.jsonl   # held-out generalization set (never trained)
SFT_CKPT=joint_ckpt
RL_CKPT=rl_ckpt
HF_REPO="${HF_REPO:-}"                  # e.g. "yourname/small_fable-planner"; leave empty to skip HF push

HF_ARGS=""
if [[ -n "$HF_REPO" ]]; then
  HF_ARGS="--hf_repo $HF_REPO --ckpt_every_min 15"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo " small_fable A100 run  $(date)"
echo " BASE  : $BASE"
echo " DATA  : $DATA"
echo " RESUME: ${RESUME:-none}"
echo "=========================================="

nvidia-smi | head -12

# ── Stage 0: build SFT corpus from traces ────────────────────────────────────
echo ""
echo "── Stage 0: traces → SFT corpus ──"

# Build plan_vocab.json from ALL three sets (so every token is covered even for eval).
python -u traces_to_sft.py \
  --traces  $TR/hard_reasoning_traces_1000.jsonl \
            $TR/hard_reasoning_traces_2000.jsonl \
            $TR/hard_reasoning_traces_3000.jsonl \
  --answers $TR/answers_1000.jsonl \
            $TR/answers_2000.jsonl \
            $TR/answers_3000.jsonl \
  --out /tmp/_all_vocab.jsonl --vocab_out plan_vocab.json
echo "[s0] plan_vocab.json written ($(python -c "import json; v=json.load(open('plan_vocab.json')); print(len(v['vocab']),'tokens')"))"

# Training corpus = sets 1000 + 2000.
python -u traces_to_sft.py \
  --traces  $TR/hard_reasoning_traces_1000.jsonl \
            $TR/hard_reasoning_traces_2000.jsonl \
  --answers $TR/answers_1000.jsonl \
            $TR/answers_2000.jsonl \
  --out $DATA --vocab_out /tmp/_v.json
echo "[s0] train corpus: $(wc -l < $DATA) rows → $DATA"

# Held-out generalization = set 3000 (flipped answers, never seen in training).
python -u traces_to_sft.py \
  --traces  $TR/hard_reasoning_traces_3000.jsonl \
  --answers $TR/answers_3000.jsonl \
  --out $DATA_3K --vocab_out /tmp/_v.json
echo "[s0] generalization corpus (set 3000): $(wc -l < $DATA_3K) rows → $DATA_3K"

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
echo ""
echo "── Stage 1: SFT (A100 config) ──"
# All flags from configs/sft_a100.yaml; explicit here so the script is self-contained.
python -u train_sft.py \
  --base "$BASE" \
  --data $DATA --train 1800 --held 200 \
  --dtype bfloat16 \
  --bs 16 \
  --lr 2e-5 --lr_min 8e-8 --warmup_frac 0.05 \
  --lam_resp 1.0 --lam_kl 0.5 \
  --max_resp 256 --plan_max_len 24 \
  --curriculum \
  --stage1_epochs 4 --stage2_epochs 2 --hard_err_rate 0.75 --hard_samples 8 \
  --probe 32 \
  --out $SFT_CKPT \
  --ckpt_every_min 15 \
  --metrics_out sft_metrics.jsonl \
  --device cuda \
  $RESUME $HF_ARGS
echo "[s1] SFT done → $SFT_CKPT"
echo "     Check sft_metrics.jsonl: ablation_gap and gap_content should be POSITIVE."

# ── Stage 2a: offline rollouts ────────────────────────────────────────────────
echo ""
echo "── Stage 2a: offline rollouts (A100 config) ──"
python -u rollout_offline.py \
  --base "$BASE" \
  --sft_ckpt $SFT_CKPT \
  --data $DATA --train 400 \
  --dtype bfloat16 \
  --group 16 --temp 1.3 --top_p 0.98 \
  --max_resp 256 \
  --out rl_rollouts.jsonl \
  --report pre_rl_filter_report.csv \
  --device cuda
echo "[s2a] rollouts done → rl_rollouts.jsonl"
echo "      Check pre_rl_filter_report.csv: zero_var fraction should be <30%."

# ── Stage 2b: GRPO ───────────────────────────────────────────────────────────
echo ""
echo "── Stage 2b: offline GRPO (A100 config) ──"
python -u train_grpo_offline.py \
  --base "$BASE" \
  --rollouts rl_rollouts.jsonl \
  --sft_ckpt $SFT_CKPT \
  --data $DATA \
  --out $RL_CKPT \
  --dtype bfloat16 \
  --inner_epochs 3 \
  --lr 1e-4 --clip_eps 0.2 \
  --beta_plan 1.0 --beta_ce 0.1 \
  --max_resp 256 \
  --kl_stop 0.12 \
  --held 32 \
  --maxent --gamma 2.0 \
  --ckpt_every_min 15 \
  --device cuda \
  $RESUME $HF_ARGS
echo "[s2b] GRPO done → $RL_CKPT"
echo "      Check: adapter |ΔL2|>0; held_reward moved; plan_entropy stable."

# ── Stage 3: evaluate ─────────────────────────────────────────────────────────
echo ""
echo "── Stage 3: compare SFT vs SFT+RL (three-way ablation) ──"
# --sample is required to see RL effects; greedy hides small changes.
# Reports gap_content (gold − random plan) and gap_presence (random − no plan).
python -u compare.py \
  --base "$BASE" \
  --sft_ckpt $SFT_CKPT \
  --rl_ckpt $RL_CKPT \
  --data $DATA --train 1800 --held 200 \
  --dtype bfloat16 \
  --max_resp 256 \
  --sample --temp 0.7 \
  --group 8 \
  --device cuda

echo ""
echo "── Stage 3b: generalization on held-out set 3000 (flipped answers) ──"
python - <<'PY'
import torch, json
from model_joint import JointModel
from train_sft import eval_held, load_rows

gen = load_rows('dataset/traces_sft_3000.jsonl')
print(f"[gen] set 3000 size: {len(gen)} rows")
for ckpt, label in [('joint_ckpt', 'SFT'), ('rl_ckpt', 'SFT + RL')]:
    m = JointModel.from_checkpoint(
        'Qwen/Qwen2.5-1.5B-Instruct', ckpt, device='cuda',
        dtype=torch.bfloat16, is_trainable=False)
    m.eval()
    r = eval_held(m, gen[:100], max_resp=256, sample=True, temp=0.7)
    print(f"[gen/{label}] {json.dumps({k: round(v, 3) for k, v in r.items()})}")
    del m; torch.cuda.empty_cache()
PY

echo ""
echo "=========================================="
echo " DONE  $(date)"
echo " Checkpoints: $SFT_CKPT/  $RL_CKPT/"
if [[ -n "$HF_REPO" ]]; then
  echo " HF mirror : https://huggingface.co/$HF_REPO"
fi
echo "=========================================="
