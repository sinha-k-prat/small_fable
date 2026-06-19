# Running `small_fable` on a single SSHed A100

End-to-end recipe for the **interleaved (agentic) pipeline** + **planner-only RL with the KL-to-SFT
answer anchor** on one A100 (40 GB or 80 GB). This is the same pipeline as `train_all_colab.ipynb`,
but as plain shell commands you can run over SSH and leave attached in `tmux`.

> An A100 has far more headroom than the Colab T4. You can drop gradient checkpointing, raise batch
> size, and train on more rows — see **§7 A100 tuning**.

---

## 0 · Prerequisites

- An A100 with recent NVIDIA drivers + CUDA 12.x (`nvidia-smi` should print the GPU).
- Python 3.10+ and `git`.
- A Hugging Face account + a **write** token (https://huggingface.co/settings/tokens) — used to
  stream checkpoints off the box so a dropped SSH session never loses progress.

```bash
ssh user@your-a100-host
nvidia-smi                       # confirm the A100 is visible
```

## 1 · Clone

```bash
git clone https://github.com/sinha-k-prat/small_fable.git
cd small_fable
```

## 2 · Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt huggingface_hub
# peft can pull a torchao that clashes with this transformers; remove it if present:
pip uninstall -y torchao 2>/dev/null || true
```

## 3 · Hugging Face login (checkpoint streaming)

```bash
export HF_TOKEN=hf_xxx_your_write_token        # or: huggingface-cli login
huggingface-cli whoami                          # prints your username
export HF_REPO="$(huggingface-cli whoami | head -1)/small_fable-planner"
python -c "from huggingface_hub import create_repo; import os; \
  create_repo(os.environ['HF_REPO'], repo_type='model', exist_ok=True, private=False); \
  print('checkpoints ->', 'https://huggingface.co/'+os.environ['HF_REPO'])"
```

## 4 · Run inside tmux (survives SSH drops)

```bash
tmux new -s fable                # detach with Ctrl-b d ; reattach with: tmux attach -t fable
source .venv/bin/activate        # if a fresh shell
export HF_TOKEN=hf_xxx HF_REPO="$(huggingface-cli whoami | head -1)/small_fable-planner"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 5 · The pipeline

### Stage 0 — build the interleaved corpus (CPU, seconds)

`--interleaved` keeps per-turn `turns:[{plan,response}]` and writes `plan_vocab.json` with the
`BOP`/`FINALIZE_ALL` markers. Train = sets 1000+2000; held-out generalization = set 3000 (flipped).

```bash
TR=traces; DATA=dataset/traces_sft.jsonl

# vocab from ALL three sets (so every token, incl. eval-only ones, is covered):
python -u traces_to_sft.py --interleaved \
  --traces $TR/hard_reasoning_traces_1000.jsonl $TR/hard_reasoning_traces_2000.jsonl $TR/hard_reasoning_traces_3000.jsonl \
  --answers $TR/answers_1000.jsonl $TR/answers_2000.jsonl $TR/answers_3000.jsonl \
  --out /tmp/_all.jsonl --vocab_out plan_vocab.json

# train corpus = sets 1000 + 2000:
python -u traces_to_sft.py --interleaved \
  --traces $TR/hard_reasoning_traces_1000.jsonl $TR/hard_reasoning_traces_2000.jsonl \
  --answers $TR/answers_1000.jsonl $TR/answers_2000.jsonl \
  --out $DATA --vocab_out /tmp/_v.json

# held-out generalization = set 3000 (flipped answers, never trained):
python -u traces_to_sft.py --interleaved \
  --traces $TR/hard_reasoning_traces_3000.jsonl --answers $TR/answers_3000.jsonl \
  --out dataset/traces_sft_3000.jsonl --vocab_out /tmp/_v.json
```

### Stage 1 — interleaved SFT (checkpoints + resumes via HF)

Watch the held line: `ablation_gap` (drop plan) and `shuffle_gap` (random primitives) should go
**positive** — proof the executor genuinely needs the plan.

```bash
python -u train_sft.py --interleaved \
  --data $DATA --train 800 --held 60 \
  --epochs 4 --lr 2e-5 --bs 4 \
  --max_turns 6 --max_plan 12 --max_resp 256 --probe 0 \
  --out joint_ckpt --device cuda \
  --resume --hf_repo "$HF_REPO" --ckpt_every_min 10
# per-epoch metrics also append to sft_metrics.jsonl
```

### Stage 2a — interleaved rollouts (sampling, no training)

```bash
python -u rollout_offline.py --interleaved \
  --sft_ckpt joint_ckpt --data $DATA --train 120 \
  --group 8 --temp 1.2 --top_p 0.98 --max_resp 256 --max_turns 6 \
  --out rl_rollouts.jsonl --device cuda
```

### Stage 2b — planner-only GRPO + KL-to-SFT answer anchor

`--freeze_executor` freezes the LoRA and sets `lam_resp=0`; `--beta_ce 0` drops the plan CE so the
loss is **pure advantage×policy on the planner**. `--kl_resp 0.1` adds the
**KL(executor resp_new ‖ SFT resp_old)** anchor: because the executor is frozen, that KL is nonzero
only through the plan prefix, so it stops the planner from steering the frozen executor's answers
away from SFT. (Reference = the cached behavior logp, which **is** the SFT executor, since rollouts
were generated from `joint_ckpt`.)

```bash
python -u train_grpo_offline.py --interleaved --freeze_executor \
  --rollouts rl_rollouts.jsonl --sft_ckpt joint_ckpt --data $DATA \
  --out rl_ckpt --inner_epochs 3 --lr 1e-4 --clip_eps 0.2 \
  --beta_plan 1.0 --beta_ce 0 --kl_resp 0.1 \
  --held 16 --max_resp 256 --device cuda \
  --resume --hf_repo "$HF_REPO" --ckpt_every_min 10
# log line shows: plan_approx_kl=... kl_resp(to-SFT)=... held_reward=...
```

> Tune `--kl_resp`: raise it (e.g. `0.3`) if RL answers drift into degenerate text; lower it
> (e.g. `0.03`) if the anchor is too tight and reward won't move.

### Stage 3 — evaluate

```bash
# generalization on the unseen, flipped set 3000:
python - <<'PY'
import torch
from model_joint import JointModel
from train_sft import eval_held_interleaved, load_rows
gen = load_rows('dataset/traces_sft_3000.jsonl')[:60]
for ckpt,label in [('joint_ckpt','SFT'),('rl_ckpt','SFT + planner-only RL')]:
    m = JointModel.from_checkpoint('Qwen/Qwen2.5-1.5B-Instruct', ckpt, device='cuda'); m.eval()
    r = eval_held_interleaved(m, gen, max_turns=6, max_plan=12, max_resp=256, sample=True)
    print(f'[{label}] SET 3000 (unseen, flipped):', {k:round(v,3) for k,v in r.items()})
    del m; torch.cuda.empty_cache()
PY

# transparent per-turn head-to-head on one hard prompt:
python - <<'PY'
import torch
from model_joint import JointModel
P=('A snail is at the bottom of a 12-meter well. Each day it climbs 4 meters, '
   'but each night it slides back 3 meters. On which day does it first reach the top?')
for ckpt,label in [('joint_ckpt','SFT only'),('rl_ckpt','SFT + planner-only GRPO')]:
    m = JointModel.from_checkpoint('Qwen/Qwen2.5-1.5B-Instruct', ckpt, device='cuda'); m.eval()
    print('\n'+'='*72+f'\n{label}\n'+'='*72)
    pi,pa = m.batch_prompts([P])
    rec = m.run_interleaved(pi[0], pa[0], sample=True, temp=0.7,
                            max_turns=6, max_plan=12, max_resp=256, verbose=True)
    print('FINAL:', m.interleaved_answer_text(rec)[:400])
    del m; torch.cuda.empty_cache()
PY
```

---

## 6 · Resume after a crash / disconnect

Every stage with `--resume` reloads `train_state.pt` from `--out` (pulled from HF if local is gone)
and continues from the next inner step. To recover a box that was wiped:

```bash
python -c "from huggingface_hub import snapshot_download; import os; \
  snapshot_download(repo_id=os.environ['HF_REPO'], allow_patterns=['joint_ckpt/*','rl_ckpt/*'], local_dir='.')"
```

Then re-run the **same** Stage 1 / 2b command with `--resume` — it no-ops the finished work and picks
up where it stopped. (The resume guard refuses a checkpoint whose `data`/`rollouts`/`n_plan` differ,
so changing the vocab or corpus correctly starts fresh.)

## 7 · A100 tuning (vs the Colab T4 defaults)

The commands above are T4-safe. On an A100 you have room to go faster/bigger:

| Knob | T4 default | A100 80 GB | A100 40 GB |
|---|---|---|---|
| SFT `--bs` | 2 | 8 | 4 |
| `--train` (SFT) | 800 | 1500 (≈ corpus − held) | 1200 |
| rollout `--train` | 120 | 300+ | 200 |
| rollout `--group` | 8 | 16 | 8–12 |
| gradient checkpointing | needed | optional | optional |

- **bf16 base**: the base Qwen already loads in bf16; only the planner head + plan_emb are fp32 (kept
  fp32 on purpose for stable logits). No change needed.
- **Gradient checkpointing** is auto-enabled for memory; on an 80 GB A100 you can comment it out for a
  speed-up (it's the `gradient_checkpointing_enable()` call in `train_grpo_offline.py` / `train_sft.py`).
- Run `watch -n5 nvidia-smi` in a second tmux pane to size batch up until ~85% memory.

## 8 · One-shot driver (optional)

To chain everything unattended, drop this into `run_a100.sh` and `bash run_a100.sh` inside tmux —
each stage `--resume`s, so re-running after any failure is safe:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_REPO:?set HF_REPO}"; : "${HF_TOKEN:?set HF_TOKEN}"
TR=traces; DATA=dataset/traces_sft.jsonl

python -u traces_to_sft.py --interleaved --traces $TR/hard_reasoning_traces_1000.jsonl $TR/hard_reasoning_traces_2000.jsonl $TR/hard_reasoning_traces_3000.jsonl --answers $TR/answers_1000.jsonl $TR/answers_2000.jsonl $TR/answers_3000.jsonl --out /tmp/_all.jsonl --vocab_out plan_vocab.json
python -u traces_to_sft.py --interleaved --traces $TR/hard_reasoning_traces_1000.jsonl $TR/hard_reasoning_traces_2000.jsonl --answers $TR/answers_1000.jsonl $TR/answers_2000.jsonl --out $DATA --vocab_out /tmp/_v.json
python -u traces_to_sft.py --interleaved --traces $TR/hard_reasoning_traces_3000.jsonl --answers $TR/answers_3000.jsonl --out dataset/traces_sft_3000.jsonl --vocab_out /tmp/_v.json

python -u train_sft.py --interleaved --data $DATA --train 800 --held 60 --epochs 4 --lr 2e-5 --bs 4 --max_turns 6 --max_plan 12 --max_resp 256 --probe 0 --out joint_ckpt --device cuda --resume --hf_repo "$HF_REPO" --ckpt_every_min 10
python -u rollout_offline.py --interleaved --sft_ckpt joint_ckpt --data $DATA --train 120 --group 8 --temp 1.2 --top_p 0.98 --max_resp 256 --max_turns 6 --out rl_rollouts.jsonl --device cuda
python -u train_grpo_offline.py --interleaved --freeze_executor --rollouts rl_rollouts.jsonl --sft_ckpt joint_ckpt --data $DATA --out rl_ckpt --inner_epochs 3 --lr 1e-4 --clip_eps 0.2 --beta_plan 1.0 --beta_ce 0 --kl_resp 0.1 --held 16 --max_resp 256 --device cuda --resume --hf_repo "$HF_REPO" --ckpt_every_min 10
echo "done -> https://huggingface.co/$HF_REPO/tree/main"
```
