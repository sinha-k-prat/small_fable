#!/usr/bin/env python3
"""
checkpointing.py — mid-training checkpoint + resume, with optional Hugging Face push.

Both train_sft.py (single-stage) and train_grpo_offline.py use a Checkpointer to save the model
PLUS a train_state.pt (optimizer / scheduler / position / RNG) every --ckpt_every_min minutes (and
at every epoch / inner-epoch boundary). With --hf_repo set, each checkpoint is also uploaded
(overwriting the same path) so a run can RESUME after a Colab restart by re-downloading the dir.

Resume granularity is MID-EPOCH:
  - SFT  records (epoch, batch_idx)        and skips already-done batches on resume.
  - GRPO records (inner_epoch, group_idx)  and skips already-done groups on resume.

A completed run leaves a train_state whose position is past the end, so re-running with --resume is
a safe no-op (the loop range is empty).
"""
import os, time
import torch

STATE_FILE = "train_state.pt"


def save_train_state(out_dir, state):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, STATE_FILE))


def load_train_state(ckpt_dir):
    """Return the saved train_state dict, or None if absent."""
    if not ckpt_dir:
        return None
    p = os.path.join(ckpt_dir, STATE_FILE)
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def restore_optimizer(opt, state, device):
    """Load optimizer state and move its tensors onto `device` (state is saved on CPU)."""
    if state and state.get("optimizer"):
        opt.load_state_dict(state["optimizer"])
        for s in opt.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)


def restore_rng(state):
    if not state:
        return
    try:
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("py_rng") is not None:
            import random
            random.setstate(state["py_rng"])
    except Exception as e:
        print("[ckpt] RNG restore skipped:", e)


def scalar_args(args):
    """Pickle-light snapshot of CLI args (scalars only) for resume sanity checks."""
    return {k: v for k, v in vars(args).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}


class Checkpointer:
    """Saves model + train_state to out_dir; optionally uploads the dir to a HF model repo."""

    def __init__(self, out_dir, base, every_min=0.0, every_steps=0,
                 hf_repo=None, hf_token=None, path_in_repo=None):
        self.out_dir = out_dir
        self.base = base
        self.every_min = every_min
        self.every_steps = every_steps
        self.hf_repo = hf_repo
        self.path_in_repo = path_in_repo or os.path.basename(os.path.normpath(out_dir))
        self.api = None
        self._last = time.time()
        if hf_repo:
            tok = hf_token or os.environ.get("HF_TOKEN")
            try:
                from huggingface_hub import HfApi, create_repo
                create_repo(hf_repo, repo_type="model", exist_ok=True, private=False, token=tok)
                self.api = HfApi(token=tok)
                print(f"[ckpt] HF push enabled -> {hf_repo}/{self.path_in_repo}")
            except Exception as e:
                print(f"[ckpt] HF push disabled ({e})")

    def due(self, step):
        """True if a periodic checkpoint is due (by step count or wall-clock)."""
        if self.every_steps and step > 0 and step % self.every_steps == 0:
            return True
        if self.every_min and (time.time() - self._last) >= self.every_min * 60:
            return True
        return False

    def save(self, model, state, reason="periodic", push=True):
        model.save(self.out_dir, self.base)
        save_train_state(self.out_dir, state)
        self._last = time.time()
        pos = {k: state[k] for k in ("epoch", "batch_idx", "inner_epoch", "group_idx", "global_step")
               if k in state}
        print(f"[ckpt] saved {self.out_dir} ({reason}) {pos}", flush=True)
        if push and self.api is not None:
            try:
                self.api.upload_folder(folder_path=self.out_dir, path_in_repo=self.path_in_repo,
                                       repo_id=self.hf_repo,
                                       commit_message=f"{reason}: {self.path_in_repo}")
                print(f"[ckpt] pushed -> "
                      f"https://huggingface.co/{self.hf_repo}/tree/main/{self.path_in_repo}", flush=True)
            except Exception as e:
                print(f"[ckpt] HF push failed (continuing): {e}", flush=True)
