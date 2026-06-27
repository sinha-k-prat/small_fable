"""models.py — load & freeze Qwen2.5 0.5B / 1.5B from local cache.

torch + transformers ONLY. Default device cpu, dtype fp32. No network access.

Model facts (read from config in logic; table is for sizing):
    0.5B : hidden=896,  layers=24, intermediate=4864, vocab=151936, tie_word_embeddings=True
    1.5B : hidden=1536, layers=28, intermediate=8960, vocab=151936, tie_word_embeddings=True
HF paths: model.model.layers[i].self_attn / .mlp.{gate_proj,up_proj,down_proj}
          .input_layernorm / .post_attention_layernorm ; model.model.norm ; model.model.embed_tokens
down_proj.weight is [H, I] (no bias); its COLUMNS are writer directions in R^H.
gate_proj/up_proj.weight are [I, H]. Tied embeddings -> unembedding rows == embed_tokens.weight.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

MODEL_IDS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}


@dataclass
class ModelBundle:
    tag: str  # "0.5B" | "1.5B"
    model: nn.Module  # Qwen2ForCausalLM, eval, no-grad, fp32, cpu
    tokenizer: object  # PreTrainedTokenizerBase
    hidden: int  # config.hidden_size
    n_layers: int  # config.num_hidden_layers
    intermediate: int  # config.intermediate_size
    device: torch.device
    dtype: torch.dtype

    # ---- typed accessors to exact HF paths (never reach for a separate lm_head) ----
    def layer(self, i):
        return self.model.model.layers[i]

    def down_proj(self, i):
        return self.model.model.layers[i].mlp.down_proj  # weight [H, I]

    def gate_proj(self, i):
        return self.model.model.layers[i].mlp.gate_proj  # weight [I, H]

    def up_proj(self, i):
        return self.model.model.layers[i].mlp.up_proj  # weight [I, H]

    def input_ln(self, i):
        return self.model.model.layers[i].input_layernorm

    def post_attn_ln(self, i):
        return self.model.model.layers[i].post_attention_layernorm

    def final_norm(self):
        return self.model.model.norm

    def embed(self):
        return self.model.model.embed_tokens  # tied unembedding

    def down_proj_columns(self, i, idx):
        """[k, H] float32. down_proj.weight is [H, I]; writer column j = weight[:, j] in R^H.
        Returned TRANSPOSED so row r == writer idx[r]. ``idx`` is the top-K neuron indices."""
        W = self.down_proj(i).weight  # [H, I]
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        cols = W[:, idx_t].T  # [k, H]
        return cols.detach().to(torch.float32).cpu().numpy()

    def unembed_rows(self, token_ids):
        """[k, H] float32 — embed_tokens.weight[token_ids] (== unembedding rows, tied)."""
        ids = torch.as_tensor(list(token_ids), dtype=torch.long)
        rows = self.embed().weight[ids]  # [k, H]
        return rows.detach().to(torch.float32).cpu().numpy()


def load_model(tag, device="cpu", dtype=torch.float32):
    """Load a frozen Qwen2.5 bundle from the local HF cache (no download)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # defensive: no network
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import inspect

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODEL_IDS[tag]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    # dtype kwarg name is version-dependent: 4.44 only accepts `torch_dtype=`;
    # modern transformers (~4.56+) renames it to `dtype=` and deprecates `torch_dtype`.
    # Introspect the installed from_pretrained signature and pass whichever it accepts.
    # NOT "auto" (would give bf16 on some caches) — we force fp32 for exact reconstruction.
    _params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    _dtype_kw = "dtype" if "dtype" in _params else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=True,
        attn_implementation="eager",  # exact attn-write reconstruction on CPU
        **{_dtype_kw: dtype},
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    cfg = model.config
    bundle = ModelBundle(
        tag=tag,
        model=model,
        tokenizer=tok,
        hidden=cfg.hidden_size,
        n_layers=cfg.num_hidden_layers,
        intermediate=cfg.intermediate_size,
        device=torch.device(device),
        dtype=dtype,
    )

    # ---- cheap correctness guards (explicit shape asserts) ----
    assert cfg.tie_word_embeddings, "expected tied embeddings"
    assert cfg.hidden_act == "silu", "expected SiLU activation"
    dp = bundle.down_proj(0)
    assert dp.weight.shape == (bundle.hidden, bundle.intermediate), dp.weight.shape
    assert dp.bias is None, "down_proj must have no bias"
    assert bundle.gate_proj(0).weight.shape == (bundle.intermediate, bundle.hidden)
    assert bundle.up_proj(0).weight.shape == (bundle.intermediate, bundle.hidden)
    # tied unembedding shares storage with embed_tokens.
    # `cfg.tie_word_embeddings` (asserted above) already guarantees tying semantically.
    # The stricter data_ptr() identity check is fragile on modern transformers: with
    # tied weights `lm_head` may be re-tied lazily, materialized separately (meta/device_map
    # paths), or absent entirely (get_output_embeddings() returns None) -> spurious
    # AssertionError/AttributeError. So go through the accessor and only assert when a
    # distinct output-embedding weight is materialized; still catches real un-tying on 4.44.
    out_emb = bundle.model.get_output_embeddings()
    if out_emb is not None and getattr(out_emb, "weight", None) is not None:
        assert (
            out_emb.weight.data_ptr() == bundle.embed().weight.data_ptr()
        ), "lm_head should be tied to embed_tokens"
    return bundle
