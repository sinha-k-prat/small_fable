# `residual_orrery` — Implementation-Ready Design Doc

A novel mechanistic-interpretability visualization, packaged as a clean importable +
CLI-driven Python package. The residual stream is rendered as an **orrery**: down_proj
"writer-direction" stars fixed on the unit sphere S², and a marker that **hops** through
the residual trajectory, lighting up each layer's stars as that layer fires. 0.5B vs 1.5B
are rendered **side by side** as twin spheres advancing on a shared frame clock.

> Primary approach = DESIGN 3 (clean package + disk cache + clean module boundaries:
> `project`/`animate` are torch-free and unit-testable on cached `.npz`). Grafted in:
> DESIGN 1's mechanistic self-test asserts and additive residual reconstruction with the
> independent `layer-output` cross-check; DESIGN 2's glow-envelope/comet-tail/HUD visual
> mechanics, `Agg` backend discipline, and explicit mpl-3.2 / imageio-2.9 pitfall list.
> Every conflict between the three is resolved below and called out inline.

All facts in this doc were **verified live** against the local cache on the exact pinned
stack (see §0). The doc is detailed enough to build the package without guessing.

---

## 0. Verified ground truth (checked live on this machine)

Installed stack (confirmed): `torch 2.2.2, transformers 4.44.2, numpy 1.20.1,
matplotlib 3.2.2, scikit-learn 0.23.1, Pillow 7.2.0, imageio 2.9.0, Python 3.8.3`.

Verified by running hooks on `Qwen/Qwen2.5-0.5B-Instruct` from cache:

| fact | result |
|---|---|
| `down_proj.weight.shape` | `(H, I) = (896, 4864)`, `bias is None` |
| `gate_proj/up_proj.weight.shape` | `(I, H)` |
| MLP write identity `W_down @ a == mlp_out` | `‖·‖∞ = 4.8e-7` ✔ |
| `self_attn` forward output | **is a tuple** → use `out[0]` |
| `mlp` forward output | **plain tensor** (not a tuple) |
| residual recon `lin + attn_out`, then `+ mlp_out` vs `layer.output[0]` | `‖·‖∞ = 0.0` ✔ |
| tied unembed: `norm_row @ embed.weight.T` argmax | **matches** `model.logits[...].argmax()`; predicts `'4'` ✔ |
| `a` shape (down_proj pre-hook input) | `(4864,)` = `(I,)` ✔ |
| forward-hook signature | `(module, input, output)` — **3 positional args** |

**CRITICAL ENVIRONMENT FINDING (resolves a conflict in all three designs):**
`tokenizer.apply_chat_template(...)` **raises** `ImportError: apply_chat_template requires
jinja2>=3.1.0 ... Your version is 2.11.2`. **All three source designs called
`apply_chat_template` and would crash.** This spec therefore builds the Qwen2.5 ChatML
prompt with a **manual string template** (see §3). Do **not** call `apply_chat_template`.

Model structural facts (read from `config`, never hardcoded in logic; table is for sizing):

| | hidden `H` | layers `N` | intermediate `I` | vocab `V` | trajectory nodes `P=2N+3` |
|---|---|---|---|---|---|
| 0.5B | 896 | 24 | 4864 | 151936 | 51 |
| 1.5B | 1536 | 28 | 8960 | 151936 | 59 |

`tie_word_embeddings=True` for both (`lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`),
`hidden_act="silu"`. Trajectory node count `P = 2N+3`: `embed` + `2N` sub-writes +
`final-norm` + `unembed-target`.

Verified Qwen2DecoderLayer.forward (transformers 4.44.2): both residual adds happen
**inside** the layer's `forward`, so no single layer-output hook yields the post-attn
state → trajectory must be reconstructed from sub-module hooks (§4).

---

## 1. Package layout & module dependency graph

```
residual_orrery/
  __init__.py     # version + tiny public re-export surface
  examples.py     # the 5 prompts + manual ChatML builder (NO apply_chat_template)
  models.py       # load/freeze 0.5B & 1.5B from cache; typed accessors; weight-derived stars
  collect.py      # forward hooks -> trajectory + per-layer `a` + writer cols + unembed rows; .npz cache
  project.py      # joint PCA(3) over the union, then L2 sphere-normalize  (numpy+sklearn only)
  animate.py      # frame schedule, slerp, glow, side-by-side render -> GIF  (numpy+mpl+imageio only)
  compare.py      # two-model driver: collect (cached) -> 2 frames -> project -> render
  cli.py          # argparse entry; also re-exported as console_script
  __main__.py     # `python -m residual_orrery` -> cli.main()
out/              # GIFs; out/cache/*.npz+*.json; out/frames_debug/*.png (gitignored)
tests/test_residual_orrery.py
```

```
examples ─┐
models  ──┼─► collect ──► project ──► animate
          │                  ▲           ▲
          └───────────────► compare ─────┘ ──► cli ──► __main__
```

Dependency rules (enforced, this is the key ergonomic win):
- `models`: torch + transformers only.
- `collect`: torch + numpy; depends on `models`, `examples`.
- `project`: **numpy + sklearn only — NO torch.**
- `animate`: **numpy + matplotlib + imageio only — NO torch.**
- `compare`: orchestrates; no new heavy deps. `cli`: argparse only.

So `project`/`animate` iterate fast on cached `.npz` without loading a model.

---

## 2. `models.py`

```python
import os, torch, numpy as np
from dataclasses import dataclass
from torch import nn

MODEL_IDS = {"0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
             "1.5B": "Qwen/Qwen2.5-1.5B-Instruct"}

@dataclass
class ModelBundle:
    tag: str                       # "0.5B" | "1.5B"
    model: nn.Module               # Qwen2ForCausalLM, eval, no-grad, fp32, cpu
    tokenizer: "PreTrainedTokenizerBase"
    hidden: int                    # config.hidden_size
    n_layers: int                  # config.num_hidden_layers
    intermediate: int              # config.intermediate_size
    device: torch.device
    dtype: torch.dtype

    # typed accessors to exact HF paths (NEVER reach for a separate lm_head)
    def layer(self, i):        return self.model.model.layers[i]
    def down_proj(self, i):    return self.model.model.layers[i].mlp.down_proj   # weight [H,I]
    def gate_proj(self, i):    return self.model.model.layers[i].mlp.gate_proj   # weight [I,H]
    def up_proj(self, i):      return self.model.model.layers[i].mlp.up_proj     # weight [I,H]
    def post_attn_ln(self, i): return self.model.model.layers[i].post_attention_layernorm
    def final_norm(self):      return self.model.model.norm
    def embed(self):           return self.model.model.embed_tokens              # tied unembedding

    def down_proj_columns(self, i: int, idx: np.ndarray) -> np.ndarray:
        """[k, H] float32. down_proj.weight is [H, I]; writer column j = weight[:, j] in R^H.
        Returned TRANSPOSED so row r == writer idx[r]. `idx` is the top-K neuron indices."""
        W = self.down_proj(i).weight                      # [H, I]
        return W[:, torch.as_tensor(idx)].T.detach().to(torch.float32).cpu().numpy()

    def unembed_rows(self, token_ids) -> np.ndarray:
        """[k, H] float32 — embed_tokens.weight[token_ids] (== unembedding rows, tied)."""
        ids = torch.as_tensor(list(token_ids))
        return self.embed().weight[ids].detach().to(torch.float32).cpu().numpy()


def load_model(tag: str, *, device="cpu", dtype=torch.float32) -> ModelBundle:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # defensive: no network
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODEL_IDS[tag]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, local_files_only=True,
        attn_implementation="eager",                      # avoid SDPA path differences on CPU
    ).to(device)
    model.eval(); model.requires_grad_(False)
    cfg = model.config
    b = ModelBundle(tag, model, tok, cfg.hidden_size, cfg.num_hidden_layers,
                    cfg.intermediate_size, torch.device(device), dtype)
    # cheap correctness guards
    assert cfg.tie_word_embeddings and cfg.hidden_act == "silu"
    assert b.down_proj(0).weight.shape == (b.hidden, b.intermediate)
    assert b.down_proj(0).bias is None
    return b
```

Resolved conflicts / pitfalls:
- `torch_dtype=torch.float32` (NOT `"auto"` → would give bf16 on some caches and break the
  fp32 asserts). transformers 4.44.2 uses `torch_dtype=`, not the newer `dtype=` alias.
- `local_files_only=True` + `HF_HUB_OFFLINE=1` (both models are cached; never download).
- `attn_implementation="eager"` (explicit) so attn-write reconstruction is exact on CPU.
- Tied embeddings → always read `embed_tokens.weight`; there is no separate unembedding.

---

## 3. `examples.py`  (MANUAL ChatML — apply_chat_template is broken here)

```python
EXAMPLES = [
    "What is 17 plus 25? Reply with just the number.",
    "Give one word that means the opposite of 'hot'.",
    "Complete the sentence: The capital of France is",
    "Is 7 a prime number? Answer yes or no.",
    "Reverse the letters of the word 'cat'.",
]

_CHATML = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

def build_input_ids(bundle, prompt: str) -> "torch.LongTensor":
    """Qwen2.5 ChatML prompt, built as a plain string (jinja2 is 2.11.2 here, so
    tokenizer.apply_chat_template RAISES ImportError — verified). Returns ids [1, T]."""
    text = _CHATML.format(prompt=prompt)
    ids = bundle.tokenizer(text, return_tensors="pt").input_ids
    return ids.to(bundle.device)
```

The traced position is `pos = ids.shape[1] - 1` (the final `'\n'` after `assistant`, i.e.
the token whose residual produces the first generated logit; verified `argmax → '4'` for
the arithmetic prompt). One forward pass, `use_cache=False`, **no generation loop**.

For the GIF we animate **one** chosen prompt (`--example`, default 0). The other 4 prompts'
last-token trajectories optionally enrich the PCA fit only (§5.1, `--enrich-pca`, default on).

---

## 4. `collect.py` — hooks, exact shapes, reconstruction, self-tests, cache

### 4.1 Data structures

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

class NodeKind(str, Enum):
    EMBED="embed"; ATTN="attn"; MLP="mlp"; FINAL="final"; UNEMBED="unembed"

@dataclass
class TrajNode:
    kind: NodeKind
    layer: int                       # -1 for embed/final/unembed
    h: np.ndarray                    # [H] float32 — cumulative residual point (or unembed dir)
    a: Optional[np.ndarray] = None   # [I] float32, only for kind==MLP (down_proj input)

@dataclass
class RunCollection:
    tag: str
    instruction: str
    input_ids: np.ndarray            # [T] int64
    last_pos: int
    pred_token_id: int
    pred_token_str: str
    nodes: list                      # ordered TrajNode: embed,(attn0,mlp0)...(attnN-1,mlpN-1),final,unembed
    topk_idx: dict                   # layer -> [K] int   (top-K |a_j| writers for THIS token)
    topk_a:   dict                   # layer -> [K] float32 (|a_j| values, for glow size)
    down_cols: dict                  # layer -> [K, H] float32 (writer dirs == down_proj cols, transposed)
    unembed_dir: np.ndarray          # [H] float32 (unembedding row of pred token)
    H: int; N: int; I: int; topk: int
```

`nodes` length is exactly `P = 2N+3`. Node→(kind,layer) map is deterministic:
node 0 `(EMBED,-1)`; node `2L+1` `(ATTN,L)`; node `2L+2` `(MLP,L)`; node `2N+1` `(FINAL,-1)`;
node `2N+2` `(UNEMBED, pred_token_id)`. Only `MLP` nodes carry `a`.

### 4.2 Hook points (all sampled at column `pos`, `.detach().float().cpu().numpy()`)

Run once under `torch.no_grad()`, `model(ids, use_cache=False)`. **Forward hooks take 3
args `(module, input, output)`; forward-pre hooks take 2 `(module, input)`** (verified).

| Quantity | Hook target | Type | Extract (shape) |
|---|---|---|---|
| `embed` row | `embed_tokens` | forward | `out[0, pos, :]`  → `[H]` |
| layer-`L` input `h_in[L]` | `layers[L]` | **forward_pre** | `inp[0][0, pos, :]` → `[H]` |
| attn delta `Δattn[L]` | `layers[L].self_attn` | forward | `out[0][0, pos, :]` → `[H]`  (**out is a tuple**) |
| mlp delta `Δmlp[L]` | `layers[L].mlp` | forward | `out[0, pos, :]` → `[H]`  (out is a plain tensor) |
| firing `a[L]` | `layers[L].mlp.down_proj` | **forward_pre** | `inp[0][0, pos, :]` → `[I]` |
| layer output (xcheck only) | `layers[L]` | forward | `out[0][0, pos, :]` → `[H]` |
| final-norm point | `model.norm` | forward | `out[0, pos, :]` → `[H]` |

`a[L]` captured as the **literal input to `down_proj`** is exactly `silu(gate(x))*up(x)`
(down_proj has no bias), so `W_down @ a[L]` reproduces the MLP write bit-for-bit — no need
to recompute SiLU. Verified `‖W@a - mlp_out‖∞ = 4.8e-7`.

### 4.3 Trajectory reconstruction (additive — exact, with independent cross-check)

```
h[0]        = embed_out                      # (EMBED, -1)
h[2L+1]     = h_in[L] + Δattn[L]             # (ATTN, L)  == residual after attn write
h[2L+2]     = h[2L+1] + Δmlp[L]              # (MLP,  L)  == residual after mlp write
h[2N+1]     = norm_out                       # (FINAL,-1)
h[2N+2]     = unembed_dir                     # (UNEMBED, pred_id)  (a direction; see §6)
```

This is exact because the residual is literally `prev + delta` in the verified forward.
**Cross-check (DESIGN 1's strength, kept):** we *also* hook each `layers[L]` forward output
and assert `‖h[2L+2] - layer_out[L]‖∞ < 1e-4` (verified 0.0). And we capture `h_in[L]`
independently, asserting `‖h[2L+2] - h_in[L+1]‖∞ < 1e-4` (residual continuity). These prove
the reconstruction matches the model rather than re-implementing it.

### 4.4 Top-K writer selection (per animated token)

For each layer `L`: `idx = np.argpartition(np.abs(a[L]), -K)[-K:]` then sort by `|a|` desc.
`topk_idx[L]=idx`, `topk_a[L]=np.abs(a[L])[idx]`, `down_cols[L]=bundle.down_proj_columns(L, idx)`
→ `[K, H]`. Default `K=48` (range 40–60). Total drawn stars per panel = `N*K` (≈1152 / 1344).
**Resolved:** ATTN nodes have no writer stars (attention is not an MLP write). When the hop is
on `(ATTN, L)` we light layer L's MLP stars using `a[L]` (per the brief: "layer L's node,
attn OR mlp") — we do not invent attention stars.

### 4.5 Public API + lifecycle

```python
def collect_run(bundle, instruction, *, topk=48, self_test=True) -> RunCollection:
    ids = build_input_ids(bundle, instruction); pos = ids.shape[1]-1
    store = {}; handles = []
    try:
        # register all hooks from the §4.2 table (pos-closures; .detach().float().cpu())
        with torch.no_grad():
            out = bundle.model(ids, use_cache=False)
    finally:
        for h in handles: h.remove()          # ALWAYS remove
    # build nodes via §4.3; top-K via §4.4; pred = out.logits[0,pos].argmax()
    rc = ...
    if self_test: _self_test(bundle, rc, out, store)   # §9 asserts
    return rc

def collect_all(bundle, instructions=EXAMPLES, *, topk=48) -> list: ...
```

### 4.6 Disk cache (`.npz` + `.json` sidecar, no pickle of custom classes)

```python
def collect_all_cached(bundle, instructions=EXAMPLES, *, topk=48,
                       cache_dir="out/cache") -> list:
    """key = sha1(tag | transformers.__version__ | topk | tuple(instructions)
                  | tokenizer.name_or_path). Arrays -> np.savez(key.npz) under flat keys
    (f'{ex}/node{n}/h', f'{ex}/node{n}/a', f'{ex}/L{l}/cols', .../idx, .../a,
     f'{ex}/unembed_dir'); scalars (pred ids/strs, T, last_pos, shapes) -> key.json.
    Load with np.load(..., allow_pickle=False). Miss -> collect_all -> save."""
```

Memory note: only `a` is large (`I≤8960 × N≤28 ≈ 250k floats ≈ 1 MB`) — fine on CPU.

---

## 5. `project.py` — joint PCA(3) + sphere  (numpy + sklearn only)

### 5.1 The single shared 3D frame (per model)

Fit **one** `PCA(n_components=3)` per model over the **union** of everything that will be
drawn for that model, so trajectory points, writer stars, and unembed stars share one R³
frame. Union `X = [M, H] float32`:
- all trajectory `node.h` for the chosen prompt (and, if `enrich_pca`, the other 4 prompts'
  trajectories — fit-only, stabilizes a degenerate frame),
- the **drawn** writer columns: `down_cols[L]` for every layer (already top-K),
- the unembed rows (pred token + optional small candidate set).

```python
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

@dataclass
class SphereFrame:
    pca: PCA
    mean_: np.ndarray            # == pca.mean_, kept explicit
    explained_var_ratio: np.ndarray
    H: int
    def project(self, V):        # [*,H] -> [*,3]
        return self.pca.transform(np.asarray(V, np.float32))
    def project_sphere(self, V, eps=1e-8):
        Y = self.project(V); n = np.linalg.norm(Y, axis=-1, keepdims=True)
        return Y / np.maximum(n, eps)

def fit_sphere_frame(runs, *, subsample_cols=1500, seed=0) -> SphereFrame:
    X = _build_union(runs, subsample_cols, seed)          # [M, H] float32
    pca = PCA(n_components=3, svd_solver="full").fit(X)    # centers internally via mean_
    return SphereFrame(pca, pca.mean_.copy().astype(np.float32),
                       pca.explained_variance_ratio_.copy(), runs[0].H)
```

sklearn 0.23.1 correctness (resolved across designs):
- `PCA(n_components=3, svd_solver="full")` is deterministic and supported; full SVD for
  these tiny M. **Do NOT pass `random_state`** to `"full"` (only `"randomized"` uses it) —
  avoid the randomized solver entirely for determinism on old sklearn.
- `PCA.fit` stores `mean_` and `transform` subtracts it → a single shared affine origin for
  **all** drawn objects. Writer columns / unembed rows are directions, but we deliberately
  pass them through the **same** `transform` (same `mean_`) so every object lives in one
  chart, then L2-normalize to S². Center **once over the union**; never re-fit per object class.
- `np.float32` in, but PCA upcasts to float64 internally — fine.
- If `subsample_cols < N*K`, random-subsample columns **for the fit only** with
  `np.random.RandomState(seed)`; all `N*K` columns are still projected for drawing.

### 5.2 Render-ready projected bundle

```python
@dataclass
class ProjectedRun:
    tag: str
    traj_sphere: np.ndarray          # [P, 3] on S^2, ordered hop path
    node_kinds: list                 # parallel to traj_sphere: (NodeKind, layer)
    stars_sphere: dict               # layer -> [K, 3] writer stars on S^2
    stars_a: dict                    # layer -> [K] raw |a_j| (glow magnitude; normed at render)
    unembed_sphere: np.ndarray       # [3] target star on S^2
    pred_token_str: str
    N: int; topk: int

def project_run(frame: SphereFrame, run: RunCollection) -> ProjectedRun: ...
```

The `(UNEMBED, pred_id)` node's sphere position **is** `frame.project_sphere(unembed_dir)`,
the same point as the unembed star drawn in the constellation — so the marker's final hop
lands exactly on the star that then glows (visual identity guaranteed; §6, §7.4).

### 5.3 Per-model frames (resolved conflict, unanimous across designs)

0.5B and 1.5B have different `H` (896 vs 1536) → **a shared PCA basis is mathematically
impossible** without a fabricated alignment map, which we deliberately do not invent. Each
model gets its **own** `SphereFrame`. The side-by-side compares *routing shape/structure*,
not a common coordinate system. Panel titles state "own PCA frame".

---

## 6. The unembedding node (direction → point)

`unembed_dir = embed_tokens.weight[pred_id]` (`[H]`). Its sphere point is
`frame.project_sphere(unembed_dir[None])[0]`. This is both the terminal trajectory node and
the target star, so "where the marker ends" == "which star lights up".

---

## 7. `animate.py` — slerp hop + glow + side-by-side GIF  (numpy + mpl + imageio only)

Headless discipline: at module top, **before importing pyplot**:
```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: registers '3d' projection on mpl 3.2
import matplotlib.cm as cm                    # cm.get_cmap (mpl-3.2 safe)
```

### 7.1 Frame schedule

```python
@dataclass
class HopSchedule:
    total_frames: int
    seg_of_frame: np.ndarray     # [F] int   segment (great-circle hop) index
    t_of_frame:   np.ndarray     # [F] float in [0,1], eased
    node_of_frame: np.ndarray    # [F] int   active node (= source node while t<1)

def build_schedule(P, total_frames=60, ease="smoothstep", hold_end=6) -> HopSchedule:
    """segments = P-1. Distribute `total_frames - hold_end` frames across segments
    (base = q//segments, remainder spread to the first frames). Append `hold_end`
    frames pinned at the last node (so the final glow reads). CLI clamps
    total_frames >= (P-1)+hold_end so every node is visited."""
```
Per-panel `P` differs (51 vs 59); both panels are driven by the **same `total_frames`** so
they finish together, each mapping that budget onto its own segment count.

### 7.2 Marker motion — slerp on S² (great-circle)

```python
def slerp(p0, p1, t, eps=1e-7):
    d = np.clip(np.dot(p0, p1), -1.0, 1.0); w = np.arccos(d)
    if w < eps:                       # coincident -> lerp+renorm
        v = (1-t)*p0 + t*p1
    elif w > np.pi - eps:             # antipodal -> half great-circle about an orthogonal axis
        v = _antipodal_path(p0, p1, t)
    else:
        s = np.sin(w); v = (np.sin((1-t)*w)/s)*p0 + (np.sin(t*w)/s)*p1
    return v / np.maximum(np.linalg.norm(v), eps)
```
`--no-slerp` falls back to 3D lerp+renorm. The antipodal guard matters: PCA can place a write
nearly opposite its predecessor.

### 7.3 Glow mechanic (the core encoding)

Two star populations per panel:
- **Resting constellation:** all `N*K` writer stars always drawn faint/cool
  (`size≈6`, slate RGBA `(0.55,0.60,0.72,0.18)`) so the full constellation is visible.
- **Active layer light-up:** when the active node is `(ATTN,L)` or `(MLP,L)`, layer L's
  K stars light up, driven by `a[L]`:
  - **size** `s_j = S_MIN + (S_MAX-S_MIN)*nrm_j`, `nrm = minmax(|a[L]|)∈[0,1]`, `S_MIN=8, S_MAX=130`.
  - **color/alpha** `cm.get_cmap("plasma")(nrm_j)`, alpha `0.35→1.0`. (Use `cm.get_cmap`,
    NOT `plt.colormaps[...]` which is newer.) Pass `c` as a precomputed `[K,4]` RGBA array.
  - **glow envelope** over the active segment: `env(t)=1-|2t-1|` (triangular pulse) multiplies
    intensity so layers pulse as the marker passes, then dim back.
- **Marker:** single bright sphere `MARKER_S=90`, white; plus a fading **comet tail** of the
  last ≤6 interpolated positions (decreasing alpha) to show direction of travel.
- **Target star:** `unembed_sphere` drawn as a distinct gold star; in the final `hold_end`
  frames it ramps to full glow with an expanding halo (second larger low-alpha scatter) and
  is labeled with `pred_token_str` via `ax.text(...)`.

### 7.4 Per-panel renderer (mpl-3.2 specifics — must follow exactly)

```python
def render_panel(ax, proj, fstate, style):
    ax.cla()
    _draw_wire_sphere(ax, style)        # np.outer cos/sin grid, ~18x18, plot_wireframe
                                        #   linewidth=0.3, color=(1,1,1,0.08)
    _draw_static_stars(ax, proj)        # all writer stars, dim
    _draw_active_glow(ax, proj, fstate) # active layer lit, sized by |a|
    _draw_unembed(ax, proj, fstate)     # candidates dim; target gold; final halo+label
    _draw_marker_and_tail(ax, fstate)
    # equal cube (mpl 3.2 has NO set_box_aspect; do NOT set_aspect('equal') on 3D -> raises):
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim): lim(-1.05, 1.05)
    ax.set_axis_off(); ax.grid(False)
    ax.view_init(elev=style.elev, azim=fstate.azim)   # slow orbit across frames
    ax.text2D(0.02, 0.95, fstate.hud, transform=ax.transAxes, color="white", fontsize=9)
```
- `ax.scatter(xs,ys,zs, s=sizes, c=rgba, depthshade=False, edgecolors='none')`
  (`depthshade=False` keeps glow uniform; supported in 3.2).
- HUD via `ax.text2D(..., transform=ax.transAxes)` (exists in 3.2).
- **No `constrained_layout`**; use `fig.subplots_adjust(...)`.

### 7.5 Side-by-side render loop (reuse ONE figure; per-frame PNG → GIF)

```python
def render_compare_gif(proj_a, proj_b, *, out, total_frames=60, style=STYLE,
                       dpi=110, fps=12, orbit_turns=0.5, use_slerp=True,
                       keep_frames=False) -> str:
    schA = build_schedule(proj_a.traj_sphere.shape[0], total_frames)
    schB = build_schedule(proj_b.traj_sphere.shape[0], total_frames)
    stA = _precompute_states(proj_a, schA, total_frames, use_slerp, orbit_turns, style)
    stB = _precompute_states(proj_b, schB, total_frames, use_slerp, orbit_turns, style)
    fig = plt.figure(figsize=(12, 6), dpi=dpi)
    axL = fig.add_subplot(1, 2, 1, projection='3d')
    axR = fig.add_subplot(1, 2, 2, projection='3d')
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.02)
    frames = []
    for i in range(total_frames):
        render_panel(axL, proj_a, stA[i], style)
        render_panel(axR, proj_b, stB[i], style)
        axL.set_title(f"{proj_a.tag} (own PCA frame)", color="w", fontsize=11)
        axR.set_title(f"{proj_b.tag} (own PCA frame)", color="w", fontsize=11)
        fig.suptitle(stA[i].global_hud, color="w", fontsize=12)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, facecolor="black")
        frames.append(imageio.imread(buf.getvalue()))
        if keep_frames: _dump_png(buf, i)
    plt.close(fig)
    imageio.mimsave(out, frames, duration=1.0/fps, loop=0)   # imageio 2.9.0 signature
    return out
```
imageio 2.9.0 (resolved): `mimsave(uri, frames, duration=1/fps, loop=0)` — `duration` is
**seconds per frame**, `loop=0` = infinite. Use `imageio.imread` (NOT `imageio.v2`). No
`FuncAnimation`, no `FFMpegWriter`, no ffmpeg. Reuse one figure (`ax.cla()` per frame) to
avoid leaks. `render_single_gif(proj, ...)` is the one-panel variant.

### 7.6 `FrameState` + style

```python
@dataclass
class FrameState:
    marker_xyz: np.ndarray; tail: list; active_layer: int; active_kind: str
    glow_norm: np.ndarray|None      # [K] for active layer, else None
    glow_env: float; target_glow: float; azim: float; hud: str; global_hud: str

@dataclass
class GlowStyle:
    bg="black"; sphere_wire=(1,1,1,0.08)
    static_rgba=(0.55,0.60,0.72,0.18); static_s=6
    cmap_name="plasma"; s_min=8.0; s_max=130.0
    marker_rgba=(1,1,1,1.0); marker_s=90; tail_len=6; tail_alpha0=0.5
    target_rgba=(1.0,0.84,0.0,1.0); elev=18; azim0=-60
STYLE = GlowStyle()
```

---

## 8. `compare.py` — two-model driver

```python
@dataclass
class CompareResult:
    runs: dict; frames: dict; projected: dict; gif_paths: list

def build_compare(*, tags=("0.5B","1.5B"), instructions=EXAMPLES, example_idx=0,
                  topk=48, device="cpu", cache_dir="out/cache", enrich_pca=True):
    """Load both bundles; collect_all_cached each; per model fit_sphere_frame
    (enrich with all 5 trajectories if enrich_pca else just the chosen one);
    project_run the chosen example. Pure data, no rendering -> fully unit-testable."""

def run_compare(*, tags=("0.5B","1.5B"), instructions=EXAMPLES, example_idx=0,
                out_dir="out", total_frames=60, topk=48, device="cpu", dpi=110,
                fps=12, use_slerp=True, keep_frames=False, enrich_pca=True) -> CompareResult:
    cr = build_compare(tags=tags, instructions=instructions, example_idx=example_idx,
                       topk=topk, device=device, enrich_pca=enrich_pca)
    a, b = cr.projected[tags[0]], cr.projected[tags[1]]
    out = os.path.join(out_dir, f"compare_ex{example_idx}_{tags[0]}_vs_{tags[1]}.gif")
    render_compare_gif(a, b, out=out, total_frames=total_frames, dpi=dpi, fps=fps,
                       use_slerp=use_slerp, keep_frames=keep_frames)
    cr.gif_paths.append(out); return cr
```
Never force a shared PCA (different H). Twin spheres advance on a shared frame clock.

---

## 9. Self-tests baked into `collect_run` (mechanistic asserts; also in tests/)

1. **MLP-write identity:** `‖W_down[L] @ a[L] − Δmlp[L]‖∞ < 1e-4` ∀L. (verified 4.8e-7)
2. **Layer-output cross-check:** `‖h[2L+2] − layer_out[L]‖∞ < 1e-4` ∀L. (verified 0.0)
3. **Residual continuity:** `‖h[2L+2] − h_in[L+1]‖∞ < 1e-4` ∀L<N-1.
4. **Tied-unembed prediction:** `argmax(norm_row @ embed.weight.T) == logits[0,pos].argmax()`. (verified)
5. **Shape gate:** `down_proj.weight==(H,I)`, `down_cols[L]==(K,H)`, `a[L]==(I,)`, `len(nodes)==2N+3`.

---

## 10. `cli.py` + `__main__.py`

```python
def build_parser():
    p = argparse.ArgumentParser(prog="residual_orrery")
    p.add_argument("--models", nargs="+", default=["0.5B","1.5B"], choices=["0.5B","1.5B"])
    p.add_argument("--example", type=int, default=0)        # index into EXAMPLES
    p.add_argument("--prompt", default=None)                # overrides --example
    p.add_argument("--out", default="out")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--topk", type=int, default=48)          # 40..60
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--no-slerp", dest="use_slerp", action="store_false")
    p.add_argument("--no-enrich-pca", dest="enrich_pca", action="store_false")
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.add_argument("--single", action="store_true")         # one-panel per model
    p.add_argument("--keep-frames", action="store_true")
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    # clamp frames; --single -> render_single_gif per model; else run_compare(...)
    # print resolved settings + gif paths; return 0
```
`__main__.py`: `from .cli import main; raise SystemExit(main())`.

CPU budget: one forward per model (no generation); PCA on ~1–2k points (instant); 60 frames
× 2 panels × (~1.2–1.4k scatter pts + coarse wireframe) at dpi 110 → a few minutes. Knobs to
stay in budget: `--frames`, `--topk`, `--dpi`, wireframe stride.

---

## 11. `__init__.py`

```python
from .models import load_model, ModelBundle, MODEL_IDS
from .examples import EXAMPLES, build_input_ids
from .collect import collect_run, collect_all, collect_all_cached, RunCollection, NodeKind
from .project import fit_sphere_frame, project_run, SphereFrame, ProjectedRun
from .animate import render_compare_gif, render_single_gif, GlowStyle, STYLE
from .compare import build_compare, run_compare, CompareResult
__version__ = "0.1.0"
```

---

## 12. tests/test_residual_orrery.py (fast: 0.5B, 1 short prompt, frames=8, topk=6, dpi=70)

1. `apply_chat_template` is NOT called anywhere (grep guard) — ChatML built manually.
2. MLP-write identity (#9.1); layer-output cross-check (#9.2); residual continuity (#9.3).
3. `a` identity: down_proj pre-hook input == `silu(gate(x))*up(x)` recomputed (<1e-4).
4. Writer-column identity: `down_cols[L].T @ a[L] == Δmlp[L]` (<1e-4).
5. Tied unembed: `embed.weight.data_ptr()==lm_head.weight.data_ptr()` (verified True).
6. Sphere: all projected norms ≈1±1e-5; `slerp(p0,p1,0)==p0`, `==p1` at 1; antipodal stays unit.
7. Schedule: visits every node when `frames ≥ (P-1)+hold_end`; lengths consistent.
8. PCA frame: `project` shape `[N,3]`; `explained_var_ratio` sums ≤1.
9. Cache round-trip: second `collect_all_cached` hits cache, arrays equal.
10. GIF smoke: `render_compare_gif(frames=8, topk=6, dpi=70)` writes a non-empty `.gif`;
    `imageio.get_reader` reads ≥1 frame.

---

## 13. Key correctness call-outs (why this runs on the pinned stack)

- **`apply_chat_template` RAISES (jinja2 2.11.2 < 3.1.0)** → manual ChatML string (§3). All
  three source designs missed this; it is the one change that would otherwise crash everything.
- **Forward hooks take `(module, input, output)` (3 args); pre-hooks `(module, input)`.**
- **`self_attn` output is a tuple → `out[0]`; `mlp` output is a plain tensor.** Verified.
- **`down_proj.weight` is `[H,I]` → columns are writers**; expose transposed `[K,H]`;
  `W_down @ a == mlp_write` unit-tested. Never confuse with `gate/up` `[I,H]`.
- **Tied embeddings** → unembedding rows = `embed_tokens.weight[ids]`; no `lm_head`.
- **`a` = literal down_proj input** (pre-hook) = `silu(gate(x))*up(x)`; no SiLU recompute.
- **Additive trajectory reconstruction** is exact and independently cross-checked vs layer
  output (0.0) and residual continuity.
- **Per-model PCA** (H 896 vs 1536); twin spheres on a shared frame clock; titled "own PCA frame".
- **PCA**: `svd_solver="full"`, no `random_state`, center once over the union.
- **mpl 3.2**: `Axes3D` import registers `'3d'`; `add_subplot(...,projection='3d')`; NO
  `set_box_aspect` (absent), NO `set_aspect('equal')` on 3D (raises) → equal cube via equal
  limits; NO `constrained_layout`; `cm.get_cmap` (not `plt.colormaps`); `text2D` for HUD;
  `depthshade=False`.
- **imageio 2.9.0**: `mimsave(..., duration=1/fps, loop=0)`; `imread` (not `v2`); no ffmpeg;
  per-frame `savefig`→`imread`→`mimsave`; reuse one figure.
- **`torch_dtype=torch.float32`** (not `"auto"`/`dtype=`); `local_files_only=True`+`HF_HUB_OFFLINE=1`;
  `attn_implementation="eager"`; `eval()`, `requires_grad_(False)`, single `no_grad` forward,
  hooks removed in `finally`.

Repo alignment: GIF/palette idioms follow `/Users/pratyushsinha/small_fable/tools/make_preview_gif.py`
(PIL-7.2 manual-anchor patterns, dark palette AMBER/TEAL/ROSE). Note the repo's
`requirements.txt` pins a *newer* transformers for training; `residual_orrery` deliberately
targets the **old** validated stack in §0 — keep the two requirement sets separate.
