# residual_orrery

A small, novel **mechanistic-interpretability visualization**: it renders a transformer's
residual stream as an *orrery* — a constellation of fixed "writer-direction" stars on the
unit sphere S², with a marker that **hops** through the residual trajectory and **lights up**
each layer's stars as that layer fires. It draws **Qwen2.5-0.5B vs Qwen2.5-1.5B side by side**
so you can watch how each model routes the *same* token through its own writer constellation.

<p align="center">
  <img src="../out/compare_ex0_0.5B_vs_1.5B.gif" alt="0.5B vs 1.5B twin-sphere comparison" width="760"><br>
  <em>compare_ex0_0.5B_vs_1.5B.gif — twin spheres animating together</em>
</p>

<p align="center">
  <img src="../out/single_ex0_0.5B.gif" alt="single-panel 0.5B orrery" width="420"><br>
  <em>single_ex0_0.5B.gif — one model, one sphere</em>
</p>

> The GIFs above are written to `out/` when you run the CLI (paths are relative to this
> README, which lives inside the `residual_orrery/` package). See **Usage** below.

---

## The mechanistic idea

### 1. The residual stream is an additive backbone

Each Qwen2 decoder layer writes to the residual stream **twice** — and both writes are pure
additions:

```text
residual = h ;  h = residual + self_attn(input_layernorm(h))        # ATTENTION write
residual = h ;  h = residual + mlp(post_attention_layernorm(h))     # MLP write
```

After the last layer:

```text
h      = model.norm(h)
logits = h @ embed_tokens.weight.T          # TIED embeddings -> unembedding == embedding
```

Because every write is `h = h + delta`, the per-token state is literally the running sum of
these deltas. `collect.py` reconstructs that exact running sum from forward hooks (it captures
the layer input `h_in[L]`, the attention delta, and the MLP delta, then re-adds them), and the
`_self_test` in `collect.py` asserts the reconstruction is exact to `1e-3` against the model's
own intermediate tensors (`layer_out`, `h_in[L+1]`).

### 2. MLP `down_proj` columns are *writer directions* in the residual basis

The MLP is

```text
mlp(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

Let `a = silu(gate(x)) * up(x)` ∈ ℝ^I (the intermediate, or "neuron", activation vector).
`down_proj.weight` has shape `[hidden, intermediate]` and **no bias**, so the MLP write is just

```text
mlp_write = down_proj.weight @ a              # a vector in R^hidden (the residual basis)
```

This is a sum over neurons `j`: each **column** `down_proj.weight[:, j]` is a fixed vector in
the residual basis ℝ^hidden — a **"writer direction"** — and it is added to the residual scaled
by `a_j`. So per token at layer `L`, writer column `j` is "firing" in proportion to `|a_j|`.
`collect.py` verifies this identity directly (`W_down[L] @ a[L] == Δmlp[L]` to `1e-3`).

`collect.py` therefore captures, for the **last prompt-token position**:
- the residual **trajectory** (the cumulative `h` after each sub-write),
- the per-layer neuron activation `a` (hooked as the *input* to `down_proj`),
- the **top-K writer columns** per layer, selected by `|a_j|` for *this* token (default K=48),
- the **unembedding row** of the predicted (argmax) token — which, thanks to tied embeddings,
  is simply `embed_tokens.weight[pred_id]`.

### 3. Project to ℝ³, then point onto the sphere

`project.py` fits **one** `PCA(3)` per model over the **union** of everything that will be
drawn: all trajectory points, a subsample of the drawn writer columns, and the unembedding
rows. That gives a single shared 3-D frame, and every 3-D vector is then **L2-normalized onto
the unit sphere S²** ("compress to 3-D and point them at the sphere").

There is one subtlety that was load-bearing: **Qwen2's residual stream has a few huge-norm
outlier dimensions** (raw projected norms span ~1.3 to ~265). If you PCA the raw vectors, those
outliers hijack the PCA mean and axes, and all ~1000+ small-norm `down_proj` columns collapse to
a single point (sphere std ≈ 0.0002 — an invisible constellation). The fix in `project.py`
(`_unit_rows`, applied in **both** `SphereFrame.project` and `_build_union`) is to
**L2-normalize every vector in ℝ^hidden *before* PCA**, so PCA captures *directional* structure.
With that, the writer stars spread out across the sphere (std ≈ 0.72) and the constellation
becomes visible.

> **Why two separate frames?** 0.5B (hidden=896) and 1.5B (hidden=1536) live in different-sized
> spaces and cannot share a PCA basis without fabricating an alignment. So each panel uses its
> **own PCA frame** (the panel titles say so). The side-by-side comparison is of routing
> **shape**, not of absolute aligned coordinates.

### 4. The hop + the light-up

`animate.py` animates a marker traveling the trajectory:

```text
embed(token) -> L0 post-attn -> L0 post-mlp -> L1 post-attn -> ... -> final-norm -> unembed(argmax)
```

Motion between consecutive sphere points is **slerp** (great-circle interpolation on S²; a
straight-then-renormalize lerp is available via `--no-slerp`). For a model with N layers the
trajectory has **P = 2N + 3** nodes (embed, then attn+mlp per layer, then final-norm, then the
unembedding target).

While the marker is on layer `L`'s node (the ATTN or MLP node), that layer's **top-K writer
stars light up**, sized and colored (plasma colormap) by the token's `|a_j|` at `L`, with a
triangular brightness pulse across the segment. At the very end, the **gold target-token star**
glows with an expanding halo and a text label of the predicted token. A short white **tail**
trails the marker, and the camera slowly orbits.

### 5. What the 0.5B-vs-1.5B difference shows

Running both panels on the same prompt makes the **routing shape** comparable at a glance:

- **0.5B**: 24 layers, intermediate 4864 — a shorter trajectory (P = 2·24+3 = 51 nodes) and a
  sparser writer constellation; the hop is quicker and the lit-up clusters are smaller.
- **1.5B**: 28 layers, intermediate 8960 — a longer trajectory (P = 2·28+3 = 59 nodes) and a
  denser constellation, with more (and differently distributed) stars lighting up per layer.

You see *how many* writer directions each model recruits, *where* on its own sphere they sit,
and *how* the residual marker threads through them to land on the same target token — i.e. two
different internal routes to the same next-token decision.

> **Note on "the answer".** Collection captures a **single greedy decode step**, so the target
> star is the **first next-token** (e.g. `'4'` for "What is 17 plus 25?"), not a full generation
> (`'42'`). That is expected and by design.

---

## Install

The package targets a deliberately old, frozen stack and is verified to run on it
(CPython 3.8; torch 2.2.2 / transformers 4.44.2 / numpy 1.20.1 / matplotlib 3.2.2 /
scikit-learn 0.23.1 / Pillow 7.2.0 / imageio 2.9.0). `pyproject.toml` declares **`>=` floors,
not pins**, so the same spec also installs on a newer environment (e.g. Colab).

From the repository root (`small_fable/`):

```bash
pip install -e .
```

This installs the `residual_orrery` package and a console script, `residual-orrery`.

> No model download is needed at runtime: `models.py` loads Qwen2.5-0.5B-Instruct and
> Qwen2.5-1.5B-Instruct from the **local Hugging Face cache** (`local_files_only=True`,
> `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set defensively).

---

## Usage

### CLI

Console script (after `pip install -e .`) or the module form — they are equivalent:

```bash
residual-orrery            --models 0.5B 1.5B --example 0 --frames 55 --topk 40 --dpi 110 --device cpu --out out
python -m residual_orrery  --models 0.5B 1.5B --example 0 --frames 55 --topk 40 --dpi 110 --device cpu --out out
```

Single panel (one model):

```bash
python -m residual_orrery --models 0.5B --single --example 0 --frames 55 --topk 40 --dpi 110 --device cpu --out out
```

These write:
- `out/compare_ex0_0.5B_vs_1.5B.gif` — the twin-sphere comparison
- `out/single_ex0_0.5B.gif` — the single-model panel
- `out/cache/*.npz` (+ `.json` sidecars) — the collection cache (so re-renders skip the forward pass)

> `--frames 55` is **auto-clamped up to 65**: the marker must visit every trajectory node
> (up to 2·28+3 = 59 for the 1.5B) plus a 6-frame end hold, so the floor is 59 + 6 = 65 frames.

#### CLI options (from `cli.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--models {0.5B,1.5B} [...]` | `0.5B 1.5B` | Which model(s) to render. |
| `--example INT` | `0` | Index into the 5 built-in `EXAMPLES`. |
| `--prompt STR` | `None` | Custom prompt (overrides `--example`). |
| `--out DIR` | `out` | Output directory (GIFs + `cache/`). |
| `--frames INT` | `60` | Frame budget (auto-clamped up so every node is visited). |
| `--topk INT` | `48` | Top-K writer columns per layer, by `\|a_j\|` (40–60 recommended). |
| `--device STR` | `cpu` | Torch device. No local GPU assumed; fp32 on CPU. |
| `--dpi INT` | `110` | Render DPI per frame. |
| `--fps INT` | `12` | GIF frame rate. |
| `--single` | off | One panel per model instead of side-by-side. |
| `--no-slerp` | slerp on | Use straight-line-then-renormalize instead of great-circle. |
| `--no-enrich-pca` | enrich on | Fit PCA on the chosen example only (default fits on all 5). |
| `--no-cache` | cache on | Ignore the on-disk collection cache. |
| `--keep-frames` | off | Also dump per-frame PNGs (`frames_debug/`). |
| `--smoke` | off | Torch-free self-check of the project/animate pipeline on synthetic data, then exit. |

A fast, model-free sanity check (no Qwen weights required):

```bash
python -m residual_orrery --smoke
```

### Python API

`project` and `animate` are **torch-free** (numpy + scikit-learn, and numpy + matplotlib +
imageio respectively), so they iterate quickly on the cached `.npz` without loading a model.

```python
from residual_orrery import (
    load_model, collect_all_cached,        # torch path (needs Qwen in the local HF cache)
    fit_sphere_frame, project_run,         # torch-free
    render_compare_gif, render_single_gif, # torch-free
    EXAMPLES,
)

# 1) collect (cached) for one model
bundle = load_model("0.5B", device="cpu")
runs   = collect_all_cached(bundle, EXAMPLES, topk=40, cache_dir="out/cache")

# 2) fit ONE PCA(3) frame over the union, project the chosen example onto S^2
frame = fit_sphere_frame(runs)            # enriches the basis with all 5 prompts
proj  = project_run(frame, runs[0])

# 3) render
render_single_gif(proj, out="out/single_ex0_0.5B.gif", total_frames=65, dpi=110)
```

Or drive the whole two-model pipeline at once:

```python
from residual_orrery import run_compare
run_compare(tags=("0.5B", "1.5B"), example_idx=0, out_dir="out",
            total_frames=65, topk=40, dpi=110, device="cpu")
```

---

## The 5 built-in examples

Medium-difficulty prompts (solvable by both models, but not single-token-trivial):

```text
0  What is 17 plus 25? Reply with just the number.
1  Give one word that means the opposite of 'hot'.
2  Complete the sentence: The capital of France is
3  Is 7 a prime number? Answer yes or no.
4  Reverse the letters of the word 'cat'.
```

> Prompts are wrapped in the Qwen2.5 **ChatML** template **by hand** in `examples.py`
> (`<|im_start|>system … <|im_start|>assistant\n`). This avoids
> `tokenizer.apply_chat_template`, which raises on the frozen jinja2 2.11.2 in the target env.

---

## How the pieces fit together

```text
examples ─┐
models  ──┼─► collect ──► project ──► animate
          └───────────────► compare ─────┘ ──► cli ──► __main__
```

| Module | Role |
| --- | --- |
| `examples.py` | The 5 prompts + the hand-built ChatML tokenizer path. |
| `models.py` | Load & freeze Qwen2.5 0.5B/1.5B from the local cache; typed accessors to the exact HF module paths; shape/tie sanity guards. |
| `collect.py` | Forward hooks → residual trajectory + per-layer neuron activation `a` + top-K `down_proj` columns + unembed row; mechanistic self-tests; `.npz`/`.json` disk cache. |
| `project.py` | Joint `PCA(3)` over the union + L2 sphere-normalize (with the pre-PCA unit-normalization fix). |
| `animate.py` | Slerp hop + per-layer light-up + side-by-side / single render → GIF via per-frame PNG and `imageio.mimsave` (no ffmpeg). |
| `compare.py` | Two-model driver: collect (cached) → per-model PCA frame → project → render. |
| `cli.py` | `argparse` entry point (and `--smoke`). |

### Environment notes baked into the code

- **No ffmpeg / no `FuncAnimation`**: GIFs are assembled from per-frame PNGs via
  `imageio.mimsave` — the most version-robust path on imageio 2.9.
- **matplotlib 3.2 3-D quirks**: 3-D axes are registered via `mpl_toolkits.mplot3d.Axes3D`;
  there is no `set_box_aspect`, so an equal cube is enforced with equal limits; and the 3-D
  panes (which default to opaque white in mpl 3.2 and would hide the black sphere) are
  explicitly painted black (`_blacken_3d_axes`).
- **CPU, fp32, `torch.no_grad()`, `model.eval()`, `requires_grad_(False)`**, eager attention
  (so the attention-write reconstruction is exact).

---

## License

MIT.
