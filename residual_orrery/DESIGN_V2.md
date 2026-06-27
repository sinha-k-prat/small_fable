# residual_orrery v2 — Implementation-Ready Spec (synthesis of Design A + Design B)

This is ONE authoritative spec for v2. It merges Design A (generation + variants, with the
load-bearing `digits`/last-run rule) and Design B (render: beacons, input marker, collective,
rescue, MLP→column links, CLI, notebook). **Every change is additive and version-safe.** All
existing modes keep working: `single`/`compare`, the torch-free `project`/`animate` path, the
`--smoke` self-check, and old `.npz`/`.json` caches all stay green.

Verified local stack: mpl 3.2.2, numpy 1.20.1, imageio 2.9.0, transformers 4.44.2, torch 2.2.2,
sklearn; jinja2 is **2.11.2** so `apply_chat_template` RAISES — keep MANUAL ChatML everywhere.
Also runs on modern Colab via the existing shims `_get_cmap` / `_zoom_in` / `_save_gif` /
`_first_tensor`. Generation verified on the cached Qwen2.5-0.5B: `17+25 -> '42'` (the single-token
`'4'` mislabel is real and fixed), `antonym -> 'Cold'`, `prime -> 'No.'` (wrong; 7 IS prime → RED).

---

## 0. Conflict resolutions (authoritative decisions)

1. **Task representation** → adopt B's frozen `Task` dataclass (carries `key/system/user/gold/
   grade/gen_tokens` in one record). Drop A's separate parallel `GOLD`/`GEN_TOKENS` lists as the
   *source of truth*; instead derive convenience lists from the `Task` records. `build_input_ids`
   still accepts a plain `str`, a `Task`, OR a `(system, user)` tuple (A's flexibility) so all
   call sites flow through.
2. **Field names on `RunCollection`/`ProjectedRun`** → unify to: `answer_text`, `is_correct`,
   `gold`, `grade_mode`. (A's `mode` is renamed `grade_mode`; B's `grade` on `Task` stays the
   per-task spelling and maps to `grade_mode`.)
3. **`collect_run` signature** → adopt B's: `generate=False, gen_tokens=24, gold="",
   grade_mode=""`. `generate=False` (default) ⇒ identical to today. `instruction` may be a `Task`,
   in which case `system/gold/grade_mode/gen_tokens` are pulled from it (these override kwargs).
4. **Generation implementation** → adopt B's verified core (clean `GenerationConfig` that nulls
   `temperature/top_p/top_k`, explicit `attention_mask`, `do_sample=False, num_beams=1`), PLUS
   A's belt-and-suspenders ChatML stop: pass `eos_token_id=[<|im_end|>, eos]` as a list. Under
   `torch.no_grad`; device-safe (ids already on `bundle.device`; decode on CPU).
5. **`digits` grader semantics** → adopt A's rule (LOAD-BEARING): take the **LAST** contiguous
   integer run in the answer (the committed final answer after the scaffold's many earlier
   numbers) and require **exact digit equality** to gold. This makes the multiplication near-miss
   `FINAL ANSWER: 8256722` render **RED** (≠ `8293662`), exactly the DISCOVERED_PROMPTS finding.
   Reject B's `dg in da` substring-for-digits (it would falsely pass `8293662` appearing anywhere
   and would mis-handle the scaffold). `add` (`42`) still grades correctly under last-run-equality.
6. **Beacon color is decoupled from `pred_token_id`** (A's headline): color comes ONLY from
   `is_correct` (generation), never from the single next-token. `pred_token_id`/`unembed_dir`
   remain the *trajectory landmark* (unchanged), used only as the label fallback when
   `answer_text` is empty (old runs / `--smoke`).
7. **Variant key for plain** → `"plain"` (back-compat: the legacy 5-prompt `EXAMPLES` list of
   strings is preserved as `EXAMPLES = [t.user for t in EXAMPLES_PLAIN]`).
8. **Per-frame pulse** → reuse `FrameState.glow_env` for the persistent beacon pulse (B's
   recommendation): no new `FrameState` field, no `fstate_frame_phase` helper.
9. **MLP→column links** → ON by default for single/compare/rescue (≤2 traces); OFF by default in
   collective (busy with N traces), gated by a `links=False` kwarg.

The traced through-layer trajectory at the last prompt token (`nodes`, `topk_*`, `down_cols`,
`unembed_dir`, all `_self_test` identities) is **byte-for-byte unchanged**. Generation is a
separate post-hoc `.generate` call that runs only when `generate=True`.

Files touched: `residual_orrery/{examples.py, collect.py, project.py, animate.py, compare.py,
cli.py, __init__.py}` and `notebooks/residual_orrery_colab.ipynb`.

---

## 1. `examples.py` — Task records, three aligned variant sets, system-aware MANUAL ChatML

### 1.1 The `Task` record (frozen dataclass)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    key: str            # stable id: "add","antonym","capital","prime","reverse","mult"
    system: object      # optional system message (str) or None -> default helpful-assistant
    user: str           # user prompt
    gold: str           # gold answer text
    grade: str          # "digits" | "substr" | "equal"  -> RunCollection.grade_mode
    gen_tokens: int     # per-task greedy cap (mult=256; others small)

TASK_KEYS = ["add", "antonym", "capital", "prime", "reverse", "mult"]
```

Gold + grader (from GROUNDED FINDINGS), index-aligned across all three variant sets:

| key | gold | grade | gen_tokens |
|---|---|---|---|
| add | `42` | `digits` | 8 |
| antonym | `cold` | `substr` | 8 |
| capital | `Paris` | `substr` | 8 |
| prime | `yes` | `substr` (7 IS prime; 0.5B says "No" → RED) | 8 |
| reverse | `tac` | `substr` | 8 |
| mult | `8293662` | `digits` | 256 |

### 1.2 System-aware MANUAL ChatML builder (jinja2-safe; back-compat exact)

Replace the single hardcoded-system `_CHATML` with a system-parameterized template. Default
system is byte-identical to the old one, so `build_chatml('x')` and `build_input_ids(bundle,'x')`
produce the SAME ids as before.

```python
_CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_DEFAULT_SYSTEM = "You are a helpful assistant."

def build_chatml(prompt, system=None):
    """Raw ChatML string. system=None -> default helpful-assistant (byte-identical to v1)."""
    return _CHATML.format(system=(system or _DEFAULT_SYSTEM), prompt=prompt)

def build_input_ids(bundle, prompt, system=None):
    """Qwen2.5 ChatML prompt -> ids [1,T] on bundle.device. MANUAL ChatML (NO apply_chat_template).
    `prompt` may be: a plain user str; a Task (system/user pulled from it); or a (system,user)
    tuple/list (its system overrides the kwarg). Back-compat: build_input_ids(bundle, str) is
    unchanged (default system)."""
    if isinstance(prompt, Task):
        system, prompt = prompt.system, prompt.user
    elif isinstance(prompt, (tuple, list)):
        system, prompt = prompt[0], prompt[1]
    text = build_chatml(prompt, system=system)
    ids = bundle.tokenizer(text, return_tensors="pt").input_ids  # [1, T]
    assert ids.dim() == 2 and ids.shape[0] == 1, ids.shape
    return ids.to(bundle.device)
```

### 1.3 The three index-aligned variant sets (SAME 6 tasks, SAME order)

```python
EXAMPLES_PLAIN: list[Task]    # the current 5 EXAMPLES + mult, plain user prompts, system=None
EXAMPLES_SIMPLE: list[Task]   # terse prompts where the small model tends to FAIL
EXAMPLES_DETAILED: list[Task] # step-by-step PROGRAMMATIC scaffold that walks the METHOD,
                              # NEVER states the answer

VARIANTS = {"plain": EXAMPLES_PLAIN, "simple": EXAMPLES_SIMPLE, "detailed": EXAMPLES_DETAILED}

# Back-compat: existing code imports EXAMPLES as a list[str] of USER prompts. Unchanged behavior
# for single/compare/cli/cache.
EXAMPLES = [t.user for t in EXAMPLES_PLAIN]

assert len(EXAMPLES_PLAIN) == len(EXAMPLES_SIMPLE) == len(EXAMPLES_DETAILED) == 6
for s in (EXAMPLES_PLAIN, EXAMPLES_SIMPLE, EXAMPLES_DETAILED):
    assert [t.key for t in s] == TASK_KEYS
```

**EXAMPLES_PLAIN** (system=None for all; preserves the legacy 5 + adds mult):
```
add     : "What is 17 plus 25? Reply with just the number."
antonym : "Give one word that means the opposite of 'hot'."
capital : "Complete the sentence: The capital of France is"
prime   : "Is 7 a prime number? Answer yes or no."
reverse : "Reverse the letters of the word 'cat'."
mult    : "Compute the exact product: 9246 x 897. Reply with just the number."
```
(The first five `user` strings MUST equal the current `EXAMPLES` verbatim so `EXAMPLES` is byte-identical and v1 caches/notebook examples don't drift.)

**EXAMPLES_SIMPLE** (system=None; terse, where the small model fails):
```
add     : "What is 17 plus 25? Reply with just the number."
antonym : "Give one word that means the opposite of 'hot'. Reply with one word."
capital : "The capital of France is"
prime   : "Is 7 a prime number? Answer yes or no."
reverse : "Reverse the letters of the word 'cat'. Reply with just the reversed word."
mult    : "Compute the exact product: 9246 x 897. Reply with just the number."
```

**EXAMPLES_DETAILED** — every task shares ONE system message (so the notebook can quote it once):
```
SYSTEM (all 6): "You are a meticulous step-by-step solver. Show ALL intermediate work explicitly
  on separate lines, never skip or summarize a step. End with a line 'FINAL ANSWER:' then only
  the answer."
```
User prompts walk the METHOD and NEVER state the answer (verify by eye: no "42", "cold", "Paris",
yes/no verdict, "tac", or "8293662" appears in any user/system string):
```
add     : "Add 17 and 25. Add column by column from the right: add the ones digits (7 + 5),
           write the ones digit of that sum and carry any tens; then add the tens digits plus
           the carry. Combine the digits."
antonym : "Find one word that is the opposite of 'hot'. State the dimension the word varies
           along, then name the word at the far opposite end of that dimension from 'hot'."
capital : "Name the capital city of France. State the country, then recall the city that is its
           seat of national government."
prime   : "Determine whether 7 is prime. A prime has exactly two distinct positive divisors, 1
           and itself. Test each integer d from 2 up to 6 and state whether d divides 7 evenly.
           If none divide it, it is prime. Answer yes or no."
reverse : "Reverse the letters of the word 'cat'. List its letters in order, numbering them.
           Then write the letters from the last numbered one to the first, concatenated."
mult    : "Compute 9246 x 897. Multiply by long multiplication: multiply the top number by each
           digit of the bottom number (write each partial product, shifted by place), then add
           the partial products."
```
`EXAMPLES_DETAILED[5]` (mult system + user) is the `DISCOVERED_PROMPTS.md` scaffold verbatim-style
and MUST be transcribed exactly (`gen_tokens=256`). The mult SIMPLE prompt `"Compute the exact
product: 9246 x 897. Reply with just the number."` is also load-bearing.

---

## 2. `collect.py` — multi-token greedy generation + correctness (additive, opt-in)

### 2.1 New `RunCollection` fields (defaults keep all old constructors + caches valid)

Append at the END of the dataclass (so `_smoke`, `_load_runs`, every positional/kw construction
still works):
```python
@dataclass
class RunCollection:
    # ... all existing fields unchanged, ending with `topk: int` ...
    # ---- v2 additive: full generated answer + correctness for the terminal beacon ----
    answer_text: str = ""        # full greedy continuation (decoded, specials skipped)
    gold: str = ""               # gold string this run was judged against ("" if unknown)
    grade_mode: str = ""         # "digits"|"substr"|"equal"|"" (correctness normalizer)
    is_correct: object = None    # True / False / None(unknown) — bool or None
```

### 2.2 Greedy generation helper (B's verified core + A's ChatML stop)

```python
def _greedy_answer(bundle, ids, max_new_tokens):
    """Greedy multi-token continuation of `ids` ([1,T] on bundle.device) -> decoded NEW tokens
    only, specials skipped, stripped. torch.no_grad, device-safe. Returns '' if max_new_tokens<=0.
    Version-safe: CLEAN GenerationConfig (the cached Qwen config sets temperature/top_p/top_k,
    which WARN under do_sample=False) + explicit attention_mask (pad==eos)."""
    import torch
    from transformers import GenerationConfig
    if max_new_tokens is None or max_new_tokens <= 0:
        return ""
    tok = bundle.tokenizer
    T = ids.shape[1]
    attn = torch.ones_like(ids)
    eos = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [e for e in (im_end, eos) if isinstance(e, int) and e >= 0] or eos
    gc = GenerationConfig(
        do_sample=False, num_beams=1,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=(eos if eos is not None else tok.pad_token_id),
        eos_token_id=eos_ids,                 # list -> stop on <|im_end|> OR eos
        temperature=None, top_p=None, top_k=None,
    )
    with torch.no_grad():
        out = bundle.model.generate(ids, attention_mask=attn, generation_config=gc)
    new_ids = out[0, T:].detach().to("cpu").tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()
```
Notes: separate from the traced forward (which keeps `use_cache=False` + hooks); generation runs
AFTER `finally` removed the hooks, so they never interfere. `do_sample=False, num_beams=1` is
greedy on transformers 4.44.2 and modern.

### 2.3 Normalizer + judge (A's last-run digits rule — LOAD-BEARING)

```python
import re

def _norm_alnum(s):
    """lowercase, keep [a-z0-9] only (drops spaces/punct)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _digits_only_last(s):
    """LAST contiguous integer run in the string, digits only. Handles
    'FINAL ANSWER: 8,293,662.' and the scaffold's many earlier numbers by taking the LAST run
    -> the committed answer. '' if no digits."""
    runs = re.findall(r"\d[\d,]*", str(s))
    return re.sub(r"\D", "", runs[-1]) if runs else ""

def judge(answer_text, gold, mode):
    """-> True / False / None(unknown). None when gen skipped or gold/mode missing."""
    if not gold or not mode or answer_text is None or answer_text == "":
        return None
    if mode == "digits":
        a, g = _digits_only_last(answer_text), _digits_only_last(gold)
        return None if (a == "" or g == "") else (a == g)   # EXACT last-run equality
    if mode == "equal":
        return _norm_alnum(answer_text) == _norm_alnum(gold)
    # default "substr": gold appears as a normalized substring of the answer
    g = _norm_alnum(gold)
    return (g != "") and (g in _norm_alnum(answer_text))
```
Grounded behavior produced:
- `add` `"42"`/`" 42."`/`"42"` → `digits` last-run `"42"` == gold → **True (GREEN)**. (single-token `'4'` would have been False.)
- `mult` scaffold ending `"FINAL ANSWER: 8256722"` → last run `8256722` ≠ `8293662` → **False (RED)** — the DISCOVERED_PROMPTS near-miss, correctly RED. A correct run ending `8293662` → True.
- `prime` `"No."` → substr `"yes"` not present → **False (RED)** (7 IS prime). `"Yes, 7 is prime."` → True.
- `capital` `" Paris."` → substr `"paris"` → True. `reverse` `"tac"` → True (`"cat"` → False).
- gen skipped (`generate=False` / cap≤0) → `answer_text=""` → `is_correct=None` → **AMBER** beacon.

### 2.4 Wiring into `collect_run` / `collect_all` (additive, opt-in)

```python
def collect_run(bundle, instruction, topk=48, self_test=True,
                generate=False, gen_tokens=24, gold="", grade_mode=""):
    """instruction may be a str OR a Task. If a Task: system/gold/grade/gen_tokens come from it
    (override the kwargs). generate=False (default) => identical to v1 (no .generate call)."""
    from .examples import Task
    if isinstance(instruction, Task):
        gold = instruction.gold
        grade_mode = instruction.grade
        if generate:
            gen_tokens = instruction.gen_tokens
    ids = build_input_ids(bundle, instruction)   # accepts str OR Task
    # ... (traced forward + nodes + topk + self_test EXACTLY as v1; rc built with new fields default)
    if self_test:
        _self_test(bundle, rc, out, store)
    # ---- v2: greedy answer + correctness (only when requested) ----
    if generate and gen_tokens and gen_tokens > 0:
        rc.answer_text = _greedy_answer(bundle, ids, gen_tokens)
        rc.gold = gold or ""
        rc.grade_mode = grade_mode or ""
        rc.is_correct = judge(rc.answer_text, rc.gold, rc.grade_mode)
    return rc
```
`build_input_ids` is called ONCE; reuse the same `ids` for both the traced forward (today's path)
and generation (so the system message threads through identically). The `pred_token_id` /
`unembed_dir` / `nodes` path is byte-for-byte unchanged.

```python
def collect_all(bundle, instructions=EXAMPLES, topk=48, self_test=True,
                generate=False, gen_tokens=24):
    """generate=False => legacy. When instructions are Tasks, per-task gold/grade/gen_tokens
    are pulled inside collect_run. A scalar gen_tokens is the fallback cap for non-Task / plain."""
    return [collect_run(bundle, ins, topk=topk, self_test=self_test,
                        generate=generate, gen_tokens=gen_tokens) for ins in instructions]
```

### 2.5 Cache: JSON sidecar scalars + key bump (old caches still load)

`_save_runs` — inside `meta["runs"].append({...})`:
```python
    "answer_text": rc.answer_text,
    "gold": rc.gold,
    "grade_mode": rc.grade_mode,
    "is_correct": (None if rc.is_correct is None else bool(rc.is_correct)),
```
`_load_runs` — in the `RunCollection(...)` call, read with `.get(...)` defaults:
```python
    answer_text=rm.get("answer_text", ""),
    gold=rm.get("gold", ""),
    grade_mode=rm.get("grade_mode", ""),
    is_correct=rm.get("is_correct", None),
```
`.get` defaults ⇒ OLD caches (without these keys) still load. No new arrays (scalars only; no
pickle). `_cache_key` payload gains `"v2"`, `str(bool(generate))`, `str(int(gen_tokens))` so a
generated run never aliases a v1 cache and SIMPLE vs DETAILED vs different cap get distinct files.

`collect_all_cached` threads `generate`/`gen_tokens` through to `collect_all` and into `_cache_key`.

### 2.6 `_smoke` stays green

`_smoke` builds `RunCollection` directly; the four new fields default ⇒ constructs fine, no torch.
(Optionally set `answer_text="42", is_correct=True` on one synthetic run to exercise the GREEN
beacon path in the smoke GIF.)

---

## 3. `project.py` — carry answer/correctness onto `ProjectedRun` (torch-free, additive)

Add three fields (defaults so old/smoke runs and the unembed-coincidence assert all hold):
```python
@dataclass
class ProjectedRun:
    # ... existing fields ...
    answer_text: str = ""
    is_correct: object = None    # True/False/None
    gold: str = ""
```
`project_run` copies them one-for-one from the run (`getattr(run, "answer_text", "")`, etc., so
even a duck-typed smoke run without the fields projects cleanly). Node 0 is always EMBED (the INPUT
token) — render keys off `node_kinds`, no new field needed. `unembed_sphere == traj_sphere[-1]`
assertion is preserved (beacon sits on the terminal node).

---

## 4. `animate.py` — correctness beacon, input marker, MLP→column links, collective + rescue

### 4.1 `GlowStyle` additions (all defaulted; no existing field changed)

```python
    # correctness beacon (Feature 2)
    beacon_correct_rgb: tuple = (0.20, 0.95, 0.45)   # green
    beacon_wrong_rgb:   tuple = (0.95, 0.28, 0.28)   # red
    beacon_unknown_rgb: tuple = (1.0, 0.78, 0.18)    # amber/gold (unknown)
    beacon_s: float = 90.0
    beacon_label_fs: float = 11.0
    # input token marker (Feature 2)
    input_rgba: tuple = (0.35, 1.0, 0.55, 1.0)       # green diamond at node 0
    input_s: float = 70.0
    # MLP -> writer-column links (Feature 6)
    link_rgba: tuple = (0.70, 0.80, 1.0, 0.16)       # faint cool
    link_lw: float = 0.6
    link_topn: int = 4                               # 3..5 brightest firing cols
```

### 4.2 Beacon (Feature 2) — replaces `_draw_unembed`, keeps its pulse/halo/hold mechanics

```python
def _beacon_rgb(proj, style):
    c = getattr(proj, "is_correct", None)
    if c is True:  return style.beacon_correct_rgb
    if c is False: return style.beacon_wrong_rgb
    return style.beacon_unknown_rgb                  # None -> amber

def _draw_beacon(ax, proj, fstate, style, label_prefix=""):
    """Terminal beacon at unembed_sphere: persistent, pulsing, GREEN/RED/amber by correctness,
    labeled with the generated answer (falls back to pred_token_str for old/smoke runs)."""
    t = proj.unembed_sphere
    rgb = _beacon_rgb(proj, style)
    g = fstate.target_glow                            # 0 until end-hold, ramps to 1
    pulse = 0.55 + 0.45 * fstate.glow_env             # reuse glow_env (no new FrameState field)
    base_a = 0.45 + 0.55 * g
    ax.scatter([t[0]],[t[1]],[t[2]], s=style.beacon_s*(1.0+1.2*g),
               c=[list(rgb)+[base_a]], depthshade=False, edgecolors="none")
    ax.scatter([t[0]],[t[1]],[t[2]], s=style.beacon_s*(2.2+2.5*g)*pulse,     # halo
               c=[list(rgb)+[0.16*(0.5+g)]], depthshade=False, edgecolors="none")
    label = (getattr(proj, "answer_text", "") or proj.pred_token_str).strip().replace("\n", " ")
    if len(label) > 22: label = label[:21] + "…"
    if g > 0:
        ax.text(t[0], t[1], t[2], "  "+label_prefix+label, color="white",
                fontsize=style.beacon_label_fs,
                bbox=dict(facecolor=list(rgb)+[0.35], edgecolor="none", pad=1.5))
```
Verified mpl-3.2-safe: boxed `ax.text(..., bbox=dict(facecolor=(r,g,b,a), edgecolor="none"))`.

### 4.3 Input diamond at node 0 (Feature 2)

```python
def _draw_input_marker(ax, proj, style):
    """Distinct green diamond at node 0 (the INPUT/embed token) so the trajectory start is
    identifiable. marker='D', depthshade=False, edgecolors='none' — verified on mpl 3.2.2."""
    p = proj.traj_sphere[0]
    ax.scatter([p[0]],[p[1]],[p[2]], s=style.input_s, marker="D",
               c=[style.input_rgba], depthshade=False, edgecolors="none")
```

### 4.4 MLP→writer-column links (Feature 6)

```python
def _draw_mlp_links(ax, proj, fstate, style):
    """Faint dashed links from the active MLP residual node to its top-N firing writer-column
    stars. Drawn only while an MLP layer fires, so links appear and fade with the marker.
    stars_sphere[L] is already desc-sorted by |a| in collect_run."""
    if fstate.active_kind != "mlp":
        return
    L = fstate.active_layer
    if L not in proj.stars_sphere or L not in proj.stars_a:
        return
    src = proj.traj_sphere[2 * L + 2]                 # MLP node index in the P=2N+3 layout
    stars = proj.stars_sphere[L]                      # [K,3]
    a = np.asarray(proj.stars_a[L], np.float64)       # [K] desc by |a|
    n = min(style.link_topn, stars.shape[0])
    env = fstate.glow_env
    for k in range(n):
        d = stars[k]
        w = a[k] / max(a[0], 1e-8)
        alpha = style.link_rgba[3] * w * (0.4 + 0.6 * env)
        ax.plot([src[0], d[0]], [src[1], d[1]], [src[2], d[2]],
                color=style.link_rgba[:3], alpha=alpha, linewidth=style.link_lw,
                linestyle=":", solid_capstyle="round")
```
Verified mpl-3.2-safe: dashed 3D `ax.plot([..],[..],[..], linestyle=":", alpha=..., linewidth=..)`.

### 4.5 `render_panel` wiring (one path → applies in single/compare/collective/rescue)

Add a `links=True` flag (default True; collective passes `links=False`). Draw links BEFORE the
glow (so glowing stars sit on top of their links), beacon before trace, input marker on top.
```python
def render_panel(ax, proj, fstate, style, links=True, beacon_prefix=""):
    ax.cla()
    _blacken_3d_axes(ax, style)
    _draw_wire_sphere(ax, style)
    _draw_static_stars(ax, proj, style)
    if links:
        _draw_mlp_links(ax, proj, fstate, style)      # Feature 6 (faint, under glow)
    _draw_active_glow(ax, proj, fstate, style)
    _draw_beacon(ax, proj, fstate, style, label_prefix=beacon_prefix)   # Feature 2 (was _draw_unembed)
    _draw_trace(ax, proj, fstate, style)
    _draw_input_marker(ax, proj, style)               # Feature 2 (on top)
    # ... existing limits / axis-off / view_init / _zoom_in / HUD unchanged ...
```
`render_single_gif` / `render_compare_gif` route through `render_panel` ⇒ they now show
correctness-colored beacons, input diamonds, and links with NO signature change (defaults).

### 4.6 `render_collective_gif` (Feature 4) — ONE sphere, ALL tasks overlaid

```python
def render_collective_gif(projected_runs, out, total_frames=72, style=STYLE, dpi=110, fps=12,
                          orbit_turns=0.5, use_slerp=True, keep_frames=False, hold_end=8,
                          cmap_name="tab10", links=False, title=None):
    """One shared sphere with every task trajectory overlaid, each in a distinct qualitative
    color, each ending at its OWN correctness-colored labeled beacon (label prefixed with the
    task key so overlapping beacons stay legible), slow shared orbit."""
```
Mechanics (reuse existing machinery):
- Caller fits ONE shared frame: `frame = fit_sphere_frame(rc_list)` (already accepts a LIST →
  one shared PCA frame), `projected_runs = [project_run(frame, rc) for rc in rc_list]`.
- Per-trace schedule built to a common `F = max` (same rebuild pattern as `render_compare_gif`);
  shared azimuth so all traces orbit together. `proj_j._trace_xyz = _build_trace(...)` per trace.
- Per trace `j`: `col = _get_cmap(cmap_name)(j % cmap.N)`; clone a per-trace style via
  `dataclasses.replace(style, trace_rgba=col[:3]+(0.95,), node_hot_rgba=col, input_rgba=col)` so
  `_draw_trace`/`_draw_input_marker` pick up the trace color. The BEACON stays correctness-colored
  (green/red/amber); its label is prefixed with the task key.
- Draw wire sphere + static-star union ONCE per frame, then loop traces into the SAME `ax`
  (beacon + trace + input per trace; links off by default — too busy with N traces).
- Use `_zoom_in` / `_save_gif` as elsewhere.

For the 2-model side-by-side collective, add a thin `render_collective_pair_gif(runs_a, runs_b,
out, ...)` arranging two 3D subplots (mirrors `render_compare_gif`'s figure layout) and calling the
per-frame collective draw on each axis. Single-model path calls `render_collective_gif` directly.

### 4.7 `render_rescue_gif` (Feature 5) — SIMPLE vs DETAILED, same task, same sphere

```python
def render_rescue_gif(proj_simple, proj_detailed, out, total_frames=72, style=STYLE, dpi=110,
                      fps=12, orbit_turns=0.5, use_slerp=True, keep_frames=False, hold_end=8,
                      simple_rgb=(0.95,0.45,0.20), detailed_rgb=(0.30,0.65,1.0),
                      task_label="", model_tag="", links=True):
    """Overlay two variant trajectories of ONE task on ONE shared sphere. Each trace its own
    color; each ends at its correctness-colored beacon (simple typically RED, detailed GREEN).
    A RED->GREEN flip shows as the detailed path turning to a different region. Slow orbit."""
```
A thin specialization of the collective overlay with exactly TWO traces (simple=orange,
detailed=blue) and an explanatory suptitle, e.g.
`"RESCUE — {task} on {model}: simple(orange)->{ok_s} vs detailed(blue)->{ok_d}"` where `ok_*` is
"correct"/"wrong"/"?" from each `is_correct`. `links=True` here (only two traces) so the "turn" is
mechanistically legible. Beacons stay correctness-colored; trace colors are per-variant via the
same `dataclasses.replace(style, ...)` clone.

---

## 5. `compare.py` — `build_collective`/`run_collective` + `build_rescue`/`run_rescue`

### 5.1 Collective

```python
@dataclass
class CollectiveResult:
    runs: dict          # tag -> list[RunCollection]
    frames: dict        # tag -> SphereFrame (ONE per model, fit over all tasks)
    projected: dict     # tag -> list[ProjectedRun]
    gif_paths: list = field(default_factory=list)

def build_collective(tags=("0.5B","1.5B"), variant="plain", device="cpu", topk=48,
                     cache_dir="out/cache", generate=True, gen_tokens=24, use_cache=True):
    """Per model: collect ALL tasks of VARIANTS[variant] (with generation+correctness), fit ONE
    shared frame over all tasks, project every task. Pure data; no rendering. Frees each bundle."""

def run_collective(tags=("0.5B","1.5B"), variant="plain", out_dir="out", total_frames=72,
                   device="cpu", topk=48, dpi=110, fps=12, use_slerp=True, keep_frames=False,
                   generate=True, gen_tokens=24, use_cache=True):
    """1 model -> single collective sphere GIF (render_collective_gif).
       2 models -> two collective spheres side by side (render_collective_pair_gif).
       Returns CollectiveResult with gif_paths set."""
```
`build_collective` passes `instructions=VARIANTS[variant]` (Task list) into `collect_all_cached`
with `generate=True`; per-task gold/grade/gen_tokens are pulled inside `collect_run`. `fit_sphere_
frame(rc_list)` yields the ONE shared frame; `project_run` each task.

### 5.2 Rescue

```python
@dataclass
class RescueResult:
    runs: dict          # tag -> {"simple":[RC...], "detailed":[RC...]}
    frames: dict        # tag -> list[SphereFrame]  (one shared frame per task)
    projected: dict     # tag -> list[(proj_simple, proj_detailed)]  per task
    gif_paths: list = field(default_factory=list)

def build_rescue(tags=("0.5B",), task=None, device="cpu", topk=48, cache_dir="out/cache",
                 use_cache=True):
    """Per model: collect EXAMPLES_SIMPLE and EXAMPLES_DETAILED (both generate=True; per-task
    gen_tokens from the Task, so mult=256). For each task (or just `task`), fit ONE shared frame
    over its [simple_rc, detailed_rc] so both trajectories share a PCA basis and the turn is
    comparable, then project both. `task` accepts an int index OR a key ('mult'), resolved
    against TASK_KEYS; None -> all tasks."""

def run_rescue(tags=("0.5B",), task=None, out_dir="out", total_frames=72, device="cpu", topk=48,
               dpi=110, fps=12, use_slerp=True, keep_frames=False, use_cache=True):
    """Render the rescue overlay per resolved task (one GIF each; multiplication is the
    headline). Returns RescueResult with gif_paths set."""
```
Shared-frame-per-task: `fit_sphere_frame([simple_rc, detailed_rc])` (optionally include the other
tasks of one variant as fit-only anchors). Multiplication carries `gen_tokens=256` via its Task so
the detailed scaffold is produced and the detailed beacon reflects the near-miss/flip.

`build_compare`/`run_compare` gain pass-through `generate=False, gen_tokens=24` (defaults keep v1
behavior); when set, the compare beacons become correctness-colored.

---

## 6. `cli.py` — `--mode`, `--variant`, `--task`, `--gen_tokens` (all additive)

New args (existing flags unchanged; `--smoke` unchanged and still passes — its synthetic runs have
`is_correct=None` → amber beacon, no `answer_text` → label falls back to `pred_token_str`):
```python
p.add_argument("--mode", choices=["single","compare","collective","rescue"], default="compare")
p.add_argument("--variant", choices=["plain","simple","detailed"], default="plain")
p.add_argument("--task", default=None, help="rescue/single task: int index OR key (e.g. 'mult')")
p.add_argument("--gen_tokens", type=int, default=24, help="greedy answer cap (mult auto-bumps to 256 via Task)")
p.add_argument("--no-generate", dest="generate", action="store_false")  # default generate=True
```
`main()` dispatch:
- `--mode single|compare` → `instructions = VARIANTS[args.variant]` (or `--prompt` → a one-off
  plain `Task(key='custom', system=None, user=prompt, gold='', grade='', gen_tokens=args.gen_
  tokens)`), then `run_compare(..., single=(args.mode=='single' or args.single), generate=args.
  generate, gen_tokens=args.gen_tokens)`. `--single` and `--example` still honored. With
  `--variant plain` and no gold, beacons render amber unless `--generate` finds a gold via Task.
- `--mode collective` → `run_collective(tuple(args.models), variant=args.variant, ...,
  generate=args.generate, gen_tokens=args.gen_tokens)`.
- `--mode rescue` → `run_rescue(tuple(args.models), task=args.task, ...)`. When `task` resolves to
  `"mult"`, the Task's `gen_tokens=256` is used automatically.

The existing `min_frames` frame-clamp stays. `__init__.py` adds to `__all__`: `Task`, `VARIANTS`,
`EXAMPLES_SIMPLE`, `EXAMPLES_DETAILED`, `EXAMPLES_PLAIN`, `TASK_KEYS`, and (torch-free side)
`render_collective_gif`, `render_collective_pair_gif`, `render_rescue_gif`; behind the `_TORCH_OK`
guard: `build_collective`, `run_collective`, `CollectiveResult`, `build_rescue`, `run_rescue`,
`RescueResult`.

---

## 7. Notebook cell — RESCUE incl. multiplication

Add a code cell to `notebooks/residual_orrery_colab.ipynb` (the headline artifact):
```python
from residual_orrery.compare import run_rescue
from IPython.display import Image, display
# Headline: multiplication rescue on the 1.5B (simple FAILS -> detailed near-miss path turns).
res = run_rescue(tags=("1.5B",), task="mult", out_dir="out", total_frames=84,
                 device="cpu", dpi=110, fps=12)
for g in res.gif_paths:
    print(g); display(Image(filename=g))
# Optional: all tasks side-quest
res_all = run_rescue(tags=("1.5B",), task=None, out_dir="out", total_frames=72)
for g in res_all.gif_paths:
    print(g); display(Image(filename=g))
```

---

## 8. Invariants preserved (regression contract)

- Traced through-layer trajectory at the last prompt token is **unchanged**; generation only sets
  `answer_text`/`gold`/`grade_mode`/`is_correct` for the beacon. Traced forward keeps
  `use_cache=False` + hooks; generation is a separate post-hoc `.generate` (hooks already removed).
- All `_self_test` mechanistic identities run untouched on collect.
- `project`/`animate` stay torch-free; `--smoke` keeps passing (new fields default to neutral →
  amber beacon, `pred_token_str` label).
- `unembed_sphere == traj_sphere[-1]` assertion preserved (beacon sits on the terminal node).
- Beacon color decoupled from `pred_token_id` (the trajectory landmark) — color is ONLY from
  `is_correct`.
- Version-safety: every new draw primitive (diamond marker, dashed 3D link, boxed text,
  qualitative `tab10`/`tab20` cmap) verified on mpl 3.2.2; reuse `_get_cmap`/`_zoom_in`/`_save_gif`/
  `_first_tensor`. Generation under `torch.no_grad`, device-safe (cpu/cuda), via clean
  `GenerationConfig` + `attention_mask` verified on transformers 4.44.2 (Qwen2.5-0.5B).
- No `apply_chat_template` anywhere (jinja2 2.11.2) — MANUAL ChatML; system threaded via string
  formatting only. Old `.npz`/`.json` caches still load (`.get` defaults); `_cache_key` v2 bump
  prevents aliasing.
