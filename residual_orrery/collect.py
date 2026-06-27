"""collect.py — forward hooks -> residual trajectory + per-layer firing `a`
+ writer columns + unembed rows; mechanistic self-tests; .npz disk cache.

torch + numpy; depends on models, examples.

Verified shapes / facts (transformers 4.44.2 AND modern ~4.55, eager attn, fp32, CPU):
  * forward hook signature: (module, input, output)  — 3 positional args
  * forward_pre hook signature: (module, input)       — 2 positional args
  * self_attn forward output is tuple-or-tensor across versions  -> coerce via _first_tensor
        - 4.44 : 3-tuple (attn_output, attn_weights, past_key_value) -> out[0]
        - ~4.55: 2-tuple (attn_output, attn_weights) (or a bare tensor on some paths)
  * decoder layer (layers[L]) forward output is tuple-or-tensor  -> coerce via _first_tensor
        - 4.44 : tuple (hidden_states, ...) -> out[0]
        - ~4.55: bare tensor [B,T,H] -> over-indexing out[0][0,pos,:] would IndexError
  * mlp forward output is a plain tensor in BOTH versions  -> use out
  * embed_tokens / model.norm forward output is a plain tensor in BOTH versions -> use out
  * down_proj pre-hook input == silu(gate(x))*up(x) == `a`, shape [I]
  * W_down @ a == mlp_write  (down_proj has no bias)  -> reproduces the MLP write
  * residual is literally prev + delta in the forward, so trajectory recon is exact.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# torch is imported LAZILY inside the collection functions only. The dataclasses
# (RunCollection/TrajNode/NodeKind), project.py, animate.py and the --smoke path are
# torch-free, so they import + run on a machine without torch (e.g. a render-only env).

from .examples import EXAMPLES, build_input_ids


class NodeKind(str, Enum):
    EMBED = "embed"
    ATTN = "attn"
    MLP = "mlp"
    FINAL = "final"
    UNEMBED = "unembed"


@dataclass
class TrajNode:
    kind: NodeKind
    layer: int  # -1 for embed/final; pred_token_id for unembed
    h: np.ndarray  # [H] float32 — cumulative residual point (or unembed direction)
    a: object = None  # [I] float32, only for kind==MLP (down_proj input)


@dataclass
class RunCollection:
    tag: str
    instruction: str
    input_ids: np.ndarray  # [T] int64
    last_pos: int
    pred_token_id: int
    pred_token_str: str
    nodes: list  # ordered TrajNode, length P = 2N+3
    topk_idx: dict  # layer -> [K] int
    topk_a: dict  # layer -> [K] float32 (|a_j| values, for glow size)
    down_cols: dict  # layer -> [K, H] float32 (writer dirs == down_proj cols, transposed)
    unembed_dir: np.ndarray  # [H] float32 (unembedding row of pred token)
    H: int
    N: int
    I: int
    topk: int
    # ---- v2 additive: full generated answer + correctness for the terminal beacon ----
    answer_text: str = ""        # full greedy continuation (decoded, specials skipped)
    gold: str = ""               # gold string this run was judged against ("" if unknown)
    grade_mode: str = ""         # "digits"|"substr"|"equal"|"" (correctness normalizer)
    is_correct: object = None    # True / False / None(unknown) — bool or None


# ----------------------------------------------------------------------------
# node index <-> (kind, layer) map.  P = 2N + 3.
#   node 0          -> (EMBED, -1)
#   node 2L+1       -> (ATTN, L)
#   node 2L+2       -> (MLP, L)
#   node 2N+1       -> (FINAL, -1)
#   node 2N+2       -> (UNEMBED, pred_token_id)
# ----------------------------------------------------------------------------
def node_count(n_layers):
    return 2 * n_layers + 3


def _np(t):
    """detach -> float32 -> cpu -> numpy, 1-D copy."""
    import torch
    return t.detach().to(torch.float32).cpu().numpy().copy()


def _first_tensor(x):
    """Coerce a module forward output that may be a tuple (older transformers,
    e.g. 4.44 decoder layer / self_attn) or a plain tensor (modern ~4.55, where
    the Qwen2 decoder layer was refactored to ``return hidden_states``). Returns
    the hidden-states tensor either way. No-op (returns x) when already a tensor."""
    return x[0] if isinstance(x, tuple) else x


# ----------------------------------------------------------------------------
# v2: greedy multi-token answer + normalized correctness (additive, opt-in)
# ----------------------------------------------------------------------------
def _greedy_answer(bundle, ids, max_new_tokens):
    """Greedy multi-token continuation of `ids` ([1,T] on bundle.device) -> decoded NEW
    tokens only, specials skipped, stripped. torch.no_grad, device-safe. Returns '' if
    max_new_tokens<=0. Version-safe: CLEAN GenerationConfig (the cached Qwen config sets
    temperature/top_p/top_k, which WARN under do_sample=False) + explicit attention_mask;
    ChatML stop via eos_token_id=[<|im_end|>, eos]."""
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


def _norm_alnum(s):
    """lowercase, keep [a-z0-9] only (drops spaces/punct)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _digits_only_last(s):
    """LAST contiguous integer run in the string, digits only. Handles
    'FINAL ANSWER: 8,293,662.' and the scaffold's many earlier numbers by taking the LAST
    run -> the committed answer. '' if no digits."""
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


def collect_run(bundle, instruction, topk=48, self_test=True,
                generate=False, gen_tokens=24, gold="", grade_mode=""):
    """Run one forward pass, capture the trajectory for the last prompt token.

    Returns a RunCollection. All capture happens at column ``pos`` (last prompt token).

    ``instruction`` may be a str OR a ``Task``. If a Task: system/gold/grade/gen_tokens come
    from it (and override the kwargs). ``generate=False`` (default) => identical to v1 (no
    ``.generate`` call). The traced through-layer trajectory is byte-for-byte unchanged;
    generation is a separate post-hoc call that only sets answer_text/correctness.
    """
    import torch
    from .examples import Task
    if isinstance(instruction, Task):
        gold = instruction.gold
        grade_mode = instruction.grade
        if generate:
            gen_tokens = instruction.gen_tokens
    ids = build_input_ids(bundle, instruction)  # [1, T]; accepts str OR Task
    T = ids.shape[1]
    pos = T - 1
    N = bundle.n_layers
    H = bundle.hidden
    I = bundle.intermediate

    store = {
        "embed": None,  # [H]
        "h_in": {},  # L -> [H]   (layer input, forward_pre on layers[L])
        "attn": {},  # L -> [H]   (attn delta, self_attn forward out, tuple-or-tensor coerced)
        "mlp": {},  # L -> [H]    (mlp delta, mlp forward out)
        "a": {},  # L -> [I]      (down_proj pre-hook input)
        "layer_out": {},  # L -> [H] (layers[L] forward out, tuple-or-tensor coerced) — cross-check only
        "norm": None,  # [H]
    }
    handles = []

    def mk_embed_hook():
        def hook(module, inp, out):  # forward: (module, input, output)
            store["embed"] = _np(out[0, pos, :])  # [H]

        return hook

    def mk_layer_pre_hook(L):
        def hook(module, inp):  # forward_pre: (module, input)
            store["h_in"][L] = _np(inp[0][0, pos, :])  # [H]

        return hook

    def mk_attn_hook(L):
        def hook(module, inp, out):  # self_attn out: tuple (4.44/4.55) or bare tensor -> coerce
            t = _first_tensor(out)  # [B,T,H] in both versions
            store["attn"][L] = _np(t[0, pos, :])  # [H]

        return hook

    def mk_mlp_hook(L):
        def hook(module, inp, out):  # mlp out is a plain tensor
            store["mlp"][L] = _np(out[0, pos, :])  # [H]

        return hook

    def mk_downproj_pre_hook(L):
        def hook(module, inp):  # forward_pre on down_proj: input == a
            store["a"][L] = _np(inp[0][0, pos, :])  # [I]

        return hook

    def mk_layerout_hook(L):
        def hook(module, inp, out):  # layers[L] out: tuple (4.44) or bare tensor (~4.55) -> coerce
            t = _first_tensor(out)  # [B,T,H] in both versions
            store["layer_out"][L] = _np(t[0, pos, :])  # [H]

        return hook

    def mk_norm_hook():
        def hook(module, inp, out):  # model.norm out is a plain tensor
            store["norm"] = _np(out[0, pos, :])  # [H]

        return hook

    try:
        handles.append(bundle.embed().register_forward_hook(mk_embed_hook()))
        handles.append(bundle.final_norm().register_forward_hook(mk_norm_hook()))
        for L in range(N):
            lyr = bundle.layer(L)
            handles.append(lyr.register_forward_pre_hook(mk_layer_pre_hook(L)))
            handles.append(lyr.register_forward_hook(mk_layerout_hook(L)))
            handles.append(lyr.self_attn.register_forward_hook(mk_attn_hook(L)))
            handles.append(lyr.mlp.register_forward_hook(mk_mlp_hook(L)))
            handles.append(
                bundle.down_proj(L).register_forward_pre_hook(mk_downproj_pre_hook(L))
            )

        with torch.no_grad():
            out = bundle.model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()  # ALWAYS remove

    logits = out.logits  # [1, T, V]
    pred_id = int(logits[0, pos].argmax().item())
    pred_str = bundle.tokenizer.decode([pred_id])

    # ---- shape sanity on captured tensors ----
    assert store["embed"].shape == (H,), store["embed"].shape
    assert store["norm"].shape == (H,), store["norm"].shape
    for L in range(N):
        assert store["h_in"][L].shape == (H,)
        assert store["attn"][L].shape == (H,)
        assert store["mlp"][L].shape == (H,)
        assert store["a"][L].shape == (I,), (store["a"][L].shape, (I,))
        assert store["layer_out"][L].shape == (H,)

    # ---- trajectory reconstruction (additive, exact) ----
    nodes = []
    nodes.append(TrajNode(NodeKind.EMBED, -1, store["embed"]))
    h = store["embed"]
    for L in range(N):
        h_attn = store["h_in"][L] + store["attn"][L]  # (ATTN, L)
        nodes.append(TrajNode(NodeKind.ATTN, L, h_attn))
        h_mlp = h_attn + store["mlp"][L]  # (MLP, L)
        nodes.append(TrajNode(NodeKind.MLP, L, h_mlp, a=store["a"][L]))
        h = h_mlp
    nodes.append(TrajNode(NodeKind.FINAL, -1, store["norm"]))
    unembed_dir = bundle.unembed_rows([pred_id])[0]  # [H]
    nodes.append(TrajNode(NodeKind.UNEMBED, pred_id, unembed_dir))

    assert len(nodes) == node_count(N), (len(nodes), node_count(N))

    # ---- top-K writer selection per layer (by |a_j| for THIS token) ----
    K = int(min(topk, I))
    topk_idx, topk_a, down_cols = {}, {}, {}
    for L in range(N):
        a = store["a"][L]  # [I]
        absA = np.abs(a)
        idx = np.argpartition(absA, -K)[-K:]
        idx = idx[np.argsort(absA[idx])[::-1]]  # sort desc by |a|
        idx = idx.astype(np.int64)
        topk_idx[L] = idx
        topk_a[L] = absA[idx].astype(np.float32)  # [K]
        cols = bundle.down_proj_columns(L, idx)  # [K, H]
        assert cols.shape == (K, H), cols.shape
        down_cols[L] = cols

    # Store a JSON-serializable instruction string (Task -> its user prompt) so the cache
    # sidecar stays plain-JSON; the trajectory/answer are what matter downstream.
    instr_str = instruction.user if isinstance(instruction, Task) else instruction
    rc = RunCollection(
        tag=bundle.tag,
        instruction=instr_str,
        input_ids=ids[0].detach().cpu().numpy().astype(np.int64),
        last_pos=pos,
        pred_token_id=pred_id,
        pred_token_str=pred_str,
        nodes=nodes,
        topk_idx=topk_idx,
        topk_a=topk_a,
        down_cols=down_cols,
        unembed_dir=unembed_dir.astype(np.float32),
        H=H,
        N=N,
        I=I,
        topk=K,
    )

    if self_test:
        _self_test(bundle, rc, out, store)

    # ---- v2: greedy answer + correctness (only when requested) ----
    # Reuse the SAME ids (the traced forward's hooks were removed in `finally`), so the
    # system message threads through identically. The pred_token_id / unembed_dir / nodes
    # path above is byte-for-byte unchanged.
    if generate and gen_tokens and gen_tokens > 0:
        rc.answer_text = _greedy_answer(bundle, ids, gen_tokens)
        rc.gold = gold or ""
        rc.grade_mode = grade_mode or ""
        rc.is_correct = judge(rc.answer_text, rc.gold, rc.grade_mode)
    return rc


def _self_test(bundle, rc, out, store):
    """Mechanistic asserts (§9). Cheap: a handful of vector norms."""
    import torch
    N, H, I = rc.N, rc.H, rc.I
    pos = rc.last_pos
    # fp32 CPU; spec uses 1e-4, loosen slightly for safety. Under reduced precision
    # (bf16/fp16, e.g. a bf16 cache on CUDA) the additive identities are far looser,
    # so gate the tolerance on the bundle dtype to avoid spurious self-test failures.
    tol = 1e-3 if bundle.dtype == torch.float32 else 2e-1

    # 1. MLP-write identity: ||W_down[L] @ a[L] - Δmlp[L]||inf < tol
    for L in range(N):
        W = bundle.down_proj(L).weight.detach().to(torch.float32).cpu().numpy()  # [H,I]
        a = store["a"][L]  # [I]
        recon = W @ a  # [H]
        err = np.max(np.abs(recon - store["mlp"][L]))
        assert err < tol, ("mlp-write identity", L, err)

    # 2. Layer-output cross-check: ||h[2L+2] - layer_out[L]||inf < tol
    for L in range(N):
        h_mlp = rc.nodes[2 * L + 2].h
        err = np.max(np.abs(h_mlp - store["layer_out"][L]))
        assert err < tol, ("layer-output xcheck", L, err)

    # 3. Residual continuity: ||h[2L+2] - h_in[L+1]||inf < tol  for L < N-1
    for L in range(N - 1):
        h_mlp = rc.nodes[2 * L + 2].h
        err = np.max(np.abs(h_mlp - store["h_in"][L + 1]))
        assert err < tol, ("residual continuity", L, err)

    # 4. Tied-unembed prediction: argmax(norm_row @ embed.weight.T) == logits argmax
    norm_row = torch.as_tensor(store["norm"])  # [H]
    Wt = bundle.embed().weight.detach().to(torch.float32).cpu()  # [V, H]
    pred_from_norm = int((Wt @ norm_row).argmax().item())
    assert pred_from_norm == rc.pred_token_id, (
        "tied-unembed argmax",
        pred_from_norm,
        rc.pred_token_id,
    )

    # 5. Shape gate
    assert bundle.down_proj(0).weight.shape == (H, I)
    for L in range(N):
        assert rc.down_cols[L].shape == (rc.topk, H)
        assert store["a"][L].shape == (I,)
    assert len(rc.nodes) == node_count(N)


def collect_all(bundle, instructions=EXAMPLES, topk=48, self_test=True,
                generate=False, gen_tokens=24):
    """Collect a RunCollection for each instruction (in order). ``generate=False`` => legacy.
    When instructions are Tasks, per-task gold/grade/gen_tokens are pulled inside collect_run;
    a scalar ``gen_tokens`` is the fallback cap for non-Task / plain instructions."""
    return [collect_run(bundle, ins, topk=topk, self_test=self_test,
                        generate=generate, gen_tokens=gen_tokens) for ins in instructions]


# ----------------------------------------------------------------------------
# Disk cache: flat .npz arrays + .json sidecar (no pickle of custom classes).
# ----------------------------------------------------------------------------
def _instr_repr(ins):
    """Stable repr for an instruction in the cache key. A Task contributes its
    system+user+gold+grade+gen so SIMPLE vs DETAILED never alias."""
    from .examples import Task
    if isinstance(ins, Task):
        return repr((ins.key, ins.system, ins.user, ins.gold, ins.grade, ins.gen_tokens))
    return repr(ins)


def _cache_key(bundle, instructions, topk, generate=False, gen_tokens=24):
    import transformers

    payload = "|".join(
        [
            "v2",
            bundle.tag,
            transformers.__version__,
            str(int(topk)),
            str(bool(generate)),
            str(int(gen_tokens)),
            repr(tuple(_instr_repr(i) for i in instructions)),
            str(getattr(bundle.tokenizer, "name_or_path", "")),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _save_runs(runs, npz_path, json_path):
    arrays = {}
    meta = {"n_runs": len(runs), "runs": []}
    for e, rc in enumerate(runs):
        for n, nd in enumerate(rc.nodes):
            arrays[f"ex{e}/node{n}/h"] = nd.h.astype(np.float32)
            if nd.a is not None:
                arrays[f"ex{e}/node{n}/a"] = nd.a.astype(np.float32)
        for L in rc.down_cols:
            arrays[f"ex{e}/L{L}/cols"] = rc.down_cols[L].astype(np.float32)
            arrays[f"ex{e}/L{L}/idx"] = rc.topk_idx[L].astype(np.int64)
            arrays[f"ex{e}/L{L}/a"] = rc.topk_a[L].astype(np.float32)
        arrays[f"ex{e}/unembed_dir"] = rc.unembed_dir.astype(np.float32)
        arrays[f"ex{e}/input_ids"] = rc.input_ids.astype(np.int64)
        node_meta = [(nd.kind.value, int(nd.layer)) for nd in rc.nodes]
        meta["runs"].append(
            {
                "tag": rc.tag,
                "instruction": rc.instruction,
                "last_pos": rc.last_pos,
                "pred_token_id": rc.pred_token_id,
                "pred_token_str": rc.pred_token_str,
                "node_meta": node_meta,
                "layers": sorted(int(L) for L in rc.down_cols),
                "H": rc.H,
                "N": rc.N,
                "I": rc.I,
                "topk": rc.topk,
                "answer_text": rc.answer_text,
                "gold": rc.gold,
                "grade_mode": rc.grade_mode,
                "is_correct": (None if rc.is_correct is None else bool(rc.is_correct)),
            }
        )
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **arrays)
    with open(json_path, "w") as f:
        json.dump(meta, f)


def _load_runs(npz_path, json_path):
    with open(json_path) as f:
        meta = json.load(f)
    npz = np.load(npz_path, allow_pickle=False)
    runs = []
    for e, rm in enumerate(meta["runs"]):
        nodes = []
        for n, (kind, layer) in enumerate(rm["node_meta"]):
            h = npz[f"ex{e}/node{n}/h"]
            akey = f"ex{e}/node{n}/a"
            a = npz[akey] if akey in npz.files else None
            nodes.append(TrajNode(NodeKind(kind), int(layer), h, a))
        topk_idx, topk_a, down_cols = {}, {}, {}
        for L in rm["layers"]:
            down_cols[int(L)] = npz[f"ex{e}/L{L}/cols"]
            topk_idx[int(L)] = npz[f"ex{e}/L{L}/idx"]
            topk_a[int(L)] = npz[f"ex{e}/L{L}/a"]
        runs.append(
            RunCollection(
                tag=rm["tag"],
                instruction=rm["instruction"],
                input_ids=npz[f"ex{e}/input_ids"],
                last_pos=rm["last_pos"],
                pred_token_id=rm["pred_token_id"],
                pred_token_str=rm["pred_token_str"],
                nodes=nodes,
                topk_idx=topk_idx,
                topk_a=topk_a,
                down_cols=down_cols,
                unembed_dir=npz[f"ex{e}/unembed_dir"],
                H=rm["H"],
                N=rm["N"],
                I=rm["I"],
                topk=rm["topk"],
                answer_text=rm.get("answer_text", ""),
                gold=rm.get("gold", ""),
                grade_mode=rm.get("grade_mode", ""),
                is_correct=rm.get("is_correct", None),
            )
        )
    return runs


def collect_all_cached(
    bundle, instructions=EXAMPLES, topk=48, cache_dir="out/cache", use_cache=True,
    generate=False, gen_tokens=24,
):
    """Cached variant of collect_all. Miss -> collect_all -> save; hit -> load .npz/.json.
    ``generate``/``gen_tokens`` thread through into the cache key (so a generated run never
    aliases a v1 cache, and SIMPLE vs DETAILED vs different caps get distinct files)."""
    key = _cache_key(bundle, instructions, topk, generate=generate, gen_tokens=gen_tokens)
    npz_path = os.path.join(cache_dir, key + ".npz")
    json_path = os.path.join(cache_dir, key + ".json")
    if use_cache and os.path.exists(npz_path) and os.path.exists(json_path):
        try:
            return _load_runs(npz_path, json_path)
        except Exception:
            pass  # fall through to recompute on any corruption
    runs = collect_all(bundle, instructions, topk=topk,
                       generate=generate, gen_tokens=gen_tokens)
    _save_runs(runs, npz_path, json_path)
    return runs
