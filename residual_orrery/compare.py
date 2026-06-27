"""compare.py — two-model driver: collect (cached) -> per-model PCA frame -> project
-> render side-by-side GIF.

Orchestration only; no new heavy deps. Never forces a shared PCA basis (different H).
``build_compare`` is pure data (no rendering) so it is fully unit-testable.
"""

import os
from dataclasses import dataclass, field

from .examples import (
    EXAMPLES,
    VARIANTS,
    EXAMPLES_SIMPLE,
    EXAMPLES_DETAILED,
    TASK_KEYS,
    resolve_task,
)
from .models import load_model
from .collect import collect_all_cached
from .project import fit_sphere_frame, project_run
from .animate import (
    render_compare_gif,
    render_single_gif,
    render_collective_gif,
    render_collective_pair_gif,
    render_rescue_gif,
)


@dataclass
class CompareResult:
    runs: dict  # tag -> list[RunCollection]
    frames: dict  # tag -> SphereFrame
    projected: dict  # tag -> ProjectedRun (chosen example)
    gif_paths: list = field(default_factory=list)


def build_compare(
    tags=("0.5B", "1.5B"),
    instructions=EXAMPLES,
    example_idx=0,
    topk=48,
    device="cpu",
    cache_dir="out/cache",
    enrich_pca=True,
    use_cache=True,
    generate=False,
    gen_tokens=512,
):
    """Load both bundles, collect (cached), fit a per-model PCA frame, project the
    chosen example. Returns a CompareResult with no rendering done. ``generate=True`` makes
    the beacons correctness-colored (per-task gold/grade/cap pulled when instructions are
    Tasks)."""
    runs, frames, projected = {}, {}, {}
    for tag in tags:
        bundle = load_model(tag, device=device)
        rc_list = collect_all_cached(
            bundle, instructions, topk=topk, cache_dir=cache_dir, use_cache=use_cache,
            generate=generate, gen_tokens=gen_tokens,
        )
        runs[tag] = rc_list
        chosen = rc_list[example_idx]
        fit_runs = rc_list if enrich_pca else [chosen]
        frame = fit_sphere_frame(fit_runs)
        frames[tag] = frame
        projected[tag] = project_run(frame, chosen)
        # free the model promptly; project/animate are torch-free from here.
        del bundle
    return CompareResult(runs=runs, frames=frames, projected=projected, gif_paths=[])


def run_compare(
    tags=("0.5B", "1.5B"),
    instructions=EXAMPLES,
    example_idx=0,
    out_dir="out",
    total_frames=60,
    topk=48,
    device="cpu",
    dpi=110,
    fps=12,
    use_slerp=True,
    keep_frames=False,
    enrich_pca=True,
    single=False,
    use_cache=True,
    generate=False,
    gen_tokens=512,
):
    """Full pipeline -> GIF(s) in out_dir. Returns CompareResult with gif_paths set."""
    os.makedirs(out_dir, exist_ok=True)
    cr = build_compare(
        tags=tags,
        instructions=instructions,
        example_idx=example_idx,
        topk=topk,
        device=device,
        cache_dir=os.path.join(out_dir, "cache"),
        enrich_pca=enrich_pca,
        use_cache=use_cache,
        generate=generate,
        gen_tokens=gen_tokens,
    )

    if single or len(tags) == 1:
        for tag in tags:
            out = os.path.join(out_dir, "single_ex%d_%s.gif" % (example_idx, tag))
            render_single_gif(
                cr.projected[tag],
                out=out,
                total_frames=total_frames,
                dpi=dpi,
                fps=fps,
                use_slerp=use_slerp,
                keep_frames=keep_frames,
            )
            cr.gif_paths.append(out)
        return cr

    a, b = cr.projected[tags[0]], cr.projected[tags[1]]
    out = os.path.join(
        out_dir, "compare_ex%d_%s_vs_%s.gif" % (example_idx, tags[0], tags[1])
    )
    render_compare_gif(
        a,
        b,
        out=out,
        total_frames=total_frames,
        dpi=dpi,
        fps=fps,
        use_slerp=use_slerp,
        keep_frames=keep_frames,
    )
    cr.gif_paths.append(out)
    return cr


# ----------------------------------------------------------------------------
# v2: collective (Feature 4) — ALL tasks of a variant on ONE shared sphere per model
# ----------------------------------------------------------------------------
@dataclass
class CollectiveResult:
    runs: dict          # tag -> list[RunCollection]
    frames: dict        # tag -> SphereFrame (ONE per model, fit over all tasks)
    projected: dict     # tag -> list[ProjectedRun]
    gif_paths: list = field(default_factory=list)


def build_collective(
    tags=("0.5B", "1.5B"),
    variant="plain",
    device="cpu",
    topk=48,
    cache_dir="out/cache",
    generate=True,
    gen_tokens=512,
    use_cache=True,
):
    """Per model: collect ALL tasks of VARIANTS[variant] (with generation+correctness), fit
    ONE shared frame over all tasks, project every task. Pure data; no rendering. Frees each
    bundle. Each ProjectedRun gets a ``_task_key`` for beacon labels."""
    instructions = VARIANTS[variant]
    runs, frames, projected = {}, {}, {}
    for tag in tags:
        bundle = load_model(tag, device=device)
        rc_list = collect_all_cached(
            bundle, instructions, topk=topk, cache_dir=cache_dir, use_cache=use_cache,
            generate=generate, gen_tokens=gen_tokens,
        )
        runs[tag] = rc_list
        frame = fit_sphere_frame(rc_list)        # ONE shared frame over all tasks
        frames[tag] = frame
        projs = []
        for t, rc in zip(instructions, rc_list):
            pj = project_run(frame, rc)
            pj._task_key = t.key
            projs.append(pj)
        projected[tag] = projs
        del bundle
    return CollectiveResult(runs=runs, frames=frames, projected=projected, gif_paths=[])


def run_collective(
    tags=("0.5B", "1.5B"),
    variant="plain",
    out_dir="out",
    total_frames=72,
    device="cpu",
    topk=48,
    dpi=110,
    fps=12,
    use_slerp=True,
    keep_frames=False,
    generate=True,
    gen_tokens=512,
    use_cache=True,
):
    """1 model -> single collective sphere GIF (render_collective_gif).
       2 models -> two collective spheres side by side (render_collective_pair_gif).
       Returns CollectiveResult with gif_paths set."""
    os.makedirs(out_dir, exist_ok=True)
    cr = build_collective(
        tags=tags, variant=variant, device=device, topk=topk,
        cache_dir=os.path.join(out_dir, "cache"),
        generate=generate, gen_tokens=gen_tokens, use_cache=use_cache,
    )
    if len(tags) == 1:
        tag = tags[0]
        projs = cr.projected[tag]
        out = os.path.join(out_dir, "collective_%s_%s.gif" % (variant, tag))
        render_collective_gif(
            projs, out=out, total_frames=total_frames, dpi=dpi, fps=fps,
            use_slerp=use_slerp, keep_frames=keep_frames,
            labels=[p._task_key for p in projs],
        )
        cr.gif_paths.append(out)
        return cr

    a, b = cr.projected[tags[0]], cr.projected[tags[1]]
    out = os.path.join(out_dir, "collective_%s_%s_vs_%s.gif" % (variant, tags[0], tags[1]))
    render_collective_pair_gif(
        a, b, out=out, total_frames=total_frames, dpi=dpi, fps=fps,
        use_slerp=use_slerp, keep_frames=keep_frames,
        labels_a=[p._task_key for p in a], labels_b=[p._task_key for p in b],
    )
    cr.gif_paths.append(out)
    return cr


# ----------------------------------------------------------------------------
# v2: rescue (Feature 5) — SIMPLE vs DETAILED, same task, same per-task shared frame
# ----------------------------------------------------------------------------
@dataclass
class RescueResult:
    runs: dict          # tag -> {"simple":[RC...], "detailed":[RC...]}
    frames: dict        # tag -> list[SphereFrame]  (one shared frame per task)
    projected: dict     # tag -> list[(proj_simple, proj_detailed)]  per task
    gif_paths: list = field(default_factory=list)


def build_rescue(
    tags=("0.5B",),
    task=None,
    device="cpu",
    topk=48,
    cache_dir="out/cache",
    use_cache=True,
):
    """Per model: collect EXAMPLES_SIMPLE and EXAMPLES_DETAILED (both generate=True; per-task
    gen_tokens from the Task, so mult=256). For each task (or just ``task``), fit ONE shared
    frame over its [simple_rc, detailed_rc] so both trajectories share a PCA basis, then
    project both. ``task`` accepts an int index OR a key ('mult'); None -> all tasks."""
    idx = resolve_task(task)
    sel = range(len(TASK_KEYS)) if idx is None else [idx]
    runs, frames, projected = {}, {}, {}
    for tag in tags:
        bundle = load_model(tag, device=device)
        rc_simple = collect_all_cached(
            bundle, EXAMPLES_SIMPLE, topk=topk, cache_dir=cache_dir, use_cache=use_cache,
            generate=True,
        )
        rc_detailed = collect_all_cached(
            bundle, EXAMPLES_DETAILED, topk=topk, cache_dir=cache_dir, use_cache=use_cache,
            generate=True,
        )
        runs[tag] = {"simple": rc_simple, "detailed": rc_detailed}
        fr_list, pj_list = [], []
        for i in sel:
            s_rc, d_rc = rc_simple[i], rc_detailed[i]
            frame = fit_sphere_frame([s_rc, d_rc])      # ONE shared frame per task
            ps = project_run(frame, s_rc)
            pd = project_run(frame, d_rc)
            ps._task_key = TASK_KEYS[i]
            pd._task_key = TASK_KEYS[i]
            fr_list.append(frame)
            pj_list.append((ps, pd))
        frames[tag] = fr_list
        projected[tag] = pj_list
        del bundle
    return RescueResult(runs=runs, frames=frames, projected=projected, gif_paths=[])


def run_rescue(
    tags=("0.5B",),
    task=None,
    out_dir="out",
    total_frames=72,
    device="cpu",
    topk=48,
    dpi=110,
    fps=12,
    use_slerp=True,
    keep_frames=False,
    use_cache=True,
):
    """Render the rescue overlay per resolved task (one GIF each; multiplication is the
    headline). Returns RescueResult with gif_paths set."""
    os.makedirs(out_dir, exist_ok=True)
    rr = build_rescue(
        tags=tags, task=task, device=device, topk=topk,
        cache_dir=os.path.join(out_dir, "cache"), use_cache=use_cache,
    )
    for tag in tags:
        for (ps, pd) in rr.projected[tag]:
            key = getattr(ps, "_task_key", "task")
            out = os.path.join(out_dir, "rescue_%s_%s.gif" % (key, tag))
            render_rescue_gif(
                ps, pd, out=out, total_frames=total_frames, dpi=dpi, fps=fps,
                use_slerp=use_slerp, keep_frames=keep_frames,
                task_label=key, model_tag=tag,
            )
            rr.gif_paths.append(out)
    return rr
