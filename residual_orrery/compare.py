"""compare.py — two-model driver: collect (cached) -> per-model PCA frame -> project
-> render side-by-side GIF.

Orchestration only; no new heavy deps. Never forces a shared PCA basis (different H).
``build_compare`` is pure data (no rendering) so it is fully unit-testable.
"""

import os
from dataclasses import dataclass, field

from .examples import EXAMPLES
from .models import load_model
from .collect import collect_all_cached
from .project import fit_sphere_frame, project_run
from .animate import render_compare_gif, render_single_gif


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
):
    """Load both bundles, collect (cached), fit a per-model PCA frame, project the
    chosen example. Returns a CompareResult with no rendering done."""
    runs, frames, projected = {}, {}, {}
    for tag in tags:
        bundle = load_model(tag, device=device)
        rc_list = collect_all_cached(
            bundle, instructions, topk=topk, cache_dir=cache_dir, use_cache=use_cache
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
