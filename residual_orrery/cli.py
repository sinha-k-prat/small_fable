"""cli.py — argparse entry point. Also reachable via ``python -m residual_orrery``."""

import argparse
import os

from .examples import EXAMPLES, VARIANTS, TASK_KEYS, Task, resolve_task


def build_parser():
    p = argparse.ArgumentParser(
        prog="residual_orrery",
        description="Render the residual orrery: 0.5B vs 1.5B twin spheres of writer-direction stars.",
    )
    p.add_argument(
        "--models", nargs="+", default=["0.5B", "1.5B"], choices=["0.5B", "1.5B"]
    )
    p.add_argument("--mode", choices=["single", "compare", "collective", "rescue"],
                   default="compare", help="render mode (default: compare)")
    p.add_argument("--variant", choices=["plain", "simple", "detailed"], default="plain",
                   help="instruction variant for single/compare/collective")
    p.add_argument("--task", default=None,
                   help="rescue/single task: int index OR key (e.g. 'mult')")
    p.add_argument("--gen_tokens", type=int, default=24,
                   help="greedy answer cap (mult auto-bumps to 256 via Task)")
    p.add_argument("--no-generate", dest="generate", action="store_false",
                   help="skip greedy generation (beacons render amber)")
    p.set_defaults(generate=True)
    p.add_argument("--example", type=int, default=0, help="index into EXAMPLES / variant tasks")
    p.add_argument("--prompt", default=None, help="override --example with a custom prompt")
    p.add_argument("--out", default="out")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--topk", type=int, default=48, help="top-K writer cols per layer (40..60)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--no-slerp", dest="use_slerp", action="store_false")
    p.add_argument("--no-enrich-pca", dest="enrich_pca", action="store_false")
    p.add_argument("--no-cache", dest="use_cache", action="store_false")
    p.add_argument("--single", action="store_true", help="one panel per model")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="torch-free self-check of project/animate on synthetic data, then exit",
    )
    return p


def _smoke():
    """Fast torch-free self-check: build synthetic runs, project, render a tiny GIF."""
    import tempfile
    import numpy as np
    from .project import fit_sphere_frame, project_run
    from .animate import render_compare_gif, slerp, build_schedule

    # synthetic RunCollection-shaped object (duck-typed for project.py)
    from .collect import RunCollection, TrajNode, NodeKind, node_count

    def make_run(tag, H, N, I, K):
        rng = np.random.RandomState(0 if tag == "0.5B" else 1)
        nodes = []
        nodes.append(TrajNode(NodeKind.EMBED, -1, rng.randn(H).astype(np.float32)))
        for L in range(N):
            nodes.append(TrajNode(NodeKind.ATTN, L, rng.randn(H).astype(np.float32)))
            nodes.append(
                TrajNode(NodeKind.MLP, L, rng.randn(H).astype(np.float32), rng.randn(I).astype(np.float32))
            )
        nodes.append(TrajNode(NodeKind.FINAL, -1, rng.randn(H).astype(np.float32)))
        ud = rng.randn(H).astype(np.float32)
        nodes.append(TrajNode(NodeKind.UNEMBED, 5, ud))
        assert len(nodes) == node_count(N)
        topk_idx = {L: np.arange(K) for L in range(N)}
        topk_a = {L: np.abs(rng.randn(K)).astype(np.float32) for L in range(N)}
        down_cols = {L: rng.randn(K, H).astype(np.float32) for L in range(N)}
        return RunCollection(
            tag=tag,
            instruction="smoke",
            input_ids=np.array([1, 2, 3], np.int64),
            last_pos=2,
            pred_token_id=5,
            pred_token_str="X",
            nodes=nodes,
            topk_idx=topk_idx,
            topk_a=topk_a,
            down_cols=down_cols,
            unembed_dir=ud,
            H=H,
            N=N,
            I=I,
            topk=K,
        )

    # slerp endpoints + unit invariants
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    assert np.allclose(slerp(p0, p1, 0.0), p0, atol=1e-6)
    assert np.allclose(slerp(p0, p1, 1.0), p1, atol=1e-6)
    assert abs(np.linalg.norm(slerp(p0, -p0, 0.5)) - 1.0) < 1e-6  # antipodal stays unit

    ra = make_run("0.5B", 32, 4, 64, 6)
    rb = make_run("1.5B", 48, 5, 80, 6)
    fa = fit_sphere_frame([ra])
    fb = fit_sphere_frame([rb])
    pa = project_run(fa, ra)
    pb = project_run(fb, rb)
    # sphere norms ~ 1
    assert np.allclose(np.linalg.norm(pa.traj_sphere, axis=1), 1.0, atol=1e-5)
    sch = build_schedule(pa.traj_sphere.shape[0], total_frames=10)
    assert sch.total_frames >= pa.traj_sphere.shape[0] - 1

    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "smoke.gif")
    render_compare_gif(pa, pb, out=out, total_frames=8, dpi=70, fps=8)
    import imageio

    rdr = imageio.get_reader(out)
    n = sum(1 for _ in rdr)
    assert n >= 1, "gif had no frames"
    print("SMOKE OK: torch-free project/animate pipeline works. gif frames=%d -> %s" % (n, out))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        return _smoke()

    os.makedirs(args.out, exist_ok=True)

    # clamp frames so every node is visited (P up to 2*28+3=59, +hold 6)
    min_frames = 59 + 6
    frames = max(args.frames, min_frames) if args.frames < min_frames else args.frames

    # ---- mode: collective ----
    if args.mode == "collective":
        from .compare import run_collective
        print("residual_orrery: mode=collective variant=%s models=%s frames=%d device=%s"
              % (args.variant, args.models, frames, args.device))
        cr = run_collective(
            tags=tuple(args.models), variant=args.variant, out_dir=args.out,
            total_frames=frames, device=args.device, topk=args.topk, dpi=args.dpi,
            fps=args.fps, use_slerp=args.use_slerp, keep_frames=args.keep_frames,
            generate=args.generate, gen_tokens=args.gen_tokens, use_cache=args.use_cache,
        )
        for g in cr.gif_paths:
            print("  wrote GIF: %s" % g)
        return 0

    # ---- mode: rescue ----
    if args.mode == "rescue":
        from .compare import run_rescue
        print("residual_orrery: mode=rescue task=%r models=%s frames=%d device=%s"
              % (args.task, args.models, frames, args.device))
        rr = run_rescue(
            tags=tuple(args.models), task=args.task, out_dir=args.out,
            total_frames=frames, device=args.device, topk=args.topk, dpi=args.dpi,
            fps=args.fps, use_slerp=args.use_slerp, keep_frames=args.keep_frames,
            use_cache=args.use_cache,
        )
        for tag in args.models:
            for (ps, pd) in rr.projected.get(tag, []):
                print("  %s %s: simple=%r(%s) detailed=%r(%s)" % (
                    tag, getattr(ps, "_task_key", "?"),
                    ps.answer_text, ps.is_correct, pd.answer_text, pd.is_correct))
        for g in rr.gif_paths:
            print("  wrote GIF: %s" % g)
        return 0

    # ---- mode: single | compare ----
    # instructions are the chosen variant's Task records (so gold/grade/cap thread through);
    # --prompt overrides with a one-off plain Task; --task selects which task index to render.
    instructions = list(VARIANTS[args.variant])
    example_idx = args.example
    if args.task is not None:
        example_idx = resolve_task(args.task)
    if args.prompt is not None:
        instructions = [Task(key="custom", system=None, user=args.prompt,
                             gold="", grade="", gen_tokens=args.gen_tokens)]
        example_idx = 0

    chosen = instructions[example_idx]
    single = (args.mode == "single") or args.single

    from .compare import run_compare

    print("residual_orrery: mode=%s variant=%s models=%s example=%d topk=%d frames=%d dpi=%d device=%s"
          % (args.mode, args.variant, args.models, example_idx, args.topk, frames, args.dpi, args.device))
    print("  prompt: %r" % chosen.user)

    cr = run_compare(
        tags=tuple(args.models),
        instructions=instructions,
        example_idx=example_idx,
        out_dir=args.out,
        total_frames=frames,
        topk=args.topk,
        device=args.device,
        dpi=args.dpi,
        fps=args.fps,
        use_slerp=args.use_slerp,
        keep_frames=args.keep_frames,
        enrich_pca=args.enrich_pca,
        single=single,
        use_cache=args.use_cache,
        generate=args.generate,
        gen_tokens=args.gen_tokens,
    )
    for tag in args.models:
        pj = cr.projected.get(tag)
        if pj is not None:
            extra = ""
            if getattr(pj, "answer_text", ""):
                extra = "  answer=%r correct=%s" % (pj.answer_text, pj.is_correct)
            print("  %s predicted token: %r%s" % (tag, pj.pred_token_str, extra))
    for g in cr.gif_paths:
        print("  wrote GIF: %s" % g)
    return 0


if __name__ == "__main__":  # allow `python -m residual_orrery.cli`
    raise SystemExit(main())
