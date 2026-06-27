"""cli.py — argparse entry point. Also reachable via ``python -m residual_orrery``."""

import argparse
import os

from .examples import EXAMPLES


def build_parser():
    p = argparse.ArgumentParser(
        prog="residual_orrery",
        description="Render the residual orrery: 0.5B vs 1.5B twin spheres of writer-direction stars.",
    )
    p.add_argument(
        "--models", nargs="+", default=["0.5B", "1.5B"], choices=["0.5B", "1.5B"]
    )
    p.add_argument("--example", type=int, default=0, help="index into EXAMPLES")
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

    # resolve prompt / instructions
    instructions = list(EXAMPLES)
    example_idx = args.example
    if args.prompt is not None:
        instructions = [args.prompt]
        example_idx = 0

    # clamp frames so every node is visited (P up to 2*28+3=59, +hold 6)
    min_frames = 59 + 6
    frames = max(args.frames, min_frames) if args.frames < min_frames else args.frames

    from .compare import run_compare

    print("residual_orrery: models=%s example=%d topk=%d frames=%d dpi=%d device=%s"
          % (args.models, example_idx, args.topk, frames, args.dpi, args.device))
    print("  prompt: %r" % instructions[example_idx])

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
        single=args.single,
        use_cache=args.use_cache,
    )
    for tag in args.models:
        pj = cr.projected.get(tag)
        if pj is not None:
            print("  %s predicted token: %r" % (tag, pj.pred_token_str))
    for g in cr.gif_paths:
        print("  wrote GIF: %s" % g)
    return 0


if __name__ == "__main__":  # allow `python -m residual_orrery.cli`
    raise SystemExit(main())
