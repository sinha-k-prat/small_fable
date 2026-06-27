"""animate.py — slerp hop + per-layer down_proj-column light-up + side-by-side GIF.

numpy + matplotlib + imageio ONLY — NO torch.

Headless / mpl-3.2 / imageio-2.9 discipline (must follow exactly):
  * matplotlib.use("Agg") BEFORE importing pyplot.
  * 3D via ``from mpl_toolkits.mplot3d import Axes3D`` (registers '3d') +
    fig.add_subplot(..., projection='3d').
  * NO set_box_aspect (absent in 3.2); NO set_aspect('equal') on 3D (raises) ->
    equal cube via equal limits.
  * NO constrained_layout; use fig.subplots_adjust.
  * cm.get_cmap(...) (NOT plt.colormaps[...]); ax.text2D for HUD; depthshade=False.
  * imageio 2.9.0: per-frame savefig -> imread -> mimsave(uri, frames,
    duration=1/fps, loop=0). NO FuncAnimation / FFMpegWriter / ffmpeg. Reuse one figure.
"""

import io
import os
from dataclasses import dataclass, field

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402  (fallback path only)
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers '3d')

import imageio  # noqa: E402


def _get_cmap(name):
    """Version-safe colormap accessor. matplotlib>=3.6 exposes mpl.colormaps[name];
    cm.get_cmap was REMOVED in 3.9. Old local mpl 3.2 has neither colormaps[] nor a
    removal, so it falls through to cm.get_cmap. Works on 3.2 AND modern Colab 3.10."""
    try:
        return matplotlib.colormaps[name]  # mpl >= 3.6
    except (AttributeError, KeyError):
        return cm.get_cmap(name)  # mpl < 3.6


def _save_gif(out, frames, fps):
    """Version-safe GIF write. imageio v2 takes duration as seconds/frame; v3 changed
    the kwargs, so fall back progressively to the most basic mimsave call."""
    try:
        imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
    except TypeError:
        try:
            imageio.mimsave(out, frames, duration=1.0 / fps)
        except TypeError:
            imageio.mimsave(out, frames)


# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
@dataclass
class GlowStyle:
    bg: str = "black"
    sphere_wire: tuple = (1, 1, 1, 0.08)
    static_rgba: tuple = (0.55, 0.60, 0.72, 0.18)
    static_s: float = 6.0
    cmap_name: str = "plasma"
    s_min: float = 8.0
    s_max: float = 130.0
    marker_rgba: tuple = (1.0, 1.0, 1.0, 1.0)
    marker_s: float = 90.0
    tail_len: int = 6
    tail_alpha0: float = 0.5
    target_rgba: tuple = (1.0, 0.84, 0.0, 1.0)
    elev: float = 18.0
    azim0: float = -60.0
    wire_stride: int = 18


STYLE = GlowStyle()


# ----------------------------------------------------------------------------
# slerp on S^2 (great-circle)
# ----------------------------------------------------------------------------
def _antipodal_path(p0, p1, t, eps=1e-7):
    """Half great-circle about an axis orthogonal to p0 (p0, p1 ~ antipodal)."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(p0, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ortho = ref - np.dot(ref, p0) * p0
    ortho = ortho / max(np.linalg.norm(ortho), eps)
    w = np.pi * t
    return np.cos(w) * p0 + np.sin(w) * ortho


def slerp(p0, p1, t, eps=1e-7):
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    d = np.clip(np.dot(p0, p1), -1.0, 1.0)
    w = np.arccos(d)
    if w < eps:  # coincident -> lerp + renorm
        v = (1 - t) * p0 + t * p1
    elif w > np.pi - eps:  # antipodal -> half great-circle about an orthogonal axis
        v = _antipodal_path(p0, p1, t, eps)
    else:
        s = np.sin(w)
        v = (np.sin((1 - t) * w) / s) * p0 + (np.sin(t * w) / s) * p1
    return v / max(np.linalg.norm(v), eps)


def _lerp_renorm(p0, p1, t, eps=1e-7):
    v = (1 - t) * np.asarray(p0) + t * np.asarray(p1)
    return v / max(np.linalg.norm(v), eps)


# ----------------------------------------------------------------------------
# frame schedule
# ----------------------------------------------------------------------------
@dataclass
class HopSchedule:
    total_frames: int
    seg_of_frame: np.ndarray  # [F] int   segment (hop) index
    t_of_frame: np.ndarray  # [F] float in [0,1], eased
    node_of_frame: np.ndarray  # [F] int  active source node


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def build_schedule(P, total_frames=60, ease="smoothstep", hold_end=6):
    """Distribute frames across the P-1 segments; pin the last `hold_end` at the end node."""
    segments = max(P - 1, 1)
    hold_end = int(min(hold_end, max(total_frames - segments, 0)))
    moving = total_frames - hold_end
    if moving < segments:  # ensure every node visited
        moving = segments
        total_frames = moving + hold_end
    base = moving // segments
    rem = moving - base * segments
    counts = np.full(segments, base, dtype=int)
    counts[:rem] += 1  # spread remainder to the first segments

    seg_of_frame = []
    t_of_frame = []
    node_of_frame = []
    for s in range(segments):
        c = counts[s]
        for k in range(c):
            # t goes (0 .. <1] across the segment so frame lands on the next node only
            # at the very start of the following segment.
            frac = (k + 1) / c if c > 0 else 1.0
            seg_of_frame.append(s)
            t = _smoothstep(frac) if ease == "smoothstep" else frac
            t_of_frame.append(t)
            node_of_frame.append(s)
    # hold frames pinned at the final node
    for _ in range(hold_end):
        seg_of_frame.append(segments - 1)
        t_of_frame.append(1.0)
        node_of_frame.append(segments)  # == P-1, the last node
    return HopSchedule(
        total_frames=len(seg_of_frame),
        seg_of_frame=np.asarray(seg_of_frame, int),
        t_of_frame=np.asarray(t_of_frame, float),
        node_of_frame=np.asarray(node_of_frame, int),
    )


# ----------------------------------------------------------------------------
# per-frame state
# ----------------------------------------------------------------------------
@dataclass
class FrameState:
    marker_xyz: np.ndarray
    tail: list
    active_layer: int
    active_kind: str
    glow_norm: object  # [K] for active layer, else None
    glow_env: float
    target_glow: float
    azim: float
    hud: str
    global_hud: str


def _minmax(x, eps=1e-8):
    x = np.asarray(x, np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _precompute_states(proj, sched, total_frames, use_slerp, orbit_turns, style, hold_end=6):
    """Build a FrameState list for one panel."""
    traj = proj.traj_sphere  # [P, 3]
    P = traj.shape[0]
    states = []
    tail_positions = []
    for i in range(sched.total_frames):
        s = int(sched.seg_of_frame[i])
        t = float(sched.t_of_frame[i])
        p0 = traj[s]
        p1 = traj[min(s + 1, P - 1)]
        if use_slerp:
            xyz = slerp(p0, p1, t)
        else:
            xyz = _lerp_renorm(p0, p1, t)

        # active node: the source node of the current segment (where the hop is "on").
        active_node = int(sched.node_of_frame[i])
        active_node = min(active_node, P - 1)
        kind, layer = proj.node_kinds[active_node]

        glow_norm = None
        if kind in ("attn", "mlp") and layer in proj.stars_a:
            glow_norm = _minmax(proj.stars_a[layer])  # [K] in [0,1]
        # triangular pulse over the active segment
        glow_env = 1.0 - abs(2.0 * t - 1.0)

        # target glow ramps during the hold tail at the very end
        is_hold = i >= (sched.total_frames - hold_end)
        if is_hold:
            denom = max(hold_end - 1, 1)
            ramp = (i - (sched.total_frames - hold_end)) / denom
            target_glow = float(np.clip(ramp, 0.0, 1.0))
        else:
            target_glow = 0.0

        azim = style.azim0 + 360.0 * orbit_turns * (i / max(sched.total_frames - 1, 1))

        tail_positions.append(xyz)
        tail = [np.asarray(p) for p in tail_positions[-style.tail_len :]]

        hud = _panel_hud(proj, kind, layer)
        states.append(
            FrameState(
                marker_xyz=np.asarray(xyz),
                tail=tail,
                active_layer=int(layer),
                active_kind=str(kind),
                glow_norm=glow_norm,
                glow_env=float(glow_env),
                target_glow=target_glow,
                azim=float(azim),
                hud=hud,
                global_hud="",
            )
        )
    return states


def _panel_hud(proj, kind, layer):
    if kind == "embed":
        return "embed"
    if kind == "final":
        return "final norm"
    if kind == "unembed":
        return "-> %r" % proj.pred_token_str
    return "%s  L%d" % (kind.upper(), layer)


# ----------------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------------
def _draw_wire_sphere(ax, style):
    n = style.wire_stride
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, color=style.sphere_wire)


def _draw_static_stars(ax, proj, style):
    pts = []
    for L in sorted(proj.stars_sphere):
        pts.append(proj.stars_sphere[L])
    if not pts:
        return
    P = np.concatenate(pts, axis=0)  # [N*K, 3]
    ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        s=style.static_s,
        c=[style.static_rgba],
        depthshade=False,
        edgecolors="none",
    )


def _draw_active_glow(ax, proj, fstate, style):
    if fstate.glow_norm is None:
        return
    L = fstate.active_layer
    if L not in proj.stars_sphere:
        return
    P = proj.stars_sphere[L]  # [K, 3]
    nrm = np.asarray(fstate.glow_norm)  # [K]
    env = fstate.glow_env
    sizes = (style.s_min + (style.s_max - style.s_min) * nrm) * (0.4 + 0.6 * env)
    cmap = _get_cmap(style.cmap_name)
    rgba = cmap(nrm)  # [K, 4]
    rgba[:, 3] = (0.35 + 0.65 * nrm) * (0.3 + 0.7 * env)  # alpha 0.35->1.0, pulsed
    ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        s=sizes,
        c=rgba,
        depthshade=False,
        edgecolors="none",
    )


def _draw_unembed(ax, proj, fstate, style):
    t = proj.unembed_sphere  # [3]
    g = fstate.target_glow
    base_alpha = 0.35 + 0.65 * g
    rgba = list(style.target_rgba[:3]) + [base_alpha]
    size = 60 + 140 * g
    ax.scatter(
        [t[0]], [t[1]], [t[2]], s=size, c=[rgba], depthshade=False, edgecolors="none"
    )
    if g > 0:  # expanding halo + label during the final hold
        halo = list(style.target_rgba[:3]) + [0.18 * g]
        ax.scatter(
            [t[0]],
            [t[1]],
            [t[2]],
            s=size * (2.0 + 2.0 * g),
            c=[halo],
            depthshade=False,
            edgecolors="none",
        )
        ax.text(
            t[0], t[1], t[2], "  %s" % proj.pred_token_str, color="gold", fontsize=10
        )


def _draw_marker_and_tail(ax, fstate, style):
    tail = fstate.tail
    n = len(tail)
    for k, p in enumerate(tail[:-1]):
        frac = (k + 1) / max(n, 1)
        alpha = style.tail_alpha0 * frac
        ax.scatter(
            [p[0]],
            [p[1]],
            [p[2]],
            s=style.marker_s * 0.4 * frac,
            c=[(1.0, 1.0, 1.0, alpha)],
            depthshade=False,
            edgecolors="none",
        )
    m = fstate.marker_xyz
    ax.scatter(
        [m[0]],
        [m[1]],
        [m[2]],
        s=style.marker_s,
        c=[style.marker_rgba],
        depthshade=False,
        edgecolors="none",
    )


def _blacken_3d_axes(ax, style):
    """mpl 3.2: the 3D panes default to opaque white and hide the black figure facecolor
    + wireframe. Paint panes black and kill the grid/tick lines so the sphere shows."""
    ax.set_facecolor(style.bg)
    transparent = (0.0, 0.0, 0.0, 0.0)
    # w_xaxis/w_yaxis/w_zaxis were REMOVED in mpl 3.8; modern uses xaxis/yaxis/zaxis.
    axes3 = (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis) if hasattr(ax, "w_xaxis") \
        else (ax.xaxis, ax.yaxis, ax.zaxis)
    for axis in axes3:
        try:
            axis.set_pane_color((0.0, 0.0, 0.0, 1.0))  # solid black pane
        except Exception:
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0)); axis.pane.set_alpha(1.0)
        axis.line.set_color(transparent)
        axis._axinfo["grid"]["color"] = transparent
        axis._axinfo["grid"]["linewidth"] = 0.0


def render_panel(ax, proj, fstate, style):
    ax.cla()
    _blacken_3d_axes(ax, style)
    _draw_wire_sphere(ax, style)
    _draw_static_stars(ax, proj, style)
    _draw_active_glow(ax, proj, fstate, style)
    _draw_unembed(ax, proj, fstate, style)
    _draw_marker_and_tail(ax, fstate, style)
    # equal cube via equal limits (mpl 3.2 has NO set_box_aspect; set_aspect('equal') raises)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=style.elev, azim=fstate.azim)
    ax.text2D(
        0.02, 0.95, fstate.hud, transform=ax.transAxes, color="white", fontsize=9
    )


# ----------------------------------------------------------------------------
# render loops
# ----------------------------------------------------------------------------
def _dump_png(out_dir, i, buf):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "frame_%04d.png" % i), "wb") as f:
        f.write(buf.getvalue())


def render_compare_gif(
    proj_a,
    proj_b,
    out,
    total_frames=60,
    style=STYLE,
    dpi=110,
    fps=12,
    orbit_turns=0.5,
    use_slerp=True,
    keep_frames=False,
    hold_end=6,
):
    """Side-by-side twin-sphere GIF. Both panels share total_frames (finish together)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    schA = build_schedule(proj_a.traj_sphere.shape[0], total_frames, hold_end=hold_end)
    schB = build_schedule(proj_b.traj_sphere.shape[0], total_frames, hold_end=hold_end)
    F = max(schA.total_frames, schB.total_frames)
    # rebuild so both have identical frame counts (use the larger budget)
    schA = build_schedule(proj_a.traj_sphere.shape[0], F, hold_end=hold_end)
    schB = build_schedule(proj_b.traj_sphere.shape[0], F, hold_end=hold_end)
    F = min(schA.total_frames, schB.total_frames)

    stA = _precompute_states(proj_a, schA, F, use_slerp, orbit_turns, style, hold_end)
    stB = _precompute_states(proj_b, schB, F, use_slerp, orbit_turns, style, hold_end)

    fig = plt.figure(figsize=(12, 6), dpi=dpi, facecolor=style.bg)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.02)

    frames = []
    frame_dir = os.path.join(os.path.dirname(out) or ".", "frames_debug")
    for i in range(F):
        sa, sb = stA[i], stB[i]
        render_panel(axL, proj_a, sa, style)
        render_panel(axR, proj_b, sb, style)
        axL.set_title("%s (own PCA frame)" % proj_a.tag, color="w", fontsize=11)
        axR.set_title("%s (own PCA frame)" % proj_b.tag, color="w", fontsize=11)
        fig.suptitle(
            "residual orrery   %s   |   %s"
            % (sa.hud, sb.hud),
            color="w",
            fontsize=12,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=style.bg)
        buf.seek(0)
        frames.append(imageio.imread(buf.getvalue()))
        if keep_frames:
            _dump_png(frame_dir, i, buf)
    plt.close(fig)
    _save_gif(out, frames, fps)
    return out


def render_single_gif(
    proj,
    out,
    total_frames=60,
    style=STYLE,
    dpi=110,
    fps=12,
    orbit_turns=0.5,
    use_slerp=True,
    keep_frames=False,
    hold_end=6,
):
    """One-panel variant for a single model."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sch = build_schedule(proj.traj_sphere.shape[0], total_frames, hold_end=hold_end)
    F = sch.total_frames
    st = _precompute_states(proj, sch, F, use_slerp, orbit_turns, style, hold_end)

    fig = plt.figure(figsize=(7, 7), dpi=dpi, facecolor=style.bg)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    frames = []
    frame_dir = os.path.join(os.path.dirname(out) or ".", "frames_debug")
    for i in range(F):
        render_panel(ax, proj, st[i], style)
        ax.set_title("%s (own PCA frame)" % proj.tag, color="w", fontsize=11)
        fig.suptitle("residual orrery   %s" % st[i].hud, color="w", fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=style.bg)
        buf.seek(0)
        frames.append(imageio.imread(buf.getvalue()))
        if keep_frames:
            _dump_png(frame_dir, i, buf)
    plt.close(fig)
    _save_gif(out, frames, fps)
    return out
