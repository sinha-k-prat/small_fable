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
    bg: str = "#04050a"                                  # near-black with a hint of blue
    sphere_wire: tuple = (0.42, 0.58, 0.92, 0.32)        # brighter blue wire (was alpha 0.08)
    wire_lw: float = 0.6
    static_rgba: tuple = (0.74, 0.80, 0.96, 0.45)        # brighter, more opaque stars
    static_s: float = 11.0
    cmap_name: str = "plasma"
    s_min: float = 14.0
    s_max: float = 240.0
    marker_rgba: tuple = (1.0, 1.0, 1.0, 1.0)
    marker_s: float = 130.0
    # persistent accumulating trace (the hop path drawn on the sphere, like the sketch)
    trace_rgba: tuple = (0.20, 1.0, 0.80, 0.95)          # bright cyan/teal path
    trace_lw: float = 2.4
    node_dim_rgba: tuple = (0.55, 0.62, 0.85, 0.55)      # not-yet-visited node dot
    node_hot_rgba: tuple = (1.0, 0.95, 0.55, 1.0)        # visited node dot (warm)
    node_s: float = 26.0
    target_rgba: tuple = (1.0, 0.84, 0.0, 1.0)
    elev: float = 18.0
    azim0: float = -60.0
    wire_stride: int = 16
    zoom: float = 1.5                                    # blow the unit sphere up in-panel
    # ---- v2: correctness beacon (Feature 2) ----
    beacon_correct_rgb: tuple = (0.20, 0.95, 0.45)       # green
    beacon_wrong_rgb: tuple = (0.95, 0.28, 0.28)         # red
    beacon_unknown_rgb: tuple = (1.0, 0.78, 0.18)        # amber/gold (unknown)
    beacon_s: float = 90.0
    beacon_label_fs: float = 11.0
    # ---- v2: input token marker (Feature 2) ----
    input_rgba: tuple = (0.35, 1.0, 0.55, 1.0)           # green diamond at node 0
    input_s: float = 70.0
    # ---- v2: MLP -> writer-column links (Feature 6) ----
    link_rgba: tuple = (0.70, 0.80, 1.0, 0.16)           # faint cool
    link_lw: float = 0.6
    link_topn: int = 4                                   # 3..5 brightest firing cols


STYLE = GlowStyle()

# Dense points sampled along EACH trajectory segment for the persistent great-circle trace.
PTS_PER_SEG = 22


def _build_trace(traj, use_slerp=True):
    """Full hop path as a dense great-circle polyline on S^2 -> [M, 3], M = (P-1)*PTS_PER_SEG.
    Segment s occupies rows [s*PTS_PER_SEG : (s+1)*PTS_PER_SEG], matching the reveal count
    computed in _precompute_states, so the drawn trace exactly tracks the marker."""
    P = traj.shape[0]
    pts = []
    for s in range(P - 1):
        p0, p1 = traj[s], traj[s + 1]
        for k in range(PTS_PER_SEG):
            tt = (k + 1) / PTS_PER_SEG
            pts.append(slerp(p0, p1, tt) if use_slerp else _lerp_renorm(p0, p1, tt))
    if not pts:
        return np.zeros((0, 3), np.float64)
    return np.asarray(pts, np.float64)


def _zoom_in(ax, zoom):
    """Make the unit sphere fill more of the panel, version-safely. Modern mpl: set_box_aspect
    with a zoom kwarg. Old mpl 3.2: shrink ax.dist (smaller = closer)."""
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)   # mpl >= ~3.3
        return
    except Exception:
        pass
    try:
        ax.dist = 10.0 / float(zoom)              # mpl 3.2
    except Exception:
        pass


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
    trace_reveal: int = 0    # how many polyline points of the persistent trace to draw
    nodes_visited: int = 0   # how many trajectory nodes the hop has reached


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
        tail = [np.asarray(p) for p in tail_positions]   # full persistent tail (unused for trace)

        # persistent trace reveal: segment s rows are [s*PTS_PER_SEG : (s+1)*PTS_PER_SEG],
        # so reveal up to the marker's current fraction within its segment.
        seg = int(sched.seg_of_frame[i])
        reveal = seg * PTS_PER_SEG + int(round(t * PTS_PER_SEG))
        reveal = min(reveal, max(P - 1, 0) * PTS_PER_SEG)
        nodes_visited = min(active_node + 1, P)

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
                trace_reveal=int(reveal),
                nodes_visited=int(nodes_visited),
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
    ax.plot_wireframe(xs, ys, zs, linewidth=style.wire_lw, color=style.sphere_wire)


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


def _beacon_rgb(proj, style):
    c = getattr(proj, "is_correct", None)
    if c is True:
        return style.beacon_correct_rgb
    if c is False:
        return style.beacon_wrong_rgb
    return style.beacon_unknown_rgb                  # None -> amber


def _draw_beacon(ax, proj, fstate, style, label_prefix=""):
    """Terminal beacon at unembed_sphere: persistent, pulsing, GREEN/RED/amber by correctness,
    labeled with the generated answer (falls back to pred_token_str for old/smoke runs).
    Replaces _draw_unembed; keeps its pulse/halo/hold mechanics. Beacon color is decoupled
    from pred_token_id — it comes ONLY from is_correct."""
    t = proj.unembed_sphere  # [3]
    rgb = _beacon_rgb(proj, style)
    g = fstate.target_glow                            # 0 until end-hold, ramps to 1
    pulse = 0.55 + 0.45 * fstate.glow_env             # reuse glow_env (no new FrameState field)
    base_a = 0.45 + 0.55 * g
    ax.scatter([t[0]], [t[1]], [t[2]], s=style.beacon_s * (1.0 + 1.2 * g),
               c=[list(rgb) + [base_a]], depthshade=False, edgecolors="none")
    ax.scatter([t[0]], [t[1]], [t[2]], s=style.beacon_s * (2.2 + 2.5 * g) * pulse,    # halo
               c=[list(rgb) + [0.16 * (0.5 + g)]], depthshade=False, edgecolors="none")
    label = (getattr(proj, "answer_text", "") or proj.pred_token_str).strip().replace("\n", " ")
    if len(label) > 22:
        label = label[:21] + "…"
    if g > 0:
        ax.text(t[0], t[1], t[2], "  " + label_prefix + label, color="white",
                fontsize=style.beacon_label_fs,
                bbox=dict(facecolor=list(rgb) + [0.35], edgecolor="none", pad=1.5))


def _draw_input_marker(ax, proj, style):
    """Distinct diamond at node 0 (the INPUT/embed token) so the trajectory start is
    identifiable. marker='D', depthshade=False, edgecolors='none' — verified on mpl 3.2.2."""
    p = proj.traj_sphere[0]
    ax.scatter([p[0]], [p[1]], [p[2]], s=style.input_s, marker="D",
               c=[style.input_rgba], depthshade=False, edgecolors="none")


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
    if stars.shape[0] == 0:
        return
    n = min(style.link_topn, stars.shape[0])
    env = fstate.glow_env
    for k in range(n):
        d = stars[k]
        w = a[k] / max(a[0], 1e-8)
        alpha = float(np.clip(style.link_rgba[3] * w * (0.4 + 0.6 * env), 0.0, 1.0))
        ax.plot([src[0], d[0]], [src[1], d[1]], [src[2], d[2]],
                color=style.link_rgba[:3], alpha=alpha, linewidth=style.link_lw,
                linestyle=":", solid_capstyle="round")


def _draw_trace(ax, proj, fstate, style):
    """Persistent accumulating hop path drawn ON the sphere (great-circle polyline) plus a dot at
    every trajectory node — visited nodes warm, upcoming nodes dim. The line grows as the marker
    hops and stays drawn (during the end hold the WHOLE route is visible, like the sketch)."""
    trace = getattr(proj, "_trace_xyz", None)
    r = int(fstate.trace_reveal)
    if trace is not None and r >= 2:
        seg = trace[:r]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=style.trace_rgba[:3], alpha=style.trace_rgba[3],
                linewidth=style.trace_lw, solid_capstyle="round")
    # node dots: every trajectory node, brightened once visited
    nodes = proj.traj_sphere  # [P, 3]
    nv = int(fstate.nodes_visited)
    if nv < len(nodes):
        up = nodes[nv:]
        ax.scatter(up[:, 0], up[:, 1], up[:, 2], s=style.node_s * 0.7,
                   c=[style.node_dim_rgba], depthshade=False, edgecolors="none")
    if nv > 0:
        vis = nodes[:nv]
        ax.scatter(vis[:, 0], vis[:, 1], vis[:, 2], s=style.node_s,
                   c=[style.node_hot_rgba], depthshade=False, edgecolors="none")
    # bright moving head
    m = fstate.marker_xyz
    ax.scatter([m[0]], [m[1]], [m[2]], s=style.marker_s, c=[style.marker_rgba],
               depthshade=False, edgecolors="none")
    ax.scatter([m[0]], [m[1]], [m[2]], s=style.marker_s * 2.4,    # soft halo around the head
               c=[(1.0, 1.0, 1.0, 0.18)], depthshade=False, edgecolors="none")


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
    # equal cube via equal limits (mpl 3.2 has NO set_box_aspect; set_aspect('equal') raises)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=style.elev, azim=fstate.azim)
    _zoom_in(ax, style.zoom)   # blow the sphere up so it fills the panel
    ax.text2D(
        0.02, 0.95, fstate.hud, transform=ax.transAxes, color="white", fontsize=10
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
    proj_a._trace_xyz = _build_trace(proj_a.traj_sphere, use_slerp)
    proj_b._trace_xyz = _build_trace(proj_b.traj_sphere, use_slerp)

    fig = plt.figure(figsize=(12, 6.6), dpi=dpi, facecolor=style.bg)
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


# ----------------------------------------------------------------------------
# v2: collective (Feature 4) + rescue (Feature 5) — multiple traces on ONE sphere
# ----------------------------------------------------------------------------
import dataclasses  # noqa: E402  (used by the overlay style-clone)


def _trace_style(style, rgb):
    """Clone `style` so _draw_trace / _draw_input_marker pick up a per-trace color while the
    BEACON stays correctness-colored (we don't touch the beacon_* fields)."""
    rgb3 = tuple(rgb[:3])
    return dataclasses.replace(
        style,
        trace_rgba=rgb3 + (0.95,),
        node_hot_rgba=rgb3 + (1.0,),
        node_dim_rgba=rgb3 + (0.45,),
        input_rgba=rgb3 + (1.0,),
    )


def _draw_overlay_frame(ax, projs, states, styles, i, links, prefixes):
    """Draw one frame of N overlaid traces into the SAME 3D axis. Wire sphere + static-star
    union drawn ONCE, then each trace's glow/beacon/trace/input/links layered in."""
    ax.cla()
    _blacken_3d_axes(ax, styles[0])
    _draw_wire_sphere(ax, styles[0])
    for proj, st in zip(projs, states):
        _draw_static_stars(ax, proj, styles[0])
    for j, (proj, st, stl) in enumerate(zip(projs, states, styles)):
        fs = st[i]
        if links:
            _draw_mlp_links(ax, proj, fs, stl)
        _draw_active_glow(ax, proj, fs, stl)
        _draw_beacon(ax, proj, fs, stl, label_prefix=prefixes[j])
        _draw_trace(ax, proj, fs, stl)
        _draw_input_marker(ax, proj, stl)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=styles[0].elev, azim=states[0][i].azim)
    _zoom_in(ax, styles[0].zoom)


def _prep_overlay(projs, total_frames, hold_end, use_slerp, orbit_turns, style, colors):
    """Build per-trace schedules to a common F, per-trace state lists, per-trace styles, and
    attach _trace_xyz. Returns (states_list, styles_list, F)."""
    # common F = the largest schedule across traces (so all orbit/finish together)
    Fs = []
    for proj in projs:
        s = build_schedule(proj.traj_sphere.shape[0], total_frames, hold_end=hold_end)
        Fs.append(s.total_frames)
    F = max(Fs)
    states, styles = [], []
    for j, proj in enumerate(projs):
        sch = build_schedule(proj.traj_sphere.shape[0], F, hold_end=hold_end)
        stl = _trace_style(style, colors[j])
        st = _precompute_states(proj, sch, sch.total_frames, use_slerp, orbit_turns, stl, hold_end)
        proj._trace_xyz = _build_trace(proj.traj_sphere, use_slerp)
        states.append(st)
        styles.append(stl)
    F = min(len(s) for s in states)
    return states, styles, F


def _ok_word(proj):
    c = getattr(proj, "is_correct", None)
    return "correct" if c is True else ("wrong" if c is False else "?")


def render_collective_gif(
    projected_runs,
    out,
    total_frames=72,
    style=STYLE,
    dpi=110,
    fps=12,
    orbit_turns=0.5,
    use_slerp=True,
    keep_frames=False,
    hold_end=8,
    cmap_name="tab10",
    links=False,
    title=None,
    labels=None,
):
    """ONE shared sphere with every task trajectory overlaid, each in a distinct qualitative
    color, each ending at its OWN correctness-colored labeled beacon (label prefixed with the
    task key so overlapping beacons stay legible), slow shared orbit. links OFF by default
    (busy with N traces)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmap = _get_cmap(cmap_name)
    ncol = getattr(cmap, "N", 10)
    colors = [cmap(j % ncol) for j in range(len(projected_runs))]
    if labels is None:
        labels = [str(getattr(p, "_task_key", "") or "") for p in projected_runs]
    prefixes = [(lbl + ": ") if lbl else "" for lbl in labels]

    states, styles, F = _prep_overlay(
        projected_runs, total_frames, hold_end, use_slerp, orbit_turns, style, colors
    )

    fig = plt.figure(figsize=(7.6, 7.6), dpi=dpi, facecolor=style.bg)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    frames = []
    frame_dir = os.path.join(os.path.dirname(out) or ".", "frames_debug")
    for i in range(F):
        _draw_overlay_frame(ax, projected_runs, states, styles, i, links, prefixes)
        tag = projected_runs[0].tag if projected_runs else ""
        fig.suptitle(title or ("collective — %s (shared PCA frame)" % tag),
                     color="w", fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=style.bg)
        buf.seek(0)
        frames.append(imageio.imread(buf.getvalue()))
        if keep_frames:
            _dump_png(frame_dir, i, buf)
    plt.close(fig)
    _save_gif(out, frames, fps)
    return out


def render_collective_pair_gif(
    runs_a,
    runs_b,
    out,
    total_frames=72,
    style=STYLE,
    dpi=110,
    fps=12,
    orbit_turns=0.5,
    use_slerp=True,
    keep_frames=False,
    hold_end=8,
    cmap_name="tab10",
    links=False,
    labels_a=None,
    labels_b=None,
):
    """Two collective spheres side by side (mirrors render_compare_gif's layout): model A's
    tasks on the left sphere, model B's on the right."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmap = _get_cmap(cmap_name)
    ncol = getattr(cmap, "N", 10)
    colA = [cmap(j % ncol) for j in range(len(runs_a))]
    colB = [cmap(j % ncol) for j in range(len(runs_b))]
    if labels_a is None:
        labels_a = [str(getattr(p, "_task_key", "") or "") for p in runs_a]
    if labels_b is None:
        labels_b = [str(getattr(p, "_task_key", "") or "") for p in runs_b]
    prefA = [(l + ": ") if l else "" for l in labels_a]
    prefB = [(l + ": ") if l else "" for l in labels_b]

    stA, stylesA, FA = _prep_overlay(runs_a, total_frames, hold_end, use_slerp, orbit_turns, style, colA)
    stB, stylesB, FB = _prep_overlay(runs_b, total_frames, hold_end, use_slerp, orbit_turns, style, colB)
    F = min(FA, FB)

    fig = plt.figure(figsize=(12, 6.6), dpi=dpi, facecolor=style.bg)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.02)

    frames = []
    frame_dir = os.path.join(os.path.dirname(out) or ".", "frames_debug")
    for i in range(F):
        _draw_overlay_frame(axL, runs_a, stA, stylesA, i, links, prefA)
        _draw_overlay_frame(axR, runs_b, stB, stylesB, i, links, prefB)
        axL.set_title("%s (shared PCA frame)" % (runs_a[0].tag if runs_a else ""), color="w", fontsize=11)
        axR.set_title("%s (shared PCA frame)" % (runs_b[0].tag if runs_b else ""), color="w", fontsize=11)
        fig.suptitle("collective — all tasks per model", color="w", fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=style.bg)
        buf.seek(0)
        frames.append(imageio.imread(buf.getvalue()))
        if keep_frames:
            _dump_png(frame_dir, i, buf)
    plt.close(fig)
    _save_gif(out, frames, fps)
    return out


def render_rescue_gif(
    proj_simple,
    proj_detailed,
    out,
    total_frames=72,
    style=STYLE,
    dpi=110,
    fps=12,
    orbit_turns=0.5,
    use_slerp=True,
    keep_frames=False,
    hold_end=8,
    simple_rgb=(0.95, 0.45, 0.20),
    detailed_rgb=(0.30, 0.65, 1.0),
    task_label="",
    model_tag="",
    links=True,
):
    """Overlay two variant trajectories of ONE task on ONE shared sphere. Each trace its own
    color (simple=orange, detailed=blue); each ends at its correctness-colored beacon (simple
    typically RED, detailed GREEN). A RED->GREEN flip shows as the detailed path turning to a
    different region. Slow orbit. links=True (only two traces) so the 'turn' is legible."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    projs = [proj_simple, proj_detailed]
    colors = [simple_rgb, detailed_rgb]
    prefixes = ["simple ", "detailed "]
    states, styles, F = _prep_overlay(
        projs, total_frames, hold_end, use_slerp, orbit_turns, style, colors
    )

    ok_s, ok_d = _ok_word(proj_simple), _ok_word(proj_detailed)
    suptitle = "RESCUE — %s%s: simple(orange)->%s vs detailed(blue)->%s" % (
        task_label, (" on %s" % model_tag) if model_tag else "", ok_s, ok_d
    )

    fig = plt.figure(figsize=(7.6, 7.6), dpi=dpi, facecolor=style.bg)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.90)

    frames = []
    frame_dir = os.path.join(os.path.dirname(out) or ".", "frames_debug")
    for i in range(F):
        _draw_overlay_frame(ax, projs, states, styles, i, links, prefixes)
        fig.suptitle(suptitle, color="w", fontsize=11)
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
    proj._trace_xyz = _build_trace(proj.traj_sphere, use_slerp)

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
