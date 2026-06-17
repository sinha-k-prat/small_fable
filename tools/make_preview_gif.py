#!/usr/bin/env python3
"""
make_preview_gif.py — render assets/showcase.gif, a looping teaser of the animated
core-ideas showcase (index.html). Recreates two of the site's scenes with the same
palette + typography. Supersampled 2x for anti-aliasing. macOS system fonts.

    python3 tools/make_preview_gif.py
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

# ----- canvas -----
W, H = 800, 450
SS = 2                      # supersample factor
WS, HS = W * SS, H * SS

# ----- palette (matches index.html) -----
BG1   = (14, 15, 19)
BG2   = (22, 24, 31)
PANEL = (18, 20, 27)
LINE  = (42, 46, 58)
INK   = (233, 231, 224)
MUTED = (139, 144, 156)
AMBER = (245, 185, 66)
TEAL  = (61, 220, 151)
VIOLET= (155, 140, 255)
ROSE  = (255, 107, 139)

F = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FB = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
FM = "/System/Library/Fonts/Menlo.ttc"
def serif(px):  return ImageFont.truetype(FB, px * SS)
def serifr(px): return ImageFont.truetype(F, px * SS)
def mono(px):   return ImageFont.truetype(FM, px * SS)

def lerp(a, b, t): return a + (b - a) * t
def mix(c1, c2, t): return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))
def clamp01(t): return max(0.0, min(1.0, t))
def ease_out(t): t = clamp01(t); return 1 - (1 - t) ** 3
def ease_io(t):
    t = clamp01(t)
    return 4*t*t*t if t < 0.5 else 1 - (-2*t+2)**3/2

def S(v): return int(round(v * SS))

def text_c(d, xy, s, font, fill, anchor="lt", spacing=0):
    """Draw text with manual anchoring (PIL 7.2 has no anchor=)."""
    w, h = d.textsize(s, font=font)
    if spacing:
        w = sum(d.textsize(ch, font=font)[0] for ch in s) + spacing * (len(s) - 1)
    x, y = xy
    if anchor[0] == "m": x -= w / 2
    elif anchor[0] == "r": x -= w
    if anchor[1] == "m": y -= h / 2
    elif anchor[1] == "b": y -= h
    if spacing:
        cx = x
        for ch in s:
            d.text((cx, y), ch, font=font, fill=fill)
            cx += d.textsize(ch, font=font)[0] + spacing
    else:
        d.text((x, y), s, font=font, fill=fill)

def round_rect(d, box, r, outline=None, fill=None, width=1):
    x0, y0, x1, y1 = box
    if fill:
        d.rectangle([x0+r, y0, x1-r, y1], fill=fill)
        d.rectangle([x0, y0+r, x1, y1-r], fill=fill)
        for cx, cy in [(x0+r,y0+r),(x1-r,y0+r),(x0+r,y1-r),(x1-r,y1-r)]:
            d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
    if outline:
        d.arc([x0,y0,x0+2*r,y0+2*r], 180, 270, fill=outline, width=width)
        d.arc([x1-2*r,y0,x1,y0+2*r], 270, 360, fill=outline, width=width)
        d.arc([x0,y1-2*r,x0+2*r,y1], 90, 180, fill=outline, width=width)
        d.arc([x1-2*r,y1-2*r,x1,y1], 0, 90, fill=outline, width=width)
        d.line([x0+r,y0,x1-r,y0], fill=outline, width=width)
        d.line([x0+r,y1,x1-r,y1], fill=outline, width=width)
        d.line([x0,y0+r,x0,y1-r], fill=outline, width=width)
        d.line([x1,y0+r,x1,y1-r], fill=outline, width=width)

def base_frame():
    img = Image.new("RGB", (WS, HS), BG1)
    d = ImageDraw.Draw(img)
    # subtle vertical gradient
    for y in range(HS):
        t = (y / HS) ** 1.3
        d.line([(0, y), (WS, y)], fill=mix(BG2, BG1, t))
    return img, d

def header(d, caption, formula, accent, fade=1.0):
    brand = mix(BG1, AMBER, fade)
    text_c(d, (S(48), S(34)), "S M A L L _ F A B L E", mono(11), brand, "lm")
    text_c(d, (S(48), S(64)), "Core Ideas, in Motion", serif(30), mix(BG1, INK, fade), "lm")
    # accent kicker + caption (below the stage)
    text_c(d, (S(48), S(378)), caption, serifr(21), mix(BG1, INK, fade), "lm")
    text_c(d, (S(48), S(410)), formula, mono(12), mix(BG1, accent, fade), "lm")

def stage_box(d):
    round_rect(d, [S(46), S(96), S(754), S(338)], S(16), outline=LINE, fill=PANEL, width=SS)

# --------------------------------------------------------------------------- scenes
def scene_advantage(d, p):
    """Group-relative advantage: mean line draws, then bars grow above/below."""
    stage_box(d)
    x0, x1 = S(120), S(700)
    meanY = S(228)
    rewards = [0.92, 0.18, 0.70, 0.12, 1.0, 0.40, 0.85, 0.30]
    mean = sum(rewards) / len(rewards)
    scale = S(95)
    # mean line reveal (first 35%)
    lp = ease_out(p / 0.35)
    lx = int(lerp(x0, x1, lp))
    for xx in range(x0, lx, S(12)):
        d.line([(xx, meanY), (min(xx + S(6), lx), meanY)], fill=TEAL, width=SS)
    if lp > 0.3:
        text_c(d, (x1, meanY - S(14)), "group mean", mono(12), TEAL, "rm")
    # bars (after 30%)
    n = len(rewards)
    bw = S(46); gap = (x1 - x0 - n * bw) / (n - 1)
    for i, r in enumerate(rewards):
        bx = x0 + i * (bw + gap)
        bp = ease_out((p - 0.30 - i * 0.055) / 0.4)
        if bp <= 0: continue
        full = (r - mean) * scale
        cur = full * bp
        col = TEAL if r >= mean else ROSE
        top = meanY - cur if r >= mean else meanY
        bot = meanY if r >= mean else meanY - cur
        round_rect(d, [bx, min(top, bot), bx + bw, max(top, bot)], S(5), fill=col)

def scene_bell(d, p):
    """MaxEnt prompt weighting: bell curve draws, weighted dots pop."""
    stage_box(d)
    x0, x1 = S(120), S(680)
    yb = S(300); peak = S(150)
    d.line([(x0, yb), (x1, yb)], fill=LINE, width=SS)
    text_c(d, (x0, yb + S(14)), "p_q=0", mono(11), MUTED, "lm")
    text_c(d, (x1, yb + S(14)), "p_q=1", mono(11), MUTED, "rm")
    text_c(d, ((x0 + x1) // 2, yb + S(14)), "0.5", mono(11), VIOLET, "mm")
    # bell reveal
    def bell_y(t): return yb - peak * math.exp(-((t - 0.5) / 0.18) ** 2 / 2)
    rev = ease_io(clamp01(p / 0.7))
    pts = []
    steps = 120
    for i in range(int(steps * rev) + 1):
        t = i / steps
        pts.append((int(lerp(x0, x1, t)), int(bell_y(t))))
    if len(pts) > 1:
        d.line(pts, fill=VIOLET, width=3 * SS, joint="curve")
    # weighted sample dots
    for sp in [0.0, 0.18, 0.5, 0.72, 1.0]:
        dp = ease_out((p - 0.45 - sp * 0.1) / 0.4)
        if dp <= 0: continue
        w = math.exp(-((sp - 0.5) / 0.18) ** 2 / 2)
        cx = int(lerp(x0, x1, sp)); cy = int(bell_y(sp))
        rr = int((S(5) + S(10) * w) * dp)
        col = AMBER if w > 0.05 else mix(AMBER, PANEL, 0.7)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)

# --------------------------------------------------------------------------- timeline
frames = []
def add(scene, caption, formula, accent, n, p_from=0.0, p_to=1.0, fade=1.0):
    for k in range(n):
        img, d = base_frame()
        f = fade if not callable(fade) else fade(k / max(1, n - 1))
        header(d, caption, formula, accent, f)
        scene(d, lerp(p_from, p_to, k / max(1, n - 1)))
        frames.append(img.resize((W, H), Image.LANCZOS))

# intro fade-in on scene 1
add(scene_advantage, "Group-Relative Advantage", "A_i = r_i - mean(r_group)", TEAL,
    8, 0.0, 0.10, fade=lambda t: ease_out(t))
add(scene_advantage, "Group-Relative Advantage", "A_i = r_i - mean(r_group)", TEAL, 26, 0.10, 1.0)
add(scene_advantage, "Group-Relative Advantage", "A_i = r_i - mean(r_group)", TEAL, 8, 1.0, 1.0)
# crossfade-ish: quick fade out then scene 2 fade in
add(scene_advantage, "Group-Relative Advantage", "A_i = r_i - mean(r_group)", TEAL,
    5, 1.0, 1.0, fade=lambda t: 1 - ease_out(t))
add(scene_bell, "MaxEnt Prompt Weighting", "weight peaks at p_q = 0.5", VIOLET,
    6, 0.0, 0.10, fade=lambda t: ease_out(t))
add(scene_bell, "MaxEnt Prompt Weighting", "weight peaks at p_q = 0.5", VIOLET, 26, 0.10, 1.0)
add(scene_bell, "MaxEnt Prompt Weighting", "weight peaks at p_q = 0.5", VIOLET, 10, 1.0, 1.0)
add(scene_bell, "MaxEnt Prompt Weighting", "weight peaks at p_q = 0.5", VIOLET,
    5, 1.0, 1.0, fade=lambda t: 1 - ease_out(t))

# Build a global palette seeded with EVERY brand color (plus bg gradient shades) so no
# scene's accent gets remapped to the wrong hue. Then map all frames onto it.
seed = Image.new("RGB", (256, 64))
sd = ImageDraw.Draw(seed)
swatches = [BG1, BG2, PANEL, LINE, INK, MUTED, AMBER, TEAL, VIOLET, ROSE]
for i, c in enumerate(swatches):                       # solid swatches (top half)
    sd.rectangle([i * 25, 0, (i + 1) * 25, 32], fill=c)
for y in range(32, 64):                                # bg gradient (bottom half)
    t = (y - 32) / 32
    sd.line([(0, y), (256, y)], fill=mix(BG2, BG1, t))
pal_src = seed.convert("P", palette=Image.ADAPTIVE, colors=128)
qframes = [f.quantize(colors=128, palette=pal_src, dither=Image.NONE) for f in frames]

# dump two full-fade verification frames as PNG (not part of the gif)
if os.environ.get("DUMP"):
    frames[20].save("/tmp/full_a.png")     # mid scene 1
    frames[len(frames)-20].save("/tmp/full_b.png")  # mid scene 2

os.makedirs("assets", exist_ok=True)
out = "assets/showcase.gif"
qframes[0].save(out, save_all=True, append_images=qframes[1:], duration=70, loop=0, optimize=True)
print(f"wrote {out}  ({len(qframes)} frames, {os.path.getsize(out)//1024} KB)")
