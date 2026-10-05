"""Timeline of the Yoshino PV. render(t) returns the float RGB frame at time t (seconds).

The cut list follows the structure and timing of the reference opening movie; every character
shot uses the repository's Yoshino artwork, and everything else is drawn in code.
"""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

import backgrounds
from common import *  # noqa: F401,F403
from common import _premul

DURATION = 114.47

import json
from pathlib import Path

_BEATS = json.load(open(Path(__file__).resolve().parent / "beats.json"))
BEAT_T = np.array(_BEATS["beats"], np.float64)
DOWNBEAT = _BEATS["downbeat_offset"]


def last_beat(t):
    i = int(np.searchsorted(BEAT_T, t, "right")) - 1
    if i < 0:
        return None, False
    return float(BEAT_T[i]), (i - DOWNBEAT) % 4 == 0


def next_beat(t):
    i = int(np.searchsorted(BEAT_T, t, "left"))
    return float(BEAT_T[i]) if i < len(BEAT_T) else None

# --------------------------------------------------------------------------- face anchors (source px)
CARD_FACE = {
    "card1": (610, 370), "card2": (500, 360), "card3": (580, 440), "card4": (630, 450), "card5": (660, 540),
    "card6": (690, 370), "card7": (530, 380), "card8": (330, 450), "card9": (590, 410), "card10": (690, 330),
    "card11": (580, 490), "card12": (600, 560), "card13": (590, 370), "card15": (670, 380), "card16": (580, 390),
    "card17": (720, 370), "card18": (400, 620), "card19": (430, 470), "card20": (580, 340), "card21": (600, 310),
    "card22": (630, 350),
    "Yoshino SSR1": (650, 370), "Yoshino SSR2": (680, 420), "Yoshino SSR3": (620, 365), "Yoshino SSR4": (630, 330),
    "Yoshino SSR5": (360, 320),
}
CUT_FACE = {  # fraction of image size
    "tachie1": (0.54, 0.29), "tachie2": (0.476, 0.318), "tachie3": (0.427, 0.286), "tachie4": (0.52, 0.30),
    "tachie5": (0.367, 0.266), "tachie8": (0.416, 0.30), "tachie9": (0.47, 0.32),
    "ysn1": (0.49, 0.353), "ysn2": (0.51, 0.155), "ysn3": (0.568, 0.256), "ysn4": (0.495, 0.26), "ysn5": (0.488, 0.209),
    "ysn6": (0.423, 0.297),
}


def face(name):
    if name in CARD_FACE:
        return CARD_FACE[name]
    w, h = size_of(name)
    fx, fy = CUT_FACE[name]
    return fx * w, fy * h


def put(dst, name, x, y, height=None, scale=None, ang=0.0, opacity=1.0, flip=False, anchor=None):
    """Place a cut-out with its face (or anchor fraction) at (x, y)."""
    w, h = size_of(name)
    if scale is None:
        scale = height / h
    ax, ay = face(name) if anchor is None else (anchor[0] * w, anchor[1] * h)
    return place(dst, name, ax, ay, x, y, scale, ang, opacity, flip)


SILHOUETTE = np.float32([0.90, 0.36, 0.62])


def put_reveal(dst, name, x, y, scale, opacity, color_u):
    """Cut-out drawn as a flat pink silhouette that dissolves into full colour (color_u 0..1)."""
    if opacity <= 0.001:
        return dst
    w, h = size_of(name)
    fx, fy = face(name)
    lf = to_f(warp_layer(name, fx, fy, x, y, scale))
    a = lf[..., 3:4]
    cu = smooth(clamp01(color_u))
    if cu < 1:  # white rim so the silhouette separates from the pink background
        rim = cv2.dilate(np.ascontiguousarray(a[..., 0]), np.ones((7, 7), np.uint8))
        rim = cv2.GaussianBlur(rim, (0, 0), 1.5)[..., None]
        rl = np.concatenate([rim * np.float32([1.0, 0.96, 0.98]), rim], -1)
        over(dst, rl, opacity * (1 - cu))
    sil = a * SILHOUETTE
    rgb = sil * (1 - cu) + lf[..., :3] * cu
    # a soft light lift while the colour comes in
    rgb = np.minimum(rgb + a * 0.25 * pulse(cu, 0.0, 0.35, 1.0), a)
    return over(dst, np.concatenate([rgb, a], -1), opacity)


# --------------------------------------------------------------------------- prepared assets


def extract_on_white(name, thresh=46.0, enclosed=False, protect=()):
    """Alpha from a logo drawn on white.

    White connected to the border is background. With enclosed=True, enclosed white areas that are
    pure white (inside the heart, letter counters, gaps between the rows) are background too, while
    the off-white letter fills are kept; `protect` boxes (x0, y0, x1, y1) are never cut.
    """
    a = img(name)
    rgb = a[..., :3].astype(np.float32)
    dark = 255.0 - rgb.min(axis=2)
    cand = (dark < (12 if enclosed else thresh)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    keep_bg = np.zeros(n, bool)
    keep_bg[list(border)] = True
    if enclosed:
        mean_dark = np.bincount(lab.ravel(), weights=dark.ravel(), minlength=n) / np.maximum(stats[:, 4], 1)
        for i in range(1, n):
            if stats[i, 4] < 300 or mean_dark[i] >= 4.5:
                continue
            x, y, w, h = stats[i, :4]
            if any(x >= p[0] and y >= p[1] and x + w <= p[2] and y + h <= p[3] for p in protect):
                continue
            keep_bg[i] = True
    bg = keep_bg[lab]
    ring = cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~bg
    alpha = np.ones(dark.shape, np.float32)
    alpha[bg] = 0.0
    alpha[ring] = np.clip((dark[ring] - 10) / (thresh - 10), 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
    prem = np.clip(rgb - (1 - alpha[..., None]) * 255.0, 0, 255)
    out = np.dstack([prem, alpha * 255]).astype(np.uint8)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
    return out


def compose_group():
    """All-Yoshino group shot assembled from the cut-out artwork."""
    GW, GH = 1180, 885
    bgim = backgrounds.draw_group_backdrop((GW, GH))
    dst = np.asarray(bgim, np.float32) / 255.0
    canvas = np.zeros((GH, GW, 3), np.float32)
    canvas[:] = dst
    layout = [  # name, face x, face y, scale
        ("tachie9", 250, 215, 1.55),
        ("tachie5", 950, 200, 1.6),
        ("tachie3", 610, 235, 1.45),
        ("tachie2", 280, 520, 1.65),
        ("tachie8", 905, 545, 1.6),
        ("tachie4", 585, 600, 1.62),
    ]
    for name, x, y, s in layout:
        lay = _place_canvas(name, x, y, s, GW, GH)
        a = lay[..., 3:4]
        canvas = canvas * (1 - a) + lay[..., :3]
    return _premul(np.dstack([np.clip(canvas, 0, 1) * 255, np.full((GH, GW), 255)]).astype(np.uint8))


def compose_action():
    GW, GH = 1180, 885
    canvas = np.asarray(backgrounds.draw_sky_burst((GW, GH)), np.float32) / 255.0
    layout = [
        ("tachie1", 960, 150, 0.95, -12),
        ("ysn2", 230, 250, 0.25, 8),
        ("ysn3", 950, 420, 0.22, -6),
        ("ysn6", 260, 650, 0.17, 10),
        ("ysn4", 600, 330, 0.30, -4),
    ]
    for name, x, y, s, ang in layout:
        lay = _place_canvas(name, x, y, s, GW, GH, ang)
        canvas = canvas * (1 - lay[..., 3:4]) + lay[..., :3]
    return _premul(np.dstack([np.clip(canvas, 0, 1) * 255, np.full((GH, GW), 255)]).astype(np.uint8))


def _place_canvas(name, x, y, s, GW, GH, ang=0.0):
    src = img(name)
    fx, fy = face(name)
    level = 0
    while s * 2 ** level < 0.5:
        level += 1
    from common import mip
    src = mip(name, level)
    f = 2 ** level
    M = m_place(fx / f, fy / f, x, y, s * f, ang)
    lay = cv2.warpAffine(src, M, (GW, GH), flags=cv2.INTER_LINEAR, borderValue=0)
    return lay.astype(np.float32) / 255.0


def prepare():
    backgrounds.build_all()
    from common import BUILD
    for key, fn in (("logo_title", lambda: img("logo_yoshinon")),  # user-supplied cut-out title logo
                    ("logo_brand", lambda: extract_on_white("logo1")),
                    ("group", compose_group),
                    ("action", compose_action)):
        if not (BUILD / f"{key}.npy").exists():
            register(key, fn())
    for name in ("ysn3", "ysn2", "ysn5", "ysn1", "card10", "card15", "card13", "Yoshino SSR1", "Yoshino SSR2", "Yoshino SSR3",
                 "Yoshino SSR4", "Yoshino SSR5", "card9", "group"):
        sketch(name)


# --------------------------------------------------------------------------- building blocks


@lru_cache(None)
def checker():
    a = img("haikei").astype(np.float32) / 255.0
    tile = np.concatenate([a[..., :3], a[..., :3]], axis=1)
    return cv2.resize(tile, (W, H), interpolation=cv2.INTER_AREA)


CHECKER_SPEED = 40.0  # px/s, the checker background rises


def checker_at(t, boost=1.0):
    """The pink checker asset scrolling upward; boost > 1 strengthens its (very pale) pattern."""
    off = (t * CHECKER_SPEED) % H
    c = cv2.warpAffine(checker(), np.float32([[1, 0, 0], [0, 1, -off]]), (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
    if boost != 1.0:
        m = np.float32([0.97, 0.93, 0.95])
        c = np.clip(m + (c - m) * boost, 0, 1)
    return c


FLARE = None


def flare():
    global FLARE
    if FLARE is None:
        from common import BUILD
        FLARE = np.load(BUILD / "flare.npy")
    return FLARE


def title_logo(dst, cx=W / 2, cy=H * 0.47, st=0.8, opacity=1.0, glow=0.0, flip=False, mono=0.0, bright=0.0):
    """YOSHINON RHYME logo; st=1 fills the frame."""
    if opacity <= 0.001:
        return dst
    w, h = size_of("logo_title")
    s = W / w * st
    lay = warp_layer("logo_title", w / 2, h / 2, cx, cy, s, 0.0, flip)
    lf = to_f(lay)
    if mono or bright:
        a = lf[..., 3:4]
        col = lf[..., :3]
        col = col * (1 - mono) + a * np.float32([1.0, 0.94, 0.97]) * mono
        col = np.minimum(col + a * bright, a)
        lf = np.concatenate([col, a], -1)
    if glow > 0:
        g = blur(np.ascontiguousarray(lf[..., 3]), 10)
        gl = np.clip(g * 1.4, 0, 1)[..., None] * glow
        dst[:] = screen(dst, gl * np.float32([1.0, 0.82, 0.92]))
    return over(dst, lf, opacity)


def heart_ribbon(dst, u, cx=W / 2, cy=H * 0.47, st=0.8, opacity=1.0, glow=0.0, tint=None):
    """Draw-on animation of the ribbon heart using the three supplied key drawings."""
    if opacity <= 0.001 or u <= 0:
        return dst
    layers = []
    if u < 0.5:
        k3 = clamp01(1 - (u - 0.25) / 0.1) * clamp01(u / 0.08)
        k2 = clamp01((u - 0.22) / 0.08)
        if k3 > 0:
            layers.append(("heart3", 1.0, k3, 0.9 + 0.2 * u))
        if k2 > 0:
            layers.append(("heart2", 1.0, k2, 1.0))
    else:
        k2 = clamp01(1 - (u - 0.55) / 0.2)
        if k2 > 0:
            layers.append(("heart2", 1.0, k2, 1.0))
    acc = np.zeros((H, W, 4), np.float32)
    for name, base_scale, k, extra in layers:
        w, h = size_of(name)
        s = W / w * st * extra
        lf = to_f(warp_layer(name, w / 2, h / 2, cx, cy, s))
        acc = acc * (1 - lf[..., 3:4] * k) + lf * k
    if u > 0.45:
        w, h = size_of("heart1")
        s = W / w * st
        lf = to_f(warp_layer("heart1", w / 2, h / 2, cx, cy, s))
        sweep = clamp01((u - 0.45) / 0.55)
        xn, yn = xy_norm()
        ang = (np.arctan2(yn * H - cy, xn * W - cx) - math.radians(120)) % (2 * math.pi)
        m = np.clip((sweep * 2 * math.pi * 1.08 - ang) / 0.25, 0, 1)[..., None]
        lf = lf * m
        acc = acc * (1 - lf[..., 3:4]) + lf
    if tint is not None:
        a = acc[..., 3:4]
        acc = np.concatenate([np.minimum(acc[..., :3] * 0.4 + a * np.float32(tint) * 0.6, a), a], -1)
    if glow > 0:
        g = blur(np.ascontiguousarray(acc[..., 3]), 7)
        dst[:] = screen(dst, np.clip(g * 1.6, 0, 1)[..., None] * np.float32([1.0, 0.8, 0.92]) * glow)
    return over(dst, acc, opacity)


SQ_SOFT = [
    (0.30, 0.42, 0.62, 20, 6, (0.97, 0.62, 0.78), 0.45),
    (0.72, 0.58, 0.55, -12, -5, (0.99, 0.74, 0.86), 0.45),
    (0.55, 0.25, 0.40, 35, 9, (1.0, 0.88, 0.93), 0.5),
    (0.15, 0.85, 0.45, 5, -7, (0.96, 0.56, 0.74), 0.35),
    (0.90, 0.15, 0.38, 28, 8, (0.98, 0.68, 0.82), 0.4),
]
SQ_VIVID = [
    (0.25, 0.35, 0.70, 24, 14, (0.95, 0.38, 0.64), 0.55),
    (0.78, 0.62, 0.66, -18, -12, (0.98, 0.55, 0.75), 0.6),
    (0.58, 0.12, 0.45, 40, 18, (1.0, 0.82, 0.90), 0.55),
    (0.10, 0.90, 0.50, 8, -16, (0.93, 0.32, 0.58), 0.5),
]


def soft_pink(rgb, k=0.25):
    return screen(rgb, np.float32([1.0, 0.86, 0.92]), k)


# --------------------------------------------------------------------------- text


TAGLINE = "YORITA YOSHINO  FAN MOVIE"
COPYRIGHT = ("©Bandai Namco Entertainment Inc.", "fan-made movie")
CATCH_MAIN = "神さびて　愛らしく――それはひとりの偶像の物語。"
CATCH_RUBY = "神様"
NAME_JP = "依田芳乃"
NAME_EN = "Yoshino Yorita"
YOSHINO_COLOR = (0xC4, 0xBC, 0xB7)  # 依田芳乃 image colour #C4BCB7
YOSHINO_DARK = (122, 108, 101)


def tagline_layer(y=H * 0.605):
    return text_layer(TAGLINE, FONT_ROUND_B, 15, W / 2, y, anchor="c", fill=(232, 110, 150), spacing=2,
                      stroke=2, stroke_fill=(255, 255, 255))


CATCH_IN = (12.72, 13.4)
CATCH_OUT = 16.75  # sparkle-out starts here, sweeping left to right
CATCH_SWEEP = 0.4
CATCH_SIZE = 34


@lru_cache(None)
def catch_layer():
    """Catch copy with ruby 神様 over 偶像, centred in the frame, plus per-glyph screen boxes."""
    im, pad, xs, widths = text_image(CATCH_MAIN, FONT_SERIF, CATCH_SIZE, fill=(255, 255, 255), stroke=4,
                                     stroke_fill=(220, 80, 140), glow=8, glow_col=(255, 130, 186), glow_k=1.0)
    i = CATCH_MAIN.index("偶像")
    gx = xs[i]
    gw = widths[i] + widths[i + 1]
    rim, rpad, _, rwid = text_image(CATCH_RUBY, FONT_SERIF, 17, fill=(255, 255, 255), stroke=2,
                                    stroke_fill=(234, 108, 160), glow=4, glow_col=(255, 150, 196), spacing=3)
    tw = im.size[0] - 2 * pad
    x0 = W / 2 - tw / 2
    y0 = H / 2 - CATCH_SIZE * 0.7
    ox = int(x0 - pad)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.alpha_composite(im, (ox, int(y0 - pad)))
    rw = sum(rwid)
    rx = x0 + (gx - pad) + gw / 2 - rw / 2 + 1
    canvas.alpha_composite(rim, (int(rx - rpad), int(y0 - pad - 8)))
    lay = pil_to_layer(canvas)
    glyph_x = [ox + x for x in xs]
    # column -> glyph index, so every column (glyph, stroke, glow, ruby) leaves with its glyph
    col = np.zeros(W, np.int32)
    for k, gx_k in enumerate(glyph_x):
        col[int(max(0, gx_k - 6)):] = k
    return lay, x0, x0 + tw, tuple(glyph_x), tuple(widths), (y0 - 14, y0 + CATCH_SIZE * 1.35), col


def _glyph_start(k, n):
    return CATCH_OUT + CATCH_SWEEP * k / max(1, n - 1)


@lru_cache(None)
def catch_stars():
    _, _, _, gx, gw, (ya, yb), _ = catch_layer()
    rng = np.random.default_rng(17)
    stars = []
    for k, (x, w) in enumerate(zip(gx, gw)):
        if CATCH_MAIN[k] in "　―":
            continue
        for j in range(3 if j_is_big(k) else 2):
            stars.append((_glyph_start(k, len(gx)) + rng.uniform(0.0, 0.1), x + rng.uniform(0.15, 0.85) * w,
                          rng.uniform(ya + 6, yb - 6), rng.uniform(11, 22), rng.uniform(0, 0.8),
                          rng.uniform(-40, -14), rng.uniform(-10, 10)))
    return stars


def j_is_big(k):
    return k % 2 == 0


def star_poly(x, y, r, ang, thin=0.16):
    pts = []
    for k in range(8):
        th = ang + k * math.pi / 4
        rr = r if k % 2 == 0 else r * thin
        pts.append((x + math.cos(th) * rr, y + math.sin(th) * rr))
    return pts


def catch_sparkles(dst, t):
    """Four-point stars that flash up as each glyph vanishes, then drift up and fade."""
    polys, dots = [], []
    life = 0.34
    for t0, x, y, r, ang, vy, vx in catch_stars():
        u = (t - t0) / life
        if u <= 0 or u >= 1:
            continue
        k = math.sin(math.pi * u) ** 1.5
        px, py = x + vx * u, y + vy * u
        polys.append(star_poly(px, py, r * (0.3 + 0.9 * k), ang + u * 1.6))
        dots.append((px, py, 3.5 * k))
    if not polys:
        return dst
    m = poly_mask(polys)
    for px, py, rad in dots:
        cv2.circle(m, (int(px * 16), int(py * 16)), max(1, int(rad * 16)), 255, -1, cv2.LINE_AA, 4)
    glow = cv2.GaussianBlur(m, (0, 0), 7).astype(np.float32) / 255.0
    dst[:] = screen(dst, np.clip(glow * 2.2, 0, 1)[..., None] * np.float32([1.0, 0.62, 0.84]), 0.9)
    fill_mask(dst, m, (1.0, 0.98, 0.92), 1.0)
    return dst


def catchcopy(dst, t):
    if t < CATCH_IN[0] or t > CATCH_OUT + CATCH_SWEEP + 0.5:
        return dst
    lay, x0, x1, gx, gw, _, col = catch_layer()
    lf = to_f(lay)
    xn = np.arange(W, dtype=np.float32)
    rev = lerp(x0 - 20, x1 + 60, smooth(seg(t, *CATCH_IN)))
    m = np.clip((rev - xn) / 50.0, 0, 1)
    if t >= CATCH_OUT:
        n = len(gx)
        starts = np.array([_glyph_start(k, n) for k in range(n)], np.float32)[col]
        v = np.clip((t - starts) / 0.16, 0, 1)
        flash = np.sin(np.pi * np.clip(v / 0.5, 0, 1)) * (v < 0.5)
        fade = np.clip((v - 0.35) / 0.65, 0, 1)
        m = m * (1 - fade * fade * (3 - 2 * fade))
        a = lf[..., 3:4]
        lf = np.concatenate([np.minimum(lf[..., :3] + a * flash[None, :, None] * 0.6, a), a], -1)
    over(dst, lf * m[None, :, None], 1.0)
    if t >= CATCH_OUT:
        catch_sparkles(dst, t)
    return dst


def name_plate(dst, tau, side):
    """Name plate for the character intros; side='r' puts it bottom-right."""
    if tau < 0:
        return dst
    x = W * 0.955 if side == "r" else W * 0.045
    info = text_layer(NAME_JP, FONT_ROUND, 84, x, H * 0.735, anchor=side if side == "r" else "l",
                      fill=YOSHINO_COLOR, stroke=8, stroke_fill=(255, 255, 255), outer=(3, YOSHINO_DARK),
                      glow=6, glow_col=(90, 78, 72), glow_k=0.45, spacing=2)
    lay, xs, widths = info
    lf = to_f(lay)
    out = np.zeros_like(lf)
    # contiguous, non-overlapping slices so each glyph (with its stroke) is animated exactly once
    cuts = [int(xs[0]) - 40] + [int(round(x - 6)) for x in xs[1:]] + [int(xs[-1] + widths[-1]) + 40]
    for i in range(len(xs)):
        u = clamp01((tau - i * 0.07) / 0.3)
        if u <= 0:
            continue
        a, b = max(0, cuts[i]), min(W, cuts[i + 1])
        piece = np.zeros_like(lf)
        piece[:, a:b] = lf[:, a:b]
        sc = 1.0 + (1 - ease_out(u)) * 0.8
        if sc > 1.001:
            cx, cy = (a + b) / 2, H * 0.735 + 50
            piece = cv2.warpAffine(piece, m_place(cx, cy, cx, cy, sc), (W, H), flags=cv2.INTER_LINEAR)
        out += piece * ease_out(u)
    over(dst, np.clip(out, 0, 1))
    ui = clamp01((tau - 0.35) / 0.4)
    if ui > 0:
        en = text_layer(NAME_EN, FONT_ROUND_B, 24, x - (6 if side == "r" else -6), H * 0.735 + 104,
                        anchor=side if side == "r" else "l", fill=(255, 255, 255), stroke=3,
                        stroke_fill=YOSHINO_DARK, spacing=4)
        text_on(dst, en, opacity=ease_out(ui), scatter=ui)
    return dst


CREDITS = [
    ("l", [("出演", "依田芳乃"), ("CV", "高田憂希")]),
    ("r", [("企画", "藤原　肇"), ("イラスト素材", "バンダイナムコ")]),
    ("l", [("Theme song", "『true my heart』"), ("歌", "ave;new feat.佐倉紗織"),
           ("作詞", "a.k.a.dRESS & 佐倉紗織"), ("作曲・編曲", "a.k.a.dRESS (ave;new)")]),
    ("r", [("映像・エフェクト", "Claude"), ("プログラム", "Python"), ("スペシャルサンクス", "すべてのプロデューサーさん")]),
]


def credit_page(dst, idx, tau, out_u):
    side, entries = CREDITS[idx]
    x = W * 0.06 if side == "l" else W * 0.94
    y = H * 0.12
    k = 0
    for label, value in entries:
        lab = text_layer(label, FONT_ROUND_B, 18, x, y, anchor=side, fill=(226, 84, 140), stroke=3,
                         stroke_fill=(255, 255, 255), spacing=2)
        val = text_layer(value, FONT_ROUND_B, 30, x, y + 30, anchor=side, fill=(255, 255, 255), stroke=4,
                         stroke_fill=(226, 92, 150), spacing=2, glow=4, glow_col=(255, 160, 200), glow_k=0.6)
        u_in = clamp01((tau - 0.15 - k * 0.18) / 0.7)
        op = ease_out(u_in) * (1 - out_u)
        text_on(dst, lab, opacity=op, scatter=u_in)
        text_on(dst, val, opacity=op, scatter=u_in)
        y += 104 if len(entries) < 4 else 96
        k += 1
    if idx == 3:
        u_in = clamp01((tau - 0.8) / 0.8)
        for j, line in enumerate(COPYRIGHT):
            c = text_layer(line, FONT_ROUND_B, 18, W * 0.94, H * 0.86 + j * 26, anchor="r", fill=(255, 255, 255),
                           stroke=3, stroke_fill=(226, 92, 150), spacing=1)
            text_on(dst, c, opacity=ease_out(u_in) * (1 - out_u), scatter=u_in)
    return dst


def starring_text(dst, t):
    x = W * 0.05
    rows = [("STARRING", FONT_ROUND_B, 18, (236, 96, 150), 0, H * 0.79),
            ("『依田芳乃』", FONT_ROUND, 34, (255, 255, 255), 4, H * 0.79 + 26),
            ("CV.高田憂希", FONT_ROUND_B, 20, (255, 255, 255), 3, H * 0.79 + 76)]
    for k, (txt, fs, sz, col, st, y) in enumerate(rows):
        u = clamp01((t - 7.15 - k * 0.15) / 0.6)
        if u <= 0:
            continue
        info = text_layer(txt, fs, sz, x, y, fill=col, stroke=max(st, 2), stroke_fill=(232, 96, 150) if st else (255, 255, 255),
                          spacing=2)
        text_on(dst, info, opacity=ease_out(u) * (1 - seg(t, 10.0, 10.4)), scatter=u)
    return dst


def copyright_text(dst, t, t0):
    u = clamp01((t - t0) / 0.7)
    if u <= 0:
        return dst
    for j, line in enumerate(COPYRIGHT):
        c = text_layer(line, FONT_ROUND_B, 14, W * 0.955, H * 0.895 + j * 20, anchor="r", fill=(236, 110, 160),
                       stroke=2, stroke_fill=(255, 255, 255), spacing=1)
        text_on(dst, c, opacity=ease_out(u), scatter=u)
    return dst


# --------------------------------------------------------------------------- code-drawn emotes


def emote_drop(dst, x, y, s, a):
    if a <= 0:
        return dst
    pts = []
    for k in range(40):
        th = k / 40 * 2 * math.pi
        r = 1 - 0.55 * max(0, math.sin(th)) ** 3
        pts.append((x + math.cos(th) * s * 0.5 * r, y - math.sin(th) * s * 0.8 * (0.9 if math.sin(th) < 0 else 1.3)))
    m_out = poly_mask([[(x + (px - x) * 1.18, y + (py - y) * 1.12) for px, py in pts]])
    fill_mask(dst, m_out, (0.28, 0.5, 0.86), a)
    fill_mask(dst, poly_mask([pts]), (0.72, 0.88, 1.0), a)
    fill_mask(dst, poly_mask([rect_poly(x - s * 0.14, y - s * 0.05, s * 0.12, s * 0.35, -15)]), (1, 1, 1), a)
    return dst


def emote_bang(dst, x, y, s, a, ang=12):
    if a <= 0:
        return dst
    c, si = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def rot(px, py):
        return (x + px * c - py * si, y + px * si + py * c)

    bar = [rot(-0.22 * s, -1.2 * s), rot(0.22 * s, -1.2 * s), rot(0.09 * s, 0.25 * s), rot(-0.09 * s, 0.25 * s)]
    big = [rot(px * 1.35 - (x if False else 0), py) for px, py in [(-0.22 * s, -1.25 * s), (0.22 * s, -1.25 * s),
                                                                (0.09 * s, 0.28 * s), (-0.09 * s, 0.28 * s)]]
    dx, dy = rot(0, 0.55 * s)
    fill_mask(dst, poly_mask([big]), (1, 1, 1), a)
    fill_mask(dst, poly_mask([bar]), (0.92, 0.12, 0.18), a)
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(dx * 16), int(dy * 16)), int(0.2 * s * 16), 255, -1, cv2.LINE_AA, 4)
    m2 = np.zeros((H, W), np.uint8)
    cv2.circle(m2, (int(dx * 16), int(dy * 16)), int(0.14 * s * 16), 255, -1, cv2.LINE_AA, 4)
    fill_mask(dst, m, (1, 1, 1), a)
    fill_mask(dst, m2, (0.92, 0.12, 0.18), a)
    return dst


def sparkle(dst, x, y, s, a, col=(1.0, 1.0, 0.92)):
    if a <= 0:
        return dst
    pts = []
    for k in range(8):
        th = k / 8 * 2 * math.pi
        r = s if k % 2 == 0 else s * 0.22
        pts.append((x + math.cos(th) * r, y + math.sin(th) * r))
    fill_mask(dst, poly_mask([pts]), col, a)
    return dst


def notes(dst, t, x, y, a):
    if a <= 0:
        return dst
    for k, (ch, dx, dy, sz, col) in enumerate((("♪", 0, 0, 54, (240, 92, 150)), ("♫", 60, -46, 46, (255, 150, 60)),
                                               ("♪", -50, -70, 40, (120, 150, 240)))):
        bob = math.sin(t * 7 + k * 1.7) * 6
        info = text_layer(ch, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz, int(x + dx), int(y + dy),
                          anchor="c", fill=col, stroke=4, stroke_fill=(255, 255, 255))
        lay = info[0]
        if bob:
            lay = np.roll(lay, int(bob), axis=0)
        over(dst, lay, a * clamp01((t - k * 0.12) * 4))
    return dst


# --------------------------------------------------------------------------- transitions


@lru_cache(None)
def _diag(ang):
    xn, yn = xy_norm()
    a = math.radians(ang)
    c = xn * W * math.cos(a) + yn * H * math.sin(a)
    return (c - c.min()) / (c.max() - c.min())


def slash_wipe(A, B, u, ang=-62, sk=None):
    c = _diag(ang)
    p = lerp(-0.12, 1.12, smooth(u))
    mB = np.clip((p - c) / 0.015, 0, 1)[..., None]
    out = A * (1 - mB) + B * mB
    band = np.exp(-((c - p) / 0.06) ** 2)[..., None]
    out = out * (1 - band * 0.85) + band * 0.85 * np.float32([1.0, 0.92, 0.96])
    if sk is not None:
        over(out, to_f(sk) * band, 1.0)
    return out


def transition(A, B, u, kind):
    if kind == "fade":
        return mix(A, B, smooth(u))
    if kind == "white":
        w = (1 - abs(2 * u - 1)) ** 0.6
        return to_white(mix(A, B, smooth(u)), w)
    if kind == "blur":
        base = A if u < 0.5 else B
        amt = 0.35 * (1 - abs(2 * u - 1))
        return to_white(zoom_blur(base, amt), 0.35 * (1 - abs(2 * u - 1)))
    if kind == "wipe":
        return slash_wipe(A, B, u)
    if kind == "pink":
        w = 1 - abs(2 * u - 1)
        return wash(mix(A, B, smooth(u)), w * 0.9)
    return B


# --------------------------------------------------------------------------- shots


def kb(name, p0, p1, t0, t1, wash_k=0.0, adj=None, ang=0.0, extra=None, sk=0.0):
    """Ken-Burns shot from (cx, cy, zoom) p0 to p1 over [t0, t1]."""
    def render(t):
        u = smooth(seg(t, t0, t1)) * 0.6 + seg(t, t0, t1) * 0.4
        cx, cy, z = (lerp(a, b, u * 2.5) if i < 2 else lerp(a, b, u) for i, (a, b) in enumerate(zip(p0, p1)))
        z *= (1 + 0.20 * u) * (1 + 0.14 * (1 - ease_out(seg(t, t0, t0 + 0.4))))
        fr = cover(name, cx, cy, z, ang)
        if adj:
            fr = adjust(fr, **adj(u))
        if sk > 0:
            w, h = size_of(name)
            s = max(W / w, H / h) * z
            hw, hh = W / (2 * s), H / (2 * s)
            cxx = min(max(cx, hw), w - hw)
            cyy = min(max(cy, hh), h - hh)
            fr = to_white(fr, sk * 0.8)
            over(fr, sketch_layer(name, cxx, cyy, W / 2, H / 2, s), sk)
        wk = wash_k(u) if callable(wash_k) else wash_k
        if wk:
            fr = wash(fr, wk)
        if extra:
            extra(fr, t)
        return fr
    render.params = (name, p0, p1, t0, t1, wash_k, adj, ang, extra, sk)
    return render


def fz(name, zoom):
    x, y = face(name)
    return (x, y, zoom)


def fzo(name, zoom, dx=0, dy=0):
    x, y = face(name)
    return (x + dx, y + dy, zoom)


class Shot:
    def __init__(self, t0, render, tin="cut", tdur=0.0):
        self.t0, self.render, self.tin, self.tdur = t0, render, tin, tdur


def run_shots(shots, t):
    i = 0
    for k, s in enumerate(shots):
        if t >= s.t0:
            i = k
    s = shots[i]
    cur = s.render(t)
    if i > 0 and s.tin != "cut" and t < s.t0 + s.tdur:
        prev = shots[i - 1].render(t)
        cur = transition(prev, cur, (t - s.t0) / s.tdur, s.tin)
    return cur


def _drop(fr, t):
    return emote_drop(fr, 760, 210, 70, smooth(seg(t, 54.5, 54.62)))


def _notes18(fr, t):
    return notes(fr, t - 56.7, 700, 210, smooth(seg(t, 56.7, 56.9)))


def _notes12(fr, t):
    return notes(fr, t - 71.35, 830, 200, smooth(seg(t, 71.35, 71.5)))


def _bang(fr, t):
    emote_bang(fr, 820, 300, 110, smooth(seg(t, 77.22, 77.3)))
    return fr


def _sparkles(fr, t):
    for k, (x, y) in enumerate(((180, 160), (860, 120), (900, 520), (120, 560))):
        a = max(0.0, math.sin((t - 60.2) * 5 + k * 1.7))
        sparkle(fr, x, y, 26 + 10 * a, a * 0.9)
    return fr


MONTAGE = [
    Shot(44.6, kb("card9", fzo("card9", 1.25, 60, 20), fzo("card9", 1.38, -40, 0), 44.6, 47.5)),
    Shot(47.45, kb("card1", fz("card1", 1.45), fzo("card1", 1.6, -10, -15), 47.45, 49.7), "white", 0.5),
    Shot(49.65, kb("card7", fz("card7", 1.9), fzo("card7", 2.05, -10, -10), 49.65, 51.4,
                   adj=lambda u: dict(bright=0.12 * (1 - u), warm=0.06 * (1 - u))), "fade", 0.3),
    Shot(51.35, kb("card21", fzo("card21", 1.3, 0, 90), fzo("card21", 1.38, 10, 80), 51.35, 52.3), "fade", 0.25),
    Shot(52.2, kb("card21", fz("card21", 2.0), fzo("card21", 2.1, 10, -8), 52.2, 53.0), "fade", 0.25),
    Shot(53.0, kb("card11", fzo("card11", 1.25, 0, 30), fz("card11", 1.9), 53.0, 54.1), "fade", 0.12),
    Shot(54.45, kb("card11", fzo("card11", 2.1, 30, -20), fzo("card11", 2.2, 30, -20), 54.45, 54.95, extra=_drop),
         "white", 0.35),
    Shot(54.95, kb("card11", fzo("card11", 3.2, 0, 70), fzo("card11", 3.4, 0, 70), 54.95, 55.5, wash_k=0.55), "fade", 0.12),
    Shot(55.7, kb("card18", fzo("card18", 1.55, 20, 0), fzo("card18", 1.7, 0, -20), 55.7, 58.25, extra=_notes18),
         "white", 0.4),
    Shot(58.75, kb("card8", fzo("card8", 1.35, -30, 80), fzo("card8", 1.5, 40, -20), 58.75, 60.25), "white", 0.6),
    Shot(60.2, kb("card4", fz("card4", 1.75), fzo("card4", 1.9, 0, -10), 60.2, 62.05, extra=_sparkles), "fade", 0.25),
    Shot(62.0, kb("card15", fz("card15", 2.1), fz("card15", 2.2), 62.0, 62.7, wash_k=0.75, sk=0.6), "fade", 0.15),
    Shot(62.65, kb("card15", fz("card15", 1.75), fzo("card15", 1.82, -10, -5), 62.65, 63.5), "fade", 0.15),
    Shot(63.6, kb("card15", fzo("card15", 1.45, -60, 40), fzo("card15", 1.55, -80, 20), 63.6, 65.3), "white", 0.3),
    Shot(65.25, kb("card10", fzo("card10", 1.15, 0, 40), fzo("card10", 1.2, 0, 40), 65.25, 66.6,
                   wash_k=lambda u: 0.9 - 0.4 * u, sk=0.7), "white", 0.5),
    Shot(66.5, kb("card10", fzo("card10", 1.18, 0, 50), fzo("card10", 1.22, 0, 50), 66.5, 67.25), "fade", 0.35),
    Shot(67.2, kb("card10", fz("card10", 2.2), fzo("card10", 2.3, 0, -5), 67.2, 68.0)),
    Shot(67.95, kb("card5", fzo("card5", 1.3, 0, -20), fzo("card5", 1.3, 0, -20), 67.95, 68.25), "fade", 0.3),
    Shot(68.2, kb("card5", fz("card5", 2.2), fz("card5", 2.3), 68.2, 68.75), "fade", 0.3),
    Shot(68.7, kb("card5", fzo("card5", 1.5, 0, -20), fzo("card5", 1.55, 0, -20), 68.7, 69.25), "fade", 0.2),
    Shot(69.2, kb("card13", fz("card13", 1.7), fz("card13", 1.75), 69.2, 69.75, wash_k=0.8, sk=0.5), "fade", 0.15),
    Shot(69.7, kb("card13", fzo("card13", 1.45, 0, 30), fz("card13", 1.6), 69.7, 71.1), "fade", 0.2),
    Shot(71.1, kb("card12", fzo("card12", 1.5, 40, -40), fzo("card12", 1.6, 30, -40), 71.1, 72.85, extra=_notes12),
         "blur", 0.35),
    Shot(72.85, kb("card16", fzo("card16", 1.6, 0, 20), fzo("card16", 1.72, -10, 10), 72.85, 74.6), "blur", 0.35),
    Shot(74.5, kb("card2", fzo("card2", 1.45, 0, 30), fz("card2", 1.6), 74.5, 76.25), "wipe", 0.35),
    Shot(76.2, kb("card2", fz("card2", 2.4), fzo("card2", 2.5, 0, -6), 76.2, 76.75), "fade", 0.2),
    Shot(76.72, kb("card17", fz("card17", 2.3), fzo("card17", 2.6, 10, 0), 76.72, 76.97)),
    Shot(76.97, kb("card6", fz("card6", 1.9), fzo("card6", 2.1, -10, 0), 76.97, 77.22)),
    Shot(77.22, kb("card19", fzo("card19", 1.7, 40, 20), fzo("card19", 1.8, 40, 20), 77.22, 77.47, extra=_bang)),
    Shot(77.47, kb("card20", fz("card20", 2.0), fzo("card20", 2.15, 0, -6), 77.47, 77.72)),
    Shot(77.72, kb("card22", fz("card22", 2.1), fz("card22", 2.3), 77.72, 77.97)),
    Shot(77.97, kb("card3", fz("card3", 2.0), fzo("card3", 2.3, 10, 0), 77.97, 78.22)),
    Shot(78.22, kb("Yoshino SSR4", fz("Yoshino SSR4", 1.8), fz("Yoshino SSR4", 2.1), 78.22, 78.47)),
    Shot(78.47, kb("card8", fz("card8", 2.2), fz("card8", 2.5), 78.47, 78.72)),
]


def add_punch_ins(shots, end):
    """Insert a tighter framing of the same card on a beat in the middle of each long shot."""
    out = []
    for i, sh in enumerate(shots):
        out.append(sh)
        t1 = shots[i + 1].t0 if i + 1 < len(shots) else end
        params = getattr(sh.render, "params", None)
        if params is None or t1 - sh.t0 < 1.3:
            continue
        cands = [float(b) for b in BEAT_T if sh.t0 + 0.6 < b < t1 - 0.45]
        if not cands:
            continue
        b = cands[len(cands) // 2]
        name, p0, p1, _, _, wash_k, adj, ang, extra, sk = params
        z = max(p0[2], p1[2]) * 1.32
        fx, fy = face(name)
        out.append(Shot(b, kb(name, (fx, fy, z), (fx, fy - 6, z * 1.05), b, t1, wash_k, adj, ang, extra, sk)))
    return out


def nearest_beat(t, tol=0.2):
    i = int(np.argmin(np.abs(BEAT_T - t)))
    return float(BEAT_T[i]) if abs(BEAT_T[i] - t) <= tol else t


EIGHTHS = np.sort(np.concatenate([BEAT_T, (BEAT_T[:-1] + BEAT_T[1:]) / 2]))


def snap_cuts(shots, end):
    """Snap cut points to the beat grid; fast cuts (< 0.45 s) use the half-beat grid. Order is kept."""
    orig = [sh.t0 for sh in shots]
    for i in range(1, len(shots)):
        gap = min(orig[i] - orig[i - 1], (orig[i + 1] if i + 1 < len(orig) else end) - orig[i])
        grid, tol = (EIGHTHS, 0.12) if gap < 0.45 else (BEAT_T, 0.2)
        j = int(np.argmin(np.abs(grid - orig[i])))
        cand = float(grid[j]) if abs(grid[j] - orig[i]) <= tol else orig[i]
        if cand > shots[i - 1].t0 + 0.15:
            shots[i].t0 = cand
    return shots


MONTAGE = add_punch_ins(snap_cuts(MONTAGE, 78.72), 78.72)


def action_shot(t):
    u = seg(t, 78.72, 79.85)
    z = lerp(1.0, 1.08, u)
    fr = cover("action", 590, 442, z)
    amt = 0.45 * (1 - smooth(seg(t, 78.72, 79.0))) + 0.08
    fr = zoom_blur(fr, amt * 0.6, n=8)
    return fr


MONTAGE.append(Shot(78.72, action_shot, "blur", 0.25))


# --------------------------------------------------------------------------- segments


def sketch_tinted(dst, name, fx, fy, dx, dy, scale, ang, color, opacity, reveal=None):
    """Pink line-art of an asset, recoloured, optionally revealed outward from (dx, dy)."""
    if opacity <= 0.001:
        return dst
    lay = to_f(sketch_layer(name, fx, fy, dx, dy, scale, ang))
    a = lay[..., 3:4]
    if reveal is not None:
        xn, yn = xy_norm()
        d = np.sqrt((xn * W - dx) ** 2 + (yn * H - dy) ** 2)
        a = a * np.clip((reveal - d) / 60.0, 0, 1)[..., None]
    col = np.concatenate([a * np.float32(color), a], -1)
    return over(dst, col, opacity)


# opening line-art: one sketch per half beat, each replacing the previous
# (asset, face x, face y as frame fraction, scale, angle from -> to, drift dx, dy, colour, start)
OPEN_SKETCHES = [
    ("ysn3", 0.42, 0.45, 0.51, -6, -2, 60, -14, (0.84, 0.30, 0.78), 0.894),
    ("Yoshino SSR4", 0.62, 0.40, 1.5, 5, 2, -70, 8, (0.95, 0.40, 0.66), 1.120),
    ("ysn2", 0.38, 0.50, 0.52, -9, -5, 50, -24, (0.70, 0.42, 0.90), 1.347),
    ("ysn5", 0.60, 0.42, 0.56, 4, 1, -60, -10, (0.90, 0.34, 0.72), 1.603),
    ("ysn1", 0.45, 0.46, 0.55, -4, 0, 40, 10, (0.86, 0.38, 0.80), 1.858),
]
OPEN_END = 2.6


def seg_opening(t):
    if t < 0.35:
        return full(0.0)
    if t < 0.9:
        return full(smooth(seg(t, 0.35, 0.9)) ** 1.4)
    fr = to_white(checker_at(t), 0.45)
    starts = [o[-1] for o in OPEN_SKETCHES] + [OPEN_END]
    for i, (name, x, y, sc, a0, a1, ddx, ddy, col, tb) in enumerate(OPEN_SKETCHES):
        te = starts[i + 1]
        if t < tb or t > te + 0.07:
            continue
        u = seg(t, tb, te)
        if name in CARD_FACE:  # card art: scale relative to cover size
            w, h = size_of(name)
            s = max(W / w, H / h) * sc
        else:
            s = sc
        fx, fy = face(name)
        dx = x * W + ddx * ease_out(u)
        dy = y * H + ddy * ease_out(u)
        rv = lerp(0, 1400, ease_out(seg(t, tb, tb + 0.16)))
        op = smooth(seg(t, tb, tb + 0.05)) * (1 - seg(t, te, te + 0.07))
        sketch_tinted(fr, name, fx, fy, dx, dy, s * (1 + 0.06 * u), lerp(a0, a1, ease_out(u)), col, op, reveal=rv)
    fr = to_white(fr, pulse(t, 2.15, 2.27, 2.5) * 0.85)
    return fr


def brand_logo(dst, op, t, scale=0.78, tag_t0=None):
    w, h = size_of("logo_brand")
    s = scale * W / w
    lay = warp_layer("logo_brand", 512, 372, W / 2, H * 0.47, s)
    over(dst, lay, op)
    if tag_t0 is not None:
        u = clamp01((t - tag_t0) / 0.6)
        if u > 0:
            text_on(dst, tagline_layer(), opacity=op * ease_out(u), scatter=u)
    return dst


def seg_brand(t):
    if t < 2.6:
        fr = seg_opening(t)
        fr = to_white(fr, seg(t, 2.3, 2.6) * 0.5)
        base = to_white(checker_at(t), 0.25)
        fr = mix(fr, base, smooth(seg(t, 2.3, 2.6)))
    else:
        fr = to_white(checker_at(t), 0.25)
    # white then soft squares (3.15-4.0)
    fr = to_white(fr, smooth(seg(t, 3.1, 3.3)))
    if t > 3.3:
        squares(fr, t, SQ_SOFT, grow=lerp(0.7, 1.0, smooth(seg(t, 3.3, 4.0))), opacity=smooth(seg(t, 3.3, 3.9)))
    # Yoshino 1 (4.0 - 5.0): cut-out at upper left, face clear of the logo
    if 3.95 < t < 5.1:
        u = ease_out(seg(t, 3.95, 4.3))
        out_u = ease_in(seg(t, 4.8, 5.05))
        cl = fr.copy()
        put_reveal(cl, "ysn4", lerp(120, 195, u) + 700 * out_u, 207, 0.31, 1.0, seg(t, 4.087, 4.37))
        fr = mix(fr, cl, smooth(seg(t, 3.95, 4.15)) * (1 - smooth(seg(t, 4.9, 5.05))))
    # vivid squares (5.0 - 6.0)
    if t > 4.85:
        base = full(0, (0.99, 0.78, 0.87))
        squares(base, t, SQ_VIVID, grow=lerp(0.8, 1.2, seg(t, 4.85, 6.6)))
        fr = mix(fr, base, smooth(seg(t, 4.85, 5.05)))
    # Yoshino 2 (6.0 - 6.6): cut-out at upper right
    if t > 5.9:
        u = ease_out(seg(t, 5.9, 6.3))
        cl = fr.copy()
        put_reveal(cl, "ysn6", lerp(900, 830, u), 207, 0.40, 1.0, seg(t, 6.258, 6.5))
        fr = mix(fr, cl, smooth(seg(t, 5.9, 6.1)))
    lop = smooth(seg(t, 2.3, 2.6))
    if t < 2.6:
        sc = lerp(0.9, 0.78, ease_out(seg(t, 2.3, 2.6)))
    else:
        sc = 0.78
    brand_logo(fr, lop, t, sc, tag_t0=2.5)
    return fr


def seg_starring(t):
    base = full(0, (0.99, 0.80, 0.88))
    squares(base, t, SQ_VIVID, grow=1.1)
    squares(base, t * 0.7, SQ_SOFT, grow=1.3, opacity=0.7)
    for name, x, y, sc, t0 in (("tachie4", 240, 250, 1.7, 6.85), ("ysn5", 585, 395, 0.32, 7.15),
                               ("tachie3", 830, 560, 1.4, 7.0)):
        put(base, name, x, y, scale=sc, opacity=smooth(seg(t, t0, t0 + 0.45)))
    starring_text(base, t)
    base = to_white(base, smooth(seg(t, 10.0, 10.45)))
    return base


QUICK = [("ysn2", 11.72, 12.05, 2.6), ("card21", 12.05, 12.38, 2.4), ("card16", 12.38, 12.7, 2.3), ("card2", 12.7, 13.0, 2.4)]


def shrine(t, z0, z1, t0, t1):
    """Night sakura avenue (user-supplied background), slowly pushing toward the vanishing point."""
    u = seg(t, t0, t1)
    fr = cover("bg_sakura_night", 700, 470, lerp(z0, z1, u))
    petals(fr, t, n=55, seed=2, opacity=0.85)
    return fr


def seg_shrine(t):
    if t < 11.72:
        fr = shrine(t, 1.0, 1.07, 10.45, 11.7)
        fr = to_white(fr, 1 - smooth(seg(t, 10.45, 10.8)))
        fr = to_white(fr, smooth(seg(t, 11.4, 11.62)))
        return fr
    if t < 13.0:
        for i, (name, a, b, z) in enumerate(QUICK):
            if a <= t < b or (i == len(QUICK) - 1 and t >= a):
                u = seg(t, a, b)
                x, y = face(name)
                fr = cover(name, x + lerp(-10, 10, u), y + 20, z * (1 + 0.05 * u))
                if i == 0:
                    fr = to_white(fr, 1 - smooth(seg(t, 11.72, 11.85)))
                if i > 0 and t < a + 0.07:
                    pn, pa, pb, pz = QUICK[i - 1]
                    px, py = face(pn)
                    fr = mix(cover(pn, px + 10, py + 20, pz * 1.05), fr, seg(t, a, a + 0.07))
                petals(fr, t, n=30, seed=4, scale=1.6)
                if i == 3:
                    fr = soft_pink(fr, 0.25)
                return fr
    fr = shrine(t, 1.08, 1.13, 13.0, 14.0)
    fr = mix(cover("card2", face("card2")[0], face("card2")[1] + 20, 2.5), fr, smooth(seg(t, 13.0, 13.12)))
    fr = to_white(fr, smooth(seg(t, 13.3, 14.0)), (1.0, 0.92, 0.95))
    return fr


def group_frame(t, z):
    return cover("group", 590, 442, z)


def seg_group(t):
    z = lerp(1.12, 1.0, ease_out(seg(t, 14.0, 16.0))) + 0.03 * seg(t, 16.0, 19.5)
    fr = group_frame(t, z)
    if t < 15.7:
        k = 1 - smooth(seg(t, 14.9, 15.7))
        fr = wash(blur(fr, 6 * k), 0.95 * k)
        fr = to_white(fr, 0.55 * k * (1 - seg(t, 14.0, 15.0) * 0.3), (1.0, 0.92, 0.95))
    petals(fr, t, n=45, seed=6, opacity=smooth(seg(t, 15.0, 15.6)))
    # heart ribbon draw-on
    if 17.0 <= t < 18.45:
        heart_ribbon(fr, seg(t, 17.0, 18.0), st=0.84, glow=0.6)
    # logo hit
    if t >= 18.0:
        hit = pulse(t, 18.0, 18.2, 18.55)
        lu = ease_out(seg(t, 18.15, 18.6))
        if t > 18.15:
            title_logo(fr, st=lerp(1.25, 0.84, lu), opacity=smooth(seg(t, 18.15, 18.3)), glow=0.8 * (1 - lu) + 0.25)
        fr = zoom_blur(fr, 0.32 * hit)
        fr = to_white(fr, 0.4 * hit)
    if t > 19.2:
        k = smooth(seg(t, 19.2, 19.8))
        bgp = wash(blur(group_frame(t, z), 14), 0.9)
        bgp = to_white(bgp, 0.3 + 0.4 * seg(t, 19.6, 20.0), (1.0, 0.93, 0.96))
        title_logo(bgp, st=0.84, glow=0.25)
        fr = mix(fr, bgp, k)
    return fr


def seg_title(t):
    fr = checker_at(t, 2.4)
    sq = smooth(seg(t, 22.8, 24.0))
    if sq > 0:
        squares(fr, t, SQ_SOFT, grow=lerp(0.5, 1.25, seg(t, 22.8, 26.0)), opacity=sq)
        squares(fr, t * 0.8, SQ_VIVID[:2], grow=lerp(0.4, 1.0, seg(t, 23.0, 26.0)), opacity=sq * 0.6)
    if t < 20.13:
        fr = to_white(fr, 0.5, (1.0, 0.93, 0.96))
        title_logo(fr, st=0.84, opacity=0.8, flip=True, glow=0.3)
    elif t < 20.75:
        u = seg(t, 20.13, 20.75)
        fr = to_white(fr, 0.6 * (1 - u))
        title_logo(fr, st=lerp(0.9, 0.82, ease_out(u)), opacity=0.35 + 0.65 * smooth(seg(t, 20.5, 20.75)),
                   mono=0.6 * (1 - u))
    else:
        title_logo(fr, st=0.82 - 0.02 * seg(t, 20.75, 26.0))
    copyright_text(fr, t, 21.95)
    fr = to_white(fr, smooth(seg(t, 25.9, 26.5)))
    return fr


INTROS = [  # start, closeup source, closeup zoom, pan (dx0, dy0, dx1, dy1), tachie, side of tachie
    (26.5, "Yoshino SSR1", 2.4, (-30, 30, 20, -10), "Yoshino SSR1 tachie", "l"),
    (30.2, "Yoshino SSR2", 2.25, (40, 20, -20, -10), "Yoshino SSR2 tachie", "r"),
    (33.7, "Yoshino SSR3", 2.4, (-20, 40, 20, 0), "Yoshino SSR3 tachie", "l"),
    (37.2, "Yoshino SSR4", 1.55, (40, 30, -30, -10), "Yoshino SSR4 tachie", "r"),
    (40.7, "Yoshino SSR5", 2.9, (-5, 5, 50, 45), "Yoshino SSR5 tachie", "l"),
]
INTRO_END = 44.6
INTRO_LEN = [INTROS[i + 1][0] - INTROS[i][0] for i in range(4)] + [INTRO_END - INTROS[4][0]]


def face_at(name, zoom, tx, ty, dx=0.0, dy=0.0):
    """Source centre that puts the face of `name` at screen point (tx, ty) for a cover() view."""
    x, y = face(name)
    w, h = size_of(name)
    s = max(W / w, H / h) * zoom
    return x - (tx - W / 2) / s + dx, y - (ty - H / 2) / s + dy


def closeup_view(k, tau):
    start, name, z, (dx0, dy0, dx1, dy1), _, side = INTROS[k]
    u = clamp01(tau / (INTRO_LEN[k] + 0.4))
    zz = z * (1 + 0.15 * u)
    pb = next_beat(start + 0.7)
    if pb is not None and start + tau >= pb:
        zz *= 1.12
    tx = W * 0.64 if side == "l" else W * 0.36  # keep the face clear of the standing art
    cx, cy = face_at(name, zz, tx, H * 0.40, lerp(dx0, dx1, u) * 1.2, lerp(dy0, dy1, u) * 1.2)
    return name, cx, cy, zz


def intro_frame(k, t):
    start, name, z, pan, tachie, side = INTROS[k]
    tau = t - start
    nm, cx, cy, zz = closeup_view(k, tau)
    fr = cover(nm, cx, cy, zz)
    if zz > 2.5:  # strong upscale (SSR5: she is small in the card) -> light unsharp mask
        fr = np.clip(fr + (fr - cv2.GaussianBlur(fr, (0, 0), 1.6)) * 0.6, 0, 1)
    fr = soft_pink(fr, 0.12)
    xn, yn = xy_norm()
    ht = smooth(clamp01((tau - 1.1) / 0.5))
    if ht > 0:
        if side == "l":
            field = np.clip(1.15 - xn / 0.48, 0, 1) * ht
        else:
            field = np.clip(1.15 - (1 - xn) / 0.48, 0, 1) * ht
        halftone(fr, field, (1.0, 1.0, 1.0), 0.55, cell=22)
        halftone(fr, np.clip((yn - 0.72) / 0.28, 0, 1) * ht, (0.98, 0.62, 0.78), 0.5, cell=16, ang=30, square=False)
    entry = nearest_beat(start + 1.45, 0.25) - start
    tu = ease_out(clamp01((tau - entry) / 0.45))
    if tu > 0:
        tx = W * 0.20 if side == "l" else W * 0.80
        off = (1 - tu) * (-160 if side == "l" else 160)
        w, h = size_of(tachie)
        fx = face(tachie)[0] if tachie in CUT_FACE else 0.45 * w
        place(fr, tachie, fx, 0, tx + off, 6, 0.98, 0, tu)
        if tu < 1:
            place(fr, tachie, fx, 0, tx + off * 2.2, 6, 0.98, 0, (1 - tu) * 0.35)
    petal_shower(fr, tau, n=56, seed=100 + k)  # each new CG brings a fall of petals
    name_plate(fr, tau - entry - 0.15, "r" if side == "l" else "l")
    if k == 0:
        fr = to_white(fr, 1 - smooth(seg(t, 26.5, 27.05)))
    return fr


def seg_intros(t):
    k = 0
    for i, it in enumerate(INTROS):
        if t >= it[0]:
            k = i
    fr = intro_frame(k, t)
    if k > 0 and t < INTROS[k][0] + 0.38:
        prev = intro_frame(k - 1, t)
        nm, cx, cy, zz = closeup_view(k, t - INTROS[k][0])
        w, h = size_of(nm)
        s = max(W / w, H / h) * zz
        sk = sketch_layer(nm, cx, cy, W / 2, H / 2, s)
        fr = slash_wipe(prev, fr, seg(t, INTROS[k][0], INTROS[k][0] + 0.38), ang=-62 if k % 2 else 242, sk=sk)
    return fr


def seg_montage(t):
    if t < INTRO_END + 0.38:
        prev = intro_frame(4, t)
        cur = MONTAGE[0].render(t)
        sk = sketch_layer("card9", face("card9")[0] + 60, face("card9")[1] + 20, W / 2, H / 2,
                          max(W / 1280, H / 824) * 1.25)
        return slash_wipe(prev, cur, seg(t, INTRO_END, INTRO_END + 0.38), ang=-62, sk=sk)
    fr = run_shots(MONTAGE, t)
    # white flashes between beats
    fr = to_white(fr, pulse(t, 54.0, 54.3, 54.5))
    fr = to_white(fr, pulse(t, 55.4, 55.6, 55.9))
    if 58.1 < t < 58.9:
        u = seg(t, 58.15, 58.8)
        fr = to_white(fr, pulse(t, 58.1, 58.3, 58.85))
        for k, (col, ang0) in enumerate((((0.95, 0.3, 0.75), 0.3), ((1.0, 0.82, 0.2), 2.4), ((0.98, 0.5, 0.2), 4.2))):
            r = 60 + 520 * ease_out(u)
            x = W / 2 + math.cos(ang0) * r * 0.9
            y = H / 2 + math.sin(ang0) * r * 0.6
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (int(x), int(y)), (int(90 + 140 * u), int(40 + 60 * u)), math.degrees(ang0), 0, 360, 255, -1,
                        cv2.LINE_AA)
            m = cv2.GaussianBlur(m, (0, 0), 6)
            fill_mask(fr, m, col, 0.8 * (1 - u))
        petals(fr, t, n=60, seed=9, scale=2.0, burst=(0.5, 0.5, ease_out(u)), opacity=1 - u)
    if 70.95 < t < 71.15:
        fr = zoom_blur(fr, 0.3 * pulse(t, 70.95, 71.1, 71.15))
    fr = to_white(fr, smooth(seg(t, 79.5, 79.85)))
    return fr


def seg_waterfall(t):
    u = seg(t, 79.8, 84.2)
    fr = cover("bg_waterfall", lerp(640, 760, u), 380, lerp(1.0, 1.06, u))
    fl = flare()
    shift = int(lerp(-40, 40, u))
    fr = screen(fr, np.roll(fl, shift, axis=1), 0.75)
    fr = to_white(fr, 1 - smooth(seg(t, 79.8, 80.2)))
    if t > 83.35:
        x, y = face("ysn5")
        cl = cover("ysn5", x + lerp(-20, 20, seg(t, 83.35, 85.0)), y + 40, 2.4)
        cl = wash(cl, 0.85)
        cl = to_white(cl, 0.25)
        fr = mix(fr, cl, smooth(seg(t, 83.35, 84.0)))
    fr = to_white(fr, smooth(seg(t, 84.7, 85.15)))
    return fr


CREDIT_PAGES = [(85.2, "card17", 1.25, (0, 40)), (89.0, "card20", 1.3, (0, 60)), (92.6, "card16", 1.3, (40, 40)),
                (96.0, "card22", 1.35, (-60, 60))]


def credit_bg(i, t):
    t0, name, z, (dx, dy) = CREDIT_PAGES[i]
    u = seg(t, t0, t0 + 4.2)
    side = CREDITS[i][0]
    zz = z * (1 + 0.12 * u)
    tx = W * 0.66 if side == "l" else W * 0.34  # face on the side opposite the text
    cx, cy = face_at(name, zz, tx + lerp(-60, 60, u), H * 0.45)
    fr = cover(name, cx, cy, zz)
    fr = wash(fr, 0.62)
    fr = to_white(fr, 0.18, (1.0, 0.9, 0.95))
    xn, yn = xy_norm()
    specs = [(0.18 if side == "l" else 0.82, 0.38, 0.75, 18, 3, (0.95, 0.42, 0.66), 0.42),
             (0.10 if side == "l" else 0.90, 0.85, 0.5, -10, -4, (1.0, 0.78, 0.88), 0.5),
             (0.6, -0.05, 0.5, 30, 5, (0.98, 0.6, 0.78), 0.3)]
    squares(fr, t - t0, specs)
    return fr


def seg_credits(t):
    i = 0
    for k, p in enumerate(CREDIT_PAGES):
        if t >= p[0]:
            i = k
    t0 = CREDIT_PAGES[i][0]
    t1 = CREDIT_PAGES[i + 1][0] if i + 1 < len(CREDIT_PAGES) else 99.1
    fr = credit_bg(i, t)
    if i > 0 and t < t0 + 0.45:
        prev = credit_bg(i - 1, t)
        credit_page(prev, i - 1, t - CREDIT_PAGES[i - 1][0], 1.0)
        fr = mix(prev, fr, smooth(seg(t, t0, t0 + 0.45)))
    credit_page(fr, i, t - t0, smooth(seg(t, t1 - 0.4, t1)) if i < 3 else 0.0)
    if i == 0:
        fr = to_white(fr, 1 - smooth(seg(t, 85.2, 85.6)))
    fr = to_white(fr, smooth(seg(t, 98.7, 99.1)), (0.93, 0.92, 0.93))
    return fr


def seg_finale(t):
    if t < 100.0:
        g = 0.93
        if t > 99.3:
            g = lerp(0.93, 0.03, smooth(seg(t, 99.3, 99.8)))
        fr = full(g)
        heart_ribbon(fr, seg(t, 99.28, 99.98), st=0.84, glow=1.0, tint=(1.0, 0.92, 0.96))
        return fr
    if t < 101.0:
        g = lerp(0.03, 0.42, smooth(seg(t, 100.15, 100.7)))
        fr = full(0, (g, g * 0.94, g * 0.97))
        fr = to_white(fr, smooth(seg(t, 100.7, 101.0)) * 0.9, (1.0, 0.93, 0.96))
        hit = pulse(t, 100.0, 100.08, 100.3)
        title_logo(fr, st=lerp(1.1, 0.84, ease_out(seg(t, 100.0, 100.35))), mono=0.85 * (1 - seg(t, 100.6, 101.0)),
                   bright=0.2 * (1 - seg(t, 100.5, 101.0)), glow=1.2 - 0.8 * seg(t, 100.5, 101.0))
        fr = zoom_blur(fr, 0.4 * hit)
        return fr
    fr = checker_at(t, 2.4)
    if t < 101.62:
        title_logo(fr, st=0.84, glow=0.25)
    elif t < 101.8:
        title_logo(fr, st=0.84, flip=True, opacity=0.7, glow=0.3)
        fr = to_white(fr, 0.3)
    elif t < 102.5:
        u = seg(t, 101.8, 102.5)
        fr = to_white(fr, 0.6 * (1 - u))
        title_logo(fr, st=lerp(0.9, 0.84, ease_out(u)), opacity=0.3 + 0.7 * smooth(seg(t, 102.2, 102.5)),
                   mono=0.6 * (1 - u))
    else:
        title_logo(fr, st=0.84 - 0.02 * seg(t, 102.5, 107.0), opacity=1 - smooth(seg(t, 106.9, 107.75)))
    return fr


def seg_end(t):
    fr = full(1.0)
    op = smooth(seg(t, 108.55, 109.0)) * (1 - smooth(seg(t, 112.4, 113.0)))
    brand_logo(fr, op, t, 0.62, tag_t0=108.6)
    if t > 113.15:
        g = 1 - smooth(seg(t, 113.15, 114.05))
        fr = full(g * 0.98)
    return fr


FAN = (6.33, 6.79)
FAN_BAND = 28.0  # degrees of pink ribs behind the leading edge
FAN_RIBS = (np.float32([1.0, 0.74, 0.86]), np.float32([0.96, 0.56, 0.75]))


def fan_wipe(A, B, u):
    """A folding fan opening from below the frame, left to right; its pink ribs sweep A away to reveal B."""
    cx, cy = W / 2, H + 30
    xn, yn = xy_norm()
    ang = np.degrees(np.arctan2(cy - yn * H, xn * W - cx))
    theta = lerp(185.0, -FAN_BAND - 5.0, smooth(u))
    d = ang - theta  # > 0: already swept
    wA = np.clip(-d / 1.2 + 0.5, 0, 1)[..., None]
    wB = np.clip((d - FAN_BAND) / 1.2 + 0.5, 0, 1)[..., None]
    wP = np.clip(1 - wA - wB, 0, 1)
    rib = (np.floor(np.clip(d, 0, FAN_BAND) / 4.0) % 2)[..., None]
    pink = FAN_RIBS[0] * (1 - rib) + FAN_RIBS[1] * rib
    r = np.sqrt((xn * W - cx) ** 2 + (yn * H - cy) ** 2)[..., None]
    pink = pink + (1 - pink) * np.clip((r - 300) / 900, 0, 1) * 0.35  # paler towards the outer edge
    out = A * wA + B * wB + pink * wP
    edge = np.exp(-(d / 0.9) ** 2)[..., None]
    return to_white_arr(out, edge * 0.7)


def to_white_arr(rgb, k):
    return rgb * (1 - k) + k


SEGMENTS = [
    (0.0, seg_opening), (2.3, seg_brand), (FAN[1], seg_starring), (10.45, seg_shrine), (14.0, seg_group),
    (20.0, seg_title), (26.5, seg_intros), (INTRO_END, seg_montage), (79.8, seg_waterfall), (85.2, seg_credits),
    (99.1, seg_finale), (107.75, seg_end),
]


# (start, end, screen-lift, saturation): shots measured darker or more saturated than the rest
SHOT_CORRECTIONS = [
    (10.45, 11.72, 0.22, 0.85), (13.0, 14.0, 0.22, 0.85),
    (44.6, 47.5, 0.26, 0.62), (49.65, 51.4, 0.08, 0.8), (52.2, 53.0, 0.04, 0.85),
    (55.7, 58.2, 0.02, 0.8), (58.75, 60.3, 0.14, 0.75), (72.85, 74.75, 0.22, 0.8),
    (77.72, 77.97, 0.2, 0.85), (79.8, 83.6, 0.16, 1.0), (5.9, 6.6, 0.0, 0.8),
]


def shot_correct(fr, t):
    for a, b, lift, sat in SHOT_CORRECTIONS:
        k = smooth(seg(t, a - 0.08, a + 0.04)) * (1 - smooth(seg(t, b - 0.04, b + 0.08)))
        if k <= 0:
            continue
        out = fr + (1 - fr) * lift
        lum = (out @ np.float32([0.3, 0.59, 0.11]))[..., None]
        out = lum + (out - lum) * sat
        fr = mix(fr, np.clip(out, 0, 1), k)
    return fr


def grade(fr, g=1.0):
    """Soft, bright, pastel-pink look: diffusion, glow, lifted blacks, lower contrast and saturation."""
    if g <= 0:
        return fr
    out = fr * 0.68 + blur(fr, 1.8) * 0.32
    out = screen(out, blur(out, 16), 0.3)
    lum = (out @ np.float32([0.3, 0.59, 0.11]))[..., None]
    out = lum + (out - lum) * 0.9
    out = 0.5 + (out - 0.5) * 1.06
    out = np.clip(out, 0, 1) ** 0.88
    out = out * 0.95 + 0.05
    out = out * np.float32([1.0, 0.965, 0.975]) + np.float32([0.025, 0.0, 0.012])
    return mix(fr, np.clip(out, 0, 1), g)


def grade_amount(t):
    g = smooth(seg(t, 0.5, 1.0)) * (1 - smooth(seg(t, 113.15, 113.6)))
    if 99.3 < t < 100.75:
        g *= 0.35
    return g


def render(t):
    fn = SEGMENTS[0][1]
    for t0, f in SEGMENTS:
        if t >= t0:
            fn = f
    if FAN[0] <= t < FAN[1]:
        fr = fan_wipe(seg_brand(t), seg_starring(t), seg(t, *FAN))
    else:
        fr = fn(t)
    fr = catchcopy(fr, t)
    fr = shot_correct(np.clip(fr, 0, 1), t)
    fr = grade(fr, grade_amount(t))
    return np.clip(fr, 0, 1)
