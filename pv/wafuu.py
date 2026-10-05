"""Yoshino-palette wafuu motifs drawn in code: sakura blossoms, mizuhiki cords and knots, seigaiha waves.

All functions draw into float RGB frames (H, W, 3) in place, like the helpers in common.py.
"""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import H, W, clamp01, ease_out, fill_mask, lerp, mix, smooth

# --------------------------------------------------------------------------- palette (Yoshino image colour #C4BCB7)
YOS = np.float32([0xC4, 0xBC, 0xB7]) / 255
IVORY = np.float32([0.975, 0.958, 0.940])
GREIGE_LIGHT = np.float32([0.900, 0.875, 0.860])
TAUPE = np.float32([0.560, 0.500, 0.470])
SAKURA = np.float32([0.965, 0.800, 0.835])
SAKURA_DEEP = np.float32([0.900, 0.580, 0.660])
GOLD = np.float32([0.870, 0.730, 0.450])
VERMILION = np.float32([0.800, 0.250, 0.250])


def ivory_wash(rgb, k):
    return rgb * (1 - k) + IVORY * k


# --------------------------------------------------------------------------- sakura blossom


@lru_cache(None)
def _petal_outline(n=40):
    """One rounded petal from the centre (0, 0) to the tip (0, -1), with the sakura notch at the tip."""
    pts = []
    for i in range(n):
        ph = i / n * 2 * math.pi
        x = 0.37 * math.sin(ph)
        y = -0.52 - 0.50 * math.cos(ph)
        if y < -0.82:  # notch: pull the tip centre down
            y += 0.20 * max(0.0, 1 - abs(x) / 0.15)
        pts.append((x, y))
    return np.float32(pts)


def blossom_polys(cx, cy, r, ang_deg):
    base = _petal_outline()
    polys = []
    for k in range(5):
        a = math.radians(ang_deg) + k * 2 * math.pi / 5
        c, s = math.cos(a), math.sin(a)
        x = cx + (base[:, 0] * c - base[:, 1] * s) * r
        y = cy + (base[:, 0] * s + base[:, 1] * c) * r
        polys.append(np.stack([x, y], 1))
    return polys


def _mask(polys, blur=0.0):
    m = np.zeros((H, W), np.uint8)
    for p in polys:  # one call per polygon so overlaps are unioned, not XOR-ed
        cv2.fillPoly(m, [np.round(np.asarray(p) * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m


def blossom(dst, cx, cy, r, ang, col, opacity=1.0, centre=True, blur=0.0):
    if opacity <= 0.001 or r < 1:
        return dst
    fill_mask(dst, _mask(blossom_polys(cx, cy, r, ang), blur), col, opacity)
    if centre and r > 12:
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(cx * 16), int(cy * 16)), int(r * 0.16 * 16), 255, -1, cv2.LINE_AA, 4)
        fill_mask(dst, m, SAKURA_DEEP if col is not SAKURA_DEEP else IVORY, opacity * 0.8)
    return dst


def blossom_field(dst, t, specs, grow=1.0, opacity=1.0):
    """Large soft blossoms slowly turning: spec = (cx, cy, diameter (frac of W), ang0, angvel, colour, alpha)."""
    for cx, cy, size, a0, av, col, al in specs:
        blossom(dst, cx * W, cy * H, size * W * grow / 2, a0 + av * t, col, al * opacity, centre=False, blur=2.0)
    return dst


@lru_cache(None)
def _cover_layout(seed=5):
    rng = np.random.default_rng(seed)
    cells = []
    nx, ny = 6, 5
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5 + rng.uniform(-0.3, 0.3)) / nx * W
            y = (j + 0.5 + rng.uniform(-0.3, 0.3)) / ny * H
            d = math.hypot(x - W / 2, y - H / 2) / math.hypot(W / 2, H / 2)
            cells.append((x, y, rng.uniform(150, 190), rng.uniform(0, 72), rng.integers(0, 3), d,
                          rng.uniform(-60, 60)))
    return cells


COVER_COLS = (IVORY, SAKURA, GREIGE_LIGHT)


def sakura_cover(A, B, u):
    """Transition: blossoms bloom outward from the centre to cover A, then fall away to reveal B."""
    out = (A if u < 0.5 else B).copy()
    cover = 1 - abs(2 * u - 1)  # 0 -> 1 -> 0
    out = ivory_wash(out, smooth(clamp01(cover * 1.4 - 0.25)) * 0.9)
    for x, y, r, a0, ci, d, spin in _cover_layout():
        if u < 0.5:
            k = ease_out(clamp01((u * 2 - d * 0.45) / 0.55))
        else:
            k = 1 - smooth(clamp01(((u - 0.5) * 2 - (1 - d) * 0.45) / 0.55))
        if k <= 0:
            continue
        drift = (1 - k) * 40 if u >= 0.5 else 0.0
        blossom(out, x, y + drift, r * k, a0 + spin * u, COVER_COLS[ci], 0.95, centre=True)
    return out


def blossom_burst(dst, u, n=26, seed=9, cx=W / 2, cy=H / 2):
    """Blossoms flung outward from a point, turning and fading (u 0..1)."""
    if u <= 0 or u >= 1:
        return dst
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.3, 1.0) * 620 * ease_out(u)
        r = rng.uniform(16, 40) * (0.6 + 0.4 * u)
        col = COVER_COLS[rng.integers(0, 3)] if rng.uniform() < 0.6 else SAKURA_DEEP
        blossom(dst, cx + math.cos(a) * dist, cy + math.sin(a) * dist * 0.8, r, rng.uniform(0, 72) + 200 * u, col,
                (1 - u) ** 0.7)
    return dst


# --------------------------------------------------------------------------- mizuhiki


MIZUHIKI = (VERMILION, GOLD, IVORY)


def _normals(pts):
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)


def cords(dst, pts, width=7.0, gap=None, cols=MIZUHIKI, opacity=1.0):
    """Three parallel mizuhiki cords along a polyline (N, 2)."""
    if opacity <= 0.001 or len(pts) < 2:
        return dst
    pts = np.asarray(pts, np.float64)
    gap = width * 1.05 if gap is None else gap
    nrm = _normals(pts)
    for k, col in enumerate(cols):
        off = (k - (len(cols) - 1) / 2) * gap
        p = np.round((pts + nrm * off) * 16).astype(np.int32)
        m_edge = np.zeros((H, W), np.uint8)
        cv2.polylines(m_edge, [p], False, 255, int(width + 2), cv2.LINE_AA, 4)
        fill_mask(dst, m_edge, np.float32(col) * 0.62, opacity)
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [p], False, 255, int(width), cv2.LINE_AA, 4)
        fill_mask(dst, m, col, opacity)
        hl = np.round((pts + nrm * (off - width * 0.18)) * 16).astype(np.int32)
        m_hl = np.zeros((H, W), np.uint8)
        cv2.polylines(m_hl, [hl], False, 255, max(1, int(width * 0.25)), cv2.LINE_AA, 4)
        fill_mask(dst, m_hl, np.minimum(np.float32(col) + 0.25, 1.0), opacity * 0.7)
    return dst


@lru_cache(None)
def _knot_path(n=420):
    """Awaji-musubi-like knot: three lobes (trefoil projection) with two tails falling from the bottom."""
    t = np.linspace(0.35, 2 * math.pi - 0.35, n)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    pts = np.stack([x, -y], 1) / 3.2
    # tails: continue downward from each open end
    left = np.stack([np.linspace(pts[0, 0], -0.55, 40), np.linspace(pts[0, 1], 1.25, 40)], 1)[::-1]
    right = np.stack([np.linspace(pts[-1, 0], 0.55, 40), np.linspace(pts[-1, 1], 1.25, 40)], 1)
    return np.concatenate([left, pts, right], 0)


def knot(dst, cx, cy, size, progress=1.0, ang=0.0, opacity=1.0, width=None):
    """Draw the knot centred at (cx, cy); progress draws it on from the middle outward."""
    if opacity <= 0.001 or progress <= 0:
        return dst
    p = _knot_path()
    n = len(p)
    half = int(n / 2 * clamp01(progress))
    seg_pts = p[n // 2 - half: n // 2 + half + 1]
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    xy = np.stack([seg_pts[:, 0] * c - seg_pts[:, 1] * s, seg_pts[:, 0] * s + seg_pts[:, 1] * c], 1) * size
    xy += np.float64([cx, cy])
    return cords(dst, xy, width=width or max(2.5, size * 0.06), opacity=opacity)


def cord_band(dst, p0, p1, wave=10.0, phase=0.0, width=8.0, opacity=1.0, n=60):
    """Straight-ish cords from p0 to p1 with a gentle wave, used as a wipe edge."""
    t = np.linspace(0, 1, n)
    x = lerp(p0[0], p1[0], t)
    y = lerp(p0[1], p1[1], t)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    w = np.sin(t * 6.0 + phase) * wave
    return cords(dst, np.stack([x + nx * w, y + ny * w], 1), width=width, opacity=opacity)


# --------------------------------------------------------------------------- seigaiha


@lru_cache(None)
def seigaiha_mask(R=34):
    """Overlapping concentric-arc wave pattern as a (H, W) float mask of the lines."""
    S = 2
    im = Image.new("L", (W * S, H * S), 0)
    d = ImageDraw.Draw(im)
    r = R * S
    row = 0
    y = -r
    while y < H * S + r:
        x0 = -r if row % 2 else 0
        x = x0
        while x < W * S + 2 * r:
            d.ellipse([x - r, y - r, x + r, y + r], fill=0)
            for k, rr in enumerate((1.0, 0.74, 0.48, 0.22)):
                q = r * rr
                d.ellipse([x - q, y - q, x + q, y + q], outline=255, width=max(1, int(S * 1.6)))
            x += 2 * r
        y += r // 2
        row += 1
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def seigaiha(dst, field, col, opacity=0.5, R=34):
    m = seigaiha_mask(R) * np.clip(field, 0, 1) * opacity
    m = m[..., None]
    dst *= 1 - m
    dst += np.float32(col) * m
    return dst


def ribbon_flash(rgb, k):
    """Soft ivory flash used in place of a pure white flash."""
    return ivory_wash(rgb, k)


__all__ = [n for n in dir() if not n.startswith("_")] + ["mix"]
