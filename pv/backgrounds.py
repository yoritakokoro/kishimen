"""Code-drawn background plates and decorative layers (no image generation, no external assets).

Every plate is drawn with PIL polygons/ellipses at 2x and downsampled, then cached in build/.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import BUILD, H, W

SS = 2  # supersampling factor


def _grad(size, stops, vertical=True):
    """Linear gradient image from [(pos, (r,g,b)), ...]."""
    w, h = size
    n = h if vertical else w
    pos = np.linspace(0, 1, n)
    cols = np.zeros((n, 3), np.float32)
    ps = [p for p, _ in stops]
    for c in range(3):
        cols[:, c] = np.interp(pos, ps, [col[c] for _, col in stops])
    arr = np.broadcast_to(cols[:, None, :], (h, w, 3)) if vertical else np.broadcast_to(cols[None, :, :], (h, w, 3))
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _layer(size, col=(255, 255, 255)):
    # transparent pixels keep a matching colour so blurring does not pull in dark fringes
    return Image.new("RGBA", size, col + (0,))


def _clouds(base, rng, n, area, scale, col=(255, 255, 255), alpha=200, blur=26):
    lay = _layer(base.size)
    d = ImageDraw.Draw(lay)
    x0, y0, x1, y1 = area
    for _ in range(n):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        for _ in range(9):
            r = rng.uniform(0.5, 1.0) * scale
            ox, oy = rng.normal(0, scale * 1.1), rng.normal(0, scale * 0.25)
            d.ellipse([cx + ox - r * 1.4, cy + oy - r * 0.7, cx + ox + r * 1.4, cy + oy + r * 0.7],
                      fill=col + (alpha,))
    base.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur)))


def draw_group_backdrop(size):
    """Soft white-pink backdrop with bokeh circles for the all-Yoshino group shot."""
    BW, BH = size[0] * SS, size[1] * SS
    rng = np.random.default_rng(3)
    base = _grad((BW, BH), [(0, (250, 246, 242)), (0.6, (236, 228, 222)), (1, (222, 212, 206))]).convert("RGBA")
    lay = _layer(base.size)
    d = ImageDraw.Draw(lay)
    for _ in range(60):
        x, y = rng.uniform(0, BW), rng.uniform(0, BH)
        r = rng.uniform(20, 90) * SS
        col = [(255, 255, 255), (240, 214, 220), (232, 224, 218)][rng.integers(0, 3)]
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (int(rng.uniform(60, 150)),))
    base.alpha_composite(lay.filter(ImageFilter.GaussianBlur(5 * SS)))
    return base.convert("RGB").resize(size, Image.LANCZOS)


def draw_sky_burst(size):
    """Bright sky with radial speed lines for the action group shot."""
    BW, BH = size
    base = _grad((BW, BH), [(0, (90, 168, 240)), (0.7, (190, 228, 252)), (1, (255, 250, 240))]).convert("RGBA")
    lay = _layer(base.size)
    d = ImageDraw.Draw(lay)
    cx, cy = BW * 0.5, BH * 0.55
    rng = np.random.default_rng(5)
    for k in range(64):
        a = k / 64 * 2 * math.pi + rng.uniform(-0.03, 0.03)
        da = rng.uniform(0.01, 0.03)
        R = BW * 1.2
        d.polygon([(cx + math.cos(a) * 40, cy + math.sin(a) * 40), (cx + math.cos(a - da) * R, cy + math.sin(a - da) * R),
                   (cx + math.cos(a + da) * R, cy + math.sin(a + da) * R)], fill=(255, 255, 255, 90))
    base.alpha_composite(lay.filter(ImageFilter.GaussianBlur(2)))
    return base.convert("RGB")


def draw_flare(size=(W, H)):
    """Rainbow lens ring and glare, returned as an additive RGB float array."""
    BW, BH = size
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    cx, cy, R = BW * 0.55, BH * 1.3, BH * 0.78
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.exp(-((r - R) / (BH * 0.035)) ** 2)
    hue = np.clip((r - R) / (BH * 0.07) + 0.5, 0, 1)
    rgb = np.stack([np.clip(1.5 - np.abs(hue * 4 - 0.5), 0, 1), np.clip(1.5 - np.abs(hue * 4 - 2.0), 0, 1),
                    np.clip(1.5 - np.abs(hue * 4 - 3.5), 0, 1)], -1)
    out = rgb * ring[..., None] * 0.2
    gx, gy = BW * 0.55, -BH * 0.12
    g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (BH * 0.45) ** 2))
    out += np.float32([1.0, 0.97, 0.88]) * g[..., None] * 0.65
    for k, (u, rad, a) in enumerate(((0.35, 30, 0.10), (0.55, 18, 0.12), (0.75, 46, 0.06))):
        hx, hy = gx + (BW * 0.4 - gx) * u * 1.6, gy + (BH * 0.7 - gy) * u * 1.6
        h = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / rad ** 2) ** 2)
        out += np.float32([0.7, 0.9, 1.0]) * h[..., None] * a
    return out.astype(np.float32)


def build_all():
    BUILD.mkdir(exist_ok=True)
    if not (BUILD / "flare.npy").exists():
        np.save(BUILD / "flare.npy", draw_flare())


if __name__ == "__main__":
    build_all()
    print("built", sorted(p.name for p in BUILD.iterdir()))
