"""Shared helpers: asset loading, compositing, effects and text for the PV renderer.

All frames are float32 RGB arrays in [0, 1] with shape (H, W, 3).
Layers are premultiplied RGBA uint8 arrays.
"""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path(__file__).resolve().parent / "build"
W, H = 1024, 768
FPS = 30

FONT_ROUND = "/usr/share/fonts/opentype/mplus/Mplus2-Black.otf"
FONT_ROUND_B = "/usr/share/fonts/opentype/mplus/Mplus2-ExtraBold.otf"
FONT_ROUND_M = "/usr/share/fonts/opentype/mplus/Mplus2-Medium.otf"
FONT_SERIF = ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 0)
FONT_SERIF_R = ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0)
FONT_SANS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0)

# --------------------------------------------------------------------------- math


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def seg(t, a, b):
    """Progress of t through [a, b] as 0..1."""
    if b <= a:
        return 1.0 if t >= a else 0.0
    return clamp01((t - a) / (b - a))


def smooth(u):
    return u * u * (3 - 2 * u)


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_in(u):
    return u ** 3


def lerp(a, b, u):
    return a + (b - a) * u


def pulse(t, a, peak, b):
    """0 -> 1 between a..peak, 1 -> 0 between peak..b."""
    if t <= a or t >= b:
        return 0.0
    if t < peak:
        return smooth((t - a) / (peak - a))
    return smooth((b - t) / (b - peak))


# --------------------------------------------------------------------------- assets


def _premul(rgba):
    a = rgba[..., 3:4].astype(np.uint16)
    out = rgba.copy()
    out[..., :3] = ((rgba[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    return out


def _load_rgba(path, maxdim=None):
    im = Image.open(path).convert("RGBA")
    if maxdim and max(im.size) > maxdim:
        s = maxdim / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    return _premul(np.asarray(im))


ASSET_MAXDIM = {"ysn": 3000}


@lru_cache(None)
def img(name):
    """Premultiplied RGBA uint8 for a repo image (name without extension) or a build image."""
    BUILD.mkdir(exist_ok=True)
    cache = BUILD / f"{name}.npy"
    if cache.exists():
        return np.load(cache)
    for ext in (".png", ".jpg"):
        p = ROOT / f"{name}{ext}"
        if p.exists():
            break
    else:
        raise FileNotFoundError(name)
    maxdim = ASSET_MAXDIM["ysn"] if name.startswith("ysn") else None
    arr = _load_rgba(p, maxdim)
    np.save(cache, arr)
    return arr


def register(name, arr):
    """Store a generated layer so img(name) returns it."""
    BUILD.mkdir(exist_ok=True)
    np.save(BUILD / f"{name}.npy", arr)
    img.cache_clear()


@lru_cache(None)
def mip(name, level):
    a = img(name)
    for _ in range(level):
        a = cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    return a


def size_of(name):
    a = img(name)
    return a.shape[1], a.shape[0]


# --------------------------------------------------------------------------- geometry


def m_place(sx, sy, dx, dy, scale, ang=0.0):
    """Affine matrix mapping source point (sx, sy) to (dx, dy) with scale and rotation (deg)."""
    a = math.radians(ang)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    return np.float32([[c, -s, dx - c * sx + s * sy], [s, c, dy - s * sx - c * sy]])


def warp_layer(name, sx, sy, dx, dy, scale, ang=0.0, flip=False):
    """Warp a named asset into a full-frame premultiplied RGBA uint8 layer."""
    level = 0
    while scale * (2 ** level) < 0.5 and level < 5:
        level += 1
    src = mip(name, level)
    f = 2 ** level
    s = scale * f
    sx, sy = sx / f, sy / f
    if flip:
        src = src[:, ::-1]
        sx = src.shape[1] - sx
    interp = cv2.INTER_CUBIC if s > 1.2 else cv2.INTER_LINEAR
    out = cv2.warpAffine(src, m_place(sx, sy, dx, dy, s, ang), (W, H), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def warp_arr(arr, M, border=cv2.BORDER_CONSTANT):
    return cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def to_f(layer):
    return layer.astype(np.float32) * (1.0 / 255.0)


def over(dst, layer, opacity=1.0):
    """Composite a premultiplied RGBA (uint8 or float) layer over dst in place."""
    if opacity <= 0.001:
        return dst
    lf = to_f(layer) if layer.dtype == np.uint8 else layer
    a = lf[..., 3:4] * opacity
    dst *= 1.0 - a
    dst += lf[..., :3] * opacity
    return dst


def place(dst, name, fx, fy, dx, dy, scale, ang=0.0, opacity=1.0, flip=False, fn=None):
    """Draw asset with its point (fx, fy) (pixels in source) at (dx, dy)."""
    if opacity <= 0.001:
        return dst
    lay = warp_layer(name, fx, fy, dx, dy, scale, ang, flip)
    if fn is not None:
        lay = fn(lay)
    return over(dst, lay, opacity)


def cover(name, cx, cy, zoom=1.0, ang=0.0, clamp=True):
    """Full-frame view of an image centred on source point (cx, cy)."""
    w, h = size_of(name)
    s = max(W / w, H / h) * zoom
    if clamp:
        hw, hh = W / (2 * s), H / (2 * s)
        cx = min(max(cx, hw), w - hw) if w > 2 * hw else w / 2
        cy = min(max(cy, hh), h - hh) if h > 2 * hh else h / 2
    lay = warp_layer(name, cx, cy, W / 2, H / 2, s, ang)
    f = to_f(lay)
    rgb = f[..., :3]
    if lay[..., 3].min() < 250:
        rgb += 1.0 - f[..., 3:4]
    return rgb


def full(v, col=None):
    if col is None:
        return np.full((H, W, 3), v, np.float32)
    return np.zeros((H, W, 3), np.float32) + np.float32(col)


def mix(a, b, u):
    if u <= 0:
        return a
    if u >= 1:
        return b
    return a * (1 - u) + b * u


def to_white(rgb, u, col=(1.0, 1.0, 1.0)):
    if u <= 0:
        return rgb
    return rgb * (1 - u) + np.float32(col) * u


def screen(a, b, k=1.0):
    return 1.0 - (1.0 - a) * (1.0 - b * k)


# --------------------------------------------------------------------------- effects


def blur(rgb, sigma):
    if sigma < 0.3:
        return rgb
    if sigma > 6:
        f = 4
        small = cv2.resize(rgb, (W // f, H // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def bloom(rgb, k=0.18, sigma=18, thresh=0.55):
    small = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = np.clip((small - thresh) / (1 - thresh), 0, 1)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return screen(rgb, big, k)


def zoom_blur(rgb, amount, cx=W / 2, cy=H / 2, n=10):
    if amount <= 0.002:
        return rgb
    acc = np.zeros_like(rgb)
    for i in range(n):
        s = 1.0 + amount * i / (n - 1)
        acc += cv2.warpAffine(rgb, m_place(cx, cy, cx, cy, s), (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return acc / n


PINK_DARK = np.float32([0.60, 0.50, 0.50])  # Yoshino duotone: dusty rose-taupe
PINK_LIGHT = np.float32([0.985, 0.965, 0.955])  # ivory


def wash(rgb, k, dark=PINK_DARK, light=PINK_LIGHT, keep=0.25):
    """Pink duotone wash (k=0 none, 1 fully washed)."""
    if k <= 0:
        return rgb
    lum = rgb[..., 0:1] * 0.3 + rgb[..., 1:2] * 0.59 + rgb[..., 2:3] * 0.11
    lum = np.clip(lum * 0.75 + 0.3, 0, 1)
    duo = dark * (1 - lum) + light * lum
    duo = duo * (1 - keep) + rgb * keep
    return rgb * (1 - k) + duo * k


def adjust(rgb, bright=0.0, sat=1.0, warm=0.0):
    out = rgb
    if sat != 1.0:
        lum = rgb[..., 0:1] * 0.3 + rgb[..., 1:2] * 0.59 + rgb[..., 2:3] * 0.11
        out = lum + (out - lum) * sat
    if warm:
        out = out + np.float32([warm, warm * 0.3, -warm])
    if bright:
        out = out + bright
    return np.clip(out, 0, 1)


@lru_cache(None)
def sketch(name, sigma_div=420.0, color=(0.84, 0.30, 0.78)):
    """Pink line-art version of an asset (premultiplied RGBA uint8), via colour-dodge edge filter."""
    key = f"sketch_{name}"
    p = BUILD / f"{key}.npy"
    if p.exists():
        return np.load(p)
    a = img(name).astype(np.float32) / 255.0
    rgb = a[..., :3] + (1 - a[..., 3:4])
    gray = rgb @ np.float32([0.3, 0.59, 0.11])
    inv = 1 - gray
    sig = max(a.shape[:2]) / sigma_div
    bl = cv2.GaussianBlur(inv, (0, 0), sig)
    dodge = np.clip(gray / np.maximum(1 - bl, 1e-3), 0, 1)
    line = np.clip((1 - dodge) * 3.2, 0, 1) ** 0.85
    shade = np.clip(1 - gray, 0, 1) * 0.12
    alpha = np.clip(line + shade, 0, 1) * a[..., 3]
    out = np.empty(a.shape, np.float32)
    out[..., :3] = np.float32(color) * alpha[..., None]
    out[..., 3] = alpha
    out = (out * 255 + 0.5).astype(np.uint8)
    np.save(p, out)
    return out


def sketch_layer(name, sx, sy, dx, dy, scale, ang=0.0):
    key = f"sketch_{name}"
    sketch(name)
    return warp_layer(key, sx, sy, dx, dy, scale, ang)


@lru_cache(None)
def _ht_coords(cell, ang):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = math.radians(ang)
    u = (xx * math.cos(a) + yy * math.sin(a)) / cell
    v = (-xx * math.sin(a) + yy * math.cos(a)) / cell
    fu = u - np.round(u)
    fv = v - np.round(v)
    return np.maximum(np.abs(fu), np.abs(fv)), np.sqrt(fu * fu + fv * fv)


@lru_cache(None)
def xy_norm():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx / W, yy / H


def halftone(dst, field, color=(1.0, 1.0, 1.0), opacity=0.6, cell=20, ang=45, square=True):
    """Overlay a dot screen; field (H, W) in 0..1 sets the dot size."""
    dsq, dci = _ht_coords(cell, ang)
    d = dsq if square else dci
    r = 0.5 * np.clip(field, 0, 1)
    m = np.clip((r - d) * cell + 0.5, 0, 1) * opacity
    m = m[..., None]
    dst *= 1 - m
    dst += np.float32(color) * m
    return dst


def poly_mask(polys, shift=4):
    m = np.zeros((H, W), np.uint8)
    f = 1 << shift
    pts = [np.round(np.asarray(p, np.float64) * f).astype(np.int32) for p in polys]
    cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_AA, shift=shift)
    return m


def fill_mask(dst, mask, color, opacity=1.0):
    a = mask.astype(np.float32)[..., None] * (opacity / 255.0)
    dst *= 1 - a
    dst += np.float32(color) * a
    return dst


def square_poly(cx, cy, size, ang):
    a = math.radians(ang)
    pts = []
    for k in range(4):
        th = a + math.pi / 4 + k * math.pi / 2
        r = size / math.sqrt(2)
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def rect_poly(cx, cy, w, h, ang):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + x * c - y * s, cy + x * s + y * c))
    return pts


def squares(dst, t, specs, grow=1.0, opacity=1.0):
    """Rotating translucent squares. spec = (cx, cy, size, ang0, angvel, color, alpha)."""
    for cx, cy, size, a0, av, col, al in specs:
        m = poly_mask([square_poly(cx * W, cy * H, size * W * grow, a0 + av * t)])
        fill_mask(dst, m, col, al * opacity)
    return dst


# --------------------------------------------------------------------------- petals


@lru_cache(None)
def _petals(n, seed):
    rng = np.random.default_rng(seed)
    return dict(
        x=rng.uniform(-0.1, 1.1, n), y=rng.uniform(-0.2, 1.2, n),
        vx=rng.uniform(-0.09, -0.02, n), vy=rng.uniform(0.07, 0.16, n),
        rot=rng.uniform(0, 6.28, n), rv=rng.uniform(-2.5, 2.5, n),
        tumble=rng.uniform(1.0, 4.0, n), ph=rng.uniform(0, 6.28, n),
        size=rng.uniform(0.6, 1.4, n), col=rng.integers(0, 3, n), depth=rng.uniform(0, 1, n),
    )


_PETAL = None


def _petal_shape():
    global _PETAL
    if _PETAL is None:
        pts = []
        for k in range(24):
            th = k / 24 * 2 * math.pi
            r = 1.0
            x, y = math.cos(th) * 0.55, math.sin(th)
            if y < -0.75:  # notch at the tip
                y = -0.75 + (y + 0.75) * 0.2 - 0.12 * (1 - abs(x) / 0.3 if abs(x) < 0.3 else 0)
            pts.append((x * r, y * r))
        _PETAL = np.float32(pts)
    return _PETAL


PETAL_COLS = [(1.0, 0.80, 0.87), (0.99, 0.70, 0.80), (1.0, 0.92, 0.95)]


def petals(dst, t, n=40, seed=1, scale=1.0, speed=1.0, opacity=1.0, burst=None):
    if opacity <= 0.01:
        return dst
    P = _petals(n, seed)
    shape = _petal_shape()
    masks = [np.zeros((H, W), np.uint8) for _ in PETAL_COLS]
    f = 16
    for i in range(n):
        tt = t * speed
        x = (P["x"][i] + P["vx"][i] * tt + 0.03 * math.sin(P["ph"][i] + tt * 1.7)) % 1.2 - 0.1
        y = (P["y"][i] + P["vy"][i] * tt) % 1.4 - 0.2
        if burst is not None:
            bx, by, bu = burst
            dx, dy = P["x"][i] - 0.5, P["y"][i] - 0.5
            x = bx + dx * bu * 1.6
            y = by + dy * bu * 1.6
        sz = (7 + 9 * P["depth"][i]) * P["size"][i] * scale
        rot = P["rot"][i] + P["rv"][i] * tt
        sq = 0.35 + 0.65 * abs(math.sin(P["ph"][i] + tt * P["tumble"][i]))
        c, s = math.cos(rot), math.sin(rot)
        px = shape[:, 0] * sz * sq
        py = shape[:, 1] * sz
        X = x * W + px * c - py * s
        Y = y * H + px * s + py * c
        pts = np.round(np.stack([X, Y], 1) * f).astype(np.int32)
        cv2.fillPoly(masks[P["col"][i]], [pts], 255, lineType=cv2.LINE_AA, shift=4)
    for m, col in zip(masks, PETAL_COLS):
        fill_mask(dst, m, col, 0.92 * opacity)
    return dst


@lru_cache(None)
def _shower(n, seed):
    rng = np.random.default_rng(seed)
    return dict(
        x=rng.uniform(-0.05, 1.1, n), y0=rng.uniform(-0.75, -0.04, n),
        vx=rng.uniform(-0.10, -0.02, n), vy=rng.uniform(0.24, 0.44, n),
        rot=rng.uniform(0, 6.28, n), rv=rng.uniform(-3.0, 3.0, n),
        tumble=rng.uniform(1.5, 4.5, n), ph=rng.uniform(0, 6.28, n),
        size=rng.uniform(0.7, 1.3, n), col=rng.integers(0, 3, n), depth=rng.uniform(0, 1, n) ** 1.5,
    )


SHOWER_COLS = [(0.99, 0.70, 0.81), (0.96, 0.58, 0.74), (1.0, 0.86, 0.91)]


def petal_shower(dst, tau, n=42, seed=1, scale=1.0, opacity=1.0):
    """A burst of petals that starts above the frame at tau = 0 and falls through it, swaying."""
    if tau < 0 or opacity <= 0.01:
        return dst
    P = _shower(n, seed)
    shape = _petal_shape()
    masks = [np.zeros((H, W), np.uint8) for _ in SHOWER_COLS]
    for i in range(n):
        d = P["depth"][i]
        fall = 0.75 + 0.5 * d  # nearer petals fall faster
        y = P["y0"][i] + P["vy"][i] * fall * tau
        if y > 1.15:
            continue
        x = P["x"][i] + P["vx"][i] * tau + 0.04 * math.sin(P["ph"][i] + tau * 2.2)
        sz = (8 + 14 * d) * P["size"][i] * scale
        rot = P["rot"][i] + P["rv"][i] * tau
        sq = 0.35 + 0.65 * abs(math.sin(P["ph"][i] + tau * P["tumble"][i]))
        c, s = math.cos(rot), math.sin(rot)
        px = shape[:, 0] * sz * sq
        py = shape[:, 1] * sz
        X = x * W + px * c - py * s
        Y = y * H + px * s + py * c
        pts = np.round(np.stack([X, Y], 1) * 16).astype(np.int32)
        cv2.fillPoly(masks[P["col"][i]], [pts], 255, lineType=cv2.LINE_AA, shift=4)
    for m, col in zip(masks, SHOWER_COLS):
        fill_mask(dst, m, col, 0.95 * opacity)
    return dst


# --------------------------------------------------------------------------- text


def font(spec, size):
    if isinstance(spec, tuple):
        return ImageFont.truetype(spec[0], size, index=spec[1])
    return ImageFont.truetype(spec, size)


def pil_to_layer(im):
    return _premul(np.asarray(im.convert("RGBA")))


def text_image(text, fspec, size, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0),
               glow=0, glow_col=(255, 255, 255), glow_k=1.0, spacing=0, gradient=None, shadow=None, outer=None):
    """Render a text line to a tight PIL RGBA image. Returns (image, origin_offset)."""
    f = font(fspec, size)
    pad = stroke + glow * 3 + 8 + (abs(shadow[0]) + abs(shadow[1]) if shadow else 0) + (outer[0] if outer else 0)
    widths = [f.getlength(ch) + spacing for ch in text]
    tw = int(sum(widths) + 2 * pad)
    asc, desc = f.getmetrics()
    th = int(asc + desc + 2 * pad)
    base = Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    def draw_glyphs(im, col, sw, swc, dx=0, dy=0):
        d = ImageDraw.Draw(im)
        x = pad
        for ch, wd in zip(text, widths):
            d.text((x + dx, pad + dy), ch, font=f, fill=col, stroke_width=sw, stroke_fill=swc)
            x += wd

    if glow:
        g = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_glyphs(g, glow_col + (255,), stroke + 2, glow_col + (255,))
        g = g.filter(ImageFilter.GaussianBlur(glow))
        if glow_k != 1.0:
            arr = np.asarray(g).copy()
            arr[..., 3] = np.clip(arr[..., 3].astype(np.float32) * glow_k, 0, 255).astype(np.uint8)
            g = Image.fromarray(arr)
        base.alpha_composite(g)
    if shadow:
        s = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_glyphs(s, shadow[2], stroke, shadow[2], shadow[0], shadow[1])
        base.alpha_composite(s.filter(ImageFilter.GaussianBlur(1.2)))
    if outer:  # thin contour outside the main stroke
        o = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_glyphs(o, outer[1], stroke + outer[0], outer[1])
        base.alpha_composite(o)
    if stroke:
        s = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_glyphs(s, stroke_fill, stroke, stroke_fill)
        base.alpha_composite(s)
    if gradient:
        m = Image.new("L", base.size, 0)
        draw_glyphs(m, 255, 0, 0)
        top, bot = gradient
        gy = np.linspace(0, 1, th)[:, None, None]
        grad = (np.float32(top) * (1 - gy) + np.float32(bot) * gy).astype(np.uint8)
        grad = np.broadcast_to(grad, (th, tw, 3))
        gim = Image.fromarray(np.dstack([grad, np.asarray(m)]).astype(np.uint8), "RGBA")
        base.alpha_composite(gim)
    else:
        fg = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_glyphs(fg, fill, 0, fill)
        base.alpha_composite(fg)
    xs = [pad + sum(widths[:i]) for i in range(len(text))]
    return base, pad, xs, widths


@lru_cache(None)
def text_layer(text, fspec, size, x, y, anchor="l", **kw):
    """Full-frame premultiplied text layer; (x, y) is the top-left of the glyph box (anchor l/r/c)."""
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}
    im, pad, xs, widths = text_image(text, fspec, size, **kw)
    tw = im.size[0] - 2 * pad
    if anchor == "r":
        x0 = x - tw
    elif anchor == "c":
        x0 = x - tw / 2
    else:
        x0 = x
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.alpha_composite(im, (int(round(x0 - pad)), int(round(y - pad)))) if x0 - pad >= 0 and y - pad >= 0 else \
        canvas.paste(im, (int(round(x0 - pad)), int(round(y - pad))), im)
    return pil_to_layer(canvas), tuple(x0 + v - pad for v in xs), tuple(widths)


def text_on(dst, layer_info, opacity=1.0, reveal=None, scatter=None, t=0.0):
    """Composite a text layer. reveal=(x_px) soft left-to-right wipe. scatter=progress 0..1 per-glyph assembly."""
    lay, xs, widths = layer_info
    if opacity <= 0.001:
        return dst
    lf = to_f(lay)
    if reveal is not None:
        xn = np.arange(W, dtype=np.float32)
        m = np.clip((reveal - xn) / 40.0, 0, 1)[None, :, None]
        lf = lf * m
    if scatter is not None and scatter < 1.0:
        out = np.zeros_like(lf)
        rng = np.random.default_rng(len(xs) * 7 + 3)
        for i, (x0, wd) in enumerate(zip(xs, widths)):
            x0i, x1i = int(max(0, x0 - 4)), int(min(W, x0 + wd + 8))
            if x1i <= x0i:
                continue
            delay = rng.uniform(0, 0.5)
            u = clamp01((scatter - delay) / 0.5)
            if u <= 0:
                continue
            dx = int(round((1 - ease_out(u)) * rng.uniform(-60, 60)))
            dy = int(round((1 - ease_out(u)) * rng.uniform(-30, 30)))
            sl = lf[:, x0i:x1i] * ease_out(u)
            sl = np.roll(sl, (dy, 0), axis=(0, 1))
            a0, a1 = max(0, x0i + dx), min(W, x1i + dx)
            if a1 > a0:
                out[:, a0:a1] += sl[:, a0 - (x0i + dx):a1 - (x0i + dx)]
        lf = np.clip(out, 0, 1)
    return over(dst, lf, opacity)
