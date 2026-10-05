"""Render the Yoshino PV.

  python3 render.py                    # full movie -> out/yoshinon_rhyme_pv.mp4
  python3 render.py --sheet 1 2.8 5    # contact sheet of stills at the given times
  python3 render.py --from 26 --to 45  # render a section only
"""
import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenes  # noqa: E402
from common import FPS, H, W  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def frame_bytes(i):
    t = i / FPS
    fr = scenes.render(t)
    return (fr * 255 + 0.5).astype(np.uint8).tobytes()


def sheet(times, path, cols=4, tw=400):
    th = tw * H // W
    rows = (len(times) + cols - 1) // cols
    sh = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    with Pool(4) as pool:
        frames = pool.map(_still, times)
    for k, (t, fr) in enumerate(zip(times, frames)):
        im = Image.fromarray(fr).resize((tw, th), Image.LANCZOS)
        sh.paste(im, ((k % cols) * tw, (k // cols) * th))
    sh.save(path)
    print("wrote", path)


def _still(t):
    return (scenes.render(t) * 255 + 0.5).astype(np.uint8)


def render_movie(path, t0, t1, crf=17):
    OUT.mkdir(exist_ok=True)
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, b in enumerate(pool.imap(frame_bytes, range(n0, n1), chunksize=4)):
            proc.stdin.write(b)
            if k % 150 == 0:
                print(f"frame {n0 + k}/{n1}  t={(n0 + k) / FPS:.2f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", nargs="*", type=float)
    ap.add_argument("--sheet-out", default=str(OUT / "sheet.png"))
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=scenes.DURATION)
    ap.add_argument("--out", default=str(OUT / "yoshinon_rhyme_pv.mp4"))
    a = ap.parse_args()
    scenes.prepare()
    OUT.mkdir(exist_ok=True)
    if a.sheet:
        sheet(a.sheet, a.sheet_out)
        return
    render_movie(a.out, a.t0, a.t1)


if __name__ == "__main__":
    main()
