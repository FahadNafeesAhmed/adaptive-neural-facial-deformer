"""Turn rendered PNG frames into an MP4 and a README-sized GIF.

    uv run --extra media python scripts/make_media.py runs/solver/demo/frames runs/solver/demo/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path, help="folder of frame_XXXX.png from blender_render.py")
    ap.add_argument("out", type=Path, help="output path without extension")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-width", type=int, default=900)
    ap.add_argument("--gif-step", type=int, default=2, help="keep every Nth frame in the GIF")
    args = ap.parse_args()

    paths = sorted(args.frames.glob("frame_*.png"))
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")
    frames = [Image.open(p).convert("RGB") for p in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import imageio_ffmpeg

    w, h = frames[0].size
    writer = imageio_ffmpeg.write_frames(str(args.out.with_suffix(".mp4")), (w, h), fps=args.fps,
                                         codec="libx264", pix_fmt_out="yuv420p", quality=8)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(np.asarray(f)))
    writer.close()

    scale = args.gif_width / w
    small = [f.resize((args.gif_width, int(h * scale)), Image.LANCZOS) for f in frames[:: args.gif_step]]
    small[0].save(args.out.with_suffix(".gif"), save_all=True, append_images=small[1:], optimize=True,
                  duration=int(1000 * args.gif_step / args.fps), loop=0)
    print(f"wrote {args.out.with_suffix('.mp4')} and {args.out.with_suffix('.gif')} from {len(frames)} frames")


if __name__ == "__main__":
    main()
