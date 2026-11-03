"""Screen Key: a simple green/blue keyer that bakes into the Alpha Hint, for the CorridorKey 2.0
Blender add-on.

Run inside a background Blender, which reads every image format Blender does:

    blender -b --factory-startup --python screen_key_worker.py -- <request.json>

The request lists the frames: [plate number, footage path, hint path or null]. For each frame the
worker keys the screen colour out of the footage, multiplies the result into the hint (when there
is one), and writes a white-on-black PNG named after the plate number. A frame that's already
written is skipped, so several workers can share one request (each takes every Nth frame).

    {"frames": [[1001, "C:/.../plate_01001.png", "C:/.../hint_01001.png"], ...],
     "color": "GREEN" | "BLUE", "threshold": 0.04, "softness": 0.08,
     "out_dir": "...", "prefix": "hint_", "slice": [0, 4]}
"""

from __future__ import annotations

import json
import os
import struct
import sys
import zlib

import numpy as np


# --- the maths (no Blender here: tested on its own) -----------------------------------------------


GAMMA = 2.2  # the live keyer's Gamma node undoes the same


def to_display(linear: np.ndarray) -> np.ndarray:
    """Linear light to display-like values: the colour difference is judged there, so a dark,
    shadowed screen still reads as green (in linear light it barely differs from black)."""
    return np.clip(linear, 0.0, None) ** (1.0 / GAMMA)


def screen_matte(rgb: np.ndarray, color: str, threshold: float, softness: float) -> np.ndarray:
    """1 where the footage is not the screen, 0 where it is: the colour difference (green minus
    the larger of red and blue, or blue minus red/green, in display-like values) through a
    smoothstep from `threshold` to `threshold + softness`. The same sums as the live keyer inside
    the shader node."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    difference = g - np.maximum(r, b) if color == "GREEN" else b - np.maximum(r, g)
    t = np.clip((difference - threshold) / max(softness, 1e-6), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def hint_value(rgba: np.ndarray) -> np.ndarray:
    """A hint image as one matte: its grey level, times its alpha (a cut-out works as well as a
    black-and-white matte)."""
    return rgba[..., :3].mean(axis=-1) * rgba[..., 3]


def fit(matte: np.ndarray, height: int, width: int) -> np.ndarray:
    """A matte resized (nearest pixel) to the footage's size, when the hint is another size."""
    if matte.shape == (height, width):
        return matte
    rows = (np.arange(height) * matte.shape[0] // height).clip(0, matte.shape[0] - 1)
    cols = (np.arange(width) * matte.shape[1] // width).clip(0, matte.shape[1] - 1)
    return matte[rows][:, cols]


def write_grey_png(path: str, matte_bottom_up: np.ndarray) -> None:
    """An 8-bit greyscale PNG (the rows of a Blender image run bottom to top)."""
    grey = (np.clip(matte_bottom_up[::-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    height, width = grey.shape
    raw = b"".join(b"\x00" + grey[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path + ".part", "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
    os.replace(path + ".part", path)  # only whole frames ever appear under their name


# --- Blender: reading the images -------------------------------------------------------------------


def _pixels(path: str):
    """(H x W x 4 float array, bottom row first, whether it was 8-bit sRGB) via Blender."""
    import bpy

    image = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = image.size
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        srgb = not image.is_float and "srgb" in image.colorspace_settings.name.lower()
        return flat.reshape(height, width, 4), srgb
    finally:
        bpy.data.images.remove(image)


def main() -> None:
    request_path = sys.argv[sys.argv.index("--") + 1]
    with open(request_path, encoding="utf-8") as f:
        request = json.load(f)
    color = request.get("color", "GREEN")
    threshold = float(request.get("threshold", 0.04))
    softness = float(request.get("softness", 0.08))
    out_dir, prefix = request["out_dir"], request.get("prefix", "hint_")
    start, step = request.get("slice", [0, 1])
    os.makedirs(out_dir, exist_ok=True)
    hints: dict[str, np.ndarray] = {}  # one still hint, read once
    try:
        for number, footage, hint in request["frames"][start::step]:
            target = os.path.join(out_dir, f"{prefix}{int(number):06d}.png")
            if os.path.exists(target):
                continue
            rgba, srgb = _pixels(footage)
            rgb = rgba[..., :3] if srgb else to_display(rgba[..., :3])  # 8-bit sRGB already is
            matte = screen_matte(rgb, color, threshold, softness)
            if hint:
                if hint not in hints:
                    value = hint_value(_pixels(hint)[0])
                    if len(hints) > 2:  # a sequence: don't keep every frame
                        hints.clear()
                    hints[hint] = value
                matte = matte * fit(hints[hint], *matte.shape)
            write_grey_png(target, matte)
    except Exception as exc:  # the add-on shows this line
        with open(os.path.join(out_dir, f"error_{start}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
