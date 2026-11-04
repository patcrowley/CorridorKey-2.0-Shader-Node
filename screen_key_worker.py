"""Screen Key: a simple green/blue keyer that bakes into the Alpha Hint, for the CorridorKey 2.0
Blender add-on.

Run inside a background Blender, which reads every image format Blender does:

    blender -b --factory-startup --python screen_key_worker.py -- <request.json>

The request lists the frames: [plate number, footage path, hint path or null]. For each frame the
worker keys the screen colour out of the footage, multiplies the result into the hint (when there
is one), and writes a white-on-black PNG named after the plate number. A frame that's already
written is skipped, so several workers can share one request (each takes every Nth frame).

    {"frames": [[1001, "C:/.../plate_01001.png", "C:/.../hint_01001.png"], ...],
     "direction": [x, y, z] (screen_direction of the picked colour), "threshold": 0.04, "softness": 0.08,
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


GREEN, BLUE = (0.20, 0.62, 0.25), (0.15, 0.25, 0.65)  # typical screens, display-like values


def screen_direction(display_color) -> np.ndarray:
    """The screen colour's hue as a unit vector: the colour minus its grey level, normalised.
    A pixel's score is its dot product with this: how far it leans towards the screen colour
    (grey and white score 0, skin and other opposite colours score below 0)."""
    colour = np.asarray(display_color, dtype=np.float64)
    chroma = colour - colour.mean()
    length = float(np.linalg.norm(chroma))
    if length < 0.02:  # a grey pick has no hue to key: green
        return screen_direction(GREEN)
    return chroma / length


def screen_matte(rgb: np.ndarray, direction, threshold: float, softness: float) -> np.ndarray:
    """1 where the footage is not the screen, 0 where it is: each pixel's lean towards the screen
    colour (display-like values) through a smoothstep from `threshold` to `threshold + softness`.
    The same sums as the live keyer inside the shader node. `direction` may be "GREEN"/"BLUE"."""
    if isinstance(direction, str):
        direction = screen_direction(BLUE if direction == "BLUE" else GREEN)
    score = rgb[..., :3] @ np.asarray(direction, dtype=rgb.dtype)
    t = np.clip((score - threshold) / max(softness, 1e-6), 0.0, 1.0)
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


def write_grey_png(path: str, matte_bottom_up: np.ndarray, depth: int = 8) -> None:
    """A greyscale PNG, 8 or 16 bits (the rows of a Blender image run bottom to top)."""
    top = 65535.0 if depth == 16 else 255.0
    grey = (np.clip(matte_bottom_up[::-1], 0.0, 1.0) * top + 0.5).astype(">u2" if depth == 16 else np.uint8)
    height, width = grey.shape
    raw = b"".join(b"\x00" + grey[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path + ".part", "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
    os.replace(path + ".part", path)  # only whole frames ever appear under their name


CLEAN_BINS = 256  # the clean floor's histogram: share of the floor colour, 0-2 in 256 steps


def floor_grid(origin, across, along, size):
    """The floor texels' centres in the plane's own space: (rows = along, cols = across, 4)."""
    cols, rows = size
    u = (np.arange(cols) + 0.5) / cols
    v = (np.arange(rows) + 0.5) / rows
    points = (np.asarray(origin, np.float64)[None, None, :] + u[None, :, None] * np.asarray(across, np.float64)
              + v[:, None, None] * np.asarray(along, np.float64))
    return np.concatenate([points, np.ones((rows, cols, 1))], axis=2)


def floor_samples(rgb: np.ndarray, display: bool, grid: np.ndarray, view, colour, direction):
    """One frame seen from the floor: for each texel, the floor's brightness there as a share of
    the floor colour, and whether it's measurable (in frame, in front, the screen's hue: not the
    actor or the set). `view`: plane-to-camera matrix (16, row-major), lens, kx, ky, sx, sy."""
    matrix = np.asarray(view[:16], np.float64).reshape(4, 4)
    lens, kx, ky, sx, sy = view[16:21]
    camera = grid.reshape(-1, 4) @ matrix.T
    depth = -camera[:, 2]
    ahead = depth > 1e-3
    depth = np.maximum(depth, 1e-3)
    u = camera[:, 0] * lens * kx / depth + 0.5 + sx
    v = camera[:, 1] * lens * ky / depth + 0.5 + sy
    inside = ahead & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    height, width = rgb.shape[:2]
    px = np.clip((u * width).astype(np.int64), 0, width - 1)
    py = np.clip((v * height).astype(np.int64), 0, height - 1)  # rows run bottom to top, as v
    seen = rgb[py, px, :3].astype(np.float64)
    if display:
        linear, shown = np.where(seen <= 0.04045, seen / 12.92, ((seen + 0.055) / 1.055) ** 2.4), seen
    else:
        linear = np.clip(seen, 0.0, None)
        shown = to_display(linear)
    colour = np.asarray(colour, np.float64)
    share = linear @ colour / max(float(colour @ colour), 1e-9)
    # the floor's hue however dark (as the add-on's Hue Gate), and not as dark as a dark suit
    direction = np.asarray(direction, np.float64)
    own = to_display(colour)
    own = max(float(own @ direction) / (float(np.linalg.norm(own)) + 0.03), 1e-3)
    hueness = (shown @ direction) / (np.linalg.norm(shown, axis=-1) + 0.03) / own
    floor = (hueness > 0.45) & (share > 0.15)
    return share, inside & floor


def add_to_histogram(histogram: np.ndarray, share: np.ndarray, valid: np.ndarray) -> None:
    bins = histogram.shape[1]
    index = np.clip((share * (bins / 2.0)).astype(np.int64), 0, bins - 1)
    texels = np.nonzero(valid)[0]
    histogram += np.bincount(texels * bins + index[texels], minlength=histogram.size).reshape(histogram.shape).astype(histogram.dtype)


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    """A Gaussian blur (radius in texels), edges kept (each side renormalised)."""
    if radius < 0.5:
        return values
    out = values
    for axis in (0, 1):
        n = out.shape[axis]
        x = np.arange(n)
        kernel = np.exp(-0.5 * ((x[:, None] - x[None, :]) / radius) ** 2)
        kernel /= kernel.sum(axis=1, keepdims=True)
        out = kernel @ out if axis == 0 else out @ kernel.T
    return out


def clean_floor_level(histogram: np.ndarray, size, shadow_time: float, softness: float) -> np.ndarray:
    """The clean floor from every frame's samples: per texel, the middle of its brightness once
    the darkest `shadow_time` share of the frames it was seen in is set aside (shadows move on,
    the lighting stays), blurred over `softness` texels; texels never seen take their
    surroundings' level. (rows, cols), a share of the floor colour."""
    cols, rows = size
    bins = histogram.shape[1]
    counts = histogram.sum(axis=1)
    target = (shadow_time + (1.0 - shadow_time) * 0.5) * counts
    running = np.cumsum(histogram, axis=1)
    index = np.argmax(running >= np.maximum(target, 1)[:, None], axis=1)
    below = np.where(index > 0, running[np.arange(len(index)), index - 1], 0)
    inside = histogram[np.arange(len(index)), index]
    within = np.clip((target - below) / np.maximum(inside, 1), 0.0, 1.0)
    level = ((index + within) * (2.0 / bins)).reshape(rows, cols)
    known = (counts >= 3).reshape(rows, cols).astype(np.float64)
    if known.sum() == 0:
        return np.ones((rows, cols))
    level = np.where(known > 0, level, 0.0)
    filled = level.copy()
    for radius in (4.0, 16.0, 64.0):  # holes take their surroundings' level, near ones first
        weight = _blur(known, radius)
        guess = _blur(level * known, radius) / np.maximum(weight, 1e-9)
        filled = np.where((known == 0) & (weight > 1e-3) & (filled == 0), guess, filled)
    filled = np.where(filled == 0, float(np.median(level[known > 0])), filled)
    return np.clip(_blur(filled, softness), 0.0, 2.0)


# --- Blender: reading the images -------------------------------------------------------------------


def _pixels(path: str):
    """(H x W x 4 float array, bottom row first, whether it's display-encoded) via Blender."""
    import bpy

    image = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = image.size
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        # 8-bit images hold display-encoded values already, whatever Blender tags them (a PNG
        # loads as "sRGB" or "Rec.1886" depending on the setup); float images (EXR) are linear
        display = not image.is_float
        return flat.reshape(height, width, 4), display
    finally:
        bpy.data.images.remove(image)


def main() -> None:
    request_path = sys.argv[sys.argv.index("--") + 1]
    with open(request_path, encoding="utf-8") as f:
        request = json.load(f)
    direction = request.get("direction") or request.get("color", "GREEN")
    threshold = float(request.get("threshold", 0.04))
    softness = float(request.get("softness", 0.08))
    out_dir, prefix = request["out_dir"], request.get("prefix", "hint_")
    start, step = request.get("slice", [0, 1])
    os.makedirs(out_dir, exist_ok=True)
    if request.get("mode") == "clean_floor":
        clean_floor_frames(request, start, step)
        return
    if request.get("mode") == "clean_floor_combine":
        clean_floor_combine(request)
        return
    hints: dict[str, np.ndarray] = {}  # one still hint, read once
    try:
        for number, footage, hint in request["frames"][start::step]:
            target = os.path.join(out_dir, f"{prefix}{int(number):06d}.png")
            if os.path.exists(target):
                continue
            rgba, srgb = _pixels(footage)
            rgb = rgba[..., :3] if srgb else to_display(rgba[..., :3])  # 8-bit sRGB already is
            matte = screen_matte(rgb, direction, threshold, softness)
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


def clean_floor_frames(request: dict, start: int, step: int) -> None:
    """The clean-floor bake, sampling: this worker's share of the frames, each looked at from the
    floor, into one histogram per texel (saved as histogram_<start>.npy; progress_<start>.txt
    counts the frames done)."""
    work = request["work"]
    size = request["size"]
    grid = floor_grid(*request["grid"], size)
    histogram = np.zeros((size[0] * size[1], CLEAN_BINS), np.uint16)
    progress = os.path.join(work, f"progress_{start}.txt")
    try:
        done = 0
        for number, footage, view in request["frames"][start::step]:
            rgba, display = _pixels(footage)
            share, valid = floor_samples(rgba, display, grid, view, request["colour"], request["direction"])
            add_to_histogram(histogram, share, valid)
            done += 1
            if done % 5 == 0:
                with open(progress + ".part", "w") as f:
                    f.write(str(done))
                os.replace(progress + ".part", progress)
        np.save(os.path.join(work, f"histogram_{start}.npy"), histogram)
        with open(progress + ".part", "w") as f:
            f.write(str(done))
        os.replace(progress + ".part", progress)
    except Exception as exc:  # the add-on shows this line
        with open(os.path.join(work, f"error_{start}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}")
        raise


def clean_floor_combine(request: dict) -> None:
    """The clean-floor bake, finishing: the workers' histograms into one 16-bit PNG over the
    plane's UVs (value / 2: up to twice the floor colour)."""
    work = request["work"]
    try:
        names = [n for n in os.listdir(work) if n.startswith("histogram_") and n.endswith(".npy")]
        histogram = sum(np.load(os.path.join(work, n)).astype(np.uint32) for n in names)
        level = clean_floor_level(histogram, request["size"], float(request.get("shadow_time", 0.8)),
                                  float(request.get("softness", 2.0)))
        write_grey_png(os.path.join(request["out_dir"], "clean_floor.png"), level / 2.0, depth=16)
    except Exception as exc:
        with open(os.path.join(work, "error_combine.txt"), "w", encoding="utf-8") as f:
            f.write(f"{type(exc).__name__}: {exc}")
        raise


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


if __name__ == "__main__":
    main()
