"""CorridorKey 2.0 (free, GitHub) worker for the Blender add-on.

Run with the free CorridorKey 2.0 install's own Python, from its folder:

    <venv python> free_server.py <CorridorKey2.0 folder>

It loads the model once and keys on request, so previews after the first skip the load. One JSON
object per line on stdin; one per line on stdout (everything the library prints goes to stderr):

    -> {"id": 3, "cmd": "key", "input": "<frames dir>", "output_dir": "<dir>",
        "hint_first": null, "hint_video": "<dir>",
        "cfg": {..., "screen": "GREEN" or "BLUE", "input_matrix": null or 3x3 to linear Rec.709}}
    <- {"id": 3, "progress": {"fraction": 0.5, "detail": "keying"}}     (as it goes)
    <- {"id": 3, "ok": true, "exr_dir": "<dir of index-numbered EXRs>", "suffix": "cutout"}
    -> {"id": 4, "cmd": "park"}            (model to system memory, GPU freed)
    <- {"id": 4, "ok": true}

Closing stdin ends it. Nothing here touches Blender.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_PROTOCOL = sys.stdout
_KEYING = [None]  # the id of the request being keyed


def say(message: dict) -> None:
    _PROTOCOL.write(json.dumps(message) + "\n")
    _PROTOCOL.flush()


# How far a key has got, from what the library prints at each step: (pattern, fraction, detail).
_STEPS = (
    (re.compile(r"^\s*probed \d+ frames"), 0.05, "reading the frames"),
    (re.compile(r"^\s*Auto VRAM"), 0.10, "fitting to the GPU"),
    (re.compile(r"^\s*RVM continuous prepass"), 0.20, "first pass (matting)"),
    (re.compile(r"^\s*frozen VFM prepass"), 0.35, "second pass (depth and features)"),
    (re.compile(r"^\s*Hann stage1"), 0.50, "keying"),
    (re.compile(r"^\s*wrote \d+ output"), 0.95, "saving"),
)
_WINDOW = re.compile(r"^\s*window (\d+)/(\d+)")


def _step(line: str) -> None:
    rid = _KEYING[0]
    if rid is None:
        return
    found = _WINDOW.match(line)
    if found:
        done, total = int(found.group(1)), max(1, int(found.group(2)))
        detail = "finishing the frames" if done >= total else f"keying, window {done + 1} of {total}"
        say({"id": rid, "progress": {"fraction": 0.5 + 0.35 * done / total, "detail": detail}})
        return
    for pattern, fraction, detail in _STEPS:
        if pattern.match(line):
            say({"id": rid, "progress": {"fraction": fraction, "detail": detail}})
            return


_GPU_ROOM_GB = 2.5  # left free for Blender (viewport, textures, display) while a key runs
_GPU_ROOM_SHARE = 0.15  # or this share of the card, whichever is more


def _leave_room_on_gpu(runtime, torch) -> None:
    """The library sizes its windows to the GPU memory free when a key starts, with a 0.3 GB margin.
    Blender shares the card and takes more as it goes (the viewport, 4K textures), so a long render
    could run out of memory part-way; on Windows the driver then spills into system memory and the
    key crawls (a 108-frame 4K render sat at its first window for 20 minutes). Size the windows as
    if a couple of GB fewer were free."""
    select = getattr(runtime, "select_vram_profile", None)
    if select is None or not hasattr(runtime, "apply_auto_vram"):
        return

    def apply_auto_vram(cfg, shape_hw, num_frames):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        budget = (free_bytes + torch.cuda.memory_reserved()) / 2**30
        room = max(_GPU_ROOM_GB, _GPU_ROOM_SHARE * total_bytes / 2**30)
        budget = max(budget - room, 1.0)
        chunk, stride, low_vram, estimate = select(shape_hw, num_frames, budget)
        cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
        print(f"  Auto VRAM: {budget:.1f} GB to use ({room:.1f} GB left for Blender) at "
              f"{shape_hw[1]}x{shape_hw[0]} -> hann_chunk={chunk}, hann_stride={stride}, "
              f"low_vram={low_vram} (estimated peak {estimate:.1f} GB)")

    runtime.apply_auto_vram = apply_auto_vram


_SCREEN = ["GREEN"]  # the screen of the shot being keyed: what its despill takes out


def _despill_by_screen(runtime, torch) -> None:
    """The library's despill takes out green alone; on a blue screen it would pull a costume's real
    green out and leave the blue spill. Swap in one that takes out the shot's own screen colour:
    the excess of that channel over the other two's average, given back to them half each."""
    original = getattr(runtime, "_despill_average_srgb_torch", None)
    if original is None:
        return

    def despill(fg, strength: float = 0.5):
        if _SCREEN[0] != "BLUE":
            return original(fg, strength)
        if strength <= 0.0:
            return fg
        r, g, b = fg[:, 0:1], fg[:, 1:2], fg[:, 2:3]
        spill = (b - (r + g) * 0.5).clamp(min=0.0)
        full = torch.cat([r + spill * 0.5, g + spill * 0.5, b - spill], dim=1)
        return full if strength >= 1.0 else fg * (1.0 - strength) + full * strength

    runtime._despill_average_srgb_torch = despill


_CONVERTED_KEEP = 300  # converted frames kept for reuse (Live and Cache Ahead come back to the same ones)


def _to_rec709(folder: str, output_dir: str, matrix) -> str:
    """Copies of a folder's EXR frames converted to linear Rec.709 by `matrix` (3x3, rows), in a
    folder of the same name (the clip's name comes from it); other frames are linked as they are.
    The library reads an EXR's numbers as Rec.709, so a wide-gamut plate would key with the wrong
    colours. The plates themselves are never changed. Converted frames are kept (half float) and
    reused while the plate and the matrix are the same."""
    import hashlib
    import shutil
    import tempfile

    import cv2
    import numpy as np

    target = os.path.join(output_dir, "_rec709", os.path.basename(os.path.normpath(folder)))
    os.makedirs(target, exist_ok=True)
    m = np.asarray(matrix, dtype=np.float32)
    store = os.path.join(tempfile.gettempdir(), "corridorkey_v2", "rec709",
                         hashlib.sha1(m.tobytes()).hexdigest()[:12])
    os.makedirs(store, exist_ok=True)
    flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]

    def place(source, dest):
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)

    for name in sorted(os.listdir(folder)):
        source, dest = os.path.join(folder, name), os.path.join(target, name)
        if not name.lower().endswith(".exr"):
            place(source, dest)
            continue
        st = os.stat(source)  # the plate itself (staged as a hard link where the disk allows)
        kept = os.path.join(store, f"{st.st_ino or abs(hash(os.path.realpath(source)))}_{st.st_size}_{int(st.st_mtime)}.exr")
        if not os.path.isfile(kept):
            image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] < 3:
                place(source, dest)
                continue
            image = image.astype(np.float32, copy=False)
            rgb = image[..., 2::-1] @ m.T  # OpenCV keeps BGR
            image[..., :3] = np.clip(rgb[..., ::-1], -65000.0, 65000.0)
            partial = kept + ".part.exr"
            if not cv2.imwrite(partial, image, flags):
                raise RuntimeError(f"couldn't write the converted frame {kept}")
            os.replace(partial, kept)
        else:
            os.utime(kept)  # recently used
        place(kept, dest)
    try:  # keep the store from growing without end: the least recently used go first
        frames = sorted((os.path.getmtime(os.path.join(store, n)), n) for n in os.listdir(store) if n.endswith(".exr"))
        for _when, n in frames[:-_CONVERTED_KEEP]:
            os.remove(os.path.join(store, n))
    except OSError:
        pass
    return target


class _Steps:
    """The library's prints and progress bars stay out of the protocol (they go to stderr, the
    log, as before); each line is also read for how far the key has got."""

    def __init__(self, stream) -> None:
        self.stream, self.pending = stream, ""

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.pending += text
        *lines, self.pending = re.split(r"[\r\n]", self.pending)
        for line in lines:
            try:
                _step(line)
            except Exception:  # progress is a nicety: never let it stop a key
                pass
        return len(text)

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


sys.stdout = _Steps(sys.stderr)


def main() -> None:
    root = os.path.abspath(sys.argv[1])
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from pathlib import Path

        import torch

        from corridorkey.runtime import (
            InferenceConfig,
            RuntimePaths,
            load_model,
            run_inference,
        )
        from corridorkey.video_io import WriterConfig, raw_clip_name
        import corridorkey.runtime as runtime
    except Exception as exc:  # a broken install: say so and stop
        say({"event": "error", "message": f"Couldn't load CorridorKey 2.0: {exc}"})
        traceback.print_exc()
        return
    if not torch.cuda.is_available():
        say({"event": "error", "message": "CorridorKey 2.0 (free) needs an NVIDIA GPU with CUDA."})
        return
    _despill_by_screen(runtime, torch)
    _leave_room_on_gpu(runtime, torch)
    say({"event": "loading"})
    try:
        model = load_model(RuntimePaths.from_package(root), device="cuda")
    except Exception as exc:
        say({"event": "error", "message": f"Couldn't load the model: {exc}"})
        traceback.print_exc()
        return
    say({"event": "ready", "gpu": torch.cuda.get_device_name(0)})

    suffixes = {"alpha": "alpha", "fg": "fg", "cutout": "cutout"}
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
        except ValueError:
            continue
        rid, cmd = request.get("id"), request.get("cmd")
        if cmd == "park":
            model.cpu()
            torch.cuda.empty_cache()
            say({"id": rid, "ok": True})
            continue
        if cmd != "key":
            say({"id": rid, "ok": False, "message": f"unknown command {cmd!r}"})
            continue
        try:
            c = request.get("cfg") or {}
            outputs = set(c.get("outputs") or ["cutout"])
            chunk = int(c.get("hann_chunk") or 80)
            cfg = InferenceConfig(
                hann_chunk=chunk,
                hann_stride=max(1, int(c.get("hann_stride") or chunk // 2)),
                auto_vram=bool(c.get("auto_vram", True)),
                carry_hint=bool(c.get("carry_hint", True)),
                hint_quality=float(c.get("hint_quality", 0.95)),
                carry_hint_quality=float(c.get("carry_hint_quality", 0.95)),
                model_scale=float(c.get("model_scale", 1.0)),
                low_vram=bool(c.get("low_vram", False)),
                outputs=outputs,
                temp_dir=Path(c.get("temp_dir") or os.path.join(request["output_dir"], "_temp")),
                frame_dir_linear=bool(c.get("frame_dir_linear", False)),
                despill_strength=float(c.get("despill_strength", 0.5)),
                output_format="exr",
                exr_dtype=str(c.get("exr_dtype", "half")),
            )
            _SCREEN[0] = "BLUE" if str(c.get("screen", "GREEN")).upper() == "BLUE" else "GREEN"
            _KEYING[0] = rid
            plates = request["input"]
            if c.get("input_matrix") and os.path.isdir(plates):
                say({"id": rid, "progress": {"fraction": 0.02, "detail": "converting the footage's colours"}})
                plates = _to_rec709(plates, request["output_dir"], c["input_matrix"])
            run_inference(
                model,
                plates,
                request["output_dir"],
                cfg,
                WriterConfig(),
                initial_hint_path=request.get("hint_first"),
                hint_video_path=request.get("hint_video"),
            )
            name = sorted(outputs)[0] if "cutout" not in outputs else "cutout"
            exr_dir = os.path.join(
                request["output_dir"], f"{raw_clip_name(plates)}_{suffixes[name]}_exr"
            )
            _KEYING[0] = None
            say({"id": rid, "ok": True, "exr_dir": exr_dir, "suffix": suffixes[name]})
        except Exception as exc:
            _KEYING[0] = None
            traceback.print_exc()
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            if "out of memory" in message.lower():
                message = "Out of GPU memory: try Low GPU Memory or a lower Internal Resolution."
            say({"id": rid, "ok": False, "message": message})
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
