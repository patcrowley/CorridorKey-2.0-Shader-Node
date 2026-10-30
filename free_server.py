"""CorridorKey 2.0 (free, GitHub) worker for the Blender add-on.

Run with the free CorridorKey 2.0 install's own Python, from its folder:

    <venv python> free_server.py <CorridorKey2.0 folder>

It loads the model once and keys on request, so previews after the first skip the load. One JSON
object per line on stdin; one per line on stdout (everything the library prints goes to stderr):

    -> {"id": 3, "cmd": "key", "input": "<frames dir>", "output_dir": "<dir>",
        "hint_first": null, "hint_video": "<dir>", "cfg": {...}}
    <- {"id": 3, "ok": true, "exr_dir": "<dir of index-numbered EXRs>", "suffix": "cutout"}
    -> {"id": 4, "cmd": "park"}            (model to system memory, GPU freed)
    <- {"id": 4, "ok": true}

Closing stdin ends it. Nothing here touches Blender.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_PROTOCOL = sys.stdout
sys.stdout = sys.stderr  # the library's prints and progress bars stay out of the protocol


def say(message: dict) -> None:
    _PROTOCOL.write(json.dumps(message) + "\n")
    _PROTOCOL.flush()


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
    except Exception as exc:  # a broken install: say so and stop
        say({"event": "error", "message": f"Couldn't load CorridorKey 2.0: {exc}"})
        traceback.print_exc()
        return
    if not torch.cuda.is_available():
        say({"event": "error", "message": "CorridorKey 2.0 (free) needs an NVIDIA GPU with CUDA."})
        return
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
            run_inference(
                model,
                request["input"],
                request["output_dir"],
                cfg,
                WriterConfig(),
                initial_hint_path=request.get("hint_first"),
                hint_video_path=request.get("hint_video"),
            )
            name = sorted(outputs)[0] if "cutout" not in outputs else "cutout"
            exr_dir = os.path.join(
                request["output_dir"], f"{raw_clip_name(request['input'])}_{suffixes[name]}_exr"
            )
            say({"id": rid, "ok": True, "exr_dir": exr_dir, "suffix": suffixes[name]})
        except Exception as exc:
            traceback.print_exc()
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            if "out of memory" in message.lower():
                message = "Out of GPU memory: try Low GPU Memory or a lower Internal Resolution."
            say({"id": rid, "ok": False, "message": message})
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
