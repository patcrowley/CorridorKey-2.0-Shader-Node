"""SAM 3 alpha-hint maker for the CorridorKey 2.0 Blender add-on.

Run with SAM 3's own Python:  <venv python> sam3_worker.py <request.json>

The request names the frames, the text prompt and where the mattes go; the worker writes one
white-on-black PNG per frame (every match of the prompt, merged) named after its plate number.
Progress goes to stdout as JSON lines; everything the libraries print goes to stderr.

    {"mode": "frame" | "video", "frames_dir": dir of frames named 000000.png.. (video mode),
     "frame_paths": [...] (frame mode), "numbers": [plate number of each frame, in order],
     "prompt": "person", "prompt_index": 0, "threshold": 0.5,
     "checkpoint": ".../sam3.pt", "out_dir": "...", "prefix": "hint_"}

SAM 3 is Meta's, under the SAM License (see the LICENSE next to its weights).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import types

_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def say(**message) -> None:
    _PROTOCOL.write(json.dumps(message) + "\n")
    _PROTOCOL.flush()


def _stand_in_for_triton() -> None:
    """SAM 3 imports a Triton kernel for one distance transform. Without Triton (usual on Windows),
    the same transform is done with OpenCV instead."""
    try:
        import triton  # noqa: F401

        return
    except Exception:
        pass
    import cv2
    import numpy as np
    import torch

    def edt_triton(data):
        out = [
            torch.from_numpy(cv2.distanceTransform(d.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 0))
            for d in data
        ]
        return torch.stack(out).to(device=data.device, dtype=torch.float32)

    module = types.ModuleType("sam3.model.edt")
    module.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = module


def _fast_frame_reading() -> None:
    """SAM 3 reads a folder of frames one at a time before it tracks anything. The same reading,
    on every CPU core: frames decode in parallel into the same float16 tensor SAM 3 builds."""
    try:
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from sam3.model import io_utils

        load_one = io_utils._load_img_as_tensor
        exts = io_utils.IMAGE_EXTS
    except Exception:
        return  # a different SAM 3: its own reader

    def load_folder(image_folder, image_size, offload_video_to_cpu, img_mean, img_std, async_loading_frames):
        names = [p for p in os.listdir(image_folder) if os.path.splitext(p)[-1].lower() in exts]
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if not names:
            raise RuntimeError(f"no images found in {image_folder}")
        paths = [os.path.join(image_folder, n) for n in names]
        images = torch.zeros(len(paths), 3, image_size, image_size, dtype=torch.float16)

        def one(i):
            image, height, width = load_one(paths[i], image_size)
            images[i] = image
            return height, width

        with ThreadPoolExecutor(max_workers=max(2, min(16, (os.cpu_count() or 4)))) as pool:
            sizes = list(pool.map(one, range(len(paths))))
        mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        if not offload_video_to_cpu:
            images, mean, std = images.cuda(), mean.cuda(), std.cuda()
        images -= mean
        images /= std
        return images, sizes[0][0], sizes[0][1]

    io_utils.load_video_frames_from_image_folder = load_folder


_FRAME_BYTES = 3 * 1008 * 1008 * 2  # one frame as SAM 3 keeps it: 1008 x 1008 RGB, float16
_STATE_BYTES = 3 * 2**20  # what tracking keeps per frame (masks, memory): a generous guess
_HEADROOM = 3 * 2**30  # the model's working memory while it runs


def _where_frames_go(count: int):
    """(frames in system RAM?, tracking state in system RAM?): on the GPU whenever they fit in
    what's free now (the model already loaded), in system RAM only when they don't."""
    import torch

    free, _total = torch.cuda.mem_get_info()
    video, state = count * _FRAME_BYTES, count * _STATE_BYTES
    if video + state + _HEADROOM <= free:
        return False, False
    if video + _HEADROOM <= free:
        return False, True
    return True, state + _HEADROOM > free


def _track(predictor, request, numbers, prompt, threshold, out_dir, prefix, offload):
    """Track the prompt through the frames; (frames with nothing found, when tracking began)."""
    import numpy as np
    from PIL import Image

    total = len(numbers)
    offload_video, offload_state = offload
    try:
        free_gb = round(predictor_free_gb(), 1)
    except Exception:
        free_gb = None
    say(event="memory", frames="system RAM" if offload_video else "GPU",
        state="system RAM" if offload_state else "GPU", free_gb=free_gb)
    reading = time.monotonic()
    session = predictor.handle_request({
        "type": "start_session",
        "resource_path": request["frames_dir"],
        "offload_video_to_cpu": offload_video,
        "offload_state_to_cpu": offload_state,
    })["session_id"]
    say(event="stage", stage="frames", seconds=round(time.monotonic() - reading, 1))
    say(event="progress", done=0, total=total)
    work_started = time.monotonic()
    start = max(0, min(int(request.get("prompt_index", 0)), total - 1))
    predictor.handle_request({
        "type": "add_prompt", "session_id": session, "frame_index": start, "text": prompt,
        "output_prob_thresh": threshold,
    })
    done: set[int] = set()
    empty = 0
    first = sorted(os.listdir(request["frames_dir"]))[0]
    with Image.open(os.path.join(request["frames_dir"], first)) as probe:
        size = (probe.height, probe.width)
    for answer in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session,
        "propagation_direction": "both", "start_frame_index": start,
        "output_prob_thresh": threshold,
    }):
        index = int(answer["frame_index"])
        masks = answer["outputs"].get("out_binary_masks")
        if masks is not None and len(masks):
            mask = np.asarray(masks).any(axis=0).astype(np.float32)
        else:
            mask = np.zeros(size, np.float32)
            if index not in done:
                empty += 1
        _save(mask, os.path.join(out_dir, f"{prefix}{numbers[index]:06d}.png"))
        done.add(index)
        say(event="progress", done=len(done), total=total)
    predictor.handle_request({"type": "close_session", "session_id": session})
    return empty, work_started


def predictor_free_gb() -> float:
    import torch

    return torch.cuda.mem_get_info()[0] / 2**30


def _save(mask, path: str) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        request = json.load(f)
    numbers = request["numbers"]
    out_dir = request["out_dir"]
    prefix = request.get("prefix", "hint_")
    prompt = request["prompt"].strip() or "person"
    threshold = float(request.get("threshold", 0.5))
    os.makedirs(out_dir, exist_ok=True)
    try:
        import numpy as np
        import torch
        from PIL import Image
    except Exception as exc:
        say(event="error", message=f"SAM 3 isn't installed properly: {exc}")
        return
    if not torch.cuda.is_available():
        say(event="error", message="SAM 3 needs an NVIDIA GPU with CUDA.")
        return
    try:  # faster float32 maths on RTX cards: what SAM 3.1's own predictor sets
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass
    _stand_in_for_triton()
    _fast_frame_reading()
    started = time.monotonic()
    say(event="loading")
    empty = 0
    try:
        if request["mode"] == "frame":
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(checkpoint_path=request["checkpoint"], load_from_HF=False)
            processor = Sam3Processor(model, confidence_threshold=threshold)
            say(event="stage", stage="load", seconds=round(time.monotonic() - started, 1))
            work_started = time.monotonic()
            paths = request["frame_paths"]
            say(event="progress", done=0, total=len(paths))
            for i, path in enumerate(paths):
                image = Image.open(path).convert("RGB")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = processor.set_image(image)
                    state = processor.set_text_prompt(prompt=prompt, state=state)
                probs = state.get("masks_logits")
                if probs is None or len(probs) == 0:
                    mask = np.zeros((image.height, image.width), np.float32)
                    empty += 1
                else:
                    mask = probs.float().amax(dim=0)[0].cpu().numpy()
                _save(mask, os.path.join(out_dir, f"{prefix}{numbers[i]:06d}.png"))
                say(event="progress", done=i + 1, total=len(paths))
        else:
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor

            predictor = Sam3VideoPredictor(checkpoint_path=request["checkpoint"])
            say(event="stage", stage="load", seconds=round(time.monotonic() - started, 1))
            total = len(numbers)
            offload = _where_frames_go(total)
            while True:
                try:
                    empty, work_started = _track(predictor, request, numbers, prompt, threshold,
                                                 out_dir, prefix, offload)
                    break
                except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is one
                    if "out of memory" not in str(exc).lower() or offload == (True, True):
                        raise
                    # it didn't fit after all: again, with what doesn't fit in system RAM
                    for session in list(getattr(predictor, "_all_inference_states", {})):
                        try:
                            predictor.handle_request({"type": "close_session", "session_id": session})
                        except Exception:
                            pass
                    torch.cuda.empty_cache()
                    offload = (True, True) if offload[0] else (True, offload[1])
    except Exception as exc:
        traceback.print_exc()
        message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        if "out of memory" in message.lower():
            message = "Out of GPU memory: make the hint for a shorter range, or close other GPU apps."
        say(event="error", message=message)
        return
    say(event="stage", stage="work", seconds=round(time.monotonic() - work_started, 1))
    say(event="done", empty=empty)


if __name__ == "__main__":
    main()
