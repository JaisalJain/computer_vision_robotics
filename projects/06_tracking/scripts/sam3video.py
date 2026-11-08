"""SAM 3 text-prompted video tracking helpers (Hugging Face ``transformers`` implementation).

SAM 3 (``facebook/sam3``, gated: accept the licence on the model page and log in with
``hf auth login``) detects every instance of a text concept in each frame and tracks them with a
SAM 2 style memory. The ``transformers`` post-processing (mask NMS between detections, hole
filling / sprinkle removal) relies on the optional ``kernels`` package, which downloads compiled
CUDA kernels at runtime. We do not install it; :func:`enable_postprocessing` plugs in small
PyTorch / OpenCV equivalents instead, so the model behaves as intended.
"""

from __future__ import annotations

import time

import numpy as np


class TorchCvUtils:
    """Drop-in for the two ``kernels-community/cv-utils`` ops used by SAM 3 video."""

    @staticmethod
    def generic_nms(ious, probs, iou_threshold: float, use_iou_matrix: bool = True):
        """Greedy NMS on a precomputed IoU matrix; returns kept indices (descending score)."""
        import torch

        order = torch.argsort(probs, descending=True).tolist()
        suppressed = torch.zeros(len(order), dtype=torch.bool, device=probs.device)
        keep = []
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            suppressed |= ious[i] > iou_threshold
        return torch.tensor(keep, dtype=torch.long, device=probs.device)

    @staticmethod
    def cc_2d(mask, get_counts: bool = True):
        """8-connected components of (N, 1, H, W) uint8 masks: labels and per-pixel area."""
        import cv2
        import torch

        m = mask.detach().to("cpu").numpy().astype(np.uint8)
        labels = np.zeros(m.shape, np.int32)
        counts = np.zeros(m.shape, np.int32)
        for i in range(m.shape[0]):
            _, lab, stats, _ = cv2.connectedComponentsWithStats(m[i, 0], connectivity=8)
            labels[i, 0] = lab
            counts[i, 0] = stats[:, cv2.CC_STAT_AREA].astype(np.int32)[lab]
        return torch.from_numpy(labels).to(mask.device), torch.from_numpy(counts).to(mask.device)


def enable_postprocessing() -> None:
    """Install :class:`TorchCvUtils` as SAM 3 video's NMS / connected-components backend."""
    import transformers.models.sam3_video.modeling_sam3_video as m

    m.cv_utils_kernel = TorchCvUtils()


def load_sam3_video(device: str = "cuda"):
    """SAM 3 video model (bf16) + processor from the Hugging Face cache."""
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    enable_postprocessing()
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return model.eval(), processor


def track_concept(model, processor, frames: list[np.ndarray], text: str, device: str = "cuda",
                  keep_masks: set[int] | None = None, on_frame=None,
                  state_device: str = "cpu") -> dict:  # fmt: skip
    """Run text-prompted detection + tracking over ``frames`` (RGB uint8).

    Args:
        state_device: where the per-frame tracker state (memory features, low-res masks of every
            object in every processed frame) is kept. It grows linearly with frames x objects;
            on the GPU, 100 frames with ~10 people exceed 8 GB, so it lives in CPU RAM by default
            and only the frames the memory attention needs are copied to the GPU.
        keep_masks: frame indices (0-based) whose full-resolution masks are returned.
        on_frame: optional callback ``(frame_idx, out)`` with the post-processed outputs
            (``object_ids``, ``scores``, ``boxes`` xyxy, ``masks``), e.g. to render a GIF.

    Returns:
        ``rows`` in MOTChallenge format ``(frame, id, x, y, w, h, score)`` (ids start at 1),
        ``masks`` {frame_idx: (ids, masks)}, ``seconds``, ``fps`` and ``peak_vram_mb``.
    """
    import torch

    from cvlab.device import track_vram

    keep_masks = keep_masks or set()
    rows, masks = [], {}
    with track_vram() as mem:
        session = processor.init_video_session(
            video=frames, inference_device=device, processing_device="cpu",
            video_storage_device="cpu", inference_state_device=state_device, dtype=torch.bfloat16,
        )  # fmt: skip
        session = processor.add_text_prompt(inference_session=session, text=text)
        t0 = time.perf_counter()
        stream = model.propagate_in_video_iterator(inference_session=session,
                                                   max_frame_num_to_track=len(frames))  # fmt: skip
        for out in stream:
            post = processor.postprocess_outputs(session, out)
            f = int(out.frame_idx)
            ids = post["object_ids"].cpu().numpy()
            boxes = post["boxes"].float().cpu().numpy()
            scores = post["scores"].float().cpu().numpy()
            for i, b, s in zip(ids, boxes, scores):
                rows.append((f + 1, int(i) + 1, b[0], b[1], b[2] - b[0], b[3] - b[1], s))
            if f in keep_masks:
                masks[f] = (ids + 1, post["masks"].cpu().numpy())
            if on_frame is not None:
                on_frame(f, post)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    out_rows = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    out_rows[:, 2:6] = out_rows[:, 2:6].round(2)
    return {"rows": out_rows, "masks": masks, "seconds": seconds, "fps": len(frames) / seconds,
            "peak_vram_mb": mem.peak_mb}  # fmt: skip
