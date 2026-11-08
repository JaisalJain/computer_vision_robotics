"""DINOv3 appearance embeddings as a re-identification (ReID) encoder for BoT-SORT.

BoT-SORT can fuse motion (IoU) with appearance: every detection is cropped, embedded, and a
track/detection pair is also allowed to match when the cosine distance of their embeddings
is small. Ultralytics ships a dedicated person-ReID model (``yolo26n-reid.onnx``, trained to
tell identities apart). :class:`DinoV3ReID` plugs in a *general-purpose* self-supervised
backbone instead, DINOv3 ViT-S/16 (``facebook/dinov3-vits16-pretrain-lvd1689m``, gated:
accept the DINOv3 licence on the model page and log in with ``hf auth login``), so the two
can be compared on the same detections.

The encoder follows the Ultralytics ReID call convention, ``encoder(img_bgr, dets_xywh)`` ->
one feature vector per detection (``None`` for an empty crop), where ``dets_xywh`` holds
centre-x, centre-y, width and height in pixels.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_boxes(img: np.ndarray, xyxy: np.ndarray) -> list[np.ndarray]:
    """Crop ``(x1, y1, x2, y2)`` boxes (clipped to the image) from an HxWx3 image."""
    h, w = img.shape[:2]
    out = []
    for x1, y1, x2, y2 in np.asarray(xyxy, dtype=np.float64).reshape(-1, 4):
        x1, x2 = int(np.clip(np.floor(x1), 0, w)), int(np.clip(np.ceil(x2), 0, w))
        y1, y2 = int(np.clip(np.floor(y1), 0, h)), int(np.clip(np.ceil(y2), 0, h))
        out.append(img[y1:y2, x1:x2])
    return out


class DinoV3ReID:
    """DINOv3 ViT-S/16 crop embeddings (L2-normalised CLS token, optionally + mean patch token).

    Args:
        device: Torch device for the backbone.
        crop_hw: Every crop is resized to this (height, width) in pixels; 256 x 128 is the usual
            pedestrian-ReID aspect ratio and a multiple of the 16 px patch size (16 x 8 patches).
        pooling: ``"cls"`` (global CLS token) or ``"cls+mean"`` (CLS concatenated with the mean
            of the patch tokens).
        half: Run the backbone in fp16 on CUDA.
    """

    def __init__(self, device: str = "cuda", crop_hw: tuple[int, int] = (256, 128),
                 pooling: str = "cls", half: bool = True, model_id: str = MODEL_ID):  # fmt: skip
        import torch
        from transformers import AutoModel

        if pooling not in ("cls", "cls+mean"):
            raise ValueError(pooling)
        self.device = torch.device(device)
        self.dtype = torch.float16 if half and self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self.crop_hw, self.pooling = tuple(crop_hw), pooling

    def embed_crops(self, crops_rgb: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Embed RGB crops -> ``(N, D)`` float32, L2-normalised rows (D = 384 or 768)."""
        import cv2
        import torch

        h, w = self.crop_hw
        feats = []
        for s in range(0, len(crops_rgb), batch_size):
            chunk = [cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
                     for c in crops_rgb[s : s + batch_size]]  # fmt: skip
            x = (np.stack(chunk).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device, self.dtype)  # (B, 3, H, W)
            with torch.inference_mode():
                tokens = self.model(pixel_values=x).last_hidden_state  # (B, 1 + 4 reg + P, 384)
            f = tokens[:, 0]  # CLS token after the final LayerNorm
            if self.pooling == "cls+mean":
                f = torch.cat([f, tokens[:, self.n_prefix :].mean(dim=1)], dim=1)
            feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
        dim = self.model.config.hidden_size * (2 if self.pooling == "cls+mean" else 1)
        return np.concatenate(feats) if feats else np.zeros((0, dim), np.float32)

    def __call__(self, img_bgr: np.ndarray, dets_xywh: np.ndarray) -> list[np.ndarray | None]:
        """Ultralytics ReID interface: BGR frame + ``(cx, cy, w, h, ...)`` rows -> features."""
        d = np.asarray(dets_xywh, dtype=np.float64).reshape(len(dets_xywh), -1)[:, :4]
        xyxy = np.concatenate([d[:, :2] - d[:, 2:] / 2, d[:, :2] + d[:, 2:] / 2], axis=1)
        crops = crop_boxes(img_bgr[..., ::-1], xyxy)  # BGR -> RGB view
        valid = [c.shape[0] > 1 and c.shape[1] > 1 for c in crops]
        emb = iter(self.embed_crops([np.ascontiguousarray(c) for c, v in zip(crops, valid) if v]))
        return [next(emb) if v else None for v in valid]
