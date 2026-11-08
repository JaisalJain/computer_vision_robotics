"""CPU tests for the SAM 3 post-processing replacement of project 06 (no model is loaded)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/06_tracking/scripts"))

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
from sam3video import TorchCvUtils  # noqa: E402


def test_generic_nms_suppresses_overlaps_by_score():
    ious = torch.tensor([[1.0, 0.8, 0.0], [0.8, 1.0, 0.05], [0.0, 0.05, 1.0]])
    probs = torch.tensor([0.6, 0.9, 0.7])
    keep = TorchCvUtils.generic_nms(ious, probs, iou_threshold=0.1)
    assert keep.tolist() == [1, 2]  # 0 overlaps the higher-scoring 1; 2 is separate


def test_cc_2d_labels_and_areas():
    m = np.zeros((1, 1, 6, 8), np.uint8)
    m[0, 0, 1:3, 1:3] = 1  # 4-pixel blob
    m[0, 0, 4, 5:8] = 1  # 3-pixel blob
    labels, counts = TorchCvUtils.cc_2d(torch.from_numpy(m))
    lab, cnt = labels[0, 0].numpy(), counts[0, 0].numpy()
    assert lab[1, 1] == lab[2, 2] != lab[4, 6]
    assert cnt[1, 1] == 4 and cnt[4, 6] == 3
    assert cnt[0, 0] == 6 * 8 - 7  # background component area
