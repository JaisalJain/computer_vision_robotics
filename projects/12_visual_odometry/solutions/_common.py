"""Shared helpers for the exercise solutions (run them from the repository root)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vo  # noqa: E402

from cvlab import tum  # noqa: E402
from cvlab.io import data_dir  # noqa: E402

SEQUENCES = {
    "fr1_xyz": ("fr1_xyz", "rgbd_dataset_freiburg1_xyz"),
    "fr1_desk": ("fr1_desk", "rgbd_dataset_freiburg1_desk"),
}


def load(name: str = "fr1_xyz") -> tum.TumSequence:
    return tum.load_sequence(data_dir("tum", *SEQUENCES[name]))


def gray(seq, k: int) -> np.ndarray:
    return cv2.imread(str(seq.rgb_paths[k]), cv2.IMREAD_GRAYSCALE)


__all__ = ["SEQUENCES", "gray", "load", "vo"]
