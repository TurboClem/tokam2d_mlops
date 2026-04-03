"""
Unit tests for transforms, metrics and inference utilities.

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transforms.augmentations import (
    Compose,
    IntensityJitter,
    MixUp,
    PerImageMinMaxNormalize,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
    get_train_transform,
    get_val_transform,
)
from src.evaluation.metrics import compute_ap50
from src.evaluation.inference import weighted_box_fusion


# ── helpers ──────────────────────────────────────────────────────────────────

def dummy_target(n_boxes=2):
    boxes = torch.tensor([[50.0, 50.0, 150.0, 150.0], [200.0, 200.0, 300.0, 300.0]][:n_boxes])
    return {"boxes": boxes, "labels": torch.ones(n_boxes, dtype=torch.int64), "frame_index": "test-0"}


def dummy_image(c=3, h=512, w=512):
    return torch.rand(c, h, w)


# ── transform tests ───────────────────────────────────────────────────────────

class TestToTensor:
    def test_numpy_becomes_tensor(self):
        img = np.random.rand(3, 64, 64).astype(np.float32)
        out, _ = ToTensor()(img, dummy_target())
        assert isinstance(out, torch.Tensor) and out.dtype == torch.float32

    def test_tensor_passthrough(self):
        img = torch.rand(3, 64, 64)
        out, _ = ToTensor()(img, dummy_target())
        assert torch.allclose(img, out)


class TestPerImageMinMaxNormalize:
    def test_output_in_zero_one(self):
        img = torch.rand(3, 64, 64) * 100 + 500   # arbitrary range
        out, _ = PerImageMinMaxNormalize()(img, dummy_target())
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6

    def test_constant_image_becomes_zeros(self):
        img = torch.ones(3, 32, 32) * 7.0
        out, _ = PerImageMinMaxNormalize()(img, dummy_target())
        assert torch.all(out == 0)


class TestRandomHFlip:
    def test_always_flip_moves_boxes(self):
        img = torch.zeros(3, 512, 512)
        tgt = {"boxes": torch.tensor([[100.0, 50.0, 200.0, 150.0]]),
               "labels": torch.ones(1, dtype=torch.int64), "frame_index": "t"}
        _, out = RandomHFlip(p=1.0)(img, tgt)
        # x1_new = 512 - 200 = 312
        assert out["boxes"][0, 0].item() == pytest.approx(312.0)
        assert out["boxes"][0, 2].item() == pytest.approx(412.0)

    def test_never_flip_preserves_boxes(self):
        img = dummy_image()
        tgt = dummy_target()
        orig = tgt["boxes"].clone()
        _, out = RandomHFlip(p=0.0)(img, tgt)
        assert torch.allclose(out["boxes"], orig)


class TestRandomVFlip:
    def test_always_flip_moves_boxes(self):
        img = torch.zeros(3, 512, 512)
        tgt = {"boxes": torch.tensor([[50.0, 100.0, 150.0, 200.0]]),
               "labels": torch.ones(1, dtype=torch.int64), "frame_index": "t"}
        _, out = RandomVFlip(p=1.0)(img, tgt)
        # y1_new = 512 - 200 = 312
        assert out["boxes"][0, 1].item() == pytest.approx(312.0)
        assert out["boxes"][0, 3].item() == pytest.approx(412.0)


class TestIntensityJitter:
    def test_output_clipped_to_zero_one(self):
        img = torch.rand(3, 64, 64)
        out, _ = IntensityJitter(brightness=0.4, contrast=0.4, p=1.0)(img, dummy_target())
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


class TestGetValTransform:
    def test_numpy_input(self):
        img = np.random.rand(3, 128, 128).astype(np.float32)
        out, _ = get_val_transform()(img, dummy_target())
        assert isinstance(out, torch.Tensor)
        assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


# ── AP50 tests ────────────────────────────────────────────────────────────────

class _PerfectModel(torch.nn.Module):
    def __init__(self, gt_boxes):
        super().__init__()
        self.gt = gt_boxes

    def forward(self, images):
        return [{"boxes": self.gt, "scores": torch.ones(len(self.gt))}]

    def eval(self): return self


class _ZeroModel(torch.nn.Module):
    def forward(self, images):
        return [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0)}]

    def eval(self): return self


class _MockLoader:
    def __init__(self, imgs, tgts):
        self.data = [(imgs, tgts)]

    def __iter__(self):
        return iter(self.data)


class TestComputeAP50:
    def _loader(self, gt_boxes):
        tgt = {"boxes": gt_boxes, "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
               "frame_index": "test-0"}
        return _MockLoader([dummy_image()], [tgt])

    def test_perfect_predictions_give_one(self):
        gt = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
        ap = compute_ap50(_PerfectModel(gt), self._loader(gt), "cpu")
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_no_predictions_give_zero(self):
        gt = torch.tensor([[50.0, 50.0, 150.0, 150.0]])
        ap = compute_ap50(_ZeroModel(), self._loader(gt), "cpu")
        assert ap == pytest.approx(0.0, abs=0.01)


# ── WBF tests ─────────────────────────────────────────────────────────────────

class TestWeightedBoxFusion:
    def test_identical_boxes_fuse_to_one(self):
        box = np.array([[100.0, 100.0, 200.0, 200.0]])
        boxes, scores = weighted_box_fusion([box, box], [np.array([0.9]), np.array([0.8])])
        assert len(boxes) == 1
        assert scores[0] == pytest.approx(0.85)

    def test_non_overlapping_boxes_kept_separate(self):
        b1 = np.array([[0.0, 0.0, 50.0, 50.0]])
        b2 = np.array([[400.0, 400.0, 500.0, 500.0]])
        boxes, scores = weighted_box_fusion([b1], [np.array([0.9])],
                                            iou_thresh=0.4, img_size=512)
        assert len(boxes) == 1

    def test_empty_input_returns_empty(self):
        boxes, scores = weighted_box_fusion([np.zeros((0, 4))], [np.zeros(0)])
        assert len(boxes) == 0 and len(scores) == 0
