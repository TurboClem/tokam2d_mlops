"""
Data augmentation transforms for Tokamak blob detection.

Extracted from the best-performing competition submission.

Key design decision: Per-image min-max normalization (NOT dataset statistics).
This is applied in TokamDataset.__getitem__ already on raw sensor data, then
again after augmentation via PerImageMinMaxNormalize to handle range drift.
"""

import math
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tgt):
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt


class ToTensor:
    def __call__(self, img, tgt):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        return img, tgt


class PerImageMinMaxNormalize:
    """Normalize each image independently to [0, 1] using its own min/max.

    Applied after augmentation to re-normalise any range drift caused by
    intensity transforms. This is the key normalisation choice that outperformed
    dataset-level statistics in competition experiments.
    """

    def __call__(self, img, tgt):
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn) if mx > mn else torch.zeros_like(img)
        return img, tgt


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            _, w = img.shape[-2:]
            img = TF.hflip(img)
            if tgt["boxes"] is not None and len(tgt["boxes"]):
                b = tgt["boxes"].clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
                tgt["boxes"] = b
        return img, tgt


class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            h, _ = img.shape[-2:]
            img = TF.vflip(img)
            if tgt["boxes"] is not None and len(tgt["boxes"]):
                b = tgt["boxes"].clone()
                b[:, [1, 3]] = h - b[:, [3, 1]]
                tgt["boxes"] = b
        return img, tgt


class RandomRotation:
    def __init__(self, deg=20, p=0.6):
        self.deg = deg
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            angle = random.uniform(-self.deg, self.deg)
            _, h, w = img.shape
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            if tgt["boxes"] is not None and len(tgt["boxes"]):
                cos_a = math.cos(math.radians(angle))
                sin_a = math.sin(math.radians(angle))
                cx, cy = w / 2, h / 2
                new_boxes = []
                for x1, y1, x2, y2 in tgt["boxes"]:
                    corners = torch.tensor(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32
                    )
                    corners[:, 0] -= cx
                    corners[:, 1] -= cy
                    nx = corners[:, 0] * cos_a - corners[:, 1] * sin_a + cx
                    ny = corners[:, 0] * sin_a + corners[:, 1] * cos_a + cy
                    new_boxes.append(
                        [
                            nx.min().clamp(0, w),
                            ny.min().clamp(0, h),
                            nx.max().clamp(0, w),
                            ny.max().clamp(0, h),
                        ]
                    )
                tgt["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
        return img, tgt


class GaussianBlur:
    def __init__(self, k=7, sigma=(0.1, 3.0), p=0.5):
        self.blur = torchvision.transforms.GaussianBlur(k, sigma)
        self.p = p

    def __call__(self, img, tgt):
        return (self.blur(img), tgt) if random.random() < self.p else (img, tgt)


class IntensityJitter:
    """Grayscale-safe brightness/contrast jitter directly on float tensors.

    Uses additive brightness and multiplicative contrast (no colour ops),
    which is appropriate for single-channel physics simulation images.
    """

    def __init__(self, brightness=0.4, contrast=0.4, p=0.9):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            b = random.uniform(-self.brightness, self.brightness)
            c = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = img.mean()
            img = torch.clamp((img - mean) * c + mean + b, 0.0, 1.0)
        return img, tgt


class GaussianNoise:
    def __init__(self, std=0.08, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = torch.clamp(img + torch.randn_like(img) * self.std, 0, 1)
        return img, tgt


class RandomErasing:
    def __init__(self, p=0.4, scale=(0.02, 0.2)):
        self.p = p
        self.scale = scale

    def __call__(self, img, tgt):
        if random.random() < self.p:
            _, h, w = img.shape
            area = h * w * random.uniform(*self.scale)
            ar = random.uniform(0.3, 3.3)
            ph, pw = int(np.sqrt(area * ar)), int(np.sqrt(area / ar))
            if ph < h and pw < w:
                t, l = random.randint(0, h - ph), random.randint(0, w - pw)
                img[:, t : t + ph, l : l + pw] = torch.rand_like(
                    img[:, t : t + ph, l : l + pw]
                )
        return img, tgt


class ResizedCropWithBoxes:
    def __init__(self, size, scale=(0.4, 1.0), ratio=(0.75, 1.33)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, tgt):
        _, h, w = img.shape
        for _ in range(20):
            area = h * w * random.uniform(*self.scale)
            ar = random.uniform(*self.ratio)
            ch, cw = int(np.sqrt(area / ar)), int(np.sqrt(area * ar))
            if not (0 < ch <= h and 0 < cw <= w):
                continue
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            if tgt["boxes"] is not None and len(tgt["boxes"]):
                in_crop = [
                    b[2] > left and b[0] < left + cw and b[3] > top and b[1] < top + ch
                    for b in tgt["boxes"]
                ]
                if not any(in_crop) and random.random() > 0.3:
                    continue
            img_c = TF.resize(img[:, top : top + ch, left : left + cw], self.size)
            if tgt["boxes"] is not None and len(tgt["boxes"]):
                sx, sy = self.size[1] / cw, self.size[0] / ch
                b = tgt["boxes"].clone().float()
                b[:, [0, 2]] = (b[:, [0, 2]] - left).clamp(0, cw) * sx
                b[:, [1, 3]] = (b[:, [1, 3]] - top).clamp(0, ch) * sy
                valid = (b[:, 2] - b[:, 0] > 1) & (b[:, 3] - b[:, 1] > 1)
                tgt["boxes"] = b[valid] if valid.any() else None
                tgt["labels"] = (
                    torch.ones(valid.sum(), dtype=torch.int64) if valid.any() else None
                )
            return img_c, tgt
        return TF.resize(img, self.size), tgt


class ElasticTransform:
    def __init__(self, alpha=60.0, sigma=6.0, p=0.35):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            c, h, w = img.shape
            np_img = img.numpy()
            dx = np.random.randn(h, w) * self.sigma * self.alpha / (h - 1)
            dy = np.random.randn(h, w) * self.sigma * self.alpha / (w - 1)
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            out = np.zeros_like(np_img)
            for i in range(c):
                out[i] = map_coordinates(
                    np_img[i],
                    [(ys + dy).ravel(), (xs + dx).ravel()],
                    order=1,
                    mode="reflect",
                ).reshape(h, w)
            img = torch.from_numpy(out).float()
        return img, tgt


class MixUp:
    """Blend image with a random sample from the training set."""

    def __init__(self, alpha=0.4, p=0.4):
        self.alpha = alpha
        self.p = p
        self._ds = self._idx = None

    def set_dataset(self, ds, idx):
        self._ds = ds
        self._idx = idx

    def __call__(self, img, tgt):
        if random.random() < self.p and self._ds is not None:
            mix_img, mix_tgt = self._ds[random.choice(self._idx)]
            lam = max(np.random.beta(self.alpha, self.alpha), 0.5)
            img = lam * img + (1 - lam) * mix_img.to(img.device)
            if tgt["boxes"] is not None and mix_tgt["boxes"] is not None:
                tgt["boxes"] = torch.cat([tgt["boxes"], mix_tgt["boxes"]], 0)
                tgt["labels"] = torch.ones(len(tgt["boxes"]), dtype=torch.int64)
        return img, tgt


class AugTransform:
    """Combines base pipeline with MixUp (requires dataset access)."""

    def __init__(self, base, mixup):
        self.base = base
        self.mixup = mixup

    def __call__(self, img, tgt):
        return self.mixup(*self.base(img, tgt))


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

IMG_SIZE = (512, 512)


def get_train_transform():
    """Full aggressive training pipeline (best competition config)."""
    return Compose(
        [
            IntensityJitter(brightness=0.4, contrast=0.4, p=0.9),
            GaussianNoise(std=0.08, p=0.5),
            GaussianBlur(k=7, sigma=(0.1, 3.0), p=0.5),
            ResizedCropWithBoxes(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            RandomHFlip(0.5),
            RandomVFlip(0.3),
            RandomRotation(deg=25, p=0.6),
            RandomErasing(p=0.35),
            ElasticTransform(alpha=60.0, sigma=6.0, p=0.35),
            ToTensor(),
            PerImageMinMaxNormalize(),
        ]
    )


def get_val_transform():
    """Minimal validation pipeline."""
    return Compose([ToTensor(), PerImageMinMaxNormalize()])


def get_pl_transform():
    """Lighter augmentation for pseudo-labeled turb_i data (noisy labels)."""
    return Compose(
        [
            IntensityJitter(brightness=0.2, contrast=0.2, p=0.8),
            GaussianNoise(std=0.05, p=0.4),
            GaussianBlur(k=5, sigma=(0.1, 2.0), p=0.4),
            ResizedCropWithBoxes(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            RandomHFlip(0.5),
            RandomVFlip(0.3),
            ToTensor(),
            PerImageMinMaxNormalize(),
        ]
    )
