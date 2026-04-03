"""
MMD-based domain adaptation (Phase 3 of training).

Aligns backbone/FPN features of labeled source frames (blob_dwi)
toward unlabeled target frames (turb_i) using Maximum Mean Discrepancy.
Only the backbone and RPN are updated — the detection heads are frozen.
"""

import logging

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MMD kernel
# ---------------------------------------------------------------------------

def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xx = (x ** 2).sum(1, keepdim=True)
    yy = (y ** 2).sum(1, keepdim=True)
    return torch.exp(-(xx + yy.t() - 2 * (x @ y.t())) / (2 * sigma ** 2))


def mmd_loss(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    sigmas: tuple = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Multi-kernel MMD. Lower = distributions are closer."""
    loss = source_feat.new_zeros(1).squeeze()
    for s in sigmas:
        loss = (
            loss
            + _gaussian_kernel(source_feat, source_feat, s).mean()
            + _gaussian_kernel(target_feat, target_feat, s).mean()
            - 2 * _gaussian_kernel(source_feat, target_feat, s).mean()
        )
    return loss / len(sigmas)


# ---------------------------------------------------------------------------
# FPN hook
# ---------------------------------------------------------------------------

class _FPNHook:
    def __init__(self, fpn: nn.Module):
        self.feat: torch.Tensor | None = None
        self._h = fpn.register_forward_hook(self._fn)

    def _fn(self, m, inp, out):
        feat = (
            out.get("0", list(out.values())[0])
            if isinstance(out, dict)
            else (out[0] if isinstance(out, (list, tuple)) else out)
        )
        self.feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)

    def remove(self):
        self._h.remove()


# ---------------------------------------------------------------------------
# DA fine-tuning
# ---------------------------------------------------------------------------

def domain_adaptation_finetune(
    model: nn.Module,
    labeled_dataset,
    labeled_indices: list,
    turb_h5_path,
    device: str,
    val_transform,
    da_epochs: int = 3,
    da_lr: float = 3e-5,
    lambda_mmd: float = 0.4,
    batch_size: int = 2,
) -> nn.Module:
    """
    Fine-tune backbone + FPN with MMD feature alignment.

    Uses ALL turb frames (typically only 8) as the target — small enough
    to stack them into memory, which simplifies the inner loop.

    Args:
        model:           Trained Faster R-CNN (will be moved to `device`).
        labeled_dataset: Full labeled TokamDataset.
        labeled_indices: Indices to use as source domain.
        turb_h5_path:    Path to turb_i HDF5 file.
        device:          "cuda" or "cpu".
        val_transform:   Validation transform (ToTensor + PerImageMinMaxNormalize).
        da_epochs, da_lr, lambda_mmd, batch_size: hyperparameters.

    Returns:
        Fine-tuned model in eval mode on CPU.
    """
    from src.model.dataset import TransformSubset

    def collate_fn(batch):
        return tuple(zip(*batch))

    logger.info(
        "=== Domain Adaptation | λ=%.2f  epochs=%d  lr=%.0e ===",
        lambda_mmd, da_epochs, da_lr,
    )

    # Source loader (labeled frames with val normalisation)
    src_ds = TransformSubset(labeled_dataset, labeled_indices, val_transform)
    src_ld = DataLoader(
        src_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Pre-load ALL turb frames (typically 8 — fits easily in memory)
    with h5py.File(str(turb_h5_path), "r") as f:
        key = list(f.keys())[0]
        raw = f[key][:]

    turb_frames = []
    for frame in raw:
        frame = frame.astype(np.float32)
        if frame.ndim == 2:
            frame = frame[np.newaxis]
        t = torch.from_numpy(frame)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        img, _ = val_transform(t, {"boxes": None, "labels": None, "frame_index": -1})
        turb_frames.append(img)

    turb_stack = torch.stack(turb_frames).to(device)
    logger.info(
        "Source: %d labeled frames | Target: %d turb frames",
        len(src_ds), len(turb_frames),
    )

    hook = _FPNHook(model.backbone.fpn)
    da_params = list(model.backbone.parameters()) + list(model.rpn.parameters())
    opt = torch.optim.AdamW(da_params, lr=da_lr, weight_decay=1e-4)
    model.to(device)

    for epoch in range(da_epochs):
        det_sum = mmd_sum = steps = 0
        model.train()

        for imgs, tgts in src_ld:
            imgs = [img.to(device) for img in imgs]
            tgts_d = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in tgts
            ]
            vi = [img for img, t in zip(imgs, tgts_d) if t["boxes"] is not None and len(t["boxes"]) > 0]
            vt = [t for t in tgts_d if t["boxes"] is not None and len(t["boxes"]) > 0]
            if not vi:
                continue

            det_loss = sum(model(vi, vt).values())
            src_feat = hook.feat

            # Turb forward (backbone only, no detection heads)
            model.eval()
            with torch.enable_grad():
                fd = model.backbone(turb_stack)
                finest = fd.get("0", list(fd.values())[0])
                trb_feat = F.adaptive_avg_pool2d(finest, (1, 1)).squeeze(-1).squeeze(-1)
            model.train()

            mmd = (
                mmd_loss(src_feat, trb_feat)
                if src_feat is not None
                else det_loss.new_zeros(1).squeeze()
            )
            total = det_loss + lambda_mmd * mmd

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(da_params, 1.0)
            opt.step()

            det_sum += det_loss.item()
            mmd_sum += mmd.item()
            steps += 1

        logger.info(
            "DA Epoch %d/%d | det=%.4f | mmd=%.4f",
            epoch + 1, da_epochs, det_sum / max(1, steps), mmd_sum / max(1, steps),
        )

    hook.remove()
    logger.info("Domain adaptation complete.")
    return model.eval().to("cpu")
