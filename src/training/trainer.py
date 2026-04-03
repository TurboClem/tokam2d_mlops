"""
3-phase training pipeline for Tokamak blob detection.

Phase 1 — Supervised warm-up on blob_dwi frames only (blob_i hurts).
Phase 2 — Iterative pseudo-labeling on turb_i with a strict quality gate.
Phase 3 — MMD domain adaptation (backbone + FPN only).

All hyperparameters come from config/config.yaml, loaded via the cfg dict.
"""

import logging
import math

import mlflow
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.domain_adaptation import domain_adaptation_finetune, generate_pseudo_labels
from src.evaluation import compute_ap50
from src.model import PseudoLabelDataset, TransformSubset, build_model
from src.transforms import AugTransform, MixUp, get_pl_transform, get_train_transform, get_val_transform

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def _select_blob_dwi_indices(dataset) -> list[int]:
    """Return only blob_dwi frame indices.

    KEY INSIGHT from competition: training on blob_i frames alongside blob_dwi
    consistently hurt leaderboard performance. blob_dwi only = best results.
    """
    dwi = [i for i, fid in enumerate(dataset.idx_to_frame) if "blob_dwi" in fid]
    other = [i for i, fid in enumerate(dataset.idx_to_frame) if "blob_dwi" not in fid]
    logger.info(
        "blob_dwi: %d frames | other labeled (excluded): %d frames", len(dwi), len(other)
    )
    if not dwi:
        logger.warning("No blob_dwi frames found — using ALL labeled frames.")
        return list(range(len(dataset)))
    return dwi


def _run_epoch(model, loader, optimizer, device, tag: str = "") -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total, n = 0.0, 0
    for imgs, tgts in loader:
        imgs = [img.to(device) for img in imgs]
        tgts_d = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in tgts
        ]
        vi = [img for img, t in zip(imgs, tgts_d) if t["boxes"] is not None and len(t["boxes"]) > 0]
        vt = [t for t in tgts_d if t["boxes"] is not None and len(t["boxes"]) > 0]
        if not vi:
            continue
        loss = sum(model(vi, vt).values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    avg = total / max(1, n)
    logger.info("  [%s] loss: %.4f", tag, avg)
    return avg


def train(full_dataset, turb_h5_path, cfg: dict, device: str) -> torch.nn.Module:
    """
    Run the full 3-phase training pipeline.

    Args:
        full_dataset:  TokamDataset with all labeled frames.
        turb_h5_path:  Path to turb_i HDF5, or None.
        cfg:           Parsed config dict (from config/config.yaml).
        device:        "cuda" or "cpu".

    Returns:
        Trained model in eval mode on CPU.
    """
    tc = cfg["training"]
    mc = cfg["model"]
    pl = cfg["pseudo_labeling"]
    da = cfg["domain_adaptation"]

    train_indices = _select_blob_dwi_indices(full_dataset)
    logger.info("Training on %d blob_dwi frames.", len(train_indices))

    val_tf = get_val_transform()

    # =========================================================================
    # PHASE 1 — Supervised warm-up on blob_dwi only
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1 — Supervised warm-up (%d epochs, blob_dwi only)", tc["phase1_epochs"])
    logger.info("=" * 60)

    base_tf = get_train_transform()
    mixup = MixUp(alpha=0.4, p=0.4)
    mixup.set_dataset(full_dataset, train_indices)
    aug_tf = AugTransform(base_tf, mixup)

    train_ds = TransformSubset(full_dataset, train_indices, aug_tf)
    train_ld = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = build_model(
        score_thresh=mc["score_thresh"],
        nms_thresh=mc["nms_thresh"],
        detections_per_img=mc["detections_per_img"],
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tc["phase1_epochs"], eta_min=1e-5
    )

    for ep in range(tc["phase1_epochs"]):
        loss = _run_epoch(model, train_ld, opt, device, tag=f"P1 {ep + 1}/{tc['phase1_epochs']}")
        scheduler.step()
        mlflow.log_metric("phase1_loss", loss, step=ep)

    # =========================================================================
    # PHASE 2 — Iterative pseudo-labeling on turb_i
    # =========================================================================
    if turb_h5_path is not None:
        for pl_iter in range(1, pl["iterations"] + 1):
            logger.info("=" * 60)
            logger.info("PHASE 2 — Pseudo-label iteration %d/%d", pl_iter, pl["iterations"])
            logger.info("=" * 60)

            model.eval().to(device)
            pl_frames, pl_targets = generate_pseudo_labels(model, turb_h5_path, device, val_tf)
            mlflow.log_metric("pl_accepted_frames", len(pl_frames), step=pl_iter)

            if len(pl_frames) < pl["min_frames"]:
                logger.warning(
                    "Only %d frames passed quality gate (need >= %d). Skipping PL round.",
                    len(pl_frames), pl["min_frames"],
                )
                continue

            pl_tf = get_pl_transform()
            pl_ds = PseudoLabelDataset(pl_frames, pl_targets, transform=pl_tf)

            oversample = max(pl["labeled_oversample"], math.ceil(len(pl_ds) / max(1, len(train_indices))))
            lab_ds = TransformSubset(full_dataset, train_indices * oversample, aug_tf)
            combined = ConcatDataset([lab_ds, pl_ds])
            combo_ld = DataLoader(
                combined,
                batch_size=tc["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            logger.info(
                "Combined: %d labeled (×%d) + %d pseudo-labeled",
                len(lab_ds), oversample, len(pl_ds),
            )

            pl_lr = tc["lr"] * pl["lr_factor"] * (0.5 ** (pl_iter - 1))
            pl_opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=pl_lr,
                weight_decay=1e-4,
            )
            model.train().to(device)

            for ep in range(pl["epochs_per_iter"]):
                loss = _run_epoch(
                    model, combo_ld, pl_opt, device,
                    tag=f"PL-{pl_iter} {ep + 1}/{pl['epochs_per_iter']}",
                )
                mlflow.log_metric(f"pl{pl_iter}_loss", loss, step=ep)
    else:
        logger.info("Skipping pseudo-labeling (no turb_i found).")

    # =========================================================================
    # PHASE 3 — MMD domain adaptation
    # =========================================================================
    if turb_h5_path is not None and da["enabled"]:
        model = domain_adaptation_finetune(
            model=model,
            labeled_dataset=full_dataset,
            labeled_indices=train_indices,
            turb_h5_path=turb_h5_path,
            device=device,
            val_transform=val_tf,
            da_epochs=da["epochs"],
            da_lr=da["lr"],
            lambda_mmd=da["lambda_mmd"],
            batch_size=tc["batch_size"],
        )
    else:
        logger.info("Skipping domain adaptation.")
        model.eval().to("cpu")

    logger.info("Training complete.")
    return model
