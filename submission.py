"""
submission.py – Entry point called by the Codabench ingestion program.

This file is the only interface required by the competition platform.
It delegates everything to the modular src/ package.
"""

import sys
from pathlib import Path

# Ensure src/ is importable when called by ingestion_program
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml

from src.domain_adaptation import domain_adaptation_finetune
from src.evaluation import compute_dataset_stats, compute_turb_stats
from src.model import build_model, TransformSubset
from src.transforms import AdvancedAugmentation, get_train_transform, get_val_transform

from ingestion_program.tokam2d_utils.dataset import TokamDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(training_dir, n_folds: int = 5):
    """
    Train the Faster R-CNN detector with K-Fold CV + optional domain adaptation.

    Called by ingestion_program/ingestion.py.

    Args:
        training_dir: Path to the training data directory.
        n_folds:      Number of cross-validation folds.

    Returns:
        Trained torch.nn.Module in eval mode on CPU.
    """
    cfg = yaml.safe_load(open(Path(__file__).parent / "config" / "config.yaml"))
    tc = cfg["training"]
    mc = cfg["model"]
    da = cfg["domain_adaptation"]

    training_path = Path(training_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset & normalisation ───────────────────────────────────────────────
    full_dataset = TokamDataset(training_path, include_unlabeled=False)
    total_frames = len(full_dataset)
    if total_frames == 0:
        raise ValueError("No labeled data found in training directory.")

    turb_h5 = next(training_path.glob("turb_i*.h5"), None)
    if turb_h5 is not None:
        mean, std = compute_turb_stats(turb_h5, num_samples=da["turb_stats_samples"])
    else:
        mean, std = compute_dataset_stats(full_dataset, num_samples=100)

    img_size = tuple(cfg["data"]["img_size"])
    base_train_tf = get_train_transform(mean=mean.tolist(), std=std.tolist(), img_size=img_size)
    val_tf = get_val_transform(mean=mean.tolist(), std=std.tolist())

    # ── K-Fold CV ─────────────────────────────────────────────────────────────
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=tc["random_seed"])
    best_ap, best_state = 0.0, None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(total_frames))):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        train_tf = AdvancedAugmentation(base_train_tf, full_dataset, list(train_idx))
        train_set = TransformSubset(full_dataset, list(train_idx), train_tf)
        val_set = TransformSubset(full_dataset, list(val_idx), val_tf)

        train_loader = DataLoader(train_set, batch_size=tc["batch_size"], shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)

        model = build_model(score_thresh=mc["score_thresh"], nms_thresh=mc["nms_thresh"],
                            detections_per_img=mc["detections_per_img"])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **tc["lr_scheduler"])

        from src.evaluation import compute_ap50
        fold_best_ap, fold_best_state, patience_counter = 0.0, None, 0

        for epoch in range(tc["max_epochs"]):
            model.train()
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in t.items()} for t in targets]
                valid_imgs = [i for i, t in zip(images, targets_dev)
                              if t["boxes"] is not None and len(t["boxes"]) > 0]
                valid_tgts = [t for t in targets_dev
                              if t["boxes"] is not None and len(t["boxes"]) > 0]
                if not valid_imgs:
                    continue
                losses = sum(model(valid_imgs, valid_tgts).values())
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
                optimizer.step()

            ap = compute_ap50(model, val_loader, device)
            scheduler.step(ap)
            print(f"  Epoch {epoch + 1}/{tc['max_epochs']} | AP50={ap:.4f}")

            if ap > fold_best_ap:
                fold_best_ap = ap
                fold_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= tc["early_stopping_patience"]:
                    break

        print(f"Fold {fold + 1} best AP50: {fold_best_ap:.4f}")
        if fold_best_ap > best_ap:
            best_ap = fold_best_ap
            best_state = fold_best_state

    # ── Final model ───────────────────────────────────────────────────────────
    final_model = build_model(score_thresh=mc["score_thresh"], nms_thresh=mc["nms_thresh"],
                              detections_per_img=mc["detections_per_img"], pretrained=False)
    if best_state:
        final_model.load_state_dict(best_state)

    # ── Domain adaptation ─────────────────────────────────────────────────────
    if da["enabled"] and turb_h5 is not None:
        final_model = domain_adaptation_finetune(
            model=final_model.to(device),
            labeled_dataset=full_dataset,
            labeled_indices=list(range(total_frames)),
            turb_h5_path=turb_h5,
            device=device,
            turb_mean=mean if isinstance(mean, torch.Tensor) else torch.tensor(mean),
            turb_std=std if isinstance(std, torch.Tensor) else torch.tensor(std),
            num_epochs=da["num_epochs"],
            lambda_mmd=da["lambda_mmd"],
            lr=da["lr"],
            batch_size=da["batch_size"],
        )

    return final_model.eval().to("cpu")
