"""
main.py – Training entry point for Tokamak blob detection.

Usage:
    python main.py                            # uses config/config.yaml defaults
    python main.py --data-dir /path/to/train  # override data directory
    python main.py --no-pl --no-da            # supervised-only (fast baseline)

The script:
  1. Loads config/config.yaml (CLI flags override individual keys).
  2. Runs the 3-phase training pipeline with full MLflow tracking.
  3. Saves the final model to models/best_model.pth.
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import torch
import yaml

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.tokam2d_utils.dataset import TokamDataset  # noqa: E402
from src.training import train
from src.model import save_model
from src.utils import setup_logging


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train Tokamak blob detector")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data-dir", default=None, help="Override data.train_dir in config")
    p.add_argument("--no-pl", action="store_true", help="Disable pseudo-labeling")
    p.add_argument("--no-da", action="store_true", help="Disable domain adaptation")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.data_dir:
        cfg["data"]["train_dir"] = args.data_dir
    if args.no_pl:
        cfg["pseudo_labeling"]["enabled"] = False
    if args.no_da:
        cfg["domain_adaptation"]["enabled"] = False

    setup_logging(cfg["artifacts"]["log_file"])
    logger = logging.getLogger(__name__)

    torch.manual_seed(cfg["training"]["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    training_path = Path(cfg["data"]["train_dir"])
    if not training_path.exists():
        logger.error("Training directory not found: %s", training_path)
        sys.exit(1)

    # Discover turb_i
    turb_h5 = next(training_path.glob("turb_i*.h5"), None)
    if turb_h5:
        logger.info("Found turb_i: %s", turb_h5)
    else:
        logger.warning("turb_i not found — pseudo-labeling and DA will be skipped.")
        cfg["pseudo_labeling"]["enabled"] = False
        cfg["domain_adaptation"]["enabled"] = False

    # Load dataset
    logger.info("Loading dataset from %s", training_path)
    full_dataset = TokamDataset(training_path, include_unlabeled=False)
    logger.info("Total labeled frames: %d", len(full_dataset))
    if len(full_dataset) == 0:
        logger.error("No labeled frames found.")
        sys.exit(1)

    # MLflow run
    mlfc = cfg["mlflow"]
    mlflow.set_tracking_uri(mlfc["tracking_uri"])
    mlflow.set_experiment(mlfc["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "phase1_epochs": cfg["training"]["phase1_epochs"],
            "lr": cfg["training"]["lr"],
            "batch_size": cfg["training"]["batch_size"],
            "pl_enabled": cfg["pseudo_labeling"]["enabled"],
            "pl_score_thresh": cfg["pseudo_labeling"]["score_thresh"],
            "pl_iterations": cfg["pseudo_labeling"]["iterations"],
            "da_enabled": cfg["domain_adaptation"]["enabled"],
            "da_lambda_mmd": cfg["domain_adaptation"]["lambda_mmd"],
        })
        mlflow.log_artifact("config/config.yaml")

        model = train(full_dataset, turb_h5, cfg, device)

        model_path = Path(cfg["artifacts"]["model_path"])
        save_model(model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Done. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
