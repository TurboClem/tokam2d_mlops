"""
api.py – FastAPI inference server for Tokamak blob detection.

Endpoints:
    GET  /          → health check
    GET  /info      → model metadata
    POST /predict   → detect blobs in an uploaded HDF5 frame

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Environment variables (set in .env, never commit):
    MODEL_PATH   path to .pth weights   (default: models/best_model.pth)
    DEVICE       "cpu" or "cuda"        (default: auto-detect)
"""

import io
import logging
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.model import build_model
from src.transforms import get_val_transform
from src.evaluation import predict_with_tta

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tokamak Blob Detector",
    description=(
        "Faster R-CNN (ResNet-50 FPN) trained on TOKAM2D simulation data.\n\n"
        "Upload an HDF5 frame and get bounding boxes around plasma blobs.\n"
        "Inference uses hflip+vflip TTA fused with Weighted Box Fusion."
    ),
    version="1.0.0",
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pth"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_model: torch.nn.Module | None = None
_val_transform = get_val_transform()


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model weights not found at {MODEL_PATH}. "
                "Run `python main.py` to train first."
            )
        _model = build_model(pretrained=False)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE).eval()
        logger.info("Model loaded: %s on %s", MODEL_PATH, DEVICE)
    return _model


@app.on_event("startup")
async def _startup():
    try:
        _get_model()
    except RuntimeError as e:
        logger.warning("Model not loaded at startup: %s", e)


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.get("/info", summary="Model info")
def info():
    return {
        "architecture": "fasterrcnn_resnet50_fpn",
        "weights": str(MODEL_PATH),
        "device": DEVICE,
        "inference": "TTA (hflip + vflip) + Weighted Box Fusion",
        "input": "HDF5 file (.h5) with density frames",
        "output": "list of {box: [x1,y1,x2,y2], score: float}",
    }


@app.post("/predict", summary="Detect blobs in an HDF5 frame")
async def predict(file: UploadFile = File(...)):
    """
    Upload an HDF5 file containing one or more tokamak density frames.
    Returns detections for the **first** frame found.

    The HDF5 file must have a top-level dataset (any name) with shape
    (N, H, W) or (H, W).
    """
    if not (file.filename or "").endswith(".h5"):
        raise HTTPException(status_code=400, detail="File must be a .h5 HDF5 file.")

    try:
        import h5py

        contents = await file.read()
        with h5py.File(io.BytesIO(contents), "r") as f:
            key = list(f.keys())[0]
            data = f[key]
            frame = (data[0] if data.ndim == 3 else data[:]).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read HDF5: {e}")

    # Normalise and convert to tensor (C, H, W)
    if frame.ndim == 2:
        frame = frame[np.newaxis]
    t = torch.from_numpy(frame)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)

    # Apply val transform (PerImageMinMaxNormalize)
    img, _ = _val_transform(t, {"boxes": None, "labels": None, "frame_index": -1})

    try:
        model = _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    boxes, scores = predict_with_tta(model, img, DEVICE)

    detections = [
        {"box": box.tolist(), "score": float(score)}
        for box, score in zip(boxes, scores)
    ]
    return JSONResponse({"file": file.filename, "n_detections": len(detections), "detections": detections})
