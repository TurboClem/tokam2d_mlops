# Tokamak Blob Detection — MLOps Project

**ENSAE Paris — Mise en Production (2024-2025)**

Automated detection of plasma blobs in 2D tokamak simulation data, using Faster R-CNN with a 3-phase training pipeline: supervised warm-up, iterative pseudo-labeling, and MMD domain adaptation.

---

## Scientific context

TOKAM2D is a 2D fluid turbulence simulation code developed at CEA/IRFM (Cadarache) that models plasma dynamics in the scrape-off layer of tokamak reactors. The simulation produces density field movies in which overdense "blobs" propagate outward — their detection and characterisation is important for understanding turbulent transport in fusion devices.

The training data comes from two simulation regimes:
- **blob_i / blob_dwi** — isolated single-blob movies (labeled, used for training)
- **turb_i** — turbulent regime, unlabeled (used for domain adaptation)
- **turb_dwi** — realistic turbulent test set (evaluation target)

---

## Architecture: 3-phase training pipeline

```
Phase 1 — Supervised warm-up
  blob_dwi frames only  (blob_i hurts leaderboard — key finding)
  Strong augmentation + MixUp
  Cosine LR schedule

Phase 2 — Iterative pseudo-labeling on turb_i
  Run model on turb_i frames
  Quality gate: score >= 0.55, area >= 100px², aspect ratio <= 8
  If < 3 frames pass gate → skip (avoid poisoning)
  Labeled data oversampled 4× vs pseudo-labels
  Lower LR (noisy labels)
  Repeat 3 times

Phase 3 — MMD domain adaptation
  Align backbone/FPN features: blob_dwi → turb_i
  Only backbone + RPN updated
  lambda_mmd = 0.4

Inference: hflip + vflip TTA fused with Weighted Box Fusion
```

---

## Project structure

```
tokam2d_mlops/
├── main.py                        # Training entry point (MLflow tracking)
├── api.py                         # FastAPI inference server (TTA + WBF)
│
├── src/
│   ├── tokam2d_utils/             # Dataset loader (HDF5 + XML annotations)
│   │   ├── dataset.py
│   │   └── xml_loader.py
│   ├── transforms/
│   │   └── augmentations.py       # All augmentation classes + pipeline factories
│   ├── model/
│   │   ├── detector.py            # build_model / save_model / load_model
│   │   └── dataset.py             # TransformSubset, PseudoLabelDataset
│   ├── training/
│   │   └── trainer.py             # 3-phase train() function
│   ├── domain_adaptation/
│   │   ├── pseudo_labels.py       # Pseudo-label generation with quality gate
│   │   └── mmd.py                 # MMD loss + FPN hook + DA fine-tuning
│   ├── evaluation/
│   │   ├── metrics.py             # compute_ap50
│   │   └── inference.py           # predict_with_tta, weighted_box_fusion
│   └── utils/
│       └── logging_setup.py
│
├── config/
│   └── config.yaml                # ALL hyperparameters (no secrets here)
├── tests/
│   └── test_pipeline.py           # pytest unit tests
│
├── .github/workflows/ci.yml       # Lint → test → Docker build/push
├── Dockerfile
├── requirements.txt
├── ruff.toml
├── .gitignore
├── .env.example                   # Template for secrets
└── LICENSE
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/tokam2d_mlops.git
cd tokam2d_mlops
pip install -r requirements.txt
```

### 2. Create your `.env` file

```bash
cp .env.example .env
# Edit .env — never commit it
```

### 3. Provide data

Data is not stored in git. Place it in `data/raw/train/`:

```
data/raw/train/
├── blob_dwi.h5
├── blob_dwi.xml      ← annotations
├── blob_i.h5
├── blob_i.xml
└── turb_i.h5         ← unlabeled, used for domain adaptation
```

Or pull from S3 on SSP Cloud (see deployment section).

### 4. Train

```bash
python main.py                             # full 3-phase pipeline
python main.py --no-pl --no-da            # supervised only (fast baseline)
python main.py --data-dir /path/to/train  # custom data path
```

Training is tracked automatically in MLflow. Start the UI with:

```bash
mlflow ui --host 0.0.0.0 --port 5000
# → http://localhost:5000
```

### 5. Run the API locally

```bash
uvicorn api:app --reload --port 8000
# → http://localhost:8000/docs
```

Then POST an `.h5` file to `/predict` to get bounding boxes.

---

## Code quality

```bash
# Lint + format check
ruff check .
ruff format --check .

# Auto-fix
ruff check --fix .
ruff format .

# Tests
python -m pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -t tokam2d .

# Run API (mount trained model weights)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=models/best_model.pth \
  tokam2d
```

---

## Deploying on SSP Cloud

### Step 1 — Push data to S3

```bash
# From a VSCode terminal on SSP Cloud
MY_BUCKET="your_sspcloud_username"
mc cp -r /path/to/local/train/ s3/${MY_BUCKET}/tokam2d/train/
```

### Step 2 — Launch MLflow

Go to **Catalogue → MLflow**, launch the service, copy its URL, then update `config/config.yaml`:

```yaml
mlflow:
  tracking_uri: "https://your-mlflow-url.lab.sspcloud.fr"
```

### Step 3 — Train

```bash
# Clone, install, then:
python main.py --data-dir /path/to/s3/mounted/train
```

Every run — including per-phase losses, pseudo-label acceptance counts, and the final model — is logged to MLflow automatically.

### Step 4 — Deploy the API

The CI pipeline pushes a Docker image to Docker Hub on every merge to `main`. On SSP Cloud:

- **Catalogue → Custom Service**
- Image: `<your-dockerhub-username>/tokam2d:latest`
- Port: `8000`
- Mount model weights from S3 or pass `MODEL_PATH` env var

### Step 5 — Continuous deployment with ArgoCD

1. Go to **Catalogue → ArgoCD**, create a new app pointing at your GitHub repo.
2. Create `k8s/deployment.yaml` (minimal example below).
3. Every push to `main` that passes CI will automatically redeploy the API.

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tokam2d-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tokam2d-api
  template:
    metadata:
      labels:
        app: tokam2d-api
    spec:
      containers:
      - name: api
        image: <your-dockerhub-username>/tokam2d:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pth"
---
apiVersion: v1
kind: Service
metadata:
  name: tokam2d-api
spec:
  selector:
    app: tokam2d-api
  ports:
  - port: 80
    targetPort: 8000
```

---

## About the data

The dataset was generated by the **TOKAM2D** 2D fluid turbulence code (CEA/IRFM). **It is not publicly available** — it was produced specifically for a Codabench competition.

### How to get more data

**Option 1 — Contact the organizers.** The competition is over; the CEA/DATAIA team may share the full training set (including `turb_dwi` annotations) on request. The competition was organized through the x-datascience-datacamp GitHub organization.

**Option 2 — Use the dev data you already have.** The file `public_dev_data.tar.gz` in the original competition repo contains the full dev-phase data. Extract it:
```bash
tar -xzf public_dev_data.tar.gz
```

**Option 3 — Generate synthetic blobs.** The open-source `blobmodel` Python package generates synthetic blob fields with configurable statistics:
```bash
pip install blobmodel
```
This won't replicate TOKAM2D exactly but gives unlimited training data for experimentation.

**Option 4 — Related open datasets.** The TCV tokamak dataset from the paper *"Tracking blobs in the turbulent edge plasma of a tokamak fusion device"* (Han et al., 2022, *Scientific Reports*) contains Gas Puff Imaging data with blob annotations. See: https://www.nature.com/articles/s41598-022-21671-w — contact the authors for data access.

---

## GitHub Actions setup (required once)

In your repo **Settings → Secrets → Actions**, add:

| Secret name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (from hub.docker.com → Account Settings → Security) |

The CI pipeline then runs on every push: lint → tests → build Docker image → push to Docker Hub (on `main` only).

---

## Checklist (ENSAE course requirements)

- [x] `.gitignore` adapted to Python + data/secrets exclusions
- [x] `README.md` with context, architecture, usage, deployment guide
- [x] `LICENSE` (MIT)
- [x] `requirements.txt` with package versions
- [x] Code quality: `ruff` linter + formatter, configured in `ruff.toml`
- [x] Modular structure: `main.py` orchestrates `src/` modules
- [x] Cookiecutter-inspired project layout
- [x] Config separated from code (`config/config.yaml`)
- [x] Secrets separated (`.env` / environment variables, never committed)
- [x] Data separated (S3, not in git — see deployment section)
- [x] Unit tests (`tests/test_pipeline.py`, `pytest`)
- [x] `Dockerfile` for portability
- [x] GitHub Actions CI: lint → test → Docker build/push
- [ ] MLflow tracking → requires running MLflow server (see deployment guide)
- [ ] API deployed on SSP Cloud → requires your account (see deployment guide)
- [ ] ArgoCD continuous deployment → requires your account (see deployment guide)

---

## License

[MIT](LICENSE)
