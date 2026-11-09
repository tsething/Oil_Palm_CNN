# Docker Deployment Guide

## Oil Palm Fruit Ripeness Classification Using CNN

This project uses two separate Docker images for training and testing a custom Convolutional Neural Network (CNN). The network classifies oil palm fruit images into five categories: Immature, Partially Ripe, Fully Ripe, Overripe, and Decayed.

## 1. Requirements

- Docker Desktop with Linux containers
- Windows 11 with WSL 2 for the demonstrated setup
- NVIDIA GPU support for GPU-accelerated training
- Project source code and the dataset for training
- A saved model checkpoint and input image for inference

The testing workflow uses CPU inference and does not require an NVIDIA GPU.

## 2. Project Structure

```text
Offline_Assignment/
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
├── requirements-test.txt
├── .dockerignore
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── docker_predict.py
├── data/
├── models/
│   └── exp2_best.pt
├── demo_images/
│   └── newoilpalm.jpg
├── docker_artifacts/
└── outputs/
    └── docker_evidence/
```

## 3. Build the Training Image

Open PowerShell in the project root:

```powershell
docker build -f Dockerfile.train -t oilpalm-train:1.0 .
```

Check the CNN architecture and GPU access:

```powershell
docker run --rm --gpus all oilpalm-train:1.0
```

## 4. Train the CNN Inside Docker

Create separate directories for Docker-generated results:

```powershell
New-Item -ItemType Directory -Force -Path "docker_artifacts\models", "docker_artifacts\outputs\training"
```

Run baseline training using the mounted dataset:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" `
  --mount "type=bind,source=$($PWD.Path)\docker_artifacts\models,target=/app/models" `
  --mount "type=bind,source=$($PWD.Path)\docker_artifacts\outputs,target=/app/outputs" `
  oilpalm-train:1.0 `
  python src/train.py
```

The container trains the CNN for 20 epochs and saves the best checkpoint according to validation loss.

Expected output files:

- `docker_artifacts/models/baseline_best.pt`
- `docker_artifacts/outputs/training/baseline_history.csv`

The original experiment checkpoints remain separate from the Docker training outputs.

## 5. Build the Testing Image

```powershell
docker build -f Dockerfile.test -t oilpalm-test:1.0 .
```

Check the testing environment:

```powershell
docker run --rm oilpalm-test:1.0
```

This command checks that the required Python libraries are available.

## 6. Predict a New Image Using CPU

Place a new oil palm fruit image in:

```text
demo_images/newoilpalm.jpg
```

Run inference:

```powershell
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\models,target=/app/models,readonly" `
  --mount "type=bind,source=$($PWD.Path)\demo_images,target=/app/demo_images,readonly" `
  oilpalm-test:1.0 `
  python src/docker_predict.py `
  --model /app/models/exp2_best.pt `
  --image /app/demo_images/newoilpalm.jpg
```

The testing container loads the saved model, applies the training preprocessing pipeline, and outputs the predicted ripeness class and softmax confidence score.

### Demonstration Result

For the supplied external sample `newoilpalm.jpg`:

- **Predicted class:** Partially Ripe
- **Softmax confidence:** 57.65%
- **Inference device:** CPU
- **Checkpoint:** `exp2_best.pt`
- **Dropout configuration:** 0.5

The external image has no independently verified ground-truth label, so this result demonstrates successful inference rather than measured prediction accuracy.

## 7. Model Evaluation

The previously selected Experiment 2 model achieved the following results on the official test split:

| Metric | Result |
|---|---:|
| Test accuracy | 82.03% |
| Macro precision | 84.46% |
| Macro recall | 82.41% |
| Macro F1-score | 83.28% |
| Test samples | 718 |

These metrics come from the original final evaluation, not from the single-image Docker demonstration.

## 8. Docker Isolation and Persistence

The training and testing images have separate responsibilities. Training generates the CNN checkpoint, while testing loads existing model weights without retraining.

Bind mounts allow dataset files to be read and generated checkpoints to remain available on the host after the training container exits.

The testing workflow uses CPU inference without relying on the host's Conda environment.

## 9. Cross-Machine Verification

Cross-machine verification must be performed separately on a second computer using the testing Docker image, saved checkpoint, inference source code, and demonstration image.

Record the testing command and output on the second computer as evidence. Do not claim successful cross-machine verification until it has actually been completed.

## 10. Reproducibility Notes

The training script uses a fixed random seed and predefined experiment settings. Exact floating-point results may still differ between hardware or execution environments.

The original final test results and the Docker-generated baseline training results belong to separate runs and should not be presented as identical experiments.
