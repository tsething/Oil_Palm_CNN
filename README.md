# Oil Palm Fruit Ripeness Classification Using CNN

## 1. Project Overview

This project uses a custom Convolutional Neural Network (CNN) to classify oil palm fruit images into five ripeness levels:

- Immature
- Partially Ripe
- Fully Ripe
- Overripe
- Decayed

The CNN was developed using PyTorch. Three training experiments were conducted to compare the effects of learning rate and dropout.

Docker was used to create separate training and testing environments. The testing container was also successfully tested on a second computer.

## 2. Final Results

The selected model was Experiment 2, with a learning rate of 0.001 and dropout of 0.5.

| Metric | Result |
|---|---:|
| Test Accuracy | 82.03% |
| Macro Precision | 84.46% |
| Macro Recall | 82.41% |
| Macro F1-score | 83.28% |
| Test Images | 718 |
| Correct Predictions | 589 |

The final model is saved as `models/exp2_best.pt`.

## 3. Dataset

**Dataset Name:** An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality Grading

**Source:** Mendeley Data

**Download:** https://doi.org/10.17632/424y96m6sw.1

The dataset contains 4,728 RGB images in five classes. The original training, validation, and testing split files were used.

One duplicate image was excluded from training, leaving:

| Split | Images Used |
|---|---:|
| Training | 3,305 |
| Validation | 704 |
| Testing | 718 |

Download and extract the dataset into the `data/` folder.

Expected structure:

```text
data/
├── Images/
│   ├── 0Immature/
│   ├── 1PartiallyRipe/
│   ├── 2FullyRipe/
│   ├── 3OverRipe/
│   └── 4Decayed/
├── Train_val_test_split/
│   ├── Training.txt
│   ├── Validation.txt
│   └── Testing.txt
└── excluded_images.txt
```

Create `excluded_images.txt` containing `Decayed322.jpg` if it is not already provided.

The dataset images are not included in this GitHub repository.

## 4. Project Structure

```text
Oil_Palm_CNN/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_experiment.py
│   ├── evaluate.py
│   └── docker_predict.py
├── data/
├── models/
│   └── exp2_best.pt
├── outputs/
├── docker_artifacts/
├── demo_images/
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
├── requirements-test.txt
├── DOCKER_GUIDE.md
└── README.md
```

Some directories are created locally when running the project and are excluded from GitHub.

## 5. Requirements

- Docker Desktop with Linux containers
- WSL 2 for Windows
- NVIDIA GPU for the demonstrated GPU training commands
- Downloaded dataset for training and full test evaluation
- Saved CNN checkpoint for prediction

The testing container can run on CPU without an NVIDIA GPU.

The demonstrated Docker images use PyTorch 2.14.0 with CUDA 12.6 support.

**Commands below use Windows PowerShell and must be executed from the project root directory.**

## 6. Docker Training

### 6.1 Build the Training Image

```powershell
docker build -f Dockerfile.train -t oilpalm-train:1.0 .
```

### 6.2 Check the Training Environment

```powershell
docker run --rm --gpus all oilpalm-train:1.0
```

### 6.3 Create Output Directories

```powershell
New-Item -ItemType Directory -Force -Path "docker_artifacts\models", "docker_artifacts\outputs\training"
```

### 6.4 Run CNN Training

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" `
  --mount "type=bind,source=$($PWD.Path)\docker_artifacts\models,target=/app/models" `
  --mount "type=bind,source=$($PWD.Path)\docker_artifacts\outputs,target=/app/outputs" `
  oilpalm-train:1.0 `
  python src/train.py
```

The container trains the baseline CNN for 20 epochs.

Generated files:

```text
docker_artifacts/models/baseline_best.pt
docker_artifacts/outputs/training/baseline_history.csv
```

The Docker baseline training achieved a best validation loss of 0.5188 at epoch 17.

This training run is separate from the original Experiment 2 model used for final evaluation.

## 7. Docker Testing

### 7.1 Build the Testing Image

```powershell
docker build -f Dockerfile.test -t oilpalm-test:1.0 .
```

### 7.2 Check the Testing Environment

```powershell
docker run --rm oilpalm-test:1.0
```

### 7.3 Evaluate the Full Test Set

Create a separate output directory:

```powershell
New-Item -ItemType Directory -Force -Path ".\docker_artifacts\test_verification"
```

Run the evaluation:

```powershell
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" `
  --mount "type=bind,source=$($PWD.Path)\models,target=/app/models,readonly" `
  --mount "type=bind,source=$($PWD.Path)\docker_artifacts\test_verification,target=/app/outputs" `
  oilpalm-test:1.0 `
  python src/evaluate.py
```

The testing container loads `exp2_best.pt` and evaluates the official test split using CPU inference.

**Verified Docker Evaluation Results:**

- Test images: 718
- Accuracy: 82.03%
- Macro Precision: 84.46%
- Macro Recall: 82.41%
- Macro F1-score: 83.28%

The results are saved under:

```text
docker_artifacts/test_verification/evaluation/
```

Use a fresh output directory for another verification run because the evaluation script protects existing final result files from being overwritten.

## 8. Predict a New Image

Place an oil palm fruit image in:

```text
demo_images/newoilpalm.jpg
```

Run:

```powershell
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\models,target=/app/models,readonly" `
  --mount "type=bind,source=$($PWD.Path)\demo_images,target=/app/demo_images,readonly" `
  oilpalm-test:1.0 `
  python src/docker_predict.py `
  --model /app/models/exp2_best.pt `
  --image /app/demo_images/newoilpalm.jpg
```

**Demonstration Result:**

- Predicted Class: Partially Ripe
- Softmax Confidence: 57.65%
- Device: CPU

The external image does not have a verified ground-truth label. Therefore, this result demonstrates successful prediction rather than measured accuracy.

## 9. Cross-Machine Verification

The testing workflow was successfully verified on a second computer.

The testing source files, saved model, and sample image were copied from the laptop to a desktop computer. The testing image was built on the desktop using Docker.

The same `newoilpalm.jpg` image was predicted as **Partially Ripe with 57.65% confidence**, matching the laptop result.

The second computer did not need a local PyTorch or Conda installation.

## 10. Reproducibility

A fixed random seed of 42 was used during training.

The training and testing Docker images provide separate environments. Bind mounts allow the dataset and saved checkpoints to remain outside the images.

The original final evaluation and the later Docker evaluation produced the same test accuracy of 82.03%.

The Docker training run was a separate baseline run, so its validation results are not expected to match the original training run exactly.

For detailed Docker instructions, refer to `DOCKER_GUIDE.md`.

## 11. Generative AI Declaration

ChatGPT was used to support programming explanations, troubleshooting, documentation, and report writing. The CNN experiments and Docker tests were executed using the project code, and the reported results came from actual experiment outputs.
