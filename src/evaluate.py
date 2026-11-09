
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from dataset import OilPalmDataset, basic_transform, CLASS_NAMES
from model import OilPalmCNN


# ==========================================
# Oil Palm CNN - Final Test Evaluation
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "exp2_best.pt"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Protect against accidental repeated testing
RESULT_PATH = OUTPUT_DIR / "final_metrics.json"

if RESULT_PATH.exists():
    raise FileExistsError(
        "Final evaluation already exists. "
        "Do not rerun the test set for model selection."
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("OIL PALM CNN - FINAL TEST EVALUATION")
print("=" * 55)
print("Device:", device)

# ------------------------------------------
# 1. Load selected trained model
# ------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = OilPalmCNN(
    num_classes=5,
    dropout_rate=checkpoint["dropout_rate"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model: Experiment 2")
print("Selected Epoch:", checkpoint["epoch"])

# ------------------------------------------
# 2. Prepare unseen testing dataset
# ------------------------------------------

test_dataset = OilPalmDataset(
    DATA_DIR,
    split="test",
    transform=basic_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

assert len(test_dataset) == 718

print("Test Images:", len(test_dataset))

# ------------------------------------------
# 3. Predict on testing images
# ------------------------------------------

all_true = []
all_pred = []

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(
            predictions.cpu().tolist()
        )

y_true = np.array(all_true)
y_pred = np.array(all_pred)

# ------------------------------------------
# 4. Calculate evaluation metrics
# ------------------------------------------

test_accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(5)),
        zero_division=0
    )
)

macro_precision = float(np.mean(precision))
macro_recall = float(np.mean(recall))
macro_f1 = float(np.mean(f1))

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

print("\n=== FINAL TEST METRICS ===")
print(f"Accuracy: {test_accuracy:.2%}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# ------------------------------------------
# 5. Per-class classification report
# ------------------------------------------

report_df = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Support": support
})

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv",
    index=False
)

print("\n=== PER-CLASS PERFORMANCE ===")
print(report_df.round(4).to_string(index=False))

print("\n=== CLASSIFICATION REPORT ===")

print(classification_report(
    y_true, y_pred,
    labels=list(range(5)),
    target_names=CLASS_NAMES,
    zero_division=0
))

# ------------------------------------------
# 6. Confusion matrix
# ------------------------------------------

cm = confusion_matrix(
    y_true, y_pred,
    labels=list(range(5))
)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    cbar_kws={"label": "Number of Images"},
    ax=ax
)

ax.set_title(
    "Oil Palm CNN - Test Confusion Matrix",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("Actual Class", fontsize=12)

plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "08_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ------------------------------------------
# 7. Wrong prediction analysis
# ------------------------------------------

wrong_indices = np.where(y_true != y_pred)[0]

print("\nWrong Predictions:", len(wrong_indices))

wrong_records = []

for idx in wrong_indices:

    image_path, _ = test_dataset.samples[idx]

    wrong_records.append({
        "Filename": image_path.name,
        "Actual": CLASS_NAMES[y_true[idx]],
        "Predicted": CLASS_NAMES[y_pred[idx]]
    })

pd.DataFrame(
    wrong_records,
    columns=["Filename", "Actual", "Predicted"]
).to_csv(
    OUTPUT_DIR / "wrong_predictions.csv",
    index=False
)

# Display up to 5 incorrect predictions
num_examples = min(5, len(wrong_indices))

if num_examples > 0:

    fig, axes = plt.subplots(
        1, num_examples,
        figsize=(3.5 * num_examples, 4)
    )

    axes = np.atleast_1d(axes)

    for ax, idx in zip(
        axes, wrong_indices[:num_examples]
    ):

        image_path, _ = test_dataset.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()

        ax.imshow(image)

        ax.set_title(
            f"Actual: {CLASS_NAMES[y_true[idx]]}\n"
            f"Predicted: {CLASS_NAMES[y_pred[idx]]}",
            fontsize=10
        )

        ax.axis("off")

    fig.suptitle(
        "Oil Palm CNN - Incorrect Predictions",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "09_wrong_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

# ------------------------------------------
# 8. Most-common-class baseline
# ------------------------------------------

train_dataset = OilPalmDataset(
    DATA_DIR,
    split="train",
    transform=None
)

train_labels = np.array([
    label for _, label in train_dataset.samples
])

majority_class = int(
    np.bincount(train_labels, minlength=5).argmax()
)

baseline_pred = np.full_like(
    y_true, majority_class
)

baseline_accuracy = accuracy_score(
    y_true, baseline_pred
)

print("\n=== SIMPLE BASELINE COMPARISON ===")
print("Majority Class:", CLASS_NAMES[majority_class])
print(f"Majority Baseline Accuracy: {baseline_accuracy:.2%}")
print(f"CNN Test Accuracy: {test_accuracy:.2%}")

# ------------------------------------------
# 9. Save final metrics
# ------------------------------------------

results = {
    "selected_model": "exp2_best.pt",
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "test_samples": len(test_dataset),
    "test_accuracy": float(test_accuracy),
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "wrong_predictions": len(wrong_indices),
    "majority_class": CLASS_NAMES[majority_class],
    "majority_baseline_accuracy": float(baseline_accuracy)
}

RESULT_PATH.write_text(
    json.dumps(results, indent=4),
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("FINAL EVALUATION COMPLETED")
print("=" * 55)
print("Results saved to:", OUTPUT_DIR)
