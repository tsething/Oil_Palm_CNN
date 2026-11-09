
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilPalmDataset, basic_transform
from model import OilPalmCNN


# ==========================================
# Oil Palm CNN - Baseline Training
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration
SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT = 0.3

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("OIL PALM CNN - BASELINE TRAINING")
print("=" * 50)
print("Device:", device)

# ------------------------------------------
# 1. Prepare datasets and dataloaders
# ------------------------------------------

train_dataset = OilPalmDataset(
    DATA_DIR, split="train",
    transform=basic_transform
)

val_dataset = OilPalmDataset(
    DATA_DIR, split="val",
    transform=basic_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# ------------------------------------------
# 2. Create CNN, loss and optimizer
# ------------------------------------------

model = OilPalmCNN(
    num_classes=5,
    dropout_rate=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ------------------------------------------
# 3. Training and validation function
# ------------------------------------------

def run_epoch(loader, training):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# ------------------------------------------
# 4. Train baseline model
# ------------------------------------------

history = []
best_val_loss = float("inf")

model_path = MODEL_DIR / "baseline_best.pt"
history_path = OUTPUT_DIR / "baseline_history.csv"

print("\nTraining started...\n")

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = run_epoch(
        train_loader, training=True
    )

    with torch.no_grad():
        val_loss, val_acc = run_epoch(
            val_loader, training=False
        )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Save the model with lowest validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "dropout_rate": DROPOUT,
            "num_classes": 5
        }, model_path)

        saved = " *BEST*"
    else:
        saved = ""

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2%}"
        f"{saved}",
        flush=True
    )

    # Save results after every epoch
    with open(
        history_path, "w",
        newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

print("\n" + "=" * 50)
print("BASELINE TRAINING COMPLETED")
print("=" * 50)
print("Best Validation Loss:", round(best_val_loss, 4))
print("Saved Model:", model_path)
print("Training History:", history_path)
