
from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilPalmDataset, basic_transform
from model import OilPalmCNN


# ==========================================
# Oil Palm CNN - Controlled Experiments
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experimental settings
EXPERIMENTS = {
    "exp1": {
        "learning_rate": 0.0005,
        "dropout_rate": 0.3
    },
    "exp2": {
        "learning_rate": 0.001,
        "dropout_rate": 0.5
    }
}

BATCH_SIZE = 16
EPOCHS = 20
SEED = 42


# ------------------------------------------
# 1. Reproducibility
# ------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------
# 2. Train or validate one epoch
# ------------------------------------------

def run_epoch(model, loader, criterion,
              optimizer, device, training):

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

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ------------------------------------------
# 3. Main experiment
# ------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        required=True
    )

    args = parser.parse_args()
    name = args.experiment
    config = EXPERIMENTS[name]

    model_path = MODEL_DIR / f"{name}_best.pt"
    history_path = OUTPUT_DIR / f"{name}_history.csv"
    summary_path = OUTPUT_DIR / f"{name}_summary.json"

    # Prevent accidentally overwriting results
    for path in [model_path, history_path, summary_path]:
        if path.exists():
            raise FileExistsError(
                f"Result already exists: {path}"
            )

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print(f"OIL PALM CNN - {name.upper()}")
    print("=" * 55)

    print("Device:", device)
    print("Learning Rate:", config["learning_rate"])
    print("Dropout:", config["dropout_rate"])
    print("Epochs:", EPOCHS)
    print("Batch Size:", BATCH_SIZE)

    train_dataset = OilPalmDataset(
        DATA_DIR, "train", basic_transform
    )

    val_dataset = OilPalmDataset(
        DATA_DIR, "val", basic_transform
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

    assert len(train_dataset) == 3305
    assert len(val_dataset) == 704

    model = OilPalmCNN(
        num_classes=5,
        dropout_rate=config["dropout_rate"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0

    print("\nTraining started!\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion,
            optimizer, device, True
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            optimizer, device, False
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Save checkpoint using validation loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            best_val_accuracy = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "dropout_rate": config["dropout_rate"],
                "num_classes": 5,
                "learning_rate": config["learning_rate"]
            }, model_path)

            marker = " *BEST*"
        else:
            marker = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
            f"{marker}",
            flush=True
        )

        # Save training history every epoch
        with history_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f, fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

    # Save experiment summary
    summary = {
        "experiment": name,
        "learning_rate": config["learning_rate"],
        "dropout_rate": config["dropout_rate"],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset)
    }

    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print(f"{name.upper()} TRAINING COMPLETED")
    print("=" * 55)
    print("Selected Epoch:", best_epoch)
    print("Best Validation Loss:", round(best_val_loss, 4))
    print("Validation Accuracy:", f"{best_val_accuracy:.2%}")
    print("Model:", model_path)
    print("History:", history_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
