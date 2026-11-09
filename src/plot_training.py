
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Oil Palm CNN - Training Visualisation
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = (
    ROOT / "outputs" / "training"
    / "baseline_history.csv"
)

FIGURE_DIR = ROOT / "outputs" / "training" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load real training results
df = pd.read_csv(HISTORY_PATH)

required_columns = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy"
]

if not set(required_columns).issubset(df.columns):
    raise ValueError("Missing columns in training history")

# Plot settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

TRAIN_COLOR = "#2563A6"
VAL_COLOR = "#E49A37"

# ==========================================
# Figure 4: Training and Validation Loss
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss",
    color=TRAIN_COLOR,
    linewidth=2.5,
    marker="o",
    markersize=4
)

ax.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss",
    color=VAL_COLOR,
    linewidth=2.5,
    marker="s",
    markersize=4
)

best_idx = df["val_loss"].idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])

ax.axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    alpha=0.6,
    label=f"Best Epoch: {best_epoch}"
)

ax.set_title(
    "Baseline CNN - Training and Validation Loss",
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_xticks(df["epoch"])
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "04_baseline_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ==========================================
# Figure 5: Training and Validation Accuracy
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(
    df["epoch"],
    df["train_accuracy"] * 100,
    label="Training Accuracy",
    color=TRAIN_COLOR,
    linewidth=2.5,
    marker="o",
    markersize=4
)

ax.plot(
    df["epoch"],
    df["val_accuracy"] * 100,
    label="Validation Accuracy",
    color=VAL_COLOR,
    linewidth=2.5,
    marker="s",
    markersize=4
)

ax.axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    alpha=0.6,
    label=f"Selected Checkpoint: Epoch {best_epoch}"
)

ax.set_title(
    "Baseline CNN - Training and Validation Accuracy",
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xticks(df["epoch"])
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "05_baseline_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ==========================================
# Training Summary
# ==========================================

print("\n" + "=" * 50)
print("BASELINE CNN TRAINING SUMMARY")
print("=" * 50)

print("Total Epochs:", len(df))
print("Selected Epoch:", best_epoch)

print(
    "Best Validation Loss:",
    round(df["val_loss"].min(), 4)
)

print(
    "Validation Accuracy at Selected Epoch:",
    f"{df.loc[best_idx, 'val_accuracy']:.2%}"
)

print(
    "Final Training Accuracy:",
    f"{df['train_accuracy'].iloc[-1]:.2%}"
)

print(
    "Final Validation Accuracy:",
    f"{df['val_accuracy'].iloc[-1]:.2%}"
)

print(
    "Final Training Loss:",
    round(df["train_loss"].iloc[-1], 4)
)

print(
    "Final Validation Loss:",
    round(df["val_loss"].iloc[-1], 4)
)

print("\nFigures saved to:", FIGURE_DIR)
