
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Oil Palm CNN - Experiment Comparison
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "outputs" / "training"
OUTPUT_DIR = TRAIN_DIR / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

experiments = {
    "Baseline": "baseline_history.csv",
    "Exp 1 - Lower LR": "exp1_history.csv",
    "Exp 2 - Higher Dropout": "exp2_history.csv"
}

colors = {
    "Baseline": "#2563A6",
    "Exp 1 - Lower LR": "#D68B35",
    "Exp 2 - Higher Dropout": "#27866F"
}

histories = {}
summary_rows = []

for name, filename in experiments.items():
    df = pd.read_csv(TRAIN_DIR / filename)
    histories[name] = df

    best = df.loc[df["val_loss"].idxmin()]

    summary_rows.append({
        "Experiment": name,
        "Selected Epoch": int(best["epoch"]),
        "Validation Loss": best["val_loss"],
        "Validation Accuracy (%)": best["val_accuracy"] * 100
    })

# ------------------------------------------
# Figure 6: Validation Loss Comparison
# ------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

for name, df in histories.items():
    ax.plot(
        df["epoch"],
        df["val_loss"],
        label=name,
        color=colors[name],
        linewidth=2.5,
        marker="o",
        markersize=4
    )

ax.set_title(
    "Validation Loss Comparison Across Experiments",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_xticks(range(1, 21))
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "06_validation_loss_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
plt.close(fig)

# ------------------------------------------
# Figure 7: Validation Accuracy Comparison
# ------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

for name, df in histories.items():
    ax.plot(
        df["epoch"],
        df["val_accuracy"] * 100,
        label=name,
        color=colors[name],
        linewidth=2.5,
        marker="o",
        markersize=4
    )

ax.set_title(
    "Validation Accuracy Comparison Across Experiments",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xticks(range(1, 21))
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "07_validation_accuracy_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
plt.close(fig)

# ------------------------------------------
# Comparison Summary
# ------------------------------------------

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("Validation Loss")

summary.to_csv(
    OUTPUT_DIR / "experiment_comparison.csv",
    index=False
)

print("\n=== EXPERIMENT COMPARISON ===")
print(summary.round(4).to_string(index=False))

print("\nSelected Model (minimum validation loss):")
print(summary.iloc[0]["Experiment"])

print("\nComparison figures saved to:", OUTPUT_DIR)
