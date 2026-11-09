
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# Oil Palm CNN - Exploratory Data Analysis
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class folder names and readable labels
CLASSES = {
    "0Immature": "Immature",
    "1PartiallyRipe": "Partially Ripe",
    "2FullyRipe": "Fully Ripe",
    "3OverRipe": "Overripe",
    "4Decayed": "Decayed"
}

# Count images
class_counts = {}

for folder, label in CLASSES.items():
    images = list((IMAGE_DIR / folder).glob("*.jpg"))
    class_counts[label] = len(images)

df = pd.DataFrame({
    "Ripeness Level": list(class_counts.keys()),
    "Number of Images": list(class_counts.values())
})

# Set professional chart style
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(11, 6))

colors = sns.color_palette("YlGn", n_colors=5)

bars = ax.bar(
    df["Ripeness Level"],
    df["Number of Images"],
    color=colors,
    width=0.65
)

# Add numerical labels above bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{int(height):,}",
        (bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

ax.set_title(
    "Oil Palm Fruit Ripeness Dataset — Class Distribution",
    fontsize=17,
    fontweight="bold",
    pad=20
)
ax.set_xlabel("Ripeness Level", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_ylim(0, max(class_counts.values()) * 1.2)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
sns.despine()

fig.tight_layout()

# Save high-resolution figure
save_path = OUTPUT_DIR / "01_class_distribution.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nClass distribution:")
print(df.to_string(index=False))
print(f"\nFigure saved to: {save_path}")


# ==========================================
# Figure 2: Sample Image Gallery
# ==========================================

import random
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

# Display 3 images from each of the 5 classes
NUM_SAMPLES = 3

fig, axes = plt.subplots(
    nrows=NUM_SAMPLES,
    ncols=len(CLASSES),
    figsize=(15, 10)
)

# Loop through the five classes
for col, (folder, label) in enumerate(CLASSES.items()):

    class_path = IMAGE_DIR / folder
    image_files = sorted(class_path.glob("*.jpg"))

    if len(image_files) < NUM_SAMPLES:
        raise ValueError(
            f"Not enough images in {folder}"
        )

    # Randomly select 3 images without replacement
    selected_images = random.sample(
        image_files,
        NUM_SAMPLES
    )

    for row, image_path in enumerate(selected_images):

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()

        axes[row, col].imshow(image)
        axes[row, col].axis("off")

        # Display class name above each column
        if row == 0:
            axes[row, col].set_title(
                label,
                fontsize=13,
                fontweight="bold",
                pad=12
            )

# Main figure title
fig.suptitle(
    "Visual Samples of Oil Palm Fruit Ripeness Levels",
    fontsize=19,
    fontweight="bold",
    y=0.99
)

# Improve spacing
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save the gallery
save_path = OUTPUT_DIR / "02_sample_image_gallery.png"

fig.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("\nSample Image Gallery generated successfully!")
print(f"Figure saved to: {save_path}")


# ==========================================
# Figure 3: Image Brightness Analysis
# ==========================================

import numpy as np

brightness_records = []

# Calculate average brightness of each image
for folder, label in CLASSES.items():

    class_path = IMAGE_DIR / folder
    image_files = sorted(class_path.glob("*.jpg"))

    for image_path in image_files:

        with Image.open(image_path) as img:
            # Convert image to grayscale
            gray_image = img.convert("L")

            # Convert image into NumPy array
            pixel_array = np.asarray(
                gray_image, dtype=np.float32
            )

            # Calculate mean brightness
            mean_brightness = pixel_array.mean()

        brightness_records.append({
            "Ripeness Level": label,
            "Mean Brightness": mean_brightness
        })

# Create DataFrame
brightness_df = pd.DataFrame(brightness_records)

# Plot brightness distribution
sns.set_theme(style="whitegrid", context="talk")

fig, ax = plt.subplots(figsize=(12, 6))

colors = sns.color_palette("YlGn", n_colors=5)

sns.boxplot(
    data=brightness_df,
    x="Ripeness Level",
    y="Mean Brightness",
    hue="Ripeness Level",
    order=list(CLASSES.values()),
    hue_order=list(CLASSES.values()),
    palette=colors,
    legend=False,
    ax=ax
)

ax.set_title(
    "Image Brightness Distribution by Ripeness Class",
    fontsize=17,
    fontweight="bold",
    pad=20
)

ax.set_xlabel("Ripeness Level", fontsize=12)
ax.set_ylabel("Mean Grayscale Brightness (0-255)", fontsize=12)
ax.set_ylim(0, 255)

ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

sns.despine()
fig.tight_layout()

# Save figure
save_path = OUTPUT_DIR / "03_brightness_distribution.png"

fig.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

# Print descriptive statistics
print("\n=== IMAGE BRIGHTNESS STATISTICS ===")

summary = brightness_df.groupby(
    "Ripeness Level", sort=False
)["Mean Brightness"].agg(
    ["count", "mean", "std", "min", "median", "max"]
)

print(summary.round(2))

print("\nBrightness Analysis completed!")
print(f"Figure saved to: {save_path}")
