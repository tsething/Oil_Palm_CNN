
from pathlib import Path
from PIL import Image

# ==========================================
# Oil Palm CNN - Dataset Verification
# ==========================================


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data"
IMAGES_PATH = DATASET_PATH / "Images"


# 2. Define the five ripeness classes
CLASSES = {
    "0Immature": "Immature",
    "1PartiallyRipe": "Partially Ripe",
    "2FullyRipe": "Fully Ripe",
    "3OverRipe": "Overripe",
    "4Decayed": "Decayed"
}

# 3. Verify dataset folder
if not IMAGES_PATH.is_dir():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGES_PATH}"
    )

# 4. Count images in each class
total_images = 0

print("=" * 45)
print("OIL PALM DATASET VERIFICATION")
print("=" * 45)

for folder, class_name in CLASSES.items():

    class_path = IMAGES_PATH / folder

    if not class_path.is_dir():
        raise FileNotFoundError(class_path)

    images = list(class_path.glob("*.jpg"))
    count = len(images)
    total_images += count

    print(f"{class_name:<18}: {count} images")

# 5. Print total number of images
print("-" * 45)
print(f"Total Images: {total_images}")

# 6. Verify sample image
sample_path = next(IMAGES_PATH.rglob("*.jpg"))
with Image.open(sample_path) as image:
    image.verify()

with Image.open(sample_path) as image:
    print(f"Image Size: {image.size}")
    print(f"Image Mode: {image.mode}")

print("=" * 45)
print("Dataset verification completed!")
