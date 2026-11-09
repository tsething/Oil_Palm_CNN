
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "Train_val_test_split"

duplicates = [
    "Decayed321.jpg",
    "Decayed322.jpg"
]

splits = {
    "Training": "Training.txt",
    "Validation": "Validation.txt",
    "Testing": "Testing.txt"
}

print("=== DUPLICATE SPLIT INVESTIGATION ===")

for split_name, filename in splits.items():
    text = (SPLIT_DIR / filename).read_text(
        encoding="utf-8-sig"
    )

    entries = text.splitlines()

    for duplicate in duplicates:
        if any(duplicate in line for line in entries):
            print(f"{duplicate} -> {split_name}")
