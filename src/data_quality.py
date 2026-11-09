
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, UnidentifiedImageError
import hashlib
import json
import re

# ==========================================
# Oil Palm CNN - Data Quality Audit
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "Images"
SPLIT_DIR = DATA_DIR / "Train_val_test_split"
OUTPUT_DIR = ROOT / "outputs" / "data_quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "0Immature": "Immature",
    "1PartiallyRipe": "Partially Ripe",
    "2FullyRipe": "Fully Ripe",
    "3OverRipe": "Overripe",
    "4Decayed": "Decayed"
}

SPLIT_FILES = {
    "Training": "Training.txt",
    "Validation": "Validation.txt",
    "Testing": "Testing.txt"
}

# ------------------------------------------
# 1. Collect all image files
# ------------------------------------------

image_paths = []
image_lookup = {}

for folder in CLASSES:
    class_dir = IMAGE_DIR / folder

    if not class_dir.is_dir():
        raise FileNotFoundError(class_dir)

    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_paths.append(path)
        image_lookup.setdefault(path.name, []).append(path)

print("=" * 50)
print("OIL PALM DATA QUALITY AUDIT")
print("=" * 50)
print("Total image files:", len(image_paths))

# ------------------------------------------
# 2. Check corrupted images and properties
# ------------------------------------------

corrupted = []
sizes = Counter()
modes = Counter()

for path in image_paths:
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.load()
            sizes[str(img.size)] += 1
            modes[img.mode] += 1

    except (OSError, ValueError, UnidentifiedImageError) as e:
        corrupted.append({
            "file": str(path.relative_to(ROOT)),
            "error": str(e)
        })

print("\n[1] IMAGE INTEGRITY")
print("Corrupted images:", len(corrupted))
print("Image sizes:", dict(sizes))
print("Image modes:", dict(modes))

# ------------------------------------------
# 3. Detect exact duplicates using SHA-256
# ------------------------------------------

def file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


hash_groups = defaultdict(list)
hash_by_name = {}

for path in image_paths:
    digest = file_hash(path)
    hash_groups[digest].append(str(path.relative_to(ROOT)))
    hash_by_name[path.name] = digest

duplicate_groups = [
    paths for paths in hash_groups.values()
    if len(paths) > 1
]

print("\n[2] EXACT DUPLICATE DETECTION")
print("Duplicate groups:", len(duplicate_groups))
print(
    "Extra duplicate files:",
    sum(len(group) - 1 for group in duplicate_groups)
)

# ------------------------------------------
# 4. Read official dataset split files
# ------------------------------------------

def read_split(filename):
    path = SPLIT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    names = []

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        # Extract JPG filename from each line
        match = re.search(
            r"([^/\\,\s]+\.jpe?g)",
            line,
            flags=re.IGNORECASE
        )

        if not match:
            raise ValueError(
                f"Unrecognised split entry: {line}"
            )

        names.append(match.group(1))

    return names


splits = {
    name: read_split(filename)
    for name, filename in SPLIT_FILES.items()
}

print("\n[3] DATASET SPLITS")

for name, filenames in splits.items():
    print(f"{name}: {len(filenames)} images")

# ------------------------------------------
# 5. Check filenames and split integrity
# ------------------------------------------

all_split_names = set().union(
    *(set(names) for names in splits.values())
)

missing_files = sorted(
    all_split_names - set(image_lookup)
)

unlisted_files = sorted(
    set(image_lookup) - all_split_names
)

ambiguous_names = sorted(
    name for name, paths in image_lookup.items()
    if len(paths) > 1
)

repeated_entries = {
    name: len(names) - len(set(names))
    for name, names in splits.items()
}

split_overlaps = {}

split_names = list(splits)

for i, a in enumerate(split_names):
    for b in split_names[i + 1:]:
        overlap = set(splits[a]) & set(splits[b])
        split_overlaps[f"{a} vs {b}"] = sorted(overlap)

# ------------------------------------------
# 6. Check cross-split content duplicates
# ------------------------------------------

hashes_by_split = {}

for name, filenames in splits.items():
    hashes_by_split[name] = {
        hash_by_name[f]
        for f in filenames
        if f in hash_by_name
    }

cross_split_duplicates = {}

for i, a in enumerate(split_names):
    for b in split_names[i + 1:]:
        common = (
            hashes_by_split[a]
            & hashes_by_split[b]
        )
        cross_split_duplicates[f"{a} vs {b}"] = len(common)

print("\n[4] SPLIT INTEGRITY")
print("Missing files:", len(missing_files))
print("Unlisted files:", len(unlisted_files))
print("Ambiguous filenames:", len(ambiguous_names))
print("Repeated split entries:", repeated_entries)

for pair, overlap in split_overlaps.items():
    print(f"Filename overlap ({pair}): {len(overlap)}")

for pair, count in cross_split_duplicates.items():
    print(f"Content overlap ({pair}): {count}")

# ------------------------------------------
# 7. Save audit report
# ------------------------------------------

report = {
    "total_images": len(image_paths),
    "class_counts": {
        label: sum(
            p.parent.name == folder
            for p in image_paths
        )
        for folder, label in CLASSES.items()
    },
    "image_sizes": dict(sizes),
    "image_modes": dict(modes),
    "corrupted_images": corrupted,
    "duplicate_groups": duplicate_groups,
    "split_counts": {
        name: len(files)
        for name, files in splits.items()
    },
    "missing_files": missing_files,
    "unlisted_files": unlisted_files,
    "ambiguous_filenames": ambiguous_names,
    "repeated_split_entries": repeated_entries,
    "filename_overlaps": split_overlaps,
    "cross_split_content_duplicates": cross_split_duplicates
}

report_path = OUTPUT_DIR / "data_quality_report.json"

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nAudit report saved to:", report_path)
print("Data Quality Audit completed!")
