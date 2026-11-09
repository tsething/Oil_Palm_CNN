
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ==========================================
# Oil Palm CNN - Custom Dataset
# ==========================================

CLASS_FOLDERS = [
    "0Immature",
    "1PartiallyRipe",
    "2FullyRipe",
    "3OverRipe",
    "4Decayed"
]

CLASS_NAMES = [
    "Immature",
    "Partially Ripe",
    "Fully Ripe",
    "Overripe",
    "Decayed"
]


class OilPalmDataset(Dataset):

    def __init__(self, data_dir, split, transform=None):

        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Images"
        self.split_dir = (
            self.data_dir / "Train_val_test_split"
        )

        self.transform = transform

        split_files = {
            "train": "Training.txt",
            "val": "Validation.txt",
            "test": "Testing.txt"
        }

        if split not in split_files:
            raise ValueError(f"Invalid split: {split}")

        split_path = self.split_dir / split_files[split]

        # Read official split file
        filenames = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        # Exclude known duplicate from training only
        if split == "train":
            exclude_path = (
                self.data_dir / "excluded_images.txt"
            )

            if not exclude_path.is_file():
                raise FileNotFoundError(
                    f"Missing exclusion list: {exclude_path}"
                )

            exclusions = {
                line.strip()
                for line in exclude_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }

            if not exclusions.issubset(set(filenames)):
                raise ValueError(
                    "An excluded filename is not in Training.txt"
                )

            filenames = [
                name for name in filenames
                if name not in exclusions
            ]

        # Build filename-to-path mapping
        image_lookup = {}

        for class_idx, folder in enumerate(CLASS_FOLDERS):
            class_path = self.image_dir / folder

            for image_path in class_path.glob("*.jpg"):
                if image_path.name in image_lookup:
                    raise ValueError(
                        f"Duplicate filename: {image_path.name}"
                    )

                image_lookup[image_path.name] = (
                    image_path, class_idx
                )

        # Build samples list
        self.samples = []

        for filename in filenames:
            if filename not in image_lookup:
                raise FileNotFoundError(
                    f"Image not found: {filename}"
                )

            self.samples.append(image_lookup[filename])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Basic preprocessing
basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
