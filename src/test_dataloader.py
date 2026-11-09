
import torch
from torch.utils.data import DataLoader

from dataset import OilPalmDataset, basic_transform

# ==========================================
# Oil Palm CNN - DataLoader Test
# ==========================================

DATA_DIR = (
    __import__("pathlib").Path(__file__)
    .resolve().parent.parent / "data"
)

train_dataset = OilPalmDataset(
    DATA_DIR, split="train",
    transform=basic_transform
)

val_dataset = OilPalmDataset(
    DATA_DIR, split="val",
    transform=basic_transform
)

test_dataset = OilPalmDataset(
    DATA_DIR, split="test",
    transform=basic_transform
)

print("=" * 45)
print("OIL PALM CNN - DATALOADER TEST")
print("=" * 45)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))

# Batch size 16 is a safe starting point
# for RTX 3050 4GB laptop GPU
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Tensor dtype:", images.dtype)
print("Pixel range:",
      round(images.min().item(), 3),
      "to",
      round(images.max().item(), 3))
print("Labels:", labels.tolist())

assert len(train_dataset) == 3305
assert len(val_dataset) == 704
assert len(test_dataset) == 718
assert images.shape == (16, 3, 256, 256)
assert labels.dtype == torch.int64

print("\nDataLoader test PASSED!")
