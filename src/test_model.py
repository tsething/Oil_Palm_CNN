import torch
from model import OilPalmCNN

# ==========================================
# Oil Palm CNN - Architecture Test
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = OilPalmCNN(
    num_classes=5,
    dropout_rate=0.3
).to(device)

# Count trainable parameters
total_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 50)
print("OIL PALM CNN - MODEL ARCHITECTURE")
print("=" * 50)

print(model)
print("\nDevice:", device)
print("Trainable Parameters:", f"{total_params:,}")

# Test with one dummy image first
dummy_input = torch.randn(
    1, 3, 256, 256,
    device=device
)

model.eval()

with torch.no_grad():
    output = model(dummy_input)

print("\nInput Shape:", dummy_input.shape)
print("Output Shape:", output.shape)

assert output.shape == (1, 5)
assert total_params == 155429

print("\nCNN Architecture Test PASSED!")
