
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import OilPalmCNN


CLASS_NAMES = [
    "Immature",
    "Partially Ripe",
    "Fully Ripe",
    "Overripe",
    "Decayed",
]


def main():
    parser = argparse.ArgumentParser(
        description="Predict oil palm fruit ripeness."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an input image."
    )
    parser.add_argument(
        "--model",
        default="/app/models/exp2_best.pt",
        help="Path to the trained CNN checkpoint."
    )
    args = parser.parse_args()

    # Use CPU so the testing container does not need a GPU.
    device = torch.device("cpu")

    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=True
    )

    model = OilPalmCNN(
        num_classes=checkpoint["num_classes"],
        dropout_rate=checkpoint["dropout_rate"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Use the same preprocessing as during training.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_index].item()

    print("=" * 45)
    print("OIL PALM FRUIT RIPENESS PREDICTION")
    print("=" * 45)
    print(f"Image: {args.image}")
    print(f"Device: {device}")
    print(f"Model dropout: {checkpoint['dropout_rate']}")
    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    main()
