
import torch
import torch.nn as nn


# ==========================================
# Oil Palm CNN - Custom Neural Network
# ==========================================

class OilPalmCNN(nn.Module):

    def __init__(self, num_classes=5, dropout_rate=0.3):
        super().__init__()

        # Block 1: Extract low-level features
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Block 2: Extract intermediate features
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Block 3: Extract higher-level features
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Reduce spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = self.classifier(x)

        return x
