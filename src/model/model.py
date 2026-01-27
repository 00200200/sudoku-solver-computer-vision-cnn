import torch
import torch.nn as nn
from torchvision import models


class ConvNet(nn.Module):
    """
    Simple Convolutional Neural Network for digit recognition.
    
    Architecture:
        - Conv2D (3 -> 32) + ReLU + MaxPool
        - Conv2D (32 -> 64) + ReLU + MaxPool
        - Fully Connected (64*7*7 -> 64) + ReLU
        - Fully Connected (64 -> num_classes)
    
    Input: 28x28x3 images
    Output: num_classes logits
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet152(nn.Module):
    """
    ResNet152 with custom classification head for digit recognition.
    
    Uses pretrained ImageNet weights with frozen early layers for transfer learning.
    Only the last 5 parameter groups and the custom FC head are trainable.
    
    Input: 224x224x3 images (ImageNet normalized)
    Output: num_classes logits
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        
        # Freeze early layers, keep last 5 parameter groups trainable
        for param in list(self.resnet.parameters())[:-5]:
            param.requires_grad = False

        # Replace classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
