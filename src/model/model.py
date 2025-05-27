import torch
import torch.nn as nn
from torchvision import models


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )  # padding=1, więc rozmiar pozostaje 28x28
        self.pool = nn.MaxPool2d(2, 2)  # pooling z rozmiarem 2x2
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )  # padding=1, więc rozmiar pozostaje 14x14
        self.fc1 = nn.Linear(
            64 * 7 * 7, 64
        )  # 64 kanały, po poolingach obraz ma wymiary 7x7
        self.fc2 = nn.Linear(64, num_classes)  # 10 klas

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Warstwa konwolucyjna + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Warstwa konwolucyjna + pooling
        x = x.flatten(start_dim=1)  # Wypłaszczanie (64 kanały, 7x7 rozmiar)
        x = torch.relu(self.fc1(x))  # Warstwa w pełni połączona
        x = self.fc2(x)  # Wyjście
        return x


class ResNet152(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet152(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
