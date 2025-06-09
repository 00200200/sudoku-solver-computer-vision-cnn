import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet
from src.preprocess.build_features import process_sudoku_image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)

    num_epochs = 8

    train_loader, test_loader = get_mnist_loaders("data/raw/MNIST", for_resnet=False)
    model = train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=num_epochs,
    )

    os.makedirs("models", exist_ok=True)
    model_name = f"models/{num_epochs}epochs_convnet_mnist_only.pkl"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as: {model_name}")

    print("\nEvaluating on MNIST test set:")
    evaluate_model(model, test_loader)

    print("\nEvaluating on Sudoku test set:")
    _, sudoku_test_loader = get_sudoku_loaders(
        "data/raw/sudoku/v1_test/v1_test",
        cell_processor=lambda img: process_sudoku_image(
            img, invert_for_mnist_compatibility=True
        ),
        for_resnet=False,
    )
    evaluate_model(model, sudoku_test_loader)
