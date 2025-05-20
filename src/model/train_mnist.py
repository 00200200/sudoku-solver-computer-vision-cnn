import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.model.model import ConvNet
from src.model.predict import evaluate_model
from src.model.train import train_model

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)  # Move model to GPU if available

    # Load MNIST data and train
    train_loader, test_loader = get_mnist_loaders("data/raw/MNIST")
    model = train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=50,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model_mnist_only.pkl")

    # Evaluate on MNIST
    print("\nEvaluating on MNIST test set:")
    evaluate_model(model, test_loader)

    # Test on Sudoku data
    print("\nEvaluating on Sudoku test set:")
    _, sudoku_test_loader = get_sudoku_loaders("data/raw/sudoku/v1_test/v1_test")
    evaluate_model(model, sudoku_test_loader)
