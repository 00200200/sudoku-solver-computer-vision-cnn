import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.model.model import ResNet152
from src.model.predict import evaluate_model
from src.model.train import train_model

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet152().to(device)  # Move model to GPU if available

    # Load Sudoku data and train
    sudoku_train_dir = "data/raw/sudoku/v1_training/v1_training"
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(sudoku_train_dir, sudoku_test_dir)
    model = train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=50,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnest_sudoku_only.pkl")

    # Evaluate on training set
    print("\nEvaluating on training set:")
    evaluate_model(model, test_loader)

    # Test on Sudoku test data
    print("\nEvaluating on Sudoku test set:")
    evaluate_model(model, test_loader)
