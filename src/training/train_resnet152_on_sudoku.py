import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ResNet152
from src.preprocess.build_features import process_sudoku_image

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet152().to(device)  # Move model to GPU if available

    sudoku_train_dirs = [
        "data/raw/sudoku/v1_training/v1_training",
        "data/raw/sudoku/v2_train/v2_train",
    ]
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(
        sudoku_train_dirs,
        cell_processor=process_sudoku_image,
        test_dir=sudoku_test_dir,
    )

    if train_loader is None or test_loader is None:
        print("Failed to load Sudoku data. Exiting training.")
        exit()

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
