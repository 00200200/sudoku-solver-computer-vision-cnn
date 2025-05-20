import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataio import get_sudoku_loaders
from src.model.model import ConvNet
from src.model.predict import evaluate_model
from src.model.train import train_model

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)  # Przenosimy model na GPU

    # Load and train
    sudoku_train_dir = "data/raw/sudoku/v1_training/v1_training"
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(sudoku_train_dir, sudoku_test_dir)
    train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=50,
    )

    torch.save(model.state_dict(), "models/model_sudoku_only.pkl")
    evaluate_model(model, test_loader)
