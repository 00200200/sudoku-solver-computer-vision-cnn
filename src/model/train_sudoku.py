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
    train_loader, val_loader = get_sudoku_loaders("data/raw/sudoku/mixed 2/mixed 2")
    train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=50,
    )

    torch.save(model.state_dict(), "models/model_sudoku_only.pkl")
    evaluate_model(model, val_loader)
