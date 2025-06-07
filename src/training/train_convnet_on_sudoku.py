import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet
from src.preprocess.build_features import process_sudoku_image

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)  # Przenosimy model na GPU

    sudoku_train_dirs = [
        "data/raw/sudoku/v1_training/v1_training",
        "data/raw/sudoku/v2_train/v2_train",
    ]
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(
        sudoku_train_dirs,
        cell_processor=process_sudoku_image,
        test_dir=sudoku_test_dir,
        for_resnet=False,
    )
    train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=44,
    )

    torch.save(model.state_dict(), "models/model_sudoku_only.pkl")
    evaluate_model(model, test_loader)
