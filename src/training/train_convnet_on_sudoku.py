import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet
from src.preprocess.build_features import process_sudoku_image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)

    num_epochs = 50

    sudoku_train_dirs = [
        "data/raw/sudoku/v1_training/v1_training",
        "data/raw/sudoku/v2_train/v2_train",
    ]
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(
        sudoku_train_dirs,
        cell_processor=lambda img: process_sudoku_image(
            img, invert_for_mnist_compatibility=True
        ),
        test_dir=sudoku_test_dir,
        for_resnet=False,
    )
    train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optim.Adam(model.parameters(), lr=0.001),
        num_epochs=num_epochs,
    )

    model_name = f"models/{num_epochs}epochs_convnet_sudoku_only.pkl"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as: {model_name}")

    evaluate_model(model, test_loader)
