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
    try:
        model.load_state_dict(torch.load("models/15epochs_convnet_mnist_only.pkl"))
        print("Loaded pre-trained MNIST model: models/15epochs_convnet_mnist_only.pkl")
    except FileNotFoundError:
        print("New model not found, trying old naming...")
        model.load_state_dict(torch.load("models/model_mnist_only.pkl"))
        print("Loaded pre-trained MNIST model: models/model_mnist_only.pkl")

    num_epochs = 40

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

    print("\nPerformance on Sudoku before fine-tuning:")
    evaluate_model(model, test_loader)

    print(f"\nFine-tuning on Sudoku data for {num_epochs} epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(
        model, train_loader, nn.CrossEntropyLoss(), optimizer, num_epochs=num_epochs
    )

    print("\nPerformance on Sudoku after fine-tuning:")
    evaluate_model(model, test_loader)

    model_name = f"models/{num_epochs}epochs_convnet_mnist_to_sudoku_finetuned.pkl"
    torch.save(model.state_dict(), model_name)
    print(f"Fine-tuned model saved as: {model_name}")
