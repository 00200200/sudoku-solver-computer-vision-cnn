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
    try:
        model.load_state_dict(torch.load("models/50epochs_convnet_sudoku_only.pkl"))
        print(
            "Loaded pre-trained Sudoku model: models/50epochs_convnet_sudoku_only.pkl"
        )
    except FileNotFoundError:
        print("New model not found, trying old naming...")
        model.load_state_dict(torch.load("models/model_sudoku_only.pkl"))
        print("Loaded pre-trained Sudoku model: models/model_sudoku_only.pkl")

    num_epochs = 10

    mnist_dir = "data/raw/MNIST"
    train_loader, test_loader = get_mnist_loaders(mnist_dir, for_resnet=False)

    print("\nPerformance on MNIST before fine-tuning:")
    evaluate_model(model, test_loader)

    print(f"\nFine-tuning on MNIST data for {num_epochs} epochs...")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(
        model, train_loader, nn.CrossEntropyLoss(), optimizer, num_epochs=num_epochs
    )

    print("\nPerformance on MNIST after fine-tuning:")
    evaluate_model(model, test_loader)

    print("\nPerformance on Sudoku after MNIST fine-tuning:")
    sudoku_dir_train = "data/raw/sudoku/v1_training/v1_training"
    sudoku_dir_test = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(
        sudoku_dir_train,
        cell_processor=lambda img: process_sudoku_image(
            img, invert_for_mnist_compatibility=True
        ),
        test_dir=sudoku_dir_test,
        for_resnet=False,
    )
    evaluate_model(model, test_loader)
    model_name = f"models/{num_epochs}epochs_convnet_sudoku_to_mnist_finetuned.pkl"
    torch.save(model.state_dict(), model_name)
    print(f"Fine-tuned model saved as: {model_name}")
