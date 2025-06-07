import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet
from src.preprocess.build_features import process_sudoku_image

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained MNIST model
    model = ConvNet().to(device)
    model.load_state_dict(torch.load("models/model_sudoku_only.pkl"))
    print("Loaded pre-trained MNIST model: models/model_sudoku_only.pkl")

    # Load Sudoku data
    mnist_dir = "data/raw/MNIST"
    train_loader, test_loader = get_mnist_loaders(mnist_dir, for_resnet=False)

    # Test before fine-tuning
    print("\nPerformance on Sudoku before fine-tuning:")
    evaluate_model(model, test_loader)

    print("\nFine-tuning on Sudoku data...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, nn.CrossEntropyLoss(), optimizer, num_epochs=10)

    # Test after fine-tuning
    print("\nPerformance on Sudoku after fine-tuning:")
    sudoku_dir_train = "data/raw/sudoku/v1_training/v1_training"
    sudoku_dir_test = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(
        sudoku_dir_train,
        cell_processor=process_sudoku_image,
        test_dir=sudoku_dir_test,
        for_resnet=False,
    )
    evaluate_model(model, test_loader)

    # Save fine-tuned model
    torch.save(model.state_dict(), "models/model_sudoku_mnist.pkl")
    print("Fine-tuned model saved as: models/model_sudoku_mnist.pkl")
