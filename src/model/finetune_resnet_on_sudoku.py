import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataio import get_sudoku_loaders
from src.model.model import ResNet152
from src.model.predict import evaluate_model
from src.model.train import train_model

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet model
    model = ResNet152().to(device)
    model.load_state_dict(torch.load("models/resnest_sudoku_only.pkl"))
    print("Loaded pre-trained ResNet model: models/resnest_sudoku_only.pkl")

    # Load Sudoku data
    sudoku_train_dir = "data/raw/sudoku/v1_training/v1_training"
    sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
    train_loader, test_loader = get_sudoku_loaders(sudoku_train_dir, sudoku_test_dir)

    # Test before fine-tuning
    print("\nPerformance on Sudoku before fine-tuning:")
    evaluate_model(model, test_loader)

    print("\nFine-tuning on Sudoku data...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, nn.CrossEntropyLoss(), optimizer, num_epochs=50)

    # Test after fine-tuning
    print("\nPerformance on Sudoku after fine-tuning:")
    evaluate_model(model, test_loader)

    # Save fine-tuned model
    torch.save(model.state_dict(), "models/resnest_sudoku_finetuned.pkl")
    print("Fine-tuned model saved as: models/resnest_sudoku_finetuned.pkl")
