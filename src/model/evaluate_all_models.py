import os
from datetime import datetime

import pandas as pd
import torch

from src.data.dataio import get_sudoku_loaders
from src.model.model import ConvNet, ResNet152
from src.model.predict import evaluate_model


def evaluate_and_save_results(model, model_name, test_loader, results_list):
    print(f"\nEvaluating {model_name}...")
    accuracy = evaluate_model(model, test_loader)
    results_list.append(
        {
            "model_name": model_name,
            "accuracy": accuracy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return results_list


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_list = []

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load test data
    _, test_loader = get_sudoku_loaders("data/raw/sudoku/v1_test/v1_test")

    # Test MNIST-only model
    model = ConvNet().to(device)
    model.load_state_dict(torch.load("models/model_mnist_only.pkl"))
    results_list = evaluate_and_save_results(
        model, "MNIST-only", test_loader, results_list
    )

    # Test MNIST-Sudoku model
    model = ConvNet().to(device)
    model.load_state_dict(torch.load("models/model_mnist_sudoku.pkl"))
    results_list = evaluate_and_save_results(
        model, "MNIST-Sudoku", test_loader, results_list
    )

    # Test Sudoku-only model
    model = ConvNet().to(device)
    model.load_state_dict(torch.load("models/model_sudoku_only.pkl"))
    results_list = evaluate_and_save_results(
        model, "Sudoku-only", test_loader, results_list
    )

    # Test ResNet Sudoku model
    model = ResNet152().to(device)
    model.load_state_dict(torch.load("models/resnest_sudoku_only.pkl"))
    results_list = evaluate_and_save_results(
        model, "ResNet-Sudoku", test_loader, results_list
    )

    # Test ResNet Finetuned model
    model = ResNet152().to(device)
    model.load_state_dict(torch.load("models/resnest_sudoku_finetuned.pkl"))
    results_list = evaluate_and_save_results(
        model, "ResNet-Sudoku-Finetuned", test_loader, results_list
    )

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/model_comparison_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    print(f"\nResults saved to: {results_file}")
    print("\nSummary of results:")
    for _, row in results_df.iterrows():
        print(f"{row['model_name']}: {row['accuracy']:.2f}% accuracy")
