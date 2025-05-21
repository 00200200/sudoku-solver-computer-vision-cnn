import os
from datetime import datetime

import pandas as pd
import torch

from src.data.dataio import get_sudoku_loaders
from src.model.model import ConvNet, ResNet152
from src.model.predict import evaluate_model
from src.preprocess.build_features import process_sudoku_image


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


def load_and_evaluate_model(
    model_class, model_path, model_name, test_loader, results_list
):
    try:
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        return evaluate_and_save_results(model, model_name, test_loader, results_list)
    except Exception as e:
        print(f"Skipping {model_name} - {e}")
        return results_list


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_list = []

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load test data
    _, test_loader = get_sudoku_loaders(
        "data/raw/sudoku/v1_test/v1_test", cell_processor=process_sudoku_image
    )

    # Test MNIST-only model
    results_list = load_and_evaluate_model(
        ConvNet, "models/model_mnist_only.pkl", "MNIST-only", test_loader, results_list
    )

    # Test MNIST-Sudoku model
    results_list = load_and_evaluate_model(
        ConvNet,
        "models/model_mnist_sudoku.pkl",
        "MNIST-Sudoku",
        test_loader,
        results_list,
    )

    # Test Sudoku-only model
    results_list = load_and_evaluate_model(
        ConvNet,
        "models/model_sudoku_only.pkl",
        "Sudoku-only",
        test_loader,
        results_list,
    )

    # Test ResNet Sudoku model if available
    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_sudoku_only.pkl",
        "ResNet-Sudoku",
        test_loader,
        results_list,
    )

    # Test ResNet Finetuned model if available
    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_sudoku_finetuned.pkl",
        "ResNet-Sudoku-Finetuned",
        test_loader,
        results_list,
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
