import os
from datetime import datetime

import pandas as pd
import torch

from src.data.dataio import get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet, ResNet152
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
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
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

    # Load test data - evaluate with ConvNet format (28x28)
    _, test_loader = get_sudoku_loaders(
        "data/raw/sudoku/v1_test/v1_test",
        cell_processor=process_sudoku_image,
        for_resnet=False,
    )

    print("Starting model evaluation on Sudoku test set...")

    # Test ConvNet models
    results_list = load_and_evaluate_model(
        ConvNet,
        "models/model_sudoku_only.pkl",
        "ConvNet-Sudoku",
        test_loader,
        results_list,
    )

    # Test ResNet152 models
    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_mnist_only.pkl",
        "ResNet152-MNIST",
        test_loader,
        results_list,
    )

    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_sudoku_only.pkl",
        "ResNet152-Sudoku",
        test_loader,
        results_list,
    )

    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_sudoku_finetuned.pkl",
        "ResNet152-Finetuned",
        test_loader,
        results_list,
    )

    # Also try alternative naming patterns
    results_list = load_and_evaluate_model(
        ResNet152,
        "models/resnest_new_sudoku_finetuned.pkl",
        "ResNet152-New-Finetuned",
        test_loader,
        results_list,
    )

    # Convert results to DataFrame and save as CSV
    if results_list:
        results_df = pd.DataFrame(results_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/model_comparison_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        print(f"\nResults saved to: {results_file}")
        print("\nSummary of results:")
        print("-" * 50)
        for _, row in results_df.iterrows():
            print(f"{row['model_name']:<25}: {row['accuracy']:>6.2f}% accuracy")

        # Find best model
        best_model = results_df.loc[results_df["accuracy"].idxmax()]
        print("-" * 50)
        print(
            f"Best performing model: {best_model['model_name']} ({best_model['accuracy']:.2f}%)"
        )
    else:
        print("No models were successfully evaluated.")
