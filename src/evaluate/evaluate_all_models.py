import glob
import os
from datetime import datetime

import pandas as pd
import torch

from src.data.dataio import get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet, ResNet152
from src.preprocess.build_features import process_sudoku_image


def evaluate_and_save_results(model, model_name, test_loader, results_list):
    """Evaluate model and append results to list."""
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
    """Load model from disk and evaluate it."""
    try:
        model = model_class().to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        return evaluate_and_save_results(model, model_name, test_loader, results_list)
    except Exception as e:
        print(f"Skipping {model_name} - {e}")
        return results_list


def get_model_class_from_filename(filename):
    """Determine model class from filename pattern."""
    if "resnest" in filename or "resnet" in filename:
        return ResNet152
    else:
        return ConvNet


def discover_all_models():
    """Discover all .pkl model files in models/ directory."""
    model_files = glob.glob("models/*.pkl")
    models_info = []

    for model_path in model_files:
        filename = os.path.basename(model_path)
        model_class = get_model_class_from_filename(filename)
        model_name = filename.replace(".pkl", "").replace("_", "-")
        models_info.append((model_class, model_path, model_name))

    return models_info


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_list = []

    os.makedirs("results", exist_ok=True)

    _, test_loader = get_sudoku_loaders(
        "data/raw/sudoku/v1_test/v1_test",
        cell_processor=process_sudoku_image,
        for_resnet=False,
    )

    print("Starting comprehensive model evaluation on Sudoku test set...")

    all_models = discover_all_models()
    print(f"Found {len(all_models)} models to evaluate:")
    for model_class, model_path, model_name in all_models:
        print(f"  - {model_name} ({model_class.__name__})")

    print("\n" + "=" * 70)

    for model_class, model_path, model_name in all_models:
        results_list = load_and_evaluate_model(
            model_class, model_path, model_name, test_loader, results_list
        )

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values("accuracy", ascending=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/model_comparison_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        print(f"\nResults saved to: {results_file}")
        print("\nModel Rankings (sorted by accuracy):")
        print("=" * 70)
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:2d}. {row['model_name']:<45}: {row['accuracy']:>6.2f}%")

        best_model = results_df.iloc[0]
        print("=" * 70)
        print(f"Best Model: {best_model['model_name']}")
        print(f"Accuracy: {best_model['accuracy']:.2f}%")
        print(f"Evaluated: {best_model['timestamp']}")

        print(f"\nTop 5 Models:")
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"{i}. {row['model_name']}: {row['accuracy']:.2f}%")
    else:
        print("No models were successfully evaluated.")
