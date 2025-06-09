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


def get_model_class_from_filename(filename):
    """Determine model class based on filename patterns"""
    if "resnest" in filename or "resnet" in filename:
        return ResNet152
    else:
        return ConvNet


def discover_all_models():
    """Discover all .pkl files in the models directory"""
    model_files = glob.glob("models/*.pkl")
    models_info = []

    for model_path in model_files:
        filename = os.path.basename(model_path)
        model_class = get_model_class_from_filename(filename)
        # Create a clean model name from filename
        model_name = filename.replace(".pkl", "").replace("_", "-")
        models_info.append((model_class, model_path, model_name))

    return models_info


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

    print("Starting comprehensive model evaluation on Sudoku test set...")

    # Discover all models
    all_models = discover_all_models()
    print(f"Found {len(all_models)} models to evaluate:")
    for model_class, model_path, model_name in all_models:
        print(f"  - {model_name} ({model_class.__name__})")

    print("\n" + "=" * 70)

    # Evaluate all discovered models
    for model_class, model_path, model_name in all_models:
        results_list = load_and_evaluate_model(
            model_class, model_path, model_name, test_loader, results_list
        )

    # Convert results to DataFrame and save as CSV
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Sort by accuracy descending
        results_df = results_df.sort_values("accuracy", ascending=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/comprehensive_model_comparison_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        print(f"\nResults saved to: {results_file}")
        print("\nComprehensive Summary of Results (sorted by accuracy):")
        print("=" * 70)
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:2d}. {row['model_name']:<45}: {row['accuracy']:>6.2f}% accuracy")

        # Find best model
        best_model = results_df.iloc[0]  # First row after sorting by accuracy desc
        print("=" * 70)
        print(f"🏆 BEST PERFORMING MODEL: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['accuracy']:.2f}%")
        print(f"   Evaluated at: {best_model['timestamp']}")

        # Show top 5 models
        print(f"\n🥇 TOP 5 MODELS:")
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            print(f"{medal} {row['model_name']}: {row['accuracy']:.2f}%")

    else:
        print("No models were successfully evaluated.")
