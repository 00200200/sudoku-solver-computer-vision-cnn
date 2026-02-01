
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.train import train_model
from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet, ResNet152
from src.preprocess.build_features import process_sudoku_image


def main():
    parser = argparse.ArgumentParser(description="Train digit recognition models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["convnet", "resnet"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "sudoku"],
        required=True,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="Path to pretrained model for fine-tuning",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output model path (default: auto-generated)",
    )
    parser.add_argument(
        "--eval-both",
        action="store_true",
        help="Evaluate on both MNIST and Sudoku after training",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    is_resnet = args.model == "resnet"
    if is_resnet:
        model = ResNet152().to(device)
        print("Model: ResNet152")
    else:
        model = ConvNet().to(device)
        print("Model: ConvNet")

    # Load pretrained weights if fine-tuning
    if args.finetune:
        try:
            model.load_state_dict(torch.load(args.finetune, map_location=device))
            print(f"Loaded pretrained model: {args.finetune}")
        except FileNotFoundError:
            print(f"Error: Pretrained model not found at {args.finetune}")
            return

    # Load dataset
    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(
            "data/raw/MNIST", batch_size=args.batch_size, for_resnet=is_resnet
        )
        dataset_name = "MNIST"
    else:  # sudoku
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
            batch_size=args.batch_size,
            for_resnet=is_resnet,
        )
        dataset_name = "Sudoku"

        if train_loader is None or test_loader is None:
            print("Failed to load Sudoku data. Exiting.")
            return

    print(f"Dataset: {dataset_name}")
    print(f"Training for {args.epochs} epochs with lr={args.lr}")

    # Evaluate before training if fine-tuning
    if args.finetune:
        print(f"\nPerformance before fine-tuning:")
        evaluate_model(model, test_loader)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        num_epochs=args.epochs,
    )

    # Evaluate after training
    print(f"\nPerformance on {dataset_name} after training:")
    evaluate_model(model, test_loader)

    # Cross-evaluation if requested
    if args.eval_both:
        if args.dataset == "mnist":
            print("\nEvaluating on Sudoku test set:")
            _, sudoku_loader = get_sudoku_loaders(
                "data/raw/sudoku/v1_test/v1_test",
                cell_processor=lambda img: process_sudoku_image(
                    img, invert_for_mnist_compatibility=True
                ),
                for_resnet=is_resnet,
            )
            evaluate_model(model, sudoku_loader)
        else:
            print("\nEvaluating on MNIST test set:")
            _, mnist_loader = get_mnist_loaders(
                "data/raw/MNIST", for_resnet=is_resnet
            )
            evaluate_model(model, mnist_loader)

    # Save model
    os.makedirs("models", exist_ok=True)
    if args.output:
        model_path = args.output
    else:
        model_type = "resnet" if is_resnet else "convnet"
        finetune_suffix = "_finetuned" if args.finetune else ""
        model_path = f"models/{args.epochs}epochs_{model_type}_{dataset_name.lower()}{finetune_suffix}.pkl"

    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
