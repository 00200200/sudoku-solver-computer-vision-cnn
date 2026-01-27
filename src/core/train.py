import time

import torch

import src.common.tools as tools
import src.data.dataio as dataio
from src.model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Train the model for specified number of epochs."""
    model.train()
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        batch_count = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            if (i + 1) % 10 == 0:
                print(f"  Processing batch {i+1}/{batch_count}", end="\r")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        print(
            f"\nEpoch {epoch+1}/{num_epochs} completed. Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s"
        )

    print("\nTraining finished.")
    return model
