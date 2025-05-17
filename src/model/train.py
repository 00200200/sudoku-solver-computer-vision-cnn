import torch

import src.common.tools as tools
import src.data.dataio as dataio
from src.model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Funkcja do trenowania modelu
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Włącz tryb treningu
    for epoch in range(num_epochs):
        model.train()  # ustawienie modelu w tryb treningowy
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            # Przenosimy dane na GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass i optymalizacja
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Obliczanie dokładności
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )
