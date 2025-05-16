import torch

import src.common.tools as tools
import src.data.dataio as dataio
from src.model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Funkcja do trenowania modelu
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Włącz tryb treningu
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zerowanie gradientów
            optimizer.zero_grad()

            # Przekazywanie danych przez model
            outputs = model(inputs)

            # Obliczanie straty
            loss = criterion(outputs, labels)

            # Obliczanie gradientów
            loss.backward()

            # Aktualizacja wag
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
