# from src.model import model
import torch

import src.common.tools as tools
import src.data.dataio as dataio
from src.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader):
    model.eval()  # ustawienie modelu w tryb ewaluacyjny
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Przenosimy dane na GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
