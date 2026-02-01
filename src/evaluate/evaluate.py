import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from src.common import tools


def evaluate_model(model, test_loader):
    """Evaluate model on test data and return accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc


class ModelResults:
    """Container for model evaluation metrics."""
    def __init__(self, y_true, y_pred, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.metrics = {}

    def calculate_metrics(self):
        self.metrics["confusion_matrix"] = confusion_matrix(self.y_true, self.y_pred)
        self.metrics["accuracy"] = accuracy_score(self.y_true, self.y_pred)

    def print_metrics(self):
        for key, value in self.metrics.items():
            print(f"{key}:\n{value}\n")
