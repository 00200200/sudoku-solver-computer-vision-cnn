import numpy as np
import torch

import src.common.tools as tools
import src.data.dataio as dataio
from src.evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_model(model, data_loader):
    """Run predictions on dataset."""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


def predict_single_image(model, image_tensor):
    """Predict class for single image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
