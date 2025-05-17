# Import libraries and classes required for this example:
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocess.build_features import process_sudoku_image


def load_image(path):
    return cv2.imread(path)


def load_dat(path):
    try:
        with open(path) as f:
            next(f)
            next(f)  # Skip metadata
            grid = []
            for line in f:
                grid.extend([int(x) for x in line.strip().split()])
        return grid
    except:
        return None


def process_cells(image):
    try:
        cells = process_sudoku_image(image)
        return cells
    except:
        return None


def load_sudoku_data(data_dir):
    """Load and preprocess all sudoku images and labels from a directory"""
    images = []
    labels = []

    for file in os.listdir(data_dir):
        if not file.endswith(".jpg"):
            continue

        # Get paths
        image_path = os.path.join(data_dir, file)
        label_path = image_path.replace(".jpg", ".dat")
        if not os.path.exists(label_path):
            continue

        # Load data
        grid_labels = load_dat(label_path)
        image = load_image(image_path)
        if not grid_labels or image is None:
            continue

        # Process image
        cells = process_cells(image)
        if not cells:
            continue

        # Add to dataset
        images.extend(cells)
        labels.extend(grid_labels)

    # Convert to arrays
    return (np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64))


class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).unsqueeze(0), self.labels[idx]


def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    # Load all data
    images, labels = load_sudoku_data(data_dir)
    print(f"Dataset size: {len(images)} samples")

    # Split dataset
    train_size = int(train_split * len(images))
    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets
    train_dataset = CustomImageDataset(images[train_indices], labels[train_indices])
    val_dataset = CustomImageDataset(images[val_indices], labels[val_indices])

    # Create loaders
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )
