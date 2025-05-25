import os
import struct

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_image(path):
    return cv2.imread(path)


def load_dat(path):
    try:
        with open(path) as f:
            next(f)
            next(f)
            return [int(x) for line in f for x in line.strip().split()]
    except FileNotFoundError:
        print(f"Warning: Label file not found at {path}, skipping image.")
        return None
    except Exception as e:
        print(f"Warning: Error loading label file {path}: {e}, skipping image.")
        return None


# MNIST loading functions
def read_idx(filename):
    with open(filename, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic == 2051:  # Images
            rows, cols = struct.unpack(">II", f.read(8))
            return (
                np.fromfile(f, dtype=np.uint8)
                .reshape(size, rows * cols)
                .astype(np.float32)
                / 255.0
            )
        elif magic == 2049:  # Labels
            return np.fromfile(f, dtype=np.uint8)


def load_mnist(data_dir):
    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    if not os.path.exists(train_images_path):
        train_images_path = os.path.join(
            data_dir, "train-images-idx3-ubyte", "train-images.idx3-ubyte"
        )

    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    if not os.path.exists(train_labels_path):
        train_labels_path = os.path.join(
            data_dir, "train-labels-idx1-ubyte", "train-labels.idx1-ubyte"
        )

    test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    if not os.path.exists(test_images_path):
        test_images_path = os.path.join(
            data_dir, "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"
        )

    test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
    if not os.path.exists(test_labels_path):
        test_labels_path = os.path.join(
            data_dir, "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"
        )

    return (
        read_idx(train_images_path),
        read_idx(train_labels_path),
        read_idx(test_images_path),
        read_idx(test_labels_path),
    )


class SudokuDataset(Dataset):
    def __init__(self, data_dirs, cell_processor):
        self.images = []
        self.labels = []
        self.cell_processor = cell_processor

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        for data_dir in data_dirs:
            print(f"Loading Sudoku dataset from: {data_dir}")
            if not os.path.isdir(data_dir):
                print(f"Warning: Directory not found {data_dir}, skipping.")
                continue

            for file in os.listdir(data_dir):
                if not file.endswith(".jpg"):
                    continue

                # Load and process image/labels
                image_path = os.path.join(data_dir, file)
                label_path = image_path.replace(".jpg", ".dat")

                if not os.path.exists(label_path):
                    continue

                labels = load_dat(label_path)
                image = load_image(image_path)

                if not labels or image is None:
                    continue

                # Process cells
                processor_output = self.cell_processor(image)
                if not processor_output or not processor_output[0]:
                    print(
                        f"Warning: No cells extracted for {image_path} using the provided processor, skipping."
                    )
                    continue

                cells = processor_output[0]

                # Add to dataset
                self.images.extend(cells)
                self.labels.extend(labels)

        # Convert to arrays
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        return image, self.labels[idx]


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).reshape(1, 28, 28)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        return image, int(self.labels[idx])


def get_sudoku_loaders(
    train_dirs, cell_processor, test_dir=None, batch_size=32, train_split=0.8
):
    train_dataset = SudokuDataset(train_dirs, cell_processor=cell_processor)
    print(f"Sudoku training dataset size: {len(train_dataset)} samples")

    if not train_dataset or len(train_dataset) == 0:
        print(
            "Warning: Sudoku training dataset is empty. Check data paths and processing."
        )
        return None, None

    if test_dir:
        test_dataset = SudokuDataset(test_dir, cell_processor=cell_processor)
        print(f"Sudoku test dataset size: {len(test_dataset)} samples")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Split dataset
        train_size = int(train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_mnist_loaders(data_dir, batch_size=32):
    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    print(
        f"MNIST dataset size: {len(train_images)} training, {len(test_images)} test samples"
    )

    return (
        DataLoader(
            MNISTDataset(train_images, train_labels),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            MNISTDataset(test_images, test_labels), batch_size=batch_size, shuffle=False
        ),
    )
