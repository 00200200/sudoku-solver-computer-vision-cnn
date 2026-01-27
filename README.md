# Sudoku Solver using Computer Vision and Deep Learning

This project implements an end-to-end pipeline for solving Sudoku puzzles from images using computer vision and deep learning techniques.

## Pipeline in Action

|                           Original Image                           |                               Extracted Grid                                |                         Solved Sudoku                          |
| :----------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Original](results/pipeline_outputs/20250605_113638_original.jpg) | ![Extracted](results/pipeline_outputs/20250605_113638_extracted_sudoku.jpg) | ![Solved](results/pipeline_outputs/20250605_113638_solved.jpg) |
|                          Raw input image                           |                           Detected & warped grid                            |                    Final solution overlaid                     |

**Best Model:** ConvNet achieves **95.68% accuracy** on digit recognition with only 50K parameters.

## Project Structure

```
src/
├── model/                 # Neural network models
│   ├── model.py          # ConvNet & ResNet152 architectures
│   ├── predict.py        # Prediction functions
│   └── solver.py         # Sudoku solving algorithm
├── training/             # Training scripts
│   ├── train_convnet_on_sudoku.py  # Best model training
│   ├── train_convnet_on_mnist.py   # MNIST baseline
│   └── train_resnet152_*.py        # ResNet variants
├── finetuning/           # Transfer learning experiments
├── evaluate/             # Model evaluation
│   └── evaluate_all_models.py     # Comprehensive comparison
├── data/                 # Data loading
├── preprocess/           # Image preprocessing
├── core/                 # Training utilities
└── pipeline.py           # Main inference pipeline
```

## Models

### ConvNet (Primary Model)

Our custom lightweight CNN architecture optimized specifically for Sudoku digit recognition:

**Architecture:**

- **Input**: 28x28 RGB images (converted from grayscale)
- **Conv1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPool**: 2x2 pooling
- **Conv2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPool**: 2x2 pooling
- **FC1**: 64 x 7 x 7 → 64 neurons, ReLU activation
- **FC2**: 64 → 10 classes (digits 0-9)

**Specifications:**

- Parameters: ~50,000 (lightweight and efficient)
- Training time: ~10-15 minutes on GPU for 50 epochs
- Inference: Real-time performance
- Memory usage: Minimal



## Results

### Best Performing Models

| Rank | Model                                          | Accuracy   | Training Approach       |
| ---- | ---------------------------------------------- | ---------- | ----------------------- |
| 1st  | **50epochs-convnet-sudoku-only**               | **95.68%** | Direct Sudoku training  |
| 2nd  | 150epochs-convnet-sudoku-very-long-small-batch | 95.52%     | Extended training       |
| 3rd  | 140epochs-convnet-sudoku-higher-lr-longer      | 95.37%     | Higher learning rate    |
| 4th  | 100epochs-convnet-sudoku-larger-batch-lower-lr | 94.75%     | Batch size optimization |


## Quick Start

### Installation

```bash
git clone https://github.com/00200200/sudoku-solver-computer-vision-cnn
cd sudoku
poetry install && poetry shell
```

### Usage

**Solve Sudoku from image:**

```bash
python src/pipeline.py
```

**Train best model:**

```bash
python src/training/train_convnet_on_sudoku.py
```

**Evaluate all models:**

```bash
python src/evaluate/evaluate_all_models.py
```

## Tech Stack

**Core:** PyTorch, OpenCV, NumPy, scikit-learn, pandas  
**Development:** Poetry, pre-commit hooks, black, isort  
**Requirements:** Python 3.9+, CUDA GPU (recommended)

## Pipeline

**Image → Grid Detection → Cell Recognition → Puzzle Solving → Solution Overlay**

1. **Sudoku Detection**: Contour detection and perspective correction
2. **Cell Extraction**: 81 individual 28x28 images
3. **Digit Recognition**: ConvNet predicts 0-9 for each cell
4. **Solving**: Backtracking algorithm completes the puzzle
5. **Output**: Solution overlaid on original image

**Best Model:** `50epochs-convnet-sudoku-only` - 95.68% accuracy, 50K parameters, <1s inference
