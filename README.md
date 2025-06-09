# Sudoku Solver using Computer Vision and Deep Learning

This project implements an end-to-end pipeline for solving Sudoku puzzles from images using computer vision and deep learning techniques.

## 📸 Pipeline in Action

|                           Original Image                           |                               Extracted Grid                                |                         Solved Sudoku                          |
| :----------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Original](results/pipeline_outputs/20250605_113638_original.jpg) | ![Extracted](results/pipeline_outputs/20250605_113638_extracted_sudoku.jpg) | ![Solved](results/pipeline_outputs/20250605_113638_solved.jpg) |
|                          Raw input image                           |                           Detected & warped grid                            |                    Final solution overlaid                     |

**🏆 Best Model:** ConvNet achieves **95.68% accuracy** on digit recognition with only 50K parameters.

## 📁 Project Structure

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

## 🧠 Models

### ConvNet (Primary Model)

Our custom lightweight CNN architecture optimized specifically for Sudoku digit recognition:

**Architecture:**

- **Input**: 28x28 RGB images (converted from grayscale)
- **Conv1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPool**: 2x2 pooling
- **Conv2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPool**: 2x2 pooling
- **FC1**: 64 _ 7 _ 7 → 64 neurons, ReLU activation
- **FC2**: 64 → 10 classes (digits 0-9)

**Specifications:**

- Parameters: ~50,000 (lightweight and efficient)
- Training time: ~10-15 minutes on GPU for 50 epochs
- Inference: Real-time performance
- Memory usage: Minimal

### ResNet152 (Experimental)

Deep residual network with transfer learning approach:

- Pre-trained on ImageNet, fine-tuned for digit recognition
- ~60M parameters (1200x larger than ConvNet)
- Significantly higher computational complexity
- Designed for complex image classification tasks

## 📊 Results

### 🏆 Best Performing Models

| Rank | Model                                          | Accuracy   | Training Approach       |
| ---- | ---------------------------------------------- | ---------- | ----------------------- |
| 🥇   | **50epochs-convnet-sudoku-only**               | **95.68%** | Direct Sudoku training  |
| 🥈   | 150epochs-convnet-sudoku-very-long-small-batch | 95.52%     | Extended training       |
| 🥉   | 140epochs-convnet-sudoku-higher-lr-longer      | 95.37%     | Higher learning rate    |
| 4️⃣   | 100epochs-convnet-sudoku-larger-batch-lower-lr | 94.75%     | Batch size optimization |

### 🔍 Key Findings

**✅ What Worked:**

- **ConvNet trained directly on Sudoku data achieved the best results** (95.68% accuracy)
- **Optimal configuration**: 50 epochs, Adam optimizer (lr=0.001), batch size=32
- Models trained specifically on Sudoku outperformed all transfer learning approaches

**❌ Transfer Learning Did Not Help:**

- MNIST pre-training surprisingly **hurt performance** rather than helped
- MNIST-to-Sudoku fine-tuned models peaked at ~94% accuracy (1.5% lower than direct training)
- MNIST-only models performed poorly on Sudoku (~13-52% accuracy)

**🎯 ResNet Consideration:**
While ResNet152 models showed potential, they faced several limitations:

- **Computational overhead**: 60M vs 50K parameters (1200x larger)
- **Training time**: Hours vs minutes
- **Inference speed**: Slower real-time performance
- **Model incompatibility**: Several ResNet models failed to load due to architecture mismatches

### 📈 Performance Analysis

**ConvNet Advantages:**

- **Efficiency**: Excellent accuracy-to-complexity ratio
- **Speed**: Fast training and inference
- **Simplicity**: Easy to deploy and maintain
- **Sufficient performance**: 95.68% accuracy meets project requirements

**Conclusion:**
For this Sudoku recognition task, the lightweight ConvNet provides the **optimal balance** between accuracy and computational efficiency. While more complex architectures like ResNet could potentially achieve higher accuracy, the **95.68% result is satisfactory** for practical Sudoku solving applications, and the added complexity is not justified.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/00200200/sudoku
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

## 📚 Tech Stack

**Core:** PyTorch, OpenCV, NumPy, scikit-learn, pandas  
**Development:** Poetry, pre-commit hooks, black, isort  
**Requirements:** Python 3.9+, CUDA GPU (recommended)

## 🔄 Pipeline

**Image → Grid Detection → Cell Recognition → Puzzle Solving → Solution Overlay**

1. **Sudoku Detection**: Contour detection and perspective correction
2. **Cell Extraction**: 81 individual 28x28 images
3. **Digit Recognition**: ConvNet predicts 0-9 for each cell
4. **Solving**: Backtracking algorithm completes the puzzle
5. **Output**: Solution overlaid on original image

**Best Model:** `50epochs-convnet-sudoku-only` - 95.68% accuracy, 50K parameters, <1s inference
