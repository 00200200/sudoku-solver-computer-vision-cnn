# Sudoku Solver using Computer Vision and Deep Learning

This project implements an end-to-end pipeline for solving Sudoku puzzles from images using computer vision and deep learning techniques.

## 📁 Project Structure

```
src/
├── model/                  # Neural network models and prediction
│   ├── model.py           # ConvNet and ResNet152 architectures
│   ├── predict.py         # Prediction functions
│   ├── solver.py          # Sudoku solving algorithm
│   └── evaluate_all_models.py # Model comparison script
├── training/              # Training scripts
│   ├── train_sudoku.py    # Train on Sudoku data
│   ├── train_mnist.py     # Train on MNIST data
│   └── train_resnet152_*.py # ResNet training variants
├── finetuning/           # Fine-tuning scripts
│   └── finetune_*.py     # Various fine-tuning approaches
├── data/                 # Data loading and preprocessing
│   └── dataio.py         # Data loaders for MNIST and Sudoku
├── preprocess/           # Image preprocessing
│   └── build_features.py # Sudoku extraction and cell processing
├── evaluate/             # Model evaluation
│   └── evaluate.py       # Evaluation metrics and functions
├── core/                 # Core training utilities
│   └── train.py          # Main training loop
├── common/               # Shared utilities
│   └── tools.py          # Helper functions
└── pipeline.py           # Main inference pipeline
```

## 🧠 Models

### ConvNet

- Lightweight CNN optimized for 28x28 digit recognition
- 2 convolutional layers + 2 fully connected layers
- ~50K parameters
- Fast training and inference

### ResNet152

- Pre-trained on ImageNet, fine-tuned for digit recognition
- Transfer learning with frozen backbone
- Modified first layer for small images (stride=1)
- ~60M parameters

## 🔬 Experiments

Approach follows a systematic methodology:

1. **MNIST Baseline**: Train models on standard MNIST digit recognition
2. **Sudoku Training**: Train directly on Sudoku cell images
3. **Transfer Learning**: Fine-tune MNIST-trained models on Sudoku data
4. **Comparison**: Evaluate all approaches on held-out Sudoku test set

## 📊 Results

### Key Findings

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Poetry (dependency management)
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/00200200/sudoku
cd sudoku
```

2. Install dependencies with Poetry:

```bash
poetry install
```

3. Activate virtual environment:

```bash
poetry shell
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

### Usage

#### Training Models

1. Train ConvNet on Sudoku data:

```bash
python src/training/train_sudoku.py
```

2. Train ResNet152 on MNIST:

```bash
python src/training/train_resnet152_on_mnist.py
```

3. Fine-tune ResNet152 on Sudoku:

```bash
python src/finetuning/finetune_resnet_on_sudoku.py
```

#### Running Inference

Solve a single Sudoku image:

```bash
python src/pipeline.py
```

The pipeline will:

- Extract Sudoku grid from image
- Recognize digits using trained model
- Solve the puzzle algorithmically
- Save results with timestamp in `results/pipeline_outputs/`

#### Model Evaluation

Compare all trained models:

```bash
python src/model/evaluate_all_models.py
```

Results saved to `results/model_comparison_TIMESTAMP.csv`

## 📚 Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **scikit-learn**: Evaluation metrics
- **pandas**: Data analysis and results storage

### Development Tools

- **Poetry**: Dependency management and packaging
- **pre-commit**: Code quality hooks
- **black**: Code formatting
- **isort**: Import sorting
- **pylint**: Static code analysis

See `pyproject.toml` for complete dependency list with versions.

## 🛠 Development

### Code Quality

This project uses pre-commit hooks to ensure code quality:
Run manual checks:

```bash
pre-commit run --all-files
```

### Project Configuration

- `pyproject.toml`: Poetry configuration and project metadata
- `.pre-commit-config.yaml`: Code quality tools configuration
- `poetry.lock`: Dependency lock file

## 🔄 Pipeline Details

### Image Processing

1. **Sudoku Detection**: Find and extract Sudoku grid using contour detection
2. **Perspective Correction**: Apply geometric transformation for top-down view
3. **Cell Extraction**: Divide grid into 81 individual cells (9x9)
4. **Preprocessing**: Convert to grayscale, resize to 28x28, normalize

### ConvNet Architecture

- Input: 28x28 grayscale images (converted to 3-channel for ResNet)
- Output: 10 classes (digits 0-9, where 0 = empty cell)
- Loss: CrossEntropyLoss
- Optimizer: Adam with lr=0.001
