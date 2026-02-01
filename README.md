# Sudoku Solver with Computer Vision & Deep Learning

Automated Sudoku puzzle solver using computer vision for grid extraction and deep learning for digit recognition.

## Pipeline in Action

|                           Original Image                           |                               Extracted Grid                                |                         Solved Sudoku                          |
| :----------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Original](results/pipeline_outputs/20250605_113638_original.jpg) | ![Extracted](results/pipeline_outputs/20250605_113638_extracted_sudoku.jpg) | ![Solved](results/pipeline_outputs/20250605_113638_solved.jpg) |
|                          Raw input image                           |                           Detected & warped grid                            |                    Final solution overlaid                     |

## Features

- **Image Processing**: Automatic Sudoku grid detection and cell extraction
- **Digit Recognition**: CNN and ResNet152 models for handwritten digit classification
- **Sudoku Solving**: Backtracking algorithm for puzzle solving
- **Transfer Learning**: Pre-training on MNIST with fine-tuning on Sudoku datasets

## Project Structure

```
src/
├── common/          # Utility functions (config, pickle)
├── core/            # Training logic
├── data/            # Dataset loaders (MNIST, Sudoku)
├── evaluate/        # Model evaluation scripts
├── model/           # Model architectures (ConvNet, ResNet152)
├── preprocess/      # Image processing pipeline
└── scripts/         # Main executable scripts
    ├── train.py     # Universal training script
    └── pipeline.py  # End-to-end Sudoku solver
```

## Quick Start

### Training a Model

Train ConvNet on MNIST:

```bash
python src/scripts/train.py --model convnet --dataset mnist --epochs 10
```

Train ConvNet on Sudoku:

```bash
python src/scripts/train.py --model convnet --dataset sudoku --epochs 50
```

Fine-tune MNIST model on Sudoku:

```bash
python src/scripts/train.py --model convnet --dataset sudoku --epochs 40 \
  --finetune models/10epochs_convnet_mnist.pkl --lr 0.0005
```

See [TRAINING.md](TRAINING.md) for more training examples.

### Solving a Sudoku Puzzle

```bash
python src/scripts/pipeline.py
```

Or use in Python:

```python
from src.scripts.pipeline import main_pipeline

solution = main_pipeline(
    image_path="path/to/sudoku.jpg",
    model_path="models/50epochs_convnet_sudoku.pkl",
    save_images=True,
    show_images=True
)
```

### Evaluating Models

Evaluate all models in the models directory:

```bash
python src/evaluate/evaluate_all_models.py
```

## Models

### ConvNet

- Simple CNN architecture for 28x28 digit images
- 2 conv layers + 2 FC layers
- Fast training and inference

### ResNet152

- Pre-trained on ImageNet with custom classification head
- Fine-tuned for digit recognition
- Higher accuracy but slower

## Dependencies

- PyTorch
- OpenCV (cv2)
- NumPy
- Pandas
- scikit-learn
- PyYAML

Install with:

```bash
poetry install
```

## Results

Models are automatically evaluated and saved in `results/` directory with timestamps.

Pipeline outputs (original image, extracted grid, solution) are saved in `results/pipeline_outputs/`.
