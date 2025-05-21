import os

import cv2
import torch

from src.model.model import ResNet152
from src.model.solver import Suduko as solve_sudoku_algorithm
from src.preprocess.build_features import process_sudoku_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def overlay_digits(base_image, grid_digits, cell_coords, color=(0, 0, 0)):
    output_image = base_image.copy()
    for i in range(9):
        for j in range(9):
            if grid_digits[i][j] != 0:
                x, y, w, h = cell_coords[i * 9 + j]
                text_size = cv2.getTextSize(
                    str(grid_digits[i][j]), cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0]
                text_position = (
                    x + w // 2 - text_size[0] // 2,
                    y + h // 2 + text_size[1] // 2,
                )
                cv2.putText(
                    output_image,
                    str(grid_digits[i][j]),
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
    return output_image


def predict_grid(model, cell_images):
    model.eval()
    grid = [[0] * 9 for _ in range(9)]

    with torch.no_grad():
        for i in range(9):
            for j in range(9):
                cell_image = cell_images[i * 9 + j]
                tensor_image = (
                    torch.from_numpy(cell_image)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                output = model(tensor_image)
                grid[i][j] = torch.max(output.data, 1)[1].item()

    return grid


def main_pipeline(image_path, model_path):
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    cells, coords, warped = process_sudoku_image(image)
    if cells is None or coords is None or warped is None:
        raise ValueError("Failed to process Sudoku image")

    # Load model and predict
    model = ResNet152().to(device)
    model.load_state_dict(torch.load(model_path))

    # Predict and solve
    grid = predict_grid(model, cells)
    grid_solution = [row[:] for row in grid]

    if solve_sudoku_algorithm(grid_solution, 0, 0):
        cv2.imshow("Original", image)
        cv2.imshow("Solution", overlay_digits(warped, grid_solution, coords))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return grid_solution
    return None


if __name__ == "__main__":
    workspace_root = "/Users/michal/Library/Code/sudoku/"
    image_path = os.path.join(
        workspace_root, "data/raw/sudoku/v1_test/v1_test/image8.jpg"
    )
    model_path = os.path.join(workspace_root, "models/resnest_sudoku_finetuned.pkl")
    main_pipeline(image_path, model_path)
