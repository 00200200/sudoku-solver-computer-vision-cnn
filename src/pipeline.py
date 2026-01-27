import os
from datetime import datetime

import cv2
import torch

from src.model.model import ConvNet, ResNet152
from src.model.solver import Sudoku as solve_sudoku_algorithm
from src.preprocess.build_features import process_sudoku_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def overlay_digits(base_image, grid_digits, cell_coords, color=(0, 0, 0)):
    """
    Overlay digit predictions onto the Sudoku grid image.
    
    Args:
        base_image: Base image to draw on
        grid_digits: 9x9 grid of digit predictions
        cell_coords: List of (x, y, w, h) coordinates for each cell
        color: Color tuple for the text (B, G, R format)
    
    Returns:
        Image with overlaid digits
    """
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
    """
    Predict digits for each cell in the Sudoku grid.
    
    Args:
        model: Trained digit recognition model
        cell_images: List of 81 preprocessed cell images
    
    Returns:
        9x9 grid of predicted digits (0-9)
    """
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
                tensor_image = tensor_image.repeat(1, 3, 1, 1)

                output = model(tensor_image)
                grid[i][j] = torch.max(output.data, 1)[1].item()

    return grid


def save_results(
    original_image, warped_sudoku, solved_image, output_dir="results/pipeline_outputs"
):
    """
    Save pipeline results with timestamp.
    
    Args:
        original_image: Original input image
        warped_sudoku: Extracted and warped Sudoku grid
        solved_image: Sudoku grid with solution overlay
        output_dir: Directory to save results
    
    Returns:
        Timestamp string used for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(f"{output_dir}/{timestamp}_original.jpg", original_image)
    cv2.imwrite(f"{output_dir}/{timestamp}_extracted_sudoku.jpg", warped_sudoku)
    cv2.imwrite(f"{output_dir}/{timestamp}_solved.jpg", solved_image)

    print(f"Results saved to {output_dir}/ with timestamp {timestamp}")
    return timestamp


def main_pipeline(image_path, model_path, save_images=True, show_images=True):
    """
    Complete pipeline to process a Sudoku image and solve it.
    
    Args:
        image_path: Path to input Sudoku image
        model_path: Path to trained digit recognition model
        save_images: Whether to save intermediate and final results
        show_images: Whether to display images in windows
    
    Returns:
        Solved grid (9x9 list) or None if solving failed
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    cells, coords, warped = process_sudoku_image(image)
    if cells is None or coords is None or warped is None:
        raise ValueError("Failed to process Sudoku image")

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    grid = predict_grid(model, cells)
    grid_solution = [row[:] for row in grid]

    if solve_sudoku_algorithm(grid_solution, 0, 0):
        solved_image = overlay_digits(warped, grid_solution, coords)

        if save_images:
            timestamp = save_results(image, warped, solved_image)
            print(
                f"Pipeline completed successfully. Results saved with timestamp: {timestamp}"
            )

        if show_images:
            cv2.imshow("Original", image)
            cv2.imshow("Extracted Sudoku", warped)
            cv2.imshow("Solution", solved_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return grid_solution
    else:
        print("Failed to solve the Sudoku puzzle")
        return None


if __name__ == "__main__":
    workspace_root = "/Users/michal/Library/Code/sudoku/"
    image_path = os.path.join(
        workspace_root, "data/raw/sudoku/v1_test/v1_test/image8.jpg"
    )

    model_path = os.path.join(workspace_root, "models/50epochs_convnet_sudoku_only.pkl")
    main_pipeline(image_path, model_path)
