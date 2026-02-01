import cv2
import numpy as np

import src.common.tools as tools
import src.data.dataio as dataio


def perspective_transform(image, corners):
    """Apply perspective transformation for top-down view."""
    def order_corner_points(corners):
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        corners = [(corner[0], corner[1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = (
            corners[3],
            corners[0],
            corners[1],
            corners[2],
        )
        return (top_l, top_r, bottom_r, bottom_l)

    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Calculate width as max distance between bottom or top corners
    width_A = np.sqrt(
        ((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2)
    )
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Calculate height as max distance between left or right corners
    height_A = np.sqrt(
        ((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2)
    )
    height_B = np.sqrt(
        ((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2)
    )
    height = max(int(height_A), int(height_B))

    # Define target dimensions for the warped image
    dimensions = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    ordered_corners = np.array(ordered_corners, dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    return cv2.warpPerspective(image, matrix, (width, height))


def extract_cells_with_coords_from_warped_image(image):
    """Extract individual cells from warped Sudoku grid."""
    h, w = image.shape[:2]
    cell_h, cell_w = h // 9, w // 9

    cells = []
    for i in range(9):
        for j in range(9):
            x, y = j * cell_w, i * cell_h
            cells.append(
                {
                    "image": image[y : y + cell_h, x : x + cell_w],
                    "coords": (x, y, cell_w, cell_h),
                }
            )
    return cells


def finding_sudoku_mask(image):
    """Create binary mask to detect Sudoku grid."""
    sudoku_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sudoku_blur = cv2.GaussianBlur(sudoku_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        sudoku_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )
    dilate = cv2.dilate(thresh, kernel=np.ones((3, 3), np.uint8), iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, np.ones((3, 3)))

    return closing


def extract_sudoku_grid(image, mask):
    """Extract corners of Sudoku grid from mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)

    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # If we don't have 4 points, use minimum area rectangle
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = np.array(box)

    corners = approx.reshape(4, 2)

    return corners


def process_sudoku_image(image, invert_for_mnist_compatibility=True):
    """Process Sudoku image: extract grid, warp, and extract cells."""
    try:
        mask = finding_sudoku_mask(image.copy())
        corners = extract_sudoku_grid(image.copy(), mask)
        warped = perspective_transform(image, corners)

        cells_data = extract_cells_with_coords_from_warped_image(warped)
        processed_cells = []
        coords = []

        for cell in cells_data:
            gray = cv2.cvtColor(cell["image"], cv2.COLOR_BGR2GRAY)

            if invert_for_mnist_compatibility:
                # MNIST format: black background, white digits
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                # Original format: white background, black digits
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

            processed = cv2.resize(binary, (28, 28)) / 255.0

            processed_cells.append(processed)
            coords.append(cell["coords"])

        return processed_cells, coords, warped
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


if __name__ == "__main__":
    config = tools.load_config()
    base = config["base"]
    image_path = base + "sudoku/mixed 2/mixed 2/image2.jpg"

    image = cv2.imread(image_path)
    cv2.imshow("Original Image", image)

    mask = finding_sudoku_mask(image)
    cv2.imshow("Mask", mask)

    contour = extract_sudoku_grid(image, mask)

    warped = perspective_transform(image, contour)
    cv2.imshow("Extracted Sudoku", warped)

    processed_cells, coords_on_warped, warped_display = process_sudoku_image(
        image, invert_for_mnist_compatibility=True
    )
    if processed_cells:
        print(f"Successfully processed {len(processed_cells)} cells.")
        cv2.imshow("Warped Sudoku for Display", warped_display)

    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
