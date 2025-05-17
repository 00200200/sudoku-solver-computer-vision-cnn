import cv2
import numpy as np

import src.common.tools as tools
import src.data.dataio as dataio

# GaussianBlur -  Rozmycie
# dilate - Dylatacja (poszerzenie białych obszarów)
# erode - Erozja (zmniejszenie białych obszarów)
# findContours - Wykrywanie konturów
# getPerspectiveTransform - Transformacja perspektywy rozni sie od warpPerspective tym ze nie zmienia wielkosci obrazu
#  # warpPerspective - Transformacja perspektywy


# Przeprocesowanie danych z sudoku:
# znalezienie maski sudoku, ✔️
# wyciąć sudoku na podstawie maski i dostosować perspektywę, ✔️
# wyciąć poszczególne pola
def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0], corner[1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = (
            corners[3],
            corners[0],
            corners[1],
            corners[2],
        )
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(
        ((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2)
    )
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(
        ((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2)
    )
    height_B = np.sqrt(
        ((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2)
    )
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_each_cell(image):
    # Znajdź kontury wszystkich komórek
    height, width = image.shape[:2]
    cell_height = height // 9
    cell_width = width // 9
    cells = []
    for i in range(9):
        cells_row = []
        for j in range(9):
            x = j * cell_width
            y = i * cell_height
            cell = image[y : y + cell_height, x : x + cell_width]
            cells_row.append(cell)
        cells.append(cells_row)

    return cells


def finding_sudoku_mask(image):
    sudoku_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # BLur
    sudoku_blur = cv2.GaussianBlur(sudoku_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        sudoku_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )
    dilate = cv2.dilate(thresh, kernel=np.ones((3, 3), np.uint8), iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, np.ones((3, 3)))

    return closing


def extract_sudoku_grid(image, mask):
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find Contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw Contour on the image
    cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)

    peri = cv2.arcLength(largest_contour, True)  # Get the perimeter
    approx = cv2.approxPolyDP(
        largest_contour, 0.02 * peri, True
    )  # Przyblizanie konturu 2 % obwodu ( True Zamkniety kontur)

    # If we still don't have 4 points
    if len(approx) != 4:
        rect = cv2.minAreaRect(
            largest_contour
        )  # MinAreaRect zwraca prostokat, ktory ma najmniejsza powierzchnie i zawiera kontur
        box = cv2.boxPoints(
            rect
        )  # BoxPoints zwraca 4 punkty, ktore sa wierzcholkami prostokata
        approx = np.array(box)

    corners = approx.reshape(4, 2)  # Reshape to 4x2 array

    return corners


def process_sudoku_image(image):
    try:
        mask = finding_sudoku_mask(image)
        contour = extract_sudoku_grid(image, mask)
        warped = perspective_transform(image, contour)
        cells = crop_each_cell(warped)

        processed_cells = []
        for row in cells:
            for cell in row:
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized / 255.0
                processed_cells.append(normalized)

        return processed_cells
    except Exception as e:
        print(f"Error processing Sudoku image: {e}")
        return None


def preprocess(config):
    # Load the data
    rawdatapath = config["datarawdirectory"] + config["dataname"] + ".csv"
    [X, y] = dataio.load(rawdatapath)

    # # Save intermediate products
    # savepath = config["datainterimdirectory"]
    # # dataio.save(X_train, y_train, savepath + "train.csv")
    # # dataio.save(X_test, y_test, savepath + "test.csv")

    # # Save final products
    # savepath = config["dataprocesseddirectory"]
    # # dataio.save(X_train_scaled, y_train, savepath + "train.csv")
    # # dataio.save(X_test_scaled, y_test, savepath + "test.csv")


if __name__ == "__main__":
    config = tools.load_config()
    base = config["base"]
    image_path = base + "sudoku/mixed 2/mixed 2/image2.jpg"

    # 1. Wczytaj obraz
    image = cv2.imread(image_path)
    cv2.imshow("Oryginalny obraz", image)

    # 2. Utwórz maskę sudoku
    mask = finding_sudoku_mask(image)
    cv2.imshow("Maska", mask)

    # 3. Wytnij sudoku z obrazu
    contour = extract_sudoku_grid(image, mask)
    # cv2.imshow("Wyciete sudoku", contour)

    # 4. Zastosuj transformację perspektywy
    warped = perspective_transform(image, contour)
    cv2.imshow("Wyciete sudoku", warped)

    cells = crop_each_cell(warped)
    print(len(cells))
    print("Naciśnij dowolny klawisz, aby zakończyć...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
