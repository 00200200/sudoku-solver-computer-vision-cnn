import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.common.tools as tools
import src.data.dataio as dataio

# GaussianBlur -  Rozmycie
# dilate - Dylatacja (poszerzenie białych obszarów)
# erode - Erozja (zmniejszenie białych obszarów)
# findContours - Wykrywanie konturów
# getPerspectiveTransform - Transformacja perspektywy rozni sie od warpPerspective tym ze nie zmienia wielkosci obrazu
#  # warpPerspective - Transformacja perspektywy


def finding_sudoku_mask(image):
    sudoku_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sudoku_gray


def preprocess(config):
    # Load the data
    rawdatapath = config["datarawdirectory"] + config["dataname"] + ".csv"
    [X, y] = dataio.load(rawdatapath)

    # Save intermediate products
    savepath = config["datainterimdirectory"]
    # dataio.save(X_train, y_train, savepath + "train.csv")
    # dataio.save(X_test, y_test, savepath + "test.csv")

    # Save final products
    savepath = config["dataprocesseddirectory"]
    # dataio.save(X_train_scaled, y_train, savepath + "train.csv")
    # dataio.save(X_test_scaled, y_test, savepath + "test.csv")


if __name__ == "__main__":
    config = tools.load_config()
    base = config["base"]
    image_path = base + "sudoku/mixed 2/mixed 2/image2.jpg"

    image = cv2.imread(image_path)
    mask = finding_sudoku_mask(image)

    cv2.imshow("mask", mask)
    print("Naciśnij dowolny klawisz, aby zakończyć...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
