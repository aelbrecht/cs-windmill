import glob

import imageio
import numpy as np
import rasterio
from cv2 import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .image import remove_noise_satellite

default_land_cov_path = "./data/E000N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
default_sat_path = "./data/crs.tiff"
default_examples_path = "./examples/*/*.png"
block_size = 30


class SatelliteData:
    _sat_path = ""
    _sat_img = None
    _sat_set = None
    _cov_path = ""
    _cov_img = None
    _cov_set = None

    def __init__(self, satellite_path: str = default_sat_path, land_coverage_path: str = default_land_cov_path):
        self._sat_path = satellite_path
        self._cov_path = land_coverage_path
        self._load_dataset()

    def satellite_image(self):
        return self._sat_img

    def land_coverage_image(self):
        return self._cov_img

    def satellite(self):
        return self._sat_set

    def land_coverage(self):
        return self._cov_set

    # loads datasets from disk
    def _load_dataset(self):
        # retrieve satellite data
        self._sat_set = rasterio.open(self._sat_path)
        self._sat_img = self._sat_set.read(1)

        # retrieve land coverage data
        # https://lcviewer.vito.be/download
        self._cov_set = rasterio.open(self._cov_path)
        self._cov_img = self._cov_set.read(1)

        # create bitmap from land coverage data
        self._cov_img[self._cov_img != 200] = 0  # 200 is specified as "water-body"
        self._cov_img[self._cov_img == 200] = 1

        # dilate land bitmap (using erode as ocean is 1 in this case)
        kernel = np.ones((5, 5), np.uint8)
        self._cov_img = cv.erode(self._cov_img, kernel, iterations=5)  # 5 iterations seems to remove most beaches


def load_samples(dataset_dir: str = default_examples_path):
    examples = glob.glob(dataset_dir, recursive=True)
    images = []
    labels = []

    valid = ["windmills", "boats", "ocean", "beach"]
    target = ["windmills", "boats"]

    for path in examples:

        category = path.split("/")[2]
        if category not in valid:
            continue

        src = imageio.imread(path)
        src = np.expand_dims(remove_noise_satellite(src) / 255.0, -1)

        # import images oriented 4 ways
        for i in range(4):
            images.append(np.rot90(src.copy(), i))
            if category in target:
                labels.append(1)
            else:
                labels.append(0)

    return images, labels


def load_samples_split(dataset_dir: str = default_examples_path):
    images, labels = load_samples(dataset_dir=dataset_dir)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    x_train = np.stack(x_train)
    x_test = np.stack(x_test)
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    return x_train, x_test, y_train, y_test
