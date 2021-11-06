import numpy as np
from cv2 import cv2 as cv

kernel1 = np.ones((3, 3), np.float32) / 25
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def get_satellite_raw_uint8(coord, block_size, img_data, dataset):
    x, y = dataset.index(float(coord['lon']), float(coord['lat']))
    img = img_data[x - block_size // 2:x + block_size // 2, y - block_size // 2:y + block_size // 2]
    img = (img / np.max(img) * 255).astype(np.uint8)
    return img


def remove_noise_satellite(img):
    mask = img / 16
    mask = np.power(mask, 2)
    mask = cv.filter2D(mask, -1, kernel1)
    avg_val = np.average(mask) + 10
    mask[mask < avg_val] = np.multiply(mask[mask < avg_val], 0.5)
    mask = cv.dilate(mask, kernel2, iterations=2)
    img = np.copy(img) * mask.astype(np.float32)
    img *= 255.0 / img.max()
    return img


def get_filtered_satellite_float32(coord, block_size, img_data, dataset):
    img = get_satellite_raw_uint8(coord, block_size, img_data, dataset)
    img = remove_noise_satellite(img)
    img = np.expand_dims(img, -1)
    return img
