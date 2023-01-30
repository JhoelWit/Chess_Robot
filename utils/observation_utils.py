import numpy as np


def normalize_img(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))