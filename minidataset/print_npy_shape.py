from pathlib import Path

import numpy as np


DEFAULT_NPY_PATH = Path(__file__).with_name("env_scale_separa_pooled_windowed.npy")


def output_shape(file_path=DEFAULT_NPY_PATH):
    """Print and return the shape of a .npy array."""
    array = np.load(file_path, mmap_mode="r")
    print(array.shape)
    return array.shape


if __name__ == "__main__":
    output_shape()
