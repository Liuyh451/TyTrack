from pathlib import Path

import numpy as np


INPUT_PATH = Path(__file__).with_name("env_scale_separa_pooled_windowed.npy")
OUTPUT_PATH = Path(__file__).with_name("env_scale_separa_pooled_windowed_first1.npy")


def keep_first_dim_one(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Keep only the first item on axis 0 and save it as a new .npy file."""
    array = np.load(input_path, mmap_mode="r")
    sliced = array[:1]
    np.save(output_path, sliced)
    print(f"saved: {output_path}")
    print(f"shape: {sliced.shape}")
    return sliced.shape


if __name__ == "__main__":
    keep_first_dim_one()
