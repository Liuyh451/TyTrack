from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    "combined_mask_test.pt",
    "X_test.pt",
    "y_test.pt",
]
INDEX = 59
OUTPUT_SUFFIX = "_idx60"


def slice_first_dim_index_60(file_names=INPUT_FILES, index=INDEX):
    """Keep the 60th item on axis 0 for each .pt file and save new files."""
    saved_shapes = {}

    for file_name in file_names:
        input_path = BASE_DIR / file_name
        data = torch.load(input_path, map_location="cpu")

        if not hasattr(data, "shape"):
            raise TypeError(f"{input_path} does not contain a tensor-like object.")

        if data.shape[0] <= index:
            raise IndexError(
                f"{input_path} first dimension is {data.shape[0]}, "
                f"cannot keep index {index}."
            )

        sliced = data[index : index + 1]
        output_path = input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}{input_path.suffix}")
        torch.save(sliced, output_path)

        saved_shapes[str(output_path)] = tuple(sliced.shape)
        print(f"{input_path.name}: {tuple(data.shape)} -> {tuple(sliced.shape)}")
        print(f"saved: {output_path}")

    return saved_shapes


if __name__ == "__main__":
    slice_first_dim_index_60()
