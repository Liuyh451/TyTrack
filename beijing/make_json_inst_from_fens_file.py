import argparse
import json
from pathlib import Path

import torch

from make_json_inst_auto import parse_ensemble_to_fixed_models


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = CURRENT_DIR / "FENS-2026062112.txt"
DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "data"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_member_records(payload: dict):
    for member_obj in payload.get("data", []):
        if not isinstance(member_obj, dict):
            continue
        for key, records in member_obj.items():
            if key.isdigit() and isinstance(records, list):
                for record in records:
                    if isinstance(record, dict):
                        yield record


def infer_typhoon_code(payload: dict) -> str:
    for record in iter_member_records(payload):
        xuhao = record.get("xuhao")
        if xuhao is not None:
            return str(xuhao)
    raise ValueError("Cannot infer typhoon code: no record contains xuhao.")


def infer_data_time(payload: dict) -> str:
    for record in iter_member_records(payload):
        data_time = record.get("datetime")
        if data_time:
            return str(data_time)
    raise ValueError("Cannot infer data time: no record contains datetime.")


def safe_time_dir(data_time: str) -> str:
    return data_time.replace(" ", "-").replace(":", "-")


def save_tensors(payload: dict, output_dir: Path, max_models: int) -> None:
    x, mask, y = parse_ensemble_to_fixed_models(payload, max_models=max_models)
    if x is None:
        raise ValueError("Failed to parse ensemble payload into tensors.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(x), output_dir / "x_results.pt")
    torch.save(torch.tensor(mask), output_dir / "x_masks.pt")
    torch.save(torch.tensor(y), output_dir / "y.pt")

    print(f"Saved x_results.pt: shape={tuple(x.shape)}")
    print(f"Saved x_masks.pt:  shape={tuple(mask.shape)}")
    print(f"Saved y.pt:        shape={tuple(y.shape)}")
    print(f"Output directory:  {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build model input tensors from a local FENS ensemble JSON file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Local FENS JSON/txt file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--typhoon-code",
        default=None,
        help="Override typhoon code. Defaults to xuhao from the file.",
    )
    parser.add_argument(
        "--data-time",
        default=None,
        help="Override data time. Defaults to datetime from the file.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=8,
        help="Maximum ensemble members to keep. Default: 8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = load_json(args.input)

    if payload.get("code") != 200:
        raise ValueError(f"Unexpected payload code: {payload.get('code')}")

    typhoon_code = str(args.typhoon_code or infer_typhoon_code(payload))
    data_time = str(args.data_time or infer_data_time(payload))
    output_dir = args.output_root / typhoon_code / safe_time_dir(data_time)

    print(f"Input file:       {args.input}")
    print(f"Typhoon code:     {typhoon_code}")
    print(f"Data time:        {data_time}")
    save_tensors(payload, output_dir, args.max_models)

    print()
    print("Run inference with:")
    print(
        "python infer_case_auto_plot.py "
        f"--ty_number {typhoon_code} --report_time {safe_time_dir(data_time)}"
    )


if __name__ == "__main__":
    main()
