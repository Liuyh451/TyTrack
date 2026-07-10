import argparse
import os
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPECTED_FILES = ("x_results.pt", "x_masks.pt", "y.pt")


def find_latest_result_dir(data_dir: Path) -> Path:
    result_files = list(data_dir.glob("*/*/x_results.pt"))
    if not result_files:
        raise FileNotFoundError(f"未在 {data_dir} 下找到 x_results.pt")

    latest_file = max(result_files, key=lambda path: path.stat().st_mtime)
    return latest_file.parent


def resolve_result_dir(args) -> Path:
    if args.result_dir:
        return Path(args.result_dir).resolve()

    if args.typhoon_code and args.time:
        return DATA_DIR / str(args.typhoon_code) / args.time

    return find_latest_result_dir(DATA_DIR)


def load_tensor(result_dir: Path, filename: str):
    path = result_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    return torch.load(path, map_location="cpu")


def print_tensor(name: str, tensor, full: bool):
    print(f"\n{name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(f"dtype: {tensor.dtype}")

    if name == "x_masks.pt":
        print(f"valid count: {int(tensor.sum().item())}")

    if full:
        print(tensor)
    else:
        print(tensor)


def main():
    parser = argparse.ArgumentParser(description="打印已保存的台风预报张量结果")
    parser.add_argument("--result-dir", help="直接指定结果目录")
    parser.add_argument("--typhoon-code", help="台风编号，例如 20260054")
    parser.add_argument("--time", help="时间目录，例如 2026-07-07-18-00-00")
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整打印大张量，默认也会打印 tensor 内容",
    )
    args = parser.parse_args()

    result_dir = resolve_result_dir(args)
    print(f"结果目录: {result_dir}")

    missing = [name for name in EXPECTED_FILES if not (result_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"结果目录缺少文件: {missing}")

    torch.set_printoptions(precision=4, sci_mode=False, threshold=10000)

    x = load_tensor(result_dir, "x_results.pt")
    mask = load_tensor(result_dir, "x_masks.pt")
    y = load_tensor(result_dir, "y.pt")

    print_tensor("x_results.pt", x, args.full)
    print_tensor("x_masks.pt", mask, args.full)
    print_tensor("y.pt", y, args.full)


if __name__ == "__main__":
    main()
