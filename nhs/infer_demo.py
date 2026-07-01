import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Inference demo placeholder.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--typhoon-id", required=True)
    parser.add_argument("--typhoon-name", required=True)
    parser.add_argument("--report-time", required=True)
    return parser.parse_args()


def run(sample_dir, typhoon_id, typhoon_name, report_time):
    sample_dir = Path(sample_dir)
    expected_files = ["x.npy", "x_masks.npy", "y.npy"]
    missing = [name for name in expected_files if not (sample_dir / name).exists()]

    if missing:
        raise FileNotFoundError(f"missing sample files for {typhoon_id}: {missing}")

    print(
        "[infer-demo] ready for inference: "
        f"typhoon={typhoon_id} {typhoon_name}, "
        f"report_time={report_time}, sample_dir={sample_dir}"
    )
    return 0


def main():
    args = parse_args()
    return run(args.sample_dir, args.typhoon_id, args.typhoon_name, args.report_time)


if __name__ == "__main__":
    raise SystemExit(main())
