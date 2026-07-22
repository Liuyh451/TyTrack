import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


EARTH_RADIUS_KM = 6371.0088
FORECAST_LEAD_HOURS = (12, 24, 36, 48, 60, 72, 96, 120)
CSV_FIELDS = (
    "report_time",
    "lead_hour",
    "valid_time",
    "predicted_lat",
    "predicted_lon",
    "actual_lat",
    "actual_lon",
    "error_km",
)


def normalize_ty_code(ty_code) -> str:
    value = str(ty_code).strip()
    if len(value) == 4 and value.isdigit():
        return f"20{value}"
    return value


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def read_real_track(path: Path) -> dict[datetime, tuple[float, float]]:
    track = {}
    if not path.exists():
        return track

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            fields = raw_line.split()
            if not fields or fields[0] == "time":
                continue
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: invalid real-track line")
            point_time = datetime.strptime(
                f"{fields[0]} {fields[1]}", "%Y-%m-%d %H:%M:%S"
            )
            track[point_time] = (float(fields[2]), float(fields[3]))
    return track


def read_npy_forecast(path: Path) -> dict[int, tuple[float, float]]:
    values = np.load(path)
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[2] < 1:
        raise ValueError(f"{path}: expected [sample, lead, channel, lat_lon]")
    if values.shape[1] > len(FORECAST_LEAD_HOURS) or values.shape[3] != 2:
        raise ValueError(f"{path}: unsupported forecast shape {values.shape}")

    predictions = values[0, :, -1, :]
    return {
        lead_hour: (float(prediction[0]), float(prediction[1]))
        for lead_hour, prediction in zip(FORECAST_LEAD_HOURS, predictions)
        if prediction[0] != 0 and prediction[1] != 0
    }


def discover_forecasts(
    npy_dir: Path,
    current_report_time: str,
    lookback: int,
) -> list[tuple[str, Path]]:
    forecasts = []
    if npy_dir.exists():
        for path in npy_dir.glob("*.npy"):
            report_time = path.stem
            if len(report_time) != 10 or not report_time.isdigit():
                continue
            if report_time < current_report_time:
                forecasts.append((report_time, path))

    forecasts.sort(key=lambda item: item[0], reverse=True)
    if lookback:
        forecasts = forecasts[:lookback]
    return forecasts


def load_csv_state(path: Path) -> tuple[set[tuple[str, str]], str | None]:
    keys = set()
    last_report_time = None
    if not path.exists():
        return keys, last_report_time

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0] or row[0] == "report_time" or len(row) < 2:
                continue
            keys.add((row[0], row[1]))
            last_report_time = row[0]
    return keys, last_report_time


def append_csv_rows(path: Path, rows: list[dict]) -> int:
    existing_keys, last_report_time = load_csv_state(path)
    new_rows = [
        row
        for row in rows
        if (row["report_time"], row["lead_hour"]) not in existing_keys
    ]
    new_rows.sort(key=lambda row: (row["report_time"], int(row["lead_hour"])))
    if not new_rows:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()

        previous_report_time = last_report_time
        for row in new_rows:
            if previous_report_time and row["report_time"] != previous_report_time:
                f.write("\n")
            writer.writerow(row)
            previous_report_time = row["report_time"]
    return len(new_rows)


def evaluate_recent_forecasts(
    ty_code,
    current_report_time: str,
    npy_root: Path,
    real_track_path: Path,
    eval_root: Path,
    lookback: int = 0,
) -> dict:
    if lookback < 0:
        raise ValueError("lookback must be zero or greater")

    normalized_ty_code = normalize_ty_code(ty_code)
    forecasts = discover_forecasts(
        Path(npy_root) / normalized_ty_code,
        current_report_time,
        lookback,
    )
    real_track = read_real_track(Path(real_track_path))
    rows = []

    for report_time, forecast_path in forecasts:
        report_dt = datetime.strptime(report_time, "%Y%m%d%H")
        for lead_hour, (predicted_lat, predicted_lon) in read_npy_forecast(
            forecast_path
        ).items():
            valid_dt = report_dt + timedelta(hours=lead_hour)
            actual = real_track.get(valid_dt)
            if actual is None:
                continue
            actual_lat, actual_lon = actual
            rows.append(
                {
                    "report_time": report_time,
                    "lead_hour": str(lead_hour),
                    "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_lat": f"{predicted_lat:.3f}",
                    "predicted_lon": f"{predicted_lon:.3f}",
                    "actual_lat": f"{actual_lat:.3f}",
                    "actual_lon": f"{actual_lon:.3f}",
                    "error_km": f"{haversine_km(predicted_lat, predicted_lon, actual_lat, actual_lon):.3f}",
                }
            )

    csv_path = Path(eval_root) / normalized_ty_code / f"mde_{normalized_ty_code}.csv"
    added_count = append_csv_rows(csv_path, rows)
    return {
        "forecast_count": len(forecasts),
        "matched_count": len(rows),
        "added_count": added_count,
        "csv_path": csv_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NPY typhoon forecasts against real-track points."
    )
    parser.add_argument("--ty-code", required=True)
    parser.add_argument("--current-report-time", required=True)
    parser.add_argument("--npy-root", type=Path, default=Path("output_npy"))
    parser.add_argument("--real-track", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=Path("eval"))
    parser.add_argument(
        "--lookback",
        type=int,
        default=0,
        help="Number of recent report times to evaluate; 0 evaluates all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_recent_forecasts(
        args.ty_code,
        args.current_report_time,
        args.npy_root,
        args.real_track,
        args.eval_root,
        args.lookback,
    )
    print(
        f"MDE evaluation: forecasts={result['forecast_count']} "
        f"matched={result['matched_count']} added={result['added_count']} "
        f"csv={result['csv_path']}"
    )


if __name__ == "__main__":
    main()
