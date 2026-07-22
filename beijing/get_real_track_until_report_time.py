import argparse
import gzip
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen


BEIJING_TZ = timezone(timedelta(hours=8))
DEFAULT_URL_TEMPLATE = "https://cdn.oss.wushikj.com/data/typhoon/{year}/{ty_code}.json"


def normalize_ty_code(ty_code) -> str:
    ty_code = str(ty_code).strip()
    if len(ty_code) == 4 and ty_code.isdigit():
        return f"20{ty_code}"
    return ty_code


def parse_time(value: str) -> datetime:
    """
    Parse report_time or typhoon point time as Beijing time.

    Supported examples:
    - 2026062312
    - 2026-06-23 12:00:00
    - 2026-06-23-12-00-00
    - 2026-06-23T12:00:00
    """
    raw = str(value).strip()
    normalized = raw.replace("Z", "+00:00")

    if len(raw) == 10 and raw.isdigit():
        dt = datetime.strptime(raw, "%Y%m%d%H")
    else:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d-%H-%M-%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            dt = datetime.fromisoformat(normalized)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=BEIJING_TZ)
    return dt.astimezone(BEIJING_TZ)


def format_report_time(value: str) -> str:
    return parse_time(value).strftime("%Y%m%d%H")


def load_typhoon_json(ty_code: str, source: str | None = None) -> dict:
    ty_code = normalize_ty_code(ty_code)
    if source:
        path = Path(source)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
        url = source
    else:
        year = str(ty_code)[:4]
        url = DEFAULT_URL_TEMPLATE.format(year=year, ty_code=ty_code)

    with urlopen(url, timeout=30) as response:
        content = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()
        if encoding == "gzip" or content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)
        return json.loads(content.decode("utf-8-sig"))


def extract_real_track(payload: dict, report_time: str) -> list[dict]:
    cutoff = parse_time(report_time)
    points = []

    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        if item.get("lat") is None or item.get("lng") is None or not item.get("time"):
            continue

        point_time = parse_time(item["time"])
        if point_time > cutoff:
            continue

        points.append(
            {
                "time": point_time.strftime("%Y-%m-%d %H:%M:%S"),
                "lat": float(item["lat"]),
                "lng": float(item["lng"]),
                "power": item.get("power"),
                "speed": item.get("speed"),
                "pressure": item.get("pressure"),
                "strong": item.get("strong"),
            }
        )

    return sorted(points, key=lambda point: point["time"])


def save_track_txt(points: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("time lat lng power speed pressure strong\n")
        for point in points:
            f.write(
                f"{point['time']} {point['lat']:.1f} {point['lng']:.1f} "
                f"{point['power']} {point['speed']} {point['pressure']} {point['strong']}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get real typhoon track points until the given report time."
    )
    parser.add_argument(
        "--ty_code",
        default="2611",
        help="Typhoon code, for example 202608.",
    )
    parser.add_argument(
        "--report_time",
        default="2026-07-15 08:00:00",
        help="Cutoff time in Beijing time, for example 2026-06-23 12:00:00.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional local JSON path or URL. Defaults to cdn.oss.wushikj.com.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output txt path. Defaults to output/<ty_code>/real_track_<ty_code>_<report_time>.txt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ty_code = normalize_ty_code(args.ty_code)
    payload = load_typhoon_json(ty_code, args.source)
    points = extract_real_track(payload, args.report_time)

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("output")
            / ty_code
            / f"real_track_{ty_code}_{format_report_time(args.report_time)}.txt"
        )

    save_track_txt(points, output_path)
    print(f"Saved {len(points)} real track points -> {output_path}")


if __name__ == "__main__":
    main()
