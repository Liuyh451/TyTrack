import os
from datetime import datetime

import requests


JSON_URL = "https://cdn.oss.wushikj.com/data/typhoon/2026/202606.json"
SOURCE_NAME = "南海所"
TXT_NAME = "forecast_records.txt"


def fetch_json(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"JSON 获取失败: {exc}")
        return None


def first_value(data, keys):
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return ""


def parse_time(value):
    return datetime.fromisoformat(value)


def format_number(value, digits=2):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect_nanhai_forecasts(json_data):
    records = []

    for info in json_data.get("data", []):
        report_time = info.get("time")
        if not report_time:
            continue

        for source_data in info.get("data", []):
            if source_data.get("source") != SOURCE_NAME:
                continue

            forecasts = []
            for forecast in source_data.get("data", []):
                forecast_time = forecast.get("time")
                if not forecast_time:
                    continue

                report_dt = parse_time(report_time)
                forecast_dt = parse_time(forecast_time)
                lead_hour = (forecast_dt - report_dt).total_seconds() / 3600

                forecasts.append(
                    {
                        "lead_hour": lead_hour,
                        "forecast_time": forecast_time,
                        "lat": first_value(forecast, ["lat", "latitude"]),
                        "lon": first_value(forecast, ["lng", "lon", "longitude"]),
                        "pressure": first_value(
                            forecast,
                            ["pressure", "center_pressure", "min_pressure", "air_pressure", "pres"],
                        ),
                        "wind": first_value(
                            forecast,
                            ["speed", "wind_speed", "max_wind_speed", "wind", "vmax"],
                        ),
                    }
                )

            if forecasts:
                records.append(
                    {
                        "report_time": report_time,
                        "forecasts": sorted(forecasts, key=lambda item: item["lead_hour"]),
                    }
                )

    return sorted(records, key=lambda item: parse_time(item["report_time"]))


def write_txt(json_data, records):
    typhoon_id = str(json_data.get("tfid") or json_data.get("id") or "202606")
    typhoon_name = str(json_data.get("ename") or json_data.get("name") or typhoon_id).lower().strip()
    save_dir = os.path.join("data", "case", "wpo", typhoon_name)
    os.makedirs(save_dir, exist_ok=True)

    txt_path = os.path.join(save_dir, TXT_NAME)
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"台风编号: {typhoon_id}\n")
        f_txt.write(f"台风名称: {typhoon_name}\n")
        f_txt.write(f"预报机构: {SOURCE_NAME}\n")
        f_txt.write("=" * 80 + "\n")
        f_txt.write("起报时间\t预报时效(h)\t预报时间\t纬度\t经度\t最小气压\t最大风速\n")

        for record in records:
            for forecast in record["forecasts"]:
                f_txt.write(
                    "\t".join(
                        [
                            record["report_time"],
                            format_number(forecast["lead_hour"], 0),
                            forecast["forecast_time"],
                            format_number(forecast["lat"]),
                            format_number(forecast["lon"]),
                            str(forecast["pressure"]),
                            str(forecast["wind"]),
                        ]
                    )
                    + "\n"
                )

    return txt_path


def main():
    json_data = fetch_json(JSON_URL)
    if not json_data:
        return

    records = collect_nanhai_forecasts(json_data)
    txt_path = write_txt(json_data, records)

    print(f"处理完成: {txt_path}")
    print(f"起报时间数量: {len(records)}")
    print(f"预报记录数量: {sum(len(record['forecasts']) for record in records)}")


if __name__ == "__main__":
    main()
