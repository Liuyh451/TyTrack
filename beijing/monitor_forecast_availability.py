import argparse
import base64
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent
MONITOR_DIR = BASE_DIR / "monitor_logs"
STATE_PATH = MONITOR_DIR / "forecast_availability_state.json"
CHECK_LOG_PATH = MONITOR_DIR / "forecast_availability_checks.jsonl"
SUMMARY_CSV_PATH = MONITOR_DIR / "forecast_availability_first_seen.csv"

BEIJING_TZ = timezone(timedelta(hours=8))

ACTIVE_TYPHOON_URL = (
    "http://106.120.73.242/wg-cmes/"
    "cmes-typhoonocean-internal/api/tcRealtime/getActiveTyphoon"
)
SUBJECTIVE_FORECAST_URL = (
    "http://106.120.73.242/wg-cmes/"
    "cmes-typhoonocean-internal/api/tcRealtime/getTyphoonInfoByTypeAndTime"
)

INSTITUTIONS = [
    ("韩国", "RKSLWTKO"),
    ("中国台湾", "FENGQING"),
    ("中国香港", "VHHHWTSS"),
    ("菲律宾", "RPMMWTPH"),
    ("中国", "BABJWTPQ"),
    ("日本", "RJTDSUBJ"),
    ("美国", "PGTWSUBJ"),
]


def now_bj() -> datetime:
    return datetime.now(BEIJING_TZ)


def generate_access_key(user_id: str, security_key: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    value = user_id + timestamp
    md5_string = hashlib.md5((value + security_key).encode("utf-8")).hexdigest()
    signature = value + md5_string
    return base64.b64encode(signature.encode("utf-8")).decode("utf-8")


def normalize_report_time(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    if len(text) == 14 and text.isdigit():
        return (
            f"{text[0:4]}-{text[4:6]}-{text[6:8]} "
            f"{text[8:10]}:{text[10:12]}:{text[12:14]}"
        )
    return text.replace("T", " ").replace("+08:00", "").replace("Z", "")


def parse_report_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=BEIJING_TZ)


def candidate_report_times(now: datetime, lookback_cycles: int) -> List[str]:
    target_hour = (now.hour // 6) * 6
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    return [
        (target - timedelta(hours=6 * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(lookback_cycles)
    ]


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"first_seen": {}}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    MONITOR_DIR.mkdir(exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp_path.replace(STATE_PATH)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    MONITOR_DIR.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_summary_csv(row: Dict[str, Any]) -> None:
    MONITOR_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "first_seen_at",
        "report_time",
        "delay_hours",
        "cycle_hour",
        "xuhao",
        "engname",
        "tfbh",
        "available_institutions",
    ]
    exists = SUMMARY_CSV_PATH.exists()
    with SUMMARY_CSV_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def extract_nested_value(typhoon: Dict[str, Any], key: str, zone: str = "W") -> str:
    value = typhoon.get(key)
    if value not in (None, ""):
        return str(value)

    for item in typhoon.values():
        if not isinstance(item, list):
            continue
        for record in item:
            if not isinstance(record, dict):
                continue
            if zone and record.get("zone") != zone:
                continue
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def has_zone(typhoon: Dict[str, Any], zone: str = "W") -> bool:
    if typhoon.get("zone") == zone:
        return True
    for value in typhoon.values():
        if isinstance(value, list):
            for record in value:
                if isinstance(record, dict) and record.get("zone") == zone:
                    return True
    return False


def extract_active_typhoons(payload: Dict[str, Any], zone: str = "W") -> List[Dict[str, str]]:
    typhoons = payload.get("data", [])
    if not isinstance(typhoons, list):
        return []

    result = []
    for typhoon in typhoons:
        if not isinstance(typhoon, dict):
            continue
        xuhao = typhoon.get("xuhao")
        if xuhao is None:
            continue
        if not has_zone(typhoon, zone):
            continue

        result.append(
            {
                "xuhao": str(xuhao),
                "engname": str(
                    typhoon.get("engname")
                    or typhoon.get("enname")
                    or extract_nested_value(typhoon, "engname", zone)
                    or ""
                ),
                "tfbh": extract_nested_value(typhoon, "tfbh", zone),
            }
        )
    return result


def list_forecast_records(payload: Dict[str, Any], fcst_code: str) -> List[Dict[str, Any]]:
    data_list = payload.get("data", [])
    if not isinstance(data_list, list):
        return []

    records: List[Dict[str, Any]] = []
    for item in data_list:
        if not isinstance(item, dict):
            continue

        value = item.get(fcst_code)
        if isinstance(value, list):
            records.extend(record for record in value if isinstance(record, dict))
            continue

        fcst_type = str(item.get("FCSTType") or item.get("fcstType") or "").upper()
        if fcst_type == fcst_code:
            for key, nested in item.items():
                if str(key).endswith("_old") or not isinstance(nested, list):
                    continue
                if nested and isinstance(nested[0], dict):
                    records.extend(record for record in nested if isinstance(record, dict))
    return records


def has_valid_report_time(records: Iterable[Dict[str, Any]], report_time: str) -> bool:
    for record in records:
        if normalize_report_time(record.get("datetime")) != report_time:
            continue
        if record.get("fcsthour") is None:
            continue
        if record.get("lat") is None or record.get("lon") is None:
            continue
        return True
    return False


def request_json(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    try:
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        status_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None, "response json is not an object", status_code
        return payload, None, status_code
    except Exception as exc:
        return None, str(exc), 0


def check_once(args) -> None:
    check_time = now_bj()
    check_time_text = check_time.strftime("%Y-%m-%d %H:%M:%S")
    report_times = candidate_report_times(check_time, args.lookback_cycles)
    state = load_state()
    first_seen = state.setdefault("first_seen", {})

    access_key = generate_access_key(args.user_id, args.security_key)
    headers = {"typhoon-access-key": access_key}
    session = requests.Session()

    active_payload, active_error, active_status = request_json(
        session, ACTIVE_TYPHOON_URL, headers, timeout=args.timeout
    )

    active_typhoons: List[Dict[str, str]] = []
    if active_payload and active_payload.get("code") == 200:
        active_typhoons = extract_active_typhoons(active_payload, args.zone)

    check_record: Dict[str, Any] = {
        "checked_at": check_time_text,
        "active_status_code": active_status,
        "active_api_code": active_payload.get("code") if active_payload else None,
        "active_error": active_error,
        "active_count": len(active_typhoons),
        "report_times": report_times,
        "items": [],
    }

    print(f"[{check_time_text}] active_typhoons={len(active_typhoons)}")

    for typhoon in active_typhoons:
        for report_time in report_times:
            available_institutions = []
            institution_details = []

            for institution_name, fcst_code in INSTITUTIONS:
                params = {
                    "xuhao": typhoon["xuhao"],
                    "fcstType": fcst_code,
                    "dataTime": report_time,
                }
                payload, error, status = request_json(
                    session,
                    SUBJECTIVE_FORECAST_URL,
                    headers,
                    params=params,
                    timeout=args.timeout,
                )

                api_code = payload.get("code") if payload else None
                records = list_forecast_records(payload, fcst_code) if payload else []
                available = bool(payload and api_code == 200 and has_valid_report_time(records, report_time))
                if available:
                    available_institutions.append(fcst_code)

                institution_details.append(
                    {
                        "institution": institution_name,
                        "fcst_code": fcst_code,
                        "http_status": status,
                        "api_code": api_code,
                        "record_count": len(records),
                        "available": available,
                        "error": error,
                    }
                )

            is_available = bool(available_institutions)
            item_record = {
                "xuhao": typhoon["xuhao"],
                "engname": typhoon["engname"],
                "tfbh": typhoon["tfbh"],
                "report_time": report_time,
                "available": is_available,
                "available_institutions": available_institutions,
                "institution_details": institution_details,
            }
            check_record["items"].append(item_record)

            if is_available:
                state_key = f"{typhoon['xuhao']}|{report_time}"
                if state_key not in first_seen:
                    report_dt = parse_report_time(report_time)
                    delay_hours = round((check_time - report_dt).total_seconds() / 3600, 3)
                    first_seen[state_key] = {
                        "first_seen_at": check_time_text,
                        "report_time": report_time,
                        "delay_hours": delay_hours,
                        "xuhao": typhoon["xuhao"],
                        "engname": typhoon["engname"],
                        "tfbh": typhoon["tfbh"],
                        "available_institutions": available_institutions,
                    }
                    append_summary_csv(
                        {
                            "first_seen_at": check_time_text,
                            "report_time": report_time,
                            "delay_hours": delay_hours,
                            "cycle_hour": report_dt.strftime("%H"),
                            "xuhao": typhoon["xuhao"],
                            "engname": typhoon["engname"],
                            "tfbh": typhoon["tfbh"],
                            "available_institutions": "|".join(available_institutions),
                        }
                    )

                print(
                    f"  available xuhao={typhoon['xuhao']} report_time={report_time} "
                    f"institutions={','.join(available_institutions)}"
                )

    append_jsonl(CHECK_LOG_PATH, check_record)
    save_state(state)
    print(f"check log: {CHECK_LOG_PATH}")
    print(f"first-seen summary: {SUMMARY_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="监控外网台风预报数据各起报时次的可用延迟")
    parser.add_argument("--loop", action="store_true", help="持续运行，每隔 interval 秒检查一次")
    parser.add_argument("--interval", type=int, default=3600, help="循环检查间隔，默认 3600 秒")
    parser.add_argument("--lookback-cycles", type=int, default=8, help="向前检查多少个 6 小时时次")
    parser.add_argument("--timeout", type=int, default=30, help="接口超时时间，秒")
    parser.add_argument("--zone", default="W", help="只监控指定区域，默认 W")
    parser.add_argument("--user-id", default="1000004")
    parser.add_argument("--security-key", default="nN2hJQEN3v7Y7sPmDvthdvkQXapuZV")
    args = parser.parse_args()

    while True:
        check_once(args)
        if not args.loop:
            break
        print(f"sleep {args.interval} seconds...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
