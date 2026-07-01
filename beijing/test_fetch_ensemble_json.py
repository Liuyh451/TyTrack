import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import requests


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "debug_api_json"


def setup_logger():
    logger = logging.getLogger("TestFetchEnsembleJson")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    OUTPUT_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        OUTPUT_DIR / f"test_fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON: {path}")


def summarize_response(data: Dict[str, Any]) -> str:
    raw_data = data.get("data")
    if isinstance(raw_data, list):
        data_summary = f"list len={len(raw_data)}"
    else:
        data_summary = type(raw_data).__name__

    return (
        f"code={data.get('code')}, "
        f"success={data.get('success')}, "
        f"msg={data.get('msg')}, "
        f"data={data_summary}"
    )


def extract_xuhao_by_zone(data: Union[str, Dict[str, Any]]) -> List[str]:
    if isinstance(data, str):
        data = json.loads(data)

    xuhao_list = []
    if not data.get("success") or "data" not in data:
        return xuhao_list

    typhoons = data["data"]
    for typhoon in typhoons:
        xuhao = typhoon.get("xuhao")
        if xuhao is None:
            continue

        for _, value in typhoon.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    if item.get("zone") == "W":
                        xuhao_list.append(str(xuhao))
                        break
                else:
                    continue
                break

    return xuhao_list


def fetch_json(session: requests.Session, url: str, headers: Dict[str, str], params=None):
    response = session.get(url, headers=headers, params=params, timeout=30)
    try:
        data = response.json()
    except Exception:
        data = {
            "_http_status": response.status_code,
            "_text": response.text,
        }

    if isinstance(data, dict):
        data["_http_status"] = response.status_code
        data["_request_url"] = response.url

    return data


def main():
    from key import generate_access_key

    user_id = "1000004"
    security_key = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV"
    fcst_type = "FENSENS"
    target_times = [
        "2026-06-21 12:00:00",
        "2026-06-22 12:00:00",
    ]

    access_key = generate_access_key(user_id, security_key)
    headers = {"typhoon-access-key": access_key}

    active_url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getActiveTyphoon"
    ensemble_url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getTcEnsembleForecast"

    session = requests.Session()

    logger.info("Fetching active typhoons")
    active_data = fetch_json(session, active_url, headers)
    save_json(OUTPUT_DIR / "active_typhoon.json", active_data)

    if isinstance(active_data, dict):
        logger.info(f"Active response: {summarize_response(active_data)}")

    typhoon_codes = extract_xuhao_by_zone(active_data if isinstance(active_data, dict) else {})
    logger.info(f"Typhoon codes from active API: {typhoon_codes}")

    if not typhoon_codes:
        logger.warning("Active API returned no W-zone typhoon codes; ensemble requests will be skipped.")
        return

    for data_time in target_times:
        time_dir = OUTPUT_DIR / data_time.replace("-", "").replace(":", "").replace(" ", "_")

        for typhoon_code in typhoon_codes:
            params = {
                "typhoonCode": str(typhoon_code),
                "fcstType": str(fcst_type),
                "dataTime": str(data_time),
            }

            logger.info(f"Fetching ensemble: {params}")
            ensemble_data = fetch_json(session, ensemble_url, headers, params=params)

            filename = f"ensemble_{typhoon_code}_{fcst_type}_{data_time.replace('-', '').replace(':', '').replace(' ', '')}.json"
            save_json(time_dir / filename, ensemble_data)

            if isinstance(ensemble_data, dict):
                logger.info(f"Ensemble response: {summarize_response(ensemble_data)}")


if __name__ == "__main__":
    main()
