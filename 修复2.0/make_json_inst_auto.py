import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Union, Dict, Any

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("TyphoonProcessor")
    logger.setLevel(logging.INFO)

    # 閬垮厤閲嶅娣诲姞 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 鏂囦欢 handler
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 鎺у埗鍙?handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# =====================================================
# 宸ュ叿鍑芥暟
# =====================================================
def to_utc_timestamp(time_str):
    if pd.isna(time_str) or time_str is None:
        return np.nan

    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))

        utc_time = dt.astimezone(timezone.utc)

        return int(utc_time.timestamp())

    except Exception:
        return np.nan


# =====================================================
# 解析集合预报
# =====================================================
def parse_ensemble_to_fixed_models(response_json, max_models=8):
    data_list = response_json.get("data", [])

    if not data_list:
        logger.warning("No data found.")
        return None, None, None

    # 提取所有集合成员
    members = []

    for member_obj in data_list:
        for key, value in member_obj.items():
            if not key.isdigit():
                continue
            if not isinstance(value, list):
                continue
            if len(value) == 0:
                continue

            value = sorted(value, key=lambda x: x.get("fcsthour", 0))

            unique = {}
            for rec in value:
                hour = rec.get("fcsthour")
                if hour is None:
                    continue
                if hour not in unique:
                    unique[hour] = rec

            member_records = list(unique.values())

            if len(member_records) > 0:
                members.append(member_records)

    logger.info(f"原始成员数: {len(members)}")

    # 过滤无效成员
    valid_members = []

    for recs in members:
        has_valid = any(
            r.get("lat") is not None and r.get("lon") is not None for r in recs
        )
        if has_valid:
            valid_members.append(recs)

    logger.info(f"有效成员数: {len(valid_members)}")

    selected_members = valid_members[:max_models]
    num_models = len(selected_members)
    logger.info(f"实际使用成员数: {num_models}")

    padded_members = selected_members + [None] * (max_models - num_models)

    max_timesteps = (
        max(len(recs) for recs in selected_members) if selected_members else 0
    )
    logger.info(f"时间步长: {max_timesteps}")

    if max_timesteps == 0:
        logger.error("没有有效的时间步数据，返回 None")
        return None, None, None

    # 起报时间
    first_record = selected_members[0][0]
    base_dt_str = first_record.get("datetime")
    base_timestamp = to_utc_timestamp(base_dt_str) if base_dt_str else 0

    feature_list = []
    mask_list = []

    for recs in padded_members:
        if recs is None:
            features = np.zeros((max_timesteps, 6), dtype=np.float32)
            mask = np.zeros(max_timesteps, dtype=bool)
        else:
            df = pd.DataFrame(recs)
            df["time"] = base_timestamp
            df["pre_time"] = df["fcsthour"].astype(float)
            df["lat_model"] = df["lat"].astype(float)
            df["lng_model"] = df["lon"].astype(float)
            df["pressure_model"] = pd.to_numeric(df["pressure"], errors="coerce")
            df["speed_model"] = pd.to_numeric(df["windv"], errors="coerce")
            features = df[
                [
                    "time",
                    "pre_time",
                    "lat_model",
                    "lng_model",
                    "pressure_model",
                    "speed_model",
                ]
            ].values
            cur_len = len(features)
            if cur_len < max_timesteps:
                pad = np.zeros((max_timesteps - cur_len, 6), dtype=np.float32)
                features = np.vstack([features, pad])
            mask = np.zeros(max_timesteps, dtype=bool)
            mask[:cur_len] = True
            features = np.nan_to_num(features, nan=0.0)

        feature_list.append(features)
        mask_list.append(mask)

    X_features = np.stack(feature_list, axis=1)  # (T, M, 6)
    mask = np.stack(mask_list, axis=1)  # (T, M)

    one_hot = np.eye(max_models, dtype=np.float32)
    X_onehot = np.tile(one_hot, (max_timesteps, 1, 1))
    X = np.concatenate([X_features, X_onehot], axis=-1)  # (T, M, 14)

    X = X[np.newaxis, ...]  # (1, T, M, 14)
    mask = mask[np.newaxis, ...]  # (1, T, M)

    # 构造 y
    T = X.shape[1]
    y = np.zeros((1, T, 1, 4), dtype=np.float32)
    for t in range(T):
        valid_idx = np.where(mask[0, t, :])[0]
        if len(valid_idx) > 0:
            lead_time = X[0, t, valid_idx[0], 1]
            y[0, t, 0, 3] = lead_time

    return X, mask, y


def extract_xuhao_by_zone(data: Union[str, Dict[str, Any]]) -> List[int]:
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

        for key, value in typhoon.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    if item.get("zone") == "W":
                        xuhao_list.append(str(xuhao))
                        break
                else:
                    continue
                break
    return xuhao_list


# ========== 涓诲嚱鏁?==========
def parse_latest_ensemble_time(response_json: Dict[str, Any]) -> Optional[str]:
    if response_json.get("code") != 200:
        return None

    data = response_json.get("data")
    if not isinstance(data, dict):
        return None

    data_times = data.get("dataTime", [])
    if not isinstance(data_times, list) or not data_times:
        return None

    valid_times = []
    for time_str in data_times:
        if not isinstance(time_str, str):
            continue
        try:
            datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        valid_times.append(time_str)

    return max(valid_times) if valid_times else None


def fetch_latest_ensemble_time(
    headers, typhoon_code: str, fcst_type: str
) -> Optional[str]:
    url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/listTcEnsembleTime"
    params = {"typhoonCode": str(typhoon_code), "fcstType": str(fcst_type)}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error(f"获取集合预报时间失败: {resp.status_code}")
            return None

        data = resp.json()
        latest_time = parse_latest_ensemble_time(data)
        if latest_time is None:
            logger.warning(
                f"集合预报时间列表为空或不可解析: typhoonCode={typhoon_code}, "
                f"fcstType={fcst_type}, code={data.get('code')}, msg={data.get('msg')}"
                f"rawData={data}"
            )
            return None

        logger.info(
            f"最新集合预报时间: typhoonCode={typhoon_code}, "
            f"fcstType={fcst_type}, dataTime={latest_time}"
        )
        return latest_time

    except Exception as e:
        logger.error(f"请求集合预报时间异常: {e}")
        return None


def fetch_and_process(
    data_time: str,
    fcst_type: str = "FENSENS",
    access_key: str = None,
    user_id: str = "1000004",
    security_key: str = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV",
):
    if access_key is None:
        from key import generate_access_key

        access_key = generate_access_key(user_id, security_key)

    headers = {"typhoon-access-key": access_key}

    # 获取台风编号列表
    url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getActiveTyphoon"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"获取台风编号列表失败: {e}")
        return []
    data = response.json()
    # 根据区域筛选活跃台风编号
    xuhao_list = extract_xuhao_by_zone(data)
    logger.info(f"待处理的台风列表: {xuhao_list}")
    if not xuhao_list:
        logger.info("没有符合条件的台风需要处理")
        return []

    url_ens = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getTcEnsembleForecast"
    success_codes = []

    for typhoon_code in xuhao_list:
        logger.info(f"\n正在处理台风编号: {typhoon_code}")
        # 获取最新集合预报时间
        latest_data_time = fetch_latest_ensemble_time(headers, typhoon_code, fcst_type)
        if latest_data_time is None:
            logger.warning(f"台风 {typhoon_code} 未获取到最新集合预报时间，舍弃")
            continue
        safe_time = latest_data_time.replace(" ", "-").replace(":", "-")
        logger.info(f"安全目录时间: {safe_time}")
        save_dir = os.path.join(BASE_DIR, "data", str(typhoon_code), safe_time)
        # save_dir = f"data/{typhoon_code}/{safe_time}"
        expected_files = ["x_results.pt", "x_masks.pt", "y.pt"]
        if all(os.path.exists(os.path.join(save_dir, name)) for name in expected_files):
            logger.info(f"台风 {typhoon_code} 时次 {latest_data_time} 已处理过，跳过")
            # success_codes.append(typhoon_code)
            continue

        params = {
            "typhoonCode": typhoon_code,
            "fcstType": fcst_type,
            "dataTime": latest_data_time,
        }
        try:
            resp = requests.get(url_ens, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                logger.error(f"请求失败: {resp.status_code}")
                continue
            data = resp.json()
            if data.get("code") != 200:
                logger.error(f"API错误: {data.get('msg')}")
                continue
        except Exception as e:
            logger.error(f"请求集合预报数据异常: {e}")
            continue

        X, mask, y = parse_ensemble_to_fixed_models(data, max_models=8)
        if X is None:
            logger.warning(
                f"台风 {typhoon_code} 数据解析失败，X is None，跳过，rawData={data}"
            )
            continue

        logger.info(f"X shape: {X.shape}")
        logger.info(f"mask shape: {mask.shape}")
        logger.info(f"y shape: {y.shape}")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(torch.tensor(X), f"{save_dir}/x_results.pt")
        torch.save(torch.tensor(mask), f"{save_dir}/x_masks.pt")
        torch.save(torch.tensor(y), f"{save_dir}/y.pt")
        logger.info(f"台风 {typhoon_code} 数据保存成功 -> {save_dir}")
        logger.info(f"台风 {typhoon_code} 数据保存成功 -> {os.path.abspath(save_dir)}")

        # success_codes.append(typhoon_code)
        # 将 success_codes.append 的内容改为 safe_time：
        success_codes.append((typhoon_code, safe_time))

    logger.info(f"\n处理完成，成功保存的台风: {success_codes}")
    return success_codes


if __name__ == "__main__":
    start_time = "2026-06-04 00:00:00"
    result = fetch_and_process(start_time, fcst_type="FENSENS")
    logger.info(f"最终结果: {result}")
