import os

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import torch

from key import generate_access_key


SUBJECTIVE_INSTITUTIONS = [
    "韩国",
    "中国台湾",
    "中国香港",
    "菲律宾",
    "南海所",
    "中国",
    "日本",
    "美国",
]

SUBJECTIVE_FCST_CODES = {
    "韩国": ("RKSLWTKO",),
    "中国台湾": ("FENGQING",),
    "中国香港": ("VHHHWTSS",),
    "菲律宾": ("RPMMWTPH",),
    "南海所": (),
    "中国": ("BABJWTPQ",),
    "日本": ("RJTDSUBJ",),
    "美国": ("PGTWSUBJ",),
}

SUBJECTIVE_NAME_KEYWORDS = {
    "韩国": ("韩国",),
    "中国台湾": ("台湾",),
    "中国香港": ("香港",),
    "菲律宾": ("菲律宾",),
    "南海所": ("南海",),
    "中国": ("中央台", "中国主观", "中央气象台"),
    "日本": ("日本",),
    "美国": ("美国",),
}

SUBJECTIVE_FORECAST_URL = (
    "http://10.40.168.50:28000/"
    "cmes-typhoonOcean-internal/api/tcRealtime/getTyphoonByParams"
)
SINGLE_SUBJECTIVE_FORECAST_URL = (
    "http://10.40.168.50:28000/"
    "cmes-typhoonOcean/api/tcRealtime/getTyphoonInfoByTypeAndTime/"
)


def fetch_typhoon_data(url):
    """Fetch typhoon data from the given URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def one_hot_encode(source, sources_list):
    """Generate one-hot encoding for source."""
    return [1 if source == s else 0 for s in sources_list]


def calculate_mean(column, default_value):
    """Calculate mean of a column with NaN handling."""
    non_nan_values = column[~np.isnan(column)]
    if len(non_nan_values) > 0:
        return np.mean(non_nan_values)
    return default_value


def to_utc_timestamp(time_str):
    if pd.isna(time_str) or time_str is None:
        return np.nan
    try:
        # 解析时间字符串
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))

        # 如果没有时区信息，默认它是中国时间（UTC+8）
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))  # 设置为北京时间

        # 转换为 UTC 时间：减去 8 小时
        utc_time = dt.astimezone(timezone.utc)

        # 返回 Unix 时间戳（以秒为单位）
        return int(utc_time.timestamp())
    except ValueError:
        return np.nan


def _clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _to_float(value):
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _item_fcst_type(item):
    for key in ("FCSTType", "fcstType", "code"):
        value = _clean_text(item.get(key))
        if value:
            return value.upper()
    return ""


def _item_text(item):
    keys = ("FCSTType", "fcstType", "code", "cn", "chineseName", "name")
    return " ".join(_clean_text(item.get(key)) for key in keys)


def _item_matches_institution(item, institution):
    fcst_type = _item_fcst_type(item)
    if fcst_type in SUBJECTIVE_FCST_CODES.get(institution, ()):
        return True

    text = _item_text(item)
    return any(keyword in text for keyword in SUBJECTIVE_NAME_KEYWORDS.get(institution, ()))


def _records_from_forecast_item(item):
    fcst_type = _item_fcst_type(item)
    preferred_keys = [fcst_type, _clean_text(item.get("code")).upper()]

    for key in preferred_keys:
        if key and isinstance(item.get(key), list):
            return item[key]

    for key, value in item.items():
        if str(key).endswith("_old"):
            continue
        if not isinstance(value, list):
            continue
        if not value or not isinstance(value[0], dict):
            continue
        if any(field in value[0] for field in ("fcsthour", "lat", "lon", "validtime")):
            return value

    return []


def _item_time_value(item):
    item_time = _to_float(item.get("time"))
    if not np.isnan(item_time):
        return item_time

    records = _records_from_forecast_item(item)
    if records:
        ts = to_utc_timestamp(records[0].get("datetime"))
        if not np.isnan(ts):
            return ts

    return float("-inf")


def _find_institution_forecast_item(data_list, institution):
    matches = [
        item for item in data_list
        if isinstance(item, dict) and _item_matches_institution(item, institution)
    ]
    if not matches:
        return None
    return max(matches, key=_item_time_value)


def _sort_and_dedupe_records(records):
    sortable_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        fcsthour = _to_float(record.get("fcsthour"))
        if np.isnan(fcsthour):
            continue
        sortable_records.append((fcsthour, record))

    sortable_records.sort(key=lambda item: item[0])

    unique = {}
    for fcsthour, record in sortable_records:
        if fcsthour not in unique:
            unique[fcsthour] = record

    return list(unique.values())


def _records_to_track_rows(records):
    rows = []
    for record in records:
        rows.append({
            "time": record.get("datetime"),
            "pre_time": _to_float(record.get("fcsthour")),
            "lat_model": _to_float(record.get("lat")),
            "lng_model": _to_float(record.get("lon")),
            "pressure_model": _to_float(record.get("pressure")),
            "speed_model": _to_float(record.get("windv", record.get("movespeed"))),
        })
    return rows


def build_subjective_tracks(response_json, institutions=None):
    institutions = list(institutions or SUBJECTIVE_INSTITUTIONS)
    data_list = response_json.get("data", [])
    if not isinstance(data_list, list):
        data_list = []

    typhoon_tracks = {inst: [] for inst in institutions}
    selected_fcst_types = {}

    for institution in institutions:
        item = _find_institution_forecast_item(data_list, institution)
        if item is None:
            selected_fcst_types[institution] = None
            continue

        records = _sort_and_dedupe_records(_records_from_forecast_item(item))
        typhoon_tracks[institution] = _records_to_track_rows(records)
        selected_fcst_types[institution] = _item_fcst_type(item)

    return typhoon_tracks, selected_fcst_types


def _normalize_report_time(report_time):
    text = _clean_text(report_time)
    if not text:
        return ""
    if "," in text:
        text = text.split(",")[-1].strip()
    if len(text) == 14 and text.isdigit():
        return (
            f"{text[0:4]}-{text[4:6]}-{text[6:8]} "
            f"{text[8:10]}:{text[10:12]}:{text[12:14]}"
        )
    return text


def _filter_records_by_issue_time(records, report_time=None):
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return []

    normalized_report_time = _normalize_report_time(report_time)
    if normalized_report_time:
        return [
            record for record in records
            if _normalize_report_time(record.get("datetime")) == normalized_report_time
        ]

    issue_times = [
        _normalize_report_time(record.get("datetime"))
        for record in records
        if _normalize_report_time(record.get("datetime"))
    ]
    if not issue_times:
        return records

    latest_issue_time = max(issue_times, key=to_utc_timestamp)
    return [
        record for record in records
        if _normalize_report_time(record.get("datetime")) == latest_issue_time
    ]


def _records_from_single_center_json(response_json, fcst_code):
    data_list = response_json.get("data", [])
    if not isinstance(data_list, list):
        return []

    fcst_code = _clean_text(fcst_code).upper()
    records = []

    for item in data_list:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get(fcst_code), list):
            records.extend(item[fcst_code])
            continue
        if _item_fcst_type(item) == fcst_code:
            records.extend(_records_from_forecast_item(item))

    return records


def build_subjective_tracks_from_single_responses(response_by_fcst_code, report_time=None, institutions=None):
    institutions = list(institutions or SUBJECTIVE_INSTITUTIONS)
    typhoon_tracks = {inst: [] for inst in institutions}
    selected_fcst_types = {}

    for institution in institutions:
        codes = SUBJECTIVE_FCST_CODES.get(institution, ())
        if not codes:
            selected_fcst_types[institution] = None
            continue

        fcst_code = codes[0]
        response_json = (
            response_by_fcst_code.get(fcst_code)
            or response_by_fcst_code.get(institution)
        )
        if not response_json:
            selected_fcst_types[institution] = fcst_code
            continue

        records = _records_from_single_center_json(response_json, fcst_code)
        records = _filter_records_by_issue_time(records, report_time)
        records = _sort_and_dedupe_records(records)
        typhoon_tracks[institution] = _records_to_track_rows(records)
        selected_fcst_types[institution] = fcst_code

    return typhoon_tracks, selected_fcst_types


def _tracks_to_tensors(typhoon_tracks, selected_fcst_types, institutions=None):
    institutions = list(institutions or SUBJECTIVE_INSTITUTIONS)
    max_timesteps = max(1, max((len(records) for records in typhoon_tracks.values()), default=0))

    X, mask = process_data(
        typhoon_tracks,
        max_timesteps,
        list(range(max_timesteps)),
        institutions,
    )
    X = X[np.newaxis, ...]
    mask = mask[np.newaxis, ...]
    y = _build_y_from_x_mask(X, mask)
    return X, mask, y, selected_fcst_types


def parse_single_subjective_jsons_to_fixed_institutions(response_by_fcst_code, report_time=None, institutions=None):
    institutions = list(institutions or SUBJECTIVE_INSTITUTIONS)
    typhoon_tracks, selected_fcst_types = build_subjective_tracks_from_single_responses(
        response_by_fcst_code,
        report_time=report_time,
        institutions=institutions,
    )
    return _tracks_to_tensors(typhoon_tracks, selected_fcst_types, institutions)


def process_subjective_single_api_data(response_by_fcst_code, report_time=None, institutions=None):
    X, mask, y = parse_subjective_to_fixed_institutions(
        response_by_fcst_code,
        report_time=report_time,
    )
    selected_fcst_types = {
        inst: (SUBJECTIVE_FCST_CODES.get(inst, (None,))[0] if SUBJECTIVE_FCST_CODES.get(inst) else None)
        for inst in (institutions or SUBJECTIVE_INSTITUTIONS)
    }
    return X, mask, y, selected_fcst_types


def parse_subjective_to_fixed_institutions(response_by_fcst_code, report_time=None):
    """
    解析 8 个主观预报单接口返回，并按固定机构顺序构造 X、mask、y。
    response_by_fcst_code 的 key 使用 FCSTType，例如 FENGQING、BABJWTPQ。
    """
    institutions = SUBJECTIVE_INSTITUTIONS
    normalized_report_time = _normalize_report_time(report_time)

    records_by_model = []
    max_timesteps = 0

    for institution in institutions:
        codes = SUBJECTIVE_FCST_CODES.get(institution, ())
        fcst_code = codes[0] if codes else None
        records = []

        response_json = response_by_fcst_code.get(fcst_code) if fcst_code else None
        data_list = response_json.get("data", []) if isinstance(response_json, dict) else []
        if not isinstance(data_list, list):
            data_list = []

        for item in data_list:
            if not isinstance(item, dict):
                continue

            value = item.get(fcst_code)
            if isinstance(value, list):
                records.extend(record for record in value if isinstance(record, dict))
                continue

            for key, value in item.items():
                if str(key).endswith("_old"):
                    continue
                if key != fcst_code or not isinstance(value, list):
                    continue
                records.extend(record for record in value if isinstance(record, dict))

        if normalized_report_time:
            records = [
                record for record in records
                if _normalize_report_time(record.get("datetime")) == normalized_report_time
            ]
        elif records:
            latest_issue_time = ""
            latest_timestamp = float("-inf")
            for record in records:
                issue_time = _normalize_report_time(record.get("datetime"))
                if not issue_time:
                    continue
                timestamp = to_utc_timestamp(issue_time)
                if not np.isnan(timestamp) and timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_issue_time = issue_time

            if latest_issue_time:
                records = [
                    record for record in records
                    if _normalize_report_time(record.get("datetime")) == latest_issue_time
                ]

        sortable_records = []
        for record in records:
            fcsthour = _to_float(record.get("fcsthour"))
            if np.isnan(fcsthour):
                continue
            sortable_records.append((fcsthour, record))
        sortable_records.sort(key=lambda item: item[0])

        deduped_records = []
        seen_hours = set()
        for fcsthour, record in sortable_records:
            if fcsthour in seen_hours:
                continue
            seen_hours.add(fcsthour)
            deduped_records.append(record)

        records_by_model.append(deduped_records)
        max_timesteps = max(max_timesteps, len(deduped_records))

    if max_timesteps == 0:
        return None, None, None

    feature_list = []
    mask_list = []

    for records in records_by_model:
        features = np.zeros((max_timesteps, 6), dtype=np.float32)
        model_mask = np.zeros(max_timesteps, dtype=bool)

        for t, record in enumerate(records):
            issue_timestamp = to_utc_timestamp(record.get("datetime"))
            if np.isnan(issue_timestamp):
                issue_timestamp = 0.0

            fcsthour = _to_float(record.get("fcsthour"))
            lat = _to_float(record.get("lat"))
            lon = _to_float(record.get("lon"))
            pressure = _to_float(record.get("pressure"))
            windv = _to_float(record.get("windv", record.get("movespeed")))

            row = np.array(
                [issue_timestamp, fcsthour, lat, lon, pressure, windv],
                dtype=np.float32,
            )
            features[t] = np.nan_to_num(row, nan=0.0)
            model_mask[t] = not np.isnan(lat) and not np.isnan(lon)

        feature_list.append(features)
        mask_list.append(model_mask)

    X_features = np.stack(feature_list, axis=1)
    mask = np.stack(mask_list, axis=1)

    one_hot = np.eye(len(institutions), dtype=np.float32)
    X_onehot = np.tile(one_hot, (max_timesteps, 1, 1))
    X = np.concatenate([X_features, X_onehot], axis=-1)

    X = X[np.newaxis, ...]
    mask = mask[np.newaxis, ...]

    T = X.shape[1]
    y = np.zeros((1, T, 1, 4), dtype=np.float32)
    for t in range(T):
        valid_idx = np.where(mask[0, t, :])[0]
        if len(valid_idx) > 0:
            y[0, t, 0, 3] = X[0, t, valid_idx[0], 1]

    return X, mask, y


def _build_y_from_x_mask(X, mask):
    B, T, _, _ = X.shape
    y = np.zeros((B, T, 1, 4), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            valid_idx = np.where(mask[b, t, :])[0]
            if len(valid_idx) > 0:
                y[b, t, 0, 3] = X[b, t, valid_idx[0], 1]
    return y


def parse_subjective_json_to_fixed_institutions(response_json, institutions=None):
    institutions = list(institutions or SUBJECTIVE_INSTITUTIONS)
    typhoon_tracks, selected_fcst_types = build_subjective_tracks(response_json, institutions)
    return _tracks_to_tensors(typhoon_tracks, selected_fcst_types, institutions)


def save_subjective_tensors(response_json, save_dir="data", institutions=None):
    X, mask, y, selected_fcst_types = parse_subjective_json_to_fixed_institutions(
        response_json,
        institutions,
    )

    os.makedirs(save_dir, exist_ok=True)
    torch.save(torch.tensor(X), os.path.join(save_dir, "x_results.pt"))
    torch.save(torch.tensor(mask), os.path.join(save_dir, "x_masks.pt"))
    torch.save(torch.tensor(y), os.path.join(save_dir, "y.pt"))

    print("selected fcst types:", selected_fcst_types)
    print("results", torch.tensor(X).shape)
    print("mask", torch.tensor(mask).shape)
    print("y", torch.tensor(y).shape)
    return y, X, mask, selected_fcst_types


def save_subjective_tensors_from_single_responses(
        response_by_fcst_code,
        report_time=None,
        save_dir="data",
        institutions=None):
    X, mask, y, selected_fcst_types = process_subjective_single_api_data(
        response_by_fcst_code,
        report_time=report_time,
        institutions=institutions,
    )
    if X is None:
        raise ValueError("No valid subjective forecast records found.")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(torch.tensor(X), os.path.join(save_dir, "x_results.pt"))
    torch.save(torch.tensor(mask), os.path.join(save_dir, "x_masks.pt"))
    torch.save(torch.tensor(y), os.path.join(save_dir, "y.pt"))

    print("selected fcst types:", selected_fcst_types)
    print("results", torch.tensor(X).shape)
    print("mask", torch.tensor(mask).shape)
    print("y", torch.tensor(y).shape)
    return y, X, mask, selected_fcst_types


def get_subjective_inst_data(querytyphoon_by_code, data_time=None, fcst_type=None, save_dir="data"):
    user_id = "1000004"
    security_key = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV"
    access_key = generate_access_key(user_id, security_key)

    headers = {"typhoon-access-key": access_key}
    allowed_codes = None
    if fcst_type:
        allowed_codes = {
            code.strip().upper()
            for code in str(fcst_type).split(",")
            if code.strip()
        }

    response_by_fcst_code = {}
    for institution in SUBJECTIVE_INSTITUTIONS:
        codes = SUBJECTIVE_FCST_CODES.get(institution, ())
        if not codes:
            continue

        fcst_code = codes[0]
        if allowed_codes is not None and fcst_code not in allowed_codes:
            continue

        payload = {
            "xuhao": querytyphoon_by_code,
            "fcstType": fcst_code,
        }
        if data_time:
            payload["time"] = data_time

        try:
            response = requests.request(
                "GET",
                SINGLE_SUBJECTIVE_FORECAST_URL,
                headers=headers,
                data=payload,
                timeout=30,
            )
            response.raise_for_status()
            response_by_fcst_code[fcst_code] = response.json()
        except Exception as exc:
            print(f"fetch {institution}({fcst_code}) failed: {exc}")

    return save_subjective_tensors_from_single_responses(
        response_by_fcst_code,
        report_time=data_time,
        save_dir=save_dir,
    )


def process_data(typhoon_tracks, max_timesteps, list_timestep1, orglist):
    # 第一步：确保所有机构都有默认 DataFrame
    filled_typhoon_tracks = {}
    for inst in orglist:
        if inst not in typhoon_tracks or not typhoon_tracks[inst]:
            # 创建默认 DataFrame
            default_df = pd.DataFrame({
                'time': [None] * max_timesteps,
                'pre_time': [None] * max_timesteps,
                'lat_model': [np.nan] * max_timesteps,
                'lng_model': [np.nan] * max_timesteps,
                'pressure_model': [np.nan] * max_timesteps,
                'speed_model': [np.nan] * max_timesteps
            })
            filled_typhoon_tracks[inst] = default_df
        else:
            if isinstance(typhoon_tracks[inst], list):
                filled_typhoon_tracks[inst] = pd.DataFrame(typhoon_tracks[inst])
            else:
                filled_typhoon_tracks[inst] = typhoon_tracks[inst].copy()

    typhoon_tracks = filled_typhoon_tracks

    input_sequences = []
    num_institutions = len(orglist)
    one_hot = np.eye(num_institutions)
    institution_to_index = {inst: idx for idx, inst in enumerate(orglist)}

    # 掩码：标记每个模型在每个时间步是否有有效数据
    mask = np.zeros((num_institutions, max_timesteps), dtype=bool)

    for model_idx, (model_name, trackdf) in enumerate(typhoon_tracks.items()):
        if isinstance(trackdf, list):
            track = pd.DataFrame(trackdf)
        else:
            track = trackdf.copy()

        # 转换 time 字段为 UTC 时间戳
        def to_utc_timestamp(time_str):
            if pd.isna(time_str) or time_str is None:
                return np.nan
            try:
                # 解析时间字符串
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))

                # 如果没有时区信息，默认它是中国时间（UTC+8）
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))  # 设置为北京时间

                # 转换为 UTC 时间：减去 8 小时
                utc_time = dt.astimezone(timezone.utc)

                # 返回 Unix 时间戳（以秒为单位）
                return int(utc_time.timestamp())
            except ValueError:
                return np.nan

        track['time'] = track['time'].apply(to_utc_timestamp)

        # pre_time 转换为 float
        track['pre_time'] = track['pre_time'].astype(float)
        track = track.drop_duplicates(subset=['pre_time'], keep='first').reset_index(drop=True)

        # 获取当前 track 的长度
        current_length = len(track)

        # 如果当前长度小于 max_timesteps，则补充 NaN 行
        if current_length < max_timesteps:
            nan_row = pd.DataFrame(
                [{col: np.nan for col in track.columns} for _ in range(max_timesteps - current_length)])
            padded_track = pd.concat([track, nan_row], ignore_index=True)
        else:
            padded_track = track.copy()

        # 更新 mask：经纬度都存在才视为有效
        valid = (
            ~padded_track['lat_model'].isna()
            & ~padded_track['lng_model'].isna()
        ).values
        mask[model_idx, :] = valid

        # 添加 one-hot 编码
        inst_idx = institution_to_index[model_name]
        one_hot_encoding = one_hot[inst_idx]
        feature_columns = ['time', 'pre_time', 'lat_model', 'lng_model', 'pressure_model', 'speed_model']
        features = padded_track[feature_columns].values

        track_with_id = np.hstack([
            features,
            np.tile(one_hot_encoding, (max_timesteps, 1))
        ])

        input_sequences.append(track_with_id)

    # 构造 full_input
    feature_dim = input_sequences[0].shape[1]
    full_input = np.zeros((len(orglist), max_timesteps, feature_dim), dtype=np.float32)
    for i, seq in enumerate(input_sequences):
        full_input[i, :, :] = seq.astype(np.float32)

    X = full_input.transpose(1, 0, 2)  # shape: (timesteps, models, features+id)
    mask = mask.T  # shape: (timesteps, models)

    # 处理 NaN
    new_arrays = []
    new_masks = []
    for array, mask_row in zip(X, mask):
        array = array.astype(float)
        if np.isnan(array).any():
            df = pd.DataFrame(array)
            for col in range(df.shape[1]):
                df[col] = df[col].fillna(0.0)
            array = df.values
        new_arrays.append(array)
        new_masks.append(mask_row)

    return np.array(new_arrays), np.stack(new_masks, axis=0)


def downsample_data(data, pool):
    time_steps, channels, height, width = data.shape
    data = data.reshape(time_steps * channels, 1, height, width)
    data = pool(data)  # 下采样
    data = data.reshape(time_steps, channels, data.shape[2], data.shape[3])  # 恢复维度
    return data

def get_inst_info(data, source, max_timesteps,start_time,typhoon_tracks_all,typhoon_tracks,y):
    babj_data = []
    key = source
    for typhoon_info in data['data']:  # 遍历 data 列表中的每个台风信息
        forecasts = typhoon_info[key]
        if not forecasts:
            continue  # 当前机构无预报数据，跳过
        if key in typhoon_info:
            forecasts = typhoon_info[key]  # 获取 BABJWTPQ 对应的预报数据列表

            babj_data.extend(forecasts)  # 添加到最终结果中
    max_timesteps = max(max_timesteps, len(babj_data))
    # 当前机构有预报数据，标记为有效
    has_valid_forecast = True
    for record in babj_data:
        #todo 时间点
        report_time_str = record['datetime'],
        if report_time_str!=start_time:
             continue
        typhoon_tracks[source].append({
            'time': record['datetime'],
            'pre_time': record['fcsthour'],
            'lat_model': record['lat'],
            'lng_model': record['lon'],
            'pressure_model': None,
            'speed_model': None
        })
        if source == 'BABJWTPQ':
                time = record['datetime']
                true_time = to_utc_timestamp(time)
                y.append({
                    'time': true_time,
                    'lat_model': record['lat'],
                    'lng_model': record['lon'],
                    'pre_time': record['fcsthour']
                })
    # 将当前起报时间点的数据加入最终列表
    typhoon_tracks_all.append(typhoon_tracks)
    # 打印提取后的数据长度和示例
    print(f"共提取 {len(babj_data)} 条 BABJWTPQ 数据")
    print(typhoon_tracks)
    return max_timesteps

def get_inst_data(start_time):
    # 示例使用
    user_id = "1000004"
    security_key = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV"

    access_key = generate_access_key(user_id, security_key)
    print(type(access_key))
    print("typhoon-access-key:", access_key)
    latest_xuhao = get_activate_typ()
    xuhao = latest_xuhao
    time = start_time
    typhoon_tracks_all = []
    y = []
    # 初始化 max_timesteps 为 0
    max_timesteps = 0
    #中国，日本，ECMFJSXX
    institutions = ['韩国', '中国台湾', '中国香港', '菲律宾', '南海所', '中国', '日本', '美国']
    typhoon_tracks = {inst: [] for inst in institutions}
    url_2 = 'http://10.40.168.50:28000/cmes-typhoonOcean/api/tcRealtime/getTyphoonInfoByTypeAndTime/'
    fcstType = ['RKSLWTKO', 'FENGQING', '中国香港', '菲律宾', '南海所', 'BABJWTPQ', 'RJTDSUBJ', 'PGTWSUBJ']
    for i in fcstType:
        payload = {
            "xuhao": xuhao,
            "fcstType": i,
            "time": time
        }
        headers = {"typhoon-access-key": access_key}
        if i not in ['RKSLWTKO', 'BABJWTPQ', 'RJTDSUBJ', 'PGTWSUBJ']:
           continue
        else:
            response = requests.request("GET", url_2, headers=headers, data=payload)
            print(response.text)
            data = response.json()
            # 假设 data 是已经加载好的 JSON 数据
            key = i
            max_timesteps = get_inst_info(data, key, max_timesteps, start_time, typhoon_tracks_all, typhoon_tracks, y)
            max_timesteps = max(1, max_timesteps)
    # 处理每个机构
    results = []
    masks = []
    list_timestep1 = list(range(max_timesteps))  # 动态生成 pre_time 列表
    for typhoon_tracks in typhoon_tracks_all:
        result, mask = process_data(typhoon_tracks, max_timesteps, list_timestep1, institutions)
        results.append(result)
        masks.append(mask)
    # 最终结果：(样本数, max_timesteps, num_models, feature_dim + num_models)
    for i, result in enumerate(results):
        print(f"Result {i} shape:", result.shape)

    final_result = np.stack(results, axis=0)
    tensor_results = [torch.tensor(result) for result in results]
    tensor_masks = [torch.tensor(mask) for mask in masks]
    stacked_tensor_results = torch.stack(tensor_results, dim=0)
    stacked_tensor_masks = torch.stack(tensor_masks, dim=0)
    print("results",stacked_tensor_results.shape)#([1, 5, 8, 14])
    print("mask",stacked_tensor_masks.shape)#([1, 5, 8])
    y = np.array([[d['time'], d['lat_model'], d['lng_model'], d['pre_time']] for d in y], dtype=np.float32)
    y = np.stack(y, axis=0)
    print("y----------------",y.shape)
    y = torch.tensor(y)
    torch.save(y, 'data/y.pt')
    # 确保 max_timesteps 至少为 1
    torch.save(stacked_tensor_results, 'data/x_results.pt')
    torch.save(stacked_tensor_masks, 'data/x_masks.pt')
    print("Data processing complete and saved as .pt files.")
    return y, stacked_tensor_results, stacked_tensor_masks
def get_activate_typ():

    url = "http://10.40.168.50:28000/cmes-typhoonOcean/api/tcRealtime/getActiveTyphoon"

    payload = {}
    headers = {}
    user_id = "1000004"
    security_key = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV"

    access_key = generate_access_key(user_id, security_key)
    headers = {"typhoon-access-key": access_key}
    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 200:
        from datetime import datetime

        if response.status_code == 200:
            try:
                data = response.json()
                typhoon_data_list = data.get('data', [])

                if not typhoon_data_list:
                    print("没有台风数据")
                    return None

                all_typhoons = []

                for item in typhoon_data_list:
                    babj_list = item.get('BABJWTPQ', [])
                    for ty in babj_list:
                        if 'datetime' in ty and 'xuhao' in ty:
                            all_typhoons.append(ty)

                if not all_typhoons:
                    print("BABJWTPQ 中没有有效台风记录")
                    return None

                # 找到 datetime 最新的台风记录
                latest_typhoon = max(
                    all_typhoons,
                    key=lambda x: datetime.fromisoformat(x['datetime'].replace(" ", "T"))
                )

                print("最新台风信息:", latest_typhoon)
                return latest_typhoon['xuhao']

            except Exception as e:
                print("解析失败:", e)
                return None
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return None


