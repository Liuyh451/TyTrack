import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


INSTITUTIONS = [
    "韩国",
    "中国台湾",
    "中国香港",
    "菲律宾",
    "南海所",
    "中国",
    "日本",
    "美国",
]

FCST_CODES = {
    "韩国": "RKSLWTKO",
    "中国台湾": "FENGQING",
    "中国香港": "VHHHWTSS",
    "菲律宾": "RPMMWTPH",
    "南海所": None,
    "中国": "BABJWTPQ",
    "日本": "RJTDSUBJ",
    "美国": "PGTWSUBJ",
}


def to_utc_timestamp(time_str):
    if not time_str:
        return np.nan
    try:
        dt = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        return int(dt.astimezone(timezone.utc).timestamp())
    except (TypeError, ValueError):
        return np.nan


def to_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_report_time(report_time):
    if not report_time:
        return ""
    text = str(report_time).strip()
    if len(text) == 14 and text.isdigit():
        return (
            f"{text[0:4]}-{text[4:6]}-{text[6:8]} "
            f"{text[8:10]}:{text[10:12]}:{text[12:14]}"
        )
    return text


def infer_fcst_code(payload):
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        fcst_type = item.get("FCSTType") or item.get("fcstType")
        if fcst_type:
            return str(fcst_type).upper()
        for key, value in item.items():
            if key.endswith("_old"):
                continue
            if isinstance(value, list):
                return str(key).upper()
    raise ValueError("Cannot infer FCSTType from JSON.")


def parse_subjective_to_fixed_institutions(response_by_fcst_code, report_time=None):
    """
    解析不同发报中心（机构）的台风预报数据，并将其转化为固定维度的张量（Tensor），
    用于后续深度学习模型（如多模式集成预测）的输入。
    """
    # 统一格式化目标起报时间
    normalized_report_time = normalize_report_time(report_time)
    if not normalized_report_time:
        raise ValueError("report_time is required.")
    records_by_model = [] # 用于存储各个机构清洗后的时序预报记录
    max_timesteps = 0     # 记录所有机构中最长的预报时效步数（序列长度T）

    # 1. 遍历预设的机构列表，提取并清洗每个机构的数据
    for institution in INSTITUTIONS:
        fcst_code = FCST_CODES[institution]
        records = []

        # 安全地从 JSON 响应中提取当前机构的数据列表
        response_json = response_by_fcst_code.get(fcst_code) if fcst_code else None
        data_list = response_json.get("data", []) if isinstance(response_json, dict) else []
        if not isinstance(data_list, list):
            data_list = []

        # 遍历数据列表，提取核心预报记录
        for item in data_list:
            if not isinstance(item, dict):
                continue
            value = item.get(fcst_code)
            if isinstance(value, list):
                records.extend(record for record in value if isinstance(record, dict))

        # 2. 筛选特定起报时间的记录
        records = [
            record for record in records
            if normalize_report_time(record.get("datetime")) == normalized_report_time
        ]
        # 3. 按预报时效 (fcsthour) 排序并去重
        sortable_records = []
        for record in records:
            fcsthour = to_float(record.get("fcsthour"))
            if not np.isnan(fcsthour):
                sortable_records.append((fcsthour, record))
        sortable_records.sort(key=lambda item: item[0]) # 按预报时效升序排列

        deduped_records = []
        seen_hours = set()
        for fcsthour, record in sortable_records:
            if fcsthour in seen_hours:
                continue # 如果同一预报时效有多条记录，只保留第一条
            seen_hours.add(fcsthour)
            deduped_records.append(record)

        records_by_model.append(deduped_records)
        max_timesteps = max(max_timesteps, len(deduped_records)) # 更新全局最大时间步长

    # 如果所有机构都没有有效数据，直接返回 None
    if max_timesteps == 0:
        return None, None, None

    feature_list = []
    mask_list = []

    # 4. 构建特征矩阵 (Features) 和 掩码矩阵 (Mask)
    for records in records_by_model:
        # 初始化当前机构的特征矩阵 [T, 6] (6个气象特征) 和掩码 [T]
        features = np.zeros((max_timesteps, 6), dtype=np.float32)
        model_mask = np.zeros(max_timesteps, dtype=bool)

        for t, record in enumerate(records):
            # 获取起报时间戳，异常则填0
            issue_timestamp = to_utc_timestamp(record.get("datetime"))
            if np.isnan(issue_timestamp):
                issue_timestamp = 0.0

            # 提取具体的台风气象要素
            fcsthour = to_float(record.get("fcsthour"))
            lat = to_float(record.get("lat"))
            lon = to_float(record.get("lon"))
            pressure = to_float(record.get("pressure"))
            windv = to_float(record.get("windv", record.get("movespeed"))) # 优先取 windv，其次取 movespeed

            # 拼接为单步特征向量
            row = np.array(
                [issue_timestamp, fcsthour, lat, lon, pressure, windv],
                dtype=np.float32,
            )
            # 将 NaN 替换为 0.0，防止模型计算出现 NaN 梯度
            features[t] = np.nan_to_num(row, nan=0.0)
            
            # 生成 Mask：只有经纬度均为有效数字时，该步才算作有效数据
            model_mask[t] = not np.isnan(lat) and not np.isnan(lon)

        feature_list.append(features)
        mask_list.append(model_mask)

    # 5. 组合多模式数据并添加机构的独热编码 (One-Hot)
    # 将列表沿机构维度拼接，X_features 形状: [T, 机构数, 6]
    X_features = np.stack(feature_list, axis=1)
    # mask 形状: [T, 机构数]
    mask = np.stack(mask_list, axis=1)

    # 生成机构身份的独热编码，形状: [机构数, 机构数]
    one_hot = np.eye(len(INSTITUTIONS), dtype=np.float32)
    # 将独热编码扩展到所有时间步，形状: [T, 机构数, 机构数]
    X_onehot = np.tile(one_hot, (max_timesteps, 1, 1))
    
    # 将气象特征与机构独热编码在特征维度拼接
    # X 最终特征维度 = 6 + 机构数
    X = np.concatenate([X_features, X_onehot], axis=-1)

    # 增加 Batch 维度 (Batch=1)，适配 PyTorch/TensorFlow 输入
    # X 形状: [1, T, 机构数, 6+机构数], mask 形状: [1, T, 机构数]
    X = X[np.newaxis, ...]
    mask = mask[np.newaxis, ...]

    # 6. 构建占位/辅助目标变量 y
    T = X.shape[1]
    y = np.zeros((1, T, 1, 4), dtype=np.float32)
    for t in range(T):
        # 寻找当前时间步 t 中，有有效预报的机构索引
        valid_idx = np.where(mask[0, t, :])[0]
        if len(valid_idx) > 0:
            # 取第一个有效机构的预报时效 (fcsthour, 即 X 的索引 1)，赋值给 y 的第 4 个通道
            y[0, t, 0, 3] = X[0, t, valid_idx[0], 1]

    return X, mask, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).with_name("sample_single_subjective.json"),
    )
    parser.add_argument("--report-time", default="2024-09-27 12:00:00")
    args = parser.parse_args()

    with args.json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    fcst_code = infer_fcst_code(payload)
    X, mask, y = parse_subjective_to_fixed_institutions(
        {fcst_code: payload},
        report_time=args.report_time,
    )

    if X is None:
        raise ValueError("No valid records found.")

    np.set_printoptions(precision=3, suppress=True)
    print("fcst_code:", fcst_code)
    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("y shape:", y.shape)
    print("X:")
    print(X)


if __name__ == "__main__":
    main()
