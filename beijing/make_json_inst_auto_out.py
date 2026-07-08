import os
import sys
import json
import requests
import numpy as np
import torch
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Union, Dict, Any

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

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


# =====================================================
# 工具函数
# =====================================================
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

    print(f"[ERROR] 无法从 payload 中推断出 FCSTType, payload 键值: {list(payload.keys())}")
    raise ValueError("Cannot infer FCSTType from JSON.")


def parse_subjective_to_fixed_institutions(response_by_fcst_code, report_time=None):
    """
    解析不同发报中心（机构）的台风预报数据，并将其转化为固定维度的张量（Tensor），
    用于后续深度学习模型（如多模式集成预测）的输入。
    """
    # 统一格式化目标起报时间
    normalized_report_time = normalize_report_time(report_time)
    if not normalized_report_time:
        print("[ERROR] 缺少必要的 report_time 参数！")
        raise ValueError("report_time is required.")

    print(f"[INFO] 开始解析多模式预报数据，目标起报时间: {normalized_report_time}")

    records_by_model = []  # 用于存储各个机构清洗后的时序预报记录
    max_timesteps = 0  # 记录所有机构中最长的预报时效步数（序列长度T）

    # 1. 遍历预设的机构列表，提取并清洗每个机构的数据
    for institution in INSTITUTIONS:
        fcst_code = FCST_CODES.get(institution)
        if not fcst_code:
            print(f"[WARNING] 机构 {institution} 在 FCST_CODES 字典中找不到对应的代码，保留空位。")
            records_by_model.append([])
            continue

        records = []
        response_json = response_by_fcst_code.get(fcst_code) if fcst_code else None

        if not response_json:
            data_list = []
        else:
            data_list = (
                response_json.get("data", []) if isinstance(response_json, dict) else []
            )
            if not isinstance(data_list, list):
                print(f"[WARNING] 机构 {institution}({fcst_code}) 返回的 'data' 字段不是列表格式！")
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
            record
            for record in records
            if normalize_report_time(record.get("datetime")) == normalized_report_time
        ]

        if not records:
            records_by_model.append([])
            continue

        # 3. 按预报时效 (fcsthour) 排序并去重
        sortable_records = []
        for record in records:
            fcsthour = to_float(record.get("fcsthour"))
            if not np.isnan(fcsthour):
                sortable_records.append((fcsthour, record))
        sortable_records.sort(key=lambda item: item[0])  # 按预报时效升序排列

        deduped_records = []
        seen_hours = set()
        for fcsthour, record in sortable_records:
            if fcsthour in seen_hours:
                continue  # 如果同一预报时效有多条记录，只保留第一条
            seen_hours.add(fcsthour)
            deduped_records.append(record)

        records_by_model.append(deduped_records)
        max_timesteps = max(max_timesteps, len(deduped_records))  # 更新全局最大时间步长

    # 如果所有机构都没有有效数据，直接返回 None
    if max_timesteps == 0:
        print(f"[WARNING] 所有机构在 {normalized_report_time} 时刻均无有效预报数据，函数将返回 None。")
        return None, None, None

    feature_list = []
    mask_list = []

    # 4. 构建特征矩阵 (Features) 和 掩码矩阵 (Mask)
    for i, records in enumerate(records_by_model):
        institution_name = INSTITUTIONS[i]
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
            windv = to_float(
                record.get("windv", record.get("movespeed"))
            )  # 优先取 windv，其次取 movespeed

            # 拼接为单步特征向量
            row = np.array(
                [issue_timestamp, fcsthour, lat, lon, pressure, windv],
                dtype=np.float32,
            )
            # 将 NaN 替换为 0.0，防止模型计算出现 NaN 梯度
            features[t] = np.nan_to_num(row, nan=0.0)

            # 生成 Mask：只有经纬度均为有效数字时，该步才算作有效数据
            is_valid = not np.isnan(lat) and not np.isnan(lon)
            model_mask[t] = is_valid

        feature_list.append(features)
        mask_list.append(model_mask)

    # 5. 组合多模式数据并添加机构的独热编码 (One-Hot)
    X_features = np.stack(feature_list, axis=1)
    mask = np.stack(mask_list, axis=1)

    one_hot = np.eye(len(INSTITUTIONS), dtype=np.float32)
    X_onehot = np.tile(one_hot, (max_timesteps, 1, 1))

    X = np.concatenate([X_features, X_onehot], axis=-1)

    X = X[np.newaxis, ...]
    mask = mask[np.newaxis, ...]

    # 6. 构建占位/辅助目标变量 y
    T = X.shape[1]
    y = np.zeros((1, T, 1, 4), dtype=np.float32)
    for t in range(T):
        valid_idx = np.where(mask[0, t, :])[0]
        if len(valid_idx) > 0:
            y[0, t, 0, 3] = X[0, t, valid_idx[0], 1]

    print(f"[INFO] 张量构建成功！最终输出形状 -> X: {X.shape}, mask: {mask.shape}, y: {y.shape}")

    return X, mask, y


def _has_zone(typhoon: Dict[str, Any], zone: str = "W") -> bool:
    """检查台风是否属于指定的区域 (默认 'W')。"""
    if typhoon.get("zone") == zone:
        return True

    for value in typhoon.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if any(item.get("zone") == zone for item in value):
                return True

    return False


def extract_typhoon_info_by_zone(
    data: Union[str, Dict[str, Any]], zone: str = "W"
) -> List[Dict[str, str]]:
    """根据指定的区域提取活跃台风的基础信息 (序号、英文名、台风编号)。"""
    if isinstance(data, str):
        data = json.loads(data)

    typhoon_infos = []
    if not data.get("success") or "data" not in data:
        return typhoon_infos

    typhoons = data["data"]
    for typhoon in typhoons:
        xuhao = typhoon.get("xuhao")
        if xuhao is None:
            continue

        # 仅保留活跃状态的台风
        if typhoon.get("activityTyphoon") is False:
            continue

        # 检查是否在目标区域内
        if not _has_zone(typhoon, zone):
            continue

        typhoon_infos.append(
            {
                "xuhao": str(xuhao),
                "engname": str(typhoon.get("engname") or ""),
                "tfbh": str(typhoon.get("tfbh") or ""),
            }
        )

    return typhoon_infos


# =====================================================
# API 交互与主流程
# =====================================================
def parse_latest_ensemble_time(response_json: Dict[str, Any]) -> Optional[str]:
    """从 API 响应中解析并返回最新的集合预报时间。"""
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
            # 验证时间格式是否合法
            datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        valid_times.append(time_str)

    # 返回最大的时间字符串（即最新时间）
    return max(valid_times) if valid_times else None



def fetch_and_process(
    data_time: str,
    access_key: str = None,
    user_id: str = "1000004",
    security_key: str = "nN2hJQEN3v7Y7sPmDvthdvkQXapuZV",
) -> List[Tuple[str, str, str, str]]:
    """
    主控函数：拉取活跃台风，按传入起报时间循环下载 8 家主观预报，
    处理成 PyTorch Tensor 并落盘保存。
    """
    # 动态生成 API 密钥
    if access_key is None:
        try:
            from key import generate_access_key
            access_key = generate_access_key(user_id, security_key)
        except ImportError:
            print("[ERROR] 无法导入 key.py 模块或 generate_access_key 函数。")
            return []

    headers = {"typhoon-access-key": access_key}

    # 1. 获取活跃台风列表
    url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getActiveTyphoon"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] 获取台风编号列表失败: {e}")
        return []

    data = response.json()

    # 筛选活跃台风并保留元数据
    typhoon_infos = extract_typhoon_info_by_zone(data)
    print(f"[INFO] 待处理的台风列表: {typhoon_infos}")
    if not typhoon_infos:
        print("[INFO] 没有符合条件的台风需要处理")
        return []

    url_subjective = (
        "http://10.40.168.50:28000/"
        "cmes-typhoonOcean/api/tcRealtime/getTyphoonInfoByTypeAndTime/"
    )
    success_codes = []

    # 2. 遍历每个台风进行预报数据的拉取与处理
    for typhoon_info in typhoon_infos:
        typhoon_code = typhoon_info["xuhao"]
        engname = typhoon_info["engname"]
        tfbh = typhoon_info["tfbh"]
        
        print(f"\n" + "="*50)
        print(f"[INFO] 正在处理台风编号: {typhoon_code}, engname: {engname}, tfbh: {tfbh}")

        # 格式化时间字符串作为文件夹名称 (避免包含非法字符)
        report_time = normalize_report_time(data_time)
        if not report_time:
            print("[ERROR] data_time 不能为空")
            return success_codes

        safe_time = report_time.replace(" ", "-").replace(":", "-")

        # 设定本地保存路径
        save_dir = os.path.join(BASE_DIR, "data", str(typhoon_code), safe_time)
        expected_files = ["x_results.pt", "x_masks.pt", "y.pt"]

        # 检查是否已经处理过此文件
        if all(os.path.exists(os.path.join(save_dir, name)) for name in expected_files):
            print(f"[INFO] 台风 {typhoon_code} 时次 {report_time} 已处理过，跳过")
            continue

        # 循环请求 8 家主观预报机构，按 FCSTType code 暂存响应
        response_by_fcst_code = {}
        for institution in INSTITUTIONS:
            center_code = FCST_CODES.get(institution)
            if not center_code:
                continue

            payload = {
                "xuhao": typhoon_code,
                "fcstType": center_code,
                "time": report_time,
            }
            try:
                resp = requests.request(
                    "GET",
                    url_subjective,
                    headers=headers,
                    data=payload,
                    timeout=30,
                )
                if resp.status_code != 200:
                    print(f"[ERROR] 请求 {institution}({center_code}) 失败: {resp.status_code}")
                    continue

                center_data = resp.json()
                if center_data.get("code") != 200:
                    print(f"[ERROR] {institution}({center_code}) API错误: code={center_data.get('code')}, msg={center_data.get('msg')}")
                    continue

                response_by_fcst_code[center_code] = center_data
            except Exception as e:
                print(f"[ERROR] 请求 {institution}({center_code}) 预报数据异常: {e}")
                continue

        # 3. 将原始 JSON 解析为深度学习所需的特征矩阵
        X, mask, y = parse_subjective_to_fixed_institutions(
            response_by_fcst_code,
            report_time=report_time,
        )
        if X is None:
            print(f"[WARNING] 台风 {typhoon_code} 数据解析失败，X is None，跳过")
            continue

        # 4. 保存 Tensor 为文件
        os.makedirs(save_dir, exist_ok=True)
        torch.save(torch.tensor(X), os.path.join(save_dir, "x_results.pt"))
        torch.save(torch.tensor(mask), os.path.join(save_dir, "x_masks.pt"))
        torch.save(torch.tensor(y), os.path.join(save_dir, "y.pt"))

        print(f"[SUCCESS] 台风 {typhoon_code} 数据已保存至 -> {os.path.abspath(save_dir)}")

        # 记录处理成功的台风信息
        success_codes.append((typhoon_code, safe_time, engname, tfbh))

    print(f"\n[INFO] 处理完成，成功保存的台风: {success_codes}")
    return success_codes


if __name__ == "__main__":
    start_time = "2026-06-04 00:00:00"
    result = fetch_and_process(start_time)