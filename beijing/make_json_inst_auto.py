import os
import sys
import json
import logging
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
# 日志配置 (同时输出到控制台和文件)
# =====================================================
def setup_logger():
    """初始化并配置日志记录器。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("TyphoonProcessor")
    logger.setLevel(logging.INFO)  # 日常运行建议设为 INFO，排查具体数据时可改为 DEBUG

    # 避免重复添加 handler 导致日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler：将日志写入每天的文件
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件里可以记录更详细的 DEBUG 信息
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台 handler：在终端输出日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台保持清爽，只看 INFO 以上
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


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

    # 【日志注入点】推断失败通常意味着上游 API 接口数据结构发生了重大变化
    logger.error(
        f"无法从 payload 中推断出 FCSTType, payload 键值: {list(payload.keys())}"
    )
    raise ValueError("Cannot infer FCSTType from JSON.")


def parse_subjective_to_fixed_institutions(response_by_fcst_code, report_time=None):
    """
    解析不同发报中心（机构）的台风预报数据，并将其转化为固定维度的张量（Tensor），
    用于后续深度学习模型（如多模式集成预测）的输入。
    """
    # 统一格式化目标起报时间
    normalized_report_time = normalize_report_time(report_time)
    if not normalized_report_time:
        # 【日志注入点】关键参数缺失拦截
        logger.error("缺少必要的 report_time 参数！")
        raise ValueError("report_time is required.")

    # 【日志注入点】记录函数入口和核心参数
    logger.info(f"开始解析多模式预报数据，目标起报时间: {normalized_report_time}")

    records_by_model = []  # 用于存储各个机构清洗后的时序预报记录
    max_timesteps = 0  # 记录所有机构中最长的预报时效步数（序列长度T）

    # 1. 遍历预设的机构列表，提取并清洗每个机构的数据
    for institution in INSTITUTIONS:
        fcst_code = FCST_CODES.get(institution)
        if not fcst_code:
            logger.warning(
                f"机构 {institution} 在 FCST_CODES 字典中找不到对应的代码，保留空位。"
            )
            records_by_model.append([])
            continue

        records = []

        # 安全地从 JSON 响应中提取当前机构的数据列表
        response_json = response_by_fcst_code.get(fcst_code) if fcst_code else None

        if not response_json:
            # 【日志注入点】某些机构没返回数据是常态，用 debug 或 info 记录即可，不要用 error
            logger.info(f"数据源中未包含机构 {institution}({fcst_code}) 的预报数据。")
            data_list = []
        else:
            data_list = (
                response_json.get("data", []) if isinstance(response_json, dict) else []
            )
            if not isinstance(data_list, list):
                logger.warning(
                    f"机构 {institution}({fcst_code}) 返回的 'data' 字段不是列表格式！"
                )
                data_list = []

        # 遍历数据列表，提取核心预报记录
        for item in data_list:
            if not isinstance(item, dict):
                continue
            value = item.get(fcst_code)
            if isinstance(value, list):
                records.extend(record for record in value if isinstance(record, dict))

        logger.debug(f"机构 {institution} 提取到原始轨迹点共 {len(records)} 个。")

        # 2. 筛选特定起报时间的记录
        records = [
            record
            for record in records
            if normalize_report_time(record.get("datetime")) == normalized_report_time
        ]

        if not records:
            logger.debug(
                f"机构 {institution} 在指定时间 {normalized_report_time} 下没有预报轨迹点。"
            )
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
                logger.debug(
                    f"机构 {institution} 发现重复的预报时效 {fcsthour}h，已去重。"
                )
                continue  # 如果同一预报时效有多条记录，只保留第一条
            seen_hours.add(fcsthour)
            deduped_records.append(record)

        records_by_model.append(deduped_records)
        max_timesteps = max(max_timesteps, len(deduped_records))  # 更新全局最大时间步长
        logger.info(
            f"机构 {institution} 解析完成，共获取 {len(deduped_records)} 个有效预报时效。"
        )

    # 如果所有机构都没有有效数据，直接返回 None
    if max_timesteps == 0:
        # 【日志注入点】这是业务层面的“空跑”，非常重要，需要警告
        logger.warning(
            f"所有机构在 {normalized_report_time} 时刻均无有效预报数据，函数将返回 None。"
        )
        return None, None, None

    logger.debug(
        f"所有机构数据提取完毕，全局最大时间步长 max_timesteps = {max_timesteps}"
    )

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

            if not is_valid:
                logger.debug(
                    f"机构 {institution_name} 在 fcsthour={fcsthour} 时经纬度数据无效(lat={lat}, lon={lon})，mask 设为 False"
                )

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

    # 【日志注入点】成功出口，打印张量形状，方便核对深度学习模型输入层维度
    logger.info(
        f"张量构建成功！最终输出形状 -> X: {X.shape}, mask: {mask.shape}, y: {y.shape}"
    )

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


def _find_nested_value(typhoon: Dict[str, Any], key: str, zone: str = "W") -> str:
    """从发报中心数组里提取台风元信息，比如 tfbh/engname。"""
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

        # 检查是否在目标区域内
        if not _has_zone(typhoon, zone):
            continue

        typhoon_infos.append(
            {
                "xuhao": str(xuhao),
                "engname": (
                    str(typhoon.get("engname") or typhoon.get("enname") or "")
                    or _find_nested_value(typhoon, "engname", zone)
                ),
                "tfbh": _find_nested_value(typhoon, "tfbh", zone),
            }
        )

    return typhoon_infos


# =====================================================
# API 交互与主流程
# =====================================================

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
            logger.error("无法导入 key.py 模块或 generate_access_key 函数。")
            return []

    headers = {"typhoon-access-key": access_key}

    # 1. 获取活跃台风列表
    url = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getActiveTyphoon"
    # url = "http://106.120.73.242/wg-cmes/cmes-typhoonocean-internal/api/tcRealtime/getActiveTyphoon"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"获取台风编号列表失败: {e}")
        return []

    data = response.json()

    # 筛选活跃台风并保留元数据
    typhoon_infos = extract_typhoon_info_by_zone(data)
    logger.info(f"待处理的台风列表: {typhoon_infos}")
    if not typhoon_infos:
        logger.info("没有符合条件的台风需要处理")
        return []

    url_subjective = (
        "http://10.40.168.50:28000/"
        "cmes-typhoonOcean/api/tcRealtime/getTyphoonInfoByTypeAndTime/"
    )
    # url_subjective = (
    #     "http://106.120.73.242/wg-cmes/cmes-typhoonocean-internal/api/tcRealtime/getTyphoonInfoByTypeAndTime"
    # )
    success_codes = []

    # 2. 遍历每个台风进行预报数据的拉取与处理
    for typhoon_info in typhoon_infos:
        typhoon_code = typhoon_info["xuhao"]
        engname = typhoon_info["engname"]
        tfbh = typhoon_info["tfbh"]
        logger.info(
            f"\n正在处理台风编号: {typhoon_code}, engname: {engname}, tfbh: {tfbh}"
        )

        # 格式化时间字符串作为文件夹名称 (避免包含非法字符)
        report_time = normalize_report_time(data_time)
        if not report_time:
            logger.error("data_time 不能为空")
            return success_codes

        safe_time = report_time.replace(" ", "-").replace(":", "-")
        logger.info(f"安全目录时间: {safe_time}")

        # 设定本地保存路径
        save_dir = os.path.join(BASE_DIR, "data", str(typhoon_code), safe_time)
        expected_files = ["x_results.pt", "x_masks.pt", "y.pt"]

        # 检查是否已经处理过此文件
        if all(os.path.exists(os.path.join(save_dir, name)) for name in expected_files):
            logger.info(f"台风 {typhoon_code} 时次 {report_time} 已处理过，跳过")
            continue

        # 循环请求 8 家主观预报机构，按 FCSTType code 暂存响应
        response_by_fcst_code = {}
        for institution in INSTITUTIONS:
            center_code = FCST_CODES.get(institution)
            if not center_code:
                logger.info(f"机构 {institution} 暂无 fcstType code，保留空位")
                continue

            payload = {
                "xuhao": typhoon_code,
                "fcstType": center_code,
                "dataTime": report_time,
            }
            try:
                resp = requests.request(
                    "GET",
                    url_subjective,
                    headers=headers,
                    params=payload,
                    timeout=30,
                )
                if resp.status_code != 200:
                    logger.error(
                        f"请求 {institution}({center_code}) 失败: {resp.status_code}"
                    )
                    continue

                center_data = resp.json()
                if center_data.get("code") != 200:
                    logger.error(
                        f"{institution}({center_code}) API错误: "
                        f"code={center_data.get('code')}, msg={center_data.get('msg')}"
                    )
                    continue

                response_by_fcst_code[center_code] = center_data
                logger.info(f"获取 {institution}({center_code}) 数据成功")
            except Exception as e:
                logger.error(f"请求 {institution}({center_code}) 预报数据异常: {e}")
                continue

        # 3. 将原始 JSON 解析为深度学习所需的特征矩阵
        X, mask, y = parse_subjective_to_fixed_institutions(
            response_by_fcst_code,
            report_time=report_time,
        )
        if X is None:
            logger.warning(f"台风 {typhoon_code} 数据解析失败，X is None，跳过")
            continue

        logger.info(f"X shape: {X.shape}")
        logger.info(f"mask shape: {mask.shape}")
        logger.info(f"y shape: {y.shape}")

        # 4. 保存 Tensor 为文件
        os.makedirs(save_dir, exist_ok=True)
        torch.save(torch.tensor(X), os.path.join(save_dir, "x_results.pt"))
        torch.save(torch.tensor(mask), os.path.join(save_dir, "x_masks.pt"))
        torch.save(torch.tensor(y), os.path.join(save_dir, "y.pt"))

        logger.info(f"台风 {typhoon_code} 数据保存成功 -> {os.path.abspath(save_dir)}")

        # 记录处理成功的台风信息
        success_codes.append((typhoon_code, safe_time, engname, tfbh))

    logger.info(f"\n处理完成，成功保存的台风: {success_codes}")
    return success_codes


if __name__ == "__main__":
    start_time = "2026-07-09 00:00:00"
    result = fetch_and_process(start_time)
    logger.info(f"最终结果: {result}")
