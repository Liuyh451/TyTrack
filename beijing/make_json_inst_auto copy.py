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

# =====================================================
# 日志配置 (同时输出到控制台和文件)
# =====================================================
def setup_logger():
    """初始化并配置日志记录器。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("TyphoonProcessor")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler 导致日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler：将日志写入每天的文件
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 控制台 handler：在终端输出日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()


# =====================================================
# 工具函数
# =====================================================
def to_utc_timestamp(time_str: str) -> Union[int, float]:
    """将 ISO 格式的时间字符串转换为 UTC 时间戳。"""
    if pd.isna(time_str) or time_str is None:
        return np.nan

    try:
        # 将 "Z" 替换为 "+00:00" 以兼容 fromisoformat
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        # 如果没有时区信息，默认假定为东八区 (北京时间)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))

        # 转换为 UTC 并返回时间戳
        utc_time = dt.astimezone(timezone.utc)
        return int(utc_time.timestamp())

    except Exception:
        return np.nan


# =====================================================
# 核心数据处理模块
# =====================================================
def parse_ensemble_to_fixed_models(response_json: Dict[str, Any], max_models: int = 8) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    解析集合预报数据，对齐时间步并填充至固定数量的模型 (max_models)。
    返回: X (特征), mask (掩码), y (标签)
    """
    data_list = response_json.get("data", [])

    if not data_list:
        logger.warning("No data found.")
        return None, None, None

    # 提取所有集合成员
    members = []

    for member_obj in data_list:
        for key, value in member_obj.items():
            # 过滤出键名为数字（代表模型/成员编号）的列表
            if not key.isdigit():
                continue
            if not isinstance(value, list) or len(value) == 0:
                continue

            # 按预报时效 (fcsthour) 排序
            value = sorted(value, key=lambda x: x.get("fcsthour", 0))

            # 去重：确保每个预报时效只有一个记录
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

    # 过滤无效成员：确保成员中至少有一条记录包含经纬度
    valid_members = []
    for recs in members:
        has_valid = any(
            r.get("lat") is not None and r.get("lon") is not None for r in recs
        )
        if has_valid:
            valid_members.append(recs)

    logger.info(f"有效成员数: {len(valid_members)}")

    # 截断或保留指定最大数量的模型成员
    selected_members = valid_members[:max_models]
    num_models = len(selected_members)
    logger.info(f"实际使用成员数: {num_models}")

    # 用 None 补齐缺失的模型位置
    padded_members = selected_members + [None] * (max_models - num_models)

    # 找出所有模型中最长的时间步长
    max_timesteps = max(len(recs) for recs in selected_members) if selected_members else 0
    logger.info(f"时间步长: {max_timesteps}")

    if max_timesteps == 0:
        logger.error("没有有效的时间步数据，返回 None")
        return None, None, None

    # 获取起报时间的基准时间戳
    first_record = selected_members[0][0]
    base_dt_str = first_record.get("datetime")
    base_timestamp = to_utc_timestamp(base_dt_str) if base_dt_str else 0

    feature_list = []
    mask_list = []

    # 遍历补齐后的成员，提取特征并构建矩阵
    for recs in padded_members:
        if recs is None:
            # 缺失的模型用 0 填充，掩码为 False
            features = np.zeros((max_timesteps, 6), dtype=np.float32)
            mask = np.zeros(max_timesteps, dtype=bool)
        else:
            # 构建有效模型的 DataFrame
            df = pd.DataFrame(recs)
            df["time"] = base_timestamp
            df["pre_time"] = df["fcsthour"].astype(float)
            df["lat_model"] = df["lat"].astype(float)
            df["lng_model"] = df["lon"].astype(float)
            df["pressure_model"] = pd.to_numeric(df["pressure"], errors="coerce")
            df["speed_model"] = pd.to_numeric(df["windv"], errors="coerce")
            
            # 提取需要的 6 列特征
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
            # 如果当前模型的时间步长不够最大步长，用 0 补齐
            if cur_len < max_timesteps:
                pad = np.zeros((max_timesteps - cur_len, 6), dtype=np.float32)
                features = np.vstack([features, pad])
                
            mask = np.zeros(max_timesteps, dtype=bool)
            mask[:cur_len] = True
            
            # 处理 NaN 值
            features = np.nan_to_num(features, nan=0.0)

        feature_list.append(features)
        mask_list.append(mask)

    # 堆叠所有模型的特征和掩码
    X_features = np.stack(feature_list, axis=1)  # 形状: (T, M, 6)
    mask = np.stack(mask_list, axis=1)           # 形状: (T, M)

    # 生成模型编号的 One-hot 编码并与特征拼接
    one_hot = np.eye(max_models, dtype=np.float32)
    X_onehot = np.tile(one_hot, (max_timesteps, 1, 1))
    X = np.concatenate([X_features, X_onehot], axis=-1)  # 形状: (T, M, 14)

    # 增加 Batch 维度
    X = X[np.newaxis, ...]      # 形状: (1, T, M, 14)
    mask = mask[np.newaxis, ...] # 形状: (1, T, M)

    # 构造目标变量 y
    T = X.shape[1]
    y = np.zeros((1, T, 1, 4), dtype=np.float32)
    for t in range(T):
        valid_idx = np.where(mask[0, t, :])[0]
        if len(valid_idx) > 0:
            # 提取第一个有效模型的预报时效 (pre_time) 作为标签里的 lead_time
            lead_time = X[0, t, valid_idx[0], 1]
            y[0, t, 0, 3] = lead_time

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


def extract_typhoon_info_by_zone(data: Union[str, Dict[str, Any]], zone: str = "W") -> List[Dict[str, str]]:
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


def fetch_latest_ensemble_time(headers: Dict[str, str], typhoon_code: str, fcst_type: str) -> Optional[str]:
    """调用接口获取特定台风和预报类型的最新预报时间。"""
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
                f"fcstType={fcst_type}, code={data.get('code')}, msg={data.get('msg')} "
                f"rawData={data}"
            )
            return None

        logger.info(f"最新集合预报时间: typhoonCode={typhoon_code}, fcstType={fcst_type}, dataTime={latest_time}")
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
) -> List[Tuple[str, str, str, str]]:
    """
    主控函数：拉取活跃台风、获取最新预报时次、下载集合数据、处理成 PyTorch Tensor 并落盘保存。
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

    url_ens = "http://10.40.168.50:28000/cmes-typhoonOcean-internal/api/tcRealtime/getTcEnsembleForecast"
    success_codes = []

    # 2. 遍历每个台风进行预报数据的拉取与处理
    for typhoon_info in typhoon_infos:
        typhoon_code = typhoon_info["xuhao"]
        engname = typhoon_info["engname"]
        tfbh = typhoon_info["tfbh"]
        logger.info(f"\n正在处理台风编号: {typhoon_code}, engname: {engname}, tfbh: {tfbh}")
        
        # 获取最新集合预报时间
        latest_data_time = fetch_latest_ensemble_time(headers, typhoon_code, fcst_type)
        if latest_data_time is None:
            logger.warning(f"台风 {typhoon_code} 未获取到最新集合预报时间，舍弃")
            continue
            
        # 格式化时间字符串作为文件夹名称 (避免包含非法字符)
        safe_time = latest_data_time.replace(" ", "-").replace(":", "-")
        logger.info(f"安全目录时间: {safe_time}")
        
        # 设定本地保存路径
        save_dir = os.path.join(BASE_DIR, "data", str(typhoon_code), safe_time)
        expected_files = ["x_results.pt", "x_masks.pt", "y.pt"]
        
        # 检查是否已经处理过此文件
        if all(os.path.exists(os.path.join(save_dir, name)) for name in expected_files):
            logger.info(f"台风 {typhoon_code} 时次 {latest_data_time} 已处理过，跳过")
            continue

        # 请求实际的集合预报数据
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

        # 3. 将原始 JSON 解析为深度学习所需的特征矩阵
        X, mask, y = parse_ensemble_to_fixed_models(data, max_models=8)
        if X is None:
            logger.warning(f"台风 {typhoon_code} 数据解析失败，X is None，跳过，rawData={data}")
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
    start_time = "2026-06-04 00:00:00"
    result = fetch_and_process(start_time, fcst_type="FENSENS")
    logger.info(f"最终结果: {result}")