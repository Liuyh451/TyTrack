import os
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================= 配置区 =================

# 所有输出都落在脚本所在目录，避免从不同工作目录启动时写到别处。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_NAME = "forecast_records.txt"
NANHAI_WIND_NAME = "nanhai_wind.txt"

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

ALLOWED_HOURS = [2, 8, 14, 20]  # 北京时间

PRE_TIME_LIST = [12, 24, 36, 48, 60, 72, 96, 120]


class TyphoonDataError(Exception):
    """台风样本处理异常，由主控脚本捕获并记录。"""

    pass


class NoSampleError(TyphoonDataError):
    """当前起报时次没有可用预报样本。"""

    pass


# ================= 数据加载函数 =================


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build typhoon forecast samples without ground truth."
    )

    parser.add_argument(
        "report_time", help="Beijing report time, format YYYYMMDDHH, e.g. 2026062214."
    )

    parser.add_argument("json_url", help="Typhoon detail JSON URL.")

    args = parser.parse_args()

    try:
        args.report_dt = datetime.strptime(args.report_time, "%Y%m%d%H")
    except ValueError as exc:
        raise SystemExit("report_time must use YYYYMMDDHH, e.g. 2026062214") from exc

    return args


def fetch_json(url):

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    except Exception as e:
        raise TyphoonDataError(f"JSON fetch failed: {url}: {e}") from e


def get_forecast_wind(forecast):
    """提取预报中的风速字段，源数据缺失时按 0 处理。"""

    for key in ("speed", "wind", "wind_speed", "windv", "vmax"):
        value = forecast.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def save_nanhai_wind_forecast(typhoon_tracks, save_dir):
    """保存南海所各预报时效的风速，缺失值补 0。"""

    wind_by_hour = {hour: 0.0 for hour in PRE_TIME_LIST}

    for item in typhoon_tracks.get("南海所", []):
        try:
            pre_time = int(round(float(item.get("pre_time", 0))))
        except (TypeError, ValueError):
            continue

        if pre_time in wind_by_hour:
            wind_by_hour[pre_time] = float(item.get("wind_model") or 0.0)

    wind_path = os.path.join(save_dir, NANHAI_WIND_NAME)
    with open(wind_path, "w", encoding="utf-8") as f:
        f.write("pre_time wind_speed\n")
        for hour in PRE_TIME_LIST:
            f.write(f"{hour} {wind_by_hour[hour]:.1f}\n")

    return wind_path


# ================= X 构造函数 =================


def process_single_x(tracks, max_steps, orglist):
    num_inst = len(orglist)
    one_hot_eye = np.eye(num_inst)
    inst_sequences = []
    mask_matrix = np.zeros((num_inst, max_steps), dtype=bool)
    for m_idx, inst in enumerate(orglist):
        data_list = tracks.get(inst, [])

        # 无数据
        if not data_list:
            ts = np.zeros(max_steps)
            pre = np.zeros(max_steps)
            lat = np.zeros(max_steps)
            lng = np.zeros(max_steps)

            mask_matrix[m_idx, :] = False

        else:
            df = pd.DataFrame(data_list)

            # 补齐长度
            if len(df) < max_steps:
                nan_df = pd.DataFrame(
                    np.nan, index=range(max_steps - len(df)), columns=df.columns
                )

                df = pd.concat([df, nan_df], ignore_index=True)

            else:
                df = df.iloc[:max_steps]

            # 北京时间 -> UTC timestamp
            def to_ts(row):

                if "time" not in row or pd.isna(row["time"]):
                    return 0

                return int(
                    (
                        datetime.fromisoformat(row["time"]) - timedelta(hours=8)
                    ).timestamp()
                )

            ts = df.apply(to_ts, axis=1).values

            pre = (
                df["pre_time"].fillna(0).values
                if "pre_time" in df
                else np.zeros(max_steps)
            )

            lat = (
                df["lat_model"].fillna(0).values
                if "lat_model" in df
                else np.zeros(max_steps)
            )

            lng = (
                df["lng_model"].fillna(0).values
                if "lng_model" in df
                else np.zeros(max_steps)
            )

            mask_matrix[m_idx, :] = (
                ~df["lat_model"].isna().values if "lat_model" in df else False
            )

        # 前6维
        base_feat = np.stack(
            [ts, pre, lat, lng, np.zeros_like(lat), np.zeros_like(lat)], axis=1
        )

        # one-hot机构编码
        ids = np.tile(one_hot_eye[m_idx], (max_steps, 1))

        inst_sequences.append(np.hstack([base_feat, ids]))

    # [T, M, D]
    return (
        np.array(inst_sequences).transpose(1, 0, 2).astype(np.float32),
        mask_matrix.T,
    )


# ================= 主逻辑 =================


def process_typhoon(json_data, report_dt):

    # 处理单个台风详情 JSON，按 report_time 精确生成一个起报时次的样本。
    if not json_data:
        raise TyphoonDataError("empty typhoon detail JSON")

    target_typhoon_name = json_data["ename"].upper().strip()

    ty_name = json_data["ename"].lower().strip()

    ty_code = json_data.get("ty_code", "").strip()

    save_name = f"{ty_code}_{ty_name}" if ty_code else ty_name

    # report_key 作为目录层级，保证同一台风不同起报时次不会互相覆盖。
    report_key = report_dt.strftime("%Y%m%d%H")

    SAVE_DIR = os.path.join(SCRIPT_DIR, "data", save_name, report_key)

    os.makedirs(SAVE_DIR, exist_ok=True)
    txt_path = os.path.join(SAVE_DIR, TXT_NAME)
    begin_t = datetime.fromisoformat(json_data["begin_time"])
    end_t = datetime.fromisoformat(json_data["end_time"])
    all_x = []
    all_masks = []
    all_y = []
    print(f"[INFO] Processing typhoon: {target_typhoon_name}")
    print(f"精准匹配起报时间 (BJ): {report_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"台风记录: {target_typhoon_name}\n")
        f_txt.write("=" * 60 + "\n")

        for info in json_data.get("data", []):
            bj_time_str = info["time"]

            dt_bj = datetime.fromisoformat(bj_time_str)

            # 时间过滤
            # 传入 report_time 时只保留完全相等的起报点；不传时才走原批量逻辑。
            if dt_bj != report_dt:
                continue

            if not (begin_t <= dt_bj <= end_t):
                raise TyphoonDataError(
                    f"report time {report_dt} is outside typhoon lifetime "
                    f"{begin_t} - {end_t}"
                )

            # ================= 提取预报数据 =================

            typhoon_tracks = {inst: [] for inst in INSTITUTIONS}

            has_forecast = False

            if "data" in info:
                for source_data in info["data"]:
                    src = source_data.get("source")

                    if src not in INSTITUTIONS:
                        continue

                    forecasts = source_data.get("data", [])

                    if not forecasts:
                        continue

                    has_forecast = True

                    for f in forecasts:
                        f_dt = datetime.fromisoformat(f["time"])

                        typhoon_tracks[src].append(
                            {
                                "time": bj_time_str,
                                "pre_time": ((f_dt - dt_bj).total_seconds() / 3600),
                                "lat_model": f.get("lat"),
                                "lng_model": f.get("lng"),
                                "wind_model": get_forecast_wind(f),
                            }
                        )

            if not has_forecast:
                continue

            # ================= 构造 X =================
            x_sample, m_sample = process_single_x(
                typhoon_tracks, len(PRE_TIME_LIST), INSTITUTIONS
            )

            # ================= 构造 Y =================
            # [lat, lon, timestamp, leadtime]
            y_sample = np.zeros((len(PRE_TIME_LIST), 1, 4), dtype=np.float32)
            has_valid_forecast = bool(m_sample.any())
            utc_report = dt_bj - timedelta(hours=8)

            if has_valid_forecast:
                for t_idx, offset in enumerate(PRE_TIME_LIST):
                    utc_target = utc_report + timedelta(hours=offset)

                    # lat/lon 保持为 0
                    y_sample[t_idx, 0, 2] = int(utc_target.timestamp())
                    y_sample[t_idx, 0, 3] = offset

            # ================= 保存样本 =================
            all_x.append(x_sample)
            all_masks.append(m_sample)
            all_y.append(y_sample)
            # ================= 写 TXT =================

            f_txt.write(f"\n[起报时间 (BJ)]: {bj_time_str}\n")

            f_txt.write("-" * 40 + "\n")

            for inst in INSTITUTIONS:
                data = typhoon_tracks.get(inst, [])

                if data:
                    f_txt.write(f"机构: {inst}\n")

                    for p in data:
                        f_txt.write(
                            f"  + {p['pre_time']:>3.0f}h "
                            f"| Lat: {p['lat_model']:>5.2f} "
                            f"| Lon: {p['lng_model']:>6.2f}\n"
                        )

                else:
                    f_txt.write(f"机构: {inst} (无数据)\n")

            f_txt.write("-" * 40 + "\n")

            print(f"[OK] Recorded report time: {bj_time_str}")

    # ================= 保存 =================

    if all_x:
        wind_path = save_nanhai_wind_forecast(typhoon_tracks, SAVE_DIR)

        # 当前 ocean 环境里的 torch 导入会触发 DLL 问题；这里用 npy 保存中间样本。
        np.save(os.path.join(SAVE_DIR, "x.npy"), np.stack(all_x))

        np.save(os.path.join(SAVE_DIR, "x_masks.npy"), np.stack(all_masks))

        np.save(os.path.join(SAVE_DIR, "y.npy"), np.stack(all_y))

        print(f"\n[OK] Processing complete. Samples: {len(all_x)}")

        print("Saved files:")
        print(f"  - {SAVE_DIR}/x.npy")
        print(f"  - {SAVE_DIR}/x_masks.npy")
        print(f"  - {SAVE_DIR}/y.npy")
        print(f"  - {wind_path}")

        return len(all_x)

    else:
        raise NoSampleError(
            "no sample generated; check report time or source forecasts"
        )


def process_typhoon_url(json_url, report_dt):
    """请求单个台风详情并生成样本；失败时抛 TyphoonDataError。"""

    json_data = fetch_json(json_url)
    return process_typhoon(json_data, report_dt)


def main():

    args = parse_args()

    total = process_typhoon_url(args.json_url, args.report_dt)

    print(f"\nAll processing complete. Samples: {total}")


if __name__ == "__main__":
    main()
