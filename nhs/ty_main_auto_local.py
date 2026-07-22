import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import get_real_track_until_report_time as REAL_TRACK
import evaluate_mde as MDE_EVALUATOR

# 获取当前脚本所在绝对路径
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取某年所有台风列表的 API 地址
YEAR_LIST_URL = "https://cdn.oss.wushikj.com/data/typhoon/year/{year}.json"
# 获取单个台风详细信息的 API 地址
DETAIL_URL = "https://cdn.oss.wushikj.com/data/typhoon/{year}/{typhoon_id}.json"
# 每天固定的四个起报时间（北京时间）
REPORT_HOURS = (2, 8, 14, 20)
REAL_TRACK_LOOKBACK_STEPS = 4
# 推理所需的核心样本文件清单
SAMPLE_FILES = ("x.npy", "x_masks.npy", "y.npy")
TEST_TY_CODE = "2609"
TEST_REPORT_TIME = "2026071008"
MODEL_RUNTIME_DIR = SCRIPT_DIR.parent / "beijing"
MDE_LOOKBACK_REPORTS = 0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


class TyphoonTaskError(Exception):
    """主控任务异常。"""


def load_data_module():
    """数据脚本文件名里有空格，使用 importlib 按路径加载成库。"""
    # 指定需要动态加载的获取数据脚本路径
    module_path = SCRIPT_DIR / "get_typhoon_data _no_gt.py"
    spec = importlib.util.spec_from_file_location("nhs_typhoon_data", module_path)
    if spec is None or spec.loader is None:
        raise TyphoonTaskError(f"cannot load data module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 加载外部数据处理模块
DATA_MODULE = load_data_module()


def setup_logger():
    # 在当前脚本目录下创建日志文件夹
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    # 日志文件按天命名
    log_file = log_dir / f"auto_demo_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("NhsAutoDemo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 设置统一的日志格式：时间 - 级别 - 信息
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 配置文件输出处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 配置控制台输出处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 初始化全局日志记录器
LOGGER = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the typhoon test case or live task.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the original active-typhoon scheduler instead of the fixed test case.",
    )
    parser.add_argument(
        "--now",
        help="Current Beijing time used with --live, e.g. 202606241849.",
    )
    return parser.parse_args()


def parse_now(value):
    # 如果没有提供时间参数，则默认获取当前北京时间
    if not value:
        return datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)

    # 尝试解析用户传入的各种时间格式
    for fmt in ("%Y%m%d%H%M", "%Y%m%d%H", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    # 如果格式都不匹配，抛出异常
    raise TyphoonTaskError(
        "--now must use YYYYMMDDHHMM, YYYYMMDDHH, or YYYY-MM-DDTHH:MM:SS"
    )


def nearest_report_time(now):
    """返回最近且不在未来的北京时起报时间，例如 18:49 -> 当天 14:00。"""
    report_date = now.date()
    chosen_hour = None

    # 遍历规定的起报时间，找到最后一个小于当前时间的小时
    for hour in REPORT_HOURS:
        if now.hour >= hour:
            chosen_hour = hour

    # 如果当前时间早于当天的第一个起报时间(比如凌晨1点)，则倒退回前一天的最后一次起报时间(20点)
    if chosen_hour is None:
        report_date -= timedelta(days=1)
        chosen_hour = REPORT_HOURS[-1]

    return datetime.combine(report_date, datetime.min.time()).replace(hour=chosen_hour)


def fetch_json(url):
    # 封装带有超时控制和错误处理的 HTTP GET 请求
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise TyphoonTaskError(f"request failed: {url}: {exc}") from exc
    except ValueError as exc:
        raise TyphoonTaskError(f"invalid JSON: {url}: {exc}") from exc


def fetch_active_typhoons(year):
    """从年度列表中筛选 current=1 的活跃台风，可能同时返回多个。"""
    payload = fetch_json(YEAR_LIST_URL.format(year=year))
    if not isinstance(payload, list):
        raise TyphoonTaskError(
            f"unexpected year list payload: {type(payload).__name__}"
        )

    # 过滤出当前正在活跃（current=1）且具备台风编号（ty_code）的数据
    return [
        item for item in payload if item.get("current") == 1 and item.get("ty_code")
    ]


def sample_dir_for(typhoon, report_key):
    """样本固定保存到脚本目录下的 data/{ty_code}_{ename}/{report_time}。"""
    ty_code = str(typhoon["ty_code"])
    # 获取台风英文名并格式化（转小写、去首尾空格），若没有则直接使用编号替代
    ty_name = str(typhoon.get("ename") or ty_code).lower().strip()
    # 拼接出样本保存的具体路径
    return SCRIPT_DIR / "data" / f"{ty_code}_{ty_name}" / report_key


def sample_exists(sample_dir):
    """三个核心样本文件都存在，才认为这个台风时次已经处理过。"""
    return all((sample_dir / name).exists() for name in SAMPLE_FILES)


def inference_output_exists(ty_code, report_key):
    """检查 output 目录下是否已经存在当前台风和起报时间的推理结果 txt。"""
    output_dir = SCRIPT_DIR / "output" / str(ty_code)

    # 如果输出目录不存在，那结果肯定不在
    if not output_dir.exists():
        return False

    # 匹配结尾是 _{report_key}.txt 的文件
    # 例如: T_SEVP_C_SCSIOEns_20260624130327_P_TYPHOON_TF_202608_2026062414.txt
    matches = [
        path
        for path in output_dir.glob(f"*_{report_key}.txt")
        if not path.name.startswith("real_track_")
    ]
    return len(matches) > 0


def select_real_track_points(payload, report_dt, steps=REAL_TRACK_LOOKBACK_STEPS):
    for index in range(steps):
        observation_dt = report_dt - timedelta(hours=6 * index)
        points = REAL_TRACK.extract_real_track(
            payload, observation_dt.strftime("%Y%m%d%H")
        )
        expected_time = observation_dt.strftime("%Y-%m-%d %H:%M:%S")
        if points and points[-1]["time"] == expected_time:
            return points, observation_dt
    return [], None


def fetch_real_track(ty_code, report_dt, output_path):
    normalized_ty_code = REAL_TRACK.normalize_ty_code(ty_code)
    report_key = report_dt.strftime("%Y%m%d%H")
    try:
        payload = REAL_TRACK.load_typhoon_json(normalized_ty_code)
        points, observation_dt = select_real_track_points(payload, report_dt)
        if observation_dt is None:
            LOGGER.warning(
                "real track unavailable in fallback window: "
                "typhoon=%s requested_report_time=%s",
                normalized_ty_code,
                report_key,
            )
            return None
        added_count = REAL_TRACK.save_track_txt(points, output_path)
        LOGGER.info(
            "real track updated: typhoon=%s requested_report_time=%s "
            "observation_time=%s received=%s appended=%s path=%s",
            normalized_ty_code,
            report_key,
            observation_dt.strftime("%Y%m%d%H"),
            len(points),
            added_count,
            output_path,
        )
        return output_path
    except Exception as exc:
        LOGGER.warning(
            "real track fetch failed: typhoon=%s report_time=%s reason=%s",
            normalized_ty_code,
            report_key,
            exc,
        )
        return None


def run_inference(typhoon, sample_dir, report_key):
    """调用真实推理脚本，推理脚本结构保持不动，只调整输入参数。"""
    try:
        # 构建执行命令及参数列表
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "infer_case_auto_plot.py"),
            "--data_path",
            str(SCRIPT_DIR / "data"),
            "--ty_number",
            str(typhoon["ty_code"]),
            "--ty_name",
            str(typhoon.get("ename") or typhoon["ty_code"]),
            "--report_time",
            report_key,
            "--trgpath",
            str(MODEL_RUNTIME_DIR / "checkpoints"),
        ]

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            path
            for path in (str(MODEL_RUNTIME_DIR), existing_pythonpath)
            if path
        )
        env["PYTHONIOENCODING"] = "utf-8"
        
        # 启动子进程执行推理脚本
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            capture_output=True, # 捕获标准输出和标准错误
            text=True,           # 以字符串形式返回输出
            encoding="utf-8",
            errors="replace",
            timeout=1200,        # 设置超时时间为 20 分钟
        )
        
        # 记录推理过程中的标准输出日志
        if result.stdout:
            LOGGER.info("infer stdout:\n%s", result.stdout)
        # 记录推理过程中的警告或错误输出日志
        if result.stderr:
            LOGGER.warning("infer stderr:\n%s", result.stderr)
            
        # 如果返回码不为 0，说明脚本执行失败，抛出异常
        if result.returncode != 0:
            raise TyphoonTaskError(f"infer failed, return code={result.returncode}")
    except Exception:
        # 捕获并记录推理过程中的所有未预期异常
        LOGGER.exception(
            "inference failed: typhoon=%s report_time=%s",
            typhoon.get("ty_code"),
            report_key,
        )


def run_mde_evaluation(ty_code, report_key, real_track_path):
    try:
        result = MDE_EVALUATOR.evaluate_recent_forecasts(
            ty_code=ty_code,
            current_report_time=report_key,
            npy_root=SCRIPT_DIR / "output_npy",
            real_track_path=real_track_path,
            eval_root=SCRIPT_DIR / "eval",
            lookback=MDE_LOOKBACK_REPORTS,
        )
        LOGGER.info(
            "MDE evaluation complete: typhoon=%s report_time=%s forecasts=%s "
            "matched=%s added=%s csv=%s",
            ty_code,
            report_key,
            result["forecast_count"],
            result["matched_count"],
            result["added_count"],
            result["csv_path"],
        )
    except Exception:
        LOGGER.exception(
            "MDE evaluation failed: typhoon=%s report_time=%s",
            ty_code,
            report_key,
        )


def process_one_typhoon(typhoon, report_dt):
    # 获取当前台风编号及目标起报时间字符串格式
    ty_code = str(typhoon["ty_code"])
    report_key = report_dt.strftime("%Y%m%d%H")
    
    # 确定样本存放路径
    sample_dir = sample_dir_for(typhoon, report_key)
    normalized_ty_code = REAL_TRACK.normalize_ty_code(ty_code)
    real_track_path = sample_dir.parent / f"real_track_{normalized_ty_code}.txt"
    fetch_real_track(ty_code, report_dt, real_track_path)

    # 检查样本是否已存在，不存在则请求 API 拉取数据
    if sample_exists(sample_dir):
        LOGGER.info(
            "sample exists, skip data fetch: typhoon=%s report_time=%s",
            ty_code,
            report_key,
        )
    else:
        detail_url = DETAIL_URL.format(year=ty_code[:4], typhoon_id=ty_code)
        LOGGER.info("fetch sample: typhoon=%s report_time=%s", ty_code, report_key)
        # 调用外部模块下载处理数据
        DATA_MODULE.process_typhoon_url(detail_url, report_dt)

    # 再次检查样本是否存在（确保下载成功后才能进行后续推理）
    if sample_exists(sample_dir):
        # 新增检查逻辑：如果 txt 输出文件已存在，则跳过推理
        if inference_output_exists(ty_code, report_key):
            LOGGER.info(
                "inference output txt already exists, skip inference: typhoon=%s report_time=%s",
                ty_code,
                report_key,
            )
        else:
            # 样本齐备且结果未生成时，执行推理任务
            run_inference(typhoon, sample_dir, report_key)
    else:
        # 样本确实不存在或下载失败，跳过推理
        LOGGER.warning(
            "sample missing, skip inference: typhoon=%s report_time=%s",
            ty_code,
            report_key,
        )

    run_mde_evaluation(ty_code, report_key, real_track_path)


def run_once(now=None):
    """单次调度：确定起报时间 -> 找活跃台风 -> 拉样本 -> 调推理 demo。"""
    # 确定基准时间
    now = now or datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    # 获取最近有效的起报时间及对应的年份
    report_dt = nearest_report_time(now)
    year = report_dt.year

    # 打印当前时间和匹配到的起报时间信息
    LOGGER.info("current Beijing time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("selected report time: %s", report_dt.strftime("%Y%m%d%H"))

    # 从远程接口拉取本年度当前正在活跃的台风列表
    active_typhoons = fetch_active_typhoons(year)
    LOGGER.info(
        "active typhoons: %s",
        ", ".join(
            f"{item.get('ty_code')}({item.get('ename')})" for item in active_typhoons
        )
        or "none",
    )

    # 遍历每个活跃台风，依次处理
    for typhoon in active_typhoons:
        try:
            process_one_typhoon(typhoon, report_dt)
        except DATA_MODULE.NoSampleError as exc:
            # 捕获已知的数据缺失异常并记录警告
            LOGGER.warning(
                "no sample, skip typhoon: typhoon=%s report_time=%s reason=%s",
                typhoon.get("ty_code"),
                report_dt.strftime("%Y%m%d%H"),
                exc,
            )
        except Exception:
            # 捕获其余未知异常，确保不影响列表中其他台风的处理
            LOGGER.exception("typhoon failed: typhoon=%s", typhoon.get("ty_code"))


def run_test_case():
    ty_code = REAL_TRACK.normalize_ty_code(TEST_TY_CODE)
    report_dt = datetime.strptime(TEST_REPORT_TIME, "%Y%m%d%H")
    detail_url = DETAIL_URL.format(year=ty_code[:4], typhoon_id=ty_code)
    payload = fetch_json(detail_url)
    typhoon = {
        "ty_code": str(payload.get("ty_code") or ty_code),
        "ename": str(payload.get("ename") or ty_code),
    }

    LOGGER.info(
        "test mode: typhoon=%s(%s) report_time=%s",
        typhoon["ty_code"],
        typhoon["ename"],
        TEST_REPORT_TIME,
    )
    process_one_typhoon(typhoon, report_dt)


def main():
    # 脚本的主入口点
    try:
        args = parse_args()
        if args.live:
            run_once(parse_now(args.now))
        else:
            run_test_case()
    except Exception:
        LOGGER.exception("task failed")


if __name__ == "__main__":
    main()
