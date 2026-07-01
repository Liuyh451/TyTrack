import argparse
import importlib.util
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_LIST_URL = "https://cdn.oss.wushikj.com/data/typhoon/year/{year}.json"
DETAIL_URL = "https://cdn.oss.wushikj.com/data/typhoon/{year}/{typhoon_id}.json"
REPORT_HOURS = (2, 8, 14, 20)
SAMPLE_FILES = ("x.npy", "x_masks.npy", "y.npy")


class TyphoonTaskError(Exception):
    """主控任务异常。"""


def load_data_module():
    """数据脚本文件名里有空格，使用 importlib 按路径加载成库。"""

    module_path = SCRIPT_DIR / "get_typhoon_data _no_gt.py"
    spec = importlib.util.spec_from_file_location("nhs_typhoon_data", module_path)
    if spec is None or spec.loader is None:
        raise TyphoonTaskError(f"cannot load data module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA_MODULE = load_data_module()


def setup_logger():
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"auto_demo_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("NhsAutoDemo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Run active typhoon task once.")
    # 生产环境不用传参数；这个参数只用于本地复现某个当前时间。
    parser.add_argument("--now", help="Test time, e.g. 202606241849.")
    return parser.parse_args()


def parse_now(value):
    if not value:
        return datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)

    for fmt in ("%Y%m%d%H%M", "%Y%m%d%H", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise TyphoonTaskError("--now must use YYYYMMDDHHMM, YYYYMMDDHH, or YYYY-MM-DDTHH:MM:SS")


def nearest_report_time(now):
    """返回最近且不在未来的北京时起报时间，例如 18:49 -> 当天 14:00。"""

    report_date = now.date()
    chosen_hour = None

    for hour in REPORT_HOURS:
        if now.hour >= hour:
            chosen_hour = hour

    if chosen_hour is None:
        report_date -= timedelta(days=1)
        chosen_hour = REPORT_HOURS[-1]

    return datetime.combine(report_date, datetime.min.time()).replace(hour=chosen_hour)


def fetch_json(url):
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
        raise TyphoonTaskError(f"unexpected year list payload: {type(payload).__name__}")

    return [item for item in payload if item.get("current") == 1 and item.get("ty_code")]


def sample_dir_for(typhoon, report_key):
    """样本固定保存到脚本目录下的 data/{ty_code}_{ename}/{report_time}。"""

    ty_code = str(typhoon["ty_code"])
    ty_name = str(typhoon.get("ename") or ty_code).lower().strip()
    return SCRIPT_DIR / "data" / f"{ty_code}_{ty_name}" / report_key


def sample_exists(sample_dir):
    """三个核心样本文件都存在，才认为这个台风时次已经处理过。"""

    return all((sample_dir / name).exists() for name in SAMPLE_FILES)


def run_inference(typhoon, sample_dir, report_key):
    """调用真实推理脚本，推理脚本结构保持不动，只调整输入参数。"""

    try:
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
        ]
        LOGGER.info("infer command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1200,
        )
        if result.stdout:
            LOGGER.info("infer stdout:\n%s", result.stdout)
        if result.stderr:
            LOGGER.warning("infer stderr:\n%s", result.stderr)
        if result.returncode != 0:
            raise TyphoonTaskError(f"infer failed, return code={result.returncode}")
    except Exception:
        LOGGER.exception("inference failed: typhoon=%s report_time=%s", typhoon.get("ty_code"), report_key)


def process_one_typhoon(typhoon, report_dt):
    ty_code = str(typhoon["ty_code"])
    report_key = report_dt.strftime("%Y%m%d%H")
    sample_dir = sample_dir_for(typhoon, report_key)

    if sample_exists(sample_dir):
        LOGGER.info("sample exists, skip data fetch: typhoon=%s report_time=%s", ty_code, report_key)
    else:
        detail_url = DETAIL_URL.format(year=ty_code[:4], typhoon_id=ty_code)
        LOGGER.info("fetch sample: typhoon=%s report_time=%s", ty_code, report_key)
        DATA_MODULE.process_typhoon_url(detail_url, report_dt)

    if sample_exists(sample_dir):
        run_inference(typhoon, sample_dir, report_key)
    else:
        LOGGER.warning("sample missing, skip inference: typhoon=%s report_time=%s", ty_code, report_key)


def run_once(now=None):
    """单次调度：确定起报时间 -> 找活跃台风 -> 拉样本 -> 调推理 demo。"""

    now = now or datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    report_dt = nearest_report_time(now)
    year = report_dt.year

    LOGGER.info("current Beijing time: %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("selected report time: %s", report_dt.strftime("%Y%m%d%H"))

    active_typhoons = fetch_active_typhoons(year)
    LOGGER.info(
        "active typhoons: %s",
        ", ".join(f"{item.get('ty_code')}({item.get('ename')})" for item in active_typhoons) or "none",
    )

    for typhoon in active_typhoons:
        try:
            process_one_typhoon(typhoon, report_dt)
        except DATA_MODULE.NoSampleError as exc:
            LOGGER.warning(
                "no sample, skip typhoon: typhoon=%s report_time=%s reason=%s",
                typhoon.get("ty_code"),
                report_dt.strftime("%Y%m%d%H"),
                exc,
            )
        except Exception:
            LOGGER.exception("typhoon failed: typhoon=%s", typhoon.get("ty_code"))


def main():
    try:
        args = parse_args()
        # run_once(parse_now(args.now))
        run_once(parse_now("202606241849"))
    except Exception:
        LOGGER.exception("task failed")


if __name__ == "__main__":
    main()
