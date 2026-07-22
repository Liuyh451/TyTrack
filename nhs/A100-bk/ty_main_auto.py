import argparse
import importlib.util
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# 获取当前脚本所在绝对路径
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取某年所有台风列表的 API 地址
YEAR_LIST_URL = "https://cdn.oss.wushikj.com/data/typhoon/year/{year}.json"
# 获取单个台风详细信息的 API 地址
DETAIL_URL = "https://cdn.oss.wushikj.com/data/typhoon/{year}/{typhoon_id}.json"
# 每天固定的四个起报时间（北京时间）
REPORT_HOURS = (2, 8, 14, 20)
# 推理所需的核心样本文件清单
SAMPLE_FILES = ("x.npy", "x_masks.npy", "y.npy")


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
    parser = argparse.ArgumentParser(description="Run active typhoon task once.")
    # 生产环境不用传参数；这个参数只用于本地复现某个当前时间。
    parser.add_argument("--now", help="Test time, e.g. 202606241849.")
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
    matches = list(output_dir.glob(f"*_{report_key}.txt"))
    return len(matches) > 0


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
        ]
        
        # 启动子进程执行推理脚本
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
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


def process_one_typhoon(typhoon, report_dt):
    # 获取当前台风编号及目标起报时间字符串格式
    ty_code = str(typhoon["ty_code"])
    report_key = report_dt.strftime("%Y%m%d%H")
    
    # 确定样本存放路径
    sample_dir = sample_dir_for(typhoon, report_key)

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


def main():
    # 脚本的主入口点
    try:
        args = parse_args()
        run_once(parse_now(args.now))
    except Exception:
        LOGGER.exception("task failed")


if __name__ == "__main__":
    main()