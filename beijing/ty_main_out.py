import time
import datetime
import subprocess
import traceback
import logging
import sys
from pathlib import Path
import make_json_inst_auto_out

current_dir = Path(__file__).resolve().parent


# =========================
# 日志配置
# =========================
def setup_logger():
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"task_{datetime.datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("TyphoonTask")
    logger.setLevel(logging.INFO)

    # 清除可能存在的旧 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件 handler（追加，按天区分文件名）
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

# =========================
# 测试配置开关
# =========================
TEST_MODE = False  # 生产环境设为 False，使用 Crontab 定时调用时自动判断


MAX_REPORT_TIME_LOOKBACKS = 4


def get_target_report_time(now):
    """
    根据北京时间返回最近一个已完成的 00/06/12/18 起报时次。
    例如 11:52 -> 当日 06:00:00，00:20 -> 当日 00:00:00。
    """
    target_hour = (now.hour // 6) * 6
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    return target_time.strftime("%Y-%m-%d %H:00:00")


def get_candidate_report_times(now, max_lookbacks=MAX_REPORT_TIME_LOOKBACKS):
    """
    从当前 6 小时时次开始向前回退，避免最新时次尚未落盘时一直空跑。
    例如 00:42 -> 00:00, 前一日 18:00, 前一日 12:00, 前一日 06:00。
    """
    target_hour = (now.hour // 6) * 6
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    return [
        (target_time - datetime.timedelta(hours=6 * i)).strftime("%Y-%m-%d %H:00:00")
        for i in range(max_lookbacks)
    ]


def run_once(report_time):
    """
    report_time: 格式为 "YYYY-MM-DD HH:00:00"
    """
    try:
        logger.info(f"开始请求数据（拉起子进程）: {report_time}")

        success_list = make_json_inst_auto_out.fetch_and_process(report_time)
        # xuhao_list = [20260045]  # 您的自定义列表（覆盖返回值）
        if not success_list:
            logger.warning("没有需要处理的台风")
            return []
        for success_item in success_list:
            ty_number, report_time, *typhoon_meta = success_item
            engname = typhoon_meta[0] if len(typhoon_meta) > 0 else ""
            tfbh = typhoon_meta[1] if len(typhoon_meta) > 1 else ""
            # 启动外部推理脚本
            logger.info(f"启动推理进程: typhoon {ty_number}, report_time {report_time}")
            logger.info(f"Typhoon metadata: engname {engname}, tfbh {tfbh}")
            result = subprocess.run(
                [
                    "python",
                    "infer_case_auto_plot.py",
                    "--ty_number",
                    str(ty_number),
                    "--report_time",
                    str(report_time),
                    "--engname",
                    str(engname),
                    "--tfbh",
                    str(tfbh),
                ],
                cwd=str(current_dir),
                capture_output=True,
                text=True,
                timeout=1200,
            )

            logger.info("--- 子进程标准输出 (stdout) ---")
            logger.info(result.stdout)

            if result.returncode != 0:
                logger.error(
                    f"❌ infer_case_auto_plot.py 运行失败 (返回码 {result.returncode})"
                )
                logger.error("--- 子进程错误输出 (stderr) ---")
                logger.error(result.stderr)
            else:
                logger.info("✅ infer_case_auto_plot.py 运行成功")

        return success_list

    except Exception:
        logger.exception("run_once 异常:")
        return []


def main():
    logger.info("系统启动（单次执行模式）")

    # 测试模式：使用固定时间
    if TEST_MODE:
        logger.info("⚠️ 当前处于【测试模式】，使用固定测试时间...")
        test_run_time = "2026-06-04 00:00:00"  # 可修改为需要的测试时间
        run_once(test_run_time)
        logger.info("测试执行完毕。")
        return

    # 正常模式：根据当前时间自动判断应该请求的 report_time
    beijing_tz = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(beijing_tz)
    candidate_report_times = get_candidate_report_times(now)
    logger.info(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"本次候选数据时间: {candidate_report_times}")

    for report_time in candidate_report_times:
        logger.info(f"尝试请求数据时间: {report_time}")
        success_list = run_once(report_time)
        if success_list:
            logger.info(f"数据时间 {report_time} 处理成功，停止回退")
            break
        logger.warning(f"数据时间 {report_time} 暂无可用结果，尝试上一时次")
    else:
        logger.warning("所有候选时次均未获取到可用数据")

    logger.info("任务执行完成。")


if __name__ == "__main__":
    main()
