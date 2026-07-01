import time
import datetime
import subprocess
import traceback
import logging
import sys
from pathlib import Path
import make_json_inst_auto

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
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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


def get_target_report_time(now):
    """
    根据当前时间，返回应该请求的 report_time：
    - 如果当前时间 < 12:00，返回当日 00:00:00
    - 如果当前时间 >= 12:00，返回当日 12:00:00
    """
    if now.hour < 12:
        target_hour = 0
    else:
        target_hour = 12
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    return target_time.strftime("%Y-%m-%d %H:00:00")


def run_once(report_time):
    """
    report_time: 格式为 "YYYY-MM-DD HH:00:00"
    """
    try:
        logger.info(f"开始请求数据（拉起子进程）: {report_time}")

        success_list = make_json_inst_auto.fetch_and_process(report_time)
        # xuhao_list = [20260045]  # 您的自定义列表（覆盖返回值）
        if not success_list:
            logger.warning("没有需要处理的台风")
            return []
        for ty_number, report_time in success_list:
            # 启动外部推理脚本
            logger.info(f"启动推理进程: typhoon {ty_number}, report_time {report_time}")
            result = subprocess.run(
                [
                    "python",
                    "infer_case_auto_plot.py",
                    "--ty_number",
                    str(ty_number),
                    "--report_time",
                    str(report_time),
                ],
                cwd=str(current_dir),
                capture_output=True,
                text=True,
                timeout=1200,
            )

            logger.info("--- 子进程标准输出 (stdout) ---")
            logger.info(result.stdout)

            if result.returncode != 0:
                logger.error(f"❌ infer_case_auto_plot.py 运行失败 (返回码 {result.returncode})")
                logger.error("--- 子进程错误输出 (stderr) ---")
                logger.error(result.stderr)
            else:
                logger.info("✅ infer_case_auto_plot.py 运行成功")

    except Exception:
        logger.exception("run_once 异常:")


def main():
    logger.info("系统启动（单次执行模式）")

    # 测试模式：使用固定时间
    if TEST_MODE:
        logger.info("⚠️ 当前处于【测试模式】，使用固定测试时间...")
        test_run_time = '2026-06-04 00:00:00'   # 可修改为需要的测试时间
        run_once(test_run_time)
        logger.info("测试执行完毕。")
        return

    # 正常模式：根据当前时间自动判断应该请求的 report_time
    now = datetime.datetime.now()
    report_time = get_target_report_time(now)
    logger.info(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"本次将请求的数据时间: {report_time}")
    run_once(report_time)
    logger.info("任务执行完成。")


if __name__ == "__main__":
    main()