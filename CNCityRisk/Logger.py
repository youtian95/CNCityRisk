"""
精简日志模块：统一记录 CNCityRisk 的运行信息、Python 警告和未处理异常。

使用流程：程序入口先调用 ``init_root_logger()``，其他模块通过
``get_logger(__name__)`` 获取记录器。进度条继续直接显示在终端，不写入日志。

输入：日志文件名、日志级别及是否捕获警告和未处理异常。
输出：终端日志和保存在 ``CNCityRisk/logs`` 中的 UTF-8 日志文件。
"""

from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import sys


LOG_DIR = Path(__file__).parent / "logs"
_logger_initialized = False
_original_excepthook = sys.excepthook


def get_logger(module_name=None):
    """按模块名获取记录器，并将日志交给已配置的根记录器处理。"""
    return logging.getLogger(module_name or "CNCityRisk")


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """把主线程中未处理的异常连同 traceback 写入日志。"""
    if issubclass(exc_type, KeyboardInterrupt):
        _original_excepthook(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger()
    if not logger.handlers:
        _original_excepthook(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "程序因未处理异常终止",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def init_root_logger(log_file=None, level=logging.INFO, capture_stderr=True):
    """
    在主进程初始化根记录器。

    ``capture_stderr`` 为兼容现有调用而保留；为 True 时捕获 Python warning
    和主线程未处理异常，但不会替换 ``sys.stderr``。
    """
    global _logger_initialized

    if multiprocessing.current_process().name != "MainProcess":
        return None
    if _logger_initialized:
        return logging.getLogger()

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"CNCityRisk_{timestamp}.log"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 关闭并移除旧处理器，避免重复输出或占用旧日志文件。
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    ))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    root_logger.addHandler(console_handler)

    if capture_stderr:
        # 捕获 Python warning 和未处理异常；tqdm 的 stderr 进度条不受影响。
        logging.captureWarnings(True)
        sys.excepthook = _log_uncaught_exception

    _logger_initialized = True
    root_logger.info("Logging system initialized")
    root_logger.info("Log file saved at: %s", log_path)
    return root_logger
