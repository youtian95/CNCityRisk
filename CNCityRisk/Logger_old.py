"""
日志模块：统一配置 CNCityRisk 的文件日志与终端日志。

使用流程：程序入口先调用 ``init_root_logger()``，各模块再通过
``get_logger(__name__)`` 获取日志记录器。初始化后，写入 ``sys.stderr`` 的
未捕获异常、警告及第三方库错误会同时显示在 CMD 中并保存到当前日志文件；
进度条的动态刷新片段仅显示在终端，不写入日志。

输入：日志文件名、日志级别及是否捕获标准错误流。
输出：已配置的 ``logging.Logger``；日志文件保存在 ``CNCityRisk/logs``。
"""

from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import sys
import threading
from typing import TextIO


# 日志文件统一存放在包目录下，方便从源码运行和打包后定位。
LOG_DIR = Path(__file__).parent / "logs"

# 避免主进程重复添加处理器或重复包装标准错误流。
_logger_initialized = False


class _StderrTee:
    """将标准错误流同时写回终端，并逐行转交给日志记录器。"""

    def __init__(self, original_stream: TextIO, logger: logging.Logger):
        """保存原始错误流和目标记录器，并初始化线程安全的行缓冲区。"""
        self._original_stream = original_stream
        self._logger = logger
        self._buffer = ""
        self._lock = threading.RLock()
        self._local = threading.local()

    def write(self, message: str) -> int:
        """保留 CMD 原始输出，并将非进度条的完整 stderr 文本写入日志。"""
        if not message:
            return 0

        # 日志处理器自身报错时可能再次写入 stderr；此处防止无限递归。
        if getattr(self._local, "is_logging", False):
            return self._original_stream.write(message)

        with self._lock:
            written = self._original_stream.write(message)
            self._original_stream.flush()

            # tqdm 等进度条使用回车符原地刷新；这些片段只用于终端显示，
            # 若在每次 flush 时写入文件，会制造大量伪 ERROR 日志。
            if "\r" in message:
                return written if written is not None else len(message)

            self._buffer += message

            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._log_line(line.rstrip("\r"))

        return written if written is not None else len(message)

    def flush(self) -> None:
        """刷新终端，并把尚未以换行符结尾的错误文本写入日志。"""
        with self._lock:
            self._original_stream.flush()
            if self._buffer:
                line = self._buffer.rstrip("\r")
                self._buffer = ""
                self._log_line(line)

    def _log_line(self, line: str) -> None:
        """记录一行非空错误文本，同时标记来源为标准错误流。"""
        if not line.strip():
            return

        self._local.is_logging = True
        try:
            self._logger.error("[stderr] %s", line)
        finally:
            self._local.is_logging = False

    @property
    def encoding(self):
        """沿用原始流编码，兼容检查 ``sys.stderr.encoding`` 的第三方库。"""
        return getattr(self._original_stream, "encoding", None)

    @property
    def errors(self):
        """沿用原始流的编码错误处理方式。"""
        return getattr(self._original_stream, "errors", None)

    def isatty(self) -> bool:
        """返回原始流是否连接到交互式终端。"""
        return self._original_stream.isatty()

    def fileno(self) -> int:
        """返回原始流文件描述符，供需要底层句柄的库使用。"""
        return self._original_stream.fileno()

    def __getattr__(self, name):
        """把未显式实现的流属性转交给原始 stderr，保持第三方库兼容性。"""
        return getattr(self._original_stream, name)


def setup_logger(name=None, log_file=None, level=logging.INFO, console_output=True):
    """配置一个同时写入 UTF-8 文件和可选终端的日志记录器。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 未指定文件名时，使用当前时间保证不同运行不会互相覆盖。
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{timestamp}.log"
    else:
        log_path = LOG_DIR / log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清理当前记录器的旧处理器，避免重复输出同一条日志。
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console_output:
        # 使用 stdout，避免 stderr 捕获器再次捕获 logging 自己的终端输出。
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(module_name=None):
    """按模块名获取记录器；日志会向已配置的根记录器传播。"""
    logger_name = module_name if module_name else "CNCityRisk"
    return logging.getLogger(logger_name)


def init_root_logger(log_file=None, level=logging.INFO, capture_stderr=True):
    """仅在主进程初始化根记录器，并可捕获所有 Python 标准错误输出。"""
    global _logger_initialized

    # Windows 多进程会重新导入入口模块，子进程不应重复创建日志文件。
    if multiprocessing.current_process().name != "MainProcess":
        return None

    if _logger_initialized:
        return logging.getLogger()

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"CNCityRisk_{timestamp}.log"

    root_logger = setup_logger(name=None, log_file=log_file, level=level)

    # 在记录初始化信息之后再替换 stderr，确保捕获器使用完整配置的根记录器。
    if capture_stderr and not isinstance(sys.stderr, _StderrTee):
        sys.stderr = _StderrTee(sys.stderr, root_logger)

    _logger_initialized = True
    root_logger.info("Logging system initialized")
    root_logger.info("Log file saved at: %s", LOG_DIR / log_file)

    return root_logger
