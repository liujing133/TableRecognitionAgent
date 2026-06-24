"""
日志工具模块 - 兼容 loguru 风格的标准 logging 封装
实现类型：[规则]

如果环境中安装了 loguru，优先使用 loguru；
否则自动回退到 Python 标准库 logging，提供兼容接口。

用法（与 loguru 完全相同）：
    from utils.logger import logger
    logger.info("xxx")
    logger.debug("xxx")
    logger.warning("xxx")
    logger.error("xxx")
"""

import sys
import logging
from pathlib import Path

# ── 尝试使用 loguru ──────────────────────────

_USE_LOGURU = False
logger = None

try:
    from loguru import logger as _loguru_logger

    # 移除默认 handler，添加带格式的 stderr handler
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        colorize=True,
    )
    logger = _loguru_logger
    _USE_LOGURU = True
except ImportError:
    pass

# ── 回退到标准 logging ──────────────────────

if logger is None:

    class _CompatibleLogger:
        """
        与 loguru 使用方式兼容的 logging 包装器。

        覆盖 loguru 的 .debug/.info/.warning/.error 方法，
        使代码在无 loguru 的环境中也可以运行。
        """

        def __init__(self):
            self._logger = logging.getLogger("table_agent")
            self._logger.setLevel(logging.DEBUG)

            # 避免重复添加 handler
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setLevel(logging.DEBUG)
                fmt = logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S",
                )
                handler.setFormatter(fmt)
                self._logger.addHandler(handler)

            self._level = logging.DEBUG  # 默认级别

        def debug(self, msg, *args, **kwargs):
            self._logger.debug(str(msg), *args, **kwargs)

        def info(self, msg, *args, **kwargs):
            self._logger.info(str(msg), *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self._logger.warning(str(msg), *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            self._logger.error(str(msg), *args, **kwargs)

        def exception(self, msg, *args, **kwargs):
            self._logger.exception(str(msg), *args, **kwargs)

        def remove(self):
            """兼容 loguru.remove()"""
            self._logger.handlers.clear()

        def add(self, sink, level="DEBUG", format=None, colorize=None, **kwargs):
            """兼容 loguru.add()"""
            if isinstance(sink, str) and Path(sink).suffix:
                # 文件 sink
                fh = logging.FileHandler(sink, encoding="utf-8")
                fh.setLevel(getattr(logging, level.upper(), logging.DEBUG))
                if format:
                    fh.setFormatter(logging.Formatter(format))
                self._logger.addHandler(fh)
            else:
                # 流 sink
                handler = logging.StreamHandler(sink or sys.stderr)
                handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
                if format:
                    handler.setFormatter(logging.Formatter(format.replace("<green>", "").replace("</green>", "")
                                                            .replace("<level>", "").replace("</level>", "")
                                                            .replace("{time:", "%(asctime")  # minimal compat
                                                            .replace("{level}", "%(levelname)-7s")
                                                            .replace("{message}", "%(message)s")))
                self._logger.addHandler(handler)

        def _get_level(self):
            return self._logger.level

        def _set_level(self, level):
            self._logger.setLevel(level)

        # 底层 logger 对象（供需要直接访问的地方使用）
        @property
        def core(self):
            return self._logger

    logger = _CompatibleLogger()

# 判断是否正在使用 loguru（供调试工具检查）
def is_loguru() -> bool:
    return _USE_LOGURU
