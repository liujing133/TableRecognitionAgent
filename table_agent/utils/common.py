"""
工具函数模块
实现类型：[规则] 纯工具函数，无模型调用

职责：
  1. YAML配置加载与缓存
  2. trace_id生成（配合组长全链路追踪）
  3. 函数耗时计时装饰器
  4. 图像合法性校验
  5. 置信度分级映射
  6. 调试图像保存工具
"""

import os
import cv2
import uuid
import yaml
import time
import numpy as np
from pathlib import Path
from table_agent.utils.logger import logger


# 配置文件路径
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
_config_cache = None


def load_config() -> dict:
    """加载yaml配置，带缓存"""
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"配置文件不存在：{_CONFIG_PATH}")
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_cfg(section: str) -> dict:
    """获取指定配置节"""
    return load_config().get(section, {})


def new_trace_id() -> str:
    """生成全局唯一trace_id"""
    return str(uuid.uuid4())


def timeit(fn):
    """
    装饰器：记录函数耗时到日志。
    使用 try/finally 确保即使函数抛出异常也能记录耗时。
    """
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            return result
        finally:
            elapsed = (time.time() - t0) * 1000
            logger.debug(f"[{fn.__module__}.{fn.__name__}] 耗时 {elapsed:.1f} ms")
    wrapper.__name__ = fn.__name__
    return wrapper


def validate_image(img: np.ndarray, name: str = "image") -> None:
    """
    检查numpy图像合法性，不合法直接抛异常。
    实现类型：[规则]
    """
    if img is None:
        raise ValueError(f"{name} 为 None，图像加载失败")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} 类型错误，期望 np.ndarray，得到 {type(img)}")
    if img.ndim not in (2, 3):
        raise ValueError(f"{name} 维度错误：{img.ndim}，期望 2（灰度）或 3（BGR/RGB）")
    if img.size == 0:
        raise ValueError(f"{name} 是空数组")


def classify_confidence(score: float) -> str:
    """
    将浮点置信度映射为等级字符串。
    实现类型：[规则]
    返回: 'high' | 'medium' | 'low'
    """
    cfg = get_cfg("confidence")
    if score >= cfg.get("high", 0.80):
        return "high"
    elif score >= cfg.get("medium", 0.60):
        return "medium"
    else:
        return "low"


def save_debug_image(
    img: np.ndarray,
    name: str,
    output_dir: str = None,
    trace_id: str = "",
) -> str:
    """
    保存调试用图像（仅在 DEBUG 模式下生效）。
    实现类型：[规则]

    Args:
        img: 要保存的图像 (BGR 或灰度)
        name: 文件名（不含后缀）
        output_dir: 输出目录，None时使用 ./debug_output/
        trace_id: 追踪ID，用于文件命名区分

    Returns:
        保存的完整路径，如果未保存则返回空字符串
    """
    if os.environ.get("TABLE_AGENT_DEBUG", "").lower() not in ("1", "true", "yes"):
        return ""

    if output_dir is None:
        output_dir = "debug_output"

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{trace_id}" if trace_id else ""
    save_path = str(save_dir / f"{name}{tag}.png")

    success = cv2.imwrite(save_path, img)
    if success:
        logger.debug(f"调试图像已保存：{save_path}")
    else:
        logger.warning(f"调试图像保存失败：{save_path}")
    return save_path
