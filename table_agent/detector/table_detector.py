"""
YOLOv8 表格区域检测模块
实现类型：[模型] YOLOv8深度学习目标检测

职责：
  1. 加载YOLOv8权重（自训练权重 or 通用预训练权重）
  2. 对预处理后的图像推理，输出所有表格的边界框
  3. 输出含置信度的检测结果，支持低置信预警
  4. 封装为可被流水线直接调用的标准接口

为什么用YOLOv8而不用规则：
  规则方法（霍夫线检测）只能找到矩形框，
  无法区分"这是一个表格区域"还是"这是装饰边框/图片"。
  YOLOv8通过监督学习，能真正理解"表格"语义。

为什么不用大模型（LLM/VLM）：
  YOLOv8推理单张图片仅需50~200ms（CPU），
  LLM/VLM推理同等任务需要5~30秒，成本差100倍以上，
  且YOLOv8输出结构化坐标，天然满足后续需求。
"""

import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from table_agent.utils.logger import logger

from table_agent.utils.common import get_cfg, validate_image, classify_confidence, timeit

# 项目根目录（用于解析相对路径）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class DetectedTable:
    """单个检测到的表格"""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) 像素坐标
    confidence: float                  # YOLOv8检测置信度 0~1
    confidence_level: str              # 'high' | 'medium' | 'low'
    page_idx: int = 0                  # 所在页码（0-based）
    table_idx: int = 0                 # 页内表格序号（0-based）
    warning: Optional[str] = None      # 低置信时的预警信息


@dataclass
class DetectionResult:
    """整页检测结果"""
    tables: List[DetectedTable] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)   # (H, W)
    model_used: str = ""
    has_warning: bool = False


# ─────────────────────────────────────────────
# 核心类
# ─────────────────────────────────────────────

class TableDetector:
    """
    YOLOv8表格检测器
    实现类型：[模型] YOLOv8

    使用优先级：
      1. config.yaml 中 model_path 指定的自训练权重（组长提供）
      2. 若不存在，使用 yolov8n.pt 通用预训练权重兜底
    """

    def __init__(self):
        self._cfg = get_cfg("detector")
        self._model = None  # 懒加载
        self._model_name = ""

    # ── 公开入口 ──────────────────────────────

    @timeit
    def detect(
        self,
        img: np.ndarray,
        page_idx: int = 0,
        trace_id: str = "",
    ) -> DetectionResult:
        """
        对单张页面图像执行表格检测。

        Args:
            img: BGR numpy图像（经预处理模块处理后的图）
            page_idx: 当前页码，用于结果溯源
            trace_id: 全链路追踪ID

        Returns:
            DetectionResult，含所有检测到的表格及置信度
        """
        tag = f"[trace={trace_id}]" if trace_id else ""
        validate_image(img, "detector_input")

        model = self._get_model()
        h, w = img.shape[:2]

        conf_thr = self._cfg.get("conf_threshold", 0.45)
        iou_thr = self._cfg.get("iou_threshold", 0.45)
        imgsz = self._cfg.get("imgsz", 1024)
        device = self._cfg.get("device", "cpu")

        # YOLOv8推理
        results = model.predict(
            source=img,
            conf=conf_thr,
            iou=iou_thr,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        tables: List[DetectedTable] = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            logger.info(f"{tag} 第{page_idx}页未检测到表格")
            return DetectionResult(
                tables=[],
                image_shape=(h, w),
                model_used=self._model_name,
                has_warning=False,
            )

        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            level = classify_confidence(conf)

            warning = None
            if level in ("medium", "low"):
                warning = (
                    f"表格#{idx} 置信度偏低({conf:.2f})，"
                    f"建议人工复核（页码={page_idx}，坐标=[{x1},{y1},{x2},{y2}]）"
                )
                logger.warning(f"{tag} {warning}")

            tables.append(DetectedTable(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                confidence_level=level,
                page_idx=page_idx,
                table_idx=idx,
                warning=warning,
            ))

        has_warning = any(t.warning for t in tables)
        logger.info(
            f"{tag} 第{page_idx}页检测到{len(tables)}个表格，"
            f"预警={has_warning}"
        )

        return DetectionResult(
            tables=tables,
            image_shape=(h, w),
            model_used=self._model_name,
            has_warning=has_warning,
        )

    # ── 模型懒加载 ────────────────────────────

    @staticmethod
    def _resolve_model_path(path: str) -> Path:
        """
        解析模型路径：
          1. 如果是绝对路径，直接使用
          2. 如果是相对路径，先以项目根目录为基准
          3. 再以当前工作目录为基准
        """
        if not path:
            return Path()
        p = Path(path)
        if p.is_absolute():
            return p
        rooted = _PROJECT_ROOT / p
        if rooted.exists():
            return rooted
        return p

    def _get_model(self):
        """懒加载YOLOv8模型，首次调用时初始化"""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "未找到 ultralytics 包，请执行：pip install ultralytics"
            )

        # 优先使用自训练权重
        custom_path = self._resolve_model_path(self._cfg.get("model_path", ""))
        if custom_path and custom_path.exists():
            logger.info(f"加载自训练表格检测权重：{custom_path}")
            self._model = YOLO(str(custom_path))
            self._model_name = str(custom_path.name)
        else:
            # 兜底：使用YOLOv8n通用预训练权重
            fallback = self._cfg.get("fallback_model", "yolov8n.pt")
            fallback_path = self._resolve_model_path(fallback)

            if fallback_path and fallback_path.exists():
                logger.info(f"加载通用预训练权重：{fallback_path}")
                self._model = YOLO(str(fallback_path))
                self._model_name = str(fallback_path.name)
            else:
                logger.warning(
                    f"未找到自训练权重 {custom_path}，"
                    f"本地兜底权重 {fallback} 也不存在，"
                    f"将自动下载YOLOv8n通用预训练权重"
                )
                self._model = YOLO(fallback)
                self._model_name = fallback

        return self._model

    def is_model_ready(self) -> bool:
        """检查模型是否可用（用于健康检查接口）"""
        try:
            self._get_model()
            return True
        except Exception:
            return False
