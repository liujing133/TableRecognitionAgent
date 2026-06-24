"""
PaddleOCR 文字提取模块
实现类型：[模型] PaddleOCR离线模型

职责：
  1. 对表格裁剪子图执行OCR，提取每行文字及其坐标
  2. 存储每个文字块的置信度
  3. 按行列坐标将文字块归属到单元格区域（供TSR模块使用）
  4. 后处理：过滤低置信字符、基础格式规整

为什么用PaddleOCR：
  - 中英文混合识别准确率业界领先
  - 完全离线，支持私有化部署，数据不出域
  - 轻量级模型在CPU可运行（单张<500ms）
  - 相比Tesseract，中文识别准确率显著更高

为什么不用大模型（GPT-4V/Qwen-VL）：
  - LLM每次调用需要网络请求，违反私有化部署要求
  - 成本高（约$0.01/张，批量处理不可接受）
  - PaddleOCR的准确率已满足指标要求（>=95%%字符准确率）
"""

import re
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from table_agent.utils.logger import logger

from table_agent.utils.common import get_cfg, validate_image, timeit


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    raw_polygon: List[List[int]]


@dataclass
class OCRResult:
    blocks: List[TextBlock] = field(default_factory=list)
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    has_warning: bool = False
    warning_detail: Optional[str] = None


class OCREngine:
    def __init__(self):
        self._cfg = get_cfg("ocr")
        self._ocr = None

    @timeit
    def extract(self, img: np.ndarray, trace_id: str = "") -> OCRResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        validate_image(img, "ocr_input")
        ocr = self._get_ocr()
        drop_score = self._cfg.get("drop_score", 0.5)
        raw_results = ocr.ocr(img, cls=True)

        if not raw_results or raw_results[0] is None:
            logger.info(f"{tag} OCR未识别到文字（可能为空白表格区域）")
            return OCRResult()

        blocks = []
        confidences = []

        for line in raw_results[0]:
            if line is None:
                continue
            polygon, (text, conf) = line
            if conf < drop_score:
                continue
            pts = np.array(polygon, dtype=np.int32)
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            text = self._postprocess_text(text)
            blocks.append(TextBlock(text=text, confidence=float(conf), bbox=(x1, y1, x2, y2), raw_polygon=polygon))
            confidences.append(float(conf))

        if not blocks:
            return OCRResult()

        avg_conf = float(np.mean(confidences))
        low_conf_blocks = [b for b in blocks if b.confidence < 0.7]
        has_warning = len(low_conf_blocks) > len(blocks) * 0.2
        warning_detail = None
        if has_warning:
            warning_detail = f"OCR低置信块占比过高：{len(low_conf_blocks)}/{len(blocks)}，平均置信度={avg_conf:.2f}，建议检查图像质量"
            logger.warning(f"{tag} {warning_detail}")

        logger.info(f"{tag} OCR识别{len(blocks)}个文字块，平均置信度={avg_conf:.2f}")
        return OCRResult(blocks=blocks, avg_confidence=avg_conf, low_confidence_count=len(low_conf_blocks), has_warning=has_warning, warning_detail=warning_detail)

    @timeit
    def assign_to_cells(self, ocr_result: OCRResult, cell_bboxes: List[Tuple[int, int, int, int]]) -> List[str]:
        cell_texts = [""] * len(cell_bboxes)
        for block in ocr_result.blocks:
            bx1, by1, bx2, by2 = block.bbox
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            best_idx = -1
            best_area = float("inf")
            for i, (cx1, cy1, cx2, cy2) in enumerate(cell_bboxes):
                if cx1 <= cx <= cx2 and cy1 <= cy <= cy2:
                    area = (cx2 - cx1) * (cy2 - cy1)
                    if area < best_area:
                        best_area = area
                        best_idx = i
            if best_idx >= 0:
                existing = cell_texts[best_idx]
                cell_texts[best_idx] = existing + "\n" + block.text if existing else block.text
        return cell_texts

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise RuntimeError("未找到 paddleocr 包，请执行：pip install paddlepaddle paddleocr")
        logger.info("初始化PaddleOCR（首次加载会自动下载模型，约200MB）...")
        self._ocr = PaddleOCR(
            lang=self._cfg.get("lang", "ch"),
            use_gpu=self._cfg.get("use_gpu", False),
            use_angle_cls=self._cfg.get("use_angle_cls", True),
            det_db_thresh=self._cfg.get("det_db_thresh", 0.3),
            det_db_box_thresh=self._cfg.get("det_db_box_thresh", 0.5),
            rec_batch_num=self._cfg.get("rec_batch_num", 6),
            show_log=self._cfg.get("show_log", False),
        )
        logger.info("PaddleOCR 初始化完成")
        return self._ocr

    def _postprocess_text(self, text):
        if not text:
            return text
        text = text.strip()
        text = self._full_to_half(text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _full_to_half(text):
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(" ")
            else:
                result.append(char)
        return "".join(result)
