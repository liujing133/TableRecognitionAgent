"""
EasyOCR 文字提取模块，替代PaddleOCR规避Windows底层BUG
"""
import re
import cv2
import numpy as np
import easyocr
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
        self._reader = None

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        logger.info("加载EasyOCR中英文模型...")
        # 改成项目根models目录，关闭自动下载
        model_dir = r"./table_agent/models"
        self._reader = easyocr.Reader(
            ['ch_sim', 'en'],
            model_storage_directory=model_dir,
            download_enabled=False
        )
        logger.info("EasyOCR 加载完成")
        return self._reader

    @timeit
    def extract(self, img: np.ndarray, trace_id: str = "") -> OCRResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        validate_image(img, "ocr_input")
        reader = self._get_reader()
        drop_score = self._cfg.get("drop_score", 0.5)

        # 限制大图，降低卡顿与崩溃概率
        h, w = img.shape[:2]
        max_side = 1200
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        raw_results = reader.readtext(img)
        if not raw_results:
            logger.info(f"{tag} OCR未识别到文字（空白表格区域）")
            return OCRResult()

        blocks = []
        confidences = []
        for polygon, text, conf in raw_results:
            if conf < drop_score:
                continue
            pts = np.array(polygon, dtype=np.int32)
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            text = self._postprocess_text(text)
            blocks.append(TextBlock(
                text=text,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                raw_polygon=polygon
            ))
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