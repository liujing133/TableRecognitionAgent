"""RapidOCR 表格专用OCR，替代EasyOCR，解决单字漏检、文字切割问题"""
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from table_agent.utils.logger import logger
from table_agent.utils.common import get_cfg, validate_image, timeit
from rapidocr_onnxruntime import RapidOCR

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
        logger.info("加载 RapidOCR PP-OCRv4 模型...")
        self._reader = RapidOCR()
        logger.info("RapidOCR 加载完成")
        return self._reader

    @timeit
    def extract(self, img: np.ndarray, trace_id: str = "") -> OCRResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        validate_image(img, "ocr_input")
        reader = self._get_reader()
        drop_score = self._cfg.get("drop_score", 0.5)

        # 限制大图缩放，防止OCR卡顿
        h, w = img.shape[:2]
        max_side = 1200
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # RapidOCR推理：[[[x1,y1,x2,y2], text, conf], ...]
        raw_results, elapse = reader(img, use_cls=False)
        if not raw_results:
            logger.info(f"{tag} OCR未识别到文字（空白表格区域）")
            return OCRResult()

        blocks = []
        for item in raw_results:
            polygon, text, conf = item
            # polygon格式 [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

            text = self._postprocess_text(text)
            pts_int = [[int(p[0]), int(p[1])] for p in polygon]
            blocks.append(TextBlock(
                text=text,
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                raw_polygon=pts_int
            ))

        # 同行近距离文字合并（解决空格拆分多块）
        row_groups = {}
        HORIZONTAL_GAP_THRESHOLD = 10
        for blk in blocks:
            cy = (blk.bbox[1] + blk.bbox[3]) / 2
            key = round(cy, 8)
            if key not in row_groups:
                row_groups[key] = []
            row_groups[key].append(blk)

        merged_blocks = []
        for row_blks in row_groups.values():
            row_blks.sort(key=lambda x: x.bbox[0])
            if not row_blks:
                continue
            current_group = [row_blks[0]]
            for blk in row_blks[1:]:
                last = current_group[-1]
                gap = blk.bbox[0] - last.bbox[2]
                if gap <= HORIZONTAL_GAP_THRESHOLD:
                    current_group.append(blk)
                else:
                    texts = [b.text.strip() for b in current_group]
                    full_text = " ".join(texts)
                    min_x1 = min(b.bbox[0] for b in current_group)
                    min_y1 = min(b.bbox[1] for b in current_group)
                    max_x2 = max(b.bbox[2] for b in current_group)
                    max_y2 = max(b.bbox[3] for b in current_group)
                    avg_conf = sum(b.confidence for b in current_group) / len(current_group)
                    poly = [[min_x1, min_y1], [max_x2, min_y1], [max_x2, max_y2], [min_x1, max_y2]]
                    merged_blocks.append(TextBlock(
                        text=full_text,
                        confidence=avg_conf,
                        bbox=(min_x1, min_y1, max_x2, max_y2),
                        raw_polygon=poly
                    ))
                    current_group = [blk]
            # 收尾分组
            texts = [b.text.strip() for b in current_group]
            full_text = " ".join(texts)
            min_x1 = min(b.bbox[0] for b in current_group)
            min_y1 = min(b.bbox[1] for b in current_group)
            max_x2 = max(b.bbox[2] for b in current_group)
            max_y2 = max(b.bbox[3] for b in current_group)
            avg_conf = sum(b.confidence for b in current_group) / len(current_group)
            poly = [[min_x1, min_y1], [max_x2, min_y1], [max_x2, max_y2], [min_x1, max_y2]]
            merged_blocks.append(TextBlock(
                text=full_text,
                confidence=avg_conf,
                bbox=(min_x1, min_y1, max_x2, max_y2),
                raw_polygon=poly
            ))

        blocks = merged_blocks
        if not blocks:
            return OCRResult()

        # 统计置信信息
        confidences = [b.confidence for b in blocks]
        avg_conf = float(np.mean(confidences))
        low_conf_blocks = [b for b in blocks if b.confidence < 0.7]
        has_warning = len(low_conf_blocks) > len(blocks) * 0.2
        warning_detail = None
        if has_warning:
            warning_detail = f"OCR低置信块占比过高：{len(low_conf_blocks)}/{len(blocks)}，平均置信度={avg_conf:.2f}"
            logger.warning(f"{tag} {warning_detail}")
        logger.info(f"{tag} RapidOCR识别{len(blocks)}个文字块，平均置信度={avg_conf:.2f}")
        return OCRResult(
            blocks=blocks,
            avg_confidence=avg_conf,
            low_confidence_count=len(low_conf_blocks),
            has_warning=has_warning,
            warning_detail=warning_detail
        )

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
                combine = f"{cell_texts[best_idx]} {block.text}".strip()
                combine = re.sub(r"\s+", " ", combine)
                cell_texts[best_idx] = combine
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