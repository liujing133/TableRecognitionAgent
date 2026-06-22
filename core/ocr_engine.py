from paddleocr import PaddleOCR
import yaml
import numpy as np
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class OCREngine:
    def __init__(self):
        #self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=cfg["model"]["ocr_use_gpu"])
        self.ocr = None

    def extract_text(self, table_img: np.ndarray, trace_id: str):
        """返回所有文字块：四点坐标、文本、置信度"""
        res = self.ocr.ocr(table_img, cls=True)
        text_blocks = []
        for line in res[0]:
            pts = line[0]
            text, score = line[1][0], line[1][1]
            text_blocks.append({"points": pts, "text": text, "score": score})
        log_trace(trace_id, "OCREngine", {"text_block_num": len(text_blocks)})
        return text_blocks