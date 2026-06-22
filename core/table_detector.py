from ultralytics import YOLO
import yaml
import numpy as np
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class TableDetector:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")

    def detect(self, img: np.ndarray, trace_id: str):
        """输出页面所有表格检测框"""
        results = self.model(img, conf=0.5)
        table_boxes = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy()
            for box in boxes:
                table_boxes.append(box.tolist())
        log_trace(trace_id, "TableDetector", {"box_count": len(table_boxes)})
        return table_boxes