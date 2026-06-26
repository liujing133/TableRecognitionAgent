# core/tsr_parser.py
import os
import yaml
import json
import numpy as np
import cv2
from PIL import Image
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

try:
    from core import spatial_clustering
except Exception:
    import spatial_clustering

class TSRParser:
    def __init__(self, device=None, model_path=None):
        self.model_path = model_path or cfg.get("model", {}).get("tsr_model_path", "models/tableformer_light")
        self.device = device or "cpu"
        self.model = None
        self.processor = None
        self.id2label = {}
        self.score_threshold = cfg.get("model", {}).get("tsr_score_threshold", 0.2)
        self.model_loaded = False
        if os.path.exists(self.model_path):
            try:
                self.load_model()
                self.model_loaded = True
            except Exception as e:
                log_trace("init", "TSRParser", {"model_load_error": str(e)})
                self.model_loaded = False
        else:
            log_trace("init", "TSRParser", {"model_missing": self.model_path})

    def load_model(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        import torch

        model_dir = self.model_path
        if os.path.isfile(self.model_path):
            model_dir = os.path.dirname(self.model_path)

        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.model.to(torch.device(self.device))
        self.id2label = self.model.config.id2label
        log_trace("load", "TSRParser", {"model_path": model_dir, "note": "TableFormer model loaded"})

    def normalize_text_blocks(self, text_blocks):
        normalized = []
        for block in text_blocks:
            text = block.get("text") or block.get("ocr_text") or block.get("content") or ""
            score = float(block.get("score", block.get("confidence", 0.0) or 0.0))
            if "points" in block and block["points"]:
                pts = np.array(block["points"], dtype=float)
                x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            elif all(k in block for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = float(block["x1"]), float(block["y1"]), float(block["x2"]), float(block["y2"])
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
            else:
                continue
            normalized.append({
                "points": pts.tolist(),
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "score": score,
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            })
        return normalized

    def bbox_to_xyxy(self, bbox):
        if not bbox:
            return [0.0, 0.0, 0.0, 0.0]
        if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        if len(bbox) == 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        raise ValueError("无法识别bbox格式")

    def get_cell_center(self, cell):
        x1, y1, x2, y2 = self.bbox_to_xyxy(cell.get("bbox", []))
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def point_in_box(self, center, box):
        x, y = center
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def merge_bboxes(self, boxes):
        xyxys = [self.bbox_to_xyxy(b) for b in boxes if b]
        if not xyxys:
            return [0.0, 0.0, 0.0, 0.0]
        xs = [x for b in xyxys for x in (b[0], b[2])]
        ys = [y for b in xyxys for y in (b[1], b[3])]
        return [min(xs), min(ys), max(xs), max(ys)]

    def detect_table_structure(self, table_img):
        if self.processor is None or self.model is None:
            raise RuntimeError("TSR TableFormer 模型尚未加载")

        import torch

        if isinstance(table_img, np.ndarray):
            image = cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(table_img, Image.Image):
            image = table_img.convert("RGB")
        else:
            raise RuntimeError("table_img 必须为 numpy.ndarray 或 PIL.Image.Image")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        detections = self.processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        results = []
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            label_name = self.id2label.get(int(label), str(int(label)))
            results.append({
                "label": label_name,
                "score": float(score.item()),
                "box": [float(box[0].item()), float(box[1].item()), float(box[2].item()), float(box[3].item())],
            })
        return results

    def _compute_cell_union_bbox(self, cell):
        """计算一个网格格子的合并边界框（处理多边形或 xyxy 格式）"""
        bbox = cell.get("bbox", [])
        if not bbox:
            return [0.0, 0.0, 0.0, 0.0]
        return self.bbox_to_xyxy(bbox)


    def _boxes_overlap(self, box_a, box_b):
        """计算两个 bbox 的 IoU（对称）"""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        area_a = max((xa2 - xa1) * (ya2 - ya1), 1.0)
        area_b = max((xb2 - xb1) * (yb2 - yb1), 1.0)
        iou = inter / (area_a + area_b - inter)
        return iou


    def apply_structure_predictions(self, grid, predictions):
        """
        用 TableFormer 预测结果修正网格：
        - "table column header" / "table projected row header" → 标记 is_header
        - "table spanning cell" → 合并单元格
        - 其他标签 → 按 bbox 重叠找到对应格子
        """
        if not grid:
            return grid, False

        has_multi_header = any(
            pred["label"] in {"table column header", "table projected row header"}
            for pred in predictions
        )

        # ---- 第1步：处理 header 标签 ----
        header_labels = {"table column header", "table projected row header"}
        for pred in predictions:
            if pred["label"] not in header_labels:
                continue
            span_box = pred["box"]
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    cell_box = self._compute_cell_union_bbox(cell)
                    center = self.get_cell_center(cell)
                    # 用点检测（宽松）或低 IoU 检测
                    if self.point_in_box(center, span_box) or self._boxes_overlap(cell_box, span_box) >= 0.05:
                        cell["is_header"] = True

        # ---- 第2步：处理 spanning cell 标签（仅真正跨格的才合并） ----
        span_labels = {"table spanning cell"}
        spans = [pred for pred in predictions if pred["label"] in span_labels]
        for span in spans:
            span_box = span["box"]
            contained = []
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    cell_box = self._compute_cell_union_bbox(cell)
                    iou = self._boxes_overlap(cell_box, span_box)
                    if iou >= 0.05:
                        contained.append({"r": r, "c": c, "cell": cell, "iou": iou})

            if len(contained) < 2:
                continue  # 至少覆盖2格才算合并

            rows_set = sorted({item["r"] for item in contained})
            cols_set = sorted({item["c"] for item in contained})
            top_cell = grid[rows_set[0]][cols_set[0]]

            top_cell["rowspan"] = len(rows_set)
            top_cell["colspan"] = len(cols_set)
            merged_texts = []
            merged_bboxes = []
            for r in rows_set:
                for c in cols_set:
                    cell = grid[r][c]
                    if cell.get("text"):
                        merged_texts.append(cell["text"])
                    merged_bboxes.append(cell.get("bbox", []))
                    if r != rows_set[0] or c != cols_set[0]:
                        cell["text"] = ""
                        cell["rowspan"] = 1
                        cell["colspan"] = 1
            top_cell["text"] = " ".join([t for t in merged_texts if t]).strip()
            top_cell["bbox"] = self.merge_bboxes(merged_bboxes)

        # ---- 第3步：多级表头启发式检测 ----
        try:
            if not has_multi_header and len(grid) >= 2:

                def avg_text_len(r):
                    texts = [cell.get("text", "") for cell in r]
                    texts = [t for t in texts if t]
                    return sum(len(t) for t in texts) / (len(texts) or 1)

                first_len = avg_text_len(grid[0])
                second_len = avg_text_len(grid[1])
                if first_len < second_len * 0.6:
                    has_multi_header = True
        except Exception:
            pass

        return grid, bool(has_multi_header)


    def normalize_table_struct(self, table_struct):
        rows = table_struct.get("rows", [])
        for r in rows:
            for cell in r.get("cells", []):
                cell.setdefault("text", "")
                cell.setdefault("rowspan", 1)
                cell.setdefault("colspan", 1)
                cell.setdefault("is_header", False)
                cell.setdefault("bbox", cell.get("bbox", []))
        table_struct.setdefault("has_multi_header", False)
        table_struct.setdefault("has_merge_cell", False)
        table_struct.setdefault("fallback_used", False)
        return table_struct

    def infer_tableformer(self, text_blocks, page_meta=None, table_img=None):
        text_blocks = self.normalize_text_blocks(text_blocks)
        if not text_blocks:
            return self.fallback_parse(text_blocks, page_meta)

        grid = spatial_clustering.cluster_rows_cols(text_blocks, cfg)
        if not grid:
            return self.fallback_parse(text_blocks, page_meta)

        if table_img is None:
            raise RuntimeError("TableFormer 推理需要 table_img 参数")

        predictions = self.detect_table_structure(table_img)
        log_trace("infer", "TSRParser", {"predictions": predictions})
        grid, has_multi_header = self.apply_structure_predictions(grid, predictions)

        rows = []
        for row in grid:
            cells = []
            for cell in row:
                bbox = self.bbox_to_xyxy(cell.get("bbox", []))
                cells.append({
                    "text": cell.get("text", ""),
                    "rowspan": int(cell.get("rowspan", 1)),
                    "colspan": int(cell.get("colspan", 1)),
                    "is_header": bool(cell.get("is_header", False)),
                    "bbox": bbox,
                })
            rows.append({"cells": cells})

        table_struct = {
            "rows": rows,
            "has_multi_header": has_multi_header,
            "has_merge_cell": any(cell["colspan"] > 1 or cell["rowspan"] > 1 for row in rows for cell in row["cells"]),
        }
        return self.normalize_table_struct(table_struct)

    def fallback_parse(self, text_blocks, page_meta=None, trace_id="trace"):
        normalized = self.normalize_text_blocks(text_blocks)
        grid = spatial_clustering.cluster_rows_cols(normalized if normalized else text_blocks, cfg)
        refined = []
        for row in grid:
            new_row = []
            for i, cell in enumerate(row):
                c = {
                    "text": cell.get("text", "") if isinstance(cell, dict) else "",
                    "rowspan": 1,
                    "colspan": 1,
                    "is_header": bool(cell.get("is_header", False)) if isinstance(cell, dict) else False,
                    "bbox": self.bbox_to_xyxy(cell.get("bbox", [])) if isinstance(cell, dict) else [0.0, 0.0, 0.0, 0.0],
                }
                if c["text"] == "" and i > 0 and row[i - 1].get("text", "") != "":
                    c["colspan"] = 2
                    c["text"] = row[i - 1].get("text", "") + "(续)"
                new_row.append(c)
            refined.append({"cells": new_row})
        table_struct = {"rows": refined, "has_multi_header": False, "has_merge_cell": False, "fallback_used": True}
        return self.normalize_table_struct(table_struct)

    def parse(self, text_blocks: list, trace_id: str, table_img=None, page_meta=None, use_fallback=False):
        log_trace(trace_id, "TSRParser", {"cell_count": len(text_blocks), "use_fallback": use_fallback})
        if not use_fallback and self.model_loaded:
            try:
                res = self.infer_tableformer(text_blocks, page_meta, table_img=table_img)
                if res:
                    res = self.normalize_table_struct(res)
                    res["fallback_used"] = False
                    log_trace(trace_id, "TSRParser", {"fallback_used": False})
                    return res
            except NotImplementedError:
                pass
            except Exception as e:
                log_trace(trace_id, "TSRParser", {"model_infer_error": str(e)})
        res = self.fallback_parse(text_blocks, page_meta, trace_id)
        log_trace(trace_id, "TSRParser", {"fallback_used": True})
        return res
