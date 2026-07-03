import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from table_agent.utils.logger import logger
from table_agent.utils.common import get_cfg, classify_confidence, timeit

try:
    from rapid_table import RapidTable, RapidTableInput, ModelType
    RAPID_AVAIL = True
except ImportError:
    logger.error("rapid-table 未安装，请执行 pip install rapid-table rapidocr_onnxruntime")
    RAPID_AVAIL = False

# 数据结构完全兼容原有流水线
@dataclass
class DetectedTable:
    bbox: Tuple[int, int, int, int]
    confidence: float
    confidence_level: str
    page_idx: int = 0
    table_idx: int = 0
    warning: Optional[str] = None

@dataclass
class DetectionResult:
    tables: List[DetectedTable] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)
    model_used: str = "RapidTable ONNX表格检测器"
    has_warning: bool = False

class TableDetector:
    def __init__(self):
        self._cfg = get_cfg("detector")
        self.conf_thresh = self._cfg.get("conf_threshold", 0.4)
        # 微调参数：缩小底部扩充，增加左右收缩防止右侧溢出
        self.bottom_expand = 82
        self.left_right_shrink = 0
        self.table_engine = None
        if RAPID_AVAIL:
            try:
                input_cfg = RapidTableInput(
                    model_type=ModelType.PPSTRUCTURE_ZH,
                    use_ocr=False
                )
                self.table_engine = RapidTable(input_cfg)
            except Exception as e:
                logger.error(f"RapidTable初始化失败: {str(e)}")
                self.table_engine = None

    @timeit
    def detect(self, img: np.ndarray, page_idx=0, trace_id="") -> DetectionResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        h, w = img.shape[:2]
        tables = []

        if self.table_engine is None:
            logger.error(f"{tag} 表格引擎未初始化")
            return DetectionResult(tables=[], image_shape=(h, w), has_warning=True)

        try:
            result = self.table_engine(img)
            cell_groups = result.cell_bboxes
            if not cell_groups:
                logger.info(f"{tag} 未检测到任何表格")
                return DetectionResult(tables=[], image_shape=(h, w), has_warning=False)

            for idx, cells in enumerate(cell_groups):
                xs = []
                ys = []
                for cell in cells:
                    coords = list(map(int, cell[:4]))
                    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])
                # 原始包围盒
                tx1, ty1 = min(xs), min(ys)
                tx2, ty2 = max(xs), max(ys)

                # 修复1：左右向内收缩，解决右侧多出一大块
                tx1 = tx1 + self.left_right_shrink
                tx2 = tx2 - self.left_right_shrink
                # 修复2：少量向下扩展补齐尾行，数值缩小避免过度拉伸
                ty2 = ty2 + self.bottom_expand

                # 全局边界限制，杜绝越界
                tx1 = max(0, tx1)
                ty1 = max(0, ty1)
                tx2 = min(w, tx2)
                ty2 = min(h, ty2)

                # 过滤过小畸形框
                tw = tx2 - tx1
                th = ty2 - ty1
                if tw < 25 or th < 25:
                    continue

                conf = 0.85
                conf_level = classify_confidence(conf)
                warn_msg = f"表格{idx}置信度{conf:.2f}" if conf_level != "high" else None
                tables.append(DetectedTable(
                    bbox=(tx1, ty1, tx2, ty2),
                    confidence=conf,
                    confidence_level=conf_level,
                    page_idx=page_idx,
                    table_idx=idx,
                    warning=warn_msg
                ))

            has_warning = any(t.warning is not None for t in tables)
            logger.info(f"{tag} 底部扩充{self.bottom_expand}像素，左右收缩{self.left_right_shrink}像素")
            return DetectionResult(
                tables=tables,
                image_shape=(h, w),
                model_used="RapidTable ONNX表格检测器",
                has_warning=has_warning
            )
        except TypeError as e:
            if "NoneType" in str(e):
                logger.error(f"{tag} 缺失rapidocr依赖，请执行 pip install rapidocr_onnxruntime")
            raise
        except ValueError as e:
            if "too many values to unpack" in str(e):
                logger.error(f"{tag} 单元格坐标长度异常：{str(e)}")
            return DetectionResult(tables=[], image_shape=(h, w), has_warning=True)
        except Exception as e:
            logger.error(f"{tag} 表格检测异常：{str(e)}")
            return DetectionResult(tables=[], image_shape=(h, w), has_warning=True)