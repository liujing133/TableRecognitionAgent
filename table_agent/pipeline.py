"""
组员1模块统一对外接口
实现类型：[规则] 流程编排，无模型调用

这个文件是组员1所有模块的统一出口，组长的FastAPI直接import这里。
输入：图像（文件路径或ndarray）
输出：标准化结构体，包含：
  - 预处理结果（矫正后图像、二值图等）
  - 检测结果（所有表格坐标、置信度）
  - 每张表格的裁剪子图
  - 每张表格的OCR文字块（含坐标、置信度）
  - 整体预警信息
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Optional
from table_agent.utils.logger import logger

from table_agent.preprocessor.image_preprocessor import ImagePreprocessor, PreprocessResult
from table_agent.detector.table_detector import TableDetector, DetectionResult, DetectedTable
from table_agent.ocr.ocr_engine import OCREngine, OCRResult
from table_agent.utils.common import new_trace_id, timeit


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class TableCandidate:
    """单张检测到的表格的完整信息（组长TSR模块的输入）"""
    detection: DetectedTable          # 检测结果（坐标、置信度）
    crop_bgr: np.ndarray              # 裁剪后的表格BGR图像
    crop_binary: np.ndarray           # 裁剪后的二值图（供无边框表格聚类）
    ocr_result: OCRResult             # 该表格的OCR结果


@dataclass
class PageProcessResult:
    """
    单页处理完整结果，传递给组长/组员2的TSR模块
    """
    trace_id: str
    page_idx: int
    preprocess: PreprocessResult      # 预处理中间结果
    detection: DetectionResult        # 检测结果（含所有表格）
    tables: List[TableCandidate]      # 每张表格的候选（裁剪图+OCR）
    has_warning: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        """便捷属性：返回本页检测到的表格数量"""
        return len(self.tables)


# ─────────────────────────────────────────────
# 流水线
# ─────────────────────────────────────────────

class Member1Pipeline:
    """
    组员1负责的完整流水线：
    图像 → 预处理 → YOLOv8检测 → 裁剪 → PaddleOCR
    """

    def __init__(self):
        self._pre = ImagePreprocessor()
        self._det = TableDetector()
        self._ocr = OCREngine()

    @timeit
    def process_page(
        self,
        source: Union[str, Path, np.ndarray],
        page_idx: int = 0,
        trace_id: Optional[str] = None,
    ) -> PageProcessResult:
        """
        处理单页图像，返回完整结构体。

        Args:
            source: 图像路径或BGR ndarray
            page_idx: 当前页码（0-based）
            trace_id: 外部传入的trace_id，None时自动生成

        Returns:
            PageProcessResult
        """
        if trace_id is None:
            trace_id = new_trace_id()
        tag = f"[trace={trace_id}][page={page_idx}]"

        warnings: List[str] = []

        # ── Step 1: 图像预处理 ────────────────
        logger.info(f"{tag} Step1: 图像预处理开始")
        pre_result = self._pre.process(source, trace_id=trace_id)

        # ── Step 2: YOLOv8表格检测 ────────────
        logger.info(f"{tag} Step2: 表格检测开始")
        det_result = self._det.detect(
            pre_result.processed, page_idx=page_idx, trace_id=trace_id
        )
        if det_result.has_warning:
            warnings.extend(
                [t.warning for t in det_result.tables if t.warning]
            )

        # ── Step 3: 裁剪 + OCR ───────────────
        logger.info(f"{tag} Step3: 裁剪+OCR开始，共{len(det_result.tables)}张表格")
        candidates: List[TableCandidate] = []

        for table in det_result.tables:
            # 裁剪BGR图（给TSR / OCR用）
            crop_bgr = self._pre.crop_table_region(
                pre_result.processed, table.bbox
            )
            # 裁剪二值图（直接传入单通道，给无边框表格聚类用）
            crop_binary = self._pre.crop_table_region(
                pre_result.binary, table.bbox
            )

            # OCR识别
            ocr_result = self._ocr.extract(crop_bgr, trace_id=trace_id)
            if ocr_result.has_warning and ocr_result.warning_detail:
                warnings.append(ocr_result.warning_detail)

            candidates.append(TableCandidate(
                detection=table,
                crop_bgr=crop_bgr,
                crop_binary=crop_binary,
                ocr_result=ocr_result,
            ))

        has_warning = len(warnings) > 0
        if has_warning:
            logger.warning(f"{tag} 本页存在{len(warnings)}条预警")

        logger.info(f"{tag} 页面处理完成，表格数={len(candidates)}")

        return PageProcessResult(
            trace_id=trace_id,
            page_idx=page_idx,
            preprocess=pre_result,
            detection=det_result,
            tables=candidates,
            has_warning=has_warning,
            warnings=warnings,
        )

    @timeit
    def process_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
        trace_id: Optional[str] = None,
    ) -> List[PageProcessResult]:
        """
        批量处理多页图像。
        实现类型：[规则] 循环编排

        Args:
            sources: 图像路径或ndarray列表（每页一张）
            trace_id: 外部传入的trace_id，None时自动生成

        Returns:
            每页的PageProcessResult列表
        """
        if trace_id is None:
            trace_id = new_trace_id()

        results: List[PageProcessResult] = []
        for idx, src in enumerate(sources):
            logger.info(f"[trace={trace_id}] 开始处理第{idx}页...")
            result = self.process_page(src, page_idx=idx, trace_id=trace_id)
            results.append(result)

        logger.info(
            f"[trace={trace_id}] 批量处理完成，共{len(results)}页，"
            f"总表格数={sum(r.table_count for r in results)}"
        )
        return results

    def health_check(self) -> dict:
        """
        健康检查，供组长的FastAPI /health 接口调用
        返回各子模块状态
        """
        return {
            "preprocessor": "ok",  # 纯规则，无需检查
            "detector": "ok" if self._det.is_model_ready() else "model_not_loaded",
            "ocr": "ok",           # PaddleOCR懒加载，不在此处检查
        }


# ─────────────────────────────────────────────
# 序列化工具（配合组长的HTTP接口）
# ─────────────────────────────────────────────

def serialize_page_result(result: PageProcessResult) -> dict:
    """
    将PageProcessResult转为JSON可序列化的dict，
    供组长的FastAPI response使用。
    numpy数组不序列化（太大），只序列化元数据。
    """
    return {
        "trace_id": result.trace_id,
        "page_idx": result.page_idx,
        "has_warning": result.has_warning,
        "warnings": result.warnings,
        "deskew_angle": result.preprocess.deskew_angle,
        "preprocess_steps": result.preprocess.steps,
        "image_shape": list(result.detection.image_shape),
        "model_used": result.detection.model_used,
        "table_count": result.table_count,
        "tables": [
            {
                "table_idx": t.detection.table_idx,
                "page_idx": t.detection.page_idx,
                "bbox": list(t.detection.bbox),
                "detection_confidence": round(t.detection.confidence, 4),
                "detection_level": t.detection.confidence_level,
                "detection_warning": t.detection.warning,
                "crop_shape": list(t.crop_bgr.shape[:2]),
                "ocr_block_count": len(t.ocr_result.blocks),
                "ocr_avg_confidence": round(t.ocr_result.avg_confidence, 4),
                "ocr_has_warning": t.ocr_result.has_warning,
                "ocr_warning": t.ocr_result.warning_detail,
                "ocr_blocks": [
                    {
                        "text": b.text,
                        "confidence": round(b.confidence, 4),
                        "bbox": list(b.bbox),
                    }
                    for b in t.ocr_result.blocks
                ],
            }
            for t in result.tables
        ],
    }
