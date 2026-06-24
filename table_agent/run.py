"""
组员1模块演示脚本
实现类型：[规则] 流程编排

功能：
  1. 处理单张测试图像，展示完整流程（预处理 → 检测 → OCR）
  2. 批量处理 tests/samples/ 下的所有测试图像
  3. 可视化输出（检测框绘制、各阶段对比图）
  4. 性能统计（各模块耗时）

用法：
  python run.py                          # 处理单张默认测试图
  python run.py --all                    # 批量处理所有测试图
  python run.py --image path/to/img.jpg  # 处理指定图片
  python run.py --visualize              # 保存可视化结果
  python run.py --test                   # 仅运行规则类测试（不需要模型）
"""

import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import Member1Pipeline, serialize_page_result
from table_agent.utils.common import new_trace_id
from table_agent.utils.logger import logger


def draw_detections(img, tables, show_confidence=True):
    """在图像上绘制所有检测到的表格边界框"""
    canvas = img.copy()
    for t in tables:
        if hasattr(t, "bbox"):
            bbox = t.bbox
            conf = getattr(t, "confidence", 0)
            level = getattr(t, "confidence_level", "medium")
        elif hasattr(t, "detection"):
            bbox = t.detection.bbox
            conf = t.detection.confidence
            level = t.detection.confidence_level
        else:
            continue
        x1, y1, x2, y2 = bbox
        color_map = {"high": (0, 200, 0), "medium": (0, 200, 200), "low": (0, 0, 200)}
        color = color_map.get(level, (200, 0, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if show_confidence:
            label = f"{conf:.2f} [{level}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def print_result(result):
    """打印处理结果的详细信息"""
    print(f"\n{'='*60}")
    print(f"  Result (trace_id: {result.trace_id}, page: {result.page_idx})")
    print(f"{'='*60}")
    print(f"  Steps: {', '.join(result.preprocess.steps)}")
    print(f"  Deskew: {result.preprocess.deskew_angle:.2f} deg")
    print(f"  Model: {result.detection.model_used}")
    print(f"  Image: {result.detection.image_shape}")
    print(f"  Tables: {result.table_count}")
    for i, t in enumerate(result.tables):
        det = t.detection if hasattr(t, "detection") else t
        print(f"\n  Table #{i}: bbox={det.bbox}, conf={det.confidence:.4f} [{det.confidence_level}]")
        if det.warning:
            print(f"    WARNING: {det.warning}")
        if hasattr(t, "ocr_result"):
            ocr = t.ocr_result
            print(f"    OCR blocks: {len(ocr.blocks)}, avg_conf={ocr.avg_confidence:.4f}")
            for j, b in enumerate(ocr.blocks[:5]):
                print(f'      [{j}] "{b.text}" (conf={b.confidence:.2f})')
            if len(ocr.blocks) > 5:
                print(f"      ... +{len(ocr.blocks)-5} more")
    if result.has_warning:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    - {w}")
    print(f"{'='*60}\n")


def process_single(pipeline, image_path, page_idx=0, save_vis=False):
    """处理单张图像"""
    logger.info(f"Processing: {image_path}")
    t0 = time.time()
    result = pipeline.process_page(image_path, page_idx=page_idx, trace_id=new_trace_id())
    elapsed = (time.time() - t0) * 1000
    print_result(result)
    print(f"  Time: {elapsed:.1f} ms\n")
    return result


def process_all_samples(pipeline, save_vis=False):
    """批量处理所有测试样本"""
    samples_dir = PROJECT_ROOT / "tests" / "samples"
    if not samples_dir.exists():
        logger.warning(f"Test samples not found: {samples_dir}")
        return
    images = sorted(samples_dir.glob("*.jpg"))
    if not images:
        logger.warning("No test images found")
        return
    logger.info(f"Processing {len(images)} test images...")
    for idx, img_path in enumerate(images):
        print(f"\n[{idx+1}/{len(images)}] {img_path.name}")
        t0 = time.time()
        result = pipeline.process_page(str(img_path), page_idx=idx)
        elapsed = time.time() - t0
        print(f"  Tables: {result.table_count}, Time: {elapsed*1000:.0f}ms")
        if save_vis:
            Path("visualizations").mkdir(exist_ok=True)
            vis = draw_detections(result.preprocess.original, result.tables)
            cv2.imwrite(f"visualizations/{img_path.stem}_result.png", vis)
    print(f"\nDone processing {len(images)} images")


def main():
    parser = argparse.ArgumentParser(description="Member 1 pipeline demo")
    parser.add_argument("--image", type=str, default=None, help="Image path")
    parser.add_argument("--all", action="store_true", help="Process all test samples")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations")
    parser.add_argument("--test", action="store_true", help="Run rule-based tests only")
    parser.add_argument("--list-samples", action="store_true", help="List test samples")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.list_samples:
        samples_dir = PROJECT_ROOT / "tests" / "samples"
        if not samples_dir.exists():
            print("Run: python tests/generate_test_samples.py")
            return
        for img in sorted(samples_dir.glob("*")):
            print(f"  {img.name} ({img.stat().st_size/1024:.1f} KB)")
        return

    if args.test:
        print("Running rule-based tests...")
        import numpy as np
        from utils.common import classify_confidence, new_trace_id, validate_image
        from preprocessor.image_preprocessor import ImagePreprocessor
        from ocr.ocr_engine import OCREngine, OCRResult, TextBlock
        from pipeline import serialize_page_result, PageProcessResult
        from preprocessor.image_preprocessor import PreprocessResult
        from detector.table_detector import DetectionResult

        # Utils tests
        assert classify_confidence(0.9) == "high"
        assert classify_confidence(0.7) == "medium"
        assert classify_confidence(0.3) == "low"
        assert len({new_trace_id() for _ in range(50)}) == 50
        validate_image(np.zeros((10, 10, 3), dtype=np.uint8))

        # Preprocessor tests
        pre = ImagePreprocessor()
        assert hasattr(pre, '_binarize') and hasattr(pre, '_sharpen')
        gray_img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        result = pre.process(gray_img)
        assert result.processed is not None
        assert result.gray.ndim == 2 and result.binary.ndim == 2
        assert set(np.unique(result.binary)).issubset({0, 255})
        crop = pre.crop_table_region(gray_img, (50, 50, 200, 150), padding=0)
        assert crop.shape == (100, 150, 3)

        # OCR tests
        engine = OCREngine()
        assert engine._full_to_half("１２３") == "123"
        assert engine._full_to_half("ＡＢＣ") == "ABC"
        assert engine._postprocess_text("  hello  ") == "hello"
        assert engine._postprocess_text("a   b") == "a b"

        # assign_to_cells
        blocks = [TextBlock(text="姓名", confidence=0.95, bbox=(10, 10, 50, 30), raw_polygon=[])]
        cell_texts = engine.assign_to_cells(OCRResult(blocks=blocks), [(0, 0, 100, 50)])
        assert cell_texts[0] == "姓名"

        # Serialize tests
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_binary = np.zeros((100, 100), dtype=np.uint8)
        mock_pre = PreprocessResult(mock_img, mock_img, mock_gray, mock_binary, 0.0, ["load"])
        mock_det = DetectionResult(tables=[], image_shape=(100, 100), model_used="test", has_warning=False)
        data = serialize_page_result(PageProcessResult(trace_id="test", page_idx=0, preprocess=mock_pre, detection=mock_det, tables=[]))
        assert data["trace_id"] == "test"
        assert data["table_count"] == 0

        # Pipeline health check
        health = Member1Pipeline().health_check()
        assert "preprocessor" in health and "detector" in health and "ocr" in health

        print("  All rule-based tests passed!")
        return

    pipeline = Member1Pipeline()

    if args.all:
        process_all_samples(pipeline, save_vis=args.visualize)
        return

    if args.image:
        img_path = args.image
    else:
        default = PROJECT_ROOT / "tests" / "samples" / "normal_table.jpg"
        if default.exists():
            img_path = str(default)
        else:
            print("No test image. Run: python tests/generate_test_samples.py")
            return

    if not Path(img_path).exists():
        print(f"Image not found: {img_path}")
        return

    process_single(pipeline, img_path, save_vis=args.visualize)


if __name__ == "__main__":
    main()
