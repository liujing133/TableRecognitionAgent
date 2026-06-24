
"""
组员1模块测试用例
运行方式：在项目根目录执行 python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCommonUtils:
    def test_confidence_classify_high(self):
        from utils.common import classify_confidence
        assert classify_confidence(0.90) == "high"
        assert classify_confidence(0.80) == "high"

    def test_confidence_classify_medium(self):
        from utils.common import classify_confidence
        assert classify_confidence(0.70) == "medium"
        assert classify_confidence(0.60) == "medium"

    def test_confidence_classify_low(self):
        from utils.common import classify_confidence
        assert classify_confidence(0.30) == "low"

    def test_confidence_classify_boundary(self):
        from utils.common import classify_confidence
        assert classify_confidence(0.7999) == "medium"
        assert classify_confidence(0.5999) == "low"
        assert classify_confidence(0.0) == "low"
        assert classify_confidence(1.0) == "high"

    def test_new_trace_id_unique(self):
        from utils.common import new_trace_id
        ids = {new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_new_trace_id_format(self):
        from utils.common import new_trace_id
        tid = new_trace_id()
        assert len(tid) == 36
        assert tid.count("-") == 4

    def test_validate_image_ok(self):
        from utils.common import validate_image
        validate_image(np.zeros((100, 100, 3), dtype=np.uint8))
        validate_image(np.zeros((100, 100), dtype=np.uint8))

    def test_validate_image_none(self):
        from utils.common import validate_image
        with pytest.raises(ValueError):
            validate_image(None)

    def test_validate_image_type_error(self):
        from utils.common import validate_image
        with pytest.raises(TypeError):
            validate_image("not_an_image")

    def test_validate_image_empty(self):
        from utils.common import validate_image
        with pytest.raises(ValueError):
            validate_image(np.array([]))

    def test_get_cfg_returns_dict(self):
        from utils.common import get_cfg
        assert isinstance(get_cfg("preprocessor"), dict)

    def test_load_config_has_all_sections(self):
        from utils.common import load_config
        cfg = load_config()
        for section in ["preprocessor", "detector", "ocr", "confidence"]:
            assert section in cfg


class TestImagePreprocessor:
    def _make_gray_img(self, h=400, w=300):
        return np.ones((h, w, 3), dtype=np.uint8) * 200

    def test_process_returns_result(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        result = pre.process(self._make_gray_img())
        assert result is not None
        assert result.processed is not None
        assert result.gray is not None
        assert result.binary is not None

    def test_process_shape_consistent(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        img = self._make_gray_img(400, 300)
        result = pre.process(img)
        assert result.processed.shape == img.shape

    def test_binarize_output_is_binary(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        result = pre.process(self._make_gray_img())
        unique_vals = np.unique(result.binary)
        assert set(unique_vals).issubset({0, 255})

    def test_gray_is_2d(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        result = pre.process(self._make_gray_img())
        assert result.gray.ndim == 2

    def test_resize_large_image(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        big_img = np.zeros((6000, 5000, 3), dtype=np.uint8)
        result = pre.process(big_img)
        h, w = result.processed.shape[:2]
        assert max(h, w) <= 4096
        assert "resize" in result.steps

    def test_crop_table_region_basic(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        img = np.zeros((500, 600, 3), dtype=np.uint8)
        crop = pre.crop_table_region(img, (100, 100, 300, 250), padding=0)
        assert crop.shape[0] == 150
        assert crop.shape[1] == 200

    def test_crop_with_padding_clips(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = pre.crop_table_region(img, (0, 0, 50, 50), padding=20)
        assert crop.shape[0] <= 200
        assert crop.shape[1] <= 200

    def test_steps_logged(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        result = pre.process(self._make_gray_img())
        assert "load" in result.steps
        assert "denoise" in result.steps
        assert "binarize" in result.steps

    def test_load_from_ndarray(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        result = pre.process(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is not None

    def test_invalid_source_raises(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            pre.process("/nonexistent/path/image.jpg")

    def test_process_single_channel(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        gray = np.zeros((100, 100), dtype=np.uint8)
        result = pre.process(gray)
        assert result.processed.ndim == 3
        assert result.gray.ndim == 2

    def test_deskew_no_lines(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        random_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed, angle = pre._deskew(random_img)
        assert angle == 0.0

    def test_original_preserved(self):
        from preprocessor.image_preprocessor import ImagePreprocessor
        pre = ImagePreprocessor()
        img = self._make_gray_img()
        original_before = img.copy()
        result = pre.process(img)
        assert np.array_equal(result.original, original_before)


class TestOCRPostprocess:
    def test_full_to_half_digits(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._full_to_half("０１２") == "012"

    def test_full_to_half_letters(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._full_to_half("ＡＢＣ") == "ABC"

    def test_full_width_space(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._full_to_half("　") == " "

    def test_postprocess_strip(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._postprocess_text("  hello  ") == "hello"

    def test_postprocess_merge_spaces(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._postprocess_text("a   b   c") == "a b c"

    def test_postprocess_empty(self):
        from ocr.ocr_engine import OCREngine
        engine = OCREngine()
        assert engine._postprocess_text("") == ""

    def test_assign_to_cells_basic(self):
        from ocr.ocr_engine import OCREngine, OCRResult, TextBlock
        engine = OCREngine()
        blocks = [
            TextBlock(text="姓名", confidence=0.95, bbox=(10, 10, 50, 30), raw_polygon=[]),
            TextBlock(text="张三", confidence=0.92, bbox=(110, 10, 150, 30), raw_polygon=[]),
        ]
        ocr_result = OCRResult(blocks=blocks)
        cell_bboxes = [(0, 0, 100, 50), (100, 0, 200, 50)]
        cell_texts = engine.assign_to_cells(ocr_result, cell_bboxes)
        assert cell_texts[0] == "姓名"
        assert cell_texts[1] == "张三"

    def test_assign_to_cells_no_match(self):
        from ocr.ocr_engine import OCREngine, OCRResult, TextBlock
        engine = OCREngine()
        blocks = [TextBlock(text="溢出文字", confidence=0.9, bbox=(500, 500, 600, 520), raw_polygon=[])]
        ocr_result = OCRResult(blocks=blocks)
        cell_texts = engine.assign_to_cells(ocr_result, [(0, 0, 100, 100)])
        assert cell_texts[0] == ""

    def test_assign_to_cells_merge(self):
        from ocr.ocr_engine import OCREngine, OCRResult, TextBlock
        engine = OCREngine()
        blocks = [
            TextBlock(text="联系", confidence=0.95, bbox=(10, 10, 40, 25), raw_polygon=[]),
            TextBlock(text="方式", confidence=0.93, bbox=(10, 26, 40, 45), raw_polygon=[]),
        ]
        ocr_result = OCRResult(blocks=blocks)
        cell_texts = engine.assign_to_cells(ocr_result, [(0, 0, 100, 50)])
        assert cell_texts[0] == "联系\n方式"


class TestPipelineSerialize:
    def test_serialize_empty_result(self):
        from pipeline import serialize_page_result, PageProcessResult
        from preprocessor.image_preprocessor import PreprocessResult
        from detector.table_detector import DetectionResult
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = np.zeros((100, 100), dtype=np.uint8)
        binary = np.zeros((100, 100), dtype=np.uint8)
        mock_pre = PreprocessResult(img, img, gray, binary, 0.0, ["load"])
        mock_det = DetectionResult(tables=[], image_shape=(100, 100), model_used="test", has_warning=False)
        result = PageProcessResult(trace_id="test-id", page_idx=0, preprocess=mock_pre, detection=mock_det, tables=[])
        serialized = serialize_page_result(result)
        assert serialized["trace_id"] == "test-id"
        assert serialized["table_count"] == 0

    def test_page_result_table_count(self):
        from pipeline import PageProcessResult, TableCandidate
        from preprocessor.image_preprocessor import PreprocessResult
        from detector.table_detector import DetectionResult, DetectedTable
        from ocr.ocr_engine import OCRResult
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = np.zeros((100, 100), dtype=np.uint8)
        binary = np.zeros((100, 100), dtype=np.uint8)
        detected = DetectedTable(bbox=(0, 0, 50, 50), confidence=0.9, confidence_level="high", table_idx=0)
        candidate = TableCandidate(detection=detected, crop_bgr=np.zeros((50, 50, 3), dtype=np.uint8), crop_binary=np.zeros((50, 50), dtype=np.uint8), ocr_result=OCRResult())
        result = PageProcessResult(trace_id="t", page_idx=0, preprocess=PreprocessResult(img, img, gray, binary, 0.0, []), detection=DetectionResult(tables=[detected, detected], image_shape=(100, 100), model_used="test", has_warning=False), tables=[candidate, candidate])
        assert result.table_count == 2

    def test_detection_result_empty(self):
        from detector.table_detector import DetectionResult
        result = DetectionResult()
        assert result.tables == []
        assert not result.has_warning

    def test_ocr_result_empty(self):
        from ocr.ocr_engine import OCRResult
        r = OCRResult()
        assert r.blocks == []
        assert r.avg_confidence == 0.0
        assert not r.has_warning

    def test_health_check_structure(self):
        from pipeline import Member1Pipeline
        pipeline = Member1Pipeline()
        health = pipeline.health_check()
        assert "preprocessor" in health
        assert "detector" in health
        assert "ocr" in health
