"""
图像预处理模块
实现类型：[规则] 纯OpenCV规则算法，零模型调用，零网络请求

职责：
  1. 图像加载与尺寸归一化
  2. 透视矫正（纠偏倾斜拍摄的文档）
  3. 去噪（快速非局部均值去噪）
  4. 自适应二值化（增强低对比度扫描件）
  5. 锐化（提升OCR文字清晰度）
  6. 表格区域裁剪（按检测框裁出子图）

为什么这一步可以不用大模型：
  图像几何矫正、去噪、二值化均有成熟的经典CV算法，
  效果稳定、速度快（单张<50ms）、完全确定性，无需LLM。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from table_agent.utils.logger import logger

from table_agent.utils.common import get_cfg, load_config, validate_image, timeit


class PreprocessResult:
    """预处理结果封装"""
    def __init__(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        gray: np.ndarray,
        binary: np.ndarray,
        deskew_angle: float,
        steps: List[str],
    ):
        self.original = original
        self.processed = processed
        self.gray = gray
        self.binary = binary
        self.deskew_angle = deskew_angle
        self.steps = steps


class ImagePreprocessor:
    """图像预处理器 - [规则] 全部为OpenCV确定性算法"""

    def __init__(self):
        self._cfg = get_cfg("preprocessor")

    @timeit
    def process(
        self,
        source: Union[str, Path, np.ndarray],
        trace_id: str = "",
    ) -> PreprocessResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        steps = []

        img = self._load(source)
        validate_image(img, "input_image")
        original = img.copy()
        steps.append("load")
        logger.debug(f"{tag} 图像加载完成，尺寸={img.shape}")

        img, resized = self._resize_if_needed(img)
        if resized:
            steps.append("resize")

        img = self._denoise(img)
        steps.append("denoise")

        img, angle = self._deskew(img)
        steps.append(f"deskew({angle:.2f}°)")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = self._binarize(gray)
        steps.append("binarize")

        if self._cfg.get("sharpen_enabled", True):
            img = self._sharpen(img)
            steps.append("sharpen")

        logger.info(f"{tag} 预处理完成，步骤={steps}")
        return PreprocessResult(
            original=original, processed=img,
            gray=gray, binary=binary,
            deskew_angle=angle, steps=steps,
        )

    @timeit
    def crop_table_region(
        self, img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = None,
    ) -> np.ndarray:
        if padding is None:
            padding = load_config().get("detector", {}).get("padding", 8)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        crop = img[y1:y2, x1:x2]
        validate_image(crop, "crop_result")
        return crop

    @timeit
    def process_with_visualization(self, source, trace_id=""):
        result = self.process(source, trace_id=trace_id)
        h, w = result.processed.shape[:2]
        scale = min(300 / h, 1.0)
        disp_h, disp_w = int(h * scale), int(w * scale)

        def thumb(img):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        segments = [
            ("原始", thumb(result.original)),
            ("灰度", thumb(result.gray)),
            ("去噪+矫正后", thumb(result.processed)),
            ("二值图", thumb(result.binary)),
        ]
        gap = 10
        total_w = (disp_w + gap) * 2
        total_h = (disp_h + gap) * 2
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 220
        for idx, (label, img) in enumerate(segments):
            row, col = divmod(idx, 2)
            x = col * (disp_w + gap)
            y = row * (disp_h + gap)
            canvas[y:y+disp_h, x:x+disp_w] = img
            cv2.putText(canvas, label, (x+4, y+disp_h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return result, canvas

    def _load(self, source):
        if isinstance(source, np.ndarray):
            return source.copy()
        path = str(source)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法加载图像：{path}")
        return img

    def _resize_if_needed(self, img):
        max_size = self._cfg.get("max_image_size", 4096)
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_size:
            return img, False
        scale = max_size / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"图像缩放：{w}x{h} → {new_w}x{new_h}")
        return img, True

    def _denoise(self, img):
        h = self._cfg.get("denoise_h", 10)
        h_color = self._cfg.get("denoise_h_color", 10)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, 7, 21)

    def _deskew(self, img):
        threshold = self._cfg.get("deskew_threshold", 0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return img, 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return img, 0.0
        median_angle = float(np.median(angles))
        if abs(median_angle) < threshold:
            return img, median_angle
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.debug(f"倾斜矫正：旋转 {median_angle:.2f}°")
        return corrected, median_angle

    def _binarize(self, gray):
        block_size = self._cfg.get("binarize_block_size", 15)
        c = self._cfg.get("binarize_c", 8)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    def _sharpen(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)
