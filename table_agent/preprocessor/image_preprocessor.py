"""
图像预处理模块（优化版）
解决问题：小图片模糊、清晰图过度处理、拍照图边框消失
核心改进：自适应预处理策略 + 动态参数调整
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
from table_agent.utils.logger import logger

from table_agent.utils.common import get_cfg, load_config, validate_image, timeit


class PreprocessResult:
    """预处理结果封装（保持原有结构）"""
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
    """图像预处理器 - 自适应策略优化版"""

    def __init__(self):
        self._cfg = get_cfg("preprocessor")
        # 新增：自适应判断阈值配置
        self._small_img_threshold = self._cfg.get("small_img_threshold", 600)  # 小图判定阈值（长边<600）
        self._clarity_threshold = self._cfg.get("clarity_threshold", 30)      # 清晰度阈值（方差法）
        self._brightness_threshold = self._cfg.get("brightness_threshold", 0.3)  # 光照不均阈值

    # 新增：清晰度评估（方差法，值越高越清晰）
    def _evaluate_clarity(self, gray: np.ndarray) -> float:
        """使用拉普拉斯算子方差评估图像清晰度"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    # 新增：光照均匀性评估
    def _evaluate_brightness(self, gray: np.ndarray) -> float:
        """评估图像光照均匀性（0~1，值越低越不均）"""
        # 分块计算亮度方差
        h, w = gray.shape
        block_h, block_w = h // 4, w // 4
        brightness_blocks = []
        for i in range(4):
            for j in range(4):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                brightness_blocks.append(np.mean(block) / 255.0)
        return 1 - (np.var(brightness_blocks) / np.mean(brightness_blocks))  # 均匀性得分

    # 新增：小图判断
    def _is_small_image(self, img: np.ndarray) -> bool:
        """判断是否为小图片（长边<阈值）"""
        h, w = img.shape[:2]
        return max(h, w) < self._small_img_threshold

    # 新增：光照补偿（针对拍照图）
    def _light_compensation(self, img: np.ndarray) -> np.ndarray:
        """自适应光照补偿，提升暗部细节"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 自适应直方图均衡化（限制对比度，避免过曝）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 新增：边缘增强（保护表格边框）
    def _enhance_edges(self, img: np.ndarray) -> np.ndarray:
        """增强边缘，避免锐化导致边框断裂"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        # 边缘增强掩码
        edge_mask = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        # 原图 + 边缘增强
        enhanced = cv2.addWeighted(img, 1.0, cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR), 0.2, 0)
        return enhanced

    @timeit
    def process(
        self,
        source: Union[str, Path, np.ndarray],
        trace_id: str = "",
    ) -> PreprocessResult:
        tag = f"[trace={trace_id}]" if trace_id else ""
        steps = []

        # 1. 基础加载
        img = self._load(source)
        validate_image(img, "input_image")
        original = img.copy()
        steps.append("load")
        logger.debug(f"{tag} 图像加载完成，尺寸={img.shape}")

        # 2. 自适应判断（核心）
        is_small = self._is_small_image(img)
        gray_init = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clarity_score = self._evaluate_clarity(gray_init)
        is_clear = clarity_score > self._clarity_threshold
        brightness_score = self._evaluate_brightness(gray_init)
        is_uneven_light = brightness_score < self._brightness_threshold

        logger.info(
            f"{tag} 图像特征：小图={is_small}，清晰度={clarity_score:.2f}，光照均匀性={brightness_score:.2f}"
        )

        # 3. 自适应缩放（小图跳过缩放）
        img, resized = self._resize_if_needed(img, skip=is_small)
        if resized:
            steps.append("resize")

        # 4. 自适应去噪（清晰图/小图跳过强去噪）
        if not (is_small or is_clear):
            img = self._denoise(img)
            steps.append("denoise")
        else:
            steps.append("denoise(skipped)")

        # 5. 倾斜矫正（保留，所有图都需要）
        img, angle = self._deskew(img)
        steps.append(f"deskew({angle:.2f}°)")

        # 6. 光照补偿（仅拍照图/光照不均图）
        if is_uneven_light:
            img = self._light_compensation(img)
            steps.append("light_compensation")

        # 7. 灰度+二值化（自适应参数）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if is_clear and not is_uneven_light:
            # 清晰图：简化二值化（低阈值）
            binary = self._binarize(gray, block_size=9, c=4)
            steps.append("binarize(light)")
        elif is_uneven_light:
            # 光照不均图：自适应二值化（大区块+高补偿）
            binary = self._binarize(gray, block_size=25, c=12)
            steps.append("binarize(heavy)")
        else:
            # 普通图：默认参数
            binary = self._binarize(gray)
            steps.append("binarize(default)")

        # 8. 自适应锐化（小图/清晰图：轻量锐化；其他：增强边缘+锐化）
        if self._cfg.get("sharpen_enabled", True):
            if is_small or is_clear:
                # 小图/清晰图：轻量锐化（避免模糊）
                img = self._sharpen(img, light=True)
                steps.append("sharpen(light)")
            else:
                # 其他图：先增强边缘再锐化（保护边框）
                img = self._enhance_edges(img)
                img = self._sharpen(img, light=False)
                steps.append("enhance_edges + sharpen")
        else:
            steps.append("sharpen(skipped)")

        logger.info(f"{tag} 预处理完成，步骤={steps}")
        return PreprocessResult(
            original=original, processed=img,
            gray=gray, binary=binary,
            deskew_angle=angle, steps=steps,
        )

    # 重载：缩放逻辑（支持跳过）
    def _resize_if_needed(self, img, skip: bool = False, tag: str = ""):
        if skip:
            logger.debug(f"{tag} 小图跳过缩放")
            return img, False
        max_size = self._cfg.get("max_image_size", 4096)
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_size:
            return img, False
        scale = max_size / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"{tag} 图像缩放：{w}x{h} → {new_w}x{new_h}")
        return img, True

    # 重载：二值化（支持动态参数）
    def _binarize(self, gray, block_size=None, c=None):
        block_size = block_size or self._cfg.get("binarize_block_size", 15)
        c = c or self._cfg.get("binarize_c", 8)
        # 确保block_size为奇数
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    # 重载：锐化（支持轻量/标准模式）
    def _sharpen(self, img, light: bool = True):
        if light:
            # 轻量锐化（小图/清晰图）
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32)
        else:
            # 标准锐化（普通图）
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    # 以下方法保持原有逻辑（_load/_denoise/_deskew/crop_table_region等）
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