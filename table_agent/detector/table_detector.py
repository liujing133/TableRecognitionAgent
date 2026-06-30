"""
Qwen2.5-VL 表格区域检测模块
实现类型：[模型] 多模态大模型（Ollama部署）

职责：
  1. 调用本地Ollama的Qwen2.5-VL模型
  2. 对预处理后的图像推理，输出所有表格的边界框
  3. 输出含置信度的检测结果，支持低置信预警
  4. 封装为可被流水线直接调用的标准接口

替换说明：
  - 原YOLOv8通过目标检测输出表格框，现通过VLM解析图像输出表格框
  - 置信度由VLM模拟（因VLM无原生置信度输出，按解析稳定性分级）
"""

import os
import json
import base64
import numpy as np
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
# 保留原有工具类依赖（需确保项目中存在这些模块）
from table_agent.utils.logger import logger
from table_agent.utils.common import get_cfg, validate_image, classify_confidence, timeit

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# 数据结构（保持与原代码完全一致，确保接口兼容）
# ─────────────────────────────────────────────

@dataclass
class DetectedTable:
    """单个检测到的表格"""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) 像素坐标
    confidence: float                  # 置信度 0~1（VLM模拟）
    confidence_level: str              # 'high' | 'medium' | 'low'
    page_idx: int = 0                  # 所在页码（0-based）
    table_idx: int = 0                 # 页内表格序号（0-based）
    warning: Optional[str] = None      # 低置信时的预警信息


@dataclass
class DetectionResult:
    """整页检测结果"""
    tables: List[DetectedTable] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)   # (H, W)
    model_used: str = ""
    has_warning: bool = False


# ─────────────────────────────────────────────
# 工具函数（复用vl_demo.py中的base64转换逻辑）
# ─────────────────────────────────────────────

def img_to_base64(img: np.ndarray, max_size: int = 1024) -> str:
    """
    OpenCV图像（np.ndarray）转base64字符串
    Args:
        img: BGR格式的numpy图像
        max_size: 图像最大边长限制（防止过大导致模型处理失败）
    Returns:
        base64编码字符串
    """
    # 缩放图片：宽高任一维度超过max_size则等比例缩小
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 编码为png格式
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("图片编码失败")
    return base64.b64encode(buf).decode("utf-8")


# ─────────────────────────────────────────────
# 核心类（替换YOLO为Qwen2.5-VL）
# ─────────────────────────────────────────────

class TableDetector:
    """
    Qwen2.5-VL表格检测器（Ollama部署）
    实现类型：[模型] 多模态大模型

    配置依赖：
      - 需在config.yaml中补充ollama相关配置（见下方说明）
    """

    def __init__(self):
        self._cfg = get_cfg("detector")
        # Ollama配置（从config.yaml读取，无则用默认值）
        self._ollama_url = self._cfg.get("ollama_url", "http://127.0.0.1:11434/api/chat")
        self._model_name = self._cfg.get("ollama_model", "qwen2.5vl:7b-q4_K_M")
        self._max_img_size = self._cfg.get("max_img_size", 1024)
        self._temperature = self._cfg.get("temperature", 0.01)
        self._timeout = self._cfg.get("timeout", 300)  # 5分钟超时

    # ── 公开入口（保持原接口不变） ──────────────────────────────

    @timeit
    def detect(
        self,
        img: np.ndarray,
        page_idx: int = 0,
        trace_id: str = "",
    ) -> DetectionResult:
        """
        对单张页面图像执行表格检测（接口与原YOLO版本完全兼容）。

        Args:
            img: BGR numpy图像（经预处理模块处理后的图）
            page_idx: 当前页码，用于结果溯源
            trace_id: 全链路追踪ID

        Returns:
            DetectionResult，含所有检测到的表格及置信度
        """
        tag = f"[trace={trace_id}]" if trace_id else ""
        validate_image(img, "detector_input")

        h, w = img.shape[:2]
        try:
            # 1. 图像转base64
            img_b64 = img_to_base64(img, self._max_img_size)
            
            # 2. 调用Ollama的Qwen2.5-VL模型
            table_boxes = self._call_ollama_vlm(img_b64, tag)
            
            # 3. 解析结果为DetectedTable列表
            tables = self._parse_vlm_result(table_boxes, page_idx, h, w, tag)
            
            # 4. 检查是否有预警
            has_warning = any(t.warning for t in tables)
            
            logger.info(
                f"{tag} 第{page_idx}页检测到{len(tables)}个表格，"
                f"预警={has_warning}（模型：{self._model_name}）"
            )

            return DetectionResult(
                tables=tables,
                image_shape=(h, w),
                model_used=self._model_name,
                has_warning=has_warning,
            )

        except Exception as e:
            logger.error(f"{tag} 第{page_idx}页表格检测失败：{str(e)}")
            return DetectionResult(
                tables=[],
                image_shape=(h, w),
                model_used=self._model_name,
                has_warning=True,
            )

    # ── 私有方法：调用Ollama VLM ────────────────────────────

    def _call_ollama_vlm(self, img_b64: str, tag: str) -> List[Tuple[int, int, int, int]]:
        """
        调用Ollama部署的Qwen2.5-VL模型，获取表格边界框
        Returns:
            表格框列表 [(x1,y1,x2,y2), ...]
        """
        # 强约束Prompt，强制只输出表格框的JSON，减少模型废话
        prompt = f"""
你是专业的表格区域检测专家，严格遵守以下规则：
1. 识别范围包含：
   - 合并单元格表格（无论是否有边框）；
   - 无边框/少线表格（通过行列文字对齐、空白间隔判断）；
   - 跨页续表（当前页仅标注可见区域）；
   - 多级/嵌套表头表格（整体标注为一个表格框）；
   - 扫描噪声背景下的表格（忽略斑点、模糊、倾斜干扰）；
2. 每个表格输出一个边界框，格式为 [x1, y1, x2, y2]（像素坐标，左上角为原点）；
3. 坐标必须是整数，且在图像范围内；
4. 禁止输出任何解释/说明，仅返回标准JSON数组；
5. 无表格时返回空数组 []；
6. 输出示例：[[10,20,300,400], [50,60,200,300]]

补充判断规则：
- 无边框表格：通过文字行/列的对齐关系、单元格空白间距判定表格边界；
- 合并单元格：不拆分单元格，按表格整体外轮廓标注；
- 扫描噪声：即使表格边缘模糊/有斑点，仍标注可见的最小外框；
- 嵌套表格：外层表格标注整体框，内层子表格单独标注；
"""

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "temperature": self._temperature
        }

        # 发送请求到Ollama
        try:
            resp = requests.post(
                self._ollama_url,
                json=payload,
                timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"{tag} 模型处理图片超时（超时时间：{self._timeout}秒）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{tag} 请求Ollama服务失败：{str(e)}")

        # 解析返回结果
        content = resp.json()["message"]["content"]
        
        # 清洗返回内容，提取纯JSON数组（防止模型额外加文字）
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            raise RuntimeError(f"{tag} VLM返回内容无有效JSON：{content}")
        
        json_str = content[start:end]
        try:
            box_list = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{tag} VLM返回JSON解析失败：{json_str} | 错误：{str(e)}")

        # 验证格式
        if not isinstance(box_list, list):
            raise RuntimeError(f"{tag} VLM返回非数组格式：{box_list}")
        
        return box_list

    # ── 私有方法：解析VLM结果为DetectedTable ────────────────────

    def _parse_vlm_result(
        self,
        box_list: List,
        page_idx: int,
        img_h: int,
        img_w: int,
        tag: str
    ) -> List[DetectedTable]:
        """
        将VLM返回的表格框列表解析为DetectedTable对象列表
        自动兼容多层嵌套：[ [框1,框2] ] / [框1] / [框1,框2] 三种格式
        """
        # 第一步：扁平化所有坐标，提取纯四元框数组
        flat_boxes = []

        def extract_boxes(arr):
            """递归提取所有长度=4的坐标数组"""
            for item in arr:
                if isinstance(item, list):
                    if len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                        # 标准坐标框
                        flat_boxes.append([int(round(v)) for v in item])
                    else:
                        extract_boxes(item)

        extract_boxes(box_list)
        if not flat_boxes:
            logger.warning(f"{tag} 未提取到任何合法表格坐标")
            return []

        # 第二步：过滤+生成表格对象
        tables = []
        for idx, box in enumerate(flat_boxes):
            x1, y1, x2, y2 = box
            # 坐标合法性校验（防止越界）
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1 + 1, min(x2, img_w))  # 确保x2 > x1
            y2 = max(y1 + 1, min(y2, img_h))  # 确保y2 > y1
            # 过滤极小无效框
            w = x2 - x1
            h = y2 - y1
            if w < 20 or h < 20:
                logger.warning(f"{tag} 表格#{idx} 尺寸过小，跳过 {box}")
                continue

            # VLM模拟置信度
            box_area = w * h
            img_area = img_w * img_h
            area_ratio = box_area / img_area if img_area > 0 else 0.0
            if 0.001 < area_ratio < 0.95:
                confidence = 0.90
            elif 0.0001 < area_ratio <= 0.001 or 0.95 <= area_ratio < 0.99:
                confidence = 0.70
            else:
                confidence = 0.50
            confidence_level = classify_confidence(confidence)
            warning = None
            if confidence_level in ("medium", "low"):
                warning = (
                    f"表格#{idx} 置信度偏低({confidence:.2f})，"
                    f"建议人工复核（页码={page_idx}，坐标=[{x1},{y1},{x2},{y2}]）"
                )
                logger.warning(f"{tag} {warning}")
            tables.append(DetectedTable(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                confidence_level=confidence_level,
                page_idx=page_idx,
                table_idx=idx,
                warning=warning,
            ))
        return tables


# ─────────────────────────────────────────────
# 补充依赖（若项目中未引入cv2，需添加）
# ─────────────────────────────────────────────
try:
    import cv2
except ImportError:
    raise RuntimeError("未找到cv2包，请执行：pip install opencv-python")