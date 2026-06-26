import cv2
import numpy as np
from table_agent.ocr.ocr_engine import OCREngine

# 测试空白图像OCR
ocr = OCREngine()
test_img = np.zeros((500, 500, 3), dtype=np.uint8)
result = ocr.extract(test_img)
print(f"测试结果：识别到{len(result.blocks)}个文字块，平均置信度={result.avg_confidence}")