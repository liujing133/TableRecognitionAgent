import cv2
import numpy as np
from PIL import Image

def preprocess_table_img(img: np.ndarray):
    """
    扫描件矫正、去噪、二值化、裁剪
    实现类型：规则算法，不消耗大模型
    """
    # 1. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 高斯降噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. 二值化
    _, bin_img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 4. 透视倾斜矫正
    coords = np.column_stack(np.where(bin_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    corrected = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return corrected

def crop_table_region(img: np.ndarray, box: list):
    """根据检测框裁剪表格区域，附带页面锚点坐标"""
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2], {"x1":x1, "y1":y1, "x2":x2, "y2":y2}