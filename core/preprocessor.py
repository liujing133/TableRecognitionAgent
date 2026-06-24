import cv2
import numpy as np
from PIL import Image

def preprocess_table_img(img: np.ndarray):
    """
    扫描件矫正、去噪、二值化、裁剪
    实现类型：规则算法，不消耗大模型
    """
    h, w = img.shape[:2]
    # 1. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 高斯降噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. 二值化（去掉INV，白底黑字正常阈值，避免前景像素错乱）
    _, bin_img = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 只提取白色文字区域（白底黑字：文字=255）
    coords = np.column_stack(np.where(bin_img == 255))
    # 无文字直接返回原图，不做旋转
    if len(coords) == 0:
        return img.copy()

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # 优化角度矫正逻辑，杜绝90度翻转
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle

    # 拦截异常大角度：超过±30度判定为计算错误，放弃矫正
    if abs(angle) > 30:
        rot_angle = 0
    else:
        rot_angle = -angle

    # 旋转矩阵
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rot_angle, 1.0)
    corrected = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return corrected

def crop_table_region(img: np.ndarray, box: list):
    """根据检测框裁剪表格区域，附带页面锚点坐标"""
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2], {"x1":x1, "y1":y1, "x2":x2, "y2":y2}