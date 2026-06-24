"""
合成测试图像生成脚本
实现类型：[规则] 纯OpenCV绘图操作，无模型调用

生成6种典型场景的测试图像：
  1. normal_table.jpg    - 标准带边框表格（基准测试）
  2. skewed_table.jpg    - 倾斜拍摄的表格（测试透视矫正）
  3. noisy_table.jpg     - 含噪声的扫描件（测试去噪）
  4. low_contrast.jpg    - 低对比度表格（测试二值化）
  5. sparse_table.jpg    - 稀疏表格（测试表格检测 - 大空白区域）
  6. dense_table.jpg     - 密集表格（测试多行多列表格）

用法：
  python tests/generate_test_samples.py
  # 在 tests/samples/ 目录下生成测试图片
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


OUTPUT_DIR = Path(__file__).parent / "samples"


def _draw_table(
    img: np.ndarray,
    rows: int,
    cols: int,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
    texts: List[List[str]],
    border_color: Tuple[int, int, int] = (0, 0, 0),
    border_thickness: int = 1,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    在图像上绘制一个带边框和文字的表格。
    实现类型：[规则]
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 0, 0)
    font_thickness = 1

    for r in range(rows):
        for c in range(cols):
            x0 = x + c * cell_w
            y0 = y + r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            # 填充背景
            cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, -1)

            # 绘制边框
            cv2.rectangle(img, (x0, y0), (x1, y1), border_color, border_thickness)

            # 写入文字
            if r < len(texts) and c < len(texts[r]):
                text = texts[r][c]
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x0 + (cell_w - text_size[0]) // 2
                text_y = y0 + (cell_h + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)


def generate_normal_table() -> np.ndarray:
    """标准表格 - 有边框，正常光照"""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    texts = [
        ["姓名", "年龄", "部门", "职位"],
        ["张三", "28", "技术部", "高级工程师"],
        ["李四", "32", "产品部", "产品经理"],
        ["王五", "26", "市场部", "市场专员"],
        ["赵六", "35", "技术部", "架构师"],
        ["陈七", "30", "人事部", "HR主管"],
    ]
    _draw_table(img, rows=len(texts), cols=4, x=50, y=50, cell_w=160, cell_h=50, texts=texts)
    return img


def generate_skewed_table(angle: float = -7.0) -> np.ndarray:
    """倾斜表格 - 模拟扫描时放置不正"""
    normal = generate_normal_table()
    h, w = normal.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    skewed = cv2.warpAffine(normal, M, (w, h), borderValue=(200, 200, 200))
    return skewed


def generate_noisy_table(noise_level: int = 30) -> np.ndarray:
    """含噪声的表格 - 模拟老旧扫描件"""
    normal = generate_normal_table()
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level, normal.shape).astype(np.int16)
    noisy = np.clip(normal.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def generate_low_contrast() -> np.ndarray:
    """低对比度表格 - 模拟淡色打印/复印褪色"""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 230  # 浅灰背景

    texts = [
        ["项目", "负责人", "截止日期", "状态"],
        ["系统升级", "张三", "2026-07-01", "进行中"],
        ["接口开发", "李四", "2026-06-28", "已完成"],
        ["测试用例", "王五", "2026-07-05", "未开始"],
        ["文档编写", "赵六", "2026-07-10", "进行中"],
    ]
    # 使用浅灰色边框和文字模拟低对比度
    border_color = (180, 180, 180)
    _draw_table(img, rows=len(texts), cols=4, x=50, y=50, cell_w=160, cell_h=50, texts=texts, border_color=border_color)
    # 文字也改为浅灰
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(len(texts)):
        for c in range(len(texts[r])):
            x0 = 50 + c * 160
            y0 = 50 + r * 50
            text = texts[r][c]
            # 在原有位置重写浅灰色文字
            cv2.putText(img, text, (x0 + 5, y0 + 32), font, 0.4, (140, 140, 140), 1)
    return img


def generate_sparse_table() -> np.ndarray:
    """稀疏大表格 - 只有几行数据，用于检测大空白区域"""
    h, w = 700, 900
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    texts = [
        ["编号", "名称", "规格", "数量", "单价", "备注"],
        ["001", "螺钉 M6×20", "不锈钢", "500", "0.50", ""],
        ["002", "螺母 M6", "碳钢", "300", "0.30", ""],
        ["003", "垫圈 Φ6", "弹簧钢", "800", "0.10", ""],
    ]
    _draw_table(img, rows=len(texts), cols=6, x=30, y=30, cell_w=130, cell_h=45, texts=texts)
    return img


def generate_dense_table() -> np.ndarray:
    """密集表格 - 多行多列，模拟复杂数据表"""
    h, w = 800, 1000
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    headers = ["序号", "姓名", "语文", "数学", "英语", "物理", "化学", "生物", "总分", "排名"]
    rows_data = [
        [str(i), f"学生{i}", str(80 + (i * 3) % 20), str(75 + (i * 5) % 25),
         str(85 + (i * 2) % 15), str(70 + (i * 7) % 25), str(78 + (i * 4) % 20),
         str(82 + (i * 3) % 18), str(600 + i * 10), str(i)]
        for i in range(1, 13)
    ]
    texts = [headers] + rows_data
    _draw_table(img, rows=len(texts), cols=10, x=20, y=20, cell_w=90, cell_h=40, texts=texts)
    return img


def generate_all(output_dir: Path = OUTPUT_DIR) -> List[str]:
    """生成所有测试图像"""
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("normal_table.jpg", generate_normal_table, "标准带边框表格"),
        ("skewed_table.jpg", generate_skewed_table, "倾斜拍摄表格（-7°旋转）"),
        ("noisy_table.jpg", generate_noisy_table, "含高斯噪声表格"),
        ("low_contrast.jpg", generate_low_contrast, "低对比度表格"),
        ("sparse_table.jpg", generate_sparse_table, "稀疏大表格"),
        ("dense_table.jpg", generate_dense_table, "密集成绩表"),
    ]

    generated = []
    for filename, gen_fn, desc in generators:
        img = gen_fn()
        save_path = str(output_dir / filename)
        cv2.imwrite(save_path, img)
        generated.append(save_path)
        print(f"  ✅ {filename} - {desc} ({img.shape[1]}x{img.shape[0]})")

    # 额外生成一个多表格页面（2个表在同一页）
    multi_img = np.ones((900, 800, 3), dtype=np.uint8) * 255
    texts1 = [["课程", "学分", "成绩"], ["数学", "4", "92"], ["英语", "3", "88"]]
    texts2 = [["项目", "分值", "评级"], ["期中", "30", "A"], ["期末", "70", "A+"]]
    _draw_table(multi_img, rows=len(texts1), cols=3, x=30, y=30, cell_w=150, cell_h=45, texts=texts1)
    _draw_table(multi_img, rows=len(texts2), cols=3, x=30, y=450, cell_w=150, cell_h=45, texts=texts2)

    multi_path = str(output_dir / "multi_table.jpg")
    cv2.imwrite(multi_path, multi_img)
    generated.append(multi_path)
    print(f"  ✅ multi_table.jpg - 单页多表格场景 (800x900)")

    print(f"\n🎯 共生成 {len(generated)} 张测试图像到: {output_dir}")
    return generated


if __name__ == "__main__":
    print(f"生成合成测试图像...")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    generate_all()
