# core/spatial_clustering.py
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import yaml


def _estimate_eps(coords):
    """基于文字块高度分布自适应估计 eps"""
    if len(coords) < 2:
        return 10.0
    heights = []
    for item in coords:
        pts = item[4]  # polygon or bbox
        pts_np = np.array(pts, dtype=float)
        if pts_np.ndim == 2 and pts_np.shape[1] == 2:
            h = float(np.max(pts_np[:, 1]) - np.min(pts_np[:, 1]))
        elif pts_np.ndim == 1 and pts_np.size == 4:
            h = float(pts_np[3] - pts_np[1])
        else:
            h = 10.0
        heights.append(h)
    heights = np.array(heights)
    # 取中位数高度 × 1.2 作为行聚类 eps
    median_h = float(np.median(heights))
    return max(median_h * 1.2, 5.0)


def _get_bbox_x_range(pts):
    """从多边形或 xyxy 格式提取 x 范围"""
    pts_np = np.array(pts, dtype=float)
    if pts_np.ndim == 2 and pts_np.shape[1] == 2:
        return float(np.min(pts_np[:, 0])), float(np.max(pts_np[:, 0]))
    elif pts_np.ndim == 1 and pts_np.size == 4:
        return float(pts_np[0]), float(pts_np[2])
    return 0.0, 0.0


def _get_bbox_y_range(pts):
    pts_np = np.array(pts, dtype=float)
    if pts_np.ndim == 2 and pts_np.shape[1] == 2:
        return float(np.min(pts_np[:, 1])), float(np.max(pts_np[:, 1]))
    elif pts_np.ndim == 1 and pts_np.size == 4:
        return float(pts_np[1]), float(pts_np[3])
    return 0.0, 0.0


def _align_columns_across_rows(sorted_rows):
    """
    跨行对齐列：用最长行（参考行）的 x 范围作为列边界参考，
    其他行按 x 重叠匹配到对应列。避免 DBSCAN 过分割问题。
    """
    if not sorted_rows:
        return sorted_rows

    # 1. 找参考行（格子最多的行）
    ref_idx = max(range(len(sorted_rows)), key=lambda i: len(sorted_rows[i]))
    ref_row = sorted_rows[ref_idx]

    # 2. 从参考行定义列边界
    col_boundaries = []
    for item in ref_row:
        x1, x2 = _get_bbox_x_range(item[4])
        col_boundaries.append((x1, x2))

    if len(col_boundaries) < 1:
        return sorted_rows

    # 3. 每行的 text block 按 x 重叠匹配到对应列
    aligned_rows = []
    for row_items in sorted_rows:
        col_dict = {}
        for item in row_items:
            x1, x2 = _get_bbox_x_range(item[4])
            best_col = -1
            best_overlap = 0
            for ci, (bx1, bx2) in enumerate(col_boundaries):
                overlap = max(0.0, min(x2, bx2) - max(x1, bx1))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = ci
            if best_col >= 0 and best_overlap > 0:
                if best_col in col_dict:
                    # 同列多个 text block → 合并文本和 bbox
                    existing = col_dict[best_col]
                    existing[2] = existing[2] + " " + item[2]
                    e_pts = np.array(existing[4], dtype=float)
                    n_pts = np.array(item[4], dtype=float)
                    if e_pts.ndim == 2 and n_pts.ndim == 2:
                        xs = np.concatenate([e_pts[:, 0], n_pts[:, 0]])
                        ys = np.concatenate([e_pts[:, 1], n_pts[:, 1]])
                        existing[4] = [[float(np.min(xs)), float(np.min(ys))],
                                       [float(np.max(xs)), float(np.min(ys))],
                                       [float(np.max(xs)), float(np.max(ys))],
                                       [float(np.min(xs)), float(np.max(ys))]]
                else:
                    col_dict[best_col] = list(item)

        # 按列顺序输出，缺列补空
        new_row = []
        for ci in range(len(col_boundaries)):
            if ci in col_dict:
                new_row.append(col_dict[ci])
            else:
                new_row.append(["", "", "", 0.0, []])
        aligned_rows.append(new_row)

    return aligned_rows


def cluster_rows_cols(text_blocks, cfg):
    """
    输入: OCR识别的文字块列表，兼容两种格式：
    - 格式1: {"points": [[x1,y1],...], "text": str, "score": float}
    - 格式2: {"bbox": [x1,y1,x2,y2], "text": str, "confidence": float}
    输出: 初始化的行列网格（带空位填充）
    """
    if not text_blocks:
        return []
    # 1. 提取所有文字框信息
    centers = []
    for block in text_blocks:
        # 兼容 points/bbox 两种格式
        if "points" in block and block["points"]:
            pts = np.array(block["points"])
        elif "bbox" in block and len(block["bbox"]) == 4:
            # 从bbox转换为四点格式 [x1,y1, x2,y1, x2,y2, x1,y2]
            x1, y1, x2, y2 = block["bbox"]
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            continue  # 跳过无效块
        
        # 兼容 score/confidence 字段
        score = block.get("score", block.get("confidence", 0.0))
        
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        centers.append([cx, cy, block["text"], score, pts])
    
    if not centers:  # 无有效文字块时直接返回空
        return []
    
    centers = sorted(centers, key=lambda x: x[1])  # 按垂直方向初步排序
    coords = np.array([[c[0], c[1]] for c in centers])
    # 2. 行聚类 (DBSCAN，基于垂直坐标 cy)
    eps_row = _estimate_eps(centers)
    row_labels = DBSCAN(eps=eps_row, min_samples=1).fit(coords[:, 1].reshape(-1, 1)).labels_
    # 3. 按行标签分组，行内按水平坐标 cx 排序
    rows_dict = defaultdict(list)
    for label, item in zip(row_labels, centers):
        rows_dict[label].append(item)
    sorted_rows = []
    for label in sorted(rows_dict.keys()):
        row_items = rows_dict[label]
        row_items.sort(key=lambda x: x[0])
        sorted_rows.append(row_items)
    # 4. 跨行列对齐
    aligned_rows = _align_columns_across_rows(sorted_rows)
    # 5. 构建基础网格
    final_grid = []
    for row_items in aligned_rows:
        cells = []
        for item in row_items:
            text = item[2] if isinstance(item[2], str) else ""
            score = item[3] if isinstance(item[3], (int, float)) else 0.0
            raw_bbox = item[4]
            if isinstance(raw_bbox, np.ndarray):
                pts_np = raw_bbox
            elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) > 0:
                pts_np = np.array(raw_bbox, dtype=float)
            else:
                pts_np = np.array([])
            if pts_np.ndim == 2 and pts_np.shape[1] == 2:
                x1 = float(np.min(pts_np[:, 0]))
                y1 = float(np.min(pts_np[:, 1]))
                x2 = float(np.max(pts_np[:, 0]))
                y2 = float(np.max(pts_np[:, 1]))
                bbox_display = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                bbox_display = []
            cells.append({
                "text": text,
                "score": score,
                "bbox": bbox_display,
                "rowspan": 1,
                "colspan": 1,
            })
        final_grid.append(cells)

    # ========== 修复后的合并单元格推断逻辑（无未定义变量） ==========
    if not final_grid:
        return final_grid
    # 获取参考列边界（第一行）
    ref_row = final_grid[0]
    col_boundaries = []
    for cell in ref_row:
        pts = np.array(cell["bbox"], dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            x1 = float(np.min(pts[:, 0]))
            x2 = float(np.max(pts[:, 0]))
            col_boundaries.append((x1, x2))
    max_col_cnt = len(col_boundaries)
    if max_col_cnt == 0:
        return final_grid

    # 逐行计算单元格跨列colspan
    for ri, row in enumerate(final_grid):
        inferred_cols = 0
        new_cells = []
        for cell in row:
            pts = np.array(cell["bbox"], dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 2:
                cx1 = float(np.min(pts[:, 0]))
                cx2 = float(np.max(pts[:, 0]))
            else:
                cx1, cx2 = 0, 0
            # 计算当前单元格覆盖多少列
            span = sum(1 for (bx1, bx2) in col_boundaries if not (cx2 < bx1 or cx1 > bx2))
            span = max(1, span)
            cell["colspan"] = span
            new_cells.append(cell)
            inferred_cols += span
        # 替换当前行
        final_grid[ri] = new_cells

    # 可选：过滤全空白行，解决导出大量空行空白单元格
    clean_grid = []
    for r in final_grid:
        has_text = any(c["text"].strip() for c in r)
        if has_text:
            clean_grid.append(r)
    return clean_grid