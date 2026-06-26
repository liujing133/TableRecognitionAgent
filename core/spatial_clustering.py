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
    """跨行对齐列：用所有行的 x 中位数位置构建统一列边界"""
    if not sorted_rows:
        return sorted_rows

    # 收集所有 cell 的 x 中心位置
    all_x_centers = []
    for row_items in sorted_rows:
        for item in row_items:
            x1, x2 = _get_bbox_x_range(item[4])
            all_x_centers.append((x1 + x2) / 2.0)

    if len(all_x_centers) < 2:
        return sorted_rows

    # 用 DBSCAN 对所有 x_center 做一维列聚类，得到全局列簇
    x_coords = np.array(all_x_centers).reshape(-1, 1)
    # eps 取 x 方向间距中位数的 0.6 倍
    sorted_x = np.sort(all_x_centers)
    gaps = np.diff(sorted_x)
    median_gap = float(np.median(gaps)) if len(gaps) > 0 else 20.0
    eps_col = max(median_gap * 0.6, 5.0)
    col_labels = DBSCAN(eps=eps_col, min_samples=1).fit(x_coords).labels_

    # 计算每列的中心位置（取该簇所有 x 的中位数）
    col_centers = {}
    for label, cx in zip(col_labels, all_x_centers):
        if label not in col_centers:
            col_centers[label] = []
        col_centers[label].append(cx)

    sorted_cols = sorted(col_centers.keys(), key=lambda k: np.median(col_centers[k]))
    col_idx_map = {label: i for i, label in enumerate(sorted_cols)}

    # 重新排列每个行，将 cell 分配到对应的列位置
    aligned_rows = []
    idx = 0
    for row_items in sorted_rows:
        # 为每行创建一个 dict: col_index -> cell
        col_dict = {}
        for item in row_items:
            x1, x2 = _get_bbox_x_range(item[4])
            cx_item = (x1 + x2) / 2.0
            # 找到最近的列
            best_col = None
            min_dist = float('inf')
            for label, cx_list in col_centers.items():
                col_cx = np.median(cx_list)
                dist = abs(cx_item - col_cx)
                if dist < min_dist:
                    min_dist = dist
                    best_col = label
            if best_col is not None:
                ci = col_idx_map[best_col]
                # 如果该列已有 cell，合并文本（多行文本块属于同一格）
                if ci in col_dict:
                    existing = col_dict[ci]
                    existing[2] = existing[2] + " " + item[2]
                    # 合并 bbox
                    e_pts = np.array(existing[4], dtype=float)
                    n_pts = np.array(item[4], dtype=float)
                    if e_pts.ndim == 2:
                        xs = np.concatenate([e_pts[:, 0], n_pts[:, 0]])
                        ys = np.concatenate([e_pts[:, 1], n_pts[:, 1]])
                        merged_pts = [[float(np.min(xs)), float(np.min(ys))],
                                      [float(np.max(xs)), float(np.min(ys))],
                                      [float(np.max(xs)), float(np.max(ys))],
                                      [float(np.min(xs)), float(np.max(ys))]]
                        existing[4] = merged_pts
                else:
                    col_dict[ci] = list(item)

        # 按列顺序输出
        new_row = []
        for ci in range(len(sorted_cols)):
            if ci in col_dict:
                new_row.append(col_dict[ci])
            else:
                # 缺列 → 插空
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

    # 5. 构建最终网格
    final_grid = []
    for row_items in aligned_rows:
        cells = []
        for item in row_items:
            text = item[2] if isinstance(item[2], str) else ""
            score = item[3] if isinstance(item[3], (int, float)) else 0.0
            raw_bbox = item[4]
            # 处理 numpy array 或 list/tuple
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

    # 5. 统一列数并推断合并单元格
    if final_grid:
        max_col_cnt = max([len(row) for row in final_grid])
        # 5a. 计算列边界：用最长的行（或所有行的众数行）确定 x 分区
        ref_row_idx = max(range(len(final_grid)), key=lambda i: len(final_grid[i]))
        ref_cells = final_grid[ref_row_idx]
        # 计算每列的 x 起始/结束边界
        col_boundaries = []
        for cell in ref_cells:
            pts = np.array(cell["bbox"], dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 2:
                x1, x2 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
            elif pts.ndim == 1 and pts.size == 4:
                x1, x2 = float(pts[0]), float(pts[2])
            else:
                continue
            col_boundaries.append((x1, x2))
        if not col_boundaries:
            col_boundaries = [(float(i) / max_col_cnt, float(i + 1) / max_col_cnt) for i in range(max_col_cnt)]

        # 5b. 检测列数偏少的行：推断 colspan
        for row in final_grid:
            if len(row) >= max_col_cnt:
                continue
            # 该行物理格子少于标准列数 → 可能存在合并单元格
            inferred_cols = 0
            new_cells = []
            for cell in row:
                pts = np.array(cell["bbox"], dtype=float)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    cx1, cx2 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                elif pts.ndim == 1 and pts.size == 4:
                    cx1, cx2 = float(pts[0]), float(pts[2])
                else:
                    cx1, cx2 = 0.0, 0.0
                cx_mid = (cx1 + cx2) / 2.0
                # 计算此 Cell 跨越了几列
                span = 1
                for bi, (bx1, bx2) in enumerate(col_boundaries):
                    if bi < inferred_cols:
                        continue
                    if cx1 <= bx2 and cx2 >= bx1:
                        # 此格子的 x 范围与该列相交
                        if inferred_cols == bi:
                            span = 1
                            inferred_cols += 1
                        else:
                            # 跳过了一些列 → 需要补 colspan
                            gap = bi - inferred_cols
                            if gap > 0:
                                inferred_cols += gap
                                span += gap
                            inferred_cols += 1
                            span += (bi + 1 - inferred_cols) if bi + 1 > inferred_cols else 0
                            break
                # 更精确的跨度计算：以 x 区间覆盖的列数
                actual_span = max(1, sum(1 for bx1, bx2 in col_boundaries
                                         if not (cx2 < bx1 or cx1 > bx2)))
                if actual_span > 1:
                    cell["colspan"] = actual_span
                    inferred_cols += actual_span - 1
                new_cells.append(cell)
                inferred_cols += 1
                if inferred_cols >= max_col_cnt:
                    break

            # 用新推断的单元格替换
            row.clear()
            row.extend(new_cells)
            # 补空到 max_col_cnt
            while len(row) < max_col_cnt:
                row.append({"text": "", "score": 0, "bbox": [], "rowspan": 1, "colspan": 1})

    return final_grid
