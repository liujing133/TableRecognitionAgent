# core/spatial_clustering.py
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def _estimate_eps(coords):
    if len(coords) < 2:
        return 10.0
    heights = []
    for item in coords:
        pts = item[4]
        pts_np = np.array(pts, dtype=float)
        if pts_np.ndim == 2 and pts_np.shape[1] == 2:
            h = float(np.max(pts_np[:, 1]) - np.min(pts_np[:, 1]))
        elif pts_np.ndim == 1 and pts_np.size == 4:
            h = float(pts_np[3] - pts_np[1])
        else:
            h = 10.0
        heights.append(h)
    heights = np.array(heights)
    median_h = float(np.median(heights))
    # 收紧DBSCAN，保证表格能正常分行
    return max(median_h * 1.15, 8.0)

def get_bounds(pts):
    """兼容 4点二维数组 / xyxy一维数组，输出 x1,y1,x2,y2 float"""
    pts_np = np.array(pts, dtype=float)
    if pts_np.ndim == 2 and pts_np.shape[1] == 2:
        x1 = np.min(pts_np[:, 0])
        y1 = np.min(pts_np[:, 1])
        x2 = np.max(pts_np[:, 0])
        y2 = np.max(pts_np[:, 1])
    elif pts_np.ndim == 1 and pts_np.size == 4:
        x1, y1, x2, y2 = pts_np[0], pts_np[1], pts_np[2], pts_np[3]
    else:
        x1 = y1 = x2 = y2 = 0.0
    return x1, y1, x2, y2

def _fuse_vertical_blocks_in_row(row_item_list, vert_tol=22, overlap_ratio=0.05):
    """单一行簇内融合垂直紧贴文字，不跨表格行"""
    if len(row_item_list) <= 1:
        return row_item_list
    fused = []
    used = set()
    item_cnt = len(row_item_list)
    # 先按文字垂直中线从上到下，再按x从左到右，文字顺序正常
    def sort_key(item):
        cx, cy = item[0], item[1]
        return (cy, cx)
    row_item_list.sort(key=sort_key)
    for i in range(item_cnt):
        if i in used:
            continue
        base = row_item_list[i]
        bx1, by1, bx2, by2 = get_bounds(base[4])
        fuse_text = base[2]
        fx1, fy1, fx2, fy2 = bx1, by1, bx2, by2
        for j in range(i+1, item_cnt):
            if j in used:
                continue
            curr = row_item_list[j]
            cx1, cy1, cx2, cy2 = get_bounds(curr[4])
            vert_gap = cy1 - fy2
            # 间隙过大直接跳过
            if vert_gap > vert_tol:
                continue
            # 防止跨大行合并
            if cy1 > fy2 + vert_tol * 1.5:
                continue
            # 水平重叠校验
            ovl_x1 = max(bx1, cx1)
            ovl_x2 = min(bx2, cx2)
            ovl_w = ovl_x2 - ovl_x1
            base_w = bx2 - bx1
            if ovl_w / base_w < overlap_ratio:
                continue
            # 合并文本与包围盒
            fuse_text += curr[2]
            fx1 = min(fx1, cx1)
            fy1 = min(fy1, cy1)
            fx2 = max(fx2, cx2)
            fy2 = max(fy2, cy2)
            used.add(j)
        new_pts = [[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]]
        new_item = [
            np.mean([fx1, fx2]),
            np.mean([fy1, fy2]),
            fuse_text,
            base[3],
            new_pts
        ]
        fused.append(new_item)
    fused.sort(key=lambda x: x[0])
    return fused

def _align_columns_across_rows(sorted_rows):
    if not sorted_rows:
        return sorted_rows
    ref_idx = max(range(len(sorted_rows)), key=lambda i: len(sorted_rows[i]))
    ref_row = sorted_rows[ref_idx]
    col_boundaries = []
    for item in ref_row:
        x1, _, x2, _ = get_bounds(item[4])
        col_boundaries.append((x1, x2))
    if len(col_boundaries) < 1:
        return sorted_rows
    aligned_rows = []
    for row_items in sorted_rows:
        col_dict = {}
        for item in row_items:
            x1, _, x2, _ = get_bounds(item[4])
            best_col = -1
            best_overlap = 0
            for ci, (bx1, bx2) in enumerate(col_boundaries):
                overlap = max(0.0, min(x2, bx2) - max(x1, bx1))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = ci
            if best_col >= 0 and best_overlap > 0:
                if best_col in col_dict:
                    exist = col_dict[best_col]
                    exist[2] = exist[2] + " " + item[2]
                    ex1, ey1, ex2, ey2 = get_bounds(exist[4])
                    cx1, cy1, cx2, cy2 = get_bounds(item[4])
                    nx1, ny1 = min(ex1, cx1), min(ey1, cy1)
                    nx2, ny2 = max(ex2, cx2), max(ey2, cy2)
                    exist[4] = [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]]
                else:
                    col_dict[best_col] = list(item)
        new_row = []
        for ci in range(len(col_boundaries)):
            new_row.append(col_dict.get(ci, ["", "", "", 0.0, []]))
        aligned_rows.append(new_row)

    # 关闭跨多行兜底拼接，避免不同行文字乱合并
    return aligned_rows

def cluster_rows_cols(text_blocks, cfg):
    if not text_blocks:
        return []
    centers = []
    for block in text_blocks:
        if "points" in block and block["points"]:
            pts = np.array(block["points"])
        elif "bbox" in block and len(block["bbox"]) == 4:
            x1, y1, x2, y2 = block["bbox"]
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            continue
        score = block.get("score", block.get("confidence", 0.0))
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        centers.append([cx, cy, block["text"], score, pts])
    
    if not centers:
        return []
    centers = sorted(centers, key=lambda x: x[1])
    coords = np.array([[c[0], c[1]] for c in centers])

    # DBSCAN一维Y分行，保证表格垂直行结构
    eps_row = _estimate_eps(centers)
    row_labels = DBSCAN(eps=eps_row, min_samples=1).fit(coords[:, 1].reshape(-1, 1)).labels_
    rows_dict = defaultdict(list)
    for label, item in zip(row_labels, centers):
        rows_dict[label].append(item)
    
    sorted_rows = []
    for label in sorted(rows_dict.keys()):
        raw_row = rows_dict[label]
        # 行内融合：仅同一表格行内垂直小字合并
        fused_row = _fuse_vertical_blocks_in_row(raw_row, vert_tol=22, overlap_ratio=0.05)
        sorted_rows.append(fused_row)
    
    aligned_rows = _align_columns_across_rows(sorted_rows)
    final_grid = []
    for row_items in aligned_rows:
        cells = []
        for item in row_items:
            text = item[2] if isinstance(item[2], str) else ""
            score = item[3] if isinstance(item[3], (int, float)) else 0.0
            raw_pts = item[4]
            # 修复报错：统一使用安全边界读取
            x1, y1, x2, y2 = get_bounds(raw_pts)
            bbox_display = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            cells.append({
                "text": text,
                "score": score,
                "bbox": bbox_display,
                "rowspan": 1,
                "colspan": 1,
            })
        final_grid.append(cells)
    
    # 计算colspan
    if not final_grid:
        return final_grid
    ref_row = final_grid[0]
    col_boundaries = []
    for cell in ref_row:
        pts = np.array(cell["bbox"], dtype=float)
        cx1, cy1, cx2, cy2 = get_bounds(pts)
        col_boundaries.append((cx1, cx2))
    max_col_cnt = len(col_boundaries)
    if max_col_cnt == 0:
        return final_grid
    for ri, row in enumerate(final_grid):
        new_cells = []
        for cell in row:
            pts = np.array(cell["bbox"], dtype=float)
            cx1, cy1, cx2, cy2 = get_bounds(pts)
            span = sum(1 for (bx1, bx2) in col_boundaries if not (cx2 < bx1 or cx1 > bx2))
            cell["colspan"] = max(1, span)
            new_cells.append(cell)
        final_grid[ri] = new_cells
    
    # 过滤空白行
    clean_grid = []
    for r in final_grid:
        if any(c["text"].strip() for c in r):
            clean_grid.append(r)
    return clean_grid