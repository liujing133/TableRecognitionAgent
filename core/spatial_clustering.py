# core/spatial_clustering.py
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import yaml

def cluster_rows_cols(text_blocks, cfg):
    """
    输入: OCR识别的文字块列表，兼容两种格式：
    - 格式1: {"points": [[x1,y1],...], "text": str, "score": float}
    - 格式2: {"bbox": [x1,y1,x2,y2], "text": str, "confidence": float}
    输出: 初始化的行列网格（带空位填充）
    """
    if not text_blocks:
        return []
    
    # 1. 提取所有文字框的中心点坐标 (cx, cy)
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
    
    # 2. 行聚类 (DBSCAN，基于垂直坐标cy)
    # eps 设为平均文字高度的 0.8 倍左右，可根据图片尺寸动态调整
    img_h = max(coords[:, 1]) - min(coords[:, 1]) if len(coords)>0 else 0
    eps_row = max(img_h / 20, 5)  # 动态阈值，防止单张图过密或过疏
    row_labels = DBSCAN(eps=eps_row, min_samples=1).fit(coords[:, 1].reshape(-1, 1)).labels_
    
    # 3. 按行标签分组，并在每一行内按水平坐标 cx 排序
    rows_dict = defaultdict(list)
    for label, item in zip(row_labels, centers):
        rows_dict[label].append(item)
    
    sorted_rows = []
    for label in sorted(rows_dict.keys()):
        row_items = rows_dict[label]
        row_items.sort(key=lambda x: x[0])  # 按 cx 排序
        sorted_rows.append(row_items)
    
    # 4. 构建初始网格
    final_grid = []
    for row_items in sorted_rows:
        cells = []
        for item in row_items:
            cells.append({
                "text": item[2],
                "score": item[3],
                "bbox": item[4].tolist(),
                "rowspan": 1,
                "colspan": 1
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