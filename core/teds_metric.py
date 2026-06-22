def calc_teds(pred_table: dict, gold_table: dict) -> float:
    """
    表格树编辑距离相似度TEDS [0,1]
    用于评估还原质量，输出置信度
    """
    # 简化实现：基于行列、合并单元格结构匹配
    pred_rows = pred_table.get("rows", [])
    gold_rows = gold_table.get("rows", [])
    row_match = min(len(pred_rows), len(gold_rows)) / max(len(pred_rows), len(gold_rows)) if max(len(pred_rows), len(gold_rows))>0 else 0
    cell_sim = 0.0
    # 单元格合并属性匹配简化计算
    for r in range(min(len(pred_rows), len(gold_rows))):
        p_cells = pred_rows[r]["cells"]
        g_cells = gold_rows[r]["cells"]
        cell_sim += min(len(p_cells), len(g_cells)) / max(len(p_cells), len(g_cells)) if max(len(p_cells), len(g_cells))>0 else 0
    if min(len(pred_rows), len(gold_rows)) > 0:
        cell_sim /= min(len(pred_rows), len(gold_rows))
    teds = 0.6 * row_match + 0.4 * cell_sim
    return round(teds, 3)

def get_warning_level(teds_score: float, cfg: dict) -> str:
    """根据阈值输出预警等级"""
    high = cfg["teds_threshold"]["high"]
    mid = cfg["teds_threshold"]["mid"]
    if teds_score >= high:
        return "high_conf"
    elif teds_score >= mid:
        return "mid_conf_warn"
    else:
        return "low_conf_alert"