def _table_to_html(table: dict) -> str:
    rows = table.get("rows", [])
    html_parts = ["<table>"]
    for row in rows:
        html_parts.append("<tr>")
        for cell in row.get("cells", []):
            tag = "th" if cell.get("is_header", False) else "td"
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            text = (cell.get("text", "") or "").strip()
            attrs = ""
            if colspan > 1: attrs += f' colspan="{colspan}"'
            if rowspan > 1: attrs += f' rowspan="{rowspan}"'
            html_parts.append(f"<{tag}{attrs}>{text}</{tag}>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)


def _normalized_edit_dist(a: str, b: str) -> float:
    if a == b: return 0.0
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(n, m)


def _calc_structure_score(pred_table: dict, gold_table: dict) -> dict:
    pred_rows = pred_table.get("rows", [])
    gold_rows = gold_table.get("rows", [])
    if len(gold_rows) > 0:
        row_match = 1.0 - abs(len(pred_rows) - len(gold_rows)) / max(len(pred_rows), len(gold_rows))
    else:
        row_match = 1.0 if len(pred_rows) == 0 else 0.0
    col_match = merge_match = cell_text_match = 0.0
    n_compare = min(len(pred_rows), len(gold_rows))
    if n_compare > 0:
        col_scores, merge_scores, text_scores = [], [], []
        for r in range(n_compare):
            pc, gc = pred_rows[r].get("cells", []), gold_rows[r].get("cells", [])
            n_cells = min(len(pc), len(gc))
            if n_cells == 0: continue
            col_scores.append(n_cells / max(len(pc), len(gc)))
            for c in range(n_cells):
                p_cs, g_cs = int(pc[c].get("colspan", 1)), int(gc[c].get("colspan", 1))
                p_rs, g_rs = int(pc[c].get("rowspan", 1)), int(gc[c].get("rowspan", 1))
                if p_cs == g_cs and p_rs == g_rs: merge_scores.append(1.0)
                elif abs(p_cs - g_cs) <= 1 and abs(p_rs - g_rs) <= 1: merge_scores.append(0.5)
                else: merge_scores.append(0.0)
                pt, gt = (pc[c].get("text", "") or "").strip(), (gc[c].get("text", "") or "").strip()
                if gt: text_scores.append(1.0 if pt == gt else (0.5 if pt and (pt in gt or gt in pt) else 0.0))
                else: text_scores.append(1.0 if not pt else 0.0)
        if col_scores: col_match = sum(col_scores) / len(col_scores)
        if merge_scores: merge_match = sum(merge_scores) / len(merge_scores)
        if text_scores: cell_text_match = sum(text_scores) / len(text_scores)
    return {"row_match": row_match, "col_match": col_match, "merge_match": merge_match, "cell_text_match": cell_text_match}


def calc_teds(pred_table: dict, gold_table: dict) -> float:
    pred_html = _table_to_html(pred_table)
    gold_html = _table_to_html(gold_table)
    html_ed = _normalized_edit_dist(pred_html, gold_html)
    html_sim = 1.0 - html_ed
    struct = _calc_structure_score(pred_table, gold_table)
    teds = 0.40 * html_sim + 0.25 * struct["row_match"] + 0.15 * struct["col_match"] + 0.10 * struct["merge_match"] + 0.10 * struct["cell_text_match"]
    return round(max(0.0, min(1.0, teds)), 3)


def get_warning_level(teds_score: float, cfg: dict) -> str:
    high = cfg["teds_threshold"]["high"]
    mid = cfg["teds_threshold"]["mid"]
    if teds_score >= high: return "high_conf"
    elif teds_score >= mid: return "mid_conf_warn"
    else: return "low_conf_alert"