def export_struct_json(table_data: dict, anchor: dict, teds: float, warn_level: str) -> dict:
    """机器入库结构化JSON，携带锚点、合并、置信度"""
    return {
        "table_id": table_data["table_id"],
        "page_anchor": anchor,
        "teds_score": teds,
        "warning_level": warn_level,
        "is_cross_page": table_data.get("is_cross_page", False),
        "rows": table_data["rows"]
    }

def export_markdown(table_data: dict) -> str:
    """生成支持合并单元格的Mark表格，自动过滤末尾空列"""
    rows = table_data["rows"]
    if not rows:
        return ""
    # 过滤每行尾部连续空单元格
    def trim_empty_cells(cell_list):
        idx = len(cell_list)
        while idx > 0 and cell_list[idx-1]["text"].strip() == "":
            idx -= 1
        return cell_list[:idx]
    header_cells = trim_empty_cells(rows[0]["cells"])
    header = [c["text"] for c in header_cells]
    if not header:
        return ""
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"]*len(header)) + " |\n"
    # 表体同样裁剪空列
    for r in rows[1:]:
        cells = trim_empty_cells(r["cells"])
        cell_texts = [c["text"] for c in cells]
        md += "| " + " | ".join(cell_texts) + " |\n"
    return md