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
    """生成支持合并单元格的Markdown表格"""
    rows = table_data["rows"]
    if not rows:
        return ""
    # 表头
    header = [c["text"] for c in rows[0]["cells"]]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"]*len(header)) + " |\n"
    # 表体
    for r in rows[1:]:
        cells = [c["text"] for c in r["cells"]]
        md += "| " + " | ".join(cells) + " |\n"
    return md