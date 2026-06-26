import importlib.util
import yaml
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
sim_thresh = cfg["cross_page"]["sim_threshold"]

_tokenizer = None
_bert = None
_torch_available = importlib.util.find_spec("torch") is not None
_transformers_available = importlib.util.find_spec("transformers") is not None


def _load_bert_model():
    global _tokenizer, _bert
    if _tokenizer is None or _bert is None:
        from transformers import BertTokenizer, BertModel
        local_bert_path = "./models/bert-base-chinese"
        _tokenizer = BertTokenizer.from_pretrained(local_bert_path)
        _bert = BertModel.from_pretrained(local_bert_path)


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def _header_to_string(table: dict, max_rows: int = 1) -> str:
    rows = table.get("rows", [])[:max_rows]
    header_lines = []
    for row in rows:
        line = " ".join([_normalize_text(c.get("text", "")) for c in row.get("cells", []) if c.get("text")])
        if line:
            header_lines.append(line)
    return " | ".join(header_lines)


def _header_rows_to_strings(table: dict, max_rows: int = 3) -> list:
    """返回多行表头的每行独立字符串列表（用于多级表头逐行对比）"""
    rows = table.get("rows", [])[:max_rows]
    result = []
    for row in rows:
        line = " ".join([_normalize_text(c.get("text", "")) for c in row.get("cells", []) if c.get("text")])
        result.append(line)
    return result


def _header_column_count(table: dict) -> int:
    rows = table.get("rows", [])
    if not rows:
        return 0
    # 基于第一行统计列数，考虑 colspan
    total_cols = 0
    for cell in rows[0].get("cells", []):
        total_cols += int(cell.get("colspan", 1))
    return total_cols


def _calc_similarity_features(text1: str, text2: str) -> tuple[float, float]:
    text1 = _normalize_text(text1)
    text2 = _normalize_text(text2)
    if not text1 or not text2:
        return 0.0, 0.0

    set1 = set(text1.split())
    set2 = set(text2.split())
    union = len(set1 | set2)
    overlap = 0.0
    if union > 0:
        overlap = len(set1 & set2) / union

    bert_sim = 0.0
    if _transformers_available and _torch_available:
        try:
            _load_bert_model()
            import torch
            inputs1 = _tokenizer(text1, return_tensors="pt", truncation=True, max_length=32)
            inputs2 = _tokenizer(text2, return_tensors="pt", truncation=True, max_length=32)
            with torch.no_grad():
                out1 = _bert(**inputs1)
                out2 = _bert(**inputs2)
            vec1 = out1.last_hidden_state[:, 0, :]
            vec2 = out2.last_hidden_state[:, 0, :]
            bert_sim = float(torch.nn.functional.cosine_similarity(vec1, vec2).item())
        except Exception:
            bert_sim = 0.0

    return overlap, bert_sim


def _calc_sim(text1: str, text2: str) -> float:
    text1 = _normalize_text(text1)
    text2 = _normalize_text(text2)
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0

    overlap, bert_sim = _calc_similarity_features(text1, text2)
    if overlap >= 0.5:
        return round(0.5 + 0.5 * overlap, 3)
    if bert_sim > 0:
        score = 0.3 * overlap + 0.7 * ((bert_sim + 1) / 2)
    else:
        score = overlap
    return round(score, 3)


def _row_to_text(row: dict) -> str:
    return " ".join([_normalize_text(c.get("text", "")) for c in row.get("cells", []) if c.get("text")])


def _header_rows_match(last_header_lines: list, curr_start_idx: int, curr_rows: list) -> int:
    """检查 curr_rows 中从 curr_start_idx 开始的连续行是否匹配 last_header_lines。
    返回匹配上的行数（0 表示不匹配）。"""
    n_header = len(last_header_lines)
    matched = 0
    for i in range(n_header):
        if curr_start_idx + i >= len(curr_rows):
            break
        curr_text = _row_to_text(curr_rows[curr_start_idx + i])
        if not curr_text and i < n_header - 1:
            matched += 1
            continue
        if not curr_text:
            break
        last_text = last_header_lines[i]
        if not last_text:
            matched += 1
            continue
        # 用 overlap 检测
        set1 = set(last_text.split())
        set2 = set(curr_text.split())
        union = len(set1 | set2)
        if union == 0:
            matched += 1
            continue
        overlap = len(set1 & set2) / union
        # 如果当前行包含数字而上一页对应行不含数字，则不像表头
        if any(char.isdigit() for char in curr_text) and not any(char.isdigit() for char in last_text):
            if overlap < 0.9:
                break
        if overlap >= 0.75 and len(set2) >= len(set1) * 0.75:
            matched += 1
            continue
        break
    return matched


def _strip_redundant_header_rows(last_table: dict, curr_table: dict) -> list:
    if not last_table.get("rows") or not curr_table.get("rows"):
        return curr_table.get("rows", [])

    last_header_lines = _header_rows_to_strings(last_table, max_rows=3)
    # 过滤掉空行
    last_header_lines = [l for l in last_header_lines if l]
    if not last_header_lines:
        return curr_table.get("rows", [])

    rows = curr_table.get("rows", [])
    keep_idx = 0
    n_header = len(last_header_lines)

    # 尝试匹配多级表头（最多跳过 n_header 行）
    while keep_idx < len(rows):
        matched = _header_rows_match(last_header_lines, keep_idx, rows)
        if matched > 0:
            keep_idx += matched
            continue
        # 如果当前行是空行，跳过
        if not _row_to_text(rows[keep_idx]):
            keep_idx += 1
            continue
        break

    return rows[keep_idx:]


def merge_cross_page(last_table: dict, curr_table: dict, trace_id: str):
    """匹配上一页尾表与当前页首表，拼接完整逻辑表格"""
    if not last_table or not curr_table:
        return curr_table

    # 表头相似度：仅用第一行对比（避免数据行噪声）
    last_header = _header_to_string(last_table, max_rows=1)
    curr_header = _header_to_string(curr_table, max_rows=1)
    last_cols = _header_column_count(last_table)
    curr_cols = _header_column_count(curr_table)
    sim = _calc_sim(last_header, curr_header)
    log_trace(trace_id, "CrossPageMerge", {
        "header_sim": sim,
        "last_header": last_header,
        "curr_header": curr_header,
        "last_cols": last_cols,
        "curr_cols": curr_cols,
    })

    if sim >= sim_thresh and last_cols > 0 and last_cols == curr_cols:
        # 去重剥离：使用多行表头匹配（支持多级表头）
        merged_rows = last_table.get("rows", []) + _strip_redundant_header_rows(last_table, curr_table)
        merged = {
            "table_id": last_table.get("table_id", ""),
            "rows": merged_rows,
            "is_cross_page": True
        }
        log_trace(trace_id, "CrossPageMergeSuccess", {"merged_rows": len(merged_rows), "sim": sim})
        return merged

    log_trace(trace_id, "CrossPageMergeSkip", {"sim": sim, "threshold": sim_thresh})
    return curr_table
