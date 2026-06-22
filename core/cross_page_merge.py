from transformers import BertTokenizer, BertModel
import torch
import yaml
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
sim_thresh = cfg["cross_page"]["sim_threshold"]

tokenizer = None
bert = None

def load_bert_model():
    global tokenizer, bert
    if tokenizer is None or bert is None:
        # 本地离线路径
        local_bert_path = "./models/bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(local_bert_path)
        bert = BertModel.from_pretrained(local_bert_path)

def get_text_emb(text: str):
    load_bert_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:,0,:]

def calc_sim(vec1, vec2) -> float:
    cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2).item()
    return round((cos_sim + 1)/2, 3)

def merge_cross_page(last_table: dict, curr_table: dict, trace_id: str):
    """匹配上一页尾表与当前页首表，拼接完整逻辑表格"""
    last_header = " ".join([c["text"] for c in last_table["rows"][0]["cells"]])
    curr_header = " ".join([c["text"] for c in curr_table["rows"][0]["cells"]])
    sim = calc_sim(get_text_emb(last_header), get_text_emb(curr_header))
    log_trace(trace_id, "CrossPageMerge", {"header_sim": sim})
    if sim >= sim_thresh and len(last_table["rows"][0]["cells"]) == len(curr_table["rows"][0]["cells"]):
        # 去除当前页重复表头，合并行
        merge_rows = last_table["rows"] + curr_table["rows"][1:]
        merged = {
            "table_id": last_table["table_id"],
            "rows": merge_rows,
            "is_cross_page": True
        }
        return merged
    return curr_table
