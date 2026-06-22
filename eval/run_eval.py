import os
import json
import yaml
from core.teds_metric import calc_teds
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def run_batch_eval(test_data_dir: str):
    """批量评测数据集，输出TEDS指标报告"""
    trace_id = "eval_batch_" + str(os.urandom(4).hex())
    total_simple_teds = []
    total_complex_teds = []
    cross_page_correct = 0
    total_cross = 0
    for fname in os.listdir(test_data_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(test_data_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        pred = data["pred_table"]
        gold = data["gold_table"]
        teds = calc_teds(pred, gold)
        if data["table_type"] == "simple":
            total_simple_teds.append(teds)
        elif data["table_type"] == "complex":
            total_complex_teds.append(teds)
        if data["is_cross_page"]:
            total_cross +=1
            if pred["is_cross_page"] and len(pred["rows"]) > len(gold["rows"][1:]):
                cross_page_correct +=1
    # 计算指标
    avg_simple = sum(total_simple_teds)/len(total_simple_teds) if total_simple_teds else 0
    avg_complex = sum(total_complex_teds)/len(total_complex_teds) if total_complex_teds else 0
    cross_acc = cross_page_correct / total_cross if total_cross>0 else 0
    report = {
        "avg_simple_table_teds": round(avg_simple,3),
        "avg_complex_table_teds": round(avg_complex,3),
        "cross_page_merge_acc": round(cross_acc,3)
    }
    log_trace(trace_id, "EVAL_REPORT", report)
    print("====评测指标报告====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == "__main__":
    run_batch_eval("./eval/test_dataset")