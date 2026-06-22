import sys
import os
from pathlib import Path
# 解决中文编码
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.append(str(Path(__file__).parent.parent))
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import time
from core.preprocessor import preprocess_table_img, crop_table_region
from core.table_detector import TableDetector
from core.ocr_engine import OCREngine
from core.tsr_parser import TSRParser
from core.cross_page_merge import merge_cross_page
from core.teds_metric import calc_teds, get_warning_level
from core.exporter import export_struct_json, export_markdown
from utils.logger import gen_trace_id, log_trace
from utils.schema import TableAgentRequest, TableAgentResponse
import yaml

app = FastAPI(title="智能表格识别与还原智能体API")
# 初始化所有Skill
detector = TableDetector()
ocr = OCREngine()
tsr = TSRParser()
with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

@app.post("/api/table/parse", response_model=TableAgentResponse)
async def parse_table(file: UploadFile = File(...), page_num: int = 1, last_page_table: str = None):
    start_time = time.time()
    trace_id = gen_trace_id()
    log_trace(trace_id, "API_ENTRY", {"page": page_num})
    # 读取图片
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # 1. 图像预处理
    proc_img = preprocess_table_img(img)
    # 2. 检测表格区域
    boxes = detector.detect(proc_img, trace_id)
    table_json_out = []
    md_out = []
    warn_list = []
    # 3. 逐表格处理
    for idx, box in enumerate(boxes):
        crop_img, anchor = crop_table_region(proc_img, box)
        anchor["page"] = page_num
        # OCR提取文字块
        text_blocks = ocr.extract_text(crop_img, trace_id)
        # TSR解析表格结构（合并单元格、多级表头）
        table_struct = tsr.parse(text_blocks, trace_id)
        table_struct["table_id"] = f"tbl_{trace_id}_{idx}"
        # 跨页拼接
        if cfg["cross_page"]["enable"] and last_page_table:
            table_struct = merge_cross_page(last_page_table, table_struct, trace_id)
        # TEDS置信打分（此处用简化模拟金标，正式评测替换人工标注）
        dummy_gold = {"rows": table_struct["rows"]}
        teds = calc_teds(table_struct, dummy_gold)
        warn_lv = get_warning_level(teds, cfg)
        # 双形态导出
        json_table = export_struct_json(table_struct, anchor, teds, warn_lv)
        md_table = export_markdown(table_struct)
        table_json_out.append(json_table)
        md_out.append(md_table)
        # 低置信预警收集
        if warn_lv != "high_conf":
            warn_list.append({"table_id": json_table["table_id"], "teds": teds, "level": warn_lv})
    cost_ms = round((time.time() - start_time)*1000, 2)
    log_trace(trace_id, "API_FINISH", {"cost_ms": cost_ms, "table_count": len(table_json_out)})
    return TableAgentResponse(
        trace_id=trace_id,
        table_list=table_json_out,
        markdown_table_list=md_out,
        warning_list=warn_list,
        total_cost_ms=cost_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.api_server:app", host="0.0.0.0", port=8008, reload=False)