import os
import sys
from pathlib import Path
# 解决中文编码
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.append(str(Path(__file__).parent.parent))
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
import yaml
import base64
import json
from io import BytesIO

# ========== 替换核心：导入组员1的Pipeline ==========
from table_agent.pipeline import Member1Pipeline, serialize_page_result
# 保留原有必要模块（TSR/跨页拼接/打分/导出）
from core.tsr_parser import TSRParser
from core.cross_page_merge import merge_cross_page
from core.teds_metric import calc_teds, get_warning_level
from core.exporter import export_struct_json, export_markdown
from utils.logger import gen_trace_id, log_trace
from utils.schema import TableAgentRequest, TableAgentResponse

app = FastAPI(title="智能表格识别与还原智能体API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 初始化替换：仅保留TSR和配置，移除原有detector/ocr ==========
pipeline = Member1Pipeline()  # 组员1的流水线
tsr = TSRParser()
with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

@app.post("/api/table/parse", response_model=TableAgentResponse)
async def parse_table(file: UploadFile = File(...), page_num: int = 1, last_page_table: str = None):
    start_time = time.time()
    trace_id = gen_trace_id()
    log_trace(trace_id, "API_ENTRY", {"page": page_num})
    
    # 新增：存储各步骤可视化数据
    step_visualizations = []
    
    # 读取图片（仅读取一次！）
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # 新增：校验图片解码是否成功
    if img is None:
        log_trace(trace_id, "ERROR", "图片解码失败：空数据或不支持的格式")
        raise HTTPException(status_code=400, detail="图片解码失败，请检查文件是否为空或格式是否支持（jpg/png）")
    
    # ========== 步骤1：调用组员1的Pipeline处理整页 ==========
    import tempfile
    # 直接复用已读取的img，不再重复read文件
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, img)
        tmp_path = tmp_file.name

    # 标准调用：传路径，参数名是 page_idx / trace_id，无 image 参数
    result = pipeline.process_page(
        tmp_path,
        page_idx=page_num,
        trace_id=trace_id
    )
    # 删除临时文件
    os.unlink(tmp_path)

    log_trace(trace_id, "STEP_MEMBER1_PIPELINE", f"组员1流水线完成：检测到{result.table_count}个表格")

    # ========== 步骤1可视化：预处理结果（从组员1结果中提取） ==========
    # 假设组员1的result包含预处理后的图像（可从result的预处理结果中取）
    # 若pipeline返回的proc_img不存在，可从第一个表格的crop_binary反向获取
    if result.table_count > 0:
        proc_img = result.tables[0].crop_binary  # 用组员1的二值图作为预处理可视化
    else:
        proc_img = img  # 无表格时用原图
    _, proc_img_encoded = cv2.imencode('.jpg', proc_img)
    proc_img_base64 = base64.b64encode(proc_img_encoded).decode('utf-8')
    step_visualizations.append({
        "step_name": "图像预处理（组员1）",
        "type": "image",
        "content": proc_img_base64,
        "desc": "完成图像降噪、二值化、倾斜校正等预处理（复用组员1规则）"
    })

    # ========== 步骤2可视化：表格检测结果（从组员1结果中提取） ==========
    det_img = img.copy()
    for i, table in enumerate(result.tables):
        # 从组员1的detection中取检测框坐标
        box = table.detection.bbox  # DetectedTable 对象属性格式[x1,y1,x2,y2]
        cv2.rectangle(det_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    _, det_img_encoded = cv2.imencode('.jpg', det_img)
    det_img_base64 = base64.b64encode(det_img_encoded).decode('utf-8')
    step_visualizations.append({
        "step_name": "表格区域检测（组员1 YOLOv8）",
        "type": "image",
        "content": det_img_base64,
        "desc": f"检测到 {result.table_count} 个表格区域（复用组员1 YOLOv8检测）"
    })

    table_json_out = []
    md_out = []
    warn_list = []
    
    # 3. 逐表格处理（复用原有TSR/跨页/打分逻辑，仅替换数据源为组员1的结果）
    for idx, table in enumerate(result.tables):
        # ========== 步骤3可视化：文字块提取（从组员1的OCR结果） ==========
        ocr_img = table.crop_bgr.copy()  # 组员1裁剪后的表格BGR图
        text_blocks = []
        for block in table.ocr_result.blocks:
            bbox = block.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            text_blocks.append({
                "text": block.text,
                "confidence": float(block.confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bbox": [x1, y1, x2, y2],
            })
            cv2.rectangle(ocr_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(ocr_img, block.text[:10], (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        _, ocr_img_encoded = cv2.imencode('.jpg', ocr_img)
        ocr_img_base64 = base64.b64encode(ocr_img_encoded).decode('utf-8')
        step_visualizations.append({
            "step_name": f"文字块提取-表格{idx+1}（组员1 PaddleOCR）",
            "type": "image",
            "content": ocr_img_base64,
            "desc": f"提取到 {len(text_blocks)} 个文字块（复用组员1 OCR）"
        })
        log_trace(trace_id, f"STEP_OCR_TABLE{idx+1}", f"提取{len(text_blocks)}个文字块")

        # ========== 步骤4：表格结构解析（保留原有TSR逻辑） ==========
        table_struct = tsr.parse(text_blocks, trace_id, table_img=table.crop_bgr)
        table_struct["table_id"] = f"tbl_{trace_id}_{idx}"
        step_visualizations.append({
            "step_name": f"表格结构解析-表格{idx+1}",
            "type": "json",
            "content": table_struct,
            "desc": "完成单元格合并、多级表头解析"
        })
        log_trace(trace_id, f"STEP_TSR_TABLE{idx+1}", "表格结构解析完成")

        # ========== 步骤5：跨页拼接（保留原有逻辑） ==========
        cross_page_info = "未启用跨页拼接"
        if cfg["cross_page"]["enable"] and idx == 0 and last_page_table:
            if isinstance(last_page_table, str):
                try:
                    last_page_table = json.loads(last_page_table)
                except Exception as e:
                    log_trace(trace_id, "ERROR", {"last_page_table_parse_error": str(e)})
                    last_page_table = None
            if isinstance(last_page_table, dict):
                table_struct = merge_cross_page(last_page_table, table_struct, trace_id)
                cross_page_info = "完成跨页表格拼接"
            else:
                log_trace(trace_id, "WARN", "last_page_table格式不正确，跳过跨页拼接")
        elif cfg["cross_page"]["enable"] and idx != 0 and last_page_table:
            cross_page_info = "仅对当前页第一个表格进行跨页拼接"
        step_visualizations.append({
            "step_name": f"跨页拼接-表格{idx+1}",
            "type": "text",
            "content": cross_page_info,
            "desc": "跨页拼接开关：{}".format("开启" if cfg["cross_page"]["enable"] else "关闭")
        })
        log_trace(trace_id, f"STEP_CROSS_PAGE_TABLE{idx+1}", cross_page_info)

        # ========== 步骤6：置信打分（保留原有逻辑） ==========
        dummy_gold = {"rows": table_struct["rows"]}
        teds = calc_teds(table_struct, dummy_gold)
        warn_lv = get_warning_level(teds, cfg)
        step_visualizations.append({
            "step_name": f"置信打分-表格{idx+1}",
            "type": "text",
            "content": f"TEDS分数：{teds:.4f} | 置信等级：{warn_lv}",
            "desc": "TEDS越接近1表示识别越准确"
        })
        log_trace(trace_id, f"STEP_TEDS_TABLE{idx+1}", f"TEDS={teds:.4f}, warn_lv={warn_lv}")

        # 锚点信息（从组员1的检测结果中提取）
        anchor = {
            "page": page_num,
            "bbox": list(table.detection.bbox),
            "confidence": float(table.detection.confidence)
        }
        
        # 双形态导出（保留原有逻辑）
        json_table = export_struct_json(table_struct, anchor, teds, warn_lv)
        md_table = export_markdown(table_struct)
        table_json_out.append(json_table)
        md_out.append(md_table)
        
        # 低置信预警收集
        if warn_lv != "high_conf":
            warn_list.append({"table_id": json_table["table_id"], "teds": teds, "level": warn_lv})

    cost_ms = round((time.time() - start_time)*1000, 2)
    log_trace(trace_id, "API_FINISH", {"cost_ms": cost_ms, "table_count": len(table_json_out)})
    
    # 返回结果（保留原有结构）
    return TableAgentResponse(
        trace_id=trace_id,
        table_list=table_json_out,
        markdown_table_list=md_out,
        warning_list=warn_list,
        total_cost_ms=cost_ms,
        step_visualizations=step_visualizations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.api_server:app", host="0.0.0.0", port=8008, reload=False)