from loguru import logger
import yaml
import uuid

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

logger.add(cfg["log"]["log_path"], rotation="500MB", encoding="utf-8")

def gen_trace_id() -> str:
    """生成全局唯一trace_id，贯穿全链路审计"""
    return str(uuid.uuid4())

def log_trace(trace_id: str, stage: str, content: dict):
    """记录四级执行轨迹：任务->智能体->Skill->模型"""
    logger.info(f"[TRACE_ID:{trace_id}] STAGE:{stage} DATA:{content}")