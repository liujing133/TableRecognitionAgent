from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class TableAgentRequest(BaseModel):
    """接口入参规范（供课题2解析智能体调用）"""
    trace_id: str
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    page_num: int = Field(description="当前页码，溯源锚点")
    last_page_table: Optional[Dict] = Field(None, description="上一页缓存表格，用于跨页拼接")

class TableAgentResponse(BaseModel):
    """标准化输出"""
    trace_id: str
    table_list: List[Dict]
    markdown_table_list: List[str]
    warning_list: List[Dict]
    total_cost_ms: float