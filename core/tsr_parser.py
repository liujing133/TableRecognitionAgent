# core/tsr_parser.py
import yaml
from utils.logger import log_trace

with open("config/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

class TSRParser:
    def __init__(self):
        # 加载轻量化TableFormer TSR模型，此处简化封装
        self.model_path = cfg["model"]["tsr_model_path"]

    def parse(self, text_blocks: list, trace_id: str):
        """
        解析文字块，输出带rowspan/colspan、多级表头的表格结构
        :param text_blocks: OCR输出文字坐标文本列表
        :param trace_id: 全链路追踪ID
        :return: 结构化表格dict
        """
        # 模拟TSR模型推理，实际项目替换TableFormer推理逻辑
        rows = [
            {
                "cells": [
                    {"text": "序号", "rowspan": 1, "colspan": 1},
                    {"text": "名称", "rowspan": 1, "colspan": 1},
                    {"text": "金额", "rowspan": 1, "colspan": 1}
                ]
            },
            {
                "cells": [
                    {"text": "1", "rowspan": 1, "colspan": 1},
                    {"text": "测试项目", "rowspan": 1, "colspan": 1},
                    {"text": "1000", "rowspan": 1, "colspan": 1}
                ]
            }
        ]
        table_struct = {
            "rows": rows,
            "has_multi_header": False,
            "has_merge_cell": False
        }
        log_trace(trace_id, "TSRParser", {"cell_count": len(text_blocks), "row_num": len(rows)})
        return table_struct