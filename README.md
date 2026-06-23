# TableRecognitionAgent
课题 8 智能表格识别与还原智能体完整代码设计  

整体规范：Python + FastAPI + OpenCV + YOLOv8 + PaddleOCR + TableFormer，满足可私有化、API 服务化、可追溯、双形态输出、TEDS 置信评估、跨页表格拼接全部课题要求；代码分层解耦，严格区分「规则模块 / 小模型模块 / 业务逻辑模块 / 接口层」，自带日志溯源、批量评测、badcase 导出。  

项目目录结构（工程化标准，交付物规范）  

table_agent/  
├── config/                # 配置文件  
│   └── settings.yaml      # 置信阈值、模型路径、跨页开关  
├── core/                  # 核心能力Skill分层  
│   ├── preprocessor.py     # 图像预处理（规则算法）  
│   ├── table_detector.py  # YOLOv8表格区域检测  
│   ├── ocr_engine.py      # PaddleOCR文字提取  
│   ├── tsr_parser.py      # TSR表格结构解析(合并单元格/多级表头)  
│   ├── borderless_infer.py# 无边框表格布局推理  
│   ├── cross_page_merge.py# 跨页续表拼接逻辑  
│   ├── teds_metric.py     # TEDS结构相似度打分（置信度）  
│   └── exporter.py        # JSON/Markdown双形态导出  
├── service/  
│   └── api_server.py      # FastAPI 标准HTTP接口服务  
├── utils/  
│   ├── logger.py          # 全链路trace日志，可审计溯源  
│   ├── schema.py          # 统一入参出参数据规范  
│   └── tools.py           # 通用工具函数  
├── eval/  
│   ├── run_eval.py        # 自动化评测脚本  
│   └── badcase_save.py    # 失败案例聚类存储  
├── models/                # 本地离线模型权重（私有化，不上传外网）  
├── test_demo/             # 测试样例、调用demo  
├── Dockerfile             # 容器化部署（加分交付物）  
├── requirements.txt       # 依赖清单  
└── README.md              # 接口文档、部署说明  

运行使用流程  

安装依赖：pip install -r requirements.txt  
配置模型权重放入models/文件夹  
启动服务：python service/api_server.py  
课题 2 解析智能体调用http://127.0.0.1:8008/api/table/parse上传表格图像  
FastAPI 自带可视化调试页面http://127.0.0.1:8008/docs  
批量评测：python eval/run_eval.py自动输出验收指标报告  