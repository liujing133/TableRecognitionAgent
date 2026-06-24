# 组员1模块 - 图像底层 + 检测 OCR

## 目录结构

```
table_agent/
├── config/
│   ├── __init__.py
│   └── config.yaml              # 所有参数配置（阈值、路径等）
├── preprocessor/
│   ├── __init__.py
│   └── image_preprocessor.py    # 图像预处理 [规则] OpenCV
├── detector/
│   ├── __init__.py
│   └── table_detector.py        # YOLOv8表格检测 [模型]
├── ocr/
│   ├── __init__.py
│   └── ocr_engine.py            # PaddleOCR文字提取 [模型]
├── utils/
│   ├── __init__.py
│   ├── common.py                # 工具函数（配置、trace_id、计时、校验、置信度分级）
│   └── logger.py                # 日志模块（兼容loguru/标准logging）
├── tests/
│   ├── __init__.py
│   ├── test_member1.py          # 35+ 测试用例
│   ├── generate_test_samples.py # 合成测试图像生成
│   └── samples/                 # 7张合成测试图像
├── pipeline.py                  # 三模块统一接口（组长从这里调用）
├── run.py                       # 演示脚本（支持 --image / --all / --test 等）
├── requirements.txt             # 依赖
├── test.jpg                     # 原始测试图像
├── yolov8n.pt                   # YOLOv8通用预训练权重（兜底用）
└── README_member1.md            # 本说明文档
```

## 各模块职责

| 模块 | 能力 | 实现类型 | 关键算法 |
|------|------|----------|---------|
| 预处理 | 图像加载、中文路径支持 | 规则 | cv2.imdecode + np.fromfile |
| 预处理 | 超大图缩放 | 规则 | 等比例缩放 + INTER_AREA |
| 预处理 | 透视矫正 | 规则 | 霍夫直线检测 + 仿射变换 |
| 预处理 | 快速去噪 | 规则 | fastNlMeansDenoisingColored |
| 预处理 | 自适应二值化 | 规则 | 高斯自适应阈值 |
| 预处理 | 拉普拉斯锐化 | 规则 | 卷积核滤波 |
| 预处理 | 可视化对比图 | 规则 | 四图拼接 + 标签 |
| 检测 | 表格定位 | 模型 | YOLOv8 |
| 检测 | 置信度分级 | 规则 | 三段式阈值 |
| OCR | 文字提取 | 模型 | PaddleOCR |
| OCR | 全半角转换 | 规则 | Unicode映射 |
| OCR | 文字归属单元格 | 规则 | 中心点落点（最小面积匹配） |
| 流水线 | 流程编排 | 规则 | Pipeline模式 |
| 流水线 | 批量处理 | 规则 | 循环编排 |
| 流水线 | 序列化 | 规则 | dict转换 |

## 安装环境

```bash
pip install -r requirements.txt
# 注意：PaddleOCR需要先装paddlepaddle
```

## 快速演示

```bash
# 1. 生成合成测试图像
python tests/generate_test_samples.py

# 2. 运行规则类测试（不需要模型）
python run.py --test

# 3. 处理单张测试图像
python run.py

# 4. 批量处理所有测试图像
python run.py --all
```

## 组长调用方式

```python
from pipeline import Member1Pipeline, serialize_page_result

pipeline = Member1Pipeline()

# 处理单页
result = pipeline.process_page("path/to/image.jpg", page_idx=0, trace_id="xxx")
# result.table_count      → 本页表格数
# result.tables[i].crop_bgr     → 裁剪出的表格BGR图（给组员2的TSR用）
# result.tables[i].crop_binary  → 二值图（给组员2的无边框聚类用）
# result.tables[i].ocr_result   → OCR文字块（含坐标和置信度）
# result.tables[i].detection    → 检测框坐标和置信度

# 批量处理
results = pipeline.process_batch(["page1.jpg", "page2.jpg"])

# 序列化为JSON（FastAPI response用）
data = serialize_page_result(result)

# 健康检查
health = pipeline.health_check()
```

## 运行全部测试

```bash
cd table_agent
python -m pytest tests/ -v

# 或者直接：
python run.py --test
```

## 关于models文件夹

组长提供的模型文件（`bert-base-chinese/` 和 `tableformer_light/`）在：
`D:\电脑管家迁移文件\xwechat_files\...\models\`

这些是组员2的TSR模块和组长接口服务需要用的，但组员1模块当前不直接依赖它们。如果后续需要将 `tableformer_light` 引入表格检测流程，需要：
1. 训练 YOLOv8 专用权重替换 `yolov8n.pt`
2. 或通过组长协调接口接入
