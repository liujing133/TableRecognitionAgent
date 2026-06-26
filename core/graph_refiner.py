# core/graph_refiner.py
import importlib.util
import os
import numpy as np
from sklearn.neighbors import kneighbors_graph

_torch_available = importlib.util.find_spec("torch") is not None

# ---------- 单例 GNN 模型（只加载一次） ----------
_GNN_MODEL_INSTANCE = None
_GNN_WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gnn_weights.pt")


def _get_gnn_model(node_feat_dim=6, hidden_dim=32):
    global _GNN_MODEL_INSTANCE
    if _GNN_MODEL_INSTANCE is not None:
        return _GNN_MODEL_INSTANCE, True

    if not _torch_available:
        return None, False

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GraphSAGE_Layer(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim)

        def forward(self, x, adj):
            neighbor_sum = torch.mm(adj, x)
            concat = torch.cat([x, neighbor_sum], dim=1)
            return F.relu(self.linear(concat))

    class LightweightGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = GraphSAGE_Layer(node_feat_dim, hidden_dim)
            self.layer2 = GraphSAGE_Layer(hidden_dim, 2)

        def forward(self, x, adj):
            h = self.layer1(x, adj)
            return self.layer2(h, adj)

    model = LightweightGNN()
    weights_loaded = False

    # 尝试加载预训练权重
    weight_path = _GNN_WEIGHT_PATH
    if os.path.isfile(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            weights_loaded = True
        except Exception:
            pass

    model.eval()
    _GNN_MODEL_INSTANCE = model
    return model, weights_loaded


def _bbox_to_center(bbox):
    if not bbox:
        return 0.0, 0.0
    pts = np.array(bbox, dtype=float)
    if pts.ndim == 1 and pts.size == 4:
        x1, y1, x2, y2 = pts.tolist()
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if pts.ndim == 2 and pts.shape[1] == 2:
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        return float(cx), float(cy)
    return 0.0, 0.0


def build_adj_from_grid(grid_cells):
    nodes = []
    for row in grid_cells:
        for cell in row:
            if cell.get("bbox") is not None:
                cx, cy = _bbox_to_center(cell["bbox"])
                nodes.append([cx, cy])
            else:
                nodes.append([0, 0])

    if len(nodes) < 2:
        if _torch_available:
            import torch
            return torch.eye(len(nodes)), nodes
        return np.eye(len(nodes), dtype=np.float32), nodes

    nodes_np = np.array(nodes)
    adj = kneighbors_graph(nodes_np, n_neighbors=min(3, len(nodes)), mode='connectivity', include_self=True)
    adj = adj.toarray().astype(np.float32)
    if _torch_available:
        import torch
        adj_torch = torch.tensor(adj)
    else:
        adj_torch = adj
    return adj_torch, nodes_np


def _heuristic_merge_empty_cells(grid_cells):
    """确定性启发式降级：空单元格向左/向上合并"""
    refined_rows = []
    for row in grid_cells:
        refined_row = []
        for c_idx, cell in enumerate(row):
            cell_copy = cell.copy()
            text = cell_copy.get("text", "")
            if text == "" and len(row) > 1:
                # 优先向左合并
                if c_idx > 0 and row[c_idx - 1].get("text", "") != "":
                    cell_copy["colspan"] = max(int(row[c_idx - 1].get("colspan", 1)), 2)
                    cell_copy["text"] = row[c_idx - 1]["text"] + "(续)"
            refined_row.append(cell_copy)
        refined_rows.append(refined_row)
    return refined_rows


def _gnn_inference(grid_cells, model, adj, nodes):
    """运行 GNN 推理（仅在有预训练权重时使用）"""
    import torch
    node_feats = []
    for row in grid_cells:
        for cell in row:
            bbox = cell.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = np.array(bbox, dtype=float).tolist() if isinstance(np.array(bbox), np.ndarray) else bbox
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                if isinstance(bbox[0], (list, tuple)):
                    xs = [float(p[0]) for p in bbox]
                    ys = [float(p[1]) for p in bbox]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            text = cell.get("text", "")
            node_feats.append([
                x1 / 1000.0, y1 / 1000.0,
                1.0 if text else 0.0,
                float(min(len(text), 20)),
                float(cell.get("score", 0.0)),
                width / height,
            ])
    node_feats = torch.tensor(node_feats, dtype=torch.float32)
    with torch.no_grad():
        logits = model(node_feats, adj)

    refined_rows = []
    idx = 0
    for row in grid_cells:
        refined_row = []
        for c_idx, cell in enumerate(row):
            cell_copy = cell.copy()
            prob_merge = float(torch.sigmoid(logits[idx, 0]).item())
            if cell_copy.get("text", "") == "" and len(row) > 1 and prob_merge > 0.65:
                if c_idx > 0 and row[c_idx - 1].get("text", "") != "":
                    cell_copy["colspan"] = max(int(row[c_idx - 1].get("colspan", 1)), 2)
                    cell_copy["text"] = row[c_idx - 1]["text"] + "(续)"
            refined_row.append(cell_copy)
            idx += 1
        refined_rows.append(refined_row)
    return refined_rows


def refine_with_gnn(grid_cells, trace_id="gnn"):
    """
    对外接口：优化行列关系和合并单元格

    策略：
    1. 如果 torch 可用且预训练权重存在 → 用 GNN 推理
    2. 否则 → 用确定性启发式降级（不随机乱猜）
    """
    model, weights_loaded = _get_gnn_model()

    if weights_loaded:
        try:
            adj, nodes = build_adj_from_grid(grid_cells)
            return _gnn_inference(grid_cells, model, adj, nodes)
        except Exception:
            pass

    # 确定降级：不随机，直接用启发式
    return _heuristic_merge_empty_cells(grid_cells)