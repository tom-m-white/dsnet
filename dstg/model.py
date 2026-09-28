"""Simplified topology ablation; explicit interpretations are in decisions.md.

Input: T x 1024 float features. Output: T importance scores in [0, 1].
Graph preparation can be cached for training because input features are fixed.
Default forward includes graph construction, for end-to-end inference timing.
"""
import torch
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv

from .graph import build_graphs, to_edge_index

GRAPH_NAMES = ("fwd", "bwd", "omni")


class DSTG(nn.Module):
    def __init__(self, graphs=("fwd", "bwd", "omni"), input_dim=1024,
                 hidden_dim=192, heads=1, dropout=0.0, graph_config=None):
        super().__init__()
        if len(set(graphs)) != len(graphs) or set(graphs) - set(GRAPH_NAMES):
            raise ValueError("graphs must be a unique subset of fwd, bwd, omni (empty enables control)")
        self.graphs = tuple(g for g in GRAPH_NAMES if g in graphs)
        self.input_dim = input_dim
        self.graph_config = dict(graph_config or {})
        # VideoSAGE Fig. 2 / released SPELL: only the middle layer is shared.
        self.gat = nn.ModuleDict({g: GATConv(
            input_dim, hidden_dim, heads=heads, concat=False,
            edge_dim=1, dropout=dropout, fill_value=1.0) for g in self.graphs})
        self.shared_sage = SAGEConv(hidden_dim, hidden_dim) if self.graphs else None
        self.independent_sage = nn.ModuleDict({
            g: SAGEConv(hidden_dim, hidden_dim) for g in self.graphs
        })
        self.head = nn.Linear(hidden_dim, 1)
        # Graph-free diagnostic: per-frame 1024 -> 192 -> 1 MLP. No mixing,
        # graph preparation or inactive graph parameters. Existing V1-V5 unchanged.
        self.input_projection = nn.Linear(input_dim, hidden_dim) if not self.graphs else None

    def prepare_graphs(self, x):
        """Algorithm 1 on fixed raw features; PyG rows are source, target."""
        if not self.graphs:
            return {}
        arrays = build_graphs(x.detach().cpu().numpy(), **self.graph_config)
        adjacency = dict(zip(("fwd", "omni", "bwd"), arrays))
        result = {}
        for name in self.graphs:
            indices, weights = to_edge_index(adjacency[name])
            result[name] = (torch.from_numpy(indices).to(x.device),
                            torch.from_numpy(weights).to(device=x.device, dtype=x.dtype))
        return result

    def forward(self, x, edges=None):
        if x.ndim != 2 or x.shape[1] != self.input_dim or x.shape[0] == 0:
            raise ValueError(f"expected nonempty T x {self.input_dim} input")
        if not self.graphs:
            return self.head(self.input_projection(x).relu()).squeeze(-1).sigmoid()
        if edges is None:
            edges = self.prepare_graphs(x)
        total = None
        for name in self.graphs:
            edge_index, edge_weight = edges[name]
            h = self.gat[name](x, edge_index, edge_attr=edge_weight[:, None]).relu()
            h = self.shared_sage(h, edge_index).relu()
            h = self.independent_sage[name](h, edge_index)
            total = h if total is None else total + h
        return self.head(total).squeeze(-1).sigmoid()
