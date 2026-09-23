"""Step 1-3 from the meeting: prove torch, torch_geometric and h5py import and work.

Run from anywhere:  python check_env.py
"""
import torch

print("=== 1) PyTorch ===")
print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
x = torch.randn(3, 3, device="cuda" if torch.cuda.is_available() else "cpu")
print("tiny matmul OK:", (x @ x.T).shape)

print("\n=== 2) PyTorch Geometric (graph neural nets) ===")
import torch_geometric
from torch_geometric.nn import GCNConv

print("torch_geometric version:", torch_geometric.__version__)
# Toy graph: 4 nodes (think: 4 video frames), edges connect neighbouring frames.
feats = torch.randn(4, 8)
edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
out = GCNConv(8, 8)(feats, edges)
print("GCNConv on a 4-node graph OK, output shape:", tuple(out.shape))

print("\n=== 3) h5py ===")
import h5py
from pathlib import Path

print("h5py version:", h5py.__version__)
data_dir = Path(__file__).parent / "DSNet" / "datasets"
for path in sorted(data_dir.glob("*.h5")):
    with h5py.File(path, "r") as f:
        print(f"opened {path.name:45s} -> {len(f.keys())} videos")
