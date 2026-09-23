"""Test Algorithm 1 in isolation on a 10-frame toy example (run: python test_graph.py).

Checks: omni symmetry, zero outside the window, forward = upper-triangular, backward =
lower-triangular, forward + backward = omni, and decay monotonicity.
"""
import numpy as np

from dstg.graph import MODES, build_graphs, temporal_decay, to_edge_index

np.set_printoptions(precision=3, suppress=True, linewidth=150)

T, D, W = 10, 8, 3
rng = np.random.default_rng(0)
# Toy video: 3 "scenes" of similar frames, so the similarity term has visible structure.
X = np.repeat(rng.random((3, D)), 4, axis=0)[:T] + 0.05 * rng.random((T, D))

fwd, omni, bwd = build_graphs(X, mode="exponential", decay=0.7, fusion=0.5, window=W)

print(f"toy input: T={T} frames, D={D} features, window w_t={W}, exponential decay 0.7, fusion 0.5")
print("\ndecay A[t] for each mode (distance = 1..w_t):")
for m in MODES:
    print(f"  {m:12s}", temporal_decay(m, 0.7, W))

print("\nomni Y (undirected):\n", omni)
print("\nforward Y_f:\n", fwd)
print("\nbackward Y_b:\n", bwd)

checks = {
    "omni is symmetric": np.allclose(omni, omni.T),
    "forward is upper-triangular": np.allclose(fwd, np.triu(fwd, 1)),
    "backward is lower-triangular": np.allclose(bwd, np.tril(bwd, -1)),
    "forward + backward == omni": np.allclose(fwd + bwd, omni),
    "backward == forward transposed": np.allclose(bwd, fwd.T),
    "no self-loops (zero diagonal)": np.allclose(np.diag(omni), 0),
    "zero outside the window": all(
        omni[i, j] == 0 for i in range(T) for j in range(T) if abs(i - j) > W),
    "every in-window pair is non-zero": all(
        omni[i, j] != 0 for i in range(T) for j in range(T) if 0 < abs(i - j) <= W),
    "weights in [0, 1] after normalization": omni.min() >= 0 and omni.max() <= 1 + 1e-9,
    "strongest edge normalized to 1": abs(omni.max() - 1) < 1e-9,
    # squaring is monotonic, so ordering is kept while weak edges are pushed down harder:
    "squaring keeps edge ordering": np.array_equal(
        np.argsort(omni[omni > 0]), np.argsort(np.sqrt(omni[omni > 0]))),
    "weakest/strongest ratio is squared": abs(
        omni[omni > 0].min() - np.sqrt(omni[omni > 0]).min() ** 2) < 1e-9,
    "decay decreases with distance": all(
        np.all(np.diff(temporal_decay(m, 0.7, 8)) <= 1e-12) for m in MODES),
}
print()
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

ei, ew = to_edge_index(fwd)
print(f"\nforward as PyG edges: edge_index {ei.shape}, edge_weight {ew.shape}, "
      f"first 5 edges {list(zip(ei[0, :5], ei[1, :5]))}")
assert all(checks.values()), "a structural check failed"
print("\nall checks passed")
