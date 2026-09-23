"""Algorithm 1 of DSTG-VS: video frame features -> three weighted adjacency matrices.

Standalone: numpy only, no model and no PyG. See decisions.md for how the paper was read
(backward graph is `tril` not `triu`; line-20 normalization divisor).

    from dstg.graph import build_graphs
    fwd, omni, bwd = build_graphs(X, mode="exponential", decay=0.7, fusion=0.5, window=20)
"""
import numpy as np

NORM_DIVISOR = 2.0  # line 20: "Y = Y0 / max(Y0) 2" read as "/ 2"; see decisions.md
MODES = ("exponential", "linear", "logarithmic")


def temporal_decay(mode, decay, window):
    """A[t] for t = 0..window-1, using distance = t + 1 (line 4 of Algorithm 1)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    dt = np.arange(1, window + 1, dtype=np.float64)
    if mode == "exponential":
        return decay ** dt
    if mode == "linear":
        return np.maximum(0.0, 1.0 - decay * dt / window)
    return np.maximum(0.0, 1.0 - decay * np.log(dt) / np.log(window))


def frame_similarity(X):
    """Lines 1-2: cosine similarity between frames, diagonal removed, scaled to max 1."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.maximum(norms, 1e-12)
    S = unit @ unit.T
    np.fill_diagonal(S, 0.0)
    peak = S.max()
    return S / peak if peak > 0 else S


def build_graphs(X, mode="exponential", decay=0.7, fusion=0.5, window=20):
    """Return (forward, omni, backward) weighted adjacency matrices, each T x T.

    :param X: frame features, T x D.
    :param mode: decay function, one of MODES.
    :param decay: decay coefficient ℓ.
    :param fusion: fusion ratio ð, the weight given to frame similarity vs. temporal decay.
    :param window: temporal window w_t; frame i links only to frames within w_t of it.
    """
    X = np.asarray(X, dtype=np.float64)
    T = len(X)
    S = frame_similarity(X)
    A = temporal_decay(mode, decay, window)

    Y0 = np.zeros((T, T), dtype=np.float64)
    for i in range(T):  # lines 13-19
        # paper line 15 reads je = min(i + w_t, T), which with an exclusive slice links only
        # w_t - 1 neighbours and leaves A[w_t - 1] unused; +1 uses the whole decay vector.
        js, je = i + 1, min(i + window + 1, T)
        if js >= je:
            continue
        W = A[:je - js] * (1 - fusion) + S[i, js:je] * fusion
        Y0[i, js:je] = W
        Y0[js:je, i] = W

    peak = Y0.max()
    Y = (Y0 / peak / NORM_DIVISOR) if peak > 0 else Y0  # line 20

    forward = np.triu(Y, 1)      # i -> later frames only
    backward = np.tril(Y, -1)    # i -> earlier frames only (paper prints triu; see decisions.md)
    return forward, Y, backward


def to_edge_index(adj):
    """Weighted adjacency -> (edge_index [2, E] int64, edge_weight [E] float32) for PyG."""
    src, dst = np.nonzero(adj)
    return np.stack([src, dst]).astype(np.int64), adj[src, dst].astype(np.float32)
