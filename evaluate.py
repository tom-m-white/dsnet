"""Shared evaluation ruler for video summarization: F-score, Kendall's tau, Spearman's rho.

No model dependency. Input is a predicted importance score per SAMPLED frame (one per row of
`features` in the h5, i.e. aligned with `picks`). Output is F-score (%), tau, rho.

Protocol (see decisions.md for why each choice was made):
  F-score   DSNet / DR-DSN keyshot protocol: scores -> frame scores via `picks` -> shot scores
            (mean over KTS shot) -> 0/1 knapsack with a 15% length budget -> binary summary ->
            F1 vs each annotator's `user_summary`; TVSum = mean over annotators, SumMe = max.
  tau/rho   Per annotator, then averaged over annotators (Otani et al., CVPR 2019), with ties
            handled as in the CA-SUM / CSTA code: kendalltau(rankdata(p), rankdata(t)) = tau-b,
            spearmanr with average ranks.
            TVSum: each annotator's raw 1-5 ratings from ydata-tvsum50-anno.tsv, normalized by
                   max and subsampled every 15 frames (CA-SUM / CSTA convention).
            SumMe: each annotator's binary `user_summary` at `picks` (SumMe annotators only ever
                   gave keep/skip selections). `summe_protocol="csta"` instead reproduces CSTA:
                   binary summary vs the MEAN of the annotators, at the original frame rate.
  Averaging: per video, then mean over the test videos of the split.

Usage as a library:
    from evaluate import Evaluator
    ev = Evaluator("summe")                     # or "tvsum"
    ev.evaluate_video("video_1", scores)        # -> {"fscore": .., "tau": .., "rho": ..}
    ev.evaluate_split({"video_1": s1, ...})     # -> mean over videos

Usage from the command line (scores saved as .npz: one array per video name):
    python evaluate.py --dataset summe --scores my_scores.npz --split-file DSNet/splits/summe.yml --split 0
    python evaluate.py --dataset tvsum --self-check      # human + random baselines as sanity check
"""
import argparse
import csv
import warnings
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.stats import kendalltau, rankdata, spearmanr

ROOT = Path(__file__).parent
H5_PATHS = {
    "summe": ROOT / "DSNet" / "datasets" / "eccv16_dataset_summe_google_pool5.h5",
    "tvsum": ROOT / "DSNet" / "datasets" / "eccv16_dataset_tvsum_google_pool5.h5",
}
TVSUM_ANNO = ROOT / "tvsum_original" / "ydata-tvsum50-v1_1" / "data" / "ydata-tvsum50-anno.tsv"
SUMMARY_PROPORTION = 0.15


# ----------------------------------------------------------------------------- building blocks

def knapsack(values, weights, capacity):
    """0/1 knapsack by dynamic programming. Returns the indices of the packed items."""
    values = np.asarray(values, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    n = len(values)
    best = np.zeros(capacity + 1, dtype=np.int64)
    keep = np.zeros((n, capacity + 1), dtype=bool)
    for i in range(n):
        w, v = weights[i], values[i]
        if w > capacity:
            continue
        candidate = best[:capacity + 1 - w] + v
        improved = candidate > best[w:]
        keep[i, w:] = improved
        best[w:] = np.where(improved, candidate, best[w:])
    packed, c = [], capacity
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            packed.append(i)
            c -= weights[i]
    return sorted(packed)


def scores_to_summary(scores, change_points, n_frames, nfps, picks, proportion=SUMMARY_PROPORTION):
    """Sampled-frame scores -> binary keyshot summary over the original frames (DSNet protocol)."""
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i, lo in enumerate(picks):
        hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[lo:hi] = scores[i]
    # shot score = mean frame score, scaled to int as in DSNet / OR-Tools
    shot_scores = [int(1000 * frame_scores[a:b + 1].mean()) for a, b in change_points]
    packed = knapsack(shot_scores, nfps, int(n_frames * proportion))
    summary = np.zeros(n_frames, dtype=bool)
    for s in packed:
        a, b = change_points[s]
        summary[a:b + 1] = True
    return summary


def f1(pred, gt):
    overlap = (pred & gt).sum()
    if overlap == 0:
        return 0.0
    precision, recall = overlap / pred.sum(), overlap / gt.sum()
    return float(2 * precision * recall / (precision + recall))


def rank_correlations(pred, annotators):
    """Mean over annotators of (tau, rho), CA-SUM/CSTA style tie handling."""
    taus, rhos = [], []
    for true in annotators:
        taus.append(kendalltau(rankdata(pred), rankdata(true))[0])
        rhos.append(spearmanr(pred, true)[0])
    return float(np.mean(taus)), float(np.mean(rhos))


def load_tvsum_annotations(path=TVSUM_ANNO):
    """{video_index (1-50) -> list of 20 per-annotator score arrays, normalized, every 15th frame}.

    Row order of the .tsv matches video_1..video_50 of the h5 (same convention as CA-SUM/CSTA).
    """
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    video_ids = list(dict.fromkeys(r[0] for r in rows))  # youtube ids in file order
    per_video = {vid: [] for vid in video_ids}
    for r in rows:
        s = np.array([float(x) for x in r[2].split(",")])
        per_video[r[0]].append((s / s.max())[::15])
    return {i + 1: per_video[vid] for i, vid in enumerate(video_ids)}


# ----------------------------------------------------------------------------- evaluator

class Evaluator:
    def __init__(self, dataset, h5_path=None, tvsum_anno_path=TVSUM_ANNO, summe_protocol="per_annotator"):
        assert dataset in ("summe", "tvsum"), dataset
        assert summe_protocol in ("per_annotator", "csta"), summe_protocol
        self.dataset = dataset
        self.summe_protocol = summe_protocol
        self.h5 = h5py.File(h5_path or H5_PATHS[dataset], "r")
        self.tvsum_anno = load_tvsum_annotations(tvsum_anno_path) if dataset == "tvsum" else None

    def video(self, name):
        v = self.h5[name]
        return dict(
            change_points=v["change_points"][...], n_frames=int(v["n_frames"][()]),
            nfps=v["n_frame_per_seg"][...], picks=v["picks"][...],
            user_summary=v["user_summary"][...].astype(bool),
        )

    def annotator_scores(self, name, v):
        """Per-annotator ground truth at the sampled-frame rate (one array per annotator)."""
        if self.dataset == "tvsum":
            anno = self.tvsum_anno[int(name.split("_")[-1])]
            n = len(v["picks"])
            assert all(len(a) == n for a in anno), f"{name}: tsv length != number of sampled frames"
            return anno
        return v["user_summary"][:, v["picks"]].astype(np.float32)

    def evaluate_video(self, name, scores):
        v = self.video(name)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        assert len(scores) == len(v["picks"]), (
            f"{name}: expected {len(v['picks'])} scores (one per sampled frame), got {len(scores)}")

        summary = scores_to_summary(scores, v["change_points"], v["n_frames"], v["nfps"], v["picks"])
        f1s = [f1(summary, u) for u in v["user_summary"]]
        fscore = max(f1s) if self.dataset == "summe" else float(np.mean(f1s))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant input -> NaN correlation; reported, not hidden
            if self.dataset == "summe" and self.summe_protocol == "csta":
                gt = v["user_summary"].mean(0)
                tau = float(kendalltau(rankdata(summary), rankdata(gt))[0])
                rho = float(spearmanr(summary, gt)[0])
            else:
                tau, rho = rank_correlations(scores, self.annotator_scores(name, v))
        return {"fscore": 100 * fscore, "tau": tau, "rho": rho}

    def evaluate_split(self, scores_by_video):
        per_video = {k: self.evaluate_video(k, s) for k, s in scores_by_video.items()}
        mean = {m: float(np.mean([r[m] for r in per_video.values()])) for m in ("fscore", "tau", "rho")}
        return mean, per_video

    # ------------------------------------------------------------------ sanity checks

    def human_baseline(self, names):
        """Leave-one-out: each annotator's own scores treated as the prediction, vs the others."""
        taus, rhos = [], []
        for name in names:
            v = self.video(name)
            anno = self.annotator_scores(name, v)
            t, r = zip(*[rank_correlations(a, [b for j, b in enumerate(anno) if j != i])
                         for i, a in enumerate(anno)])
            taus.append(np.mean(t))
            rhos.append(np.mean(r))
        return float(np.mean(taus)), float(np.mean(rhos))

    def random_baseline(self, names, seed=0):
        rng = np.random.default_rng(seed)
        return self.evaluate_split({n: rng.random(len(self.video(n)["picks"])) for n in names})[0]


def test_keys(split_file, split_idx):
    split = yaml.safe_load(open(split_file))[split_idx]
    return [Path(k).name for k in split["test_keys"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("summe", "tvsum"))
    p.add_argument("--scores", help=".npz file: one score array per video name (e.g. video_1)")
    p.add_argument("--split-file", help="DSNet split yaml; restricts to that split's test videos")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--summe-protocol", default="per_annotator", choices=("per_annotator", "csta"))
    p.add_argument("--self-check", action="store_true", help="print human + random baselines")
    args = p.parse_args()

    ev = Evaluator(args.dataset, summe_protocol=args.summe_protocol)
    names = test_keys(args.split_file, args.split) if args.split_file else sorted(ev.h5.keys())

    if args.self_check:
        tau, rho = ev.human_baseline(names)
        print(f"human (leave-one-out) over {len(names)} videos: tau={tau:.3f} rho={rho:.3f}")
        r = ev.random_baseline(names)
        print(f"random scores:            F={r['fscore']:.2f}%  tau={r['tau']:.3f} rho={r['rho']:.3f}")

    if args.scores:
        data = np.load(args.scores)
        mean, per_video = ev.evaluate_split({n: data[n] for n in names})
        for n, r in per_video.items():
            print(f"  {n:9s} F={r['fscore']:6.2f}%  tau={r['tau']:.3f}  rho={r['rho']:.3f}")
        print(f"{args.dataset} ({len(names)} videos): F={mean['fscore']:.2f}%  "
              f"tau={mean['tau']:.3f}  rho={mean['rho']:.3f}")


if __name__ == "__main__":
    main()
