"""Run a DSNet checkpoint and save its per-sampled-frame importance scores to .npz for evaluate.py.

The score saved is the one DSNet itself feeds into the knapsack: after NMS, each sampled frame
gets the highest confidence of any predicted segment covering it (vsumm_helper.bbox2summary).

    python export_dsnet_scores.py --model anchor-based \
        --ckpt DSNet/models/pretrain_ab_basic/checkpoint/summe.yml.0.pt \
        --dataset summe --out scores/ab_pretrained_summe_split0.npz
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "DSNet" / "src"))
from helpers import bbox_helper  # noqa: E402
from modules.model_zoo import get_model  # noqa: E402

H5 = {"summe": "eccv16_dataset_summe_google_pool5.h5", "tvsum": "eccv16_dataset_tvsum_google_pool5.h5"}


def dsnet_scores(model, seq, nms_thresh):
    seq_len = len(seq)
    with torch.no_grad():
        cls, bboxes = model.predict(torch.from_numpy(seq).unsqueeze(0))
    bboxes = np.clip(bboxes, 0, seq_len).round().astype(np.int32)
    cls, bboxes = bbox_helper.nms(cls, bboxes, nms_thresh)
    score = np.zeros(seq_len, dtype=np.float32)
    for c, (lo, hi) in zip(cls, bboxes):
        score[lo:hi] = np.maximum(score[lo:hi], c)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("anchor-based", "anchor-free"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True, choices=("summe", "tvsum"))
    p.add_argument("--out", required=True)
    p.add_argument("--nms-thresh", type=float, default=None, help="default: 0.5 AB, 0.4 AF")
    args = p.parse_args()
    nms = args.nms_thresh or (0.5 if args.model == "anchor-based" else 0.4)

    model = get_model(args.model, base_model="attention", num_feature=1024, num_hidden=128,
                      anchor_scales=[4, 8, 16, 32], num_head=8)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    out = {}
    with h5py.File(ROOT / "DSNet" / "datasets" / H5[args.dataset], "r") as f:
        for name in f:  # all videos; evaluate.py picks the split's test videos
            out[name] = dsnet_scores(model, f[name]["features"][...].astype(np.float32), nms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"saved {len(out)} videos -> {args.out}")


if __name__ == "__main__":
    main()
