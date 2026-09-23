"""Look inside the DSNet .h5 datasets and the splits/ folder.

Answers from the meeting notes:
  - What is inside each h5 file (the "features")?
  - TVSum (50 videos) vs SumMe (25 videos): how do they differ?
  - Prove that splits/*.yml define separate TRAINING and TESTING sets.

Run:  python inspect_datasets.py
"""
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).parent / "DSNet"
DATA = ROOT / "datasets"


def describe_dataset(name):
    path = DATA / f"eccv16_dataset_{name}_google_pool5.h5"
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        print(f"\n{'=' * 70}\n{name.upper()}: {path.name}\n{'=' * 70}")
        print(f"number of videos: {len(keys)}")

        v = f[keys[0]]
        print(f"\nfields stored for one video ({keys[0]}):")
        for field in v.keys():
            d = v[field]
            val = d[()] if d.shape == () else f"shape={d.shape} dtype={d.dtype}"
            if isinstance(val, bytes):
                val = val.decode()
            print(f"  {field:18s} {val}")

        if "user_summary" not in v:
            return None  # OVP / YouTube only have features/gtscore/gtsummary

        n_frames = np.array([f[k]["n_frames"][()] for k in keys])
        n_steps = np.array([f[k]["features"].shape[0] for k in keys])
        n_users = np.array([f[k]["user_summary"].shape[0] for k in keys])
        n_shots = np.array([f[k]["change_points"].shape[0] for k in keys])
        gaps = np.concatenate([np.diff(f[k]["picks"][...]) for k in keys])
        feat_dim = v["features"].shape[1]

        print("\nsummary across all videos:")
        print(f"  feature vector per sampled frame: {feat_dim} numbers "
              f"(GoogLeNet pool5 CNN features)")
        print(f"  original frames/video: min {n_frames.min()}, "
              f"max {n_frames.max()}, mean {n_frames.mean():.0f}")
        print(f"  sampled steps/video  : min {n_steps.min()}, "
              f"max {n_steps.max()}, mean {n_steps.mean():.0f}")
        print(f"  sampling gap between picks (frames): "
              f"{sorted(set(gaps.tolist()))}  -> 1 of every 15 frames kept")
        print(f"  annotators (users) per video: {sorted(set(n_users.tolist()))}")
        print(f"  shots (KTS segments)/video: min {n_shots.min()}, "
              f"max {n_shots.max()}, mean {n_shots.mean():.0f}")
        if "video_name" in v:
            names = [f[k]["video_name"][()] for k in keys[:5]]
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            print(f"  example real video names: {names}")
        return set(keys)


def check_splits(split_file, all_keys):
    splits = yaml.safe_load(open(ROOT / "splits" / split_file))
    print(f"\n--- splits/{split_file}: {len(splits)} train/test splits ---")
    tested = []
    for i, s in enumerate(splits):
        train = {Path(k).name for k in s["train_keys"]}
        test = {Path(k).name for k in s["test_keys"]}
        overlap = train & test
        print(f"  fold {i}: train={len(train):3d}  test={len(test):3d}  "
              f"overlap={len(overlap)}  covers all videos: {train | test == all_keys}")
        assert not overlap, "train and test must not share videos!"
        tested += list(test)
    counts = Counter(tested)
    print(f"  distinct videos ever used for testing: {len(counts)} of {len(all_keys)}")
    print(f"  never tested: {len(all_keys - set(counts))} videos; "
          f"{{times tested: #videos}} = {dict(sorted(Counter(counts.values()).items()))}")
    print("  -> the 5 'splits' are independent random 80/20 splits, NOT a true "
          "5-fold partition")


if __name__ == "__main__":
    tvsum = describe_dataset("tvsum")
    summe = describe_dataset("summe")
    describe_dataset("ovp")
    describe_dataset("youtube")

    print(f"\n{'=' * 70}\nTRAIN / TEST SPLITS\n{'=' * 70}")
    check_splits("tvsum.yml", tvsum)
    check_splits("summe.yml", summe)
