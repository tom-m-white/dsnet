# DSNet investigation — setup, results, and notes

Repo investigated: <https://github.com/li-plus/DSNet> (cloned into `DSNet/`)
Paper: *DSNet: A Flexible Detect-to-Summarize Network for Video Summarization*, IEEE TIP 2020.

**TL;DR:** DSNet installs and runs on this machine (RTX 4060, Windows 11). The authors'
pretrained models reproduce the paper's numbers exactly. Training from scratch lands close to them
(see table below). The data is **not videos**. Each h5 file holds precomputed CNN "features": 1024
numbers for every 15th frame. While digging I found a few things about the evaluation protocol
that matter if we plan to compare against this method (see [Things to pay attention to](#things-to-pay-attention-to)).

---

## 1. What DSNet does (plain English)

*Video summarization* means taking a long video and picking the ~15% of it that best represents the whole.

DSNet treats this like **object detection, but along time**. Where an object detector draws boxes
around objects in an image, DSNet draws "boxes" (time intervals) around the important parts
of a video. It then gives each interval a score and keeps the best-scoring shots, up to 15% of
the video's length (a knapsack problem).

Two variants:
- **Anchor-based**: proposes fixed-length candidate intervals (4, 8, 16, 32 steps) at every time step, scores them, and refines their boundaries.
- **Anchor-free**: at every time step, predicts how far the important segment stretches left and right.

The "base model" that reads the frame sequence can be `attention` (default), `lstm`, `bilstm`, `linear`, or
`gcn`. `gcn` is a **graph** neural network in which every frame is a node and similar frames are
connected by edges. That option is the reason the repo needs `torch_geometric`.

---

## 2. Setup (what I did)

The repo pins Python 3.6 / torch 1.1 (from 2019). Those versions can't use an RTX 40-series GPU, so I used a modern stack:

```bash
conda create -n dsnet python=3.10
conda activate dsnet
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric h5py pyyaml tqdm pytest numpy==1.23.5 ortools==9.7.2996 opencv-python==4.8.1.78 "protobuf<5"
```

(conda is installed at `C:\Users\white\miniconda3` but isn't on PATH. Use the **Anaconda Prompt**,
or call `C:\Users\white\miniconda3\envs\dsnet\python.exe` directly.)

Datasets and pretrained models were downloaded from the README's Dropbox links into `DSNet/datasets/` and `DSNet/models/`.

### Local patches (2 small fixes, both needed only because the libraries are newer)
| File | Change | Why |
|---|---|---|
| `src/helpers/vsumm_helper.py` | Import shim for the OR-Tools knapsack solver | OR-Tools renamed its API. It is the same dynamic-programming solver underneath. |
| `src/modules/models.py` (GCN) | `edge_indices / seq_len` → `//` | In torch ≥1.5, `/` returns floats, so `--base-model gcn` crashed. `//` restores the integer division the authors intended. |

The evaluation math itself is untouched.

---

## 3. The checklist from the meeting

### ✅ (1) `import torch`, using conda / ✅ (2) `import torch_geometric` / ✅ (3) h5py opens the files
```bash
python check_env.py
```
```
torch version : 2.4.1+cu121   CUDA available: True   GPU: NVIDIA GeForce RTX 4060 Laptop GPU
torch_geometric version: 2.8.0.post1   GCNConv on a 4-node graph OK
opened eccv16_dataset_ovp_google_pool5.h5      -> 50 videos
opened eccv16_dataset_summe_google_pool5.h5    -> 25 videos
opened eccv16_dataset_tvsum_google_pool5.h5    -> 50 videos
opened eccv16_dataset_youtube_google_pool5.h5  -> 39 videos
```

### ✅ Unit tests
`pytest tests`: 25 passed, 2 failed. Both failures come from the test file using an old nose-style
`setup()` method that pytest 8 no longer calls. They are not DSNet bugs.

### ✅ "What's in the h5 files / the features?"
```bash
python inspect_datasets.py
```
**We don't have the videos. We have what a CNN "measured" about them.** Here is the hospital analogy
from the meeting. At a check-up you don't hand over your whole body. They draw blood and run a fixed
panel of tests, and each test gives a number. In the same way, for **every 15th frame** of a video,
a pretrained image network (GoogLeNet, `pool5` layer) produced a **fixed panel of 1024 numbers**
describing what is in that frame. DSNet only ever sees these number panels, never pixels.

Fields stored per video (e.g. TVSum `video_1`):

| Field | Shape | Meaning |
|---|---|---|
| `features` | (707, 1024) | 707 sampled frames × 1024 CNN numbers. **This is the model input.** |
| `picks` | (707,) | Which original frame each row came from: 0, 15, 30, ... |
| `n_frames` | 10597 | Length of the original video in frames |
| `n_steps` | 707 | Number of sampled frames (≈ n_frames / 15) |
| `gtscore` | (707,) | Averaged human "importance" score per sampled frame (the **training** target) |
| `user_summary` | (20, 10597) | Each annotator's 0/1 keep-or-drop choice for every original frame (the **test** ground truth) |
| `change_points` | (71, 2) | Start/end frame of each shot, found by the KTS algorithm |
| `n_frame_per_seg` | (71,) | Length of each shot |
| `video_name` | SumMe only | e.g. "Air_Force_One" |

> **About "15" and "470" in the notes:** It is 15 *frames*, not 15 pixels. One frame out of every
> 15 is kept. **470 is the average number of sampled frames (rows of features) per TVSum video.**
> (Original TVSum videos average 7,047 frames, and 7047 / 15 ≈ 470.)

### ✅ TVSum (50) vs SumMe (25)
The other dataset in the notes is **SumMe**.

| | **TVSum** | **SumMe** |
|---|---|---|
| Videos | 50 | 25 |
| Content | YouTube videos in 10 categories (news, how-to, vlogs, ...) | Personal / egocentric videos (sports, holidays, events) |
| Annotators per video | 20 | 15–18 |
| How humans labeled | Each rated **importance scores** (1–5) per 2-second chunk | Each **picked a summary** directly (keep/drop) |
| Avg. length | 7,047 frames (~470 feature rows) | 4,393 frames (~293 rows) |
| Shots per video | 17–130 (avg 47) | 7–65 (avg 30) |
| F-score protocol in code | **average** over annotators | **max** over annotators (best-matching human) |
| Typical F-score | ~62% | ~50% |

Because the two datasets use different protocols (avg vs. max), **their numbers can't be compared
with each other**. SumMe is harder and noisier (few videos, and humans disagree more).
OVP (50) and YouTube (39) are also included. They are used only as extra *training* data in the
"augmented" and "transfer" settings, never for testing.

### ✅ "Prove there is a training and testing dataset" (the `splits/` folder)
`splits/*.yml` lists which videos go in `train_keys` and which go in `test_keys`, 5 times over:

| File | Setting | Train on | Test on |
|---|---|---|---|
| `tvsum.yml` / `summe.yml` | **Canonical** | 80% of that dataset | the other 20% |
| `*_aug.yml` | **Augmented** | 80% of that dataset **+ all of the other 3 datasets** | the same 20% |
| `*_trans.yml` | **Transfer** | **only the other 3 datasets** | all of that dataset |

Proof from `inspect_datasets.py`: in every split, train (40 videos) and test (10 videos) have
**zero overlap**, and together they cover all 50 TVSum videos. For SumMe the split is 20 train / 5 test.

### ✅ Run DSNet: training → evaluation
**Pretrained models (authors' checkpoints) → `evaluate.py`**:
```bash
cd DSNet/src
python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
python evaluate.py anchor-free  --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```
**Trained from scratch here → `evaluate.py`**:
```bash
python train.py    anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
python evaluate.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
```
`train.py` writes one checkpoint per split to `models/<name>/checkpoint/`. `evaluate.py` loads those
checkpoints and scores each split's test videos. Full training takes about 45 minutes on the 4060.

| F-score (%) | TVSum | SumMe |
|---|---|---|
| Paper, anchor-based | 62.05 | 50.19 |
| **Pretrained anchor-based (my run)** | **62.05** ✅ | **50.19** ✅ |
| Paper, anchor-free | 61.86 | 51.18 |
| **Pretrained anchor-free (my run)** | **61.86** ✅ | **51.18** ✅ |
| **Anchor-based trained from scratch (my run)** | **62.32** | **48.32** |

Training from scratch gets within about 2 points of the paper, which is normal run-to-run variation.
SumMe varies more because each test split has only 5 videos. Per-split details are in
`DSNet/models/ab_basic/log.txt` (training) and `eval_log.txt` (evaluation).

The F-scores that `evaluate.py` prints for my trained checkpoints are **identical** to the "max"
F-scores that `train.py` logged. That shows the train → evaluate pipeline is consistent, and it
also demonstrates point 2 below.

---

## Things to pay attention to

These are the points that matter for the question "is the way we evaluate correct?"

1. **The 5 "splits" are not true 5-fold cross-validation.** They are 5 *independent random* 80/20
   splits. As a result, **18 of the 50 TVSum videos are never tested**, while one video is tested 4 times.
   SumMe has the same pattern: 9 of 25 videos are never tested. This convention was inherited from earlier papers (DR-DSN, VASNet).
2. **The test set is also used to choose the checkpoint.** `train.py` evaluates on `test_keys` after
   every epoch and saves the epoch with the **highest test F-score**. There is no separate validation
   set, so the reported numbers are optimistic. You can see this in `log.txt`: the
   "F-score cur/max" column moves a few points from epoch to epoch, and only the max is kept.
3. **TVSum and SumMe are scored differently** (average vs. max over annotators). Anyone
   comparing against DSNet has to use the same protocol and the same split files, or the numbers aren't comparable.
4. **The features are fixed and old** (2016 GoogLeNet). Without the raw videos we can't
   re-extract features with a newer backbone. That requires the original TVSum/SumMe videos,
   and `src/make_dataset.py` shows how to do it.
5. **The summary length is fixed at 15%**, and the summary is built from KTS shots via the knapsack step.
   So the F-score depends partly on shot boundaries that the model doesn't control.

---

## Files I added
| File | Purpose |
|---|---|
| `check_env.py` | Checklist items 1–3 (torch, torch_geometric, h5py) |
| `inspect_datasets.py` | Dumps h5 contents, compares TVSum vs SumMe, verifies the train/test splits |
| `README.md` | This file |

## Open questions for the professor
- Is the goal to **use DSNet as a baseline** in our own work, or to **audit its evaluation**? Points 1–2 above matter a lot for either.
- Should I train the other variants next (anchor-free, the `gcn` base model, augmented/transfer splits)?
- Do we have (or want) the raw videos, so that we could extract our own features?
