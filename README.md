# Video summarization: a shared evaluation ruler + a graph-topology ablation

Research code for an undergraduate research project at Gonzaga University on **video
summarization** — given a long video, pick the ~15% of it that best represents the whole.

The repo contains two things:

1. **`evaluate.py`** — one evaluation implementation shared by every model in the project, so that
   numbers from different people are comparable. It takes a plain array of predicted importance
   scores and returns **F-score, Kendall's τ and Spearman's ρ**. No model dependency, no deep
   learning framework needed.
2. **`dstg/`** — a reimplementation of the graph construction and model from
   [DSTG-VS](https://doi.org/10.1016/j.patcog.2026.113478) (Li et al., *Pattern Recognition*, 2026),
   used to test which of its three graph branches (forward, backward, omni) are actually needed.

[DSNet](https://github.com/li-plus/DSNet) (Zhu et al., IEEE TIP 2020) is used as a reference model
and baseline. It is cloned locally rather than vendored; see [Working with the DSNet clone](#working-with-the-dsnet-clone).

**Status:** work in progress. `evaluate.py` and the graph construction are done and tested; the
model and the ablation runs are not finished yet.

---

## Why a shared `evaluate.py`

Published τ/ρ numbers in this field are not always comparable, because papers do not agree on the
protocol and often do not state which one they used. The two conventions found in released code:

| | TVSum | SumMe |
|---|---|---|
| [CA-SUM](https://github.com/e-apostolidis/CA-SUM) | per annotator, raw ratings from the original `.tsv` | not computed |
| [CSTA](https://github.com/thswodnjs3/CSTA) | per annotator, raw ratings from the original `.tsv` | annotators averaged, then correlated with the binary summary |

This repo uses the **per-annotator** protocol of
[Otani et al., CVPR 2019](https://arxiv.org/abs/1903.11328) for both datasets: correlate the
prediction with each annotator separately, then average. Averaging annotators first produces a
smoother curve than any real person's and inflates the result — measurably so: under the
CSTA-style SumMe protocol, **random scores already score τ ≈ 0.09**, while the per-annotator
protocol puts random at ≈ 0.00.

**Validation.** The human leave-one-out baseline reproduces the published values exactly, which is
the check that the protocol matches prior work:

| | Published (Otani et al.) | This repo |
|---|---|---|
| TVSum, human τ / ρ | 0.177 / 0.204 | **0.177 / 0.204** |
| SumMe, human τ / ρ | 0.205 / 0.213 | 0.210 / 0.210 |
| Random scores τ / ρ | 0.000 | 0.001 / 0.002 |

The F-score implementation matches DSNet's own to the last digit on every split-0 test video of both
datasets, with no shared code between them.

---

## Setup

Python 3.10, PyTorch 2.4.1 (CUDA 12.1), developed on Windows 11 with an RTX 4060.

```bash
conda create -y -n dsnet python=3.10
conda activate dsnet
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric h5py pyyaml tqdm pytest "scipy<1.14" numpy==1.23.5 ortools==9.7.2996 opencv-python==4.8.1.78 "protobuf<5"
```

The pinned versions matter: DSNet's original code uses `np.bool` and the pre-9.8 OR-Tools knapsack
API, both removed in newer releases. `numpy==1.23.5` and `ortools==9.7.2996` keep that code running
unmodified. `evaluate.py` itself needs only numpy, scipy, h5py and pyyaml.

Check the environment with `python check_env.py`.

### Data

Preprocessed features (h5) from the DSNet release — 1024-dim GoogLeNet pool5 features for every
15th frame, plus human annotations, KTS shot boundaries and the 5 train/test splits:

```bash
mkdir -p DSNet/datasets && cd DSNet/datasets
curl -L -o dsnet_datasets.zip https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip?dl=1
unzip dsnet_datasets.zip
```

For TVSum τ/ρ you also need the **original** TVSum release, which contains the per-annotator 1–5
ratings (`ydata-tvsum50-anno.tsv`). The h5 file does not: its per-annotator field is binary.

```bash
mkdir -p tvsum_original && cd tvsum_original
curl -L -O https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz   # 644 MB
tar -xzf tvsum50_ver_1_1.tgz && cd ydata-tvsum50-v1_1 && unzip ydata-tvsum50-data.zip
```

**The TVSum release is for non-commercial research use and may not be redistributed.** It is
gitignored here; download your own copy.

`python inspect_datasets.py` prints what is inside the h5 files and verifies the splits.

---

## Using `evaluate.py`

Scores are **one value per sampled frame** — the rows of `features` in the h5, aligned with `picks`.

As a library:

```python
from evaluate import Evaluator

ev = Evaluator("summe")                      # or "tvsum"
ev.evaluate_video("video_1", scores)         # {"fscore": .., "tau": .., "rho": ..}
mean, per_video = ev.evaluate_split({"video_1": s1, "video_7": s7})
```

From the command line, with scores saved as an `.npz` (one array per video name):

```bash
python evaluate.py --dataset summe --scores my_scores.npz --split-file DSNet/splits/summe.yml --split 0
python evaluate.py --dataset tvsum --self-check     # human + random baselines
```

Options: `--summe-protocol csta` reproduces CSTA's SumMe numbers for comparison with published
results; `--self-check` prints the human and random baselines that validate the protocol.

**Note:** DSNet ships its own `src/evaluate.py`. If you put `DSNet/src` on `sys.path`, import this
one *first*, or you will silently get the other module.

---

## Reference numbers

DSNet anchor-based (attention), trained here from scratch, per-annotator protocol:

| | F-score | τ | ρ |
|---|---|---|---|
| SumMe split 0, 300 epochs, final model | 44.59% | 0.060 | 0.072 |
| SumMe, published (best epoch by test F-score) | 50.19% | – | – |

The 5.6-point gap is the cost of **not** selecting the model on the test set. DSNet's trainer keeps
whichever epoch scored best on the test split; the runs here use a fixed epoch count and the final
model (`--final-epoch`, added by our patch). Every run in this project follows the latter.

`python walkthrough.py` traces one video end to end — features, attention, anchors, NMS, knapsack,
scoring — printing real intermediate values, which is the fastest way to understand the pipeline.

---

## Working with the DSNet clone

`DSNet/` is upstream's own git repository and is **gitignored** here, so it is never committed into
this repo. Local changes to it live in `patches/dsnet_local.diff`:

```bash
git clone https://github.com/li-plus/DSNet.git
git -C DSNet apply ../patches/dsnet_local.diff
```

The patch covers the OR-Tools import shim, a `//` fix in the GCN branch (both needed only because
the libraries are newer than 2019), and the `--final-epoch` flag. **After editing anything inside
`DSNet/`, regenerate the patch**, or the change exists only on your machine:

```bash
git -C DSNet diff > patches/dsnet_local.diff
```

---

## Repo layout

| Path | What it is |
|---|---|
| `evaluate.py` | the shared ruler: F-score, τ, ρ |
| `export_dsnet_scores.py` | runs a DSNet checkpoint, saves scores as `.npz` for `evaluate.py` |
| `dstg/graph.py` | DSTG-VS Algorithm 1: features → forward / omni / backward adjacency matrices |
| `test_graph.py` | structural tests for the above, on a 10-frame toy example |
| `check_env.py`, `inspect_datasets.py`, `walkthrough.py` | environment check, dataset tour, end-to-end trace |
| `patches/` | local changes to the DSNet clone |
| `decisions.md` | log of every judgment call, with the reasoning |

**`decisions.md` is the important file.** Where a paper is ambiguous, it records which reading was
implemented and why — including two places where DSTG-VS's Algorithm 1 can be read more than one way.

---

## Acknowledgments

- [DSNet](https://github.com/li-plus/DSNet) — Zhu, Lu, Li, Zhou, *IEEE TIP* 2020 (MIT licensed)
- [DSTG-VS](https://doi.org/10.1016/j.patcog.2026.113478) — Li, Jia, Xu, Wang, Meyer, Tan, *Pattern Recognition* 2026
- [CA-SUM](https://github.com/e-apostolidis/CA-SUM) and [CSTA](https://github.com/thswodnjs3/CSTA) — reference implementations of the correlation protocols
- Otani, Nakashima, Rahtu, Heikkilä, *Rethinking the Evaluation of Video Summaries*, CVPR 2019
- Datasets: TVSum (Song et al., CVPR 2015) and SumMe (Gygli et al., ECCV 2014); preprocessed features from [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

Datasets carry their own licenses and are not redistributed here.
