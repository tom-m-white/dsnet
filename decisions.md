# Decisions log

These are judgment calls and findings, newest are first (top).

---

## 2026-09-27: Repository organization for the agreed research workflow

The repo was quite messy, so in order to keep tidiness (especially when transfering this to Dakotas Laptop),
there were several changes made. Most of which was moving things to their dedicated folders. Details on the changes below:
Keep the main training/evaluation/benchmark commands, their configs, and this decision
log at the root. Group supporting material without discarding research evidence:

| Previous location | Current location / command |
|---|---|
| `test_*.py` | `tests/`; `python -m unittest discover -s tests -v` |
| `check_env.py` | `scripts/check_env.py`; `python -m scripts.check_env` |
| `inspect_datasets.py` | `scripts/inspect_datasets.py`; `python -m scripts.inspect_datasets` |
| `export_dsnet_scores.py` | `scripts/export_dsnet_scores.py`; `python -m scripts.export_dsnet_scores` |
| `walkthrough.py` | `scripts/walkthrough.py`; `python -m scripts.walkthrough` |
| `HANDOFF.md` | [docs/handoff.md](docs/handoff.md) |
| `BASELINE.md` | [docs/baseline.md](docs/baseline.md) |
| Long README background/setup | [docs/reference.md](docs/reference.md) |

Run the module commands from the repository root. Historical entries below retain
their original names; this table maps them to current locations. README now provides
the current status, the professor's conventions, everyday commands and a file map.

As states previously, Dr. Wang wants reproducibility in the paper, in order to have
none of the critical files where change/modified. This means
Preserve datasets, DSNet, patches, checkpoints, frame scores, all official results,
and training-only validation evidence. They support reproducibility and the paper.
Research implementation and config files were not relocated or changed, so saved
source hashes and the required `results/{variant}_{dataset}_{split}_{seed}.json`
contract remain valid. The separate DSNet checkout is untouched.

The Verification of the success of this move was done with 
all 10 unit tests and 13 toy graph checks pass after the moves. The
environment checker, dataset inspection, DSNet walkthrough and exporter help command
run from their new module paths. Documentation links resolve. Recorded source hashes
match all five official baseline runs and all 15 validation runs.

---

## 2026-09-26: Training-only validation completed; generalization remains weak


Below is an attempt at getting higher F-Scores, with an investiagation into why
generalization has been so weak. (My current theroy is that there is a lack of data,
specifically, most videos do not change much from frame to frame and so the total "verity"
of frames is limited in an already small dataset. The model is around 800k params,
its possible to memorize and overfit, which fits with what ive been seeing in these tests.)

Completed all 15 predeclared runs in `20260926_generalization_v1`: three
conditions x five SumMe outer partitions x seed 0, 300 epochs each. Every model was
initialized afresh and fitted on 16 videos; its four inner-validation videos were
not used for gradient updates. No existing 20-video-trained checkpoint was reused.
The five corresponding outer-test videos were never evaluated by this experiment.

Purpose: investigate whether the training/held-out gap in the first V5 run
reflects poor generalization and whether two simple, one-factor regularization
changes improve it. These are inner-validation numbers, not new test scores;
they are not directly comparable with the original 39.88% five-split test mean.

Final-epoch means over the five validation partitions (F-score in percent):


| Condition | Fit MSE | Validation MSE | Validation F | Validation tau | Validation rho |
|---|---:|---:|---:|---:|---:|
| Baseline: dropout 0, weight decay 1e-7 | 0.000734 | 0.034170 | 35.6916 | -0.012549 | -0.015298 |
| Weight decay 1e-4 | 0.004179 | 0.039841 | 38.3169 | 0.003806 | 0.004649 |
| Attention dropout 0.3 | 0.000861 | 0.033101 | 38.6042 | -0.000609 | -0.000800 |
| Fixed random-score reference | -- | -- | 41.1110 | 0.014667 | 0.017886 |

The random reference uses one deterministic draw per video and the same draw across
conditions. It is a sanity reference, not a Monte Carlo estimate or significance test.
Being below this particular draw does not establish statistical inferiority to random.

Paired final-epoch validation F-scores:

| Outer partition | Baseline | Weight decay 1e-4 | Attention dropout 0.3 |
|---|---:|---:|---:|
| 0 | 39.28 | 35.49 | 38.71 |
| 1 | 37.42 | 38.80 | 41.52 |
| 2 | 31.06 | 44.62 | 45.22 |
| 3 | 38.75 | 37.41 | 36.53 |
| 4 | 31.94 | 35.27 | 31.05 |

Interpretation and decisions:

- The baseline's mean fitting F-score is 64.71% and fitting tau is 0.3424, versus
  validation F=35.69% and tau=-0.0125. This supports the generalization concern on
  videos not used for fitting; it is not limited to the previously observed test set.
- Stronger weight decay raises mean validation F by **2.6253 points**, improving
  three of five partitions. However, validation MSE is worse on **all five** partitions.
  Tau improves on four partitions but its mean remains near zero. A smaller fit/validation
  gap caused by worse fitting error is not itself improved generalization.
- Attention dropout raises mean validation F by **2.9125 points**, but improves it
  on only **two of five** partitions. The +14.15-point gain on partition 2 strongly
  influences the mean. Validation MSE improves on four partitions (mean reduction
  about 3.13%), but final fitting error remains tiny and mean tau/rho remain near zero.
- Learning curves do not justify a blanket claim that validation MSE always increases
  with training: baseline mean validation MSE is 0.034848 at epoch 25 and 0.034170 at
  epoch 300, while fitting MSE falls from 0.014795 to 0.000734. Most of that fitting
  improvement does not transfer. For stronger weight decay, mean validation MSE rises
  from 0.032991 at epoch 25 to 0.039841 at epoch 300. Earlier points are diagnostic;
  **we did not select earlier checkpoints or change the final-300 protocol**.
- None of these settings resolves the weak ranking/generalization result. Do not
  promote a setting into the official experiment based on pooled validation F alone.
  Original `config.yaml`, model implementation, official checkpoints and test results
  remain unchanged. The full 150-run ablation has not been launched.
- These 20 validation placements contain **15 distinct videos** across overlapping
  outer partitions; only one initialization seed was used. Results are exploratory,
  not five independent test estimates. Keep the cross-partition selection caveat
  recorded in the protocol below.

**Validation and provenance:** three new partition/IO boundary tests pass. Audited all
15 final results for completion at epoch 300, matching fit/validation membership across
conditions, exclusion of the corresponding outer-test videos, and unchanged source
hashes. Initial aggregate metrics agree within 1e-5; sub-micro differences in rank
metrics are consistent with CUDA numerical nondeterminism, so exact bitwise agreement
is not claimed. The run manifest and summary both report `complete`.

Files: `validation_config.yaml`, `validate_generalization.py`, `test_validation.py`;
raw results, progress curves and diagnostic checkpoints under
`results/validation/20260926_generalization_v1/`; aggregate results in that directory's
`summary.json`. Preserve these ignored outputs separately when sharing the repository.

---

## 2026-09-26: Training-only validation launched to investigate generalization

**Why:** the first fixed-300 V5 baseline fit its training targets very closely
(MSE 0.000855) but had much higher held-out MSE (0.035315). We are investigating
generalization, not optimizing against the original test scores. This experiment
tests whether regularization improves performance on unseen videos drawn solely
from each original training partition.

**Predeclared protocol:** `validation_config.yaml`, experiment
`20260926_generalization_v1`; runner `validate_generalization.py`.

- V5, SumMe, all five original outer splits, seed 0: 15 runs total.
- Within each original 20-video training partition, randomly reserve **four whole
  videos** for inner validation and fit on the other 16. Never split adjacent frames
  from the same video between fitting and validation. Partition seed is 271828 plus
  outer split index; the same partition, initialization seed and shuffled video order
  are used across conditions. Dropout necessarily adds its own stochasticity.
- Original five-video outer test sets are excluded from all data reads, gradient
  updates, metrics and checkpoint selection in this runner. Their identifiers are
  retained only to audit the exclusion. Boundary tests reject out-of-partition reads
  and evaluation before any IO/model execution.
- Three conditions fixed **before inspecting this experiment's validation results**:

| Condition | GAT attention dropout | Adam weight decay |
|---|---:|---:|
| baseline | 0 | 1e-7 |
| weight_decay | 0 | 1e-4 |
| attention_dropout | 0.3 | 1e-7 |

- Each condition changes one factor. The dropout is on GAT attention coefficients,
  not newly introduced feature dropout. No architecture change; hidden size 192,
  one head, learning rate 5e-4, MSE, 300 epochs remain fixed.
- Record fit/validation MSE, F-score, per-annotator tau/rho and per-video metrics at
  epochs 0, 1, 25, 50, 100, 150, 200, 250 and 300. Curves diagnose memorization;
  **all comparisons of final models use epoch 300**. No early stopping or validation-best
  checkpoint is used. Include one fixed random-score validation reference per partition.
- Save only final diagnostic checkpoints. Official baseline files and configuration
  stay unchanged. Results/progress/manifests are under
  `results/validation/20260926_generalization_v1/`; they must not enter official test
  aggregation. Source hashes, exact video lists and resolved settings are recorded.
- These are exploratory inner-validation results, not replacement test scores.
  Four validation videos per split and one initialization seed are noisy. Original
  outer splits overlap: a video can belong to different roles in different outer
  splits. Consequently pooled validation results cannot choose a single global
  setting and also preserve all the old outer tests as untouched. Any later nested
  selection must be made separately within each outer training partition, or use a
  genuinely new holdout for an unbiased global selection estimate. The original
  baseline test results were already observed; that history cannot be undone.

Status at launch: exclusion tests passed. Completed results are recorded immediately above.

---

## 2026-09-26: V5 baseline completed, below target :(

See -> \docs\baseline.md for more information on V5s completion.

All five SumMe splits, seed 0, 300 epochs, final checkpoint:
**F=39.88211%, tau=-0.0197034, rho=-0.0240480**. Across-split sample SD:
F=6.07043 points, tau=0.0292942, rho=0.0357950. Per-split F:
41.30 / 44.73 / 39.12 / 44.41 / 29.84%. Compare against the existing DSNet
five-split mean of 42.59055%, not its split-0 result of 44.59%.

Final-checkpoint training-target MSE averages 0.000855 versus 0.035315 held out;
this is strong evidence of overfitting. Preserve the results without changing
epochs or choosing test-selected checkpoints. The 150-run sweep has not been
launched. Next investigation should use training-only validation, with protocol
changes versioned and applied consistently. 

All seven unit tests and 13 toy graph checks pass. CPU checkpoint replay agrees
with the saved CUDA predictions on all five split-0 test videos. Local GPU/CPU
inference benchmarks run; Jetson/Pi deployment and power measurement remain Dakota's
physical-hardware work. Added only `threadpoolctl==3.6.0` to the existing environment
to cap NumPy BLAS as well as Torch threads during benchmarking. Loaded serial
OpenMP pools may remain at one thread; actual pool sizes are saved in benchmark JSON.

---

## 2026-09-26: Software implementation choices made

(Ive tried my best to follow other papers for implementation based procedures, see below)

### Five variants, one implementation

| Variant | Active branches |
|---|---|
| V1 | forward |
| V2 | backward |
| V3 | omni |
| V4 | forward + backward |
| V5 | forward + backward + omni |

`DSTG(graphs=...)` selects both the registered branch parameters and executed branches.
Everything else is identical. Removing branches  changes capacity as well as compute;
this is a branch-removal ablation, not a parameter-matched topology comparison.
The labels were not supplied in the pasted messages, so the table above establishes
our convention for Dakota. Three seeds (0, 1, 2), five splits, two datasets and five
variants would give the requested 150 runs. The first baseline uses seed 0.

### Graph choices

- Retain cosine similarity and an inclusive distance window (`1 <= |i-j| <= W`).
  Cosine removes feature-magnitude dependence; the inclusive interpretation uses every
  decay coefficient calculated in Algorithm 1. No extra top-k selection or threshold.
- Checked every feature row in both H5 datasets: maximum absolute L2 norm error from
  1 is `5.960464477539063e-08` in each dataset. Thus raw dot product and cosine are
  effectively equivalent on these supplied features. This closes the practical
  concern about the unused `X_unit` in Algorithm 1 without changing the data.
- Retain the directed lower-triangular backward correction, no adjacency self-loops,
  and squared normalization, verified visually on page 4 of the provided PDF.
- `edge_index[0]` is source and `[1]` is target: forward sends earlier to later.
  This is a directional branch, not a causal summarizer: similarity normalization,
  backward/omni branches and final video summarization can use the whole video.
- A one-frame input has empty graph edges. For logarithmic decay with W=1, use
  weight 1 for the sole distance instead of the undefined log(1)/log(1).

### Architecture and weighting

Related work resolves the unclear sharing language in DSTG-VS:
[VideoSAGE Fig. 2 and Section 3](https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Chaves_VideoSAGE_Video_Summarization_with_Graph_Representation_Learning_CVPRW_2024_paper.pdf)
explicitly shares only the second layer across the three graph modules. Its
[released SPELL implementation](https://github.com/IntelLabs/GraVi-T/blob/main/gravit/models/context_reasoning/spell.py)
also has separate first/third layers and a single reused middle layer. We use:

`per-branch GATConv(1024,192) -> ReLU -> shared SAGEConv(192,192) -> ReLU -> per-branch SAGEConv(192,192)`.

Sum active branch outputs, then a shared linear 192->1 head and sigmoid. One attention
head, no tuned hyperparameters, no dropout (p=0), no normalization layers. This is a
simplified reimplementation: no ViL, GLF, global pooling, CLS/positional encoding or
the paper's separate multistage dimensionality reduction; the first GAT projects
1024->192 directly. K=4/10 is retained in config as paper metadata, **unused** because
it controls global pooling and GLF is omitted. The fixed input features construct
graphs; learned features do not rebuild graph weights during training.

PyG's standard SAGEConv has no edge-weight argument. We use the adjacency values as
scalar `edge_attr` in GAT (`edge_dim=1`), so temporal decay and similarity actually
affect attention; both SAGE layers then use standard unweighted neighbor means.
This is an explicit interpretation, not an assertion of equivalence to unpublished
DSTG-VS code. We do not discard weights from the entire network or pass them as the
unrelated SAGE `size` argument. GAT adds self-loops with edge attribute 1; SAGE uses
its default separate root transform without adjacency self-loops.
References: [GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html),
[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html).

### Training, evaluation and saved runs (things ive followed/did)

- All hyperparameters are in `config.yaml`; training CLI accepts only variant,
  dataset, split and seed. Adam, MSE against unmodified `gtscore`, one video per
  update, shuffled train videos each epoch, 300 fixed epochs, final weights only.
- Seed Python, NumPy and Torch; CPU threads fixed to four. CUDA scatter operations
  can be nondeterministic, so seeded runs are not claimed bitwise reproducible.
- Cache fixed training graph edges for speed. No held-out score is computed during
  training, and test videos never update the optimizer. Evaluate once after epoch 300.
- Write `results/{variant}_{dataset}_{split}_{seed}.json`, with top-level `fscore`
  in percent, `tau`, `rho`, identity, per-video metrics, resolved config, source hashes,
  split hash and environment. Matching checkpoints/scores go in `models/` and `scores/`.
  Refuse overwrites. A completed result is written atomically after its artifacts.
- Aggregation requires all five splits for every requested seed and matching source,
  config, split definition and locked evaluation protocol. Splits overlap: descriptive
  standard deviations are not independent-sample confidence intervals.
- No minimum F-score is enforced in code. Report poor performance without selecting
  a different epoch or tuning on held-out results.

### Correction to scores

Re-evaluated all five saved DSNet final-300 score files with the existing evaluator:
mean F=42.59055%, tau=-0.0080953, rho=-0.0097252. Split 0 is still F=44.59066%,
tau=0.0599416, rho=0.0722548. Compare new five-split means against 42.59%, and split 0
against 44.59%; do not mix these.

The historical statement below that the 5.6-point gap is *entirely caused* by removing
test-set selection is stronger than these measurements establish. A published aggregate
and our single split are not a controlled paired comparison. Keep 5.60 points as the
numerical difference, but causal attribution requires matching splits, seeds, training
and implementation with selection policy as the only change. Do not put the stronger
claim into the manuscript without that experiment.

### Benchmark boundary and hardware section

`benchmark.py` provides a linear stand-in and the same trained-model interface.
10 calls, first 2 discarded, sample standard deviation of the remaining 8; synchronize
CUDA before and after each timed call; `eval()` and `inference_mode()`; fixed CPU threads.
Timing includes graph construction and its transfers, excludes input allocation, feature
extraction, loading and evaluation. All variants currently pay the same dense graph
construction cost; branch savings are in selected edge conversion and neural inference.
Measure on an idle device, separately from training. Synthetic inputs are uniform [0,1]
with shape 1000 x 1024; this measures processed feature rows, not raw camera-video FPS.

CUDA memory is peak Torch allocated/reserved memory, including model/input. CPU memory is
sampled process RSS (including native allocations), measured in an additional warm pass
to avoid timing interference; it is a lower bound and can miss short peaks. Allocator
caching means the baseline-subtracted RSS is not a reliable activation-memory measure.
Energy stays null until Dakota measures it on real hardware. Do not invent wattage or
use laptop power numbers as Jetson/Pi measurements. Integrate timestamped power samples
over a clearly bounded inference interval (watts times seconds = joules), recording the
rail, sampling interval, idle treatment and device power mode; do not sum overlapping rails.

| Device | Model | Memory | Verification |
|---|---|---|---|
| Toms local laptop GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 8188 MiB reported VRAM | nvidia-smi, 2026-09-26; driver 595.97 |
| Toms local machine | Intel Core Ultra 185H | 33,945,935,872 bytes OS-visible RAM | psutil, 2026-09-26 |
| Lab workstation | Pending Dakota | Pending | |
| Jetson | Pending Dakota's exact module/model | Pending | Requires physical access |
| Raspberry Pi 5 | Pi 5 per team message; not inspected | Pending RAM size | Requires physical access |

Existing environment verified without reinstall: Python in conda `dsnet`, Torch
2.4.1+cu121, PyG 2.8.0.post1, h5py 3.16.0. CUDA and a PyG convolution both work.

---

## 2026-09-23: LOCKED by Wang: per-annotator τ/ρ for all runs

All 150 runs use the **per-annotator** protocol (`evaluate.py` default). Reason: under the CSTA
protocol random scores already reach τ ≈ 0.09, so it does not zero out its own null baseline.
The contrast to quote in the paper, same model / split / scores, SumMe split 0:

| Protocol | τ | ρ |
|---|---|---|
| Per-annotator (locked) | 0.060 | 0.072 |
| CSTA (averaged annotators vs. binary summary) | 0.187 | 0.209 |

**Confirmed by Wang: DSTG-VS Table 2 has its ρ/τ column headers transposed**, and the same
transposition carries into Section 4.3's text. The V2Xum-LLaMA row appears to be entered the other
way round, so the table is not internally consistent. This will be noted in the paper.

**Measured effect of test-set model selection:** F = 44.59% (fixed 300 epochs, final model) vs.
DSNet's published 50.19% (best epoch chosen on the test set) = **a 5.6-point gap** on SumMe.
This is the reference line for the new model.

## 2026-09-23: Repo hygiene: separating our work from the DSNet clone

`DSNet/` is upstream's own git repo, so it is listed in `.gitignore` and never committed here. Our
changes to it (5 files) are kept as **`patches/dsnet_local.diff`** instead. To restore them on a new
machine: clone DSNet, then `git -C DSNet apply ../patches/dsnet_local.diff`. **Regenerate the patch
after any further DSNet edit:** `git -C DSNet diff > patches/dsnet_local.diff`.
Also ignored: `tvsum_original/` (license forbids redistribution), `datasets/`, `models/`, `scores/`,
`results/`.

## 2026-09-23: Algorithm 1 (graph construction) — reading of the paper

Implemented in `dstg/graph.py` (numpy only, no model, no PyG) and tested in isolation on a 10-frame
toy example (`test_graph.py`, 12 structural checks).

**Almost certainly a typo in the paper (not yet confirmed with the authors)**

1. **Line 22, `Y_b = triu(Y, -1)` → implemented as `tril(Y, -1)`.** As printed the backward graph
   would be upper-triangular and nearly identical to the omni graph, which would make the backward
   branch redundant by construction. Wang flagged the same reading and directed the lower-triangular
   implementation. Since `Y` is symmetric, `Y_b = Y_f^T`.

**Checked and read correctly — no change needed**

2. **Line 20 is `Y = (Y_0 / max(Y_0))^2`** — squared, not "divided by 2". The exponent is lost when
   the PDF text is extracted, which is how I first misread it; the rendered page is unambiguous.
   Squaring is what "sparse normalization" means here: it is monotonic, so edge ordering is
   unchanged, but weak edges are pushed toward zero harder than strong ones.
3. **Decay uses Δt = t + 1** (line 4), so the nearest neighbour gets `ℓ¹`, not `ℓ⁰`.
4. **No self-loops from `S`:** line 1 subtracts `diag(X·X^T)`. GATConv adds self-loops itself;
   SAGEConv instead keeps a separate root weight for the node, so neither layer needs them in `S`.

**Ambiguous, I need to ask Wang, implementation suggested**

5. **Window bounds, lines 14-16.** `j_s : j_e` can be read exclusively (Python) or inclusively
   (maths), and both are shape-consistent. Exclusive gives `w_t - 1` neighbours per side and leaves
   the last decay value `A[w_t - 1]` computed but unused; inclusive gives `w_t` neighbours and uses
   all of `A`. **here is my implementation suggestion, we use** (`j_e = min(i + w_t + 1, T)`), because a
   computed-but-unused decay value points that way. At SumMe's `w_t = 20`: 20 vs 19 neighbours per side.
6. **Line 1 defines `X_unit = ||X||_2` and then never uses it** — `S = X·X^T - diag(X·X^T)` is built
   from the raw `X`. **Here is my Implementation suggestion`S` as cosine similarity** (row-normalized `X`), since otherwise
   `X_unit` is dead and "frame similarity matrix" normally means cosine. Unlike item 5, this one
   changes the values, so it is the more important of the two to confirm.

---

## 2026-09-22: `evaluate.py` protocol ("follow what previous papers have done")

The DSTG-VS paper reports τ/ρ (citing Kendall 1945 on ties) but does not spell out the procedure.
It takes its split protocol from Apostolidis et al. [19]. So I read the evaluation code of two methods
in its Table 2: **CA-SUM** (Apostolidis et al., the same group as [19]) and **CSTA**.

| Choice | What previous code does | What `evaluate.py` does |
|---|---|---|
| TVSum ground truth for τ/ρ | CA-SUM and CSTA both use per-annotator raw ratings from `ydata-tvsum50-anno.tsv`, normalized by max, subsampled `[::15]` | Same |
| Aggregation | Correlate with each annotator, then average over annotators, then over videos | Same |
| Ties | `kendalltau(rankdata(p), rankdata(t))` (= τ-b) and `spearmanr` (average ranks) | Same |
| What is correlated (TVSum) | The model's predicted importance scores, *before* the knapsack | Same. For DSNet: the per-step score that feeds DSNet's knapsack (`export_dsnet_scores.py`) |
| SumMe τ/ρ | **CA-SUM: not computed** ("applicable only for TVSum", which explains the "–" entries in Table 2). **CSTA: averages the annotators' binary masks** and correlates that with the model's **binary summary** at the original frame rate | **Default: per-annotator** (each annotator's binary `user_summary` at `picks` vs. predicted scores), following Wang's rule. `--summe-protocol csta` reproduces CSTA. |
| F-score | DSNet / DR-DSN keyshot protocol, 15% knapsack, avg (TVSum) / max (SumMe) | Same. Re-implemented with no model or OR-Tools dependency; matches DSNet's own F-score to 0.00 on every split-0 test video (both datasets). |

**Validation of the protocol:**
- **Human leave-one-out baseline, TVSum: τ = 0.177, ρ = 0.204.** This exactly matches the published
  "Human" row (Otani et al.; also in DSTG-VS Table 2).
- Human, SumMe: τ = 0.210, ρ = 0.210 (published: 0.205 / 0.213). Close, but not exact.
- Random scores: τ, ρ ≈ 0.00 under the per-annotator protocol, as expected.

**Finding 1: the CSTA SumMe protocol is biased.** Random scores get **τ = 0.092, ρ = 0.102** under
it, not 0. The knapsack output is correlated with the averaged masks no matter what the input scores
are. For the same pretrained DSNet on SumMe split 0:

| SumMe split 0, pretrained DSNet (AB) | τ | ρ |
|---|---|---|
| Per-annotator, raw scores (default) | 0.009 | 0.011 |
| CSTA protocol | 0.235 | 0.263 |

So published SumMe τ/ρ numbers may not be comparable with each other unless their protocol is known.
**To discuss:** which protocol the DSTG-VS SumMe numbers used.

**Finding 2: random scores get a high F-score** (TVSum 55.3%, SumMe 40.8%), confirming Otani et al.
DSNet's margin over random on TVSum is about 7 points.

**Possible issue in DSTG-VS Table 2 (worth double-checking):** the column header reads ρ then τ,
but the Human row's TVSum values (0.177, 0.204) are Otani's **τ = 0.177, ρ = 0.204**, and CSTA's
published values also appear in τ, ρ order. The column labels may be swapped.

**Other judgment calls:**
- Score resolution: one score per sampled frame (the rows of `features`). The `.tsv` subsampled with
  `[::15]` has exactly the same length for all 50 videos (verified).
- `.tsv` row order ↔ `video_1..50`: CA-SUM and CSTA assume these match. **Verified:** the
  per-annotator frame counts equal `n_frames` for all 50 videos, with 20 annotators each.
- Constant predictions give NaN correlations. They are left as NaN, not replaced, so they get noticed.
- Name clash: `DSNet/src/evaluate.py` also exists. Import ours before adding `DSNet/src` to `sys.path`.

## 2026-09-22: RESULT: SumMe split 0

DSNet anchor-based, 300 epochs, final model (no selection), `evaluate.py` default protocol:

| | F-score | τ | ρ |
|---|---|---|---|
| **SumMe split 0 (deliverable)** | **44.59%** | **0.060** | **0.072** |
| Same model, CSTA protocol (for comparison with published numbers) | 44.59% | 0.187 | 0.209 |
| SumMe, other splits (1/2/3/4) | 51.9 / 37.7 / 42.5 / 36.3% | 0.018 / −0.057 / 0.028 / −0.090 | 0.022 / −0.068 / 0.034 / −0.109 |

- The F-score from `evaluate.py` equals the one `train.py` logged at epoch 299 (0.4459), so the two ways of computing it agree.
- F is just below the expected 45–55%, and τ/ρ are inside the expected 0.05–0.25.
- **Caution:** each split has only 5 test videos, and τ ranges from −0.09 to +0.06 across splits.
  A split-0 number on its own is noisy, so the mean over splits (or more seeds) is more trustworthy.

## 2026-09-22: Deliverable model: no test-set selection

Added `--final-epoch` to DSNet's trainers (`[local patch]`): it trains for a fixed number of epochs and
keeps the **last** model. Used 300 epochs, DSNet's default (not tuned). Anchor-based, attention base
model, SumMe, all 5 splits trained; the deliverable is split 0.

## 2026-09-22: TVSum original annotations

Downloaded `tvsum50_ver_1_1.tgz` (644 MB) from the authors' page (people.csail.mit.edu/yalesong/tvsum)
into `tvsum_original/`. It contains `ydata-tvsum50-anno.tsv` (50 videos × 20 annotators, shot-level
1–5 ratings expanded to every frame) and the **raw videos** (`ydata-tvsum50-video.zip`). The license
restricts it to non-commercial research use and forbids redistribution, so keep it out of any public repo.

---

## 2026-09-21: Per-annotator ground truth for τ/ρ: what our data actually has

**Question (from Wang):** Does our copy of the data contain per-annotator *continuous* importance scores,
so τ/ρ can be computed per annotator and then averaged (Otani et al., CVPR 2019)?

**Finding: No, not in either h5 file.** The fields stored for every video, in every file, are:
`change_points, features, gtscore, gtsummary, n_frame_per_seg, n_frames, n_steps, picks, user_summary`
(plus `video_name` for SumMe). The only per-annotator field is `user_summary`, which is **binary (0/1)**.

| | SumMe | TVSum |
|---|---|---|
| `user_summary` | 15–18 annotators × every original frame, 0/1 | 20 annotators × every original frame, 0/1 |
| What `gtscore` is | **Exactly** the mean of the annotators' binary masks at the sampled frames (max difference 0.0 on video_1) | **Not** derived from `user_summary` (Spearman 0.47 with the mean mask). It is the averaged 1–5 importance ratings. |
| Is per-annotator ground truth recoverable from our copy? | **Yes.** SumMe annotators made binary selections in the first place, so `user_summary` *is* their raw annotation (to confirm against the original SumMe release). | **No.** Each annotator's binary mask is a 15% knapsack summary built from their 1–5 ratings, so information is lost. The raw ratings are in `ydata-tvsum50-anno.tsv` from the original TVSum release. |

**Implication:** for the current deliverable (SumMe split 0), per-annotator τ/ρ can be computed with
the data we have, using each annotator's binary mask. For TVSum we need to download the original
annotation file.

---

## 2026-09-21: Open judgment calls for τ/ρ (resolved 2026-09-22: follow previous papers; see above)

1. **What counts as DSNet's "predicted score"?** DSNet predicts segments, not per-frame scores, so a
   per-step score has to be defined. τ/ρ depend a lot on the choice (preliminary, pretrained, SumMe split 0):

   | Score definition | τ | ρ |
   |---|---|---|
   | Anchor-based, after NMS (max score of boxes covering each step; this is what feeds the knapsack) | 0.009 | 0.011 |
   | Anchor-based, raw (max over the 4 anchors per step, no NMS) | 0.025 | 0.030 |
   | Anchor-free, raw score × centerness | 0.048 | 0.058 |
   | Anchor-free, raw score only | 0.021 | 0.026 |

   All are at or below the expected range (0.05–0.25). *Proposal:* use the score that feeds the
   knapsack, because it is the same score the F-score is computed from. To be confirmed.
2. **Resolution:** compare at the sampled-step level (annotator masks taken at `picks`), not at the
   original frame rate. This matches the resolution of `gtscore`.
3. **Ties:** binary annotator vectors contain massive ties. `scipy.stats.kendalltau` defaults to τ-b,
   which corrects for ties. Keep the default and state it.
4. **Which model?** The pretrained checkpoints (and ours) were **selected on the test set**. Per the
   professor, final runs use a fixed epoch count and the final model. The deliverable needs a model
   retrained that way. How many epochs is still to be decided.

---

## 2026-09-19: Earlier findings

- **Environment:** the repo pins Python 3.6 / torch 1.1, which can't run on an RTX 40-series GPU.
  Used Python 3.10, torch 2.4.1+cu121, numpy 1.23.5, ortools 9.7. Two compatibility patches, both
  marked `[local patch]`: the OR-Tools knapsack import, and GCN `/` → `//`. The evaluation math is unchanged.
- **Splits:** the 5 splits are independent random 80/20 splits, not true 5-fold. 18 of 50 TVSum
  videos and 9 of 25 SumMe videos are never tested.
- **Model selection on the test set:** `train.py` keeps the epoch with the best test F-score. Decision
  (professor): use a fixed epoch count and the final model from now on.
- **Model vs. human agreement:** on **TVSum `video_35`** (tvsum split 0, pretrained anchor-based
  checkpoint `tvsum.yml.0.pt`), DSNet F = 0.733 vs. the 20 annotators, while leave-one-out
  human-vs-human F = 0.629. Reproduce with `python walkthrough.py`.
