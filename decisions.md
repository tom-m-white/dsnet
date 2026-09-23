# Decisions log

These are judgment calls and findings, newest are first.

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
