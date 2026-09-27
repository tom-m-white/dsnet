# V5 baseline: September 26, 2026

**Below the requested acceptance range.** The simplified three-branch model completed
all five SumMe splits, seed 0, 300 epochs, final weights, per-annotator evaluation.
This is an honest baseline result, not a successful reproduction of DSTG-VS.

| Split | V5 F-score (%) | V5 tau | V5 rho | Existing DSNet F-score (%) |
|---|---:|---:|---:|---:|
| 0 | 41.30 | -0.03555 | -0.04339 | 44.59 |
| 1 | 44.73 | -0.03755 | -0.04587 | 51.92 |
| 2 | 39.12 | -0.01974 | -0.02400 | 37.67 |
| 3 | 44.41 | 0.03103 | 0.03792 | 42.51 |
| 4 | 29.84 | -0.03671 | -0.04490 | 36.27 |
| **Mean** | **39.88** | **-0.01970** | **-0.02405** | **42.59** |

Across-split sample standard deviations: F=6.07 percentage points, tau=0.02929,
rho=0.03579. These overlapping splits do not provide five independent samples.
The new mean is 2.71 points below the existing DSNet mean and below Wang's 45-55%
acceptable range. No checkpoint was selected using a test score.

## What the diagnostic shows

Re-evaluating the final checkpoints against their training targets gives mean per-video
MSE **0.000855** on training data and **0.035315** on test data (averaged over splits).
Mean within-video Pearson correlation with `gtscore` is 0.984-0.991 on training data,
but -0.107 to 0.088 on test data. These diagnostic Pearson correlations are distinct
from the reported per-annotator tau/rho. The gap indicates strong overfitting; it does
not by itself prove the architecture is the only cause or rule out all implementation issues.

Verified: graph structure; branch selection and inactive parameters; gradients for
all variants; shared middle-layer calls; edge weights affecting attention; source-to-target
direction; cached/uncached agreement; nonempty/degenerate inputs; strict aggregation;
benchmark warmup and reporting. All 7 unit tests and the 13 original toy checks pass.
A one-epoch integration smoke run wrote valid metrics/checkpoints/scores to a separate
temporary directory. Reloading the trained split-0 checkpoint on CPU reproduces saved
CUDA predictions for all five test videos within rtol=1e-4 / atol=2e-6.

## Hardware interface check (stats)

On a RTX 4060 Laptop GPU, the trained split-0 V5 model processed synthetic
1000 x 1024 input in **52.87 +/- 1.82 ms** (8 timed runs after 2 warmups, sample SD),
about **18,914 feature rows/second**. It has *exactly* **888,577 parameters** and peak Torch
CUDA allocated memory of **84,730,368 bytes**. This is a local development measurement,
not the lab workstation/Jetson/Pi comparison. CPU inference and the dummy benchmark
also ran successfully. Power and energy have **not** been measured.

Timing includes dense graph construction and internal CPU/GPU transfers, excluding
feature extraction. GPU activity, power modes and thermals need controlled measurement
for paper tables. See `decisions.md` for the memory measurement caveats.

## Files and reproduction

- Accuracy results: `results/V5_summe_{0..4}_0.json`.
- Final checkpoints: `models/V5_summe_{0..4}_0.pt`.
- Saved scores: `scores/V5_summe_{0..4}_0.npz`.
- Summary: `results/V5_summe_seed0_summary.json`.
- Fit diagnostic: `results/V5_summe_seed0_fit_diagnostic.json`.
- Local GPU benchmark: `results/benchmarks/V5_summe_0_0_cuda.json`.

```powershell
conda activate dsnet
python aggregate.py --variant V5 --dataset summe --seeds 0
python -m unittest discover -s tests -v
```

Generated outputs are ignored by Git. Preserve/copy them separately when sharing.