# Signal diagnostics requested by Dr. Wang

Completed September 27, 2026. Purpose: investigate the failure to generalize before
spending time on the 150-run sweep or changing the architecture. These checks use
the existing V5 final checkpoints and the same per-annotator evaluator.

| Condition | F-score (%) | Kendall tau | Spearman rho |
|---|---:|---:|---:|
| V5, own training videos | 63.4007 | 0.340697 | 0.416211 |
| V5, held-out videos | 39.8821 | -0.019703 | -0.024048 |
| Random scores, held-out videos, 100 draws | 40.7973 | -0.000068 | -0.000083 |
| Graph-free control, held-out videos | 37.6195 | -0.020513 | -0.025082 |

All model rows are means across five SumMe splits, seed 0, 300 fixed epochs,
final checkpoint. Random scores are uniform [0,1), seeds 10000 through 10099;
each overlapping video uses the same draw within a seed. Standard deviations
across random five-split means are 1.9178 F-score points, 0.007680 tau and
0.009375 rho. These describe random-score variation on this fixed dataset,
not confidence intervals for generalization. A high chance F-score can coexist
with zero rank correlation under the keyshot/max-annotator evaluation protocol.

## Findings

- Training tau is positive in every split (0.3360 to 0.3467), so a learnable
  signal reaches the output on training videos. The train/test gap supports
  overfitting or poor generalization; it does not establish a single cause.
- Across 25 test-video placements, within-video predicted-score standard
  deviation is 0.02821 to 0.10316, and range is 0.17044 to 0.53280. None meet
  the declared near-constant threshold of 0.001 for either measure. This rules
  out near-constant final outputs at that threshold, not every form of hidden
  feature over-smoothing.
- Output length, target length and picks length equal T throughout all 125
  train/test placements. Picks are ordered and in range. SumMe targets match
  annotation means sampled at picks. Reloaded test predictions match saved
  predictions. A node-permutation test with remapped edges checks row identity;
  a sentinel test checks sampled scores expand to the correct original-frame
  interval. No CLS token is created or stripped in this simplified model.
  This checks the supplied H5 pipeline, not the original raw-video feature extraction.
- `graphs=[]` uses a per-frame 1024 -> 192 ReLU -> 1 sigmoid head with 196,993
  parameters. It uses the same trainer, split, seed, optimizer, loss, epochs and
  evaluator as V5, with no graph operations. This is a diagnostic control, not
  a parameter-matched topology ablation. Its test tau is negative in all five
  splits, so the control does not isolate the graph branches as the cause.

## Decision

Keep the 150-run sweep on hold. Do not launch the conditional residual experiment
yet: removing the graphs did not restore positive held-out correlation. Report
these results to Dr. Wang before choosing the next experiment. These diagnostics
do not prove that dataset size alone causes the problem, or that a residual
connection could never help. Subsequent model selection belongs on training-only
validation; these test results must not become a tuning leaderboard.

## Evidence and reproduction

- `results/diagnostics/20260927_signal/checkpoint_checks.json`: per-video score
  spread, train/test metrics, and all 100 random draws.
- `results/diagnostics/20260927_signal/graph_free/summary.json`: control summary;
  sibling `models/`, `scores/`, and `results/` contain all five runs.
- `diagnostic_config.yaml`: declared control and random-baseline settings.
- `python -m scripts.diagnose_signal` and `python -m scripts.run_graph_free_control`
  reproduce the checks from the repository root with datasets available. Both
  protect existing results. For a fresh diagnostic experiment, change its name
  and provide the historical `source_before` snapshot under that experiment.
- The source snapshot in `results/diagnostics/20260927_signal/source_before/`
  preserves the implementation that produced the original checkpoints. It was
  copied before adding graph-free support; its manifest contains file hashes.
  The historical repository commit is `32629e04950e0364aeccebe3818b12821e15d029`.
  Existing V1-V5 state keys and parameter initialization order are preserved;
  all 125 checkpoint predictions agree with the historical model implementation
  within rtol=1e-4 and atol=2e-6. Historical model-source hashes correctly differ
  from the edited model file; do not relabel the old runs with new hashes.

The [research folder](https://drive.google.com/drive/folders/1UQVa6sIBhWPkS6UU8rK5nOZaa02C-djH)
contains directly browsable `Checkpoints` (25 .pt files) and `Results` (31 .json
files). All filenames and file sizes were verified against local originals.
The earlier archive containing source was removed from Drive after the user
clarified the sharing scope. Source snapshots remain local for reproducibility.
The user is handling collaborator access.
