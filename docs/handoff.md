# Software and hardware handoff

The runnable model is `dstg.model.DSTG`; the variant mapping and training settings
are in `config.yaml`. Decisions and limitations are in `decisions.md`.

## Run in Tom's existing environment

```powershell
conda activate dsnet
python -m scripts.check_env
python -m tests.test_graph
python -m unittest discover -s tests -v
python train.py --variant V5 --dataset summe --split 0 --seed 0
python aggregate.py --variant V5 --dataset summe --seeds 0
```

Training refuses to overwrite existing results, checkpoints or scores. Preserve them
before rerunning an identity. `aggregate.py` requires all five splits. Add seeds `1 2`
to aggregate the eventual three-seed experiment. The canonical 150-run grid is five
variants x two datasets x five splits x seeds 0, 1, 2. Assess the V5 baseline and resolve
any implementation defects before spending time on the entire grid; do not tune on test scores.

## Generalization diagnostic (separate from official runs)

`python validate_generalization.py` runs the fixed suite in `validation_config.yaml`:
V5, five SumMe partitions, baseline/weight-decay/attention-dropout conditions,
16 fitting videos and four inner-validation videos per original training partition.
It never evaluates the corresponding original test videos. Existing experiment
outputs are protected from overwriting. The experiment was launched on September 26;
inspect its manifest before trying to run the same experiment again.

Outputs are under `results/validation/20260926_generalization_v1/`, including a
manifest, per-run progress and final JSON, final diagnostic checkpoints and a suite
summary. These **are not test results** and must not enter Dakota's accuracy table.
The final comparison uses epoch 300; earlier curve points are diagnostic only.
`python -m unittest tests.test_validation -v` checks video partitions and read/evaluation guards.

## Result contract for Dakota

`results/V5_summe_0_0.json` is an accuracy result. Stable top-level fields:

| Field | Meaning |
|---|---|
| schema_version | 1 |
| status | complete |
| variant / dataset / split / seed | Run identity |
| fscore | Percentage, e.g. 45.0 means 45%, not 0.45% |
| tau / rho | Per-annotator correlations |
| graphs | Active branch names |
| epochs / selection / protocol | 300 / final_epoch / per_annotator |
| per_video | Metrics keyed by test video |
| parameter_count | Active registered model parameters |
| config / environment / source_sha256 | Provenance |

Checkpoints are `models/{same_identity}.pt`; frame scores are `scores/{same_identity}.npz`.
Dataset files and outputs are ignored by Git. Copy the needed checkpoints explicitly.
Only load trusted local `.pt` files.

## Benchmark interface

```powershell
python benchmark.py --variant dummy --dataset summe --split 0 --seed 0
python benchmark.py --variant V5 --dataset summe --split 0 --seed 0
```

`dummy` needs Torch, PyYAML, psutil and threadpoolctl, and does not import PyG. Real model inference
also needs NumPy and PyG. `benchmark_config.yaml` controls device, CPU threads and
the fixed 1000-frame protocol. Set `device: cpu` for a CPU run. Use an otherwise idle
device and consistent power mode. Results are in `results/benchmarks/` and are kept
separate from accuracy aggregation. Do not compare graph-cached timing against the
default end-to-end feature-to-score timing. Read the memory field labels carefully:
sampled process RSS and Torch CUDA allocation are different quantities.

The program leaves energy null. Dakota must collect timestamped Jetson power or
inline-meter readings on the Pi, record the measurement boundary, and integrate
power over inference. Real video timing can be an additional experiment, but keep
it distinct from the fixed synthetic 1000-row comparison. Record exact lab hardware,
OS, libraries, cooling, thread count and power mode alongside the results.

Reuse/export Tom's conda environment for the workstation; ARM needs its own compatible
inference installation. A Windows environment is not a transferable ARM binary image.
If replacing PyG on ARM, verify numerical output against this implementation before
using timings. A different convolution is not an equivalent hardware benchmark.

## Working backward from October 10

- September 26-28: finish V5 baseline and implementation checks; freeze protocol and
  labels; Dakota confirms hardware access and stand-in benchmarking.
- September 29-October 3: complete the 150-run grid if the baseline is accepted;
  benchmark all five trained variants under the same settings.
- October 4-6: validate energy logs and result completeness; produce accuracy/cost
  tables and paired comparisons using the same splits and seeds.
- October 7-9: manuscript, limitations, reproducibility package and professor review.
- October 10: submission deadline supplied by the team; confirm conference formatting
  and submission cutoff separately.
