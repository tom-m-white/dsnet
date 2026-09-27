# Video summarization: graph-branch ablation

Research with Dr. Wong: measure what accuracy and runtime cost change when forward,
backward and omni graph branches are removed from a simplified DSTG-VS model.
Tom owns the software; Dakota owns hardware benchmarking and energy measurements.

## Current status

- One model class implements all five variants; training, evaluation and benchmarking run.
- V5 SumMe baseline: **39.88% F-score**, tau=-0.0197, rho=-0.0240 over five splits,
  seed 0, 300 epochs, final checkpoint. This is below the requested acceptance range.
- All 15 training-only validation runs are complete. The tested regularization changes
  did not resolve weak generalization. These are diagnostics, not replacement test results.
- The full 150-run sweep and Jetson/Pi energy measurements remain pending.

Start with [decisions.md](decisions.md) for the rationale and latest results,
[the baseline report](docs/baseline.md) for the first V5 experiment, and
[Dakota's handoff](docs/handoff.md) for the result format and hardware workflow.

## Everyday commands

Run from this repository's root using the existing environment:

```powershell
conda activate dsnet
python -m scripts.check_env
python -m unittest discover -s tests -v
```

Training accepts only the four agreed arguments; hyperparameters live in `config.yaml`:

```powershell
python train.py --variant V5 --dataset summe --split 0 --seed 0
python aggregate.py --variant V5 --dataset summe --seeds 0
```

The existing V5 seed-0 runs are already complete, so training those identities again
will refuse to overwrite them. Aggregation requires all five splits for each seed.
Official runs use 300 fixed epochs, final weights and per-annotator tau/rho.

| Variant | Active graphs |
|---|---|
| V1 | forward |
| V2 | backward |
| V3 | omni |
| V4 | forward + backward |
| V5 | forward + backward + omni |

For the standalone ten-frame graph check, run `python -m tests.test_graph`.
For model-independent score evaluation, use `python evaluate.py --help`.

## Benchmarking and validation

```powershell
python benchmark.py --variant dummy --dataset summe --split 0 --seed 0
python benchmark.py --variant V5 --dataset summe --split 0 --seed 0
```

Benchmark settings are in `benchmark_config.yaml`; measure on an idle device.
Existing benchmark files are protected against overwrites. Energy remains unmeasured
until Dakota records physical power readings. See [handoff](docs/handoff.md).

`python validate_generalization.py` runs the suite in `validation_config.yaml`.
The current suite is already complete and refuses to overwrite its outputs. Its
results stay under `results/validation/`, separate from official test aggregation.

## Where things live

| Location | Purpose |
|---|---|
| `dstg/` | Graph construction and the single model class |
| `train.py`, `evaluate.py`, `aggregate.py` | Official accuracy workflow |
| `benchmark.py` | Dakota's inference benchmark and linear stand-in |
| `validate_generalization.py` | Training-only generalization diagnostics |
| `config.yaml`, `benchmark_config.yaml`, `validation_config.yaml` | Settings for those workflows |
| `decisions.md` | Required research decision log and results |
| `tests/` | Graph, model, reporting and validation-boundary checks |
| `scripts/` | Environment check, dataset inspection and DSNet reference utilities |
| `docs/` | Baseline report, handoff and detailed evaluation/setup reference |
| `patches/` | Our compatibility and final-epoch changes to the separate DSNet clone |
| `results/`, `models/`, `scores/` | Local results, saved checkpoints and frame scores |
| `DSNet/`, `tvsum_original/` | Local upstream clone and datasets |

## Keeping the research reproducible

Accuracy results retain the required path `results/{variant}_{dataset}_{split}_{seed}.json`.
Checkpoints, scores, datasets and the separate DSNet clone are ignored by Git, but
remain important: preserve and copy them explicitly when sharing the project.
Do not remove them as clutter. Original recorded source hashes remain meaningful.

Reuse the working environment rather than reinstalling it for routine work. For a
new machine, dataset locations, evaluation details, DSNet patch restoration and
research acknowledgments are retained in [the reference guide](docs/reference.md).
