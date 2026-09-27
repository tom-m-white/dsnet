"""Aggregate complete, compatible final-epoch runs; never silently omit missing runs."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def aggregate(variant, dataset, seeds, directory=ROOT / "results"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    rows, signature = [], None
    for seed in seeds:
        for split in range(5):
            path = Path(directory) / f"{variant}_{dataset}_{split}_{seed}.json"
            r = json.loads(path.read_text(encoding="utf-8"))
            identity = (r["variant"], r["dataset"], r["split"], r["seed"])
            if identity != (variant, dataset, split, seed):
                raise ValueError(f"Run identity mismatch: {path}")
            if (r["status"], r["selection"], r["protocol"], r["epochs"]) != (
                    "complete", "final_epoch", "per_annotator", 300):
                raise ValueError(f"Not a locked-protocol 300-epoch run: {path}")
            current = json.dumps({k: r[k] for k in (
                "config", "source_sha256", "splits_sha256", "graphs", "fscore_unit")}, sort_keys=True)
            if signature is not None and current != signature:
                raise ValueError(f"Incompatible source, configuration or splits: {path}")
            signature = current
            if not all(np.isfinite(r[k]) for k in ("fscore", "tau", "rho")):
                raise ValueError(f"Nonfinite metrics: {path}")
            rows.append(r)
    metrics = ("fscore", "tau", "rho")
    return {
        "variant": variant, "dataset": dataset, "seeds": seeds, "n_runs": len(rows),
        "fscore_unit": "percent",
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in metrics},
        "std_over_runs_ddof1": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in metrics},
        "per_seed_mean": {str(seed): {k: float(np.mean([r[k] for r in rows if r["seed"] == seed]))
                                      for k in metrics} for seed in seeds},
        "runs": [{k: r[k] for k in ("split", "seed", *metrics)} for r in rows],
        "note": "Splits overlap; run standard deviations are descriptive, not independent-sample confidence intervals.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", required=True, choices=("summe", "tvsum"))
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(aggregate(**vars(args)), indent=2, allow_nan=False))
