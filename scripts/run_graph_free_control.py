"""Same trainer/evaluator/settings as V5, with graphs=[]; diagnostic output only."""
import copy
import json

import numpy as np
import yaml

from train import ROOT, load_config, run


def main():
    diagnostic = yaml.safe_load((ROOT / "diagnostic_config.yaml").read_text())
    config = copy.deepcopy(load_config())
    variant = diagnostic["graph_free_variant"]
    config["variants"][variant] = diagnostic["graph_free_graphs"]
    output = ROOT / "results" / "diagnostics" / diagnostic["experiment"] / "graph_free"
    rows = [run(variant, diagnostic["dataset"], split, diagnostic["seed"],
                config=config, output_root=output) for split in diagnostic["splits"]]
    summary = {"status": "complete", "role": "graph_free_diagnostic_not_topology_ablation",
               "definition": "Per-frame Linear(1024,192), ReLU, Linear(192,1), sigmoid; no graph operations.",
               "config": diagnostic, "parameter_count": rows[0]["parameter_count"],
               "mean": {m: float(np.mean([r[m] for r in rows])) for m in ("fscore", "tau", "rho")},
               "per_split": [{k: r[k] for k in ("split", "fscore", "tau", "rho")} for r in rows]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
