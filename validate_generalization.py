"""Training-only, video-disjoint generalization diagnostic; never evaluates outer tests.

The CLI runs the predeclared suite in validation_config.yaml. Outputs are isolated
from official accuracy results. All conditions retain the final epoch, not the
validation-best checkpoint. Original model, trainer, and config are unchanged.
"""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import yaml

from dstg.model import DSTG
from evaluate import Evaluator
from train import ROOT, load_config


def partition_training(split_data, count, seed):
    train = [Path(k).name for k in split_data["train_keys"]]
    forbidden = [Path(k).name for k in split_data["test_keys"]]
    if len(set(train)) != len(train) or set(train) & set(forbidden):
        raise ValueError("duplicate training videos or train/test overlap")
    if not 0 < count < len(train):
        raise ValueError("validation size must leave at least one fitting video")
    ordered = sorted(train)
    random.Random(seed).shuffle(ordered)
    validation, fitting = sorted(ordered[:count]), sorted(ordered[count:])
    assert not (set(fitting) & set(validation))
    assert set(fitting + validation) == set(train)
    return fitting, validation, forbidden


def read_allowed(h5, names, allowed, model, device):
    """Check the whole request BEFORE reading any feature or target data."""
    if not set(names) <= set(allowed):
        raise ValueError("attempt to read a video outside the allowed training partition")
    videos = {}
    for name in names:
        x = torch.tensor(h5[name]["features"][...], device=device)
        y = torch.tensor(h5[name]["gtscore"][...], device=device).float().reshape(-1)
        videos[name] = (x, y, model.prepare_graphs(x))
    return videos


def atomic_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def measure(model, videos, names, evaluator, allowed):
    if not set(names) <= set(allowed):
        raise ValueError("attempt to evaluate outer-test or unknown videos")
    model.eval()
    scores, errors = {}, {}
    with torch.inference_mode():
        for name in names:
            x, target, edges = videos[name]
            prediction = model(x, edges)
            errors[name] = float(torch.nn.functional.mse_loss(prediction, target).item())
            scores[name] = prediction.cpu().numpy()
    means, per_video = evaluator.evaluate_split(scores)
    if not all(np.isfinite(list(v.values())).all() for v in per_video.values()):
        raise FloatingPointError("nonfinite diagnostic metric")
    return {"mse": float(np.mean(list(errors.values()))), **means,
            "per_video": {n: {**per_video[n], "mse": errors[n]} for n in names}}


def run_one(condition, split, seed, base, experiment, output_dir):
    cfg = copy.deepcopy(base)
    changes = experiment["conditions"][condition]
    cfg["model"]["dropout"] = changes["dropout"]
    cfg["training"]["weight_decay"] = changes["weight_decay"]
    variant, dataset = experiment["variant"], experiment["dataset"]
    tc, dc = cfg["training"], cfg["datasets"][dataset]
    split_path = ROOT / dc["splits"]
    split_data = yaml.safe_load(split_path.read_text())[split]
    fit, validation, forbidden = partition_training(
        split_data, experiment["inner_validation_videos"], experiment["partition_seed"] + split)
    allowed = fit + validation
    run_id = f"{variant}_{dataset}_{split}_{seed}"
    directory = Path(output_dir) / condition
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f"{run_id}.json"
    checkpoint_path = directory / f"{run_id}.pt"
    progress_path = directory / f"{run_id}.progress.json"
    if any(p.exists() for p in (result_path, checkpoint_path, progress_path)):
        raise FileExistsError(f"Existing diagnostic run: {directory / run_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(tc["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    device = tc["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DSTG(graphs=cfg["variants"][variant], **cfg["model"], graph_config=dc["graph"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ev_kwargs = {"tvsum_anno_path": ROOT / dc["annotations"]} if dataset == "tvsum" else {}
    evaluator = Evaluator(dataset, h5_path=ROOT / dc["h5"], **ev_kwargs)
    sources = ["validate_generalization.py", "dstg/model.py", "dstg/graph.py", "evaluate.py"]
    result = {
        "schema_version": 1, "status": "running", "experiment": experiment["experiment"],
        "condition": condition, "variant": variant, "dataset": dataset, "split": split, "seed": seed,
        "purpose": experiment["purpose"], "role": "inner_validation_diagnostic_not_test_result",
        "fit_keys": fit, "validation_keys": validation, "excluded_outer_test_keys": forbidden,
        "config": cfg, "validation_config": experiment, "history": [],
        "selection": "final_epoch", "protocol": "per_annotator", "fscore_unit": "percent",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "environment": {"torch": str(torch.__version__), "pyg": torch_geometric.__version__,
                        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None},
    }
    started = time.perf_counter()
    try:
        videos = read_allowed(evaluator.h5, allowed, allowed, model, device)
        # Baseline independent of optimization, evaluated only on inner validation.
        random_scores = {n: np.random.default_rng(seed + i).random(len(videos[n][0]))
                         for i, n in enumerate(validation)}
        result["random_validation"], _ = evaluator.evaluate_split(random_scores)
        order_rng = random.Random(seed)
        epochs_to_measure = set(experiment["evaluation_epochs"]) | {0, tc["epochs"]}
        for epoch in range(tc["epochs"] + 1):
            if epoch:
                model.train()
                order = fit.copy()
                order_rng.shuffle(order)
                for name in order:
                    x, target, edges = videos[name]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.mse_loss(model(x, edges), target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss at epoch {epoch}: {name}")
                    loss.backward()
                    optimizer.step()
            if epoch in epochs_to_measure:
                row = {"epoch": epoch,
                       "fit": measure(model, videos, fit, evaluator, allowed),
                       "validation": measure(model, videos, validation, evaluator, allowed)}
                result["history"].append(row)
                result["elapsed_seconds"] = time.perf_counter() - started
                atomic_json(progress_path, result)
                print(f"{condition} {run_id} epoch {epoch}: fit MSE={row['fit']['mse']:.5f}, "
                      f"val MSE={row['validation']['mse']:.5f}, val F={row['validation']['fscore']:.2f}, "
                      f"val tau={row['validation']['tau']:.4f}", flush=True)
        result["status"] = "complete"
        result["final"] = result["history"][-1]
        result["elapsed_seconds"] = time.perf_counter() - started
        torch.save({"state_dict": model.cpu().state_dict(), "result": result}, checkpoint_path)
        atomic_json(result_path, result)
        atomic_json(progress_path, result)
        return result
    finally:
        evaluator.h5.close()


def summarize(results, experiment):
    summary = {"experiment": experiment["experiment"], "status": "complete",
               "role": "inner_validation_diagnostic_not_test_result", "conditions": {}}
    for condition in experiment["conditions"]:
        rows = [r for r in results if r["condition"] == condition]
        expected = len(experiment["outer_splits"]) * len(experiment["seeds"])
        if len(rows) != expected or any(r["status"] != "complete" for r in rows):
            raise ValueError("incomplete validation suite")
        final = {group: {m: float(np.mean([r["final"][group][m] for r in rows]))
                         for m in ("mse", "fscore", "tau", "rho")} for group in ("fit", "validation")}
        curve = [{"epoch": h["epoch"], **{
            group: {m: float(np.mean([next(x for x in r["history"] if x["epoch"] == h["epoch"])[group][m]
                                     for r in rows])) for m in ("mse", "fscore", "tau", "rho")}
            for group in ("fit", "validation")}} for h in rows[0]["history"]]
        summary["conditions"][condition] = {
            "n_runs": len(rows), "final": final, "curve": curve,
            "per_split": [{"split": r["split"], "seed": r["seed"],
                           "validation": {m: r["final"]["validation"][m]
                                          for m in ("mse", "fscore", "tau", "rho")}} for r in rows],
            "random_validation": {m: float(np.mean([r["random_validation"][m] for r in rows]))
                                  for m in ("fscore", "tau", "rho")},
        }
    return summary


def main():
    experiment = yaml.safe_load((ROOT / "validation_config.yaml").read_text())
    base = load_config()
    output = ROOT / "results" / "validation" / experiment["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Experiment exists; do not overwrite: {output}")
    manifest = {"base_config": base, "experiment": experiment,
                "created_unix_seconds": time.time(), "status": "running"}
    atomic_json(manifest_path, manifest)
    results = []
    try:
        for split in experiment["outer_splits"]:
            for seed in experiment["seeds"]:
                for condition in experiment["conditions"]:
                    results.append(run_one(condition, split, seed, base, experiment, output))
        atomic_json(output / "summary.json", summarize(results, experiment))
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
