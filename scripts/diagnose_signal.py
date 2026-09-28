"""Dr. Wang's checkpoint checks; no fitting and no hyperparameter selection here."""
import importlib.util
import json
import time

import numpy as np
import torch
import yaml

from dstg.model import DSTG
from evaluate import Evaluator
from train import ROOT


def spread(scores):
    return {"min": float(scores.min()), "max": float(scores.max()),
            "mean": float(scores.mean()), "std": float(scores.std()),
            "range": float(np.ptp(scores)), "unique": int(len(np.unique(scores)))}


def main():
    cfg = yaml.safe_load((ROOT / "diagnostic_config.yaml").read_text())
    out = ROOT / "results" / "diagnostics" / cfg["experiment"]
    path = out / "checkpoint_checks.json"
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(cfg["dataset"])
    report = {"config": cfg, "splits": [], "cls": "No CLS token is created or removed in this simplified implementation.",
              "alignment_scope": "H5 feature rows, gtscore, picks, annotation sampling and model row identity; not raw-video extraction verification."}
    spec = importlib.util.spec_from_file_location("dstg.pre_diagnostic_model", out / "source_before/dstg/model.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    try:
        for split in cfg["splits"]:
            run_id = f"V5_{cfg['dataset']}_{split}_{cfg['seed']}"
            saved = torch.load(ROOT / "models" / f"{run_id}.pt", map_location="cpu", weights_only=False)
            meta = saved["result"]; c = meta["config"]
            kwargs = dict(graphs=meta["graphs"], **c["model"], graph_config=c["datasets"][cfg["dataset"]]["graph"])
            model = DSTG(**kwargs).to(device).eval()
            model.load_state_dict(saved["state_dict"])
            previous = old.DSTG(**kwargs).to(device).eval()
            previous.load_state_dict(saved["state_dict"])
            row = {"split": split, "groups": {}}
            with np.load(ROOT / "scores" / f"{run_id}.npz") as stored, torch.inference_mode():
                for group in ("train_keys", "test_keys"):
                    predictions, details = {}, {}
                    for name in meta[group]:
                        video = ev.h5[name]
                        x = torch.tensor(video["features"][...], device=device)
                        picks = video["picks"][...]
                        y = video["gtscore"][...].reshape(-1)
                        edges = model.prepare_graphs(x)
                        pred = model(x, edges).cpu().numpy()
                        np.testing.assert_allclose(pred, previous(x, edges).cpu().numpy(), rtol=1e-4, atol=2e-6)
                        assert pred.shape == y.shape == picks.shape == (len(x),)
                        assert picks[0] == 0 and np.all(np.diff(picks) > 0)
                        assert picks[-1] < int(video["n_frames"][()])
                        np.testing.assert_allclose(y, video["user_summary"][...][:, picks].mean(0), atol=1e-6)
                        if group == "test_keys":
                            np.testing.assert_allclose(pred, stored[name], rtol=1e-4, atol=2e-6)
                        stats = spread(pred)
                        stats.update(T=len(x), target_std=float(y.std()), mse=float(np.mean((pred-y)**2)),
                                     alignment_checks_passed=True,
                                     near_constant_std=stats["std"] < cfg["near_constant_std_threshold"],
                                     near_constant_range=stats["range"] < cfg["near_constant_range_threshold"])
                        predictions[name] = pred
                        details[name] = stats
                    mean, per_video = ev.evaluate_split(predictions)
                    row["groups"][group] = {"metrics": mean, "per_video": {
                        name: {**details[name], **per_video[name]} for name in predictions}}
                    np.savez(out / f"{run_id}_{group}_scores.npz", **predictions)
            report["splits"].append(row)
            print(f"split {split}: train {row['groups']['train_keys']['metrics']}; test {row['groups']['test_keys']['metrics']}", flush=True)
        # Cache per-video random evaluations so overlapping splits use identical draws.
        names = sorted({n for r in report["splits"] for n in r["groups"]["test_keys"]["per_video"]})
        draws = []
        for repeat in range(cfg["random_repeats"]):
            rng = np.random.default_rng(cfg["random_seed_start"] + repeat)
            values = {name: ev.evaluate_video(name, rng.random(len(ev.h5[name]["picks"]))) for name in names}
            per_split = []
            for r in report["splits"]:
                keys = r["groups"]["test_keys"]["per_video"]
                per_split.append({m: float(np.mean([values[n][m] for n in keys])) for m in ("fscore", "tau", "rho")})
            draws.append({"seed": cfg["random_seed_start"] + repeat, "per_split": per_split,
                          "mean": {m: float(np.mean([v[m] for v in per_split])) for m in ("fscore", "tau", "rho")}})
            if (repeat + 1) % 20 == 0:
                print(f"random baseline {repeat+1}/{cfg['random_repeats']}", flush=True)
        report["random"] = {"draws": draws,
            "mean": {m: float(np.mean([d["mean"][m] for d in draws])) for m in ("fscore", "tau", "rho")},
            "std_across_random_draws": {m: float(np.std([d["mean"][m] for d in draws], ddof=1)) for m in ("fscore", "tau", "rho")},
            "note": "Monte Carlo variability of the five-split mean, not uncertainty across independent video datasets."}
        report["means"] = {g: {m: float(np.mean([r["groups"][g]["metrics"][m] for r in report["splits"]]))
                               for m in ("fscore", "tau", "rho")} for g in ("train_keys", "test_keys")}
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "complete"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"means": report["means"], "random": report["random"]["mean"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    finally:
        ev.h5.close()


if __name__ == "__main__":
    main()
