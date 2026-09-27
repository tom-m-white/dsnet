"""Fixed-epoch training, with a single held-out evaluation after the final epoch."""
import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import yaml

from dstg.model import DSTG
from evaluate import Evaluator

ROOT = Path(__file__).resolve().parent


def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def run(variant, dataset, split, seed, config=None, output_root=ROOT):
    cfg = config if config is not None else load_config()
    if variant not in cfg["variants"]:
        raise ValueError(f"Unknown variant {variant}; define its agreed graphs in config.yaml")
    if seed < 0 or not 0 <= split < 5:
        raise ValueError("seed must be nonnegative and split must be 0..4")
    tc, dc = cfg["training"], cfg["datasets"][dataset]
    if tc["epochs"] < 1:
        raise ValueError("epochs must be positive")
    run_id = f"{variant}_{dataset}_{split}_{seed}"
    output_root = Path(output_root)
    result_path = output_root / "results" / f"{run_id}.json"
    checkpoint_path = output_root / "models" / f"{run_id}.pt"
    scores_path = output_root / "scores" / f"{run_id}.npz"
    for path in (result_path, checkpoint_path, scores_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(tc["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    device = tc["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DSTG(graphs=cfg["variants"][variant], **cfg["model"],
                 graph_config=dc["graph"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                                 weight_decay=tc["weight_decay"])
    split_path = ROOT / dc["splits"]
    split_data = yaml.safe_load(split_path.read_text())[split]
    train_names = [Path(k).name for k in split_data["train_keys"]]
    test_names = [Path(k).name for k in split_data["test_keys"]]
    if set(train_names) & set(test_names):
        raise ValueError("train/test overlap")
    ev_kwargs = {"tvsum_anno_path": ROOT / dc["annotations"]} if dataset == "tvsum" else {}
    ev = Evaluator(dataset, h5_path=ROOT / dc["h5"], summe_protocol="per_annotator", **ev_kwargs)
    started = time.perf_counter()
    try:
        # Cache only the fixed graph topology; learned activations are recomputed.
        videos = {}
        for name in train_names:
            x = torch.tensor(ev.h5[name]["features"][...], device=device)
            y = torch.tensor(ev.h5[name]["gtscore"][...], device=device).float().reshape(-1)
            videos[name] = (x, y, model.prepare_graphs(x))
        history = []
        rng = random.Random(seed)
        for epoch in range(tc["epochs"]):
            model.train()
            order = train_names.copy()
            rng.shuffle(order)
            losses = []
            for name in order:
                x, target, edges = videos[name]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x, edges), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch + 1}, {name}")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history.append(float(np.mean(losses)))
            if epoch == 0 or (epoch + 1) % tc["log_every"] == 0:
                print(f"{run_id}: epoch {epoch+1}/{tc['epochs']}, MSE={history[-1]:.6f}, "
                      f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        model.eval()
        scores = {}
        with torch.inference_mode():
            for name in test_names:
                x = torch.tensor(ev.h5[name]["features"][...], device=device)
                scores[name] = model(x).cpu().numpy()
        metrics, per_video = ev.evaluate_split(scores)
        if not all(np.isfinite(list(row.values())).all() for row in per_video.values()):
            raise FloatingPointError("Nonfinite evaluation metric; inspect constant predictions/annotations")
        git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()
        source_files = ["train.py", "evaluate.py", "dstg/model.py", "dstg/graph.py"]
        result = {
            "schema_version": 1, "status": "complete", "variant": variant,
            "dataset": dataset, "split": split, "seed": seed,
            "graphs": list(model.graphs), "epochs": tc["epochs"],
            "selection": "final_epoch", "protocol": "per_annotator",
            "fscore_unit": "percent", **metrics, "per_video": per_video,
            "config": cfg, "train_keys": train_names, "test_keys": test_names,
            "training_mse": history, "elapsed_seconds": time.perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "pyg": torch_geometric.__version__, "device": str(device),
                            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None},
            "git_revision": git_revision,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
            "splits_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        }
        torch.save({"state_dict": model.cpu().state_dict(), "result": result}, checkpoint_path)
        np.savez(scores_path, **scores)
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(result_path)
        print(json.dumps(metrics), flush=True)
        return result
    finally:
        ev.h5.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", required=True, choices=("summe", "tvsum"))
    parser.add_argument("--split", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
