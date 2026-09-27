"""Inference timing interface for Dakota; physical power measurement is separate.

Use --variant dummy for a single linear layer, or a trained V1..V5 checkpoint.
The measured path includes graph construction and CPU/GPU transfers within forward,
but excludes feature extraction, input allocation, checkpoint loading and evaluation.
Do not run alongside training or other GPU workloads when collecting paper numbers.
"""
import argparse
import hashlib
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent


def benchmark(model, x, repeats=10, warmup=2, cpu_threads=4, rss_poll_seconds=0.001):
    if not 0 <= warmup < repeats or repeats - warmup < 2:
        raise ValueError("need at least two measured iterations after warmup")
    if cpu_threads < 1 or rss_poll_seconds <= 0:
        raise ValueError("thread count and sampling interval must be positive")
    torch.set_num_threads(cpu_threads)
    model.eval()
    device = x.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings = []
    with threadpool_limits(limits=cpu_threads), torch.inference_mode():
        pools = [{k: pool.get(k) for k in ("internal_api", "num_threads", "version")}
                 for pool in threadpool_info()]
        for _ in range(warmup):
            model(x)
            synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Memory is a separate pass so the sampling thread does not bias timing.
        for _ in range(repeats - warmup):
            synchronize()
            start = time.perf_counter()
            output = model(x)
            synchronize()
            timings.append(time.perf_counter() - start)
            if output.numel() != x.shape[0] or not torch.isfinite(output).all():
                raise ValueError("model must return one finite importance score per frame")
            del output
        cuda_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        cuda_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        process = psutil.Process()
        baseline = process.memory_info().rss
        peak_rss = [baseline]
        stop = threading.Event()

        def sample_memory():
            while not stop.is_set():
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
                stop.wait(rss_poll_seconds)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        try:
            model(x)
            synchronize()
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        finally:
            stop.set()
            sampler.join()
    mean = statistics.mean(timings)
    return {
        "latency_mean_seconds": mean, "latency_std_seconds": statistics.stdev(timings),
        "latency_std_convention": "sample_ddof_1", "latency_seconds": timings,
        "frames_per_second": x.shape[0] / mean,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "peak_cuda_allocated_bytes": cuda_peak, "peak_cuda_reserved_bytes": cuda_reserved,
        "peak_process_rss_sampled_bytes": peak_rss[0], "baseline_process_rss_bytes": baseline,
        "memory_note": "RSS is a sampled process-wide lower bound in a separate warm inference pass; "
                       "CUDA peak is allocator memory during timed passes, not total device usage.",
        "measured_repeats": repeats - warmup, "discarded_warmup": warmup,
        "cpu_threads": cpu_threads, "rss_poll_seconds": rss_poll_seconds,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "native_thread_pools": pools,
        "input_shape": list(x.shape), "device": str(device),
        "timing_scope": "features_to_scores_including_graph_construction",
        "energy_joules_per_video": None,
        "power_measurement_status": "not_measured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("dummy", "V1", "V2", "V3", "V4", "V5"))
    parser.add_argument("--dataset", required=True, choices=("summe", "tvsum"))
    parser.add_argument("--split", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "benchmark_config.yaml").read_text())
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    device = cfg.pop("device")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_num_interop_threads(1)
    training_metadata = None
    if args.variant == "dummy":
        model = torch.nn.Linear(cfg["input_dim"], 1)
    else:
        from dstg.model import DSTG
        checkpoint = ROOT / "models" / f"{args.variant}_{args.dataset}_{args.split}_{args.seed}.pt"
        # These are locally generated research checkpoints; do not load untrusted .pt files.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        training_metadata = saved["result"]
        model_cfg = training_metadata["config"]
        model = DSTG(graphs=training_metadata["graphs"], **model_cfg["model"],
                     graph_config=model_cfg["datasets"][args.dataset]["graph"])
        model.load_state_dict(saved["state_dict"])
    model.to(device)
    x = torch.rand(cfg.pop("frames"), cfg.pop("input_dim"), device=device)
    result = benchmark(model, x, **cfg)
    result.update(schema_version=1, **vars(args), python=platform.python_version(),
                  torch=str(torch.__version__), operating_system=platform.platform(),
                  gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
                  trained_checkpoint=training_metadata is not None,
                  checkpoint_source_sha256=training_metadata["source_sha256"] if training_metadata else None)
    result["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["benchmark_config_sha256"] = hashlib.sha256((ROOT / "benchmark_config.yaml").read_bytes()).hexdigest()
    path = ROOT / "results" / "benchmarks" / f"{args.variant}_{args.dataset}_{args.split}_{args.seed}_{device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {path}")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
