"""Protect reporting against incomplete or mixed experiments."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from aggregate import aggregate
from benchmark import benchmark


class PipelineTests(unittest.TestCase):
    def test_aggregate_rejects_missing_and_mismatched_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "status": "complete", "selection": "final_epoch", "protocol": "per_annotator",
                "epochs": 300, "variant": "V5", "dataset": "summe", "seed": 0,
                "config": {"epochs": 300}, "source_sha256": {"model": "a"},
                "splits_sha256": "b", "graphs": ["fwd", "bwd", "omni"],
                "fscore_unit": "percent", "fscore": 40.0, "tau": 0.1, "rho": 0.2,
            }
            for split in range(4):
                Path(tmp, f'V5_summe_{split}_0.json').write_text(json.dumps(dict(base, split=split)))
            with self.assertRaises(FileNotFoundError):
                aggregate("V5", "summe", [0], tmp)
            last = Path(tmp, 'V5_summe_4_0.json')
            last.write_text(json.dumps(dict(base, split=4)))
            summary = aggregate("V5", "summe", [0], tmp)
            self.assertEqual(summary["n_runs"], 5)
            self.assertEqual(summary["mean"]["fscore"], 40.0)
            last.write_text(json.dumps(dict(base, split=4, source_sha256={"model": "changed"})))
            with self.assertRaises(ValueError):
                aggregate("V5", "summe", [0], tmp)

    def test_dummy_benchmark_protocol(self):
        result = benchmark(torch.nn.Linear(1024, 1), torch.rand(1000, 1024), cpu_threads=2)
        self.assertEqual(result["parameter_count"], 1025)
        self.assertEqual(len(result["latency_seconds"]), 8)
        self.assertEqual(result["discarded_warmup"], 2)
        self.assertAlmostEqual(result["frames_per_second"], 1000 / result["latency_mean_seconds"])
        self.assertIsNone(result["peak_cuda_allocated_bytes"])
        self.assertIsNone(result["energy_joules_per_video"])
        # Some loaded OpenMP runtimes remain serial; none may exceed the cap.
        self.assertTrue(all(1 <= p["num_threads"] <= 2 for p in result["native_thread_pools"]))


if __name__ == "__main__":
    unittest.main()
