"""Topology, edge-weight, gradient and cached-inference contract tests."""
import unittest

import numpy as np
import torch

from dstg.graph import build_graphs, temporal_decay
from dstg.model import DSTG
from train import load_config


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.x = torch.rand(10, 1024)

    def test_all_variants_and_removed_branches(self):
        counts = []
        for graphs in load_config()["variants"].values():
            model = DSTG(graphs=graphs)
            self.assertEqual(set(model.gat), set(graphs))
            self.assertEqual(set(model.independent_sage), set(graphs))
            calls = []
            handle = model.shared_sage.register_forward_hook(lambda *args: calls.append(1))
            y = model(self.x)
            handle.remove()
            self.assertEqual(len(calls), len(graphs))
            self.assertEqual(tuple(y.shape), (10,))
            self.assertTrue(torch.isfinite(y).all() and (y >= 0).all() and (y <= 1).all())
            (y - torch.linspace(0, 1, 10)).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])
        self.assertLess(counts[2], counts[3])
        self.assertLess(counts[3], counts[4])

    def test_cached_graphs_and_direction(self):
        model = DSTG().eval()
        edges = model.prepare_graphs(self.x)
        self.assertTrue((edges["fwd"][0][0] < edges["fwd"][0][1]).all())
        self.assertTrue((edges["bwd"][0][0] > edges["bwd"][0][1]).all())
        torch.testing.assert_close(model(self.x), model(self.x, edges))
        reordered = DSTG(graphs=["omni", "bwd", "fwd"])
        reordered.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(self.x), reordered(self.x))

    def test_weights_affect_attention(self):
        model = DSTG(graphs=["omni"]).eval()
        edges = model.prepare_graphs(self.x)
        ei, ew = edges["omni"]
        changed = {"omni": (ei, torch.linspace(0, 10, len(ew)))}
        self.assertGreater((model(self.x, edges) - model(self.x, changed)).abs().max().item(), 1e-7)

    def test_signal_flows_source_to_target(self):
        model = DSTG(graphs=["fwd"]).eval()
        edges = model.prepare_graphs(self.x)
        altered = self.x.clone()
        altered[-1] += 10
        # Last frame cannot influence any earlier frame through forward edges.
        torch.testing.assert_close(model(self.x, edges)[:-1], model(altered, edges)[:-1])

    def test_degenerate_graphs_and_validation(self):
        for a in build_graphs(np.zeros((1, 3))):
            self.assertTrue(np.isfinite(a).all())
            self.assertEqual(a.sum(), 0)
        self.assertEqual(temporal_decay("logarithmic", 0.7, 1)[0], 1)
        for x in (np.empty((0, 3)), np.array([[np.nan]]), np.ones(3)):
            with self.assertRaises(ValueError):
                build_graphs(x)
        for graphs in (["fwd", "fwd"], ["unknown"]):
            with self.assertRaises(ValueError):
                DSTG(graphs=graphs)

    def test_graph_free_control_is_rowwise_and_has_no_graph_parameters(self):
        model = DSTG(graphs=[])
        self.assertEqual(model.prepare_graphs(self.x), {})
        self.assertEqual(len(model.gat), 0)
        self.assertIsNone(model.shared_sage)
        y = model(self.x)
        changed = self.x.clone()
        changed[4] += 5
        other = model(changed)
        keep = torch.arange(len(y)) != 4
        torch.testing.assert_close(y[keep], other[keep])
        self.assertNotEqual(y[4].item(), other[4].item())
        y.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_output_rows_follow_node_ids_with_remapped_edges(self):
        model = DSTG().eval()
        edges = model.prepare_graphs(self.x)
        permutation = torch.randperm(len(self.x))
        inverse = torch.argsort(permutation)
        remapped = {g: (inverse[ei], ew) for g, (ei, ew) in edges.items()}
        torch.testing.assert_close(model(self.x, edges)[permutation],
                                   model(self.x[permutation], remapped), atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(model(self.x[:1]).shape), (1,))


if __name__ == "__main__":
    unittest.main()
