import unittest

from validate_generalization import partition_training, read_allowed, measure


class ValidationBoundaryTests(unittest.TestCase):
    def test_video_disjoint_deterministic_partition(self):
        split = {"train_keys": [f"data/video_{i}" for i in range(20)],
                 "test_keys": [f"data/video_{i}" for i in range(20, 25)]}
        fit, val, test = partition_training(split, 4, 12)
        self.assertEqual((len(fit), len(val), len(test)), (16, 4, 5))
        self.assertFalse(set(fit) & set(val) or set(fit + val) & set(test))
        self.assertEqual((fit, val, test), partition_training(split, 4, 12))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            partition_training({"train_keys": ["v1", "v2"], "test_keys": ["v2"]}, 1, 0)

    def test_disallowed_reads_and_evaluation_rejected_before_io(self):
        # None objects would fail if IO or model execution preceded the guard.
        with self.assertRaises(ValueError):
            read_allowed(None, ["test_video"], ["train_video"], None, None)
        with self.assertRaises(ValueError):
            measure(None, None, ["test_video"], None, ["train_video"])


if __name__ == "__main__":
    unittest.main()
