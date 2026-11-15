import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from graphbench.cora import condition_edges, run_seed, summarize
from graphbench.ego_audit import audit, degree_bucket


class EgoAuditTests(unittest.TestCase):
    def test_degree_bucket_boundaries(self):
        self.assertEqual(degree_bucket(1), "degree_at_most_1")
        self.assertEqual(degree_bucket(2), "degree_2_to_10")
        self.assertEqual(degree_bucket(10), "degree_2_to_10")
        self.assertEqual(degree_bucket(11), "degree_above_10")

    def test_reverse_duplicate_edges_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("# triangle with a leaf\na b\nb a\nb c\na c\nc d\n", encoding="utf-8")
            report = audit(path)

        self.assertEqual(report["nodes"], 4)
        self.assertEqual(report["undirected_edges"], 4)
        self.assertEqual(report["class_counts"], {"degree_2_to_10": 3, "degree_at_most_1": 1})
        self.assertEqual(report["majority_class_accuracy"], 0.75)
        self.assertEqual(report["target_definition_accuracy"], 1.0)

    def test_invalid_edge_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("# bad edge\na\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"graph.txt:2: expected two node IDs"):
                audit(path)


class CoraExperimentTests(unittest.TestCase):
    def setUp(self):
        self.graph = Data(
            x=torch.eye(6),
            y=torch.tensor([0, 1, 0, 1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]]),
            train_mask=torch.tensor([True, True, False, False, False, False]),
            val_mask=torch.tensor([False, False, True, True, False, False]),
            test_mask=torch.tensor([False, False, False, False, True, True]),
        )

    def test_edge_controls_preserve_matched_shapes(self):
        edges = self.graph.edge_index
        self.assertTrue(torch.equal(condition_edges(self.graph, "citation_edges", 42), edges))
        self.assertEqual(condition_edges(self.graph, "self_loops_only", 42).shape, (2, 0))

        shuffled = condition_edges(self.graph, "shuffled_node_ids", 42)
        self.assertFalse(torch.equal(shuffled, edges))
        self.assertTrue(torch.equal(shuffled, condition_edges(self.graph, "shuffled_node_ids", 42)))
        original_degrees = torch.bincount(edges[0], minlength=self.graph.num_nodes).sort().values
        shuffled_degrees = torch.bincount(shuffled[0], minlength=self.graph.num_nodes).sort().values
        self.assertTrue(torch.equal(original_degrees, shuffled_degrees))

    def test_training_uses_validation_to_choose_a_reproducible_checkpoint(self):
        first = run_seed(self.graph, 2, "citation_edges", seed=42, epochs=4, patience=2)
        second = run_seed(self.graph, 2, "citation_edges", seed=42, epochs=4, patience=2)
        self.assertEqual(first, second)
        self.assertIn(first["best_epoch"], range(1, 5))
        self.assertGreaterEqual(first["validation_loss"], 0)
        self.assertIn(first["test_accuracy"], (0.0, 0.5, 1.0))

    def test_single_seed_standard_deviation_is_unavailable(self):
        runs = [
            {"condition": condition, "test_accuracy": 0.5, "test_macro_f1": 0.4}
            for condition in ("citation_edges", "self_loops_only", "shuffled_node_ids")
        ]
        self.assertEqual(summarize(runs)["citation_edges"]["test_accuracy"], {"mean": 0.5, "sample_sd": None})


if __name__ == "__main__":
    unittest.main()
