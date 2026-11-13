"""Original-paper evaluation and integrity tests, using temporary fixtures only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from sklearn.metrics import f1_score
from scipy.stats import kendalltau

from prepare_original import prepare
from protocol import atomic_json, correlations, evaluate_rows, load_json, loss_tensor, predictions, preference, sha
import report as report_module


def prediction_rows(labels, deltas):
    return [dict(comparison_id=f"test_{i:06d}", gold_id=f"gold_{i % 2}", source_split="test", source_row_index=i,
                 label=label, score_a=.5 + delta / 2, score_b=.5 - delta / 2, delta=delta,
                 prediction=preference(delta)) for i, (label, delta) in enumerate(zip(labels, deltas))]


class ProtocolTests(unittest.TestCase):
    def test_strict_epsilon_boundary_both_directions(self):
        epsilon = 1e-4
        self.assertEqual(preference(0), 0)
        self.assertEqual(preference(epsilon), 1)
        self.assertEqual(preference(-epsilon), -1)
        self.assertEqual(preference(math.nextafter(epsilon, 0)), 0)
        self.assertEqual(preference(math.nextafter(-epsilon, 0)), 0)
        self.assertEqual(preference(math.nextafter(epsilon, math.inf)), 1)
        self.assertEqual(preference(math.nextafter(-epsilon, -math.inf)), -1)

    def test_nonfinite_preference_fails(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                preference(value)

    def test_weighted_f1_matches_sklearn_and_differs_from_macro(self):
        labels = [1] * 6 + [0] * 2 + [-1] * 2
        rows = prediction_rows(labels, [.2] * 10)
        values = evaluate_rows(rows)
        self.assertAlmostEqual(values["accuracy"], .6)
        self.assertAlmostEqual(values["f1_weighted"], .45)
        self.assertAlmostEqual(values["f1_macro"], .25)
        self.assertAlmostEqual(values["f1_weighted"], f1_score(labels, [1] * 10, labels=[-1, 0, 1], average="weighted", zero_division=0))
        self.assertAlmostEqual(values["f1_micro"], values["accuracy"])
        self.assertEqual(values["tie_recall"], 0)

    def test_global_statistics_retain_duplicate_content_rows(self):
        rows = prediction_rows([-1, 0, 1], [-.2, 0, .2])
        duplicate = copy.deepcopy(rows[0])
        duplicate.update(comparison_id="test_duplicate", source_row_index=3)
        result = evaluate_rows(rows + [duplicate])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["continuous"]["kendall_tau_b"], 1)
        self.assertEqual(result["aggregation"], "global original rows, duplicates retained")
        with self.assertRaises(ValueError):
            evaluate_rows(rows + [rows[0]])

    def test_score_label_delta_validation(self):
        rows = prediction_rows([1], [.2])
        for field, value in (("score_a", math.nan), ("score_b", 1.1), ("delta", .3), ("label", 2)):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                evaluate_rows(bad)

    def test_constant_correlations_are_explicitly_undefined(self):
        result = correlations([0, 0, 0], [-1, 0, 1])
        self.assertEqual(result["undefined"], "constant input")
        self.assertIsNone(result["kendall_tau_b"])
        self.assertIsNone(result["kendall_tau_c"])
        self.assertIsNone(result["spearman_rho"])

    def test_prediction_sign_is_a_b_not_reversed(self):
        source = [dict(comparison_id="test_000000", gold_id="g", source_split="test", source_row_index=0,
                       label=1, gold_claim="gold", A="a", B="b")]
        actual = predictions(source, {("gold", "a"): .8, ("gold", "b"): .2})[0]
        self.assertAlmostEqual(actual["delta"], .6)
        self.assertEqual(actual["prediction"], 1)

    def test_reference_comparison_uses_one_discrete_vector_and_rank_pairs(self):
        rows = prediction_rows([-1, -1, 0, 0, 1, 1], [-.8, .1, -.2, .2, -.1, .8])
        diagnostics = evaluate_rows(rows)
        actual = report_module.paper_compatible(rows, diagnostics)
        encoding = {1: [1, 2], -1: [2, 1], 0: [1, 1]}
        predicted_flat = [value for row in rows for value in encoding[row["prediction"]]]
        human_flat = [value for row in rows for value in encoding[row["label"]]]
        expected = kendalltau(predicted_flat, human_flat, variant="b").statistic
        self.assertEqual(list(actual), ["accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman"])
        self.assertAlmostEqual(actual["patenteval_rank_pair_tau_b"], expected)
        self.assertEqual(actual["discrete_spearman"], diagnostics["discrete"]["spearman_rho"])
        self.assertNotAlmostEqual(actual["patenteval_rank_pair_tau_b"], diagnostics["continuous"]["kendall_tau_b"])
        self.assertEqual(set(report_module.report_numeric_matches(actual)), set(actual))

    def test_constant_non_draw_labels_remain_variable_after_rank_encoding(self):
        rows = prediction_rows([-1, 0, 1], [.2, .2, .2])
        diagnostics = evaluate_rows(rows)
        self.assertIsNone(diagnostics["discrete"]["kendall_tau_b"])
        actual = report_module.paper_compatible(rows, diagnostics)
        self.assertIsNotNone(actual["patenteval_rank_pair_tau_b"])
        tied = prediction_rows([-1, 0, 1], [0, 0, 0])
        self.assertIsNone(report_module.paper_compatible(tied, evaluate_rows(tied))["patenteval_rank_pair_tau_b"])

    def test_equation5_loss_directions_tolerance_and_fp32_gradient(self):
        try:
            import torch
        except (ImportError, OSError):
            self.skipTest("torch unavailable locally; formal CPU container must run this test")
        a = torch.tensor([.7, .4, .7, .4, .505, .52], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([.5] * 6, dtype=torch.float32)
        labels = torch.tensor([1, 1, -1, -1, 0, 0])
        actual = loss_tensor(a, b, labels, .1, .01)
        expected = torch.tensor([0, .2, .3, 0, 0, .01], dtype=torch.float32).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(actual.dtype, torch.float32)
        actual.backward()
        self.assertLess(float(a.grad[1]), 0)  # A should score higher for +1.
        self.assertGreater(float(a.grad[2]), 0)  # A should score lower for -1.
        self.assertEqual(float(a.grad[4]), 0)  # An allowed equal-quality gap.
        self.assertGreater(float(a.grad[5]), 0)  # Pull an excessive tie gap back.


class ReportIntegrityTests(unittest.TestCase):
    """All synthetic weights/results below live only in an auto-cleaned tempdir."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_paper_report_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.data = self.root / "data"
        self.data.mkdir()
        hashes = {}
        for split, n in (("train", 1044), ("test", 184)):
            source = [dict(gold_claim=f"Synthetic reference {i % 20}", A=f"candidate A {i}", B=f"candidate B {i}",
                           human_eval={"Quality": (i % 3) - 1}) for i in range(n)]
            path = self.data / f"quality_{split}.json"
            atomic_json(path, source)
            hashes[path.name] = sha(path)
        atomic_json(self.data / "source_manifest.json", {"repository": "synthetic-unit-test-only", "downloads": [{"file": name, "sha256": digest} for name, digest in hashes.items()]})
        prepare(self.data, self.output)
        patcher = patch.object(report_module, "FIXED_SOURCE_HASHES", hashes)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.author = self.root / "author_checkpoint"
        self.author.mkdir()
        header = json.dumps({"synthetic": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        header += b" " * ((-len(header)) % 8)
        self.weight_bytes = struct.pack("<Q", len(header)) + header + struct.pack("<f", 0)
        (self.author / "model.safetensors").write_bytes(self.weight_bytes)
        self.config = load_json(Path(__file__).parent / "experiment_config.json")
        self.config["author_checkpoint"]["weights_sha256"] = sha(self.author / "model.safetensors")
        atomic_json(self.output / "experiment_config.json", self.config)
        files = {name: sha(self.output / name) for name in report_module.INPUT_FILES}
        files.update({name: sha(Path(__file__).parent / name) for name in report_module.CORE_FILES})
        self.fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        atomic_json(self.output / "source_fingerprint.json", {"fingerprint": self.fingerprint, "files": files})
        dataset = load_json(self.output / "dataset.json")
        self.pred = []
        for original in dataset[1044:]:
            delta = original["label"] * .2
            self.pred.append({key: original[key] for key in ("comparison_id", "gold_id", "source_split", "source_row_index", "label")})
            self.pred[-1].update(score_a=.5 + delta / 2, score_b=.5 - delta / 2, delta=delta, prediction=preference(delta))
        for length in (2048, 4096):
            pred = [dict(row, max_length=length, checkpoint="released_author") for row in self.pred]
            atomic_json(self.output / "author" / f"predictions_{length}.json", pred)
            atomic_json(self.output / "author" / f"result_{length}.json",
                        dict(status="complete", fingerprint=self.fingerprint, model=self.config["author_checkpoint"], max_length=length,
                             padding="max_length", precision="float32", model_eval=True, elapsed_seconds=1,
                             metrics=evaluate_rows(pred), inference_code_sha256=sha(Path(__file__).parent / "author_modeling.py"),
                             test_used_for_training_or_selection=False))
        # Minimum loss is tied at epochs 5 and 6; the prespecified choice is 5.
        history = [dict(epoch=i + 1, validation_loss=[.5, .4, .3, .2, .1, .1, .2, .3, .4, .5][i],
                        validation_accuracy=.5, validation_continuous_tau=.2) for i in range(10)]
        for seed in (42, 43):
            folder = self.output / f"seed_{seed}"
            result = dict(status="complete", seed=seed, primary=seed == 42, fingerprint=self.fingerprint,
                          assumptions=self.config["assumptions_not_disclosed_by_authors"], history=history,
                          elapsed_seconds=1, peak_vram_bytes=1, checkpoints={})
            for name, epoch, selection in (("final", 10, "prespecified final epoch"), ("validation_best", 5, "minimum validation loss")):
                checkpoint = folder / name
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.safetensors").write_bytes(self.weight_bytes)
                atomic_json(checkpoint / "config.json", {"synthetic_test_only": True})
                metadata = dict(seed=seed, epoch=epoch, selection=selection, fingerprint=self.fingerprint, settings=self.config["training"])
                atomic_json(checkpoint / "reproduction_metadata.json", metadata)
                atomic_json(folder / f"{name}_test_predictions.json", self.pred)
                result["checkpoints"][name] = dict(metadata=metadata, metrics=evaluate_rows(self.pred), files={p.name: sha(p) for p in checkpoint.iterdir()})
            atomic_json(folder / "result.json", result)

    def run_report(self):
        return report_module.report(self.output, self.data, self.author)

    def test_complete_six_conditions_never_claims_full_protocol_reproduced(self):
        revision = {"revision": "synthetic_evaluation_only_test", "model_training_changed": False, "reference": report_module.REFERENCE_CODE}
        atomic_json(self.output / "evaluation_protocol_revision.json", revision)
        result = self.run_report()
        self.assertEqual(result["status"], "complete", result.get("errors"))
        self.assertFalse(result["paper_protocol_reproduced"])
        actual = load_json(self.output / "reproduction_results.json")
        self.assertEqual([row["condition"] for row in actual["rows"]],
                         ["author_2048", "author_4096", "seed_42_final", "seed_42_validation_best", "seed_43_final", "seed_43_validation_best"])
        self.assertEqual(actual["primary_condition"], "seed_42_final")
        self.assertEqual(actual["rows"][3]["epoch"], 5)
        self.assertEqual(actual["rows"][2]["epoch"], 10)
        self.assertEqual(actual["schema_version"], 2)
        self.assertEqual(actual["evaluation_protocol_revision"], revision)
        self.assertEqual(actual["rows"][0]["paper_compatible"]["patenteval_rank_pair_tau_b"], 1.)
        self.assertEqual(actual["rows"][0]["metrics"]["continuous"]["kendall_tau_b"], 1.)
        self.assertNotIn("rounded_numeric_matches", actual["rows"][0])
        self.assertEqual(set(actual["rows"][0]["report_numeric_matches"]), set(actual["rows"][0]["paper_compatible"]))
        manifest = load_json(self.output / "sha_manifest.json")
        self.assertIn("evaluation_protocol_revision.json", [entry["path"] for entry in manifest["files"]])
        for filename in ("report.html", "comparison.png", "comparison.csv", "reproduction_results.json", "verify.json", "sha_manifest.json"):
            self.assertTrue((self.output / filename).is_file(), filename)
        spec = importlib.util.spec_from_file_location("paper_independent_verifier_test", Path(__file__).parent / "independent_verify.py")
        independent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(independent)
        with patch.object(independent, "FIXED_SOURCE_HASHES", report_module.FIXED_SOURCE_HASHES):
            verified = independent.verify(self.output, self.data)
        self.assertEqual(verified["status"], "passed")
        self.assertLessEqual(verified["maximum_absolute_difference"], 1e-10)
        self.assertEqual(verified["conditions"][0]["paper_compatible"], actual["rows"][0]["paper_compatible"])

    def test_missing_seed_result_refuses_complete(self):
        (self.output / "seed_43" / "result.json").unlink()
        self.assertEqual(self.run_report()["status"], "incomplete")
        self.assertFalse((self.output / "report.html").exists())

    def test_original_test_order_is_required(self):
        path = self.output / "author" / "predictions_2048.json"
        atomic_json(path, list(reversed(load_json(path))))
        result = self.run_report()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("reordered", result["errors"][0])

    def test_exact_source_hash_is_required(self):
        with (self.data / "quality_test.json").open("ab") as handle:
            handle.write(b" ")
        result = self.run_report()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("source bytes", result["errors"][0])

    def test_wrong_validation_epoch_refused_without_test_selection(self):
        path = self.output / "seed_43" / "result.json"
        value = load_json(path)
        value["checkpoints"]["validation_best"]["metadata"]["epoch"] = 6
        atomic_json(path, value)
        result = self.run_report()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("prespecified", result["errors"][0])

    def test_changed_weight_file_is_refused(self):
        (self.output / "seed_42" / "final" / "model.safetensors").write_bytes(b"tampered")
        result = self.run_report()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("SHA256", result["errors"][0])

    def test_changed_recorded_metrics_is_refused(self):
        path = self.output / "author" / "result_4096.json"
        value = load_json(path)
        value["metrics"]["accuracy"] = .5
        atomic_json(path, value)
        self.assertEqual(self.run_report()["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
