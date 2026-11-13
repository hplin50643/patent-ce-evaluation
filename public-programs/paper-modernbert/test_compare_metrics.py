"""Independent numerical/source checks; no synthetic report is ever generated."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import warnings

import compare_metrics as comparison
from compare_metrics import compare_pair, preference, score_metrics


def prediction_rows(labels, deltas, groups=None):
    result = []
    for index, (label, delta) in enumerate(zip(labels, deltas)):
        # Keep exact boundary values by placing one score at zero.
        result.append({
            "comparison_id": f"test_{index:06d}",
            "gold_id": groups[index] if groups else f"g{index % 3}",
            "source_split": "test", "source_row_index": index,
            "label": int(label), "score_a": max(delta, 0.0),
            "score_b": max(-delta, 0.0), "delta": float(delta),
            "prediction": preference(delta),
        })
    return result


def independent_metrics(rows):
    """Expanded vectors/SciPy plus label-wise counts; no production helpers."""
    from scipy import stats
    labels = [row["label"] for row in rows]
    deltas = [row["score_a"] - row["score_b"] for row in rows]
    predicted = [0 if abs(delta) < 0.0001 else (1 if delta > 0 else -1) for delta in deltas]
    encoding = {1: (1, 2), -1: (2, 1), 0: (1, 1)}
    human_flat = [value for label in labels for value in encoding[label]]
    predicted_flat = [value for label in predicted for value in encoding[label]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau = stats.kendalltau(human_flat, predicted_flat, variant="b").statistic
        rho = stats.spearmanr(labels, predicted).statistic
    ties = labels.count(0)
    per_class = []
    for label in (-1, 0, 1):
        true_positive = sum(y == label and p == label for y, p in zip(labels, predicted))
        support = labels.count(label)
        predicted_count = predicted.count(label)
        precision = true_positive / predicted_count if predicted_count else 0.
        recall = true_positive / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class.append((support, f1))
    return {
        "accuracy": sum(y == p for y, p in zip(labels, predicted)) / len(labels),
        "f1_weighted": sum(support * f1 for support, f1 in per_class) / len(labels),
        "f1_macro": sum(f1 for _, f1 in per_class) / 3,
        "patenteval_rank_pair_tau_b": float(tau) if math.isfinite(tau) else None,
        "discrete_spearman": float(rho) if math.isfinite(rho) else None,
        "tie_recall": sum(y == 0 and p == 0 for y, p in zip(labels, predicted)) / ties if ties else None,
    }


class OptionalMetricAssertions:
    def assert_optional_close(self, actual, expected, places=12):
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertAlmostEqual(actual, expected, places=places)


class MetricTests(OptionalMetricAssertions, unittest.TestCase):
    def test_strict_tie_boundary_and_positive_direction(self):
        self.assertEqual([preference(value) for value in (-0.0001, -0.0000999, 0, 0.0000999, 0.0001)],
                         [-1, 0, 0, 0, 1])
        rows = prediction_rows([-1, 0, 1], [-0.2, 0, 0.2])
        values = score_metrics(rows)
        for name in comparison.PRIMARY_METRICS:
            self.assertEqual(values[name], 1.0)
        self.assertEqual(values["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_closed_form_matches_independent_scipy_and_label_counts_randomized(self):
        rng = random.Random(93)
        for _ in range(100):
            n = rng.randrange(3, 200)
            labels = [rng.choice((-1, 0, 1)) for _ in range(n)]
            deltas = [rng.choice((-.7, -.0001, -.00001, 0., .00001, .0001, .5)) for _ in range(n)]
            rows = prediction_rows(labels, deltas)
            actual, expected = score_metrics(rows), independent_metrics(rows)
            for key, value in expected.items():
                self.assert_optional_close(actual[key], value)

    def test_same_row_shuffle_is_metric_invariant(self):
        rows = prediction_rows([1, 0, -1, 1, -1, 0, 1], [.2, 0, -.1, -.2, -.3, .1, .3],
                               ["z", "a", "z", "b", "a", "b", "z"])
        shuffled = deepcopy(rows)
        random.Random(3).shuffle(shuffled)
        self.assertEqual(score_metrics(rows), score_metrics(shuffled))
        self.assertEqual(score_metrics(rows)["group_macro"]["group_order"], ["a", "b", "z"])

    def test_swapping_all_candidate_directions_and_labels_preserves_metrics(self):
        rows = prediction_rows([1, 0, -1, 1, -1, 0], [.2, 0, -.1, -.2, -.3, .1])
        swapped = deepcopy(rows)
        for row in swapped:
            row["score_a"], row["score_b"] = row["score_b"], row["score_a"]
            row["delta"] *= -1
            row["label"] *= -1
            row["prediction"] *= -1
        original_metrics, inverted = score_metrics(rows), score_metrics(swapped)
        for key in comparison.PRIMARY_METRICS + ("f1_macro", "tie_recall"):
            self.assert_optional_close(inverted[key], original_metrics[key])
        wrong_direction = deepcopy(swapped)
        for row, original in zip(wrong_direction, rows):
            row["label"] = original["label"]
        self.assertLess(score_metrics(wrong_direction)["accuracy"], original_metrics["accuracy"])

    def test_global_undefined_is_null_and_group_diagnostic_zero_is_explicit(self):
        rows = prediction_rows([1, 0, -1], [0, 0, 0])
        actual = score_metrics(rows)
        self.assertIsNone(actual["patenteval_rank_pair_tau_b"])
        self.assertIsNone(actual["discrete_spearman"])
        self.assertEqual(actual["group_macro"]["patenteval_rank_pair_tau_b"], 0)
        self.assertEqual(len(actual["group_macro"]["zero_assigned_gold_ids"]["patenteval_rank_pair_tau_b"]), 3)
        same_non_tie = score_metrics(prediction_rows([1, 1, 1], [.2, .2, .2]))
        self.assertEqual(same_non_tie["patenteval_rank_pair_tau_b"], 1.)
        self.assertIsNone(same_non_tie["discrete_spearman"])

    def test_invalid_predictions_fail_early(self):
        base = prediction_rows([-1, 0, 1], [-.2, 0, .2])
        variations = []
        bad = deepcopy(base); bad[0]["delta"] *= -1; variations.append(bad)
        bad = deepcopy(base); bad[0]["prediction"] *= -1; variations.append(bad)
        bad = deepcopy(base); bad[0]["threshold"] = .01; variations.append(bad)
        bad = deepcopy(base); bad[0]["label"] = True; variations.append(bad)
        bad = deepcopy(base); bad[0]["score_a"] = float("nan"); variations.append(bad)
        bad = deepcopy(base); bad[0]["score_a"] = 1.1; variations.append(bad)
        bad = deepcopy(base); bad[1]["comparison_id"] = bad[0]["comparison_id"]; variations.append(bad)
        for rows in variations:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    score_metrics(rows)
        with self.assertRaises(ValueError):
            score_metrics([])


class PairedBootstrapTests(OptionalMetricAssertions, unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_compare_metrics_test_")
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        originals = [
            {"gold_claim": f"synthetic reference {index % 20}", "A": f"A{index}", "B": f"B{index}",
             "human_eval": {"Quality": (index % 3) - 1}}
            for index in range(184)
        ]
        payload = json.dumps(originals, ensure_ascii=False).encode("utf-8")
        (self.data_dir / "quality_test.json").write_bytes(payload)
        self.expected_hash = hashlib.sha256(payload).hexdigest()
        self.patch = patch.object(comparison, "EXPECTED_TEST_SHA256", self.expected_hash)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        golds = [hashlib.sha256(row["gold_claim"].encode()).hexdigest() for row in originals]
        labels = [row["human_eval"]["Quality"] for row in originals]
        left_deltas = [label * .3 if index % 4 else -label * .3 for index, label in enumerate(labels)]
        right_deltas = [label * .3 for label in labels]
        self.left = prediction_rows(labels, left_deltas, golds)
        self.right = prediction_rows(labels, right_deltas, golds)

    def test_pair_source_checks_exact_order_labels_gold_and_no_seed_pooling(self):
        result = compare_pair(self.left, self.right, bootstrap=30, data_dir=self.data_dir)
        self.assertEqual(result["source_check"]["status"], "passed")
        self.assertEqual(result["n"], 184)
        self.assertEqual(result["n_gold"], 20)
        variations = []
        shuffled = deepcopy(self.right); shuffled[0], shuffled[1] = shuffled[1], shuffled[0]; variations.append(shuffled)
        changed = deepcopy(self.right); changed[0]["label"] = 1; variations.append(changed)
        changed = deepcopy(self.right); changed[0]["gold_id"] = "wrong"; variations.append(changed)
        changed = deepcopy(self.right); changed[0]["source_split"] = "train"; variations.append(changed)
        changed = deepcopy(self.right); changed[0]["source_row_index"] = 1; variations.append(changed)
        changed = deepcopy(self.right); changed[0]["A"] = "swapped source candidate"; variations.append(changed)
        variations.extend((self.right[:-1], self.right + deepcopy(self.right)))
        for rows in variations:
            with self.subTest(n=len(rows)):
                with self.assertRaises(ValueError):
                    compare_pair(self.left, rows, bootstrap=2, data_dir=self.data_dir)
        both_shuffled_left, both_shuffled_right = deepcopy(self.left), deepcopy(self.right)
        random.Random(19).shuffle(both_shuffled_left)
        random.Random(19).shuffle(both_shuffled_right)
        with self.assertRaisesRegex(ValueError, "order/ID"):
            compare_pair(both_shuffled_left, both_shuffled_right, bootstrap=2, data_dir=self.data_dir)

    def test_bootstrap_global_recomputation_matches_expanded_row_reference(self):
        # The groups have unequal row counts, so averaging per-gold primary
        # scores would disagree with this explicit whole-row reconstruction.
        n_resamples = 80
        result = compare_pair(self.left, self.right, bootstrap=n_resamples, seed=42, data_dir=self.data_dir)
        golds = sorted({row["gold_id"] for row in self.left})
        by_left = {gold: [row for row in self.left if row["gold_id"] == gold] for gold in golds}
        by_right = {gold: [row for row in self.right if row["gold_id"] == gold] for gold in golds}
        rng = random.Random(42)
        samples = {name: [] for name in comparison.PRIMARY_METRICS}
        for _ in range(n_resamples):
            chosen = rng.choices(golds, k=20)
            a = independent_metrics([row for gold in chosen for row in by_left[gold]])
            b = independent_metrics([row for gold in chosen for row in by_right[gold]])
            for name in samples:
                if a[name] is not None and b[name] is not None:
                    samples[name].append(b[name] - a[name])
        import numpy as np
        for name, values in samples.items():
            actual = result["bootstrap"]["metrics"][name]
            self.assertEqual(actual["valid_resamples"], len(values))
            expected_ci = np.quantile(values, [.025, .975])
            for computed, expected in zip(actual["ci95"], expected_ci):
                self.assertAlmostEqual(computed, float(expected), places=12)
        self.assertGreater(result["difference"]["accuracy"], 0)
        self.assertEqual(result["bootstrap"]["group_order"], golds)

    def test_swapped_model_roles_negate_points_and_intervals(self):
        forward = compare_pair(self.left, self.right, bootstrap=100, data_dir=self.data_dir)
        reverse = compare_pair(self.right, self.left, bootstrap=100, data_dir=self.data_dir)
        self.assertEqual(forward["bootstrap"]["draws_sha256"], reverse["bootstrap"]["draws_sha256"])
        for name in comparison.COMPARISON_METRICS:
            self.assert_optional_close(reverse["difference"][name], -forward["difference"][name])
            for left, right in zip(reverse["bootstrap"]["metrics"][name]["ci95"],
                                   reversed(forward["bootstrap"]["metrics"][name]["ci95"])):
                self.assertAlmostEqual(left, -right, places=12)

    def test_same_seed_reproducible_and_different_seed_changes_draw_fingerprint(self):
        first = compare_pair(self.left, self.right, bootstrap=50, seed=42, data_dir=self.data_dir)
        second = compare_pair(self.left, self.right, bootstrap=50, seed=42, data_dir=self.data_dir)
        third = compare_pair(self.left, self.right, bootstrap=50, seed=43, data_dir=self.data_dir)
        self.assertEqual(first, second)
        self.assertNotEqual(first["bootstrap"]["draws_sha256"], third["bootstrap"]["draws_sha256"])

    def test_undefined_global_resamples_are_disclosed_and_not_assigned_zero(self):
        tied = deepcopy(self.right)
        for row in tied:
            row.update(score_a=0., score_b=0., delta=0., prediction=0)
        result = compare_pair(self.left, tied, bootstrap=50, data_dir=self.data_dir)
        for name in ("patenteval_rank_pair_tau_b", "discrete_spearman"):
            metric = result["bootstrap"]["metrics"][name]
            self.assertIsNone(result["difference"][name])
            self.assertIsNone(metric["ci95"])
            self.assertEqual(metric["valid_resamples"], 0)
            self.assertEqual(metric["undefined_resamples"], 50)
            self.assertFalse(metric["eligible_for_directional_interpretation"])
        self.assertEqual(result["bootstrap"]["metrics"]["accuracy"]["valid_resamples"], 50)

    def test_less_than_95_percent_valid_is_not_directionally_interpreted(self):
        sparse = deepcopy(self.right)
        only_gold = sparse[0]["gold_id"]
        for row in sparse:
            if row["gold_id"] != only_gold:
                row.update(score_a=0., score_b=0., delta=0., prediction=0)
        result = compare_pair(self.left, sparse, bootstrap=200, data_dir=self.data_dir)
        metric = result["bootstrap"]["metrics"]["patenteval_rank_pair_tau_b"]
        self.assertGreater(metric["valid_resamples"], 0)
        self.assertLess(metric["valid_fraction"], .95)
        self.assertFalse(metric["eligible_for_directional_interpretation"])
        self.assertEqual(metric["interpretation"], "insufficient_defined_resamples")

    def test_source_hash_mismatch_fails_and_explicit_data_dir_overrides_environment(self):
        with patch.dict("os.environ", {"PATENT_CE_DATA_DIR": str(self.data_dir)}):
            result = compare_pair(self.left, self.right, bootstrap=2)
            self.assertEqual(result["source_check"]["source_sha256"], self.expected_hash)
            with self.assertRaises(FileNotFoundError):
                compare_pair(self.left, self.right, bootstrap=2, data_dir=self.data_dir / "missing")
        path = self.data_dir / "quality_test.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            compare_pair(self.left, self.right, bootstrap=2, data_dir=self.data_dir)


if __name__ == "__main__":
    unittest.main()
