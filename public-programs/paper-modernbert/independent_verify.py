"""Independently verify the completed paired ModernBERT comparison.

No imports from compare_metrics, protocol, report, reference_tau, or their
helpers are used. Point estimates and every one of the 10,000 bootstrap draws
per seed use expanded row arrays, SciPy statistics, and sklearn classification
metrics. In particular, the rank-pair tau uses explicit two-item rank encoding;
it does not reuse the production contingency/phi implementation.

CLI: --output-dir NEW_RUN --baseline-dir COMPLETED_LONGFORMER --data-dir RAW_DATA
Writes independent_verification.json, with status=failed if any required formal
artifact is missing or inconsistent. It never fabricates unfinished model data.
Large checkpoint blobs are not rehashed here: report.collect verifies their
full hashes; this verifier checks raw inputs, metadata, predictions, report/CSV
values, fixed protocol and the complete bootstrap calculation independently.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Callable
import csv


TOLERANCE = 1e-10
EPSILON = 1e-4
LABELS = [-1, 0, 1]
PRIMARY = ["accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman"]
GROUP = ["patenteval_rank_pair_tau_b", "continuous_kendall_tau_b", "discrete_spearman"]
BOOTSTRAP_KEYS = PRIMARY + ["f1_macro", "tie_recall"] + [f"group_macro_{key}" for key in GROUP]
SOURCE_HASHES = {
    "quality_train.json": "20c3e88862b0a8cd3ce6e48607f9887a29c4502d4e87633b41095be194bdca48",
    "quality_test.json": "10e68f47f2f4329f92c6d60936d23f4eaab6b852150b8fa1b039d8d348868039",
}
MODELS = {
    "longformer": ("allenai/longformer-base-4096", "301e6a42cb0d9976a6d6a26a079fef81c18aa895"),
    "modernbert": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
}
CONDITIONS = [f"{model}_seed{seed}_{selection}" for model in MODELS
              for seed in (42, 43) for selection in ("final", "validation_best")]
HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finite(value: Any, name: str) -> float:
    require(not isinstance(value, bool), f"{name}: unexpected boolean")
    result = float(value)
    require(math.isfinite(result), f"{name}: nonfinite number")
    return result


class Audit:
    def __init__(self) -> None:
        self.numeric_values_checked = 0
        self.maximum_absolute_difference = 0.0
        self.maximum_difference_location: str | None = None

    def number(self, expected: float | None, actual: Any, name: str) -> None:
        if expected is None:
            require(actual is None, f"{name}: undefined value must remain null")
            return
        difference = abs(float(expected) - finite(actual, name))
        self.numeric_values_checked += 1
        if difference > self.maximum_absolute_difference:
            self.maximum_absolute_difference = difference
            self.maximum_difference_location = name
        require(difference <= TOLERANCE, f"{name}: absolute difference {difference} exceeds {TOLERANCE}")

    def csv_value(self, expected: Any, actual: str, name: str) -> None:
        if expected is None:
            require(actual == "", f"{name}: null CSV value must be empty")
        elif isinstance(expected, bool):
            require(actual == str(expected), f"{name}: CSV boolean differs")
        elif isinstance(expected, (int, float)):
            self.number(expected, actual, name)
        else:
            require(actual == str(expected), f"{name}: CSV value differs")


class IndependentCalculator:
    """Direct library calculations on expanded row arrays, with explicit ranks."""

    def __init__(self) -> None:
        import numpy as np
        import scipy
        from scipy import stats
        import sklearn
        from sklearn import metrics
        self.np, self.stats, self.sklearn = np, stats, metrics
        self.encoding = np.array([[2, 1], [1, 1], [1, 2]], dtype=np.int64)
        self.versions = {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}

    def optional_correlation(self, function, a, b, **kwargs) -> float | None:
        if len(a) < 2 or not self.np.any(a != a[0]) or not self.np.any(b != b[0]):
            return None
        result = float(function(a, b, **kwargs).statistic)
        return result if math.isfinite(result) else None

    def arrays(self, rows: list[dict[str, Any]]):
        labels = self.np.array([row["label"] for row in rows], dtype=self.np.int64)
        delta = self.np.array([float(row["score_a"]) - float(row["score_b"]) for row in rows], dtype=self.np.float64)
        predicted = self.np.where(self.np.abs(delta) < EPSILON, 0, self.np.where(delta > 0, 1, -1))
        return labels, predicted, delta

    def metrics(self, labels, predicted, delta=None) -> dict[str, Any]:
        np, stats, classification = self.np, self.stats, self.sklearn
        precision, recall, f1, support = classification.precision_recall_fscore_support(
            labels, predicted, labels=LABELS, zero_division=0,
        )
        matrix = classification.confusion_matrix(labels, predicted, labels=LABELS)
        human_ranks = self.encoding[labels + 1].reshape(-1)
        predicted_ranks = self.encoding[predicted + 1].reshape(-1)
        non_tie = labels != 0
        result = {
            "n": len(labels),
            "accuracy": float(classification.accuracy_score(labels, predicted)),
            "f1_weighted": float(np.average(f1, weights=support)),
            "f1_macro": float(np.mean(f1)),
            "f1_micro": float(classification.accuracy_score(labels, predicted)),
            "tie_recall": float(recall[1]) if support[1] else None,
            "patenteval_rank_pair_tau_b": self.optional_correlation(
                stats.kendalltau, human_ranks, predicted_ranks, variant="b",
            ),
            "discrete_spearman": self.optional_correlation(stats.spearmanr, labels, predicted),
            "confusion_matrix": matrix.tolist(),
            "label_order": list(LABELS),
            "non_tie_accuracy": float(np.mean(labels[non_tie] == predicted[non_tie])) if np.any(non_tie) else None,
            "reverse_rate": float(np.mean(predicted[non_tie] == -labels[non_tie])) if np.any(non_tie) else None,
        }
        if delta is not None:
            result["continuous"] = {
                "kendall_tau_b": self.optional_correlation(stats.kendalltau, delta, labels, variant="b"),
                "kendall_tau_c": self.optional_correlation(stats.kendalltau, delta, labels, variant="c"),
                "spearman_rho": self.optional_correlation(stats.spearmanr, delta, labels),
            }
            result["discrete"] = {
                "kendall_tau_b": self.optional_correlation(stats.kendalltau, predicted, labels, variant="b"),
                "kendall_tau_c": self.optional_correlation(stats.kendalltau, predicted, labels, variant="c"),
                "spearman_rho": result["discrete_spearman"],
            }
            for name in ("continuous", "discrete"):
                result[name]["undefined"] = (
                    "constant input" if result[name]["kendall_tau_b"] is None else None
                )
        return result

    def full(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels, predicted, delta = self.arrays(rows)
        result = self.metrics(labels, predicted, delta)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["gold_id"]].append(index)
        group_results = []
        for gold, indices in sorted(groups.items()):
            selected = self.np.array(indices)
            values = self.metrics(labels[selected], predicted[selected], delta[selected])
            group_results.append({
                "gold_id": gold, "n": len(indices),
                "patenteval_rank_pair_tau_b": values["patenteval_rank_pair_tau_b"],
                "continuous_kendall_tau_b": values["continuous"]["kendall_tau_b"],
                "discrete_spearman": values["discrete_spearman"],
            })
        result["n_gold"] = len(groups)
        result["group_macro"] = {
            "n_gold": len(groups), "group_order": sorted(groups), "per_gold": group_results,
            **{key: float(self.np.mean([row[key] if row[key] is not None else 0.0 for row in group_results]))
               for key in GROUP},
        }
        return result

    def bootstrap(
        self, left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]],
        n_resamples: int = 10000, seed: int = 42,
    ) -> dict[str, Any]:
        np = self.np
        labels, left_predicted, _ = self.arrays(left_rows)
        right_labels, right_predicted, _ = self.arrays(right_rows)
        require(np.array_equal(labels, right_labels), "Bootstrap labels are not paired")
        gold_order = sorted({row["gold_id"] for row in left_rows})
        indices = [np.array([index for index, row in enumerate(left_rows) if row["gold_id"] == gold])
                   for gold in gold_order]
        left_groups = {row["gold_id"]: row for row in self.full(left_rows)["group_macro"]["per_gold"]}
        right_groups = {row["gold_id"]: row for row in self.full(right_rows)["group_macro"]["per_gold"]}
        group_differences = np.array([
            [(right_groups[gold][key] if right_groups[gold][key] is not None else 0.0)
             - (left_groups[gold][key] if left_groups[gold][key] is not None else 0.0) for key in GROUP]
            for gold in gold_order
        ])
        samples = {key: [] for key in BOOTSTRAP_KEYS}
        generator = random.Random(seed)
        draw_hash = hashlib.sha256()
        for _ in range(n_resamples):
            draw = generator.choices(range(len(gold_order)), k=len(gold_order))
            draw_hash.update(bytes(draw))
            expanded = np.concatenate([indices[index] for index in draw])
            # All rows, including repeated group multiplicities, are actually
            # expanded before the independent library calls below.
            a = self.metrics(labels[expanded], left_predicted[expanded])
            b = self.metrics(labels[expanded], right_predicted[expanded])
            for key in PRIMARY + ["f1_macro", "tie_recall"]:
                if a[key] is not None and b[key] is not None:
                    samples[key].append(b[key] - a[key])
            diagnostics = np.mean(group_differences[draw], axis=0)
            for key, value in zip(GROUP, diagnostics):
                samples[f"group_macro_{key}"].append(float(value))
        return {
            "n_resamples": n_resamples, "seed": seed, "group_order": gold_order,
            "draws_sha256": draw_hash.hexdigest(),
            "metrics": {
                key: {
                    "ci95": np.quantile(values, [.025, .975], method="linear").tolist() if values else None,
                    "valid_resamples": len(values), "undefined_resamples": n_resamples - len(values),
                    "valid_fraction": len(values) / n_resamples,
                    "fraction_bootstrap_positive": sum(value > 0 for value in values) / len(values) if values else None,
                } for key, values in samples.items()
            },
        }


def _check_prediction_rows(rows, original, name: str, audit: Audit) -> None:
    require(isinstance(rows, list) and len(rows) == 184, f"{name}: exactly 184 rows required")
    for index, (row, expected) in enumerate(zip(rows, original)):
        for key in ("comparison_id", "gold_id", "source_split", "source_row_index", "label"):
            require(row.get(key) == expected[key], f"{name}[{index}].{key}: source/order mismatch")
        require(type(row["source_row_index"]) is int and type(row["label"]) is int, f"{name}: source indices/labels must be integers")
        for key in ("gold_claim", "A", "B"):
            if key in row:
                require(row[key] == expected[key], f"{name}: source text/direction differs")
        a, b = finite(row["score_a"], "score_a"), finite(row["score_b"], "score_b")
        require(0 <= a <= 1 and 0 <= b <= 1, f"{name}: scores outside [0,1]")
        audit.number(a - b, row["delta"], f"{name}[{index}].delta")
        require(abs((a - b) - row["delta"]) <= 1e-12, f"{name}: delta inconsistent with scores")
        predicted = 0 if abs(a - b) < EPSILON else (1 if a > b else -1)
        require(type(row.get("prediction")) is int and row["prediction"] == predicted,
                f"{name}: strict tie/sign rule differs")
        if "threshold" in row:
            require(row["threshold"] == EPSILON, f"{name}: threshold was fitted/changed")


def _check_metrics(expected, recorded, name: str, audit: Audit, *, paper_keys: bool) -> None:
    keys = ["n", "n_gold", "accuracy", "f1_weighted", "f1_macro", "f1_micro", "tie_recall"]
    if paper_keys:
        keys += ["patenteval_rank_pair_tau_b", "discrete_spearman", "non_tie_accuracy", "reverse_rate"]
    for key in keys:
        audit.number(expected[key], recorded[key], f"{name}.{key}")
    require(recorded["confusion_matrix"] == expected["confusion_matrix"], f"{name}: confusion matrix differs")
    require(recorded["label_order"] == LABELS, f"{name}: confusion label order differs")
    if "continuous" in recorded:
        for key in recorded["continuous"]:
            if key == "undefined":
                require(recorded["continuous"][key] == expected["continuous"][key], f"{name}: undefined annotation differs")
            else:
                audit.number(expected["continuous"][key], recorded["continuous"][key], f"{name}.continuous.{key}")
    if "discrete" in recorded:
        for key in recorded["discrete"]:
            if key == "undefined":
                require(recorded["discrete"][key] == expected["discrete"][key], f"{name}: undefined annotation differs")
            else:
                audit.number(expected["discrete"][key], recorded["discrete"][key], f"{name}.discrete.{key}")
    if "discrete_kendall_tau_b" in recorded:
        audit.number(expected["discrete"]["kendall_tau_b"], recorded["discrete_kendall_tau_b"], f"{name}.discrete_tau")
    if "group_macro" in recorded:
        group = recorded["group_macro"]
        require(group["group_order"] == expected["group_macro"]["group_order"], f"{name}: group order differs")
        audit.number(expected["n_gold"], group["n_gold"], f"{name}.group_macro.n_gold")
        for key in GROUP:
            audit.number(expected["group_macro"][key], group[key], f"{name}.group_macro.{key}")
            undefined = [row["gold_id"] for row in expected["group_macro"]["per_gold"] if row[key] is None]
            require(group["zero_assigned_gold_ids"][key] == undefined, f"{name}: undefined group diagnostics differ")
        require(len(group["per_gold"]) == 20, f"{name}: 20 per-gold diagnostics required")
        for left, right in zip(expected["group_macro"]["per_gold"], group["per_gold"]):
            require(left["gold_id"] == right["gold_id"] and left["n"] == right["n"], f"{name}: group identity differs")
            for key in GROUP:
                audit.number(left[key], right[key], f"{name}.per_gold.{left['gold_id']}.{key}")


def _verify_source(output, baseline, data_dir, track: Callable[[Path], Path], audit: Audit):
    config = read_json(track(output / "experiment_config.json"))
    old_config = read_json(track(baseline / "experiment_config.json"))
    lock_path = track(output / "baseline_lock.json")
    lock = read_json(lock_path)
    require(digest(lock_path) == config["baseline_experiment"]["lock_sha256"], "Baseline lock hash differs")
    raw = {}
    for filename, expected_hash in SOURCE_HASHES.items():
        path = track(data_dir / filename)
        require(digest(path) == expected_hash == lock["raw_source_sha256"][filename], "Frozen raw source SHA256 differs")
        raw[filename] = read_json(path)
    require(len(raw["quality_train.json"]) == 1044 and len(raw["quality_test.json"]) == 184, "Raw source counts differ")
    for model, current in (("longformer", old_config), ("modernbert", config)):
        require((current["base_model"]["model_id"], current["base_model"]["revision"]) == MODELS[model], f"{model}: base checkpoint differs")
        training = current["training"]
        require(training["seeds"] == [42, 43] and training["primary_seed"] == 42, "Prespecified seed roles differ")
        require(training["epochs"] == 10 and training["batch_size_comparisons"] == 4
                and training["precision"] == "float32", "Fixed epochs/FP32/true-batch protocol differs")
        require(training["primary_checkpoint"] == "final epoch 10", "Final-epoch primary checkpoint changed")
    for key, value in old_config["training"].items():
        if key != "global_attention_mask":
            require(config["training"][key] == value, f"Shared training setting differs: {key}")
    require(config["evaluation"]["epsilon"] == EPSILON and config["evaluation"]["secondary_seed"] == 43,
            "Tie threshold/secondary seed changed")
    require(config["evaluation"]["primary_metrics"] == PRIMARY, "Primary metric vector differs")
    require(config["evaluation"]["no_test_set_tuning"] is True
            and config["evaluation"]["no_selecting_seed_or_epoch_by_test"] is True, "Test-selection prohibition missing")
    require(config["evaluation"]["bootstrap"]["replicates"] == 10000
            and config["evaluation"]["bootstrap"]["seed"] == 42, "Bootstrap protocol differs")
    require(digest(baseline / "experiment_config.json") == config["baseline_experiment"]["config_sha256"], "Locked Longformer config differs")
    for filename in ("dataset.json", "split_manifest.json"):
        new_path, old_path = track(output / filename), track(baseline / filename)
        require(digest(new_path) == digest(old_path), f"Models used different {filename}")
        if filename in lock["files"]:
            require(digest(old_path) == lock["files"][filename], f"Baseline lock differs for {filename}")
    dataset = read_json(output / "dataset.json")
    expected_dataset = []
    for split in ("train", "test"):
        for index, row in enumerate(raw[f"quality_{split}.json"]):
            expected_dataset.append({
                "source_split": split, "source_row_index": index, "comparison_id": f"{split}_{index:06d}",
                "gold_id": hashlib.sha256(row["gold_claim"].encode("utf-8")).hexdigest(),
                "gold_claim": row["gold_claim"], "A": row["A"], "B": row["B"],
                "label": row["human_eval"]["Quality"],
            })
    require(dataset == expected_dataset, "Dataset text/order/labels/duplicate identities differ from original source")
    original_test = expected_dataset[1044:]
    require(len({row["gold_id"] for row in original_test}) == 20, "Expected 20 reference groups")
    split = read_json(output / "split_manifest.json")
    require(len(split["folds"]) == 1 and split["folds"][0]["fold"] == 0, "Expected one original split")
    indices = list(range(1044))
    random.Random(42).shuffle(indices)
    expected_ids = {
        "validation": [f"train_{i:06d}" for i in indices[:105]],
        "train": [f"train_{i:06d}" for i in indices[105:]],
        "test": [f"test_{i:06d}" for i in range(184)],
    }
    for name, identifiers in expected_ids.items():
        require(split["folds"][0][name]["comparison_ids"] == identifiers, f"{name}: fixed original split/order differs")
    training_golds = {row["gold_id"] for row in expected_dataset[:1044]}
    original_overlap = sum(row["gold_id"] in training_golds for row in original_test)
    require(original_overlap == 184, "Published-split overlap audit differs")
    fp_path = track(output / "source_fingerprint.json")
    fingerprint = read_json(fp_path)
    for name, expected_hash in fingerprint["files"].items():
        require(Path(name).name == name, "Unexpected fingerprint path")
        path = output / name if name in ("experiment_config.json", "dataset.json", "split_manifest.json", "baseline_lock.json") else HERE / name
        track(path)
        require(digest(path) == expected_hash, f"Frozen ModernBERT source/input differs: {name}")
    computed_fp = hashlib.sha256(json.dumps(fingerprint["files"], sort_keys=True).encode()).hexdigest()
    require(fingerprint["fingerprint"] == computed_fp, "ModernBERT fingerprint differs")
    require(read_json(track(baseline / "source_fingerprint.json"))["fingerprint"] == config["baseline_experiment"]["fingerprint"], "Longformer fingerprint differs")
    gate = read_json(track(output / "preflight.json"))
    require(gate["status"] == "passed" and gate["fingerprint"] == computed_fp
            and gate["precision"] == "float32" and gate["batch_comparisons"] == 4
            and gate["batch_sequences"] == 8, "ModernBERT FP32 batch-four gate missing")
    replay = read_json(track(output / "longformer_replay.json"))
    require(replay["status"] == "passed" and len(replay["checkpoints"]) == 4
            and all(row["passed"] and row["class_predictions_identical"]
                    and row["maximum_absolute_score_difference"] <= 1e-6 for row in replay["checkpoints"]),
            "Locked Longformer SDK replay did not pass")
    return config, old_config, original_test, computed_fp, lock


def verify(
    output_dir: str | Path, baseline_dir: str | Path, data_dir: str | Path,
    *, progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output, baseline, data_dir = (Path(value).resolve() for value in (output_dir, baseline_dir, data_dir))
    audit = Audit()
    tracked: dict[str, str] = {}

    def track(path: Path) -> Path:
        key = str(path.resolve())
        actual = digest(path)
        if key in tracked:
            require(tracked[key] == actual, f"Source changed during verification: {key}")
        else:
            tracked[key] = actual
        return path

    calculator = IndependentCalculator()
    config, old_config, original_test, fingerprint, lock = _verify_source(output, baseline, data_dir, track, audit)
    aggregate = read_json(track(output / "comparison.json"))
    require(aggregate["status"] == "complete" and aggregate["schema_version"] == 1, "Completed formal comparison is required")
    require(aggregate["fingerprint"] == fingerprint and aggregate["settings"] == config, "Report settings/fingerprint differ")
    require(aggregate["primary_comparison"] == "seed42 final epoch10; ModernBERT minus fixed Longformer", "Primary model/seed/checkpoint changed")
    require(aggregate["no_test_set_tuning"] is True, "Report must preserve fixed test protocol")
    for name, expected in aggregate["evaluation_source_sha256"].items():
        require(name in ("report.py", "compare_metrics.py") and digest(track(HERE / name)) == expected,
                "Reported evaluation source differs")
    require([row["condition"] for row in aggregate["rows"]] == CONDITIONS, "Eight conditions must keep their fixed order")
    computed, raw_predictions, expected_csv, all_csv_rows = {}, {}, [], []
    for model, folder, cfg, fp in (
        ("longformer", baseline, old_config, config["baseline_experiment"]["fingerprint"]),
        ("modernbert", output, config, fingerprint),
    ):
        for seed in (42, 43):
            result = read_json(track(folder / f"seed_{seed}" / "result.json"))
            require(result["status"] == "complete" and result["seed"] == seed and result["fingerprint"] == fp
                    and result["primary"] is (seed == 42), "Training result/model/seed identity differs")
            require(result["assumptions"] == cfg["assumptions_not_disclosed_by_authors"], "Training assumptions differ")
            history = result["history"]
            require([row["epoch"] for row in history] == list(range(1, 11)), "Exactly ten completed epochs required")
            require(all(finite(row["validation_loss"], "validation loss") >= 0 for row in history), "Invalid validation loss")
            best_epoch = min(history, key=lambda row: row["validation_loss"])["epoch"]
            for selection in ("final", "validation_best"):
                condition = f"{model}_seed{seed}_{selection}"
                details = result["checkpoints"][selection]
                metadata = details["metadata"]
                epoch = 10 if selection == "final" else best_epoch
                require(metadata["seed"] == seed and metadata["epoch"] == epoch and metadata["fingerprint"] == fp
                        and metadata["settings"] == cfg["training"], f"{condition}: checkpoint metadata differs")
                require(metadata["selection"] == ("prespecified final epoch" if selection == "final" else "minimum validation loss"),
                        f"{condition}: checkpoint selected outside fixed rule")
                saved_metadata = read_json(track(folder / f"seed_{seed}" / selection / "reproduction_metadata.json"))
                require(saved_metadata == metadata, f"{condition}: checkpoint metadata differs from result")
                predictions = read_json(track(folder / f"seed_{seed}" / f"{selection}_test_predictions.json"))
                _check_prediction_rows(predictions, original_test, condition, audit)
                values = calculator.full(predictions)
                _check_metrics(values, details["metrics"], f"{condition}.result.metrics", audit, paper_keys=False)
                reported = aggregate["rows"][CONDITIONS.index(condition)]
                role = "primary" if seed == 42 and selection == "final" else "prespecified supplement"
                identity = {"condition": condition, "model": model, "seed": seed,
                            "checkpoint": selection, "epoch": epoch, "role": role,
                            "train_seconds": result["elapsed_seconds"], "peak_vram_bytes": result["peak_vram_bytes"]}
                for key, value in identity.items():
                    if isinstance(value, float):
                        audit.number(value, reported[key], f"{condition}.report.{key}")
                    else:
                        require(reported[key] == value, f"{condition}.report.{key}: differs")
                _check_metrics(values, reported["metrics"], f"{condition}.report.metrics", audit, paper_keys=True)
                computed[condition], raw_predictions[condition] = values, predictions
                expected_csv.append({**identity, **{key: values[key] for key in PRIMARY + ["f1_macro", "tie_recall"]}})
                all_csv_rows.extend({"condition": condition, **row} for row in predictions)
    with track(output / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.DictReader(handle))
    require(len(table) == 8, "Comparison CSV must contain all eight conditions")
    for expected, row in zip(expected_csv, table):
        require(set(row) == set(expected), "Comparison CSV columns differ")
        for key, value in expected.items():
            audit.csv_value(value, row[key], f"comparison.csv.{expected['condition']}.{key}")
    with track(output / "all_test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, table = list(reader.fieldnames or []), list(reader)
    require(len(table) == 8 * 184, "Prediction CSV must contain exactly 1472 original condition-rows")
    expected_fields = list(dict.fromkeys(key for row in all_csv_rows for key in row))
    require(fields == expected_fields, "Prediction CSV source/metadata columns differ")
    for index, (expected, row) in enumerate(zip(all_csv_rows, table)):
        for key in fields:
            audit.csv_value(expected.get(key), row[key], f"all_test_predictions.csv[{index}].{key}")
    bootstrap_checks = {}
    for seed in (42, 43):
        a, b = f"longformer_seed{seed}_final", f"modernbert_seed{seed}_final"
        reported = aggregate["comparisons"][str(seed)]
        require(reported["schema_version"] == 1 and reported["difference_direction"] == "modernbert_minus_longformer"
                and reported["n"] == 184 and reported["n_gold"] == 20 and reported["primary_metric_names"] == PRIMARY,
                "Comparison identity/difference direction differs")
        _check_metrics(computed[a], reported["longformer"], f"pair{seed}.longformer", audit, paper_keys=True)
        _check_metrics(computed[b], reported["modernbert"], f"pair{seed}.modernbert", audit, paper_keys=True)
        require(reported["source_check"]["status"] == "passed"
                and reported["source_check"]["source_sha256"] == SOURCE_HASHES["quality_test.json"]
                and reported["source_check"]["original_order_preserved"] is True, "Comparison source audit differs")
        bootstrap_report = reported["bootstrap"]
        require(bootstrap_report["n_resamples"] == 10000 and bootstrap_report["seed"] == 42
                and bootstrap_report["n_gold"] == 20 and bootstrap_report["unit"] == "gold_id"
                and bootstrap_report["minimum_valid_fraction"] == .95, "Bootstrap settings differ")
        started = time.perf_counter()
        independent = calculator.bootstrap(raw_predictions[a], raw_predictions[b], n_resamples=10000, seed=42)
        require(bootstrap_report["group_order"] == independent["group_order"]
                and bootstrap_report["draws_sha256"] == independent["draws_sha256"], "Bootstrap pairing/order/random draws differ")
        require(set(bootstrap_report["metrics"]) == set(BOOTSTRAP_KEYS), "Bootstrap metric vector differs")
        differences = {}
        for key in BOOTSTRAP_KEYS:
            if key.startswith("group_macro_"):
                name = key.removeprefix("group_macro_")
                left, right = computed[a]["group_macro"][name], computed[b]["group_macro"][name]
            else:
                left, right = computed[a][key], computed[b][key]
            point = right - left if left is not None and right is not None else None
            audit.number(point, reported["difference"][key], f"pair{seed}.point.{key}")
            actual, reference = bootstrap_report["metrics"][key], independent["metrics"][key]
            audit.number(point, actual["point_difference"], f"pair{seed}.bootstrap.{key}.point")
            for field in ("valid_resamples", "undefined_resamples"):
                require(actual[field] == reference[field], f"pair{seed}.{key}.{field}: differs")
            audit.number(reference["valid_fraction"], actual["valid_fraction"], f"pair{seed}.{key}.valid_fraction")
            audit.number(reference["fraction_bootstrap_positive"], actual["fraction_bootstrap_positive"], f"pair{seed}.{key}.fraction_positive")
            if reference["ci95"] is None:
                require(actual["ci95"] is None, f"pair{seed}.{key}: undefined interval must remain null")
            else:
                require(isinstance(actual["ci95"], list) and len(actual["ci95"]) == 2, "CI must have two endpoints")
                for index, value in enumerate(reference["ci95"]):
                    audit.number(value, actual["ci95"][index], f"pair{seed}.{key}.ci95[{index}]")
            eligible = reference["valid_fraction"] >= .95 and point is not None
            require(actual["eligible_for_directional_interpretation"] is eligible, "Undefined/95%-valid interpretation gate differs")
            if point is None:
                interpretation = "point_estimate_undefined"
            elif not eligible:
                interpretation = "insufficient_defined_resamples"
            elif reference["ci95"][0] > 0:
                interpretation = "positive_in_this_fixed_seed_comparison"
            elif reference["ci95"][1] < 0:
                interpretation = "negative_in_this_fixed_seed_comparison"
            else:
                interpretation = "interval_includes_zero"
            require(actual["interpretation"] == interpretation, "Bootstrap direction interpretation differs")
            differences[key] = {"point_difference": point, **reference, "eligible_for_directional_interpretation": eligible}
        bootstrap_checks[str(seed)] = {
            "status": "passed", "n_resamples": 10000, "training_seeds_pooled": False,
            "elapsed_seconds": time.perf_counter() - started, "metrics": differences,
            "draws_sha256": independent["draws_sha256"],
        }
        if progress is not None:
            progress({"stage": "independent_full_bootstrap", "seed": seed,
                      "status": "passed", "elapsed_seconds": bootstrap_checks[str(seed)]["elapsed_seconds"]})
    for path, expected in tracked.items():
        require(digest(Path(path)) == expected, f"Source artifact changed during verification: {path}")
    return {
        "status": "passed", "independent_implementation": True, "paper_protocol_reproduced": False,
        "checked_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "absolute_tolerance": TOLERANCE, "numeric_values_checked": audit.numeric_values_checked,
        "maximum_absolute_difference": audit.maximum_absolute_difference,
        "maximum_difference_location": audit.maximum_difference_location,
        "reference_libraries": calculator.versions, "python_version": sys.version,
        "verifier_sha256": digest(Path(__file__)), "source_sha256": tracked,
        "raw_source_sha256": dict(SOURCE_HASHES), "fingerprint": fingerprint,
        "n_conditions": 8, "n_test_rows_per_condition": 184, "prediction_csv_rows_verified": 1472,
        "primary_comparison": "seed42 final epoch10 ModernBERT minus fixed Longformer",
        "bootstrap_verification": bootstrap_checks,
        "checks": [
            "raw_source_sha256_and_exact_original_text_labels_order_duplicate_identities",
            "same_original_939_105_184_split_and_shared_training_parameters",
            "fixed_base_revisions_seed42_primary_seed43_separate_sensitivity",
            "all_eight_prespecified_checkpoints_final_epoch10_and_earliest_validation_loss_minimum",
            "strict_abs_delta_lt_1e4_and_positive_A_better_direction",
            "scipy_expanded_rank_pair_tau_and_discrete_spearman",
            "sklearn_accuracy_weighted_macro_micro_f1_tie_recall_confusion",
            "all_eight_individual_results_report_rows_and_both_csv_sources",
            "two_separate_10000_draw_whole_group_expanded_row_bootstraps",
            "all_nine_metric_intervals_linear_percentiles_valid_undefined_counts_and_95_percent_gate",
            "original_split_content_overlap_is_not_unseen_reference_generalization",
        ],
        "scope_note": (
            "Complete scalar-library recomputation of all 20,000 paired group-bootstrap draws; no production closed-form imports. "
            "Input/code/metadata/prediction hashes are checked here. Large model-weight blobs are not rehashed; "
            "their complete byte integrity is verified by report.collect. Fixed-seed group intervals do not establish cross-seed stability."
        ),
        "conditions": [{"condition": name, "computed_metrics": computed[name]} for name in CONDITIONS],
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = verify(
            arguments.output_dir, arguments.baseline_dir, arguments.data_dir,
            progress=lambda event: print(json.dumps(event), flush=True),
        )
    except Exception as exc:
        versions = {}
        for name in ("numpy", "scipy", "scikit-learn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        result = {
            "status": "failed", "independent_implementation": True, "paper_protocol_reproduced": False,
            "errors": [f"{type(exc).__name__}: {exc}"], "reference_libraries": versions,
            "absolute_tolerance": TOLERANCE,
            "checked_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        }
    _write(arguments.output_dir / "independent_verification.json", result)
    print(json.dumps({key: result.get(key) for key in (
        "status", "numeric_values_checked", "maximum_absolute_difference", "errors",
    )}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
