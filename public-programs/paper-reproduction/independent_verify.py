"""Independent original-split statistics verification; no protocol/report imports.

Reads all six prespecified predictions and their finished reports, recomputes
statistics with SciPy/scikit-learn, and writes one verification JSON. It does
not load models, choose a condition, change predictions or fit a threshold.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

TOLERANCE = 1e-10
EPSILON = 1e-4
LABELS = [-1, 0, 1]
CORE_FILES = ("run_reproduction.py", "author_modeling.py", "protocol.py", "prepare_original.py")
FIXED_SOURCE_HASHES = {"quality_train.json": "20c3e88862b0a8cd3ce6e48607f9887a29c4502d4e87633b41095be194bdca48",
                       "quality_test.json": "10e68f47f2f4329f92c6d60936d23f4eaab6b852150b8fa1b039d8d348868039"}
CONDITIONS = ("author_2048", "author_4096", "seed_42_final", "seed_42_validation_best", "seed_43_final", "seed_43_validation_best")
TARGETS = {"accuracy": .565, "f1_reported_averaging_unspecified": .574, "kendall_tau": .477, "spearman_rho": .602}
REFERENCE_PAPER = "https://aclanthology.org/2024.naacl-long.147.pdf#page=9"
REFERENCE_CODE = "https://github.com/ZoeYou/PatentEval/blob/d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427"
PAPER_VECTOR = ("accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    number = float(value)
    require(math.isfinite(number), f"{name}: nonfinite number")
    return number


def compare(actual, expected, name, differences):
    if isinstance(actual, dict):
        require(isinstance(expected, dict), f"{name}: dictionary required")
        for key, value in actual.items():
            require(key in expected, f"{name}.{key}: missing")
            compare(value, expected[key], f"{name}.{key}", differences)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), f"{name}: list mismatch")
        for index, value in enumerate(actual):
            compare(value, expected[index], f"{name}[{index}]", differences)
    elif isinstance(actual, float):
        error = abs(actual - finite(expected, name))
        require(error <= TOLERANCE, f"{name}: recomputed {actual!r}, recorded {expected!r}")
        differences.append({"path": name, "independent": actual, "recorded": float(expected), "absolute_difference": error})
    else:
        require(actual == expected, f"{name}: independently computed {actual!r}, recorded {expected!r}")


def _data_dir(output, explicit):
    choices = [Path(explicit)] if explicit else [Path("/data"), output.parent / "data", Path(__file__).resolve().parents[2] / "data" / "Patent-CE"]
    for candidate in choices:
        if all((candidate / name).is_file() for name in FIXED_SOURCE_HASHES):
            return candidate
    raise ValueError("Original source files unavailable; provide --data-dir")


def verify_sources(output, data_dir, tracked):
    config, dataset, split, frozen = [read_json(output / name) for name in ("experiment_config.json", "dataset.json", "split_manifest.json", "source_fingerprint.json")]
    tracked.extend(output / name for name in ("experiment_config.json", "dataset.json", "split_manifest.json", "source_fingerprint.json"))
    require(config["training"]["seeds"] == [42, 43] and config["training"]["primary_seed"] == 42 and config["training"]["epochs"] == 10, "Prespecified seed/epoch policy changed")
    require(config["evaluation"]["epsilon"] == EPSILON and config["evaluation"]["author_lengths"] == [2048, 4096], "Prespecified epsilon/author lengths changed")
    require(config["evaluation"]["no_test_set_tuning"] is True and config["evaluation"]["no_selecting_seed_or_epoch_by_test"] is True, "Test-set selection is outside the declared experiment")
    require(config["paper_quality_targets"] == TARGETS, "Paper target values changed")
    expected_hashes = {name: digest(output / name) for name in ("experiment_config.json", "dataset.json", "split_manifest.json")}
    for name in CORE_FILES:
        path = Path(__file__).parent / name
        expected_hashes[name] = digest(path)
        tracked.append(path)
    require(frozen["files"] == expected_hashes, "Frozen configuration/data/split/training-source SHA256 mismatch")
    fingerprint = hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest()
    require(frozen["fingerprint"] == fingerprint, "Frozen fingerprint is invalid")
    require(split["dataset_sha256"] == expected_hashes["dataset.json"], "Prepared dataset differs from split manifest SHA256")
    require(isinstance(dataset, list) and len(dataset) == 1228 and len({row["comparison_id"] for row in dataset}) == 1228, "All 1228 original row identities are required")
    require(len({row["gold_id"] for row in dataset}) == 20, "Expected 20 original reference groups")
    raw_hashes = {}
    for source_split, count, offset in (("train", 1044, 0), ("test", 184, 1044)):
        path = data_dir / f"quality_{source_split}.json"
        raw_hashes[path.name] = digest(path)
        require(raw_hashes[path.name] == FIXED_SOURCE_HASHES[path.name] == split["source_hashes"][path.name], f"{path.name}: original source SHA256 mismatch")
        tracked.append(path)
        raw = read_json(path)
        require(len(raw) == count, "Original source row count changed")
        for index, original in enumerate(raw):
            row = dataset[offset + index]
            require(row["comparison_id"] == f"{source_split}_{index:06d}" and row["source_split"] == source_split and row["source_row_index"] == index, "Original source row order/identity changed")
            require(type(row["label"]) is int and row["label"] in LABELS and row["label"] == original["human_eval"]["Quality"], "Original Quality labels changed")
            require(all(row[key] == original[key] for key in ("gold_claim", "A", "B")), "Original claim text changed")
            require(row["gold_id"] == hashlib.sha256(row["gold_claim"].encode()).hexdigest(), "Gold hash differs from exact reference text")
    require(len(split["folds"]) == 1, "Original experiment requires exactly one split")
    indices = list(range(1044))
    random.Random(42).shuffle(indices)
    expected = {"train": [f"train_{i:06d}" for i in indices[105:]], "validation": [f"train_{i:06d}" for i in indices[:105]], "test": [f"test_{i:06d}" for i in range(184)]}
    for name, ids in expected.items():
        require(split["folds"][0][name]["comparison_ids"] == ids, f"{name}: fixed original split changed")
    return config, dataset[1044:], fingerprint, raw_hashes


def check_predictions(path, original, metadata):
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == 184, f"{path}: all 184 test rows required")
    require([r["comparison_id"] for r in rows] == [r["comparison_id"] for r in original], f"{path}: test IDs/order changed")
    for row, source in zip(rows, original):
        for name in ("comparison_id", "gold_id", "source_split", "source_row_index", "label"):
            require(row[name] == source[name], f"{path.name}: source {name} mismatch")
        require(type(row["label"]) is int and row["label"] in LABELS, "Invalid preference label")
        a, b, delta = (finite(row[key], key) for key in ("score_a", "score_b", "delta"))
        require(0 <= a <= 1 and 0 <= b <= 1, "Candidate scores outside sigmoid range")
        require(abs(delta - (a - b)) <= TOLERANCE, "Delta differs from score_a - score_b")
        prediction = 0 if abs(delta) < EPSILON else (1 if delta > 0 else -1)
        require(type(row.get("prediction")) is int and row["prediction"] == prediction, "Prediction violates strict abs(delta) < 1e-4")
        for name, value in metadata.items():
            require(row.get(name) == value, f"Prediction {name} metadata differs")
    return rows


def recompute(rows, stats, sklearn_metrics):
    labels = [row["label"] for row in rows]
    deltas = [float(row["delta"]) for row in rows]
    predicted = [0 if abs(value) < EPSILON else (1 if value > 0 else -1) for value in deltas]
    def correlation(values):
        if len(set(values)) < 2 or len(set(labels)) < 2:
            return {"kendall_tau_b": None, "kendall_tau_c": None, "spearman_rho": None, "undefined": "constant input"}
        result = {"kendall_tau_b": float(stats.kendalltau(labels, values, variant="b", nan_policy="raise").statistic),
                  "kendall_tau_c": float(stats.kendalltau(labels, values, variant="c", nan_policy="raise").statistic),
                  "spearman_rho": float(stats.spearmanr(labels, values, nan_policy="raise").statistic), "undefined": None}
        require(all(value is None or math.isfinite(value) for value in result.values()), "Undefined/nonfinite SciPy statistic")
        return result
    matrix = sklearn_metrics.confusion_matrix(labels, predicted, labels=LABELS)
    ties = labels.count(0)
    return dict(n=len(rows), n_gold=len({row["gold_id"] for row in rows}),
                accuracy=float(sklearn_metrics.accuracy_score(labels, predicted)),
                f1_weighted=float(sklearn_metrics.f1_score(labels, predicted, labels=LABELS, average="weighted", zero_division=0)),
                f1_macro=float(sklearn_metrics.f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0)),
                f1_micro=float(sklearn_metrics.f1_score(labels, predicted, labels=LABELS, average="micro", zero_division=0)),
                confusion_matrix=matrix.tolist(), label_order=LABELS, tie_recall=float(matrix[1, 1] / ties) if ties else None,
                continuous=correlation(deltas), discrete=correlation(predicted), epsilon=EPSILON,
                tie_policy="abs(delta) < epsilon", aggregation="global original rows, duplicates retained")


def independent_rank_pair_tau(predicted, labels, stats):
    """Independently implement the cited rank-pair encoding, without helper imports."""
    require(len(predicted) == len(labels), "Rank-pair vectors must have equal length")
    encoding = {1: (1, 2), -1: (2, 1), 0: (1, 1)}
    require(all(value in encoding for value in [*predicted, *labels]), "Invalid rank-pair preference")
    predicted_flat = [rank for value in predicted for rank in encoding[value]]
    human_flat = [rank for value in labels for rank in encoding[value]]
    if not predicted_flat or len(set(predicted_flat)) < 2 or len(set(human_flat)) < 2:
        return None
    return finite(stats.kendalltau(predicted_flat, human_flat, variant="b", nan_policy="raise").statistic, "rank-pair tau-b")


def paper_compatible(rows, metrics, stats):
    predicted = [0 if abs(row["delta"]) < EPSILON else (1 if row["delta"] > 0 else -1) for row in rows]
    return dict(accuracy=metrics["accuracy"], f1_weighted=metrics["f1_weighted"],
                patenteval_rank_pair_tau_b=independent_rank_pair_tau(predicted, [row["label"] for row in rows], stats),
                discrete_spearman=metrics["discrete"]["spearman_rho"])


def report_numeric_matches(values):
    pairs = {"accuracy": (values["accuracy"], TARGETS["accuracy"]),
             "f1_weighted": (values["f1_weighted"], TARGETS["f1_reported_averaging_unspecified"]),
             "patenteval_rank_pair_tau_b": (values["patenteval_rank_pair_tau_b"], TARGETS["kendall_tau"]),
             "discrete_spearman": (values["discrete_spearman"], TARGETS["spearman_rho"])}
    return {key: value is not None and abs(value - target) < .0005 + 1e-12 for key, (value, target) in pairs.items()}


def _csv_expected(condition, metrics, comparison):
    result = {key: condition[key] for key in ("condition", "role", "seed", "checkpoint", "epoch", "max_length")}
    result.update({f"paper_compatible_{key}": value for key, value in comparison.items()})
    result.update({key: metrics[key] for key in ("n", "accuracy", "f1_weighted", "f1_macro", "f1_micro", "tie_recall")})
    for mode in ("continuous", "discrete"):
        for key in ("kendall_tau_b", "kendall_tau_c", "spearman_rho"):
            result[f"{mode}_{key}"] = metrics[mode][key]
    result.update({f"report_match_{key}": value for key, value in report_numeric_matches(comparison).items()})
    return result


def verify(output_dir, data_dir=None):
    import numpy
    import scipy
    import scipy.stats as stats
    import sklearn
    import sklearn.metrics as sklearn_metrics

    output = Path(output_dir).resolve()
    tracked, differences = [], []
    raw_dir = _data_dir(output, data_dir)
    initial_paths = [output / name for name in ("experiment_config.json", "dataset.json", "split_manifest.json", "source_fingerprint.json", "reproduction_results.json", "comparison.csv")]
    initial_paths += [Path(__file__).parent / name for name in CORE_FILES]
    evaluation_paths = [Path(__file__).parent / name for name in ("report.py", "reference_tau.py")]
    initial_paths += evaluation_paths
    initial_paths += [raw_dir / name for name in FIXED_SOURCE_HASHES]
    for length in (2048, 4096):
        initial_paths.extend((output / "author" / f"result_{length}.json", output / "author" / f"predictions_{length}.json"))
    for seed in (42, 43):
        initial_paths.extend(output / f"seed_{seed}" / name for name in ("result.json", "final_test_predictions.json", "validation_best_test_predictions.json"))
    revision_path = output / "evaluation_protocol_revision.json"
    if revision_path.is_file():
        initial_paths.append(revision_path)
    initial_hashes = {str(path.resolve()): digest(path) for path in initial_paths}
    config, original, fingerprint, raw_hashes = verify_sources(output, raw_dir, tracked)
    aggregate_path, csv_path = output / "reproduction_results.json", output / "comparison.csv"
    aggregate = read_json(aggregate_path)
    require(aggregate["status"] == "complete" and aggregate["execution_verification"] == "passed", "Finished reproduction report is required")
    require(aggregate.get("schema_version") == 2, "Reference-aligned report schema version 2 is required")
    require(aggregate["paper_protocol_reproduced"] is False, "Numerical audit must not claim recovery of undisclosed author protocol")
    require(aggregate["fingerprint"] == fingerprint and aggregate["raw_source_sha256"] == raw_hashes, "Report fingerprint/source hashes differ")
    require(aggregate["primary_condition"] == "seed_42_final", "Primary condition changed after test")
    require([row["condition"] for row in aggregate["rows"]] == list(CONDITIONS), "All six conditions must retain prespecified presentation order")
    require(aggregate["settings"] == config, "Report settings differ from frozen configuration")
    require(aggregate["evaluation_protocol"]["vector"] == list(PAPER_VECTOR), "Paper comparison must use the fixed four-metric vector")
    require(aggregate["evaluation_protocol"]["rank_encoding"] == {"1": [1, 2], "-1": [2, 1], "0": [1, 1]}, "Rank-pair encoding differs from the cited reference")
    require(aggregate["evaluation_protocol"]["reference_paper"] == REFERENCE_PAPER and aggregate["evaluation_protocol"]["reference_code"] == REFERENCE_CODE, "Pinned primary reference sources differ")
    require(aggregate["evaluation_code_sha256"] == {path.name: digest(path) for path in evaluation_paths}, "Evaluation-layer source SHA256 differs")
    tracked.extend(evaluation_paths)
    if revision_path.is_file():
        require(aggregate["evaluation_protocol_revision"] == read_json(revision_path), "Evaluation revision record differs")
        tracked.append(revision_path)
    else:
        require(aggregate["evaluation_protocol_revision"] is None, "Declared evaluation revision record is missing locally")
    tracked.extend((aggregate_path, csv_path))
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_fields, csv_rows = list(reader.fieldnames or ()), list(reader)
    require([row["condition"] for row in csv_rows] == list(CONDITIONS), "CSV conditions missing, duplicated or reordered")
    recovered = []
    for position, identifier in enumerate(CONDITIONS):
        if identifier.startswith("author_"):
            length = int(identifier.split("_")[1])
            result_path = output / "author" / f"result_{length}.json"
            predictions_path = output / "author" / f"predictions_{length}.json"
            result = read_json(result_path)
            require(result["status"] == "complete" and result["fingerprint"] == fingerprint, "Author result incomplete or fingerprint mismatch")
            require(result["model"] == config["author_checkpoint"] and result["max_length"] == length, "Author model/length differs")
            require(result["precision"] == "float32" and result["padding"] == "max_length" and result["model_eval"] is True and result["test_used_for_training_or_selection"] is False, "Author inference protocol differs")
            require(result["inference_code_sha256"] == digest(Path(__file__).parent / "author_modeling.py"), "Author inference source SHA256 differs")
            recorded_metrics = result["metrics"]
            condition = dict(condition=identifier, role="released_checkpoint_replay", seed=None, checkpoint="released_author", epoch=None, max_length=length)
            metadata = {"max_length": length, "checkpoint": "released_author"}
        else:
            _, seed_text, checkpoint = identifier.split("_", 2)
            seed = int(seed_text)
            result_path = output / f"seed_{seed}" / "result.json"
            predictions_path = output / f"seed_{seed}" / f"{checkpoint}_test_predictions.json"
            result = read_json(result_path)
            require(result["status"] == "complete" and result["fingerprint"] == fingerprint and result["seed"] == seed and result["primary"] is (seed == 42), "Seed result identity/fingerprint differs")
            require(result["assumptions"] == config["assumptions_not_disclosed_by_authors"], "Training assumptions differ")
            history = result["history"]
            require([entry["epoch"] for entry in history] == list(range(1, 11)), "Ten complete training epochs are required")
            require(all(finite(entry["validation_loss"], "validation loss") >= 0 for entry in history), "Invalid validation loss")
            epoch = 10 if checkpoint == "final" else min(history, key=lambda entry: entry["validation_loss"])["epoch"]
            details = result["checkpoints"][checkpoint]
            saved = details["metadata"]
            selection = "prespecified final epoch" if checkpoint == "final" else "minimum validation loss"
            require(saved["seed"] == seed and saved["epoch"] == epoch and saved["selection"] == selection and saved["fingerprint"] == fingerprint and saved["settings"] == config["training"], "Checkpoint selection/settings differ")
            recorded_metrics = details["metrics"]
            role = "primary" if seed == 42 and checkpoint == "final" else ("prespecified_validation_supplement" if seed == 42 else "prespecified_seed_sensitivity")
            condition = dict(condition=identifier, role=role, seed=seed, checkpoint=checkpoint, epoch=epoch, max_length=config["training"]["max_length"])
            metadata = {}
        tracked.extend((result_path, predictions_path))
        rows = check_predictions(predictions_path, original, metadata)
        metrics = recompute(rows, stats, sklearn_metrics)
        comparison = paper_compatible(rows, metrics, stats)
        compare(metrics, recorded_metrics, f"{identifier}.result.metrics", differences)
        compare(condition, aggregate["rows"][position], f"{identifier}.report.identity", differences)
        compare(metrics, aggregate["rows"][position]["metrics"], f"{identifier}.report.metrics", differences)
        compare(comparison, aggregate["rows"][position]["paper_compatible"], f"{identifier}.report.paper_compatible", differences)
        compare(report_numeric_matches(comparison), aggregate["rows"][position]["report_numeric_matches"], f"{identifier}.report.matches", differences)
        expected_csv = _csv_expected(condition, metrics, comparison)
        require(csv_fields == list(expected_csv), "CSV columns differ from complete fixed comparison schema")
        for name, value in expected_csv.items():
            recorded = csv_rows[position][name]
            if value is None:
                require(recorded == "", f"{identifier}.csv.{name}: null must remain empty")
            elif isinstance(value, bool):
                require(recorded == str(value), f"{identifier}.csv.{name}: boolean mismatch")
            elif isinstance(value, float):
                compare(value, finite(recorded, name), f"{identifier}.csv.{name}", differences)
            else:
                require(str(value) == recorded, f"{identifier}.csv.{name}: identity/integer mismatch")
        recovered.append(dict(**condition, metrics=metrics, paper_compatible=comparison, report_numeric_matches=report_numeric_matches(comparison)))
    sources = {str(path.resolve()): digest(path) for path in sorted(set(tracked))}
    require(sources == initial_hashes, "Source artifacts changed during verification")
    return dict(status="passed", independent_implementation=True, paper_protocol_reproduced=False,
                absolute_tolerance=TOLERANCE, numeric_values_checked=len(differences),
                maximum_absolute_difference=max((entry["absolute_difference"] for entry in differences), default=0.),
                reference_libraries={"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
                python_version=sys.version, verifier_sha256=digest(__file__), source_sha256=sources,
                raw_source_sha256=raw_hashes, fingerprint=fingerprint, n_conditions=6, n_test_rows_per_condition=184,
                checks=["all_six_prespecified_conditions", "exact_original_184_test_ids_order_labels_and_claim_sources", "original_source_and_dataset_hashes",
                        "frozen_configuration_split_training_source_fingerprint", "strict_epsilon_0.0001", "finite_candidate_scores_and_deltas",
                        "global_scipy_continuous_and_discrete_tau_b_tau_c_spearman", "sklearn_accuracy_weighted_macro_micro_f1_tie_recall_confusion",
                        "independent_patenteval_rank_pair_tau_b_and_fixed_paper_vector", "evaluation_only_revision_and_source_hashes",
                        "all_individual_result_metrics", "reproduction_results_metrics_and_fixed_roles", "comparison_csv_all_columns",
                        "prespecified_seed42_final_primary_and_validation_loss_minima"],
                scope_note="Numerical and input/code provenance verification. Model weight-byte integrity is covered by the separate report verifier; this helper does not load or rehash large weights.",
                conditions=recovered, numeric_checks=differences)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = verify(arguments.output_dir, arguments.data_dir)
    except Exception as exc:
        versions = {}
        for name in ("numpy", "scipy", "scikit-learn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        result = {"status": "failed", "paper_protocol_reproduced": False,
                  "errors": [f"{type(exc).__name__}: {exc}"], "reference_libraries": versions, "absolute_tolerance": TOLERANCE}
    result["checked_at_taipei"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    target = arguments.output_dir / "independent_verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("numeric_checks", "conditions", "source_sha256")}, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
