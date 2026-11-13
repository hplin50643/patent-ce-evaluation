"""Independent, fixed-threshold comparison on the 184 original Patent-CE rows.

Public API
==========
score_metrics(rows) -> dict
    The four flat primary keys are ``accuracy``, ``f1_weighted``,
    ``patenteval_rank_pair_tau_b`` and ``discrete_spearman``. Additional keys
    include f1_macro, f1_micro, tie_recall, confusion_matrix, continuous and
    group_macro. group_macro contains fixed-gold-order per_gold records and
    equal-weight diagnostics; it never substitutes for the global primary metrics.
    This function supports arbitrary nonempty diagnostic fixtures.

compare_pair(lf_rows, mb_rows, bootstrap=10000, seed=42, *, data_dir=None) -> dict
    Requires both complete original test files in their original 184-row order,
    including source_split/source_row_index, IDs, labels and gold hashes. The
    data_dir argument, PATENT_CE_DATA_DIR environment variable, or repository
    data/Patent-CE directory supplies the hash-verified original source.
    Returns schema_version=1; longformer/modernbert hold score_metrics outputs;
    difference holds ModernBERT minus Longformer point estimates; bootstrap
    contains n_resamples, seed, n_gold, group_order and metrics. Each bootstrap
    metric holds point_difference, ci95, valid_resamples, undefined_resamples,
    valid_fraction, eligible_for_directional_interpretation and interpretation.
    source_check records the authoritative row audit; limitations records scope.

One call compares one seed/checkpoint pair. Calls never pool seeds or choose a
checkpoint. The classifier is strictly abs(score_a-score_b) < 1e-4 for a tie.

PatentEval encodes +1/-1/0 as [1,2]/[2,1]/[1,1], flattens the rank pairs, then
computes tau-b. Its binary rank contingency permits an exact closed-form phi
coefficient. Discrete Spearman is Pearson correlation of marginal midranks.
Both can therefore be recomputed from a preference confusion matrix on each
whole-reference-group bootstrap draw, without repeatedly calling sklearn/SciPy.
The production implementation imports neither the old protocol nor reference_tau.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import random
from typing import Any, Iterable


EPSILON = 1e-4
LABELS = (-1, 0, 1)
PRIMARY_METRICS = (
    "accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman",
)
GROUP_METRICS = (
    "patenteval_rank_pair_tau_b", "continuous_kendall_tau_b", "discrete_spearman",
)
COMPARISON_METRICS = PRIMARY_METRICS + ("f1_macro", "tie_recall") + tuple(
    f"group_macro_{name}" for name in GROUP_METRICS
)
EXPECTED_TEST_SHA256 = "10e68f47f2f4329f92c6d60936d23f4eaab6b852150b8fa1b039d8d348868039"
REFERENCE_IMPLEMENTATION_URL = (
    "https://github.com/ZoeYou/PatentEval/blob/"
    "d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427"
)
MINIMUM_VALID_FRACTION = 0.95


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def preference(delta: float) -> int:
    delta = _finite(delta, "delta")
    return 0 if abs(delta) < EPSILON else (1 if delta > 0 else -1)


def _validated(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(rows)
    if not records:
        raise ValueError("Prediction rows must not be empty")
    seen = set()
    normalized = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a dictionary")
        identifier, gold = row.get("comparison_id"), row.get("gold_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Each row requires a nonempty comparison_id")
        if identifier in seen:
            raise ValueError(f"Duplicate comparison_id: {identifier}")
        seen.add(identifier)
        if not isinstance(gold, str) or not gold:
            raise ValueError("Each row requires a nonempty gold_id")
        label = row.get("label")
        if type(label) is not int or label not in LABELS:
            raise ValueError("label must be an integer in {-1,0,1}")
        a, b = _finite(row.get("score_a"), "score_a"), _finite(row.get("score_b"), "score_b")
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("Scores must be in [0,1]")
        delta = _finite(row.get("delta"), "delta")
        if not math.isclose(delta, a - b, rel_tol=0, abs_tol=1e-12):
            raise ValueError("delta must equal score_a - score_b (positive means A is better)")
        predicted = preference(delta)
        if "prediction" in row and (type(row["prediction"]) is not int or row["prediction"] != predicted):
            raise ValueError("Stored prediction disagrees with strict <1e-4 preference")
        if "threshold" in row and _finite(row["threshold"], "threshold") != EPSILON:
            raise ValueError("A fitted/different threshold is not allowed")
        normalized.append(dict(row, score_a=a, score_b=b, delta=delta, prediction=predicted))
    return normalized


def _clamp_correlation(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _pearson(x: list[float], y: list[float]) -> float | None:
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    vx = math.fsum((value - mx) ** 2 for value in x)
    vy = math.fsum((value - my) ** 2 for value in y)
    if not vx or not vy:
        return None
    numerator = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    return _clamp_correlation(numerator / math.sqrt(vx * vy))


def _midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        average = (start + end + 1) / 2.0
        for offset in range(start, end):
            result[order[offset]] = average
        start = end
    return result


def _raw_tau(x: list[float], y: list[float]) -> float | None:
    concordant = discordant = tied_x_only = tied_y_only = 0
    for index, (left_x, left_y) in enumerate(zip(x, y)):
        for right in range(index + 1, len(x)):
            dx, dy = left_x - x[right], left_y - y[right]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x_only += 1
            elif dy == 0:
                tied_y_only += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt((comparable + tied_x_only) * (comparable + tied_y_only))
    return (concordant - discordant) / denominator if denominator else None


def _matrix(rows: list[dict[str, Any]]) -> tuple[int, ...]:
    counts = [0] * 9
    for row in rows:
        counts[(row["label"] + 1) * 3 + row["prediction"] + 1] += 1
    return tuple(counts)


def _rank_pair_tau(m: list[int] | tuple[int, ...]) -> float | None:
    # Counts of flattened (human_rank, predicted_rank), ordered as (1,1),
    # (1,2), (2,1), (2,2). Every original comparison contributes twice.
    a = m[0] + m[1] + m[3] + 2 * m[4] + m[5] + m[7] + m[8]
    b = m[2] + m[3] + m[5] + m[6]
    c = m[1] + m[2] + m[6] + m[7]
    d = m[0] + m[8]
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denominator if denominator else None


def _discrete_spearman(m: list[int] | tuple[int, ...], support: list[int], predicted: list[int]) -> float | None:
    total = sum(support)
    ranks = []
    for marginal in (support, predicted):
        offset = 0
        values = []
        for count in marginal:
            values.append(offset + (count + 1) / 2.0)
            offset += count
        ranks.append(values)
    center = (total + 1) / 2.0
    vx = math.fsum(count * (value - center) ** 2 for count, value in zip(support, ranks[0]))
    vy = math.fsum(count * (value - center) ** 2 for count, value in zip(predicted, ranks[1]))
    if not vx or not vy:
        return None
    covariance = math.fsum(
        m[i * 3 + j] * (ranks[0][i] - center) * (ranks[1][j] - center)
        for i in range(3) for j in range(3)
    )
    return _clamp_correlation(covariance / math.sqrt(vx * vy))


def _matrix_metrics(m: list[int] | tuple[int, ...]) -> dict[str, float | None]:
    support = [sum(m[index:index + 3]) for index in (0, 3, 6)]
    predicted = [m[index] + m[index + 3] + m[index + 6] for index in range(3)]
    total = sum(support)
    if not total:
        raise ValueError("Cannot evaluate an empty confusion matrix")
    f1 = [2 * m[index * 4] / (support[index] + predicted[index])
          if support[index] + predicted[index] else 0.0 for index in range(3)]
    accuracy = (m[0] + m[4] + m[8]) / total
    non_tie = support[0] + support[2]
    return {
        "accuracy": accuracy,
        "f1_weighted": math.fsum(n * value for n, value in zip(support, f1)) / total,
        "patenteval_rank_pair_tau_b": _rank_pair_tau(m),
        "discrete_spearman": _discrete_spearman(m, support, predicted),
        "f1_macro": math.fsum(f1) / 3,
        "f1_micro": accuracy,
        "tie_recall": m[4] / support[1] if support[1] else None,
        "non_tie_accuracy": (m[0] + m[8]) / non_tie if non_tie else None,
        "reverse_rate": (m[2] + m[6]) / non_tie if non_tie else None,
    }


def _score_validated(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _matrix(rows)
    result: dict[str, Any] = _matrix_metrics(matrix)
    labels = [row["label"] for row in rows]
    deltas = [row["delta"] for row in rows]
    predicted = [row["prediction"] for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["gold_id"]].append(row)
    per_gold = []
    zero_assigned: dict[str, list[str]] = {name: [] for name in GROUP_METRICS}
    totals = {name: 0.0 for name in GROUP_METRICS}
    for gold, group in sorted(groups.items()):
        metrics = _matrix_metrics(_matrix(group))
        continuous_tau = _raw_tau(
            [row["delta"] for row in group], [row["label"] for row in group],
        )
        item = {
            "gold_id": gold, "n": len(group),
            "patenteval_rank_pair_tau_b": metrics["patenteval_rank_pair_tau_b"],
            "continuous_kendall_tau_b": continuous_tau,
            "discrete_spearman": metrics["discrete_spearman"],
            "zero_assigned_metrics": [],
        }
        for name in GROUP_METRICS:
            if item[name] is None:
                zero_assigned[name].append(gold)
                item["zero_assigned_metrics"].append(name)
            else:
                totals[name] += item[name]
        per_gold.append(item)
    result.update({
        "n": len(rows), "n_gold": len(groups),
        "confusion_matrix": [list(matrix[index:index + 3]) for index in (0, 3, 6)],
        "label_order": list(LABELS),
        "epsilon": EPSILON,
        "tie_policy": "strict abs(delta) < 1e-4; positive delta means A is better",
        "aggregation": "global original rows; original duplicate identities retained",
        "rank_pair_encoding": {"1": [1, 2], "-1": [2, 1], "0": [1, 1]},
        "reference_implementation": REFERENCE_IMPLEMENTATION_URL,
        "continuous": {
            "kendall_tau_b": _raw_tau(deltas, labels),
            "spearman_rho": _pearson(_midranks(deltas), _midranks(labels)),
        },
        "discrete_kendall_tau_b": _raw_tau(predicted, labels),
        "group_macro": {
            "n_gold": len(groups),
            **{name: totals[name] / len(groups) for name in GROUP_METRICS},
            "group_order": sorted(groups),
            "undefined_policy": "Keep every gold equally weighted; undefined per-gold diagnostics contribute zero",
            "zero_assigned_gold_ids": zero_assigned,
            "per_gold": per_gold,
        },
    })
    return result


def score_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute fixed-threshold primary metrics and explicitly named diagnostics.

    Global undefined correlations remain None. Only equal-weight group-macro
    diagnostics use declared zero contributions for undefined groups. Input row
    order does not change these statistics; compare_pair additionally enforces
    the original source order before using any predictions in a real comparison.
    """
    return _score_validated(_validated(rows))


def _source_reference(data_dir: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if data_dir is None:
        data_dir = os.environ.get("PATENT_CE_DATA_DIR")
    directory = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data" / "Patent-CE"
    source = directory / "quality_test.json"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_TEST_SHA256:
        raise ValueError("Original quality_test.json SHA256 differs from the frozen source")
    originals = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(originals, list) or len(originals) != 184:
        raise ValueError("Exactly 184 original test rows are required")
    expected = []
    for index, row in enumerate(originals):
        label = row["human_eval"]["Quality"]
        if type(label) is not int or label not in LABELS:
            raise ValueError("Invalid original source label")
        expected.append({
            "comparison_id": f"test_{index:06d}", "source_split": "test",
            "source_row_index": index,
            "gold_id": hashlib.sha256(row["gold_claim"].encode("utf-8")).hexdigest(),
            "label": label, "gold_claim": row["gold_claim"], "A": row["A"], "B": row["B"],
        })
    if len({row["gold_id"] for row in expected}) != 20:
        raise ValueError("The frozen original test must contain 20 reference groups")
    return expected, {"source_file": str(source.resolve()), "source_sha256": digest}


def _check_source(rows: list[dict[str, Any]], expected: list[dict[str, Any]], name: str) -> None:
    if len(rows) != 184:
        raise ValueError(f"{name}: exactly 184 original test rows required; do not pool seeds")
    for index, (row, original) in enumerate(zip(rows, expected)):
        if row.get("comparison_id") != original["comparison_id"]:
            raise ValueError(f"{name}: original test row order/ID mismatch at index {index}")
        for key in ("source_split", "source_row_index", "gold_id", "label"):
            if row.get(key) != original[key] or (key == "source_row_index" and type(row.get(key)) is not int):
                raise ValueError(f"{name}: original source {key} mismatch at index {index}")
        for key in ("gold_claim", "A", "B"):
            if key in row and row[key] != original[key]:
                raise ValueError(f"{name}: original source {key} text/direction mismatch at index {index}")


def _points(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        **{name: metrics[name] for name in PRIMARY_METRICS + ("f1_macro", "tie_recall")},
        **{f"group_macro_{name}": metrics["group_macro"][name] for name in GROUP_METRICS},
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return right - left if left is not None and right is not None else None


def _percentile(values: list[float], quantile: float) -> float:
    index = (len(values) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def compare_pair(
    lf_rows: Iterable[dict[str, Any]], mb_rows: Iterable[dict[str, Any]],
    bootstrap: int = 10000, seed: int = 42, *, data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare one paired seed/checkpoint using whole-gold percentile bootstrap.

    The direction is always ModernBERT minus Longformer. Every draw samples 20
    sorted-reference groups with replacement, retains all rows in each selected
    group (including multiplicity), and recomputes global metrics. Undefined
    global differences are omitted metric-by-metric with counts/fractions kept;
    directional CI interpretation requires >=95% defined draws and a defined
    point estimate. Intervals are pointwise/descriptive, conditional on a fixed
    training seed. This routine never interprets original-split content overlap
    as independent generalization and never selects among seeds/checkpoints.
    """
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("bootstrap must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    left, right = _validated(lf_rows), _validated(mb_rows)
    expected, source = _source_reference(data_dir)
    _check_source(left, expected, "Longformer")
    _check_source(right, expected, "ModernBERT")
    metrics_left, metrics_right = _score_validated(left), _score_validated(right)
    points_left, points_right = _points(metrics_left), _points(metrics_right)
    differences = {name: _difference(points_left[name], points_right[name]) for name in COMPARISON_METRICS}
    groups_left: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_right: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_left, row_right in zip(left, right):
        if (row_left["comparison_id"], row_left["gold_id"], row_left["label"]) != (
            row_right["comparison_id"], row_right["gold_id"], row_right["label"],
        ):
            raise ValueError("Paired row identities or label directions differ")
        groups_left[row_left["gold_id"]].append(row_left)
        groups_right[row_right["gold_id"]].append(row_right)
    gold_order = sorted(groups_left)
    group_scores_left = {row["gold_id"]: row for row in metrics_left["group_macro"]["per_gold"]}
    group_scores_right = {row["gold_id"]: row for row in metrics_right["group_macro"]["per_gold"]}
    components = []
    for gold in gold_order:
        group_differences = tuple(
            (group_scores_right[gold][name] or 0.0) - (group_scores_left[gold][name] or 0.0)
            for name in GROUP_METRICS
        )
        components.append((_matrix(groups_left[gold]), _matrix(groups_right[gold]), group_differences))
    samples: dict[str, list[float]] = {name: [] for name in COMPARISON_METRICS}
    rng = random.Random(seed)
    draw_hash = hashlib.sha256()
    for _ in range(bootstrap):
        draw = rng.choices(range(len(gold_order)), k=len(gold_order))
        draw_hash.update(bytes(draw))
        matrix_left, matrix_right = [0] * 9, [0] * 9
        macro_differences = [0.0] * len(GROUP_METRICS)
        for group_index in draw:
            cells_left, cells_right, macro = components[group_index]
            for cell in range(9):
                matrix_left[cell] += cells_left[cell]
                matrix_right[cell] += cells_right[cell]
            for metric_index, value in enumerate(macro):
                macro_differences[metric_index] += value
        draw_left, draw_right = _matrix_metrics(matrix_left), _matrix_metrics(matrix_right)
        for name in PRIMARY_METRICS + ("f1_macro", "tie_recall"):
            difference = _difference(draw_left[name], draw_right[name])
            if difference is not None:
                samples[name].append(difference)
        for name, value in zip(GROUP_METRICS, macro_differences):
            samples[f"group_macro_{name}"].append(value / len(gold_order))
    intervals = {}
    for name, values in samples.items():
        values.sort()
        ci = [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None
        valid_fraction = len(values) / bootstrap
        eligible = valid_fraction >= MINIMUM_VALID_FRACTION and differences[name] is not None
        if differences[name] is None:
            interpretation = "point_estimate_undefined"
        elif not eligible:
            interpretation = "insufficient_defined_resamples"
        elif ci[0] > 0:
            interpretation = "positive_in_this_fixed_seed_comparison"
        elif ci[1] < 0:
            interpretation = "negative_in_this_fixed_seed_comparison"
        else:
            interpretation = "interval_includes_zero"
        intervals[name] = {
            "point_difference": differences[name], "ci95": ci,
            "valid_resamples": len(values), "undefined_resamples": bootstrap - len(values),
            "valid_fraction": valid_fraction,
            "eligible_for_directional_interpretation": eligible,
            "interpretation": interpretation,
            "fraction_bootstrap_positive": sum(value > 0 for value in values) / len(values) if values else None,
        }
    return {
        "schema_version": 1,
        "difference_direction": "modernbert_minus_longformer",
        "primary_metric_names": list(PRIMARY_METRICS),
        "n": len(left), "n_gold": len(gold_order),
        "longformer": metrics_left, "modernbert": metrics_right,
        "difference": differences,
        "bootstrap": {
            "method": "paired whole-reference-group percentile bootstrap; recompute global metrics on every draw",
            "unit": "gold_id", "n_gold": len(gold_order), "group_order": gold_order,
            "n_resamples": bootstrap, "seed": seed, "confidence_level": 0.95,
            "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
            "draws_sha256": draw_hash.hexdigest(),
            "undefined_global_policy": "Metric-wise finite differences only; disclose undefined counts; do not replace by zero",
            "multiple_testing_adjustment": "None; metric-wise intervals are descriptive and pointwise",
            "metrics": intervals,
        },
        "source_check": {
            "status": "passed", **source, "n_original_test": 184,
            "original_order_preserved": True, "same_row_labels_and_gold_ids": True,
            "matched_fields": ["comparison_id", "source_split", "source_row_index", "gold_id", "label"],
            "raw_text_fields_verified_when_present": ["gold_claim", "A", "B"],
        },
        "limitations": [
            "The original split shares all 20 reference groups and candidate content with training; this is not unseen-reference generalization.",
            "Whole-group resampling reflects the 20 observed references at a fixed training seed, not variation across training seeds.",
            "Compare seed42 final with seed42 final; seed43 is a separate matched sensitivity comparison; do not pool them into 368 independent rows.",
            "Validation-best checkpoints are supplementary; this function never selects a model, seed or epoch using test metrics.",
            "Equal-weight group-macro correlations are diagnostics; global primary metrics are recomputed after resampling all selected rows.",
        ],
    }
