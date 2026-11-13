"""Preserve Patent-CE's published split for a paper-reproduction experiment.

Every original record, including duplicates, retains its source order, text and
label. Only a 105-row validation subset is drawn from the original training set.
The seed, Python shuffle and ceil rounding are this experiment's defaults: the
paper reports a 10% validation subset but does not disclose these details.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any


SEED = 42
VALIDATION_FRACTION = 0.1
EXPECTED_ROWS = {"train": 1044, "test": 184}
EXPECTED_GOLD_COUNT = 20
FIXED_SOURCE_HASHES = {
    "quality_train.json": "20c3e88862b0a8cd3ce6e48607f9887a29c4502d4e87633b41095be194bdca48",
    "quality_test.json": "10e68f47f2f4329f92c6d60936d23f4eaab6b852150b8fa1b039d8d348868039",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _expected_hashes(data_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = data_dir / "source_manifest.json"
    if not path.exists():
        return dict(FIXED_SOURCE_HASHES), {
            "method": "fixed approved upstream SHA256 values",
            "source_manifest_sha256": None,
        }
    payload = path.read_bytes()
    original = json.loads(payload.decode("utf-8-sig"))
    expected = {
        item["file"]: item["sha256"] for item in original["downloads"]
        if item.get("file") in FIXED_SOURCE_HASHES
    }
    if set(expected) != set(FIXED_SOURCE_HASHES):
        raise ValueError("source_manifest.json must identify both original data-file hashes")
    return expected, {
        "method": "checked against source_manifest.json downloads",
        "source_manifest_sha256": _sha256(payload),
        "repository": original.get("repository"),
        "commit": original.get("commit"),
        "license": original.get("license"),
        "paper": original.get("paper"),
    }


def _validate_row(original: Any, location: str) -> int:
    if not isinstance(original, dict):
        raise ValueError(f"{location}: expected an object")
    for field in ("gold_claim", "A", "B"):
        if not isinstance(original.get(field), str) or not original[field].strip():
            raise ValueError(f"{location}: {field} must be a nonempty string")
    evaluation = original.get("human_eval")
    label = evaluation.get("Quality") if isinstance(evaluation, dict) else None
    if type(label) is not int or label not in (-1, 0, 1):
        raise ValueError(f"{location}: Quality must be an integer in -1, 0, 1")
    return label


def _exact_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return row["gold_claim"], row["A"], row["B"], row["label"]


def _canonical_key(row: dict[str, Any]) -> tuple[str, str, str]:
    low, high = sorted((row["A"], row["B"]))
    return row["gold_claim"], low, high


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    return {
        "comparisons": len(rows),
        "gold_groups": len({row["gold_id"] for row in rows}),
        "unique_candidates": len({row[key] for row in rows for key in ("A", "B")}),
        "labels": {str(label): labels[label] for label in (-1, 0, 1)},
    }


def _duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_labels: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        canonical_labels[_canonical_key(row)].add(
            row["label"] if row["A"] <= row["B"] else -row["label"],
        )
    unique_exact = len({_exact_key(row) for row in rows})
    return {
        "rows": len(rows),
        "exact_unique_comparisons": unique_exact,
        "exact_duplicate_rows_preserved": len(rows) - unique_exact,
        "canonical_unique_comparisons": len(canonical_labels),
        "canonical_duplicate_rows_preserved": len(rows) - len(canonical_labels),
        "canonical_comparisons_with_conflicting_labels": sum(
            len(labels) > 1 for labels in canonical_labels.values()
        ),
        "identical_candidate_rows_preserved": sum(row["A"] == row["B"] for row in rows),
    }


def _overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    gold_left, gold_right = ({row["gold_id"] for row in rows} for rows in (left, right))
    candidate_left, candidate_right = (
        {row[key] for row in rows for key in ("A", "B")} for rows in (left, right)
    )
    pair_left, pair_right = (
        {(row["gold_claim"], row[key]) for row in rows for key in ("A", "B")}
        for rows in (left, right)
    )
    exact_left, exact_right = ({_exact_key(row) for row in rows} for rows in (left, right))
    canonical_left, canonical_right = (
        {_canonical_key(row) for row in rows} for rows in (left, right)
    )
    ids_left, ids_right = ({row["comparison_id"] for row in rows} for rows in (left, right))
    return {
        "row_ids_shared": len(ids_left & ids_right),
        "gold_groups_shared": len(gold_left & gold_right),
        "left_rows_gold_seen_in_right": sum(row["gold_id"] in gold_right for row in left),
        "right_rows_gold_seen_in_left": sum(row["gold_id"] in gold_left for row in right),
        "candidate_strings_shared": len(candidate_left & candidate_right),
        "reference_candidate_pairs_shared": len(pair_left & pair_right),
        "right_pair_instances_seen_in_left": sum(
            (row["gold_claim"], row[key]) in pair_left for row in right for key in ("A", "B")
        ),
        "exact_comparisons_shared": len(exact_left & exact_right),
        "right_exact_comparison_rows_seen_in_left": sum(_exact_key(row) in exact_left for row in right),
        "canonical_comparisons_shared": len(canonical_left & canonical_right),
        "right_canonical_comparison_rows_seen_in_left": sum(
            _canonical_key(row) in canonical_left for row in right
        ),
    }


def prepare(data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write preserved dataset and one original-split fold; return the manifest.

    dataset.json keeps source train rows followed by source test rows. Split ID
    lists for train/validation follow the shuffled index order; the test list
    follows the exact original test order. Content overlaps are reported rather
    than rejected, because they belong to the published split being reproduced.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    expected_hashes, source_verification = _expected_hashes(data_dir)
    source_files: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    dataset: list[dict[str, Any]] = []
    for source_split in ("train", "test"):
        filename = f"quality_{source_split}.json"
        payload = (data_dir / filename).read_bytes()
        actual_hash = _sha256(payload)
        if actual_hash != expected_hashes[filename]:
            raise ValueError(f"Source hash mismatch: {filename}")
        original_rows = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(original_rows, list) or len(original_rows) != EXPECTED_ROWS[source_split]:
            raise ValueError(f"{filename}: expected {EXPECTED_ROWS[source_split]} original rows")
        source_files[source_split] = {
            "file": filename, "sha256": actual_hash, "rows": len(original_rows),
            "bytes": len(payload), "verified_sha256": True,
        }
        sources[source_split] = []
        for index, original in enumerate(original_rows):
            label = _validate_row(original, f"{filename}:{index}")
            row = {
                "source_split": source_split,
                "source_row_index": index,
                "comparison_id": f"{source_split}_{index:06d}",
                "gold_id": _sha256(original["gold_claim"].encode("utf-8")),
                "gold_claim": original["gold_claim"],
                "A": original["A"],
                "B": original["B"],
                "label": label,
            }
            dataset.append(row)
            sources[source_split].append(row)
    if len({row["gold_id"] for row in dataset}) != EXPECTED_GOLD_COUNT:
        raise ValueError(f"Expected {EXPECTED_GOLD_COUNT} original gold reference groups")

    indices = list(range(EXPECTED_ROWS["train"]))
    random.Random(SEED).shuffle(indices)
    validation_n = math.ceil(VALIDATION_FRACTION * len(indices))
    selected = {
        "train": [sources["train"][index] for index in indices[validation_n:]],
        "validation": [sources["train"][index] for index in indices[:validation_n]],
        "test": list(sources["test"]),
    }
    fold: dict[str, Any] = {"fold": 0}
    for name, rows in selected.items():
        fold[name] = {
            "gold_ids": sorted({row["gold_id"] for row in rows}),
            "comparison_ids": [row["comparison_id"] for row in rows],
            "counts": _counts(rows),
        }
    overlap = {
        f"{left}_{right}": _overlap(selected[left], selected[right])
        for left, right in itertools.combinations(("train", "validation", "test"), 2)
    }
    if any(item["row_ids_shared"] for item in overlap.values()):
        raise ValueError("Source row IDs must be disjoint between train/validation/test")
    assigned_ids = [identifier for name in selected for identifier in fold[name]["comparison_ids"]]
    if len(assigned_ids) != len(dataset) or len(set(assigned_ids)) != len(dataset):
        raise ValueError("Original rows must each be assigned exactly once")
    fold["audit"] = {
        "row_ids_disjoint": True,
        "every_original_row_assigned_once": True,
        "content_overlap_policy": "Record published content overlaps without rejecting or removing them",
        "overlap": overlap,
    }
    dataset_payload = _json_bytes(dataset)
    manifest = {
        "schema_version": 1,
        "experiment": "original-paper-split-reproduction",
        "seed": SEED,
        "source_files": source_files,
        "source_hashes": {entry["file"]: entry["sha256"] for entry in source_files.values()},
        "source_verification": source_verification,
        "dataset_file": "dataset.json",
        "dataset_sha256": _sha256(dataset_payload),
        "source_row_index_base": 0,
        "comparison_id_policy": "Source split plus six-digit zero-based row index; duplicates retain unique IDs",
        "text_policy": "Exact original strings, labels and train-then-test order; no normalization or deduplication",
        "gold_id_policy": "SHA256 of the exact original UTF-8 gold_claim string",
        "validation_selection": {
            "fraction_of_original_train": VALIDATION_FRACTION,
            "seed": SEED,
            "rounding": "ceil",
            "validation_rows": validation_n,
            "training_rows": len(indices) - validation_n,
            "algorithm": "Python random.Random(42).shuffle(indices); first 105 validation, remaining 939 training",
            "split_list_order": "Shuffled index order for train/validation; unchanged source order for test",
            "paper_reported": "10% of training samples used for validation",
            "author_undisclosed_defaults_chosen_here": [
                "random seed 42", "ceil rounding", "Python random shuffle", "split-list ordering",
            ],
        },
        "counts": _counts(dataset),
        "folds": [fold],
        "audit": {
            "original_train": _duplicate_audit(sources["train"]),
            "original_test": _duplicate_audit(sources["test"]),
            "combined": _duplicate_audit(dataset),
            "original_train_test_overlap": _overlap(sources["train"], sources["test"]),
            "row_ids_disjoint": True,
            "original_test_order_unchanged": True,
            "source_order_and_duplicates_preserved": True,
            "content_overlaps_are_reported_not_errors": True,
        },
    }
    _atomic_write(out_dir / "dataset.json", dataset_payload)
    _atomic_write(out_dir / "split_manifest.json", _json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "Patent-CE",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = prepare(arguments.data_dir, arguments.output_dir)
    print(json.dumps({
        "comparisons": manifest["counts"]["comparisons"],
        "gold_groups": manifest["counts"]["gold_groups"],
        "split_rows": {name: manifest["folds"][0][name]["counts"]["comparisons"]
                       for name in ("train", "validation", "test")},
        "dataset_sha256": manifest["dataset_sha256"],
    }))


if __name__ == "__main__":
    main()
