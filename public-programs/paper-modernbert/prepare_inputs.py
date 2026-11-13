"""Copy the completed Longformer experiment's exact split, verifying frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import os


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def immutable_copy(source, target):
    data = source.read_bytes()
    if target.exists() and target.read_bytes() != data:
        raise ValueError(f"Immutable input differs: {target.name}")
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        temporary.replace(target)


def prepare(baseline, data_dir, output, config_path):
    output.mkdir(parents=True, exist_ok=True)
    config = read(config_path)
    lock_path = Path(__file__).parent / "baseline_lock.json"
    if digest(lock_path) != config["baseline_experiment"]["lock_sha256"]:
        raise ValueError("Baseline lock file changed")
    lock = read(lock_path)
    for name, expected in lock["files"].items():
        path = (baseline / name).resolve()
        if not path.is_relative_to(baseline.resolve()) or digest(path) != expected:
            raise ValueError(f"Longformer baseline artifact differs: {name}")
    for name, expected in lock["raw_source_sha256"].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"Raw public data differs: {name}")
    if read(baseline / "source_fingerprint.json")["fingerprint"] != config["baseline_experiment"]["fingerprint"]:
        raise ValueError("Longformer training fingerprint differs")
    for name in ("dataset.json", "split_manifest.json"):
        immutable_copy(baseline / name, output / name)
    immutable_copy(lock_path, output / "baseline_lock.json")
    immutable_copy(config_path, output / "experiment_config.json")
    rows = read(output / "dataset.json")
    if len(rows) != 1228 or sum(r["source_split"] == "test" for r in rows) != 184:
        raise ValueError("Original public row count changed")
    for split in ("train", "test"):
        raw = read(data_dir / f"quality_{split}.json")
        prepared = [row for row in rows if row["source_split"] == split]
        if len(raw) != len(prepared):
            raise ValueError("Original source count changed")
        for i, (source, row) in enumerate(zip(raw, prepared)):
            if any(source[key] != row[key] for key in ("gold_claim", "A", "B")):
                raise ValueError("Raw claim text/order changed")
            if source["human_eval"]["Quality"] != row["label"] or type(row["label"]) is not int:
                raise ValueError("Raw Quality label changed")
            if row["source_row_index"] != i or row["comparison_id"] != f"{split}_{i:06d}":
                raise ValueError("Original comparison identity changed")
            if row["gold_id"] != hashlib.sha256(row["gold_claim"].encode()).hexdigest():
                raise ValueError("Reference grouping changed")
    manifest = read(output / "split_manifest.json")
    sizes = {name: len(manifest["folds"][0][name]["comparison_ids"]) for name in ("train", "validation", "test")}
    if sizes != dict(train=939, validation=105, test=184):
        raise ValueError("Fixed split changed")
    result = dict(status="passed", baseline_run=lock["baseline_run_id"], baseline_files_verified=len(lock["files"]),
                  dataset_sha256=digest(output / "dataset.json"), split_sha256=digest(output / "split_manifest.json"),
                  raw_source_sha256=lock["raw_source_sha256"], counts=sizes,
                  method="byte-identical copy of completed original-split Longformer experiment; duplicates and overlap retained")
    (output / "input_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline_dir, args.data_dir, args.output_dir, args.config)))
