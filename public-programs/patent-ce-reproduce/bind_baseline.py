"""Stage the frozen ModernBERT code against a newly trained Longformer control."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


PROGRAMS = Path(__file__).resolve().parents[1]
TRAINING_SOURCE = ("run_reproduction.py", "author_modeling.py", "protocol.py", "prepare_original.py")
SELECTIONS = {"final": "prespecified final epoch", "validation_best": "minimum validation loss"}
HISTORICAL_REPORT_TEXT = {
    "上一輪 Longformer 的 Accuracy 高於論文，但 F1 與 Spearman 較低；這不代表已全面超越或精確重現論文。":
        "論文的數字列於主表供對照，本次的差值以重新訓練的 Longformer 為基準。",
    "四項主要指標提高，不代表平手類別也改善。":
        "勝負與平手分開計算，以呈現模型在兩類比較上的表現。",
    "本輪ModernBERT對勝負排序表現較強，對平手的召回仍偏低，應與主要數字一起解讀。":
        "各模型與種子的實際判對數列於表中。",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def baseline_hashes(baseline, historical_lock):
    hashes = {}
    for name in historical_lock["files"]:
        path = (baseline / name).resolve(strict=True)
        require(path.is_relative_to(baseline) and path.is_file(), f"Invalid Longformer artifact: {name}")
        require(path.stat().st_size > 0, f"Empty Longformer artifact: {name}")
        hashes[name] = sha256(path)
    return hashes


def validate_predictions(predictions, original_test, threshold):
    require(len(predictions) == len(original_test) == 184, "Expected the complete original 184-row test set")
    for prediction, source in zip(predictions, original_test):
        for key in ("comparison_id", "gold_id", "source_split", "source_row_index", "label"):
            require(prediction[key] == source[key], f"Longformer test identity differs: {key}")
        scores = (prediction["score_a"], prediction["score_b"])
        require(all(math.isfinite(score) and 0 <= score <= 1 for score in scores), "Invalid Longformer score")
        delta = scores[0] - scores[1]
        expected = 0 if abs(delta) < threshold else (1 if delta > 0 else -1)
        require(math.isclose(delta, prediction["delta"], abs_tol=1e-12, rel_tol=0)
                and prediction["prediction"] == expected, "Longformer prediction violates the fixed threshold")


def verify_baseline(baseline, data, lock, modern_config):
    hashes = baseline_hashes(baseline, lock)
    for name, expected in lock["raw_source_sha256"].items():
        require(sha256(data / name) == expected, f"Original Patent-CE source differs: {name}")

    # These inputs were fixed independently of the fitted weights and predictions.
    for name in ("experiment_config.json", "dataset.json", "split_manifest.json"):
        require(hashes[name] == lock["files"][name], f"Original split or settings differ: {name}")
    expected_files = {name: hashes[name] for name in ("experiment_config.json", "dataset.json", "split_manifest.json")}
    expected_files.update({name: sha256(PROGRAMS / "paper-reproduction" / name) for name in TRAINING_SOURCE})
    fingerprint = hashlib.sha256(json.dumps(expected_files, sort_keys=True).encode()).hexdigest()
    require(fingerprint == modern_config["baseline_experiment"]["fingerprint"], "Published Longformer training source differs")
    require(read_json(baseline / "source_fingerprint.json") == {"fingerprint": fingerprint, "files": expected_files},
            "Longformer run used different source or settings")

    config = read_json(baseline / "experiment_config.json")
    test_rows = [row for row in read_json(baseline / "dataset.json") if row["source_split"] == "test"]
    for seed in config["training"]["seeds"]:
        prefix = f"seed_{seed}"
        result = read_json(baseline / prefix / "result.json")
        require(result["status"] == "complete" and result["seed"] == seed and result["fingerprint"] == fingerprint
                and result["primary"] is (seed == config["training"]["primary_seed"]), f"Invalid completed Longformer seed {seed}")
        require(result["assumptions"] == config["assumptions_not_disclosed_by_authors"], f"Longformer seed {seed} assumptions differ")
        history = result["history"]
        epochs = config["training"]["epochs"]
        require([entry["epoch"] for entry in history] == list(range(1, epochs + 1))
                and all(math.isfinite(entry["validation_loss"]) and entry["validation_loss"] >= 0 for entry in history),
                f"Longformer seed {seed} has incomplete epoch history")
        best_epoch = min(history, key=lambda entry: entry["validation_loss"])["epoch"]
        for selection, policy in SELECTIONS.items():
            checkpoint = f"{prefix}/{selection}"
            details = result["checkpoints"][selection]
            metadata = dict(seed=seed, epoch=epochs if selection == "final" else best_epoch,
                            selection=policy, fingerprint=fingerprint, settings=config["training"])
            require(details["metadata"] == read_json(baseline / checkpoint / "reproduction_metadata.json") == metadata,
                    f"Longformer checkpoint selection or settings differ: {checkpoint}")
            actual_files = {name.removeprefix(checkpoint + "/"): value for name, value in hashes.items()
                            if name.startswith(checkpoint + "/")}
            require(details["files"] == actual_files, f"Longformer checkpoint hashes differ: {checkpoint}")
            validate_predictions(read_json(baseline / prefix / f"{selection}_test_predictions.json"),
                                 test_rows, config["evaluation"]["epsilon"])
    return fingerprint, hashes


def update_report_text(path):
    text = path.read_text(encoding="utf-8")
    for previous, current in HISTORICAL_REPORT_TEXT.items():
        require(text.count(previous) == 1, "Published report wording changed; review the fresh-run wording")
        text = text.replace(previous, current)
    path.write_bytes(text.encode("utf-8"))


def bind(baseline_dir, data_dir, output_code_dir):
    baseline = baseline_dir.resolve(strict=True)
    data = data_dir.resolve(strict=True)
    target = output_code_dir.resolve()
    source = PROGRAMS / "paper-modernbert"
    require(not target.exists(), f"Choose a new staging directory: {target}")
    require(not target.is_relative_to(source), "Place the staged copy outside the published ModernBERT source directory")
    lock_path = source / "baseline_lock.json"
    config = read_json(source / "experiment_config.json")
    require(sha256(lock_path) == config["baseline_experiment"]["lock_sha256"], "Published baseline lock differs from its config")
    lock = read_json(lock_path)
    fingerprint, hashes = verify_baseline(baseline, data, lock, config)
    lock.update(baseline_run_id=baseline.name, files=hashes)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".modernbert-stage-", dir=target.parent) as temporary:
        staged = Path(temporary) / "code"
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        update_report_text(staged / "report.py")
        write_json(staged / "baseline_lock.json", lock)
        config["baseline_experiment"].update(
            run_id=baseline.name, fingerprint=fingerprint, config_sha256=hashes["experiment_config.json"],
            lock_sha256=sha256(staged / "baseline_lock.json"),
            role="newly retrained paired-seed Longformer control with the published settings and split",
        )
        write_json(staged / "experiment_config.json", config)
        staged.rename(target)
    return {"status": "bound", "baseline_run_id": baseline.name, "baseline_fingerprint": fingerprint,
            "locked_files": len(hashes), "staged_code": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-code-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = bind(args.baseline_dir, args.data_dir, args.output_code_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Longformer baseline binding failed: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
