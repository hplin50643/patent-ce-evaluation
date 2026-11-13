#!/usr/bin/env python3
"""Regenerate the published split and independently verify the saved scores on CPU."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LONGFORMER = Path("outputs/paper-reproduction/paper_longformer_original_20261108")
MODERNBERT = Path("outputs/paper-modernbert/paper_modernbert_original_20261109")
DATA = ROOT / "data/Patent-CE"
BASELINE_PROGRAM = ROOT / "public-programs/paper-reproduction/prepare_original.py"
VERIFIER = ROOT / "public-programs/paper-modernbert/independent_verify.py"


def stage_run(run_path, temporary_root, extra_files=()):
    relative_paths = ["experiment_config.json", "source_fingerprint.json", *extra_files]
    for seed in (42, 43):
        seed_path = Path(f"seed_{seed}")
        relative_paths.append(seed_path / "result.json")
        for selection in ("final", "validation_best"):
            relative_paths.extend((
                seed_path / f"{selection}_test_predictions.json",
                seed_path / selection / "reproduction_metadata.json",
            ))

    target = temporary_root / run_path
    for name in relative_paths:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / run_path / name, destination)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="Save a copy of the verification record")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="patent-ce-verify-") as temporary:
        temporary_root = Path(temporary).resolve()
        baseline = stage_run(LONGFORMER, temporary_root)
        output = stage_run(MODERNBERT, temporary_root, extra_files=(
            "all_test_predictions.csv", "baseline_lock.json", "comparison.csv", "comparison.json",
            "longformer_replay.json", "preflight.json",
        ))
        subprocess.run([sys.executable, str(BASELINE_PROGRAM), "--data-dir", str(DATA),
                        "--output-dir", str(baseline)], check=True)
        split_bytes = (baseline / "split_manifest.json").read_bytes()
        for run_path in (LONGFORMER, MODERNBERT):
            if split_bytes != (ROOT / run_path / "split_manifest.json").read_bytes():
                raise ValueError(f"Regenerated split differs from {run_path / 'split_manifest.json'}")
        shutil.copyfile(baseline / "dataset.json", output / "dataset.json")
        shutil.copyfile(baseline / "split_manifest.json", output / "split_manifest.json")

        subprocess.run([sys.executable, str(VERIFIER), "--output-dir", str(output),
                        "--baseline-dir", str(baseline), "--data-dir", str(DATA)], check=True)
        if arguments.json_out is not None:
            verification = json.loads((output / "independent_verification.json").read_text(encoding="utf-8"))
            hashes = {}
            for name, value in verification["source_sha256"].items():
                path = Path(name)
                origin = temporary_root if path.is_relative_to(temporary_root) else ROOT
                hashes[path.relative_to(origin).as_posix()] = value
            verification["source_sha256"] = hashes
            arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
            arguments.json_out.write_bytes((json.dumps(verification, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
