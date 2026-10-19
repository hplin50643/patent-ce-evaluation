#!/usr/bin/env python3
"""Score raw Patent-CE pairs with the fixed local PatClaimEval snapshot.

This worker preserves every source character. The official 2,048-token result
and a 4,096-token sensitivity result are both computed for every unique pair.
Successful pair results are fsynced to a resumable, provenance-bound JSONL log.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKER_VERSION = "1.0.0"
MODEL_REPOSITORY = "lj408/PatClaimEval-Quality"
MODEL_REVISION = "c959e676f7eb61e32b0f8d9c4cc7d72a4aeb94d5"
EXPECTED_MODEL_HASHES = {
    "model.safetensors": "90fe3d9634fa8ac8b91e3f4c27a0479110c10c99046278701418955aace1ae49",
    "modeling_contrastive.py": "b1d2385ca371fc77908732cef39eb30411f14930d874c832121df3b48488dd12",
    "config.json": "8aeb826892c29a9b211d53920db5abe5d6baed14b1f6046e80c8ba657a53463d",
    "tokenizer.json": "727009a8214ddfa5af1deedf1006d4d06e8e51e54aa5f03566263d4e19bfcdce",
    "tokenizer_config.json": "18ef73773003fd10846845fa3adc26ce0bc0880de148effa2e684cd80061a0c8",
    "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}
REGRESSION_TARGETS = {
    "5592d76bdf2072d8a9c1cd5aaf0e82b191391ffcc323a0f86fe9b923d8eeb4b1": 0.817566454410553,
    "3013114acdf650d76d915792d6b7d131e6fc8e93b9e276f18c720000d12a6082": 0.13473302125930786,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pair_identifier(gold_claim: str, candidate: str) -> str:
    payload = json.dumps([gold_claim, candidate], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_input(path: Path) -> tuple[str, list[dict[str, str]]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("pairs"), list) or not value["pairs"]:
        raise ValueError("Input must contain a nonempty pairs list")
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value["pairs"]):
        if not isinstance(item, dict):
            raise ValueError(f"Pair {index} must be an object")
        for key in ("pair_id", "gold_claim", "candidate"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Pair {index}: {key} must be a nonempty string")
        pair_id = pair_identifier(item["gold_claim"], item["candidate"])
        if item["pair_id"] != pair_id:
            raise ValueError(f"Pair {index}: pair_id does not match raw source strings")
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id in input: {pair_id}")
        seen.add(pair_id)
        pairs.append({key: item[key] for key in ("pair_id", "gold_claim", "candidate")})
    return hashlib.sha256(raw).hexdigest(), pairs


def inspect_model(model_dir: Path) -> dict[str, Any]:
    for name, expected in EXPECTED_MODEL_HASHES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Fixed model hash mismatch for {name}: {actual} != {expected}")
    revision_path = model_dir / "SOURCE_REVISION.txt"
    if not revision_path.is_file() or f"Pinned revision: {MODEL_REVISION}" not in revision_path.read_text(encoding="utf-8"):
        raise ValueError("Fixed model SOURCE_REVISION.txt mismatch")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "model_dir": str(model_dir),
        "weights_bytes": (model_dir / "model.safetensors").stat().st_size,
        "file_hashes": {path.name: sha256_file(path) for path in sorted(model_dir.iterdir()) if path.is_file()},
    }


def validate_result(result: Any) -> None:
    if not isinstance(result, dict):
        raise ValueError("Checkpoint pair result is not an object")
    for key in ("patclaimeval", "patclaimeval_4096"):
        number = result.get(key)
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"Invalid {key}: {number!r}")
    tokens = result.get("pair_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= 4096:
        raise ValueError(f"Invalid pair_tokens: {tokens!r}")
    if result.get("truncated") is not (tokens > 2048) or result.get("truncated_4096") is not False:
        raise ValueError("Checkpoint truncation metadata is inconsistent")


def read_checkpoint(path: Path, header: dict[str, Any], pair_ids: set[str]) -> tuple[dict[str, dict[str, Any]], int]:
    """Ignore only an incomplete trailing write; reject any other corruption."""
    if not path.exists():
        return {}, 0
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete_length = 0
    records = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index != len(lines) - 1:
                raise ValueError("Incomplete non-final checkpoint line")
            break
        complete_length += len(line)
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid checkpoint line {index + 1}") from exc
    if not records or records[0] != header:
        raise ValueError("Checkpoint input/settings/model/environment signature mismatch")
    scores: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records[1:], start=2):
        if not isinstance(record, dict) or record.get("record_type") != "pair":
            raise ValueError(f"Invalid checkpoint record at line {index}")
        pair_id = record.get("pair_id")
        if pair_id not in pair_ids or pair_id in scores:
            raise ValueError(f"Unknown or duplicate checkpoint pair_id at line {index}: {pair_id}")
        validate_result(record.get("result"))
        scores[pair_id] = record["result"]
    return scores, complete_length


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(canonical_bytes(record) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def regression_checks(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for pair_id, expected in REGRESSION_TARGETS.items():
        if pair_id not in scores:
            continue
        observed = scores[pair_id]["patclaimeval"]
        difference = abs(observed - expected)
        checks[pair_id] = {"expected": expected, "observed": observed, "absolute_difference": difference, "tolerance": 2e-6, "passed": difference <= 2e-6}
        if difference > 2e-6:
            raise ValueError(f"Published first-test regression failed for {pair_id}: {observed} vs {expected}")
    return checks


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = utc_now()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "patclaim_progress.json"
    checkpoint_path = output_dir / "patclaim_checkpoint.jsonl"
    output_path = output_dir / "patclaim_scores.json"
    if input_path in (progress_path, checkpoint_path, output_path):
        raise ValueError("Input path aliases a worker output")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    input_sha256, pairs = load_input(input_path)
    model_dir = Path(__file__).resolve().parents[2] / "models" / "PatClaimEval-Quality"
    model_metadata = inspect_model(model_dir)
    settings = {
        "worker_version": WORKER_VERSION,
        "device": "cpu",
        "threads": args.threads,
        "seed": 42,
        "max_length": 2048,
        "sensitivity_max_length": 4096,
        "score_all_pairs_at_both_lengths": True,
        "padding": "max_length",
        "truncation": True,
        "preprocessing": "none; preserve raw input strings including whitespace",
        "local_files_only": True,
        "model_eval": True,
        "inference_mode": True,
        "deterministic_algorithms": True,
        "dtype": "float32",
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "offline_environment": {key: os.environ[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM")},
    }
    signature_context = {
        "input_sha256": input_sha256,
        "settings": settings,
        "model_revision": MODEL_REVISION,
        "model_file_hashes": model_metadata["file_hashes"],
        "environment": environment,
        "script_sha256": sha256_file(Path(__file__)),
    }
    header = {"record_type": "header", "stage": "patclaim", **signature_context}
    pair_ids = {pair["pair_id"] for pair in pairs}
    scores, complete_length = read_checkpoint(checkpoint_path, header, pair_ids)
    if output_path.exists() and not checkpoint_path.exists():
        raise FileExistsError("A score file exists without its provenance checkpoint; choose a new output directory")
    if checkpoint_path.exists():
        with checkpoint_path.open("r+b") as handle:
            handle.truncate(complete_length)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        with checkpoint_path.open("xb") as handle:
            append_record(handle, header)
    regression_checks(scores)
    initial_count = len(scores)
    progress = {
        "stage": "patclaim", "status": "loading", "input_sha256": input_sha256,
        "total_pairs": len(pairs), "completed_pairs": len(scores), "resumed_pairs": initial_count,
        "started_at_utc": started_at, "updated_at_utc": utc_now(), "settings": settings,
    }
    atomic_json(progress_path, progress)
    print(json.dumps({key: progress[key] for key in ("stage", "status", "total_pairs", "resumed_pairs")}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    token_lengths = {}
    for pair in pairs:
        tokens = len(tokenizer(pair["gold_claim"], pair["candidate"], truncation=False, add_special_tokens=True)["input_ids"])
        if tokens > 4096:
            raise ValueError(f"Pair {pair['pair_id']} has {tokens} tokens, exceeding full-text capacity 4096; no scores skipped")
        token_lengths[pair["pair_id"]] = tokens
        if pair["pair_id"] in scores and scores[pair["pair_id"]]["pair_tokens"] != tokens:
            raise ValueError("Resumed tokenizer length disagrees with checkpoint")
    model = None
    if len(scores) < len(pairs):
        spec = importlib.util.spec_from_file_location("patclaim_fixed_snapshot", model_dir / "modeling_contrastive.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import local model class")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model = module.ContrastiveModel.from_pretrained(model_dir, local_files_only=True).to(torch.device("cpu"))
        model.eval()
    progress["status"] = "running"
    with checkpoint_path.open("ab") as checkpoint:
        for pair in pairs:
            pair_id = pair["pair_id"]
            if pair_id in scores:
                continue
            pair_started = time.perf_counter()
            with torch.inference_mode():
                score_2048 = float(model.score_pair(pair["gold_claim"], pair["candidate"], tokenizer, torch.device("cpu"), max_length=2048))
                seconds_2048 = time.perf_counter() - pair_started
                sensitivity_started = time.perf_counter()
                score_4096 = float(model.score_pair(pair["gold_claim"], pair["candidate"], tokenizer, torch.device("cpu"), max_length=4096))
                seconds_4096 = time.perf_counter() - sensitivity_started
            result = {
                "patclaimeval": score_2048,
                "patclaimeval_4096": score_4096,
                "pair_tokens": token_lengths[pair_id],
                "truncated": token_lengths[pair_id] > 2048,
                "truncated_4096": False,
                "elapsed_seconds_2048": seconds_2048,
                "elapsed_seconds_4096": seconds_4096,
                "completed_at_utc": utc_now(),
            }
            validate_result(result)
            regression_checks({pair_id: result})
            append_record(checkpoint, {"record_type": "pair", "pair_id": pair_id, "result": result})
            scores[pair_id] = result
            progress.update({"completed_pairs": len(scores), "updated_at_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started, "last_pair_id": pair_id})
            atomic_json(progress_path, progress)
            print(json.dumps({"stage": "patclaim", "completed": len(scores), "total": len(pairs), "pair_seconds": seconds_2048 + seconds_4096, "pair_tokens": result["pair_tokens"]}), flush=True)
    if sha256_file(input_path) != input_sha256:
        raise ValueError("Input changed during inference")
    final = {
        "stage": "patclaim", "status": "complete", "input_sha256": input_sha256,
        "input_path": str(input_path), "settings": settings, "environment": environment,
        "model": model_metadata, "script_sha256": signature_context["script_sha256"],
        "started_at_utc": started_at, "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started, "resumed_pairs": initial_count,
        "pair_count": len(pairs), "truncated_unique_pairs": sum(item["truncated"] for item in scores.values()),
        "max_pair_tokens": max(token_lengths.values()), "regression_checks": regression_checks(scores),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scores": {pair["pair_id"]: scores[pair["pair_id"]] for pair in pairs},
    }
    atomic_json(output_path, final)
    progress.update({"status": "complete", "completed_pairs": len(scores), "updated_at_utc": utc_now(), "elapsed_seconds": final["elapsed_seconds"], "output_sha256": sha256_file(output_path)})
    atomic_json(progress_path, progress)
    print(json.dumps({"stage": "patclaim", "status": "complete", "pairs": len(scores), "output": str(output_path)}), flush=True)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        failure = {"stage": "patclaim", "status": "failed", "updated_at_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        try:
            progress_path = args.output_dir.resolve() / "patclaim_progress.json"
            if progress_path.parent.exists() and progress_path != args.input.resolve():
                atomic_json(progress_path, failure)
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
