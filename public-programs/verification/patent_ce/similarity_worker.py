"""Offline, resumable candidate/reference metrics for Patent-CE.

Run with LLM4DPCG/.venv/Scripts/python.exe. Input text is passed unchanged to
the metrics; BERTScore's own strip and 512-token right truncation are audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[3]
LLM_ROOT = WORKSPACE / "LLM4DPCG"
MODEL_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
SCORE_KEYS = ("bleu4", "rouge1_f1", "rougeL_f1", "bertscore_f1")
TOKEN_KEYS = ("gold_tokens", "candidate_tokens")
TRUNCATION_KEYS = ("gold_truncated", "candidate_truncated")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_id(gold: str, candidate: str) -> str:
    raw = json.dumps([gold, candidate], ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_pairs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("Input must contain a pairs array")
    pairs = payload["pairs"]
    if not pairs:
        raise ValueError("Input pairs cannot be empty")
    seen = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair {index} must be an object")
        gold, candidate = pair.get("gold_claim"), pair.get("candidate")
        if not isinstance(gold, str) or not isinstance(candidate, str):
            raise ValueError(f"Pair {index} texts must be strings")
        if not gold.strip() or not candidate.strip():
            raise ValueError(f"Pair {index} has empty text")
        expected = pair_id(gold, candidate)
        if pair.get("pair_id") != expected:
            raise ValueError(f"Pair {index} raw-text SHA-256 does not match")
        if expected in seen:
            raise ValueError(f"Duplicate pair_id: {expected}")
        seen.add(expected)
    return pairs


def validate_score(score: dict) -> None:
    if not isinstance(score, dict):
        raise ValueError("Score must be an object")
    for key in SCORE_KEYS:
        value = score.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid numeric score: {key}")
        lower = -1.000001 if key == "bertscore_f1" else 0.0
        upper = 1.000001 if key == "bertscore_f1" else 1.0
        if not lower <= value <= upper:
            raise ValueError(f"Score outside expected range: {key}={value}")
    for key in TOKEN_KEYS:
        if type(score.get(key)) is not int or score[key] < 2:
            raise ValueError(f"Invalid token count: {key}")
    for key in TRUNCATION_KEYS:
        if type(score.get(key)) is not bool:
            raise ValueError(f"Invalid truncation flag: {key}")
    if score["gold_truncated"] != (score["gold_tokens"] > 512):
        raise ValueError("gold_truncated does not match token count")
    if score["candidate_truncated"] != (score["candidate_tokens"] > 512):
        raise ValueError("candidate_truncated does not match token count")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("ab") as handle:
        for record in records:
            handle.write(canonical(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_checkpoint(path: Path, signatures: dict, valid_ids: set[str]) -> tuple[dict, str | None]:
    """Validate every complete line; discard only an interrupted final line."""
    scores: dict[str, dict] = {}
    bert_hash = None
    if not path.exists():
        return scores, bert_hash
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    good_bytes = 0
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index != len(lines) - 1:
                raise ValueError("Incomplete checkpoint line before file end")
            break
        record = json.loads(line)
        for key, value in signatures.items():
            if record.get(key) != value:
                raise ValueError(f"Checkpoint {key} mismatch; use a new output directory")
        identifier = record.get("pair_id")
        if identifier not in valid_ids or identifier in scores:
            raise ValueError(f"Unknown or duplicate checkpoint pair: {identifier}")
        score = record.get("score")
        validate_score(score)
        if record.get("score_sha256") != object_hash(score):
            raise ValueError("Checkpoint score checksum mismatch")
        current_hash = record.get("bertscore_hashcode")
        if not isinstance(current_hash, str) or not current_hash:
            raise ValueError("Missing BERTScore hashcode in checkpoint")
        if bert_hash is not None and current_hash != bert_hash:
            raise ValueError("Inconsistent BERTScore hashcode in checkpoint")
        bert_hash = current_hash
        scores[identifier] = score
        good_bytes += len(line)
    if good_bytes != len(data):
        with path.open("r+b") as handle:
            handle.truncate(good_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return scores, bert_hash


def configure_environment(threads: int) -> None:
    os.environ.update({
        "HF_HOME": str(LLM_ROOT / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(LLM_ROOT / ".cache" / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_EVALUATE_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_XET": "1",
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
    })


def metric_file_info(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


class SimilarityEvaluator:
    def __init__(self, threads: int, batch_size: int):
        configure_environment(threads)
        import torch
        import numpy as np
        from evaluate import load
        from transformers import AutoTokenizer
        import bert_score

        torch.set_num_threads(threads)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        self.threads = threads
        self.batch_size = batch_size
        self.hashcode = None
        ref = LLM_ROOT / ".cache" / "huggingface" / "hub" / "models--roberta-large" / "refs" / "main"
        if not ref.is_file() or ref.read_text().strip() != MODEL_REVISION:
            raise ValueError("Offline roberta-large main ref is not the pinned revision")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-large", revision=MODEL_REVISION, use_fast=False, local_files_only=True)
        if self.tokenizer.model_max_length != 512 or self.tokenizer.truncation_side != "right":
            raise ValueError("Expected 512-token right truncation")
        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        versions = {}
        for name in ("torch", "numpy", "evaluate", "bert-score", "transformers", "rouge-score", "datasets"):
            versions[name] = importlib.metadata.version(name)
        self.environment = {
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "bert_score_internal_version": bert_score.__version__,
            "metric_sources": {
                name: metric_file_info(Path(inspect.getfile(type(metric))))
                for name, metric in (("bleu", self.bleu), ("rouge", self.rouge), ("bertscore", self.bertscore))
            },
            "bert_score_utils": metric_file_info(Path(inspect.getfile(bert_score.utils))),
            "worker": metric_file_info(Path(__file__)),
        }
        self._token_counts: dict[str, int] = {}

    def token_count(self, text: str) -> int:
        # Mirrors bert_score.utils.sent_encode, except truncation is disabled.
        text = text.strip()
        if text not in self._token_counts:
            self._token_counts[text] = len(self.tokenizer.encode(
                text, add_special_tokens=True, add_prefix_space=True,
                truncation=False, verbose=False))
        return self._token_counts[text]

    def score(self, pairs: list[dict]) -> list[dict]:
        references = [pair["gold_claim"] for pair in pairs]
        candidates = [pair["candidate"] for pair in pairs]
        bleu = [self.bleu.compute(predictions=[cand], references=[[gold]],
                                  max_order=4, smooth=False)["bleu"]
                for gold, cand in zip(references, candidates)]
        rouge = self.rouge.compute(predictions=candidates, references=references,
                                  rouge_types=["rouge1", "rougeL"],
                                  use_aggregator=False, use_stemmer=False)
        bert = self.bertscore.compute(
            predictions=candidates, references=references, lang="en",
            model_type="roberta-large", num_layers=17, idf=False,
            device="cpu", batch_size=self.batch_size, nthreads=self.threads,
            rescale_with_baseline=False, use_fast_tokenizer=False)
        self.hashcode = bert["hashcode"]
        result = []
        for i, pair in enumerate(pairs):
            gold_tokens = self.token_count(pair["gold_claim"])
            candidate_tokens = self.token_count(pair["candidate"])
            score = {
                "bleu4": float(bleu[i]), "rouge1_f1": float(rouge["rouge1"][i]),
                "rougeL_f1": float(rouge["rougeL"][i]), "bertscore_f1": float(bert["f1"][i]),
                "gold_tokens": gold_tokens, "candidate_tokens": candidate_tokens,
                "gold_truncated": gold_tokens > 512, "candidate_truncated": candidate_tokens > 512,
            }
            validate_score(score)
            result.append(score)
        return result


def run(input_path: Path, output_dir: Path, threads: int = 4, batch_size: int = 4) -> dict:
    started = time.monotonic()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    input_sha = file_hash(input_path)
    pairs = read_pairs(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "similarity_progress.json"
    checkpoint_path = output_dir / "similarity_checkpoint.jsonl"
    output_path = output_dir / "similarity_scores.json"
    settings = {
        "device": "cpu", "seed": 42, "threads": threads, "batch_size": batch_size,
        "offline": True, "normalization": "No custom normalization; raw strings passed to metrics",
        "bleu4": {"max_order": 4, "smooth": False, "tokenizer": "Tokenizer13a", "aggregation": "individual pair"},
        "rouge": {"types": ["rouge1", "rougeL"], "use_stemmer": False, "use_aggregator": False, "value": "fmeasure"},
        "bertscore": {"lang": "en", "model_type": "roberta-large", "model_revision": MODEL_REVISION,
                      "num_layers": 17, "idf": False, "rescale_with_baseline": False,
                      "use_fast_tokenizer": False, "value": "f1", "max_length_including_special_tokens": 512,
                      "truncation_side": "right", "text_strip": "BERTScore library internal strip only"},
    }
    signatures = {"input_sha256": input_sha, "settings_sha256": object_hash(settings)}
    atomic_json(progress_path, {"stage": "similarity", "status": "initializing", **signatures,
                              "completed_pairs": 0, "total_pairs": len(pairs), "updated_at": now()})
    evaluator = SimilarityEvaluator(threads, batch_size)
    signatures["environment_sha256"] = object_hash(evaluator.environment)
    valid_ids = {pair["pair_id"] for pair in pairs}
    scores, old_bert_hash = read_checkpoint(checkpoint_path, signatures, valid_ids)
    pending = [pair for pair in pairs if pair["pair_id"] not in scores]
    bert_hash = old_bert_hash
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            batch_scores = evaluator.score(batch)
            if bert_hash is not None and evaluator.hashcode != bert_hash:
                raise ValueError("Resumed BERTScore hashcode differs from checkpoint")
            bert_hash = evaluator.hashcode
            records = [{**signatures, "pair_id": pair["pair_id"], "score": score,
                        "score_sha256": object_hash(score), "bertscore_hashcode": bert_hash}
                       for pair, score in zip(batch, batch_scores)]
            append_records(checkpoint_path, records)
            scores.update({pair["pair_id"]: score for pair, score in zip(batch, batch_scores)})
            progress = {"stage": "similarity", "status": "running", **signatures,
                        "completed_pairs": len(scores), "total_pairs": len(pairs),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "updated_at": now(), "bertscore_hashcode": bert_hash}
            atomic_json(progress_path, progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
        if set(scores) != valid_ids:
            raise ValueError("Completed score set does not match input IDs")
        result = {"stage": "similarity", "status": "complete", **signatures,
                  "settings": settings, "environment": evaluator.environment,
                  "bertscore_hashcode": bert_hash, "model_revision": MODEL_REVISION,
                  "input_path": str(input_path), "completed_at": now(),
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "scores": {pair["pair_id"]: scores[pair["pair_id"]] for pair in pairs}}
        atomic_json(output_path, result)
        atomic_json(progress_path, {"stage": "similarity", "status": "complete", **signatures,
                                  "completed_pairs": len(scores), "total_pairs": len(pairs),
                                  "elapsed_seconds": result["elapsed_seconds"], "updated_at": now(),
                                  "output": str(output_path), "output_sha256": file_hash(output_path)})
        return result
    except BaseException as error:
        atomic_json(progress_path, {"stage": "similarity", "status": "failed", **signatures,
                                  "completed_pairs": len(scores), "total_pairs": len(pairs),
                                  "updated_at": now(), "error": str(error),
                                  "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    result = run(args.input, args.output_dir, args.threads, args.batch_size)
    print(json.dumps({"stage": result["stage"], "status": result["status"],
                      "pairs": len(result["scores"]), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
