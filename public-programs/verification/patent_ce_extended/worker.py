#!/usr/bin/env python3
"""Offline, provenance-bound evaluation of eight additional Patent-CE metrics.

Upstream source and model files are read-only. CPU/local-path compatibility
changes are applied in memory and recorded, while scoring algorithms remain
the published implementations. Smoke results always live under smoke/METHOD.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
import types
from typing import Any


METHODS = ("bleu1", "rouge2_f1", "meteor", "bartscore", "moverscore_v2",
           "simcse", "unieval_relevance", "alignscore")
NEURAL = METHODS[3:]
SCHEMA = 2
SEED = 42
TOLERANCE = 1e-5


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def raw_pair_id(gold: str, candidate: str) -> str:
    data = json.dumps([gold, candidate], ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_input(path: Path) -> dict:
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), list) or not data["pairs"]:
        raise ValueError("Input requires a nonempty pairs list")
    seen = set()
    for pair in data["pairs"]:
        if not isinstance(pair, dict):
            raise ValueError("Pair must be an object")
        for field in ("gold_claim", "candidate"):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ValueError("Pair text must be a nonempty raw string")
        identifier = raw_pair_id(pair["gold_claim"], pair["candidate"])
        if identifier != pair.get("pair_id") or identifier in seen:
            raise ValueError("Invalid or duplicate raw pair ID")
        seen.add(identifier)
    return data


def training_populations(data: dict) -> tuple[list[str], list[str]]:
    if not isinstance(data.get("records"), list):
        raise ValueError("MoverScore requires prepared records with training split")
    gold, candidates = {}, {}
    for row in data["records"]:
        if row.get("split") == "train":
            gold.setdefault(row["gold_claim"], None)
            candidates.setdefault(row["A"], None)
            candidates.setdefault(row["B"], None)
    if not gold or not candidates:
        raise ValueError("Missing nonempty training IDF populations")
    return list(gold), list(candidates)


def select_smoke_pairs(pairs: list[dict]) -> list[dict]:
    selected = []
    for condition in (lambda p: p["gold_claim"] == p["candidate"],
                      lambda p: p["gold_claim"] != p["candidate"]):
        found = next((p for p in pairs if condition(p)), None)
        if found is not None and found not in selected:
            selected.append(found)
    longest = max(pairs, key=lambda p: len(p["gold_claim"]) + len(p["candidate"]))
    if longest not in selected:
        selected.append(longest)
    return selected


def validate_result(method: str, result: dict) -> None:
    if not isinstance(result, dict) or type(result.get("score")) not in (int, float):
        raise ValueError("Result score must be numeric")
    score = result["score"]
    if not math.isfinite(score):
        raise ValueError("Nonfinite score")
    if method == "bartscore":
        valid = score <= TOLERANCE
    elif method == "simcse":
        valid = -1 - TOLERANCE <= score <= 1 + TOLERANCE
    else:
        valid = -TOLERANCE <= score <= 1 + TOLERANCE
    if not valid:
        raise ValueError(f"Unexpected {method} score: {score}")
    metadata = result.get("metadata")
    if not isinstance(metadata, dict) or type(metadata.get("truncated")) is not bool:
        raise ValueError("Every result requires an explicit actual truncation flag")
    for field in ("gold_tokens", "candidate_tokens"):
        if type(metadata.get(field)) is not int or metadata[field] < 0:
            raise ValueError(f"Invalid untruncated token count: {field}")
    for field in ("gold_truncated", "candidate_truncated", "pair_truncated"):
        if field in metadata and type(metadata[field]) is not bool:
            raise ValueError(f"Invalid token truncation attribution: {field}")
    canonical(metadata)  # Reject NaN or unsupported data in nested metadata.


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("ab") as stream:
        stream.write(canonical({**record, "record_sha256": value_hash(record)}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_checkpoint(path: Path, signatures: dict, identifiers: set[str], method: str) -> tuple[dict, float]:
    scores, elapsed = {}, 0.0
    if not path.exists():
        return scores, elapsed
    raw = path.read_bytes()
    complete_bytes = 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # Only an interrupted, unterminated final write may be removed.
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Checkpoint record must be an object")
        checksum = record.pop("record_sha256", None)
        if checksum != value_hash(record):
            raise ValueError("Checkpoint complete-line checksum mismatch")
        for key, expected in signatures.items():
            if record.get(key) != expected:
                raise ValueError(f"Checkpoint {key} mismatch; choose a new output directory")
        identifier = record.get("pair_id")
        if identifier not in identifiers or identifier in scores:
            raise ValueError("Unknown or duplicate checkpoint pair")
        result = record.get("result")
        validate_result(method, result)
        if record.get("result_sha256") != value_hash(result):
            raise ValueError("Checkpoint result checksum mismatch")
        current_elapsed = record.get("elapsed_seconds")
        if type(current_elapsed) not in (int, float) or not math.isfinite(current_elapsed) or current_elapsed < elapsed:
            raise ValueError("Invalid checkpoint elapsed provenance")
        elapsed = float(current_elapsed)
        scores[identifier] = result
        complete_bytes += len(line)
    if complete_bytes != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(complete_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    return scores, elapsed


def runtime_environment(threads: int) -> dict:
    versions = {}
    for name in ("torch", "transformers", "numpy", "nltk", "rouge-score", "pyemd",
                 "scikit-learn", "scipy", "pytorch-lightning", "spacy", "sentencepiece", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "device": "cpu", "threads": threads, "seed": SEED,
            "offline": True}


def configure(assets: dict, threads: int) -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_EVALUATE_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(threads),
                       "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                       "NLTK_DATA": str(assets["nltk_data_dir"]), "HF_HUB_DISABLE_XET": "1"})


def method_settings(method: str, data: dict, threads: int, batch_size: int, smoke: bool) -> dict:
    common = {"method": method, "seed": SEED, "device": "cpu", "threads": threads,
              "batch_size": batch_size, "smoke": smoke,
              "normalization": "Raw source strings; upstream tokenizer behavior only",
              "score_direction": "higher is better", "orientation": "gold reference then candidate"}
    specific = {
        "bleu1": {"tokenizer": "Tokenizer13a", "max_order": 1, "smooth": False},
        "rouge2_f1": {"type": "rouge2", "value": "fmeasure", "use_stemmer": False},
        "meteor": {"tokenizer": "nltk.word_tokenize", "alpha": 0.9, "beta": 3, "gamma": 0.5,
                   "matching": "NLTK lowercase, exact/stem/WordNet synonyms"},
        "bartscore": {"model": "facebook/bart-large-cnn", "max_length": 1024,
                      "score": "mean target-token log likelihood", "direction": "gold -> candidate"},
        "moverscore_v2": {"model": "distilbert-base-uncased", "version": "moverscore_v2",
                          "n_gram": 1, "stop_words": [], "remove_subwords": True, "max_length": 512,
                          "idf": "separate de-duplicated raw training gold/candidate populations"},
        "simcse": {"model": "princeton-nlp/sup-simcse-bert-base-uncased", "pooler": "cls",
                   "max_length": 128, "normalize_to_unit": True, "score": "paired cosine"},
        "unieval_relevance": {"model": "MingZhong/unieval-sum", "dimension": "relevance",
                              "overall": False, "max_length": 1024, "source": "",
                              "score": "P(Yes)/(P(Yes)+P(No))"},
        "alignscore": {"model": "roberta-large", "checkpoint": "AlignScore-large.ckpt",
                       "evaluation_mode": "nli_sp", "max_length": 512,
                       "chunking": "official NLTK sentences and approximately 350-word reference chunks",
                       "aggregation": "max reference chunk per candidate sentence, then sentence mean",
                       "exact_subpair_memoization": "all three official heads by exact raw chunk/sentence; batch_size=1",
                       "unused_mlm_logits": "skip under eval without mlm labels; all modules/checkpoint parameters retained"},
    }
    common.update(specific[method])
    if method == "moverscore_v2":
        gold, candidates = training_populations(data)
        common["idf_training_population_sha256"] = value_hash({"gold": gold, "candidates": candidates})
        common["idf_reference_documents"] = len(gold)
        common["idf_candidate_documents"] = len(candidates)
    return common


def validate_assets(manifest: dict, method: str) -> dict:
    if manifest.get("status") not in ("ready", "complete"):
        raise ValueError("Asset manifest is not ready/complete")
    relevant = manifest.get("assets", {}).get(method, {}) if method in NEURAL else manifest.get("lexical", {})
    if method in NEURAL and not relevant:
        raise ValueError(f"No {method} asset entry")
    inventory = dict(relevant.get("file_hashes", {}))
    if method == "meteor":
        inventory.update(manifest.get("nltk_file_hashes", {}))
    if method == "alignscore":
        inventory.update(manifest.get("nltk_file_hashes", {}))
    checked = {}
    for name, expected in inventory.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Missing absolute asset file: {name}")
        actual = file_hash(path)
        if isinstance(expected, dict):
            expected = expected.get("sha256")
        if actual != expected:
            raise ValueError(f"Asset SHA-256 mismatch: {name}")
        checked[name] = actual
    if not checked:
        raise ValueError("Asset entry must include verified source/resource/weight hashes")
    lock = load_json(Path(manifest["environment_lock_path"]))
    # Setup timestamps/logs are not part of an inference identity. Package/source
    # content and pinned requirements remain bound even if setup is replayed.
    semantic_lock = {key: value for key, value in lock.items()
                     if key not in ("created_at_utc", "created_at", "updated_at", "elapsed_seconds")}
    semantic_asset = {key: relevant[key] for key in
                      ("model_dir", "source_dir", "repository", "revision", "source_commit",
                       "checkpoint_path", "module_revision", "source", "upstream_files")
                      if key in relevant}
    return {"asset_entry": semantic_asset, "validated_file_hashes": checked,
            "environment_lock_semantic_sha256": value_hash(semantic_lock)}


def load_source(path: Path, name: str, patches: list[tuple[str, str]] | None = None) -> tuple[types.ModuleType, dict]:
    source = path.read_text(encoding="utf-8")
    applied = []
    for before, after in patches or []:
        count = source.count(before)
        if count < 1:
            raise ValueError(f"Pinned source compatibility patch no longer matches: {before}")
        source = source.replace(before, after)
        applied.append({"before": before, "after": after, "matches": count})
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, {"path": str(path), "original_sha256": file_hash(path),
                    "runtime_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "patches": applied}


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))


def individual_metadata(tokenizer, gold: str, candidate: str, limit: int) -> dict:
    a, b = token_count(tokenizer, gold), token_count(tokenizer, candidate)
    return {"gold_tokens": a, "candidate_tokens": b, "limit": limit,
            "gold_truncated": a > limit, "candidate_truncated": b > limit,
            "truncated": a > limit or b > limit, "token_counts_include_special_tokens": True}


class ExactHeadMemoizer:
    """Memoize independent batch-one model outputs, retaining native order/math.

    Tensor functions are injected so the integrity/reordering logic can be tested
    without importing torch or performing model inference.
    """
    def __init__(self, inference, stack, clone):
        self.inference, self.stack, self.clone = inference, stack, clone
        self.cache = {}
        self.requested = self.misses = self.hits = 0

    def __call__(self, premise, hypothesis):
        if isinstance(premise, str) and isinstance(hypothesis, str):
            premise, hypothesis = [premise], [hypothesis]
        if len(premise) != len(hypothesis) or not premise:
            raise ValueError("AlignScore memoizer requires equal nonempty input lists")
        keys = list(zip(premise, hypothesis))
        pending = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if pending:
            heads = self.inference([key[0] for key in pending], [key[1] for key in pending])
            if len(heads) != 3 or any(head is None or len(head) != len(pending) for head in heads):
                raise ValueError("AlignScore nli_sp must return all three native heads")
            for index, key in enumerate(pending):
                self.cache[key] = tuple(self.clone(head[index]) for head in heads)
        self.requested += len(keys)
        self.misses += len(pending)
        self.hits += len(keys) - len(pending)
        return tuple(self.stack([self.cache[key][head] for key in keys]) for head in range(3))

    def diagnostics(self) -> dict:
        return {"scope": "current process only; cache is rebuilt on interrupted resume",
                "requested_model_subpairs": self.requested, "unique_model_subpairs_inferred": self.misses,
                "exact_cache_hits": self.hits, "cached_three_head_tuples": len(self.cache)}


class Adapter:
    """Small orientation/metadata adapter around the unchanged author scorers."""
    def __init__(self, method: str, manifest: dict, data: dict, out: Path,
                 threads: int, batch_size: int, signatures: dict):
        self.method, self.batch_size = method, batch_size
        self.provenance = {"source_adapters": [], "algorithm_changes": []}
        self.embedding_cache = {}
        self.asset = manifest.get("assets", {}).get(method, {})
        self.threads, self.out, self.signatures = threads, out, signatures
        import numpy as np
        import torch
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.np = np
        if method == "bleu1":
            source = Path(manifest["bleu_source_dir"])
            self.bleu, provenance = load_source(source / "nmt_bleu.py", "patce_nmt_bleu")
            tok, token_provenance = load_source(source / "tokenizer_13a.py", "patce_tokenizer13a")
            self.tokenizer = tok.Tokenizer13a()
            self.provenance["source_adapters"] += [provenance, token_provenance]
        elif method == "rouge2_f1":
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=False)
            self.provenance["source_file"] = str(Path(inspect.getfile(rouge_scorer)))
            self.provenance["source_sha256"] = file_hash(Path(self.provenance["source_file"]))
        elif method == "meteor":
            import nltk
            from nltk.translate import meteor_score
            self.nltk, self.meteor = nltk, meteor_score
            self.provenance["source_file"] = inspect.getfile(meteor_score)
            self.provenance["source_sha256"] = file_hash(Path(self.provenance["source_file"]))
            # Resolve required resources without invoking any NLTK downloader.
            nltk.data.find("tokenizers/punkt_tab/english/")
            meteor_score.wordnet.ensure_loaded()
        elif method == "bartscore":
            module, prov = load_source(Path(self.asset["source_dir"]) / "bart_score.py", "patce_bartscore")
            self.scorer = module.BARTScorer(device="cpu", max_length=1024,
                                            checkpoint=self.asset["model_dir"])
            self.tokenizer = self.scorer.tokenizer
            self.provenance["source_adapters"].append(prov)
        elif method == "moverscore_v2":
            self._load_mover(data)
        elif method == "simcse":
            module, prov = load_source(Path(self.asset["source_dir"]) / "simcse" / "tool.py", "patce_simcse_tool")
            self.scorer = module.SimCSE(self.asset["model_dir"], device="cpu", pooler="cls")
            self.scorer.model.eval()
            self.tokenizer = self.scorer.tokenizer
            self.provenance["source_adapters"].append(prov)
        elif method == "unieval_relevance":
            source = Path(self.asset["source_dir"])
            self.uni_utils, p1 = load_source(source / "utils.py", "patce_unieval_utils")
            self.uni_module, p2 = load_source(source / "metric" / "scorer.py", "patce_unieval_scorer")
            self.scorer = self.uni_module.UniEvaluator(self.asset["model_dir"], max_length=1024, device="cpu")
            self.tokenizer = self.scorer.tokenizer
            self.provenance["source_adapters"] += [p1, p2]
        elif method == "alignscore":
            self._load_align()
        else:
            raise ValueError(method)

    def _load_mover(self, data: dict) -> None:
        path = Path(self.asset["source_dir"]) / "moverscore_v2.py"
        local = repr(self.asset["model_dir"])
        patches = [
            ("device = 'cuda'", "device = 'cpu'"),
            ("AutoTokenizer.from_pretrained(model_name, do_lower_case=True)",
             f"AutoTokenizer.from_pretrained({local}, do_lower_case=True, local_files_only=True)"),
            ("AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)",
             f"AutoModel.from_pretrained({local}, output_hidden_states=True, output_attentions=True, local_files_only=True)"),
            ("dtype=np.float)", "dtype=np.float64)"),
            ("import matplotlib.pyplot as plt", "plt = None  # Visualization-only dependency is not used by scoring."),
        ]
        os.environ.pop("MOVERSCORE_MODEL", None)  # Preserve the upstream logical DistilBERT branch.
        self.mover, provenance = load_source(path, "patce_moverscore_v2", patches)
        self.tokenizer = self.mover.tokenizer
        if self.tokenizer.model_max_length != 512:
            raise ValueError("MoverScore requires the fixed 512-token DistilBERT tokenizer")
        self.provenance["source_adapters"].append(provenance)
        gold, candidates = training_populations(data)
        population = {"gold": gold, "candidates": candidates}
        idf_path = self.out / "moverscore_v2_idf.json"
        population_hash = value_hash(population)
        if idf_path.exists():
            payload = load_json(idf_path)
            if payload.get("population_sha256") != population_hash or payload.get("assets_sha256") != self.signatures["assets_sha256"]:
                raise ValueError("Existing MoverScore IDF population/assets mismatch")
        else:
            ref_idf = self.mover.get_idf_dict(gold, nthreads=self.threads)
            cand_idf = self.mover.get_idf_dict(candidates, nthreads=self.threads)
            payload = {"population_sha256": population_hash, "assets_sha256": self.signatures["assets_sha256"],
                       "gold_raw_documents": len(gold), "candidate_raw_documents": len(candidates),
                       "reference": {"default": math.log(len(gold)+1), "values": dict(ref_idf)},
                       "candidate": {"default": math.log(len(candidates)+1), "values": dict(cand_idf)}}
            atomic_json(idf_path, payload)
        def restore(value):
            default = float(value["default"])
            return defaultdict(lambda: default, {int(k): float(v) for k,v in value["values"].items()})
        self.ref_idf, self.cand_idf = restore(payload["reference"]), restore(payload["candidate"])
        self.provenance["idf"] = {"path": str(idf_path), "sha256": file_hash(idf_path),
                                  "population_sha256": population_hash,
                                  "reference_documents": len(gold), "candidate_documents": len(candidates)}

    def _load_align(self) -> None:
        if self.batch_size != 1:
            raise ValueError("Verified exact AlignScore memoization requires batch_size=1")
        root = Path(self.asset["source_dir"]) / "src"
        sys.path.insert(0, str(root))
        import alignscore.inference as inference
        import alignscore.alignscore as implementation
        path = Path(inspect.getfile(inference))
        before = "BERTAlignModel(model=model).load_from_checkpoint(checkpoint_path=ckpt_path, strict=False).to(self.device)"
        after = "BERTAlignModel.load_from_checkpoint(checkpoint_path=ckpt_path, map_location='cpu', strict=True, model=model, using_pretrained=False).to(self.device)"
        source = path.read_text(encoding="utf-8")
        if source.count(before) != 1:
            raise ValueError("Pinned AlignScore checkpoint-loading source changed")
        adapted = source.replace(before, after)
        exec(compile(adapted, str(path), "exec"), inference.__dict__)
        implementation.Inferencer = inference.Inferencer
        self.scorer = implementation.AlignScore(model=self.asset["model_dir"], batch_size=self.batch_size,
                                               device="cpu", ckpt_path=self.asset["checkpoint_path"],
                                               evaluation_mode="nli_sp", verbose=False)
        self.tokenizer = self.scorer.model.tokenizer
        import torch
        self.align_original_forward = self.scorer.model.model.forward
        state_keys_before = list(self.scorer.model.model.state_dict())
        forward_source = textwrap.dedent(inspect.getsource(self.align_original_forward))
        forward_before = "prediction_scores = self.mlm_head(base_model_output.last_hidden_state)"
        forward_after = "prediction_scores = self.mlm_head(base_model_output.last_hidden_state) if self.training or 'mlm_label' in batch else None"
        if forward_source.count(forward_before) != 1:
            raise ValueError("Pinned AlignScore unused-MLM forward assignment changed")
        optimized_source = forward_source.replace(forward_before, forward_after)
        namespace = dict(self.align_original_forward.__func__.__globals__)
        exec(compile(optimized_source, str(path.with_name("model.py")), "exec"), namespace)
        self.align_optimized_forward = types.MethodType(namespace["forward"], self.scorer.model.model)
        self.scorer.model.model.forward = self.align_optimized_forward
        if list(self.scorer.model.model.state_dict()) != state_keys_before:
            raise ValueError("Unused-MLM optimization changed checkpoint state keys")
        self.align_optimization_comparisons = []
        self.align_uncached_inference = self.scorer.model.inference
        self.align_memoizer = ExactHeadMemoizer(self.align_uncached_inference, torch.stack,
                                               lambda value: value.detach().clone())
        self.scorer.model.inference = self.align_memoizer
        self.provenance["source_adapters"].append({"path": str(path), "original_sha256": file_hash(path),
                    "runtime_source_sha256": hashlib.sha256(adapted.encode("utf-8")).hexdigest(),
                    "patches": [{"before": before, "after": after, "matches": 1}],
                    "reason": "CPU/local backbone initialization; strict full checkpoint restoration; inference math unchanged"})
        self.provenance["exact_subpair_memoization"] = {
            "key": "exact raw reference chunk and candidate sentence", "outputs": "all three native heads",
            "batch_size": 1, "aggregation": "unchanged native max then mean",
            "smoke_reference": "bypasses cache and calls original native inference"}
        self.provenance["unused_mlm_optimization"] = {
            "source_path": str(path.with_name("model.py")), "source_file_sha256": file_hash(path.with_name("model.py")),
            "original_forward_source_sha256": hashlib.sha256(forward_source.encode()).hexdigest(),
            "optimized_forward_source_sha256": hashlib.sha256(optimized_source.encode()).hexdigest(),
            "patch": {"before": forward_before, "after": forward_after, "matches": 1},
            "reason": "Native inference never reads prediction_logits; backbone and all three score heads are unchanged",
            "mlm_module_and_checkpoint_parameters_retained": True,
            "state_dict_keys_sha256": value_hash(state_keys_before),
            "smoke_reference": "original unmodified forward; compare all three head tensors on identical subpairs"}

    def diagnostics(self) -> dict:
        return {"alignscore_exact_subpair_cache": self.align_memoizer.diagnostics()} if self.method == "alignscore" else {}

    def uni_inputs(self, gold: str, candidate: str) -> list[str]:
        return self.uni_utils.add_question(dimension="relevance", output=[candidate], src=[""],
                                           ref=[gold], task="summarization")

    def align_metadata(self, gold: str, candidate: str) -> dict:
        from nltk import sent_tokenize
        sentences = sent_tokenize(gold) or [""]
        count = len(gold.strip().split()) // 350 + 1
        chunk_sentences = max(len(sentences) // count, 1)
        chunks = [" ".join(sentences[i:i+chunk_sentences]) for i in range(0, len(sentences), chunk_sentences)]
        hypotheses = sent_tokenize(candidate)
        if not hypotheses:
            raise ValueError("AlignScore produced no candidate sentences")
        lengths = []
        raw_subpairs = []
        fallback = 0
        for chunk in chunks:
            for sentence in hypotheses:
                raw_subpairs.append((chunk, sentence))
                length = len(self.tokenizer.encode(chunk, sentence, add_special_tokens=True,
                                                   truncation=False, verbose=False))
                lengths.append(length)
                if token_count(self.tokenizer, sentence) > 509:
                    fallback += 1
        return {"gold_tokens": token_count(self.tokenizer, gold),
                "candidate_tokens": token_count(self.tokenizer, candidate), "limit": 512,
                "gold_chunks": len(chunks), "candidate_sentences": len(hypotheses),
                "chunk_sentence_pairs": len(lengths), "maximum_pair_tokens": max(lengths),
                "unique_exact_chunk_sentence_pairs": len(set(raw_subpairs)),
                "within_pair_duplicate_subpairs": len(raw_subpairs)-len(set(raw_subpairs)),
                "pair_tokens": lengths, "truncated_model_pairs": sum(n > 512 for n in lengths),
                "potential_long_candidate_fallback_pairs": fallback,
                "truncated": any(n > 512 for n in lengths), "chunking_is_truncation": False}

    def score(self, pair: dict) -> dict:
        gold, candidate = pair["gold_claim"], pair["candidate"]
        m = self.method
        if m == "bleu1":
            a, b = self.tokenizer(gold), self.tokenizer(candidate)
            value = self.bleu.compute_bleu([[a]], [b], max_order=1, smooth=False)[0]
            metadata = {"gold_tokens": len(a), "candidate_tokens": len(b), "truncated": False}
        elif m == "rouge2_f1":
            value = self.scorer.score(gold, candidate)["rouge2"].fmeasure
            metadata = {"gold_tokens": len(self.scorer._tokenizer.tokenize(gold)),
                        "candidate_tokens": len(self.scorer._tokenizer.tokenize(candidate)), "truncated": False}
        elif m == "meteor":
            a, b = self.nltk.word_tokenize(gold), self.nltk.word_tokenize(candidate)
            value = self.meteor.single_meteor_score(a, b, alpha=.9, beta=3, gamma=.5)
            metadata = {"gold_tokens": len(a), "candidate_tokens": len(b), "truncated": False}
        elif m == "bartscore":
            value = self.scorer.score([gold], [candidate], batch_size=self.batch_size)[0]
            metadata = individual_metadata(self.tokenizer, gold, candidate, 1024)
        elif m == "moverscore_v2":
            value = self.mover.word_mover_score([gold], [candidate], self.ref_idf, self.cand_idf,
                        stop_words=[], n_gram=1, remove_subwords=True, batch_size=self.batch_size, device="cpu")[0]
            metadata = individual_metadata(self.tokenizer, gold, candidate, 512)
        elif m == "simcse":
            for text in (gold, candidate):
                if text not in self.embedding_cache:
                    self.embedding_cache[text] = self.scorer.encode(text, device="cpu", return_numpy=True,
                        normalize_to_unit=True, max_length=128, batch_size=self.batch_size)
            a, b = self.embedding_cache[gold], self.embedding_cache[candidate]
            value = self.np.dot(a, b) / (self.np.linalg.norm(a)*self.np.linalg.norm(b))
            metadata = individual_metadata(self.tokenizer, gold, candidate, 128)
        elif m == "unieval_relevance":
            inputs = self.uni_inputs(gold, candidate)
            value = self.scorer.score(inputs, batch_size=self.batch_size)[0]
            combined = token_count(self.tokenizer, inputs[0])
            metadata = {"gold_tokens": token_count(self.tokenizer, gold),
                        "candidate_tokens": token_count(self.tokenizer, candidate),
                        "pair_tokens": combined, "pair_truncated": combined > 1024,
                        "limit": 1024, "truncated": combined > 1024,
                        "input_order": "question, candidate, reference", "individual_truncation_attribution": "not inferred"}
        elif m == "alignscore":
            value = self.scorer.score(contexts=[gold], claims=[candidate])[0]
            metadata = self.align_metadata(gold, candidate)
        else:
            raise ValueError(m)
        result = {"score": float(value), "metadata": metadata}
        validate_result(m, result)
        return result

    def official_reference(self, pair: dict) -> float:
        """A second author entry point/check invocation for smoke equivalence."""
        gold, cand = pair["gold_claim"], pair["candidate"]
        if self.method == "simcse":
            # Official similarity uses its exact 128-token default and paired scalar inputs.
            return float(self.scorer.similarity(gold, cand, device="cpu"))
        if self.method == "unieval_relevance":
            # The author SumEvaluator delegates this exact relevance query to its scorer.
            input_text = "question: Is this summary relevant to the reference? </s> summary: " + cand + " </s> reference: " + gold
            return float(self.scorer.score([input_text], batch_size=self.batch_size)[0])
        if self.method == "alignscore":
            return self._align_official_reference(pair)
        # These adapters call the unmodified author scoring function directly.
        return self.score(pair)["score"]

    def _align_official_reference(self, pair: dict) -> float:
        """Compare original/optimized heads, returning the uncached original score."""
        import torch
        inference = self.scorer.model
        cached = inference.inference
        def compare(premises, hypotheses):
            inference.model.forward = self.align_original_forward
            started = time.monotonic()
            try:
                native = self.align_uncached_inference(premises, hypotheses)
            finally:
                inference.model.forward = self.align_optimized_forward
            native_elapsed = time.monotonic()-started
            started = time.monotonic()
            optimized = self.align_uncached_inference(premises, hypotheses)
            optimized_elapsed = time.monotonic()-started
            heads = []
            for name, original, actual in zip(("regression", "binary", "nli"), native, optimized):
                error = float((original-actual).abs().max().item())
                equal = bool(torch.equal(original, actual))
                if not torch.allclose(original, actual, rtol=0, atol=1e-7):
                    raise ValueError(f"AlignScore unused-MLM head parity failed: {name} {error}")
                heads.append({"head": name, "torch_equal": equal, "maximum_absolute_error": error,
                              "shape": list(original.shape), "dtype": str(original.dtype)})
            self.align_optimization_comparisons.append({
                "pair_id": pair["pair_id"], "subpairs": len(premises), "heads": heads,
                "original_unmodified_forward_seconds": native_elapsed,
                "optimized_forward_seconds": optimized_elapsed,
                "speedup_ratio": native_elapsed/optimized_elapsed})
            return native
        inference.inference = compare
        try:
            return float(inference.inference_per_example(pair["gold_claim"], pair["candidate"]))
        finally:
            inference.inference = cached
            inference.model.forward = self.align_optimized_forward


def make_adapter(method: str, assets: dict, data: dict, out: Path, threads: int,
                 batch_size: int, signatures: dict) -> Adapter:
    return Adapter(method, assets, data, out, threads, batch_size, signatures)


def read_state(path: Path, signatures: dict) -> dict | None:
    if not path.exists():
        return None
    state = load_json(path)
    checksum = state.pop("state_sha256", None)
    if checksum != value_hash(state):
        raise ValueError("Worker state checksum mismatch")
    for key, value in signatures.items():
        if state["signatures"].get(key) != value:
            raise ValueError(f"Worker state {key} mismatch")
    return state


def write_state(path: Path, state: dict) -> None:
    atomic_json(path, {**state, "state_sha256": value_hash(state)})


def run(method: str, input_path: Path, output_dir: Path, assets_path: Path,
        threads: int = 4, batch_size: int = 1, smoke: bool = False,
        resume_only: bool = False, make_proof: bool = True) -> dict:
    started = time.monotonic()
    input_path, assets_path = input_path.resolve(), assets_path.resolve()
    out = output_dir.resolve() / "smoke" / method if smoke else output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, manifest = read_input(input_path), load_json(assets_path)
    configure(manifest, threads)
    environment = runtime_environment(threads)
    settings = method_settings(method, data, threads, batch_size, smoke)
    pairs = select_smoke_pairs(data["pairs"]) if smoke else data["pairs"]
    verified_assets = validate_assets(manifest, method)
    signatures = {"schema_version": SCHEMA, "method": method,
                  "input_sha256": file_hash(input_path), "settings_sha256": value_hash(settings),
                  "worker_sha256": file_hash(Path(__file__)), "assets_sha256": value_hash(verified_assets),
                  "environment_sha256": value_hash(environment)}
    state_path = out / f"{method}_state.json"
    checkpoint = out / f"{method}_checkpoint.jsonl"
    result_path = out / f"{method}_scores.json"
    progress_path = out / f"{method}_progress.json"
    identifiers = {p["pair_id"] for p in pairs}
    state = read_state(state_path, signatures)
    scores, previous_elapsed = read_checkpoint(checkpoint, signatures, identifiers, method)
    if scores and state is None:
        raise ValueError("Checkpoint exists without its bound provenance state")
    pending = [p for p in pairs if p["pair_id"] not in scores]
    if not pending and result_path.exists():
        prior = load_json(result_path)
        for key, value in signatures.items():
            if prior.get(key) != value:
                raise ValueError(f"Completed output {key} mismatch")
        if prior.get("complete") is not True or prior.get("scores") != scores:
            raise ValueError("Completed result/checkpoint mismatch")
        return prior  # Preserve the original first-run runtime and result bytes.
    if resume_only:
        raise ValueError("Resume-only proof required a complete bound result; inference was prohibited")
    atomic_json(progress_path, {"method": method, "status": "initializing", "completed_pairs": len(scores),
                              "total_pairs": len(pairs), "updated_at": now(), **signatures})
    adapter = None
    try:
        adapter = make_adapter(method, manifest, data, out, threads, batch_size, signatures)
        provenance = {**verified_assets, **adapter.provenance,
                      "assets_manifest_path": str(assets_path), "input_path": str(input_path)}
        if state is None:
            state = {"signatures": signatures, "settings": settings, "environment": environment,
                     "provenance": provenance, "started_at": now()}
            write_state(state_path, state)
        elif state["provenance"] != provenance:
            raise ValueError("Resumed adapter provenance differs from checkpoint")
        for pair in pending:
            result = adapter.score(pair)
            elapsed = previous_elapsed + time.monotonic() - started
            append_checkpoint(checkpoint, {**signatures, "pair_id": pair["pair_id"], "result": result,
                                           "result_sha256": value_hash(result), "elapsed_seconds": elapsed})
            scores[pair["pair_id"]] = result
            progress = {**signatures, "status": "running", "completed_pairs": len(scores),
                        "total_pairs": len(pairs), "elapsed_seconds": round(elapsed, 3), "updated_at": now(),
                        "runtime_diagnostics": adapter.diagnostics() if hasattr(adapter, "diagnostics") else {}}
            atomic_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
        smoke_checks = None
        if smoke:
            comparisons = []
            for pair in pairs:
                reference = adapter.official_reference(pair)
                actual = scores[pair["pair_id"]]["score"]
                error = abs(reference-actual)
                if not math.isfinite(reference) or error > TOLERANCE:
                    raise ValueError(f"Official scorer smoke equivalence failed: {method} {error}")
                comparisons.append({"pair_id": pair["pair_id"], "adapter": actual,
                                    "official_reference": reference, "absolute_error": error})
            smoke_checks = {"status": "passed", "absolute_tolerance": TOLERANCE,
                            "reference": "unchanged author scoring functions; CPU/local restoration adapters recorded",
                            "comparisons": comparisons}
            if method == "alignscore" and hasattr(adapter, "align_optimization_comparisons"):
                head_comparisons = adapter.align_optimization_comparisons
                if len(head_comparisons) != len(pairs):
                    raise ValueError("Missing actual AlignScore original/optimized head comparisons")
                proof = {"status": "passed", "method": method, "verified_at": now(),
                         **signatures, "head_absolute_tolerance": 1e-7,
                         "all_head_tensors_exactly_equal": all(h["torch_equal"] for row in head_comparisons for h in row["heads"]),
                         "native_forward_unmodified": True, "native_reference_bypasses_cache": True,
                         "checkpoint_restoration": "strict full state; modules and parameters retained",
                         "subpair_comparisons": head_comparisons,
                         "optimization": adapter.provenance["unused_mlm_optimization"]}
                proof_path = out / "alignscore_optimization_parity.json"
                atomic_json(proof_path, proof)
                smoke_checks["optimization_parity"] = {"path": str(proof_path), "sha256": file_hash(proof_path)}
        if set(scores) != identifiers:
            raise ValueError("Final pair coverage mismatch")
        result = {**signatures, "complete": True, "smoke": smoke, "settings": settings,
                  "provenance": state["provenance"], "environment": environment,
                  "scores": {p["pair_id"]: scores[p["pair_id"]] for p in pairs},
                  "evaluated_pairs": len(pairs), "total_input_pairs": len(data["pairs"]),
                  "started_at": state["started_at"], "completed_at": now(),
                  "elapsed_seconds": round(previous_elapsed + time.monotonic()-started, 3),
                  "runtime_diagnostics": adapter.diagnostics() if hasattr(adapter, "diagnostics") else {},
                  "smoke_validation": smoke_checks}
        atomic_json(result_path, result)
        atomic_json(progress_path, {**signatures, "status": "complete", "completed_pairs": len(scores),
                                  "total_pairs": len(pairs), "elapsed_seconds": result["elapsed_seconds"],
                                  "updated_at": now(), "result_sha256": file_hash(result_path)})
        if make_proof:
            prove_resume(method, input_path, out, assets_path, threads, batch_size, smoke)
        return result
    except BaseException as error:
        atomic_json(progress_path, {**signatures, "status": "failed", "completed_pairs": len(scores),
                                  "total_pairs": len(pairs), "updated_at": now(), "error": str(error),
                                  "traceback": traceback.format_exc()})
        raise


def prove_resume(method: str, input_path: Path, completed_out: Path, assets_path: Path,
                 threads: int, batch_size: int, smoke: bool) -> dict:
    """Exercise the actual completed-result path with inference explicitly prohibited."""
    result_path = completed_out / f"{method}_scores.json"
    before = file_hash(result_path)
    original = load_json(result_path)
    with tempfile.TemporaryDirectory(prefix=f"patce_{method}_resume_") as temporary:
        base = Path(temporary)
        copied_out = base / "smoke" / method if smoke else base
        copied_out.mkdir(parents=True, exist_ok=True)
        for suffix in ("state.json", "checkpoint.jsonl", "scores.json"):
            name = f"{method}_{suffix}"
            shutil.copyfile(completed_out / name, copied_out / name)
        resumed = run(method, input_path, base, assets_path, threads, batch_size, smoke,
                      resume_only=True, make_proof=False)
        if resumed["scores"] != original["scores"] or file_hash(copied_out / f"{method}_scores.json") != before:
            raise AssertionError("Actual no-inference resume changed scores/output")
    if file_hash(result_path) != before:
        raise AssertionError("Resume proof changed original output")
    proof = {"status": "passed", "method": method, "verified_at": now(),
             "pairs": len(original["scores"]), "inference_calls": 0,
             "actual_run_resume_only_guard": True, "scores_identical": True,
             "original_output_unchanged": True, "original_output_sha256": before,
             "preserved_first_run_elapsed_seconds": original["elapsed_seconds"]}
    atomic_json(completed_out / f"{method}_resume_verification.json", proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Threads and batch size must be positive")
    result = run(args.method, args.input, args.output_dir, args.assets,
                 args.threads, args.batch_size, args.smoke)
    print(json.dumps({"method": args.method, "complete": result["complete"],
                      "pairs": len(result["scores"]), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
