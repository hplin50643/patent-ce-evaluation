#!/usr/bin/env python3
"""Fetch only fixed public inference assets and audit the isolated CPU runtime.

Bootstrap with bootstrap.sh; run this CLI with --output-dir RUN_DIRECTORY.
No model inference is performed. Upstream source bytes remain unmodified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = ROOT / "public-programs" / "models" / "patent-ce-extended"
MODEL_REVISIONS = {
    "bartscore": ("facebook/bart-large-cnn", "37f520fa929c961707657b28798b30c003dd100b", "model.safetensors"),
    "moverscore_v2": ("distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be", "model.safetensors"),
    "simcse": ("princeton-nlp/sup-simcse-bert-base-uncased", "2d82fab19ac3a73a20dd20333d27eb8a52d6e97f", "pytorch_model.bin"),
    "unieval_relevance": ("MingZhong/unieval-sum", "ed8da5a8aa01b19d49a5f8738d382ced54b73ad0", "pytorch_model.bin"),
    "alignscore": ("yzha/AlignScore", "8509e78d25bb914939fc585c626500c9b2944249", "AlignScore-large.ckpt"),
}
SOURCES = {
    "bartscore": ("neulab/BARTScore", "248f511cb34ae3753fc81f7d7a945de5bfe33458"),
    "moverscore_v2": ("AIPHES/emnlp19-moverscore", "0459a3b3b3bd73baa0cc515a355228ee5a2887e0"),
    "simcse": ("princeton-nlp/SimCSE", "49fa580a853752ede55b8c76d9debf748e214d3f"),
    "unieval_relevance": ("maszhongming/UniEval", "d33e7b6cfebe97b2bafe435adbd818230d5a416a"),
    "alignscore": ("yuh-zha/AlignScore", "a0936d5afee642a46b22f6c02a163478447aa493"),
}
TOKENIZER_FILES = {
    "config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.json", "vocab.json", "merges.txt", "vocab.txt", "spiece.model", "added_tokens.json",
    "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md",
}
LOCK = threading.RLock()
BLEU_UPSTREAM = {
    "nmt_bleu.py": ("https://raw.githubusercontent.com/tensorflow/nmt/0be864257a76c151eef20ea689755f08bc1faf4e/nmt/scripts/bleu.py", "d35b20182f82f22f1c31232df1d25977328cf011178fcd381c2d82ee4116b40b"),
    "tokenizer_13a.py": ("https://raw.githubusercontent.com/huggingface/evaluate/a5db2ad54aaeadf15c7aa1f056bc6f7bbeeb7e5f/metrics/bleu/tokenizer_13a.py", "496ba6b982897605ca52ac06595a0e59496c8563e26a28b47f14a0cfba725a91"),
    "bleu.py": ("https://raw.githubusercontent.com/huggingface/evaluate/a5db2ad54aaeadf15c7aa1f056bc6f7bbeeb7e5f/metrics/bleu/bleu.py", "41aebd59ddb0306117c049edf6a2f71b978cf0c5cce1da63f0e1551987f34315"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inventory(directory):
    return {str(path.resolve()): sha256(path) for path in sorted(Path(directory).rglob("*"))
            if path.is_file() and not any(part in {".cache", "__pycache__", ".git"} for part in path.parts)}


class Recorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.path = output_dir / "asset_downloads.jsonl"
        self.started = time.perf_counter()
        self.completed = {}

    def record(self, event, **details):
        value = {"time_utc": now(), "event": event, **details}
        with LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(value, ensure_ascii=False), flush=True)
            atomic_json(self.output_dir / "assets_progress.json", {
                "status": "running", "updated_at_utc": now(), "elapsed_seconds": time.perf_counter() - self.started,
                "completed_methods": sorted(self.completed), "last_event": value,
            })


def fetch_bytes(url):
    last = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Patent-CE-local-research-assets/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 15))
    raise last


def source_wanted(method, name):
    common = name.lower().startswith(("license", "copying")) or name == "README.md"
    if common:
        return True
    return {
        "bartscore": name == "bart_score.py",
        "moverscore_v2": name == "moverscore_v2.py",
        "simcse": name.startswith("simcse/") and name.endswith(".py"),
        "unieval_relevance": (name.startswith("metric/") and name.endswith(".py")) or name == "utils.py",
        "alignscore": (name.startswith("src/alignscore/") and name.endswith(".py")) or name == "pyproject.toml",
    }[method]


def download_sources(method, recorder):
    repository, commit = SOURCES[method]
    directory = ASSET_ROOT / "sources" / method
    directory.mkdir(parents=True, exist_ok=True)
    tree = json.loads(fetch_bytes(f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"))
    names = [item["path"] for item in tree["tree"] if item["type"] == "blob" and source_wanted(method, item["path"])]
    if not names:
        raise RuntimeError(f"No source files selected for {method}")
    for name in names:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repository}/{commit}/{name}"
        data = fetch_bytes(url)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        recorder.record("source_file", method=method, source=url, path=str(target), bytes=len(data), sha256=sha256(target))
    return directory


def download_model(method, recorder):
    from huggingface_hub import HfApi, hf_hub_download
    repository, revision, weights = MODEL_REVISIONS[method]
    directory = ASSET_ROOT / "weights" / method
    directory.mkdir(parents=True, exist_ok=True)
    available = HfApi(token=False).list_repo_files(repository, revision=revision, token=False)
    names = [name for name in available if name == weights or name in TOKENIZER_FILES]
    if weights not in names:
        raise RuntimeError(f"Missing fixed weight file {repository}/{weights}")
    for name in sorted(names, key=lambda name: (name == weights, name)):
        recorder.record("download_start", method=method, repository=repository, revision=revision, filename=name)
        last = None
        for attempt in range(5):
            try:
                path = Path(hf_hub_download(repository, filename=name, revision=revision, local_dir=directory, token=False))
                last = None
                break
            except Exception as exc:
                last = exc
                recorder.record("download_retry", method=method, filename=name, attempt=attempt + 1,
                                error_type=type(exc).__name__, error=str(exc))
                time.sleep(min(2 ** attempt, 20))
        if last:
            raise last
        recorder.record("download_complete", method=method, filename=name, path=str(path), bytes=path.stat().st_size,
                        sha256=sha256(path), source=f"https://huggingface.co/{repository}/resolve/{revision}/{name}")
    forbidden = [path for path in directory.rglob("*") if path.is_file() and
                 path.name.startswith(("optimizer", "scheduler", "rng_state", "training_args"))]
    if forbidden:
        raise RuntimeError(f"Training-only artifacts found: {forbidden}")
    return directory


def setup_method(method, recorder):
    source_dir = download_sources(method, recorder)
    model_dir = download_model(method, recorder)
    repository, revision, _ = MODEL_REVISIONS[method]
    result = {"model_dir": str(model_dir), "source_dir": str(source_dir), "repository": repository,
              "revision": revision, "source_repository": f"https://github.com/{SOURCES[method][0]}",
              "source_commit": SOURCES[method][1], "source_modifications": [],
              "compatibility_adapter": "Runtime adapter in worker.py; upstream source bytes remain unchanged",
              "file_hashes": {**inventory(source_dir), **inventory(model_dir)}}
    if method == "alignscore":
        backbone = ROOT / "LLM4DPCG" / ".cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59"
        if not (backbone / "model.safetensors").is_file():
            # A cold machine need not run or modify the original five-metric setup.
            from huggingface_hub import HfApi, hf_hub_download
            backbone = ASSET_ROOT / "weights" / "roberta_large"
            backbone.mkdir(parents=True, exist_ok=True)
            backbone_repo = "FacebookAI/roberta-large"
            backbone_rev = "722cf37b1afa9454edce342e7895e588b6ff1d59"
            names = HfApi(token=False).list_repo_files(backbone_repo, revision=backbone_rev, token=False)
            for name in names:
                if name in TOKENIZER_FILES or name == "model.safetensors":
                    recorder.record("backbone_download_start", filename=name, repository=backbone_repo, revision=backbone_rev)
                    hf_hub_download(backbone_repo, filename=name, revision=backbone_rev, local_dir=backbone, token=False)
        result["checkpoint_path"] = str(model_dir / "AlignScore-large.ckpt")
        result["checkpoint_dir"] = str(model_dir)
        result["model_dir"] = str(backbone)
        result["backbone_repository"] = "FacebookAI/roberta-large"
        result["backbone_revision"] = "722cf37b1afa9454edce342e7895e588b6ff1d59"
        result["file_hashes"].update(inventory(backbone))
    with LOCK:
        recorder.completed[method] = result
        atomic_json(recorder.output_dir / "assets_partial.json", {"status": "partial", "assets": recorder.completed})
    recorder.record("method_assets_complete", method=method)
    return method, result


def environment_audit(output_dir):
    modules = ["torch", "transformers", "numpy", "scipy", "pytorch_lightning", "torchmetrics", "pyemd",
               "spacy", "nltk", "datasets", "pyarrow", "sentencepiece", "safetensors", "sklearn", "prettytable"]
    checks = {}
    for name in modules:
        try:
            importlib.import_module(name)
            checks[name] = "passed"
        except Exception as exc:
            checks[name] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    import torch
    import spacy
    checks["spacy_en_core_web_sm"] = "passed" if spacy.load("en_core_web_sm").meta["version"] == "3.7.1" else "failed"
    value = {"status": "passed" if all(item == "passed" for item in checks.values()) else "failed",
             "created_at_utc": now(), "python": sys.version, "python_executable": sys.executable,
             "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
             "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())),
             "import_tests": checks, "requirements_core_sha256": sha256(Path(__file__).with_name("requirements-core.txt")),
             "requirements_lock_sha256": sha256(Path(__file__).with_name("requirements-lock.txt"))}
    atomic_json(output_dir / "extended_environment_lock.json", value)
    if value["status"] != "passed":
        raise RuntimeError("Isolated environment import tests failed; see extended_environment_lock.json")
    return value


def setup_lexical(recorder):
    import nltk
    destination = ASSET_ROOT / "nltk_data"
    destination.mkdir(parents=True, exist_ok=True)
    resources = ["wordnet", "punkt_tab", "omw-1.4"]
    for name in resources:
        recorder.record("nltk_resource_start", resource=name)
        if not nltk.download(name, download_dir=str(destination), quiet=True, raise_on_error=True):
            raise RuntimeError(f"NLTK download failed: {name}")
        recorder.record("nltk_resource_complete", resource=name)
    nltk.data.path.insert(0, str(destination))
    for name in ["corpora/wordnet.zip", "corpora/omw-1.4.zip", "tokenizers/punkt_tab/english/"]:
        nltk.data.find(name)
    source = ROOT / "LLM4DPCG/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--bleu/9e0985c1200e367cce45605ce0ecb5ede079894e0f24f54613fca08eeb8aff76"
    target = ASSET_ROOT / "sources" / "bleu"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("nmt_bleu.py", "tokenizer_13a.py", "bleu.py"):
        url, expected = BLEU_UPSTREAM[name]
        if (source / name).is_file() and sha256(source / name) == expected:
            shutil.copy2(source / name, target / name)
        else:
            data = fetch_bytes(url)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Pinned BLEU source hash mismatch: {name}")
            (target / name).write_bytes(data)
        if sha256(target / name) != expected:
            raise ValueError(f"Copied BLEU source hash mismatch: {name}")
    return {"nltk_data_dir": str(destination), "nltk_resources": resources, "nltk_file_hashes": inventory(destination),
            "bleu_source_dir": str(target), "lexical": {"file_hashes": {**inventory(target), **inventory(destination)},
              "source": "Pinned byte-identical Hugging Face evaluate / TensorFlow NMT BLEU implementation",
              "module_revision": source.name, "upstream_files": BLEU_UPSTREAM}}


def verify_completed_manifest(output_dir):
    """Keep completed bound manifests byte-identical on a preflight rerun."""
    manifest_path = output_dir / "assets.json"
    if not manifest_path.is_file():
        return False
    before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    if set(manifest["assets"]) != set(MODEL_REVISIONS):
        raise ValueError("Completed assets manifest method inventory changed")
    all_hashes = {**manifest.get("nltk_file_hashes", {}), **manifest["lexical"]["file_hashes"]}
    for method, asset in manifest["assets"].items():
        repository, revision, _ = MODEL_REVISIONS[method]
        if asset["repository"] != repository or asset["revision"] != revision or asset["source_commit"] != SOURCES[method][1]:
            raise ValueError(f"Completed {method} provenance differs from fixed setup")
        all_hashes.update(asset["file_hashes"])
    for path, expected in all_hashes.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f"Completed asset integrity mismatch: {path}")
    environment_path = Path(manifest["environment_lock_path"])
    if sha256(environment_path) != manifest["environment_lock_sha256"]:
        raise ValueError("Completed environment-lock hash mismatch")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    packages = dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()))
    if environment["packages"] != packages or environment["python_executable"] != sys.executable:
        raise ValueError("Installed environment differs from completed setup")
    for name, key in (("requirements-core.txt", "requirements_core_sha256"), ("requirements-lock.txt", "requirements_lock_sha256")):
        if sha256(Path(__file__).with_name(name)) != environment[key]:
            raise ValueError(f"Completed requirement hash mismatch: {name}")
    proof = {"status": "passed", "verified_at_utc": now(), "assets_manifest_sha256_before": before,
             "assets_manifest_sha256_after": sha256(manifest_path), "preserved_manifest_bytes": True,
             "environment_lock_sha256": sha256(environment_path), "verified_file_count": len(all_hashes),
             "inference_performed": False, "downloads_performed": False}
    atomic_json(output_dir / "assets_resume_verification.json", proof)
    print(json.dumps({"event": "existing_assets_verified", **proof}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-workers", type=int, default=3)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for key, value in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "HF_HUB_DISABLE_XET": "1", "HF_HUB_ETAG_TIMEOUT": "60", "HF_HUB_DOWNLOAD_TIMEOUT": "120"}.items():
        os.environ[key] = value
    recorder = Recorder(output_dir)
    try:
        if verify_completed_manifest(output_dir):
            return 0
        recorder.record("environment_audit_start")
        environment = environment_audit(output_dir)
        recorder.record("environment_audit_complete", python=platform.python_version())
        lexical = setup_lexical(recorder)
        methods = {}
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = {executor.submit(setup_method, method, recorder): method for method in MODEL_REVISIONS}
            for future in as_completed(futures):
                key, value = future.result()
                methods[key] = value
        manifest = {"schema_version": "1.0", "status": "complete", "created_at_utc": now(),
                    "asset_root": str(ASSET_ROOT), "assets": methods, **lexical,
                    "environment_lock_path": str(output_dir / "extended_environment_lock.json"),
                    "environment_lock_sha256": sha256(output_dir / "extended_environment_lock.json"),
                    "setup_script_sha256": sha256(Path(__file__)),
                    "download_log_path": str(recorder.path), "api_inference_requests": 0,
                    "training_assets_downloaded": False}
        atomic_json(output_dir / "assets.json", manifest)
        recorder.record("all_assets_complete", methods=sorted(methods), elapsed_seconds=time.perf_counter() - recorder.started)
        atomic_json(output_dir / "assets_progress.json", {"status": "complete", "updated_at_utc": now(),
                    "completed_methods": sorted(methods), "elapsed_seconds": time.perf_counter() - recorder.started,
                    "assets_manifest_sha256": sha256(output_dir / "assets.json")})
        return 0
    except Exception as exc:
        value = {"status": "failed", "updated_at_utc": now(), "error_type": type(exc).__name__, "error": str(exc),
                 "traceback": traceback.format_exc(), "completed_methods": sorted(recorder.completed)}
        atomic_json(output_dir / "assets_progress.json", value)
        print(json.dumps(value, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
