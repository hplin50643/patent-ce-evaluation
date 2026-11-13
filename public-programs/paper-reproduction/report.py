"""Verify and report six prespecified original-split reproduction conditions.

Completion means the declared experiment was executed and checked. It never
asserts recovery of the authors' undisclosed full training/evaluation protocol.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import sys

from prepare_original import FIXED_SOURCE_HASHES
from protocol import atomic_json, evaluate_rows, load_json, preference, sha
from reference_tau import patenteval_tau

CORE_FILES = ("run_reproduction.py", "author_modeling.py", "protocol.py", "prepare_original.py")
INPUT_FILES = ("experiment_config.json", "dataset.json", "split_manifest.json")
LABELS = [-1, 0, 1]
TARGETS = {"accuracy": .565, "f1_reported_averaging_unspecified": .574, "kendall_tau": .477, "spearman_rho": .602}
REFERENCE_PAPER = "https://aclanthology.org/2024.naacl-long.147.pdf#page=9"
REFERENCE_CODE = "https://github.com/ZoeYou/PatentEval/blob/d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427"
PAPER_COMPARISON_POLICY = {
    "revision": "reference_rank_pair_tau_comparison_v2",
    "vector": ["accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman"],
    "rank_encoding": {"1": [1, 2], "-1": [2, 1], "0": [1, 1]},
    "kendall_input": "Concatenate every two-item ranking in original row order, then global scipy tau-b",
    "spearman_input": "Global discrete {-1,0,1} preferences versus human labels",
    "reference_paper": REFERENCE_PAPER, "reference_code": REFERENCE_CODE,
    "evidence": "All four displayed metrics match six previously audited baselines at paper precision",
    "provenance_limit": "Reference-based reconstruction; PatClaimEval's actual evaluation script remains unpublished",
    "revision_scope": "Evaluation definition/presentation only; frozen model, data, threshold, seeds, epochs and training source are unchanged",
    "frozen_config_primary_correlation_note": "The earlier continuous definition remains in the frozen configuration for provenance and is retained as diagnostics, not the paper comparison vector",
}


def _assert(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    number = float(value)
    _assert(math.isfinite(number), f"{name} must be finite")
    return number


def _same(actual, recorded, location="metrics"):
    if isinstance(actual, dict):
        _assert(isinstance(recorded, dict), f"{location}: dictionary required")
        for key, value in actual.items():
            _assert(key in recorded, f"{location}.{key}: missing")
            _same(value, recorded[key], f"{location}.{key}")
    elif isinstance(actual, list):
        _assert(isinstance(recorded, list) and len(actual) == len(recorded), f"{location}: list mismatch")
        for index, value in enumerate(actual):
            _same(value, recorded[index], f"{location}[{index}]")
    elif isinstance(actual, float):
        _assert(math.isclose(actual, _finite(recorded, location), rel_tol=0, abs_tol=1e-12), f"{location}: recomputed value differs")
    else:
        _assert(actual == recorded, f"{location}: value differs")


def _locate_data(output, explicit=None):
    choices = [Path(explicit)] if explicit else [Path("/data"), output.parent / "data", Path(__file__).resolve().parents[2] / "data" / "Patent-CE"]
    for candidate in choices:
        if all((candidate / filename).is_file() for filename in FIXED_SOURCE_HASHES):
            return candidate.resolve()
    raise ValueError("Original raw data files are required for source-hash verification; provide --data-dir")


def _author_weights(config, explicit=None):
    spec = config["author_checkpoint"]
    candidates = [Path(explicit)] if explicit else []
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(cache) if cache else home / "hub"
    candidates.append(hub / ("models--" + spec["model_id"].replace("/", "--")) / "snapshots" / spec["revision"])
    candidates.append(Path(__file__).resolve().parents[1] / "models" / "PatClaimEval-Quality")
    for candidate in candidates:
        weights = candidate / "model.safetensors"
        if weights.is_file():
            _assert(sha(weights) == spec["weights_sha256"], "Released author weights do not match the pinned SHA256")
            return weights.resolve()
    raise ValueError("Pinned released author weights not available locally; provide --author-checkpoint-dir (no download is attempted)")


def _verify_inputs(output, data_dir):
    config, dataset, manifest, fingerprint = [load_json(output / name) for name in (*INPUT_FILES, "source_fingerprint.json")]
    _assert(config.get("schema_version") == 1, "Unsupported experiment configuration")
    _assert(config["training"]["seeds"] == [42, 43] and config["training"]["primary_seed"] == 42, "Prespecified seeds must remain 42 primary and 43 sensitivity")
    _assert(config["training"]["epochs"] == 10, "Prespecified final epoch must remain 10")
    _assert(config["evaluation"]["author_lengths"] == [2048, 4096], "Both prespecified author replay lengths are required")
    _assert(config["evaluation"]["epsilon"] == 1e-4, "The paper tie epsilon must remain 1e-4")
    _assert(config["evaluation"]["no_test_set_tuning"] is True and config["evaluation"]["no_selecting_seed_or_epoch_by_test"] is True, "Test-set selection is outside the declared protocol")
    _assert(config["paper_quality_targets"] == TARGETS, "Paper targets changed")
    files = {name: sha(output / name) for name in INPUT_FILES}
    files.update({name: sha(Path(__file__).parent / name) for name in CORE_FILES})
    _assert(fingerprint["files"] == files, "Frozen configuration, data, split or training source hashes differ")
    expected = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    _assert(fingerprint["fingerprint"] == expected, "source_fingerprint.json fingerprint is invalid")
    _assert(manifest["dataset_sha256"] == files["dataset.json"], "Prepared dataset SHA256 differs from manifest")
    _assert(isinstance(dataset, list) and len(dataset) == 1228, "All 1228 original records, including duplicates, are required")
    _assert(len({row["comparison_id"] for row in dataset}) == 1228, "Duplicate source-row identities")
    _assert(len({row["gold_id"] for row in dataset}) == 20, "Expected 20 original gold groups")
    raw_hashes = {}
    for split, size, offset in (("train", 1044, 0), ("test", 184, 1044)):
        filename = f"quality_{split}.json"
        raw_hashes[filename] = sha(data_dir / filename)
        _assert(raw_hashes[filename] == FIXED_SOURCE_HASHES[filename], f"{filename}: not the approved original source bytes")
        _assert(manifest["source_hashes"][filename] == raw_hashes[filename], f"{filename}: manifest source hash mismatch")
        raw = load_json(data_dir / filename)
        _assert(len(raw) == size, f"{filename}: original row count changed")
        for index, original in enumerate(raw):
            row = dataset[offset + index]
            _assert(row["comparison_id"] == f"{split}_{index:06d}" and row["source_split"] == split and row["source_row_index"] == index, "Original row identity/order changed")
            _assert(row["label"] == original["human_eval"]["Quality"] and type(row["label"]) is int and row["label"] in LABELS, "Original Quality label changed")
            _assert(all(row[key] == original[key] for key in ("gold_claim", "A", "B")), "Original claim strings changed")
            _assert(row["gold_id"] == hashlib.sha256(row["gold_claim"].encode("utf-8")).hexdigest(), "Gold hash differs from exact source text")
    _assert(len(manifest["folds"]) == 1, "Original-split reproduction requires one fixed split")
    fold = manifest["folds"][0]
    indices = list(range(1044))
    random.Random(42).shuffle(indices)
    expected_ids = {"train": [f"train_{index:06d}" for index in indices[105:]],
                    "validation": [f"train_{index:06d}" for index in indices[:105]],
                    "test": [f"test_{index:06d}" for index in range(184)]}
    for name, ids in expected_ids.items():
        _assert(fold[name]["comparison_ids"] == ids, f"{name} IDs/order differ from fixed 939/105/184 split")
    return config, dataset, manifest, expected, raw_hashes


def _check_predictions(path, original_test, epsilon, required_metadata=None):
    rows = load_json(path)
    _assert(isinstance(rows, list) and len(rows) == 184, f"{path.name}: expected all 184 predictions")
    _assert([row["comparison_id"] for row in rows] == [row["comparison_id"] for row in original_test], f"{path.name}: test IDs missing, duplicated or reordered")
    for row, original in zip(rows, original_test):
        for key in ("comparison_id", "gold_id", "source_split", "source_row_index", "label"):
            _assert(row[key] == original[key], f"{path.name}: {key} differs from original test")
        _assert(row["prediction"] == preference(_finite(row["delta"], "delta"), epsilon), f"{path.name}: prediction violates strict epsilon")
        for key, value in (required_metadata or {}).items():
            _assert(row.get(key) == value, f"{path.name}: {key} metadata differs")
    return evaluate_rows(rows, epsilon)


def _verify_weight_files(folder, recorded, metadata):
    _assert(isinstance(recorded, dict), "Checkpoint file manifest missing")
    _assert({"model.safetensors", "config.json", "reproduction_metadata.json"}.issubset(recorded), "Checkpoint hash manifest omits weights, config or reproduction metadata")
    verified = []
    for name, digest in recorded.items():
        path = (folder / name).resolve()
        _assert(path.is_relative_to(folder.resolve()) and path.is_file(), "Checkpoint file missing or outside checkpoint directory")
        _assert(path.stat().st_size > 0 and sha(path) == digest, f"Checkpoint file SHA256 mismatch: {name}")
        verified.append(path)
    _same(metadata, load_json(folder / "reproduction_metadata.json"), "checkpoint_metadata")
    return verified


def paper_compatible(rows, metrics):
    predicted = [preference(float(row["delta"]), metrics["epsilon"]) for row in rows]
    labels = [row["label"] for row in rows]
    return dict(accuracy=metrics["accuracy"], f1_weighted=metrics["f1_weighted"],
                patenteval_rank_pair_tau_b=patenteval_tau(predicted, labels),
                discrete_spearman=metrics["discrete"]["spearman_rho"])


def report_numeric_matches(values):
    values = {"accuracy": (values["accuracy"], TARGETS["accuracy"]),
              "f1_weighted": (values["f1_weighted"], TARGETS["f1_reported_averaging_unspecified"]),
              "patenteval_rank_pair_tau_b": (values["patenteval_rank_pair_tau_b"], TARGETS["kendall_tau"]),
              "discrete_spearman": (values["discrete_spearman"], TARGETS["spearman_rho"])}
    return {key: value is not None and abs(value - target) < .0005 + 1e-12 for key, (value, target) in values.items()}


def collect(output_dir, data_dir=None, author_checkpoint_dir=None):
    output = Path(output_dir).resolve()
    raw_dir = _locate_data(output, data_dir)
    config, dataset, manifest, fingerprint, raw_hashes = _verify_inputs(output, raw_dir)
    original_test = dataset[1044:]
    author_weight = _author_weights(config, author_checkpoint_dir)
    files = [output / name for name in (*INPUT_FILES, "source_fingerprint.json")]
    rows, errors = [], []
    for length in (2048, 4096):
        try:
            result_path, prediction_path = output / "author" / f"result_{length}.json", output / "author" / f"predictions_{length}.json"
            result = load_json(result_path)
            _assert(result["status"] == "complete" and result["fingerprint"] == fingerprint, "Released checkpoint result incomplete or fingerprint mismatch")
            _assert(result["model"] == config["author_checkpoint"], "Released checkpoint id/revision/weights hash differs")
            _assert(result["max_length"] == length and result["padding"] == "max_length" and result["precision"] == "float32" and result["model_eval"] is True, "Released checkpoint inference settings differ")
            _assert(result["test_used_for_training_or_selection"] is False, "Author replay must not train or select on test")
            _assert(result["inference_code_sha256"] == sha(Path(__file__).parent / "author_modeling.py"), "Released inference source hash differs")
            metrics = _check_predictions(prediction_path, original_test, config["evaluation"]["epsilon"], {"max_length": length, "checkpoint": "released_author"})
            _same(metrics, result["metrics"], f"author_{length}.metrics")
            comparison = paper_compatible(load_json(prediction_path), metrics)
            elapsed = _finite(result["elapsed_seconds"], "elapsed_seconds")
            _assert(elapsed >= 0, "Negative author replay runtime")
            rows.append(dict(condition=f"author_{length}", label=f"作者公開權重重評估・長度 {length}", role="released_checkpoint_replay", seed=None, checkpoint="released_author", epoch=None, max_length=length,
                             metrics=metrics, paper_compatible=comparison, report_numeric_matches=report_numeric_matches(comparison), elapsed_seconds=elapsed))
            files.extend((result_path, prediction_path))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"author_{length}: {exc}")
    for seed in (42, 43):
        folder = output / f"seed_{seed}"
        try:
            result_path = folder / "result.json"
            result = load_json(result_path)
            _assert(result["status"] == "complete" and result["fingerprint"] == fingerprint, "Training result incomplete or fingerprint mismatch")
            _assert(result["seed"] == seed and result["primary"] is (seed == 42), "Prespecified seed/primary metadata differs")
            _assert(result["assumptions"] == config["assumptions_not_disclosed_by_authors"], "Undisclosed assumption list differs from frozen configuration")
            history = result["history"]
            _assert([entry["epoch"] for entry in history] == list(range(1, 11)), "All ten completed epochs are required")
            for entry in history:
                _assert(_finite(entry["validation_loss"], "validation_loss") >= 0, "Invalid validation loss")
                _finite(entry["validation_accuracy"], "validation_accuracy")
                if entry["validation_continuous_tau"] is not None:
                    _finite(entry["validation_continuous_tau"], "validation tau")
            minimum = min(history, key=lambda entry: entry["validation_loss"])["epoch"]
            _assert(set(result["checkpoints"]) == {"final", "validation_best"}, "Both prespecified checkpoints are required")
            elapsed = _finite(result["elapsed_seconds"], "training elapsed_seconds")
            peak = _finite(result["peak_vram_bytes"], "peak_vram_bytes")
            _assert(elapsed >= 0 and peak >= 0, "Invalid runtime/memory metadata")
            files.append(result_path)
            for checkpoint in ("final", "validation_best"):
                details = result["checkpoints"][checkpoint]
                metadata = details["metadata"]
                epoch = 10 if checkpoint == "final" else minimum
                selection = "prespecified final epoch" if checkpoint == "final" else "minimum validation loss"
                _assert(metadata["seed"] == seed and metadata["epoch"] == epoch and metadata["selection"] == selection and metadata["fingerprint"] == fingerprint, "Checkpoint was not chosen by the prespecified final/validation policy")
                _assert(metadata["settings"] == config["training"], "Saved checkpoint settings differ")
                verified = _verify_weight_files(folder / checkpoint, details["files"], metadata)
                prediction_path = folder / f"{checkpoint}_test_predictions.json"
                metrics = _check_predictions(prediction_path, original_test, config["evaluation"]["epsilon"])
                _same(metrics, details["metrics"], f"seed_{seed}.{checkpoint}.metrics")
                comparison = paper_compatible(load_json(prediction_path), metrics)
                role = "primary" if seed == 42 and checkpoint == "final" else ("prespecified_validation_supplement" if seed == 42 else "prespecified_seed_sensitivity")
                if role == "primary":
                    label = "Seed 42・主要重訓：第 10 epoch"
                elif seed == 42:
                    label = f"Seed 42・驗證補充：loss 最低（epoch {epoch}）"
                else:
                    label = "Seed 43・敏感度補充：" + ("第 10 epoch" if checkpoint == "final" else f"驗證 loss 最低（epoch {epoch}）")
                rows.append(dict(condition=f"seed_{seed}_{checkpoint}", label=label, role=role, seed=seed, checkpoint=checkpoint, epoch=epoch, max_length=config["training"]["max_length"],
                                 metrics=metrics, paper_compatible=comparison, report_numeric_matches=report_numeric_matches(comparison), elapsed_seconds=elapsed, peak_vram_bytes=peak))
                files.extend(verified + [prediction_path])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"seed_{seed}: {exc}")
    expected_conditions = ["author_2048", "author_4096", "seed_42_final", "seed_42_validation_best", "seed_43_final", "seed_43_validation_best"]
    if [row["condition"] for row in rows] != expected_conditions:
        errors.append("All six prespecified conditions must be present in their fixed presentation order")
    context = dict(output=output, config=config, manifest=manifest, fingerprint=fingerprint, raw_source_sha256=raw_hashes,
                   author_weights={"path": str(author_weight), "sha256": config["author_checkpoint"]["weights_sha256"]},
                   rows=rows, files=files, errors=errors)
    return context


def _flatten(row):
    metrics = row["metrics"]
    result = {key: row[key] for key in ("condition", "role", "seed", "checkpoint", "epoch", "max_length")}
    result.update({f"paper_compatible_{key}": value for key, value in row["paper_compatible"].items()})
    result.update({key: metrics[key] for key in ("n", "accuracy", "f1_weighted", "f1_macro", "f1_micro", "tie_recall")})
    for mode in ("continuous", "discrete"):
        result.update({f"{mode}_{key}": metrics[mode][key] for key in ("kendall_tau_b", "kendall_tau_c", "spearman_rho")})
    result.update({f"report_match_{key}": value for key, value in row["report_numeric_matches"].items()})
    return result


def _figure(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    display_names = {"author_2048": "Released\n2048", "author_4096": "Released\n4096",
                     "seed_42_final": "Primary seed42\nfinal epoch10", "seed_42_validation_best": "Supplement seed42\nvalidation-best",
                     "seed_43_final": "Sensitivity seed43\nfinal epoch10", "seed_43_validation_best": "Sensitivity seed43\nvalidation-best"}
    names = [display_names[row["condition"]] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for shift, key, label in ((-.18, "accuracy", "Accuracy"), (.18, "f1_weighted", "Weighted F1 (inferred averaging)")):
        axes[0].bar(x + shift, [row["paper_compatible"][key] for row in rows], .36, label=label)
    axes[0].axhline(.565, color="#64748b", linestyle="--", label="Paper accuracy .565")
    axes[0].axhline(.574, color="#d97706", linestyle=":", label="Paper F1 .574")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("All six prespecified conditions; no test selection")
    axes[0].legend(fontsize=8)
    for shift, key, label in ((-.18, "patenteval_rank_pair_tau_b", "PatentEval pair-rank tau-b"), (.18, "discrete_spearman", "Discrete preference rho")):
        axes[1].bar(x + shift, [row["paper_compatible"][key] if row["paper_compatible"][key] is not None else float("nan") for row in rows], .36, label=label)
    axes[1].axhline(.477, color="#64748b", linestyle="--", label="Paper tau .477")
    axes[1].axhline(.602, color="#d97706", linestyle=":", label="Paper rho .602")
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Reference-supported comparison; author evaluator unavailable")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    path = output / "comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fmt(value, percent=False):
    return "未定義（常數輸出）" if value is None else (f"{value:.1%}" if percent else f"{value:.4f}")


def _html(output, result):
    rows, main_rows = [], []
    roles = {"released_checkpoint_replay": "公開權重重評估", "primary": "主要重訓",
             "prespecified_validation_supplement": "預先指定的驗證補充", "prespecified_seed_sensitivity": "預先指定的 seed 敏感度補充"}
    for row in result["rows"]:
        metric = row["metrics"]
        values = [f"<strong>{html.escape(row['label'])}</strong>" if row["role"] == "primary" else html.escape(row["label"])]
        values += [_fmt(metric[key], True) for key in ("accuracy", "f1_weighted", "f1_macro")]
        values += [_fmt(metric[mode][key]) for mode in ("continuous", "discrete") for key in ("kendall_tau_b", "kendall_tau_c", "spearman_rho")]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
        values = [f"<strong>{html.escape(row['label'])}</strong>" if row["role"] == "primary" else html.escape(row["label"])]
        values.append(roles[row["role"]])
        values += [_fmt(row["paper_compatible"][key], True) for key in ("accuracy", "f1_weighted")]
        values += [_fmt(row["paper_compatible"][key]) for key in ("patenteval_rank_pair_tau_b", "discrete_spearman")]
        main_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    assumptions = "".join(f"<li>{html.escape(value)}</li>" for value in result["assumptions_not_disclosed_by_authors"])
    overlap = result["published_split_audit"].get("original_train_test_overlap", {})
    matches = []
    for row in result["rows"]:
        matched = [name for name, value in row["report_numeric_matches"].items() if value]
        matches.append(f"<li>{html.escape(row['label'])}：{html.escape('、'.join(matched)) if matched else '沒有指標在論文顯示精度下吻合'}</li>")
    revision_note = '<a href="evaluation_protocol_revision.json">口徑修正記錄</a>' if result.get("evaluation_protocol_revision") is not None else "來源與編碼規則已記錄於結果 JSON"
    text = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PatClaimEval 論文數字重現檢查</title>
<style>body{{font:16px/1.8 system-ui,'Microsoft JhengHei',sans-serif;color:#172033;background:#f4f7fb;margin:0}}main{{max-width:1400px;margin:auto;padding:32px 24px}}section{{background:white;padding:22px;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #dce4ef;text-align:left}}th{{background:#eaf1fa}}.note{{background:#fff8e8;border-left:4px solid #d97706;padding:14px}}img{{max-width:100%}}a{{color:#1759b0}}</style>
<main><h1>PatClaimEval：依原始切分重做論文實驗</h1><p>驗證完成時間（台北）：{result['generated_at_taipei']}。範圍為公開 Quality 資料的原始 184 筆測試記錄；原始重複記錄與順序均保留。</p>
<section><h2>結果應如何理解</h2><p class="note">六個預先指定條件均已完成來源、權重雜湊與數值重算檢查。這代表本次實驗完整執行，不代表已還原作者未公開的完整 protocol，也不因某個數字吻合就宣稱重現成功。</p>
<p>主要條件固定為 seed 42 的第 10 epoch；同 seed 的驗證 loss 最低 checkpoint、seed 43 的兩個 checkpoint，以及作者公開權重的兩種長度全部列出。沒有依測試結果選 seed、epoch、門檻或最接近論文的條件，也沒有把不同條件的指標拼成同一列。</p>
<p>論文 Quality 目標：Accuracy <strong>56.5%</strong>、F1 <strong>57.4%</strong>、Kendall τ <strong>0.477</strong>、Spearman ρ <strong>0.602</strong>。主表使用同一組固定三分類預測計算 Accuracy、weighted F1、引用方法的 rank-pair τ-b，以及離散 Spearman。</p></section>
<section><h2>本輪調查：評估口徑修正</h2><p>依 PatClaimEval 所引用的 <a href="{REFERENCE_PAPER}">PatentEval（Zuo 等，2024）§7，第 9 頁</a>及其<a href="{REFERENCE_CODE}">固定 commit 官方程式</a>，每筆偏好 +1、−1、0 分別編碼成 [1,2]、[2,1]、[1,1]，將 N×2 排名依原順序展平後計算全域 τ-b。這與先前直接對 N 個連續分數差或三分類標籤計算 τ-b 的數字不同。</p>
<p>這個引用方法搭配 weighted F1、離散 Spearman，能讓六個既有 baseline 的四欄數字同時對上論文顯示精度，提供強的口徑證據；仍未取得 PatClaimEval 原作者實際評估程式，因此屬有參照來源支持的重建。這次只修正評估定義與呈現，模型、資料、門檻、seed、epoch 與訓練來源 fingerprint 均未改動。{revision_note}。</p></section>
<section><h2>六個條件：論文可比較口徑</h2><div style="overflow:auto"><table><tr><th>固定條件</th><th>條件角色</th><th>Accuracy</th><th>Weighted F1</th><th>PatentEval rank-pair τ-b</th><th>離散 Spearman ρ</th></tr><tr style="background:#fff8e8"><td><strong>論文表列目標</strong></td><td>文獻參考值，非本次 run</td><td>56.5%</td><td>57.4%（以 weighted 對照）</td><td>0.477</td><td>0.602</td></tr>{''.join(main_rows)}</table></div><img src="comparison.png" alt="同一固定預測向量的四項論文可比較數值"></section>
<section><h2>完整診斷：原始 global 統計全部保留</h2><p>下表保留每個條件的連續分數差及三分類標籤 τ-b／τ-c／ρ、macro F1。這些診斷不混入上方四欄主表；連續 τ 不再當作論文 τ 的重現數字。</p><div style="overflow:auto"><table><tr><th>固定條件</th><th>Accuracy</th><th>Weighted F1</th><th>Macro F1</th><th>連續 τ-b</th><th>連續 τ-c</th><th>連續 ρ</th><th>離散 τ-b</th><th>離散 τ-c</th><th>離散 ρ</th></tr>{''.join(rows)}</table></div></section>
<section><h2>主表四欄於顯示精度下的數字吻合</h2><ul>{''.join(matches)}</ul><p>僅檢查固定 paper_compatible 四欄是否落在論文三位小數的四捨五入區間；numeric match 不等於取得原作者程式或完整 protocol 重現。沒有從診斷表挑較接近的定義，也沒有替各指標挑不同模型條件。</p></section>
<section><h2>資料隔離與已知限制</h2><p>為重做論文，保留公開原始 train/test 切分與其內容重疊。本報告不是獨立參考群組的泛化測試；先前完成的 group CV 是另一項實驗，沒有改動。原始 train/test 共有 {overlap.get('gold_groups_shared', '未提供')} 個參考群組、{overlap.get('canonical_comparisons_shared', '未提供')} 組去方向化比較內容重疊。</p>
<p>目前公開資料與 checkpoint 只有整體品質 Quality；其餘特徵完整性、概念清晰度、術語一致性、邏輯連結四個維度無法用這份資料重現。作者完整訓練程式與多項設定未公開，因此以下假設需逐項保留。</p><ul>{assumptions}<li>F1 averaging 未明載；weighted 為 baseline 鑑識支持的推論。</li><li>Rank-pair τ-b 有引用官方程式及六個 baseline 支持；PatClaimEval 原作者實際評估程式仍未公開。</li></ul></section>
<section><h2>可檢查的輸出</h2><p>平手規則固定為 <code>abs(score_A-score_B) &lt; 0.0001</code>，全域評估原始 184 記錄；沒有另行擬合平手門檻。四個新 checkpoint 均核對實際權重檔的 SHA256，作者權重亦與鎖定 revision 的 SHA256 比對。</p>
<p><a href="comparison.csv">完整比較 CSV</a> · <a href="reproduction_results.json">全部結果與假設 JSON</a> · <a href="verify.json">執行驗證</a> · <a href="sha_manifest.json">SHA256 清單</a> · <a href="{html.escape(result['paper'])}">論文原文</a></p></section></main></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")


def report(output_dir, data_dir=None, author_checkpoint_dir=None):
    output = Path(output_dir).resolve()
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    try:
        context = collect(output, data_dir, author_checkpoint_dir)
        if context["errors"]:
            raise ValueError("; ".join(context["errors"]))
        config = context["config"]
        revision_path = output / "evaluation_protocol_revision.json"
        revision = load_json(revision_path) if revision_path.is_file() else None
        result = dict(schema_version=2, status="complete", generated_at_taipei=now, paper=config["paper"],
                      execution_verification="passed", paper_protocol_reproduced=False,
                      reproduction_verdict="not established: reference-supported evaluation reconstruction is not the unavailable author protocol",
                      primary_condition="seed_42_final", presentation_order="prespecified; no test-based selection", rows=context["rows"],
                      fingerprint=context["fingerprint"], raw_source_sha256=context["raw_source_sha256"], author_weights=context["author_weights"],
                      assumptions_not_disclosed_by_authors=config["assumptions_not_disclosed_by_authors"],
                      inference_only_assumptions=["F1 weighted averaging is inferred from baseline agreement, not explicitly documented", "Rank-pair tau follows cited PatentEval official code and six-baseline agreement; actual PatClaimEval evaluator is unavailable"],
                      evaluation_protocol=PAPER_COMPARISON_POLICY, evaluation_protocol_revision=revision,
                      evaluation_code_sha256={name: sha(Path(__file__).parent / name) for name in ("report.py", "reference_tau.py")},
                      available_dimension="Quality only", unavailable_dimensions=["Completeness", "Clarity", "Consistency", "Linkage"],
                      published_split_audit=context["manifest"]["audit"], paper_targets=TARGETS,
                      settings=config)
        atomic_json(output / "reproduction_results.json", result)
        records = [_flatten(row) for row in result["rows"]]
        with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        figure = _figure(output, result["rows"])
        _html(output, result)
        files = set(context["files"] + [figure, output / "reproduction_results.json", output / "comparison.csv", output / "report.html"])
        if revision_path.is_file():
            files.add(revision_path)
        diagnostics = output / "diagnostics"
        if diagnostics.is_dir():
            files.update(path for path in diagnostics.rglob("*") if path.is_file())
        manifest = {"algorithm": "sha256", "files": [{"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(files)],
                    "external_inputs": {"raw_source_sha256": context["raw_source_sha256"], "released_author_weights": context["author_weights"],
                                        "evaluation_code_sha256": result["evaluation_code_sha256"]}}
        atomic_json(output / "sha_manifest.json", manifest)
        verification = dict(status="complete", execution_verified=True, paper_protocol_reproduced=False, checked_at_taipei=now,
                            n_conditions=6, n_test_rows_per_condition=184, n_new_weight_checkpoints=4,
                            fingerprint=context["fingerprint"], raw_source_sha256=context["raw_source_sha256"],
                            author_weights=context["author_weights"], sha_manifest_sha256=sha(output / "sha_manifest.json"),
                            checks_passed=["original_source_hashes_and_exact_order_text_labels", "fixed_939_105_184_split", "frozen_input_and_training_code_fingerprint",
                                           "184_ordered_test_predictions_per_condition", "released_weights_sha256", "four_checkpoint_file_hashes",
                                           "final_epoch10_and_earliest_minimum_validation_loss_selection", "recomputed_metrics_for_all_six_conditions", "reference_rank_pair_paper_comparison_and_full_global_diagnostics", "fixed_presentation_order_no_test_selection"], errors=[])
        atomic_json(output / "verify.json", verification)
        return verification
    except (OSError, KeyError, TypeError, ValueError) as exc:
        failure = dict(status="incomplete", execution_verified=False, paper_protocol_reproduced=False, checked_at_taipei=now,
                       errors=[f"{type(exc).__name__}: {exc}"], expected_conditions=["author_2048", "author_4096", "seed_42_final", "seed_42_validation_best", "seed_43_final", "seed_43_validation_best"])
        atomic_json(output / "verify.json", failure)
        return failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--author-checkpoint-dir", type=Path)
    args = parser.parse_args(argv)
    result = report(args.output_dir, args.data_dir, args.author_checkpoint_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
