"""Verify the fixed paired experiments and write a Traditional Chinese comparison."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import math
from pathlib import Path

from prepare_inputs import digest, read
from protocol import evaluate_rows, atomic_json
from compare_metrics import score_metrics, compare_pair

KEYS = ("accuracy", "f1_weighted", "patenteval_rank_pair_tau_b", "discrete_spearman")
LABELS = ("Accuracy", "Weighted F1", "PatentEval rank-pair τ-b", "離散 Spearman ρ")
SOURCE = Path(__file__).parent


def require(value, message):
    if not value:
        raise ValueError(message)


def same(a, b, name="value"):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"{name}: keys differ")
        for key in a:
            same(a[key], b[key], name + "." + key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f"{name}: length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{name}[{i}]")
    elif isinstance(a, float):
        require(isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=0, abs_tol=1e-10), f"{name}: numeric difference")
    else:
        require(a == b, f"{name}: differs")


def check_predictions(rows, original):
    require(len(rows) == 184, "Expected 184 original test rows")
    for prediction, source in zip(rows, original):
        for key in ("comparison_id", "gold_id", "source_split", "source_row_index", "label"):
            require(prediction[key] == source[key], "Prediction source/order/label differs")
        a, b = prediction["score_a"], prediction["score_b"]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in (a, b)), "Invalid scores")
        delta = a - b
        require(abs(delta - prediction["delta"]) < 1e-12, "Delta differs from scores")
        expected = 0 if abs(delta) < 1e-4 else (1 if delta > 0 else -1)
        require(prediction["prediction"] == expected, "Strict fixed tie threshold violated")


def collect(output, baseline, data_dir):
    output, baseline, data_dir = [Path(p).resolve() for p in (output, baseline, data_dir)]
    config = read(output / "experiment_config.json")
    lock = read(output / "baseline_lock.json")
    require(digest(output / "baseline_lock.json") == config["baseline_experiment"]["lock_sha256"], "Baseline lock changed")
    files = [output / n for n in ("experiment_config.json", "baseline_lock.json", "dataset.json", "split_manifest.json", "input_verification.json", "preflight.json", "longformer_replay.json", "source_fingerprint.json", "tokenization.json", "base_model_provenance.json")]
    for name, expected in lock["files"].items():
        path = (baseline / name).resolve()
        require(path.is_relative_to(baseline) and digest(path) == expected, f"Locked Longformer artifact changed: {name}")
    for name, expected in lock["raw_source_sha256"].items():
        require(digest(data_dir / name) == expected, "Original public data changed")
    for name in ("dataset.json", "split_manifest.json"):
        require(digest(output / name) == digest(baseline / name), "Data/split differs between models")
    fingerprint = read(output / "source_fingerprint.json")
    for name, expected in fingerprint["files"].items():
        path = output / name if name in ("experiment_config.json", "dataset.json", "split_manifest.json", "baseline_lock.json") else SOURCE / name
        require(digest(path) == expected, "Frozen ModernBERT input/training code changed: " + name)
    calculated = hashlib.sha256(json.dumps(fingerprint["files"], sort_keys=True).encode()).hexdigest()
    require(calculated == fingerprint["fingerprint"], "ModernBERT fingerprint changed")
    gate, replay = read(output / "preflight.json"), read(output / "longformer_replay.json")
    require(gate["status"] == "passed" and gate["fingerprint"] == calculated and gate["precision"] == "float32"
            and gate["batch_comparisons"] == 4 and gate["batch_sequences"] == 8, "FP32 batch-four gate failed")
    require(gate["parameter_dtypes"] == ["torch.float32"] and gate["reference_compile"] is False
            and gate["save_reload_max_abs_error"] <= 1e-6, "Capability precision/reload guard failed")
    provenance = read(output / "base_model_provenance.json")
    require(provenance["model_id"] == config["base_model"]["model_id"] and provenance["revision"] == config["base_model"]["revision"], "Base model provenance differs")
    require(replay["status"] == "passed" and len(replay["checkpoints"]) == 4, "SDK replay gate failed")
    require(all(row["passed"] and row["class_predictions_identical"] and row["maximum_absolute_score_difference"] <= 1e-6
                for row in replay["checkpoints"]), "Longformer replay scores differ")
    tokens = read(output / "tokenization.json")
    require(tokens["truncated_unique_pairs"] == 0 and len(tokens["rows"]) == 1228, "Unexpected truncation/tokenization coverage")
    require(digest(output / "tokenization.json") == gate["tokenization_sha256"], "Tokenization changed after gate")
    for source, token in zip(read(output / "dataset.json"), tokens["rows"]):
        require(source["comparison_id"] == token["comparison_id"] and all(0 < token[name] <= 4096 for name in ("tokens_a", "tokens_b"))
                and token["truncated_a"] is False and token["truncated_b"] is False, "Per-row tokenizer record differs")
    old_config = read(baseline / "experiment_config.json")
    for key, value in old_config["training"].items():
        if key != "global_attention_mask":
            same(value, config["training"][key], "shared_training." + key)
    original = [r for r in read(output / "dataset.json") if r["source_split"] == "test"]
    rows, predictions = [], {}
    for model, folder, cfg, fp in (("longformer", baseline, old_config, config["baseline_experiment"]["fingerprint"]),
                                   ("modernbert", output, config, calculated)):
        for seed in (42, 43):
            result_path = folder / f"seed_{seed}" / "result.json"
            result = read(result_path)
            require(result["status"] == "complete" and result["seed"] == seed and result["fingerprint"] == fp, "Incomplete/unlocked training result")
            require([r["epoch"] for r in result["history"]] == list(range(1, 11)), "Ten epochs not completed")
            require(result["assumptions"] == cfg["assumptions_not_disclosed_by_authors"], "Recorded assumptions changed")
            if model == "modernbert":
                require(result["attention_implementation"] == gate["attention_implementation"] and result["reference_compile"] is False, "Training backend differs from gate")
            minimum = min(result["history"], key=lambda r: r["validation_loss"])["epoch"]
            files.append(result_path)
            for selection in ("final", "validation_best"):
                checkpoint = folder / f"seed_{seed}" / selection
                details = result["checkpoints"][selection]
                metadata = details["metadata"]
                require(metadata["seed"] == seed and metadata["epoch"] == (10 if selection == "final" else minimum), "Unprespecified checkpoint chosen")
                require(metadata["selection"] == ("prespecified final epoch" if selection == "final" else "minimum validation loss"), "Selection rule changed")
                require(metadata["fingerprint"] == fp and metadata["settings"] == cfg["training"], "Checkpoint settings differ")
                same(metadata, read(checkpoint / "reproduction_metadata.json"))
                require("model.safetensors" in details["files"], "Missing model weights")
                for name, expected in details["files"].items():
                    path = (checkpoint / name).resolve()
                    require(path.is_relative_to(checkpoint.resolve()) and digest(path) == expected, "Checkpoint file hash mismatch")
                    files.append(path)
                path = folder / f"seed_{seed}" / f"{selection}_test_predictions.json"
                predicted = read(path)
                check_predictions(predicted, original)
                same(evaluate_rows(predicted), details["metrics"], f"{model}.{seed}.{selection}.recorded_metrics")
                independent = score_metrics(predicted)
                for key in ("accuracy", "f1_weighted", "f1_macro", "tie_recall"):
                    same(independent[key], details["metrics"][key], f"{model}.{key}.independent")
                identifier = f"{model}_seed{seed}_{selection}"
                predictions[identifier] = predicted
                rows.append(dict(condition=identifier, model=model, seed=seed, checkpoint=selection,
                                 epoch=metadata["epoch"], role="primary" if seed == 42 and selection == "final" else "prespecified supplement",
                                 metrics=independent, train_seconds=result["elapsed_seconds"], peak_vram_bytes=result["peak_vram_bytes"]))
                files.append(path)
    comparisons = {str(seed): compare_pair(predictions[f"longformer_seed{seed}_final"], predictions[f"modernbert_seed{seed}_final"],
                                          bootstrap=config["evaluation"]["bootstrap"]["replicates"], seed=42, data_dir=data_dir) for seed in (42, 43)}
    return dict(config=config, fingerprint=calculated, rows=rows, comparisons=comparisons, predictions=predictions,
                files=files, replay=replay, preflight=gate, tokenization=tokens, baseline_lock=lock)


def fmt(value, percentage=False):
    if value is None:
        return "未定義"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def describe(comparison):
    higher, lower, positive, negative, uncertain = [], [], [], [], []
    for key, label in zip(KEYS, LABELS):
        delta = comparison["difference"][key]
        if delta is not None and delta > 0:
            higher.append(label)
        elif delta is not None and delta < 0:
            lower.append(label)
        item = comparison["bootstrap"]["metrics"][key]
        bounds = item["ci95"]
        if item["eligible_for_directional_interpretation"] and bounds is not None and bounds[0] > 0:
            positive.append(label)
        elif item["eligible_for_directional_interpretation"] and bounds is not None and bounds[1] < 0:
            negative.append(label)
        else:
            uncertain.append(label)
    sentences = []
    for values, phrase in ((higher, "點估計提高"), (lower, "點估計降低"), (positive, "群組bootstrap區間全高於0"),
                           (negative, "群組bootstrap區間全低於0"), (uncertain, "尚無明確方向證據（區間含0或有效抽樣不足）")):
        if values:
            sentences.append("、".join(values) + phrase)
    return "；".join(sentences) + "。"


def write_plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, label, target in zip(axes.flat, KEYS, ("Accuracy", "Weighted F1 (inferred paper averaging)", "PatentEval rank-pair tau-b", "Discrete Spearman rho"), (.565, .574, .477, .602)):
        for shift, model, color in ((-.18, "longformer", "#3b82b6"), (.18, "modernbert", "#db7e28")):
            values = [next(r for r in rows if r["model"] == model and r["seed"] == seed and r["checkpoint"] == "final")["metrics"][key] for seed in (42, 43)]
            axis.bar(np.arange(2) + shift, [float("nan") if v is None else v for v in values], .36, label=model, color=color)
        axis.axhline(target, color="#666", linestyle="--", label="Paper reference")
        axis.set_xticks([0, 1], ["Seed 42 (primary)", "Seed 43 (sensitivity)"])
        axis.set_ylim(0 if key in KEYS[:2] else -1, 1)
        axis.set_title(label)
        axis.legend(fontsize=8)
    fig.suptitle("Original public split; final epoch 10; no test-based selection")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)


def write_html(output, result):
    main, supplementary, diagnostic_rows = [], [], []
    for row in result["rows"]:
        cells = [f"{row['model']} · seed {row['seed']} · {row['checkpoint']} (epoch {row['epoch']})"]
        cells.extend(fmt(row["metrics"][key], key in KEYS[:2]) for key in KEYS)
        cells.extend((f"{row['train_seconds']/60:.1f}", f"{row['peak_vram_bytes']/2**30:.2f}"))
        target = main if row["checkpoint"] == "final" else supplementary
        target.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in cells) + "</tr>")
        if row["checkpoint"] == "final":
            matrix = row["metrics"]["confusion_matrix"]
            tied = sum(matrix[1])
            ordered = sum(matrix[0]) + sum(matrix[2])
            ordered_correct = matrix[0][0] + matrix[2][2]
            values = (f"{row['model']} · seed {row['seed']}", fmt(row["metrics"]["f1_macro"], True),
                      fmt(row["metrics"]["tie_recall"], True), f"{matrix[1][1]}/{tied}", f"{ordered_correct}/{ordered}")
            diagnostic_rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    ci_rows = []
    interpretations = {"positive_in_this_fixed_seed_comparison": "此固定種子有正向差異證據", "negative_in_this_fixed_seed_comparison": "此固定種子有負向差異證據",
                       "interval_includes_zero": "區間含0，尚無明確方向證據", "insufficient_defined_resamples": "有效抽樣不足95%", "point_estimate_undefined": "點估計未定義"}
    for seed in (42, 43):
        comparison = result["comparisons"][str(seed)]
        for key, label in zip(KEYS, LABELS):
            item = comparison["bootstrap"]["metrics"][key]
            bounds = item["ci95"]
            interval = "未定義" if bounds is None else f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"
            ci_rows.append(f"<tr><td>{seed}</td><td>{label}</td><td>{fmt(comparison['difference'][key])}</td><td>{interval}</td><td>{item['valid_resamples']}/10000</td><td>{html.escape(interpretations[item['interpretation']])}</td></tr>")
    header = "<tr><th>固定條件</th><th>Accuracy</th><th>Weighted F1</th><th>Rank-pair τ-b</th><th>離散 Spearman ρ</th><th>訓練分鐘</th><th>峰值 GiB</th></tr>"
    text = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ModernBERT 替換 Longformer：原始切分比較</title>
<style>body{{font:16px/1.8 system-ui,'Microsoft JhengHei',sans-serif;background:#f4f7fa;color:#172033;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}section{{background:white;padding:22px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dce4ec;text-align:left}}th{{background:#e8f0f8}}img{{max-width:100%}}.note{{padding:14px;background:#fff6df}}a{{color:#1759b0}}</style>
<main><h1>ModernBERT-base 替換 Longformer 的訓練比較</h1><p>完成時間（台北）：{result['created_at_taipei']}。範圍：英文請求項整體品質 Quality；原始公開切分。</p>
<section><h2>實際結論</h2><p><strong>主要 seed42：</strong>{html.escape(result['summaries']['42'])}</p><p><strong>敏感度 seed43：</strong>{html.escape(result['summaries']['43'])}</p><p class="note">主要比較固定為 seed42 的第10輪 ModernBERT 減去上一輪同種子的 Longformer；seed43 是預先指定的敏感度補充。沒有根據 test 選種子、輪數、平手門檻或最佳指標。</p><p>上一輪 Longformer 的 Accuracy 高於論文，但 F1 與 Spearman 較低；這不代表已全面超越或精確重現論文。本報告逐項呈現實際差值及群組 bootstrap 區間。</p></section>
<section><h2>相同資料與訓練規則</h2><p>直接逐位元組沿用前輪 dataset 與 split：939訓練、105驗證、184測試，共20種參考請求項。兩模型均全參數微調、FP32、每批4筆比較、10輪、AdamW lr=5e-6、weight decay=.01、m=.1、n=.01；無scheduler/warmup/梯度裁切。最長4096 tokens，原生tokenizer文字配對、動態padding至512倍數；本輪沒有截斷原文。</p>
<p>Longformer保留既有預訓練pooler；ModernBERT沒有該層，因此新增768→768 Dense+Tanh，再接Linear→Sigmoid，pooler以N(0,.02)/zero初始化。評分頭結構等價，但初始化來源不同；結果評估的是實際替換方案，不能只歸因於純架構。</p>
<p>ModernBERT使用固定revision <code>{result['settings']['base_model']['revision']}</code>、Transformers4.50.3、SDPA優先且reference_compile=False；實際預檢選用 <strong>{result['preflight']['attention_implementation']}</strong>。前輪Longformer訓練為4.46.2，新環境已對其四checkpoint重評估，全部分數誤差≤1e-6、分類完全一致；這是推論相容性檢查，不能消除訓練SDK版本差異。</p></section>
<section><h2>主要與敏感度結果：固定第10輪</h2><div style="overflow:auto"><table>{header}<tr><td>論文文獻參考</td><td>56.50%</td><td>57.40%*</td><td>0.4770</td><td>0.6020</td><td>—</td><td>—</td></tr>{''.join(main)}</table></div><p>*論文未明載F1 averaging，本輪採六個baseline共同支持的weighted口徑。τ依引用PatentEval的rank-pair展平τ-b；ρ為固定三分類偏好與人類標籤的Spearman。平手嚴格採<code>abs(delta)&lt;1e-4</code>。所有四欄均來自同一條件的同一組預測。</p><img src="comparison.png" alt="兩模型在兩種子固定第10輪的四項比較"></section>
<section><h2>ModernBERT−Longformer 的成對差值</h2><p>各seed分開，以20個參考原文SHA群組配對重抽10000次（seed42），每次抽整群的全部比較並重新計算全域指標，取2.5%與97.5%百分位。Accuracy/F1差值使用0～1尺度，乘100為百分點。若undefined抽樣使有效比例&lt;95%，不作方向性推論；區間跨0表示尚無明確方向證據。</p><div style="overflow:auto"><table><tr><th>Seed</th><th>指標</th><th>點差值</th><th>95%區間</th><th>有效抽樣</th><th>判讀</th></tr>{''.join(ci_rows)}</table></div><p>這些區間衡量本資料群組重抽的不確定性，兩個seed沒有合併成368筆獨立樣本。只有兩個種子，不能據此聲稱跨種子穩定改善。</p></section>
<section><h2>驗證集選模補充</h2><p>以下僅依最低驗證loss選擇，同分較早輪；不替代主要第10輪。時間與峰值顯存均為該種子完整10輪訓練的run值，並非只訓練到補充權重所在輪數的耗時。</p><div style="overflow:auto"><table>{header}{''.join(supplementary)}</table></div></section>
<section><h2>平手辨識與分類診斷</h2><p class="note">四項主要指標提高，不代表平手類別也改善。下表分開顯示原始49筆人類平手與135筆勝負比較；門檻仍固定為1e-4。本輪ModernBERT對勝負排序表現較強，對平手的召回仍偏低，應與主要數字一起解讀。</p><div style="overflow:auto"><table><tr><th>第10輪條件</th><th>Macro F1</th><th>平手 recall</th><th>平手判對／總數</th><th>非平手判對／總數</th></tr>{''.join(diagnostic_rows)}</table></div><p>平手recall差值與其群組bootstrap區間也保留於完整統計JSON；此診斷沒有改動分類門檻、權重選擇或任何原始預測。</p></section>
<section><h2>限制與驗證</h2><p>原始公開train/test的20個參考群組及全部測試候選內容均有重疊，為對照前輪而保留；這不是未見專利的泛化測試。之前按參考群組分開的五折實驗另存，不能與此輪數字直接混用。只有Quality公開標註；不能外推到其他四維度或中文請求項。margin、seed等未公布設定沿用前輪預設，沒有藉此聲稱重現作者完整protocol。</p>
<p>已檢查原始資料/分割雜湊、八組184筆預測順序與標籤、兩種子各10輪、八份正式checkpoint完整性、固定分類規則及獨立混淆矩陣統計。新模型先通過最長實際輸入的真FP32 batch4前向/反向/更新及存取權重一致性。</p>
<p><a href="comparison.json">完整統計與設定JSON</a> · <a href="comparison.csv">比較表CSV</a> · <a href="all_test_predictions.csv">逐筆預測CSV</a> · <a href="verify.json">執行驗證</a> · <a href="sha_manifest.json">檔案雜湊</a> · <a href="comparison.svg">向量圖</a></p></section>
<section><h2>原始來源</h2><p><a href="https://aclanthology.org/2025.acl-long.190.pdf">PatClaimEval論文</a> · <a href="https://huggingface.co/answerdotai/ModernBERT-base">ModernBERT官方模型卡</a> · <a href="https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/modernbert">所用版本官方API</a> · <a href="https://aclanthology.org/2024.naacl-long.147.pdf#page=9">引用評估方法</a> · <a href="https://github.com/ZoeYou/PatentEval/blob/d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427">引用官方程式</a></p></section></main></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")


def report(output, baseline, data_dir):
    output = Path(output).resolve()
    context = collect(output, baseline, data_dir)
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    result = dict(schema_version=1, status="complete", created_at_taipei=now, fingerprint=context["fingerprint"],
                  settings=context["config"], primary_comparison="seed42 final epoch10; ModernBERT minus fixed Longformer",
                  rows=context["rows"], comparisons=context["comparisons"], preflight=context["preflight"],
                  summaries={seed: describe(comparison) for seed, comparison in context["comparisons"].items()},
                  longformer_replay=context["replay"], no_test_set_tuning=True,
                  limitations=["original public split has content overlap", "only two seeds and 20 references", "different native tokenizers and pooler initialization", "Longformer training SDK 4.46.2 versus ModernBERT 4.50.3; inference replay passed", "Quality English only"],
                  evaluation_source_sha256={name: digest(SOURCE / name) for name in ("report.py", "compare_metrics.py")})
    atomic_json(output / "comparison.json", result)
    records = [{k: r[k] for k in ("condition", "model", "seed", "checkpoint", "epoch", "role", "train_seconds", "peak_vram_bytes")}
               | {key: r["metrics"][key] for key in (*KEYS, "f1_macro", "tie_recall")} for r in context["rows"]]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    prediction_rows = [dict(condition=name, **row) for name, rows in context["predictions"].items() for row in rows]
    fields = list(dict.fromkeys(key for row in prediction_rows for key in row))
    with (output / "all_test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(prediction_rows)
    write_plot(output, context["rows"])
    write_html(output, result)
    paths = set(context["files"] + [output / name for name in ("comparison.json", "comparison.csv", "all_test_predictions.csv", "comparison.png", "comparison.svg", "report.html")])
    manifest = dict(algorithm="sha256", files=[dict(path=str(p.relative_to(output)) if p.is_relative_to(output) else "baseline/" + str(p.relative_to(Path(baseline).resolve())),
                                                  size_bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(paths)])
    atomic_json(output / "sha_manifest.json", manifest)
    verified = dict(status="passed", execution_verified=True, completed_seeds=[42, 43], n_conditions=8,
                    n_test_rows_per_condition=184, n_new_checkpoints=4, n_fixed_baseline_checkpoints=4,
                    sha_manifest_sha256=digest(output / "sha_manifest.json"), fingerprint=context["fingerprint"],
                    checked_at_taipei=now, inference_sdk_replay="passed", paper_protocol_reproduced=False)
    atomic_json(output / "verify.json", verified)
    return verified


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(report(args.output_dir, args.baseline_dir, args.data_dir)))
