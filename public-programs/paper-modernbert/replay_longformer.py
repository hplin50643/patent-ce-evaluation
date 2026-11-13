"""Gate SDK compatibility by replaying fixed Longformer checkpoints in the new image."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from author_longformer import ContrastiveModel
from prepare_inputs import read, digest


def replay(baseline, output):
    settings = read(output / "experiment_config.json")
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("V100 GPU required")
    torch.backends.cuda.matmul.allow_tf32 = False
    original = [r for r in read(output / "dataset.json") if r["source_split"] == "test"]
    tolerance = 1e-6
    entries, reused, started = [], {}, time.monotonic()
    for seed in (42, 43):
        for selection in ("final", "validation_best"):
            checkpoint = baseline / f"seed_{seed}" / selection
            weight_sha = digest(checkpoint / "model.safetensors")
            expected = read(baseline / f"seed_{seed}" / f"{selection}_test_predictions.json")
            if len(expected) != 184:
                raise ValueError("Incomplete baseline predictions")
            if weight_sha not in reused:
                model = ContrastiveModel.from_pretrained(checkpoint, local_files_only=True, attn_implementation="eager").float().cuda().eval()
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
                scores = {}
                with torch.inference_mode():
                    for row in original:
                        for key in ("A", "B"):
                            pair = (row["gold_claim"], row[key])
                            if pair not in scores:
                                encoded = tokenizer(*pair, truncation=False, return_token_type_ids=False)
                                if len(encoded["input_ids"]) > 4096:
                                    raise ValueError("Baseline source length changed")
                                inputs = tokenizer.pad([encoded], padding=True, pad_to_multiple_of=512, return_tensors="pt")
                                inputs = {k: v.cuda() for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
                                scores[pair] = float(model.compute_score(**inputs).reshape(-1)[0])
                reused[weight_sha] = scores
                del model
                torch.cuda.empty_cache()
            scores = reused[weight_sha]
            maximum, classes_equal, actual = 0., True, []
            for row, before in zip(original, expected):
                for name in ("comparison_id", "gold_id", "label", "source_row_index"):
                    if row[name] != before[name]:
                        raise ValueError("Baseline prediction identity/order changed")
                a, b = scores[(row["gold_claim"], row["A"])], scores[(row["gold_claim"], row["B"])]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a, b)):
                    raise ValueError("Nonfinite/out-of-range replay score")
                delta = a - b
                predicted = 0 if abs(delta) < 1e-4 else (1 if delta > 0 else -1)
                maximum = max(maximum, abs(a - before["score_a"]), abs(b - before["score_b"]))
                classes_equal &= predicted == before["prediction"]
                actual.append(dict(comparison_id=row["comparison_id"], score_a=a, score_b=b, delta=delta, prediction=predicted))
            entry = dict(seed=seed, checkpoint=selection, weights_sha256=weight_sha, n=184,
                         maximum_absolute_score_difference=maximum, class_predictions_identical=classes_equal,
                         passed=maximum <= tolerance and classes_equal)
            entries.append(entry)
            (output / f"longformer_replay_seed{seed}_{selection}.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
    result = dict(status="passed" if all(e["passed"] for e in entries) else "failed", score_absolute_tolerance=tolerance,
                  precision="float32", model_eval=True, threshold=0.0001, tie_policy="strict abs(delta)<epsilon",
                  torch=torch.__version__, transformers=__import__("transformers").__version__,
                  checkpoints=entries, unique_weight_sets_replayed=len(reused), elapsed_seconds=time.monotonic() - started,
                  scope="Inference compatibility only; original Longformer training used Transformers 4.46.2; ModernBERT uses 4.50.3")
    (output / "longformer_replay.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise ValueError("Longformer replay compatibility failed; do not start ModernBERT training")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(replay(args.baseline_dir, args.output_dir)))
