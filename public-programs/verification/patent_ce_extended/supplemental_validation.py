#!/usr/bin/env python3
"""Validate shortest/median real samples after the complete, frozen full run.

No full-result/checkpoint is modified. Every worker invocation runs sequentially
in a new process, keeping the fixed CPU settings and the complete training IDF
population. Re-running uses the worker's independently validated resume path.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location("supplemental_frozen_worker", Path(__file__).with_name("worker.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def select_samples(data: dict) -> tuple[list[dict], dict]:
    rows = [(len(p["gold_claim"])+len(p["candidate"]), i, p) for i,p in enumerate(data["pairs"])]
    median = statistics.median(row[0] for row in rows)
    shortest = min(rows, key=lambda row: (row[0], row[1]))
    middle = min(rows, key=lambda row: (abs(row[0]-median), row[1]))
    if shortest[2]["pair_id"] == middle[2]["pair_id"]:
        raise ValueError("Two distinct shortest/median samples are required")
    descriptions = []
    for name, row in (("shortest",shortest),("closest_to_median",middle)):
        length,index,pair = row
        descriptions.append({"role":name,"pair_id":pair["pair_id"],"original_pair_index_zero_based":index,
                             "combined_characters":length,"gold_characters":len(pair["gold_claim"]),
                             "candidate_characters":len(pair["candidate"]),
                             "gold_copy":pair["gold_claim"] == pair["candidate"]})
    return [shortest[2],middle[2]], {
        "length_definition":"len(raw gold_claim)+len(raw candidate), Unicode code points; no normalization",
        "rule":"minimum combined length and minimum absolute distance to the population median; ties use original pair order",
        "population_pairs":len(rows),"median_combined_characters":median,"samples":descriptions}


def completed_full_results(out: Path, data: dict, input_sha: str, worker_sha: str, environment_sha: str) -> dict:
    expected = {p["pair_id"] for p in data["pairs"]}
    stages = {}
    for method in w.METHODS:
        path = out/f"{method}_scores.json"
        if not path.is_file():
            raise ValueError("All eight full methods must finish before supplemental inference: "+method)
        stage = w.load_json(path)
        if (stage.get("complete") is not True or stage.get("smoke") is not False or
            stage.get("input_sha256") != input_sha or stage.get("worker_sha256") != worker_sha or
            stage.get("environment_sha256") != environment_sha or set(stage.get("scores",{})) != expected):
            raise ValueError("Incomplete or differently bound full result: "+method)
        for result in stage["scores"].values():
            w.validate_result(method,result)
        proof = w.load_json(out/f"{method}_resume_verification.json")
        if (proof.get("status") != "passed" or proof.get("inference_calls") != 0 or
            proof.get("pairs") != len(expected) or proof.get("original_output_sha256") != w.file_hash(path) or
            proof.get("scores_identical") is not True or proof.get("original_output_unchanged") is not True or
            proof.get("preserved_first_run_elapsed_seconds") != stage["elapsed_seconds"]):
            raise ValueError("Invalid full no-inference resume proof: "+method)
        stages[method] = stage
    return stages


def compare_scores(method: str, formal: dict, supplemental: dict, identifiers: set[str]) -> dict:
    if set(supplemental["scores"]) != identifiers:
        raise ValueError("Supplemental pair coverage mismatch")
    for field in ("worker_sha256","assets_sha256","environment_sha256"):
        if supplemental.get(field) != formal.get(field):
            raise ValueError("Supplemental/full signature mismatch: "+field)
    left = {k:v for k,v in formal["settings"].items() if k != "smoke"}
    right = {k:v for k,v in supplemental["settings"].items() if k != "smoke"}
    if left != right:
        raise ValueError("Supplemental scoring settings differ from formal settings")
    validation = supplemental.get("smoke_validation",{})
    comparisons = validation.get("comparisons",[])
    if validation.get("status") != "passed" or {r["pair_id"] for r in comparisons} != identifiers or len(comparisons) != len(identifiers):
        raise ValueError("Missing official supplemental comparisons")
    official = {row["pair_id"]:row for row in comparisons}
    results = []
    for pid in sorted(identifiers):
        actual = supplemental["scores"][pid]
        w.validate_result(method,actual)
        original = formal["scores"][pid]
        error = abs(actual["score"]-original["score"])
        official_error = abs(actual["score"]-official[pid]["official_reference"])
        if official[pid]["adapter"] != actual["score"]:
            raise ValueError("Official comparison does not match its saved adapter score")
        if error > w.TOLERANCE or official_error > w.TOLERANCE or not all(
                math.isfinite(n) for n in [error,official_error]):
            raise ValueError("Supplemental official/formal score parity failed: "+method)
        if actual["metadata"] != original["metadata"]:
            raise ValueError("Supplemental deterministic per-pair metadata differs: "+method)
        results.append({"pair_id":pid,"formal_score":original["score"],"supplemental_score":actual["score"],
                        "official_reference":official[pid]["official_reference"],
                        "formal_absolute_error":error,"official_absolute_error":official_error,
                        "metadata_identical":True})
    return {"status":"passed","signatures_identical":["worker_sha256","assets_sha256","environment_sha256"],
            "settings_identical_except_smoke":True,"absolute_tolerance":w.TOLERANCE,"comparisons":results}


def parse_resources(path: Path) -> dict:
    """Read actual GNU time output; missing fields never become invented zeros."""
    raw = path.read_text(encoding="utf-8")
    def field(label):
        match = re.search(r"^\s*"+re.escape(label)+r":\s*(\S+)\s*$",raw,re.MULTILINE)
        if not match:
            raise ValueError("Missing actual GNU time resource field: "+label)
        return match.group(1)
    parts = field("Elapsed (wall clock) time (h:mm:ss or m:ss)").split(":")
    if len(parts) not in (2,3):
        raise ValueError("Unrecognized actual GNU time wall-time format")
    wall = sum(float(value)*60**power for power,value in enumerate(reversed(parts)))
    values = {"peak_rss_kbytes":int(field("Maximum resident set size (kbytes)")),
              "wall_seconds":wall,"user_cpu_seconds":float(field("User time (seconds)")),
              "system_cpu_seconds":float(field("System time (seconds)")),
              "exit_status":int(field("Exit status"))}
    if values["peak_rss_kbytes"] <= 0 or values["exit_status"] != 0 or any(
            not math.isfinite(values[key]) or values[key] < 0 for key in
            ("wall_seconds","user_cpu_seconds","system_cpu_seconds")):
        raise ValueError("Invalid or unsuccessful measured subprocess resources")
    return {**values,"raw_path":str(path),"raw_sha256":w.file_hash(path)}


def combined_resources(previous: list[dict], current: dict, context: dict) -> dict:
    """Retain first-run measurements when a later helper call only resumes."""
    for measurement in previous:
        if measurement["context"] != context:
            raise ValueError("Prior resource measurement belongs to different scoring inputs")
        path = Path(measurement["raw_path"])
        parsed = parse_resources(path)
        if w.file_hash(path) != measurement["raw_sha256"] or any(
                parsed[key] != measurement[key] for key in
                ("peak_rss_kbytes","wall_seconds","user_cpu_seconds","system_cpu_seconds","exit_status")):
            raise ValueError("Prior resource measurement changed")
    records = [*previous,{**current,"context":context}]
    return {"measurement":"GNU /usr/bin/time -v",
            "scope":"entire independent supplemental worker subprocess: loading, two samples, official parity, and resume proof; not a full-224 peak",
            "rss_accounting":"GNU time Maximum resident set size (kbytes); not summed simultaneous process-tree RAM",
            "peak_rss_kbytes":max(row["peak_rss_kbytes"] for row in records),
            "initial_peak_rss_kbytes":records[0]["peak_rss_kbytes"],
            "initial_wall_seconds":records[0]["wall_seconds"],
            "latest_wall_seconds":records[-1]["wall_seconds"],
            "all_measured_invocations_wall_seconds":sum(row["wall_seconds"] for row in records),
            "measurements":records}


def run(input_path: Path, out: Path, assets_path: Path, threads=4, batch_size=1) -> dict:
    start = time.monotonic()
    if threads < 1 or batch_size < 1:
        raise ValueError("Threads and batch size must be positive integers")
    input_path,out,assets_path = input_path.resolve(),out.resolve(),assets_path.resolve()
    data = w.read_input(input_path)
    worker_path = Path(__file__).with_name("worker.py")
    worker_sha,input_sha = w.file_hash(worker_path),w.file_hash(input_path)
    environment_sha = w.value_hash(w.runtime_environment(threads))
    full = completed_full_results(out,data,input_sha,worker_sha,environment_sha)
    if any(stage["settings"]["batch_size"] != batch_size for stage in full.values()):
        raise ValueError("Requested supplemental batch size differs from the completed formal run")
    time_binary = Path("/usr/bin/time")
    if not time_binary.is_file():
        raise ValueError("GNU /usr/bin/time is required for actual supplemental peak-RSS measurement; run in WSL")
    samples,selection = select_samples(data)
    identifiers = {p["pair_id"] for p in samples}
    subdir = out/"supplemental_sample_validation"
    subdir.mkdir(parents=True,exist_ok=True)
    subset_path = subdir/"pairs.json"
    subset = {"schema_version":1,"validation_only":True,"parent_input_sha256":input_sha,
              "source_manifest":data.get("source_manifest",{}),"records":data["records"],"pairs":samples}
    if subset_path.exists():
        if w.load_json(subset_path) != subset:
            raise ValueError("Existing supplemental input differs; do not overwrite a bound checkpoint")
    else:
        w.atomic_json(subset_path,subset)
    summary_path = out/"supplemental_sample_validation.json"
    prior = w.load_json(summary_path) if summary_path.is_file() else {}
    if prior and (prior.get("original_input_sha256") != input_sha or
                  prior.get("worker_sha256") != worker_sha or
                  prior.get("environment_sha256") != environment_sha or
                  prior.get("subset_input_sha256") != w.file_hash(subset_path)):
        raise ValueError("Existing supplemental summary belongs to different inputs or execution identity")
    if set(prior.get("methods",{})) - set(w.METHODS):
        raise ValueError("Existing supplemental summary contains unknown methods")
    gold,candidates = w.training_populations(data)
    population_sha = w.value_hash({"gold":gold,"candidates":candidates})
    summary = {"schema_version":1,"status":"running","complete":False,"started_at":w.now(),
               "worker_sha256":worker_sha,"helper_sha256":w.file_hash(Path(__file__)),
               "environment_sha256":environment_sha,"original_input_sha256":input_sha,
               "assets_manifest_sha256":w.file_hash(assets_path),"assets_manifest_path":str(assets_path),
               "original_pairs_sha256":w.value_hash(data["pairs"]),
               "subset_input_sha256":w.file_hash(subset_path),"subset_pairs_sha256":w.value_hash(samples),
               "selection":selection,"full_methods_finished_before_supplemental":list(w.METHODS),
               "training_records_retained_exactly":subset["records"] == data["records"],
               "retained_records":len(data["records"]),
               "retained_training_records":sum(r.get("split")=="train" for r in data["records"]),
               "retained_records_sha256":w.value_hash(data["records"]),
               "training_records_sha256":w.value_hash([r for r in data["records"] if r.get("split")=="train"]),
               "mover_training_population":{"gold_documents":len(gold),"candidate_documents":len(candidates),
                                             "population_sha256":population_sha},
               "execution":"all methods sequential after formal completion; independent subset/checkpoints; frozen worker",
               "methods":copy.deepcopy(prior.get("methods",{}))}
    w.atomic_json(summary_path,summary)
    try:
        for method in w.METHODS:
            print(json.dumps({"event":"supplemental_start","method":method}),flush=True)
            command = [sys.executable,"-u",str(worker_path),"--method",method,"--input",str(subset_path),
                       "--output-dir",str(subdir),"--assets",str(assets_path),"--threads",str(threads),
                       "--batch-size",str(batch_size),"--smoke"]
            method_out = subdir/"smoke"/method
            result_path = method_out/f"{method}_scores.json"
            completed_before = result_path.is_file() and w.load_json(result_path).get("complete") is True
            resources_path = subdir/f"{method}_resources_{time.time_ns()}_{os.getpid()}.txt"
            timed_command = [str(time_binary),"-v","-o",str(resources_path),*command]
            with (subdir/f"{method}_console.log").open("w",encoding="utf-8") as log:
                subprocess.run(timed_command,stdout=log,stderr=subprocess.STDOUT,check=True)
            measured = {**parse_resources(resources_path),"measured_at":w.now(),
                        "completed_output_existed_before_invocation":completed_before,
                        "timed_command":timed_command}
            result = w.load_json(result_path)
            if (result.get("complete") is not True or result.get("smoke") is not True or
                result.get("input_sha256") != summary["subset_input_sha256"]):
                raise ValueError("Invalid supplemental worker output: "+method)
            compared = compare_scores(method,full[method],result,identifiers)
            resource_context = {"method":method,"worker_sha256":worker_sha,
                                "input_sha256":summary["subset_input_sha256"],
                                "assets_sha256":result["assets_sha256"],"environment_sha256":environment_sha}
            compared["resources"] = combined_resources(
                prior.get("methods",{}).get(method,{}).get("resources",{}).get("measurements",[]),
                measured,resource_context)
            resume_path = method_out/f"{method}_resume_verification.json"
            resume = w.load_json(resume_path) if resume_path.is_file() else {}
            if (resume.get("status") != "passed" or resume.get("inference_calls") != 0 or
                resume.get("pairs") != 2 or resume.get("original_output_sha256") != w.file_hash(result_path) or
                resume.get("scores_identical") is not True or resume.get("original_output_unchanged") is not True or
                resume.get("preserved_first_run_elapsed_seconds") != result["elapsed_seconds"]):
                # Completed worker results may predate a proof after interruption;
                # this explicit path is guarded against any inference in worker.run.
                resume = w.prove_resume(method,subset_path,method_out,assets_path,threads,batch_size,True)
            compared.update({"result_path":str(result_path),"result_sha256":w.file_hash(result_path),
                             "formal_result_sha256":w.file_hash(out/f"{method}_scores.json"),
                             "formal_resume_proof_sha256":w.file_hash(out/f"{method}_resume_verification.json"),
                             "assets_sha256":result["assets_sha256"],"environment_sha256":result["environment_sha256"],
                             "worker_sha256":result["worker_sha256"],"zero_inference_resume":resume,
                             "resume_proof_path":str(resume_path),"resume_proof_sha256":w.file_hash(resume_path)})
            if method == "moverscore_v2":
                original_idf = w.load_json(out/"moverscore_v2_idf.json")
                supplemental_idf_path = method_out/"moverscore_v2_idf.json"
                supplemental_idf = w.load_json(supplemental_idf_path)
                if original_idf != supplemental_idf or original_idf["population_sha256"] != population_sha:
                    raise ValueError("Supplemental IDF differs from formal training-only IDF")
                compared["idf"] = {"identical_to_formal":True,"population_sha256":population_sha,
                                   "formal_file_sha256":w.file_hash(out/"moverscore_v2_idf.json"),
                                   "supplemental_file_sha256":w.file_hash(supplemental_idf_path),
                                   "canonical_content_sha256":w.value_hash(original_idf)}
            if method == "alignscore":
                proof_path = method_out/"alignscore_optimization_parity.json"
                proof = w.load_json(proof_path)
                if (proof.get("status") != "passed" or proof.get("worker_sha256") != worker_sha or
                    proof.get("input_sha256") != summary["subset_input_sha256"] or
                    proof.get("assets_sha256") != result["assets_sha256"] or
                    proof.get("environment_sha256") != environment_sha or
                    proof.get("native_forward_unmodified") is not True or
                    proof.get("native_reference_bypasses_cache") is not True or
                    {r["pair_id"] for r in proof.get("subpair_comparisons",[])} != identifiers or
                    any(h["maximum_absolute_error"] > 1e-7 for row in proof["subpair_comparisons"] for h in row["heads"])):
                    raise ValueError("Invalid supplemental native/optimized three-head proof")
                compared["align_optimization_proof"] = {"path":str(proof_path),"sha256":w.file_hash(proof_path),
                                                        "all_head_tensors_exactly_equal":proof["all_head_tensors_exactly_equal"]}
            summary["methods"][method] = compared
            w.atomic_json(summary_path,summary)
            print(json.dumps({"event":"supplemental_complete","method":method,"pairs":2}),flush=True)
        summary.update(status="passed",complete=True,completed_at=w.now(),elapsed_seconds=time.monotonic()-start)
        w.atomic_json(summary_path,summary)
        return summary
    except BaseException as error:
        summary.update(status="failed",error=str(error),updated_at=w.now())
        w.atomic_json(summary_path,summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Threads and batch size must be positive integers")
    result = run(args.input,args.output_dir,args.assets,args.threads,args.batch_size)
    print(json.dumps({"status":result["status"],"methods":len(result["methods"]),"pairs_per_method":2}),flush=True)
