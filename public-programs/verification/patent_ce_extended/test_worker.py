"""Integrity and resume tests; no model downloads or neural inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("extended_worker", Path(__file__).with_name("worker.py"))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
supp_spec = importlib.util.spec_from_file_location("supplemental_validation", Path(__file__).with_name("supplemental_validation.py"))
supp = importlib.util.module_from_spec(supp_spec)
supp_spec.loader.exec_module(supp)


def pair(gold, candidate):
    return {"pair_id": w.raw_pair_id(gold, candidate), "gold_claim": gold, "candidate": candidate}


def result(score=.5):
    return {"score": score, "metadata": {"gold_tokens": 2, "candidate_tokens": 3, "truncated": False}}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "pairs.json"
        self.pairs = [pair(" device\n", "device"), pair("device", "device"), pair("another", "sample")]
        self.data = {"pairs": self.pairs, "records": [
            {"split": "train", "gold_claim": " raw ", "A": "a", "B": "b"},
            {"split": "train", "gold_claim": " raw ", "A": "a", "B": " c "},
            {"split": "test", "gold_claim": "leak", "A": "leak-a", "B": "leak-b"}]}
        w.atomic_json(self.input, self.data)
        resource = self.root / "source.py"
        resource.write_text("source version one", encoding="utf-8")
        lock = self.root / "environment.json"
        w.atomic_json(lock, {"created_at_utc": "yesterday", "packages": {"test": "1"}})
        self.assets = self.root / "assets.json"
        self.manifest = {"status": "complete", "created_at_utc": "yesterday", "nltk_data_dir": str(self.root),
                         "lexical": {"file_hashes": {str(resource): w.file_hash(resource)}},
                         "environment_lock_path": str(lock)}
        w.atomic_json(self.assets, self.manifest)
        self.out = self.root / "output"
        self.calls = []
        owner = self

        class FakeAdapter:
            provenance = {"scorer": "test-only"}

            def score(self, item):
                owner.calls.append(item["pair_id"])
                return result(.5)

            def official_reference(self, item):
                return .5

        self.factory = FakeAdapter
        self.environment_patch = patch.object(w, "runtime_environment", return_value={"versions": {"test": "1"}})
        self.environment_patch.start()
        self.configure_patch = patch.object(w, "configure")
        self.configure_patch.start()

    def tearDown(self):
        self.environment_patch.stop()
        self.configure_patch.stop()
        self.temporary.cleanup()

    def execute(self, **kwargs):
        with patch.object(w, "make_adapter", side_effect=lambda *a: self.factory()):
            return w.run("bleu1", self.input, self.out, self.assets, **kwargs)

    def test_raw_ids_and_training_dedup_preserve_whitespace_and_exclude_test(self):
        self.assertNotEqual(w.raw_pair_id(" device\n", "device"), w.raw_pair_id("device", "device"))
        self.assertEqual(w.training_populations(self.data), ([" raw "], ["a", "b", " c "]))
        self.assertEqual(w.read_input(self.input), self.data)

    def test_modified_raw_text_and_duplicate_pair_are_rejected(self):
        modified = copy.deepcopy(self.data)
        modified["pairs"][0]["gold_claim"] = "device"
        w.atomic_json(self.input, modified)
        with self.assertRaises(ValueError):
            w.read_input(self.input)
        modified = copy.deepcopy(self.data)
        modified["pairs"].append(modified["pairs"][0])
        w.atomic_json(self.input, modified)
        with self.assertRaises(ValueError):
            w.read_input(self.input)

    def test_strict_score_and_truncation_validation(self):
        for score in [float("nan"), float("inf"), -.01, 1.01, True]:
            with self.assertRaises(ValueError):
                w.validate_result("bleu1", result(score))
        w.validate_result("bartscore", result(-14.2))
        w.validate_result("simcse", result(-.4))
        bad = result()
        bad["metadata"]["truncated"] = None
        with self.assertRaises(ValueError):
            w.validate_result("bleu1", bad)

    def checkpoint_line(self, signatures=None):
        return {**(signatures or {"signature": "fixed"}), "pair_id": self.pairs[0]["pair_id"],
                "result": result(), "result_sha256": w.value_hash(result()), "elapsed_seconds": 1.25}

    def test_only_unterminated_checkpoint_tail_is_recovered(self):
        path = self.root / "checkpoint.jsonl"
        w.append_checkpoint(path, self.checkpoint_line())
        complete = path.read_bytes()
        with path.open("ab") as stream:
            stream.write(b'{"interrupted":')
        scores, elapsed = w.read_checkpoint(path, {"signature": "fixed"}, {p["pair_id"] for p in self.pairs}, "bleu1")
        self.assertEqual(path.read_bytes(), complete)
        self.assertEqual(len(scores), 1)
        self.assertEqual(elapsed, 1.25)

    def test_complete_corrupt_line_is_not_silently_discarded(self):
        path = self.root / "checkpoint.jsonl"
        path.write_bytes(b'{"invalid":}\n')
        before = path.read_bytes()
        with self.assertRaises(json.JSONDecodeError):
            w.read_checkpoint(path, {}, set(), "bleu1")
        self.assertEqual(before, path.read_bytes())

    def test_checkpoint_signature_checksum_and_duplicate_are_rejected(self):
        for change in ("signature", "checksum", "duplicate"):
            path = self.root / (change + ".jsonl")
            line = self.checkpoint_line()
            if change == "signature":
                line["signature"] = "other"
            elif change == "checksum":
                line["result"]["score"] = .9
            w.append_checkpoint(path, line)
            if change == "duplicate":
                w.append_checkpoint(path, line)
            with self.assertRaises(ValueError):
                w.read_checkpoint(path, {"signature": "fixed"}, {p["pair_id"] for p in self.pairs}, "bleu1")

    def test_complete_resume_executes_zero_inference_and_preserves_output(self):
        original = self.execute()
        self.assertEqual(len(self.calls), 3)
        path = self.out / "bleu1_scores.json"
        before = path.read_bytes()
        with patch.object(w, "make_adapter", side_effect=AssertionError("Inference prohibited")):
            resumed = w.run("bleu1", self.input, self.out, self.assets)
        self.assertEqual(original, resumed)
        self.assertEqual(before, path.read_bytes())
        proof = w.load_json(self.out / "bleu1_resume_verification.json")
        self.assertEqual(proof["inference_calls"], 0)
        self.assertEqual(proof["preserved_first_run_elapsed_seconds"], original["elapsed_seconds"])

    def test_partial_resume_scores_only_pending_pairs(self):
        owner = self

        class Interrupted(self.factory):
            def score(self, item):
                if len(owner.calls) == 1:
                    raise RuntimeError("simulated interruption")
                return super().score(item)

        normal = self.factory
        self.factory = Interrupted
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(self.calls, [self.pairs[0]["pair_id"]])
        self.factory = normal
        final = self.execute()
        self.assertEqual(self.calls, [p["pair_id"] for p in self.pairs])
        self.assertEqual(len(final["scores"]), 3)

    def test_resume_rejects_asset_bytes_changed_even_if_manifest_unchanged(self):
        self.execute()
        source = Path(next(iter(self.manifest["lexical"]["file_hashes"])))
        source.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Asset SHA-256 mismatch"):
            self.execute()

    def test_setup_timestamps_do_not_invalidate_semantic_resume(self):
        original = self.execute()
        self.manifest["created_at_utc"] = "tomorrow"
        self.manifest["download_log_path"] = "changed-log"
        w.atomic_json(self.assets, self.manifest)
        lock = Path(self.manifest["environment_lock_path"])
        document = w.load_json(lock)
        document["created_at_utc"] = "tomorrow"
        w.atomic_json(lock, document)
        self.assertEqual(original, self.execute())

    def test_package_change_invalidates_bound_resume(self):
        self.execute()
        lock = Path(self.manifest["environment_lock_path"])
        document = w.load_json(lock)
        document["packages"]["test"] = "2"
        w.atomic_json(lock, document)
        with self.assertRaisesRegex(ValueError, "assets_sha256 mismatch"):
            self.execute()

    def test_smoke_is_physically_separate_and_compares_official_entrypoint(self):
        final = self.execute(smoke=True)
        self.assertTrue((self.out / "smoke/bleu1/bleu1_scores.json").exists())
        self.assertFalse((self.out / "bleu1_scores.json").exists())
        self.assertEqual(final["smoke_validation"]["status"], "passed")
        self.assertTrue(all(row["absolute_error"] == 0 for row in final["smoke_validation"]["comparisons"]))

    def test_resume_only_never_initializes_model_for_missing_result(self):
        with patch.object(w, "make_adapter", side_effect=AssertionError("Inference prohibited")):
            with self.assertRaisesRegex(ValueError, "inference was prohibited"):
                w.run("bleu1", self.input, self.out, self.assets, resume_only=True)

    def test_individual_token_flags_include_special_tokens(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(range(len(text.split()) + 2))
        self.assertEqual(w.individual_metadata(Tokenizer(), "a b c", "d", 4),
                         {"gold_tokens": 5, "candidate_tokens": 3, "limit": 4,
                          "gold_truncated": True, "candidate_truncated": False,
                          "truncated": True, "token_counts_include_special_tokens": True})

    def test_exact_subpair_cache_reassembles_all_three_heads_and_raw_order(self):
        calls = []
        values = {(" raw ", "x"): 1., ("raw", "x"): 2., ("raw", "y"): 3.}
        def native(a, b):
            calls.append(list(zip(a, b)))
            base = [values[key] for key in zip(a, b)]
            return base, [v+10 for v in base], [[v+20, v+30, v+40] for v in base]
        memo = w.ExactHeadMemoizer(native, list, copy.deepcopy)
        first = memo([" raw ", "raw", " raw "], ["x", "x", "x"])
        self.assertEqual(first, ([1.,2.,1.], [11.,12.,11.], [[21.,31.,41.],[22.,32.,42.],[21.,31.,41.]]))
        second = memo(["raw", "raw", " raw "], ["y", "x", "x"])
        self.assertEqual(second[0], [3.,2.,1.])
        self.assertEqual(calls, [[(" raw ","x"),("raw","x")], [("raw","y")]])
        self.assertEqual(memo.diagnostics()["exact_cache_hits"], 3)
        self.assertEqual(memo.diagnostics()["unique_model_subpairs_inferred"], 3)

    def test_exact_cache_rejects_incomplete_native_heads(self):
        memo = w.ExactHeadMemoizer(lambda a,b: (None,[.2],[.3]), list, copy.deepcopy)
        with self.assertRaisesRegex(ValueError, "all three"):
            memo(["a"], ["b"])
        self.assertEqual(memo.diagnostics()["requested_model_subpairs"], 0)

    def test_align_smoke_reference_bypasses_cache_and_restores_it(self):
        import torch
        calls = []
        model = types.SimpleNamespace(inference=lambda a,b: (_ for _ in ()).throw(AssertionError("Cache used")))
        native_forward, optimized_forward = object(), object()
        model.model = types.SimpleNamespace(forward=optimized_forward)
        def native(a, b):
            calls.append((a,b,model.model.forward))
            return torch.tensor([.1]), torch.tensor([.2]), torch.tensor([[.75,.2,.05]])
        model.inference_per_example = lambda a,b: model.inference([a],[b])[2][0,0]
        adapter = w.Adapter.__new__(w.Adapter)
        adapter.method = "alignscore"
        adapter.scorer = types.SimpleNamespace(model=model)
        adapter.align_uncached_inference = native
        adapter.align_original_forward = native_forward
        adapter.align_optimized_forward = optimized_forward
        adapter.align_optimization_comparisons = []
        cache = model.inference
        with patch.object(w.time, "monotonic", side_effect=[100.,101.,102.,103.]):
            self.assertEqual(adapter.official_reference(pair("a","b")), .75)
        self.assertEqual(calls, [(["a"],["b"],native_forward),(["a"],["b"],optimized_forward)])
        self.assertIs(model.inference, cache)
        self.assertIs(model.model.forward, optimized_forward)
        self.assertTrue(all(h["torch_equal"] for h in adapter.align_optimization_comparisons[0]["heads"]))

    def test_supplemental_selects_actual_shortest_and_median_with_stable_tie(self):
        pairs = [pair("甲","乙"),pair(" aa ","bb"),pair("ccc","ccc"),pair("long","longer")]
        selected,description = supp.select_samples({"pairs":pairs})
        self.assertEqual(selected,[pairs[0],pairs[1]])
        self.assertEqual(description["median_combined_characters"],6)
        self.assertEqual(description["samples"][0]["combined_characters"],2)
        self.assertEqual(description["samples"][1]["original_pair_index_zero_based"],1)

    def test_supplemental_refuses_to_start_until_all_full_methods_complete(self):
        with patch.object(supp.subprocess,"run") as process:
            with self.assertRaisesRegex(ValueError,"All eight full methods"):
                supp.completed_full_results(self.root,self.data,"input","worker","environment")
            process.assert_not_called()

    def test_supplemental_checks_official_and_formal_values_and_raw_metadata(self):
        identifiers = {p["pair_id"] for p in self.pairs[:2]}
        formal = {"settings":{"smoke":False,"threads":4},"worker_sha256":"worker",
                  "assets_sha256":"assets","environment_sha256":"env",
                  "scores":{pid:result(.5) for pid in identifiers}}
        extra = copy.deepcopy(formal)
        extra["settings"]["smoke"] = True
        extra["smoke_validation"] = {"status":"passed","comparisons":[
            {"pair_id":pid,"adapter":.5,"official_reference":.5} for pid in identifiers]}
        self.assertEqual(supp.compare_scores("bleu1",formal,extra,identifiers)["status"],"passed")
        bad = copy.deepcopy(extra)
        bad["smoke_validation"]["comparisons"][0]["official_reference"] = .9
        with self.assertRaisesRegex(ValueError,"parity failed"):
            supp.compare_scores("bleu1",formal,bad,identifiers)
        bad = copy.deepcopy(extra)
        bad["scores"][next(iter(identifiers))]["metadata"]["truncated"] = True
        with self.assertRaisesRegex(ValueError,"metadata differs"):
            supp.compare_scores("bleu1",formal,bad,identifiers)

    def test_resource_parser_requires_actual_fields_and_converts_clock(self):
        path = self.root/"resources.txt"
        path.write_text("User time (seconds): 3.2\nSystem time (seconds): 0.4\n"
                        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.4\n"
                        "Maximum resident set size (kbytes): 12345\nExit status: 0\n",encoding="utf-8")
        parsed = supp.parse_resources(path)
        self.assertAlmostEqual(parsed["wall_seconds"],3723.4)
        self.assertEqual(parsed["peak_rss_kbytes"],12345)
        path.write_text("Exit status: 0\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Missing actual"):
            supp.parse_resources(path)

    def test_resume_resource_measurement_preserves_initial_peak_and_detects_tampering(self):
        context = {"method":"test","input_sha256":"input"}
        def measurement(name,rss):
            path = self.root/name
            path.write_text("User time (seconds): 1.0\nSystem time (seconds): 0.1\n"
                            "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.2\n"
                            f"Maximum resident set size (kbytes): {rss}\nExit status: 0\n",encoding="utf-8")
            return supp.parse_resources(path)
        initial = supp.combined_resources([],measurement("first.txt",10000),context)
        resumed = supp.combined_resources(initial["measurements"],measurement("second.txt",1000),context)
        self.assertEqual(resumed["peak_rss_kbytes"],10000)
        self.assertEqual(resumed["initial_peak_rss_kbytes"],10000)
        self.assertEqual(len(resumed["measurements"]),2)
        (self.root/"first.txt").write_text("tampered",encoding="utf-8")
        with self.assertRaises(ValueError):
            supp.combined_resources(initial["measurements"],measurement("third.txt",1000),context)

    def test_failed_helper_rerun_preserves_later_methods_initial_resources(self):
        self.out.mkdir()
        selected,_ = supp.select_samples(self.data)
        source_sha = w.file_hash(self.input)
        worker_sha = w.file_hash(Path(__file__).with_name("worker.py"))
        environment = {"fixed":"fixture"}
        subset_dir = self.out/"supplemental_sample_validation"
        subset_dir.mkdir()
        subset_path = subset_dir/"pairs.json"
        w.atomic_json(subset_path,{"schema_version":1,"validation_only":True,"parent_input_sha256":source_sha,
                                  "source_manifest":{},"records":self.data["records"],"pairs":selected})
        retained = {"resources":{"measurements":[{"first_run_resource":"must survive interrupted replay"}]}}
        summary_path = self.out/"supplemental_sample_validation.json"
        w.atomic_json(summary_path,{"original_input_sha256":source_sha,"worker_sha256":worker_sha,
                                   "environment_sha256":w.value_hash(environment),
                                   "subset_input_sha256":w.file_hash(subset_path),"methods":{"simcse":retained}})
        original_is_file = Path.is_file
        def is_file(path):
            return path == Path("/usr/bin/time") or original_is_file(path)
        with patch.object(supp.w,"runtime_environment",return_value=environment), \
             patch.object(supp,"completed_full_results",return_value={m:{"settings":{"batch_size":1}} for m in w.METHODS}), \
             patch.object(Path,"is_file",is_file), \
             patch.object(supp.subprocess,"run",side_effect=RuntimeError("simulated replay failure")):
            with self.assertRaisesRegex(RuntimeError,"simulated replay failure"):
                supp.run(self.input,self.out,self.assets)
        after = w.load_json(summary_path)
        self.assertEqual(after["status"],"failed")
        self.assertEqual(after["methods"]["simcse"],retained)


if __name__ == "__main__":
    unittest.main()
