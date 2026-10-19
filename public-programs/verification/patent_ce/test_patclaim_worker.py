"""Focused provenance and crash-recovery tests; no model inference."""

import json
import tempfile
import unittest
from pathlib import Path

import patclaim_worker as worker


def valid_result(tokens=200):
    return {"patclaimeval": 0.6, "patclaimeval_4096": 0.61, "pair_tokens": tokens,
            "truncated": tokens > 2048, "truncated_4096": False}


class InputTests(unittest.TestCase):
    def test_raw_characters_are_preserved(self):
        gold, candidate = "1. A\n claim", "1. A  claim\\ntext"
        pair = {"pair_id": worker.pair_identifier(gold, candidate), "gold_claim": gold, "candidate": candidate}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps({"pairs": [pair]}, ensure_ascii=False), encoding="utf-8")
            input_hash, pairs = worker.load_input(path)
            self.assertEqual(pairs, [pair])
            self.assertEqual(input_hash, worker.sha256_file(path))
            pair["candidate"] = "normalized claim"
            path.write_text(json.dumps({"pairs": [pair]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw source"):
                worker.load_input(path)

    def test_duplicate_pairs_are_rejected(self):
        pair = {"pair_id": worker.pair_identifier("gold", "candidate"), "gold_claim": "gold", "candidate": "candidate"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(json.dumps({"pairs": [pair, pair]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                worker.load_input(path)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "checkpoint.jsonl"
        self.header = {"record_type": "header", "input_sha256": "input", "settings": {"threads": 4}}

    def write(self, *records):
        with self.path.open("wb") as handle:
            for record in records:
                worker.append_record(handle, record)

    def test_incomplete_final_write_recovers_only_committed_pairs(self):
        result = valid_result(2067)
        self.write(self.header, {"record_type": "pair", "pair_id": "one", "result": result})
        committed_length = self.path.stat().st_size
        with self.path.open("ab") as handle:
            handle.write(b'{"record_type":"pair",')
        scores, length = worker.read_checkpoint(self.path, self.header, {"one", "two"})
        self.assertEqual(scores, {"one": result})
        self.assertEqual(length, committed_length)

    def test_changed_settings_reject_checkpoint(self):
        self.write(self.header)
        changed = {**self.header, "settings": {"threads": 8}}
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            worker.read_checkpoint(self.path, changed, set())

    def test_corrupt_committed_line_is_rejected(self):
        self.write(self.header)
        with self.path.open("ab") as handle:
            handle.write(b"not-json\n")
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint line"):
            worker.read_checkpoint(self.path, self.header, set())

    def test_unknown_and_duplicate_pairs_are_rejected(self):
        record = {"record_type": "pair", "pair_id": "one", "result": valid_result()}
        self.write(self.header, record)
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            worker.read_checkpoint(self.path, self.header, {"two"})
        self.write(self.header, record, record)
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            worker.read_checkpoint(self.path, self.header, {"one"})

    def test_invalid_numeric_or_truncation_fields_rejected(self):
        for override in ({"patclaimeval": float("nan")}, {"patclaimeval": True},
                         {"pair_tokens": 4097}, {"truncated": True}, {"truncated_4096": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                worker.validate_result({**valid_result(), **override})


class RegressionTests(unittest.TestCase):
    def test_expected_first_test_scores_and_failure(self):
        pair_id, expected = next(iter(worker.REGRESSION_TARGETS.items()))
        result = {**valid_result(), "patclaimeval": expected}
        self.assertTrue(worker.regression_checks({pair_id: result})[pair_id]["passed"])
        result["patclaimeval"] += 0.001
        with self.assertRaisesRegex(ValueError, "regression failed"):
            worker.regression_checks({pair_id: result})


if __name__ == "__main__":
    unittest.main()
