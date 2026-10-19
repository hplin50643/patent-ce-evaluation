"""Integrity and interrupted-resume checks without model inference."""
import json
from pathlib import Path
import tempfile
import unittest

import similarity_worker as worker


class SimilarityWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.score = {"bleu4": 0.2, "rouge1_f1": 0.3, "rougeL_f1": 0.25,
                      "bertscore_f1": 0.8, "gold_tokens": 700, "candidate_tokens": 100,
                      "gold_truncated": True, "candidate_truncated": False}
        self.signatures = {"input_sha256": "input", "settings_sha256": "settings",
                           "environment_sha256": "environment"}

    def record(self, identifier="pair"):
        return {**self.signatures, "pair_id": identifier, "score": self.score,
                "score_sha256": worker.object_hash(self.score), "bertscore_hashcode": "modelhash"}

    def test_raw_pair_ids_preserve_whitespace(self):
        pairs = [{"pair_id": worker.pair_id(" reference ", candidate),
                  "gold_claim": " reference ", "candidate": candidate}
                 for candidate in ("candidate", " candidate ")]
        path = self.root / "pairs.json"
        path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
        self.assertNotEqual(pairs[0]["pair_id"], pairs[1]["pair_id"])
        self.assertEqual(worker.read_pairs(path), pairs)

    def test_checkpoint_recovers_only_incomplete_final_line(self):
        path = self.root / "checkpoint.jsonl"
        worker.append_records(path, [self.record()])
        valid_bytes = path.read_bytes()
        with path.open("ab") as handle:
            handle.write(b'{"unfinished":')
        scores, modelhash = worker.read_checkpoint(path, self.signatures, {"pair"})
        self.assertEqual(scores, {"pair": self.score})
        self.assertEqual(modelhash, "modelhash")
        self.assertEqual(path.read_bytes(), valid_bytes)

    def test_checkpoint_rejects_changed_inputs_and_changed_score(self):
        path = self.root / "checkpoint.jsonl"
        worker.append_records(path, [self.record()])
        changed = {**self.signatures, "input_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "input_sha256 mismatch"):
            worker.read_checkpoint(path, changed, {"pair"})
        record = self.record()
        record["score"] = {**self.score, "bleu4": 0.9}
        path.write_bytes(worker.canonical(record) + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            worker.read_checkpoint(path, self.signatures, {"pair"})

    def test_checkpoint_rejects_complete_corruption_and_duplicates(self):
        path = self.root / "checkpoint.jsonl"
        path.write_bytes(b"invalid json\n")
        with self.assertRaises(json.JSONDecodeError):
            worker.read_checkpoint(path, self.signatures, {"pair"})
        path.write_bytes(b"")
        worker.append_records(path, [self.record(), self.record()])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            worker.read_checkpoint(path, self.signatures, {"pair"})

    def test_score_rejects_nonfinite_and_wrong_truncation(self):
        with self.assertRaises(ValueError):
            worker.validate_score({**self.score, "bertscore_f1": float("nan")})
        with self.assertRaises(ValueError):
            worker.validate_score({**self.score, "gold_truncated": False})


if __name__ == "__main__":
    unittest.main()
