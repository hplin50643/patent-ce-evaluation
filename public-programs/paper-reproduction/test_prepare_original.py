"""Source-integrity tests for the faithful original-split preparation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from prepare_original import prepare


def source_rows(count: int, offset: int = 0) -> list[dict]:
    return [
        {
            "gold_claim": f"  reference {index % 20:02d} é 中文\n ",
            "A": f"candidate A {index % 60:02d}",
            "B": f"candidate B {index % 60:02d}",
            "human_eval": {"Quality": (index % 3) - 1},
        }
        for index in range(offset, offset + count)
    ]


class PrepareOriginalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "source"
        self.data_dir.mkdir()
        self.output_dir = self.root / "prepared"
        self.train = source_rows(1044)
        self.test = source_rows(184)

    def write_sources(self, train=None, test=None, bom=False) -> None:
        downloads = []
        for split, rows in (("train", self.train if train is None else train),
                            ("test", self.test if test is None else test)):
            filename = f"quality_{split}.json"
            payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
            if bom:
                payload = b"\xef\xbb\xbf" + payload
            (self.data_dir / filename).write_bytes(payload)
            downloads.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})
        (self.data_dir / "source_manifest.json").write_text(
            json.dumps({"downloads": downloads, "repository": "synthetic-unit-test-only"}),
            encoding="utf-8",
        )

    def dataset(self):
        return json.loads((self.output_dir / "dataset.json").read_text(encoding="utf-8"))

    def test_all_source_rows_duplicates_directions_text_and_order_preserved(self) -> None:
        self.train[1] = deepcopy(self.train[0])
        self.train[2] = deepcopy(self.train[0])
        self.train[2]["A"], self.train[2]["B"] = self.train[2]["B"], self.train[2]["A"]
        self.train[2]["human_eval"]["Quality"] *= -1
        self.train[3]["human_eval"]["Quality"] = 0
        self.write_sources(bom=True)
        manifest = prepare(self.data_dir, self.output_dir)
        dataset = self.dataset()
        self.assertEqual(len(dataset), 1228)
        self.assertEqual(len({row["comparison_id"] for row in dataset}), 1228)
        for split, originals, records in (("train", self.train, dataset[:1044]),
                                          ("test", self.test, dataset[1044:])):
            for index, (original, row) in enumerate(zip(originals, records)):
                self.assertEqual(row["source_split"], split)
                self.assertEqual(row["source_row_index"], index)
                self.assertEqual(row["comparison_id"], f"{split}_{index:06d}")
                for key in ("gold_claim", "A", "B"):
                    self.assertEqual(row[key], original[key])
                self.assertEqual(row["label"], original["human_eval"]["Quality"])
                self.assertEqual(row["gold_id"], hashlib.sha256(
                    original["gold_claim"].encode("utf-8"),
                ).hexdigest())
        self.assertEqual(dataset[0]["A"], dataset[1]["A"])
        self.assertNotEqual(dataset[0]["comparison_id"], dataset[1]["comparison_id"])
        self.assertEqual(dataset[2]["A"], dataset[0]["B"])
        self.assertEqual(dataset[2]["label"], -dataset[0]["label"])
        self.assertEqual(dataset[3]["label"], 0)
        self.assertTrue(manifest["audit"]["source_order_and_duplicates_preserved"])

    def test_seed42_ceil105_validation_939_train_and_test_untouched(self) -> None:
        self.write_sources()
        manifest = prepare(self.data_dir, self.output_dir)
        self.assertEqual(len(manifest["folds"]), 1)
        fold = manifest["folds"][0]
        self.assertEqual(fold["fold"], 0)
        indices = list(range(1044))
        random.Random(42).shuffle(indices)
        self.assertEqual(fold["validation"]["comparison_ids"],
                         [f"train_{index:06d}" for index in indices[:105]])
        self.assertEqual(fold["train"]["comparison_ids"],
                         [f"train_{index:06d}" for index in indices[105:]])
        self.assertEqual(fold["test"]["comparison_ids"],
                         [f"test_{index:06d}" for index in range(184)])
        split_ids = []
        for name, count in (("train", 939), ("validation", 105), ("test", 184)):
            self.assertEqual(fold[name]["counts"]["comparisons"], count)
            self.assertEqual(len(fold[name]["comparison_ids"]), count)
            split_ids.extend(fold[name]["comparison_ids"])
        self.assertEqual(len(split_ids), len(set(split_ids)))
        self.assertEqual(set(split_ids), {row["comparison_id"] for row in self.dataset()})
        selection = manifest["validation_selection"]
        self.assertEqual(selection["rounding"], "ceil")
        self.assertEqual(selection["seed"], 42)
        self.assertTrue(selection["author_undisclosed_defaults_chosen_here"])

    def test_content_overlaps_are_reported_and_not_rejected(self) -> None:
        self.write_sources()
        manifest = prepare(self.data_dir, self.output_dir)
        self.assertGreater(manifest["audit"]["combined"]["exact_duplicate_rows_preserved"], 0)
        original = manifest["audit"]["original_train_test_overlap"]
        self.assertEqual(original["row_ids_shared"], 0)
        self.assertEqual(original["gold_groups_shared"], 20)
        self.assertEqual(original["right_rows_gold_seen_in_left"], 184)
        self.assertGreater(original["candidate_strings_shared"], 0)
        self.assertGreater(original["right_exact_comparison_rows_seen_in_left"], 0)
        for audit in manifest["folds"][0]["audit"]["overlap"].values():
            self.assertEqual(audit["row_ids_shared"], 0)
            self.assertGreater(audit["gold_groups_shared"], 0)
        self.assertTrue(manifest["audit"]["content_overlaps_are_reported_not_errors"])

    def test_conflicting_content_labels_are_preserved_for_faithfulness(self) -> None:
        self.train[1] = deepcopy(self.train[0])
        self.train[1]["human_eval"]["Quality"] *= -1
        self.write_sources()
        manifest = prepare(self.data_dir, self.output_dir)
        dataset = self.dataset()
        self.assertEqual(dataset[1]["label"], -dataset[0]["label"])
        self.assertGreater(manifest["audit"]["combined"]["canonical_comparisons_with_conflicting_labels"], 0)

    def test_source_and_dataset_hashes_match_and_outputs_repeat_exactly(self) -> None:
        self.write_sources()
        first = prepare(self.data_dir, self.output_dir)
        repeated = self.root / "repeated"
        second = prepare(self.data_dir, repeated)
        self.assertEqual(first, second)
        for filename in ("dataset.json", "split_manifest.json"):
            self.assertEqual((self.output_dir / filename).read_bytes(),
                             (repeated / filename).read_bytes())
        self.assertEqual(first["dataset_sha256"], hashlib.sha256(
            (self.output_dir / "dataset.json").read_bytes(),
        ).hexdigest())
        for filename, digest in first["source_hashes"].items():
            self.assertEqual(digest, hashlib.sha256((self.data_dir / filename).read_bytes()).hexdigest())
        self.assertFalse(list(self.output_dir.glob("*.tmp")))

    def test_source_hash_mismatch_aborts_before_output(self) -> None:
        self.write_sources()
        path = self.data_dir / "quality_train.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            prepare(self.data_dir, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_original_counts_and_label_types_are_checked(self) -> None:
        self.write_sources(train=self.train[:-1])
        with self.assertRaisesRegex(ValueError, "expected 1044 original rows"):
            prepare(self.data_dir, self.output_dir)
        for label in (True, 2, 0.0, None):
            with self.subTest(label=label):
                changed = deepcopy(self.train)
                changed[0]["human_eval"]["Quality"] = label
                self.write_sources(train=changed)
                with self.assertRaisesRegex(ValueError, "Quality must be an integer"):
                    prepare(self.data_dir, self.output_dir)


if __name__ == "__main__":
    unittest.main()
