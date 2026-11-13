"""Exercise input provenance with real public rows and lightweight temporary manifests."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import prepare_inputs


class InputProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace = Path(__file__).resolve().parents[2]
        cls.original_baseline = Path("/baseline")
        cls.raw = Path("/data")
        if not (cls.original_baseline / "dataset.json").is_file():
            cls.original_baseline = workspace / "outputs/paper-reproduction/paper_longformer_original_20261108"
            cls.raw = workspace / "data/Patent-CE"
        if not (cls.original_baseline / "dataset.json").is_file() or not (cls.raw / "quality_train.json").is_file():
            raise unittest.SkipTest("Real baseline/public inputs are unavailable")
        cls.raw_hashes = {f"quality_{split}.json": prepare_inputs.digest(cls.raw / f"quality_{split}.json")
                          for split in ("train", "test")}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="modernbert-input-test-")
        self.root = Path(self.directory.name)
        self.baseline = self.root / "baseline"
        self.code = self.root / "code"
        self.output = self.root / "output"
        self.baseline.mkdir()
        self.code.mkdir()
        for name in ("dataset.json", "split_manifest.json", "source_fingerprint.json"):
            shutil.copyfile(self.original_baseline / name, self.baseline / name)
        self.config_path = self.code / "experiment_config.json"
        self.refresh_lock()
        # Only the temporary lock is reduced to three small real artifacts. Tests
        # neither create substitute model weights nor touch the actual 42-file lock.
        self.source_patch = mock.patch.object(prepare_inputs, "__file__", str(self.code / "prepare_inputs.py"))
        self.source_patch.start()

    def tearDown(self):
        self.source_patch.stop()
        self.directory.cleanup()

    def refresh_lock(self):
        lock = dict(baseline_run_id="paper_longformer_original_20261108",
                    files={name: prepare_inputs.digest(self.baseline / name)
                           for name in ("dataset.json", "split_manifest.json", "source_fingerprint.json")},
                    raw_source_sha256=self.raw_hashes)
        lock_path = self.code / "baseline_lock.json"
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        fingerprint = prepare_inputs.read(self.baseline / "source_fingerprint.json")["fingerprint"]
        config = dict(baseline_experiment=dict(lock_sha256=prepare_inputs.digest(lock_path), fingerprint=fingerprint))
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def run_prepare(self):
        return prepare_inputs.prepare(self.baseline, self.raw, self.output, self.config_path)

    def changed_dataset(self, change):
        target = self.baseline / "dataset.json"
        rows = prepare_inputs.read(target)
        change(rows)
        target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        self.refresh_lock()

    def test_exact_copy_preserves_original_split_order_and_labels(self):
        result = self.run_prepare()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["counts"], dict(train=939, validation=105, test=184))
        for name in ("dataset.json", "split_manifest.json"):
            self.assertEqual((self.output / name).read_bytes(), (self.original_baseline / name).read_bytes())
        self.assertEqual(self.run_prepare(), result)

    def test_changed_destination_is_not_silently_overwritten(self):
        self.run_prepare()
        target = self.output / "dataset.json"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Immutable input differs"):
            self.run_prepare()

    def test_changed_baseline_split_hash_is_rejected(self):
        target = self.baseline / "split_manifest.json"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Longformer baseline artifact differs: split_manifest.json"):
            self.run_prepare()

    def test_replaced_lock_is_rejected_before_copy(self):
        target = self.code / "baseline_lock.json"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Baseline lock file changed"):
            self.run_prepare()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_prepared_label_must_agree_with_original_public_row(self):
        self.changed_dataset(lambda rows: rows[0].__setitem__("label", -1 if rows[0]["label"] != -1 else 1))
        with self.assertRaisesRegex(ValueError, "Raw Quality label changed"):
            self.run_prepare()

    def test_comparison_identity_must_preserve_original_index(self):
        self.changed_dataset(lambda rows: rows[0].__setitem__("comparison_id", "train_000001"))
        with self.assertRaisesRegex(ValueError, "Original comparison identity changed"):
            self.run_prepare()

    def test_original_text_order_cannot_be_rearranged(self):
        self.changed_dataset(lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]))
        with self.assertRaisesRegex(ValueError, "Raw claim text/order changed|Original comparison identity changed"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
