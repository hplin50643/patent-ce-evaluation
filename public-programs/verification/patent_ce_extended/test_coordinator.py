"""Protect immutable v1 results and prevent partial runs posing as complete runs."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core import LEGACY, prepare

class CoordinatorTests(unittest.TestCase):
    def test_prepare_rejects_legacy_output_before_writing(self):
        for destination in [LEGACY,LEGACY/'nested']:
            with self.subTest(destination=destination),self.assertRaisesRegex(ValueError,'separate'):
                prepare(destination)

    def test_all_cannot_aggregate_smoke_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            command=[sys.executable,str(Path(__file__).parents[1]/'run_patent_ce_extended_evaluation.py'),
                     '--stage','all','--smoke','--output-dir',temporary]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('only supported with --stage score',result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()),[])

    def test_selecting_missing_legacy_score_is_not_a_silent_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            command=[sys.executable,str(Path(__file__).parents[1]/'run_patent_ce_extended_evaluation.py'),
                     '--stage','score','--methods','bleu4','--output-dir',temporary]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Run prepare',result.stderr)

if __name__=='__main__':unittest.main()
