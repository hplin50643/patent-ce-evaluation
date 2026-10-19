"""Policy and gold-cluster statistics checks with hand-derived outcomes."""
from __future__ import annotations
import math
import unittest
from core import METHODS, preference
from aggregate import classification, cluster_bootstrap


class EvaluationTests(unittest.TestCase):
    def test_strict_threshold(self):
        self.assertEqual(preference(0.),0)
        self.assertEqual(preference(math.nextafter(.0001,0)),0)
        self.assertEqual(preference(.0001),1)
        self.assertEqual(preference(-.0001),-1)
        self.assertEqual(preference(math.nextafter(-.0001,0)),0)
        with self.assertRaises(ValueError):
            preference(float('nan'))

    def test_candidate_swap(self):
        for delta in [-1.,-.0001,-.00001,0.,.00001,.0001,1.]:
            self.assertEqual(preference(-delta),-preference(delta))

    def test_hand_calculated_confusion(self):
        s=classification([-1,0,1],[-1,-1,1])
        self.assertEqual(s['confusion_matrix'],[[1,0,0],[1,0,0],[0,0,1]])
        self.assertEqual(s['correct'],2)
        self.assertAlmostEqual(s['macro_f1'],5/9)
        self.assertAlmostEqual(s['weighted_f1'],5/9)
        self.assertEqual(s['tie_recall'],0.)
        self.assertEqual(s['non_tie_accuracy'],1.)

    def test_opposite_direction_errors(self):
        s=classification([1,0,-1],[-1,1,1])
        self.assertEqual(s['reverse_count'],2)
        self.assertEqual(s['reverse_rate'],1.)
        self.assertEqual(s['accuracy'],0.)

    def test_cluster_bootstrap_shared_draws_and_row_replication(self):
        rows=[dict(gold_id='g1',human_label=1,methods={m:dict(prediction=1) for m in METHODS}),
              dict(gold_id='g2',human_label=-1,methods={m:dict(prediction=1) for m in METHODS})]
        ci,pairs=cluster_bootstrap(rows,1000,42)
        ci2,pairs2=cluster_bootstrap(rows*7,1000,42)
        self.assertEqual(ci,ci2)
        self.assertEqual(pairs,pairs2)
        self.assertEqual(ci['bleu4']['accuracy'],[0.,1.])
        self.assertTrue(all(p['ci_accuracy']==[0.,0.] for p in pairs))
        self.assertTrue(all(p['ci_macro_f1']==[0.,0.] for p in pairs))


if __name__=='__main__':
    unittest.main()
