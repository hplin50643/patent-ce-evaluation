"""Hand-checkable preference, dependence and truncation regression cases."""
from __future__ import annotations

import unittest

from aggregate import (classification, cluster_bootstrap, comparison_methods,
                       correlation, join_rows, metadata_truncation,
                       pair_truncation, build_cohorts)
from core import METHODS, PRIMARY, preference
from verify import independent_stats


class PreferenceStatisticsTests(unittest.TestCase):
    def test_manual_three_class_confusion_and_reverse_denominator(self):
        labels = [-1, -1, 0, 0, 1, 1]
        predictions = [-1, 0, 0, 1, 1, -1]
        stat = classification(labels, predictions)
        self.assertEqual(stat['confusion_matrix'], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(stat['correct'], 3)
        self.assertEqual(stat['accuracy'], .5)
        self.assertEqual(stat['macro_f1'], .5)
        self.assertEqual(stat['weighted_f1'], .5)
        self.assertEqual(stat['non_tie_n'], 4)
        self.assertEqual(stat['non_tie_accuracy'], .5)
        self.assertEqual(stat['reverse_count'], 1)
        self.assertEqual(stat['reverse_rate'], .25)
        for values in stat['classes'].values():
            self.assertEqual(values['precision'], .5)
            self.assertEqual(values['recall'], .5)
            self.assertEqual(values['f1'], .5)
        csv_rows = [dict(human_label=str(y), custom_prediction=str(p)) for y, p in zip(labels, predictions)]
        self.assertEqual(independent_stats(csv_rows, 'custom'), stat)

    def test_strict_tie_boundary_and_negative_bart_direction(self):
        self.assertEqual(preference(.00009999), 0)
        self.assertEqual(preference(-.00009999), 0)
        self.assertEqual(preference(.0001), 1)
        self.assertEqual(preference(-.0001), -1)
        self.assertEqual(preference(-2.0-(-3.0)), 1)
        self.assertEqual(preference(-3.0-(-2.0)), -1)
        with self.assertRaises(ValueError):
            preference(float('nan'))

    def test_missing_classes_do_not_disappear_from_macro_f1(self):
        stat = classification([1, 1], [1, 1])
        self.assertEqual(stat['accuracy'], 1.)
        self.assertEqual(stat['macro_f1'], 1/3)
        self.assertEqual(stat['weighted_f1'], 1.)
        self.assertEqual(stat['classes']['0']['f1'], 0.)
        empty = classification([], [])
        self.assertEqual(empty['n'], 0)
        self.assertIsNone(empty['accuracy'])
        self.assertIsNone(empty['macro_f1'])

    def test_constant_correlations_are_not_zero(self):
        self.assertEqual(correlation([1, 1, 1], [-1, 0, 1]), (None, None))
        self.assertEqual(correlation([-2, 1, 3], [0, 0, 0]), (None, None))
        tau, rho = correlation([-2, 1, 3], [-1, 0, 1])
        self.assertAlmostEqual(tau, 1.)
        self.assertAlmostEqual(rho, 1.)

    def test_bootstrap_keeps_unequal_gold_groups_and_shared_draws(self):
        rows = [dict(gold_id='short', human_label=1, methods={'a':dict(prediction=1), 'b':dict(prediction=1)})]
        rows += [dict(gold_id='long', human_label=1, methods={'a':dict(prediction=-1), 'b':dict(prediction=-1)}) for _ in range(9)]
        ci, paired = cluster_bootstrap(rows, methods=['a', 'b'], primary=['a', 'b'], replicates=1000, seed=42)
        # Drawing the short cluster twice yields accuracy 1, despite only 1/10
        # original rows being correct. An independent-row bootstrap fails this.
        self.assertEqual(ci['a']['accuracy'], [0., 1.])
        self.assertEqual(ci['a']['gold_groups'], 2)
        self.assertEqual(ci['a'], ci['b'])
        self.assertEqual(paired[0]['ci_accuracy'], [0., 0.])
        self.assertEqual(paired[0]['ci_macro_f1'], [0., 0.])
        self.assertEqual((ci, paired), cluster_bootstrap(rows, methods=['a', 'b'], primary=['a', 'b'], replicates=1000, seed=42))

    def test_pairwise_registry_excludes_supplement_from_78_main_pairs(self):
        comparisons = comparison_methods(METHODS, PRIMARY)
        self.assertEqual(len(comparisons), 79)
        self.assertEqual(comparisons[-1], ('patclaimeval_4096', 'patclaimeval'))
        self.assertEqual(sum('patclaimeval_4096' in pair for pair in comparisons), 1)

    def test_row_restoration_preserves_duplicates_raw_whitespace_and_swap(self):
        originals = [dict(record_id='one', split='test', row_number=1, gold_id='g', human_label=1,
                          gold_claim='raw  gold', A='candidate  A', B='candidate B', pair_a_id='a', pair_b_id='b'),
                     dict(record_id='two', split='test', row_number=2, gold_id='g', human_label=1,
                          gold_claim='raw  gold', A='candidate  A', B='candidate B', pair_a_id='a', pair_b_id='b'),
                     dict(record_id='swap', split='test', row_number=3, gold_id='g', human_label=-1,
                          gold_claim='raw  gold', A='candidate B', B='candidate  A', pair_a_id='b', pair_b_id='a')]
        stages = {'bartscore':dict(scores={'a':dict(score=-2., metadata=dict(truncated=False)),
                                          'b':dict(score=-3., metadata=dict(truncated=True))})}
        joined = join_rows(dict(records=originals), stages, methods=['bartscore'])
        self.assertEqual(len(joined), 3)
        self.assertEqual([row['record_id'] for row in joined], ['one', 'two', 'swap'])
        self.assertEqual(joined[0]['gold_claim'], 'raw  gold')
        self.assertEqual(joined[0]['A'], 'candidate  A')
        self.assertEqual([row['methods']['bartscore']['prediction'] for row in joined], [1, 1, -1])
        self.assertTrue(all(row['methods']['bartscore']['correct'] for row in joined))
        self.assertNotIn('methods', originals[0])

    def test_unknown_metadata_and_full_chunking_are_not_false_truncation(self):
        self.assertIsNone(metadata_truncation({}))
        self.assertIsNone(metadata_truncation(dict(gold_chunks=3, candidate_sentences=20)))
        self.assertFalse(metadata_truncation(dict(gold_chunks=3, candidate_sentences=20, truncated=False)))
        self.assertTrue(metadata_truncation(dict(gold_truncated=False, candidate_truncated=True)))
        self.assertIsNone(pair_truncation(False, None))
        self.assertTrue(pair_truncation(True, None))
        with self.assertRaises(ValueError):
            metadata_truncation(dict(truncated='False'))
        with self.assertRaises(ValueError):
            metadata_truncation(dict(truncated=False, candidate_truncated=True))

    def test_truncation_cohort_does_not_absorb_unknown_rows(self):
        rows = [dict(record_id=str(i), split='test', human_label=1, gold_id='g',
                     overlap_train_comparison=False, contains_gold_copy=False,
                     duplicate_full_row=False, methods={'bartscore':dict(truncated=flag)})
                for i, flag in enumerate([True, False, None])]
        cohorts = build_cohorts(rows, methods=['bartscore'])
        self.assertEqual(len(cohorts['test']['rows']), 3)
        self.assertEqual([row['record_id'] for row in cohorts['test_bartscore_truncated']['rows']], ['0'])
        self.assertEqual([row['record_id'] for row in cohorts['test_bartscore_untruncated']['rows']], ['1'])


if __name__ == '__main__':
    unittest.main()
