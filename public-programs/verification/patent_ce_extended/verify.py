"""Independent standard-library verification of extended CSVs and score metadata."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from core import METHODS, PRIMARY, PROJECT, atomic_json, digest, load_json, stamp


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    check(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
          math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), message)


def nullable_close(actual, expected, message):
    if expected is None:
        check(actual in ['', None], message)
    else:
        close(actual, expected, message)


def csv_bool(value):
    check(value in ['', 'True', 'False'], 'Unexpected CSV boolean representation')
    return {'':None, 'True':True, 'False':False}[value]


def independently_truncated(metadata):
    check(isinstance(metadata, dict), 'Metadata must be an object')
    flags = {key:value for key, value in metadata.items() if key == 'truncated' or key.endswith('_truncated')}
    check(all(type(value) is bool for value in flags.values()), 'Truncation flag is not a boolean')
    if 'truncated' in flags:
        check(flags['truncated'] or not any(flags.values()), 'Contradictory truncation flags')
        return flags['truncated']
    return any(flags.values()) if flags else None


def independently_selected(row, selector):
    if row['split'] != selector['split']:
        return False
    kind = selector['kind']
    if kind == 'split':
        return True
    if kind == 'flag_false':
        return csv_bool(row[selector['field']]) is False
    if kind == 'truncation':
        return csv_bool(row[selector['method']+'_truncated']) is selector['value']
    raise AssertionError('Unknown saved cohort rule')


def independent_stats(rows, method):
    counter = Counter((int(row['human_label']), int(row[method+'_prediction'])) for row in rows)
    matrix = [[counter[y, p] for p in [-1, 0, 1]] for y in [-1, 0, 1]]
    n = len(rows)
    correct = sum(counter[label, label] for label in [-1, 0, 1])
    classes = {}
    for label in [-1, 0, 1]:
        tp = counter[label, label]
        predicted = sum(counter[y, label] for y in [-1, 0, 1])
        support = sum(counter[label, p] for p in [-1, 0, 1])
        precision = tp/predicted if predicted else 0.
        recall = tp/support if support else 0.
        f1 = 2*tp/(predicted+support) if predicted+support else 0.
        classes[str(label)] = dict(precision=precision, recall=recall, f1=f1, support=support, predicted=predicted)
    non_tie = sum(int(row['human_label']) != 0 for row in rows)
    non_tie_correct = sum(int(row['human_label']) != 0 and int(row['human_label']) == int(row[method+'_prediction']) for row in rows)
    reverse = sum(int(row['human_label']) != 0 and int(row[method+'_prediction']) == -int(row['human_label']) for row in rows)
    return dict(n=n, correct=correct, accuracy=correct/n if n else None,
        error_rate=1-correct/n if n else None,
        macro_f1=sum(item['f1'] for item in classes.values())/3 if n else None,
        weighted_f1=sum(item['f1']*item['support'] for item in classes.values())/n if n else None,
        balanced_accuracy=sum(item['recall'] for item in classes.values())/3 if n else None,
        classes=classes, confusion_matrix=matrix, confusion_label_order=[-1, 0, 1],
        human_tie_n=classes['0']['support'], predicted_tie_n=classes['0']['predicted'],
        tie_recall=classes['0']['recall'] if n else None,
        non_tie_n=non_tie, non_tie_accuracy=non_tie_correct/non_tie if non_tie else None,
        reverse_count=reverse, reverse_rate=reverse/non_tie if non_tie else None)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(output_dir):
    out = Path(output_dir)
    result, prepared = load_json(out/'results.json'), load_json(out/'pairs.json')
    check(result['schema_version'] == 2, 'Wrong extended result schema')
    check(result['methods'] == METHODS and result['primary_methods'] == PRIMARY, 'Method registry mismatch')
    rows = read_csv(out/'scores.csv')
    originals = {row['record_id']:row for row in prepared['records']}
    check(len(rows) == len(originals) == 1228, 'Raw row count must remain 1228')
    check(len({row['record_id'] for row in rows}) == len(rows), 'Duplicate wide record IDs')
    ids = {pair['pair_id'] for pair in prepared['pairs']}
    check(len(ids) == 224, 'Expected 224 unique raw candidate pairs')
    input_hash = digest(out/'pairs.json')
    check(result['metadata']['input_sha256'] == input_hash, 'Prepared input hash mismatch')
    stages = {method:load_json(out/(method+'_scores.json')) for method in METHODS}
    for method, stage in stages.items():
        check(stage['schema_version'] == 2 and stage['method'] == method and stage['complete'] is True, 'Incomplete worker result:'+method)
        check(stage['input_sha256'] == input_hash and set(stage['scores']) == ids, 'Wrong worker input/pair coverage:'+method)
        check(digest(out/(method+'_scores.json')) == result['metadata']['method_output_hashes'][method], 'Worker changed after aggregation:'+method)
    raw_sources = {split:load_json(PROJECT/'data/Patent-CE'/('quality_'+split+'.json')) for split in ['train', 'test']}
    decisions = 0
    for row in rows:
        original = originals[row['record_id']]
        raw = raw_sources[row['split']][int(row['row_number'])-1]
        for key in ['gold_claim', 'A', 'B']:
            check(row[key] == original[key] == raw[key], 'Original text changed:'+row['record_id']+':'+key)
        check(int(row['human_label']) == original['human_label'] == raw['human_eval']['Quality'], 'Expert label mismatch')
        check(row['split'] == original['split'] and int(row['row_number']) == original['row_number'], 'Original order/index changed')
        check(row['gold_id'] == hashlib.sha256(row['gold_claim'].encode('utf-8')).hexdigest(), 'Gold group hash mismatch')
        for flag in ['duplicate_full_row', 'overlap_train_comparison', 'contains_gold_copy']:
            check(csv_bool(row[flag]) is original[flag], 'Audit flag mismatch:'+flag)
        for candidate in ['a', 'b']:
            key = hashlib.sha256(json.dumps([row['gold_claim'], row[candidate.upper()]], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
            check(key == row['pair_'+candidate+'_id'] == original['pair_'+candidate+'_id'], 'Candidate pair ID mismatch')
            for method in METHODS:
                value = stages[method]['scores'][key]
                check(type(value['score']) in [int, float] and math.isfinite(value['score']), 'Invalid worker numeric score')
                close(row[method+'_score_'+candidate], value['score'], 'Wide score differs from worker')
                check(csv_bool(row[method+'_'+candidate+'_truncated']) is independently_truncated(value.get('metadata', {})), 'Candidate truncation differs from worker')
        for method in METHODS:
            a, b = float(row[method+'_score_a']), float(row[method+'_score_b'])
            delta = a-b
            close(row[method+'_delta'], delta, 'Unrounded score difference mismatch')
            prediction = 0 if -0.0001 < delta < 0.0001 else (1 if delta > 0 else -1)
            check(int(row[method+'_prediction']) == prediction, 'Strict threshold prediction mismatch')
            check(csv_bool(row[method+'_correct']) is (prediction == int(row['human_label'])), 'Correct flag mismatch')
            ta, tb = csv_bool(row[method+'_a_truncated']), csv_bool(row[method+'_b_truncated'])
            combined = True if ta is True or tb is True else False if ta is False and tb is False else None
            check(csv_bool(row[method+'_truncated']) is combined, 'Comparison truncation mismatch')
            decisions += 1
    for key, cohort in result['cohorts'].items():
        items = [row for row in rows if independently_selected(row, cohort['selector'])]
        check(len(items) == cohort['n'], 'Cohort size mismatch:'+key)
        check(cohort['status'] == ('complete' if items else 'empty'), 'Cohort status mismatch')
        group_count = len({row['gold_id'] for row in items})
        for method in METHODS:
            actual, expected = independent_stats(items, method), cohort['results'][method]
            for field in ['n', 'correct', 'confusion_matrix', 'confusion_label_order', 'human_tie_n', 'predicted_tie_n', 'non_tie_n', 'reverse_count']:
                check(actual[field] == expected[field], 'Recomputed statistic mismatch:'+key+':'+method+':'+field)
            for field in ['accuracy', 'error_rate', 'macro_f1', 'weighted_f1', 'balanced_accuracy', 'tie_recall', 'non_tie_accuracy', 'reverse_rate']:
                nullable_close(actual[field], expected[field], 'Recomputed statistic mismatch:'+key+':'+method+':'+field)
            for label in ['-1', '0', '1']:
                for field in ['support', 'predicted', 'precision', 'recall', 'f1']:
                    close(actual['classes'][label][field], expected['classes'][label][field], 'Class statistic mismatch')
            check(expected['ci']['gold_groups'] == group_count, 'Wrong bootstrap cluster denominator')
            for metric in ['accuracy', 'macro_f1']:
                ci = expected['ci'][metric]
                check(ci == [None, None] if not items else 0 <= ci[0] <= ci[1] <= 1, 'Invalid confidence interval')
            if key in ['train', 'test']:
                check(expected == result['results'][key][method], 'Main/cohort results differ')
    summary = read_csv(out/'summary.csv')
    expected_keys = {(key, method) for key in result['cohorts'] for method in METHODS}
    check(len(summary) == len(expected_keys) and {(row['cohort'], row['method']) for row in summary} == expected_keys, 'Summary row/key coverage mismatch')
    for row in summary:
        stat = result['cohorts'][row['cohort']]['results'][row['method']]
        check(csv_bool(row['is_primary']) is (row['method'] in PRIMARY), 'Summary primary flag mismatch')
        for key in ['n', 'correct', 'accuracy', 'error_rate', 'macro_f1', 'weighted_f1', 'tie_recall', 'non_tie_accuracy', 'reverse_rate', 'continuous_tau', 'continuous_rho', 'discrete_tau', 'discrete_rho']:
            nullable_close(row[key], stat[key], 'Summary statistic differs from JSON:'+key)
        for metric in ['accuracy', 'macro_f1']:
            nullable_close(row[metric+'_ci_low'], stat['ci'][metric][0], 'Summary lower confidence limit mismatch')
            nullable_close(row[metric+'_ci_high'], stat['ci'][metric][1], 'Summary upper confidence limit mismatch')
    tall = read_csv(out/'predictions.csv')
    check(len(tall) == len(rows)*len(METHODS) == 17192, 'Tall prediction count mismatch')
    check(sum(csv_bool(row['is_primary']) is True for row in tall) == 15964, 'Primary prediction count mismatch')
    wide_by_id = {row['record_id']:row for row in rows}
    seen = set()
    for row in tall:
        key = row['record_id'], row['method']
        check(key not in seen and row['method'] in METHODS, 'Duplicate/unknown tall prediction')
        seen.add(key)
        wide = wide_by_id[row['record_id']]
        check(csv_bool(row['is_primary']) is (row['method'] in PRIMARY), 'Tall primary flag mismatch')
        for field in ['split', 'row_number', 'gold_id', 'human_label']:
            check(row[field] == wide[field], 'Tall base field mismatch')
        for field in ['score_a', 'score_b', 'delta', 'prediction', 'correct', 'truncated', 'a_truncated', 'b_truncated']:
            check(row[field] == wide[row['method']+'_'+field], 'Tall/wide mismatch:'+field)
    metadata_rows = read_csv(out/'pair_metadata.csv')
    metadata_json = load_json(out/'pair_metadata.json')['records']
    expected_metadata = {(method, pid) for method in METHODS for pid in ids}
    check(len(metadata_rows) == len(metadata_json) == len(expected_metadata), 'Pair metadata count mismatch')
    check({(row['method'], row['pair_id']) for row in metadata_rows} == expected_metadata, 'Pair metadata key coverage mismatch')
    json_by_key = {(row['method'], row['pair_id']):row for row in metadata_json}
    check(len(json_by_key) == len(metadata_json), 'Duplicate JSON pair metadata')
    for row in metadata_rows:
        method, pid = row['method'], row['pair_id']
        expected = stages[method]['scores'][pid]
        saved = json_by_key[method, pid]
        metadata = expected.get('metadata', {})
        check(json.loads(row['metadata_json']) == saved['metadata'] == metadata, 'Pair metadata changed from worker')
        close(row['score'], expected['score'], 'Pair metadata score mismatch')
        close(saved['score'], expected['score'], 'JSON metadata score mismatch')
        truncated = independently_truncated(metadata)
        check(csv_bool(row['truncated']) is truncated and saved['truncated'] is truncated, 'Pair metadata truncation mismatch')
        check(csv_bool(row['is_primary']) is (method in PRIMARY) and saved['is_primary'] is (method in PRIMARY), 'Pair metadata primary flag mismatch')
    for split in ['train', 'test']:
        items = [row for row in rows if row['split'] == split]
        for method in METHODS:
            audit = result['audit']['truncation'][split][method]
            check(audit['truncated_rows'] == sum(csv_bool(row[method+'_truncated']) is True for row in items), 'Truncation audit true mismatch')
            check(audit['untruncated_rows'] == sum(csv_bool(row[method+'_truncated']) is False for row in items), 'Truncation audit false mismatch')
            check(audit['unknown_rows'] == sum(csv_bool(row[method+'_truncated']) is None for row in items), 'Truncation audit unknown mismatch')
            check(audit['truncated_pair_instances'] == sum(csv_bool(row[method+'_'+candidate+'_truncated']) is True for row in items for candidate in ['a', 'b']), 'Truncation instance audit mismatch')
            check(audit['affected_row_numbers'] == [int(row['row_number']) for row in items if csv_bool(row[method+'_truncated']) is True], 'Truncation affected rows mismatch')
    for name, expected in prepared['source_hashes'].items():
        check(digest(PROJECT/'data/Patent-CE'/name) == expected, 'Original source hash changed:'+name)
    check(result['metadata']['available_dimensions'] == ['Quality'], 'Unavailable expert dimension presented as scored')
    geval = [item for item in result['paper_comparison'] if item['method'] == 'g_eval4']
    check(len(geval) == 1 and geval[0]['execution_status'] == 'not_executed' and geval[0]['our_accuracy_gap_pp'] is None, 'G-Eval must remain external reference only')
    artifacts = ['scores.csv', 'predictions.csv', 'summary.csv', 'pairs.json', 'pair_metadata.csv', 'pair_metadata.json']+[method+'_scores.json' for method in METHODS]
    report = dict(status='passed', verified_at_utc=stamp(), raw_records=len(rows), unique_pairs=len(ids),
        independently_verified_preference_decisions=decisions, primary_prediction_rows=15964,
        supplementary_prediction_rows=1228, methods_verified=len(METHODS), primary_methods_verified=len(PRIMARY),
        cohorts_verified=len(result['cohorts']), raw_text_exact_match=True, source_hashes_unchanged=True,
        tall_wide_agreement=True, metadata_exact_match=True, truncation_metadata_verified=True,
        confusion_accuracy_macro_weighted_f1_recomputed=True,
        artifact_hashes={name:digest(out/name) for name in artifacts})
    result['verification'] = report
    atomic_json(out/'verification.json', report)
    atomic_json(out/'results.json', result)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    verify(parser.parse_args().output_dir)
