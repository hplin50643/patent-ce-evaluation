"""Aggregate versioned candidate scores without changing the original evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import itertools
import math
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr

from core import (METHODS, PRIMARY, LABELS, EPSILON, PROJECT, atomic_json,
                  digest, load_json, preference, stamp)


# ACL final Tables 4 and 5, Quality column only. F1 averaging is unspecified.
PAPER_SOURCE = 'https://aclanthology.org/2025.acl-long.190.pdf'
PAPER_QUALITY = {
    'bleu1': (.522, .443, .326, .369),
    'bleu4': (.495, .420, .269, .305),
    'rouge1_f1': (.505, .429, .292, .332),
    'rouge2_f1': (.495, .420, .269, .306),
    'rougeL_f1': (.511, .434, .303, .344),
    'meteor': (.505, .428, .292, .331),
    'bertscore_f1': (.484, .418, .239, .268),
    'bartscore': (.446, .375, .164, .185),
    'moverscore_v2': (.462, .395, .210, .243),
    'simcse': (.446, .386, .165, .186),
    'unieval_relevance': (.527, .448, .337, .381),
    'alignscore': (.473, .401, .224, .255),
    'g_eval4': (.496, .469, .277, .310),
    'patclaimeval': (.565, .574, .477, .602),
}
BASE_FIELDS = ['record_id', 'split', 'row_number', 'gold_id', 'human_label',
               'gold_claim', 'A', 'B', 'pair_a_id', 'pair_b_id',
               'duplicate_full_row', 'overlap_train_comparison', 'contains_gold_copy']
METHOD_FIELDS = ['score_a', 'score_b', 'delta', 'prediction', 'correct',
                 'truncated', 'a_truncated', 'b_truncated']


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict.fromkeys(['mean', 'median', 'sd', 'q1', 'q3', 'minimum', 'maximum'])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                q1=float(np.quantile(values, .25)), q3=float(np.quantile(values, .75)),
                minimum=float(values.min()), maximum=float(values.max()))


def correlation(values, labels):
    if len(set(values)) < 2 or len(set(labels)) < 2:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        numbers = [kendalltau(values, labels, variant='b').statistic,
                   spearmanr(values, labels).statistic]
    return tuple(float(value) if math.isfinite(float(value)) else None for value in numbers)


def classification(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if len(labels) != len(predictions) or any(int(x) not in [-1, 0, 1] for x in [*labels, *predictions]):
        raise ValueError('Classification requires aligned fixed three-way labels')
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels + 1, predictions + 1), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * true_positive, support + predicted, out=np.zeros(3), where=(support + predicted) != 0)
    n = len(labels)
    non_tie = labels != 0
    reverse = non_tie & (predictions == -labels)
    return dict(n=n, correct=int(true_positive.sum()),
        accuracy=float(true_positive.sum()/n) if n else None,
        error_rate=float(1-true_positive.sum()/n) if n else None,
        macro_f1=float(f1.mean()) if n else None,
        weighted_f1=float(np.dot(f1, support)/n) if n else None,
        balanced_accuracy=float(recall.mean()) if n else None,
        classes={str(k):dict(precision=float(precision[k+1]), recall=float(recall[k+1]),
                            f1=float(f1[k+1]), support=int(support[k+1]), predicted=int(predicted[k+1]))
                 for k in [-1, 0, 1]},
        confusion_matrix=matrix.tolist(), confusion_label_order=[-1, 0, 1],
        human_tie_n=int(support[1]), predicted_tie_n=int(predicted[1]),
        tie_recall=float(recall[1]) if n else None,
        non_tie_n=int(non_tie.sum()),
        non_tie_accuracy=float((labels[non_tie] == predictions[non_tie]).mean()) if non_tie.any() else None,
        reverse_count=int(reverse.sum()),
        reverse_rate=float(reverse.sum()/non_tie.sum()) if non_tie.any() else None)


def method_statistics(rows, method):
    labels = [row['human_label'] for row in rows]
    predictions = [row['methods'][method]['prediction'] for row in rows]
    a = [row['methods'][method]['score_a'] for row in rows]
    b = [row['methods'][method]['score_b'] for row in rows]
    delta = [row['methods'][method]['delta'] for row in rows]
    values = classification(labels, predictions)
    values['continuous_tau'], values['continuous_rho'] = correlation(delta, labels)
    values['discrete_tau'], values['discrete_rho'] = correlation(predictions, labels)
    values['score_statistics'] = dict(A=describe(a), B=describe(b), delta=describe(delta), pooled_candidates=describe(a+b))
    return values


def comparison_methods(methods, primary):
    result = list(itertools.combinations([method for method in primary if method in methods], 2))
    if 'patclaimeval_4096' in methods and 'patclaimeval' in methods:
        result.append(('patclaimeval_4096', 'patclaimeval'))
    return result


def cluster_bootstrap(rows, methods=None, primary=None, replicates=10000, seed=42):
    """Resample complete gold groups, sharing every draw across all methods."""
    methods = list(METHODS if methods is None else methods)
    primary = list(PRIMARY if primary is None else primary)
    groups = sorted({row['gold_id'] for row in rows})
    if not groups:
        return {method:dict(accuracy=[None, None], macro_f1=[None, None], gold_groups=0, replicates=replicates)
                for method in methods}, []
    if replicates < 1:
        raise ValueError('Bootstrap replicate count must be positive')
    positions = {value: index for index, value in enumerate(groups)}
    tables = np.zeros((len(methods), len(groups), 3, 3), dtype=np.int64)
    for row in rows:
        for index, method in enumerate(methods):
            tables[index, positions[row['gold_id']], row['human_label']+1, row['methods'][method]['prediction']+1] += 1
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=replicates)
    matrices = np.einsum('rg,mgij->rmij', draws, tables, optimize=True)
    support, predicted = matrices.sum(axis=-1), matrices.sum(axis=-2)
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    accuracy = diagonal.sum(axis=-1)/matrices.sum(axis=(-1, -2))
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros_like(diagonal, dtype=float), where=(support+predicted) != 0).mean(axis=-1)
    ci = {method:dict(accuracy=np.quantile(accuracy[:, index], [.025, .975]).tolist(),
                     macro_f1=np.quantile(f1[:, index], [.025, .975]).tolist(),
                     gold_groups=len(groups), replicates=replicates)
          for index, method in enumerate(methods)}
    paired = []
    for a, b in comparison_methods(methods, primary):
        ia, ib = methods.index(a), methods.index(b)
        paired.append(dict(method_a=a, method_b=b,
            ci_accuracy=np.quantile(accuracy[:, ia]-accuracy[:, ib], [.025, .975]).tolist(),
            ci_macro_f1=np.quantile(f1[:, ia]-f1[:, ib], [.025, .975]).tolist()))
    return ci, paired


def metadata_truncation(metadata):
    """Unknown metadata stays unknown; chunking alone is not truncation."""
    if not isinstance(metadata, dict):
        raise TypeError('Score metadata must be an object')
    keys = [key for key in metadata if key == 'truncated' or key.endswith('_truncated')]
    for key in keys:
        if type(metadata[key]) is not bool:
            raise ValueError('Truncation metadata must be an actual boolean: '+key)
    if 'truncated' in metadata:
        if not metadata['truncated'] and any(metadata[key] for key in keys):
            raise ValueError('Conflicting aggregate and component truncation flags')
        return metadata['truncated']
    return any(metadata[key] for key in keys) if keys else None


def pair_truncation(a, b):
    if a is True or b is True:
        return True
    return False if a is False and b is False else None


def load_stages(out, prepared, methods=None):
    methods = list(METHODS if methods is None else methods)
    ids = {pair['pair_id'] for pair in prepared['pairs']}
    input_hash = digest(Path(out)/'pairs.json')
    stages = {}
    for method in methods:
        payload = load_json(Path(out)/(method+'_scores.json'))
        if payload.get('schema_version') != 2 or payload.get('method') != method or payload.get('complete') is not True:
            raise ValueError('Incomplete or mismatched method payload: '+method)
        if payload.get('input_sha256') != input_hash or set(payload['scores']) != ids:
            raise ValueError('Method does not cover exactly the prepared input: '+method)
        for pair_id, value in payload['scores'].items():
            if isinstance(value.get('score'), bool) or not isinstance(value.get('score'), (int, float)) or not math.isfinite(value['score']):
                raise ValueError('Invalid score '+method+':'+pair_id)
            metadata_truncation(value.get('metadata', {}))
        stages[method] = payload
    return stages


def join_rows(prepared, stages, methods=None):
    methods = list(METHODS if methods is None else methods)
    rows = []
    for original in prepared['records']:
        row = dict(original, methods={})
        for method in methods:
            a = stages[method]['scores'][row['pair_a_id']]
            b = stages[method]['scores'][row['pair_b_id']]
            score_a, score_b = float(a['score']), float(b['score'])
            if not math.isfinite(score_a) or not math.isfinite(score_b):
                raise ValueError('Nonfinite candidate score')
            delta = score_a-score_b
            prediction = preference(delta)
            ta, tb = metadata_truncation(a.get('metadata', {})), metadata_truncation(b.get('metadata', {}))
            row['methods'][method] = dict(score_a=score_a, score_b=score_b, delta=delta,
                prediction=prediction, correct=prediction == row['human_label'],
                truncated=pair_truncation(ta, tb), a_truncated=ta, b_truncated=tb)
        rows.append(row)
    return rows


def build_cohorts(rows, methods=None):
    methods = list(METHODS if methods is None else methods)
    definitions = {
        'test':dict(split='test', description='官方測試集 全部原始比較', selector=dict(kind='split', split='test')),
        'train':dict(split='train', description='官方訓練集 僅為已見資料描述', selector=dict(kind='split', split='train')),
        'test_no_train_comparison':dict(split='test', description='測試排除9筆訓練重複比較', selector=dict(kind='flag_false', split='test', field='overlap_train_comparison')),
        'test_no_gold_copy':dict(split='test', description='測試排除28筆參考原文複製', selector=dict(kind='flag_false', split='test', field='contains_gold_copy')),
        'train_deduplicated':dict(split='train', description='訓練排除20筆完全重複紀錄', selector=dict(kind='flag_false', split='train', field='duplicate_full_row')),
    }
    for split in ['train', 'test']:
        for method in methods:
            if any(row['split'] == split and row['methods'][method]['truncated'] is not None for row in rows):
                for flag, suffix in [(True, 'truncated'), (False, 'untruncated')]:
                    key = f'{split}_{method}_{suffix}'
                    definitions[key] = dict(split=split, description=f'{split} {LABELS.get(method, method)} '+('至少一個候選評分有截斷' if flag else '兩個候選評分均未截斷'),
                        selector=dict(kind='truncation', split=split, method=method, value=flag))
    for definition in definitions.values():
        definition['rows'] = [row for row in rows if select_row(row, definition['selector'])]
    return definitions


def select_row(row, selector):
    if row['split'] != selector['split']:
        return False
    if selector['kind'] == 'split':
        return True
    if selector['kind'] == 'flag_false':
        return row[selector['field']] is False
    if selector['kind'] == 'truncation':
        return row['methods'][selector['method']]['truncated'] is selector['value']
    raise ValueError('Unknown cohort selector')


def write_csvs(out, rows, stages):
    fields = BASE_FIELDS+[f'{method}_{key}' for method in METHODS for key in METHOD_FIELDS]
    with (out/'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict({key:row[key] for key in BASE_FIELDS}, **{
                f'{method}_{key}':row['methods'][method][key] for method in METHODS for key in METHOD_FIELDS}))
    fields = ['record_id', 'split', 'row_number', 'gold_id', 'human_label', 'method', 'is_primary']+METHOD_FIELDS
    with (out/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                writer.writerow(dict({key:row[key] for key in fields[:5]}, method=method,
                                     is_primary=method in PRIMARY, **row['methods'][method]))
    records = []
    fields = ['method', 'is_primary', 'pair_id', 'score', 'truncated', 'metadata_json']
    with (out/'pair_metadata.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for pair_id, value in stages[method]['scores'].items():
                item = dict(method=method, is_primary=method in PRIMARY, pair_id=pair_id,
                            score=value['score'], truncated=metadata_truncation(value.get('metadata', {})),
                            metadata=value.get('metadata', {}))
                records.append(item)
                writer.writerow(dict({key:item[key] for key in fields[:-1]}, metadata_json=__import__('json').dumps(item['metadata'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
    atomic_json(out/'pair_metadata.json', dict(schema_version=2, records=records))


def select_cases(test):
    selected, used = [], set()
    def add(candidates, reason):
        for row in candidates:
            if row['record_id'] not in used:
                selected.append(dict(row, reason_selected=reason))
                used.add(row['record_id'])
                return
    add(sorted([r for r in test if not r['methods']['patclaimeval']['correct']], key=lambda r:-abs(r['methods']['patclaimeval']['delta'])), 'PatClaimEval 分數差較大但未重現專家答案')
    add(sorted([r for r in test if r['methods']['patclaimeval']['correct']], key=lambda r:(sum(r['methods'][m]['correct'] for m in PRIMARY), r['row_number'])), 'PatClaimEval 正確而其他方法較多誤判')
    add(sorted([r for r in test if not r['methods']['bertscore_f1']['correct']], key=lambda r:-min(r['methods']['bertscore_f1']['score_a'], r['methods']['bertscore_f1']['score_b'])), 'BERTScore 兩候選皆高但偏好判斷錯誤')
    add([r for r in test if r['human_label'] == 0], '專家判平手')
    add([r for r in test if any(r['methods'][m]['truncated'] is True for m in PRIMARY)], '至少一種方法的輸入有截斷')
    add([r for r in test if r['overlap_train_comparison']], '測試與訓練重複的比較')
    add([r for r in test if r['contains_gold_copy']], '候選包含參考原文的複製')
    add(sorted(test, key=lambda r:(-sum(r['methods'][m]['correct'] for m in PRIMARY), r['row_number'])), '多方法一致的對照案例')
    for row in test:
        if len(selected) >= 8:
            break
        add([row], '補充案例')
    return selected


def paper_comparison(test_stats):
    result = []
    for method, (accuracy, f1, tau, rho) in PAPER_QUALITY.items():
        item = dict(method=method, label='G-Eval-4' if method == 'g_eval4' else LABELS[method],
                    dimension='Quality', source=PAPER_SOURCE, source_tables=[4, 5],
                    accuracy=accuracy, reported_f1=f1, kendall_tau=tau, spearman_rho=rho,
                    f1_averaging='not specified in source',
                    reproduction_claim=False, execution_status='not_executed' if method == 'g_eval4' else 'executed')
        current = test_stats.get(method)
        item.update(our_accuracy_gap_pp=100*(current['accuracy']-accuracy) if current else None,
                    our_macro_f1_gap_pp=100*(current['macro_f1']-f1) if current else None,
                    our_weighted_f1_gap_pp=100*(current['weighted_f1']-f1) if current else None)
        result.append(item)
    return result


def aggregate(output_dir, replicates=10000):
    out = Path(output_dir)
    prepared = load_json(out/'pairs.json')
    stages = load_stages(out, prepared)
    rows = join_rows(prepared, stages)
    definitions = build_cohorts(rows)
    cohorts, results, paired, cached = {}, {}, [], {}
    for key, definition in definitions.items():
        items = definition['rows']
        identifiers = tuple(row['record_id'] for row in items)
        if identifiers not in cached:
            statistics = {method:method_statistics(items, method) for method in METHODS}
            cis, differences = cluster_bootstrap(items, replicates=replicates)
            for method in METHODS:
                statistics[method]['ci'] = cis[method]
            cached[identifiers] = statistics, differences
        statistics, differences = cached[identifiers]
        for difference in differences:
            a, b = difference['method_a'], difference['method_b']
            paired.append(dict(difference, cohort=key,
                delta_accuracy=statistics[a]['accuracy']-statistics[b]['accuracy'],
                delta_macro_f1=statistics[a]['macro_f1']-statistics[b]['macro_f1']))
        cohorts[key] = dict(split=definition['split'], description=definition['description'],
                           selector=definition['selector'], n=len(items), status='complete' if items else 'empty', results=statistics)
        if key in ['train', 'test']:
            results[key] = statistics
        print(f'Aggregated {key}: {len(items)} rows', flush=True)
    train = definitions['train']['rows']
    test = definitions['test']['rows']
    majority = Counter(row['human_label'] for row in train).most_common(1)[0][0]
    audit = deepcopy(prepared['audit'])
    audit['truncation'] = {}
    for split in ['train', 'test']:
        items = [row for row in rows if row['split'] == split]
        audit['truncation'][split] = {method:dict(
            truncated_rows=sum(row['methods'][method]['truncated'] is True for row in items),
            untruncated_rows=sum(row['methods'][method]['truncated'] is False for row in items),
            unknown_rows=sum(row['methods'][method]['truncated'] is None for row in items),
            truncated_pair_instances=sum(row['methods'][method][key] is True for row in items for key in ['a_truncated', 'b_truncated']),
            affected_row_numbers=[row['row_number'] for row in items if row['methods'][method]['truncated'] is True]) for method in METHODS}
    write_csvs(out, rows, stages)
    atomic_json(out/'comparisons.json', rows)
    source_paths = {method:out/(method+'_scores.json') for method in METHODS}
    result = dict(schema_version=2,
        metadata=dict(created_at_utc=stamp(), seed=42, bootstrap_replicates=replicates,
            source_hashes=prepared['source_hashes'], source_manifest=prepared['source_manifest'],
            input_sha256=digest(out/'pairs.json'), python=sys.version, platform=platform.platform(),
            program_hashes={path.name:digest(path) for path in Path(__file__).parent.glob('*.py')},
            method_output_hashes={method:digest(path) for method, path in source_paths.items()},
            epsilon=EPSILON, tie_policy='abs(delta) < 0.0001; boundary is non-tie',
            correlation_policy=dict(continuous='score_a-score_b versus expert Quality label', discrete='predicted three-way label versus expert Quality label', kendall_variant='b', undefined='null for a constant input'),
            bootstrap_policy='Gold-reference clusters; g draws from g groups with replacement; retain every original row; paired draws shared across methods; percentile 95% intervals',
            available_dimensions=['Quality'], unavailable_dimensions=['Completeness', 'Clarity', 'Consistency', 'Linkage'],
            excluded_methods=dict(g_eval4='Not executed; original-paper reference values only')),
        audit=audit, methods=METHODS, primary_methods=PRIMARY, method_labels=LABELS,
        settings={method:stage.get('settings', {}) for method, stage in stages.items()},
        provenance={method:stage.get('provenance', {}) for method, stage in stages.items()},
        environments={method:stage.get('environment', {}) for method, stage in stages.items()},
        results=results, cohorts=cohorts, paired_differences=paired,
        baselines=dict(majority=dict(label=majority, train=classification([r['human_label'] for r in train], [majority]*len(train)), test=classification([r['human_label'] for r in test], [majority]*len(test))), uniform_random_expected_accuracy=1/3),
        cases=select_cases(test), paper_comparison=paper_comparison(results['test']),
        verification=dict(status='pending_independent_csv_verification', source_rows=len(rows), unique_pairs=len(prepared['pairs']), all_pairs_scored=True, raw_text_retained=True, skipped_rows=0))
    atomic_json(out/'results.json', result)
    fields = ['cohort', 'method', 'is_primary', 'n', 'correct', 'accuracy', 'error_rate', 'macro_f1', 'weighted_f1', 'tie_recall', 'non_tie_accuracy', 'reverse_rate', 'continuous_tau', 'continuous_rho', 'discrete_tau', 'discrete_rho', 'accuracy_ci_low', 'accuracy_ci_high', 'macro_f1_ci_low', 'macro_f1_ci_high']
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort, value in cohorts.items():
            for method, stat in value['results'].items():
                item = dict(cohort=cohort, method=method, is_primary=method in PRIMARY, **{key:stat[key] for key in fields[3:-4]})
                for metric in ['accuracy', 'macro_f1']:
                    item[metric+'_ci_low'], item[metric+'_ci_high'] = stat['ci'][metric]
                writer.writerow(item)
    atomic_json(out/'data_dictionary.json', dict(schema_version=2, encoding='UTF-8 BOM CSV with standard multiline quoting', scores_csv_rows=len(rows), predictions_csv_rows=len(rows)*len(METHODS), primary_prediction_rows=len(rows)*len(PRIMARY), supplementary_prediction_rows=len(rows)*(len(METHODS)-len(PRIMARY)), pair_metadata_rows=len(prepared['pairs'])*len(METHODS), metadata='Per candidate/reference pair and method; exact worker JSON retained; unknown truncation stays null/blank', human_label='Quality only: -1 B preferred, 0 tie, 1 A preferred', is_primary='False for patclaimeval_4096 sensitivity only', confusion_matrix='Rows expert, columns predicted, fixed order [-1,0,1]', score_policy='Higher scores are better; negative BART log scores are valid; no score-to-probability interpretation'))
    print('Aggregation complete: '+str(out/'results.json'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error('At least 100 bootstrap replicates are required')
    aggregate(args.output_dir, args.bootstrap_replicates)
