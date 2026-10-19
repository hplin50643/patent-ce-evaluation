"""Immutable Patent-CE preparation, shared identifiers and evaluation policy."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[3]
METHODS = ['bleu4', 'rouge1_f1', 'rougeL_f1', 'bertscore_f1', 'patclaimeval', 'patclaimeval_4096']
PRIMARY = METHODS[:5]
LABELS = {
    'bleu4': 'BLEU-4', 'rouge1_f1': 'ROUGE-1 F1', 'rougeL_f1': 'ROUGE-L F1',
    'bertscore_f1': 'BERTScore F1', 'patclaimeval': 'PatClaimEval 2048',
    'patclaimeval_4096': 'PatClaimEval 4096',
}
EPSILON = 0.0001


def stamp():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def pair_id(gold, candidate):
    return hashlib.sha256(canonical([gold, candidate]).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def preference(delta, epsilon=EPSILON):
    if not math.isfinite(delta):
        raise ValueError('Nonfinite score difference')
    if abs(delta) < epsilon:
        return 0
    return 1 if delta > 0 else -1


def comparison_key(record):
    a, b = sorted([record['A'], record['B']])
    return canonical([record['gold_claim'], a, b])


def prepare(output_dir, data_dir=None):
    output_dir = Path(output_dir)
    data_dir = Path(data_dir or PROJECT / 'data' / 'Patent-CE')
    manifest = load_json(data_dir / 'source_manifest.json')
    source_hashes = {}
    for source in manifest['downloads']:
        path = data_dir / source['file']
        actual = digest(path)
        if actual != source['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
        source_hashes[source['file']] = actual
    datasets = {split: load_json(data_dir / f'quality_{split}.json') for split in ['train', 'test']}
    assert len(datasets['train']) == 1044 and len(datasets['test']) == 184
    pairs, records, audit = {}, [], {}
    train_comparisons = {comparison_key(r) for r in datasets['train']}
    train_candidates = {pair_id(r['gold_claim'], r[k]) for r in datasets['train'] for k in ['A', 'B']}
    train_gold = {r['gold_claim'] for r in datasets['train']}
    conflict = defaultdict(set)
    for split, raw_rows in datasets.items():
        seen, pair_ids, golds = set(), set(), set()
        duplicate_count = 0
        for index, row in enumerate(raw_rows, 1):
            for field in ['gold_claim', 'A', 'B']:
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f'Invalid text {split}:{index}:{field}')
            y = row['human_eval']['Quality']
            if type(y) is not int or y not in [-1, 0, 1]:
                raise ValueError(f'Invalid expert label {split}:{index}')
            full_key = canonical(row)
            duplicate = full_key in seen
            duplicate_count += duplicate
            seen.add(full_key)
            item = dict(row, split=split, row_number=index, record_id=f'{split}_{index:04d}',
                        human_label=y, gold_id=hashlib.sha256(row['gold_claim'].encode('utf-8')).hexdigest(),
                        duplicate_full_row=duplicate,
                        overlap_train_comparison=split == 'test' and comparison_key(row) in train_comparisons,
                        contains_gold_copy=row['A'] == row['gold_claim'] or row['B'] == row['gold_claim'])
            golds.add(row['gold_claim'])
            for candidate in ['A', 'B']:
                pid = pair_id(row['gold_claim'], row[candidate])
                pair_ids.add(pid)
                pairs.setdefault(pid, dict(pair_id=pid, gold_claim=row['gold_claim'], candidate=row[candidate]))
                item[f'pair_{candidate.lower()}_id'] = pid
            conflict[comparison_key(row)].add(y if row['A'] <= row['B'] else -y)
            records.append(item)
        audit[split] = dict(rows=len(raw_rows), label_counts=dict(Counter(str(r['human_eval']['Quality']) for r in raw_rows)),
                            unique_gold=len(golds), unique_pairs=len(pair_ids), duplicate_full_rows=duplicate_count,
                            identical_candidates=sum(r['A'] == r['B'] for r in raw_rows),
                            contains_gold_copy=sum(r['A'] == r['gold_claim'] or r['B'] == r['gold_claim'] for r in raw_rows))
    test_records = [r for r in records if r['split'] == 'test']
    audit['overlap'] = dict(test_gold_seen_in_train=sum(r['gold_claim'] in train_gold for r in test_records),
                            test_pair_instances_seen_in_train=sum(r[f'pair_{k}_id'] in train_candidates for r in test_records for k in ['a', 'b']),
                            test_unique_pairs_seen_in_train=len({r[f'pair_{k}_id'] for r in test_records for k in ['a', 'b']} & train_candidates),
                            test_comparisons_seen_in_train=sum(r['overlap_train_comparison'] for r in test_records),
                            test_overlap_row_numbers=[r['row_number'] for r in test_records if r['overlap_train_comparison']],
                            contradictory_raw_comparison_groups=sum(len(values) > 1 for values in conflict.values()))
    assert len(pairs) == 224
    assert audit['overlap']['test_comparisons_seen_in_train'] == 9
    assert audit['test']['contains_gold_copy'] == 28
    result = dict(schema_version=1, created_at_taipei=stamp(), source_manifest=manifest,
                  source_hashes=source_hashes, preprocessing='Raw strings retained; metric-internal tokenization only',
                  audit=audit, pairs=list(pairs.values()), records=records)
    target = output_dir / 'pairs.json'
    if target.exists():
        prior = load_json(target)
        if prior['source_hashes'] != source_hashes or prior['pairs'] != result['pairs'] or prior['records'] != records:
            raise ValueError('Existing prepared input differs; choose a fresh output directory')
        return target
    atomic_json(target, result)
    return target
