"""Shared policy for the versioned thirteen-method Patent-CE evaluation."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[3]
LEGACY = PROJECT / 'outputs/patent_ce_evaluation_20261019_203847'
PRIMARY = ['bleu1', 'bleu4', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'meteor',
           'bertscore_f1', 'bartscore', 'moverscore_v2', 'simcse',
           'unieval_relevance', 'alignscore', 'patclaimeval']
METHODS = PRIMARY + ['patclaimeval_4096']
NEW_METHODS = ['bleu1', 'rouge2_f1', 'meteor', 'bartscore', 'moverscore_v2',
               'simcse', 'unieval_relevance', 'alignscore']
LABELS = dict(bleu1='BLEU-1', bleu4='BLEU-4', rouge1_f1='ROUGE-1 F1',
              rouge2_f1='ROUGE-2 F1', rougeL_f1='ROUGE-L F1', meteor='METEOR',
              bertscore_f1='BERTScore F1', bartscore='BARTScore CNN',
              moverscore_v2='MoverScore v2', simcse='SimCSE',
              unieval_relevance='UniEval relevance', alignscore='AlignScore large',
              patclaimeval='PatClaimEval 2048', patclaimeval_4096='PatClaimEval 4096')
EPSILON = 0.0001

def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def pair_id(gold, candidate):
    return hashlib.sha256(canonical([gold, candidate]).encode('utf-8')).hexdigest()

def preference(delta, epsilon=EPSILON):
    if not math.isfinite(delta):
        raise ValueError('Nonfinite score difference')
    return 0 if abs(delta) < epsilon else (1 if delta > 0 else -1)

def wsl_path(path):
    value = str(Path(path).resolve())
    return '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')

def prepare(output_dir, legacy_dir=LEGACY):
    """Verify immutable originals and import exact saved scores without inference."""
    out, legacy = Path(output_dir).resolve(), Path(legacy_dir).resolve()
    if out == legacy or out.is_relative_to(legacy):
        raise ValueError('The new output directory must be separate from the immutable legacy run')
    manifest = load_json(legacy / 'delivery_manifest.json')
    for name, item in manifest['contents'].items():
        path = PROJECT / name
        if digest(path) != item['sha256']:
            raise ValueError('Original delivery manifest mismatch: ' + name)
    provenance = load_json(legacy / 'model_provenance.json')
    bert = provenance['bertscore']
    snapshot = PROJECT / 'LLM4DPCG/.cache/huggingface/hub/models--roberta-large/snapshots' / bert['revision']
    checked_models = {}
    # Models are optional on a result-only clone; never misrepresent an absent model as verified.
    if snapshot.exists():
        for name, item in bert['files'].items():
            if digest(snapshot / name) != item['sha256']:
                raise ValueError('Legacy BERTScore model mismatch: ' + name)
        checked_models['bertscore'] = 'all recorded model files verified locally'
    else:
        checked_models['bertscore'] = 'weights absent; immutable saved provenance verified'
    pat_dir = PROJECT / 'public-programs/models/PatClaimEval-Quality'
    if pat_dir.exists():
        for name, expected in provenance['patclaimeval']['file_hashes'].items():
            if digest(pat_dir / name) != expected:
                raise ValueError('Legacy PatClaimEval model mismatch: ' + name)
        checked_models['patclaimeval'] = 'all recorded model files verified locally'
    else:
        checked_models['patclaimeval'] = 'weights absent; immutable saved provenance verified'
    out.mkdir(parents=True, exist_ok=True)
    old_pairs = legacy / 'pairs.json'
    prepared = load_json(old_pairs)
    assert len(prepared['pairs']) == 224 and len(prepared['records']) == 1228
    for source in prepared['source_manifest']['downloads']:
        if digest(PROJECT / 'data/Patent-CE' / source['file']) != source['sha256']:
            raise ValueError('Source data changed: ' + source['file'])
    for pair in prepared['pairs']:
        assert pair_id(pair['gold_claim'], pair['candidate']) == pair['pair_id']
    target = out / 'pairs.json'
    if target.exists() and digest(target) != digest(old_pairs):
        raise ValueError('Prepared input changed; choose a new output directory')
    if not target.exists():
        shutil.copy2(old_pairs, target)
    worker_hashes = {}
    for worker_name, methods in [('similarity_scores.json', ['bleu4', 'rouge1_f1', 'rougeL_f1', 'bertscore_f1']),
                                  ('patclaim_scores.json', ['patclaimeval', 'patclaimeval_4096'])]:
        path = legacy / worker_name
        worker = load_json(path)
        worker_hashes[worker_name] = digest(path)
        assert set(worker['scores']) == {p['pair_id'] for p in prepared['pairs']}
        for method in methods:
            scores = {}
            for pid, values in worker['scores'].items():
                value = float(values[method])
                assert math.isfinite(value)
                if method == 'bertscore_f1':
                    metadata = {k: values[k] for k in ['gold_tokens', 'candidate_tokens', 'gold_truncated', 'candidate_truncated']}
                    metadata['truncated'] = metadata['gold_truncated'] or metadata['candidate_truncated']
                elif method.startswith('patclaimeval'):
                    # Legacy helper records lengths for the official 2048 input.
                    metadata = dict(pair_tokens=values['pair_tokens'],
                                    truncated=values['pair_tokens'] > (4096 if method.endswith('4096') else 2048),
                                    max_length=4096 if method.endswith('4096') else 2048)
                else:
                    metadata = dict(truncated=False, truncation_policy='not applicable; lexical metric reads complete text')
                scores[pid] = dict(score=value, metadata=metadata)
            payload = dict(schema_version=2, method=method, input_sha256=digest(target), complete=True,
                           settings=worker['settings'], provenance=dict(imported_without_inference=True,
                           source_worker=worker_name, source_worker_sha256=digest(path),
                           source_manifest_sha256=digest(legacy / 'delivery_manifest.json'),
                           models=provenance), environment=worker['environment'], scores=scores)
            dest = out / (method + '_scores.json')
            if dest.exists() and load_json(dest) != payload:
                raise ValueError('Existing imported scores differ: ' + method)
            if not dest.exists():
                atomic_json(dest, payload)
    atomic_json(out / 'legacy_import_verification.json', dict(
        status='passed', source_manifest_sha256=digest(legacy / 'delivery_manifest.json'),
        original_files_verified=len(manifest['contents']), source_worker_hashes=worker_hashes,
        models_checked=checked_models, imported_methods=6, unique_pairs=224,
        source_rows=1228, all_scores_copied_without_rounding=True))
    return target
