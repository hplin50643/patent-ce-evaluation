"""Frozen original-split evaluation: no group averaging, threshold fitting or seed picking."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def preference(delta, epsilon=1e-4):
    if not math.isfinite(delta): raise ValueError('Nonfinite score difference')
    return 0 if abs(delta) < epsilon else (1 if delta > 0 else -1)


def loss_tensor(score_a, score_b, labels, margin, tolerance):
    import torch
    delta = score_a.float()-score_b.float()
    return torch.where(labels == 0, torch.relu(delta.abs()-tolerance),
                       torch.relu(margin-labels.float()*delta)).mean()


def correlations(values, labels):
    if len(set(values)) < 2 or len(set(labels)) < 2:
        return {'kendall_tau_b': None, 'kendall_tau_c': None, 'spearman_rho': None,
                'undefined': 'constant input'}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return dict(kendall_tau_b=float(kendalltau(values, labels, variant='b').statistic),
                    kendall_tau_c=float(kendalltau(values, labels, variant='c').statistic),
                    spearman_rho=float(spearmanr(values, labels).statistic), undefined=None)


def evaluate_rows(rows, epsilon=1e-4):
    if not rows or len({r['comparison_id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicated row identities')
    y, delta = [], []
    for row in rows:
        a,b = float(row['score_a']), float(row['score_b'])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in [a,b]): raise ValueError('Invalid scores')
        if row['label'] not in [-1,0,1]: raise ValueError('Invalid label')
        if not math.isclose(row['delta'], a-b, rel_tol=0, abs_tol=1e-12): raise ValueError('Delta mismatch')
        y.append(row['label']); delta.append(row['delta'])
    pred = [preference(d,epsilon) for d in delta]
    matrix = confusion_matrix(y,pred,labels=[-1,0,1])
    ties = sum(v==0 for v in y)
    return dict(n=len(rows), n_gold=len({r['gold_id'] for r in rows}),
                accuracy=float(accuracy_score(y,pred)),
                f1_weighted=float(f1_score(y,pred,labels=[-1,0,1],average='weighted',zero_division=0)),
                f1_macro=float(f1_score(y,pred,labels=[-1,0,1],average='macro',zero_division=0)),
                f1_micro=float(f1_score(y,pred,labels=[-1,0,1],average='micro',zero_division=0)),
                confusion_matrix=matrix.tolist(), label_order=[-1,0,1],
                tie_recall=float(matrix[1,1]/ties) if ties else None,
                continuous=correlations(delta,y), discrete=correlations(pred,y),
                epsilon=epsilon,tie_policy='abs(delta) < epsilon',aggregation='global original rows, duplicates retained')


def predictions(rows, scores, metadata=None):
    result=[]
    for row in rows:
        a,b=scores[(row['gold_claim'],row['A'])],scores[(row['gold_claim'],row['B'])]
        result.append(dict(comparison_id=row['comparison_id'],gold_id=row['gold_id'],
                           source_split=row['source_split'],source_row_index=row['source_row_index'],
                           label=row['label'],score_a=a,score_b=b,delta=a-b,
                           prediction=preference(a-b), **(metadata or {})))
    return result
