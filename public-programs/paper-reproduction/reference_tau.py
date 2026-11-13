"""Reproduce the rank-pair Kendall calculation in the cited PatentEval work.

Zuo et al. (2024), section 7, PDF page 9 (printed page 2695), represents
left wins, right wins, and draws by [1, 2], [2, 1], and [1, 1], respectively:
https://aclanthology.org/2024.naacl-long.147.pdf#page=9

The official implementation passes its N-by-2 rank arrays to scipy kendalltau:
https://github.com/ZoeYou/PatentEval/blob/d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427

Flattening is explicit here to preserve that operation across SciPy versions.
This helper neither thresholds scores nor changes supplied preference labels.
It implements the cited reference, not unavailable PatClaimEval evaluation code.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import kendalltau

REFERENCE_IMPLEMENTATION_URL = (
    "https://github.com/ZoeYou/PatentEval/blob/"
    "d96b52c41fa4357e38c15ad334b84ea13008a25c/src/kendall_tau.py#L427"
)
REFERENCE_PAPER_URL = "https://aclanthology.org/2024.naacl-long.147.pdf#page=9"


def rank_pairs(labels: Iterable[int]) -> np.ndarray:
    """Map +1 (left wins), -1 (right wins), 0 (draw) to an N-by-2 array.

    Draws have ranks [1, 1], exactly as the cited implementation specifies.
    Values outside {-1, 0, 1}, or non-vector inputs, raise ValueError.
    """
    values = np.asarray(list(labels))
    if values.ndim != 1 or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError("preference labels must be a vector containing only -1, 0, 1")
    ranks = np.ones((values.size, 2), dtype=np.int64)
    ranks[values == 1, 1] = 2
    ranks[values == -1, 0] = 2
    return ranks


def patenteval_tau(predictions: Iterable[int], labels: Iterable[int]) -> float | None:
    """Compute tau-b on explicitly flattened PatentEval rank pairs.

    Both inputs must contain one preference label per comparison. Return None
    for empty inputs or a constant flattened rank vector (e.g. all draws).
    A constant non-draw preference vector is not constant after rank mapping.
    Unequal input lengths raise ValueError.
    """
    predicted_ranks = rank_pairs(predictions)
    human_ranks = rank_pairs(labels)
    if predicted_ranks.shape != human_ranks.shape:
        raise ValueError("predictions and labels must contain the same number of comparisons")
    predicted_flat = predicted_ranks.ravel()
    human_flat = human_ranks.ravel()
    if predicted_flat.size == 0 or np.unique(predicted_flat).size < 2 or np.unique(human_flat).size < 2:
        return None
    value = float(kendalltau(predicted_flat, human_flat, variant="b").statistic)
    return value if math.isfinite(value) else None
