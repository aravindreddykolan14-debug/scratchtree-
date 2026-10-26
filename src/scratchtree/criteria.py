"""Impurity measures: entropy and information gain.

Everything here works on *class counts* rather than raw labels, which lets the
split search evaluate every candidate threshold of a feature in one vectorised
pass (see ``tree._best_split_for_feature``).
"""

from __future__ import annotations

import numpy as np


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy (in bits) of a single class-count vector.

    H(S) = - sum_k p_k * log2(p_k),   with p_k = n_k / n

    >>> entropy(np.array([5, 5]))
    1.0
    >>> entropy(np.array([10, 0]))
    0.0
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum()) + 0.0  # + 0.0 turns -0.0 into 0.0


def entropy_rows(counts: np.ndarray) -> np.ndarray:
    """Entropy of every row of a (m, n_classes) count matrix, vectorised."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, counts / n, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1) + 0.0


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Information gain of splitting ``parent`` into ``left`` and ``right``.

    IG = H(parent) - [ |L|/|S| * H(L) + |R|/|S| * H(R) ]

    All three arguments are class-count vectors.

    >>> information_gain(np.array([5, 5]), np.array([5, 0]), np.array([0, 5]))
    1.0
    """
    n = parent.sum()
    nl, nr = left.sum(), right.sum()
    if n == 0:
        return 0.0
    children = (nl / n) * entropy(left) + (nr / n) * entropy(right)
    return float(entropy(parent) - children)
