"""A decision tree classifier written from scratch (NumPy only).

Splitting criterion: entropy / information gain (ID3/C4.5 style), with binary
threshold splits on numeric features ("x[j] <= t" goes left).

Tree-size hyperparameters
-------------------------
max_depth            : maximum depth of the tree (root has depth 0).
min_samples_split    : a node needs at least this many samples to be split.
min_samples_leaf     : every leaf must keep at least this many samples.
max_leaf_nodes       : maximum number of leaves. The tree is grown *best-first*
                       (the node whose split gives the largest weighted gain is
                       expanded next), so the leaf budget is spent where it helps.
min_information_gain : a split is only accepted if its gain is at least this.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .criteria import entropy, entropy_rows


@dataclass
class Node:
    """One node of the tree. A node with ``feature is None`` is a leaf."""

    counts: np.ndarray                 # class counts of the samples reaching it
    depth: int
    entropy: float
    feature: Optional[int] = None      # index of the split feature
    threshold: Optional[float] = None  # go left if x[feature] <= threshold
    gain: float = 0.0                  # information gain of the chosen split
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    node_id: int = 0
    # Candidate split remembered while the node waits in the frontier
    _candidate: Optional[tuple] = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    @property
    def n_samples(self) -> int:
        return int(self.counts.sum())

    @property
    def prediction(self) -> int:
        return int(np.argmax(self.counts))


class DecisionTreeClassifier:
    """Decision tree classifier using entropy and information gain.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. ``None`` means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples a node must contain to be split.
    min_samples_leaf : int, default=1
        Minimum number of samples each child of a split must contain.
    max_leaf_nodes : int or None, default=None
        Maximum number of leaves (best-first growth). ``None`` means unlimited.
    min_information_gain : float, default=0.0
        Minimum information gain (in bits) required to make a split.

    Examples
    --------
    >>> from scratchtree import DecisionTreeClassifier
    >>> X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    >>> y = [0, 1, 1, 0]
    >>> clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    >>> clf.predict([[1, 1]]).tolist()
    [1]
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_leaf_nodes: Optional[int] = None,
        min_information_gain: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_information_gain = min_information_gain

    # ------------------------------------------------------------------ utils
    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_information_gain": self.min_information_gain,
        }

    def set_params(self, **params) -> "DecisionTreeClassifier":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"DecisionTreeClassifier({args})"

    def _validate_params(self) -> None:
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes < 1:
            raise ValueError("max_leaf_nodes must be >= 1 or None")
        if self.min_information_gain < 0:
            raise ValueError("min_information_gain must be >= 0")

    # -------------------------------------------------------------------- fit
    def fit(self, X, y, feature_names: Optional[Sequence[str]] = None):
        """Build the tree from training data ``X`` (n_samples, n_features) and labels ``y``."""
        self._validate_params()

        if feature_names is None and hasattr(X, "columns"):  # pandas DataFrame
            feature_names = [str(c) for c in X.columns]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y have a different number of samples")
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")

        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f"x[{j}]" for j in range(self.n_features_in_)]
        )
        if len(self.feature_names_) != self.n_features_in_:
            raise ValueError("feature_names has the wrong length")

        self._X, self._y = X, y_enc
        self._ids = itertools.count()
        self.feature_importances_ = np.zeros(self.n_features_in_)

        root_idx = np.arange(len(y_enc))
        self.root_ = self._make_node(root_idx, depth=0)
        self._grow(root_idx)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        del self._X, self._y  # don't keep the training data around
        return self

    def _make_node(self, idx: np.ndarray, depth: int) -> Node:
        counts = np.bincount(self._y[idx], minlength=self.n_classes_)
        return Node(counts=counts, depth=depth, entropy=entropy(counts), node_id=next(self._ids))

    def _grow(self, root_idx: np.ndarray) -> None:
        """Best-first growth: a heap keyed by weighted gain (n_node * IG).

        Without ``max_leaf_nodes`` the order of expansion doesn't matter and the
        result equals classic recursive (depth-first) ID3 growth.
        """
        n_leaves = 1
        tie = itertools.count()  # tiebreaker so heapq never compares Nodes
        frontier: list = []

        def push(node: Node, idx: np.ndarray) -> None:
            split = self._find_best_split(node, idx)
            if split is not None:
                node._candidate = split
                weighted_gain = split[2] * node.n_samples
                heapq.heappush(frontier, (-weighted_gain, next(tie), node, idx))

        push(self.root_, root_idx)
        while frontier:
            if self.max_leaf_nodes is not None and n_leaves >= self.max_leaf_nodes:
                break
            _, _, node, idx = heapq.heappop(frontier)
            feature, threshold, gain = node._candidate
            node._candidate = None

            go_left = self._X[idx, feature] <= threshold
            left_idx, right_idx = idx[go_left], idx[~go_left]

            node.feature, node.threshold, node.gain = feature, threshold, gain
            node.left = self._make_node(left_idx, node.depth + 1)
            node.right = self._make_node(right_idx, node.depth + 1)
            self.feature_importances_[feature] += gain * len(idx)
            n_leaves += 1  # one leaf became two

            push(node.left, left_idx)
            push(node.right, right_idx)

        for _, _, node, _ in frontier:  # discard unused candidates
            node._candidate = None

    # ------------------------------------------------------------ split search
    def _find_best_split(self, node: Node, idx: np.ndarray):
        """Return (feature, threshold, gain) of the best split, or None."""
        n = len(idx)
        # Stopping criteria (pre-pruning)
        if node.entropy == 0.0:
            return None
        if self.max_depth is not None and node.depth >= self.max_depth:
            return None
        if n < self.min_samples_split or n < 2 * self.min_samples_leaf:
            return None

        X_node, y_node = self._X[idx], self._y[idx]
        best = None
        best_gain = -np.inf
        for j in range(self.n_features_in_):
            result = self._best_split_for_feature(X_node[:, j], y_node, node.entropy)
            if result is not None and result[1] > best_gain + 1e-12:
                best_gain = result[1]
                best = (j, result[0], result[1])

        # A zero-gain split is still allowed (when min_information_gain == 0): on
        # XOR-like data no single split helps, but the next level separates it.
        if best is None or best[2] < self.min_information_gain - 1e-12:
            return None
        return best

    def _best_split_for_feature(self, x: np.ndarray, y: np.ndarray, parent_entropy: float):
        """Evaluate every threshold of one feature at once.

        After sorting by x, the class counts of the left child for the split
        "first i samples go left" are a cumulative sum of one-hot labels, so all
        n-1 candidate splits are scored with a few vectorised operations
        (O(n log n) per feature instead of O(n^2)).
        """
        n = len(x)
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]

        onehot = np.zeros((n, self.n_classes_))
        onehot[np.arange(n), ys] = 1
        left_counts = np.cumsum(onehot, axis=0)[:-1]   # split after position i
        right_counts = left_counts[-1] + onehot[-1] - left_counts

        n_left = np.arange(1, n)
        n_right = n - n_left

        # Valid positions: value actually changes, and both children are big enough
        valid = (xs[1:] > xs[:-1]) & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not valid.any():
            return None

        lc, rc = left_counts[valid], right_counts[valid]
        nl, nr = n_left[valid], n_right[valid]
        children_entropy = (nl * entropy_rows(lc) + nr * entropy_rows(rc)) / n
        gains = parent_entropy - children_entropy

        k = int(np.argmax(gains))
        pos = np.flatnonzero(valid)[k]
        threshold = (xs[pos] + xs[pos + 1]) / 2.0  # midpoint between two values
        return float(threshold), float(gains[k])

    # ----------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not hasattr(self, "root_"):
            raise RuntimeError("This DecisionTreeClassifier is not fitted yet. Call fit() first.")

    def _leaf_for(self, x: np.ndarray) -> Node:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _prepare_X(self, X) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the tree was fitted with {self.n_features_in_}")
        return X

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities = class proportions in the leaf reached."""
        X = self._prepare_X(X)
        proba = np.empty((len(X), self.n_classes_))
        for i, x in enumerate(X):
            counts = self._leaf_for(x).counts
            proba[i] = counts / counts.sum()
        return proba

    def predict(self, X) -> np.ndarray:
        """Predicted class labels (majority class of the leaf reached)."""
        X = self._prepare_X(X)
        return self.classes_[[self._leaf_for(x).prediction for x in X]]

    def score(self, X, y) -> float:
        """Mean accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # --------------------------------------------------------------- inspection
    def _nodes(self):
        self._check_fitted()
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                stack.extend([node.right, node.left])

    def get_depth(self) -> int:
        return max(node.depth for node in self._nodes())

    def get_n_leaves(self) -> int:
        return sum(node.is_leaf for node in self._nodes())

    @property
    def node_count(self) -> int:
        return sum(1 for _ in self._nodes())

    # ------------------------------------------------------------------ display
    def export_text(self, decimals: int = 3, show_proportions: bool = False) -> str:
        from .display import export_text
        return export_text(self, decimals=decimals, show_proportions=show_proportions)

    def export_dot(self, decimals: int = 3) -> str:
        from .display import export_dot
        return export_dot(self, decimals=decimals)

    def plot(self, ax=None, decimals: int = 2, fontsize: int = 9):
        from .display import plot_tree
        return plot_tree(self, ax=ax, decimals=decimals, fontsize=fontsize)
