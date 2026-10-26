import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree

from scratchtree import DecisionTreeClassifier, entropy, information_gain


# ---------------------------------------------------------------- criteria
def test_entropy_values():
    assert entropy(np.array([5, 5])) == pytest.approx(1.0)
    assert entropy(np.array([10, 0])) == 0.0
    assert entropy(np.array([1, 1, 1, 1])) == pytest.approx(2.0)
    assert entropy(np.array([9, 5])) == pytest.approx(0.940, abs=1e-3)  # classic "play tennis"


def test_information_gain_play_tennis_outlook_like():
    parent = np.array([9, 5])
    # perfect split -> gain equals parent entropy
    assert information_gain(parent, np.array([9, 0]), np.array([0, 5])) == pytest.approx(entropy(parent))
    # useless split (same proportions) -> zero gain
    assert information_gain(np.array([4, 4]), np.array([2, 2]), np.array([2, 2])) == pytest.approx(0.0)


# ---------------------------------------------------------------- behaviour
def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y) == 1.0


def test_fits_training_data_perfectly_when_unlimited():
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.score(X, y) == 1.0


def test_string_labels_and_proba():
    X, y = load_iris(return_X_y=True)
    labels = np.array(["setosa", "versicolor", "virginica"])[y]
    tree = DecisionTreeClassifier(max_depth=2).fit(X, labels)
    assert set(tree.predict(X)) <= set(labels)
    proba = tree.predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_single_class_gives_single_leaf():
    X = np.random.rand(20, 3)
    tree = DecisionTreeClassifier().fit(X, np.ones(20))
    assert tree.get_n_leaves() == 1 and tree.get_depth() == 0


# ---------------------------------------------------------------- size hyperparameters
@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5])
def test_max_depth_respected(depth):
    X, y = load_breast_cancer(return_X_y=True)
    assert DecisionTreeClassifier(max_depth=depth).fit(X, y).get_depth() <= depth


@pytest.mark.parametrize("leaves", [1, 2, 5, 10])
def test_max_leaf_nodes_respected(leaves):
    X, y = load_breast_cancer(return_X_y=True)
    assert DecisionTreeClassifier(max_leaf_nodes=leaves).fit(X, y).get_n_leaves() <= leaves


def test_min_samples_leaf_respected():
    X, y = load_breast_cancer(return_X_y=True)
    tree = DecisionTreeClassifier(min_samples_leaf=15).fit(X, y)
    assert min(n.n_samples for n in tree._nodes() if n.is_leaf) >= 15


def test_min_samples_split_respected():
    X, y = load_breast_cancer(return_X_y=True)
    tree = DecisionTreeClassifier(min_samples_split=40).fit(X, y)
    assert all(n.n_samples >= 40 for n in tree._nodes() if not n.is_leaf)


def test_min_information_gain_respected():
    X, y = load_breast_cancer(return_X_y=True)
    tree = DecisionTreeClassifier(min_information_gain=0.05).fit(X, y)
    assert all(n.gain >= 0.05 for n in tree._nodes() if not n.is_leaf)


def test_invalid_params():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1).fit([[0], [1]], [0, 1])
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[0]])


# ---------------------------------------------------------------- vs scikit-learn
@pytest.mark.parametrize("loader", [load_iris, load_wine, load_breast_cancer])
@pytest.mark.parametrize("depth", [1, 2, 3, None])
def test_matches_sklearn_entropy_tree(loader, depth):
    """Same criterion + same stopping rules => (almost always) the same tree.

    Exact equality isn't guaranteed: when two splits have exactly the same gain,
    sklearn picks among them randomly, we pick the first feature.
    """
    X, y = loader(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    ours = DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr).score(Xte, yte)
    # sklearn's result itself depends on random_state through tie-breaking,
    # so we check that ours falls within the range sklearn produces.
    sk = [SkTree(criterion="entropy", max_depth=depth, random_state=s).fit(Xtr, ytr).score(Xte, yte)
          for s in range(10)]
    assert min(sk) - 0.03 <= ours <= max(sk) + 0.03


def test_root_split_identical_to_sklearn():
    X, y = load_breast_cancer(return_X_y=True)
    ours = DecisionTreeClassifier(max_depth=1).fit(X, y)
    sk = SkTree(criterion="entropy", max_depth=1).fit(X, y)
    assert ours.root_.feature == sk.tree_.feature[0]
    assert ours.root_.threshold == pytest.approx(sk.tree_.threshold[0], abs=1e-6)


# ---------------------------------------------------------------- display
def test_display_outputs():
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y, feature_names=["sl", "sw", "pl", "pw"])
    txt = tree.export_text()
    assert "pl <= 2.450" in txt and "class:" in txt
    dot = tree.export_dot()
    assert dot.startswith("digraph") and "->" in dot

    import matplotlib
    matplotlib.use("Agg")
    ax = tree.plot()
    assert ax is not None
