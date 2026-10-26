"""Demo: fit the from-scratch tree, compare with scikit-learn, sweep the size hyperparameter.

Run:  python examples/demo.py
Needs the dev extras: pip install -e ".[dev]"
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree

from scratchtree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(exist_ok=True)

# 1) Iris: print and plot a small tree ------------------------------------
iris = load_iris()
names = ["sepal length", "sepal width", "petal length", "petal width"]
tree = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target_names[iris.target], feature_names=names)
print("Iris tree (max_depth=3)\n")
print(tree.export_text())
ax = tree.plot()
ax.set_title("scratchtree on Iris (max_depth=3)")
plt.tight_layout()
plt.savefig(OUT / "iris_tree.png", dpi=150)
plt.close()
(OUT / "iris_tree.dot").write_text(tree.export_dot())

# 2) Comparison with scikit-learn ------------------------------------------
print("\nComparison with sklearn DecisionTreeClassifier(criterion='entropy') — test accuracy, 30% hold-out\n")
print(f"{'dataset':<15}{'max_depth':>10}{'scratchtree':>13}{'sklearn':>10}{'fit time (ours)':>17}")
rows = []
for name, loader in [("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer)]:
    X, y = loader(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    for depth in [2, 4, None]:
        t0 = time.perf_counter()
        ours = DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr)
        dt = time.perf_counter() - t0
        sk = SkTree(criterion="entropy", max_depth=depth, random_state=0).fit(Xtr, ytr)
        row = (name, str(depth), ours.score(Xte, yte), sk.score(Xte, yte), dt)
        rows.append(row)
        print(f"{row[0]:<15}{row[1]:>10}{row[2]:>13.3f}{row[3]:>10.3f}{row[4]*1000:>14.1f} ms")

# 3) Size hyperparameter sweep (breast cancer) ------------------------------
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
depths = list(range(0, 11))
train_acc, test_acc, leaves = [], [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d).fit(Xtr, ytr)
    train_acc.append(t.score(Xtr, ytr))
    test_acc.append(t.score(Xte, yte))
    leaves.append(t.get_n_leaves())

print("\nmax_depth sweep on breast_cancer")
print(f"{'depth':>6}{'leaves':>8}{'train':>8}{'test':>8}")
for d, l, a, b in zip(depths, leaves, train_acc, test_acc):
    print(f"{d:>6}{l:>8}{a:>8.3f}{b:>8.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, train_acc, marker="o", label="train accuracy")
ax.plot(depths, test_acc, marker="o", label="test accuracy")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.set_title("Effect of tree size (breast cancer, scratchtree)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "depth_sweep.png", dpi=150)
plt.close()

leaf_budgets = [2, 3, 4, 6, 8, 12, 16, 24, 32]
leaf_test = [DecisionTreeClassifier(max_leaf_nodes=k).fit(Xtr, ytr).score(Xte, yte) for k in leaf_budgets]
print("\nmax_leaf_nodes sweep on breast_cancer")
for k, a in zip(leaf_budgets, leaf_test):
    print(f"  max_leaf_nodes={k:<3} test={a:.3f}")
print(f"\nImages written to {OUT}")
