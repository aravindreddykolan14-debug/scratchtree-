# scratchtree

A decision tree classifier implemented **from scratch** (NumPy only) using **entropy** and **information gain**.

- Binary threshold splits on numeric features, chosen by maximum information gain
- Tree-size hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_information_gain`
- Tree display: text (`export_text`), Graphviz (`export_dot`) and matplotlib (`plot`)
- scikit-learn–like API: `fit`, `predict`, `predict_proba`, `score`, `get_depth`, `get_n_leaves`, `feature_importances_`

## Install

```bash
pip install scratchtree-aravind            # core (numpy only)
pip install "scratchtree-aravind[plot]"    # + matplotlib for tree plots
```

## Usage

```python
from sklearn.datasets import load_iris
from scratchtree import DecisionTreeClassifier

iris = load_iris()
tree = DecisionTreeClassifier(max_depth=3).fit(
    iris.data, iris.target_names[iris.target],
    feature_names=["sepal length", "sepal width", "petal length", "petal width"],
)
print(tree.export_text())
print(tree.predict([[5.1, 3.5, 1.4, 0.2]]))   # ['setosa']

tree.plot()                       # matplotlib figure
open("tree.dot", "w").write(tree.export_dot())   # dot -Tpng tree.dot -o tree.png
```

```
|--- petal length <= 2.450  (n=150, H=1.585, IG=0.918)
|   |--- class: setosa  (n=50, H=0.000, [50, 0, 0])
|--- petal length >  2.450
|   |--- petal width <= 1.750  (n=100, H=1.000, IG=0.690)
...
```

## Hyperparameters

| parameter | default | meaning |
|---|---|---|
| `max_depth` | `None` | maximum depth (root = depth 0) |
| `min_samples_split` | `2` | minimum samples in a node to try splitting it |
| `min_samples_leaf` | `1` | minimum samples in each child of a split |
| `max_leaf_nodes` | `None` | maximum number of leaves (best-first growth) |
| `min_information_gain` | `0.0` | minimum gain (bits) for a split to be accepted |

## Development

```bash
pip install -e ".[dev]"
pytest
python examples/demo.py
```

See `report.md` for a description of the algorithm and results.
