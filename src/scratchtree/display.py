"""Ways to display a fitted tree: plain text, Graphviz DOT, and a matplotlib plot."""

from __future__ import annotations

from typing import List


def _fmt_counts(tree, node) -> str:
    return "[" + ", ".join(str(int(c)) for c in node.counts) + "]"


def _class_name(tree, node) -> str:
    return str(tree.classes_[node.prediction])


def export_text(tree, decimals: int = 3, show_proportions: bool = False) -> str:
    """Return an ASCII drawing of the tree, e.g.::

        |--- petal length <= 2.450  (IG=0.918)
        |   |--- class: setosa  (n=50, H=0.000)
        |--- petal length >  2.450
        |   |--- ...
    """
    tree._check_fitted()
    lines: List[str] = []

    def leaf_info(node) -> str:
        if show_proportions:
            props = node.counts / node.counts.sum()
            dist = "[" + ", ".join(f"{p:.2f}" for p in props) + "]"
        else:
            dist = _fmt_counts(tree, node)
        return f"class: {_class_name(tree, node)}  (n={node.n_samples}, H={node.entropy:.{decimals}f}, {dist})"

    def recurse(node, prefix: str) -> None:
        if node.is_leaf:
            lines.append(f"{prefix}|--- {leaf_info(node)}")
            return
        name = tree.feature_names_[node.feature]
        thr = f"{node.threshold:.{decimals}f}"
        lines.append(f"{prefix}|--- {name} <= {thr}  (n={node.n_samples}, H={node.entropy:.{decimals}f}, IG={node.gain:.{decimals}f})")
        recurse(node.left, prefix + "|   ")
        lines.append(f"{prefix}|--- {name} >  {thr}")
        recurse(node.right, prefix + "|   ")

    recurse(tree.root_, "")
    return "\n".join(lines)


def export_dot(tree, decimals: int = 3) -> str:
    """Return the tree in Graphviz DOT format (render with ``dot -Tpng tree.dot -o tree.png``)."""
    tree._check_fitted()
    out = [
        "digraph DecisionTree {",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#f5f5f5"];',
        '  edge [fontname="Helvetica"];',
    ]

    def recurse(node) -> None:
        if node.is_leaf:
            label = (
                f"class = {_class_name(tree, node)}\\n"
                f"samples = {node.n_samples}\\nvalue = {_fmt_counts(tree, node)}\\n"
                f"entropy = {node.entropy:.{decimals}f}"
            )
            out.append(f'  n{node.node_id} [label="{label}", fillcolor="#dbeafe"];')
            return
        label = (
            f"{tree.feature_names_[node.feature]} <= {node.threshold:.{decimals}f}\\n"
            f"samples = {node.n_samples}\\nvalue = {_fmt_counts(tree, node)}\\n"
            f"entropy = {node.entropy:.{decimals}f}\\ngain = {node.gain:.{decimals}f}"
        )
        out.append(f'  n{node.node_id} [label="{label}"];')
        out.append(f'  n{node.node_id} -> n{node.left.node_id} [label="True"];')
        out.append(f'  n{node.node_id} -> n{node.right.node_id} [label="False"];')
        recurse(node.left)
        recurse(node.right)

    recurse(tree.root_)
    out.append("}")
    return "\n".join(out)


def plot_tree(tree, ax=None, decimals: int = 2, fontsize: int = 9):
    """Draw the tree with matplotlib (``pip install scratchtree-aravind[plot]``).

    Layout: leaves are spaced evenly left-to-right (in-order), each internal
    node is centred above its children, depth sets the vertical position.
    Leaf colour = predicted class; colour intensity = purity (1 - normalised entropy).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot() needs matplotlib: pip install 'scratchtree-aravind[plot]'") from exc
    import numpy as np

    tree._check_fitted()
    positions = {}
    next_leaf_x = [0]

    def layout(node) -> float:
        if node.is_leaf:
            x = next_leaf_x[0]
            next_leaf_x[0] += 1
        else:
            x = (layout(node.left) + layout(node.right)) / 2
        positions[node.node_id] = (x, -node.depth)
        return x

    layout(tree.root_)
    n_leaves = max(next_leaf_x[0], 1)
    depth = tree.get_depth()

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 2.2 * n_leaves), max(3, 1.9 * (depth + 1))))
    ax.set_axis_off()

    cmap = colormaps.get_cmap("tab10")
    max_h = np.log2(tree.n_classes_) if tree.n_classes_ > 1 else 1.0

    def draw(node) -> None:
        x, y = positions[node.node_id]
        if not node.is_leaf:
            for child, label in ((node.left, "True"), (node.right, "False")):
                cx, cy = positions[child.node_id]
                ax.plot([x, cx], [y, cy], color="#888888", lw=1, zorder=1)
                ax.text((x + cx) / 2, (y + cy) / 2, label, fontsize=fontsize - 2,
                        ha="center", va="center", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
            text = (f"{tree.feature_names_[node.feature]} <= {node.threshold:.{decimals}f}\n"
                    f"n={node.n_samples}  H={node.entropy:.{decimals}f}\nIG={node.gain:.{decimals}f}")
        else:
            text = (f"class: {_class_name(tree, node)}\n"
                    f"n={node.n_samples}  H={node.entropy:.{decimals}f}\n{_fmt_counts(tree, node)}")

        purity = 1 - node.entropy / max_h
        r, g, b, _ = cmap(node.prediction % 10)
        alpha = 0.15 + 0.6 * purity
        face = (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=2,
                bbox=dict(boxstyle="round,pad=0.4", fc=face, ec="#444444", lw=0.8))
        if not node.is_leaf:
            draw(node.left)
            draw(node.right)

    draw(tree.root_)
    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-depth - 0.6, 0.6)
    return ax
