"""scratchtree — a decision tree classifier built from scratch with entropy and information gain."""

from .criteria import entropy, information_gain
from .display import export_dot, export_text, plot_tree
from .tree import DecisionTreeClassifier, Node

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeClassifier",
    "Node",
    "entropy",
    "information_gain",
    "export_text",
    "export_dot",
    "plot_tree",
]
