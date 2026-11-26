"""Doomscroll classifier: shared schema, data, metric and baseline code.

Nothing in this package imports torch/transformers, so it runs (and is
tested) on a plain CPU machine. The GPU-only code lives in train.py and
predict.py.
"""

from doomscroll.schema import LABELS

__all__ = ["LABELS"]
