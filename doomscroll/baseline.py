"""TF-IDF + logistic regression baseline.

If the 1.5B fine-tuned model can't beat this cheap baseline, the extra cost
isn't buying anything. It's also a check on the data: a very high baseline
score means the synthetic posts are easy to tell apart from surface words
(e.g. "skibidi" means brainrot), and real posts may be much harder.
"""

from __future__ import annotations

from typing import Sequence

from doomscroll.schema import LABELS


def make_tfidf_pipeline(seed: int = 0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(max_iter=5000, C=10.0, class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def fit_predict(train_texts: Sequence[str], train_labels: Sequence[str], test_texts: Sequence[str], seed: int = 0) -> list[str]:
    model = make_tfidf_pipeline(seed)
    model.fit(list(train_texts), list(train_labels))
    return [str(p) for p in model.predict(list(test_texts))]


def cross_val_macro_f1(texts: Sequence[str], labels: Sequence[str], groups: Sequence[int],
                       n_splits: int = 5, seed: int = 0) -> dict:
    """Group-aware stratified k-fold CV macro-F1 (near-duplicates never straddle folds)."""
    from sklearn.model_selection import StratifiedGroupKFold

    from doomscroll.metrics import macro_f1

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    texts, labels = list(texts), list(labels)
    scores = []
    for train_idx, test_idx in cv.split(texts, labels, groups):
        preds = fit_predict([texts[i] for i in train_idx], [labels[i] for i in train_idx],
                            [texts[i] for i in test_idx], seed)
        scores.append(macro_f1([labels[i] for i in test_idx], preds, LABELS))
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return {"folds": n_splits, "macro_f1_mean": mean, "macro_f1_std": std, "fold_scores": scores}
