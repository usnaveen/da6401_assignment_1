import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None, stratify=None):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)

    if n_samples != len(y):
        raise ValueError("X and y must contain the same number of samples")

    if 0 < test_size < 1:
        n_test = max(1, int(round(n_samples * test_size)))
    else:
        n_test = int(test_size)

    if not 0 < n_test < n_samples:
        raise ValueError("test_size must leave at least one train and one test sample")

    rng = np.random.RandomState(random_state)

    if stratify is None:
        indices = rng.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        stratify = np.asarray(stratify)
        if len(stratify) != n_samples:
            raise ValueError("stratify must match the number of samples")

        train_parts = []
        test_parts = []
        classes, counts = np.unique(stratify, return_counts=True)

        for cls, count in zip(classes, counts):
            cls_idx = np.where(stratify == cls)[0]
            cls_idx = cls_idx[rng.permutation(len(cls_idx))]
            cls_test = max(1, int(round(count * n_test / n_samples)))
            cls_test = min(cls_test, len(cls_idx) - 1) if len(cls_idx) > 1 else 1
            test_parts.append(cls_idx[:cls_test])
            train_parts.append(cls_idx[cls_test:])

        test_idx = np.concatenate(test_parts)
        train_idx = np.concatenate(train_parts)
        test_idx = test_idx[rng.permutation(len(test_idx))]
        train_idx = train_idx[rng.permutation(len(train_idx))]

        if len(test_idx) > n_test:
            overflow = test_idx[n_test:]
            test_idx = test_idx[:n_test]
            train_idx = np.concatenate([train_idx, overflow])
        elif len(test_idx) < n_test:
            needed = n_test - len(test_idx)
            supplement = train_idx[:needed]
            train_idx = train_idx[needed:]
            test_idx = np.concatenate([test_idx, supplement])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _per_class_counts(y_true, y_pred):
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    counts = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        counts.append((tp, fp, fn))
    return counts


def precision_score(y_true, y_pred, average="macro", zero_division=0):
    if average != "macro":
        raise ValueError("Only average='macro' is supported")
    scores = []
    for tp, fp, _ in _per_class_counts(np.asarray(y_true), np.asarray(y_pred)):
        denom = tp + fp
        scores.append(float(tp / denom) if denom else float(zero_division))
    return float(np.mean(scores)) if scores else 0.0


def recall_score(y_true, y_pred, average="macro", zero_division=0):
    if average != "macro":
        raise ValueError("Only average='macro' is supported")
    scores = []
    for tp, _, fn in _per_class_counts(np.asarray(y_true), np.asarray(y_pred)):
        denom = tp + fn
        scores.append(float(tp / denom) if denom else float(zero_division))
    return float(np.mean(scores)) if scores else 0.0


def f1_score(y_true, y_pred, average="macro", zero_division=0):
    if average != "macro":
        raise ValueError("Only average='macro' is supported")
    scores = []
    for tp, fp, fn in _per_class_counts(np.asarray(y_true), np.asarray(y_pred)):
        precision = float(tp / (tp + fp)) if (tp + fp) else float(zero_division)
        recall = float(tp / (tp + fn)) if (tp + fn) else float(zero_division)
        denom = precision + recall
        scores.append((2 * precision * recall / denom) if denom else float(zero_division))
    return float(np.mean(scores)) if scores else 0.0
