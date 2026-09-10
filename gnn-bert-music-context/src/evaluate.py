"""Evaluation metrics + baselines (B1 random, B4 PCA+MLP)."""
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


def tag_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    aucprs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            aucprs.append(average_precision_score(y_true[:, k], y_prob[:, k]))
    mean_aucpr = float(np.mean(aucprs)) if aucprs else 0.0
    return dict(macro_f1=float(macro_f1), micro_f1=float(micro_f1), mean_aucpr=mean_aucpr)


def emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = float(1 - ss_res / ss_tot)
    return dict(mae=mae, r2=r2)


def random_baseline(y_true: np.ndarray, seed=0):
    rng = np.random.default_rng(seed)
    base_rate = y_true.mean(axis=0, keepdims=True)
    y_prob = np.repeat(base_rate, y_true.shape[0], axis=0)
    y_prob = np.clip(y_prob + rng.normal(0, 1e-3, y_prob.shape), 0, 1)
    return tag_metrics(y_true, y_prob)


def pca_mlp_baseline(X_train, y_train, X_test, y_test, n_components=16, seed=42):
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)
    clf.fit(Xtr, y_train)
    y_prob = clf.predict_proba(Xte)
    if isinstance(y_prob, list):
        y_prob = np.stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in y_prob], axis=1)
    return tag_metrics(y_test, y_prob)