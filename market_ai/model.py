from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class ModelMetrics:
    n_samples: int
    accuracy: float
    roc_auc: float


def train_logreg_ts(X: pd.DataFrame, y: pd.Series, splits: int = 5) -> Tuple[Pipeline, ModelMetrics]:
    X = X.copy()
    y = y.astype(int)

    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    tscv = TimeSeriesSplit(n_splits=max(2, splits))
    accs, aucs = [], []
    # Fit final model on full data at the end; during CV, evaluate one-step-forward
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        accs.append(float(accuracy_score(yte, pred)))
        try:
            aucs.append(float(roc_auc_score(yte, p)))
        except Exception:
            aucs.append(float('nan'))

    pipe.fit(X, y)

    metrics = ModelMetrics(
        n_samples=int(len(y)),
        accuracy=float(np.nanmean(accs)) if accs else float('nan'),
        roc_auc=float(np.nanmean(aucs)) if aucs else float('nan'),
    )
    return pipe, metrics


def new_online_model() -> Pipeline:
    """Online/streaming-friendly model."""
    return Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1000, tol=1e-3))
    ])


def partial_fit_online(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    y = y.astype(int)
    clf = model.named_steps['clf']
    # Ensure classes are set on first partial_fit
    if not hasattr(clf, 'classes_'):
        clf.partial_fit(model.named_steps['scaler'].fit_transform(X), y, classes=np.array([0, 1]))
    else:
        clf.partial_fit(model.named_steps['scaler'].transform(X), y)
    return model


def save_model(model: Pipeline, model_path: Path, feature_cols: list[str], features_path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
    dump({'model': model, 'meta': meta or {}}, model_path)
    features_path.write_text(json.dumps({'feature_cols': feature_cols}, indent=2), encoding='utf-8')


def load_model(model_path: Path) -> Optional[Dict[str, Any]]:
    if not model_path.exists():
        return None
    return load(model_path)


def load_feature_cols(features_path: Path) -> Optional[list[str]]:
    if not features_path.exists():
        return None
    try:
        obj = json.loads(features_path.read_text(encoding='utf-8'))
        return list(obj.get('feature_cols', []))
    except Exception:
        return None
