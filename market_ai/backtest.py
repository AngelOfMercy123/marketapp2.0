from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@dataclass
class BacktestResult:
    n: int
    accuracy: float
    auc: float | None
    samples: pd.DataFrame


def walkforward_backtest(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> BacktestResult:
    """Simple walk-forward backtest with TimeSeriesSplit.

    Returns per-fold predictions in `samples`.
    """
    X = X.copy()
    y = y.copy()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ])

    tss = TimeSeriesSplit(n_splits=n_splits)
    preds = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        if len(np.unique(ytr)) < 2:
            continue
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_df = pd.DataFrame({
            'fold': fold,
            'y_true': yte.values,
            'y_pred': pred,
            'p_up': proba,
        }, index=yte.index)
        preds.append(fold_df)

    if not preds:
        return BacktestResult(n=0, accuracy=float('nan'), auc=None, samples=pd.DataFrame())

    samples = pd.concat(preds).sort_index()
    acc = float((samples['y_true'] == samples['y_pred']).mean())

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        if samples['y_true'].nunique() == 2:
            auc = float(roc_auc_score(samples['y_true'], samples['p_up']))
    except Exception:
        auc = None

    return BacktestResult(n=int(len(samples)), accuracy=acc, auc=auc, samples=samples)
