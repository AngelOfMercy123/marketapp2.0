from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demographics import compute_demographic_score


@dataclass(frozen=True)
class ScoreConfig:
    structural_weight: float = 0.60
    demographic_weight: float = 0.40

    # Structural sub-weights (sum not required; we normalize)
    utility_w: float = 0.25
    policy_w: float = 0.25
    leadership_w: float = 0.25
    labor_exposure_w: float = 0.15
    trend_w: float = 0.10  # e.g., manual "trend_score" or jan_score


def compute_structural_score(df_u: pd.DataFrame) -> pd.Series:
    # Fill missing with mid (5)
    u = df_u.copy()
    def col(name: str, default: float = 5.0) -> pd.Series:
        if name not in u.columns:
            return pd.Series(default, index=u.index, dtype=float)
        return pd.to_numeric(u[name], errors='coerce').fillna(default)

    utility = col('utility_score')
    policy = col('policy_score')
    leadership = col('leadership_score')
    labor_exposure = col('labor_exposure', default=5.0)
    trend = col('trend_score', default=5.0)

    weights = np.array([0.25, 0.25, 0.25, 0.15, 0.10], dtype=float)
    total = weights.sum()
    # Map labor_exposure: higher means more exposure/risk => invert to score
    labor_score = (10.0 - labor_exposure.clip(0, 10))

    score = (
        utility * weights[0]
        + policy * weights[1]
        + leadership * weights[2]
        + labor_score * weights[3]
        + trend * weights[4]
    ) / total
    return score.clip(0, 10)


def compute_company_scores(universe_df: pd.DataFrame, cfg: ScoreConfig = ScoreConfig()) -> pd.DataFrame:
    out = universe_df.copy()
    out['structural_score'] = compute_structural_score(out)
    out['demographic_score'] = compute_demographic_score(out)

    sw = float(cfg.structural_weight)
    dw = float(cfg.demographic_weight)
    denom = sw + dw if (sw + dw) != 0 else 1.0
    out['thesis_score'] = (out['structural_score'] * sw + out['demographic_score'] * dw) / denom
    out['thesis_score'] = out['thesis_score'].clip(0, 10)
    return out
