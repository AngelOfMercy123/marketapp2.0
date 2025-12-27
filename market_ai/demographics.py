from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemographicWeights:
    age_retire: float = 0.35
    physical_age: float = 0.30
    immigrant: float = 0.20
    stability: float = 0.15
    # NOTE: using gender composition in a predictive score can encode bias. Default=0.
    gender_physical: float = 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def compute_demographic_score(row: pd.Series, w: DemographicWeights = DemographicWeights()) -> tuple[float, dict]:
    """Return (score_0_to_10, breakdown).

    Expected row fields:
      avg_workforce_age (years)
      retirement_age (years)
      pct_physical_labor (0-100)
      pct_immigrant (0-100)
      workforce_stability (0-10)
      pct_female (0-100) [optional]
    """
    age = float(row.get('avg_workforce_age', np.nan))
    retire = float(row.get('retirement_age', np.nan))
    phys = float(row.get('pct_physical_labor', np.nan))
    imm = float(row.get('pct_immigrant', np.nan))
    stab = float(row.get('workforce_stability', np.nan))
    female = float(row.get('pct_female', np.nan))

    breakdown = {}

    # 1) Age vs retirement buffer: more buffer => higher score
    if np.isnan(age) or np.isnan(retire):
        age_retire_s = 0.5
        breakdown['age_retire_note'] = 'missing'
    else:
        buffer_years = max(0.0, retire - age)
        # 0 years buffer => 0, 30+ years => 1
        age_retire_s = _clip01(buffer_years / 30.0)
        breakdown['buffer_years'] = round(buffer_years, 2)

    # 2) Physical labor + age interaction: older + more physical => lower
    if np.isnan(age) or np.isnan(phys):
        physical_age_s = 0.5
        breakdown['physical_age_note'] = 'missing'
    else:
        phys01 = _clip01(phys / 100.0)
        # normalize age: 20 => 0, 60 => 1
        age01 = _clip01((age - 20.0) / 40.0)
        # risk increases with age and physical share; convert to score
        risk = phys01 * age01
        physical_age_s = 1.0 - risk
        breakdown['phys_age_risk'] = round(risk, 3)

    # 3) Immigration dependency: higher dependency => more exposure to policy/supply shocks
    if np.isnan(imm):
        immigrant_s = 0.5
        breakdown['immigrant_note'] = 'missing'
    else:
        imm01 = _clip01(imm / 100.0)
        immigrant_s = 1.0 - imm01  # 0% => 1, 100% => 0
        breakdown['imm_share'] = round(imm01, 3)

    # 4) Workforce stability: higher => higher score
    if np.isnan(stab):
        stability_s = 0.5
        breakdown['stability_note'] = 'missing'
    else:
        stability_s = _clip01(stab / 10.0)
        breakdown['stability'] = round(stab, 2)

    # 5) Optional: gender x physical share (default weight 0)
    # We do NOT assume capability differences; if you turn this on, treat it as
    # a proxy for workforce design/ergonomics/training needs in high-physical roles.
    if w.gender_physical > 0 and (not np.isnan(female)) and (not np.isnan(phys)):
        female01 = _clip01(female / 100.0)
        phys01 = _clip01(phys / 100.0)
        # penalty grows if both are high, capped modestly
        penalty = 0.25 * female01 * phys01
        gender_s = 1.0 - penalty
        breakdown['gender_phys_penalty'] = round(penalty, 3)
    else:
        gender_s = 1.0

    # Weighted blend
    total_w = (w.age_retire + w.physical_age + w.immigrant + w.stability + w.gender_physical)
    if total_w <= 0:
        total_w = 1.0

    score01 = (
        w.age_retire * age_retire_s
        + w.physical_age * physical_age_s
        + w.immigrant * immigrant_s
        + w.stability * stability_s
        + w.gender_physical * gender_s
    ) / total_w

    score10 = 10.0 * _clip01(score01)
    breakdown['score10'] = round(score10, 3)
    breakdown['components'] = {
        'age_retire_s': round(age_retire_s, 3),
        'physical_age_s': round(physical_age_s, 3),
        'immigrant_s': round(immigrant_s, 3),
        'stability_s': round(stability_s, 3),
        'gender_s': round(gender_s, 3),
    }

    return score10, breakdown


def add_demographic_scores(universe_df: pd.DataFrame, weights: DemographicWeights = DemographicWeights()) -> pd.DataFrame:
    df = universe_df.copy()
    scores = []
    for _, row in df.iterrows():
        s10, _ = compute_demographic_score(row, weights)
        scores.append(s10)
    df['demographic_score'] = scores
    return df
