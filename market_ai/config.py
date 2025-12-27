from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Centralized paths so the app can be relocated easily."""

    root: Path = Path('.')

    data_dir: Path = Path('data_prices')
    intraday_dir: Path = Path('data_intraday')
    logs_dir: Path = Path('logs')
    cache_dir: Path = Path('cache')

    universe_csv: Path = Path('universe.csv')
    watchlist_csv: Path = Path('watchlist.csv')

    model_path: Path = Path('model.joblib')
    features_path: Path = Path('feature_cols.json')
    pred_log_csv: Path = Path('predictions.csv')


@dataclass(frozen=True)
class AppConfig:
    # Prediction horizon (trading days) for the model target
    horizon_days: int = 5

    # Model training constraints
    min_history_rows: int = 220  # ~1 trading year
    test_splits: int = 6

    # Auto refresh cadence (seconds) for Streamlit
    refresh_sec: int = 60

    # HTTP basics
    user_agent: str = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/123.0.0.0 Safari/537.36'
    )
