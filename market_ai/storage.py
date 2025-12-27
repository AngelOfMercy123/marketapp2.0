from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, default_cols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV if it exists, else return empty DF."""
    if not path.exists():
        return pd.DataFrame(columns=(default_cols or []))
    try:
        return pd.read_csv(path)
    except Exception:
        # If a user manually edited and broke it, don't crash the whole app.
        return pd.DataFrame(columns=(default_cols or []))


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
