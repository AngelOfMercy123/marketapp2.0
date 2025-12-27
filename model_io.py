from __future__ import annotations

from pathlib import Path

from joblib import dump, load

MODEL_PATH = Path("model.joblib")


def load_model():
    if MODEL_PATH.exists():
        try:
            return load(MODEL_PATH)
        except Exception:
            return None
    return None


def save_model(pipe):
    dump(pipe, MODEL_PATH)
