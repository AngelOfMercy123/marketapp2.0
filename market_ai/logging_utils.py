from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, name: str = 'marketapp', level: int = logging.INFO) -> logging.Logger:
    """Create a rotating-ish logger (simple file + console)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # File handler
    fh = logging.FileHandler(log_dir / f'{name}.log', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
