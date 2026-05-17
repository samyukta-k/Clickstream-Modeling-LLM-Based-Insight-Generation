from pathlib import Path

from . import config


def ensure_dirs() -> None:
    for path in [
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.CHARTS_DIR,
        config.MODELS_DIR,
        config.PREDICTIONS_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
