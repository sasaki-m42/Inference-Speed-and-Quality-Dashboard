from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from inference_bench.paths import ARTIFACTS_DIR, REPO_ROOT


class AppConfig(BaseModel):
    db_path: Path = Field(default=ARTIFACTS_DIR / "bench.db")
    model_dir: Path = Field(default=ARTIFACTS_DIR / "models")
    lgbm_model_path: Path = Field(default=ARTIFACTS_DIR / "models" / "lgbm_model.txt")
    cpp_lr_model_path: Path = Field(default=ARTIFACTS_DIR / "models" / "cpp_lr_model.json")
    csv_data_path: Path = Field(default=REPO_ROOT / "data" / "raw" / "sample.csv")
    sample_source: str = Field(default="synthetic")
    random_seed: int = Field(default=42)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_config() -> AppConfig:
    load_dotenv(REPO_ROOT / ".env")
    return AppConfig(
        db_path=_resolve_path(os.getenv("APP_DB_PATH", ARTIFACTS_DIR / "bench.db")),
        model_dir=_resolve_path(os.getenv("APP_MODEL_DIR", ARTIFACTS_DIR / "models")),
        lgbm_model_path=_resolve_path(os.getenv("APP_LGBM_MODEL", ARTIFACTS_DIR / "models" / "lgbm_model.txt")),
        cpp_lr_model_path=_resolve_path(os.getenv("APP_CPP_MODEL", ARTIFACTS_DIR / "models" / "cpp_lr_model.json")),
        csv_data_path=_resolve_path(os.getenv("APP_CSV_DATA_PATH", REPO_ROOT / "data" / "raw" / "sample.csv")),
        sample_source=os.getenv("APP_SAMPLE_SOURCE", "synthetic"),
        random_seed=int(os.getenv("APP_RANDOM_SEED", "42")),
    )
