from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelManifest:
    schema_version: str
    created_at: int
    task_mode: str
    dataset_source: str
    py_lgbm_model_name: str
    cpp_lgbm_model_name: str
    cpp_lr_model_name: str
    score_threshold: float
    n_features: int
    notes: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "task_mode": self.task_mode,
            "dataset_source": self.dataset_source,
            "model_names": {
                "py_lgbm": self.py_lgbm_model_name,
                "cpp_lgbm": self.cpp_lgbm_model_name,
                "cpp_lr": self.cpp_lr_model_name,
            },
            "score_threshold": self.score_threshold,
            "n_features": self.n_features,
            "notes": self.notes,
        }


def default_manifest_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "model_manifest.json"


def save_manifest(path: str | Path, manifest: ModelManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_manifest(
    task_mode: str,
    dataset_source: str,
    py_lgbm_model_name: str,
    cpp_lgbm_model_name: str,
    cpp_lr_model_name: str,
    score_threshold: float,
    n_features: int,
    notes: str,
) -> ModelManifest:
    return ModelManifest(
        schema_version="1.0",
        created_at=int(time.time()),
        task_mode=task_mode,
        dataset_source=dataset_source,
        py_lgbm_model_name=py_lgbm_model_name,
        cpp_lgbm_model_name=cpp_lgbm_model_name,
        cpp_lr_model_name=cpp_lr_model_name,
        score_threshold=score_threshold,
        n_features=n_features,
        notes=notes,
    )
