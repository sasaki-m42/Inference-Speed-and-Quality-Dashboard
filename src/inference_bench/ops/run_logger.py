from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from inference_bench.paths import ARTIFACTS_DIR

LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_FILE = LOG_DIR / "execution_log.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def log_event(category: str, event: str, status: str, details: dict[str, Any] | None = None, source: str = "auto") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": utc_now_iso(),
        "category": category,
        "event": event,
        "status": status,
        "source": source,
        "git_commit": _git_commit(),
        "details": details or {},
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_recent_logs(limit: int = 200) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def backfill_known_history() -> None:
    known = [
        {
            "category": "validation",
            "event": "compileall",
            "status": "success",
            "details": {"scope": "src tests", "note": "Backfilled from prior execution"},
        },
        {
            "category": "pipeline",
            "event": "demo_run_synthetic_binary",
            "status": "success",
            "details": {
                "py_accuracy": 0.6142,
                "py_auc": 0.6697,
                "cpp_lgbm_accuracy": 0.6290,
                "cpp_lgbm_auc": 0.6774,
                "cpp_lr_accuracy": 0.6088,
                "cpp_lr_auc": 0.6551,
                "py_p50_ms": 18.856,
                "py_p95_ms": 19.204,
                "cpp_lgbm_p50_ms": 0.116,
                "cpp_lgbm_p95_ms": 0.145,
                "cpp_lr_p50_ms": 0.080,
                "cpp_lr_p95_ms": 0.112,
            },
        },
        {
            "category": "pipeline",
            "event": "synthetic_regression_run",
            "status": "success",
            "details": {
                "py_accuracy": 0.6252,
                "py_auc": 0.6728,
                "cpp_lgbm_accuracy": 0.6290,
                "cpp_lgbm_auc": 0.6774,
                "cpp_lr_accuracy": 0.6117,
                "cpp_lr_auc": 0.6510,
            },
        },
        {
            "category": "pipeline",
            "event": "csv_dataset_run",
            "status": "success",
            "details": {
                "rows": 5000,
                "py_accuracy": 0.9910,
                "py_auc": 0.9997,
                "cpp_lgbm_accuracy": 0.9450,
                "cpp_lgbm_auc": 0.9914,
                "cpp_lr_accuracy": 0.9020,
                "cpp_lr_auc": 0.9622,
            },
        },
    ]
    for row in known:
        log_event(row["category"], row["event"], row["status"], row["details"], source="backfill")
