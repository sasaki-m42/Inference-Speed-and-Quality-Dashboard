from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from inference_bench.paths import REPO_ROOT


@dataclass
class SampleBatch:
    sample_ids: np.ndarray
    flight_ids: np.ndarray
    timestamps: np.ndarray
    features: np.ndarray
    labels: np.ndarray


@dataclass
class BenchmarkResult:
    model_name: str
    backend: str
    n_samples: int
    batch_size: int
    num_threads: int
    warmup_iters: int
    measure_iters: int
    p50_ms: float
    p95_ms: float
    throughput_sps: float
    notes: str | None = None


def resolve_db_path(path: str | Path) -> Path:
    db_path = Path(path)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    return db_path


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = resolve_db_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.row_factory = sqlite3.Row
    return con


def initialize_schema(con: sqlite3.Connection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    con.executescript(schema_path.read_text(encoding="utf-8"))
    con.commit()


def reset_runtime_tables(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM predictions")
    con.execute("DELETE FROM benchmark")
    con.execute("DELETE FROM samples")
    con.commit()


def _encode_features(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(x, dtype=np.float32).tobytes()


def _decode_features(blob: bytes, n_features: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=n_features).copy()


def insert_samples(
    con: sqlite3.Connection,
    features: np.ndarray,
    labels: np.ndarray,
    flight_ids: Sequence[str],
    timestamps: Sequence[int],
    split: Sequence[str],
    source: str,
) -> int:
    n_samples, n_features = features.shape
    rows = []
    for i in range(n_samples):
        rows.append(
            (
                str(flight_ids[i]),
                int(timestamps[i]),
                source,
                str(split[i]),
                int(n_features),
                _encode_features(features[i]),
                int(labels[i]),
            )
        )

    con.executemany(
        """
        INSERT INTO samples (
          flight_id, ts, source, split, n_features, features, label
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    return n_samples


def fetch_samples(con: sqlite3.Connection, split: str | None = None) -> SampleBatch:
    if split is None:
        query = "SELECT id, flight_id, ts, n_features, features, label FROM samples ORDER BY id"
        rows = con.execute(query).fetchall()
    else:
        query = """
        SELECT id, flight_id, ts, n_features, features, label
        FROM samples
        WHERE split = ?
        ORDER BY id
        """
        rows = con.execute(query, (split,)).fetchall()

    if not rows:
        return SampleBatch(
            sample_ids=np.array([], dtype=np.int64),
            flight_ids=np.array([], dtype=object),
            timestamps=np.array([], dtype=np.int64),
            features=np.empty((0, 0), dtype=np.float32),
            labels=np.array([], dtype=np.int32),
        )

    n_features = int(rows[0]["n_features"])
    X = np.vstack([_decode_features(r["features"], n_features) for r in rows]).astype(np.float32)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    return SampleBatch(
        sample_ids=np.array([int(r["id"]) for r in rows], dtype=np.int64),
        flight_ids=np.array([str(r["flight_id"]) for r in rows], dtype=object),
        timestamps=np.array([int(r["ts"]) for r in rows], dtype=np.int64),
        features=X,
        labels=y,
    )


def replace_predictions(
    con: sqlite3.Connection,
    backend: str,
    model_name: str,
    sample_ids: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> int:
    con.execute(
        "DELETE FROM predictions WHERE backend = ? AND model_name = ?",
        (backend, model_name),
    )
    created_at = int(time.time())
    preds = (scores >= threshold).astype(np.int32)
    rows = [
        (
            int(sample_ids[i]),
            model_name,
            backend,
            float(scores[i]),
            int(preds[i]),
            created_at,
        )
        for i in range(len(sample_ids))
    ]
    con.executemany(
        """
        INSERT INTO predictions (sample_id, model_name, backend, score, pred, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    return len(rows)


def insert_benchmark(con: sqlite3.Connection, result: BenchmarkResult) -> None:
    created_at = int(time.time())
    con.execute(
        "DELETE FROM benchmark WHERE backend = ? AND model_name = ?",
        (result.backend, result.model_name),
    )
    con.execute(
        """
        INSERT INTO benchmark (
          model_name, backend, n_samples, batch_size, num_threads,
          warmup_iters, measure_iters, p50_ms, p95_ms, throughput_sps,
          created_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.model_name,
            result.backend,
            result.n_samples,
            result.batch_size,
            result.num_threads,
            result.warmup_iters,
            result.measure_iters,
            result.p50_ms,
            result.p95_ms,
            result.throughput_sps,
            created_at,
            result.notes,
        ),
    )
    con.commit()


def fetch_predictions_wide(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT
          s.id AS sample_id,
          s.flight_id,
          s.ts,
          s.label,
          py.score AS py_score,
          py.pred AS py_pred,
          cpp_lgbm.score AS cpp_lgbm_score,
          cpp_lgbm.pred AS cpp_lgbm_pred,
          cpp_lr.score AS cpp_lr_score,
          cpp_lr.pred AS cpp_lr_pred
        FROM samples s
        LEFT JOIN predictions py
          ON py.sample_id = s.id AND py.backend = 'py_lgbm'
        LEFT JOIN predictions cpp_lgbm
          ON cpp_lgbm.sample_id = s.id AND cpp_lgbm.backend = 'cpp_lgbm'
        LEFT JOIN predictions cpp_lr
          ON cpp_lr.sample_id = s.id AND cpp_lr.backend = 'cpp_lr'
        WHERE s.split = 'eval'
        ORDER BY s.id
        """
    ).fetchall()


def fetch_benchmark_rows(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT model_name, backend, n_samples, batch_size, num_threads,
               warmup_iters, measure_iters, p50_ms, p95_ms, throughput_sps,
               created_at, notes
        FROM benchmark
        ORDER BY created_at DESC
        """
    ).fetchall()


def count_rows(con: sqlite3.Connection, table: str) -> int:
    allowed = {"samples", "predictions", "benchmark"}
    if table not in allowed:
        raise ValueError(f"unsupported table: {table}")
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
