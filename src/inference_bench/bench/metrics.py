from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    auc: float
    confusion: np.ndarray


@dataclass
class BenchmarkStats:
    p50_ms: float
    p95_ms: float
    throughput_sps: float


def compute_classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    preds = (scores >= threshold).astype(np.int32)
    accuracy = float(accuracy_score(y_true, preds))
    unique = np.unique(y_true)
    if len(unique) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, scores))
    confusion = confusion_matrix(y_true, preds, labels=[0, 1])
    return ClassificationMetrics(accuracy=accuracy, auc=auc, confusion=confusion)


def _iterate_batches(x: np.ndarray, batch_size: int):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def benchmark_predictor(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkStats:
    if len(x_eval) == 0:
        return BenchmarkStats(p50_ms=float("nan"), p95_ms=float("nan"), throughput_sps=float("nan"))

    for _ in range(max(warmup_iters, 0)):
        for batch in _iterate_batches(x_eval, batch_size):
            _ = predict_fn(batch)

    latencies = []
    for _ in range(max(measure_iters, 1)):
        start = time.perf_counter()
        for batch in _iterate_batches(x_eval, batch_size):
            _ = predict_fn(batch)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)

    lat_arr = np.array(latencies, dtype=np.float64)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    mean_seconds = float(np.mean(lat_arr) / 1000.0)
    throughput = float(len(x_eval) / mean_seconds) if mean_seconds > 0 else float("inf")

    return BenchmarkStats(p50_ms=p50, p95_ms=p95, throughput_sps=throughput)
