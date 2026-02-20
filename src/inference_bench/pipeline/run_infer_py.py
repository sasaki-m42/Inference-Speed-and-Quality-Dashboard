from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np

from inference_bench.bench.metrics import benchmark_predictor, compute_classification_metrics
from inference_bench.config import load_config
from inference_bench.db.io import BenchmarkResult, connect, fetch_samples, initialize_schema, insert_benchmark, replace_predictions
from inference_bench.models.manifest import default_manifest_path, load_manifest
from inference_bench.ops.run_logger import log_event

DEFAULT_MODEL_NAME = "lgbm_binary_py"
BACKEND = "py_lgbm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Python LightGBM inference and benchmark.")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    parser.add_argument("--model", default=None, help="Path to LightGBM model file")
    parser.add_argument("--manifest", default=None, help="Path to model manifest JSON")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _normalize_scores(raw: np.ndarray, task_mode: str) -> np.ndarray:
    if task_mode == "regression":
        return np.clip(raw.astype(np.float32), 0.0, 1.0)
    return np.clip(raw.astype(np.float32), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    cfg = load_config()

    db_path = args.db or cfg.db_path
    model_path = args.model or cfg.lgbm_model_path
    manifest_path = args.manifest or default_manifest_path(cfg.model_dir)
    manifest = load_manifest(manifest_path) or {}

    task_mode = str(manifest.get("task_mode", "binary"))
    model_names = manifest.get("model_names", {})
    model_name = str(model_names.get("py_lgbm", model_names.get("py", DEFAULT_MODEL_NAME)))
    threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(manifest.get("score_threshold", 0.5))
    )

    con = connect(db_path)
    initialize_schema(con)
    eval_batch = fetch_samples(con, split="eval")

    if len(eval_batch.labels) == 0:
        raise RuntimeError("No eval samples found. Run trainer first.")

    booster = lgb.Booster(model_file=str(model_path))

    def predict_fn(x: np.ndarray) -> np.ndarray:
        try:
            return booster.predict(x, num_threads=args.threads)
        except TypeError:
            return booster.predict(x)

    scored_predict_fn = lambda x: _normalize_scores(predict_fn(x), task_mode)
    scores = scored_predict_fn(eval_batch.features)

    inserted = replace_predictions(
        con,
        backend=BACKEND,
        model_name=model_name,
        sample_ids=eval_batch.sample_ids,
        scores=scores,
        threshold=threshold,
    )

    cls = compute_classification_metrics(eval_batch.labels, scores, threshold=threshold)
    bench = benchmark_predictor(
        predict_fn=scored_predict_fn,
        x_eval=eval_batch.features,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )

    insert_benchmark(
        con,
        BenchmarkResult(
            model_name=model_name,
            backend=BACKEND,
            n_samples=int(len(eval_batch.labels)),
            batch_size=args.batch_size,
            num_threads=args.threads,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            p50_ms=bench.p50_ms,
            p95_ms=bench.p95_ms,
            throughput_sps=bench.throughput_sps,
            notes=f"Python LightGBM (task_mode={task_mode})",
        ),
    )

    log_event(
        category="inference",
        event="run_infer_py",
        status="success",
        details={
            "backend": BACKEND,
            "model_name": model_name,
            "task_mode": task_mode,
            "threshold": threshold,
            "pred_rows": inserted,
            "accuracy": cls.accuracy,
            "auc": cls.auc,
            "p50_ms": bench.p50_ms,
            "p95_ms": bench.p95_ms,
            "throughput_sps": bench.throughput_sps,
        },
    )

    print(f"[infer_py] model={model_path}")
    print(f"[infer_py] manifest={manifest_path}")
    print(f"[infer_py] task_mode={task_mode} model_name={model_name} threshold={threshold}")
    print(f"[infer_py] inserted predictions={inserted}")
    print(f"[infer_py] accuracy={cls.accuracy:.4f} auc={cls.auc:.4f}")
    print(
        "[infer_py] benchmark "
        f"p50={bench.p50_ms:.3f}ms p95={bench.p95_ms:.3f}ms throughput={bench.throughput_sps:.1f} samples/s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_event("inference", "run_infer_py", "failure", {"error": str(exc)})
        raise
