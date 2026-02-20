from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from inference_bench.bench.metrics import compute_classification_metrics
from inference_bench.config import load_config
from inference_bench.data.opensky_csv import load_opensky_csv_dataset
from inference_bench.data.synth import generate_synthetic_dataset
from inference_bench.db.io import connect, fetch_samples, initialize_schema, insert_samples, reset_runtime_tables
from inference_bench.inference.cpp_lr import CppLogisticRegression
from inference_bench.models.manifest import build_manifest, default_manifest_path, save_manifest
from inference_bench.ops.run_logger import log_event

PY_MODEL_NAME_FMT = "lgbm_{task_mode}_py"
CPP_LGBM_MODEL_NAME_FMT = "lgbm_{task_mode}_cpp"
CPP_LR_MODEL_NAME = "cpp_lr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset and train benchmark models.")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    parser.add_argument("--dataset-source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--csv-path", default=None, help="CSV dataset path when --dataset-source csv")
    parser.add_argument("--max-csv-rows", type=int, default=50000)

    parser.add_argument("--task-mode", choices=["binary", "regression"], default="binary")
    parser.add_argument("--score-threshold", type=float, default=0.5)

    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["linear", "xor", "hybrid"], default="hybrid")

    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not clear runtime tables before inserting new samples",
    )
    return parser.parse_args()


def _normalize_py_scores(raw_scores: np.ndarray, task_mode: str) -> np.ndarray:
    if task_mode == "regression":
        return np.clip(raw_scores.astype(np.float32), 0.0, 1.0)
    return np.clip(raw_scores.astype(np.float32), 0.0, 1.0)


def _load_dataset(args: argparse.Namespace, cfg):
    if args.dataset_source == "synthetic":
        return generate_synthetic_dataset(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
            mode=args.mode,
        )

    csv_path = args.csv_path or str(cfg.csv_data_path)
    if not csv_path:
        raise ValueError("--csv-path is required when --dataset-source csv")

    return load_opensky_csv_dataset(
        csv_path=csv_path,
        seed=args.seed,
        max_rows=args.max_csv_rows,
    )


def _save_cpp_lr_model(path, weights: np.ndarray, bias: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": CPP_LR_MODEL_NAME,
        "weights": weights.astype(np.float32).tolist(),
        "bias": float(bias),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_lightgbm(train_x: np.ndarray, train_y: np.ndarray, args: argparse.Namespace):
    if args.task_mode == "binary":
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            random_state=args.seed,
            n_jobs=args.threads,
            verbosity=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            random_state=args.seed,
            n_jobs=args.threads,
            verbosity=-1,
        )

    model.fit(train_x, train_y)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config()

    db_path = args.db or cfg.db_path
    con = connect(db_path)
    initialize_schema(con)

    if not args.no_reset:
        reset_runtime_tables(con)

    dataset = _load_dataset(args, cfg)

    inserted = insert_samples(
        con,
        features=dataset.features,
        labels=dataset.labels,
        flight_ids=dataset.flight_ids,
        timestamps=dataset.timestamps,
        split=dataset.split,
        source=args.dataset_source,
    )

    train = fetch_samples(con, split="train")
    eval_batch = fetch_samples(con, split="eval")

    lgbm = _train_lightgbm(
        train_x=train.features,
        train_y=train.labels.astype(np.float32) if args.task_mode == "regression" else train.labels,
        args=args,
    )

    py_model_name = PY_MODEL_NAME_FMT.format(task_mode=args.task_mode)
    cpp_lgbm_model_name = CPP_LGBM_MODEL_NAME_FMT.format(task_mode=args.task_mode)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    booster = lgbm.booster_
    booster.save_model(str(cfg.lgbm_model_path))

    importances = pd.DataFrame(
        {
            "feature": [f"f{i}" for i in range(train.features.shape[1])],
            "importance": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(cfg.model_dir / "feature_importance.csv", index=False)

    cpp_lr = CppLogisticRegression()
    cpp_lr_w, cpp_lr_b = cpp_lr.train(train.features, train.labels, epochs=280, lr=0.09, l2=5e-4)
    _save_cpp_lr_model(cfg.cpp_lr_model_path, cpp_lr_w, cpp_lr_b)

    if args.task_mode == "binary":
        py_raw_scores = lgbm.predict_proba(eval_batch.features)[:, 1]
    else:
        py_raw_scores = lgbm.predict(eval_batch.features)

    py_eval_scores = _normalize_py_scores(py_raw_scores, args.task_mode)
    cpp_lr_scores = cpp_lr.predict_batch(cpp_lr_w, cpp_lr_b, eval_batch.features)
    py_metrics = compute_classification_metrics(eval_batch.labels, py_eval_scores, threshold=args.score_threshold)
    cpp_lr_metrics = compute_classification_metrics(eval_batch.labels, cpp_lr_scores, threshold=args.score_threshold)

    manifest = build_manifest(
        task_mode=args.task_mode,
        dataset_source=args.dataset_source,
        py_lgbm_model_name=py_model_name,
        cpp_lgbm_model_name=cpp_lgbm_model_name,
        cpp_lr_model_name=CPP_LR_MODEL_NAME,
        score_threshold=args.score_threshold,
        n_features=int(train.features.shape[1]),
        notes="Dataset and model metadata for three-model benchmark runs",
    )
    manifest_path = save_manifest(default_manifest_path(cfg.model_dir), manifest)

    metrics_summary = {
        "dataset": {
            "n_samples": inserted,
            "n_features": int(train.features.shape[1]),
            "train_samples": int(len(train.labels)),
            "eval_samples": int(len(eval_batch.labels)),
            "dataset_source": args.dataset_source,
            "generator_mode": args.mode if args.dataset_source == "synthetic" else "csv",
            "seed": args.seed,
        },
        "task_mode": args.task_mode,
        "score_threshold": args.score_threshold,
        "python_lgbm": {
            "model_name": py_model_name,
            "accuracy": py_metrics.accuracy,
            "auc": py_metrics.auc,
        },
        "cpp_lr": {
            "model_name": CPP_LR_MODEL_NAME,
            "accuracy": cpp_lr_metrics.accuracy,
            "auc": cpp_lr_metrics.auc,
        },
    }

    summary_path = cfg.model_dir / "train_metrics.json"
    summary_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(
        category="training",
        event="run_trainer",
        status="success",
        details={
            "dataset_source": args.dataset_source,
            "task_mode": args.task_mode,
            "rows": inserted,
            "eval_rows": int(len(eval_batch.labels)),
            "accuracy": py_metrics.accuracy,
            "auc": py_metrics.auc,
            "model_path": str(cfg.lgbm_model_path),
            "cpp_lr_model_path": str(cfg.cpp_lr_model_path),
            "cpp_lr_accuracy": cpp_lr_metrics.accuracy,
            "cpp_lr_auc": cpp_lr_metrics.auc,
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
        },
    )

    print(f"[trainer] inserted samples: {inserted}")
    print(f"[trainer] dataset_source={args.dataset_source} task_mode={args.task_mode}")
    print(f"[trainer] lgbm model: {cfg.lgbm_model_path}")
    print(f"[trainer] cpp lr model: {cfg.cpp_lr_model_path}")
    print(f"[trainer] manifest: {manifest_path}")
    print(f"[trainer] train summary: {summary_path}")
    print(f"[trainer] py accuracy={py_metrics.accuracy:.4f}, auc={py_metrics.auc:.4f}")
    print(f"[trainer] cpp_lr accuracy={cpp_lr_metrics.accuracy:.4f}, auc={cpp_lr_metrics.auc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_event("training", "run_trainer", "failure", {"error": str(exc)})
        raise
