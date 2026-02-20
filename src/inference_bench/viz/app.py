from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, auc, f1_score, precision_score, recall_score, roc_curve

from inference_bench.config import load_config

matplotlib.use("Agg", force=True)

st.set_page_config(page_title="Inference Speed and Quality Dashboard", layout="wide")

BACKEND_ORDER = ["py_lgbm", "cpp_lgbm", "cpp_lr"]
BACKEND_LABELS = {
    "py_lgbm": "Python LightGBM",
    "cpp_lgbm": "C++ LightGBM",
    "cpp_lr": "C++ Logistic Regression",
}
BACKEND_COLORS = {
    "py_lgbm": "#2563eb",
    "cpp_lgbm": "#f59e0b",
    "cpp_lr": "#10b981",
}
POSTPROC_ON = 0.55
POSTPROC_OFF = 0.45
PARITY_EPS = 1e-12
JST = ZoneInfo("Asia/Tokyo")
CUSTOM_DB_ENABLE_ENV = "INFERENCE_BENCH_ENABLE_CUSTOM_DB"
PUBLIC_FIXED_DB_PATH = "/app/artifacts/bench.db"
REQUIRED_DB_SCHEMA: dict[str, set[str]] = {
    "samples": {"id", "flight_id", "ts", "source", "split", "label", "n_features"},
    "predictions": {"sample_id", "backend", "model_name", "score", "pred"},
    "benchmark": {
        "model_name",
        "backend",
        "n_samples",
        "batch_size",
        "num_threads",
        "warmup_iters",
        "measure_iters",
        "p50_ms",
        "p95_ms",
        "throughput_sps",
        "created_at",
        "notes",
    },
}


def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _to_sqlite_ro_uri(db_path: str | Path) -> str:
    path = Path(db_path).expanduser().resolve()
    return f"file:{quote(path.as_posix(), safe='/')}?mode=ro"


def _normalize_db_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _discover_db_candidates(default_db_path: str) -> list[str]:
    candidates: set[str] = {_normalize_db_path(default_db_path)}
    artifacts_dir = Path.cwd() / "artifacts"
    if artifacts_dir.exists():
        for db_file in artifacts_dir.rglob("*.db"):
            if db_file.is_file():
                candidates.add(str(db_file.resolve()))
    return sorted(candidates)


def _validate_db_schema(db_path: str) -> str | None:
    try:
        con = sqlite3.connect(_to_sqlite_ro_uri(db_path), uri=True)
    except Exception as exc:
        return f"DB を読み取り専用で開けません: {exc}"

    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {str(row[0]) for row in rows}
        missing_tables = sorted(set(REQUIRED_DB_SCHEMA.keys()) - table_names)
        if missing_tables:
            missing = ", ".join(missing_tables)
            return f"DB schema が不正です。必要テーブルが不足しています: {missing}"

        for table, required_cols in REQUIRED_DB_SCHEMA.items():
            col_rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            col_names = {str(row[1]) for row in col_rows}
            missing_cols = sorted(required_cols - col_names)
            if missing_cols:
                missing = ", ".join(missing_cols)
                return f"DB schema が不正です。{table} に必要列がありません: {missing}"
    finally:
        con.close()

    return None


def _sidebar_db_selector(default_db_path: str) -> str:
    st.sidebar.subheader("Data Source")

    custom_db_enabled = _is_env_flag_enabled(CUSTOM_DB_ENABLE_ENV, default=False)
    normalized_default = _normalize_db_path(default_db_path)

    if not custom_db_enabled:
        selected_db = _normalize_db_path(PUBLIC_FIXED_DB_PATH)
        st.sidebar.caption("DB (fixed)")
        st.sidebar.code(selected_db)
        if st.sidebar.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        return selected_db

    candidates = _discover_db_candidates(normalized_default)
    custom_option = "Custom path..."
    options = candidates + [custom_option]

    default_idx = 0
    if normalized_default in candidates:
        default_idx = candidates.index(normalized_default)

    selected_option = st.sidebar.selectbox("DB path", options=options, index=default_idx)
    if selected_option == custom_option:
        custom_default = st.session_state.get("db_custom_path", normalized_default)
        custom_input = st.sidebar.text_input("Custom DB path", value=custom_default, key="db_custom_path")
        selected_db = _normalize_db_path(custom_input)
    else:
        selected_db = _normalize_db_path(selected_option)

    st.sidebar.caption("Selected DB")
    st.sidebar.code(selected_db)
    st.sidebar.caption("Custom DB is enabled (dev mode).")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return selected_db


def _render_sidebar_common_info(samples: pd.DataFrame, bench_latest: pd.DataFrame, db_path: str, read_error: str | None) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("共通情報")
    st.sidebar.caption(f"DB: {db_path}")

    db_exists = Path(db_path).exists()
    if not db_exists:
        st.sidebar.warning("DB not found")
        return
    if read_error is not None:
        st.sidebar.error("DB read failed")
        return

    eval_rows = int((samples["split"] == "eval").sum()) if not samples.empty else 0
    if bench_latest.empty:
        st.sidebar.caption("最新実行: -")
        st.sidebar.caption("eval件数: -, batch=- / threads=-, warmup=- / measure=-")
        return

    latest_ts = int(bench_latest["created_at"].max())
    ts_text = datetime.fromtimestamp(latest_ts, tz=timezone.utc).astimezone(JST).strftime("%Y-%m-%d %H:%M JST")
    st.sidebar.caption(f"最新実行: {ts_text}")
    st.sidebar.caption(
        f"eval件数: {eval_rows:,}, "
        f"batch={_uniform_benchmark_value(bench_latest, 'batch_size')} / "
        f"threads={_uniform_benchmark_value(bench_latest, 'num_threads')}, "
        f"warmup={_uniform_benchmark_value(bench_latest, 'warmup_iters')} / "
        f"measure={_uniform_benchmark_value(bench_latest, 'measure_iters')}"
    )


@st.cache_data(show_spinner=False)
def _read_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(_to_sqlite_ro_uri(db_path), uri=True)
    try:
        samples = pd.read_sql_query(
            "SELECT id, flight_id, ts, source, split, label, n_features FROM samples ORDER BY id",
            con,
        )
        preds = pd.read_sql_query(
            """
            SELECT p.sample_id, p.backend, p.model_name, p.score, p.pred,
                   s.flight_id, s.ts, s.label, s.split, s.source
            FROM predictions p
            JOIN samples s ON s.id = p.sample_id
            ORDER BY p.sample_id
            """,
            con,
        )
        bench = pd.read_sql_query(
            """
            SELECT model_name, backend, n_samples, batch_size, num_threads,
                   warmup_iters, measure_iters, p50_ms, p95_ms, throughput_sps,
                   created_at, datetime(created_at, 'unixepoch') as created_at_text, notes
            FROM benchmark
            ORDER BY created_at DESC
            """,
            con,
        )
        return samples, preds, bench
    finally:
        con.close()


def _feature_importance_table(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).sort_values("importance", ascending=False)


def _latest_benchmark(bench: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "backend",
        "model_name",
        "n_samples",
        "batch_size",
        "num_threads",
        "warmup_iters",
        "measure_iters",
        "p50_ms",
        "p95_ms",
        "throughput_sps",
        "created_at",
        "created_at_text",
        "notes",
    ]
    if bench.empty:
        return pd.DataFrame(columns=cols).set_index("backend")
    latest = bench.sort_values("created_at").groupby("backend", as_index=False).tail(1).set_index("backend")
    return latest


def _compute_quality_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "backend", "label", "pred", "score"}
    if not required.issubset(set(preds.columns)):
        return pd.DataFrame(
            columns=["backend", "label", "accuracy", "precision", "recall", "f1", "auc", "n_eval"]
        ).set_index("backend")

    rows: list[dict[str, object]] = []
    eval_df = preds[preds["split"] == "eval"]
    for backend in BACKEND_ORDER:
        d = eval_df[eval_df["backend"] == backend]
        if d.empty:
            continue
        y_true = d["label"].astype(int).to_numpy()
        y_pred = d["pred"].astype(int).to_numpy()
        y_score = d["score"].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            auc_value = float("nan")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = float(auc(fpr, tpr))

        rows.append(
            {
                "backend": backend,
                "label": BACKEND_LABELS.get(backend, backend),
                "accuracy": float((y_true == y_pred).mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc": auc_value,
                "n_eval": int(len(d)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["backend", "label", "accuracy", "precision", "recall", "f1", "auc", "n_eval"],
    ).set_index("backend")


def _compute_lgbm_parity(preds: pd.DataFrame) -> float:
    required = {"split", "backend", "sample_id", "pred"}
    if not required.issubset(set(preds.columns)):
        return float("nan")

    eval_df = preds[preds["split"] == "eval"]
    py = eval_df[eval_df["backend"] == "py_lgbm"][["sample_id", "pred"]].rename(columns={"pred": "pred_py"})
    cpp = eval_df[eval_df["backend"] == "cpp_lgbm"][["sample_id", "pred"]].rename(columns={"pred": "pred_cpp"})
    merged = py.merge(cpp, on="sample_id", how="inner")
    if merged.empty:
        return float("nan")
    match_rate = (merged["pred_py"].astype(int) == merged["pred_cpp"].astype(int)).mean()
    return float(match_rate)


def _uniform_benchmark_value(bench_latest: pd.DataFrame, col: str) -> str:
    vals = bench_latest[col].dropna().unique().tolist()
    if len(vals) != 1:
        return "mixed"
    value = vals[0]
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return str(int(value)) if float(value).is_integer() else f"{value:.4g}"
    return str(value)


def _winner(series: pd.Series, maximize: bool = True) -> tuple[list[str], float]:
    clean = series.dropna()
    if clean.empty:
        return [], float("nan")
    best = float(clean.max() if maximize else clean.min())
    tol = max(1e-10, abs(best) * 1e-9)
    winners = [str(idx) for idx, value in clean.items() if abs(float(value) - best) <= tol]
    return winners, best


def _winner_text(backends: list[str]) -> str:
    if not backends:
        return "-"
    if len(backends) == 1:
        return BACKEND_LABELS.get(backends[0], backends[0])
    labels = [BACKEND_LABELS.get(name, name) for name in backends]
    return f"同率1位: {' / '.join(labels)}"


def _format_value(value: float, style: str) -> str:
    if np.isnan(value):
        return "-"
    if style == "ms":
        return f"{value:.3f} ms"
    if style == "throughput":
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M samples/s"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K samples/s"
        return f"{value:.0f} samples/s"
    if style == "percent":
        return f"{value * 100.0:.2f}%"
    return f"{value:.3f}"


def _comparison_table(quality: pd.DataFrame, bench_latest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = bench_latest.loc["py_lgbm"] if "py_lgbm" in bench_latest.index else None

    for backend in BACKEND_ORDER:
        q = quality.loc[backend] if backend in quality.index else None
        b = bench_latest.loc[backend] if backend in bench_latest.index else None

        row = {
            "Model": BACKEND_LABELS.get(backend, backend),
            "Recall (%)": (float(q["recall"]) * 100.0) if q is not None else np.nan,
            "Precision (%)": (float(q["precision"]) * 100.0) if q is not None else np.nan,
            "F1 (%)": (float(q["f1"]) * 100.0) if q is not None else np.nan,
            "Accuracy (%)": (float(q["accuracy"]) * 100.0) if q is not None else np.nan,
            "AUC (0-1)": float(q["auc"]) if q is not None else np.nan,
            "p50 (ms)": float(b["p50_ms"]) if b is not None else np.nan,
            "p95 (ms)": float(b["p95_ms"]) if b is not None else np.nan,
            "Throughput (samples/s)": float(b["throughput_sps"]) if b is not None else np.nan,
            "vs py_lgbm": "-",
        }

        if baseline is not None and b is not None:
            p50_ratio = float(baseline["p50_ms"]) / float(b["p50_ms"]) if float(b["p50_ms"]) > 0 else np.nan
            th_ratio = (
                float(b["throughput_sps"]) / float(baseline["throughput_sps"])
                if float(baseline["throughput_sps"]) > 0
                else np.nan
            )
            row["vs py_lgbm"] = f"p50 x{p50_ratio:.2f} / th x{th_ratio:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


def _plot_speed_metric(bench_latest: pd.DataFrame, metric: str, title: str, log_scale: bool = False):
    source = bench_latest.reindex(BACKEND_ORDER).dropna(subset=[metric])
    if source.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    colors = [BACKEND_COLORS.get(idx, "#6b7280") for idx in source.index]
    bars = ax.bar(source.index.tolist(), source[metric].astype(float).to_numpy(), color=colors)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    for bar in bars:
        value = bar.get_height()
        label = f"{value:,.0f}" if metric == "throughput_sps" else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_quality_bars(quality: pd.DataFrame):
    source = quality.reindex(BACKEND_ORDER).dropna(subset=["recall", "precision", "f1"], how="all")
    if source.empty:
        return None

    metrics = ["recall", "precision", "f1"]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    backends = source.index.tolist()
    max_value = 0.0
    for idx, backend in enumerate(backends):
        offset = (idx - (len(backends) - 1) / 2.0) * width
        values = [float(source.loc[backend, m]) for m in metrics]
        bars = ax.bar(x + offset, values, width=width, label=backend, color=BACKEND_COLORS.get(backend, "#6b7280"))
        for bar in bars:
            value = float(bar.get_height())
            max_value = max(max_value, value)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, max(1.05, max_value + 0.08))
    ax.set_title("Recall / Precision / F1")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_roc(preds: pd.DataFrame):
    eval_df = preds[preds["split"] == "eval"]
    if eval_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for backend in BACKEND_ORDER:
        d = eval_df[eval_df["backend"] == backend]
        if d.empty or d["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d["label"], d["score"])
        auc_value = auc(fpr, tpr)
        label = BACKEND_LABELS.get(backend, backend)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.3f})", color=BACKEND_COLORS.get(backend, "#6b7280"))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_confusion(preds: pd.DataFrame, backend: str):
    d = preds[(preds["split"] == "eval") & (preds["backend"] == backend)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(3.1, 3.1))
    disp = ConfusionMatrixDisplay.from_predictions(d["label"], d["pred"], ax=ax, cmap="Blues", colorbar=False)
    disp.ax_.set_title(backend)
    fig.tight_layout()
    return fig


def _positive_segments(x: np.ndarray, y01: np.ndarray) -> list[tuple[float, float]]:
    if len(x) == 0 or len(y01) == 0:
        return []

    x_arr = x.astype(float)
    y_arr = y01.astype(np.int32)
    if len(x_arr) != len(y_arr):
        n = min(len(x_arr), len(y_arr))
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
    if len(x_arr) == 0:
        return []

    if len(x_arr) > 1:
        diffs = np.diff(x_arr)
        valid_diffs = diffs[diffs > 0]
        default_step = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else 1.0
        last_step = float(diffs[-1]) if diffs[-1] > 0 else default_step
    else:
        default_step = 1.0
        last_step = 1.0

    edges = np.empty(len(x_arr) + 1, dtype=float)
    edges[:-1] = x_arr
    edges[-1] = x_arr[-1] + (last_step if last_step > 0 else default_step)

    segments: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_positive in enumerate(y_arr == 1):
        if is_positive and start_idx is None:
            start_idx = idx
        elif not is_positive and start_idx is not None:
            segments.append((edges[start_idx], edges[idx]))
            start_idx = None
    if start_idx is not None:
        segments.append((edges[start_idx], edges[len(y_arr)]))
    return segments


def _upward_threshold_crossings(scores: np.ndarray, threshold: float) -> np.ndarray:
    if len(scores) < 2:
        return np.array([], dtype=np.int64)
    above = np.asarray(scores >= threshold, dtype=bool)
    idx = np.where((~above[:-1]) & above[1:])[0] + 1
    return idx.astype(np.int64)


def _alarm_onsets(binary_signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(binary_signal >= 0.5, dtype=bool)
    if len(signal) == 0:
        return np.array([], dtype=np.int64)
    if len(signal) >= 2:
        idx = np.where((~signal[:-1]) & signal[1:])[0] + 1
    else:
        idx = np.array([], dtype=np.int64)
    if signal[0]:
        idx = np.r_[0, idx]
    return np.asarray(idx, dtype=np.int64)


def _postproc_hysteresis(scores: np.ndarray, on: float = POSTPROC_ON, off: float = POSTPROC_OFF) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.int32)
    state = 0
    for i, score in enumerate(scores):
        if np.isfinite(score):
            if state == 0 and score >= on:
                state = 1
            elif state == 1 and score <= off:
                state = 0
        out[i] = state
    return out


def _plot_flight_timeline(
    preds: pd.DataFrame,
    flight_id: str,
    decision_backend: str,
    threshold: float = 0.5,
    parity_eps: float = PARITY_EPS,
):
    d = preds[(preds["split"] == "eval") & (preds["flight_id"] == flight_id)].copy()
    if d.empty:
        return None, None
    d = d.sort_values("ts")
    ts_index = np.sort(d["ts"].unique())
    if len(ts_index) == 0:
        return None, None

    score_tbl = d.pivot_table(index="ts", columns="backend", values="score", aggfunc="mean").reindex(ts_index)
    label_ts = d.groupby("ts")["label"].max().reindex(ts_index).astype(float)
    available_backends = [b for b in BACKEND_ORDER if b in score_tbl.columns and not score_tbl[b].dropna().empty]
    if not available_backends:
        return None, None
    if decision_backend not in available_backends:
        decision_backend = available_backends[0]

    t0 = int(ts_index.min())
    x = ts_index.astype(np.int64) - t0

    line_styles = {
        "py_lgbm": "-",
        "cpp_lgbm": "--",
        "cpp_lr": "-",
    }
    line_alphas = {
        "py_lgbm": 0.75,
        "cpp_lgbm": 1.0,
        "cpp_lr": 0.95,
    }

    decision_scores = score_tbl[decision_backend].to_numpy(dtype=float)
    alarm_decision = _postproc_hysteresis(decision_scores, on=POSTPROC_ON, off=POSTPROC_OFF).astype(float)
    raw_crossings_idx = _upward_threshold_crossings(decision_scores, threshold)
    alarm_onset_idx = _alarm_onsets(alarm_decision)
    duty_cycle = float(np.mean(alarm_decision)) if len(alarm_decision) > 0 else float("nan")

    parity_source = "source=predictions.py_lgbm vs predictions.cpp_lgbm"
    parity_note = "py vs cpp parity: unavailable (py_lgbm/cpp_lgbm scores unavailable)"
    show_diff_subplot = False
    diff_x = np.array([], dtype=np.int64)
    diff = np.array([], dtype=np.float64)
    max_abs_diff = float("nan")
    if {"py_lgbm", "cpp_lgbm"}.issubset(score_tbl.columns):
        parity = score_tbl[["py_lgbm", "cpp_lgbm"]].dropna()
        if not parity.empty:
            diff_x = parity.index.to_numpy(dtype=np.int64) - t0
            diff = (parity["py_lgbm"] - parity["cpp_lgbm"]).to_numpy(dtype=float)
            max_abs_diff = float(np.max(np.abs(diff)))
            parity_note = f"py vs cpp parity: max|diff|={max_abs_diff:.3e} ({parity_source})"
            show_diff_subplot = bool(max_abs_diff > parity_eps)
        else:
            parity_note = "py vs cpp parity: unavailable (no overlapping py/cpp rows)"

    if show_diff_subplot:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(9.2, 6.2),
            gridspec_kw={"height_ratios": [3, 1.2, 1.2]},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(9.2, 5.0),
            gridspec_kw={"height_ratios": [3, 1.2]},
        )

    y_true = label_ts.fillna(0.0).to_numpy(dtype=float)
    gt_segments = _positive_segments(x, (y_true >= 0.5).astype(np.int32))
    for seg_start, seg_end in gt_segments:
        ax1.axvspan(
            seg_start,
            seg_end,
            color="#ef4444",
            alpha=0.05,
            label="_nolegend_",
            zorder=0,
        )

    for backend in BACKEND_ORDER:
        if backend not in score_tbl.columns:
            continue
        line = score_tbl[backend].dropna()
        if line.empty:
            continue
        x_backend = line.index.to_numpy(dtype=np.int64) - t0
        y_backend = line.to_numpy(dtype=float)
        ax1.plot(
            x_backend,
            y_backend,
            linewidth=1.6,
            label=BACKEND_LABELS.get(backend, backend),
            color=BACKEND_COLORS.get(backend, "#6b7280"),
            linestyle=line_styles.get(backend, "-"),
            alpha=line_alphas.get(backend, 0.9),
            zorder=3,
        )

    if len(alarm_onset_idx) > 0:
        onset_scores = decision_scores[alarm_onset_idx]
        finite = np.isfinite(onset_scores)
        if finite.any():
            ax1.scatter(
                x[alarm_onset_idx[finite]],
                onset_scores[finite],
                marker="^",
                s=38,
                color=BACKEND_COLORS.get(decision_backend, "#334155"),
                edgecolors="#0f172a",
                linewidths=0.4,
                label="Alarm onset (postproc)",
                zorder=5,
            )

    threshold_label = f"threshold={threshold:.2f}"
    ax1.axhline(threshold, color="#64748b", linestyle=":", linewidth=1.2, label=threshold_label, zorder=2)
    ax1.set_ylabel("P(anomaly)")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f"Per-flight score timeline (flight_id={flight_id}, threshold={threshold:.2f})")
    ax1.grid(alpha=0.25)

    decision_label = BACKEND_LABELS.get(decision_backend, decision_backend)
    ax2.step(x, y_true, where="post", color="#0f172a", linewidth=1.4, label="_nolegend_")
    ax2.fill_between(x, 0.0, y_true, step="post", color="#0f172a", alpha=0.18)
    decision_legend_label = f"Alarm decision (hysteresis, {decision_backend})"
    ax2.step(
        x,
        alarm_decision,
        where="post",
        color=BACKEND_COLORS.get(decision_backend, "#334155"),
        linewidth=1.2,
        linestyle="--",
        alpha=0.95,
        label=decision_legend_label,
    )
    ax2.set_ylabel("GT/Alarm")
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(alpha=0.2)

    if show_diff_subplot:
        ax3.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.0)
        ax3.plot(diff_x, diff, color="#475569", linewidth=1.2, label="py_lgbm - cpp_lgbm")
        y_pad = max(1e-3, max_abs_diff * 1.2)
        ax3.set_ylim(-y_pad, y_pad)
        ax3.set_ylabel("py-cpp")
        ax3.text(
            0.01,
            0.85,
            parity_note,
            transform=ax3.transAxes,
            fontsize=8,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#cbd5e1"},
        )
        ax3.set_xlabel("Time since start [s]")
        ax3.grid(alpha=0.2)
    else:
        ax2.set_xlabel("Time since start [s]")

    handles: list[object] = []
    labels: list[str] = []
    for axis in [ax1, ax2]:
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    label_to_handle: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label in {"", "_nolegend_"}:
            continue
        if label not in label_to_handle:
            label_to_handle[label] = handle

    wanted_labels = [BACKEND_LABELS[b] for b in BACKEND_ORDER if b in score_tbl.columns]
    wanted_labels.extend([threshold_label, "Alarm onset (postproc)", decision_legend_label])

    picked_handles: list[object] = []
    picked_labels: list[str] = []
    for label in wanted_labels:
        handle = label_to_handle.get(label)
        if handle is None:
            continue
        picked_handles.append(handle)
        picked_labels.append(label)

    if picked_handles:
        fig.legend(
            picked_handles,
            picked_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            frameon=True,
            fontsize=8,
            ncol=3,
        )
    fig.subplots_adjust(bottom=0.17)
    stats = {
        "decision_backend": decision_backend,
        "decision_label": decision_label,
        "raw_crossings": int(len(raw_crossings_idx)),
        "alarm_onsets": int(len(alarm_onset_idx)),
        "duty_cycle": duty_cycle,
        "parity_note": parity_note,
        "parity_diff_shown": bool(show_diff_subplot),
    }
    return fig, stats


def _render_header() -> None:
    st.title("Inference Speed and Quality Dashboard")
    st.caption("3モデル比較（Python LightGBM / C++ LightGBM / C++ Logistic Regression）")


def _render_kpi_cards(quality: pd.DataFrame, bench_latest: pd.DataFrame, parity_rate: float) -> None:
    series_quality = quality if not quality.empty else pd.DataFrame()

    entries: list[tuple[str, pd.Series, bool, str]] = [
        (
            "Recall（見逃しの少なさ）",
            series_quality["recall"] if "recall" in series_quality.columns else pd.Series(dtype=float),
            True,
            "percent",
        ),
        (
            "Precision（誤検知の少なさ）",
            series_quality["precision"] if "precision" in series_quality.columns else pd.Series(dtype=float),
            True,
            "percent",
        ),
        ("F1", series_quality["f1"] if "f1" in series_quality.columns else pd.Series(dtype=float), True, "percent"),
        (
            "最速 p50",
            bench_latest["p50_ms"] if "p50_ms" in bench_latest.columns else pd.Series(dtype=float),
            False,
            "ms",
        ),
        (
            "最高 Throughput",
            bench_latest["throughput_sps"] if "throughput_sps" in bench_latest.columns else pd.Series(dtype=float),
            True,
            "throughput",
        ),
        ("AUC", series_quality["auc"] if "auc" in series_quality.columns else pd.Series(dtype=float), True, "ratio"),
    ]

    cards: list[tuple[str, str, str]] = []
    for title, series, maximize, style in entries:
        winners, value = _winner(series, maximize=maximize)
        cards.append((title, _format_value(value, style), _winner_text(winners)))

    cards.append(("LGBM parity（一致率）", _format_value(parity_rate, "percent"), "py_lgbm vs cpp_lgbm"))

    n_cols = 3
    for row_start in range(0, len(cards), n_cols):
        row_cols = st.columns(n_cols)
        row_cards = cards[row_start : row_start + n_cols]
        for col, (title, value, subtext) in zip(row_cols, row_cards):
            col.metric(title, value)
            col.caption(subtext)


def _render_dashboard_body(
    preds: pd.DataFrame,
    quality: pd.DataFrame,
    bench_latest: pd.DataFrame,
    model_dir: Path,
) -> None:
    st.subheader("メイン比較テーブル")
    table = _comparison_table(quality, bench_latest)
    if table.empty:
        st.info("比較テーブルを作るためのデータがありません。")
    else:
        st.dataframe(
            table.style.format(
                {
                    "Recall (%)": "{:.2f}%",
                    "Precision (%)": "{:.2f}%",
                    "F1 (%)": "{:.2f}%",
                    "Accuracy (%)": "{:.2f}%",
                    "AUC (0-1)": "{:.3f}",
                    "p50 (ms)": "{:.3f}",
                    "p95 (ms)": "{:.3f}",
                    "Throughput (samples/s)": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("速度比較ブロック")
    s1, s2, s3 = st.columns(3)
    figs = [
        _plot_speed_metric(bench_latest, "p50_ms", "p50 latency (ms)"),
        _plot_speed_metric(bench_latest, "p95_ms", "p95 latency (ms)"),
        _plot_speed_metric(bench_latest, "throughput_sps", "Throughput (log scale)", log_scale=True),
    ]
    for c, fig in zip([s1, s2, s3], figs):
        if fig is not None:
            c.pyplot(fig)
        else:
            c.info("No benchmark data")

    st.subheader("品質比較ブロック")
    q1, q2 = st.columns(2)
    quality_fig = _plot_quality_bars(quality)
    roc_fig = _plot_roc(preds)
    if quality_fig is not None:
        q1.pyplot(quality_fig)
    else:
        q1.info("No quality data")
    if roc_fig is not None:
        q2.pyplot(roc_fig)
    else:
        q2.info("No ROC data")

    st.markdown("**混同行列（x3）**")
    cm_cols = st.columns(len(BACKEND_ORDER))
    for idx, backend in enumerate(BACKEND_ORDER):
        fig = _plot_confusion(preds, backend)
        if fig is not None:
            cm_cols[idx].pyplot(fig)
        else:
            cm_cols[idx].info(f"No eval predictions for {backend}")

    with st.expander("補助分析", expanded=False):
        st.markdown("- Per-flight timeline")
        eval_flights = sorted(preds.loc[preds["split"] == "eval", "flight_id"].dropna().unique())
        if eval_flights:
            selected_flight = st.selectbox("flight_id", eval_flights, key="supp_flight")
            flight_preds = preds[(preds["split"] == "eval") & (preds["flight_id"] == selected_flight)]
            backend_set = {str(b) for b in flight_preds["backend"].dropna().unique()}
            decision_options = [b for b in BACKEND_ORDER if b in backend_set]
            if decision_options:
                default_backend = next((b for b in BACKEND_ORDER if b in decision_options), decision_options[0])
                default_idx = decision_options.index(default_backend)
                decision_backend = st.selectbox(
                    "Decision source model",
                    decision_options,
                    index=default_idx,
                    key=f"supp_decision_backend_{selected_flight}",
                    format_func=lambda b: BACKEND_LABELS.get(b, b),
                )
                timeline, timeline_stats = _plot_flight_timeline(
                    preds,
                    selected_flight,
                    decision_backend=decision_backend,
                    threshold=0.5,
                )
            else:
                timeline, timeline_stats = None, None

            if timeline is not None:
                st.pyplot(timeline)
                if timeline_stats is not None:
                    duty_cycle = float(timeline_stats.get("duty_cycle", float("nan")))
                    duty_text = f"{duty_cycle:.1%}" if np.isfinite(duty_cycle) else "-"
                    st.caption(
                        " | ".join(
                            [
                                (
                                    "Decision model: "
                                    f"{timeline_stats.get('decision_backend', decision_backend)}"
                                ),
                                f"Hysteresis on={POSTPROC_ON:.2f} off={POSTPROC_OFF:.2f}",
                                f"Raw crossings={int(timeline_stats.get('raw_crossings', 0))}",
                                f"Alarm onsets={int(timeline_stats.get('alarm_onsets', 0))}",
                                f"Duty cycle={duty_text}",
                            ]
                        )
                    )
                    parity_note = str(timeline_stats.get("parity_note", "")).strip()
                    if parity_note:
                        st.caption(parity_note)
                    if not bool(timeline_stats.get("parity_diff_shown", False)):
                        st.caption(f"Diff panel hidden (eps={PARITY_EPS:.0e})")
            elif decision_options:
                st.info("No timeline data")
            else:
                st.info("No model data for selected flight")
        else:
            st.info("No eval flight_id data")

        st.markdown("- Feature importance")
        importance = _feature_importance_table(model_dir)
        if not importance.empty:
            st.dataframe(importance.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("feature_importance.csv がありません。trainer を実行してください。")

    st.caption(
        "この比較は同一評価データ・同一条件で計測しています。"
        " p50 / p95 / Throughput は predict_fn 実行時間ベースで、DB read/write などの I/O は含みません。"
    )


def main() -> None:
    cfg = load_config()
    db_path = _sidebar_db_selector(str(cfg.db_path))
    db_file = Path(db_path)
    samples = pd.DataFrame(columns=["split"])
    preds = pd.DataFrame(
        columns=[
            "sample_id",
            "backend",
            "model_name",
            "score",
            "pred",
            "flight_id",
            "ts",
            "label",
            "split",
            "source",
        ]
    )
    bench = pd.DataFrame()

    _render_header()
    if not db_file.exists():
        bench_latest = _latest_benchmark(bench)
        _render_sidebar_common_info(samples, bench_latest, db_path=db_path, read_error="DB not found")
        st.warning(f"DB not found: {db_path}")
        st.info("Run trainer/inference pipeline first.")
        st.stop()

    schema_error = _validate_db_schema(db_path)
    if schema_error is not None:
        bench_latest = _latest_benchmark(bench)
        _render_sidebar_common_info(samples, bench_latest, db_path=db_path, read_error=schema_error)
        st.warning(schema_error)
        st.stop()

    try:
        samples, preds, bench = _read_tables(db_path)
    except Exception as exc:
        read_error = str(exc)
        bench_latest = _latest_benchmark(bench)
        _render_sidebar_common_info(samples, bench_latest, db_path=db_path, read_error=read_error)
        st.warning(f"DB read failed: {read_error}")
        st.stop()

    bench_latest = _latest_benchmark(bench)
    quality = _compute_quality_metrics(preds)
    parity_rate = _compute_lgbm_parity(preds)
    _render_sidebar_common_info(samples, bench_latest, db_path=db_path, read_error=None)

    _render_kpi_cards(quality, bench_latest, parity_rate)
    _render_dashboard_body(
        preds=preds,
        quality=quality,
        bench_latest=bench_latest,
        model_dir=Path(cfg.model_dir),
    )


if __name__ == "__main__":
    main()
