from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from inference_bench.config import load_config
from inference_bench.ops.run_logger import log_event
from inference_bench.paths import REPORTS_DIR

matplotlib.use("Agg", force=True)

BACKEND_ORDER = ["py_lgbm", "cpp_lgbm", "cpp_lr"]
BACKEND_LABELS = {
    "py_lgbm": "Python LightGBM",
    "cpp_lgbm": "C++ LightGBM",
    "cpp_lr": "C++ Logistic Regression",
}
JST = ZoneInfo("Asia/Tokyo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static HTML report from SQLite benchmark data.")
    parser.add_argument("--db", default=None, help="SQLite path")
    parser.add_argument("--out", default=None, help="Output directory (default: reports/site)")
    return parser.parse_args()


def _load_data(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    samples = pd.read_sql_query("SELECT id, split, label FROM samples", con)
    preds = pd.read_sql_query(
        """
        SELECT p.sample_id, p.backend, p.score, p.pred,
               s.label, s.flight_id, s.ts, s.split
        FROM predictions p
        JOIN samples s ON s.id = p.sample_id
        """,
        con,
    )
    bench = pd.read_sql_query(
        """
        SELECT backend, model_name, p50_ms, p95_ms, throughput_sps,
               batch_size, num_threads, created_at,
               notes
        FROM benchmark
        ORDER BY created_at DESC
        """,
        con,
    )
    con.close()
    if not bench.empty:
        created = pd.to_datetime(bench["created_at"], unit="s", utc=True, errors="coerce")
        bench["created_at_text"] = created.dt.tz_convert(JST).dt.strftime("%Y-%m-%d %H:%M JST")
    else:
        bench["created_at_text"] = pd.Series(dtype=object)
    return samples, preds, bench


def _plot_benchmark(bench: pd.DataFrame, out_file: Path) -> None:
    latest = bench.sort_values("created_at").groupby("backend", as_index=False).tail(1)
    latest["backend"] = pd.Categorical(latest["backend"], categories=BACKEND_ORDER, ordered=True)
    latest = latest.sort_values("backend")
    short_labels = latest["backend"].astype(str).tolist()
    positions = list(range(len(latest)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    colors = ["#3a86ff", "#ff6f61", "#2ca02c", "#d62728"]
    colors = colors[: len(latest)]
    axes[0].bar(positions, latest["p50_ms"], color=colors)
    axes[0].set_title("p50 latency (ms)")
    axes[1].bar(positions, latest["p95_ms"], color=colors)
    axes[1].set_title("p95 latency (ms)")
    axes[2].bar(positions, latest["throughput_sps"], color=colors)
    axes[2].set_title("Throughput (samples/s)")
    for ax in axes:
        ax.set_xticks(positions, short_labels)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def _build_score_table(preds: pd.DataFrame) -> pd.DataFrame:
    eval_df = preds[preds["split"] == "eval"].copy()
    rows = []
    for backend in [b for b in BACKEND_ORDER if b in set(eval_df["backend"].unique())]:
        d = eval_df[eval_df["backend"] == backend]
        y_true = d["label"].astype(int)
        y_pred = d["pred"].astype(int)
        y_score = d["score"].astype(float)
        if d["label"].nunique() < 2:
            auc_value = float("nan")
        else:
            auc_value = float(roc_auc_score(y_true, y_score))
        rows.append(
            {
                "backend": backend,
                "implementation": BACKEND_LABELS.get(backend, backend),
                "accuracy": float((y_pred == y_true).mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc": auc_value,
            }
        )
    return pd.DataFrame(rows)


def _latest_benchmark_table(bench: pd.DataFrame) -> pd.DataFrame:
    if bench.empty:
        return pd.DataFrame()

    latest = bench.sort_values("created_at").groupby("backend", as_index=False).tail(1).copy()
    latest["backend"] = pd.Categorical(latest["backend"], categories=BACKEND_ORDER, ordered=True)
    latest = latest.sort_values("backend")
    latest["implementation"] = latest["backend"].map(BACKEND_LABELS).fillna(latest["backend"].astype(str))
    latest = latest[
        [
            "implementation",
            "p50_ms",
            "p95_ms",
            "throughput_sps",
            "batch_size",
            "num_threads",
            "created_at_text",
        ]
    ].rename(
        columns={
            "implementation": "Model",
            "p50_ms": "p50 (ms)",
            "p95_ms": "p95 (ms)",
            "throughput_sps": "Throughput (samples/s)",
            "batch_size": "batch_size",
            "num_threads": "threads",
            "created_at_text": "measured_at",
        }
    )

    latest["p50 (ms)"] = latest["p50 (ms)"].map(lambda v: f"{float(v):.3f}")
    latest["p95 (ms)"] = latest["p95 (ms)"].map(lambda v: f"{float(v):.3f}")
    latest["Throughput (samples/s)"] = latest["Throughput (samples/s)"].map(lambda v: f"{float(v):,.0f}")
    return latest


def _generate_html(
    samples: pd.DataFrame,
    preds: pd.DataFrame,
    bench: pd.DataFrame,
    score_table: pd.DataFrame,
    benchmark_image_name: str,
) -> str:
    now = datetime.now(timezone.utc).astimezone(JST).strftime("%Y-%m-%d %H:%M JST")
    sample_count = len(samples)
    eval_count = int((samples["split"] == "eval").sum()) if not samples.empty else 0
    model_count = int(score_table["backend"].nunique()) if not score_table.empty else 0

    bench_display = _latest_benchmark_table(bench)
    if not bench_display.empty:
        bench_html = bench_display.to_html(index=False, classes="table")
    else:
        bench_html = "<p>No benchmark rows.</p>"

    if not score_table.empty:
        score_display = score_table.copy().rename(
            columns={
                "implementation": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "auc": "AUC",
            }
        )
        score_display["Accuracy"] = score_display["Accuracy"].map(lambda v: f"{float(v) * 100.0:.2f}%")
        score_display["Precision"] = score_display["Precision"].map(lambda v: f"{float(v) * 100.0:.2f}%")
        score_display["Recall"] = score_display["Recall"].map(lambda v: f"{float(v) * 100.0:.2f}%")
        score_display["F1"] = score_display["F1"].map(lambda v: f"{float(v) * 100.0:.2f}%")
        score_display["AUC"] = score_display["AUC"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.3f}")
        score_display = score_display[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
        score_html = score_display.to_html(index=False, classes="table")
    else:
        score_html = "<p>No prediction rows.</p>"

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Inference Benchmark Report</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1b1f2a;
      --muted: #576076;
      --line: #d7deea;
    }}
    body {{ margin: 0; background: linear-gradient(180deg, #eef3ff 0%, var(--bg) 30%); color: var(--ink); font-family: "Segoe UI", "Hiragino Sans", sans-serif; }}
    .wrap {{ max-width: 1060px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 24px; box-shadow: 0 8px 28px rgba(25, 35, 58, 0.06); }}
    h1 {{ margin: 0; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-top: 8px; }}
    .grid {{ margin-top: 18px; display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; padding: 14px; background: #fbfdff; }}
    .card .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card .v {{ margin-top: 6px; font-size: 22px; font-weight: 700; }}
    section {{ margin-top: 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 20px; }}
    h2 {{ margin-top: 0; font-size: 20px; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .table th, .table td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 8px; }}
    .table th {{ background: #f1f6ff; }}
    .note {{ color: var(--muted); font-size: 13px; }}
    img {{ max-width: 100%; border: 1px solid var(--line); border-radius: 12px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1>Inference Speed and Quality Benchmark</h1>
      <p class=\"sub\">Python LightGBM / C++ LightGBM / C++ Logistic Regression | generated at {now}</p>
      <div class=\"grid\">
        <div class=\"card\"><div class=\"k\">Total Samples</div><div class=\"v\">{sample_count:,}</div></div>
        <div class=\"card\"><div class=\"k\">Eval Samples</div><div class=\"v\">{eval_count:,}</div></div>
        <div class=\"card\"><div class=\"k\">Compared Models</div><div class=\"v\">{model_count}</div></div>
      </div>
    </div>

    <section>
      <h2>Benchmark Snapshot</h2>
      <img src=\"assets/{benchmark_image_name}\" alt=\"benchmark comparison\" />
      <p class=\"note\">この比較は同一評価データ・同一条件で計測しています。p50 / p95 / Throughput は predict_fn 実行時間ベースで、DB read/write などの I/O は含みません。</p>
      {bench_html}
    </section>

    <section>
      <h2>Classification Metrics</h2>
      <p class=\"note\">品質指標（Accuracy / Precision / Recall / F1 / AUC）は split=eval のみで算出しています。</p>
      {score_html}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    cfg = load_config()

    db_path = Path(args.db or cfg.db_path)
    out_dir = Path(args.out) if args.out else REPORTS_DIR / "site"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    samples, preds, bench = _load_data(db_path)
    score_table = _build_score_table(preds)

    benchmark_img = "benchmark.png"
    benchmark_img_path = assets_dir / benchmark_img
    if not bench.empty:
        _plot_benchmark(bench, benchmark_img_path)

    html = _generate_html(
        samples=samples,
        preds=preds,
        bench=bench,
        score_table=score_table,
        benchmark_image_name=benchmark_img,
    )

    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    log_event(
        category="report",
        event="export_static",
        status="success",
        details={
            "db_path": str(db_path),
            "output": str(out_path),
            "sample_rows": len(samples),
            "prediction_rows": len(preds),
            "benchmark_rows": len(bench),
        },
    )
    print(f"[export_static] Wrote report: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_event("report", "export_static", "failure", {"error": str(exc)})
        raise
