from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

from inference_bench.ops.run_logger import log_event


@dataclass
class Step:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full reproducible benchmark pipeline: build -> train -> infer -> export."
    )
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

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=20)

    parser.add_argument("--skip-build", action="store_true", help="Skip C++ shared library build step")
    parser.add_argument("--skip-export", action="store_true", help="Skip static report export step")
    return parser.parse_args()


def _run_step(step: Step) -> None:
    printable = " ".join(step.command)
    print(f"[run_all] {step.name}: {printable}")
    subprocess.run(step.command, check=True)


def _build_steps(args: argparse.Namespace) -> list[Step]:
    steps: list[Step] = []

    if not args.skip_build:
        steps.extend(
            [
                Step(
                    name="build_cpp_lgbm",
                    command=[sys.executable, "-m", "inference_bench.inference.cpp_lgbm", "--build"],
                ),
                Step(
                    name="build_cpp_lr",
                    command=[sys.executable, "-m", "inference_bench.inference.cpp_lr", "--build"],
                ),
            ]
        )

    trainer_cmd = [
        sys.executable,
        "-m",
        "inference_bench.pipeline.run_trainer",
        "--dataset-source",
        args.dataset_source,
        "--max-csv-rows",
        str(args.max_csv_rows),
        "--task-mode",
        args.task_mode,
        "--score-threshold",
        str(args.score_threshold),
        "--n-samples",
        str(args.n_samples),
        "--n-features",
        str(args.n_features),
        "--seed",
        str(args.seed),
        "--mode",
        args.mode,
        "--threads",
        str(args.threads),
    ]
    if args.db:
        trainer_cmd.extend(["--db", args.db])
    if args.csv_path:
        trainer_cmd.extend(["--csv-path", args.csv_path])

    infer_base = [
        "--score-threshold",
        str(args.score_threshold),
        "--batch-size",
        str(args.batch_size),
        "--warmup-iters",
        str(args.warmup_iters),
        "--measure-iters",
        str(args.measure_iters),
        "--threads",
        str(args.threads),
    ]

    infer_py_cmd = [sys.executable, "-m", "inference_bench.pipeline.run_infer_py", *infer_base]
    infer_cpp_lgbm_cmd = [sys.executable, "-m", "inference_bench.pipeline.run_infer_cpp", *infer_base]
    infer_cpp_lr_cmd = [sys.executable, "-m", "inference_bench.pipeline.run_infer_cpp_lr", *infer_base]
    export_cmd = [sys.executable, "-m", "inference_bench.viz.export_static"]

    if args.db:
        infer_py_cmd.extend(["--db", args.db])
        infer_cpp_lgbm_cmd.extend(["--db", args.db])
        infer_cpp_lr_cmd.extend(["--db", args.db])
        export_cmd.extend(["--db", args.db])

    steps.extend(
        [
            Step(name="trainer", command=trainer_cmd),
            Step(name="infer_py_lgbm", command=infer_py_cmd),
            Step(name="infer_cpp_lgbm", command=infer_cpp_lgbm_cmd),
            Step(name="infer_cpp_lr", command=infer_cpp_lr_cmd),
        ]
    )

    if not args.skip_export:
        steps.append(Step(name="export_static", command=export_cmd))

    return steps


def main() -> None:
    args = parse_args()
    steps = _build_steps(args)
    step_names = [step.name for step in steps]

    log_event("pipeline", "run_all", "started", {"steps": step_names})
    for step in steps:
        _run_step(step)
    log_event("pipeline", "run_all", "success", {"steps": step_names})
    print("[run_all] pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_event("pipeline", "run_all", "failure", {"error": str(exc)})
        raise
