from __future__ import annotations

import argparse
import subprocess
import sys

from inference_bench.ops.run_logger import LOG_DIR, log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test suite and persist execution logs")
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Arguments passed to pytest. If omitted, defaults to -q.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    test_log_path = LOG_DIR / "latest_test_output.log"

    pytest_args = args.pytest_args if args.pytest_args else ["-q"]
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    test_log_path.write_text(proc.stdout, encoding="utf-8")

    log_event(
        category="validation",
        event="pytest",
        status="success" if proc.returncode == 0 else "failure",
        details={
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "output_log": str(test_log_path),
        },
    )

    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
