from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

from inference_bench.paths import REPO_ROOT


def _artifact_lib_candidates() -> list[Path]:
    lib_dir = REPO_ROOT / "artifacts" / "lib"
    return [
        lib_dir / "lr.so",
        lib_dir / "liblr.so",
        lib_dir / "lr.dylib",
        lib_dir / "liblr.dylib",
        lib_dir / "lr.dll",
    ]


def resolve_library_path() -> Path:
    for candidate in _artifact_lib_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "C++ logistic regression library not found under artifacts/lib. "
        "Run `python -m inference_bench.inference.cpp_lr --build` first."
    )


def ensure_compiled() -> Path:
    try:
        return resolve_library_path()
    except FileNotFoundError:
        pass

    build_dir = REPO_ROOT / "build" / "cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_dir = REPO_ROOT / "cpp"

    configure_cmd = ["cmake", "-S", str(cpp_dir), "-B", str(build_dir)]
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "lr"]

    subprocess.run(configure_cmd, check=True)
    subprocess.run(build_cmd, check=True)
    return resolve_library_path()


class CppLogisticRegression:
    def __init__(self, lib_path: Path | None = None):
        if lib_path is None:
            lib_path = ensure_compiled()
        self.lib_path = lib_path

        if sys.platform.startswith("win"):
            os_lib_dir = str(lib_path.parent)
            if hasattr(ctypes, "windll"):
                try:
                    ctypes.windll.kernel32.SetDllDirectoryW(os_lib_dir)
                except Exception:
                    pass

        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.train_lr.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.train_lr.restype = None

        self.lib.predict_lr_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.predict_lr_batch.restype = None

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 250,
        lr: float = 0.1,
        l2: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)

        n_samples, n_features = x.shape
        out_w = np.zeros(n_features, dtype=np.float32)
        out_b = np.zeros(1, dtype=np.float32)

        self.lib.train_lr(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            ctypes.c_int(epochs),
            ctypes.c_float(lr),
            ctypes.c_float(l2),
            out_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out_w, float(out_b[0])

    def predict_batch(self, weights: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)

        n_samples, n_features = x.shape
        if len(weights) != n_features:
            raise ValueError(f"weights dim mismatch: got {len(weights)}, expected {n_features}")

        out = np.zeros(n_samples, dtype=np.float32)
        self.lib.predict_lr_batch(
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(bias),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build C++ logistic regression shared library")
    parser.add_argument("--build", action="store_true", help="Configure and build the C++ library")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        path = ensure_compiled()
        print(f"Built C++ logistic regression library: {path}")


if __name__ == "__main__":
    main()
