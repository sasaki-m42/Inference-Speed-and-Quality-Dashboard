from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from inference_bench.paths import REPO_ROOT


def _artifact_lib_candidates() -> list[Path]:
    lib_dir = REPO_ROOT / "artifacts" / "lib"
    return [
        lib_dir / "lgbm_predictor.so",
        lib_dir / "liblgbm_predictor.so",
        lib_dir / "lgbm_predictor.dylib",
        lib_dir / "liblgbm_predictor.dylib",
        lib_dir / "lgbm_predictor.dll",
    ]


def resolve_library_path() -> Path:
    for candidate in _artifact_lib_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "C++ LightGBM predictor library not found under artifacts/lib. "
        "Run `python -m inference_bench.inference.cpp_lgbm --build` first."
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
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "lgbm_predictor"]

    subprocess.run(configure_cmd, check=True)
    subprocess.run(build_cmd, check=True)
    return resolve_library_path()


def find_lightgbm_native_library() -> str:
    pkg_dir = Path(lgb.__file__).resolve().parent
    candidates = [
        pkg_dir / "lib" / "lib_lightgbm.so",
        pkg_dir / "lib" / "lib_lightgbm.dylib",
        pkg_dir / "lib" / "lib_lightgbm.dll",
        pkg_dir / "lib_lightgbm.so",
        pkg_dir / "lib_lightgbm.dylib",
        pkg_dir / "lib_lightgbm.dll",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    if sys.platform.startswith("linux"):
        return "lib_lightgbm.so"
    if sys.platform == "darwin":
        return "lib_lightgbm.dylib"
    return "lib_lightgbm.dll"


class CppLightGBMPredictor:
    def __init__(
        self,
        model_path: str | Path,
        n_threads: int = 1,
        lib_path: Path | None = None,
        lgbm_lib_path: str | None = None,
    ):
        if lib_path is None:
            lib_path = ensure_compiled()
        self.lib_path = lib_path
        self.model_path = str(model_path)
        self.n_threads = max(1, int(n_threads))
        self.lgbm_lib_path = lgbm_lib_path or find_lightgbm_native_library()

        self.lib = ctypes.CDLL(str(self.lib_path))

        self.lib.lgbm_predictor_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.lgbm_predictor_init.restype = ctypes.c_int

        self.lib.lgbm_predictor_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.lgbm_predictor_predict.restype = ctypes.c_int

        self.lib.lgbm_predictor_free.argtypes = [ctypes.c_void_p]
        self.lib.lgbm_predictor_free.restype = None

        self.lib.lgbm_predictor_last_error.argtypes = []
        self.lib.lgbm_predictor_last_error.restype = ctypes.c_char_p

        self._handle = ctypes.c_void_p()
        rc = self.lib.lgbm_predictor_init(
            self.lgbm_lib_path.encode("utf-8"),
            self.model_path.encode("utf-8"),
            ctypes.c_int(self.n_threads),
            ctypes.byref(self._handle),
        )
        if rc != 0:
            raise RuntimeError(f"Failed to initialize C++ LightGBM predictor: {self._last_error()}")

    def _last_error(self) -> str:
        raw = self.lib.lgbm_predictor_last_error()
        if not raw:
            return "unknown error"
        return raw.decode("utf-8", errors="replace")

    def predict_batch(self, x: np.ndarray, predict_type: int = 0) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Input must be 2D array")

        n_samples, n_features = x.shape
        out = np.zeros(n_samples, dtype=np.float64)

        rc = self.lib.lgbm_predictor_predict(
            self._handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features),
            ctypes.c_int(predict_type),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"C++ LightGBM prediction failed: {self._last_error()}")

        return out.astype(np.float32)

    def close(self) -> None:
        if getattr(self, "_handle", None) and self._handle.value:
            self.lib.lgbm_predictor_free(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build C++ LightGBM predictor shared library")
    parser.add_argument("--build", action="store_true", help="Configure and build the C++ library")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        path = ensure_compiled()
        print(f"Built C++ LightGBM predictor library: {path}")


if __name__ == "__main__":
    main()
