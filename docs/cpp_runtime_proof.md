# C++ Runtime Proof (Docker Baseline)

目的:

- 「`cpp_lgbm` が本当に C++ LightGBM 推論を実行している」ことを、Docker 実行前提で追跡できる形で示す。
- 検証の基準は **Docker コンテナ内（Linux コンテナ）** とし、ホストOS（Windows/macOS/Linux）の差分には依存しない。
  - ※ Windows/macOS でも Docker Desktop が Linux コンテナを実行できれば同じ手順で再現可能。

実行日:

- 2026-02-11

## 1. 入口（どこから C++ 推論が呼ばれるか）

### 1.1 CLI 入口（C++系）

- 実行: `python -m inference_bench.pipeline.run_infer_cpp`
- 実装: `src/inference_bench/pipeline/run_infer_cpp.py`
  - `from inference_bench.inference.cpp_lgbm import CppLightGBMPredictor`（line 10）
  - `predictor = CppLightGBMPredictor(...)`（line 56）
  - backend は `cpp_lgbm`（line 15）

### 1.2 Python ラッパー -> C++ 共有ライブラリ

- 実装: `src/inference_bench/inference/cpp_lgbm.py`
  - `self.lib = ctypes.CDLL(str(self.lib_path))`（line 90）
  - `lgbm_predictor_init` / `lgbm_predictor_predict` を `ctypes` で呼ぶ（line 92-149）
  - `ensure_compiled()` で `artifacts/lib` 配下を解決（line 36, 84）

### 1.3 C++ 実体（LightGBM C API 呼び出し）

- 実装: `cpp/lgbm_predictor.cpp`
  - `LGBM_BoosterCreateFromModelfile`
  - `LGBM_BoosterPredictForMat`
  - `LGBM_BoosterFree`
  - `LGBM_GetLastError`
- `dlopen` / `dlsym` で LightGBM ネイティブライブラリのシンボルを解決して推論する。

### 1.4 対照（Python LightGBM 入口）

- 実装: `src/inference_bench/pipeline/run_infer_py.py`
  - `import lightgbm as lgb`（line 5）
  - `booster = lgb.Booster(model_file=...)`（line 62）
- `run_infer_py` と `run_infer_cpp` は入口が分離されている。

## 2. Docker 基準のビルド/実行手順

### 2.1 ビルド

```bash
docker compose run --rm infer_cpp bash -lc "python -m inference_bench.inference.cpp_lgbm --build"
```

実行結果:

```text
Built C++ LightGBM predictor library: /app/artifacts/lib/liblgbm_predictor.so
```

### 2.2 推論実行（C++経路）

```bash
docker compose run --rm infer_cpp
```

- `docker-compose.yml` で `infer_cpp` サービスのコマンドは
  `python -m inference_bench.pipeline.run_infer_cpp --threads 1` に固定。

## 3. 成果物（どこに出るか）

- `cpp/CMakeLists.txt` で出力先は `../artifacts/lib`
- `cpp_lgbm.py` の探索候補は以下:
  - `artifacts/lib/lgbm_predictor.so`
  - `artifacts/lib/liblgbm_predictor.so`
  - `artifacts/lib/lgbm_predictor.dylib`
  - `artifacts/lib/liblgbm_predictor.dylib`
  - `artifacts/lib/lgbm_predictor.dll`

## 4. Docker 内での確認コマンドと実行結果

### 4.1 共有ライブラリの存在

```bash
docker compose run --rm infer_cpp bash -lc "ls -l artifacts/lib"
```

```text
total 144
-rwxr-xr-x 1 root root 71712 Feb  6 23:52 liblgbm_predictor.so
-rwxr-xr-x 1 root root 70040 Feb  7 00:13 liblr.so
```

### 4.2 共有ライブラリ形式（ELF）と依存

```bash
docker compose run --rm infer_cpp bash -lc "readelf -h artifacts/lib/liblgbm_predictor.so | sed -n '1,18p'"
```

```text
ELF Header:
  Class:                             ELF64
  Type:                              DYN (Shared object file)
  Machine:                           AArch64
```

```bash
docker compose run --rm infer_cpp bash -lc "ldd artifacts/lib/liblgbm_predictor.so"
```

```text
libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6
libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1
libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6
libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6
```

```bash
docker compose run --rm infer_cpp bash -lc "readelf -d artifacts/lib/liblgbm_predictor.so | grep NEEDED"
```

```text
Shared library: [libstdc++.so.6]
Shared library: [libgcc_s.so.1]
Shared library: [libc.so.6]
```

### 4.3 LightGBM C API を参照しているコード確認

```bash
docker compose run --rm infer_cpp bash -lc "grep -R -n 'LGBM_' cpp src | head -n 20"
```

```text
cpp/lgbm_predictor.cpp:87:      load_symbol(lib, "LGBM_BoosterCreateFromModelfile"));
cpp/lgbm_predictor.cpp:88:  api->booster_free = reinterpret_cast<BoosterFreeFn>(load_symbol(lib, "LGBM_BoosterFree"));
cpp/lgbm_predictor.cpp:90:      load_symbol(lib, "LGBM_BoosterPredictForMat"));
cpp/lgbm_predictor.cpp:91:  api->get_last_error = reinterpret_cast<GetLastErrorFn>(load_symbol(lib, "LGBM_GetLastError"));
```

```bash
docker compose run --rm infer_cpp bash -lc "grep -R -n 'BoosterPredict' cpp src | head -n 20"
```

```text
cpp/lgbm_predictor.cpp:20:using BoosterPredictForMatFn = int (*)(...);
cpp/lgbm_predictor.cpp:89:  api->booster_predict_for_mat = reinterpret_cast<BoosterPredictForMatFn>(
cpp/lgbm_predictor.cpp:90:      load_symbol(lib, "LGBM_BoosterPredictForMat"));
```

## 5. 判定（Docker 基準）

- `cpp_lgbm` は `run_infer_cpp.py` から `ctypes` 経由で C++ 共有ライブラリを呼び出す経路になっている。
- C++ 実装は LightGBM C API (`LGBM_BoosterPredictForMat` など) を解決して推論している。
- したがって `py_lgbm` の `Booster.predict` 直呼びとは実装経路が異なる。
- Docker でのビルド・成果物・依存確認まで再現できるため、「C++ LightGBM 推論が実体として存在する」ことの証明資料として使用可能。
