# Runbook

この文書は、`Inference Speed and Quality Benchmark` をリポジトリ上で再現実行するための手順です。

## 1. 前提

- Docker Engine / Docker Compose が利用可能
- 推奨: 4core 以上、メモリ 8GB 以上
- 動作想定: macOS + Docker Desktop / Windows + Docker Desktop（ローカルディスク配下）
  - Windows は同期フォルダ/外部ドライブ配下ではボリュームマウントが失敗する場合があります（例: OneDrive / Google Drive / 外付けドライブ）

## 2. 最短実行（Docker）

既定は synthetic データで実行されます。

```bash
docker compose build
docker compose run --rm demo
docker compose up -d viz
```

- UI: [http://localhost:8501](http://localhost:8501)
- 終了:

```bash
docker compose down
```

## 3. 生成物

実行後に次の生成物が作られます（初期状態でリポジトリに含める必要はありません）。

- `artifacts/bench.db`
- `artifacts/models/*`
- `artifacts/lib/*`（C++共有ライブラリ）
- `artifacts/logs/execution_log.jsonl`
- `reports/site/index.html`（`export_static` 実行時）

## 4. 静的レポート生成

```bash
docker compose run --rm export_static
```

- 出力: `reports/site/index.html`

## 5. ローカル実行（任意）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m inference_bench.pipeline.run_all --threads 1
PYTHONPATH=src streamlit run src/inference_bench/viz/app.py
```

## 6. データソースに関する注意

- UI は既定でDB（`/app/artifacts/bench.db`）を表示します。
- 開発時にカスタムDBを使う場合のみ、対応する環境変数を有効化してください。

## 7. よくあるエラー

- `DB not found`: 先に `docker compose run --rm demo` を実行
- C++ライブラリ未生成: `demo` 実行で自動ビルドされる。手動時は `run_all` を利用
- `port 8501 already in use`: 既存プロセス停止後に再実行

## 8. 検証チェックリスト

1. `docker compose run --rm demo` が正常終了する
2. UIで `Refresh Data` 後に3モデル比較（品質/速度）が表示される
3. `docker compose run --rm export_static` で `reports/site/index.html` が生成される
