# Project Brief

## Goal
推論速度と精度の関係を、3つの実装で比較可能な形で可視化する。

- Python LightGBM (`py_lgbm`)
- C++ LightGBM predictor (`cpp_lgbm`)
- C++ Logistic Regression (`cpp_lr`)

## Value
- 実装ごとの性能差を同一データで比較できる
- 速度と精度を同時に見て判断できる
- 非機械学習ユーザーにも説明しやすい

## Success Criteria
- 学習・推論・計測を一連実行できる
- 3モデルの結果が同一評価データで保存される
- ダッシュボードで3モデル比較できる
