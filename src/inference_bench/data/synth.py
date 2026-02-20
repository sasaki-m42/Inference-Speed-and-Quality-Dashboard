from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from inference_bench.config import load_config
from inference_bench.db.io import connect, initialize_schema, insert_samples, reset_runtime_tables


@dataclass
class SyntheticDataset:
    features: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    flight_ids: np.ndarray
    timestamps: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _noise_col(noise: np.ndarray, idx: int) -> np.ndarray:
    if idx < noise.shape[1]:
        return noise[:, idx]
    return np.zeros(noise.shape[0], dtype=np.float32)


def _split_by_flight(
    flight_ids: np.ndarray,
    eval_ratio: float,
    seed: int,
) -> np.ndarray:
    n_samples = len(flight_ids)
    split = np.full(n_samples, "train", dtype=object)
    if n_samples == 0 or eval_ratio <= 0:
        return split

    unique_ids, first_idx, counts = np.unique(flight_ids, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    unique_ids = unique_ids[order]
    counts = counts[order]
    n_flights = len(unique_ids)

    if n_flights <= 1:
        if eval_ratio >= 0.5 and n_samples > 0:
            split[:] = "eval"
        return split

    target_eval_rows = int(round(n_samples * eval_ratio))
    target_eval_rows = max(1, min(target_eval_rows, n_samples - 1))

    rng = np.random.default_rng(seed + 1009)
    flight_order = rng.permutation(n_flights)

    selected_flights: list[int] = []
    selected_rows = 0
    for flight_pos in flight_order:
        selected_flights.append(int(flight_pos))
        selected_rows += int(counts[flight_pos])
        if selected_rows >= target_eval_rows:
            break

    if len(selected_flights) >= n_flights:
        # Keep at least one flight in train split.
        drop_idx = max(selected_flights, key=lambda idx: int(counts[idx]))
        selected_flights.remove(drop_idx)

    eval_flight_ids = unique_ids[selected_flights]
    eval_mask = np.isin(flight_ids, eval_flight_ids)
    split[eval_mask] = "eval"
    return split


def generate_synthetic_dataset(
    n_samples: int = 20000,
    n_features: int = 64,
    seed: int = 42,
    mode: str = "hybrid",
    eval_ratio: float = 0.2,
) -> SyntheticDataset:
    if n_features < 10:
        raise ValueError("n_features must be >= 10")
    if mode not in {"linear", "xor", "hybrid"}:
        raise ValueError("mode must be one of: linear, xor, hybrid")

    rng = np.random.default_rng(seed)
    noise_dim = n_features - 8
    color_id = np.empty(n_samples, dtype=np.int32)
    mark_id = np.empty(n_samples, dtype=np.int32)
    flight_ids = np.empty(n_samples, dtype=object)
    timestamps = np.empty(n_samples, dtype=np.int64)
    noise = rng.normal(0.0, 1.0, size=(n_samples, noise_dim)).astype(np.float32)

    n_flights = max(1, min(max(10, n_samples // 40), n_samples))
    base_len = n_samples // n_flights
    remainder = n_samples % n_flights
    drift_dim = min(4, noise_dim)

    start = 0
    ts_cursor = 1_700_000_000
    for flight_idx in range(n_flights):
        flight_len = base_len + (1 if flight_idx < remainder else 0)
        if flight_len == 0:
            continue

        end = start + flight_len
        flight_color = int(rng.integers(0, 4))
        flight_mark = int(rng.integers(0, 4))
        color_id[start:end] = flight_color
        mark_id[start:end] = flight_mark
        flight_ids[start:end] = f"FL-{flight_idx:04d}"
        timestamps[start:end] = np.arange(ts_cursor, ts_cursor + flight_len, dtype=np.int64)

        # A small per-flight offset keeps identity-specific characteristics.
        noise[start:end] += rng.normal(0.0, 0.15, size=(1, noise_dim)).astype(np.float32)
        if drift_dim > 0:
            drift = np.cumsum(
                rng.normal(0.0, 0.08, size=(flight_len, drift_dim)).astype(np.float32),
                axis=0,
            )
            noise[start:end, :drift_dim] += drift

        start = end
        ts_cursor += flight_len

    color_onehot = np.eye(4, dtype=np.float32)[color_id]
    mark_onehot = np.eye(4, dtype=np.float32)[mark_id]
    features = np.hstack([color_onehot, mark_onehot, noise]).astype(np.float32)

    linear_score = (
        1.2 * (color_id == 0)
        - 0.7 * (color_id == 3)
        + 1.4 * (mark_id == 1)
        - 0.9 * (mark_id == 2)
        + 0.25 * _noise_col(noise, 0)
        - 0.2 * _noise_col(noise, 1)
    )

    xor_score = (
        ((color_id % 2) ^ (mark_id % 2)).astype(np.float32) * 2.0
        - 1.0
        + 0.15 * _noise_col(noise, 2)
        - 0.1 * _noise_col(noise, 3)
    )

    if mode == "linear":
        logits = linear_score
    elif mode == "xor":
        logits = xor_score
    else:
        logits = 0.65 * linear_score + 0.35 * xor_score

    probs = _sigmoid(logits.astype(np.float32))
    labels = (rng.random(n_samples) < probs).astype(np.int32)

    split = _split_by_flight(flight_ids=flight_ids, eval_ratio=eval_ratio, seed=seed)

    return SyntheticDataset(
        features=features,
        labels=labels,
        split=split,
        flight_ids=flight_ids,
        timestamps=timestamps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark dataset and store into SQLite.")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["linear", "xor", "hybrid"], default="hybrid")
    parser.add_argument("--reset", action="store_true", help="Clear runtime tables before inserting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    db_path = args.db or cfg.db_path

    dataset = generate_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        mode=args.mode,
    )

    con = connect(db_path)
    initialize_schema(con)
    if args.reset:
        reset_runtime_tables(con)

    inserted = insert_samples(
        con,
        features=dataset.features,
        labels=dataset.labels,
        flight_ids=dataset.flight_ids,
        timestamps=dataset.timestamps,
        split=dataset.split,
        source=cfg.sample_source,
    )

    train_count = int((dataset.split == "train").sum())
    eval_count = int((dataset.split == "eval").sum())
    print(
        f"Inserted {inserted} samples into {db_path} "
        f"(train={train_count}, eval={eval_count}, dim={dataset.features.shape[1]})."
    )


if __name__ == "__main__":
    main()
