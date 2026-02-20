from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class OpenSkyDataset:
    features: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    flight_ids: np.ndarray
    timestamps: np.ndarray


def _coerce_timestamp(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64)
        if vals.max() > 10_000_000_000:
            vals = (vals // 1000).astype(np.int64)
        return vals.to_numpy()

    ts = pd.to_datetime(series, errors="coerce", utc=True)
    sec = (ts.astype("int64") // 10**9).astype(np.int64)
    sec = pd.Series(sec, index=ts.index)
    sec.loc[ts.isna()] = 0
    return sec.to_numpy(dtype=np.int64)


def _column(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def _binary_column(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            col = df[name]
            if col.dtype == bool:
                return col.astype(np.float32)
            if np.issubdtype(col.dtype, np.number):
                return pd.to_numeric(col, errors="coerce").fillna(0).clip(0, 1).astype(np.float32)
            normalized = (
                col.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
                .fillna(0.0)
            )
            return normalized.astype(np.float32)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return numer / denom


def _build_features(df: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    lat = _column(df, ["lat", "latitude"]).to_numpy(dtype=np.float32)
    lon = _column(df, ["lon", "longitude"]).to_numpy(dtype=np.float32)
    velocity = _column(df, ["velocity", "groundspeed"]).to_numpy(dtype=np.float32)
    heading = _column(df, ["heading", "track", "true_track"]).to_numpy(dtype=np.float32)
    vertical_rate = _column(df, ["vertical_rate", "vertrate"]).to_numpy(dtype=np.float32)
    baro_alt = _column(df, ["baro_altitude", "baroaltitude", "altitude"]).to_numpy(dtype=np.float32)
    geo_alt = _column(df, ["geo_altitude", "geoaltitude"]).to_numpy(dtype=np.float32)

    onground = _binary_column(df, ["onground", "on_ground"]).to_numpy(dtype=np.float32)
    spi = _binary_column(df, ["spi"]).to_numpy(dtype=np.float32)

    heading_rad = np.deg2rad(np.mod(heading, 360.0))

    ts_dt = pd.to_datetime(timestamps, unit="s", utc=True)
    hour = ts_dt.hour.to_numpy(dtype=np.float32)
    day = ts_dt.dayofweek.to_numpy(dtype=np.float32)

    speed_kmh = velocity * 3.6

    features = np.column_stack(
        [
            np.clip(lat / 90.0, -1.0, 1.0),
            np.clip(lon / 180.0, -1.0, 1.0),
            np.clip(speed_kmh / 1200.0, 0.0, 2.0),
            np.sin(heading_rad),
            np.cos(heading_rad),
            np.clip(vertical_rate / 30.0, -2.0, 2.0),
            np.clip(baro_alt / 15000.0, -1.0, 2.0),
            np.clip(geo_alt / 15000.0, -1.0, 2.0),
            onground,
            spi,
            np.sin((2.0 * np.pi * hour) / 24.0),
            np.cos((2.0 * np.pi * hour) / 24.0),
            np.sin((2.0 * np.pi * day) / 7.0),
            np.cos((2.0 * np.pi * day) / 7.0),
            np.clip(_safe_ratio(np.abs(vertical_rate), np.maximum(speed_kmh, 1.0)), 0.0, 2.0),
            np.clip((1.0 - onground) * np.minimum(speed_kmh / 400.0, 2.0), 0.0, 2.0),
        ]
    ).astype(np.float32)
    return features


def _build_proxy_labels(features: np.ndarray) -> np.ndarray:
    # Proxy target for benchmark only (OpenSky does not provide binary class labels for this task).
    score = (
        0.45 * features[:, 2]
        + 0.25 * features[:, 6]
        + 0.15 * features[:, 7]
        + 0.25 * features[:, 15]
        - 0.2 * features[:, 8]
    )
    threshold = float(np.percentile(score, 58))
    labels = (score >= threshold).astype(np.int32)

    if labels.min() == labels.max():
        labels = (score >= float(score.mean())).astype(np.int32)
    return labels


def _build_flight_ids(df: pd.DataFrame) -> np.ndarray:
    if "callsign" in df.columns:
        callsign = df["callsign"].astype(str).str.strip().replace("", np.nan)
    else:
        callsign = pd.Series([np.nan] * len(df), index=df.index)

    if "icao24" in df.columns:
        icao = df["icao24"].astype(str).str.strip().replace("", "unknown")
    else:
        icao = pd.Series(["unknown"] * len(df), index=df.index)

    flight_id = callsign.fillna(icao)
    return flight_id.astype(str).to_numpy(dtype=object)


def load_opensky_csv_dataset(
    csv_path: str | Path,
    seed: int = 42,
    eval_ratio: float = 0.2,
    max_rows: int | None = None,
) -> OpenSkyDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"OpenSky CSV not found: {path}")

    df = pd.read_csv(path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if len(df) < 100:
        raise ValueError("OpenSky CSV needs at least 100 rows for stable split")

    ts_col_candidates = ["time", "timestamp", "last_contact"]
    ts_col = next((c for c in ts_col_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError("OpenSky CSV must have one of columns: time, timestamp, last_contact")

    timestamps = _coerce_timestamp(df[ts_col])
    features = _build_features(df, timestamps)
    labels = _build_proxy_labels(features)
    flight_ids = _build_flight_ids(df)

    _, idx_eval = train_test_split(
        np.arange(len(df)),
        test_size=eval_ratio,
        random_state=seed,
        stratify=labels,
    )

    split = np.array(["train"] * len(df), dtype=object)
    split[idx_eval] = "eval"

    return OpenSkyDataset(
        features=features,
        labels=labels,
        split=split,
        flight_ids=flight_ids,
        timestamps=timestamps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview OpenSky CSV conversion")
    parser.add_argument("--csv", required=True, help="Path to OpenSky CSV")
    parser.add_argument("--max-rows", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_opensky_csv_dataset(args.csv, max_rows=args.max_rows)
    print(
        f"rows={len(ds.labels)} dim={ds.features.shape[1]} "
        f"train={(ds.split=='train').sum()} eval={(ds.split=='eval').sum()}"
    )


if __name__ == "__main__":
    main()
