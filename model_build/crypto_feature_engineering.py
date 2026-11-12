"""Build wash-trading detection features from historical BTCUSDT data.

The output of this script is the offline feature table that Phase 2 will use
to train the Isolation Forest model. It intentionally mirrors features we can
also compute from live Binance trades after Spark aggregates them into
one-minute windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT_DIR / "dataset" / "BTCUSDT_perp_1m_master.parquet"
DEFAULT_CSV = ROOT_DIR / "dataset" / "BTCUSDT.csv"
DEFAULT_OUTPUT = ROOT_DIR / "model_build" / "crypto_features.parquet"
DEFAULT_SAMPLE_OUTPUT = ROOT_DIR / "model_build" / "crypto_features_sample.csv"

REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_sell_base_vol",
    "timestamp_utc",
]

FEATURE_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "total_volume",
    "quote_volume",
    "num_trades",
    "avg_trade_size",
    "volume_per_trade",
    "price_change_pct",
    "high_low_spread_pct",
    "taker_buy_ratio",
    "taker_sell_ratio",
    "buy_sell_imbalance",
    "volume_to_price_movement_ratio",
    "rolling_volume_mean",
    "rolling_volume_std",
    "rolling_volume_zscore",
    "rolling_trade_count_mean",
    "rolling_trade_count_std",
    "rolling_trade_count_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create crypto wash-trading features from BTCUSDT historical data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input dataset path. Defaults to dataset/BTCUSDT_perp_1m_master.parquet, then dataset/BTCUSDT.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path for engineered features.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_SAMPLE_OUTPUT,
        help="Small CSV preview output path.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional number of rows to load for quick verification.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Rolling window size in rows. The dataset is one-minute data, so 60 means one hour.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=1000,
        help="Rows to write to the sample CSV preview.",
    )
    return parser.parse_args()


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path:
        resolved = input_path if input_path.is_absolute() else ROOT_DIR / input_path
        if not resolved.exists():
            raise FileNotFoundError(f"Input dataset not found: {resolved}")
        return resolved

    if DEFAULT_PARQUET.exists():
        return DEFAULT_PARQUET
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV

    raise FileNotFoundError(
        "No dataset found. Expected dataset/BTCUSDT_perp_1m_master.parquet "
        "or dataset/BTCUSDT.csv."
    )


def validate_columns(columns: Iterable[str]) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_dataset(path: Path, sample_rows: int | None) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    if path.suffix.lower() == ".parquet":
        if sample_rows:
            parquet_file = pq.ParquetFile(path)
            batches = []
            loaded_rows = 0
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index, columns=REQUIRED_COLUMNS)
                batch_df = table.to_pandas()
                batches.append(batch_df)
                loaded_rows += len(batch_df)
                if loaded_rows >= sample_rows:
                    break
            df = pd.concat(batches, ignore_index=True).head(sample_rows).copy()
        else:
            df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, nrows=sample_rows)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    validate_columns(df.columns)
    print(f"Loaded rows: {len(df):,}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "timestamp_utc"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    df = df[df["open"] > 0]
    df = df[df["num_trades"] >= 0]
    df = df[df["volume"] >= 0]

    removed = before - len(df)
    print(f"Cleaned rows: {len(df):,} (removed {removed:,})")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    epsilon = 1e-9
    features = df.copy()

    features["total_volume"] = features["volume"]
    features["avg_trade_size"] = features["volume"] / features["num_trades"].clip(lower=1)
    features["volume_per_trade"] = features["avg_trade_size"]
    features["price_change_pct"] = (
        (features["close"] - features["open"]).abs() / features["open"].clip(lower=epsilon)
    ) * 100
    features["high_low_spread_pct"] = (
        (features["high"] - features["low"]).abs() / features["open"].clip(lower=epsilon)
    ) * 100
    features["taker_buy_ratio"] = (
        features["taker_buy_base_vol"] / features["volume"].clip(lower=epsilon)
    )
    features["taker_sell_ratio"] = (
        features["taker_sell_base_vol"] / features["volume"].clip(lower=epsilon)
    )
    features["buy_sell_imbalance"] = (
        (features["taker_buy_base_vol"] - features["taker_sell_base_vol"]).abs()
        / features["volume"].clip(lower=epsilon)
    )
    features["volume_to_price_movement_ratio"] = (
        features["volume"] / features["price_change_pct"].clip(lower=epsilon)
    )

    rolling_volume = features["volume"].rolling(
        window=rolling_window, min_periods=max(5, rolling_window // 10)
    )
    features["rolling_volume_mean"] = rolling_volume.mean()
    features["rolling_volume_std"] = rolling_volume.std(ddof=0)
    features["rolling_volume_zscore"] = (
        (features["volume"] - features["rolling_volume_mean"])
        / features["rolling_volume_std"].replace(0, np.nan)
    )

    rolling_trades = features["num_trades"].rolling(
        window=rolling_window, min_periods=max(5, rolling_window // 10)
    )
    features["rolling_trade_count_mean"] = rolling_trades.mean()
    features["rolling_trade_count_std"] = rolling_trades.std(ddof=0)
    features["rolling_trade_count_zscore"] = (
        (features["num_trades"] - features["rolling_trade_count_mean"])
        / features["rolling_trade_count_std"].replace(0, np.nan)
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=FEATURE_COLUMNS)
    features = features[FEATURE_COLUMNS]
    print(f"Feature rows: {len(features):,}")
    return features.reset_index(drop=True)


def write_outputs(features: pd.DataFrame, output: Path, sample_output: Path, preview_rows: int) -> None:
    output = output if output.is_absolute() else ROOT_DIR / output
    sample_output = sample_output if sample_output.is_absolute() else ROOT_DIR / sample_output
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)

    features.to_parquet(output, index=False)
    features.head(preview_rows).to_csv(sample_output, index=False)

    print(f"Wrote feature parquet: {output}")
    print(f"Wrote preview CSV: {sample_output}")


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    df = load_dataset(input_path, args.sample_rows)
    cleaned = clean_dataset(df)
    features = engineer_features(cleaned, args.rolling_window)
    write_outputs(features, args.output, args.sample_output, args.preview_rows)

    print("\nFeature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


if __name__ == "__main__":
    main()
