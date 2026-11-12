"""Train an unsupervised crypto wash-trading anomaly model.

This Phase 2 script trains an Isolation Forest on historical BTCUSDT feature
windows. Because confirmed wash-trading labels are not publicly available, we
report anomaly rates and rule thresholds instead of true accuracy.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_PATH = ROOT_DIR / "model_build" / "crypto_features.parquet"
DEFAULT_MODEL_PATH = ROOT_DIR / "data_processor" / "wash_trading_model.pkl"
DEFAULT_FEATURE_LIST_PATH = ROOT_DIR / "data_processor" / "wash_trading_features.json"
DEFAULT_THRESHOLDS_PATH = ROOT_DIR / "data_processor" / "wash_trading_thresholds.json"
DEFAULT_SUMMARY_PATH = ROOT_DIR / "model_build" / "wash_model_training_summary.json"

MODEL_FEATURE_COLUMNS = [
    "price_change_pct",
    "high_low_spread_pct",
    "total_volume",
    "quote_volume",
    "num_trades",
    "avg_trade_size",
    "volume_per_trade",
    "taker_buy_ratio",
    "taker_sell_ratio",
    "buy_sell_imbalance",
    "volume_to_price_movement_ratio",
    "rolling_volume_zscore",
    "rolling_trade_count_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest for crypto wash-trading anomaly detection."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to engineered feature parquet file.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for trained model pickle.",
    )
    parser.add_argument(
        "--feature-list-output",
        type=Path,
        default=DEFAULT_FEATURE_LIST_PATH,
        help="Output path for model feature JSON.",
    )
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help="Output path for wash-score threshold JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output path for training summary JSON.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row sample for quick verification/training.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected anomaly fraction for Isolation Forest.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for Isolation Forest.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="Run Phase 1 feature generation first if feature file is missing.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def generate_features_if_needed(features_path: Path) -> None:
    if features_path.exists():
        return

    script = ROOT_DIR / "model_build" / "crypto_feature_engineering.py"
    print(f"Feature file missing. Running Phase 1 script: {script}")
    subprocess.run([sys.executable, str(script), "--output", str(features_path)], check=True)


def load_features(features_path: Path, sample_rows: int | None, random_state: int) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. Run Phase 1 first or use --generate-features."
        )

    print(f"Loading features: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"Loaded feature rows: {len(df):,}")

    missing_columns = sorted(set(MODEL_FEATURE_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Feature file is missing model columns: {missing_columns}")

    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_state).sort_index().reset_index(drop=True)
        print(f"Using sampled rows: {len(df):,}")

    return df


def calculate_thresholds(df: pd.DataFrame) -> dict[str, Any]:
    thresholds = {
        "schema_version": 1,
        "description": "Percentile thresholds used by the rule-based wash score.",
        "high_volume_p95": float(df["total_volume"].quantile(0.95)),
        "high_num_trades_p95": float(df["num_trades"].quantile(0.95)),
        "high_volume_price_ratio_p95": float(df["volume_to_price_movement_ratio"].quantile(0.95)),
        "low_price_change_p25": float(df["price_change_pct"].quantile(0.25)),
        "high_buy_sell_imbalance_p95": float(df["buy_sell_imbalance"].quantile(0.95)),
        "rolling_volume_zscore_alert": 3.0,
        "wash_score_suspicious": 70,
        "wash_score_warning": 40,
        "rule_weights": {
            "high_volume_low_price_movement": 30,
            "high_trade_count": 20,
            "high_volume_price_ratio": 20,
            "high_buy_sell_imbalance": 15,
            "rolling_volume_spike": 15,
        },
    }
    return thresholds


def compute_wash_score_preview(df: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=np.int16), index=df.index)
    weights = thresholds["rule_weights"]

    score += np.where(
        (df["total_volume"] >= thresholds["high_volume_p95"])
        & (df["price_change_pct"] <= thresholds["low_price_change_p25"]),
        weights["high_volume_low_price_movement"],
        0,
    )
    score += np.where(
        df["num_trades"] >= thresholds["high_num_trades_p95"],
        weights["high_trade_count"],
        0,
    )
    score += np.where(
        df["volume_to_price_movement_ratio"] >= thresholds["high_volume_price_ratio_p95"],
        weights["high_volume_price_ratio"],
        0,
    )
    score += np.where(
        df["buy_sell_imbalance"] >= thresholds["high_buy_sell_imbalance_p95"],
        weights["high_buy_sell_imbalance"],
        0,
    )
    score += np.where(
        df["rolling_volume_zscore"] >= thresholds["rolling_volume_zscore_alert"],
        weights["rolling_volume_spike"],
        0,
    )

    return score.clip(upper=100)


def train_model(df: pd.DataFrame, contamination: float, n_estimators: int, random_state: int) -> Pipeline:
    X = df[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training Isolation Forest...")
    model.fit(X)
    return model


def write_json(path: Path, payload: dict[str, Any] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_model(path: Path, model: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(model, model_file)


def main() -> None:
    args = parse_args()
    features_path = resolve_path(args.features)
    model_output = resolve_path(args.model_output)
    feature_list_output = resolve_path(args.feature_list_output)
    thresholds_output = resolve_path(args.thresholds_output)
    summary_output = resolve_path(args.summary_output)

    if args.generate_features:
        generate_features_if_needed(features_path)

    df = load_features(features_path, args.sample_rows, args.random_state)
    thresholds = calculate_thresholds(df)
    model = train_model(df, args.contamination, args.n_estimators, args.random_state)

    X = df[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    predictions = model.predict(X)
    anomaly_scores = model.decision_function(X)
    anomaly_count = int((predictions == -1).sum())
    wash_scores = compute_wash_score_preview(df, thresholds)
    suspicious_count = int(((predictions == -1) & (wash_scores >= thresholds["wash_score_suspicious"])).sum())

    summary = {
        "training_rows": int(len(df)),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "features_path": str(features_path),
        "model_output": str(model_output),
        "contamination": args.contamination,
        "n_estimators": args.n_estimators,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round((anomaly_count / len(df)) * 100, 4) if len(df) else 0.0,
        "wash_score_suspicious_count_on_training": suspicious_count,
        "wash_score_suspicious_percentage_on_training": round((suspicious_count / len(df)) * 100, 4)
        if len(df)
        else 0.0,
        "anomaly_score_min": float(np.min(anomaly_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_max": float(np.max(anomaly_scores)),
        "note": "No true accuracy is reported because confirmed wash-trading labels are unavailable.",
    }

    write_model(model_output, model)
    write_json(feature_list_output, MODEL_FEATURE_COLUMNS)
    write_json(thresholds_output, thresholds)
    write_json(summary_output, summary)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved model: {model_output}")
    print(f"Saved feature list: {feature_list_output}")
    print(f"Saved thresholds: {thresholds_output}")
    print(f"Saved summary: {summary_output}")


if __name__ == "__main__":
    main()
