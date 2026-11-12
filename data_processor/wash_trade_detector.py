import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    expr,
    from_json,
    lit,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
    to_timestamp,
    window,
)
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


APP_DIR = Path("/opt/spark-apps")
MODEL_BUILD_DIR = Path("/opt/model-build")
MODEL_PATH = APP_DIR / "wash_trading_model.pkl"
FEATURES_PATH = APP_DIR / "wash_trading_features.json"
THRESHOLDS_PATH = APP_DIR / "wash_trading_thresholds.json"


def get_db_config():
    return {
        "host": "timescaledb",
        "port": 5432,
        "database": "fraud_detection",
        "user": "user",
        "password": "password",
    }


def get_jdbc_options():
    return {
        "url": "jdbc:postgresql://timescaledb:5432/fraud_detection",
        "driver": "org.postgresql.Driver",
        "user": "user",
        "password": "password",
    }


def ensure_tables():
    ddl = """
    CREATE TABLE IF NOT EXISTS crypto_trade_windows (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        window_start TIMESTAMP NOT NULL,
        window_end TIMESTAMP NOT NULL,
        open_price DOUBLE PRECISION,
        high_price DOUBLE PRECISION,
        low_price DOUBLE PRECISION,
        close_price DOUBLE PRECISION,
        total_volume DOUBLE PRECISION,
        quote_volume DOUBLE PRECISION,
        num_trades INTEGER,
        taker_buy_base_vol DOUBLE PRECISION,
        taker_sell_base_vol DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS wash_trade_alerts (
        alert_id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        window_start TIMESTAMP NOT NULL,
        window_end TIMESTAMP NOT NULL,
        open_price DOUBLE PRECISION,
        high_price DOUBLE PRECISION,
        low_price DOUBLE PRECISION,
        close_price DOUBLE PRECISION,
        total_volume DOUBLE PRECISION,
        quote_volume DOUBLE PRECISION,
        num_trades INTEGER,
        price_change_pct DOUBLE PRECISION,
        high_low_spread_pct DOUBLE PRECISION,
        avg_trade_size DOUBLE PRECISION,
        volume_per_trade DOUBLE PRECISION,
        taker_buy_ratio DOUBLE PRECISION,
        taker_sell_ratio DOUBLE PRECISION,
        buy_sell_imbalance DOUBLE PRECISION,
        volume_to_price_movement_ratio DOUBLE PRECISION,
        rolling_volume_zscore DOUBLE PRECISION,
        rolling_trade_count_zscore DOUBLE PRECISION,
        anomaly_score DOUBLE PRECISION,
        is_anomaly BOOLEAN,
        wash_score INTEGER,
        wash_trade BOOLEAN,
        reason TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

    config = get_db_config()
    with psycopg2.connect(**config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        connection.commit()


def load_model_bundle():
    model = None
    model_error = None

    with FEATURES_PATH.open("r", encoding="utf-8") as feature_file:
        feature_columns = json.load(feature_file)

    with THRESHOLDS_PATH.open("r", encoding="utf-8") as thresholds_file:
        thresholds = json.load(thresholds_file)

    try:
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
        print("Loaded wash trading model successfully.", flush=True)
    except Exception as exc:
        model_error = str(exc)
        print(f"Initial model load failed: {exc}", flush=True)

        train_script = MODEL_BUILD_DIR / "train_wash_model.py"
        features_path = MODEL_BUILD_DIR / "crypto_features.parquet"
        can_rebuild = train_script.exists() and features_path.exists()

        if can_rebuild:
            print(
                "Trying to rebuild a container-compatible wash trading model from engineered features.",
                flush=True,
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(train_script),
                        "--features",
                        str(features_path),
                        "--model-output",
                        str(MODEL_PATH),
                        "--feature-list-output",
                        str(FEATURES_PATH),
                        "--thresholds-output",
                        str(THRESHOLDS_PATH),
                        "--summary-output",
                        "/tmp/wash_model_training_summary.json",
                    ],
                    check=True,
                )
                with MODEL_PATH.open("rb") as model_file:
                    model = pickle.load(model_file)
                model_error = None
                print(
                    "Rebuilt and loaded a container-compatible wash trading model successfully.",
                    flush=True,
                )
            except Exception as rebuild_exc:
                model_error = f"{exc}; rebuild_failed:{rebuild_exc}"
                print(
                    f"Warning: compatibility rebuild failed. Falling back to rule-based mode only. Error: {rebuild_exc}",
                    flush=True,
                )
        else:
            print(
                "Warning: engineered feature file is not mounted in the container. Falling back to rule-based mode only.",
                flush=True,
            )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "thresholds": thresholds,
        "model_error": model_error,
    }


MODEL_BUNDLE = load_model_bundle()


TRADE_SCHEMA = StructType(
    [
        StructField("trade_id", LongType(), True),
        StructField("symbol", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", DoubleType(), True),
        StructField("trade_time", StringType(), True),
        StructField("event_time", StringType(), True),
        StructField("is_buyer_maker", BooleanType(), True),
    ]
)


spark = SparkSession.builder.appName("CryptoWashTradeDetector").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def build_feature_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
    df = batch_df.copy()
    epsilon = 1e-9

    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])
    df = df.sort_values(["symbol", "window_start"]).reset_index(drop=True)

    df["price_change_pct"] = (
        (df["close_price"] - df["open_price"]).abs() / df["open_price"].clip(lower=epsilon)
    ) * 100
    df["high_low_spread_pct"] = (
        (df["high_price"] - df["low_price"]).abs() / df["open_price"].clip(lower=epsilon)
    ) * 100
    df["avg_trade_size"] = df["total_volume"] / df["num_trades"].clip(lower=1)
    df["volume_per_trade"] = df["avg_trade_size"]
    df["taker_buy_ratio"] = df["taker_buy_base_vol"] / df["total_volume"].clip(lower=epsilon)
    df["taker_sell_ratio"] = df["taker_sell_base_vol"] / df["total_volume"].clip(lower=epsilon)
    df["buy_sell_imbalance"] = (
        (df["taker_buy_base_vol"] - df["taker_sell_base_vol"]).abs()
        / df["total_volume"].clip(lower=epsilon)
    )
    df["volume_to_price_movement_ratio"] = (
        df["total_volume"] / df["price_change_pct"].clip(lower=epsilon)
    )

    grouped = df.groupby("symbol", group_keys=False)
    df["rolling_volume_mean"] = grouped["total_volume"].transform(
        lambda series: series.rolling(window=60, min_periods=5).mean()
    )
    df["rolling_volume_std"] = grouped["total_volume"].transform(
        lambda series: series.rolling(window=60, min_periods=5).std(ddof=0)
    )
    df["rolling_trade_count_mean"] = grouped["num_trades"].transform(
        lambda series: series.rolling(window=60, min_periods=5).mean()
    )
    df["rolling_trade_count_std"] = grouped["num_trades"].transform(
        lambda series: series.rolling(window=60, min_periods=5).std(ddof=0)
    )

    df["rolling_volume_zscore"] = (
        (df["total_volume"] - df["rolling_volume_mean"])
        / df["rolling_volume_std"].replace(0, np.nan)
    )
    df["rolling_trade_count_zscore"] = (
        (df["num_trades"] - df["rolling_trade_count_mean"])
        / df["rolling_trade_count_std"].replace(0, np.nan)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df["rolling_volume_zscore"] = df["rolling_volume_zscore"].fillna(0.0)
    df["rolling_trade_count_zscore"] = df["rolling_trade_count_zscore"].fillna(0.0)
    df["rolling_volume_mean"] = df["rolling_volume_mean"].fillna(df["total_volume"])
    df["rolling_trade_count_mean"] = df["rolling_trade_count_mean"].fillna(df["num_trades"])
    df["rolling_volume_std"] = df["rolling_volume_std"].fillna(0.0)
    df["rolling_trade_count_std"] = df["rolling_trade_count_std"].fillna(0.0)

    return df


def apply_rule_score(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    weights = thresholds["rule_weights"]
    score = np.zeros(len(df), dtype=np.int32)
    reasons = [[] for _ in range(len(df))]

    rule_1 = (df["total_volume"] >= thresholds["high_volume_p95"]) & (
        df["price_change_pct"] <= thresholds["low_price_change_p25"]
    )
    score += np.where(rule_1, weights["high_volume_low_price_movement"], 0)
    for index in np.where(rule_1)[0]:
        reasons[index].append("high_volume_low_price_movement")

    rule_2 = df["num_trades"] >= thresholds["high_num_trades_p95"]
    score += np.where(rule_2, weights["high_trade_count"], 0)
    for index in np.where(rule_2)[0]:
        reasons[index].append("high_trade_count")

    rule_3 = df["volume_to_price_movement_ratio"] >= thresholds["high_volume_price_ratio_p95"]
    score += np.where(rule_3, weights["high_volume_price_ratio"], 0)
    for index in np.where(rule_3)[0]:
        reasons[index].append("high_volume_price_ratio")

    rule_4 = df["buy_sell_imbalance"] >= thresholds["high_buy_sell_imbalance_p95"]
    score += np.where(rule_4, weights["high_buy_sell_imbalance"], 0)
    for index in np.where(rule_4)[0]:
        reasons[index].append("high_buy_sell_imbalance")

    rule_5 = df["rolling_volume_zscore"] >= thresholds["rolling_volume_zscore_alert"]
    score += np.where(rule_5, weights["rolling_volume_spike"], 0)
    for index in np.where(rule_5)[0]:
        reasons[index].append("rolling_volume_spike")

    df["wash_score"] = score.clip(max=100)
    df["reason"] = [", ".join(parts) if parts else "normal_market_behavior" for parts in reasons]
    return df


def apply_model(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    feature_frame = df[feature_columns].copy()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)

    if model is None:
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = False
        return df

    predictions = model.predict(feature_frame)
    scores = model.decision_function(feature_frame)
    df["anomaly_score"] = scores.astype(float)
    df["is_anomaly"] = predictions == -1
    return df


def finalize_predictions(df: pd.DataFrame, thresholds: dict, model_error: Optional[str]) -> pd.DataFrame:
    suspicious_score = thresholds["wash_score_suspicious"]
    warning_score = thresholds["wash_score_warning"]

    df["wash_trade"] = (
        (df["is_anomaly"] & (df["wash_score"] >= suspicious_score))
        | (df["wash_score"] >= suspicious_score)
        | (df["is_anomaly"] & (df["wash_score"] >= warning_score))
    )

    if model_error:
        df["reason"] = df["reason"] + f", model_fallback:{model_error}"

    df["created_at"] = datetime.utcnow()
    return df


def write_batch_to_db(feature_df: pd.DataFrame):
    if feature_df.empty:
        return

    jdbc_options = get_jdbc_options()

    raw_columns = [
        "symbol",
        "window_start",
        "window_end",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "total_volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_vol",
        "taker_sell_base_vol",
        "created_at",
    ]
    alerts_columns = [
        "symbol",
        "window_start",
        "window_end",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "total_volume",
        "quote_volume",
        "num_trades",
        "price_change_pct",
        "high_low_spread_pct",
        "avg_trade_size",
        "volume_per_trade",
        "taker_buy_ratio",
        "taker_sell_ratio",
        "buy_sell_imbalance",
        "volume_to_price_movement_ratio",
        "rolling_volume_zscore",
        "rolling_trade_count_zscore",
        "anomaly_score",
        "is_anomaly",
        "wash_score",
        "wash_trade",
        "reason",
        "created_at",
    ]

    spark.createDataFrame(feature_df[raw_columns]).write.format("jdbc").options(
        **jdbc_options, dbtable="crypto_trade_windows"
    ).mode("append").save()

    spark.createDataFrame(feature_df[alerts_columns]).write.format("jdbc").options(
        **jdbc_options, dbtable="wash_trade_alerts"
    ).mode("append").save()


def process_batch(batch_df, batch_id):
    print(f"Processing crypto batch {batch_id}", flush=True)

    if batch_df.rdd.isEmpty():
        print("Batch is empty; skipping.", flush=True)
        return

    pandas_df = batch_df.toPandas()
    if pandas_df.empty:
        print("Batch converted to empty pandas frame; skipping.", flush=True)
        return

    feature_df = build_feature_frame(pandas_df)
    feature_df = apply_rule_score(feature_df, MODEL_BUNDLE["thresholds"])
    feature_df = apply_model(feature_df, MODEL_BUNDLE)
    feature_df = finalize_predictions(
        feature_df,
        MODEL_BUNDLE["thresholds"],
        MODEL_BUNDLE["model_error"],
    )

    write_batch_to_db(feature_df)
    suspicious = int(feature_df["wash_trade"].sum())
    print(
        f"Stored batch {batch_id}: rows={len(feature_df)}, suspicious={suspicious}",
        flush=True,
    )


ensure_tables()

trade_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "broker:29092")
    .option("subscribe", "crypto_trades")
    .option("startingOffsets", "latest")
    .load()
)

trade_json = trade_stream.selectExpr("CAST(value AS STRING) AS value").select(
    from_json(col("value"), TRADE_SCHEMA).alias("trade")
)

trades = (
    trade_json.select("trade.*")
    .withColumn("trade_ts", to_timestamp(col("trade_time"), "yyyy-MM-dd HH:mm:ss"))
    .withColumn("event_ts", to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss"))
    .dropna(subset=["symbol", "price", "quantity", "trade_ts"])
)

windowed = (
    trades.groupBy(window(col("trade_ts"), "1 minute"), col("symbol"))
    .agg(
        spark_min(col("price")).alias("low_price"),
        spark_max(col("price")).alias("high_price"),
        spark_sum(col("quantity")).alias("total_volume"),
        spark_sum(col("price") * col("quantity")).alias("quote_volume"),
        count(lit(1)).alias("num_trades"),
        spark_sum(expr("CASE WHEN is_buyer_maker = false THEN quantity ELSE 0 END")).alias(
            "taker_buy_base_vol"
        ),
        spark_sum(expr("CASE WHEN is_buyer_maker = true THEN quantity ELSE 0 END")).alias(
            "taker_sell_base_vol"
        ),
        expr("min_by(price, trade_ts)").alias("open_price"),
        expr("max_by(price, trade_ts)").alias("close_price"),
    )
    .select(
        col("symbol"),
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("open_price"),
        col("high_price"),
        col("low_price"),
        col("close_price"),
        col("total_volume"),
        col("quote_volume"),
        col("num_trades"),
        col("taker_buy_base_vol"),
        col("taker_sell_base_vol"),
    )
)

(
    windowed.writeStream.foreachBatch(process_batch)
    .outputMode("update")
    .option("checkpointLocation", "/tmp/checkpoints/wash_trade_detector")
    .start()
    .awaitTermination()
)
