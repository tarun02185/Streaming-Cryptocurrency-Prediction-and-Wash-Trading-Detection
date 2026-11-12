# Streaming Cryptocurrency Prediction and Wash Trading Detection

A Dockerized real-time streaming project that extends a fraud-detection architecture into a cryptocurrency market anomaly and wash-trading-pattern detection system.

## Project Goal

The goal is to detect suspicious wash-trading-like behavior in cryptocurrency markets using streaming data, feature engineering, anomaly detection, and live dashboard visualization.

Public crypto APIs do not expose buyer/seller account identity, so this system does not claim legally confirmed wash trading. It predicts suspicious market behavior using public indicators such as abnormal volume, trade count, price movement, and buy/sell imbalance.

## Architecture

```text
Historical BTCUSDT Dataset
        -> Feature Engineering
        -> Isolation Forest Training
        -> wash_trading_model.pkl

Live Binance Trade Stream
        -> crypto_trade_producer.py
        -> Kafka topic: crypto_trades
        -> wash_trade_detector.py
        -> Isolation Forest + Wash Score Rules
        -> TimescaleDB
        -> Grafana Dashboard
```

The original banking fraud detection pipeline is still available as a working reference/demo inside the project.

## Why This Dataset

The project uses BTCUSDT 1-minute Binance-style historical data because it closely matches the real-time streaming format used in the live system.

It includes:

```text
open
high
low
close
volume
quote_volume
num_trades
taker_buy_base_vol
taker_sell_base_vol
timestamp_utc
```

This makes the offline training features and live streaming features consistent.

The dataset is not committed to GitHub because it is large. Place it locally under:

```text
dataset/
```

## Why Isolation Forest

The crypto pipeline uses:

```text
Isolation Forest + rule-based wash score
```

Isolation Forest was chosen because:

```text
It works without labels
Confirmed wash-trading labels are not publicly available
It is CPU-friendly
It is fast enough for tabular anomaly detection
It is easier to explain than deep learning models
```

The rule-based wash score explains why a market window looks suspicious, while Isolation Forest provides anomaly detection on the engineered feature set.

## Implemented Phases

### Phase 1: Dataset Preparation and Feature Engineering

Implemented in:

```text
model_build/crypto_feature_engineering.py
```

This script reads the downloaded BTCUSDT historical dataset and creates model-ready features such as:

```text
price_change_pct
high_low_spread_pct
total_volume
quote_volume
num_trades
avg_trade_size
volume_per_trade
taker_buy_ratio
taker_sell_ratio
buy_sell_imbalance
volume_to_price_movement_ratio
rolling_volume_zscore
rolling_trade_count_zscore
```

Run full feature generation:

```powershell
python model_build\crypto_feature_engineering.py
```

### Phase 2: Model Training

Implemented in:

```text
model_build/train_wash_model.py
```

Run training:

```powershell
python model_build\train_wash_model.py
```

Generated local artifacts:

```text
data_processor/wash_trading_model.pkl
data_processor/wash_trading_features.json
data_processor/wash_trading_thresholds.json
model_build/wash_model_training_summary.json
```

These generated artifacts are intentionally ignored by Git.

### Phase 3: Live API Streaming

Implemented in:

```text
data_producer/crypto_trade_producer.py
```

The producer connects to Binance public WebSocket trade streams for:

```text
BTCUSDT
ETHUSDT
SOLUSDT
DOGEUSDT
```

and writes normalized trade events to Kafka topic:

```text
crypto_trades
```

### Phase 4: Real-Time Prediction Pipeline

Implemented in:

```text
data_processor/wash_trade_detector.py
```

The detector:

```text
Reads crypto_trades from Kafka
Aggregates by 1-minute windows
Calculates engineered features
Loads Isolation Forest + thresholds
Calculates wash score
Writes results to TimescaleDB tables:
  crypto_trade_windows
  wash_trade_alerts
```

For runtime reliability, the crypto detector runs in self-contained Spark local mode inside its container so it does not compete with the original fraud streaming job for shared cluster workers.

## Dashboards

Grafana provides:

```text
Fraud Detection Dashboard
Wash Trading Detection Dashboard
```

Open Grafana:

```text
http://localhost:3000/dashboards
```

Default login:

```text
admin / admin
```

## Running The Project

Start the full stack:

```powershell
docker compose up --build -d
```

Check container status:

```powershell
docker compose ps
```

Stop containers:

```powershell
docker compose down
```

## Verification Commands

Check crypto producer logs:

```powershell
docker compose logs --tail=50 crypto-producer
```

Check wash detector logs:

```powershell
docker compose logs --tail=100 wash-detector
```

Check crypto alert counts:

```powershell
docker compose exec -T timescaledb psql -U user -d fraud_detection -c "SELECT COUNT(*) AS alert_rows, COUNT(*) FILTER (WHERE wash_trade) AS suspicious_rows FROM wash_trade_alerts;"
```

Inspect recent crypto predictions:

```powershell
docker compose exec -T timescaledb psql -U user -d fraud_detection -c "SELECT symbol, wash_score, wash_trade, reason, created_at FROM wash_trade_alerts ORDER BY created_at DESC LIMIT 10;"
```

Check the original fraud table:

```powershell
docker compose exec -T timescaledb psql -U user -d fraud_detection -c "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE is_fraud) AS frauds FROM transactions;"
```

## Attribution

This project extends an open-source real-time fraud detection architecture and adapts it for streaming cryptocurrency prediction and wash-trading-pattern detection. The original license is preserved in this repository.
