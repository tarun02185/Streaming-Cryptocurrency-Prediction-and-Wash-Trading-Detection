CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INTEGER,
    transaction_amount FLOAT,
    transaction_time TIMESTAMP,
    location VARCHAR,
    transaction_method VARCHAR,
    account_age INTEGER,
    transaction_hour INTEGER,
    user_transactions_last_24h INTEGER,
    is_fraud BOOLEAN
);

CREATE TABLE IF NOT EXISTS account_details (
    account_id INTEGER PRIMARY KEY,
    signup_time TIMESTAMP
);

INSERT INTO account_details (account_id, signup_time)
VALUES
    (11111, TIMESTAMP '2022-09-18 14:35:22'),
    (22222, TIMESTAMP '2021-08-05 09:15:37')
ON CONFLICT (account_id) DO NOTHING;

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

CREATE INDEX IF NOT EXISTS idx_crypto_trade_windows_symbol_time
    ON crypto_trade_windows (symbol, window_end DESC);

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

CREATE INDEX IF NOT EXISTS idx_wash_trade_alerts_symbol_time
    ON wash_trade_alerts (symbol, created_at DESC);
