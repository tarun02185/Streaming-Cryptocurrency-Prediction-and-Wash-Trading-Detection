import json
import os
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from websocket import WebSocketApp


class CryptoTradeProducer:
    def __init__(self, brokers, topic, symbols, max_messages=None):
        self.brokers = brokers
        self.topic = topic
        self.symbols = [symbol.strip().lower() for symbol in symbols if symbol.strip()]
        if not self.symbols:
            raise ValueError("At least one crypto symbol must be configured.")
        self.max_messages = max_messages
        self.sent_messages = 0
        self.producer = self._connect_kafka_with_retry()
        self.ws_url = self._build_ws_url()

    def _connect_kafka_with_retry(self):
        while True:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.brokers,
                    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                    retries=5,
                    linger_ms=100,
                )
                print(
                    f"Connected to Kafka brokers: {','.join(self.brokers)}",
                    flush=True,
                )
                return producer
            except Exception as exc:
                print(
                    f"Kafka is not ready yet ({exc}). Retrying in 5 seconds...",
                    flush=True,
                )
                time.sleep(5)

    def _build_ws_url(self):
        streams = "/".join(f"{symbol}@trade" for symbol in self.symbols)
        return f"wss://stream.binance.com:9443/stream?streams={streams}"

    @staticmethod
    def _format_timestamp(timestamp_ms):
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _normalize_trade(self, payload):
        data = payload.get("data", {})
        if not data:
            return None

        event_time_ms = data.get("E")
        trade_time_ms = data.get("T", event_time_ms)
        if event_time_ms is None or trade_time_ms is None:
            return None

        return {
            "trade_id": int(data["t"]),
            "symbol": str(data["s"]),
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "trade_time": self._format_timestamp(trade_time_ms),
            "event_time": self._format_timestamp(event_time_ms),
            "is_buyer_maker": bool(data["m"]),
        }

    def _handle_message(self, ws_app, message):
        try:
            payload = json.loads(message)
            trade = self._normalize_trade(payload)
            if trade is None:
                return

            self.producer.send(self.topic, value=trade)
            self.sent_messages += 1

            if self.sent_messages % 25 == 0:
                self.producer.flush()

            print(f"Produced crypto trade: {trade}", flush=True)

            if self.max_messages and self.sent_messages >= self.max_messages:
                self.producer.flush()
                print(
                    f"Reached max test messages ({self.max_messages}). Closing websocket.",
                    flush=True,
                )
                ws_app.close()
        except Exception as exc:
            print(f"Failed to process websocket message: {exc}", flush=True)

    @staticmethod
    def _handle_error(_ws_app, error):
        print(f"Binance websocket error: {error}", flush=True)

    @staticmethod
    def _handle_close(_ws_app, status_code, message):
        print(
            f"Binance websocket closed (status={status_code}, message={message})",
            flush=True,
        )

    def _handle_open(self, _ws_app):
        joined_symbols = ", ".join(symbol.upper() for symbol in self.symbols)
        print(
            f"Connected to Binance trade stream for: {joined_symbols}",
            flush=True,
        )

    def stream_forever(self):
        while True:
            try:
                print(f"Connecting to Binance websocket: {self.ws_url}", flush=True)
                ws_app = WebSocketApp(
                    self.ws_url,
                    on_open=self._handle_open,
                    on_message=self._handle_message,
                    on_error=self._handle_error,
                    on_close=self._handle_close,
                )
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except KeyboardInterrupt:
                print("Crypto producer interrupted, shutting down.", flush=True)
                break
            except Exception as exc:
                print(f"Unexpected websocket failure: {exc}", flush=True)

            if self.max_messages and self.sent_messages >= self.max_messages:
                break

            print("Reconnecting to Binance in 5 seconds...", flush=True)
            time.sleep(5)

        self.producer.flush()
        self.producer.close()


if __name__ == "__main__":
    brokers = os.getenv("KAFKA_BROKERS", "broker:29092").split(",")
    topic = os.getenv("KAFKA_TOPIC", "crypto_trades")
    symbols = os.getenv("CRYPTO_SYMBOLS", "btcusdt,ethusdt,solusdt,dogeusdt").split(",")
    max_messages = os.getenv("CRYPTO_MAX_MESSAGES")
    max_messages = int(max_messages) if max_messages else None

    producer = CryptoTradeProducer(
        brokers=brokers,
        topic=topic,
        symbols=symbols,
        max_messages=max_messages,
    )
    producer.stream_forever()
