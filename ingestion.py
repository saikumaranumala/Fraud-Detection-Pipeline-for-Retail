"""
src/data/ingestion.py
---------------------
Transaction data ingestion, schema validation, and preprocessing.
Supports batch CSV loading and real-time Kafka stream consumption.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "transaction_id", "user_id", "merchant_id", "amount",
    "timestamp", "merchant_category_code", "device_type",
    "ip_address", "billing_country", "shipping_country",
    "card_present", "review_text", "transaction_note",
]

DTYPE_MAP = {
    "transaction_id": str,
    "user_id":        str,
    "merchant_id":    str,
    "amount":         float,
    "merchant_category_code": str,
    "device_type":    str,
    "ip_address":     str,
    "billing_country":  str,
    "shipping_country": str,
    "card_present":   bool,
    "review_text":    str,
    "transaction_note": str,
}


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

class TransactionDataLoader:
    """Load and validate transaction data from CSV / Parquet files."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        if self.data_path.suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path, low_memory=False)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        df = self._validate_schema(df)
        df = self._parse_types(df)
        df = self._clean(df)
        logger.info(f"Loaded {len(df):,} rows | {df['is_fraud'].sum():,} fraud cases "
                    f"({df['is_fraud'].mean()*100:.2f}%)")
        return df

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _parse_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col, dtype in DTYPE_MAP.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill missing text fields
        df["review_text"]      = df["review_text"].fillna("")
        df["transaction_note"] = df["transaction_note"].fillna("")
        # Remove negative amounts (data errors)
        df = df[df["amount"] > 0].copy()
        # Deduplicate on transaction_id
        df = df.drop_duplicates(subset=["transaction_id"])
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real-time Kafka stream ingestion
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    bootstrap_servers: str = "localhost:9092"
    topic: str             = "transactions"
    group_id: str          = "fraud-detection-consumer"
    auto_offset_reset: str = "latest"


class TransactionStreamConsumer:
    """
    Consume real-time transaction events from Kafka.
    Each message is a JSON-encoded transaction dict.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.consumer = KafkaConsumer(
            self.config.topic,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.group_id,
            auto_offset_reset=self.config.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

    def stream(self):
        """Yield parsed transaction dicts from Kafka indefinitely."""
        logger.info(f"Listening on Kafka topic: {self.config.topic}")
        for message in self.consumer:
            txn = message.value
            txn["timestamp"] = pd.Timestamp(txn["timestamp"])
            yield txn

    def close(self):
        self.consumer.close()


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_transactions(
    n_samples: int = 10_000,
    fraud_rate: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset with realistic fraud patterns.
    Fraud rate default mirrors real e-commerce (~0.5%).
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def _make_block(n, is_fraud):
        amounts = (
            rng.exponential(scale=2000, size=n) if is_fraud
            else rng.lognormal(mean=4.5, sigma=1.2, size=n)
        )
        notes = (
            rng.choice(
                ["urgent transfer needed", "didn't authorise this",
                 "never received item", "charge me immediately"],
                size=n,
            ) if is_fraud
            else rng.choice(
                ["weekly grocery run", "subscription renewal",
                 "gift for friend", ""],
                size=n,
            )
        )
        return pd.DataFrame({
            "transaction_id": [f"TXN_{rng.integers(1e9)}" for _ in range(n)],
            "user_id":        [f"U{rng.integers(10000):05d}" for _ in range(n)],
            "merchant_id":    [f"M{rng.integers(5000):04d}" for _ in range(n)],
            "amount":         np.round(amounts, 2),
            "timestamp":      pd.date_range("2024-01-01", periods=n, freq="1min"),
            "merchant_category_code": rng.choice(
                ["5411", "5812", "7372", "4111", "5999"], size=n
            ),
            "device_type":    rng.choice(["mobile", "desktop", "tablet"], size=n),
            "ip_address":     [f"{rng.integers(1,255)}.{rng.integers(0,255)}.x.x" for _ in range(n)],
            "billing_country":  rng.choice(["US", "GB", "IN", "NG", "RU"], size=n),
            "shipping_country": rng.choice(["US", "GB", "IN", "NG", "RU"], size=n),
            "card_present":   rng.choice([True, False], size=n, p=[0.3, 0.7] if is_fraud else [0.7, 0.3]),
            "review_text":    rng.choice(
                ["poor service", "never delivered", ""] if is_fraud
                else ["great product", "fast shipping", ""], size=n
            ),
            "transaction_note": notes,
            "is_fraud":       [int(is_fraud)] * n,
        })

    df = pd.concat([_make_block(n_legit, False), _make_block(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_synthetic_transactions(n_samples=50_000)
    df.to_parquet("data/raw/transactions.parquet", index=False)
    logger.info(f"Saved {len(df):,} synthetic transactions")
