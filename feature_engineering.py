"""
src/features/feature_engineering.py
-------------------------------------
Builds all structured features used by the XGBoost and autoencoder models.

Feature groups:
  1. Velocity signals       – rolling transaction counts/amounts per user
  2. User profile baseline  – deviation from historical spend distribution
  3. Merchant risk          – historical fraud rate by merchant/MCC
  4. Device & geo signals   – country mismatch, card-not-present flag
  5. Temporal patterns      – hour-of-day, day-of-week relative to user norm
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# 1. Velocity features
# ---------------------------------------------------------------------------

class VelocityFeatures(BaseEstimator, TransformerMixin):
    """
    Rolling transaction counts and amounts per user over multiple windows.
    Fraud often shows sudden spikes in transaction frequency.
    """

    WINDOWS = {"1h": "1h", "6h": "6h", "24h": "24h", "7d": "7D"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy().sort_values("timestamp")
        df = df.set_index("timestamp")

        for label, window in self.WINDOWS.items():
            rolling = (
                df.groupby("user_id")["amount"]
                .rolling(window, min_periods=1)
            )
            df[f"txn_count_{label}"]  = rolling.count().reset_index(level=0, drop=True)
            df[f"txn_amount_{label}"] = rolling.sum().reset_index(level=0, drop=True)

        df = df.reset_index()
        return df


# ---------------------------------------------------------------------------
# 2. User baseline / deviation features
# ---------------------------------------------------------------------------

class UserBaselineFeatures(BaseEstimator, TransformerMixin):
    """
    Compare each transaction to the user's historical spend distribution.
    High z-score on amount = unusual for this specific user.
    """

    def fit(self, X: pd.DataFrame, y=None):
        stats = X.groupby("user_id")["amount"].agg(["mean", "std"]).rename(
            columns={"mean": "user_mean_amount", "std": "user_std_amount"}
        )
        stats["user_std_amount"] = stats["user_std_amount"].fillna(1.0).clip(lower=1e-6)
        self.user_stats_ = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.merge(self.user_stats_, on="user_id", how="left")

        # Fill unknown users with global stats
        df["user_mean_amount"] = df["user_mean_amount"].fillna(df["amount"].mean())
        df["user_std_amount"]  = df["user_std_amount"].fillna(df["amount"].std())

        df["amount_zscore"] = (
            (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"]
        )
        df["amount_pct_of_mean"] = df["amount"] / df["user_mean_amount"].clip(lower=1e-6)
        return df


# ---------------------------------------------------------------------------
# 3. Merchant risk features
# ---------------------------------------------------------------------------

class MerchantRiskFeatures(BaseEstimator, TransformerMixin):
    """
    Historical fraud rate per merchant and merchant category code (MCC).
    Smoothed with additive (Laplace) smoothing to handle rare merchants.
    """

    SMOOTHING = 10  # prior transaction count

    def fit(self, X: pd.DataFrame, y: pd.Series):
        global_rate = y.mean()

        def _smoothed_rate(group_col):
            stats = pd.concat([X[group_col], y.rename("is_fraud")], axis=1)
            agg = stats.groupby(group_col)["is_fraud"].agg(["sum", "count"])
            agg["smoothed_rate"] = (
                (agg["sum"] + self.SMOOTHING * global_rate)
                / (agg["count"] + self.SMOOTHING)
            )
            return agg["smoothed_rate"]

        self.merchant_rates_ = _smoothed_rate("merchant_id")
        self.mcc_rates_      = _smoothed_rate("merchant_category_code")
        self.global_rate_    = global_rate
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["merchant_fraud_rate"] = (
            df["merchant_id"].map(self.merchant_rates_).fillna(self.global_rate_)
        )
        df["mcc_fraud_rate"] = (
            df["merchant_category_code"].map(self.mcc_rates_).fillna(self.global_rate_)
        )
        return df


# ---------------------------------------------------------------------------
# 4. Geographic & device features
# ---------------------------------------------------------------------------

class GeoDeviceFeatures(BaseEstimator, TransformerMixin):
    """
    Country mismatch between billing and shipping, card-not-present flag.
    Cross-border transactions with card-not-present = elevated risk.
    """

    HIGH_RISK_COUNTRIES = {"NG", "RU", "KP", "IR"}

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["country_mismatch"] = (
            df["billing_country"] != df["shipping_country"]
        ).astype(int)
        df["high_risk_country"] = (
            df["billing_country"].isin(self.HIGH_RISK_COUNTRIES)
            | df["shipping_country"].isin(self.HIGH_RISK_COUNTRIES)
        ).astype(int)
        df["card_not_present"] = (~df["card_present"]).astype(int)
        df["cross_border_cnp"] = (
            df["country_mismatch"] & df["card_not_present"]
        ).astype(int)
        return df


# ---------------------------------------------------------------------------
# 5. Temporal features
# ---------------------------------------------------------------------------

class TemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Hour of day, day of week, and whether the transaction is outside
    the user's typical active hours (proxy for account takeover).
    """

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df["hour"] = df["timestamp"].dt.hour
        self.user_peak_hours_ = (
            df.groupby("user_id")["hour"]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else 12)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["txn_hour"]    = df["timestamp"].dt.hour
        df["txn_dow"]     = df["timestamp"].dt.dayofweek
        df["is_weekend"]  = (df["txn_dow"] >= 5).astype(int)
        df["is_night"]    = df["txn_hour"].between(0, 5).astype(int)

        peak = df["user_id"].map(self.user_peak_hours_).fillna(12)
        df["hour_deviation"] = (df["txn_hour"] - peak).abs()
        return df


# ---------------------------------------------------------------------------
# Master feature pipeline
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Velocity
    "txn_count_1h", "txn_amount_1h",
    "txn_count_6h", "txn_amount_6h",
    "txn_count_24h", "txn_amount_24h",
    "txn_count_7d", "txn_amount_7d",
    # User baseline
    "amount_zscore", "amount_pct_of_mean",
    # Merchant risk
    "merchant_fraud_rate", "mcc_fraud_rate",
    # Geo / device
    "country_mismatch", "high_risk_country",
    "card_not_present", "cross_border_cnp",
    # Temporal
    "txn_hour", "txn_dow", "is_weekend", "is_night", "hour_deviation",
    # Raw
    "amount",
]


def build_feature_pipeline(y_train: pd.Series = None):
    """
    Returns a configured feature pipeline.
    MerchantRiskFeatures requires y at fit time — pass via fit(X, y).
    """
    steps = [
        ("velocity",  VelocityFeatures()),
        ("user",      UserBaselineFeatures()),
        ("merchant",  MerchantRiskFeatures()),
        ("geo",       GeoDeviceFeatures()),
        ("temporal",  TemporalFeatures()),
    ]
    return Pipeline(steps)


if __name__ == "__main__":
    from src.data.ingestion import generate_synthetic_transactions
    df = generate_synthetic_transactions(n_samples=5000)
    y  = df.pop("is_fraud")

    pipe = build_feature_pipeline()
    df_feats = pipe.fit_transform(df, y)
    print(df_feats[FEATURE_COLUMNS].describe())
