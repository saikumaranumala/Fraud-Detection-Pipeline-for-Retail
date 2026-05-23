"""
tests/test_features.py
-----------------------
Unit tests for all feature engineering components.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

from src.data.ingestion import generate_synthetic_transactions
from src.features.feature_engineering import (
    VelocityFeatures,
    UserBaselineFeatures,
    MerchantRiskFeatures,
    GeoDeviceFeatures,
    TemporalFeatures,
    build_feature_pipeline,
    FEATURE_COLUMNS,
)
from src.features.nlp_features import (
    NLPFeatureExtractor,
    preprocess_text,
    KeywordRiskScorer,
    VADERSentimentScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_data():
    df = generate_synthetic_transactions(n_samples=500, seed=42)
    y  = df.pop("is_fraud")
    return df, y


@pytest.fixture(scope="module")
def engineered_data(sample_data):
    df, y = sample_data
    pipe  = build_feature_pipeline()
    df_e  = pipe.fit_transform(df.copy(), y)
    return df_e, y


# ---------------------------------------------------------------------------
# Data ingestion tests
# ---------------------------------------------------------------------------

class TestDataIngestion:
    def test_generates_correct_shape(self):
        df = generate_synthetic_transactions(n_samples=100, fraud_rate=0.1)
        assert len(df) == 100

    def test_fraud_rate_approximate(self):
        df = generate_synthetic_transactions(n_samples=10_000, fraud_rate=0.01, seed=0)
        assert 0.005 < df["is_fraud"].mean() < 0.02, "Fraud rate out of expected range"

    def test_no_negative_amounts(self):
        df = generate_synthetic_transactions(n_samples=500)
        assert (df["amount"] > 0).all()

    def test_required_columns_present(self):
        required = ["transaction_id", "user_id", "amount", "timestamp", "is_fraud"]
        df = generate_synthetic_transactions(n_samples=100)
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Velocity feature tests
# ---------------------------------------------------------------------------

class TestVelocityFeatures:
    def test_output_columns_created(self, sample_data):
        df, _ = sample_data
        vf = VelocityFeatures()
        out = vf.fit_transform(df.copy())
        for window in ["1h", "6h", "24h", "7d"]:
            assert f"txn_count_{window}"  in out.columns
            assert f"txn_amount_{window}" in out.columns

    def test_counts_non_negative(self, sample_data):
        df, _ = sample_data
        vf = VelocityFeatures()
        out = vf.fit_transform(df.copy())
        assert (out["txn_count_1h"] >= 0).all()
        assert (out["txn_count_24h"] >= 0).all()


# ---------------------------------------------------------------------------
# User baseline tests
# ---------------------------------------------------------------------------

class TestUserBaselineFeatures:
    def test_zscore_column_created(self, sample_data):
        df, y = sample_data
        ubf = UserBaselineFeatures()
        out = ubf.fit_transform(df.copy())
        assert "amount_zscore" in out.columns
        assert "amount_pct_of_mean" in out.columns

    def test_zscore_reasonable_range(self, sample_data):
        df, y = sample_data
        ubf = UserBaselineFeatures()
        out = ubf.fit_transform(df.copy())
        # Most z-scores should be within ±10 for typical data
        assert out["amount_zscore"].abs().quantile(0.99) < 20

    def test_no_nan_after_transform(self, sample_data):
        df, y = sample_data
        ubf = UserBaselineFeatures()
        out = ubf.fit_transform(df.copy())
        assert not out["amount_zscore"].isna().any()


# ---------------------------------------------------------------------------
# Merchant risk tests
# ---------------------------------------------------------------------------

class TestMerchantRiskFeatures:
    def test_fraud_rates_between_0_and_1(self, sample_data):
        df, y = sample_data
        mrf = MerchantRiskFeatures()
        out = mrf.fit(df, y).transform(df)
        assert out["merchant_fraud_rate"].between(0, 1).all()
        assert out["mcc_fraud_rate"].between(0, 1).all()

    def test_global_rate_fallback(self, sample_data):
        df, y = sample_data
        mrf = MerchantRiskFeatures()
        mrf.fit(df, y)
        # Unseen merchant should get global rate
        unseen = df.copy()
        unseen["merchant_id"] = "UNKNOWN_MERCHANT_XYZ"
        out = mrf.transform(unseen)
        assert (out["merchant_fraud_rate"] == pytest.approx(mrf.global_rate_)).all()


# ---------------------------------------------------------------------------
# Geo/device tests
# ---------------------------------------------------------------------------

class TestGeoDeviceFeatures:
    def test_binary_columns_are_binary(self, sample_data):
        df, _ = sample_data
        gdf = GeoDeviceFeatures()
        out = gdf.fit_transform(df)
        for col in ["country_mismatch", "high_risk_country", "card_not_present"]:
            assert out[col].isin([0, 1]).all(), f"{col} is not binary"

    def test_cross_border_cnp_subset_of_cnp(self, sample_data):
        df, _ = sample_data
        gdf = GeoDeviceFeatures()
        out = gdf.fit_transform(df)
        # cross_border_cnp can only be 1 if card_not_present is 1
        assert (out["cross_border_cnp"] <= out["card_not_present"]).all()


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_all_feature_columns_present(self, engineered_data):
        df_e, _ = engineered_data
        for col in FEATURE_COLUMNS:
            assert col in df_e.columns, f"Missing feature: {col}"

    def test_no_inf_values(self, engineered_data):
        df_e, _ = engineered_data
        numeric = df_e[FEATURE_COLUMNS].select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf values found in features"

    def test_no_all_nan_columns(self, engineered_data):
        df_e, _ = engineered_data
        for col in FEATURE_COLUMNS:
            assert not df_e[col].isna().all(), f"Column {col} is all NaN"


# ---------------------------------------------------------------------------
# NLP feature tests
# ---------------------------------------------------------------------------

class TestNLPFeatures:
    def test_preprocess_text_lowercases(self):
        assert preprocess_text("URGENT Transfer NOW!") == "urgent transfer now"

    def test_preprocess_removes_punctuation(self):
        result = preprocess_text("hello, world! test.")
        assert "," not in result and "!" not in result

    def test_keyword_scorer_flags_fraud_text(self):
        scr = KeywordRiskScorer()
        fraud_texts = pd.Series(["urgent transfer needed immediately"])
        legit_texts = pd.Series(["weekly grocery run"])
        assert scr.transform(fraud_texts)[0] > scr.transform(legit_texts)[0]

    def test_vader_negative_text_higher_risk(self):
        scr = VADERSentimentScorer()
        negative = pd.Series(["I never received this item, this is fraud"])
        positive = pd.Series(["great product, fast delivery, very happy"])
        assert scr.transform(negative)[0] > scr.transform(positive)[0]

    def test_nlp_extractor_fit_transform(self, sample_data):
        df, y = sample_data
        ext = NLPFeatureExtractor()
        out = ext.fit_transform(df, y)
        assert "nlp_risk_score" in out.columns
        assert out["nlp_risk_score"].between(0, 1).all()

    def test_nlp_scores_higher_for_fraud(self, sample_data):
        df, y = sample_data
        ext = NLPFeatureExtractor()
        out = ext.fit_transform(df, y)
        out["is_fraud"] = y.values
        fraud_mean = out[out["is_fraud"]==1]["nlp_risk_score"].mean()
        legit_mean = out[out["is_fraud"]==0]["nlp_risk_score"].mean()
        assert fraud_mean > legit_mean, "NLP score should be higher for fraud"
