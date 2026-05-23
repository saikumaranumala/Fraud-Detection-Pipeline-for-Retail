"""
src/features/nlp_features.py
------------------------------
NLP pipeline that extracts fraud-signal features from:
  - transaction_note  : free-text description typed by user/merchant
  - review_text       : customer review tied to the merchant

Two complementary signals:
  1. TF-IDF + Logistic Regression classifier  → P(fraud | text)
  2. VADER sentiment score                    → negative language flag

Combined output is a single `nlp_risk_score` in [0, 1].
"""

import re
import logging
import numpy as np
import pandas as pd
import nltk
import spacy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

nltk.download("vader_lexicon", quiet=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent keywords — domain-specific fraud language
# ---------------------------------------------------------------------------

FRAUD_INTENT_KEYWORDS = [
    "urgent", "immediately", "unauthorized", "didn't authorise",
    "never received", "not my transaction", "dispute", "chargeback",
    "stolen", "fraud", "blocked", "suspicious", "unrecognized",
]

PRESSURE_KEYWORDS = [
    "asap", "right now", "emergency", "critical", "time sensitive",
    "do it now", "hurry", "quick", "fast transfer",
]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """Combine review_text and transaction_note into a single field."""
    return (
        df["transaction_note"].fillna("") + " " + df["review_text"].fillna("")
    ).apply(preprocess_text)


# ---------------------------------------------------------------------------
# Keyword signal extractor (rule-based, no training needed)
# ---------------------------------------------------------------------------

class KeywordRiskScorer(BaseEstimator, TransformerMixin):
    """
    Count fraud-intent and pressure keywords in text.
    Returns a normalized score in [0, 1].
    """

    def fit(self, X, y=None):
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        scores = []
        for text in texts:
            words = set(text.lower().split())
            fraud_hits    = sum(1 for kw in FRAUD_INTENT_KEYWORDS if kw in text)
            pressure_hits = sum(1 for kw in PRESSURE_KEYWORDS    if kw in text)
            raw = fraud_hits * 2 + pressure_hits     # weight fraud intent higher
            scores.append(min(raw / 6.0, 1.0))       # cap at 1.0
        return np.array(scores)


# ---------------------------------------------------------------------------
# VADER sentiment scorer
# ---------------------------------------------------------------------------

class VADERSentimentScorer(BaseEstimator, TransformerMixin):
    """
    Uses VADER lexicon to score text sentiment.
    Highly negative text → higher fraud risk signal.
    Returns risk score: 1 - (compound + 1) / 2  so negative = high risk.
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        scores = []
        for text in texts:
            compound = self.sia.polarity_scores(text)["compound"]
            # compound in [-1, 1]: -1 = very negative
            risk = (1 - compound) / 2      # maps to [0, 1], negative → high
            scores.append(risk)
        return np.array(scores)


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression classifier
# ---------------------------------------------------------------------------

class TFIDFFraudClassifier(BaseEstimator, TransformerMixin):
    """
    Supervised classifier trained on labeled transaction text.
    Outputs P(fraud | text) for each record.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2), C: float = 1.0):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.C            = C
        self.pipeline_    = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                C=self.C,
                class_weight="balanced",
                max_iter=500,
                solver="lbfgs",
            )),
        ])

    def fit(self, texts: pd.Series, y: pd.Series):
        self.pipeline_.fit(texts, y)
        logger.info("TF-IDF classifier trained | "
                    f"vocab size: {len(self.pipeline_['tfidf'].vocabulary_):,}")
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        return self.pipeline_.predict_proba(texts)[:, 1]   # P(fraud)

    def predict(self, texts: pd.Series) -> np.ndarray:
        return self.pipeline_.predict(texts)


# ---------------------------------------------------------------------------
# Master NLP feature extractor
# ---------------------------------------------------------------------------

class NLPFeatureExtractor:
    """
    Orchestrates all NLP signals and combines them into a single
    `nlp_risk_score` per transaction.

    Fusion weights (tuned on validation set):
        40% TF-IDF classifier
        35% keyword rule scorer
        25% VADER sentiment
    """

    WEIGHTS = {"tfidf": 0.40, "keyword": 0.35, "sentiment": 0.25}

    def __init__(self):
        self.tfidf_clf    = TFIDFFraudClassifier()
        self.keyword_scr  = KeywordRiskScorer()
        self.vader_scr    = VADERSentimentScorer()
        self._fitted      = False

    def fit(self, df: pd.DataFrame, y: pd.Series):
        texts = combine_text_fields(df)
        self.tfidf_clf.fit(texts, y)
        # Keyword and VADER are unsupervised — no fitting needed
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")

        texts = combine_text_fields(df)
        result = df.copy()

        tfidf_scores    = self.tfidf_clf.transform(texts)
        keyword_scores  = self.keyword_scr.transform(texts)
        sentiment_scores = self.vader_scr.transform(texts)

        result["nlp_tfidf_score"]     = tfidf_scores
        result["nlp_keyword_score"]   = keyword_scores
        result["nlp_sentiment_score"] = sentiment_scores

        # Weighted fusion
        result["nlp_risk_score"] = (
            self.WEIGHTS["tfidf"]     * tfidf_scores
            + self.WEIGHTS["keyword"] * keyword_scores
            + self.WEIGHTS["sentiment"] * sentiment_scores
        )

        return result

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, y).transform(df)


# ---------------------------------------------------------------------------
# Top fraud keywords inspector (for explainability)
# ---------------------------------------------------------------------------

def get_top_fraud_keywords(extractor: NLPFeatureExtractor, n: int = 20) -> pd.DataFrame:
    """Extract top features from the TF-IDF logistic regression model."""
    tfidf = extractor.tfidf_clf.pipeline_["tfidf"]
    clf   = extractor.tfidf_clf.pipeline_["clf"]
    coefs = clf.coef_[0]
    vocab = {v: k for k, v in tfidf.vocabulary_.items()}
    top_idx = np.argsort(coefs)[-n:][::-1]
    return pd.DataFrame({
        "keyword":    [vocab[i] for i in top_idx],
        "coefficient": coefs[top_idx],
    })


if __name__ == "__main__":
    from src.data.ingestion import generate_synthetic_transactions

    df = generate_synthetic_transactions(n_samples=5000, seed=0)
    y  = df["is_fraud"]

    extractor = NLPFeatureExtractor()
    df_out = extractor.fit_transform(df, y)

    print(df_out[["nlp_tfidf_score", "nlp_keyword_score",
                  "nlp_sentiment_score", "nlp_risk_score"]].describe())

    print("\nTop fraud keywords:")
    print(get_top_fraud_keywords(extractor))
