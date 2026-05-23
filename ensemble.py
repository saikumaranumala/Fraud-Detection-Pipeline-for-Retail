"""
src/models/ensemble.py
-----------------------
Fuses three fraud signals into a final score:
  1. XGBoost classifier      (structured features)
  2. Autoencoder anomaly     (unsupervised novelty detection)
  3. NLP risk score          (text sentiment + intent keywords)

Also applies rule-based hard overrides for extreme signals.
"""

import logging
import numpy as np
import pandas as pd
import mlflow

from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    # Soft fusion weights (must sum to 1.0)
    weight_xgboost:    float = 0.50
    weight_autoencoder: float = 0.30
    weight_nlp:        float = 0.20

    # Hard override thresholds — bypass soft fusion
    velocity_1h_override: int   = 15        # > N txns in 1 hour → force flag
    amount_zscore_override: float = 5.0     # > 5 std devs from user mean
    xgboost_hard_threshold: float = 0.90   # very high XGBoost confidence alone

    # Final decision threshold
    decision_threshold: float = 0.50

    def __post_init__(self):
        total = self.weight_xgboost + self.weight_autoencoder + self.weight_nlp
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class FraudEnsemble:
    """
    Combines XGBoost, autoencoder, and NLP signals into a final fraud score.
    Logs all scoring decisions to MLflow for audit trails.
    """

    def __init__(
        self,
        xgboost_model,
        autoencoder_model,
        nlp_extractor,
        config: Optional[EnsembleConfig] = None,
    ):
        self.xgb  = xgboost_model
        self.ae   = autoencoder_model
        self.nlp  = nlp_extractor
        self.cfg  = config or EnsembleConfig()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns input DataFrame with appended scoring columns:
          - xgb_score       : XGBoost P(fraud)
          - ae_score        : Autoencoder normalized anomaly score
          - nlp_score       : NLP risk score
          - soft_score      : Weighted fusion of above
          - override_flag   : 1 if any hard override rule fired
          - fraud_score     : Final score (soft_score or 1.0 if override)
          - is_fraud_pred   : Binary prediction at decision_threshold
        """
        result = df.copy()

        # ── Individual model scores ──────────────────────────────────────
        result["xgb_score"] = self.xgb.predict_proba(df)
        result["ae_score"]  = self.ae.normalized_score(df)

        nlp_df = self.nlp.transform(df)
        result["nlp_score"] = nlp_df["nlp_risk_score"].values

        # ── Soft fusion ──────────────────────────────────────────────────
        result["soft_score"] = (
            self.cfg.weight_xgboost    * result["xgb_score"]
            + self.cfg.weight_autoencoder * result["ae_score"]
            + self.cfg.weight_nlp      * result["nlp_score"]
        )

        # ── Hard override rules ──────────────────────────────────────────
        override = pd.Series(False, index=result.index)

        if "txn_count_1h" in result.columns:
            override |= result["txn_count_1h"] > self.cfg.velocity_1h_override

        if "amount_zscore" in result.columns:
            override |= result["amount_zscore"].abs() > self.cfg.amount_zscore_override

        override |= result["xgb_score"] >= self.cfg.xgboost_hard_threshold

        result["override_flag"] = override.astype(int)

        # ── Final score ──────────────────────────────────────────────────
        result["fraud_score"] = np.where(
            override,
            np.maximum(result["soft_score"], 0.85),   # floor at 0.85 on override
            result["soft_score"],
        )

        result["is_fraud_pred"] = (
            result["fraud_score"] >= self.cfg.decision_threshold
        ).astype(int)

        result["confidence_tier"] = pd.cut(
            result["fraud_score"],
            bins=[0, 0.3, 0.6, 0.85, 1.01],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )

        return result

    def score_single(self, transaction: dict) -> dict:
        """Score a single transaction dict — used by the FastAPI endpoint."""
        df  = pd.DataFrame([transaction])
        out = self.score(df)
        row = out.iloc[0]
        return {
            "transaction_id":  transaction.get("transaction_id", "unknown"),
            "fraud_score":     round(float(row["fraud_score"]), 4),
            "is_fraud":        int(row["is_fraud_pred"]),
            "confidence_tier": str(row["confidence_tier"]),
            "override_fired":  bool(row["override_flag"]),
            "signal_breakdown": {
                "xgboost":    round(float(row["xgb_score"]), 4),
                "autoencoder": round(float(row["ae_score"]), 4),
                "nlp":        round(float(row["nlp_score"]), 4),
            },
        }

    def batch_evaluate(self, df: pd.DataFrame, y_true: pd.Series) -> dict:
        """Full evaluation report on a labeled dataset."""
        from sklearn.metrics import (
            classification_report, roc_auc_score,
            average_precision_score,
        )
        scored = self.score(df)
        preds  = scored["is_fraud_pred"]
        proba  = scored["fraud_score"]

        report = {
            "auc_roc":       roc_auc_score(y_true, proba),
            "avg_precision": average_precision_score(y_true, proba),
            "classification_report": classification_report(y_true, preds),
            "override_rate": scored["override_flag"].mean(),
        }

        logger.info(f"\nEnsemble Evaluation\n"
                    f"  AUC-ROC:  {report['auc_roc']:.4f}\n"
                    f"  Avg-Prec: {report['avg_precision']:.4f}\n"
                    f"  Override rate: {report['override_rate']*100:.1f}%\n"
                    f"{report['classification_report']}")

        return report
