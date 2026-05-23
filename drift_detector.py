"""
src/monitoring/drift_detector.py
----------------------------------
Data drift and model performance monitoring using Evidently AI.
Runs as a scheduled job (daily via Airflow) and triggers retraining
when drift score exceeds configured thresholds.
"""

import logging
import json
import smtplib
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
)

from src.features.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Column mapping for Evidently
# ---------------------------------------------------------------------------

COLUMN_MAPPING = ColumnMapping(
    target="is_fraud",
    prediction="is_fraud_pred",
    numerical_features=[
        "amount", "amount_zscore", "txn_count_1h", "txn_count_24h",
        "txn_amount_1h", "merchant_fraud_rate", "mcc_fraud_rate",
        "hour_deviation",
    ],
    categorical_features=[
        "device_type", "billing_country", "card_present",
        "country_mismatch", "is_weekend", "is_night",
    ],
)


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class FraudDriftDetector:
    """
    Compares a current production data window against a reference baseline.

    Three types of drift monitored:
      1. Feature drift  – input distribution shift (PSI / KS test)
      2. Target drift   – fraud rate shift in production
      3. Model drift    – degradation in recall/precision on labeled samples
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        fraud_rate_threshold: float = 0.015,  # alert if fraud rate > 1.5%
        recall_drop_threshold: float = 0.05,   # alert if recall drops > 5%
        report_dir: Path = REPORT_DIR,
    ):
        self.psi_threshold         = psi_threshold
        self.fraud_rate_threshold  = fraud_rate_threshold
        self.recall_drop_threshold = recall_drop_threshold
        self.report_dir            = report_dir

    def detect_feature_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        save_report: bool = True,
    ) -> dict:
        """
        Run Evidently DataDriftPreset on reference vs current.
        Returns drift summary and per-feature drift flags.
        """
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataQualityPreset(),
            *[ColumnDriftMetric(column_name=col)
              for col in FEATURE_COLUMNS if col in reference.columns],
        ])

        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=COLUMN_MAPPING,
        )

        result = report.as_dict()
        drift_summary = self._parse_drift_result(result)

        if save_report:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.report_dir / f"drift_report_{ts}.html"
            report.save_html(str(path))
            logger.info(f"Drift report saved to {path}")

        return drift_summary

    def _parse_drift_result(self, result: dict) -> dict:
        """Extract key drift metrics from Evidently result dict."""
        metrics = result.get("metrics", [])
        summary = {
            "dataset_drift_detected": False,
            "drift_share":            0.0,
            "drifted_features":       [],
            "stable_features":        [],
        }

        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                res = metric.get("result", {})
                summary["dataset_drift_detected"] = res.get("dataset_drift", False)
                summary["drift_share"]            = res.get("share_of_drifted_columns", 0.0)
                summary["n_drifted"]              = res.get("number_of_drifted_columns", 0)

            if metric.get("metric") == "ColumnDriftMetric":
                res  = metric.get("result", {})
                col  = res.get("column_name", "")
                stat = res.get("stattest_name", "")
                p    = res.get("p_value", 1.0)
                if res.get("drift_detected", False):
                    summary["drifted_features"].append({"feature": col, "p_value": p, "test": stat})
                else:
                    summary["stable_features"].append(col)

        return summary

    def detect_target_drift(self, current: pd.DataFrame) -> dict:
        """
        Check if the fraud rate in current production window has shifted
        significantly from the expected baseline.
        """
        if "is_fraud" not in current.columns:
            return {"target_drift_detected": False, "reason": "no labels available"}

        current_rate = current["is_fraud"].mean()
        is_drift     = current_rate > self.fraud_rate_threshold

        return {
            "target_drift_detected": is_drift,
            "current_fraud_rate":    round(current_rate, 6),
            "threshold":             self.fraud_rate_threshold,
            "recommendation":        "retrain" if is_drift else "monitor",
        }

    def detect_model_drift(
        self,
        reference_recall: float,
        current: pd.DataFrame,
        model,
    ) -> dict:
        """
        Compare current model recall on labeled production samples
        against the training baseline recall.
        """
        from sklearn.metrics import recall_score
        if "is_fraud" not in current.columns:
            return {"model_drift_detected": False, "reason": "no labels"}

        preds = model.predict(current)
        curr_recall = recall_score(current["is_fraud"], preds)
        drop = reference_recall - curr_recall

        return {
            "model_drift_detected": drop > self.recall_drop_threshold,
            "reference_recall":     round(reference_recall, 4),
            "current_recall":       round(curr_recall, 4),
            "recall_drop":          round(drop, 4),
            "recommendation":       "retrain" if drop > self.recall_drop_threshold else "ok",
        }

    def full_drift_check(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        model=None,
        reference_recall: float = 0.95,
    ) -> dict:
        """Run all drift checks and return consolidated report."""
        logger.info("Running full drift check ...")

        feature_drift = self.detect_feature_drift(reference, current)
        target_drift  = self.detect_target_drift(current)
        model_drift   = (
            self.detect_model_drift(reference_recall, current, model)
            if model else {"model_drift_detected": False}
        )

        any_drift = (
            feature_drift["dataset_drift_detected"]
            or target_drift["target_drift_detected"]
            or model_drift["model_drift_detected"]
        )

        report = {
            "timestamp":       datetime.now().isoformat(),
            "drift_detected":  any_drift,
            "retraining_recommended": any_drift,
            "feature_drift":   feature_drift,
            "target_drift":    target_drift,
            "model_drift":     model_drift,
        }

        # Save summary JSON
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.report_dir / f"drift_summary_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Drift check complete | Drift detected: {any_drift} | "
                    f"Retraining recommended: {any_drift}")

        if any_drift:
            self._trigger_alert(report)

        return report

    def _trigger_alert(self, report: dict):
        """Log alert — in production this triggers Airflow retraining DAG via API."""
        logger.warning(
            f"DRIFT ALERT | Feature drift: {report['feature_drift']['dataset_drift_detected']} | "
            f"Target drift: {report['target_drift']['target_drift_detected']} | "
            f"Model drift: {report['model_drift']['model_drift_detected']}"
        )
        # In production: POST to Airflow REST API to trigger retraining DAG
        # requests.post(f"{AIRFLOW_URL}/api/v1/dags/fraud_retraining/dagRuns", ...)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.ingestion import generate_synthetic_transactions
    from src.features.feature_engineering import build_feature_pipeline

    # Simulate reference (training distribution)
    ref_raw = generate_synthetic_transactions(n_samples=5000, seed=0)
    y_ref   = ref_raw.pop("is_fraud")
    pipe    = build_feature_pipeline()
    ref_eng = pipe.fit_transform(ref_raw, y_ref)
    ref_eng["is_fraud"] = y_ref.values

    # Simulate drifted current data (higher fraud rate, different amounts)
    cur_raw = generate_synthetic_transactions(n_samples=1000, fraud_rate=0.04, seed=99)
    y_cur   = cur_raw.pop("is_fraud")
    cur_eng = pipe.transform(cur_raw)
    cur_eng["is_fraud"] = y_cur.values

    detector = FraudDriftDetector()
    report   = detector.full_drift_check(ref_eng, cur_eng)
    print(json.dumps(report, indent=2, default=str))
