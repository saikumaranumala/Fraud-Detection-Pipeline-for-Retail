"""
src/models/xgboost_model.py
-----------------------------
XGBoost fraud classifier with:
  - Optuna hyperparameter optimization
  - MLflow experiment tracking
  - SHAP explainability
  - SMOTE for class imbalance handling
"""

import logging
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import shap
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    f1_score, recall_score, precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.features.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "fraud-detection-xgboost"
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba  = model.predict_proba(X)[:, 1]
    preds  = (proba >= threshold).astype(int)
    return {
        "auc_roc":          roc_auc_score(y, proba),
        "avg_precision":    average_precision_score(y, proba),
        "recall":           recall_score(y, preds),
        "precision":        precision_score(y, preds, zero_division=0),
        "f1":               f1_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _optuna_objective(trial, X_train, y_train, cv_folds: int = 3) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 50, 300),
        "tree_method":       "hist",
        "eval_metric":       "aucpr",
        "use_label_encoder": False,
        "random_state":      42,
    }

    model = XGBClassifier(**params)
    skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="average_precision")
    return scores.mean()


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class XGBoostFraudTrainer:
    """
    Trains an XGBoost fraud classifier with optional Optuna HPO.
    All runs are tracked in MLflow.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 3,
        use_smote: bool = True,
        decision_threshold: float = 0.35,   # lower than 0.5 to boost recall
    ):
        self.n_trials           = n_trials
        self.cv_folds           = cv_folds
        self.use_smote          = use_smote
        self.decision_threshold = decision_threshold
        self.model_             = None
        self.best_params_       = None
        self.explainer_         = None

    def _apply_smote(self, X, y):
        logger.info(f"Before SMOTE: {y.value_counts().to_dict()}")
        sm = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res

    def _tune(self, X_train, y_train) -> dict:
        logger.info(f"Running Optuna HPO: {self.n_trials} trials ...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(
            lambda trial: _optuna_objective(trial, X_train, y_train, self.cv_folds),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        logger.info(f"Best trial AUC-PR: {study.best_value:.4f}")
        return study.best_params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):

        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="xgboost_fraud_classifier"):

            # ── Class imbalance ──────────────────────────────────────────
            X_fit, y_fit = (
                self._apply_smote(X_train[FEATURE_COLUMNS], y_train)
                if self.use_smote
                else (X_train[FEATURE_COLUMNS], y_train)
            )

            # ── Hyperparameter tuning ────────────────────────────────────
            self.best_params_ = self._tune(X_fit, y_fit)
            mlflow.log_params(self.best_params_)
            mlflow.log_param("decision_threshold", self.decision_threshold)

            # ── Final training ───────────────────────────────────────────
            self.model_ = XGBClassifier(
                **self.best_params_,
                tree_method="hist",
                eval_metric="aucpr",
                use_label_encoder=False,
                random_state=42,
            )
            eval_set = [(X_val[FEATURE_COLUMNS], y_val)] if X_val is not None else None
            self.model_.fit(
                X_fit, y_fit,
                eval_set=eval_set,
                verbose=100,
            )

            # ── Evaluation ───────────────────────────────────────────────
            train_metrics = evaluate(self.model_, X_train[FEATURE_COLUMNS], y_train,
                                      self.decision_threshold)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()
                                 if not isinstance(v, list)})

            if X_val is not None:
                val_metrics = evaluate(self.model_, X_val[FEATURE_COLUMNS], y_val,
                                        self.decision_threshold)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()
                                     if not isinstance(v, list)})
                logger.info(f"Validation → Recall: {val_metrics['recall']:.3f} | "
                            f"F1: {val_metrics['f1']:.3f} | "
                            f"AUC-ROC: {val_metrics['auc_roc']:.3f}")

            # ── SHAP explainer ───────────────────────────────────────────
            self.explainer_ = shap.TreeExplainer(self.model_)

            # ── Log model ────────────────────────────────────────────────
            mlflow.xgboost.log_model(self.model_, artifact_path="xgboost_model")
            logger.info("Model logged to MLflow.")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.decision_threshold).astype(int)

    def explain(self, X: pd.DataFrame, max_display: int = 15):
        """Return SHAP values for a batch — used for per-alert explainability."""
        return self.explainer_.shap_values(X[FEATURE_COLUMNS])

    def explain_single(self, row: pd.DataFrame) -> dict:
        """Return top contributing features for a single transaction."""
        shap_vals = self.explainer_.shap_values(row[FEATURE_COLUMNS])
        feature_impact = dict(zip(FEATURE_COLUMNS, shap_vals[0]))
        sorted_impact  = dict(sorted(feature_impact.items(),
                                     key=lambda x: abs(x[1]), reverse=True))
        return sorted_impact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data.ingestion import generate_synthetic_transactions
    from src.features.feature_engineering import build_feature_pipeline

    df = generate_synthetic_transactions(n_samples=50_000)
    y  = df.pop("is_fraud")

    pipe   = build_feature_pipeline()
    df_eng = pipe.fit_transform(df, y)

    X_train, X_val, y_train, y_val = train_test_split(
        df_eng, y, test_size=0.2, stratify=y, random_state=42
    )

    trainer = XGBoostFraudTrainer(n_trials=30)
    trainer.fit(X_train, y_train, X_val, y_val)
