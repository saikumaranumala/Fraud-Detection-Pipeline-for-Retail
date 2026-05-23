"""
src/models/autoencoder.py
--------------------------
Autoencoder-based unsupervised anomaly detection.

Trained ONLY on legitimate transactions. At inference time, fraudulent
transactions show abnormally high reconstruction error because the encoder
has never seen those patterns.

Architecture:
    Input → [64] → [32] → [16] → [32] → [64] → Input
    Loss: MSE
    Anomaly threshold: 95th percentile of clean validation reconstruction error
"""

import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from src.features.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "fraud-detection-autoencoder"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim: int, latent_dim: int = 16) -> keras.Model:
    """
    Symmetric encoder-decoder with L2 regularization on weights.
    Batch normalization stabilizes training on highly skewed transaction data.
    """
    reg = regularizers.l2(1e-4)

    # Encoder
    inputs  = keras.Input(shape=(input_dim,), name="input")
    x       = layers.Dense(64, activation="relu", kernel_regularizer=reg)(inputs)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.2)(x)
    x       = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x       = layers.BatchNormalization()(x)
    encoded = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x       = layers.Dense(32, activation="relu", kernel_regularizer=reg)(encoded)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x       = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation="linear", name="output")(x)

    model = keras.Model(inputs, decoded, name="fraud_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AutoencoderAnomalyDetector:
    """
    Trains autoencoder on clean (non-fraud) data.
    Anomaly score = per-sample mean squared reconstruction error.
    Threshold = 95th percentile of clean validation error.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
    ):
        self.latent_dim            = latent_dim
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.threshold_percentile  = threshold_percentile
        self.scaler_               = StandardScaler()
        self.model_                = None
        self.threshold_            = None

    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        X_pred = self.model_.predict(X_scaled, verbose=0)
        return np.mean(np.square(X_scaled - X_pred), axis=1)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        X_train / X_val should be full DataFrames; y_train used to filter
        only clean samples for autoencoder training.
        """
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="autoencoder_anomaly"):

            # Train only on legitimate transactions
            X_clean = X_train.loc[y_train == 0, FEATURE_COLUMNS]
            logger.info(f"Autoencoder training on {len(X_clean):,} clean transactions")

            X_clean_sc = self.scaler_.fit_transform(X_clean)
            input_dim  = X_clean_sc.shape[1]

            self.model_ = build_autoencoder(input_dim, self.latent_dim)
            self.model_.summary(print_fn=logger.info)

            # Callbacks
            cb = [
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
            ]

            history = self.model_.fit(
                X_clean_sc, X_clean_sc,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=cb,
                verbose=1,
            )

            # Set anomaly threshold on clean validation data
            val_errors = self._reconstruction_error(
                self.scaler_.transform(X_clean)
            )
            self.threshold_ = np.percentile(val_errors, self.threshold_percentile)
            logger.info(f"Anomaly threshold ({self.threshold_percentile}th pct): "
                        f"{self.threshold_:.6f}")

            # MLflow logging
            mlflow.log_params({
                "latent_dim":   self.latent_dim,
                "epochs":       self.epochs,
                "batch_size":   self.batch_size,
                "threshold_pct": self.threshold_percentile,
            })
            mlflow.log_metric("anomaly_threshold", float(self.threshold_))
            mlflow.log_metric("min_val_loss", min(history.history["val_loss"]))

            # Evaluate on labeled val set if provided
            if X_val is not None and y_val is not None:
                scores = self.anomaly_score(X_val)
                auc    = roc_auc_score(y_val, scores)
                recall_at_threshold = (
                    (scores[y_val == 1] >= self.threshold_).mean()
                )
                mlflow.log_metric("val_auc_roc", auc)
                mlflow.log_metric("val_recall_at_threshold", recall_at_threshold)
                logger.info(f"Val AUC-ROC: {auc:.4f} | "
                            f"Recall at threshold: {recall_at_threshold:.4f}")

            mlflow.keras.log_model(self.model_, artifact_path="autoencoder")

        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-sample reconstruction error (higher = more anomalous)."""
        X_sc = self.scaler_.transform(X[FEATURE_COLUMNS])
        return self._reconstruction_error(X_sc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary flag: 1 = anomalous (reconstruction error > threshold)."""
        return (self.anomaly_score(X) >= self.threshold_).astype(int)

    def normalized_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Reconstruction error normalized to [0, 1] relative to threshold.
        Useful for ensemble fusion — values > 0.5 are above threshold.
        """
        raw   = self.anomaly_score(X)
        norm  = raw / (2 * self.threshold_ + 1e-9)
        return np.clip(norm, 0, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data.ingestion import generate_synthetic_transactions
    from src.features.feature_engineering import build_feature_pipeline

    df = generate_synthetic_transactions(n_samples=50_000, fraud_rate=0.005)
    y  = df.pop("is_fraud")

    pipe   = build_feature_pipeline()
    df_eng = pipe.fit_transform(df, y)

    X_train, X_val, y_train, y_val = train_test_split(
        df_eng, y, test_size=0.2, stratify=y, random_state=42
    )

    detector = AutoencoderAnomalyDetector(epochs=30, batch_size=128)
    detector.fit(X_train, y_train, X_val, y_val)

    scores = detector.anomaly_score(X_val)
    print(f"\nFraud mean score:   {scores[y_val==1].mean():.4f}")
    print(f"Legit mean score:   {scores[y_val==0].mean():.4f}")
    print(f"Threshold:          {detector.threshold_:.4f}")
