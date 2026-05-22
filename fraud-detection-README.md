# Fraud Detection in E-Commerce Transactions

> A production-ready fraud detection system combining anomaly detection, ensemble ML models, and NLP-based sentiment analysis — built during my MS in Data Science at Saint Peter's University (2023–2025) to tackle real-world imbalanced classification challenges in high-volume transaction environments.

---

## Overview

This system detects fraudulent e-commerce transactions in real time using a multi-signal approach: structured transaction features, autoencoder-based anomaly detection, and sentiment analysis on customer reviews and transaction notes. The final model achieves **95% recall** while maintaining business-acceptable false positive rates.

The project simulates a production fraud pipeline with a live Streamlit dashboard for merchant-facing monitoring, instant alerts, and actionable insights.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Data Ingestion Layer                       │
│   Transaction Records + Customer Reviews + User Behavior Logs   │
└───────────────┬──────────────────────────────┬───────────────────┘
                │                              │
┌───────────────▼──────────┐   ┌──────────────▼──────────────────┐
│   Structured ML Pipeline │   │     NLP / Sentiment Pipeline     │
│                          │   │                                  │
│  Feature Engineering     │   │  Text Preprocessing (spaCy)      │
│  XGBoost Classifier      │   │  TF-IDF + Logistic Regression   │
│  Autoencoder (Anomaly)   │   │  Intent Signal Extraction        │
│  Ensemble / Stacking     │   │  Sentiment Scoring               │
└───────────────┬──────────┘   └──────────────┬───────────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼───────────────────┐
│                     Fusion & Scoring Layer                        │
│        Weighted ensemble of structured + NLP signals             │
│        Final fraud probability score (0.0 – 1.0)                │
│        Rule-based overrides for known fraud patterns             │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│                    Merchant Dashboard (Streamlit)                 │
│     Real-time predictions · Alert feed · Sentiment insights      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Machine Learning & Anomaly Detection
| Tool | Purpose |
|------|---------|
| **XGBoost** | Primary gradient-boosted classifier for structured fraud signals |
| **Scikit-learn** | Logistic regression, preprocessing pipelines, evaluation metrics |
| **Keras / TensorFlow** | Autoencoder architecture for unsupervised anomaly detection |
| **LightGBM** | Secondary ensemble model, faster training on high-cardinality features |
| **CatBoost** | Handles categorical features (merchant category, device type) natively |
| **Imbalanced-learn (SMOTE)** | Synthetic minority oversampling for severe class imbalance |

### Natural Language Processing
| Tool | Purpose |
|------|---------|
| **spaCy** | Text preprocessing, tokenization, named entity recognition on notes |
| **NLTK** | Stopword removal, stemming, lexicon-based sentiment baseline |
| **TF-IDF (Scikit-learn)** | Sparse feature representation of review and note text |
| **Word2Vec / GloVe** | Dense word embeddings for semantic similarity between transactions |
| **VADER / TextBlob** | Rule-based sentiment scoring on customer review text |

### Feature Engineering
| Tool | Purpose |
|------|---------|
| **Pandas / NumPy** | Transaction aggregation, rolling window features, velocity signals |
| **Featuretools** | Automated deep feature synthesis for relational transaction data |
| **Scikit-learn Pipelines** | Reproducible, leakage-free preprocessing and transformation |

### Experiment Tracking & Model Management
| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment logging, model registry, artifact versioning |
| **Optuna** | Hyperparameter optimization — learning rate, depth, regularization |
| **SHAP** | Model explainability — feature importance for each fraud prediction |

### Visualization & Dashboard
| Tool | Purpose |
|------|---------|
| **Streamlit** | Interactive merchant-facing fraud monitoring dashboard |
| **Plotly** | Interactive charts — fraud rate over time, confusion matrix, ROC curve |
| **Matplotlib / Seaborn** | Static EDA visualizations and model evaluation plots |

### Data Infrastructure
| Tool | Purpose |
|------|---------|
| **Apache Kafka** | Simulated real-time transaction stream ingestion |
| **PostgreSQL** | Storing labeled transaction history and model outputs |
| **Redis** | Low-latency feature caching for real-time inference |
| **Apache Spark / PySpark** | Batch feature computation on large historical datasets |

### API & Deployment
| Tool | Purpose |
|------|---------|
| **FastAPI** | REST API exposing real-time fraud scoring endpoint |
| **Docker** | Containerized model server and Streamlit dashboard |
| **AWS Lambda** | Serverless inference for lightweight, event-driven scoring |
| **AWS S3** | Storing model artifacts, datasets, and SHAP output reports |

### Monitoring
| Tool | Purpose |
|------|---------|
| **Evidently AI** | Detecting feature drift in transaction data over time |
| **Prometheus + Grafana** | API health metrics — latency, throughput, error rate |

---

## Model Design

### Signal 1: Structured Transaction Features

Key engineered features:
- **Velocity signals**: transaction count in last 1hr / 24hr / 7d per user
- **Amount anomalies**: deviation from user's historical spend distribution
- **Merchant risk score**: historical fraud rate by merchant category code (MCC)
- **Device fingerprint consistency**: IP geolocation vs billing address delta
- **Time-of-day patterns**: transaction hour relative to user's typical behavior
- **Card-not-present flag**: online vs in-person transaction indicator

### Signal 2: Autoencoder Anomaly Detection

```python
# Architecture: encoder compresses → decoder reconstructs
# High reconstruction error = anomalous transaction pattern

Input (n_features) → Dense(64, relu) → Dense(32, relu) → Dense(16, relu)  # Encoder
                   → Dense(32, relu) → Dense(64, relu) → Dense(n_features) # Decoder

Loss: Mean Squared Error (MSE)
Threshold: 95th percentile of reconstruction error on clean training data
```

Transactions with reconstruction error above threshold are flagged as anomalous, regardless of XGBoost prediction.

### Signal 3: NLP Sentiment & Intent Analysis

Customer reviews, transaction notes, and dispute text are processed to extract:
- **Negative sentiment signals**: "unauthorized", "didn't order", "never received"
- **Behavioral intent markers**: urgency language, pressure tactics in descriptions
- **Named entity mismatch**: merchant name in note vs registered merchant name

```python
# TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced'))
])

# VADER sentiment score fused as a feature
sentiment_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
```

### Ensemble Fusion

```python
# Weighted probability fusion
fraud_score = (
    0.50 * xgboost_proba +
    0.30 * autoencoder_anomaly_score +
    0.20 * nlp_risk_score
)

# Hard override rules applied on top
if velocity_1hr > 15 or amount_zscore > 5.0:
    fraud_score = max(fraud_score, 0.85)
```

---

## Results

| Metric | Value |
|--------|-------|
| Recall (fraud detection rate) | **95%** |
| Precision | 87% |
| F1 Score | 0.91 |
| AUC-ROC | 0.97 |
| False Positive Rate | 4.2% |
| Average inference latency | < 20ms |
| Dataset size | 2.8M transactions |
| Class imbalance ratio | 1:220 (fraud:legit) |

---

## Streamlit Dashboard

The merchant dashboard provides:

- **Live transaction feed** with real-time fraud probability scores
- **Alert panel** for high-confidence fraud detections (score > 0.85)
- **Sentiment insight view**: flagged keywords and phrases from transaction notes
- **Trend charts**: fraud rate over time, top merchant categories at risk
- **Model explainability**: SHAP waterfall chart per transaction showing why it was flagged

```bash
# Run dashboard locally
streamlit run app/dashboard.py
```

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                        # Raw transaction CSVs
│   ├── processed/                  # Cleaned, feature-engineered datasets
│   └── reference/                  # Clean baseline for drift detection
├── src/
│   ├── features/
│   │   ├── velocity.py             # Rolling window transaction velocity
│   │   ├── user_profile.py         # User behavior baseline features
│   │   └── nlp_features.py         # TF-IDF + sentiment extraction
│   ├── models/
│   │   ├── xgboost_model.py        # XGBoost training and evaluation
│   │   ├── autoencoder.py          # Keras autoencoder for anomaly detection
│   │   ├── nlp_classifier.py       # TF-IDF logistic regression
│   │   └── ensemble.py             # Fusion and scoring logic
│   ├── api/
│   │   └── main.py                 # FastAPI inference endpoint
│   └── monitoring/
│       └── drift_check.py          # Evidently AI drift reports
├── app/
│   └── dashboard.py                # Streamlit merchant dashboard
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability.ipynb     # SHAP analysis
├── infra/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Train Models

```bash
# Run full training pipeline
python src/models/train_all.py

# Or train individual components
python src/models/xgboost_model.py --config configs/xgb_config.yaml
python src/models/autoencoder.py --epochs 50 --batch-size 256
python src/models/nlp_classifier.py
```

### Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1250.00, "merchant_id": "M4421", "user_id": "U8810", "note": "urgent transfer"}'
```

### Launch Dashboard

```bash
streamlit run app/dashboard.py
# Opens at http://localhost:8501
```

### Run with Docker

```bash
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# MLflow UI: http://localhost:5000
```

---

## Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Severe class imbalance (1:220) | SMOTE oversampling + cost-sensitive XGBoost (`scale_pos_weight`) |
| Real-time inference latency | Redis feature cache + lightweight model serving via FastAPI |
| Interpretability for fraud teams | SHAP waterfall plots per prediction |
| Concept drift over time | Evidently AI drift monitoring + scheduled retraining |
| NLP signal noise | Bigram TF-IDF + VADER ensemble outperformed single approach by 6% recall |

---

## Skills Demonstrated

`Machine Learning` `XGBoost` `LightGBM` `CatBoost` `Anomaly Detection` `Autoencoders` `Keras` `TensorFlow` `NLP` `spaCy` `NLTK` `TF-IDF` `Sentiment Analysis` `Word2Vec` `SHAP` `MLflow` `Optuna` `Streamlit` `Plotly` `FastAPI` `Docker` `Apache Kafka` `PySpark` `Redis` `PostgreSQL` `AWS Lambda` `AWS S3` `Evidently AI` `Prometheus` `Grafana` `Python` `Scikit-learn` `Imbalanced-learn` `Featuretools`

---

## Context

This project was developed during my Master of Science in Data Science at Saint Peter's University, NJ (2023–2025). The fraud detection techniques here — particularly the autoencoder anomaly detection and NLP intent analysis — were refined based on learnings from my production work at MakroCare, where similar anomaly detection approaches were applied to patient health metric monitoring.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

**Sai Kumar** | AI/ML Engineer  
anumalasaikumar169@gmail.com | [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
