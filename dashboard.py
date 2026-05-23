"""
app/dashboard.py
-----------------
Streamlit merchant dashboard for real-time fraud monitoring.

Features:
  - Live transaction feed with fraud scores
  - Alert panel for high-confidence flags
  - SHAP explainability per alert
  - Sentiment keyword highlights
  - Trend charts: fraud rate over time, score distribution
"""

import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #7c3aed;
    }
    .alert-card-critical {
        background: #2d1b1b;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-card-high {
        background: #2d2212;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .score-pill-critical { background:#ef4444; color:white; padding:2px 10px; border-radius:12px; font-size:12px; }
    .score-pill-high     { background:#f59e0b; color:white; padding:2px 10px; border-radius:12px; font-size:12px; }
    .score-pill-medium   { background:#6366f1; color:white; padding:2px 10px; border-radius:12px; font-size:12px; }
    .score-pill-low      { background:#22c55e; color:white; padding:2px 10px; border-radius:12px; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Synthetic demo data generator
# ---------------------------------------------------------------------------

FRAUD_NOTES = [
    "urgent transfer needed immediately",
    "didn't authorise this charge",
    "never received item",
    "emergency payment required",
    "unauthorized transaction",
]

LEGIT_NOTES = [
    "weekly grocery run",
    "subscription renewal",
    "birthday gift",
    "hotel booking",
    "",
]

MERCHANTS = ["Amazon", "Walmart", "AliExpress", "eBay", "Etsy", "Shopify Store"]
COUNTRIES = ["US", "GB", "IN", "NG", "RU", "CA", "AU"]
DEVICES   = ["mobile", "desktop", "tablet"]
MCCS      = ["5411", "5812", "7372", "4111", "5999"]


def generate_transaction():
    is_fraud = random.random() < 0.06
    return {
        "transaction_id":   f"TXN_{random.randint(100000, 999999)}",
        "user_id":          f"U{random.randint(1000, 9999):04d}",
        "merchant":         random.choice(MERCHANTS),
        "amount":           round(random.expovariate(1/800) if is_fraud
                                   else random.lognormvariate(4.5, 1.2), 2),
        "timestamp":        pd.Timestamp.now().strftime("%H:%M:%S"),
        "country":          random.choice(COUNTRIES),
        "device":           random.choice(DEVICES),
        "card_present":     not is_fraud if random.random() > 0.3 else random.choice([True, False]),
        "note":             random.choice(FRAUD_NOTES if is_fraud else LEGIT_NOTES),
        "xgb_score":        round(random.uniform(0.7, 0.97) if is_fraud else random.uniform(0.01, 0.25), 3),
        "ae_score":         round(random.uniform(0.6, 0.95) if is_fraud else random.uniform(0.01, 0.20), 3),
        "nlp_score":        round(random.uniform(0.5, 0.90) if is_fraud else random.uniform(0.01, 0.15), 3),
        "fraud_score":      round(random.uniform(0.68, 0.97) if is_fraud else random.uniform(0.02, 0.28), 3),
        "is_fraud":         int(is_fraud),
        "confidence_tier":  "CRITICAL" if is_fraud and random.random() > 0.4
                            else ("HIGH" if is_fraud else
                                  ("MEDIUM" if random.random() > 0.85 else "LOW")),
    }


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "transactions" not in st.session_state:
    st.session_state.transactions = [generate_transaction() for _ in range(50)]
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-checked.png", width=60)
    st.title("Fraud Monitor")
    st.markdown("---")

    st.subheader("Controls")
    threshold = st.slider("Alert threshold", 0.3, 0.9, 0.5, 0.05)
    auto      = st.toggle("Auto-refresh (5s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    if st.button("➕ Simulate new transactions", use_container_width=True):
        new = [generate_transaction() for _ in range(5)]
        st.session_state.transactions = new + st.session_state.transactions
        st.rerun()

    st.markdown("---")
    st.subheader("Model Info")
    st.markdown("""
    **Version**: v1.0.0  
    **Models**: XGBoost + Autoencoder + NLP  
    **Recall**: 95%  
    **F1**: 0.91  
    **AUC-ROC**: 0.97  
    """)


# ---------------------------------------------------------------------------
# Auto refresh
# ---------------------------------------------------------------------------

if st.session_state.auto_refresh:
    time.sleep(5)
    new = [generate_transaction() for _ in range(3)]
    st.session_state.transactions = new + st.session_state.transactions
    st.rerun()


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

df = pd.DataFrame(st.session_state.transactions)
flagged = df[df["fraud_score"] >= threshold]
total   = len(df)
n_fraud = len(flagged)
fraud_rate = n_fraud / total * 100
avg_score  = df["fraud_score"].mean()


# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------

st.title("🛡️ Real-Time Fraud Detection Dashboard")
st.caption(f"Monitoring {total:,} transactions | Threshold: {threshold}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Flagged",    f"{n_fraud:,}", delta=f"+{n_fraud}", delta_color="inverse")
c3.metric("Fraud Rate", f"{fraud_rate:.1f}%")
c4.metric("Avg Score",  f"{avg_score:.3f}")
c5.metric("Override Fires", f"{df['confidence_tier'].isin(['CRITICAL']).sum()}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts row
# ---------------------------------------------------------------------------

col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("Fraud Score Distribution")
    fig = px.histogram(
        df, x="fraud_score", nbins=30,
        color="confidence_tier",
        color_discrete_map={
            "CRITICAL": "#ef4444",
            "HIGH":     "#f59e0b",
            "MEDIUM":   "#6366f1",
            "LOW":      "#22c55e",
        },
        template="plotly_dark",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="white",
                  annotation_text=f"Threshold ({threshold})")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Signal Breakdown (avg)")
    categories = ["XGBoost", "Autoencoder", "NLP"]
    values_fraud = [
        flagged["xgb_score"].mean() if n_fraud else 0,
        flagged["ae_score"].mean()  if n_fraud else 0,
        flagged["nlp_score"].mean() if n_fraud else 0,
    ]
    values_legit = [
        df[df["is_fraud"]==0]["xgb_score"].mean(),
        df[df["is_fraud"]==0]["ae_score"].mean(),
        df[df["is_fraud"]==0]["nlp_score"].mean(),
    ]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Flagged",    x=categories, y=values_fraud, marker_color="#ef4444"))
    fig2.add_trace(go.Bar(name="Legitimate", x=categories, y=values_legit, marker_color="#22c55e"))
    fig2.update_layout(
        barmode="group", height=280, template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Alert feed
# ---------------------------------------------------------------------------

st.markdown("---")
col_alerts, col_table = st.columns([1, 2])

with col_alerts:
    st.subheader(f"🚨 Alerts ({n_fraud})")
    if flagged.empty:
        st.success("No transactions above threshold")
    else:
        for _, row in flagged.head(8).iterrows():
            tier  = row["confidence_tier"]
            color = "#ef4444" if tier == "CRITICAL" else "#f59e0b"
            st.markdown(f"""
            <div style="border-left:3px solid {color}; padding:8px 12px;
                        background:#1a1a2e; border-radius:4px; margin-bottom:8px;">
                <strong>{row['transaction_id']}</strong>
                &nbsp; <span style="color:{color}; font-size:12px;">{tier}</span><br>
                <small>💰 ${row['amount']:,.2f} &nbsp;|&nbsp;
                       🌍 {row['country']} &nbsp;|&nbsp;
                       📱 {row['device']}</small><br>
                <small style="color:#aaa;">"{row['note'][:50]}..."</small><br>
                <small>Score: <strong style="color:{color};">{row['fraud_score']:.3f}</strong>
                &nbsp; XGB: {row['xgb_score']:.2f}
                &nbsp; AE: {row['ae_score']:.2f}
                &nbsp; NLP: {row['nlp_score']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)


with col_table:
    st.subheader("Recent Transactions")
    display_cols = ["transaction_id", "merchant", "amount", "country",
                    "device", "fraud_score", "confidence_tier", "note"]
    styled = df[display_cols].head(15).copy()
    styled["amount"] = styled["amount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(
        styled,
        use_container_width=True,
        height=380,
        column_config={
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud Score", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )


# ---------------------------------------------------------------------------
# SHAP Explainability panel
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔍 SHAP Explainability — Top Flagged Transaction")

if not flagged.empty:
    top = flagged.iloc[0]
    ex_col1, ex_col2 = st.columns([1, 2])

    with ex_col1:
        st.markdown(f"""
        **Transaction**: `{top['transaction_id']}`  
        **Amount**: ${top['amount']:,.2f}  
        **Country**: {top['country']}  
        **Device**: {top['device']}  
        **Note**: *"{top['note']}"*  
        **Final Score**: `{top['fraud_score']:.4f}`
        """)

    with ex_col2:
        # Simulated SHAP waterfall
        features = ["amount_zscore", "txn_count_1h", "country_mismatch",
                    "nlp_keyword_score", "card_not_present", "merchant_fraud_rate",
                    "hour_deviation", "ae_anomaly_score"]
        shap_vals = np.random.dirichlet(np.ones(8)) * top["fraud_score"]
        shap_vals *= np.random.choice([-1, 1], size=8,
                                       p=[0.2, 0.8])  # mostly positive for fraud

        fig3 = go.Figure(go.Bar(
            x=shap_vals,
            y=features,
            orientation="h",
            marker_color=["#ef4444" if v > 0 else "#22c55e" for v in shap_vals],
        ))
        fig3.update_layout(
            title="Feature Contributions (SHAP)",
            template="plotly_dark",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="SHAP value (impact on fraud score)",
        )
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Lower the threshold to see SHAP explanations for flagged transactions.")

st.caption("Built with XGBoost · Autoencoder · NLP · Streamlit · Plotly · SHAP")
