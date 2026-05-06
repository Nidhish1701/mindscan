"""
Real-time Analytics Dashboard for the Mental Health Early Warning System.

Built with Streamlit — runs as a standalone web app alongside the FastAPI server.

Run:
    streamlit run dashboard/app.py --server.port 8501

Features:
    - Live prediction history feed
    - Label distribution chart
    - Confidence distribution histogram
    - Crisis flag rate over time
    - Interactive single-comment analyzer
    - Model performance metrics
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health Early Warning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH    = os.getenv("DB_PATH", "data/predictions.db")
MODEL_DIR  = os.getenv("MODEL_DIR", "models/distilbert/best_model")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .metric-card { background: #f8f9fa; border-radius: 12px; padding: 16px 20px; border: 1px solid #e9ecef; }
    .crisis-badge { background: #fee2e2; color: #991b1b; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .safe-badge { background: #dcfce7; color: #166534; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data(ttl=10)
def load_predictions(limit: int = 500) -> pd.DataFrame:
    """Load prediction history from SQLite, refreshes every 10 seconds."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=[
            "timestamp", "username", "text", "prediction",
            "confidence", "risk_flag", "latency_ms"
        ])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}",
        conn, parse_dates=["timestamp"]
    )
    conn.close()
    return df


def load_model_cached():
    """Load model once and cache in session state."""
    if "model" not in st.session_state:
        try:
            from src.predict import load_model
            model, tokenizer, label_encoder = load_model(MODEL_DIR)
            st.session_state["model"]         = model
            st.session_state["tokenizer"]     = tokenizer
            st.session_state["label_encoder"] = label_encoder
            st.session_state["classes"]       = list(label_encoder.classes_)
        except Exception as e:
            st.session_state["model_error"] = str(e)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🧠 MH Warning System")
    st.markdown("---")

    page = st.radio("Navigation", [
        "Dashboard",
        "Live Analyzer",
        "Prediction History",
        "Model Info",
    ])

    st.markdown("---")
    st.markdown("**System Status**")

    if os.path.exists(DB_PATH):
        conn   = sqlite3.connect(DB_PATH)
        total  = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        crisis = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_flag=1").fetchone()[0]
        conn.close()
        st.success(f"Database online — {total:,} records")
        if crisis > 0:
            st.error(f"⚠ {crisis} crisis flags")
    else:
        st.warning("No prediction data yet")

    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":
    st.title("Mental Health Detection — Analytics Dashboard")

    df = load_predictions()

    if df.empty:
        st.info("No predictions yet. Use the Live Analyzer or the API to generate predictions.")
        st.stop()

    # ---- KPI row ----
    col1, col2, col3, col4 = st.columns(4)
    total_preds = len(df)
    crisis_cnt  = df["risk_flag"].sum()
    avg_conf    = df["confidence"].mean()
    avg_lat     = df["latency_ms"].mean() if "latency_ms" in df.columns else 0

    col1.metric("Total predictions", f"{total_preds:,}")
    col2.metric("Crisis flags",      f"{int(crisis_cnt):,}",
                delta=f"{crisis_cnt/total_preds*100:.1f}% rate",
                delta_color="inverse")
    col3.metric("Avg confidence",    f"{avg_conf*100:.1f}%")
    col4.metric("Avg latency",       f"{avg_lat:.0f} ms")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Label distribution")
        label_counts = df["prediction"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        fig = px.bar(label_counts, x="Label", y="Count",
                     color="Label", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Confidence distribution")
        fig2 = px.histogram(df, x="confidence", nbins=30,
                             color_discrete_sequence=["#6366f1"])
        fig2.add_vline(x=0.7, line_dash="dash", line_color="orange",
                       annotation_text="70% threshold")
        fig2.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Crisis timeline ----
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        st.subheader("Crisis flags over time")
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        crisis_daily = df.groupby("date")["risk_flag"].sum().reset_index()
        fig3 = px.line(crisis_daily, x="date", y="risk_flag",
                       labels={"risk_flag": "Crisis flags"},
                       color_discrete_sequence=["#ef4444"])
        fig3.update_layout(height=250, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Recent predictions table ----
    st.subheader("Recent predictions")
    display_df = df[["timestamp", "text", "prediction", "confidence", "risk_flag"]].head(20).copy()
    display_df["text"]       = display_df["text"].str[:60] + "..."
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
    display_df["risk_flag"]  = display_df["risk_flag"].map({1: "⚠ CRISIS", 0: "safe"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# LIVE ANALYZER PAGE
# --------------------------------------------------

elif page == "Live Analyzer":
    st.title("Live Comment Analyzer")
    st.caption("Type or paste a Reddit comment to get an instant mental health assessment.")

    load_model_cached()

    if "model_error" in st.session_state:
        st.error(f"Model not loaded: {st.session_state['model_error']}")
        st.info("Train a model first: `python -m src.train_pipeline --model distilbert ...`")
        st.stop()

    text_input = st.text_area("Comment text", height=120,
                               placeholder="Enter a Reddit post or comment here...")

    if st.button("Analyze", type="primary") and text_input.strip():
        from src.predict import predict as run_predict

        with st.spinner("Analyzing..."):
            results = run_predict(
                [text_input],
                st.session_state["model"],
                st.session_state["tokenizer"],
                st.session_state["label_encoder"],
            )
        r = results[0]

        from api.crisis_alert import check_crisis_risk
        is_crisis = check_crisis_risk(r["prediction"], r["confidence"])

        col1, col2 = st.columns([1, 2])

        with col1:
            if is_crisis:
                st.error(f"⚠ CRISIS FLAG\n\n**{r['prediction'].upper()}**")
                st.warning("This comment has been flagged for clinical review.")
            else:
                st.success(f"**{r['prediction'].upper()}**")

            st.metric("Confidence", f"{r['confidence']*100:.1f}%")

        with col2:
            st.subheader("Probability breakdown")
            probs = dict(sorted(r["probabilities"].items(), key=lambda x: -x[1]))
            prob_df = pd.DataFrame({"Class": list(probs.keys()),
                                    "Probability": list(probs.values())})
            fig = px.bar(prob_df, x="Probability", y="Class", orientation="h",
                         color="Probability",
                         color_continuous_scale=["#bbf7d0", "#ef4444"])
            fig.update_layout(height=250, margin=dict(t=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if is_crisis:
            with st.expander("Crisis resources"):
                st.markdown("""
                **India Crisis Lines:**
                - iCall: `9152987821`
                - Vandrevala Foundation: `1860-2662-345`

                **International:** https://findahelpline.com
                """)


# --------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------

elif page == "Prediction History":
    st.title("Prediction History")

    df = load_predictions(limit=1000)
    if df.empty:
        st.info("No prediction history yet.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        labels = ["All"] + sorted(df["prediction"].unique().tolist())
        label_filter = st.selectbox("Filter by label", labels)
    with col2:
        risk_filter = st.selectbox("Risk flag", ["All", "Crisis only", "Safe only"])
    with col3:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = df.copy()
    if label_filter != "All":
        filtered = filtered[filtered["prediction"] == label_filter]
    if risk_filter == "Crisis only":
        filtered = filtered[filtered["risk_flag"] == 1]
    elif risk_filter == "Safe only":
        filtered = filtered[filtered["risk_flag"] == 0]
    filtered = filtered[filtered["confidence"] >= min_conf]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered[["timestamp","username","text","prediction",
                            "confidence","risk_flag"]].head(200),
                 use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "predictions_export.csv", "text/csv")


# --------------------------------------------------
# MODEL INFO PAGE
# --------------------------------------------------

elif page == "Model Info":
    st.title("Model Information")

    model_path = MODEL_DIR
    if os.path.exists(model_path):
        st.success(f"Model loaded from: `{model_path}`")

        label_map_path = os.path.join(model_path, "label_map.json")
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path) as f:
                label_map = json.load(f)
            st.subheader("Label mapping")
            st.json(label_map)
    else:
        st.warning(f"Model directory not found: `{model_path}`")
        st.info("Train a model first and update MODEL_DIR in your .env")

    st.subheader("System configuration")
    st.code(f"""
MODEL_DIR  = {MODEL_DIR}
DB_PATH    = {DB_PATH}
API_URL    = http://localhost:8000
DASHBOARD  = http://localhost:8501
    """)

# ---- Auto-refresh ----
if auto_refresh:
    time.sleep(10)
    st.rerun()
