from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="CTU-13 Attack Traffic Dashboard", layout="wide")


DATA_FILE = Path("CTU13_Attack_Traffic.csv")
RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def parse_label(value) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if text == "":
        return np.nan

    try:
        return float(text)
    except ValueError:
        pass

    positive_tokens = ("bot", "malware", "attack", "anomaly", "ddos", "malicious")
    negative_tokens = ("normal", "benign", "legit", "clean")

    if any(token in text for token in positive_tokens):
        return 1.0
    if any(token in text for token in negative_tokens):
        return 0.0

    return np.nan


def to_probability(score: np.ndarray) -> np.ndarray:
    min_s = float(np.min(score))
    max_s = float(np.max(score))
    if max_s - min_s < 1e-12:
        return np.full_like(score, 0.5, dtype=float)
    return (score - min_s) / (max_s - min_s)


def prepare_flow_data(df: pd.DataFrame):
    clean = df.copy()

    if "Unnamed: 0" in clean.columns:
        clean = clean.drop(columns=["Unnamed: 0"])

    for col in clean.columns:
        if col != "Label":
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    numeric_cols = [c for c in clean.columns if c != "Label"]
    clean = clean.replace([np.inf, -np.inf], np.nan)

    # Keep rows with enough feature signal, then impute remaining nulls by median.
    clean = clean.dropna(subset=numeric_cols, thresh=max(3, int(0.4 * len(numeric_cols))))
    medians = clean[numeric_cols].median(numeric_only=True)
    clean[numeric_cols] = clean[numeric_cols].fillna(medians)

    clean["FlowID"] = np.arange(1, len(clean) + 1)
    clean["LabelBinary"] = clean["Label"].apply(parse_label)

    X = clean[numeric_cols]
    y = clean["LabelBinary"]

    return clean, X, y, numeric_cols


def score_flows(X: pd.DataFrame, y: pd.Series):
    unique_labels = sorted(y.dropna().unique().tolist())
    metrics = {}

    if len(unique_labels) >= 2:
        y_use = y.fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_use,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_use,
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X)[:, 1]

        metrics["mode"] = "supervised"
        metrics["rf_accuracy"] = accuracy_score(y_test, rf_pred)
        metrics["rf_report"] = classification_report(y_test, rf_pred, output_dict=True)
        metrics["feature_importance"] = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        if has_xgboost():
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
            )
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            xgb_prob = xgb.predict_proba(X)[:, 1]

            metrics["xgb_accuracy"] = accuracy_score(y_test, xgb_pred)
            metrics["xgb_report"] = classification_report(y_test, xgb_pred, output_dict=True)
            metrics["risk_probability"] = xgb_prob
        else:
            metrics["xgb_accuracy"] = None
            metrics["xgb_report"] = None
            metrics["risk_probability"] = rf_prob
    else:
        iso = IsolationForest(
            n_estimators=300,
            contamination=0.1,
            random_state=RANDOM_STATE,
        )
        iso.fit(X)

        # More anomalous points have smaller score_samples values.
        raw_anomaly = -iso.score_samples(X)
        prob = to_probability(raw_anomaly)

        var_rank = X.var().sort_values(ascending=False)
        metrics["mode"] = "unsupervised"
        metrics["rf_accuracy"] = None
        metrics["rf_report"] = None
        metrics["xgb_accuracy"] = None
        metrics["xgb_report"] = None
        metrics["risk_probability"] = prob
        metrics["feature_importance"] = var_rank

    return metrics


def add_risk_columns(df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    scored = df.copy()
    scored["RiskScore"] = np.clip(probabilities * 100.0, 0.0, 100.0)

    bins = [-1, 35, 70, 100]
    labels = ["Low", "Medium", "High"]
    scored["RiskBand"] = pd.cut(scored["RiskScore"], bins=bins, labels=labels)
    scored["Status"] = np.where(scored["RiskScore"] > 70, "Bot", "Normal")

    return scored


def choose_axes(df: pd.DataFrame, fallback_cols: list[str]):
    preferred_pairs = [
        ("Flow Duration", "Flow Byts/s"),
        ("Tot Fwd Pkts", "TotLen Fwd Pkts"),
        ("Flow IAT Mean", "Flow IAT Std"),
    ]

    for x_col, y_col in preferred_pairs:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col

    if len(fallback_cols) >= 2:
        return fallback_cols[0], fallback_cols[1]

    return fallback_cols[0], fallback_cols[0]


def build_dashboard_figure(scored: pd.DataFrame, x_axis: str) -> go.Figure:
    top10 = scored.sort_values("RiskScore", ascending=False).head(10)
    status_counts = scored["Status"].value_counts()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top 10 Suspicious Flows",
            "Risk Score Distribution",
            f"{x_axis} vs Risk Score",
            "Bot vs Normal Flows",
        ),
        specs=[[{"type": "bar"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "pie"}]],
    )

    fig.add_trace(
        go.Bar(
            x=top10["FlowID"].astype(str),
            y=top10["RiskScore"],
            marker_color="crimson",
            name="Risk Score",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=scored["RiskScore"],
            marker_color="cyan",
            nbinsx=30,
            name="Risk Distribution",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=scored[x_axis],
            y=scored["RiskScore"],
            mode="markers",
            text=[f"FlowID: {x}" for x in scored["FlowID"]],
            marker={
                "size": 8,
                "color": scored["RiskScore"],
                "colorscale": "Turbo",
                "showscale": True,
                "opacity": 0.75,
            },
            name="Flows",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.45),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=920,
        template="plotly_dark",
        title_text="CTU-13 Threat Intelligence Dashboard",
        title_x=0.5,
        showlegend=False,
    )

    return fig


def build_insights_figure(scored: pd.DataFrame, importance: pd.Series, numeric_cols: list[str]) -> go.Figure:
    top_features = importance.head(12)
    corr_cols = [c for c in top_features.index[:8] if c in scored.columns]
    if "RiskScore" not in corr_cols:
        corr_cols = corr_cols + ["RiskScore"]
    corr = scored[corr_cols].corr(numeric_only=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Feature Signals", "Correlation Heatmap"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]],
    )

    fig.add_trace(
        go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation="h",
            marker_color="#33c3ff",
            name="Feature Signal",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "corr"},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=500, template="plotly_dark", showlegend=False)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    return fig


def main() -> None:
    st.markdown("## CTU-13 Attack Traffic Dashboard")

    if not DATA_FILE.exists():
        st.error(f"Dataset file not found: {DATA_FILE}")
        st.stop()

    raw_df = load_csv(DATA_FILE)
    if raw_df.empty:
        st.error(f"Dataset is empty: {DATA_FILE}")
        st.stop()

    prepared_df, X, y, numeric_cols = prepare_flow_data(raw_df)

    if X.empty:
        st.error("No usable numeric features after preprocessing.")
        st.stop()

    metrics = score_flows(X, y)
    scored = add_risk_columns(prepared_df, metrics["risk_probability"])

    total_flows = len(scored)
    high_risk = int((scored["RiskScore"] > 70).sum())
    avg_risk = float(scored["RiskScore"].mean())
    median_risk = float(scored["RiskScore"].median())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flows", total_flows)
    c2.metric("High Risk Flows", high_risk)
    c3.metric("Average Risk", f"{avg_risk:.2f}")
    c4.metric("Median Risk", f"{median_risk:.2f}")

    x_axis, _ = choose_axes(scored, numeric_cols)
    chart = build_dashboard_figure(scored, x_axis)
    st.plotly_chart(chart, use_container_width=True)

    insights_chart = build_insights_figure(scored, metrics["feature_importance"], numeric_cols)
    st.plotly_chart(insights_chart, use_container_width=True)

    cols_for_table = [
        c
        for c in ["FlowID", "Label", "RiskScore", "RiskBand", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Byts/s"]
        if c in scored.columns
    ]
    st.markdown("### Top Suspicious Flows")
    st.dataframe(
        scored.sort_values("RiskScore", ascending=False)[cols_for_table].head(20),
        use_container_width=True,
    )

    summary = pd.DataFrame(
        {
            "Metric": ["P95 Risk", "P99 Risk", "Low Risk Flows", "Medium Risk Flows", "High Risk Flows"],
            "Value": [
                f"{scored['RiskScore'].quantile(0.95):.2f}",
                f"{scored['RiskScore'].quantile(0.99):.2f}",
                int((scored["RiskBand"] == "Low").sum()),
                int((scored["RiskBand"] == "Medium").sum()),
                int((scored["RiskBand"] == "High").sum()),
            ],
        }
    )
    st.markdown("### Risk Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### Model Metrics")
    m1, m2 = st.columns(2)

    if metrics["rf_accuracy"] is not None:
        m1.metric("Random Forest Accuracy", f"{metrics['rf_accuracy']:.4f}")
    else:
        m1.metric("Model Mode", "Random Forest/ XGBoost")

    if metrics["xgb_accuracy"] is not None:
        m2.metric("XGBoost Accuracy", f"{metrics['xgb_accuracy']:.4f}")
    elif metrics["mode"] == "supervised":
        m2.info("xgboost not installed; using Random Forest risk probability.")
    else:
        m2.metric("Supervised Labels", "Unavailable")

    with st.expander("Detailed Reports"):
        if metrics["rf_report"] is not None:
            st.write("Random Forest Classification Report")
            st.dataframe(pd.DataFrame(metrics["rf_report"]).transpose(), use_container_width=True)
        if metrics["xgb_report"] is not None:
            st.write("XGBoost Classification Report")
            st.dataframe(pd.DataFrame(metrics["xgb_report"]).transpose(), use_container_width=True)
        st.write("Top Feature Signals")
        st.dataframe(metrics["feature_importance"].head(20).rename("signal").to_frame(), use_container_width=True)

    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Scored Flows",
        data=csv_bytes,
        file_name="ctu13_scored_flows.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
