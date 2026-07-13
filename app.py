"""
FX Forecasting -- EUR/USD Dashboard
Run:  python -m streamlit run app.py   (use venv_dl)
"""
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from predict_latest import REGISTRY, predict_latest, load_features

st.set_page_config(page_title="EUR/USD Forecasting", layout="wide")

# ---------- dark style for matplotlib ----------
DARK_BG = "#161b26"
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color":       "#e6e9ef",
    "axes.labelcolor":  "#e6e9ef",
    "xtick.color":      "#aab2c0",
    "ytick.color":      "#aab2c0",
    "axes.edgecolor":   "#3d4656",
    "grid.color":       "#2a3140",
    "legend.facecolor": DARK_BG,
    "legend.edgecolor": "#3d4656",
})
C_DL, C_DL2 = "#ffb400", "#c78c00"     # amber  = deep learning
C_ML, C_ML2 = "#4f8df5", "#31589c"     # blue   = machine learning
C_BASE = "#ff6b6b"                     # baseline lines
C_GRAY = "#8892a6"

# ---------- paths ----------
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures" / "models"
MASTER_CSV = ROOT / "data" / "interim" / "fx_master_dataset.csv"
LIVE_CSV = ROOT / "data" / "processed" / "fx_features_live.csv"

# ---------- cached loaders ----------
@st.cache_data
def load_results():
    data = {}
    p = MODELS_DIR / "all_10_models_comparison.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0)
        d.columns = [c.strip().lower() for c in d.columns]
        data["direction"] = d
    p = MODELS_DIR / "volatility_regression_results.csv"
    if p.exists():
        data["vol_reg"] = pd.read_csv(p, index_col=0)
    p = MODELS_DIR / "volatility_classification_results.csv"
    if p.exists():
        data["vol_clf"] = pd.read_csv(p, index_col=0)
    return data

@st.cache_data
def get_features():
    return load_features()

@st.cache_data
def load_master():
    if not MASTER_CSV.exists():
        return None
    return pd.read_csv(MASTER_CSV, parse_dates=["date"])

results = load_results()

# ---------- header ----------
st.title("EUR/USD Forecasting Dashboard")
st.caption("Dong Cong Gia Khang · Final Project 2026 · ML + Deep Learning")

tab0, tab1, tab2, tab3 = st.tabs(
    ["Overview", "Model Comparison", "Prediction", "EDA"]
)

# ============================================================
# TAB 0 -- OVERVIEW
# ============================================================
with tab0:
    dfm = load_master()
    if dfm is None:
        st.error(f"File not found: {MASTER_CSV}")
    elif "eurusd" not in dfm.columns:
        st.error(f"Column 'eurusd' not in master dataset. Columns: {list(dfm.columns)}")
    else:
        dfm = dfm.dropna(subset=["eurusd"]).reset_index(drop=True)
        price = dfm["eurusd"]
        logret = np.log(price).diff()
        vol20 = logret.rolling(20).std() * 100          # %/day

        # ---- KPI cards ----
        k1, k2, k3, k4 = st.columns(4)
        chg = (price.iloc[-1] / price.iloc[-2] - 1) * 100
        k1.metric("EUR/USD (last close)", f"{price.iloc[-1]:.4f}", f"{chg:+.2f}%")
        k2.metric("Realized vol (20d)", f"{vol20.iloc[-1]:.2f}% / day")
        wk = (price.iloc[-1] / price.iloc[-6] - 1) * 100 if len(price) > 6 else 0
        k3.metric("5-day change", f"{wk:+.2f}%")
        if "vixcls" in dfm.columns and dfm["vixcls"].notna().any():
            vix = dfm["vixcls"].dropna()
            k4.metric("VIX (last)", f"{vix.iloc[-1]:.1f}",
                      f"{vix.iloc[-1] - vix.iloc[-2]:+.1f}")
        else:
            k4.metric("Data points", f"{len(dfm):,}")
        st.caption(f"Master dataset — data through "
                   f"**{dfm['date'].iloc[-1].date()}**")

        # ---- price chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfm["date"], y=price, mode="lines",
            line=dict(color="#ffb400", width=1.3),
            name="EUR/USD",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title="EUR/USD — daily close",
            template="plotly_dark",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=420, margin=dict(l=40, r=20, t=60, b=30),
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                    bgcolor="#232a3a", activecolor="#ffb400",
                    font=dict(color="#e6e9ef"),
                ),
            ),
        )
        st.plotly_chart(fig)

        # ---- volatility clustering chart ----
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dfm["date"], y=vol20, mode="lines",
            line=dict(color="#4f8df5", width=1.1),
            fill="tozeroy", fillcolor="rgba(79,141,245,0.15)",
            name="20d realized vol",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%/day<extra></extra>",
        ))
        fig2.update_layout(
            title="Realized volatility (20-day) — clustering is visible: "
                  "calm and turbulent periods form clusters",
            template="plotly_dark",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=300, margin=dict(l=40, r=20, t=60, b=30),
        )
        st.plotly_chart(fig2)

        st.info("This clustering property is exactly why volatility is predictable "
                "while direction is not — see the Model Comparison tab.")

# ============================================================
# TAB 1 -- MODEL COMPARISON
# ============================================================
with tab1:
    st.subheader("Select problem formulation")
    problem = st.selectbox(
        "Problem",
        ["Direction — Classification",
         "Volatility ratio — Regression",
         "Volatility high/low — Classification"],
    )

    # ---------- DIRECTION ----------
    if problem.startswith("Direction"):
        if "direction" not in results:
            st.error("File not found: all_10_models_comparison.csv")
        else:
            df = results["direction"].copy()
            st.info("Baselines: Daily val ~50.6% · Weekly val ~51.8%. "
                    "Models are selected on **Validation**; test is reported once.")

            st.dataframe(
                df.style.format("{:.2f}").highlight_max(
                    subset=["val"], color="#2e5233"),
                width="stretch",
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            x = np.arange(len(df))
            w = 0.38
            is_dl = [m.startswith(("LSTM", "GRU")) for m in df.index]
            ax.bar(x - w/2, df["val"], w, label="Validation",
                   color=[C_DL if d else C_ML for d in is_dl])
            ax.bar(x + w/2, df["test"], w, label="Test",
                   color=[C_DL2 if d else C_ML2 for d in is_dl])
            ax.axhline(50, color=C_GRAY, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Directional Accuracy (%)")
            ax.set_ylim(42, 60)
            ax.set_title("Amber = Deep Learning · Blue = Machine Learning",
                         fontsize=10)
            ax.legend(fontsize=8)
            st.pyplot(fig)

            best = df["val"].idxmax()
            st.success(f"Best by validation: {best} — {df.loc[best,'val']:.2f}%")

    # ---------- VOLATILITY REGRESSION ----------
    elif problem.startswith("Volatility ratio"):
        if "vol_reg" not in results:
            st.error("File not found: volatility_regression_results.csv")
        else:
            df = results["vol_reg"].copy()
            st.info("Regression has **no accuracy**. Metrics: MAE (lower is better), "
                    "R² (> 0 means better than baseline). Baseline MAE = 0.082")

            st.dataframe(df.style.format("{:.4f}"), width="stretch")

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            axes[0].bar(df.index, df["val_mae"], color=C_ML)
            axes[0].axhline(0.082, color=C_BASE, ls="--", lw=1.4,
                            label="Baseline 0.082")
            axes[0].set_ylabel("Val MAE")
            axes[0].set_title("MAE (lower is better)")
            axes[0].tick_params(axis="x", rotation=30, labelsize=8)
            axes[0].legend(fontsize=8)

            colors = [C_DL if v > 0 else C_BASE for v in df["val_r2"]]
            axes[1].bar(df.index, df["val_r2"], color=colors)
            axes[1].axhline(0, color=C_GRAY, lw=1)
            axes[1].set_ylabel("Val R²")
            axes[1].set_title("R² (> 0 beats baseline)")
            axes[1].tick_params(axis="x", rotation=30, labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            best = df["val_r2"].idxmax()
            st.success(f"Best by R²: {best} — R² = {df.loc[best,'val_r2']:+.4f}")

    # ---------- VOLATILITY CLASSIFICATION ----------
    else:
        if "vol_clf" not in results:
            st.error("File not found: volatility_classification_results.csv")
        else:
            df = results["vol_clf"].copy()
            st.warning("Accuracy is ~60% but **baseline = 57%** (val/test split "
                       "43/57). The real edge is only about 3 pp — see the "
                       "`val_vs_base` column.")

            st.dataframe(df.style.format("{:.2f}"), width="stretch")

            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(df))
            ax.bar(x, df["val"], 0.55, color=C_ML)
            ax.axhline(56.99, color=C_BASE, ls="--", lw=1.5, label="Baseline 57%")
            ax.axhline(50, color=C_GRAY, ls=":", lw=1, label="Random 50%")
            for i, v in enumerate(df["val"]):
                ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8,
                        fontweight="bold", color="#e6e9ef")
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Val Accuracy (%)")
            ax.set_ylim(48, 63)
            ax.legend(fontsize=8)
            st.pyplot(fig)

            best = df["val"].idxmax()
            st.success(f"Best by validation: {best} — {df.loc[best,'val']:.2f}% "
                       f"(edge +{df.loc[best,'val']-56.99:.2f} pp)")

# ============================================================
# TAB 2 -- PREDICTION
# ============================================================
with tab2:
    st.subheader("Predict from latest data")

    TASK_LABELS = {
        "Direction — next day (UP/DOWN)":         "direction_daily",
        "Direction — next week (UP/DOWN)":        "direction_weekly",
        "Volatility — next day (regression)":     "vol_regression",
        "Volatility — high/low (classification)": "vol_classification",
    }

    c1, c2 = st.columns(2)
    with c1:
        task_label = st.selectbox("Task", list(TASK_LABELS.keys()))
    task = TASK_LABELS[task_label]
    with c2:
        model_name = st.selectbox("Model", list(REGISTRY[task].keys()))

    use_live = st.toggle("Use live data (updated via update_data.py)",
                         value=False, disabled=not LIVE_CSV.exists())
    if use_live and LIVE_CSV.exists():
        df_feat = pd.read_csv(LIVE_CSV, parse_dates=["date"])
        src_label = "live data"
        mtime = datetime.datetime.fromtimestamp(LIVE_CSV.stat().st_mtime)
        st.caption(f"Data through **{df_feat['date'].iloc[-1].date()}** "
                   f"({len(df_feat)} rows) — {src_label} · "
                   f"file generated {mtime:%Y-%m-%d %H:%M}")
    else:
        df_feat = get_features()
        st.caption(f"Data through **{df_feat['date'].iloc[-1].date()}** "
                   f"({len(df_feat)} rows) — frozen dataset snapshot")

    if st.button("Predict", type="primary"):
        with st.spinner("Running model..."):
            r = predict_latest(task, model_name, df_feat)

        if task in ("direction_daily", "direction_weekly"):
            m1, m2 = st.columns(2)
            m1.metric("Prediction", r["prediction"])
            m2.metric("P(UP)", f"{r['prob_up']:.1%}")
            if abs(r["prob_up"] - 0.5) < 0.05:
                st.warning("Probability is very close to 50% — the model has "
                           "low conviction, consistent with near-random-walk "
                           "behavior of FX.")
        elif task == "vol_regression":
            m1, m2 = st.columns(2)
            m1.metric("Forecast volatility", f"{r['vol_forecast_pct']:.3f}% / day")
            m2.metric("Scaled [0-1]", f"{r['vol_scaled']:.4f}")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Prediction", r["prediction"])
            m2.metric("P(HIGH)", f"{r['prob_high']:.1%}")

        st.caption("Academic demo only — not financial advice.")

# ============================================================
# TAB 3 -- EDA
# ============================================================
with tab3:
    st.subheader("Exploratory Data Analysis")

    st.markdown("**Figures**")
    FIGS = {
        "Volatility features EDA":        "08_eda_volatility_features.png",
        "Volatility regression results":  "08_volatility_regression.png",
        "Volatility feature importance":  "08_vol_feature_importance.png",
        "All 10 direction models":        "07_all_10_models.png",
        "Daily models comparison":        "06_all_models_comparison.png",
        "Baseline ML comparison (NB05)":  "05_model_comparison.png",
    }
    choice = st.selectbox("Figure", list(FIGS.keys()))
    p = FIG_DIR / FIGS[choice]
    if p.exists():
        st.image(str(p), width="stretch")
    else:
        st.error(f"File not found: {p.name}")

    st.divider()
    st.markdown("**Result tables (CSV)**")
    csv_files = sorted(MODELS_DIR.glob("*.csv"))
    csv_choice = st.selectbox("Table", [f.name for f in csv_files])
    csv_path = MODELS_DIR / csv_choice
    df_csv = pd.read_csv(csv_path, index_col=0)
    st.dataframe(df_csv, width="stretch")
    st.download_button(
        "Download CSV",
        data=csv_path.read_bytes(),
        file_name=csv_choice,
        mime="text/csv",
    )