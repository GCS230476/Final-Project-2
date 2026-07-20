import datetime

import pandas as pd
import streamlit as st

from app_core import GRAPH_STYLE, LIVE_CSV, VOL_VMAX

# Heavy imports (torch) live only on this page, so the story pages load fast.
from predict_latest import REGISTRY, predict_latest, load_features

st.markdown(
    "The final chapter is the system itself: the frozen research models, "
    "wired to a data pipeline that refreshes every weekday morning without "
    "anyone touching it. Pick any of the twenty models and ask it about "
    "tomorrow."
)

# ---------------- live pipeline ----------------
st.subheader("How fresh data reaches this page")
st.graphviz_chart(f"""
digraph {{
    {GRAPH_STYLE}
    cron [label="GitHub Actions\\ncron 23:30 UTC Mon-Fri\\n(06:30 Vietnam, after the\\nFX daily candle closes)"];
    upd  [label="update_data.py\\npull yfinance / FRED / ECB / COT"];
    bld  [label="build_master_dataset()\\nleak fix applied automatically"];
    feat [label="recompute the same 31 features\\n(code identical to NB04)"];
    chk  [label="consistency check\\nlive vs frozen on overlapping days", fillcolor="#3a2f1b"];
    csv  [label="fx_features_live.csv\\ncommitted to the repo"];
    dep  [label="Streamlit Cloud\\nauto-redeploys on push"];
    cron -> upd -> bld -> feat -> chk -> csv -> dep;
}}
""")
st.markdown(
    "The amber node is the trust step: on every run, the freshly-built "
    "features are compared against the frozen research snapshot on all "
    "overlapping dates. Result: agreement to 1e-16 (bit-perfect) on all "
    "core columns; the only deviation ever observed was ≤ 0.07 in the "
    "rate-spread on the last five days of the snapshot — caused by "
    "publication lag of DGS2/ECBDFR at snapshot time, where the live "
    "value is the *more* correct one. The frozen `fx_features.csv` used "
    "for the chapter-8 results is never overwritten."
)

# ---------------- prediction ----------------
st.subheader("Ask a model about tomorrow")

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

use_live = st.toggle("Use live data (auto-updated pipeline)",
                     value=LIVE_CSV.exists(), disabled=not LIVE_CSV.exists())
if use_live and LIVE_CSV.exists():
    df_feat = pd.read_csv(LIVE_CSV, parse_dates=["date"])
    mtime = datetime.datetime.fromtimestamp(LIVE_CSV.stat().st_mtime)
    st.caption(f"Data through **{df_feat['date'].iloc[-1].date()}** "
               f"({len(df_feat):,} rows) — live pipeline · file generated "
               f"{mtime:%Y-%m-%d %H:%M}")
else:
    df_feat = load_features()
    st.caption(f"Data through **{df_feat['date'].iloc[-1].date()}** "
               f"({len(df_feat):,} rows) — frozen research snapshot")

if st.button("Predict", type="primary", icon=":material/bolt:"):
    with st.spinner("Running model..."):
        r = predict_latest(task, model_name, df_feat)

    if task in ("direction_daily", "direction_weekly"):
        m1, m2 = st.columns(2)
        m1.metric("Prediction", r["prediction"])
        m2.metric("P(UP)", f"{r['prob_up']:.1%}",
                  help="The model's probability, not a promise. "
                       "Chapter 8: even the best direction model is "
                       "right only ~54% of the time.")
        if abs(r["prob_up"] - 0.5) < 0.05:
            st.warning(
                "Probability within 5 points of a coin flip — the model "
                "has low conviction. For direction this is the *normal, "
                "honest* state: chapter 1 explains why a near-50% answer "
                "is what an efficient market should produce.",
                icon=":material/balance:",
            )
    elif task == "vol_regression":
        m1, m2 = st.columns(2)
        m1.metric("Forecast volatility", f"{r['vol_forecast_pct']:.3f}% / day",
                  help="Typical calm day ~0.2-0.3%; stormy episodes "
                       "0.6-1.0%. See the chapter-4 volatility chart "
                       "for context.")
        m2.metric("Scaled [0-1]", f"{r['vol_scaled']:.4f}",
                  help=f"1.0 on this scale = {VOL_VMAX:.4f} "
                       "(≈3.2%, the wildest day in training history).")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Prediction", r["prediction"])
        m2.metric("P(HIGH)", f"{r['prob_high']:.1%}",
                  help="Probability that tomorrow's |return| exceeds "
                       "the training-years median.")

    st.caption("Academic demonstration only — not financial advice, and "
               "chapter 8's limitations apply to every number above.")
