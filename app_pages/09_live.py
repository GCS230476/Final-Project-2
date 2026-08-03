import datetime

import pandas as pd
import streamlit as st

from app_core import (FIG_REPORT, LIVE_CSV, NAIVE_PRICE, VOL_VMAX, load_master,
                      load_price)

# Heavy imports (torch) live only on this page, so the story pages load fast.
from predict_latest import REGISTRY, load_features, predict_latest, predict_price

st.markdown(
    "The final chapter is the system itself: the frozen research models, wired "
    "to a data pipeline that refreshes every weekday morning without anyone "
    "touching it. Pick a model and a tolerance, and every question the project "
    "asked is answered at once for the next session."
)

# ---------------- live pipeline ----------------
st.subheader("How fresh data reaches this page")
_, _mid, _ = st.columns([1, 2, 1])
with _mid:
    st.image(str(FIG_REPORT / "pipeline_daily.png"), width="stretch")
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

# ---------------- controls ----------------
st.subheader("Ask every model about the next session")

c1, c2 = st.columns([1, 2])
with c1:
    model_name = st.selectbox(
        "Model", list(REGISTRY["direction_daily"].keys()),
        help="Drives the direction and volatility answers. The price "
             "forecaster is a single champion chosen on validation in "
             "notebook 09, so it does not change with this setting.")
with c2:
    tol = st.segmented_control(
        "Tolerance for the price forecast (pips)",
        options=[30, 40, 50, 60, 80, 100], default=60,
        format_func=lambda v: f"±{v}",
        help="A price forecast counts as correct when it lands this close to "
             "the actual close. 1 pip = 0.0001.")
tol = tol or 60

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

# ---------------- prediction ----------------
if st.button("Predict the next session", type="primary",
             icon=":material/bolt:"):
    with st.spinner("Running every model..."):
        master = load_master()
        last_close = (float(master["eurusd"].dropna().iloc[-1])
                      if master is not None else None)
        px = predict_price(df_feat, last_close)
        out = {t: predict_latest(t, model_name, df_feat)
               for t in ("direction_daily", "direction_weekly",
                         "vol_regression", "vol_classification")}

    # ---- price -------------------------------------------------------
    st.markdown("#### :orange[Price — the level for the next close]")
    lo, hi = px["forecast"] - tol / 10000, px["forecast"] + tol / 10000
    p1, p2, p3 = st.columns(3)
    p1.metric("Today's close", f"{px['today']:.4f}")
    p2.metric("Forecast next close", f"{px['forecast']:.4f}",
              f"{px['move_pips']:+.1f} pip")
    # as a share of the actual price, not the /100 shorthand: at 1.14 a
    # 60-pip band is 0.53% of price, not 0.60%
    pct = 100 * (tol / 10000) / px["today"]
    p3.metric("Tolerance applied", f"±{tol} pip", f"±{pct:.2f}% of price",
              delta_color="off")
    st.success(f"**Forecast range: {lo:.4f} – {hi:.4f}** "
               f"(centre {px['forecast']:.4f}, ±{tol} pip) · model "
               f"{px['model']}", icon=":material/insights:")

    price = load_price()
    if price is not None:
        row = price["acc"]
        row = row[(row["tol_pip"] == tol)].set_index("model")
        if px["model"] in row.index and NAIVE_PRICE in row.index:
            a1, a2 = st.columns(2)
            a1.metric(f"How often this lands within ±{tol} pip (test)",
                      f"{row.loc[px['model'], 'test']:.1f}%")
            a2.metric("Same tolerance, 'tomorrow = today'",
                      f"{row.loc[NAIVE_PRICE, 'test']:.1f}%",
                      f"{row.loc[px['model'], 'test'] - row.loc[NAIVE_PRICE, 'test']:+.1f} pp",
                      delta_color="off")
            st.warning(
                "The naive rule scores the same. Chapter 8 explains why: the "
                "next close is essentially a random walk, so the tolerance is "
                "doing the work here, not the algorithm.",
                icon=":material/priority_high:")

    st.divider()

    # ---- direction ---------------------------------------------------
    st.markdown("#### :orange[Direction — which way it moves]")
    d1, d2 = st.columns(2)
    for col, key, label in [(d1, "direction_daily", "Next day"),
                            (d2, "direction_weekly", "Next week")]:
        r = out[key]
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.metric("Prediction", r["prediction"])
                st.metric("P(UP)", f"{r['prob_up']:.1%}")
    if min(abs(out[k]["prob_up"] - 0.5) for k in
           ("direction_daily", "direction_weekly")) < 0.05:
        st.warning(
            "At least one probability sits within 5 points of a coin flip — "
            "low conviction. For direction this is the *normal, honest* "
            "state: chapter 1 explains why a near-50% answer is what an "
            "efficient market should produce.",
            icon=":material/balance:")

    st.divider()

    # ---- volatility --------------------------------------------------
    st.markdown("#### :orange[Volatility — how much it moves]")
    reg, clf = out["vol_regression"], out["vol_classification"]
    h = reg.get("horizon_days", 1)
    v1, v2, v3 = st.columns(3)
    v1.metric(f"Forecast volatility, next {h} days",
              f"{reg['vol_forecast_pct']:.3f}% / day",
              help="The average daily move expected over the next "
                   f"{h} trading days. Typical calm stretch ~0.2-0.3%; "
                   "stormy episodes 0.6-1.0%.")
    v2.metric("Scaled [0-1]", f"{reg['vol_scaled']:.4f}",
              help=f"1.0 on this scale = {VOL_VMAX:.4f}, the most volatile "
                   "stretch in the training years.")
    v3.metric("Regime", clf["prediction"],
              f"P(HIGH) {clf['prob_high']:.1%}", delta_color="off")
    st.info(
        "This is the one block on the page with a measurable edge — chapter 8 "
        "shows the volatility models beating their baseline consistently, "
        "while direction and price level do not.",
        icon=":material/insights:")

    st.caption(f"All answers from **{model_name}** (price: {px['model']}), "
               f"data as of **{px['as_of']}**. Academic demonstration only — "
               "not financial advice, and chapter 8's limitations apply to "
               "every number above.")
