import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_core import (DARK_BG, LIT_CEILING, RENAISSANCE, load_master,
                      next_chapter)

# ---------------- abstract / introduction ----------------
st.markdown(
    "> This project asks whether machine learning can forecast the "
    "**EUR/USD** exchange rate, and answers it honestly. Using **22 "
    "financial and economic series** turned into **31 engineered "
    "features**, it trains **20 models** (5 algorithms × 4 tasks) to "
    "predict two different things about tomorrow: the **direction** of the "
    "move and its **volatility**. The headline finding is a deliberate "
    "contrast — :red[**direction is essentially unpredictable**] "
    "(~50–54%, at the coin-flip wall the Efficient Market Hypothesis "
    "predicts), while :green[**volatility is genuinely forecastable**] "
    "(beats every naive baseline). Along the way, the project uncovered "
    "and fixed a **data-leak bug** that had faked 86% accuracy — that "
    "investigation, in chapter 3, is the core research contribution."
)

with st.container(border=True):
    a, b, c, d = st.columns(4)
    a.metric("Data series", "22", help="From Yahoo Finance, FRED, ECB, "
             "CFTC — four independent sources.")
    b.metric("Engineered features", "31", help="In 6 groups, chapter 5.")
    c.metric("Models trained", "20", help="RF, XGBoost, LightGBM, LSTM, "
             "GRU × 4 tasks.")
    d.metric("Years of data", "16", help="2010 to today, ~4,150 "
             "trading days.")

st.markdown(
    "**How to read this dashboard** — the sidebar is a table of contents. "
    "The nine chapters run in the order the work actually happened: the "
    "question, the data, the leak investigation, exploration, feature "
    "engineering, methodology, the models, the results, and a live demo. "
    "Every chart carries a :blue-badge[what you see] and a "
    ":green-badge[why it matters] note. Read straight through, or jump."
)

st.markdown(
    "This first chapter shows why the *naive* question — 'what will "
    "EUR/USD be tomorrow?' — is unanswerable, and how it was reshaped "
    "into three questions that **can** be answered."
)

# ---------------- market snapshot ----------------
dfm = load_master()
if dfm is None:
    st.error("Master dataset not found. Run update_data.py first.")
    st.stop()

price = dfm["eurusd"]
logret = np.log(price).diff()
vol20 = logret.rolling(20).std() * 100

st.subheader("The subject: one number the whole world argues about")

k1, k2, k3, k4 = st.columns(4)
chg = (price.iloc[-1] / price.iloc[-2] - 1) * 100
k1.metric("EUR/USD (last close)", f"{price.iloc[-1]:.4f}", f"{chg:+.2f}%",
          help="How many US dollars one euro buys.")
k2.metric("Realized vol (20d)", f"{vol20.iloc[-1]:.2f}% / day",
          help="Standard deviation of daily log returns over the last "
               "20 trading days. The 'wind speed' of the market.")
wk = (price.iloc[-1] / price.iloc[-6] - 1) * 100 if len(price) > 6 else 0
k3.metric("5-day change", f"{wk:+.2f}%")
if "vixcls" in dfm.columns and dfm["vixcls"].notna().any():
    vix = dfm["vixcls"].dropna()
    k4.metric("VIX (last)", f"{vix.iloc[-1]:.1f}",
              f"{vix.iloc[-1] - vix.iloc[-2]:+.1f}",
              help="The market's 'fear index'. Above 20 = stressed regime.")
st.caption(f"Data through **{dfm['date'].iloc[-1].date()}** · "
           f"{len(dfm):,} trading days")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dfm["date"], y=price, mode="lines",
    line=dict(color="#ffb400", width=1.3), name="EUR/USD",
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
))
fig.update_layout(
    title="EUR/USD — daily close since 2010",
    template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    height=420, margin=dict(l=40, r=20, t=60, b=30),
    xaxis=dict(rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#232a3a", activecolor="#ffb400",
        font=dict(color="#e6e9ef"),
    )),
)
st.plotly_chart(fig)
st.markdown(
    ":material/visibility: **What you are looking at** — sixteen years of "
    "daily closing prices: 1.45 in 2011, near parity (1.00) in late 2022, "
    "around 1.15 today. The level wanders for years without any fixed "
    "average — statisticians call this **non-stationary**. "
    ":material/lightbulb: **Why it matters** — a series that drifts like "
    "this cannot be predicted at the *level*: guessing 'tomorrow equals "
    "today' is almost always nearly right and completely useless. So the "
    "project does not predict the price."
)

# ---------------- log returns ----------------
st.subheader("What we actually predict: the log return")

c1, c2 = st.columns([3, 2])
with c1:
    st.latex(r"r_t \;=\; \ln\!\left(\frac{P_t}{P_{t-1}}\right)")
    st.markdown(
        "- Price goes **up** → ratio > 1 → log return is **positive**\n"
        "- Price goes **down** → ratio < 1 → log return is **negative**\n"
        "- For small daily moves, the log return is almost exactly the "
        "percentage change — and unlike simple percentages, log returns "
        "**add up across days** (a 5-day return is the sum of five daily "
        "ones), and the return series is **stationary**: its mean and "
        "variance stay put, which is what models need to learn from."
    )
with c2:
    st.markdown("**Worked example**")
    st.table(pd.DataFrame({
        "Day": ["Tuesday", "Wednesday"],
        "Close": [1.1735, 1.1712],
    }).set_index("Day"))
    st.markdown(
        "ln(1.1712 / 1.1735) = **−0.00196 ≈ −0.2%**. "
        "Negative sign = the euro fell. A model that predicted DOWN "
        "for Wednesday was right."
    )

# ---------------- the wall ----------------
st.subheader("The wall: efficient markets")
st.markdown(
    "Tomorrow's return has two parts — a **sign** (up or down) and a "
    "**magnitude** (how big). They have opposite fates.\n\n"
    "The **Efficient Market Hypothesis (EMH)** says public information is "
    "absorbed into prices almost instantly. FX is the deepest market on "
    "earth (~7.5 trillion USD per day): any repeatable pattern about "
    "*direction* gets traded away by thousands of funds the moment it "
    "appears. Directional signals **self-destruct on discovery**, which "
    "leaves the daily sign close to a coin flip — a **random walk**."
)
m1, m2 = st.columns(2)
m1.metric("Published ceiling for this exact task", f"{LIT_CEILING}%",
          help="Guyard & Deriaz (2024): 21 algorithms plus stacking on "
               "daily EUR/USD direction. None beat 58.52%.")
m2.metric("Renaissance Technologies' win rate", f"{RENAISSANCE}%",
          help="The most successful quant fund in history is right on "
               "about half of its trades. Its edge is tiny but applied "
               "millions of times.")
st.markdown(
    "But EMH does not erase everything. Knowing that *tomorrow will be "
    "turbulent* tells you nothing about which way to trade — so that "
    "pattern is not traded away. Volatility **clusters**: stormy days "
    "follow stormy days, calm follows calm (Mandelbrot, 1963). Think "
    "wind: nobody can predict tomorrow's wind *direction*, but after a "
    "storm everyone correctly expects strong wind to continue."
)

# ---------------- three problems ----------------
st.subheader("So one impossible problem becomes three honest ones")

p1, p2, p3 = st.columns(3)
with p1:
    with st.container(border=True):
        st.markdown("**Problem 1 — Direction** (classification)")
        st.latex(r"y_t = \mathbb{1}\,[\,r_{t+1} > 0\,]")
        st.markdown(
            "Predict UP or DOWN for tomorrow (and for the week ahead). "
            "**Expected outcome: controlled failure** (~50–54%, ceiling "
            "~58%). This is the control group of the experiment."
        )
with p2:
    with st.container(border=True):
        st.markdown("**Problem 2 — Volatility** (regression)")
        st.latex(r"y_t = |\,r_{t+1}\,|")
        st.markdown(
            "Predict the *size* of tomorrow's move as a number (%/day). "
            "**Expected outcome: success** — beat the naive baseline "
            "on MAE, achieve R² > 0 out of sample."
        )
with p3:
    with st.container(border=True):
        st.markdown("**Problem 3 — Volatility** (classification)")
        st.latex(r"y_t = \mathbb{1}\,[\,|r_{t+1}| > \mathrm{med}\,]")
        st.markdown(
            "An easier version of problem 2: will tomorrow be a HIGH-vol "
            "or LOW-vol day? Threshold = median |return| of the training "
            "years. **Expected outcome: a few points above baseline.**"
        )

st.info(
    "**Central thesis of the project** — the same 31 features, the same "
    "data, the same five algorithms are given all three problems. "
    "Direction fails while volatility succeeds. Because everything else "
    "is held equal, that contrast is *evidence about the market itself*: "
    "efficiency eats directional signal, but leaves the inertia of "
    "volatility behind.",
    icon=":material/science:",
)

next_chapter("app_pages/02_data.py", "The data — 22 series from 4 sources")
