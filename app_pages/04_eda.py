import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_core import DARK_BG, FIG_EDA, fig_card, load_master, next_chapter

st.markdown(
    "With clean, correctly-aligned data, exploratory analysis asks three "
    "questions: does the target contain any structure at all, what kind of "
    "structure, and which inputs might carry it? The answers below predict "
    "the modeling outcome *before a single model is trained* — direction "
    "will be nearly impossible, volatility will be learnable."
)

# ---------------- stationarity ----------------
st.subheader("Step 1 — from a wandering price to a stable target")
fig_card(
    FIG_EDA / "02_price_vs_returns.png",
    what="the same asset seen two ways. Top: the price, drifting through "
         "multi-year trends and regimes with no fixed mean — "
         "non-stationary. Bottom: daily log returns — a flat band around "
         "zero with roughly constant spread — stationary.",
    why="models learn a mapping from past to future; that only works if "
        "the statistical rules of the game stay constant. Every feature "
        "and target in this project is therefore built from returns and "
        "differences, never from raw levels.",
)

# ---------------- distribution ----------------
st.subheader("Step 2 — returns are not 'nice': fat tails")
fig_card(
    FIG_EDA / "02_return_distribution.png",
    what="the return histogram against a fitted normal curve (left), the "
         "same on a log scale (middle), and a Q-Q plot (right). Excess "
         "kurtosis is 2.21 (a normal distribution has 0); on the log "
         "scale the bars sit visibly above the red curve far from "
         "center; in the Q-Q plot the points bend away from the line "
         "at both ends.",
    why="all three views say the same thing: **extreme days happen far "
        "more often than the bell curve promises** — these are the fat "
        "tails. Big moves are not freak accidents, they are a regular "
        "feature of FX. Predicting *when the market gets wild* — "
        "volatility — is therefore a question worth asking, and models "
        "that assume normality will understate risk.",
)

# ---------------- memory ----------------
st.subheader("Step 3 — the key experiment: where is the memory?")
fig_card(
    FIG_EDA / "02_acf_pacf.png",
    what="autocorrelation (ACF): how much a series correlates with its "
         "own past at lags 1–30. Top row: raw returns — every bar sits "
         "inside the blue significance band, indistinguishable from "
         "white noise. Bottom row: *squared* returns (a proxy for "
         "volatility) — dozens of bars poke above the band, positive "
         "correlation persisting past 30 days.",
    why="this single figure is the entire thesis in statistical form. "
        "Yesterday's return says nothing about tomorrow's return "
        "(random walk → direction is a coin flip), but yesterday's "
        "*magnitude* echoes for a month (volatility clustering → "
        "magnitude is forecastable). The 3-problem design of this "
        "project is a direct consequence of these two panels.",
)
fig_card(
    FIG_EDA / "02_volatility_clustering.png",
    what="returns (top) and squared returns (bottom) through time. The "
         "bottom panel is spiky but *grouped*: the 2010–12 euro debt "
         "crisis, 2015–16, the 2020 covid shock, the 2022 parity year — "
         "turbulence arrives in episodes, calm in long stretches.",
    why="this is what Mandelbrot saw in 1963: volatility clusters. It "
        "is the physical pattern that problems 2 and 3 are built to "
        "exploit — and unlike directional patterns it cannot be traded "
        "away, because knowing tomorrow is stormy does not say which "
        "way the wind blows.",
)

# ---------------- interactive ----------------
st.subheader("The same clustering, live in the current data")
dfm = load_master()
if dfm is not None:
    logret = np.log(dfm["eurusd"]).diff()
    vol20 = logret.rolling(20).std() * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dfm["date"], y=vol20, mode="lines",
        line=dict(color="#4f8df5", width=1.1),
        fill="tozeroy", fillcolor="rgba(79,141,245,0.15)",
        name="20d realized vol",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%/day<extra></extra>",
    ))
    fig2.update_layout(
        title="Realized volatility (20-day rolling), %/day — zoom into any "
              "episode",
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=320, margin=dict(l=40, r=20, t=60, b=30),
    )
    st.plotly_chart(fig2)
    st.markdown(
        ":material/visibility: **What you are looking at** — the rolling "
        "20-day standard deviation of daily returns, the 'wind speed' "
        "series the volatility models target. It moves in slow waves "
        "between roughly 0.2 and 1.0 %/day rather than jumping randomly. "
        ":material/lightbulb: **Why it matters** — a series with slow "
        "waves is exactly the kind of series yesterday's value helps "
        "predict. This chart updates with the live data pipeline."
    )

st.info(
    "**EDA verdict** — returns: stationary, fat-tailed, memoryless → "
    "expect direction models to hover near the coin flip. Squared "
    "returns: strong, month-long memory → expect volatility models to "
    "beat their baseline. Chapter 8 tests exactly these two predictions.",
    icon=":material/fact_check:",
)

next_chapter("app_pages/05_features.py",
             "Feature engineering — 31 features in 6 groups")
