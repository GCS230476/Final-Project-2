import pandas as pd
import streamlit as st

from app_core import (FIG_EDA, FIG_FEATURES, FIG_MODELS, fig_card,
                      load_features_frozen, next_chapter)

st.markdown(
    "Raw series are not what models eat. Chapter 4 showed levels are "
    "non-stationary, so every input is transformed into something stable — "
    "then enriched with memory (lags), context (rolling windows) and "
    "domain knowledge (engineered spreads). The result: **31 features in "
    "6 groups**, each with a stated reason to exist."
)

# ---------------- the six groups ----------------
st.subheader("The 31 features, group by group")

groups = [
    ("G1 · Log returns (4)",
     "eurusd_return, dxy_return, gold_return, oil_return",
     "np.log(p / p.shift(1))",
     "Today's move in the four core markets. The stationary heartbeat of "
     "the dataset — and the raw material for most other groups."),
    ("G2 · First differences (7)",
     "vixcls_diff, dgs2_diff, dgs10_diff, ecbdfr_diff, t10yie_diff, "
     "eur_effective_rate_diff, dtwexbgs_diff",
     "x - x.shift(1)",
     "Rates and indexes drift like prices, so we feed their daily *change*. "
     "A rate hike shows up as a spike in the diff, which is the actual "
     "news event."),
    ("G3 · Raw levels (2)",
     "vixcls_level, eur_net_position_pct",
     "unchanged",
     "Two exceptions where the level itself is meaningful and roughly "
     "bounded: VIX at 30 means fear regardless of yesterday, and COT "
     "positioning is already expressed as a percentage."),
    ("G4 · Lags (7)",
     "eurusd_return lag1/2/3 · dxy_return lag1/2 · vixcls_diff_lag1 · "
     "gold_return_lag1",
     "x.shift(k)",
     "Explicit short-term memory for models that see one row at a time "
     "(the tree models). The RNNs get memory from their 10-day window "
     "instead; the trees get it from these columns."),
    ("G5 · Rolling stats (7)",
     "eurusd_return ma5/ma10/ma20 · vol5/vol10/vol20 · dxy_return_vol10",
     "rolling mean / rolling std",
     "The ma* columns summarize recent trend; the vol* columns are "
     "*realized volatility* — the stars of the show. vol20 is literally "
     "'how stormy were the last 20 days', the direct carrier of the "
     "clustering signal from chapter 4."),
    ("G6 · Engineered (4)",
     "us_eu_rate_spread · us_eu_inflation_diff · vix_regime · day_of_week",
     "dgs2 - ecbdfr · cpiaucsl - cp0000... · vix > 20 · calendar",
     "Domain knowledge compressed into single columns: the rate spread "
     "*is* the carry-trade channel, the inflation gap *is* the "
     "purchasing-power channel, vix_regime flags calm vs stressed "
     "markets, day_of_week lets models notice weekday effects."),
]
for name, cols, formula, why in groups:
    with st.expander(name, icon=":material/category:"):
        st.markdown(f"**Columns:** `{cols}`")
        st.markdown(f"**Formula:** `{formula}`")
        st.markdown(f"**Why it exists:** {why}")

df_feat = load_features_frozen()
if df_feat is not None:
    n_feat = len([c for c in df_feat.columns
                  if c not in ("date", "target_return_next_day",
                               "target_direction")])
    st.caption(
        f"Sanity check straight from data/processed/fx_features.csv: "
        f"**{n_feat} feature columns**, {len(df_feat):,} rows, plus the "
        f"two targets (target_return_next_day = eurusd_return.shift(-1), "
        f"target_direction = its sign)."
    )

st.warning(
    "**Leak discipline** — every feature in row t is computed from "
    "information available at day t or earlier: shifts look backward, "
    "rolling windows end at t, and the two slow inflation series arrive "
    "forward-filled from their last release. The *only* column allowed "
    "to touch the future is the target itself.",
    icon=":material/lock:",
)

# ---------------- relationships between features ----------------
st.subheader("How the features relate to each other")
fig_card(
    FIG_EDA / "03_multicollinearity.png",
    what="absolute correlation between every pair of candidate inputs. "
         "Four hot spots: dxy vs dtwexbgs (0.77 — two dollar indexes), "
         "dgs2 vs dgs10 (0.77 — two points on the same yield curve), "
         "US vs EU inflation (0.74), and two near-duplicates: "
         "payems vs unrate (0.99) and ecbdfr vs ester (1.00).",
    why="highly-correlated features carry overlapping information "
        "(multicollinearity). Tree models are robust to it, but it has "
        "one visible side effect in the next figure: shared credit. "
        "When two features say the same thing, importance scores split "
        "between them, so no single column can look dominant.",
)

# ---------------- importance: the contrast ----------------
st.subheader("Which features matter? Depends on the question")
st.markdown(
    "The most instructive experiment of the project: ask the *same* "
    "Random Forest machinery which features matter for **direction** "
    "versus for **volatility**, and watch the answers diverge."
)
fig_card(
    FIG_FEATURES / "04_feature_importance.png",
    what="Random Forest feature importance for the *direction* task. "
         "The top-15 scores sit in a flat band between 0.038 and 0.046 "
         "— oil_return, gold_return_lag1, vixcls_diff_lag1, EUR/USD's "
         "own lags... nobody stands out; importance is spread thin "
         "across all 31 columns.",
    why="a flat importance profile is what 'no real signal' looks like: "
        "when nothing predicts the target, the forest distributes "
        "credit almost uniformly over noise. The market has eaten the "
        "directional information — exactly what EMH promised.",
    title="Direction: importance spread thin — nobody knows tomorrow's sign",
)
fig_card(
    FIG_MODELS / "08_vol_feature_importance.png",
    what="XGBoost feature importance for the *volatility* task. One "
         "feature towers: eurusd_return_vol20 at ~0.117, roughly triple "
         "the runner-up, with vol5, dxy_return_vol10 and vol10 (the "
         "dark bars — all realized-volatility features) plus vix_regime "
         "and the macro spreads filling the top rows.",
    why="this is volatility clustering made visible inside a model: "
        "'how stormy were the last 20 days' is by far the best "
        "predictor of how stormy tomorrow will be. The contrast with "
        "the flat direction profile above is the central thesis shown "
        "through the models' own eyes.",
    title="Volatility: one feature towers — yesterday's storm predicts "
          "tomorrow's",
)
fig_card(
    FIG_MODELS / "08_eda_volatility_features.png",
    what="left: the distribution of the volatility target (scaled 0–1) "
         "— right-skewed, most days calm (mean 0.132), a long tail of "
         "wild ones. Right: correlation of the top features with that "
         "target — the vol* family reaches |r| ≈ 0.24–0.28, followed "
         "by the inflation gap, rate spread and VIX level.",
    why="0.28 sounds modest, but compare it with direction where "
        "feature-target correlations are ~0: for volatility there is "
        "*actual signal* to harvest before any model is trained. "
        "Modest-but-real is the theme of every honest result in this "
        "project.",
)

# ---------------- consequences for modeling ----------------
st.subheader("How the features shaped the model choices")
st.markdown(
    "- **Tabular features with explicit lags** → gradient-boosted trees "
    "and random forests are natural candidates; they eat raw, unscaled "
    "columns (trees only ask 'is x > threshold?', so units are "
    "irrelevant).\n"
    "- **Sequential structure** (10 consecutive days) → recurrent "
    "networks (LSTM, GRU) get the same features as ordered windows, "
    "standardized first because neural nets are sensitive to scale.\n"
    "- **Multicollinearity** → importance charts are read as evidence "
    "*about signal existence*, not as precise rankings; no linear "
    "regression coefficients are interpreted.\n"
    "- **Weak signal environment** → small architectures with heavy "
    "regularization; chapter 7 shows what happened to the models that "
    "were too big for the signal."
)

next_chapter("app_pages/06_method.py",
             "Methodology — splits, baselines, and the rules of the game")
