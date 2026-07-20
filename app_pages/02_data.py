import pandas as pd
import streamlit as st

from app_core import FIG_EDA, GRAPH_STYLE, fig_card, load_master, next_chapter

st.markdown(
    "A model is only as honest as its data. This chapter covers what was "
    "collected, why each series earned its place, and how four sources with "
    "four different rhythms were merged into one table — without letting "
    "information from the future leak into the past."
)

# ---------------- why these series ----------------
st.subheader("Why these 22 series? Five economic channels")
st.markdown(
    "An exchange rate is the *relative price of two currencies*. Economic "
    "theory names the forces that move it; every series in the dataset "
    "represents one of those channels — nothing was collected just to make "
    "the feature count look impressive."
)

channels = pd.DataFrame({
    "Channel": [
        "Interest-rate differential",
        "Inflation differential",
        "Risk sentiment",
        "Broad USD strength",
        "Positioning and macro health",
    ],
    "Logic": [
        "Money flows to the currency that pays more. If US rates beat "
        "eurozone rates, investors sell EUR to hold USD.",
        "Inflation erodes purchasing power; the high-inflation currency "
        "loses value over time.",
        "In a panic, capital runs to the USD as a safe haven.",
        "EUR is ~58% of the dollar index basket — USD strength is "
        "almost mechanically EUR weakness.",
        "What big speculators are betting (COT), and the health of the "
        "US economy behind the dollar.",
    ],
    "Series": [
        "dff, dgs2, dgs10 (US) · ecbdfr, irltlt01ezm156n, euribor_3m, "
        "ester_overnight (EU)",
        "cpiaucsl (US CPI) · cp0000ez19m086nest (EU HICP) · t10yie "
        "(expected inflation)",
        "vixcls (fear index) · gold · oil",
        "dxy (Dollar Index) · dtwexbgs (Fed broad dollar)",
        "eur_net_position, eur_net_position_pct (CFTC COT) · unrate, "
        "payems · dexuseu, eurusd_official (reference EUR/USD)",
    ],
})
st.dataframe(channels, hide_index=True, width="stretch")
st.caption(
    "The two 'official' EUR/USD series (Fed's dexuseu, ECB's "
    "eurusd_official) are not features for prediction — they are "
    "*witnesses*. Chapter 3 shows how they saved the project."
)

# ---------------- sources ----------------
st.subheader("Four sources, four rhythms")
sources = pd.DataFrame({
    "Source": ["Yahoo Finance", "FRED (St. Louis Fed)", "ECB Data Portal",
               "CFTC (COT report)"],
    "Series": ["eurusd, dxy, gold, oil", "13 series (rates, inflation, "
               "labor, VIX, dollar indexes)", "4 series (official EUR/USD, "
               "effective rate, euribor, ester)", "2 series (speculator "
               "net positioning)"],
    "Frequency": ["Daily", "Daily / monthly", "Daily", "Weekly"],
    "Loader": ["src/data/load_yfinance.py", "src/data/load_fred.py",
               "src/data/load_ecb.py", "src/data/load_cot.py"],
})
st.dataframe(sources, hide_index=True, width="stretch")

# ---------------- pipeline ----------------
st.subheader("The pipeline")
st.graphviz_chart(f"""
digraph {{
    {GRAPH_STYLE}
    yf   [label="Yahoo Finance\\n4 price series"];
    fred [label="FRED\\n13 macro series"];
    ecb  [label="ECB\\n4 rate series"];
    cot  [label="CFTC COT\\n2 positioning series"];
    build [label="build_master_dataset()\\n· yfinance calendar = backbone\\n· left-merge FRED / ECB / COT\\n· forward-fill non-daily series\\n· ALIGNMENT FIX (chapter 3)", fillcolor="#3a2f1b"];
    master [label="fx_master_dataset.csv\\n~4,150 rows x 24 cols"];
    feats [label="Feature engineering (NB04)\\n31 features + 2 targets"];
    fx    [label="fx_features.csv\\n(frozen snapshot)"];
    models [label="20 trained models\\n4 tasks x 5 algorithms"];
    app   [label="This dashboard"];
    actions [label="GitHub Actions\\ncron, weekday mornings", fillcolor="#1b3a2f"];
    update  [label="update_data.py\\nre-pull + rebuild + consistency check", fillcolor="#1b3a2f"];
    live    [label="fx_features_live.csv", fillcolor="#1b3a2f"];

    yf -> build; fred -> build; ecb -> build; cot -> build;
    build -> master -> feats -> fx -> models -> app;
    actions -> update -> live -> app;
    update -> build [style=dashed, label="reuses"];
}}
""")
st.caption(
    "Amber node = where the data-leak fix lives; green nodes = the live "
    "pipeline (chapter 9). Every fresh data pull is forced through "
    "build_master_dataset(), so the fix can never be forgotten."
)

# ---------------- merging ----------------
st.subheader("Merging four rhythms into one daily table")
st.markdown(
    "- **Backbone**: the days on which all four Yahoo price series exist "
    "(an *inner* merge — if any core price is missing, it is not a proper "
    "trading day and the row is dropped).\n"
    "- **Left-merge** everything else onto that backbone: every trading "
    "day keeps its row; a source that has no value that day contributes "
    "an empty cell — monthly CPI only 'exists' one day per month.\n"
    "- **Forward-fill** the gaps: carry the last published value forward. "
    "On any day T the table then contains exactly what a trader standing "
    "on day T actually knew — this morning's newspaper stays the latest "
    "news until tomorrow's is printed."
)
st.warning(
    "The tempting alternative — backward-fill — would drag *future* "
    "releases into the past: the row for June 5 would contain CPI "
    "published June 10. That is **look-ahead bias**, a form of data "
    "leakage: the model looks brilliant in backtests and goes blind in "
    "production. Forward-fill is the honest choice, and it is applied "
    "to every non-price column.",
    icon=":material/history:",
)

# ---------------- data quality ----------------
st.subheader("Data quality checks")
fig_card(
    FIG_EDA / "01_yearly_coverage.png",
    what="the number of rows (trading days) per year — a steady ~250 "
         "every year, mean 242, with the current year partial.",
    why="a flat profile means no silent gaps: models never train on a "
        "year that is accidentally half-empty. This is the first thing "
        "to check before trusting any time series.",
)
fig_card(
    FIG_EDA / "01_missing_by_column.png",
    what="missing values per column *before* forward-fill. One outlier: "
         "ester_overnight is empty for 59.5% of history — the €STR rate "
         "simply did not exist before October 2019. Everything else is "
         "missing at most 19 cells (0.5%).",
    why="missingness must be explained, not just filled. Here the only "
        "large gap has a documented real-world cause (the series launch "
        "date), not a broken download.",
)

dfm = load_master()
if dfm is not None:
    st.success(
        f"Result: one table, **{len(dfm):,} rows x {dfm.shape[1]} "
        f"columns**, {dfm['date'].iloc[0].date()} to "
        f"{dfm['date'].iloc[-1].date()} — data/interim/"
        f"fx_master_dataset.csv. This file feeds everything downstream.",
        icon=":material/check_circle:",
    )

next_chapter("app_pages/03_leak.py",
             "The leak investigation — how 86% accuracy exposed a bug")
