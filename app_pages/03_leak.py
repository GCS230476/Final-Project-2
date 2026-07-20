import numpy as np
import pandas as pd
import streamlit as st

from app_core import FIG_EDA, LIT_CEILING, fig_card, load_master, next_chapter

st.markdown(
    "This is the most important chapter of the project. The first direction "
    "model scored far *too well* — and finding out why turned into a "
    "detective story about timestamps. Everything after this chapter is "
    "only trustworthy because of what was found here."
)

# ---------------- symptom ----------------
st.subheader("The symptom: a result too good to be true")
c1, c2 = st.columns(2)
c1.metric("First direction model (accuracy)", "~86%")
c2.metric("Published ceiling for the same task", f"{LIT_CEILING}%",
          delta="-27 pp below our 'result'", delta_color="off")
st.markdown(
    "A student model beating 21 published algorithms — and the entire "
    "quantitative finance industry — by nearly 30 points is not a "
    "triumph; it is an alarm. If 86% were real, this would not be a "
    "capstone, it would be a hedge fund. **A suspiciously beautiful "
    "number is a smoke detector, not a gift.** The prime suspect for any "
    "too-good result is data leakage: some path by which the answer "
    "sneaks into the question."
)

# ---------------- investigation ----------------
st.subheader("The investigation: two correlations that made no sense")
st.markdown(
    "The model leaned almost entirely on one feature: `dxy_return`, the "
    "Dollar Index. Two diagnostic correlations were computed. Correlation "
    "runs from −1 to +1 and measures whether two series move in step; "
    "since EUR is ~58% of the DXY basket, finance says the *same-day* "
    "correlation between EUR/USD and DXY must be strongly negative, "
    "around −0.8."
)
diag = pd.DataFrame({
    "Diagnostic": [
        "corr(EUR/USD return, DXY return) — same day",
        "corr(DXY return today, EUR/USD return tomorrow)",
    ],
    "Theory says": ["≈ −0.8 (mechanical link)", "≈ 0 (EMH: no free prophecy)"],
    "Measured (before fix)": ["−0.11  — almost unrelated?!",
                              "−0.82  — a perfect crystal ball?!"],
    "Verdict": ["Impossibly weak", "Impossibly strong"],
})
st.dataframe(diag, hide_index=True, width="stretch")
st.markdown(
    "Each number sat exactly where the *other* belonged. Two series that "
    "swap their same-day and next-day relationships are shouting one "
    "thing: **they are misaligned by one day.**"
)

fig_card(
    FIG_EDA / "03_correlation_heatmap.png",
    what="the full correlation matrix of daily returns and changes, "
         "computed *before* the fix. The crime scene: eurusd_ret vs "
         "dxy_ret reads −0.11, and eurusd_ret vs the ECB's official "
         "eurusd_official_ret reads just 0.56 — two versions of the "
         "same exchange rate agreeing only halfway. Meanwhile "
         "dexuseu_ret (Fed's EUR/USD) vs dxy_ret shows the expected "
         "−0.84.",
    why="the official series behave correctly with each other; only "
        "Yahoo's EUR/USD disagrees with everyone. That isolates the "
        "suspect: it is not the market, it is one vendor's timestamps.",
    title="Exhibit A — the correlation matrix (pre-fix)",
)
fig_card(
    FIG_EDA / "03_lead_lag.png",
    what="correlation between each feature and Yahoo's EUR/USD bar at "
         "shifts of −5 to +5 days. For dxy_ret the giant −0.82 bar "
         "sits at lag −1: *yesterday's* DXY matches *today's* "
         "EUR/USD bar almost perfectly.",
    why="a same-day mechanical relationship appearing at a one-day "
        "offset is the fingerprint of a timestamp error: the bar "
        "Yahoo labels 'day t' actually contains the move of day t−1.",
    title="Exhibit B — the lead-lag scan",
)
fig_card(
    FIG_EDA / "03_mutual_information.png",
    what="mutual information — a measure of how much knowing one "
         "variable tells you about another, linear or not — between "
         "each feature and the 'next-day' target. dxy_ret scores an "
         "absurd 0.87; the entire rest of the field is below 0.3.",
    why="under EMH, *nothing* should be this informative about "
        "tomorrow. A feature that 'predicts' the future this well is "
        "not a discovery — it is reading the answer key. This chart "
        "is the smoking gun that triggered the audit.",
    title="Exhibit C — mutual information with the target",
)
fig_card(
    FIG_EDA / "01_eurusd_cross_source.png",
    what="Yahoo's EUR/USD overlaid on the Fed's (dexuseu) and the "
         "ECB's (eurusd_official) versions, plus their daily "
         "differences in pips (1 pip = 0.0001). At price scale the "
         "three lines look identical; the differences bounce within "
         "±100–300 pips.",
    why="a one-day timestamp offset is invisible at price scale — "
        "which is why it survived so long — but at *return* scale it "
        "rewires which day's move lands in which row. Cross-checking "
        "one vendor against two official sources is what finally "
        "pinned the offset on Yahoo.",
    title="Exhibit D — three sources, one impostor",
)

# ---------------- mechanism ----------------
st.subheader("The mechanism: how a lazy timestamp becomes a leak")
st.markdown(
    "Yahoo stamps FX pairs (tickers ending `=X`) **one day late** relative "
    "to DXY, gold, oil and FRED. The target is built as "
    "`shift(-1)` — 'take the next row's return as the answer'. Follow the "
    "true dates:"
)
mech = pd.DataFrame({
    "Row labelled day t": ["Yahoo EUR/USD bar", "Target = next row's bar",
                           "dxy_return (correctly stamped)"],
    "Actually contains": ["the move of day t−1", "the move of day t",
                          "the move of day t"],
})
st.dataframe(mech, hide_index=True, width="stretch")
st.markdown(
    "The 'tomorrow' the model was asked to predict and the DXY feature it "
    "was given **describe the same physical day**. With a −0.83 mechanical "
    "same-day link, the model just read DXY's sign and 'predicted' an "
    "event that had already happened. Not a genius — a student copying "
    "from an answer sheet."
)

# ---------------- fix ----------------
st.subheader("The fix: three lines, applied at the source")
st.code(
    '# src/data/build_dataset.py :: load_yfinance()\n'
    'if asset == "eurusd":\n'
    '    df[asset] = df[asset].shift(-1)   # re-label bars to their true date\n'
    '    df = df.dropna(subset=[asset]).reset_index(drop=True)',
    language="python",
)
st.markdown(
    "`shift(-1)` pulls the next row's value onto the current row — every "
    "bar now sits on the date whose move it really contains. The last row "
    "loses its value (there is no next bar yet) and is dropped. The fix "
    "lives inside `build_master_dataset()`, so **every** future data pull "
    "inherits it automatically."
)

st.subheader("After the fix — verified live on today's dataset")
dfm = load_master()
if dfm is not None:
    r = np.log(dfm[["eurusd", "dxy"]]).diff()
    same_day = r["eurusd"].corr(r["dxy"])
    prophet = r["dxy"].corr(r["eurusd"].shift(-1))
    c1, c2, c3 = st.columns(3)
    c1.metric("Same-day corr (was −0.11)", f"{same_day:.3f}",
              help="Recomputed from the current fx_master_dataset.csv "
                   "every time this page loads. Theory expects ≈ −0.8.")
    c2.metric("'Prophecy' corr (was −0.82)", f"{prophet:.3f}",
              help="DXY today vs EUR/USD tomorrow. EMH expects ≈ 0: "
                   "the money machine must disappear.")
    c3.metric("Direction accuracy", "86% → ~52–54%",
              help="Uglier and true beats beautiful and fake.")
    st.success(
        "Both diagnostics snapped to their theoretical values the moment "
        "the alignment was fixed — and they are recomputed from the live "
        "dataset on every page load, so the proof travels with the data.",
        icon=":material/verified:",
    )

st.info(
    "**Permanent consequences** — (1) every pipeline that pulls fresh data "
    "must go through build_master_dataset(); (2) any new `=X` pair "
    "(GBP/USD, USD/JPY) must receive the same shift plus a correlation "
    "sanity check before being trusted; (3) every suspiciously good result "
    "is now audited by default. The 86% was never reported as a result — "
    "finding and killing it became the project's core methodology lesson.",
    icon=":material/gavel:",
)

next_chapter("app_pages/04_eda.py",
             "Exploring the data — what the cleaned series reveal")
