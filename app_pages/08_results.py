import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app_core import (AMBER_BG, BASE_DIR_DAILY, BASE_DIR_WEEKLY,
                      BASE_VOL_CLF, BASE_VOL_MAE, C_BASE, C_DL, C_DL2,
                      C_GRAY, C_ML, C_ML2, FIG_MODELS, GREEN_BG, LIT_CEILING,
                      RED_BG, VOL_VMAX, fig_card, hl_vs_baseline,
                      load_results, next_chapter, verdict)

st.markdown(
    "Twenty models, three problems, one prediction made back in chapter 4: "
    "direction should fail, volatility should work. Here is what happened — "
    "every number explained, including the ugly ones. In every table below, "
    ":green-badge[green] beats its baseline, :red-badge[red] does not, and "
    ":orange-badge[amber] marks the champion chosen on validation."
)

res = load_results()

# ============================================================
# DIRECTION
# ============================================================
st.header("Problem 1 — Direction: the expected failure")
st.markdown(verdict(False, "", "Outcome: fails as predicted — no model "
                    "clears its baseline by a meaningful margin"))

if "direction" in res:
    df = res["direction"].copy()
    base = [BASE_DIR_WEEKLY if m.strip().endswith("(W)") else BASE_DIR_DAILY
            for m in df.index]
    show = df.copy()
    show.insert(0, "baseline", base)
    show["edge vs base"] = show["val"] - show["baseline"]
    champ = show["edge vs base"].idxmax()

    def _mark_champ(row):
        return [AMBER_BG if row.name == champ else "" for _ in row]

    styled = (show.style.format("{:+.2f}", subset=["edge vs base"])
              .format("{:.2f}", subset=["baseline", "train", "val", "test"])
              .apply(lambda c: hl_vs_baseline(show["edge vs base"], 0.0),
                     subset=["edge vs base"])
              .apply(lambda c: [GREEN_BG if e > 0 else RED_BG
                                for e in show["edge vs base"]], subset=["val"])
              .apply(_mark_champ, axis=1, subset=["train", "val", "test"]))
    st.dataframe(styled, width="stretch")
    st.markdown(
        "**How to read this table** — rows are the ten direction models "
        "(D = daily horizon, W = weekly). *baseline* is the majority-class "
        "score to beat; *edge vs base* = val − baseline is the only number "
        "that counts. Green = beat baseline, red = did not, amber row = "
        "the champion. Notice how many rows are **red**: even 'good' "
        "training scores collapse below baseline on unseen data."
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    w = 0.38
    is_dl = [m.startswith(("LSTM", "GRU")) for m in df.index]
    ax.bar(x - w / 2, df["val"], w, label="Validation",
           color=[C_DL if d else C_ML for d in is_dl])
    ax.bar(x + w / 2, df["test"], w, label="Test",
           color=[C_DL2 if d else C_ML2 for d in is_dl])
    ax.axhline(50, color=C_GRAY, lw=0.8)
    ax.axhline(LIT_CEILING, color=C_BASE, ls="--", lw=1.2,
               label=f"Literature ceiling {LIT_CEILING}%")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_ylim(42, 62)
    ax.set_title("Amber = deep learning · Blue = tree models", fontsize=10)
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown(
        "**Every number, spelled out:**\n"
        f"- **GRU weekly, 54.06% val** — the champion. Weekly baseline "
        f"is {BASE_DIR_WEEKLY}%, so the true edge is **+2.27 points**, "
        "not '4 points above a coin'. Its train accuracy is 52.11% — "
        "*below* validation — meaning zero overfitting: it learned only "
        "what generalizes, which for direction is very little.\n"
        f"- **GRU daily, 53.41% val** vs ~{BASE_DIR_DAILY}% baseline → "
        "+2.8 points, same modest story.\n"
        "- **The tree wreck** — LightGBM daily hits 78.31% on train and "
        "51.9% on validation; XGBoost weekly 79.16% train, 48.09% val "
        "(*below* baseline). A 26–31 point train-val gap is textbook "
        "overfitting: boosting chased residuals that were pure noise.\n"
        "- **The test column scatters** from 44.01% to 53.52% and does "
        "not rank models the same way validation does. On ~250-350 rows "
        "of test data, ±3 points is sampling noise; this is why the "
        "protocol picks on validation and reports test once, untouched.\n"
        f"- Even the champion sits below the published ceiling of "
        f"{LIT_CEILING}% — and that ceiling itself is barely above "
        "chance. Renaissance wins 50.75% of trades. Direction near the "
        "coin flip is not our failure; it is the market working."
    )

# ============================================================
# VOL REGRESSION
# ============================================================
st.header("Problem 2 — Volatility regression: a result that did not survive "
          "audit")
st.markdown(verdict(False, "", "Outcome: fails — the apparent MAE win came "
                    "from the wrong baseline, and R² does not hold up"))
st.warning(
    "**This section was rewritten after a supervisor spotted that the "
    "predicted-vs-actual scatter was flat.** Investigating that one "
    "observation turned up three problems in what had been reported as a "
    "success. Everything below is the corrected version, reproducible by "
    "running `verify_vol_regression.py`.",
    icon=":material/policy:",
)

if "vol_reg_verified" in res:
    df = res["vol_reg_verified"].copy()
    show = df[["model", "calibrated", "val_mae", "val_mae_gain_vs_mean_pct",
               "val_mae_gain_vs_median_pct", "val_r2", "val_corr",
               "test_r2"]].rename(columns={
                   "val_mae_gain_vs_mean_pct": "gain vs MEAN base %",
                   "val_mae_gain_vs_median_pct": "gain vs MEDIAN base %"})
    styled = (show.style
              .format({"val_mae": "{:.4f}", "val_r2": "{:+.4f}",
                       "val_corr": "{:+.3f}", "test_r2": "{:+.4f}",
                       "gain vs MEAN base %": "{:+.1f}",
                       "gain vs MEDIAN base %": "{:+.1f}"})
              .apply(lambda c: [GREEN_BG if v > 0 else RED_BG for v in c],
                     subset=["val_r2", "test_r2"])
              .apply(lambda c: [GREEN_BG if v >= 5 else RED_BG for v in c],
                     subset=["gain vs MEDIAN base %"]))
    st.dataframe(styled, width="stretch", hide_index=True)
    st.markdown(
        "**How to read this table** — the target is tomorrow's |return|, "
        f"min-max scaled to [0, 1] (1.0 ≈ {VOL_VMAX:.4f}, a 3.2% day). "
        "The two *gain* columns are the same models measured against two "
        "different naive baselines, and that difference is the whole "
        "story."
    )

st.subheader("Problem 1 — the MAE win was measured against the wrong baseline")
st.markdown(
    "MAE is minimised by the **median**; R² is minimised by the **mean**. "
    "The volatility target is strongly right-skewed (most days calm, a few "
    "wild), so its median (0.101) sits well below its mean (0.132). The "
    "original baseline predicted the **mean**, which is the wrong "
    "reference for MAE.\n\n"
    "A constant predictor with **zero skill**, parked on the training "
    "median, scores MAE **0.0770**. The models score 0.0753–0.0765. So "
    "against the correct baseline the gain shrinks from a headline "
    ":red[**~9–12%**] to :red[**+0.7% to +2.2%**] — essentially nothing. "
    "The models had mostly discovered that *guessing low* pays on a "
    "skewed target, which is a property of the distribution, not "
    "forecasting skill."
)

st.subheader("Problem 2 — the recurrent nets were mis-calibrated")
st.markdown(
    "Their sigmoid output averaged **0.088** against a true mean of "
    "**0.118** — a systematic under-forecast of about 25%, which drove "
    "their R² negative (LSTM :red[−0.007], GRU :red[−0.054]) even though "
    "LSTM's correlation with reality (**0.274**) is the best of any "
    "model. The scale was right; only the offset was wrong.\n\n"
    "A linear correction fitted **on the training split only** repairs "
    "it: LSTM's validation R² moves :red[−0.007] → :green[**+0.049**], "
    "GRU's :red[−0.054] → :green[**+0.022**]. The fix is now applied in "
    "the live demo, where all five models finally agree on the same "
    "forecast range instead of the two nets sitting 25% low."
)

st.subheader("Problem 3 — the surviving R² is fragile and does not generalise")
if "vol_reg_verified" in res:
    rb1, rb2, rb3 = st.columns(3)
    rb1.metric("Validation R² (XGBoost)", "+0.075",
               help="Positive, and statistically real: correlation 0.28 "
                    "with p < 1e-11.")
    rb2.metric("…dropping the 5% wildest days", "+0.018", "−76%",
               delta_color="inverse",
               help="Most of the R² is earned on a handful of extreme "
                    "days, not on everyday forecasting.")
    rb3.metric("Test R² (2024-05 onward)", "−0.002",
               help="The edge does not survive into the held-out period.")
st.markdown(
    "Split the validation window into thirds and the R² reads "
    "**+0.054 / −0.013 / +0.031** — one sub-period is outright negative. "
    "Together with the scatter, this says what kind of skill is left: the "
    "model tracks the **slow drift of the volatility regime** (which "
    "months are calmer) but has essentially no **day-to-day** skill. "
    "Predictions cover only about a quarter of the real spread, so the "
    "cloud of points sits flat instead of following the diagonal."
)

st.info(
    "**Honest verdict** — the relationship is statistically real "
    "(correlation ≈ 0.24–0.28, p < 1e-11) but explains only ~6–8% of "
    "variance, leans on a few extreme days, and vanishes out of sample. "
    "Predicting the *magnitude* of tomorrow's move is therefore reported "
    "as a **failure**. The defensible volatility result is Problem 3 "
    "below — classifying the regime, where the edge is consistent across "
    "all five algorithms and survives the test set.",
    icon=":material/balance:",
)

fig_card(
    FIG_MODELS / "08_volatility_regression.png",
    what="predicted volatility against what actually happened. In the "
         "scatter, accurate forecasts would hug the dashed diagonal. "
         "Instead the cloud is almost horizontal: whatever the real "
         "value — 0.05 or 0.55 — the model answers about 0.12.",
    why="this is the chart that triggered the whole audit. A flat cloud "
        "means the predictions carry only about a quarter of the real "
        "spread: the model never commits to a big number, so it can "
        "never call a wild day. It is not overfitting and it is not a "
        "lack of capacity — the features simply carry too little "
        "information about tomorrow's magnitude, and a least-squares "
        "model facing an unpredictable target correctly retreats to "
        "the average. The flat line *is* the finding.",
    title="The chart that started the audit",
)

# ============================================================
# VOL CLASSIFICATION
# ============================================================
st.header("Problem 3 — Volatility classification: the confirmation")
st.markdown(verdict(True, "Outcome: works — all five models beat baseline "
                    "by ~2.6–2.9 points, and it holds on test", ""))

if "vol_clf" in res:
    df = res["vol_clf"].copy()
    champ = df["val"].idxmax()

    def _mark_champ_clf(row):
        return [AMBER_BG if row.name == champ else "" for _ in row]

    styled = (df.style.format("{:.2f}")
              .apply(lambda c: hl_vs_baseline(df["val"], BASE_VOL_CLF),
                     subset=["val"])
              .apply(lambda c: hl_vs_baseline(df["val_vs_base"], 0.0),
                     subset=["val_vs_base"])
              .apply(_mark_champ_clf, axis=1, subset=["train", "test"]))
    st.dataframe(styled, width="stretch")
    st.markdown(
        "**How to read this table** — will tomorrow's |return| land "
        "above or below the training-years median (HIGH or LOW vol)? "
        "*val_vs_base* is the honest metric: accuracy minus the "
        f"{BASE_VOL_CLF}% majority baseline. :green[Every val cell is "
        "green] (beats baseline), amber marks the champion — the opposite "
        "picture to the red-filled direction table."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    ax.bar(x, df["val"], 0.55, color=C_ML)
    ax.axhline(BASE_VOL_CLF, color=C_BASE, ls="--", lw=1.5,
               label=f"Baseline {BASE_VOL_CLF}%")
    ax.axhline(50, color=C_GRAY, ls=":", lw=1, label="Coin flip 50%")
    for i, v in enumerate(df["val"]):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8,
                fontweight="bold", color="#e6e9ef")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_ylim(48, 63)
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown(
        "**Every number, spelled out:**\n"
        "- **LSTM 59.93% val = +2.94 points over baseline** — the "
        "champion, with GRU and LightGBM at 59.8% right behind.\n"
        "- **All five models sit in a 0.4-point band** (59.56–59.93). "
        "Five very different algorithms agreeing this tightly means "
        "the ~3-point edge is a property of the *data*, not a lucky "
        "seed. Compare the direction table, where the same five "
        "algorithms scatter across 6 points.\n"
        "- **Overfitting is tamed but visible**: trees train at 68–75% "
        "(gap ~9–15 pts), the recurrent nets at 61–63% (gap ~1–3 "
        "pts). Same ranking of discipline as in direction — but here "
        "there is real signal, so everyone still clears the bar.\n"
        "- **Test holds up**: 55.3–57.9%, and the two deep models keep "
        "57.89% — the edge survives into 2025-26, unlike the "
        "regression R². Classifying the regime is more robust than "
        "predicting the exact magnitude."
    )

# ============================================================
# SYNTHESIS
# ============================================================
st.header("The verdict")
v1, v2, v3 = st.columns(3)
with v1:
    with st.container(border=True):
        st.markdown(":red-badge[:material/cancel: FAILS] &nbsp; "
                    "**Direction**")
        st.metric("Best edge vs baseline", "+2.3 pp", "≈ coin flip",
                  delta_color="off")
        st.caption("The control group. Exactly the controlled failure "
                   "EMH predicts.")
with v2:
    with st.container(border=True):
        st.markdown(":red-badge[:material/cancel: FAILS] &nbsp; "
                    "**Vol regression**")
        st.metric("MAE vs correct baseline", "+0.7 to +2.2%",
                  "R² gone on test", delta_color="off")
        st.caption("Magnitude is not predictable; found by audit.")
with v3:
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol classification**")
        st.metric("Accuracy vs baseline", f"+2.9 pp", "survives test")
        st.caption("Consistent across all five algorithms.")

st.info(
    "**The thesis, in its audited form** — same 31 features, same split, "
    "same five algorithms. Predicting *direction* fails, and so does "
    "predicting the *magnitude* of tomorrow's move. What survives is the "
    "coarser question: **is tomorrow a high- or low-volatility day?** "
    "That edge is small (+2.9 points), but it is consistent across all "
    "five algorithms and it holds on the test set — which is exactly "
    "what volatility clustering predicts and what market efficiency "
    "cannot erase.\n\n"
    "Two results in this project were killed by their own author: a fake "
    "86% caused by a data leak, and a fake MAE win caused by the wrong "
    "baseline. Both were found by asking why a number looked better than "
    "it had any right to.",
    icon=":material/school:",
)

with st.expander("Honest limitations — read before quoting any number",
                 icon=":material/warning:"):
    st.markdown(
        "- **No probability calibration yet**: P(UP) = 0.6 has not been "
        "verified to mean '60% of such days actually go up'.\n"
        "- **Weekly samples overlap** (consecutive 5-day windows share "
        "4 days), which makes weekly results look smoother than the "
        "underlying evidence.\n"
        "- **Significance tests are planned, not done**: a one-sided "
        "binomial test against the majority baseline for the three "
        "champions only. Small edges on ~700 validation days may well "
        "yield p > 0.05 — if so, that will be reported, not hidden.\n"
        "- **Single frozen test pass**: test numbers are one sample of "
        "one era (2025-26), not a guarantee.\n"
        "- **No trading claim**: an edge in accuracy is not an edge "
        "after spreads, slippage and risk. This is a forecasting "
        "study, not a strategy."
    )

next_chapter("app_pages/09_live.py",
             "Live prediction — the pipeline running on today's data")
