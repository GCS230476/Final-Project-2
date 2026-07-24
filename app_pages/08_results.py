import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app_core import (AMBER_BG, BASE_DIR_DAILY, BASE_DIR_WEEKLY,
                      BASE_VOL_CLF, BASE_VOL_MAE, BASE_VOL_MAE_MEDIAN,
                      C_BASE, C_DL, C_DL2, C_GRAY, C_ML, C_ML2, FIG_MODELS,
                      GREEN_BG, LIT_CEILING, RED_BG, VOL_HORIZON, VOL_VMAX,
                      fig_card, hl_vs_baseline, load_results, next_chapter,
                      verdict)

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
st.header("Problem 2 — Volatility regression: it works, once the question "
          "is posed properly")
st.markdown(verdict(True, "Outcome: works — best model explains 24% of "
                    "variance, and the edge survives on test", ""))
st.warning(
    "**This section was rebuilt after a supervisor noticed the "
    "predicted-vs-actual scatter was flat.** The arithmetic had been "
    "correct all along; the *question* was wrong. Diagnosing why is the "
    "most useful thing in this chapter.",
    icon=":material/policy:",
)

st.subheader("Why the first attempt could not have worked")
st.markdown(
    "The original target was **tomorrow's |return|** — a volatility "
    "estimate built from a *single observation*. Write tomorrow's move as"
)
st.latex(r"|r_{t+1}| \;=\; \sigma_{t+1} \cdot |z_{t+1}|")
st.markdown(
    "where :orange[σ] is the underlying volatility — forecastable, because "
    "it clusters — and :red[|z|] is random noise, which is not. Most of "
    "the variance of |r| comes from :red[|z|]. So **even a model that knew "
    "σ perfectly would score a low R²**: the ceiling was set by the "
    "question, not by the models. That is also why MAE and R² disagreed — "
    "MAE tolerates that noise, R² is punished by it — and why the scatter "
    "was flat."
)

st.subheader("The fix: ask about the week, not the single day")
st.markdown(
    f"Averaging |return| over the **next {VOL_HORIZON} trading days** lets "
    "the :red[|z|] noise cancel out and leaves :orange[σ] behind. Same "
    "features, same split, same algorithms — only the target changed:"
)
c_before, c_after = st.columns(2)
with c_before:
    with st.container(border=True):
        st.markdown(":red-badge[Before] &nbsp; **target = tomorrow only**")
        st.metric("Correlation with reality", "0.28")
        st.metric("MAE gain vs correct baseline", "+0.8%")
        st.metric("Test R²", "≈ 0.00")
with c_after:
    with st.container(border=True):
        st.markdown(":green-badge[After] &nbsp; "
                    f"**target = next {VOL_HORIZON} days**")
        st.metric("Correlation with reality", "0.52", "+0.24")
        st.metric("MAE gain vs correct baseline", "+16.3%", "+15.5 pp")
        st.metric("Test R²", "+0.09", "genuine")

if "vol_reg" in res:
    df = res["vol_reg"].copy()
    show = df[["val_mae", "gain_vs_median_pct", "val_r2", "val_corr",
               "test_r2"]].rename(
                   columns={"gain_vs_median_pct": "MAE gain vs median %"})
    styled = (show.style
              .format({"val_mae": "{:.4f}", "val_r2": "{:+.4f}",
                       "val_corr": "{:+.3f}", "test_r2": "{:+.4f}",
                       "MAE gain vs median %": "{:+.1f}"})
              .apply(lambda c: [GREEN_BG if v > 0 else RED_BG for v in c],
                     subset=["val_r2", "test_r2", "MAE gain vs median %"]))
    st.dataframe(styled, width="stretch")
    st.markdown(
        "**How to read this table** — the target is the mean |return| over "
        f"the next {VOL_HORIZON} days, min-max scaled to [0, 1] "
        f"(1.0 ≈ {VOL_VMAX:.4f}). *MAE gain vs median* compares each model "
        "with a zero-skill constant parked on the training **median** — the "
        f"correct reference for MAE (baseline MAE {BASE_VOL_MAE_MEDIAN}), "
        "not the mean, because the target is right-skewed."
    )

st.markdown(
    "**Every number, spelled out:**\n"
    "- :green[**LSTM leads**]: R² **+0.241**, correlation **0.517**, MAE "
    "**16.3% better** than the correct baseline, and test R² **+0.094**. "
    "Deep learning wins here because volatility is a *sequence* property "
    "and the 10-day window sees it directly.\n"
    "- :green[**GRU second**]: R² +0.149 on validation but the **best test "
    "R² of all, +0.112** — the most robust model out of sample.\n"
    "- **Random Forest** holds up (R² +0.091, +9.2% MAE) but "
    ":red[**XGBoost and LightGBM go negative**] on validation R² "
    "(−0.137, −0.124) while still training to 90%+ — they overfit the "
    "smoother target. On this problem the ranking flips: the boosted trees "
    "that dominated training are the worst generalisers.\n"
    "- **The trade-off is stated honestly**: this forecast is for the "
    f"**average of the next {VOL_HORIZON} days**, not for tomorrow "
    "specifically. It is a weaker claim than the original, and it is the "
    "claim the data actually supports."
)

st.info(
    "**What this says about the market** — the magnitude of a *single* "
    "day's move is essentially unpredictable, because one day is mostly "
    "noise. Average that noise away and the volatility signal is clearly "
    "there: a fifth to a quarter of the variance in next-week volatility "
    "is explainable from today's information. Volatility clustering is "
    "real; it just lives on a slower clock than one day.",
    icon=":material/insights:",
)

fig_card(
    FIG_MODELS / "08_volatility_regression.png",
    what="the ORIGINAL one-day formulation, kept here as the evidence. "
         "Accurate forecasts would hug the dashed diagonal; instead the "
         "cloud is almost horizontal — whatever the real value, the model "
         "answers about the same number.",
    why="this is the chart that started the rebuild. It is not a picture "
        "of a broken model, it is a picture of an unanswerable question: "
        "faced with a target that is mostly noise, a least-squares model "
        "correctly retreats to the average, and a flat cloud is the "
        "mathematically right response. Re-posing the question over five "
        "days is what turned this section from a failure into a result.",
    title="The chart that started the rebuild (one-day target)",
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
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol regression**")
        st.metric("Best R² (LSTM)", "+0.24", "+16% MAE vs baseline")
        st.caption(f"After reposing it over {VOL_HORIZON} days.")
with v3:
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol classification**")
        st.metric("Accuracy vs baseline", f"+2.9 pp", "survives test")
        st.caption("Consistent across all five algorithms.")

st.info(
    "**The thesis** — same 31 features, same split, same five algorithms, "
    "three questions. Asking *which way* the market moves fails, and "
    "keeps failing however it is asked. Asking *how much* it moves works, "
    "provided the question is posed over a horizon long enough for daily "
    "noise to cancel. Direction is the signal an efficient market erases; "
    "volatility is the one it cannot, because knowing a stormy week is "
    "coming does not tell anyone which way to trade.\n\n"
    "Two claims in this project were retired by its own author: a fake "
    "86% caused by a data leak, and an MAE 'win' that was only a win "
    "against the wrong baseline. Neither was a broken calculation — both "
    "were correct arithmetic pointed at the wrong comparison, which is "
    "the harder kind of error to catch.",
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
