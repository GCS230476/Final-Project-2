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
st.header("Problem 2 — Volatility regression: the quiet success")
st.markdown(verdict(True, "Outcome: works — all five models beat the "
                    "naive baseline on MAE", ""))

if "vol_reg" in res:
    df = res["vol_reg"].copy()
    styled = (df.style.format("{:.4f}")
              .apply(lambda c: hl_vs_baseline(df["val_mae"], BASE_VOL_MAE,
                                              higher_is_better=False),
                     subset=["val_mae"])
              .apply(lambda c: hl_vs_baseline(df["val_r2"], 0.0),
                     subset=["val_r2"]))
    st.dataframe(styled, width="stretch")
    st.markdown(
        "**How to read this table** — the target is tomorrow's "
        "|return|, min-max scaled to [0, 1] (1.0 corresponds to "
        f"{VOL_VMAX:.4f} ≈ a 3.2% day, the wildest in training "
        "history). *val_mae* = average miss, lower is better — "
        f":green[green beats the baseline {BASE_VOL_MAE}]. *val_r2* = "
        "variance explained — :green[green is above zero (real signal)], "
        ":red[red is below]. Every val MAE cell is green; that is the "
        "success that never happened for direction."
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(df.index, df["val_mae"], color=C_ML)
    axes[0].axhline(BASE_VOL_MAE, color=C_BASE, ls="--", lw=1.4,
                    label=f"Baseline {BASE_VOL_MAE}")
    axes[0].set_ylabel("Val MAE")
    axes[0].set_title("MAE — every model beats the baseline")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].legend(fontsize=8)
    colors = [C_DL if v > 0 else C_BASE for v in df["val_r2"]]
    axes[1].bar(df.index, df["val_r2"], color=colors)
    axes[1].axhline(0, color=C_GRAY, lw=1)
    axes[1].set_ylabel("Val R²")
    axes[1].set_title("R² — above zero = genuine signal")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        "**Every number, spelled out:**\n"
        f"- **All five models beat baseline MAE {BASE_VOL_MAE}** — the "
        "thing that never happened for direction happens here five "
        "times out of five. Best MAE: LSTM 0.0714, ~13% less error "
        "than the naive guess.\n"
        "- **Random Forest R² 0.0775, XGBoost 0.0744** on validation: "
        "about 7–8% of tomorrow's volatility variance explained. That "
        "sounds small — until you recall that for direction the "
        "explainable share was statistically zero. In daily FX, any "
        "positive out-of-sample R² is real, hard-won signal.\n"
        "- **The MAE/R² disagreement** (LSTM best on MAE, worst-ish on "
        "R²) is honest tension, not a bug: the deep models predict "
        "conservatively near the average level, which minimizes the "
        "*typical* miss (MAE) but surrenders the extreme days that R² "
        "weighs heavily. The trees stretch further for spikes.\n"
        "- **Test R² collapses toward zero** (RF +0.001, XGB −0.003) "
        "while test MAE stays fine (~0.070–0.074, still beating "
        "0.082). Translation: in 2025-26 the models still predict the "
        "*level* of volatility usefully, but the variance-explaining "
        "edge did not survive the regime change. Reported as-is."
    )

    fig_card(
        FIG_MODELS / "08_volatility_regression.png",
        what="XGBoost's predicted volatility (dark line) laid over the "
             "actual day-by-day volatility (light line) across the "
             "validation years, plus the same data as a scatter against "
             "the perfect-prediction diagonal.",
        why="you can see exactly *what kind* of skill the model has: "
            "the dark line rises and falls with the stormy episodes "
            "(mid-2022, late-2022) — it tracks the volatility *regime* "
            "— but it never reaches the tallest spikes, and the "
            "scatter flattens well below the diagonal for extreme "
            "days. The model knows *when the sea is rough*; it "
            "systematically under-calls *freak waves*. That is the "
            "honest shape of R² ≈ 0.08.",
        title="What 'R² = 0.08' actually looks like",
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
        st.metric("MAE vs naive baseline", "−10 to −13%", "R² > 0 on val")
        st.caption("Modest, real, honestly fragile out of regime.")
with v3:
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol classification**")
        st.metric("Accuracy vs baseline", f"+2.9 pp", "survives test")
        st.caption("Consistent across all five algorithms.")

st.info(
    "**The thesis, confirmed by the experiment** — same 31 features, "
    "same splits, same five algorithms: direction lands on its "
    "baseline; both volatility tasks beat theirs. The difference "
    "between the columns is the difference between a signal the market "
    "erases (direction) and a signal it cannot erase (clustering). An "
    "ugly true 54% taught us more than a beautiful fake 86% ever "
    "could.",
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
