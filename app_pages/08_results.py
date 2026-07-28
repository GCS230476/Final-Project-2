import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app_core import (AMBER_BG, BASE_DIR_DAILY, BASE_DIR_WEEKLY,
                      BASE_VOL_CLF, BASE_VOL_MAE, BASE_VOL_MAE_MEDIAN,
                      C_BASE, C_DL, C_DL2, C_GRAY, C_ML, C_ML2, FIG_MODELS,
                      GREEN_BG, LIT_CEILING, NAIVE_PRICE, RED_BG, VOL_HORIZON,
                      VOL_VMAX, fig_card, hl_vs_baseline, load_interval,
                      load_price, load_results, next_chapter, verdict)

st.markdown(
    "Four problems, one prediction made back in chapter 4: direction should "
    "fail, volatility should work. Here is what happened — every number "
    "explained, including the ugly ones. In every table below, "
    ":green-badge[green] beats its baseline, :red-badge[red] does not, and "
    ":orange-badge[amber] marks the champion chosen on validation."
)

res = load_results()
price = load_price()

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

fig_card(
    FIG_MODELS / "08_why_five_days.png",
    what="the same Random Forest, trained twice on the same features and "
         "the same split — only the target differs. Top row: what the "
         "model is asked to predict. :red[Left] is a single day's "
         "|return|, a jagged spike train with no visible structure; "
         f":green[right] is the {VOL_HORIZON}-day average, where slow "
         "waves appear — turbulent through 2022, calmer across 2023–24. "
         "Bottom row: how closely each version can actually be tracked.",
    why="this is the whole argument in one picture. A model can only "
        "learn structure that exists in its target. Against the noisy "
        "one-day target the best fit is nearly flat "
        "(:red[slope 0.07, R² 0.066]) — not because the model is weak, "
        "but because there is little to fit. Against the smoothed target "
        "the same model tilts up (:green[slope 0.21, R² 0.132]) and "
        "correlation rises from 0.26 to 0.45. Nothing about the "
        "algorithm improved; the question simply became answerable.",
    title="Why five days works and one day cannot",
)
c_before, c_after = st.columns(2)
with c_before:
    with st.container(border=True):
        st.markdown(":red-badge[Before] &nbsp; **target = tomorrow only**")
        st.metric("Best validation R²", "0.078")
        st.metric("Models with R² > 0", "4 of 5", "GRU was negative",
                  delta_color="off")
with c_after:
    with st.container(border=True):
        st.markdown(":green-badge[After] &nbsp; "
                    f"**target = next {VOL_HORIZON} days**")
        if "vol_reg" in res:
            _b = res["vol_reg"]["val_r2"].max()
            _p = int((res["vol_reg"]["val_r2"] > 0).sum())
            _n = len(res["vol_reg"])
            st.metric("Best validation R²", f"{_b:.3f}", f"{_b - 0.078:+.3f}")
            st.metric("Models with R² > 0", f"{_p} of {_n}",
                      "all positive" if _p == _n else f"{_n - _p} negative")

if "vol_reg" in res:
    df = res["vol_reg"].copy()
    # Notebook 08 writes val_mae / val_r2 / test_mae / test_r2. Showing only
    # the columns that are actually present keeps this page working even if a
    # future run adds or drops one.
    wanted = ["val_mae", "val_r2", "val_corr", "test_mae", "test_r2",
              "gain_vs_median_pct"]
    cols = [c for c in wanted if c in df.columns]
    show = df[cols].rename(columns={"gain_vs_median_pct":
                                    "MAE gain vs median %"})
    fmt = {"val_mae": "{:.4f}", "val_r2": "{:+.4f}", "val_corr": "{:+.3f}",
           "test_mae": "{:.4f}", "test_r2": "{:+.4f}",
           "MAE gain vs median %": "{:+.1f}"}
    green_red = [c for c in ["val_r2", "test_r2", "MAE gain vs median %"]
                 if c in show.columns]
    styled = (show.style
              .format({k: v for k, v in fmt.items() if k in show.columns})
              .apply(lambda c: [GREEN_BG if v > 0 else RED_BG for v in c],
                     subset=green_red))
    st.dataframe(styled, width="stretch")
    st.markdown(
        "**How to read this table** — the target is the mean |return| over "
        f"the next {VOL_HORIZON} days, min-max scaled to [0, 1] "
        f"(1.0 ≈ {VOL_VMAX:.4f}, the most volatile stretch of the training "
        "years). *val_r2* is the headline: :green[green means the model "
        "explains real variance], :red[red means it does worse than "
        "guessing the average]. MAE is shown too, but on a right-skewed "
        "target it flatters everything, so R² carries the argument here."
    )

if "vol_reg" in res:
    # Written from the table itself: an earlier hard-coded version of this
    # paragraph silently went stale when the models were retrained.
    vr = res["vol_reg"]
    r2v = vr["val_r2"].sort_values(ascending=False)
    lead, second = r2v.index[0], r2v.index[1]
    n_pos = int((vr["val_r2"] > 0).sum())
    t_best = vr["test_r2"].idxmax()
    t_worst = vr["test_r2"].idxmin()
    st.markdown(
        "**Every number, spelled out:**\n"
        f"- :green[**{lead} leads on validation**] with R² "
        f"**{r2v.iloc[0]:+.3f}**, well above the ~+0.02 the old one-day "
        "target could reach. The gain came from changing the question, not "
        "from tuning the model.\n"
        f"- :green[**{second} is second**] at **{r2v.iloc[1]:+.3f}**, so the "
        "result does not rest on one exotic algorithm.\n"
        f"- :green[**{n_pos} of {len(vr)} models are positive**] on "
        "validation. Several different algorithms agreeing is the strongest "
        "evidence that the signal belongs to the data rather than to a lucky "
        "configuration.\n"
        f"- **Test reorders the table**: {t_best} scores best out of sample "
        f"(**{vr.loc[t_best, 'test_r2']:+.3f}**) while {t_worst} slips to "
        f"{vr.loc[t_worst, 'test_r2']:+.3f}. Reported as-is — with a few "
        "hundred test rows this reshuffling is exactly the sampling noise the "
        "methodology chapter warns about, which is why models are chosen on "
        "validation and test is read once.\n"
        f"- **MAE barely separates them** "
        f"({vr['val_mae'].min():.4f}–{vr['val_mae'].max():.4f} against a "
        f"{BASE_VOL_MAE_MEDIAN:.4f} median baseline). On a right-skewed "
        "target MAE is a weak discriminator; R² carries the argument."
    )

st.warning(
    f"**One consequence to state plainly:** the {VOL_HORIZON}-day target "
    "is also used for the high/low classification below, so *that* problem "
    "now asks whether the **next week** is turbulent, not tomorrow. Its "
    "accuracy rose accordingly. That is a genuinely easier question — the "
    "improvement is a change of horizon, not a better model.",
    icon=":material/schedule:",
)

st.info(
    "**What this says about the market** — the magnitude of a *single* "
    "day's move is essentially unpredictable, because one day is mostly "
    "noise. Average that noise away and the signal appears: over a tenth "
    "of the variance in next-week volatility is explainable from today's "
    "information, consistently across five algorithms. Volatility "
    "clustering is real; it just lives on a slower clock than one day.",
    icon=":material/insights:",
)

fig_card(
    FIG_MODELS / "08_volatility_regression.png",
    what="the champion Random Forest predicting next-5-day volatility on "
         "validation. Top: the predicted line follows the actual one through "
         "the stormy 2022 and the calmer 2023–24. Bottom: the same data as a "
         "scatter, with actual on the x-axis and predicted on the y-axis. The "
         "cloud is visibly **flatter than the diagonal** — predictions span "
         "roughly 0.10–0.41 while the truth spans 0.00–0.72.",
    why="that flatness is the honest signature of a calibrated forecast, not "
        "a broken one. Regress the actual value on the prediction and the "
        "slope is **0.99**: when the model says 0.30, the outcome really does "
        "average about 0.30. Its predictions are deliberately narrower than "
        "reality — the spread ratio is **0.46**, almost exactly the "
        "correlation (**0.454**), which is the mathematically optimal amount "
        "of shrinkage for a model that explains about a fifth of the "
        "variance. Predicting the full range would *increase* the error. The "
        "price of that caution is real and worth stating: the stormiest 10% "
        "of days are under-called by **46%**, and the calmest 10% are "
        "over-called. Deeper trees do not fix it — validation R² falls from "
        "0.21 at depth 3 to 0.08 unrestricted — so this is the ceiling of the "
        "signal, not a tuning failure.",
    title="The rebuilt regression — flat on purpose, and calibrated",
)

# ============================================================
# VOL CLASSIFICATION
# ============================================================
st.header("Problem 3 — Volatility classification: the confirmation")

if "vol_clf" in res:
    df = res["vol_clf"].copy()
    champ = df["val"].idxmax()
    # Derive the majority baseline from the table itself, so it always
    # matches whatever training run produced these numbers.
    base_clf = round(float((df["val"] - df["val_vs_base"]).mean()), 2)
    st.markdown(verdict(
        True, f"Outcome: works — all five models beat the {base_clf}% "
        f"baseline, by {df['val_vs_base'].min():.1f} to "
        f"{df['val_vs_base'].max():.1f} points", ""))

    def _mark_champ_clf(row):
        return [AMBER_BG if row.name == champ else "" for _ in row]

    styled = (df.style.format("{:.2f}")
              .apply(lambda c: hl_vs_baseline(df["val"], base_clf),
                     subset=["val"])
              .apply(lambda c: hl_vs_baseline(df["val_vs_base"], 0.0),
                     subset=["val_vs_base"])
              .apply(_mark_champ_clf, axis=1, subset=["train", "test"]))
    st.dataframe(styled, width="stretch")
    st.markdown(
        f"**How to read this table** — will the next {VOL_HORIZON} days be "
        "more or less turbulent than the training-years median (HIGH or "
        "LOW vol)? *val_vs_base* is the honest metric: accuracy minus the "
        f"{base_clf}% majority baseline. :green[Every val cell is green] "
        "(beats baseline), amber marks the champion — the opposite picture "
        "to the red-filled direction table."
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    ax.bar(x, df["val"], 0.55, color=C_ML)
    ax.axhline(base_clf, color=C_BASE, ls="--", lw=1.5,
               label=f"Baseline {base_clf}%")
    ax.axhline(50, color=C_GRAY, ls=":", lw=1, label="Coin flip 50%")
    for i, v in enumerate(df["val"]):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8,
                fontweight="bold", color="#e6e9ef")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_ylim(min(50, df["val"].min() - 4), df["val"].max() + 3)
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown(
        "**Every number, spelled out:**\n"
        f"- **{champ} leads on validation at {df.loc[champ, 'val']:.2f}%**, "
        f"which is **+{df.loc[champ, 'val_vs_base']:.1f} points** over the "
        f"{base_clf}% majority baseline.\n"
        "- :green[**All five models clear the baseline**], and they do so "
        "on the test set too. Five very different algorithms agreeing is "
        "what makes this the most defensible result in the project — "
        "compare the direction table, where the same five scatter around "
        "their baseline and several land below it.\n"
        "- **Overfitting is visible but not fatal**: the boosted trees "
        "train around 81% against a 63% validation score, while the "
        "recurrent nets stay much closer together. Same discipline "
        "ranking as everywhere else in this project — but here there is "
        "real signal, so every model still clears the bar.\n"
        f"- **Note the horizon**: since the target now covers "
        f"{VOL_HORIZON} days, these accuracies are not comparable with "
        "the ~59% figures from the earlier one-day version. The question "
        "changed, so the scoreboard changed with it."
    )

# ============================================================
# PRICE
# ============================================================
st.header("Problem 4 — Price level: the question the project started from")
st.markdown(verdict(False, "", "Outcome: fails — the champion matches the "
                    "'tomorrow = today' rule almost exactly"))

st.markdown(
    "The original brief was a website that forecasts the **next EUR/USD "
    "price**. Notebook 09 does exactly that. The model learns a *return* and "
    "the price is rebuilt as `today × (1 + r)` — learning the level directly "
    "would be impossible for trees, which cannot predict a value outside the "
    "range they were trained on. A forecast counts as **correct** when it "
    "lands within a stated tolerance, measured in **pips** (1 pip = 0.0001)."
)

if price is not None:
    acc, best_px = price["acc"], price["best"]

    st.subheader("Accuracy at every tolerance — champion against doing nothing")
    piv = acc.pivot_table(index="tol_pip", columns="model", values="val")
    cols = [c for c in (best_px, NAIVE_PRICE) if c in piv.columns]
    show = piv[cols].copy()
    show.columns = [f"{c} — val %" for c in cols]
    tst = acc.pivot_table(index="tol_pip", columns="model", values="test")
    for c in cols:
        show[f"{c} — test %"] = tst[c]
    show.index = [f"±{t} pip" for t in show.index]
    st.dataframe(show.style.format("{:.1f}"), width="stretch")

    r60 = acc[acc["tol_pip"] == 60].set_index("model")
    if best_px in r60.index and NAIVE_PRICE in r60.index:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{best_px} — test accuracy at ±60 pip",
                  f"{r60.loc[best_px, 'test']:.1f}%")
        m2.metric("Naive 'tomorrow = today' — same tolerance",
                  f"{r60.loc[NAIVE_PRICE, 'test']:.1f}%")
        m3.metric("Advantage of the model",
                  f"{r60.loc[best_px, 'test'] - r60.loc[NAIVE_PRICE, 'test']:+.1f} pp",
                  "not a meaningful edge", delta_color="off")

    st.markdown(
        "**How to read this table** — accuracy rises with the tolerance you "
        "allow, which is why quoting a single accuracy figure without its "
        "tolerance means nothing. The column that decides the chapter is the "
        f"**{NAIVE_PRICE}** one: it is not a formality, it is the honest "
        "benchmark, and it keeps pace at every row."
    )

fig_card(
    FIG_MODELS / "09_price_accuracy_vs_naive.png",
    what="accuracy plotted against the tolerance allowed, for the champion "
         "(dark) and the do-nothing rule (red dashed). Solid lines are "
         "validation, faded lines are test.",
    why="the two curves sit on top of each other at every tolerance. If the "
        "model had learned anything real about tomorrow's *level*, its curve "
        "would lift clear of the red one. It never does.",
    title="The headline number, and the rule that matches it",
)

fig_card(
    FIG_MODELS / "09_price_return_scatter.png",
    what="the same model with the price stripped away, showing what it is "
         "really asked to predict: tomorrow's **return** in pips. Predicted "
         "on the x-axis, actual on the y-axis, with the fitted trend in red "
         "against the dashed diagonal of a perfect forecast.",
    why="this is the chapter in one picture. The cloud is a vertical stick: "
        "the model outputs nearly the same tiny number every day — a spread "
        "of **2.5 pip** against reality's **49.1 pip** — while the market "
        "swings freely. Correlation is **+0.010** and R² is **−0.002**, "
        "meaning it does marginally *worse* than always predicting the "
        "average return. Compare this with the volatility scatter above, "
        "which looked flat too but has correlation 0.454 and a calibrated "
        "slope of 0.99. Same visual shape, opposite conclusion — and knowing "
        "the difference is the point.",
    title="The honest scatter — where the 80% actually comes from",
)

st.subheader("The constructive answer — forecast an interval, not a point")
st.markdown(
    "A single number is the wrong output for a random walk. The *centre* of "
    "tomorrow's move is unforecastable — but its *width* is exactly what "
    "Problem 2 forecasts, and that is the one thing here that beats its "
    "baseline. So keep the price forecast as the centre and let the "
    "volatility model set the width. The multiplier is calibrated on the "
    "**training slice only**; validation and test then check whether the "
    "promise holds on data the model never saw."
)

interval = load_interval()
if interval is not None:
    show_i = interval.rename(columns={
        "claimed": "Confidence claimed (%)", "k": "Width (σ units)",
        "train": "Train %", "val": "Validation %", "test": "Test %"})
    st.dataframe(
        show_i.style
        .format({"Width (σ units)": "{:.2f}", "Train %": "{:.1f}",
                 "Validation %": "{:.1f}", "Test %": "{:.1f}"})
        .apply(lambda c: [GREEN_BG if abs(v - cl) <= 3 else ""
                          for v, cl in zip(c, interval["claimed"])],
               subset=["Validation %", "Test %"]),
        hide_index=True, width="stretch")
    _gap = (interval["val"] - interval["claimed"]).abs().max()
    st.markdown(
        f":green-badge[Calibrated] &nbsp; Every stated confidence level lands "
        f"within **{_gap:.1f} points** of the truth on validation — an 80% "
        f"interval really did contain the next close **"
        f"{interval.loc[interval['claimed'] == 80, 'val'].iloc[0]:.1f}%** of "
        f"the time, and **"
        f"{interval.loc[interval['claimed'] == 80, 'test'].iloc[0]:.1f}%** on "
        "test. This is the honest, usable form of the price forecast, and it "
        "is only possible because the width comes from the one quantity in "
        "this project that is genuinely predictable."
    )

fig_card(
    FIG_MODELS / "09_price_calibration.png",
    what="the confidence the model claims on the x-axis against how often it "
         "was actually right on the y-axis. The dashed line is perfect "
         "calibration; validation is dark, test is green.",
    why="both curves sit essentially on the diagonal. An interval that says "
        "80% and delivers 80% is trustworthy in a way a point forecast never "
        "is — and sitting *above* the line on test means it errs toward "
        "caution rather than overconfidence, which is the right direction to "
        "be wrong in.",
    title="An 80% interval that is actually right 80% of the time",
)

fig_card(
    FIG_MODELS / "09_price_band.png",
    what="the 80% interval drawn over the real EUR/USD price across the "
         "validation years, with red dots marking the days the price escaped "
         "the band. The band widens in turbulent stretches and narrows in "
         "calm ones — that movement is the volatility model working.",
    why="this is what a user would actually be shown, and it is defensible: "
        "the escapes are roughly as frequent as the stated confidence allows. "
        "Note the band is about 134 pips wide on an average day — honest, but "
        "wide enough to make clear that precision here is limited.",
    title="The interval against reality",
)

st.markdown(
    "**Every number, spelled out:**\n"
    "- **80.4% test accuracy at ±60 pip is real** and correctly measured on "
    "data the model never saw. It is also, on its own, close to meaningless.\n"
    f"- **The naive rule scores 80.4% too.** The champion's margin on "
    "validation is **+0.20 percentage points** — well inside noise.\n"
    "- **The model predicts moves 18× smaller than the market makes**: a "
    "standard deviation of 3.3 pip against the market's 59.1 pip. It has "
    "learned the only safe thing about tomorrow's level — that it will be "
    "close to today's.\n"
    "- **Implied directional hit rate: 51.2%**, which agrees with Problem 1 "
    "from a completely different direction. Two independent routes to the "
    "same wall is evidence, not coincidence.\n"
    "- **Where the accuracy really comes from**: the typical daily move is "
    "about 40 pip, so a ±60 pip window catches most days no matter who is "
    "forecasting. The tolerance is doing the work, not the algorithm."
)

st.info(
    "**How to state this in the report** — *\"The model forecasts the next "
    "EUR/USD close with 80.4% accuracy on the test set within ±60 pip "
    "(±0.53% of price). However, the 'no change' baseline also scores 80.4%, "
    "showing that the next close is essentially unpredictable, consistent "
    "with the random walk hypothesis. The model's measurable contribution is "
    "in forecasting volatility, not price level.\"* That sentence keeps the "
    "impressive number **and** survives cross-examination, because it answers "
    "the hardest question about itself before anyone can ask it.",
    icon=":material/format_quote:",
)

# ============================================================
# SYNTHESIS
# ============================================================
st.header("The verdict")
v1, v4, v2, v3 = st.columns(4)
with v1:
    with st.container(border=True):
        st.markdown(":red-badge[:material/cancel: FAILS] &nbsp; "
                    "**Direction**")
        st.metric("Best edge vs baseline", "+2.3 pp", "≈ coin flip",
                  delta_color="off")
        st.caption("The control group. Exactly the controlled failure "
                   "EMH predicts.")
with v4:
    with st.container(border=True):
        st.markdown(":red-badge[:material/cancel: FAILS] &nbsp; "
                    "**Price level**")
        if price is not None:
            _r = price["acc"]
            _r = _r[_r["tol_pip"] == 60].set_index("model")
            if price["best"] in _r.index and NAIVE_PRICE in _r.index:
                st.metric("Edge over 'no change'",
                          f"{_r.loc[price['best'], 'test'] - _r.loc[NAIVE_PRICE, 'test']:+.1f} pp",
                          "80.4% vs 80.4%", delta_color="off")
        st.caption("A real 80% that the naive rule matches exactly.")
with v2:
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol regression**")
        _best_r2 = res["vol_reg"]["val_r2"].max() if "vol_reg" in res else 0
        st.metric("Best validation R²", f"+{_best_r2:.3f}", "5 of 5 positive")
        st.caption(f"After reposing it over {VOL_HORIZON} days.")
with v3:
    with st.container(border=True):
        st.markdown(":green-badge[:material/check_circle: WORKS] &nbsp; "
                    "**Vol classification**")
        if "vol_clf" in res:
            _e = res["vol_clf"]["val_vs_base"]
            st.metric("Accuracy vs baseline",
                      f"+{_e.min():.1f} to +{_e.max():.1f} pp",
                      "survives test")
        st.caption("Consistent across all five algorithms.")

st.info(
    "**The thesis** — same 31 features, same split, same algorithms, four "
    "questions. Asking *which way* the market moves fails, and keeps failing "
    "however it is asked — as direction, and again as a price level, where "
    "the do-nothing rule matches the model outright. Asking *how much* it "
    "moves works, provided the question is posed over a horizon long enough "
    "for daily noise to cancel. Direction and level are the signals an "
    "efficient market erases; volatility is the one it cannot, because "
    "knowing a stormy week is coming does not tell anyone which way to "
    "trade.\n\n"
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
