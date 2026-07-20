import pandas as pd
import streamlit as st

from app_core import load_results, next_chapter

st.markdown(
    "Twenty models were trained: **4 tasks × 5 algorithms**. Two families "
    "compete — gradient-boosted trees and random forests versus recurrent "
    "neural networks — and they receive the *same 31 features* prepared in "
    "two different ways."
)

tasks = pd.DataFrame({
    "Task": ["Direction, next day", "Direction, next week",
             "Volatility, regression", "Volatility, high/low"],
    "Type": ["Classification", "Classification", "Regression",
             "Classification"],
    "Algorithms": ["RF · XGBoost · LightGBM · LSTM · GRU"] * 4,
})
st.dataframe(tasks, hide_index=True, width="stretch")

# ---------------- trees ----------------
st.subheader("Family 1 — trees: RF, XGBoost, LightGBM")
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**Random Forest — the wisdom of crowds**")
        st.markdown(
            "Hundreds of decision trees, each trained on a random "
            "subsample of rows and features, vote on the answer. "
            "Individual trees are mediocre and overconfident; the "
            "*average* of many diverse mediocre opinions is stable. "
            "Randomness is the defense against memorizing noise."
        )
with c2:
    with st.container(border=True):
        st.markdown("**XGBoost / LightGBM — the relay of specialists**")
        st.markdown(
            "Boosting builds trees *in sequence*: each new tree is "
            "trained specifically on the errors its predecessors left "
            "behind. Powerful and fast — and precisely because it "
            "chases every residual, dangerously good at memorizing "
            "noise when there is no real signal to chase. Remember "
            "that for chapter 8."
        )
st.markdown(
    "Trees consume the features **raw, unscaled**: a tree only ever asks "
    "'is `vol20` greater than 0.004?', and the answer to that question "
    "does not change under any rescaling of the column. Scaling trees' "
    "inputs would be pointless ritual."
)

# ---------------- rnns ----------------
st.subheader("Family 2 — recurrent networks: LSTM, GRU")
st.markdown(
    "Recurrent networks read the last **10 trading days** of all 31 "
    "features *in order*, carrying a hidden memory from day to day — "
    "closer to how a trader reads a chart than to a spreadsheet lookup. "
    "**LSTM** (1997) manages its memory with three gates; **GRU** (2014) "
    "is the streamlined two-gate version — fewer parameters, less to "
    "overfit, usually the better choice on small noisy data. Their "
    "inputs are standardized first (`StandardScaler`, **fitted on the "
    "training years only** — fitting it on all data would leak future "
    "means and variances backward in time)."
)
st.code(
    "# NB06/07/08 -- PyTorch, identical skeleton for LSTM and GRU\n"
    "GRU(input_size=31, hidden_size=32, num_layers=1, batch_first=True)\n"
    "head = Sequential(\n"
    "    Dropout(0.3),          # randomly silence neurons -> no memorizing\n"
    "    Linear(32, 16), ReLU(),\n"
    "    Dropout(0.3),\n"
    "    Linear(16, 1),         # one logit\n"
    ")\n"
    "loss = BCEWithLogitsLoss()  # classification tasks\n"
    "# inference: sigmoid(logit) -> P(UP) or P(HIGH)\n"
    "# vol regression: sigmoid -> [0,1] -> inverse min-max -> %/day",
    language="python",
)
st.markdown(
    "Note how *small* this is: hidden size 32, one layer, ~7k "
    "parameters. That is deliberate. Chapter 5 showed the signal is "
    "faint; a large network pointed at faint signal does not find more "
    "signal, it invents some from noise. Small capacity plus heavy "
    "dropout is the architecture saying 'I expect to know little.'"
)

# ---------------- overfitting preview ----------------
st.subheader("Same food, opposite table manners")
res = load_results()
if "direction" in res:
    d = res["direction"]
    st.markdown(
        "A preview of chapter 8, because it belongs to the story of the "
        "algorithms: on the direction task the boosted trees reach "
        f"**{d['train'].max():.0f}% on train** and collapse to "
        f"~{d.loc[[i for i in d.index if i.startswith(('XGB', 'Light', 'Random'))], 'val'].mean():.0f}% "
        "on validation — a ~25-point gap: they memorized the noise they "
        "were built to chase. The recurrent nets show **train ≈ or even "
        "below validation** (GRU weekly: 52.1 train vs 54.1 val) — "
        "regularization worked; nothing was memorized. On a no-signal "
        "task, the disciplined small model beats the powerful one."
    )

st.info(
    "**Why five algorithms per task instead of one?** Robustness of the "
    "*conclusion*. If only XGBoost failed at direction, maybe XGBoost "
    "was misconfigured. When five very different learners — bagging, "
    "boosting, two kinds of recurrence — all land within a few points "
    "of the coin flip on direction, and all clear the baseline on "
    "volatility, the pattern is about the market, not about any one "
    "model.",
    icon=":material/diversity_3:",
)

next_chapter("app_pages/08_results.py",
             "Results — every number, and what it actually says")
