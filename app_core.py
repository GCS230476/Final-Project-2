"""
Shared helpers for the storytelling dashboard.

Keep this module light: no torch / predict_latest imports here, so every
story page loads fast. Only app_pages/09_live.py imports the heavy stack.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- paths ----------
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
FIG_MODELS = ROOT / "figures" / "models"
FIG_EDA = ROOT / "figures" / "eda"
FIG_FEATURES = ROOT / "figures" / "features"
MASTER_CSV = ROOT / "data" / "interim" / "fx_master_dataset.csv"
FEATURES_CSV = ROOT / "data" / "processed" / "fx_features.csv"
LIVE_CSV = ROOT / "data" / "processed" / "fx_features_live.csv"

# ---------- colors ----------
DARK_BG = "#161b26"
C_DL, C_DL2 = "#ffb400", "#c78c00"     # amber  = deep learning
C_ML, C_ML2 = "#4f8df5", "#31589c"     # blue   = machine learning
C_BASE = "#ff6b6b"                     # baseline lines
C_GRAY = "#8892a6"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color":       "#e6e9ef",
    "axes.labelcolor":  "#e6e9ef",
    "xtick.color":      "#aab2c0",
    "ytick.color":      "#aab2c0",
    "axes.edgecolor":   "#3d4656",
    "grid.color":       "#2a3140",
    "legend.facecolor": DARK_BG,
    "legend.edgecolor": "#3d4656",
})

# ---------- project identity ----------
# NOTE for Khang: replace with the exact title you register with cô.
PROJECT_TITLE = ("Forecasting EUR/USD: what a machine can — and cannot — "
                 "say about tomorrow")
PROJECT_SUBTITLE = ("A comparative study of machine learning and deep "
                    "learning for exchange-rate direction and volatility, "
                    "under the Efficient Market Hypothesis")
AUTHOR = "Dong Cong Gia Khang"
COURSE = "Final Project · FPT Greenwich · 2026"

# ---------- result highlight colors (for pandas Styler) ----------
GREEN_BG = "background-color: rgba(46, 160, 67, 0.28)"    # beats baseline
RED_BG = "background-color: rgba(248, 81, 73, 0.24)"       # below baseline
AMBER_BG = "background-color: rgba(255, 180, 0, 0.22); font-weight: 700"

# ---------- key frozen numbers (from models/*.json and notebooks) ----------
VOL_VMAX = 0.03175612356587054      # min-max scale of the volatility target
VOL_MEDIAN_SCALED = 0.10140372247974766  # HIGH/LOW threshold (train median)
BASE_DIR_DAILY = 50.6               # majority-class baseline, val, daily
BASE_DIR_WEEKLY = 51.79             # majority-class baseline, val, weekly
BASE_VOL_CLF = 56.99                # majority-class baseline, val (43/57 split)
BASE_VOL_MAE = 0.082                # constant-prediction baseline MAE (scaled)
LIT_CEILING = 58.52                 # Guyard & Deriaz 2024, 21 algorithms
RENAISSANCE = 50.75                 # Renaissance Technologies win rate

TRAIN_END = "2022-01-01"
VAL_END = "2025-01-01"


# ---------- cached loaders ----------
@st.cache_data
def load_master():
    if not MASTER_CSV.exists():
        return None
    df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    return df.dropna(subset=["eurusd"]).reset_index(drop=True)


@st.cache_data
def load_features_frozen():
    if not FEATURES_CSV.exists():
        return None
    return pd.read_csv(FEATURES_CSV, parse_dates=["date"])


@st.cache_data
def load_results():
    data = {}
    p = MODELS_DIR / "all_10_models_comparison.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0)
        d.columns = [c.strip().lower() for c in d.columns]
        data["direction"] = d
    p = MODELS_DIR / "volatility_regression_results.csv"
    if p.exists():
        data["vol_reg"] = pd.read_csv(p, index_col=0)
    p = MODELS_DIR / "volatility_classification_results.csv"
    if p.exists():
        data["vol_clf"] = pd.read_csv(p, index_col=0)
    return data


# ---------- UI helpers ----------
def fig_card(path: Path, what: str, why: str, title: str | None = None):
    """A figure with its explanation: what the chart shows, why it matters."""
    with st.container(border=True):
        if title:
            st.markdown(f"#### :orange[{title}]")
        if path.exists():
            st.image(str(path), width="stretch")
        else:
            st.error(f"Figure not found: {path.name}")
        st.markdown(f":blue-badge[:material/visibility: What you are "
                    f"looking at]&nbsp; {what}")
        st.markdown(f":green-badge[:material/lightbulb: Why it matters]"
                    f"&nbsp; {why}")


def next_chapter(page_path: str, label: str):
    st.markdown("")
    st.page_link(page_path, label=f"Next: {label}",
                 icon=":material/arrow_forward:")


def verdict(ok: bool, win_text: str, lose_text: str) -> str:
    """Return a colored badge string for a good/bad outcome."""
    if ok:
        return f":green-badge[:material/check_circle: {win_text}]"
    return f":red-badge[:material/cancel: {lose_text}]"


def hl_vs_baseline(col, baseline, higher_is_better=True):
    """Styler helper: green if the value beats the baseline, red if not."""
    out = []
    for v in col:
        try:
            better = (v >= baseline) if higher_is_better else (v <= baseline)
        except TypeError:
            out.append("")
            continue
        out.append(GREEN_BG if better else RED_BG)
    return out


GRAPH_STYLE = """
    bgcolor="transparent";
    rankdir=LR;
    node [style="filled", fillcolor="#232a3a", fontcolor="#e6e9ef",
          color="#3d4656", fontname="Helvetica", fontsize=11, shape=box];
    edge [color="#8892a6", fontcolor="#aab2c0", fontname="Helvetica",
          fontsize=9];
"""
