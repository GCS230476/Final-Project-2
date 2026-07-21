"""
predict_latest.py -- shared inference pipeline for the Streamlit app.

Loads any trained model from models/ and predicts using the LATEST rows
of data/processed/fx_features.csv.

Tasks covered (matching filenames in models/):
  direction_daily   : random_forest / xgboost / lightgbm / lstm / gru
  direction_weekly  : weekly_*
  vol_regression    : vol_* (trees) + vol_reg_lstm / vol_reg_gru
  vol_classification: vol_clf_*

Rules baked in (same as training):
  - ML (tree) models  -> RAW features (numpy), last 1 row
  - DL (LSTM/GRU)     -> scaled features (fit-on-train scaler), last WINDOW rows
  - DL outputs a raw logit -> apply sigmoid
  - vol regression    -> prediction is in [0,1] min-max space -> inverse
                         transform with vmin/vmax from vol_meta.json
Run a self-test:  python predict_latest.py   (use venv_dl)
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FEATURES_CSV = ROOT / "data" / "processed" / "fx_features.csv"

# ----------------------------------------------------------------- meta
with open(MODELS_DIR / "direction_meta.json") as f:
    DIR_META = json.load(f)
with open(MODELS_DIR / "vol_meta.json") as f:
    VOL_META = json.load(f)

FEATURE_COLS = DIR_META["feature_cols"]          # same 31 cols in both metas
WINDOW = DIR_META["window"]                      # 10


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# ----------------------------------------------------------------- DL net
class RNNClassifier(nn.Module):
    """Same architecture as NB06/NB07/NB08 (hidden=32, head 32->16->1)."""

    def __init__(self, n_features: int, rnn_type: str = "lstm",
                 hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),          # head.0
            nn.Linear(hidden, 16),        # head.1
            nn.ReLU(),                    # head.2
            nn.Dropout(dropout),          # head.3
            nn.Linear(16, 1),             # head.4
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # raw logit


# ----------------------------------------------------------------- registry
# task -> model key -> filename.  Keys are what the Streamlit dropdown shows.
REGISTRY = {
    "direction_daily": {
        "Random Forest": "random_forest.pkl",
        "XGBoost":       "xgboost.pkl",
        "LightGBM":      "lightgbm.pkl",
        "LSTM":          "lstm.pt",
        "GRU":           "gru.pt",
    },
    "direction_weekly": {
        "Random Forest": "weekly_random_forest.pkl",
        "XGBoost":       "weekly_xgboost.pkl",
        "LightGBM":      "weekly_lightgbm.pkl",
        "LSTM":          "weekly_lstm.pt",
        "GRU":           "weekly_gru.pt",
    },
    "vol_regression": {
        "Random Forest": "vol_random_forest.pkl",
        "XGBoost":       "vol_xgboost.pkl",
        "LightGBM":      "vol_lightgbm.pkl",
        "LSTM":          "vol_reg_lstm.pt",
        "GRU":           "vol_reg_gru.pt",
    },
    "vol_classification": {
        "Random Forest": "vol_clf_random_forest.pkl",
        "XGBoost":       "vol_clf_xgboost.pkl",
        "LightGBM":      "vol_clf_lightgbm.pkl",
        "LSTM":          "vol_clf_lstm.pt",
        "GRU":           "vol_clf_gru.pt",
    },
}

# which scaler each task's DL models use
TASK_SCALER = {
    "direction_daily":    "scaler_direction.pkl",
    "direction_weekly":   "scaler_direction.pkl",
    "vol_regression":     "scaler_volatility.pkl",
    "vol_classification": "scaler_volatility.pkl",
}


# ----------------------------------------------------------------- loaders
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"fx_features.csv is missing columns: {missing}")
    return df


def load_model(task: str, name: str):
    fname = REGISTRY[task][name]
    path = MODELS_DIR / fname
    if fname.endswith(".pkl"):
        return joblib.load(path)
    # .pt -> rebuild net, load weights
    rnn_type = "lstm" if "lstm" in fname else "gru"
    net = RNNClassifier(n_features=len(FEATURE_COLS), rnn_type=rnn_type)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


# ----------------------------------------------------------------- predict
def predict_latest(task, name, df=None):
    """
    Predict from the most recent rows of fx_features.csv.
    Returns a dict ready to display in Streamlit.
    """
    if df is None:
        df = load_features()

    fname = REGISTRY[task][name]
    model = load_model(task, name)
    last_date = df["date"].iloc[-1]

    is_dl = fname.endswith(".pt")

    if is_dl:
        scaler = joblib.load(MODELS_DIR / TASK_SCALER[task])
        X = scaler.transform(df[FEATURE_COLS].values)     # scale ALL rows
        window = torch.tensor(X[-WINDOW:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = float(model(window).item())             # raw logit / value
    else:
        X_last = df[FEATURE_COLS].iloc[[-1]].values       # numpy -> no name warning
        if task == "vol_regression":
            raw = float(model.predict(X_last)[0])
        else:
            raw = float(model.predict_proba(X_last)[0, 1])  # P(class=1)

    # ------- format per task
    out = {"task": task, "model": name, "as_of": str(last_date.date())}

    if task in ("direction_daily", "direction_weekly"):
        prob_up = _sigmoid(raw) if is_dl else raw
        out["prediction"] = "UP" if prob_up >= 0.5 else "DOWN"
        out["prob_up"] = round(prob_up, 4)

    elif task == "vol_regression":
        # DL outputs a logit -> sigmoid to get back into [0,1] scaled space
        scaled = _sigmoid(raw) if is_dl else raw
        vmin, vmax = VOL_META["vmin"], VOL_META["vmax"]
        vol = scaled * (vmax - vmin) + vmin               # inverse min-max
        out["vol_scaled"] = round(scaled, 4)
        out["vol_forecast_pct"] = round(vol * 100, 4)     # % per day

    elif task == "vol_classification":
        prob_high = _sigmoid(raw) if is_dl else raw
        out["prediction"] = "HIGH VOL" if prob_high >= 0.5 else "LOW VOL"
        out["prob_high"] = round(prob_high, 4)

    return out


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    df = load_features()
    print(f"Data through {df['date'].iloc[-1].date()}  ({len(df)} rows)\n")
    for task, models in REGISTRY.items():
        print(f"== {task} ==")
        for name in models:
            try:
                r = predict_latest(task, name, df)
                shown = {k: v for k, v in r.items() if k not in ("task", "model", "as_of")}
                print(f"  {name:<14} {shown}")
            except Exception as e:
                print(f"  {name:<14} ERROR: {type(e).__name__}: {e}")
        print()