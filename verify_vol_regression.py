"""
Recompute the volatility-regression results from the frozen snapshot.

Why this script exists
----------------------
The numbers originally stored in models/volatility_regression_results.csv
could not be reproduced.  Auditing them turned up two problems:

1. The evaluation split was a chronological 70 / 18 / 12 split of the rows
   (train_rows = 2858 = 70% of 4084, as recorded in direction_meta.json),
   not the 2022-01-01 / 2025-01-01 calendar boundaries that were being
   documented.
2. The recurrent nets are biased low -- their sigmoid output averages
   ~0.088 against a true mean of ~0.118 -- which pushed their R2 negative
   even though their correlation with reality is the best of any model.

This script recomputes everything on the real split, reports each model
against BOTH the mean baseline (the right reference for R2) and the median
baseline (the right reference for MAE), and writes a CSV every number in
the dashboard can be traced back to.

    python verify_vol_regression.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from predict_latest import RNNClassifier, VOL_DL_CALIBRATION

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT_CSV = MODELS / "volatility_regression_verified.csv"

META = json.load(open(MODELS / "vol_meta.json"))
FEATS, VMIN, VMAX, WINDOW = (META["feature_cols"], META["vmin"],
                             META["vmax"], META["window"])

TREES = {"Random Forest": "vol_random_forest.pkl",
         "XGBoost": "vol_xgboost.pkl",
         "LightGBM": "vol_lightgbm.pkl"}
NETS = {"LSTM": "vol_reg_lstm.pt", "GRU": "vol_reg_gru.pt"}

TRAIN_FRAC, VAL_FRAC = 0.70, 0.18       # the split the models were trained on


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mae(y, p):
    return np.abs(y - p).mean()


def load_frame():
    df = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                     parse_dates=["date"])
    df["vol"] = (df["target_return_next_day"].abs() - VMIN) / (VMAX - VMIN)
    return df.dropna(subset=FEATS + ["vol"]).reset_index(drop=True)


def net_predict(fname, lo, hi, Xs):
    net = RNNClassifier(n_features=len(FEATS),
                        rnn_type="lstm" if "lstm" in fname else "gru")
    net.load_state_dict(torch.load(MODELS / fname, map_location="cpu",
                                   weights_only=True))
    net.eval()
    keep = [i for i in range(lo, hi) if i >= WINDOW - 1]
    wins = np.array([Xs[i - WINDOW + 1:i + 1] for i in keep])
    with torch.no_grad():
        logit = net(torch.tensor(wins, dtype=torch.float32)).squeeze(-1).numpy()
    return 1 / (1 + np.exp(-logit)), np.array(keep)


def main():
    df = load_frame()
    n = len(df)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    y = df["vol"].values
    y_tr, y_va, y_te = y[:a], y[a:b], y[b:]

    print(f"Rows {n}   train {a}   val {b-a}   test {n-b}")
    print(f"Boundaries: {df['date'].iloc[a].date()} and "
          f"{df['date'].iloc[b].date()}\n")

    # Baselines. MAE is minimised by the median, R2 by the mean, so a
    # right-skewed target makes the mean baseline flattering on MAE.
    c_mean, c_med = y_tr.mean(), np.median(y_tr)
    base = {
        "val_mae_vs_mean": mae(y_va, np.full_like(y_va, c_mean)),
        "val_mae_vs_median": mae(y_va, np.full_like(y_va, c_med)),
        "test_mae_vs_mean": mae(y_te, np.full_like(y_te, c_mean)),
        "test_mae_vs_median": mae(y_te, np.full_like(y_te, c_med)),
    }
    print(f"Baseline (always predict train mean   {c_mean:.4f}): "
          f"val MAE {base['val_mae_vs_mean']:.4f}")
    print(f"Baseline (always predict train median {c_med:.4f}): "
          f"val MAE {base['val_mae_vs_median']:.4f}\n")

    rows = []
    Xs = joblib.load(MODELS / "scaler_volatility.pkl").transform(
        df[FEATS].values)

    for label, f in TREES.items():
        m = joblib.load(MODELS / f)
        pv, pt = m.predict(df[FEATS].values[a:b]), m.predict(df[FEATS].values[b:])
        rows.append(dict(model=label, calibrated="n/a",
                         val_mae=mae(y_va, pv), val_r2=r2(y_va, pv),
                         val_corr=np.corrcoef(pv, y_va)[0, 1],
                         test_mae=mae(y_te, pt), test_r2=r2(y_te, pt)))

    for label, f in NETS.items():
        pv, kv = net_predict(f, a, b, Xs)
        pt, kt = net_predict(f, b, n, Xs)
        yv, yt = y[kv], y[kt]
        alpha, beta = VOL_DL_CALIBRATION[f]
        for tag, qv, qt in [("raw", pv, pt),
                            ("yes", alpha * pv + beta, alpha * pt + beta)]:
            rows.append(dict(model=label, calibrated=tag,
                             val_mae=mae(yv, qv), val_r2=r2(yv, qv),
                             val_corr=np.corrcoef(qv, yv)[0, 1],
                             test_mae=mae(yt, qt), test_r2=r2(yt, qt)))

    out = pd.DataFrame(rows).round(4)
    for k, v in base.items():
        out[k] = round(v, 4)
    out["val_mae_gain_vs_mean_pct"] = (
        100 * (base["val_mae_vs_mean"] - out["val_mae"])
        / base["val_mae_vs_mean"]).round(1)
    out["val_mae_gain_vs_median_pct"] = (
        100 * (base["val_mae_vs_median"] - out["val_mae"])
        / base["val_mae_vs_median"]).round(1)

    print(out[["model", "calibrated", "val_mae", "val_r2", "val_corr",
               "val_mae_gain_vs_mean_pct", "val_mae_gain_vs_median_pct",
               "test_r2"]].to_string(index=False))

    # Robustness of the best model's R2
    best = joblib.load(MODELS / TREES["XGBoost"])
    pv = best.predict(df[FEATS].values[a:b])
    print("\nRobustness of XGBoost validation R2")
    for q in (1.0, 0.99, 0.975, 0.95):
        m = y_va <= np.quantile(y_va, q)
        print(f"  keeping the calmest {q:.1%} of days: R2 = {r2(y_va[m], pv[m]):+.4f}")
    print("  by sub-period:", "  ".join(
        f"{r2(y_va[c], pv[c]):+.4f}"
        for c in np.array_split(np.arange(len(y_va)), 3)))

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
