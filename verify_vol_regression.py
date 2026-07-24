"""
Recompute the volatility-regression results, replicating notebook 08 exactly.

Why this script exists
----------------------
The dashboard used to quote numbers that lived only inside a CSV, with no
way to check them. This script rebuilds them from the frozen snapshot so
every figure in the Results chapter can be traced back to code.

It follows notebooks/08_volatility_regression.ipynb line for line:
  * target  = |tomorrow's simple return| of the raw EUR/USD price
  * split   = chronological 70 / 20 / 10 by row position
  * scaling = min-max fitted on TRAIN ONLY, then clipped to [0, 1]

On top of that it reports two things the notebook did not:
  1. MAE against a MEDIAN baseline as well as the mean baseline. MAE is
     minimised by the median, and this target is strongly right-skewed, so
     the mean baseline flatters every model.
  2. The recurrent nets with and without the bias calibration applied in
     predict_latest.py.

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
WINDOW = json.load(open(MODELS / "vol_meta.json"))["window"]

TREES = {"Random Forest": "vol_random_forest.pkl",
         "XGBoost": "vol_xgboost.pkl",
         "LightGBM": "vol_lightgbm.pkl"}
NETS = {"LSTM": "vol_reg_lstm.pt", "GRU": "vol_reg_gru.pt"}


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mae(y, p):
    return np.abs(y - p).mean()


def build():
    """Rebuild the notebook's frame, target and split."""
    feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                       parse_dates=["date"])
    cols = [c for c in feat.columns
            if c not in ["date", "target_return_next_day", "target_direction"]]
    master = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                         parse_dates=["date"])[["date", "eurusd"]]
    df = (feat.merge(master, on="date", how="left")
              .sort_values("date").reset_index(drop=True))
    df["abs_move"] = df["eurusd"].pct_change().shift(-1).abs()
    df = df.dropna(subset=["abs_move"]).reset_index(drop=True)

    n = len(df)
    i70, i90 = int(n * 0.70), int(n * 0.90)
    vmin = df["abs_move"].iloc[:i70].min()
    vmax = df["abs_move"].iloc[:i70].max()
    df["y"] = ((df["abs_move"] - vmin) / (vmax - vmin)).clip(0, 1)
    return df, cols, i70, i90, vmin, vmax


def net_predict(fname, lo, hi, Xs):
    net = RNNClassifier(n_features=Xs.shape[1],
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
    df, cols, i70, i90, vmin, vmax = build()
    n = len(df)
    y = df["y"].values
    y_tr, y_va, y_te = y[:i70], y[i70:i90], y[i90:]
    X = df[cols].values

    print(f"Rows {n}   train {i70}   val {i90-i70}   test {n-i90}")
    print(f"Boundaries: {df['date'].iloc[i70].date()} and "
          f"{df['date'].iloc[i90].date()}")
    print(f"Scaling from TRAIN only: vmin {vmin:.5f}  vmax {vmax:.5f}\n")

    c_mean, c_med = y_tr.mean(), np.median(y_tr)
    base = {"val_mae_vs_mean": mae(y_va, np.full_like(y_va, c_mean)),
            "val_mae_vs_median": mae(y_va, np.full_like(y_va, c_med))}
    print(f"Baseline, always predict train mean   ({c_mean:.4f}): "
          f"val MAE {base['val_mae_vs_mean']:.4f}   <- the notebook's baseline")
    print(f"Baseline, always predict train median ({c_med:.4f}): "
          f"val MAE {base['val_mae_vs_median']:.4f}   <- correct one for MAE\n")

    rows = []
    Xs = joblib.load(MODELS / "scaler_volatility.pkl").transform(X)

    for label, f in TREES.items():
        m = joblib.load(MODELS / f)
        pv, pt = m.predict(X[i70:i90]), m.predict(X[i90:])
        rows.append(dict(model=label, calibrated="n/a",
                         val_mae=mae(y_va, pv), val_r2=r2(y_va, pv),
                         val_corr=np.corrcoef(pv, y_va)[0, 1],
                         test_mae=mae(y_te, pt), test_r2=r2(y_te, pt)))

    for label, f in NETS.items():
        pv, kv = net_predict(f, i70, i90, Xs)
        pt, kt = net_predict(f, i90, n, Xs)
        yv, yt = y[kv], y[kt]
        alpha, beta = VOL_DL_CALIBRATION[f]
        for tag, qv, qt in [("raw", pv, pt),
                            ("yes", alpha * pv + beta, alpha * pt + beta)]:
            rows.append(dict(model=label, calibrated=tag,
                             val_mae=mae(yv, qv), val_r2=r2(yv, qv),
                             val_corr=np.corrcoef(qv, yv)[0, 1],
                             test_mae=mae(yt, qt), test_r2=r2(yt, qt)))

    out = pd.DataFrame(rows).round(4)
    out["val_mae_gain_vs_mean_pct"] = (
        100 * (base["val_mae_vs_mean"] - out["val_mae"])
        / base["val_mae_vs_mean"]).round(1)
    out["val_mae_gain_vs_median_pct"] = (
        100 * (base["val_mae_vs_median"] - out["val_mae"])
        / base["val_mae_vs_median"]).round(1)
    for k, v in base.items():
        out[k] = round(v, 4)

    print(out[["model", "calibrated", "val_mae", "val_r2", "val_corr",
               "val_mae_gain_vs_mean_pct", "val_mae_gain_vs_median_pct",
               "test_r2"]].to_string(index=False))

    best = joblib.load(MODELS / TREES["XGBoost"])
    pv = best.predict(X[i70:i90])
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
