"""
Run the whole project end to end, from pulling raw data to writing every
result the dashboard reads.

    python run_full_pipeline.py            # full run
    python run_full_pipeline.py --no-pull  # reuse the raw CSVs already on disk

Stages
------
1. pull the four raw sources and rebuild the master dataset
   (the Yahoo alignment fix lives inside build_master_dataset)
2. engineer the 31 features and the prediction targets
3. train 4 tasks x 5 algorithms and write their metrics
4. print a summary

Methodology, kept identical to the notebooks
--------------------------------------------
* chronological 70 / 20 / 10 split by row position -- never shuffled
* tree models see raw features; recurrent nets see 10-day windows of
  features standardised with a scaler fitted on TRAIN only
* baselines: majority class for the classifiers, and for the regression
  both the train mean (the reference R2 is defined against) and the train
  median (the one MAE is actually minimised by)

One deliberate change from the original notebooks
-------------------------------------------------
The regression target is the mean |return| over the NEXT 5 DAYS, not over
tomorrow alone. Tomorrow's |return| is a one-observation estimate of
volatility: writing it as |r| = sigma * |z|, most of its variance comes
from the unpredictable |z|, so a low R2 was guaranteed by the question
rather than by the models. Averaging five days lets |z| cancel and leaves
sigma. Measured effect: correlation 0.28 -> 0.45, and the MAE gain over
the correct (median) baseline goes from +0.8% to +13%.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
INTERIM = ROOT / "data" / "interim" / "fx_master_dataset.csv"
FEATURES = ROOT / "data" / "processed" / "fx_features.csv"

SEED, WINDOW, HORIZON = 42, 10, 5
TRAIN_FRAC, VAL_FRAC = 0.70, 0.20
DEVICE = "cpu"


def head(msg):
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74)


# ------------------------------------------------------------------ stage 1
def pull_and_build_master():
    head("STAGE 1  pull raw sources and rebuild the master dataset")
    import importlib
    for mod, fn in [("src.data.load_yfinance", "download_all"),
                    ("src.data.load_fred", "download_all"),
                    ("src.data.load_ecb", "download_all"),
                    ("src.data.load_cot", "download_cot_data")]:
        try:
            getattr(importlib.import_module(mod), fn)()
            print(f"  [ok]   {mod}")
        except Exception as e:
            print(f"  [warn] {mod} failed ({type(e).__name__}: {e}) -- "
                  "existing raw CSV will be used")
    from src.data.build_dataset import build_master_dataset
    build_master_dataset()


# ------------------------------------------------------------------ stage 2
FEATURE_COLS = json.load(open(MODELS / "vol_meta.json"))["feature_cols"]


def build_features():
    head("STAGE 2  engineer the 31 features and the targets")
    df = pd.read_csv(INTERIM, parse_dates=["date"]).sort_values("date") \
           .reset_index(drop=True)
    f = pd.DataFrame({"date": df["date"]})

    for c in ["eurusd", "dxy", "gold", "oil"]:                    # G1
        f[f"{c}_return"] = np.log(df[c] / df[c].shift(1))
    for c in ["vixcls", "dgs2", "dgs10", "ecbdfr", "t10yie",      # G2
              "eur_effective_rate", "dtwexbgs"]:
        f[f"{c}_diff"] = df[c].diff()
    f["vixcls_level"] = df["vixcls"]                              # G3
    f["eur_net_position_pct"] = df["eur_net_position_pct"]
    for base, lags in {"eurusd_return": [1, 2, 3], "dxy_return": [1, 2],
                       "vixcls_diff": [1], "gold_return": [1]}.items():
        for k in lags:                                            # G4
            f[f"{base}_lag{k}"] = f[base].shift(k)
    for w in (5, 10, 20):                                         # G5
        f[f"eurusd_return_ma{w}"] = f["eurusd_return"].rolling(w).mean()
        f[f"eurusd_return_vol{w}"] = f["eurusd_return"].rolling(w).std()
    f["dxy_return_vol10"] = f["dxy_return"].rolling(10).std()
    f["us_eu_rate_spread"] = df["dgs2"] - df["ecbdfr"]            # G6
    f["us_eu_inflation_diff"] = df["cpiaucsl"] - df["cp0000ez19m086nest"]
    f["vix_regime"] = (df["vixcls"] > 20).astype(int)
    f["day_of_week"] = df["date"].dt.dayofweek

    # targets
    f["target_return_next_day"] = f["eurusd_return"].shift(-1)
    f["target_direction"] = (f["target_return_next_day"] > 0).astype(int)
    f["eurusd"] = df["eurusd"]

    missing = [c for c in FEATURE_COLS if c not in f.columns]
    if missing:
        raise ValueError(f"feature mismatch: {missing}")

    f = f.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    out = f[["date"] + FEATURE_COLS + ["target_return_next_day",
                                       "target_direction"]]
    out.to_csv(FEATURES, index=False)
    print(f"  wrote {FEATURES.relative_to(ROOT)}  "
          f"{len(out)} rows x {out.shape[1]} cols")
    print(f"  coverage {out['date'].iloc[0].date()} -> "
          f"{out['date'].iloc[-1].date()}")
    return f


# ------------------------------------------------------------- shared parts
class RNNNet(nn.Module):
    """Same architecture as the notebooks: 1 layer, 32 hidden, 2 dropouts."""

    def __init__(self, n_features, hidden=32, dropout=0.3, rnn_type="LSTM"):
        super().__init__()
        cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = cls(input_size=n_features, hidden_size=hidden,
                       num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def windows(X, y, idx_lo, idx_hi):
    """10-day windows ending inside [idx_lo, idx_hi)."""
    xs, ys = [], []
    for i in range(idx_lo, idx_hi):
        if i < WINDOW - 1:
            continue
        xs.append(X[i - WINDOW + 1:i + 1])
        ys.append(y[i])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def train_rnn(rnn_type, Xtr, ytr, Xva, yva, score, epochs=80, lr=1e-3,
              weight_decay=1e-4, patience=12):
    """Train with early stopping on a validation score (higher = better)."""
    torch.manual_seed(SEED)
    net = RNNNet(Xtr.shape[2], rnn_type=rnn_type).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=64, shuffle=True)
    Xva_t = torch.tensor(Xva)
    best, best_state, wait = -np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xva_t)).numpy()
        s = score(yva, p)
        if s > best:
            best, best_state, wait = s, {k: v.clone() for k, v
                                         in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def slices(n):
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def acc(y, p):
    return accuracy_score(y, (p >= 0.5).astype(int)) * 100


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mae(y, p):
    return float(np.abs(y - p).mean())


# ------------------------------------------------------- classification task
def run_classification(f, y, tag, files, results_csv):
    X = f[FEATURE_COLS].values
    keep = ~pd.isna(y)
    X, y = X[keep], y[keep].astype(int)
    n = len(y)
    i70, i90 = slices(n)
    base = max(np.bincount(y[i70:i90])) / len(y[i70:i90]) * 100

    rows = {}
    for name, mdl in [
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1)),
        ("XGBoost", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbosity=0,
            eval_metric="logloss")),
        ("LightGBM", LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbose=-1)),
    ]:
        mdl.fit(X[:i70], y[:i70])
        rows[name] = [accuracy_score(y[:i70], mdl.predict(X[:i70])) * 100,
                      accuracy_score(y[i70:i90], mdl.predict(X[i70:i90])) * 100,
                      accuracy_score(y[i90:], mdl.predict(X[i90:])) * 100]
        joblib.dump(mdl, MODELS / files[name])

    scaler = StandardScaler().fit(X[:i70])
    Xs = scaler.transform(X)
    joblib.dump(scaler, MODELS / files["scaler"])
    Xtr, ytr = windows(Xs, y, 0, i70)
    Xva, yva = windows(Xs, y, i70, i90)
    Xte, yte = windows(Xs, y, i90, n)
    for name in ("LSTM", "GRU"):
        net = train_rnn(name, Xtr, ytr, Xva, yva, acc)
        with torch.no_grad():
            pr = [torch.sigmoid(net(torch.tensor(Z))).numpy()
                  for Z in (Xtr, Xva, Xte)]
        rows[name] = [acc(ytr, pr[0]), acc(yva, pr[1]), acc(yte, pr[2])]
        torch.save(net.state_dict(), MODELS / files[name])

    out = pd.DataFrame(rows, index=["train", "val", "test"]).T.round(2)
    out["val_vs_base"] = (out["val"] - base).round(2)
    out.to_csv(MODELS / results_csv)
    print(f"\n  {tag}   majority baseline on val = {base:.2f}%")
    print(out.to_string())
    return out


# ----------------------------------------------------------- regression task
def run_regression(f):
    head("STAGE 3c  volatility regression  (target: mean |return| over the "
         "next 5 days)")
    ret = f["eurusd"].pct_change()
    tgt = (ret.abs().shift(-1).rolling(HORIZON).mean()
              .shift(-(HORIZON - 1)))
    d = f.assign(raw=tgt).dropna(subset=FEATURE_COLS + ["raw"]) \
          .reset_index(drop=True)
    n = len(d)
    i70, i90 = slices(n)
    vmin = float(d["raw"].iloc[:i70].min())
    vmax = float(d["raw"].iloc[:i70].max())
    y = ((d["raw"] - vmin) / (vmax - vmin)).clip(0, 1).values
    X = d[FEATURE_COLS].values
    y_tr, y_va, y_te = y[:i70], y[i70:i90], y[i90:]

    b_mean = mae(y_va, np.full_like(y_va, y_tr.mean()))
    b_med = mae(y_va, np.full_like(y_va, np.median(y_tr)))
    print(f"  rows {n}   train {i70}  val {i90-i70}  test {n-i90}")
    print(f"  baseline MAE   mean {b_mean:.4f}   median {b_med:.4f} "
          "(median is the right reference for MAE)")

    rows = {}
    for name, mdl, fname in [
        ("Random Forest", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1), "vol_random_forest.pkl"),
        ("XGBoost", XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbosity=0),
         "vol_xgboost.pkl"),
        ("LightGBM", LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, verbose=-1),
         "vol_lightgbm.pkl"),
    ]:
        mdl.fit(X[:i70], y_tr)
        pv, pt = mdl.predict(X[i70:i90]), mdl.predict(X[i90:])
        rows[name] = dict(val_mae=mae(y_va, pv), val_r2=r2(y_va, pv),
                          val_corr=float(np.corrcoef(pv, y_va)[0, 1]),
                          test_mae=mae(y_te, pt), test_r2=r2(y_te, pt))
        joblib.dump(mdl, MODELS / fname)

    scaler = StandardScaler().fit(X[:i70])
    Xs = scaler.transform(X)
    joblib.dump(scaler, MODELS / "scaler_volatility.pkl")
    Xtr, ytr = windows(Xs, y, 0, i70)
    Xva, yva = windows(Xs, y, i70, i90)
    Xte, yte = windows(Xs, y, i90, n)
    for name, fname in [("LSTM", "vol_reg_lstm.pt"),
                        ("GRU", "vol_reg_gru.pt")]:
        net = train_rnn(name, Xtr, ytr, Xva, yva,
                        score=lambda a, b: -mae(a, b))
        with torch.no_grad():
            pv = torch.sigmoid(net(torch.tensor(Xva))).numpy()
            pt = torch.sigmoid(net(torch.tensor(Xte))).numpy()
        rows[name] = dict(val_mae=mae(yva, pv), val_r2=r2(yva, pv),
                          val_corr=float(np.corrcoef(pv, yva)[0, 1]),
                          test_mae=mae(yte, pt), test_r2=r2(yte, pt))
        torch.save(net.state_dict(), MODELS / fname)

    out = pd.DataFrame(rows).T
    out["gain_vs_mean_pct"] = 100 * (b_mean - out["val_mae"]) / b_mean
    out["gain_vs_median_pct"] = 100 * (b_med - out["val_mae"]) / b_med
    out = out.round(4)
    out.to_csv(MODELS / "volatility_regression_results.csv")
    print(out.to_string())

    json.dump({"vmin": vmin, "vmax": vmax, "horizon_days": HORIZON,
               "target": "mean absolute simple return over the next "
                         f"{HORIZON} trading days, min-max scaled on train",
               "baseline_val_mae_mean": round(b_mean, 4),
               "baseline_val_mae_median": round(b_med, 4),
               "feature_cols": FEATURE_COLS, "window": WINDOW},
              open(MODELS / "vol_meta.json", "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args()

    if not args.no_pull:
        pull_and_build_master()
    else:
        head("STAGE 1  skipped (--no-pull), reusing the master dataset")

    f = build_features()

    head("STAGE 3a  direction, next day")
    run_classification(
        f, f["target_direction"].where(f["target_return_next_day"].notna()),
        "direction daily",
        {"Random Forest": "random_forest.pkl", "XGBoost": "xgboost.pkl",
         "LightGBM": "lightgbm.pkl", "LSTM": "lstm.pt", "GRU": "gru.pt",
         "scaler": "scaler_direction.pkl"},
        "ml_results.csv")

    head("STAGE 3b  direction, next week")
    weekly = (f["eurusd"].pct_change(HORIZON).shift(-HORIZON) > 0).astype(float)
    weekly[f["eurusd"].pct_change(HORIZON).shift(-HORIZON).isna()] = np.nan
    run_classification(
        f, weekly, "direction weekly",
        {"Random Forest": "weekly_random_forest.pkl",
         "XGBoost": "weekly_xgboost.pkl", "LightGBM": "weekly_lightgbm.pkl",
         "LSTM": "weekly_lstm.pt", "GRU": "weekly_gru.pt",
         "scaler": "scaler_direction.pkl"},
        "weekly_results.csv")

    run_regression(f)

    head("STAGE 3d  volatility, high vs low")
    ret = f["eurusd"].pct_change()
    absmove = ret.abs().shift(-1)
    i70, _ = slices(int(absmove.notna().sum()))
    thr = absmove.dropna().iloc[:i70].median()
    run_classification(
        f, (absmove > thr).astype(float).where(absmove.notna()),
        "volatility high/low",
        {"Random Forest": "vol_clf_random_forest.pkl",
         "XGBoost": "vol_clf_xgboost.pkl",
         "LightGBM": "vol_clf_lightgbm.pkl", "LSTM": "vol_clf_lstm.pt",
         "GRU": "vol_clf_gru.pt", "scaler": "scaler_volatility.pkl"},
        "volatility_classification_results.csv")

    head("DONE -- all models retrained and every results CSV rewritten")


if __name__ == "__main__":
    main()
