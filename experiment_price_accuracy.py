"""
Price forecasting scored as accuracy.

The model predicts tomorrow's EUR/USD; a prediction counts as correct when
it lands within a stated tolerance. Accuracy is measured on validation and
test, never on training. Same chronological 70/20/10 split and the same 31
features as the rest of the project.

Two tolerances are reported:
  2 decimals  (+/- 50 pip)  -- like quoting VND to the nearest hundred
  +/- 60 pip                -- a touch wider, the level a risk desk uses

The naive rule "tomorrow equals today" is included as the baseline every
model has to beat.

    python experiment_price_accuracy.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "price_accuracy_results.csv"
FEATS = json.load(open(ROOT / "models" / "vol_meta.json"))["feature_cols"]

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
d = (feat.merge(mas, on="date", how="left")
         .sort_values("date").reset_index(drop=True))
d["nxt"] = d["eurusd"].shift(-1)
d = d.dropna(subset=FEATS + ["nxt"]).reset_index(drop=True)

n = len(d)
i70, i90 = int(n * 0.70), int(n * 0.90)
today, act = d["eurusd"].values, d["nxt"].values
X = d[FEATS].values
y = (act - today) / today          # models learn the return, not the level

Xs = StandardScaler().fit(X[:i70]).transform(X)

MODELS = {
    "Random Forest": (RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=20,
        random_state=42, n_jobs=-1), False),
    "XGBoost": (XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        reg_lambda=2.0, random_state=42, verbosity=0), False),
    "LightGBM": (LGBMRegressor(
        n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.03,
        reg_lambda=2.0, random_state=42, verbose=-1), False),
    "Ridge": (Ridge(alpha=10.0), True),
}

preds = {}
for name, (mdl, scale) in MODELS.items():
    xx = Xs if scale else X
    mdl.fit(xx[:i70], y[:i70])
    preds[name] = today * (1 + mdl.predict(xx))
preds["Naive (gia hom nay)"] = today.copy()

SL = {"train": (0, i70), "val": (i70, i90), "test": (i90, n)}
rows = []
for name, p in preds.items():
    err = np.abs(p - act) * 10000
    ok2 = np.round(p, 2) == np.round(act, 2)
    r = {"model": name}
    for s, (a, b) in SL.items():
        r[f"acc2dec_{s}"] = round(100 * ok2[a:b].mean(), 1)
    for s, (a, b) in SL.items():
        r[f"acc60pip_{s}"] = round(100 * (err[a:b] <= 60).mean(), 1)
    r["mae_pip_val"] = round(err[i70:i90].mean(), 1)
    r["mae_pip_test"] = round(err[i90:].mean(), 1)
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print("=" * 78)
print("ACCURACY -- du doan dung toi 2 chu so thap phan (+/- 50 pip)")
print("=" * 78)
print(res[["model", "acc2dec_train", "acc2dec_val", "acc2dec_test"]]
      .to_string(index=False))
print()
print("=" * 78)
print("ACCURACY -- trong +/- 60 pip")
print("=" * 78)
print(res[["model", "acc60pip_train", "acc60pip_val", "acc60pip_test",
           "mae_pip_val", "mae_pip_test"]].to_string(index=False))
print()
best = res.loc[res["acc2dec_val"].idxmax()]
print(f"Tot nhat tren validation (2 chu so): {best['model']}  "
      f"val {best['acc2dec_val']}%  test {best['acc2dec_test']}%")
print(f"\nWrote {OUT.relative_to(ROOT)}")
