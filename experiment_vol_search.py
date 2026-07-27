"""
Systematic search for the best volatility-regression setup.

Purpose is twofold: find a better configuration if one exists, and leave a
documented record of what was tried if one does not. Every cell of the
output table is a real fit on the same features and the same chronological
70/20/10 split -- only the design choice named in the row changes.

Levers swept
------------
horizon    1, 5, 10, 20 trading days averaged into the target
transform  raw ratio, or log volatility (volatility is roughly log-normal,
           so modelling logs can behave better)
model      Random Forest, XGBoost, Ridge (Ridge is the sanity check: if a
           linear model matches the trees, the signal is simply weak)

Selection follows the project rule: choose on validation, report test once.

    python experiment_vol_search.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "volatility_search_results.csv"
EPS = 1e-8

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
FEATS = [c for c in feat.columns
         if c not in ["date", "target_return_next_day", "target_direction"]]
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
base = (feat.merge(mas, on="date", how="left")
            .sort_values("date").reset_index(drop=True))
RET = base["eurusd"].pct_change()


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0),
        "Ridge": Ridge(alpha=10.0),
    }


rows = []
for horizon in (1, 5, 10, 20):
    raw = RET.abs().shift(-1)
    if horizon > 1:
        raw = raw.rolling(horizon).mean().shift(-(horizon - 1))
    for transform in ("raw", "log"):
        col = np.log(raw + EPS) if transform == "log" else raw
        d = base.assign(t=col).dropna(subset=FEATS + ["t"]).reset_index(drop=True)
        n = len(d)
        i70, i90 = int(n * 0.70), int(n * 0.90)
        y = d["t"].values
        X = d[FEATS].values
        # standardise features once; Ridge needs it, trees are indifferent
        sc = StandardScaler().fit(X[:i70])
        Xs = sc.transform(X)
        for name, mdl in make_models().items():
            xx = Xs if name == "Ridge" else X
            mdl.fit(xx[:i70], y[:i70])
            pv, pt = mdl.predict(xx[i70:i90]), mdl.predict(xx[i90:])
            rows.append(dict(
                horizon=horizon, transform=transform, model=name,
                val_r2=r2(y[i70:i90], pv), test_r2=r2(y[i90:], pt),
                val_corr=np.corrcoef(pv, y[i70:i90])[0, 1],
                train_r2=r2(y[:i70], mdl.predict(xx[:i70]))))
        print(f"  done: horizon={horizon:2d}  transform={transform}")

res = pd.DataFrame(rows).round(4)
res.to_csv(OUT, index=False)

print("\n" + "=" * 78)
print("FULL SEARCH -- validation R2 by horizon and target transform")
print("=" * 78)
piv = res.pivot_table(index=["horizon", "transform"], columns="model",
                      values="val_r2")
print(piv.round(4).to_string())

print("\n" + "=" * 78)
print("BEST SETUP PER HORIZON (chosen on validation)")
print("=" * 78)
for h in sorted(res["horizon"].unique()):
    sub = res[res["horizon"] == h]
    b = sub.loc[sub["val_r2"].idxmax()]
    print(f"  {h:2d} days | {b['model']:<14} {b['transform']:<4} | "
          f"val R2 {b['val_r2']:+.4f} | test R2 {b['test_r2']:+.4f} | "
          f"corr {b['val_corr']:.3f}")

best = res.loc[res["val_r2"].idxmax()]
print(f"\nOVERALL BEST ON VALIDATION: horizon {int(best['horizon'])} days, "
      f"{best['model']}, {best['transform']} target")
print(f"  val R2 {best['val_r2']:+.4f}   test R2 {best['test_r2']:+.4f}")
print(f"\nWrote {OUT.relative_to(ROOT)}")
