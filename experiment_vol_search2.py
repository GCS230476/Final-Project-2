"""
Stage 2 of the volatility search: can extra volatility features, or better
Random Forest settings, push validation R2 higher than the 0.159 that
stage 1 reached?

Two levers, both tested on horizons 5 and 10:
  features  the shared 31, or those plus longer-memory volatility columns
            (60-day realised vol, EWMA vol, and the ratio of short to long
            vol, which is the usual way of saying "is the market heating up")
  settings  a small grid over tree depth and leaf size

Same chronological 70/20/10 split, selection on validation.

    python experiment_vol_search2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "volatility_search2_results.csv"

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
BASE_FEATS = [c for c in feat.columns
              if c not in ["date", "target_return_next_day",
                           "target_direction"]]
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
df = (feat.merge(mas, on="date", how="left")
          .sort_values("date").reset_index(drop=True))

ret = df["eurusd"].pct_change()
lr = np.log(df["eurusd"] / df["eurusd"].shift(1))

# extra volatility memory -- all computed from information up to today only
df["vol60"] = lr.rolling(60).std()
df["vol_ewma"] = lr.ewm(span=20, adjust=False).std()
df["vol_ratio_5_60"] = lr.rolling(5).std() / lr.rolling(60).std()
df["vol_ratio_20_60"] = lr.rolling(20).std() / lr.rolling(60).std()
df["absret_ma60"] = lr.abs().rolling(60).mean()
EXTRA = ["vol60", "vol_ewma", "vol_ratio_5_60", "vol_ratio_20_60",
         "absret_ma60"]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


GRID = [dict(max_depth=d, min_samples_leaf=l)
        for d in (4, 6, 8, 12) for l in (5, 20, 50)]

rows = []
for horizon in (5, 10):
    tgt = ret.abs().shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    for fset, cols in (("31 shared", BASE_FEATS),
                       ("31 + vol memory", BASE_FEATS + EXTRA)):
        d = df.assign(t=tgt).dropna(subset=cols + ["t"]).reset_index(drop=True)
        n = len(d)
        i70, i90 = int(n * 0.70), int(n * 0.90)
        y, X = d["t"].values, d[cols].values
        for g in GRID:
            m = RandomForestRegressor(n_estimators=300, random_state=42,
                                      n_jobs=-1, **g)
            m.fit(X[:i70], y[:i70])
            pv, pt = m.predict(X[i70:i90]), m.predict(X[i90:])
            rows.append(dict(horizon=horizon, features=fset, **g,
                             val_r2=r2(y[i70:i90], pv),
                             test_r2=r2(y[i90:], pt),
                             val_corr=np.corrcoef(pv, y[i70:i90])[0, 1]))
        print(f"  done: horizon={horizon}  features={fset}")

res = pd.DataFrame(rows).round(4)
res.to_csv(OUT, index=False)

print("\n" + "=" * 78)
print("BEST SETTING PER (horizon, feature set) -- chosen on validation")
print("=" * 78)
for (h, f), sub in res.groupby(["horizon", "features"]):
    b = sub.loc[sub["val_r2"].idxmax()]
    print(f"  h={h:2d} | {f:<16} | depth {int(b['max_depth']):2d} leaf "
          f"{int(b['min_samples_leaf']):2d} | val R2 {b['val_r2']:+.4f} | "
          f"test R2 {b['test_r2']:+.4f} | corr {b['val_corr']:.3f}")

best = res.loc[res["val_r2"].idxmax()]
print("\n" + "=" * 78)
print("OVERALL BEST ON VALIDATION")
print("=" * 78)
print(f"  horizon {int(best['horizon'])} days, {best['features']}, "
      f"depth {int(best['max_depth'])}, leaf {int(best['min_samples_leaf'])}")
print(f"  val R2 {best['val_r2']:+.4f}   test R2 {best['test_r2']:+.4f}   "
      f"corr {best['val_corr']:.3f}")
print(f"\nWrote {OUT.relative_to(ROOT)}")
