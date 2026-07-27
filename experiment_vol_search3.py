"""
Stage 3: improve the model itself, without touching the 5-day horizon.

Stage 2 showed shallower trees help. This asks whether the bigger problem is
that most of the shared feature set is noise for this particular task: 17 of
the 31 columns correlate below 0.05 with volatility, so a tree spends its
splits wading through them.

Levers, all at horizon 5:
  feature subset   top-k features ranked by |correlation| with the target,
                   ranked on TRAIN ONLY so the choice never sees validation
  tree settings    depth and leaf-size grid

    python experiment_vol_search3.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "volatility_search3_results.csv"
HORIZON = 5

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
FEATS = [c for c in feat.columns
         if c not in ["date", "target_return_next_day", "target_direction"]]
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
df = (feat.merge(mas, on="date", how="left")
          .sort_values("date").reset_index(drop=True))
ret = df["eurusd"].pct_change()
df["t"] = ret.abs().shift(-1).rolling(HORIZON).mean().shift(-(HORIZON - 1))
d = df.dropna(subset=FEATS + ["t"]).reset_index(drop=True)
n = len(d)
i70, i90 = int(n * 0.70), int(n * 0.90)
y = d["t"].values


def r2(a, p):
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()


# rank features on TRAIN ONLY
tr = d.iloc[:i70]
rank = (tr[FEATS].apply(lambda c: abs(np.corrcoef(c, tr["t"])[0, 1])
                        if c.std() > 0 else 0)
        .sort_values(ascending=False))
print("Feature ranking computed on train only. Top 8:")
for k, v in rank.head(8).items():
    print(f"  {k:<26} {v:.3f}")

rows = []
for k in (4, 6, 8, 10, 15, 31):
    cols = list(rank.head(k).index)
    X = d[cols].values
    for depth in (3, 4, 5, 6, 8):
        for leaf in (5, 20, 50):
            m = RandomForestRegressor(n_estimators=400, max_depth=depth,
                                      min_samples_leaf=leaf, random_state=42,
                                      n_jobs=-1)
            m.fit(X[:i70], y[:i70])
            pv, pt = m.predict(X[i70:i90]), m.predict(X[i90:])
            rows.append(dict(n_features=k, max_depth=depth,
                             min_samples_leaf=leaf,
                             train_r2=r2(y[:i70], m.predict(X[:i70])),
                             val_r2=r2(y[i70:i90], pv),
                             test_r2=r2(y[i90:], pt),
                             val_corr=np.corrcoef(pv, y[i70:i90])[0, 1]))
    print(f"  done: top-{k} features")

res = pd.DataFrame(rows).round(4)
res.to_csv(OUT, index=False)

print("\n" + "=" * 74)
print("BEST SETTING PER FEATURE-SUBSET SIZE (chosen on validation)")
print("=" * 74)
for k, sub in res.groupby("n_features"):
    b = sub.loc[sub["val_r2"].idxmax()]
    print(f"  top-{k:<2} | depth {int(b['max_depth'])} leaf "
          f"{int(b['min_samples_leaf']):2d} | train {b['train_r2']:+.3f} | "
          f"val R2 {b['val_r2']:+.4f} | test R2 {b['test_r2']:+.4f} | "
          f"corr {b['val_corr']:.3f}")

best = res.loc[res["val_r2"].idxmax()]
print("\n" + "=" * 74)
print(f"OVERALL BEST (horizon stays at {HORIZON} days)")
print("=" * 74)
print(f"  top-{int(best['n_features'])} features, depth "
      f"{int(best['max_depth'])}, leaf {int(best['min_samples_leaf'])}")
print(f"  val R2 {best['val_r2']:+.4f}   test R2 {best['test_r2']:+.4f}   "
      f"corr {best['val_corr']:.3f}")
print(f"\nWrote {OUT.relative_to(ROOT)}")
