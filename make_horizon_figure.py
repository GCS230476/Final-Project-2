"""
Build the figure that explains why the volatility target was reposed from
one day to a five-day average.

Same features, same split, same Random Forest settings as notebook 08 --
only the target differs, so the two columns are a controlled comparison.

    python make_horizon_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "models" / "08_why_five_days.png"
HORIZON = 5

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
FEATS = [c for c in feat.columns
         if c not in ["date", "target_return_next_day", "target_direction"]]
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
df = (feat.merge(mas, on="date", how="left")
          .sort_values("date").reset_index(drop=True))

r = df["eurusd"].pct_change()
df["one_day"] = r.abs().shift(-1)
df["five_day"] = r.abs().shift(-1).rolling(HORIZON).mean().shift(-(HORIZON - 1))


def fit(col):
    d = df.dropna(subset=FEATS + [col]).reset_index(drop=True)
    n = len(d)
    i70, i90 = int(n * 0.70), int(n * 0.90)
    vmin, vmax = d[col].iloc[:i70].min(), d[col].iloc[:i70].max()
    y = ((d[col] - vmin) / (vmax - vmin)).clip(0, 1).values
    X = d[FEATS].values
    m = RandomForestRegressor(n_estimators=300, max_depth=6,
                              min_samples_leaf=20, random_state=42, n_jobs=-1)
    m.fit(X[:i70], y[:i70])
    p = m.predict(X[i70:i90])
    yv = y[i70:i90]
    r2 = 1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return d["date"].values[i70:i90], yv, p, r2, np.corrcoef(p, yv)[0, 1]


d1, y1, p1, r2_1, c1 = fit("one_day")
d5, y5, p5, r2_5, c5 = fit("five_day")

plt.style.use("default")
RED, GREEN, LIGHT = "#c0392b", "#1e7a4a", "#9fb3cc"
fig, ax = plt.subplots(2, 2, figsize=(14, 8))

# --- top row: what the model is asked to predict -------------------------
for col, (dd, yy, ttl, colr) in enumerate([
        (d1, y1, f"ONE-DAY target — mostly noise", RED),
        (d5, y5, f"{HORIZON}-DAY average target — a signal", GREEN)]):
    a = ax[0][col]
    a.plot(dd, yy, lw=0.7, color=colr)
    a.set_title(ttl, fontsize=12, fontweight="bold", color=colr)
    a.set_ylabel("Volatility ratio (0–1)")
    a.set_ylim(0, 1)
    a.grid(alpha=0.25)

# --- bottom row: how well a model can track it ---------------------------
for col, (yy, pp, r2v, cv, colr) in enumerate([
        (y1, p1, r2_1, c1, RED), (y5, p5, r2_5, c5, GREEN)]):
    a = ax[1][col]
    a.scatter(yy, pp, s=10, alpha=0.4, color="#34507a", edgecolors="none")
    lim = 1.0
    a.plot([0, lim], [0, lim], "--", color="#999", lw=1,
           label="Perfect prediction")
    slope, inter = np.polyfit(yy, pp, 1)
    xs = np.array([0, lim])
    a.plot(xs, slope * xs + inter, color=colr, lw=2,
           label=f"Model trend (slope {slope:.2f})")
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.set_xlabel("Actual")
    a.set_ylabel("Predicted")
    a.set_title(f"R² = {r2v:+.3f}   ·   correlation = {cv:.2f}",
                fontsize=12, fontweight="bold", color=colr)
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.25)

fig.suptitle("Same features, same split, same Random Forest — only the "
             "target changed", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT.relative_to(ROOT)}")
print(f"  one-day : R2={r2_1:+.4f}  corr={c1:.3f}  "
      f"pred spread={p1.std()/y1.std():.2f} of actual")
print(f"  {HORIZON}-day  : R2={r2_5:+.4f}  corr={c5:.3f}  "
      f"pred spread={p5.std()/y5.std():.2f} of actual")
