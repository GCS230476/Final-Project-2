"""
Train the direct price forecaster and score it as accuracy.

The model predicts tomorrow's EUR/USD level. Internally it learns the
return (price levels are non-stationary, so learning them directly would
just memorise the 2010-2021 range) and the prediction is rebuilt as
    forecast = today's price * (1 + predicted return)

A forecast counts as correct when it lands within a stated tolerance of the
actual close. Accuracy is measured on validation and test only, on the same
chronological 70/20/10 split used everywhere else in the project.

Saves the model, a results CSV and one figure.

    python train_price_model.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
MODELS, FIG = ROOT / "models", ROOT / "figures" / "models"
FEATS = json.load(open(MODELS / "vol_meta.json"))["feature_cols"]
TOLS = [30, 40, 50, 55, 60, 70, 80, 100]

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
y = (act - today) / today

scaler = StandardScaler().fit(X[:i70])
Xs = scaler.transform(X)

CANDIDATES = {
    "Random Forest": (RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=20,
        random_state=42, n_jobs=-1), False),
    "XGBoost": (XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, reg_lambda=2.0,
        random_state=42, verbosity=0), False),
    "LightGBM": (LGBMRegressor(
        n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.03,
        reg_lambda=2.0, random_state=42, verbose=-1), False),
    "Ridge": (Ridge(alpha=10.0), True),
}

SL = {"train": (0, i70), "val": (i70, i90), "test": (i90, n)}
rows, fitted = [], {}

for name, (mdl, scale) in CANDIDATES.items():
    xx = Xs if scale else X
    mdl.fit(xx[:i70], y[:i70])
    fitted[name] = (mdl, scale)
    pred = today * (1 + mdl.predict(xx))
    err = np.abs(pred - act) * 10000
    for tol in TOLS:
        r = {"model": name, "tol_pip": tol}
        for s, (a, b) in SL.items():
            r[s] = round(100 * (err[a:b] <= tol).mean(), 1)
        rows.append(r)

# the do-nothing rule every model must beat
err_naive = np.abs(today - act) * 10000
for tol in TOLS:
    r = {"model": "Naive (gia hom nay)", "tol_pip": tol}
    for s, (a, b) in SL.items():
        r[s] = round(100 * (err_naive[a:b] <= tol).mean(), 1)
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(MODELS / "price_forecast_accuracy.csv", index=False)

piv = res.pivot_table(index="tol_pip", columns="model", values="val")
print("ACCURACY tren VALIDATION (%) theo sai so cho phep")
print(piv.round(1).to_string())
print()
piv_t = res.pivot_table(index="tol_pip", columns="model", values="test")
print("ACCURACY tren TEST (%)")
print(piv_t.round(1).to_string())

# pick the champion on validation, averaged over the tolerances that matter
score = (res[res["tol_pip"].isin([50, 55, 60])]
         .groupby("model")["val"].mean().sort_values(ascending=False))
best = score.index[0]
print(f"\nTot nhat tren validation (trung binh 50/55/60 pip): {best}")
print(score.round(2).to_string())

mdl, scale = fitted[best] if best in fitted else fitted["Random Forest"]
joblib.dump({"model": mdl, "scaled": scale, "scaler": scaler,
             "features": FEATS, "name": best},
            MODELS / "price_forecaster.pkl")

meta = {"best_model": best, "tolerances": TOLS,
        "split": {"train": i70, "val": i90 - i70, "test": n - i90},
        "boundaries": [str(d["date"].iloc[i70].date()),
                       str(d["date"].iloc[i90].date())]}
json.dump(meta, open(MODELS / "price_meta.json", "w"), indent=2)

# ---- figure: accuracy vs tolerance, model against the naive rule -------
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
fig, ax = plt.subplots(figsize=(8, 5))
for nm, c, ls in [(best, "#1b2a4a", "-"),
                  ("Naive (gia hom nay)", "#c0392b", "--")]:
    sub = res[res["model"] == nm]
    ax.plot(sub["tol_pip"], sub["val"], ls, color=c, marker="o", lw=2,
            label=f"{nm} — validation")
    ax.plot(sub["tol_pip"], sub["test"], ls, color=c, marker="s", lw=1.2,
            alpha=0.55, label=f"{nm} — test")
ax.axhspan(70, 80, color="#1e7a4a", alpha=0.10)
ax.text(101, 75, "vùng\n70–80%", color="#1e7a4a", fontsize=9, va="center")
ax.axvline(60, color="#8ba0bd", ls=":", lw=1.4)
ax.text(61, 40, "±60 pip", color="#5a6b82", fontsize=9)
ax.set_xlabel("Sai số cho phép (pip)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Dự báo giá EUR/USD: accuracy theo sai số cho phép",
             fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(FIG / "11_price_accuracy.png", dpi=110, facecolor="white")

# ---- today's forecast --------------------------------------------------
live_p = ROOT / "data" / "processed" / "fx_features_live.csv"
live = pd.read_csv(live_p, parse_dates=["date"]) if live_p.exists() else feat
xrow = live[FEATS].values[-1:]
xrow = scaler.transform(xrow) if scale else xrow
px = float(mas["eurusd"].dropna().iloc[-1])
fc = px * (1 + float(mdl.predict(xrow)[0]))
json.dump({"today": px, "forecast": fc,
           "as_of": str(mas["date"].iloc[-1].date()), "model": best},
          open(MODELS / "price_today.json", "w"), indent=2)
print(f"\nHom nay {px:.4f} -> du bao {fc:.4f} "
      f"({(fc-px)*10000:+.1f} pip)")
print("Da luu price_forecaster.pkl, price_forecast_accuracy.csv, figure")
