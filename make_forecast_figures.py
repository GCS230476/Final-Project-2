"""
Build the numbers and charts for the "Du bao ngay mai" preview tab.

Everything here is measured the same way: fit or calibrate on the training
slice only, then report validation and test. Writes one CSV of results and
three figures.

    python make_forecast_figures.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures" / "models"
META = json.load(open(ROOT / "models" / "vol_meta.json"))
VMIN, VMAX, HZ = META["vmin"], META["vmax"], META["horizon_days"]
FEATS = META["feature_cols"]

feat = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                   parse_dates=["date"])
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
d = (feat.merge(mas, on="date", how="left")
         .sort_values("date").reset_index(drop=True))
d["am"] = (d["eurusd"].pct_change().abs().shift(-1)
           .rolling(HZ).mean().shift(-(HZ - 1)))
d["nxt"] = d["eurusd"].shift(-1)
d = d.dropna(subset=FEATS + ["am", "nxt"]).reset_index(drop=True)

n = len(d)
i70, i90 = int(n * 0.70), int(n * 0.90)
today, act = d["eurusd"].values, d["nxt"].values
dates = d["date"].values

rf = joblib.load(ROOT / "models" / "vol_random_forest.pkl")
vol = rf.predict(d[FEATS].values) * (VMAX - VMIN) + VMIN     # forecast vol
ret = (act - today) / today
z = ret / vol

SLICES = {"train": (0, i70), "val": (i70, i90), "test": (i90, n)}
rows = []

# ---- 1. accuracy of the point forecast, by tolerance -------------------
err_pip = np.abs(act - today) * 10000
for tol in (10, 20, 30, 40, 50, 60, 80, 100):
    r = {"metric": "accuracy_pip", "level": tol}
    for k, (a, b) in SLICES.items():
        r[k] = round(100 * (err_pip[a:b] <= tol).mean(), 1)
    rows.append(r)

# ---- 2. accuracy by decimal places -------------------------------------
for nd in (1, 2, 3, 4):
    ok = np.round(today, nd) == np.round(act, nd)
    r = {"metric": "accuracy_decimals", "level": nd}
    for k, (a, b) in SLICES.items():
        r[k] = round(100 * ok[a:b].mean(), 1)
    rows.append(r)

# ---- 3. calibrated confidence intervals --------------------------------
ks = {}
for conf in (0.60, 0.70, 0.80, 0.90):
    k = np.quantile(np.abs(z[:i70]), conf)      # calibrate on TRAIN only
    ks[conf] = k
    r = {"metric": "coverage", "level": int(conf * 100)}
    for s, (a, b) in SLICES.items():
        r[s] = round(100 * (np.abs(z[a:b]) <= k).mean(), 1)
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "models" / "forecast_accuracy.csv", index=False)
print(res.to_string(index=False))

# ================= figures =================
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
NAVY, GREEN, RED, GREY = "#1b2a4a", "#1e7a4a", "#c0392b", "#8ba0bd"

# --- Fig 1: calibration curve ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.2))
cov = res[res["metric"] == "coverage"]
ax.plot([55, 95], [55, 95], "--", color="#999", lw=1.2,
        label="Lý tưởng (nói sao đúng vậy)")
for col, c, mk in [("val", NAVY, "o"), ("test", GREEN, "s")]:
    ax.plot(cov["level"], cov[col], mk + "-", color=c, lw=2, ms=8,
            label=f"Thực tế trên {col}")
ax.set_xlabel("Bot tuyên bố mức tin cậy (%)")
ax.set_ylabel("Thực tế đúng bao nhiêu (%)")
ax.set_title("Bot nói 80% thì có đúng 80% không?\n"
             "Đường trùng đường đứt nét = đáng tin", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(FIG / "10_calibration.png", dpi=110, facecolor="white")

# --- Fig 2: accuracy vs tolerance --------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.2))
acc = res[res["metric"] == "accuracy_pip"]
for col, c, mk in [("val", NAVY, "o"), ("test", GREEN, "s")]:
    ax.plot(acc["level"], acc[col], mk + "-", color=c, lw=2, ms=7,
            label=f"Trên {col}")
ax.axhspan(60, 80, color=GREEN, alpha=0.10)
ax.text(84, 70, "vùng 60–80%", color=GREEN, fontsize=9, va="center")
ax.axvline(50, color=RED, ls=":", lw=1.4)
ax.text(52, 25, "±50 pip\n= đúng 2 chữ số", color=RED, fontsize=9)
ax.set_xlabel("Sai số cho phép (pip)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Cho phép sai bao nhiêu thì bot đúng bao nhiêu",
             fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIG / "10_accuracy_tolerance.png", dpi=110, facecolor="white")

# --- Fig 3: price with forecast band over validation --------------------
a, b = i70, i90
sl = slice(a, min(a + 400, b))                 # first ~400 days, readable
k80 = ks[0.80]
lo = today[sl] * (1 - k80 * vol[sl])
hi = today[sl] * (1 + k80 * vol[sl])
inside = (act[sl] >= lo) & (act[sl] <= hi)
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(dates[sl], lo, hi, color=NAVY, alpha=0.18,
                label="Khoảng dự báo 80%")
ax.plot(dates[sl], act[sl], color=NAVY, lw=1.1, label="Giá thực tế hôm sau")
ax.scatter(dates[sl][~inside], act[sl][~inside], s=18, color=RED, zorder=5,
           label=f"Rơi ra ngoài ({(~inside).sum()}/{len(inside)} ngày)")
ax.set_ylabel("EUR/USD")
ax.set_title(f"Khoảng dự báo 80% so với giá thực tế — phủ đúng "
             f"{100*inside.mean():.1f}% số ngày", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(FIG / "10_forecast_band.png", dpi=110, facecolor="white")

print("\nWrote 3 figures + models/forecast_accuracy.csv")
print(f"Band coverage on the plotted window: {100*inside.mean():.1f}%")

# ---- today's forecast, for the page -----------------------------------
live_p = ROOT / "data" / "processed" / "fx_features_live.csv"
live = pd.read_csv(live_p, parse_dates=["date"]) if live_p.exists() else feat
px = mas["eurusd"].dropna().iloc[-1]
v = float(rf.predict(live[FEATS].values[-1:])[0]) * (VMAX - VMIN) + VMIN
out = {"price": float(px), "vol": v,
       "k": {str(int(c * 100)): float(k) for c, k in ks.items()},
       "as_of": str(mas["date"].iloc[-1].date())}
json.dump(out, open(ROOT / "models" / "forecast_today.json", "w"), indent=2)
print(f"Today {px:.4f}, vol {v*100:.3f}%  -> models/forecast_today.json")
