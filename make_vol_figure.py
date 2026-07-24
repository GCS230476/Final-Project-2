"""Regenerate figures/models/08_volatility_regression.png for the 5-day
target, using LSTM (the best regression model after the rebuild)."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_full_pipeline import RNNNet, windows, FEATURE_COLS, slices, HORIZON

ROOT = Path(__file__).resolve().parent
meta = json.load(open(ROOT / "models" / "vol_meta.json"))
vmin, vmax = meta["vmin"], meta["vmax"]

f = pd.read_csv(ROOT / "data" / "processed" / "fx_features.csv",
                parse_dates=["date"])
mas = pd.read_csv(ROOT / "data" / "interim" / "fx_master_dataset.csv",
                  parse_dates=["date"])[["date", "eurusd"]]
f = f.merge(mas, on="date", how="left")
ret = f["eurusd"].pct_change()
tgt = ret.abs().shift(-1).rolling(HORIZON).mean().shift(-(HORIZON - 1))
d = f.assign(raw=tgt).dropna(subset=FEATURE_COLS + ["raw"]).reset_index(drop=True)
n = len(d)
i70, i90 = slices(n)
y = ((d["raw"] - vmin) / (vmax - vmin)).clip(0, 1).values

Xs = joblib.load(ROOT / "models" / "scaler_volatility.pkl").transform(
    d[FEATURE_COLS].values)
net = RNNNet(len(FEATURE_COLS), rnn_type="LSTM")
net.load_state_dict(torch.load(ROOT / "models" / "vol_reg_lstm.pt",
                               weights_only=True))
net.eval()
Xva, yva = windows(Xs, y, i70, i90)
with torch.no_grad():
    pred = torch.sigmoid(net(torch.tensor(Xva))).numpy()

dates = d["date"].values[i70 + (len(d[i70:i90]) - len(yva)):i90]
r2 = 1 - ((yva - pred) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()
corr = np.corrcoef(pred, yva)[0, 1]

plt.style.use("default")
NAVY, LIGHT = "#1b2a4a", "#8ba0bd"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))

ax1.plot(dates, yva, color=LIGHT, lw=1.0, label="Actual (5-day avg)")
ax1.plot(dates, pred, color=NAVY, lw=1.4, label="Predicted (LSTM)")
ax1.set_title(f"Next-5-day volatility: predicted vs actual  "
              f"(LSTM, validation — R² = {r2:+.3f}, corr = {corr:.2f})",
              fontsize=12, fontweight="bold")
ax1.set_ylabel("Volatility ratio (0–1)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

ax2.scatter(yva, pred, s=12, alpha=0.5, color=NAVY, edgecolors="none")
lim = max(yva.max(), pred.max()) * 1.05
ax2.plot([0, lim], [0, lim], "--", color="#888", lw=1, label="Perfect prediction")
a, b = np.polyfit(yva, pred, 1)
xs = np.array([0, lim])
ax2.plot(xs, a * xs + b, color="#c0392b", lw=1.6,
         label=f"Model trend (slope {a:.2f})")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_xlabel("Actual ratio")
ax2.set_ylabel("Predicted ratio")
ax2.set_title("Predicted vs actual (scatter) — the upward red trend is the "
              "skill", fontsize=12, fontweight="bold")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.25)

plt.tight_layout()
out = ROOT / "figures" / "models" / "08_volatility_regression.png"
fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print(f"wrote {out.relative_to(ROOT)}  |  R2={r2:+.4f}  corr={corr:.3f}  "
      f"pred_std/actual_std={pred.std()/yva.std():.2f}")
