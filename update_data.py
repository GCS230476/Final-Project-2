"""
update_data.py -- "Method B": pull fresh data and rebuild features for live prediction.

Pipeline (reuses the SAME code that built the training data):
  1. Re-pull raw data via src/data loaders (leak fix for eurusd is applied
     inside build_dataset.load_yfinance -- nothing to remember manually)
  2. Rebuild data/interim/fx_master_dataset.csv via build_master_dataset()
  3. Recompute the 31 features EXACTLY as NB04 (same formulas, same names)
  4. Save data/processed/fx_features_live.csv
     (the frozen fx_features.csv snapshot is NEVER overwritten)

The live file keeps the most recent row (which has no target -- we drop NaN
on FEATURE columns only, unlike NB04 which also drops the last row because
its target is NaN).

Consistency guard: overlapping dates between live and frozen features are
compared; a large mismatch means something drifted (source revision or a
pipeline bug) and is printed loudly.

Usage (venv_dl):
    python update_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTERIM_CSV = ROOT / "data" / "interim" / "fx_master_dataset.csv"
FROZEN_CSV = ROOT / "data" / "processed" / "fx_features.csv"
LIVE_CSV = ROOT / "data" / "processed" / "fx_features_live.csv"

with open(ROOT / "models" / "direction_meta.json") as f:
    FEATURE_COLS = json.load(f)["feature_cols"]


# ---------------------------------------------------------------- step 1+2
def refresh_raw_and_master():
    """Re-pull all sources, then rebuild the master dataset."""
    import importlib

    print("=" * 70)
    print("STEP 1 -- PULLING FRESH RAW DATA")
    print("=" * 70)
    for mod_name in ["load_yfinance", "load_fred", "load_ecb", "load_cot"]:
        try:
            mod = importlib.import_module(f"src.data.{mod_name}")
            if hasattr(mod, "download_all"):
                mod.download_all()
            elif hasattr(mod, "download_cot_data"):
                mod.download_cot_data()
            elif hasattr(mod, "main"):
                mod.main()
            else:
                print(f"  [WARN] {mod_name}: no download function found, "
                      f"skipped -- existing raw CSVs will be used")
        except Exception as e:
            print(f"  [WARN] {mod_name} failed ({type(e).__name__}: {e}) -- "
                  f"existing raw CSVs will be used (ffill covers gaps)")

    print()
    print("=" * 70)
    print("STEP 2 -- REBUILDING MASTER DATASET (leak fix applied inside)")
    print("=" * 70)
    from src.data.build_dataset import build_master_dataset
    build_master_dataset()


# ---------------------------------------------------------------- step 3
def build_features() -> pd.DataFrame:
    """Recompute the 31 features exactly as NB04."""
    df = pd.read_csv(INTERIM_CSV, parse_dates=["date"]) \
           .sort_values("date").reset_index(drop=True)

    feat = pd.DataFrame()
    feat["date"] = df["date"]

    # Group 1 -- log returns
    for col in ["eurusd", "dxy", "gold", "oil"]:
        feat[f"{col}_return"] = np.log(df[col] / df[col].shift(1))

    # Group 2 -- first differences
    for col in ["vixcls", "dgs2", "dgs10", "ecbdfr", "t10yie",
                "eur_effective_rate", "dtwexbgs"]:
        feat[f"{col}_diff"] = df[col].diff()

    # Group 3 -- raw
    feat["vixcls_level"] = df["vixcls"]
    feat["eur_net_position_pct"] = df["eur_net_position_pct"]

    # Group 4 -- lags
    lag_specs = {
        "eurusd_return": [1, 2, 3],
        "dxy_return": [1, 2],
        "vixcls_diff": [1],
        "gold_return": [1],
    }
    for base_col, lags in lag_specs.items():
        for k in lags:
            feat[f"{base_col}_lag{k}"] = feat[base_col].shift(k)

    # Group 5 -- rolling stats
    for window in [5, 10, 20]:
        feat[f"eurusd_return_ma{window}"] = (
            feat["eurusd_return"].rolling(window).mean())
    for window in [5, 10, 20]:
        feat[f"eurusd_return_vol{window}"] = (
            feat["eurusd_return"].rolling(window).std())
    feat["dxy_return_vol10"] = feat["dxy_return"].rolling(10).std()

    # Group 6 -- engineered
    feat["us_eu_rate_spread"] = df["dgs2"] - df["ecbdfr"]
    feat["us_eu_inflation_diff"] = df["cpiaucsl"] - df["cp0000ez19m086nest"]
    feat["vix_regime"] = (df["vixcls"] > 20).astype(int)
    feat["day_of_week"] = df["date"].dt.dayofweek

    # Drop NaN on FEATURE columns only (keep the newest row -- it has no
    # target but is exactly the row we want to predict from)
    feat = feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Sanity: all 31 training features must exist
    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    if missing:
        raise ValueError(f"Feature mismatch vs training meta: {missing}")

    return feat[["date"] + FEATURE_COLS]


# ---------------------------------------------------------------- step 4
def consistency_check(live: pd.DataFrame):
    """Compare overlapping dates between live and frozen features."""
    if not FROZEN_CSV.exists():
        print("[WARN] Frozen snapshot not found, skipping consistency check")
        return
    frozen = pd.read_csv(FROZEN_CSV, parse_dates=["date"])
    merged = frozen.merge(live, on="date", suffixes=("_frz", "_live"))
    if merged.empty:
        print("[WARN] No overlapping dates with frozen snapshot")
        return

    check_cols = ["eurusd_return", "dxy_return", "eurusd_return_vol20",
                  "us_eu_rate_spread"]
    tail = merged.tail(250)
    print("\nCONSISTENCY CHECK vs frozen snapshot (last 250 common dates):")
    worst = 0.0
    for c in check_cols:
        d = (tail[f"{c}_frz"] - tail[f"{c}_live"]).abs().max()
        worst = max(worst, d)
        print(f"  {c:<25} max abs diff = {d:.2e}")
    if worst > 1e-4:
        print("  [ALERT] Differences are larger than expected. Possible source "
              "revision or pipeline drift -- inspect before trusting live "
              "predictions.")
    else:
        print("  OK -- live pipeline reproduces the frozen features.")


def main():
    refresh_raw_and_master()

    print()
    print("=" * 70)
    print("STEP 3 -- RECOMPUTING 31 FEATURES (NB04 logic)")
    print("=" * 70)
    live = build_features()
    print(f"Live features: {live.shape[0]} rows x {live.shape[1]} cols")
    print(f"Date range: {live['date'].min().date()} -> "
          f"{live['date'].max().date()}")

    consistency_check(live)

    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    live.to_csv(LIVE_CSV, index=False)
    print(f"\nSaved: {LIVE_CSV}")
    print("Frozen snapshot untouched:", FROZEN_CSV.name)


if __name__ == "__main__":
    main()