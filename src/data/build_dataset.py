"""
Build master dataset by merging 22 series from 4 sources.

LOGIC:
1. Use daily business days as the backbone (date index)
2. Merge daily series directly
3. Forward-fill monthly/weekly series (avoid look-ahead bias)
4. Output: data/interim/fx_master_dataset.csv

ECONOMIC RATIONALE for forward-fill:
- On day T, we only know CPI from the most recent monthly release.
- Forward-fill correctly simulates "what info was available at time T".
- Backward-fill would leak future info -> data leakage.

Output schema (~24 columns):
- date: business day index
- yfinance: eurusd, dxy, gold, oil
- fred:     dff, dgs2, dgs10, ecbdfr, irltlt..., cpiaucsl, cp..., t10yie,
            unrate, payems, vixcls, dtwexbgs, dexuseu
- ecb:      eurusd_official, eur_effective_rate, euribor_3m, ester_overnight
- cot:      eur_net_position, eur_net_position_pct

Usage:
    python -m src.data.build_dataset
"""
from pathlib import Path

import pandas as pd

# ============================================================================
# Path config
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_yfinance() -> pd.DataFrame:
    """Load 4 yfinance CSVs and merge into a single DataFrame indexed by date."""
    yf_dir = RAW_DIR / "yfinance"
    dfs = []

    for asset in ["eurusd", "dxy", "gold", "oil"]:
        path = yf_dir / f"{asset}.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.rename(columns={"Date": "date", "Close": asset})
        df = df[["date", asset]]
        dfs.append(df)

    # Inner merge to keep only dates available in all 4 assets
    master = dfs[0]
    for df in dfs[1:]:
        master = master.merge(df, on="date", how="inner")

    print(f"  [yfinance] {len(master)} rows, {len(master.columns)-1} cols")
    return master


def load_fred() -> pd.DataFrame:
    """Load all FRED CSVs and merge."""
    fred_dir = RAW_DIR / "fred"
    csv_files = sorted(fred_dir.glob("*.csv"))

    if not csv_files:
        print(f"  [WARN] No FRED files found")
        return pd.DataFrame()

    # Each CSV has columns: date, <SERIES_ID>
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])

        # Lowercase column name for consistency
        series_col = [c for c in df.columns if c != "date"][0]
        df = df.rename(columns={series_col: series_col.lower()})
        dfs.append(df)

    # Outer merge (each series has different date coverage)
    fred = dfs[0]
    for df in dfs[1:]:
        fred = fred.merge(df, on="date", how="outer")

    fred = fred.sort_values("date").reset_index(drop=True)
    print(f"  [fred]     {len(fred)} rows, {len(fred.columns)-1} cols")
    return fred


def load_ecb() -> pd.DataFrame:
    """Load all ECB CSVs and merge."""
    ecb_dir = RAW_DIR / "ecb"
    csv_files = sorted(ecb_dir.glob("*.csv"))

    if not csv_files:
        print(f"  [WARN] No ECB files found")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    ecb = dfs[0]
    for df in dfs[1:]:
        ecb = ecb.merge(df, on="date", how="outer")

    ecb = ecb.sort_values("date").reset_index(drop=True)
    print(f"  [ecb]      {len(ecb)} rows, {len(ecb.columns)-1} cols")
    return ecb


def load_cot() -> pd.DataFrame:
    """Load COT data."""
    cot_path = RAW_DIR / "cot" / "eur_cot.csv"

    if not cot_path.exists():
        print(f"  [WARN] No COT file found")
        return pd.DataFrame()

    df = pd.read_csv(cot_path)
    df["date"] = pd.to_datetime(df["date"])
    # Only keep features, not raw long/short
    # Aggregate duplicate dates (CFTC publishes multiple EUR contracts on same date:
    # EURO FX + EURO FX CROSS RATES)
    df = df.groupby("date", as_index=False).agg({
        "eur_net_position": "sum",
        "eur_net_position_pct": "mean",
    })
    print(f"  [cot]      {len(df)} rows, {len(df.columns)-1} cols")
    return df


def build_master_dataset() -> pd.DataFrame:
    """
    Main function: load 4 sources and merge into master dataset.

    Steps:
        1. Load all 4 sources
        2. Use yfinance dates as backbone (daily business days)
        3. Left-merge other sources onto backbone
        4. Forward-fill monthly/weekly series
        5. Drop rows where target (eurusd) is missing
    """
    print("=" * 78)
    print("BUILDING MASTER DATASET")
    print("=" * 78)

    print("\nLoading 4 sources...")
    yf = load_yfinance()
    fred = load_fred()
    ecb = load_ecb()
    cot = load_cot()

    print("\nMerging...")

    # yfinance as backbone (daily, no gaps in business days)
    master = yf.copy()
    print(f"  Backbone (yfinance):  {len(master)} rows")

    # Left-merge FRED
    if not fred.empty:
        before = len(master)
        master = master.merge(fred, on="date", how="left")
        print(f"  After FRED merge:     {len(master)} rows "
              f"(+{len(fred.columns)-1} cols)")

    # Left-merge ECB
    if not ecb.empty:
        master = master.merge(ecb, on="date", how="left")
        print(f"  After ECB merge:      {len(master)} rows "
              f"(+{len(ecb.columns)-1} cols)")

    # Left-merge COT
    if not cot.empty:
        master = master.merge(cot, on="date", how="left")
        print(f"  After COT merge:      {len(master)} rows "
              f"(+{len(cot.columns)-1} cols)")

    # ========================================================================
    # FORWARD-FILL non-daily series
    # ========================================================================
    # All series EXCEPT the daily yfinance prices should be ffilled.
    # Reason: monthly CPI valid until next release, weekly COT until next week.
    # ========================================================================
    ffill_cols = [c for c in master.columns
                  if c not in ["date", "eurusd", "dxy", "gold", "oil"]]

    master = master.sort_values("date").reset_index(drop=True)
    master[ffill_cols] = master[ffill_cols].ffill()

    print(f"\nForward-filled {len(ffill_cols)} non-daily columns")

    # ========================================================================
    # FINAL CLEANUP
    # ========================================================================
    # Drop rows where any of the 4 core daily assets is missing
    # (these are the "trading day" indicators; if any is NaN, it's not a valid day)
    n_before = len(master)
    master = master.dropna(subset=["eurusd", "dxy", "gold", "oil"]).reset_index(drop=True)
    n_after = len(master)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing EUR/USD")

    # ========================================================================
    # SAVE
    # ========================================================================
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "fx_master_dataset.csv"
    master.to_csv(out_path, index=False)

    print(f"\n{'='*78}")
    print(f"FINAL SHAPE:  {master.shape}")
    print(f"Date range:   {master['date'].min().date()} -> {master['date'].max().date()}")
    print(f"Columns:      {list(master.columns)}")
    print(f"Saved to:     {out_path}")
    print(f"{'='*78}")

    return master


if __name__ == "__main__":
    master = build_master_dataset()

    # Quick diagnostics
    print(f"\n--- HEAD ---")
    print(master.head(3))

    print(f"\n--- TAIL ---")
    print(master.tail(3))

    print(f"\n--- MISSING VALUES (top 10) ---")
    missing = master.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.head(10))
        print(f"Total missing cells: {missing.sum():,} / "
              f"{master.size:,} ({100*missing.sum()/master.size:.2f}%)")
    else:
        print("No missing values!")