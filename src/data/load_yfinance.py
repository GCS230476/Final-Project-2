"""
Pull price data from Yahoo Finance.

Pulls 4 assets:
- EUR/USD (target)
- DXY (US Dollar Index, inversely correlated)
- Gold (safe haven indicator)
- Oil (commodity benchmark)

Note on yfinance CSV format:
- Newer yfinance versions save CSV with 3 header rows (Price/Ticker/Date)
- This module saves data with a CLEAN single-row header for downstream use

Usage:
    python -m src.data.load_yfinance
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================================================
# Path config
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_YF_DIR = PROJECT_ROOT / "data" / "raw" / "yfinance"


# ============================================================================
# 4 assets to pull
# ============================================================================
YF_ASSETS = {
    "eurusd": {
        "ticker": "EURUSD=X",
        "description": "EUR/USD spot rate",
        "category": "target",
    },
    "dxy": {
        "ticker": "DX-Y.NYB",
        "description": "US Dollar Index",
        "category": "usd_strength",
    },
    "gold": {
        "ticker": "GC=F",
        "description": "Gold futures",
        "category": "safe_haven",
    },
    "oil": {
        "ticker": "CL=F",
        "description": "Crude Oil WTI futures",
        "category": "commodity",
    },
}


def download_asset(name, meta, start_date="2010-01-01"):
    """
    Download a single asset from yfinance and save clean CSV.

    Returns:
        (success: bool, n_rows: int)
    """
    try:
        # auto_adjust=True returns adjusted prices (handles splits/dividends)
        data = yf.download(
            meta["ticker"],
            start=start_date,
            progress=False,
            auto_adjust=True,
        )

        if data is None or len(data) == 0:
            print(f"  [SKIP] {name:10s} -> empty data from yfinance")
            return False, 0

        # yfinance returns MultiIndex columns: (price_type, ticker)
        # Flatten to single-level: just keep price_type
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Reset index to make Date a column
        data = data.reset_index()

        # Keep only essential columns
        keep_cols = ["Date", "Close"]
        data = data[keep_cols]

        # Save
        RAW_YF_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_YF_DIR / f"{name}.csv"
        data.to_csv(out_path, index=False)

        print(
            f"  [OK]   {name:10s} ({meta['ticker']:12s}) "
            f"-> {len(data):5d} rows  ({meta['category']})"
        )
        return True, len(data)

    except Exception as e:
        print(f"  [FAIL] {name:10s} -> {type(e).__name__}: {e}")
        return False, 0


def download_all(start_date="2010-01-01"):
    """Download all 4 yfinance assets."""
    print(f"=" * 78)
    print(f"PULLING YFINANCE DATA (start={start_date})")
    print(f"Output dir: {RAW_YF_DIR}")
    print(f"=" * 78)

    success = 0
    total_rows = 0

    print(f"\nDownloading {len(YF_ASSETS)} assets...")

    for name, meta in YF_ASSETS.items():
        ok, n = download_asset(name, meta, start_date)
        if ok:
            success += 1
            total_rows += n
        time.sleep(1.0)  # Avoid yfinance rate limit

    print(f"\n{'='*78}")
    print(f"DONE: {success}/{len(YF_ASSETS)} assets downloaded")
    print(f"Total rows: {total_rows:,}")
    print(f"Saved to: {RAW_YF_DIR}")
    print(f"{'='*78}")

    return success, total_rows


if __name__ == "__main__":
    download_all()