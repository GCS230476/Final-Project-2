"""
Pull data from ECB Data Portal (European Central Bank).

WHY ECB IN ADDITION TO FRED:

1. ECB is the "primary source" - FRED only mirrors. ECB data is fresher.

2. ECB has data that FRED doesn't:
   - Euribor rates (EU interbank rates)
   - EUR effective exchange rate vs basket
   - ECB official EUR/USD reference rate (published 16:00 CET)

3. Cross-validation: 3 sources of EUR/USD (yfinance, FRED DEXUSEU, ECB)
   -> Comparing them detects data bugs = research-grade quality

Technical:
- ECB uses SDMX 2.1 standard (no API key needed, free)
- URL: https://data-api.ecb.europa.eu/service/data/{FLOW}/{KEY}?format=csvdata

References:
- ECB API docs: https://data.ecb.europa.eu/help/api/data
- SDMX standard: https://sdmx.org/

Usage:
    python -m src.data.load_ecb
"""
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


# ============================================================================
# Path config
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ECB_DIR = PROJECT_ROOT / "data" / "raw" / "ecb"

ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"


# ============================================================================
# 4 ECB series
# ============================================================================
ECB_SERIES = {
    # === OFFICIAL EUR/USD FROM ECB (cross-check) ===
    "eurusd_official": {
        "dataflow": "EXR",
        "key": "D.USD.EUR.SP00.A",
        "description": "EUR/USD official reference rate (daily, 16:00 CET)",
        "frequency": "daily",
        "category": "exchange_rate",
    },

    # === EUR EFFECTIVE EXCHANGE RATE (broad index) ===
    # Measures EUR "strength" vs basket of trading partners
    # Different from EUR/USD which is just 1 pair
    "eur_effective_rate": {
        "dataflow": "EXR",
        "key": "M.E03.EUR.EN00.A",
        "description": "EUR Nominal Effective Exchange Rate (EER-40, monthly)",
        "frequency": "monthly",
        "category": "eur_strength",
    },

    # === EURIBOR 3-MONTH (benchmark EU rate) ===
    # EU interbank rate - important benchmark for EUR
    "euribor_3m": {
        "dataflow": "FM",
        "key": "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
        "description": "Euribor 3-Month (monthly average)",
        "frequency": "monthly",
        "category": "eu_rates",
    },

    # === ESTER (Euro Short-Term Rate) ===
    # EU overnight rate - replaced EONIA from 2019
    "ester_overnight": {
        "dataflow": "EST",
        "key": "B.EU000A2X2A25.WT",
        "description": "ESTER overnight rate (daily)",
        "frequency": "daily",
        "category": "eu_rates",
    },
}


def download_ecb_series(series_name, series_meta, start_date="2010-01-01"):
    """
    Download a single ECB series and save to CSV.

    Returns:
        (success: bool, n_rows: int)
    """
    url = (
        f"{ECB_BASE_URL}/{series_meta['dataflow']}/{series_meta['key']}"
        f"?format=csvdata&startPeriod={start_date}"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # ECB returns CSV format
        df = pd.read_csv(StringIO(response.text))

        if df is None or len(df) == 0:
            print(f"  [SKIP] {series_name:25s} -> empty data")
            return False, 0

        # ECB CSV has many metadata columns, we only keep Date + Value
        # Standard value column is 'OBS_VALUE', date is 'TIME_PERIOD'
        if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
            df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
            df.columns = ["date", series_name]
        else:
            print(f"  [WARN] {series_name:25s} -> unexpected columns: {list(df.columns)[:5]}")
            return False, 0

        # Save
        RAW_ECB_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_ECB_DIR / f"{series_name}.csv"
        df.to_csv(out_path, index=False)

        print(
            f"  [OK]   {series_name:25s} "
            f"-> {len(df):5d} rows  "
            f"({series_meta['frequency']:7s}, {series_meta['category']})"
        )
        return True, len(df)

    except requests.exceptions.HTTPError as e:
        print(f"  [FAIL] {series_name:25s} -> HTTP {e.response.status_code}")
        return False, 0
    except Exception as e:
        print(f"  [FAIL] {series_name:25s} -> {type(e).__name__}: {e}")
        return False, 0


def download_all(start_date="2010-01-01"):
    """Download all ECB series."""
    print(f"=" * 78)
    print(f"PULLING ECB DATA (start={start_date})")
    print(f"Output dir: {RAW_ECB_DIR}")
    print(f"=" * 78)

    success = 0
    total_rows = 0

    print(f"\nDownloading {len(ECB_SERIES)} series...")

    for series_name, meta in ECB_SERIES.items():
        ok, n = download_ecb_series(series_name, meta, start_date)
        if ok:
            success += 1
            total_rows += n
        time.sleep(0.5)  # ECB has no public rate limit, but be polite

    print(f"\n{'='*78}")
    print(f"DONE: {success}/{len(ECB_SERIES)} series downloaded")
    print(f"Total rows: {total_rows:,}")
    print(f"Saved to: {RAW_ECB_DIR}")
    print(f"{'='*78}")

    return success, total_rows


if __name__ == "__main__":
    download_all()
