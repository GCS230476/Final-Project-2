"""
Pull macroeconomic data from FRED (Federal Reserve Economic Data).

WHY THESE 13 SERIES:

1. Interest Rate Parity (IRP) - primary driver of FX rates:
   - DFF, DGS2, DGS10:        US interest rates (short to long term)
   - ECBDFR, IRLTLT01EZM156N: EU interest rates (counterpart)
   -> US-EU rate differential drives capital flow

2. Purchasing Power Parity (PPP) - long-term driver:
   - CPIAUCSL:           US CPI (US inflation)
   - CP0000EZ19M086NEST: Eurozone HICP (EU inflation)
   - T10YIE:             US breakeven inflation expectations
   -> Higher inflation -> currency depreciation

3. Macro health:
   - UNRATE:  US Unemployment Rate
   - PAYEMS:  Nonfarm Payrolls (high-impact event)

4. Risk sentiment:
   - VIXCLS:   Fear index, flight-to-USD signal
   - DTWEXBGS: Broad USD strength

5. Sanity check:
   - DEXUSEU: Official USD/EUR rate from Fed
   -> Cross-check with yfinance to detect data bugs

References:
- Theory: Krugman & Obstfeld, "International Economics" Ch.14
- FRED docs: https://fred.stlouisfed.org/docs/api/fred/

Usage:
    python -m src.data.load_fred
"""
import os
import time
from pathlib import Path

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv


# ============================================================================
# Path config
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FRED_DIR = PROJECT_ROOT / "data" / "raw" / "fred"


# ============================================================================
# 13 FRED series with economic justification
# ============================================================================
FRED_SERIES = {
    # === US INTEREST RATES ===
    "DFF": {
        "description": "Effective Fed Funds Rate",
        "frequency": "daily",
        "category": "us_rates",
    },
    "DGS2": {
        "description": "US 2-Year Treasury Yield",
        "frequency": "daily",
        "category": "us_rates",
    },
    "DGS10": {
        "description": "US 10-Year Treasury Yield",
        "frequency": "daily",
        "category": "us_rates",
    },

    # === EUROZONE INTEREST RATES ===
    "ECBDFR": {
        "description": "ECB Deposit Facility Rate",
        "frequency": "daily",
        "category": "eu_rates",
    },
    "IRLTLT01EZM156N": {
        "description": "Eurozone 10-Year Government Bond Yield",
        "frequency": "monthly",
        "category": "eu_rates",
    },

    # === INFLATION (PPP theory) ===
    "CPIAUCSL": {
        "description": "US CPI All Items",
        "frequency": "monthly",
        "category": "inflation",
    },
    "CP0000EZ19M086NEST": {
        "description": "Eurozone HICP",
        "frequency": "monthly",
        "category": "inflation",
    },
    "T10YIE": {
        "description": "US 10-Year Breakeven Inflation",
        "frequency": "daily",
        "category": "inflation",
    },

    # === LABOR MARKET ===
    "UNRATE": {
        "description": "US Unemployment Rate",
        "frequency": "monthly",
        "category": "macro_health",
    },
    "PAYEMS": {
        "description": "US Nonfarm Payrolls",
        "frequency": "monthly",
        "category": "macro_health",
    },

    # === RISK SENTIMENT ===
    "VIXCLS": {
        "description": "VIX Volatility Index",
        "frequency": "daily",
        "category": "risk_sentiment",
    },

    # === USD STRENGTH ===
    "DTWEXBGS": {
        "description": "USD Trade-Weighted Index (Broad)",
        "frequency": "daily",
        "category": "usd_strength",
    },

    # === OFFICIAL EUR/USD FROM FED (cross-check) ===
    "DEXUSEU": {
        "description": "USD/EUR Exchange Rate (official)",
        "frequency": "daily",
        "category": "exchange_rate",
    },
}


def get_fred_client():
    """Initialize Fred client from API key in .env"""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Check your .env file."
        )
    if len(api_key) != 32:
        raise RuntimeError(
            f"FRED_API_KEY has unusual length ({len(api_key)} chars, "
            f"expected 32)."
        )
    return Fred(api_key=api_key)


def download_series(fred, series_id, meta, start_date="2010-01-01"):
    """
    Download a single FRED series and save to CSV.

    Returns:
        (success: bool, n_rows: int)
    """
    try:
        data = fred.get_series(series_id, observation_start=start_date)
        if data is None or len(data) == 0:
            print(f"  [SKIP] {series_id:25s} -> empty data")
            return False, 0

        df = data.reset_index()
        df.columns = ["date", series_id]

        RAW_FRED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_FRED_DIR / f"{series_id}.csv"
        df.to_csv(out_path, index=False)

        print(
            f"  [OK]   {series_id:25s} "
            f"-> {len(df):5d} rows  "
            f"({meta['frequency']:7s}, {meta['category']})"
        )
        return True, len(df)

    except Exception as e:
        print(f"  [FAIL] {series_id:25s} -> {type(e).__name__}: {e}")
        return False, 0


def download_all(start_date="2010-01-01"):
    """Download all 13 series from FRED."""
    print(f"=" * 78)
    print(f"PULLING FRED DATA (start={start_date})")
    print(f"Output dir: {RAW_FRED_DIR}")
    print(f"=" * 78)

    fred = get_fred_client()
    print(f"[OK] FRED client ready")

    success = 0
    total_rows = 0

    print(f"\nDownloading {len(FRED_SERIES)} series...")

    for series_id, meta in FRED_SERIES.items():
        ok, n = download_series(fred, series_id, meta, start_date)
        if ok:
            success += 1
            total_rows += n
        time.sleep(0.3)  # Avoid rate limit (FRED limit 120 req/min)

    print(f"\n{'='*78}")
    print(f"DONE: {success}/{len(FRED_SERIES)} series downloaded")
    print(f"Total rows: {total_rows:,}")
    print(f"Saved to: {RAW_FRED_DIR}")
    print(f"{'='*78}")

    return success, total_rows


if __name__ == "__main__":
    download_all()
