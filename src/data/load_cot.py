"""
Pull COT (Commitment of Traders) data from CFTC.

WHY COT:

1. COT reflects "smart money" positioning:
   - Non-Commercial = hedge funds, large speculators
   - Their net position is a leading indicator

2. Extreme positioning = contrarian signal:
   - Extreme Net Long -> bullish exhausted -> potential reversal down
   - Extreme Net Short -> bearish exhausted -> potential reversal up

3. Different from technical indicators:
   - Technical analysis only looks at price
   - COT looks at "who is trading" -> this is behavioral data
   - Adds a completely new dimension to the model

Technical:
- Source: CFTC weekly reports (every Friday)
- Library: cot_reports (pip)
- Data from 2010 to present
- Frequency: weekly (1 row per week)

References:
- Sanders et al. (2004) - "COT and futures trends"
- CFTC official: https://www.cftc.gov/MarketReports/CommitmentsofTraders

Usage:
    python -m src.data.load_cot
"""
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import cot_reports as cot

warnings.filterwarnings("ignore")


# ============================================================================
# Path config
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_COT_DIR = PROJECT_ROOT / "data" / "raw" / "cot"


def _find_column(df, *keywords):
    """
    Find a column in dataframe containing all keywords.
    CFTC renames columns each year, so we need dynamic search.
    """
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    raise KeyError(f"Column containing {keywords} not found")


def download_cot_data(start_year=2010):
    """
    Download COT data for EUR futures from CFTC.

    Returns:
        DataFrame with columns:
            - date
            - eur_noncomm_long
            - eur_noncomm_short
            - eur_net_position (= long - short)
            - eur_net_position_pct (normalized -1 to +1)
    """
    print("=" * 78)
    print(f"PULLING COT DATA (start_year={start_year})")
    print(f"Output dir: {RAW_COT_DIR}")
    print("=" * 78)

    current_year = datetime.now().year
    all_data = []

    print(f"\nDownloading COT yearly reports {start_year}-{current_year}...")

    success_years = 0
    for year in range(start_year, current_year + 1):
        try:
            print(f"  Fetching {year}...", end=" ", flush=True)
            df = cot.cot_year(
                year=year,
                cot_report_type="legacy_fut",
                store_txt=False,
                verbose=False,
            )
            all_data.append(df)
            success_years += 1
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"SKIP ({type(e).__name__})")

    if not all_data:
        raise RuntimeError(
            "Failed to pull COT data. Possible causes:\n"
            "  - Network blocking cftc.gov\n"
            "  - CFTC under maintenance\n"
            "  - cot_reports library out-of-date"
        )

    print(f"\n[OK] Got {success_years}/{current_year - start_year + 1} years")

    # Concat all years
    df = pd.concat(all_data, ignore_index=True)
    print(f"[OK] Total raw rows (all markets): {len(df):,}")

    # ========================================================================
    # Find columns dynamically (CFTC renames each year)
    # ========================================================================
    market_col = _find_column(df, "Market", "Name")

    # CFTC uses different date column formats over the years
    try:
        date_col = _find_column(df, "Report", "Date")
    except KeyError:
        try:
            date_col = _find_column(df, "As", "Date")
        except KeyError:
            # Fallback: first column containing "date"
            date_col = _find_column(df, "date")

    long_col = _find_column(df, "Noncommercial", "Long", "All")
    short_col = _find_column(df, "Noncommercial", "Short", "All")

    print(f"[OK] Found columns:")
    print(f"     market_col = '{market_col}'")
    print(f"     date_col   = '{date_col}'")
    print(f"     long_col   = '{long_col}'")
    print(f"     short_col  = '{short_col}'")

    # ========================================================================
    # Filter EUR futures (Euro FX on CME)
    # ========================================================================
    eur = df[df[market_col].str.contains("EURO FX", case=False, na=False)].copy()

    if len(eur) == 0:
        raise RuntimeError("'EURO FX' not found in COT data.")

    print(f"[OK] EUR FX rows: {len(eur):,}")

    # Standardize columns
    eur = eur[[date_col, long_col, short_col]].copy()
    eur.columns = ["date", "eur_noncomm_long", "eur_noncomm_short"]

    # Convert to numeric
    eur["eur_noncomm_long"] = pd.to_numeric(eur["eur_noncomm_long"], errors="coerce")
    eur["eur_noncomm_short"] = pd.to_numeric(eur["eur_noncomm_short"], errors="coerce")

    # ========================================================================
    # COT FEATURE ENGINEERING
    # ========================================================================
    # Net positioning (raw): more positive = more bullish EUR
    eur["eur_net_position"] = eur["eur_noncomm_long"] - eur["eur_noncomm_short"]

    # Net position % (normalized -1 to +1)
    # Avoids dependency on total volume (volume grows over years)
    total_oi = eur["eur_noncomm_long"] + eur["eur_noncomm_short"]
    eur["eur_net_position_pct"] = eur["eur_net_position"] / total_oi.replace(0, pd.NA)

    # CFTC stores date as YYMMDD (e.g., 240430 = April 30, 2024)
    # Must parse with correct format
    eur["date"] = pd.to_datetime(
        eur["date"].astype(str).str.zfill(6),  # "240430" -> ensure 6 chars
        format="%y%m%d",
        errors="coerce",
    )
    eur = eur.sort_values("date").dropna().reset_index(drop=True)

    # ========================================================================
    # SAVE
    # ========================================================================
    RAW_COT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_COT_DIR / "eur_cot.csv"
    eur.to_csv(out_path, index=False)

    print(f"\n{'='*78}")
    print(f"DONE: {len(eur)} weeks of EUR COT data")
    print(f"Date range: {eur['date'].min().date()} -> {eur['date'].max().date()}")
    print(f"Saved to: {out_path}")
    print(f"{'='*78}")

    # Quick stats
    print(f"\nQuick stats (Net Position %):")
    print(f"  mean:  {eur['eur_net_position_pct'].mean():+.3f}")
    print(f"  std:   {eur['eur_net_position_pct'].std():.3f}")
    print(f"  min:   {eur['eur_net_position_pct'].min():+.3f}")
    print(f"  max:   {eur['eur_net_position_pct'].max():+.3f}")

    return eur


if __name__ == "__main__":
    download_cot_data()
