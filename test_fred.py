"""
Test FRED API connection.

Purpose:
- Verify .env file loads correctly
- Verify FRED API key is valid
- Pull a simple series (Fed Funds Rate) to confirm connection

Usage:
    python test_fred.py
"""
import os
import sys
from dotenv import load_dotenv
from fredapi import Fred


def main():
    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')

    if not api_key:
        print("[ERROR] FRED_API_KEY not found in .env file")
        print("        Check:")
        print("        1. Does .env file exist? (Test-Path .env)")
        print("        2. Is format correct? (Get-Content .env)")
        print("        3. Is .env in the same folder as this script?")
        sys.exit(1)

    if len(api_key) != 32:
        print(f"[WARN] Key length is {len(api_key)} chars, FRED keys are usually 32.")

    print(f"[OK] Key loaded: {api_key[:6]}...{api_key[-4:]} (len={len(api_key)})")

    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Fred client: {e}")
        sys.exit(1)

    print("\n[INFO] Pulling 'DFF' (Fed Funds Rate) from 2024-01-01...")
    try:
        data = fred.get_series('DFF', observation_start='2024-01-01')
    except Exception as e:
        print(f"[ERROR] FRED API call failed: {e}")
        print(f"        Possible causes:")
        print(f"        - Invalid or revoked key")
        print(f"        - No internet connection")
        print(f"        - FRED under maintenance")
        sys.exit(1)

    if data is None or len(data) == 0:
        print("[ERROR] FRED returned empty data.")
        sys.exit(1)

    print(f"[OK] Pulled {len(data)} rows")
    print(f"\n--- Latest 5 observations ---")
    print(data.tail())
    print(f"\n[SUCCESS] FRED API is working correctly!")
    print(f"          Latest Fed Funds Rate: {data.iloc[-1]:.2f}% "
          f"(date: {data.index[-1].date()})")


if __name__ == "__main__":
    main()
