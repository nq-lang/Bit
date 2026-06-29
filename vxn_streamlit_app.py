"""
vxn_streamlit_app.py

Streamlit front-end for downloading ^VXN historical data and exporting
it as a CSV directly from the browser (st.download_button).

Run with:
    streamlit run vxn_streamlit_app.py
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

TICKER: str = "^VXN"
TARGET_TZ: str = "US/Eastern"


def fetch_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch OHLC data, returning None on empty results or API errors."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        return None


def process_data(df: pd.DataFrame, target_tz: str = TARGET_TZ) -> pd.DataFrame:
    """Standardize index to datetime/target timezone and drop NaNs."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(target_tz)
    else:
        df.index = df.index.tz_convert(target_tz)
    return df.dropna(how="any")


def main() -> None:
    st.set_page_config(page_title="VXN Data Downloader", layout="wide")
    st.title("Cboe NASDAQ-100 Volatility Index (^VXN) Downloader")
    st.caption(
        "Yahoo Finance hard limits: 1-minute data ≈ 7 days max, "
        "5-minute data ≈ 60 days max. Daily data goes back to ticker inception."
    )

    presets = {
        "1-minute (max 7d)": ("7d", "1m"),
        "5-minute (max 60d)": ("60d", "5m"),
        "Daily (max history, e.g. 2020-2026)": ("max", "1d"),
    }

    choice = st.selectbox("Select timeframe", list(presets.keys()))
    period, interval = presets[choice]

    if st.button("Fetch data"):
        with st.spinner(f"Fetching {TICKER} ({interval}, {period})..."):
            raw_df = fetch_data(TICKER, period, interval)

        if raw_df is None:
            st.warning("No data returned. Try a different timeframe.")
            return

        clean_df = process_data(raw_df)
        if clean_df.empty:
            st.warning("Data was empty after cleaning (all rows had NaNs).")
            return

        st.success(f"Retrieved {len(clean_df)} rows.")
        st.dataframe(clean_df.tail(50))

        csv_bytes = clean_df.to_csv().encode("utf-8")
        filename = f"VXN_{interval}_{period}.csv"
        st.download_button(
            label=f"Download {filename}",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
