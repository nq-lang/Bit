"""
vxn_streamlit_app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit web app for downloading Cboe NASDAQ-100 Volatility Index (^VXN)
intraday data via yfinance, with in-browser CSV download buttons.

Run locally:
    streamlit run vxn_streamlit_app.py
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import streamlit as st
import yfinance as yf


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VXN Intraday Downloader",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
      /* Dark financial terminal palette */
      [data-testid="stAppViewContainer"] {
          background: #0d1117;
          color: #e6edf3;
      }
      [data-testid="stSidebar"] {
          background: #161b22;
          border-right: 1px solid #21262d;
      }
      [data-testid="stSidebar"] * { color: #e6edf3 !important; }

      /* Metric cards */
      [data-testid="metric-container"] {
          background: #161b22;
          border: 1px solid #21262d;
          border-radius: 8px;
          padding: 12px 16px;
      }
      [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.72rem; }
      [data-testid="metric-container"] [data-testid="stMetricValue"] {
          color: #58a6ff !important;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          font-size: 1.35rem;
      }

      /* Section headers */
      .section-header {
          font-size: 0.7rem;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #58a6ff;
          padding: 0 0 6px 0;
          border-bottom: 1px solid #21262d;
          margin-bottom: 12px;
      }

      /* Status pills */
      .pill-success {
          display: inline-block;
          background: #0d3320;
          color: #3fb950;
          border: 1px solid #238636;
          border-radius: 20px;
          padding: 3px 12px;
          font-size: 0.78rem;
          font-weight: 600;
      }
      .pill-error {
          display: inline-block;
          background: #3d0f0f;
          color: #f85149;
          border: 1px solid #da3633;
          border-radius: 20px;
          padding: 3px 12px;
          font-size: 0.78rem;
          font-weight: 600;
      }
      .pill-warn {
          display: inline-block;
          background: #2d2100;
          color: #e3b341;
          border: 1px solid #9e6a03;
          border-radius: 20px;
          padding: 3px 12px;
          font-size: 0.78rem;
          font-weight: 600;
      }

      /* Download button */
      [data-testid="stDownloadButton"] button {
          background: #238636 !important;
          color: #ffffff !important;
          border: 1px solid #2ea043 !important;
          border-radius: 6px !important;
          font-weight: 600;
          width: 100%;
      }
      [data-testid="stDownloadButton"] button:hover {
          background: #2ea043 !important;
      }

      /* Fetch button */
      [data-testid="stButton"] > button[kind="primary"] {
          background: #1f6feb !important;
          border-color: #388bfd !important;
          color: #fff !important;
          font-weight: 700;
          border-radius: 6px;
      }

      /* DataFrame table */
      [data-testid="stDataFrame"] {
          border: 1px solid #21262d;
          border-radius: 6px;
      }

      /* Dividers */
      hr { border-color: #21262d; }

      /* Code/mono text */
      code {
          background: #161b22;
          color: #79c0ff;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 0.85em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────

TICKER: str = "^VXN"
TARGET_TZ: str = "US/Eastern"

FETCH_CONFIGS: list[dict] = [
    {
        "label":    "1-Minute  ·  7-Day Window",
        "interval": "1m",
        "period":   "7d",
        "filename": "VXN_1m_7d.csv",
        "key":      "1m_7d",
        "note":     "Maximum history available for 1m bars (Yahoo Finance limit)",
    },
    {
        "label":    "5-Minute  ·  60-Day Window",
        "interval": "5m",
        "period":   "60d",
        "filename": "VXN_5m_60d.csv",
        "key":      "5m_60d",
        "note":     "Maximum history available for 5m bars (Yahoo Finance limit)",
    },
]


# ── Core data functions ───────────────────────────────────────────────────────

def fetch_intraday(
    ticker: str,
    interval: str,
    period: str,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Download intraday data. Returns (DataFrame, None) on success
    or (None, error_message) on failure.
    """
    try:
        df: pd.DataFrame = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        return None, f"API error: {exc}"

    if df is None or df.empty:
        return None, (
            f"yfinance returned no data for {ticker} "
            f"(interval={interval}, period={period}). "
            "The market may be closed or the symbol unavailable."
        )
    return df, None


def normalise_timezone(df: pd.DataFrame, target_tz: str) -> pd.DataFrame:
    """Convert index to tz-aware DatetimeIndex in target_tz."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    eastern = pytz.timezone(target_tz)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(eastern)
    else:
        df.index = df.index.tz_convert(eastern)

    df.index.name = "Datetime"
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop NaN rows. Returns (cleaned_df, rows_removed)."""
    before = len(df)
    df = df.dropna()
    return df, before - len(df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise DataFrame to UTF-8 CSV bytes for st.download_button."""
    buf = io.BytesIO()
    df.to_csv(buf, encoding="utf-8")
    return buf.getvalue()


def build_stats(df: pd.DataFrame) -> dict:
    """Extract display statistics from a processed DataFrame."""
    stats: dict = {
        "bars":  f"{len(df):,}",
        "start": str(df.index.min().strftime("%Y-%m-%d %H:%M %Z")),
        "end":   str(df.index.max().strftime("%Y-%m-%d %H:%M %Z")),
    }
    if "Close" in df.columns:
        c = df["Close"]
        stats.update({
            "close_min":  f"{c.min():.4f}",
            "close_max":  f"{c.max():.4f}",
            "close_mean": f"{c.mean():.4f}",
            "close_std":  f"{c.std():.4f}",
        })
    return stats


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<div class="section-header">TARGET INSTRUMENT</div>', unsafe_allow_html=True)
    st.code("^VXN  —  Cboe NASDAQ-100 Volatility Index")
    st.caption("Symbol is fixed; VXN is the sole target of this pipeline.")

    st.markdown("---")
    st.markdown('<div class="section-header">DATA PULLS</div>', unsafe_allow_html=True)

    selected_keys: list[str] = []
    for cfg in FETCH_CONFIGS:
        checked = st.checkbox(cfg["label"], value=True, key=f"chk_{cfg['key']}")
        st.caption(cfg["note"])
        if checked:
            selected_keys.append(cfg["key"])

    st.markdown("---")
    st.markdown('<div class="section-header">TIMEZONE</div>', unsafe_allow_html=True)
    st.code("US/Eastern  (NYSE market time)")

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Data sourced from Yahoo Finance via "
        "**yfinance**. Intraday history windows are hard-limited by Yahoo's API.<br><br>"
        "⚠️ VXN is a calculated index — Volume column will be zero or absent; "
        "this is expected behaviour.</small>",
        unsafe_allow_html=True,
    )


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px">
      <span style="font-size:2.4rem">📉</span>
      <div>
        <h1 style="margin:0; font-size:1.75rem; color:#e6edf3; font-weight:700">
          VXN Intraday Downloader
        </h1>
        <p style="margin:0; color:#8b949e; font-size:0.9rem">
          Cboe NASDAQ-100 Volatility Index  ·  Maximum intraday history via yfinance
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Fetch trigger ─────────────────────────────────────────────────────────────

if not selected_keys:
    st.warning("Select at least one data pull in the sidebar to continue.")
    st.stop()

col_btn, col_info = st.columns([1, 3])
with col_btn:
    fetch_clicked = st.button("⬇  Fetch VXN Data", type="primary", use_container_width=True)
with col_info:
    st.markdown(
        "<small style='color:#8b949e; line-height:2.4'>"
        "Pulls the selected intervals from Yahoo Finance, normalises to "
        "US/Eastern, drops NaN rows, then prepares CSV download files.</small>",
        unsafe_allow_html=True,
    )


# ── State management ──────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = {}


# ── Fetch execution ───────────────────────────────────────────────────────────

if fetch_clicked:
    st.session_state["results"] = {}
    overall_bar = st.progress(0, text="Starting fetch pipeline…")
    n = len(selected_keys)

    for i, key in enumerate(selected_keys):
        cfg = next(c for c in FETCH_CONFIGS if c["key"] == key)
        overall_bar.progress(
            int((i / n) * 100),
            text=f"Fetching {cfg['interval']} bars ({cfg['period']})…",
        )

        with st.spinner(f"Downloading {cfg['label']}…"):
            df_raw, err = fetch_intraday(TICKER, cfg["interval"], cfg["period"])

        if err or df_raw is None:
            st.session_state["results"][key] = {"error": err or "Unknown error"}
            continue

        df_tz = normalise_timezone(df_raw, TARGET_TZ)
        df_clean, n_dropped = clean_data(df_tz)

        st.session_state["results"][key] = {
            "df":       df_clean,
            "csv":      to_csv_bytes(df_clean),
            "filename": cfg["filename"],
            "stats":    build_stats(df_clean),
            "dropped":  n_dropped,
            "label":    cfg["label"],
            "interval": cfg["interval"],
            "period":   cfg["period"],
        }

    overall_bar.progress(100, text="All fetches complete.")


# ── Results display ───────────────────────────────────────────────────────────

if st.session_state["results"]:
    st.markdown("---")
    st.markdown('<div class="section-header">RESULTS</div>', unsafe_allow_html=True)

    for key, result in st.session_state["results"].items():
        cfg = next(c for c in FETCH_CONFIGS if c["key"] == key)

        with st.expander(f"**{cfg['label']}**", expanded=True):

            if "error" in result:
                st.markdown(
                    f'<span class="pill-error">✗ FETCH FAILED</span>',
                    unsafe_allow_html=True,
                )
                st.error(result["error"])
                continue

            # ── Status row ────────────────────────────────────────────────
            s_col, _ = st.columns([3, 1])
            with s_col:
                drop_txt = (
                    f"  ·  {result['dropped']} NaN rows removed"
                    if result["dropped"] else "  ·  No NaN rows"
                )
                st.markdown(
                    f'<span class="pill-success">✓ READY</span>'
                    f'<span style="color:#8b949e; font-size:0.8rem; margin-left:10px">'
                    f'auto_adjust=True  ·  tz=US/Eastern{drop_txt}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(" ")

            # ── Metrics ───────────────────────────────────────────────────
            stats = result["stats"]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Bars", stats["bars"])
            m2.metric("First Bar",  stats["start"][:16])
            m3.metric("Last Bar",   stats["end"][:16])
            if "close_min" in stats:
                m4.metric("Close Min",  stats["close_min"])
                m5.metric("Close Max",  stats["close_max"])
                m6.metric("Close Mean", stats["close_mean"])

            st.markdown(" ")

            # ── Data preview ──────────────────────────────────────────────
            st.markdown('<div class="section-header">DATA PREVIEW  (first 50 rows)</div>', unsafe_allow_html=True)
            st.dataframe(
                result["df"].head(50),
                use_container_width=True,
                height=220,
            )

            # ── Download ──────────────────────────────────────────────────
            st.markdown(" ")
            dl_col, info_col = st.columns([1, 2])
            with dl_col:
                st.download_button(
                    label=f"⬇  Download  {result['filename']}",
                    data=result["csv"],
                    file_name=result["filename"],
                    mime="text/csv",
                    key=f"dl_{key}",
                    use_container_width=True,
                )
            with info_col:
                size_kb = len(result["csv"]) / 1024
                st.markdown(
                    f"<small style='color:#8b949e'>"
                    f"📄 <code>{result['filename']}</code>  ·  "
                    f"{size_kb:.1f} KB  ·  "
                    f"{stats['bars']} rows  ·  "
                    f"CSV with DatetimeIndex (US/Eastern)</small>",
                    unsafe_allow_html=True,
                )

    # ── Bulk timestamp ────────────────────────────────────────────────────────
    st.markdown("---")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    successful = sum(1 for r in st.session_state["results"].values() if "df" in r)
    total = len(st.session_state["results"])
    st.markdown(
        f"<small style='color:#8b949e'>Pipeline completed at <code>{ts}</code>  ·  "
        f"{successful}/{total} datasets ready for download.</small>",
        unsafe_allow_html=True,
    )