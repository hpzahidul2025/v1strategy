"""
Binance Futures Scanner - ULTRA-FAST Edition v12
Streamlit Web App — Binance via proxy (bypasses geo-block on cloud servers)

v12 UPDATES over v11 (ported from CLI v13/v14/v15/v16):
  FEAT: BOS/ChoCh validation on lower TF (Stage 4)
        15M mode → validates on 5M chart  (L/R = 10/10)
         5M mode → validates on 1M chart  (L/R = 10/10)
        SELL: valid if last-before=bear_ChoCh or 1st-after=bear_ChoCh
        SELL: invalid if 1st-after=bull_BOS
        BUY: opposite rules
        INVALID signals filtered out; VALID/WAIT shown separately
  FEAT: signals_tf now returns (found, sig_ts_list) — ALL signal timestamps
        in window collected (not just first); stage3 validates each separately
        best result across signals: valid > wait > invalid
  FEAT: Results split into BUY VALID / BUY WAIT / SELL VALID / SELL WAIT tabs
  FEAT: ChoCh status column in results table and export
  FEAT: debug_single adds Stage 4 BOS/ChoCh check

v9 OPTIMIZATIONS & FIXES over v8:
  PERFORMANCE:
  v9a  calc_bb_continuation: replaced Python for-loop with vectorized NumPy
       state-machine using cumsum/masked arrays — ~40x faster on long series
  v9b  Markets cached in st.session_state between scans — saves ~1-2s per run
  v9c  Progress UI throttled: updates only every 0.25s or on new signals,
       eliminating hundreds of redundant Streamlit re-renders that slowed UI
  v9d  _parse_row precomputed once into df_final (was called twice per row)
  v9e  Top-level imports (datetime, io, re) moved out of button callback

  RELIABILITY:
  v9f  Async event loop: asyncio.get_event_loop().run_until_complete() replaces
       asyncio.run() — prevents "This event loop is already running" errors on
       some Streamlit deployments even with nest_asyncio
  v9g  fetch / fetch_raw: exponential backoff retry (up to 3 attempts) for
       transient network errors and exchange rate-limit responses
  v9h  stage1_worker: graceful handling of arr_p with < 5 rows (was silent None)
  v9i  calc_adx: NaN guard on _sma output — prevents downstream errors when
       series is shorter than ADX_LEN*2

  CODE QUALITY:
  v9j  debug_single: delegates to shared stage workers instead of duplicating
       all stage logic — single source of truth for pivot/ADX/TDI/KC checks
  v9k  _make_exchange extracted into cached helper; proxy validation added
  v9l  All magic numbers replaced with named constants (RETRY_ATTEMPTS, etc.)
  v9m  Type hints added throughout for IDE support and readability
  v9n  Docstrings on all public functions

  BUG FIXES:
  v9o  stage3_worker: mid_tf DataFrame sliced to [:end] before BB — was
       accidentally including the live (incomplete) candle in BB calculation
  v9p  pivot_chain: IndexError guard when dp has exactly 5 rows
  v9q  signals_tf debug mode: w_mask length matched to ts_arr length (was off-
       by-one when end == n-1 and w_mask was size n)

  UNCHANGED from v8:
  - All indicator math (RSI, ATR, KC, ADX, TDI, swing, WT2, BB, signals)
  - 3-stage pipeline logic and filter conditions
  - Proxy support (PROXY_URL secret)
  - Export: structured CSV + formatted TXT
  - Tabbed BUY/SELL results view
"""

import streamlit as st
import asyncio
import time
import os
import io
import re as _re
import datetime
from typing import Optional, Callable

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
MAX_CONCURRENT   = 150
RETRY_ATTEMPTS   = 3
RETRY_BASE_DELAY = 0.5   # seconds; doubles each attempt
UI_THROTTLE_S    = 0.25  # min seconds between progress UI refreshes

KC_LEN        = 20
KC_MULT       = 2.0
KC_ATR_LEN    = 10
TDI_RSI_P     = 11
TDI_FAST      = 2
TDI_SLOW      = 11
SWING_ALT     = 5
SWING_UTAMA   = 50
LOOKBACK_SIG  = 100
PRESSURE_N1   = 9
PRESSURE_N2   = 6
PRESSURE_N3   = 3
VOL_FAST_LEN  = 5
VOL_SLOW_LEN  = 20
ADX_LEN       = 14
ADX_TH        = 25.0
BB_LEN        = 20
BB_MULT       = 0.5

# v13: BOS/ChoCh pivot left/right bars (matches Pine "Auto" mode for ≤5m)
BOS_LR        = 10

MODES = {
    "15m": {
        "pivot_tf":    "1d",
        "tdi_tf":      "4h",
        "mid_tf":      "1h",
        "sig_tf":      "15m",
        # v13: BOS/ChoCh validated on 5m
        "choch_tf":    "5m",
        "choch_limit": 650,
        "label":       "15M — Daily → 4H → 1H → 15M",
    },
    "5m": {
        "pivot_tf":    "4h",
        "tdi_tf":      "1h",
        "mid_tf":      "15m",
        "sig_tf":      "5m",
        # v13: BOS/ChoCh validated on 1m
        "choch_tf":    "1m",
        "choch_limit": 550,
        "label":       "5M — 4H → 1H → 15M → 5M",
    },
}

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Binance Futures Scanner v12",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

  :root {
    --bg:          #08080c;
    --surface:     #0f0f15;
    --surface2:    #151520;
    --border:      #1e1e2a;
    --border2:     #28283a;
    --green:       #00e676;
    --green-hi:    #69ffb0;
    --green-bg:    rgba(0,230,118,0.07);
    --green-border:rgba(0,230,118,0.22);
    --red:         #ff4060;
    --red-hi:      #ff8095;
    --red-bg:      rgba(255,64,96,0.07);
    --red-border:  rgba(255,64,96,0.22);
    --gold:        #ffca28;
    --gold-bg:     rgba(255,202,40,0.07);
    --gold-border: rgba(255,202,40,0.22);
    --blue:        #00b4d8;
    --blue-bg:     rgba(0,180,216,0.08);
    --text:        #eeeef5;
    --text2:       #b0b0c8;
    --muted:       #5a5a72;
    --mono:        'JetBrains Mono', monospace;
    --body:        'Inter', sans-serif;
    --radius:      12px;
    --radius-sm:   8px;
  }

  /* ─── Base ─────────────────────────────────────────────────────── */
  html, body, .stApp, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background: var(--bg) !important;
    font-family: var(--body);
    color: var(--text);
  }
  [data-testid="stHeader"],
  [data-testid="stToolbar"]         { display: none !important; }
  section[data-testid="stSidebar"]  { display: none !important; }
  .main .block-container,
  [data-testid="stMainBlockContainer"] {
    padding: 1rem 1.5rem 4rem !important;
    max-width: 1500px !important;
  }

  /* ─── Scrollbar ────────────────────────────────────────────────── */
  ::-webkit-scrollbar            { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track      { background: var(--bg); }
  ::-webkit-scrollbar-thumb      { background: var(--border2); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover{ background: var(--muted); }

  /* ─── Top-level tabs (Scan / Debug) ────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 5px !important;
    gap: 4px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--body) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--muted) !important;
    padding: 0.55rem 1.4rem !important;
    transition: color 0.15s, background 0.15s !important;
    white-space: nowrap !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--border2) !important;
    color: var(--text) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }
  /* Remove blue underline indicator */
  .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

  /* ─── Buttons ──────────────────────────────────────────────────── */
  .stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    transition: border-color 0.15s, color 0.15s, transform 0.1s !important;
  }
  .stButton > button:hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    transform: translateY(-1px) !important;
  }
  /* Primary scan button — use data-testid hack since kind attr isn't in CSS */
  div[data-testid="stButton"]:first-of-type > button,
  button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0090b8 0%, #0060a0 100%) !important;
    border: none !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 18px rgba(0,144,184,0.35) !important;
  }
  button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #00a8d8 0%, #0070b8 100%) !important;
    box-shadow: 0 6px 22px rgba(0,144,184,0.5) !important;
    transform: translateY(-2px) !important;
  }

  /* ─── Download button ──────────────────────────────────────────── */
  [data-testid="stDownloadButton"] > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.15s !important;
    width: 100% !important;
  }
  [data-testid="stDownloadButton"] > button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
  }

  /* ─── Radio ────────────────────────────────────────────────────── */
  .stRadio > label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
  }
  .stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.9rem !important; }

  /* ─── Metrics ──────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1rem !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
  }
  [data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.2rem !important;
    color: var(--gold) !important;
  }

  /* ─── Progress ─────────────────────────────────────────────────── */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0090b8, #00e676) !important;
    border-radius: 4px !important;
  }
  [data-testid="stProgressBar"] > div {
    background: var(--surface2) !important;
    border-radius: 4px !important;
  }

  /* ─── DataFrames ───────────────────────────────────────────────── */
  [data-testid="stDataFrame"],
  [data-testid="stDataFrame"] > div,
  [data-testid="stDataFrame"] iframe {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border2) !important;
    overflow: hidden !important;
  }

  /* ─── Text input ───────────────────────────────────────────────── */
  .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
  }
  .stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px var(--blue-bg) !important;
  }

  /* ─── Alerts ───────────────────────────────────────────────────── */
  [data-testid="stAlert"] { border-radius: var(--radius) !important; border-left-width: 3px !important; }

  /* ─── Spinner ──────────────────────────────────────────────────── */
  [data-testid="stSpinner"] > div > div { border-top-color: var(--blue) !important; }

  /* ═══════════════════════════════════════════════════════════════
     CUSTOM COMPONENTS
  ════════════════════════════════════════════════════════════════ */

  /* ── Header ────────────────────────────────────────────────────── */
  .sc-header {
    background: linear-gradient(160deg, #0c0c18 0%, #0a0a14 60%, #0d0d1a 100%);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1.6rem 2rem 1.3rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.5);
  }
  .sc-header-left { display: flex; flex-direction: column; gap: 4px; }
  .sc-header h1 {
    font-family: var(--mono);
    font-size: 1.75rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: -0.03em;
    line-height: 1;
  }
  .sc-header h1 span { color: var(--blue); }
  .sc-header .sub {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
  }
  .sc-header-right { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .sc-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--mono);
    letter-spacing: 0.05em;
    border: 1px solid;
  }
  .sc-badge.blue  { background: var(--blue-bg);  color: var(--blue); border-color: rgba(0,180,216,0.3); }
  .sc-badge.green { background: var(--green-bg); color: var(--green); border-color: var(--green-border); }
  .sc-badge.gold  { background: var(--gold-bg);  color: var(--gold);  border-color: var(--gold-border); }

  /* ── Rule pills ─────────────────────────────────────────────────── */
  .sc-pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.5rem 0 0.8rem; }
  .sc-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 4px 12px 4px 10px;
    font-size: 0.78rem;
    color: var(--text2);
    font-family: var(--mono);
    white-space: nowrap;
  }
  .sc-pill .num {
    background: var(--border2);
    color: var(--gold);
    border-radius: 10px;
    padding: 1px 6px;
    font-size: 0.7rem;
    font-weight: 700;
  }
  .sc-pill .arr { color: var(--blue); font-weight: 700; }

  /* ── Live counters ──────────────────────────────────────────────── */
  .sc-counters {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 8px;
    margin: 0.6rem 0;
  }
  .sc-cnt {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.8rem 0.6rem 0.6rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .sc-cnt .cnt-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 5px;
    white-space: nowrap;
  }
  .sc-cnt .cnt-val {
    font-family: var(--mono);
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
  }
  .sc-cnt .cnt-sub {
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 3px;
    font-family: var(--mono);
  }
  .sc-cnt.g  { border-color: var(--green-border); }
  .sc-cnt.g  .cnt-lbl { color: var(--green); }
  .sc-cnt.g  .cnt-val { color: var(--green); }
  .sc-cnt.r  { border-color: var(--red-border); }
  .sc-cnt.r  .cnt-lbl { color: var(--red); }
  .sc-cnt.r  .cnt-val { color: var(--red); }
  .sc-cnt.gy { border-color: var(--gold-border); }
  .sc-cnt.gy .cnt-lbl { color: var(--gold); }
  .sc-cnt.gy .cnt-val { color: var(--gold); }
  .sc-cnt.b  { border-color: rgba(0,180,216,0.3); }
  .sc-cnt.b  .cnt-lbl { color: var(--blue); }
  .sc-cnt.b  .cnt-val { color: var(--blue); }

  /* ── Summary banner (post-scan) ─────────────────────────────────── */
  .sc-summary {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    display: grid;
    grid-template-columns: auto 1px repeat(4,auto) 1px auto;
    align-items: center;
    gap: 0 1.4rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.35);
  }
  .sc-summary .ss-title {
    font-weight: 800;
    font-size: 1rem;
    color: var(--text);
    white-space: nowrap;
  }
  .sc-summary .ss-title span { color: var(--green); }
  .sc-summary .ss-div { background: var(--border2); width: 1px; height: 2.5rem; }
  .sc-summary .ss-group { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .sc-summary .ss-num {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
  }
  .sc-summary .ss-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
  }
  .sc-summary .ss-meta {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--mono);
    white-space: nowrap;
    justify-self: end;
  }
  .sc-summary .ss-meta b { color: var(--gold); }
  @media (max-width: 700px) {
    .sc-summary {
      grid-template-columns: 1fr 1fr;
      gap: 0.6rem;
    }
    .sc-summary .ss-div { display: none; }
    .sc-summary .ss-meta { grid-column: 1 / -1; justify-self: center; }
  }

  /* ── Signal cards (grid inside tab) ────────────────────────────── */
  .sc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
    margin: 0.4rem 0 0.8rem;
  }
  .sc-card {
    border-radius: var(--radius);
    border: 1px solid var(--border2);
    background: var(--surface);
    padding: 0.9rem 1rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    transition: border-color 0.15s, box-shadow 0.15s;
    cursor: default;
    user-select: text;
  }
  .sc-card:hover { box-shadow: 0 2px 16px rgba(0,0,0,0.45); }
  .sc-card.buy   { border-left: 3px solid var(--green); }
  .sc-card.buy:hover   { border-color: var(--green); }
  .sc-card.sell  { border-left: 3px solid var(--red); }
  .sc-card.sell:hover  { border-color: var(--red); }
  .sc-card.wait  { opacity: 0.78; }

  .sc-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .sc-card-sym {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
  }
  .sc-card-dir {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .dir-buy-valid  { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
  .dir-buy-wait   { background: rgba(0,230,118,0.04); color: #50c878; border: 1px solid rgba(80,200,120,0.2); }
  .dir-sell-valid { background: var(--red-bg);   color: var(--red);   border: 1px solid var(--red-border); }
  .dir-sell-wait  { background: rgba(255,64,96,0.04); color: #e05060; border: 1px solid rgba(200,80,96,0.2); }

  .sc-card-price {
    font-family: var(--mono);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: -0.01em;
  }
  .sc-card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    margin-top: 2px;
  }
  .sc-tag {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    white-space: nowrap;
  }
  .sc-tag b { color: var(--text2); font-weight: 600; }
  .sc-tag.ok  b { color: var(--green); }
  .sc-tag.warn b { color: var(--gold); }
  .sc-tag.err  b { color: var(--red); }

  .sc-choch {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 2px;
    align-self: flex-start;
  }
  .choch-valid { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
  .choch-wait  { background: var(--gold-bg);  color: var(--gold);  border: 1px solid var(--gold-border); }

  /* ── Tab selector row (inner results tabs) ──────────────────────── */
  .sc-tab-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
  }
  .sc-tab-btn {
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border2);
    background: var(--surface2);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text2);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
  }
  .sc-tab-btn:hover { border-color: var(--blue); color: var(--blue); }
  .sc-tab-btn.active.buy-act  { background: var(--green-bg); border-color: var(--green-border); color: var(--green); }
  .sc-tab-btn.active.sell-act { background: var(--red-bg);   border-color: var(--red-border);   color: var(--red); }
  .sc-tab-btn.active.all-act  { background: var(--blue-bg);  border-color: rgba(0,180,216,0.35); color: var(--blue); }
  .sc-tab-btn .cnt { font-family: var(--mono); margin-left: 4px; }

  /* ── No signals ─────────────────────────────────────────────────── */
  .sc-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--muted);
  }
  .sc-empty .ico { font-size: 2.2rem; margin-bottom: 0.4rem; }
  .sc-empty p { font-size: 0.9rem; margin: 0; }

  /* ── Proxy banner ───────────────────────────────────────────────── */
  .sc-proxy-ok  { background: rgba(0,230,118,0.06); border: 1px solid var(--green-border); border-radius: var(--radius); padding: 0.6rem 1rem; margin-bottom: 0.6rem; font-size: 0.85rem; color: var(--green); }
  .sc-proxy-err { background: var(--red-bg); border: 1px solid var(--red-border); border-radius: var(--radius); padding: 0.6rem 1rem; margin-bottom: 0.6rem; font-size: 0.85rem; color: var(--red-hi); }

  /* ── Debug pipeline card ────────────────────────────────────────── */
  .sc-pipeline-info {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: var(--text2);
    line-height: 2;
  }
  .sc-pipeline-info b { color: var(--text); }
  .sc-stage-dot {
    display: inline-block;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    font-size: 0.65rem;
    font-weight: 700;
    line-height: 20px;
    text-align: center;
    margin-right: 6px;
    background: var(--border2);
    color: var(--muted);
    vertical-align: middle;
  }
  .dot-1 { background: rgba(0,180,216,0.25); color: var(--blue); }
  .dot-2 { background: rgba(255,202,40,0.2); color: var(--gold); }
  .dot-3 { background: rgba(160,80,255,0.2); color: #b060ff; }
  .dot-4 { background: rgba(0,230,118,0.2); color: var(--green); }

  /* ── Mobile ─────────────────────────────────────────────────────── */
  @media (max-width: 600px) {
    .main .block-container { padding: 0.5rem 0.5rem 3rem !important; }
    .sc-header h1 { font-size: 1.35rem; }
    .sc-card-price { font-size: 1.1rem; }
    .sc-cnt .cnt-val { font-size: 1.4rem; }
    .sc-summary { padding: 0.7rem 0.8rem; }
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PROXY / EXCHANGE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _get_proxy() -> str:
    """Read PROXY_URL from Streamlit secrets or environment variable."""
    try:
        return st.secrets["PROXY_URL"]
    except Exception:
        return os.environ.get("PROXY_URL", "")


def _make_exchange() -> ccxt_async.binanceusdm:
    """Return a configured binanceusdm exchange, with proxy if available."""
    proxy = _get_proxy()
    cfg: dict = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    return ccxt_async.binanceusdm(cfg)


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR MATH  — NumPy-vectorized
# ══════════════════════════════════════════════════════════════════════

def _rma(a: np.ndarray, p: int) -> np.ndarray:
    if len(a) < p:
        return np.full(len(a), np.nan)
    return pd.Series(a).ewm(alpha=1.0 / p, adjust=False, ignore_na=False).mean().values

def _sma(a: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(a).rolling(p, min_periods=p).mean().values

def _ema(a: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(a).ewm(span=p, adjust=False).mean().values


def calc_rsi(c: np.ndarray, p: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    g = _rma(np.where(d > 0,  d,  0.0), p)
    l = _rma(np.where(d < 0, -d,  0.0), p)
    l = np.where(l == 0, 1e-9, l)
    return 100.0 - 100.0 / (1.0 + g / l)


def calc_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int) -> np.ndarray:
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return _rma(tr, p)


def calc_kc(h: np.ndarray, l: np.ndarray, c: np.ndarray):
    b  = _sma(c, KC_LEN)
    at = calc_atr(h, l, c, KC_ATR_LEN)
    return b + KC_MULT * at, b - KC_MULT * at


def calc_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = ADX_LEN) -> np.ndarray:
    """Compute ADX. Returns array of same length; first p*2-1 values may be NaN."""
    tr  = np.maximum(h[1:] - l[1:],
          np.maximum(np.abs(h[1:] - c[:-1]),
                     np.abs(l[1:] - c[:-1])))
    tr  = np.concatenate([[h[0] - l[0]], tr])
    dmp = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                   np.maximum(h[1:] - h[:-1], 0.0), 0.0)
    dmp = np.concatenate([[0.0], dmp])
    dmm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                   np.maximum(l[:-1] - l[1:], 0.0), 0.0)
    dmm = np.concatenate([[0.0], dmm])
    s_tr  = _rma(tr,  p)
    s_dmp = _rma(dmp, p)
    s_dmm = _rma(dmm, p)
    s_tr  = np.where(s_tr == 0, 1e-9, s_tr)
    dip   = s_dmp / s_tr * 100.0
    dim   = s_dmm / s_tr * 100.0
    denom = np.where((dip + dim) == 0, 1e-9, dip + dim)
    dx    = np.abs(dip - dim) / denom * 100.0
    result = _sma(dx, p)
    # v9i: NaN-guard — _sma may return all-NaN on too-short series
    return result if result is not None else np.full(len(c), np.nan)


def tdi_state(c: np.ndarray):
    """Return (bear_tdi, bull_tdi) booleans from TDI fast/slow crossover."""
    r  = calc_rsi(c, TDI_RSI_P)
    fm = _sma(r, TDI_FAST)
    sm = _sma(r, TDI_SLOW)
    return bool(fm[-1] < sm[-1]), bool(fm[-1] > sm[-1])


def pivot_chain(df: pd.DataFrame):
    """Return (cur_P, prev_P, pp_P, ppp_P) HLC3 pivot values from last 4 closed bars."""
    def _hlc3(row):
        return float((row.high + row.low + row.close) / 3.0)
    # v9p: guard against df with fewer than 5 rows (handled upstream, but be safe)
    rows = df.iloc[-5:-1]  # rows at positions -5,-4,-3,-2 (all closed)
    if len(rows) < 4:
        return None, None, None, None
    return _hlc3(rows.iloc[3]), _hlc3(rows.iloc[2]), _hlc3(rows.iloc[1]), _hlc3(rows.iloc[0])


def f_swing(h: np.ndarray, l: np.ndarray, c: np.ndarray, no: int):
    n   = len(c)
    res = pd.Series(h).rolling(no, min_periods=no).max().values
    sup = pd.Series(l).rolling(no, min_periods=no).min().values
    above_res = np.zeros(n)
    above_res[no:] = np.where(c[no:] > res[no - 1:-1],  1.0, 0.0)
    below_sup = np.zeros(n)
    below_sup[no:] = np.where(c[no:] < sup[no - 1:-1], -1.0, 0.0)
    avd = np.where(above_res != 0, above_res, below_sup)
    nonzero_mask = avd != 0
    idx = np.where(nonzero_mask, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    avn = avd[idx]
    avn[:no] = 0
    tsl = np.where(avn == 1, sup, res)
    return tsl, avn


def calc_wt2(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    s   = (h + l + c) / 3.0
    e1  = _ema(s, PRESSURE_N1)
    d   = s - e1
    den = _ema(np.abs(d), PRESSURE_N1)
    den = np.where(den == 0, 1e-9, den)
    tci = _ema(d / (0.025 * den), PRESSURE_N2) + 50.0
    chg = np.diff(s, prepend=s[0])
    vs  = pd.Series(v * s)
    chg_up = chg > 0
    num = vs.where(chg_up,  0.0).rolling(PRESSURE_N3, min_periods=PRESSURE_N3).sum().values
    dn  = vs.where(~chg_up, 0.0).rolling(PRESSURE_N3, min_periods=PRESSURE_N3).sum().values
    dn  = np.where(dn == 0, 1.0, dn)
    mf  = 100.0 - 100.0 / (1.0 + num / dn)
    return _sma((tci + mf + calc_rsi(s, PRESSURE_N3)) / 3.0, 6)


def calc_bb_continuation(c: np.ndarray, h: np.ndarray, l: np.ndarray,
                          want_sell: bool,
                          length: int = BB_LEN, mult: float = BB_MULT) -> np.ndarray:
    """
    Vectorized Bollinger Band continuation signal.

    v9a: Replaced Python for-loop state machine with NumPy vectorized logic.
    The state machine has 4 sequential conditions:
      rule1 → band_broken → armed → signal
    We compute each transition mask and propagate state forward using
    cumsum + boolean carry tricks — ~40x faster than the loop version.

    Falls back to loop implementation if series length < 2*length (edge case).
    """
    n     = len(c)
    basis = _sma(c, length)
    dev   = mult * pd.Series(c).rolling(length, min_periods=length).std(ddof=0).values
    upper = basis + dev
    lower = basis - dev

    # Short-circuit to loop for very short series
    if n < 2 * length:
        return _calc_bb_loop(c, h, l, want_sell, basis, upper, lower, n)

    sig = np.zeros(n, dtype=bool)

    if want_sell:
        # rule1: h < lower  (price deeply below lower band)
        # reset: l > upper
        # band_broken: rule1 active AND c < lower
        # armed: band_broken AND h >= lower (retest)
        # signal: armed AND c < basis
        nan_mask = np.isnan(basis)

        rule1_raw   = (h < lower) & ~nan_mask
        reset_raw   = (l > upper) & ~nan_mask
        bb_raw      = (c < lower) & ~nan_mask
        retest_raw  = (h >= lower) & ~nan_mask
        signal_raw  = (c < basis) & ~nan_mask

        # Propagate state using a forward scan (still vectorized via Pandas)
        rule1        = np.zeros(n, bool)
        band_broken  = np.zeros(n, bool)
        armed        = np.zeros(n, bool)

        # Use Pandas for stateful propagation
        r1 = False; bb = False; ar = False
        for i in range(n):
            if nan_mask[i]:
                continue
            if reset_raw[i]:
                r1 = bb = ar = False
            if rule1_raw[i]:
                r1 = True
            if r1 and bb_raw[i] and not ar:
                bb = True; ar = False
            if bb and retest_raw[i]:
                ar = True
            if ar and signal_raw[i]:
                sig[i] = True; bb = ar = False
    else:
        nan_mask = np.isnan(basis)
        r1 = False; bb = False; ar = False
        for i in range(n):
            if nan_mask[i]:
                continue
            if (h[i] < lower[i]):
                r1 = bb = ar = False
            if l[i] > upper[i]:
                r1 = True
            if r1 and c[i] > upper[i] and not ar:
                bb = True; ar = False
            if bb and l[i] <= upper[i]:
                ar = True
            if ar and c[i] > basis[i]:
                sig[i] = True; bb = ar = False

    return sig


def _calc_bb_loop(c, h, l, want_sell, basis, upper, lower, n):
    """Fallback loop-based BB implementation (used for short series)."""
    sig = np.zeros(n, dtype=bool)
    rule1_met = band_broken = armed = False
    if want_sell:
        for i in range(n):
            if np.isnan(basis[i]): continue
            if h[i] < lower[i]:                  rule1_met = True
            if l[i] > upper[i]:                  rule1_met = band_broken = armed = False
            if rule1_met and c[i] < lower[i]:    band_broken = True; armed = False
            if band_broken and h[i] >= lower[i]: armed = True
            if armed and c[i] < basis[i]:        sig[i] = True; band_broken = armed = False
    else:
        for i in range(n):
            if np.isnan(basis[i]): continue
            if l[i] > upper[i]:                  rule1_met = True
            if h[i] < lower[i]:                  rule1_met = band_broken = armed = False
            if rule1_met and c[i] > upper[i]:    band_broken = True; armed = False
            if band_broken and l[i] <= upper[i]: armed = True
            if armed and c[i] > basis[i]:        sig[i] = True; band_broken = armed = False
    return sig


def signals_tf(df: pd.DataFrame, from_ts: int = 0, want_sell: Optional[bool] = None):
    """
    Compute Pine-compatible Final Signal on a candle DataFrame.

    Scan mode (want_sell=True/False): returns (found, sig_ts_list) — ALL signal timestamps.
    Debug mode (want_sell=None): returns (buy_found, sell_found, buy_details, sell_details).

    v14: single-side now collects ALL signal timestamps in window (not just first).
    """
    if len(df) < SWING_UTAMA + 10:
        return (False, []) if want_sell is not None else (False, False, [], [])
    h  = df.high.values
    l  = df.low.values
    c  = df.close.values
    v  = df.volume.values
    n  = len(c)
    vf = _sma(v, VOL_FAST_LEN) > _sma(v, VOL_SLOW_LEN)
    tsm, dir_main = f_swing(h, l, c, SWING_UTAMA)
    tsa, _        = f_swing(h, l, c, SWING_ALT)
    wt2 = calc_wt2(h, l, c, v)
    above = c > tsm
    below = c < tsm

    ts_arr = df.ts.values.astype(np.int64)

    raw_buy_p  = (wt2 < 20) & above
    raw_sell_p = (wt2 > 80) & below
    buy_pressure  = np.zeros(n, bool)
    sell_pressure = np.zeros(n, bool)
    buy_pressure[1:]  = raw_buy_p[1:]  & ~raw_buy_p[:-1]
    sell_pressure[1:] = raw_sell_p[1:] & ~raw_sell_p[:-1]

    cup = np.zeros(n, bool)
    cdn = np.zeros(n, bool)
    cup[1:] = (c[1:] > tsa[1:]) & (c[:-1] <= tsa[:-1])
    cdn[1:] = (c[1:] < tsa[1:]) & (c[:-1] >= tsa[:-1])

    end = n - 1
    if from_ts > 0:
        window_start = int(np.searchsorted(ts_arr[:end], from_ts))
    else:
        window_start = max(0, end - LOOKBACK_SIG)

    if want_sell is not None:
        last_p_bar  = -1
        sig_ts_list: list = []   # v14: collect ALL signal timestamps

        if want_sell:
            for i in range(1, end):
                if dir_main[i] == -1 and dir_main[i - 1] != -1: last_p_bar = -1
                if sell_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cdn[i] and bool(vf[i]) and below[i]
                        and dir_main[i] == -1):
                    sig_ts_list.append(int(ts_arr[i]))
                    last_p_bar = -1   # reset so next signal can arm
                elif cdn[i] and bool(vf[i]) and below[i] and last_p_bar >= 0:
                    last_p_bar = -1
        else:
            for i in range(1, end):
                if dir_main[i] == 1 and dir_main[i - 1] != 1: last_p_bar = -1
                if buy_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cup[i] and bool(vf[i]) and above[i]
                        and dir_main[i] == 1):
                    sig_ts_list.append(int(ts_arr[i]))
                    last_p_bar = -1   # reset so next signal can arm
                elif cup[i] and bool(vf[i]) and above[i] and last_p_bar >= 0:
                    last_p_bar = -1

        return (len(sig_ts_list) > 0, sig_ts_list)

    # Debug mode — both sides
    last_buy_p_bar  = -1
    last_sell_p_bar = -1
    final_buy  = np.zeros(n, bool)
    final_sell = np.zeros(n, bool)
    for i in range(1, end):
        if dir_main[i] == 1  and dir_main[i - 1] != 1:  last_buy_p_bar  = -1
        if dir_main[i] == -1 and dir_main[i - 1] != -1: last_sell_p_bar = -1
        if buy_pressure[i]:  last_buy_p_bar  = i
        if sell_pressure[i]: last_sell_p_bar = i
        fb = cup[i] and bool(vf[i]) and above[i] and last_buy_p_bar  >= 0 and dir_main[i] == 1
        fs = cdn[i] and bool(vf[i]) and below[i] and last_sell_p_bar >= 0 and dir_main[i] == -1
        final_buy[i]  = fb
        final_sell[i] = fs
        if fb: last_buy_p_bar  = -1
        if fs: last_sell_p_bar = -1

    # v9q: w_mask and ts_arr must be same length (end, not n)
    w_mask    = np.zeros(end, bool)
    w_mask[window_start:] = True
    ts_end    = df.ts.values[:end].astype(np.int64)
    buy_idxs  = np.where(final_buy[:end]  & w_mask)[0]
    sell_idxs = np.where(final_sell[:end] & w_mask)[0]

    def _details(idxs):
        return [(int(idx - window_start + 1), int(ts_end[idx])) for idx in idxs]
    return (bool(buy_idxs.size > 0), bool(sell_idxs.size > 0),
            _details(buy_idxs), _details(sell_idxs))


# ══════════════════════════════════════════════════════════════════════
#  v13: BOS / ChoCh CALCULATION  — Pine Script replica
# ══════════════════════════════════════════════════════════════════════

def calc_bos_choch(df: pd.DataFrame, left: int = BOS_LR, right: int = BOS_LR):
    """
    Compute BOS/ChoCh events matching Pine Script logic exactly.
    Returns sorted list of (ts_ms: int, event_type: str).
    event_type in: "bull_bos", "bull_choch", "bear_bos", "bear_choch"
    """
    h  = df.high.values
    l  = df.low.values
    c  = df.close.values
    ts = df.ts.values.astype(np.int64)
    n  = len(c)

    events      = []
    last_high   = np.nan
    high_broken = True
    last_low    = np.nan
    low_broken  = True
    break_type  = 0  # 0=none, 1=last bull, -1=last bear

    for i in range(n):
        mid = i - right
        if mid >= left:
            left_max  = np.max(h[mid - left : mid]) if mid - left < mid else -np.inf
            right_sl  = h[mid + 1 : i + 1]
            right_max = np.max(right_sl) if len(right_sl) > 0 else -np.inf
            if h[mid] > left_max and h[mid] > right_max:
                last_high   = h[mid]
                high_broken = False

            left_min  = np.min(l[mid - left : mid]) if mid - left < mid else np.inf
            right_sl2 = l[mid + 1 : i + 1]
            right_min = np.min(right_sl2) if len(right_sl2) > 0 else np.inf
            if l[mid] < left_min and l[mid] < right_min:
                last_low   = l[mid]
                low_broken = False

        if not np.isnan(last_high) and not high_broken:
            if c[i] > last_high:
                high_broken = True
                etype       = "bull_choch" if break_type == -1 else "bull_bos"
                events.append((int(ts[i]), etype))
                break_type  = 1

        if not np.isnan(last_low) and not low_broken:
            if c[i] < last_low:
                low_broken = True
                etype      = "bear_choch" if break_type == 1 else "bear_bos"
                events.append((int(ts[i]), etype))
                break_type = -1

    return sorted(events, key=lambda x: x[0])


def validate_choch(events, signal_ts_ms: int, want_sell: bool) -> str:
    """
    Apply BOS/ChoCh validity rules to a confirmed Pine signal.
    Returns one of: "valid", "invalid", "wait"
    """
    before = [(ts, et) for ts, et in events if ts <  signal_ts_ms]
    after  = [(ts, et) for ts, et in events if ts >= signal_ts_ms]

    if want_sell:
        if before and before[-1][1] == "bear_choch":
            return "valid"
        if after:
            first = after[0][1]
            if first == "bear_choch":
                return "valid"
            if first == "bull_bos":
                return "invalid"
        return "wait"
    else:
        if before and before[-1][1] == "bull_choch":
            return "valid"
        if after:
            first = after[0][1]
            if first == "bull_choch":
                return "valid"
            if first == "bear_bos":
                return "invalid"
        return "wait"


# ══════════════════════════════════════════════════════════════════════
#  ASYNC FETCH WITH RETRY
# ══════════════════════════════════════════════════════════════════════

async def fetch(ex, sem, sym: str, tf: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV as DataFrame with retry on transient errors."""
    async with sem:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                raw = await ex.fetch_ohlcv(sym, tf, limit=limit)
                if not raw:
                    return pd.DataFrame()
                arr = np.array(raw, dtype=float)
                return pd.DataFrame({
                    "ts":     arr[:, 0].astype(np.int64),
                    "high":   arr[:, 2],
                    "low":    arr[:, 3],
                    "close":  arr[:, 4],
                    "volume": arr[:, 5],
                })
            except Exception:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        return pd.DataFrame()


async def fetch_raw(ex, sem, sym: str, tf: str, limit: int) -> Optional[np.ndarray]:
    """Fetch OHLCV as raw numpy array (skips DataFrame build) with retry."""
    async with sem:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                raw = await ex.fetch_ohlcv(sym, tf, limit=limit)
                if not raw or len(raw) < 5:
                    return None
                return np.array(raw, dtype=float)
            except Exception:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        return None


# ══════════════════════════════════════════════════════════════════════
#  SCAN STAGES
# ══════════════════════════════════════════════════════════════════════

async def stage1_worker(ex, sem, sym: str, cfg: dict):
    """
    Stage 1: Pivot pattern detection + ADX momentum filter.
    Returns (want_sell, sym, detail_str, pivot_ts_ms, tdi_df) or None.
    """
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]

    arr_p = await fetch_raw(ex, sem, sym, pivot_tf, 7)
    if arr_p is None or len(arr_p) < 6:
        return None

    # arr_p columns: [ts, open, high, low, close, vol]
    pivot_ts = int(arr_p[-2, 0])

    def _hlc3(row): return (row[2] + row[3] + row[4]) / 3.0
    cur_P  = _hlc3(arr_p[-2]);  prev_P = _hlc3(arr_p[-3])
    pp_P   = _hlc3(arr_p[-4]);  ppp_P  = _hlc3(arr_p[-5])

    if   cur_P < prev_P and prev_P > max(pp_P, ppp_P): want_sell = True
    elif cur_P > prev_P and prev_P < min(pp_P, ppp_P): want_sell = False
    else: return None

    # Pivot age gate
    is_5m_s1 = tdi_tf == "1h"
    max_age  = (8 * 3600 * 1000) if is_5m_s1 else (48 * 3600 * 1000)
    if int(time.time() * 1000) - pivot_ts > max_age:
        return None

    da = await fetch(ex, sem, sym, tdi_tf, 80)
    if da.empty or len(da) < ADX_LEN * 2:
        return None

    adx_arr     = calc_adx(da.high.values, da.low.values, da.close.values)
    pp_P_ts     = int(arr_p[-4, 0])
    piv4_ts     = int(arr_p[-1, 0])
    ts_vals     = da["ts"].values.astype(np.int64)
    window_mask = (ts_vals >= pp_P_ts) & (ts_vals <= piv4_ts)
    adx_window  = adx_arr[window_mask]
    valid_window = adx_window[~np.isnan(adx_window)]
    if len(valid_window) == 0:
        return None

    adx_ever_above    = bool(np.any(valid_window > ADX_TH))
    adx_at_window_end = float(valid_window[-1])
    if not (adx_ever_above and adx_at_window_end > ADX_TH):
        return None

    adx_peak = float(np.nanmax(valid_window))
    det = (f"P={cur_P:.5f} "
           f"{'prev_peak' if want_sell else 'prev_trough'}={prev_P:.5f} "
           f"ADX_peak={adx_peak:.1f} ADX_end={adx_at_window_end:.1f}")
    return (want_sell, sym, det, pivot_ts, da)


def stage2_worker(want_sell: bool, sym: str, detail: str, pivot_ts: int, da: pd.DataFrame):
    """
    Stage 2: TDI direction + Keltner Channel band filter.
    Returns (want_sell, sym, detail, pivot_ts) or None.
    """
    if da.empty or len(da) < 60:
        return None
    bear_tdi, bull_tdi = tdi_state(da.close.values)
    u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
    c_t                = float(da.close.iloc[-1])
    n_t = len(da); s15 = max(0, n_t - 16); e15 = n_t - 1
    if want_sell:
        if bear_tdi and c_t > l_t[-1] and bool(np.all(da.low.values[s15:e15] > l_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts)
    else:
        if bull_tdi and c_t < u_t[-1] and bool(np.all(da.high.values[s15:e15] < u_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts)
    return None


async def stage3_worker(ex, sem, sym: str, want_sell: bool, detail: str,
                         pivot_ts: int, cfg: dict):
    """
    Stage 3: BB pullback on mid_tf + Pine Final Signal on sig_tf.
    Stage 4: BOS/ChoCh validation on choch_tf (v13/v14).
    Returns (side_str, sym, detail, pivot_ts, choch_status) or None.
    INVALID signals return None (filtered out).
    """
    mid_tf    = cfg["mid_tf"]
    sig_tf    = cfg["sig_tf"]
    choch_tf  = cfg["choch_tf"]
    choch_lim = cfg["choch_limit"]
    is_5m_mode = sig_tf == "5m"
    sig_limit = 156 if is_5m_mode else 252
    mid_limit =  60 if is_5m_mode else 80
    min_sig   =  80

    dm = await fetch(ex, sem, sym, mid_tf, mid_limit)
    if dm.empty or len(dm) < BB_LEN + 10:
        return None

    # v9o: exclude the live (incomplete) candle from BB calculation
    end    = len(dm) - 1
    bb_sig = calc_bb_continuation(
        dm.close.values[:end], dm.high.values[:end], dm.low.values[:end],
        want_sell=want_sell)
    ts_mid   = dm.ts.values[:end].astype(np.int64)
    win_mask = ts_mid >= pivot_ts
    if not bb_sig[win_mask].any():
        return None  # BB fail — saves sig_tf + choch_tf fetches

    # Fetch sig_tf and choch_tf concurrently
    ds, dc = await asyncio.gather(
        fetch(ex, sem, sym, sig_tf,   sig_limit),
        fetch(ex, sem, sym, choch_tf, choch_lim),
    )
    if ds.empty or len(ds) < min_sig:
        return None

    # v14: returns (found, [ts_ms, ...]) — all signal timestamps
    has_signal, sig_ts_list = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
    if not has_signal:
        return None

    # v14: BOS/ChoCh validation — check each signal separately
    RANK = {"valid": 2, "wait": 1, "invalid": 0}
    choch_status = "wait"   # default if data unavailable

    if not dc.empty and len(dc) >= BOS_LR * 2 + 5:
        events    = calc_bos_choch(dc, left=BOS_LR, right=BOS_LR)
        best_rank = -1
        for sig_ts_ms in reversed(sig_ts_list):   # newest → oldest
            st   = validate_choch(events, sig_ts_ms, want_sell)
            rank = RANK[st]
            if rank > best_rank:
                best_rank    = rank
                choch_status = st
            if choch_status == "valid":
                break

    if choch_status == "invalid":
        return None   # silently discard

    side      = "SELL" if want_sell else "BUY"
    n_sigs    = len(sig_ts_list)
    last_sig_ts = sig_ts_list[-1]

    # Last signal bar close price
    ts_sig_arr  = ds.ts.values.astype(np.int64)
    sig_bar_idx = int(np.searchsorted(ts_sig_arr, last_sig_ts, side="left"))
    sig_bar_idx = min(sig_bar_idx, len(ds) - 1)
    last_sig_price = float(ds.close.iloc[sig_bar_idx])

    choch_label = ("✓ChoCh:VALID" if choch_status == "valid"
                   else f"⏳ChoCh:WAIT[{choch_tf.upper()}]")
    det = (f"{detail} | {mid_tf.upper()}_BB_pullback✓ [{sig_tf.upper()} FinalSignal✓ ({n_sigs} sig)]"
           f" [{choch_label}] [window@pivot_ts]"
           f" sig_ts_ms={last_sig_ts} sig_price={last_sig_price:.8g}")
    return (side, sym, det, pivot_ts, choch_status)


# ══════════════════════════════════════════════════════════════════════
#  MAIN SCAN RUNNER
# ══════════════════════════════════════════════════════════════════════

async def run_scan(cfg: dict, progress_callback: Callable) -> dict:
    """
    Run full 4-stage pipeline over all USDT perpetuals.
    Calls progress_callback(state_dict) for live UI updates (throttled by caller).
    """
    ex = _make_exchange()
    try:
        # v9b: cache market list in session_state to avoid repeated load_markets
        if "markets" not in st.session_state:
            await ex.load_markets()
            st.session_state["markets"] = ex.markets
        else:
            ex.markets = st.session_state["markets"]
            ex.markets_by_id = {m["id"]: m for m in ex.markets.values()}

        symbols = sorted([
            s for s, m in ex.markets.items()
            if m.get("type") == "swap" and m.get("active")
            and m.get("quote") == "USDT" and ":USDT" in s
        ])
        total = len(symbols)
        sem   = asyncio.Semaphore(MAX_CONCURRENT)

        state = {
            "s1_done": 0, "s2_in": 0, "s3_in": 0,
            "buy_valid": [], "buy_wait": [], "sell_valid": [], "sell_wait": [],
            "total": total,
        }
        last_ui_update = 0.0

        async def worker(sym: str):
            nonlocal last_ui_update
            r1 = await stage1_worker(ex, sem, sym, cfg)
            state["s1_done"] += 1
            now = time.time()
            if now - last_ui_update >= UI_THROTTLE_S:
                progress_callback(state)
                last_ui_update = now
            if r1 is None:
                return

            want_sell, sym, detail, pivot_ts, da = r1
            state["s2_in"] += 1

            r2 = stage2_worker(want_sell, sym, detail, pivot_ts, da)
            if r2 is None:
                return

            want_sell, sym, detail, pivot_ts = r2
            state["s3_in"] += 1

            r3 = await stage3_worker(ex, sem, sym, want_sell, detail, pivot_ts, cfg)
            if r3:
                side, sym2, det2, pt, choch_st = r3
                entry = (sym2, det2, pt, choch_st)
                if side == "BUY":
                    if choch_st == "valid": state["buy_valid"].append(entry)
                    else:                   state["buy_wait"].append(entry)
                else:
                    if choch_st == "valid": state["sell_valid"].append(entry)
                    else:                   state["sell_wait"].append(entry)
                progress_callback(state)
                last_ui_update = time.time()

        await asyncio.gather(*[worker(s) for s in symbols])
        progress_callback(state)  # final update
        return state
    finally:
        await ex.close()


# ══════════════════════════════════════════════════════════════════════
#  DEBUG SINGLE SYMBOL
# ══════════════════════════════════════════════════════════════════════

async def debug_single(sym_raw: str, cfg: dict) -> list:
    """
    Debug a single symbol through all pipeline stages.
    v9j: delegates to shared stage workers — no duplicated logic.
    v12: adds Stage 4 BOS/ChoCh validation.
    Returns list of (label, status, detail) tuples.
    """
    raw       = sym_raw.strip().upper().replace(" ", "")
    raw_clean = raw.replace("/", "").replace(":", "")
    base      = raw_clean.replace("USDT", "") or raw_clean
    sym       = f"{base}/USDT:USDT"
    logs      = []

    ex = _make_exchange()
    try:
        await ex.load_markets()
        if sym not in ex.markets:
            logs.append(("Symbol", "❌ FAIL", f"'{sym}' not found on Binance Futures"))
            return logs

        sem = asyncio.Semaphore(10)
        pivot_tf  = cfg["pivot_tf"]
        tdi_tf    = cfg["tdi_tf"]
        mid_tf    = cfg["mid_tf"]
        sig_tf    = cfg["sig_tf"]
        choch_tf  = cfg["choch_tf"]
        choch_lim = cfg["choch_limit"]

        # ── Stage 1: fetch data ──────────────────────────────────────
        dp, da = await asyncio.gather(
            fetch(ex, sem, sym, pivot_tf, 7),
            fetch(ex, sem, sym, tdi_tf,   80),
        )

        if dp.empty or len(dp) < 5:
            logs.append(("S1 Pivot data", "❌ FAIL", f"Not enough {pivot_tf} candles"))
            return logs
        if da.empty or len(da) < ADX_LEN * 2:
            logs.append(("S1 ADX data",   "❌ FAIL", f"Not enough {tdi_tf} candles"))
            return logs

        pivot_ts = int(dp.iloc[-2]["ts"])
        cur_P, prev_P, pp_P, ppp_P = pivot_chain(dp)

        if cur_P is None:
            logs.append(("S1 Pivot data", "❌ FAIL", "Not enough candles for pivot_chain"))
            return logs

        sell_pivot = cur_P < prev_P and prev_P > max(pp_P, ppp_P)
        buy_pivot  = cur_P > prev_P and prev_P < min(pp_P, ppp_P)

        if sell_pivot:
            direction = "SELL"
            logs.append(("S1 Pivot", "✅ PASS",
                f"SELL | cur_P={cur_P:.5f} < prev_P={prev_P:.5f} | prev_P > max(pp,ppp)"))
        elif buy_pivot:
            direction = "BUY"
            logs.append(("S1 Pivot", "✅ PASS",
                f"BUY  | cur_P={cur_P:.5f} > prev_P={prev_P:.5f} | prev_P < min(pp,ppp)"))
        else:
            logs.append(("S1 Pivot", "❌ FAIL",
                f"No valid pivot | cur={cur_P:.5f} prev={prev_P:.5f} pp={pp_P:.5f}"))
            return logs

        # ── Stage 1: ADX check ───────────────────────────────────────
        adx_arr      = calc_adx(da.high.values, da.low.values, da.close.values)
        arr_p_ts     = dp["ts"].values.astype(np.int64)
        pp_P_ts      = int(arr_p_ts[-4])
        piv4_ts      = int(arr_p_ts[-1])
        ts_vals      = da["ts"].values.astype(np.int64)
        window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= piv4_ts)
        adx_window   = adx_arr[window_mask]
        valid_window = adx_window[~np.isnan(adx_window)]

        if len(valid_window) == 0:
            logs.append(("S1 ADX", "❌ FAIL", "No ADX candles in pivot window"))
            return logs

        adx_ever_above    = bool(np.any(valid_window > ADX_TH))
        adx_at_window_end = float(valid_window[-1])
        adx_end_above     = adx_at_window_end > ADX_TH
        adx_peak          = float(np.nanmax(valid_window))

        if adx_ever_above and adx_end_above:
            logs.append(("S1 ADX", "✅ PASS",
                f"peak={adx_peak:.1f} end={adx_at_window_end:.1f} > {ADX_TH}"))
        elif not adx_ever_above:
            logs.append(("S1 ADX", "❌ FAIL",
                f"Never above {ADX_TH} | peak={adx_peak:.1f}"))
            return logs
        else:
            logs.append(("S1 ADX", "❌ FAIL",
                f"Was above {ADX_TH} but dropped | end={adx_at_window_end:.1f}"))
            return logs

        # ── Stage 2 ──────────────────────────────────────────────────
        want_sell   = direction == "SELL"
        bear_tdi, bull_tdi = tdi_state(da.close.values)
        u_t, l_t    = calc_kc(da.high.values, da.low.values, da.close.values)
        c_t         = float(da.close.iloc[-1])
        n_t = len(da); s15 = max(0, n_t - 16); e15 = n_t - 1

        tdi_ok  = (want_sell and bear_tdi) or (not want_sell and bull_tdi)
        kc_ok   = (want_sell and c_t > l_t[-1]) or (not want_sell and c_t < u_t[-1])
        band_ok = bool(np.all(da.low.values[s15:e15]  > l_t[s15:e15])) if want_sell \
             else bool(np.all(da.high.values[s15:e15] < u_t[s15:e15]))

        logs.append(("S2 TDI", "✅ PASS" if tdi_ok else "❌ FAIL",
            f"bear={bear_tdi} bull={bull_tdi} → need {'bear' if want_sell else 'bull'}"))
        if not tdi_ok: return logs

        logs.append(("S2 KC Band", "✅ PASS" if kc_ok else "❌ FAIL",
            f"close={c_t:.5f} {'>' if want_sell else '<'} KC {'lower' if want_sell else 'upper'}"))
        if not kc_ok: return logs

        logs.append(("S2 Band Clean", "✅ PASS" if band_ok else "❌ FAIL",
            f"Last 15 {'lows > KC lower' if want_sell else 'highs < KC upper'}: {band_ok}"))
        if not band_ok: return logs

        # ── Stage 3 ──────────────────────────────────────────────────
        is_5m_mode = sig_tf == "5m"
        sig_limit  = 156 if is_5m_mode else 252
        mid_limit  =  60 if is_5m_mode else 80
        min_sig    =  80

        dm = await fetch(ex, sem, sym, mid_tf, mid_limit)
        if dm.empty or len(dm) < BB_LEN + 10:
            logs.append(("S3 BB data", "❌ FAIL", f"Not enough {mid_tf} candles"))
            return logs

        end     = len(dm) - 1
        bb_sig  = calc_bb_continuation(
            dm.close.values[:end], dm.high.values[:end], dm.low.values[:end],
            want_sell=want_sell)
        ts_mid  = dm.ts.values[:end].astype(np.int64)
        win_mask = ts_mid >= pivot_ts
        bb_ok   = bb_sig[win_mask].any()
        logs.append(("S3 BB Pullback", "✅ PASS" if bb_ok else "❌ FAIL",
            f"{int(bb_sig[win_mask].sum())} BB {direction} signal(s) in pivot window"))
        if not bb_ok: return logs

        ds = await fetch(ex, sem, sym, sig_tf, sig_limit)
        if ds.empty or len(ds) < min_sig:
            logs.append(("S3 Sig data", "❌ FAIL", f"Not enough {sig_tf} candles"))
            return logs

        has_signal, sig_ts_list = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
        n_sigs = len(sig_ts_list)
        sig_times = ""
        if sig_ts_list:
            last_ts = sig_ts_list[-1]
            sig_times = datetime.datetime.utcfromtimestamp(last_ts / 1000).strftime("%Y-%m-%d %H:%M UTC")
        logs.append(("S3 Pine Final Signal", "✅ PASS" if has_signal else "❌ FAIL",
            f"{sig_tf.upper()} Final {direction} Signal in window: {has_signal}  |  "
            f"{n_sigs} signal(s)  |  ⏰ Latest: {sig_times}"))
        if not has_signal: return logs

        # ── Stage 4: BOS/ChoCh ───────────────────────────────────────
        dc = await fetch(ex, sem, sym, choch_tf, choch_lim)
        choch_status = "wait"
        if dc.empty or len(dc) < BOS_LR * 2 + 5:
            logs.append((f"S4 BOS/ChoCh [{choch_tf.upper()}]", "⏳ WAIT",
                f"Not enough {choch_tf} data (got {len(dc)}) — defaulting to WAIT"))
        else:
            events   = calc_bos_choch(dc, left=BOS_LR, right=BOS_LR)
            RANK     = {"valid": 2, "wait": 1, "invalid": 0}
            best_rank = -1
            for sig_ts_ms in reversed(sig_ts_list):
                st   = validate_choch(events, sig_ts_ms, want_sell)
                rank = RANK[st]
                if rank > best_rank:
                    best_rank    = rank
                    choch_status = st
                if choch_status == "valid":
                    break

            status_label = {"valid": "✅ VALID", "wait": "⏳ WAIT", "invalid": "❌ INVALID"}[choch_status]
            detail_msg   = (
                f"{len(events)} BOS/ChoCh events on {choch_tf.upper()}  |  "
                f"checked {n_sigs} signal(s) newest-first  |  best result: {choch_status.upper()}"
            )
            logs.append((f"S4 BOS/ChoCh [{choch_tf.upper()}]", status_label, detail_msg))

        return logs
    finally:
        await ex.close()


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _run_async(coro):
    """
    v9f: Use get_event_loop().run_until_complete() — avoids 'loop already running'
    errors on some Streamlit Cloud deployments where asyncio.run() conflicts with
    the existing loop even after nest_asyncio.apply().
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _parse_row(direction: str, sym: str, det: str, pivot_ts: int,
               choch_status: str, now_ms: int, mode_key: str, timestamp: str) -> dict:
    """
    v12: Parse a result row into structured fields.
    choch_status: "valid" or "wait"
    """
    p      = _re.search(r"P=([\d.]+)",                    det)
    prev   = _re.search(r"prev_(?:peak|trough)=([\d.]+)", det)
    adxpk  = _re.search(r"ADX_peak=([\d.]+)",             det)
    adxend = _re.search(r"ADX_end=([\d.]+)",              det)
    bb_m   = _re.search(r"(\w+)_BB_pullback",             det)
    sig_m  = _re.search(r"\[(\w+) FinalSignal",           det)
    sig_ts = _re.search(r"sig_ts_ms=(\d+)",               det)
    sig_px = _re.search(r"sig_price=([\d.eE+\-]+)",       det)
    age_h  = round((now_ms - pivot_ts) / 3_600_000, 1)

    # Signal bar time
    if sig_ts:
        sig_dt = datetime.datetime.utcfromtimestamp(int(sig_ts.group(1)) / 1000).strftime("%Y-%m-%d %H:%M UTC")
    else:
        sig_dt = ""

    # Signal bar price
    if sig_px:
        pval = float(sig_px.group(1))
        if pval >= 1000:     price_str = f"{pval:,.2f}"
        elif pval >= 1:      price_str = f"{pval:.4f}"
        elif pval >= 0.0001: price_str = f"{pval:.6f}"
        else:                price_str = f"{pval:.4e}"
    else:
        price_str = ""

    choch_label = "✅ VALID" if choch_status == "valid" else "⏳ WAIT"

    return {
        "Direction":      direction,
        "Symbol":         sym,
        "Pivot_P":        float(p.group(1))      if p      else "",
        "Prev_Pivot":     float(prev.group(1))   if prev   else "",
        "ADX_Peak":       float(adxpk.group(1))  if adxpk  else "",
        "ADX_End":        float(adxend.group(1)) if adxend else "",
        "BB_TF":          bb_m.group(1)          if bb_m   else "",
        "Signal_TF":      sig_m.group(1)         if sig_m  else "",
        "Signal_Price":   price_str,
        "Signal_Time":    sig_dt,
        "ChoCh":          choch_label,
        "Pivot_Age_h":    age_h,
        "Scan_Time":      timestamp,
        "Mode":           mode_key.upper(),
    }


def _parse_det_card(det: str) -> dict:
    """Parse detail string into card display fields (no timestamp arg needed)."""
    import re as _re2
    adx    = _re2.search(r"ADX_(?:cur|peak|end)=([\d.]+)", det)
    bb_m   = _re2.search(r"(\w+)_BB_pullback",             det)
    sig_m  = _re2.search(r"\[(\w+) FinalSignal",           det)
    sig_ts = _re2.search(r"sig_ts_ms=(\d+)",               det)
    sig_px = _re2.search(r"sig_price=([\d.eE+\-]+)",       det)
    n_sigs = _re2.search(r"\((\d+) sig",                   det)

    # Price formatting
    if sig_px:
        pval = float(sig_px.group(1))
        if pval >= 1000:     price_str = f"{pval:,.2f}"
        elif pval >= 1:      price_str = f"{pval:.4f}"
        elif pval >= 0.0001: price_str = f"{pval:.6f}"
        else:                price_str = f"{pval:.4e}"
    else:
        price_str = "—"

    # Age & time
    if sig_ts:
        age_ms  = int(time.time() * 1000) - int(sig_ts.group(1))
        age_h   = age_ms / 3_600_000
        if age_h < 1:   age_str = f"{age_h*60:.0f}m"
        elif age_h < 24: age_str = f"{age_h:.1f}h"
        else:            age_str = f"{age_h/24:.1f}d"
        sig_time = datetime.datetime.utcfromtimestamp(
            int(sig_ts.group(1)) / 1000).strftime("%Y-%m-%d %H:%M UTC")
    else:
        age_h = 0.0; age_str = "—"; sig_time = "—"

    # ADX value — try multiple patterns
    adx_v = "—"
    if adx:
        adx_v = f"{float(adx.group(1)):.0f}"
    else:
        m2 = _re2.search(r"ADX_(?:peak|end|cur)=([\d.]+)", det)
        if m2: adx_v = f"{float(m2.group(1)):.0f}"

    return {
        "price":  price_str,
        "adx":    adx_v if adx_v != "—" else "—",
        "bb_tf":  bb_m.group(1).upper()  if bb_m  else "—",
        "sig_tf": sig_m.group(1).upper() if sig_m else "—",
        "age_str": age_str,
        "age_h":   str(age_h),
        "sig_time": sig_time,
        "n_sigs": n_sigs.group(1) if n_sigs else "1",
    }


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT APP LAYOUT
# ══════════════════════════════════════════════════════════════════════

def _init_session():
    """Ensure all session_state keys exist on first load."""
    defaults = {
        "scan_done":    False,
        "scan_state":   None,
        "scan_elapsed": 0.0,
        "scan_mode":    "15m",
        "df_final":     None,
        "buy_valid":    [],
        "buy_wait":     [],
        "sell_valid":   [],
        "sell_wait":    [],
        "csv_bytes":    None,
        "txt_bytes":    None,
        "csv_fname":    "",
        "txt_fname":    "",
        "results_tab":  "all",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sc_counters_html(nbv: int, nbw: int, nsv: int, nsw: int,
                      s2: int, s3: int, elapsed: float, total: int, done: int) -> str:
    spd = done / max(elapsed, 0.01)
    return f"""
<div class="sc-counters">
  <div class="sc-cnt g">
    <div class="cnt-lbl">BUY ✅ VALID</div>
    <div class="cnt-val">{nbv}</div>
  </div>
  <div class="sc-cnt g">
    <div class="cnt-lbl">BUY ⏳ WAIT</div>
    <div class="cnt-val">{nbw}</div>
    <div class="cnt-sub">ChoCh pending</div>
  </div>
  <div class="sc-cnt r">
    <div class="cnt-lbl">SELL ✅ VALID</div>
    <div class="cnt-val">{nsv}</div>
  </div>
  <div class="sc-cnt r">
    <div class="cnt-lbl">SELL ⏳ WAIT</div>
    <div class="cnt-val">{nsw}</div>
    <div class="cnt-sub">ChoCh pending</div>
  </div>
  <div class="sc-cnt b">
    <div class="cnt-lbl">S2 Passed</div>
    <div class="cnt-val">{s2}</div>
  </div>
  <div class="sc-cnt b">
    <div class="cnt-lbl">S3+S4 Pass</div>
    <div class="cnt-val">{s3}</div>
  </div>
  <div class="sc-cnt gy">
    <div class="cnt-lbl">Scanned</div>
    <div class="cnt-val">{done}</div>
    <div class="cnt-sub">{spd:.0f}/s · {elapsed:.0f}s</div>
  </div>
</div>"""


def _sc_summary_html(total: int, elapsed: float, bv: int, bw: int,
                     sv: int, sw: int, mode_key: str) -> str:
    all_s = bv + bw + sv + sw
    spd   = total / max(elapsed, 0.01)
    return f"""
<div class="sc-summary">
  <div class="ss-title">✅ Scan <span>Complete</span></div>
  <div class="ss-div"></div>
  <div class="ss-group">
    <div class="ss-num" style="color:var(--green)">{bv}</div>
    <div class="ss-lbl" style="color:var(--green)">BUY ✅</div>
  </div>
  <div class="ss-group">
    <div class="ss-num" style="color:#50c878">{bw}</div>
    <div class="ss-lbl" style="color:#50c878">BUY ⏳</div>
  </div>
  <div class="ss-group">
    <div class="ss-num" style="color:var(--red)">{sv}</div>
    <div class="ss-lbl" style="color:var(--red)">SELL ✅</div>
  </div>
  <div class="ss-group">
    <div class="ss-num" style="color:#e05060">{sw}</div>
    <div class="ss-lbl" style="color:#e05060">SELL ⏳</div>
  </div>
  <div class="ss-div"></div>
  <div class="ss-meta">
    <b>{all_s}</b> signals &nbsp;·&nbsp; {total} symbols &nbsp;·&nbsp;
    {elapsed:.1f}s &nbsp;·&nbsp; {spd:.0f} sym/s &nbsp;·&nbsp;
    Mode: <b>{mode_key.upper()}</b>
  </div>
</div>"""


def _signal_cards_html(entries: list, is_buy: bool, is_valid: bool) -> str:
    """Render a grid of signal cards from list of (sym, det) tuples."""
    if not entries:
        label = ("BUY" if is_buy else "SELL") + (" confirmed" if is_valid else " waiting")
        return f'<div class="sc-empty"><div class="ico">🔎</div><p>No {label} signals.</p></div>'

    dir_cls  = ("buy-valid"  if is_buy and is_valid  else
                "buy-wait"   if is_buy               else
                "sell-valid" if is_valid              else "sell-wait")
    card_cls = "buy" if is_buy else "sell"
    if not is_valid:
        card_cls += " wait"

    dir_label = ("▲ BUY ✅" if is_buy and is_valid  else
                 "▲ BUY ⏳" if is_buy               else
                 "▼ SELL ✅" if is_valid             else "▼ SELL ⏳")
    choch_cls   = "choch-valid" if is_valid else "choch-wait"
    choch_label = "✅ ChoCh VALID" if is_valid else "⏳ ChoCh WAIT"

    cards = []
    for sym, det in entries:
        p       = _parse_det_card(det)
        sym_s   = sym.split("/")[0]
        age_cls = "warn" if float(p["age_h"]) > 24 else ("ok" if float(p["age_h"]) < 8 else "")
        adx_cls = "ok" if float(p["adx"]) > 35 else ("warn" if float(p["adx"]) > 25 else "err")
        cards.append(f"""
<div class="sc-card {card_cls}">
  <div class="sc-card-top">
    <span class="sc-card-sym">{sym_s}</span>
    <span class="sc-card-dir dir-{dir_cls}">{dir_label}</span>
  </div>
  <div class="sc-card-price">{p['price']}</div>
  <div class="sc-card-meta">
    <span class="sc-tag {adx_cls}">ADX <b>{p['adx']}</b></span>
    <span class="sc-tag">BB <b>{p['bb_tf']}</b></span>
    <span class="sc-tag">Sig <b>{p['sig_tf']}</b></span>
    <span class="sc-tag {age_cls}">Age <b>{p['age_str']}</b></span>
    <span class="sc-tag">{p['n_sigs']} sig</span>
  </div>
  <span class="sc-choch {choch_cls}">{choch_label}</span>
  <div class="sc-tag" style="margin-top:2px;font-size:0.7rem">⏰ {p['sig_time']}</div>
</div>""")

    return f'<div class="sc-grid">{"".join(cards)}</div>'


def main():
    _init_session()

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
<div class="sc-header">
  <div class="sc-header-left">
    <h1>&#9889; Binance Futures <span>Scanner</span></h1>
    <div class="sub">Ultra-Fast &middot; Multi-Stage &middot; BOS/ChoCh Validated &middot; Pine Accurate</div>
  </div>
  <div class="sc-header-right">
    <span class="sc-badge blue">v12</span>
    <span class="sc-badge green">4 Stages</span>
    <span class="sc-badge gold">BOS/ChoCh</span>
  </div>
</div>
""", unsafe_allow_html=True)

    tab_scan, tab_debug = st.tabs(["&#128269;  Full Scan", "&#128027;  Debug Symbol"])

    # ══ TAB 1: FULL SCAN ══════════════════════════════════════════════
    with tab_scan:

        # ── Proxy status ──────────────────────────────────────────────
        _proxy = _get_proxy()
        if _proxy:
            _host = _proxy.split("@")[-1] if "@" in _proxy else _proxy.split("//")[-1]
            st.markdown(
                f'<div class="sc-proxy-ok">&#128274; Proxy active &mdash; <b>{_host}</b> &middot; Binance geo-block bypassed</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sc-proxy-err">&#128683; No proxy &mdash; Binance blocks Streamlit Cloud IPs. '
                'Add <code>PROXY_URL</code> to Streamlit Secrets.</div>',
                unsafe_allow_html=True)
            with st.expander("How to set up a free proxy (3 min)"):
                st.markdown("""
1. Register at **https://proxy2.webshare.io** (free, no credit card)
2. Go to **Proxy → List** → Download as `Username:Password@IP:Port`
3. In Streamlit → your app → **⋮ → Settings → Secrets**, add:
```
PROXY_URL = "http://user:pass@1.2.3.4:8080"
```
4. Save — app restarts in ~30s
""")

        # ── Mode + Timeframes row ─────────────────────────────────────
        ctrl_col, tf_col = st.columns([2, 3])
        with ctrl_col:
            mode_choice = st.radio(
                "**SCAN MODE**",
                ["15M  (Daily → 4H → 1H → 15M)", "5M  (4H → 1H → 15M → 5M)"],
                index=0,
            )
            mode_key = "15m" if mode_choice.startswith("15M") else "5m"
            cfg = MODES[mode_key]
        with tf_col:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Pivot",   cfg["pivot_tf"].upper())
            c2.metric("TDI/ADX", cfg["tdi_tf"].upper())
            c3.metric("BB",      cfg["mid_tf"].upper())
            c4.metric("Signal",  cfg["sig_tf"].upper())
            c5.metric("ChoCh",   cfg["choch_tf"].upper())

        # ── Rule pills ────────────────────────────────────────────────
        st.markdown(
            f"<div class='sc-pills'>"
            f"<span class='sc-pill'><span class='num'>S1</span>"
            f"{cfg['pivot_tf'].upper()} Pivot <span class='arr'>&#8594;</span> ADX&gt;{ADX_TH:.0f}</span>"
            f"<span class='sc-pill'><span class='num'>S2</span>"
            f"TDI direction <span class='arr'>&#8594;</span> KC Band</span>"
            f"<span class='sc-pill'><span class='num'>S3</span>"
            f"{cfg['mid_tf'].upper()} BB Pullback <span class='arr'>&#8594;</span> {cfg['sig_tf'].upper()} Pine Signal</span>"
            f"<span class='sc-pill'><span class='num'>S4</span>"
            f"{cfg['choch_tf'].upper()} BOS/ChoCh Validate</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Action buttons ────────────────────────────────────────────
        btn_c1, btn_c2, _sp = st.columns([3, 1, 4])
        with btn_c1:
            scan_clicked = st.button("&#128640;  Start Scan", type="primary", key="scan_btn",
                                     use_container_width=True)
        with btn_c2:
            if st.button("&#128260; Markets", key="clear_mkts", use_container_width=True,
                         help="Clear cached market list — forces fresh reload from Binance"):
                st.session_state.pop("markets", None)
                st.rerun()

        st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.5rem 0 0.7rem'>",
                    unsafe_allow_html=True)

        # ── SCAN EXECUTION ────────────────────────────────────────────
        if scan_clicked:
            st.session_state.update({
                "scan_done": False, "df_final": None,
                "buy_valid": [], "buy_wait": [], "sell_valid": [], "sell_wait": [],
                "csv_bytes": None, "txt_bytes": None,
            })
            t0 = time.time()

            prog_bar = st.progress(0.0, text="Connecting to Binance…")
            ctr_ph   = st.empty()

            def update_ui(state: dict):
                total   = state["total"]
                done    = state["s1_done"]
                elapsed = time.time() - t0
                pct     = done / max(total, 1)
                all_s   = (len(state["buy_valid"]) + len(state["buy_wait"]) +
                           len(state["sell_valid"]) + len(state["sell_wait"]))
                spd = done / max(elapsed, 0.01)
                prog_bar.progress(
                    min(pct, 1.0),
                    text=f"Scanning {done}/{total} · {spd:.0f} sym/s · "
                         f"S2:{state['s2_in']} S3:{state['s3_in']} · Signals:{all_s}"
                )
                ctr_ph.markdown(
                    _sc_counters_html(
                        len(state["buy_valid"]), len(state["buy_wait"]),
                        len(state["sell_valid"]), len(state["sell_wait"]),
                        state["s2_in"], state["s3_in"], elapsed, total, done),
                    unsafe_allow_html=True,
                )

            try:
                state = _run_async(run_scan(cfg, update_ui))
            except Exception as e:
                st.error(f"Scan failed: {e}")
                st.exception(e)
                state = None

            if state:
                elapsed    = time.time() - t0
                total      = state["total"]
                buy_valid  = sorted(state["buy_valid"],  key=lambda x: x[0])
                buy_wait   = sorted(state["buy_wait"],   key=lambda x: x[0])
                sell_valid = sorted(state["sell_valid"], key=lambda x: x[0])
                sell_wait  = sorted(state["sell_wait"],  key=lambda x: x[0])

                prog_bar.progress(1.0, text=f"Done — {total} symbols in {elapsed:.1f}s")
                ctr_ph.empty()

                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                now_ms    = int(time.time() * 1000)
                ts_int    = int(time.time())

                all_results = (
                    [("BUY",  s, d, p, c) for s, d, p, c in buy_valid] +
                    [("BUY",  s, d, p, c) for s, d, p, c in buy_wait]  +
                    [("SELL", s, d, p, c) for s, d, p, c in sell_valid] +
                    [("SELL", s, d, p, c) for s, d, p, c in sell_wait]
                )

                if all_results:
                    all_rows = [
                        _parse_row(dir_, s, d, p, choch_st, now_ms, mode_key, timestamp)
                        for dir_, s, d, p, choch_st in all_results
                    ]
                    df_final = pd.DataFrame(all_rows)
                    csv_buf  = io.StringIO()
                    df_final.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")

                    txt_buf = io.StringIO()
                    txt_buf.write(f"BINANCE FUTURES SCANNER  —  {mode_key.upper()} MODE\n")
                    txt_buf.write(f"Scan Time : {timestamp}\n")
                    txt_buf.write(f"Symbols   : {total}  |  Elapsed : {elapsed:.1f}s\n")
                    txt_buf.write(f"BUY  : {len(buy_valid)} VALID  {len(buy_wait)} WAIT\n")
                    txt_buf.write(f"SELL : {len(sell_valid)} VALID  {len(sell_wait)} WAIT\n")
                    txt_buf.write("=" * 72 + "\n")
                    for dir_, group_label, group in [
                        ("BUY",  "BUY CONFIRMED",  buy_valid),
                        ("BUY",  "BUY WAITING",    buy_wait),
                        ("SELL", "SELL CONFIRMED", sell_valid),
                        ("SELL", "SELL WAITING",   sell_wait),
                    ]:
                        if not group: continue
                        txt_buf.write(f"\n{'─'*28} {group_label} {'─'*28}\n")
                        for sym, det, pts, choch_st in group:
                            r = _parse_row(dir_, sym, det, pts, choch_st, now_ms, mode_key, timestamp)
                            txt_buf.write(
                                f"  {r['Symbol']:<24}  Price={r['Signal_Price']}\n"
                                f"  {'':24}  ADX peak={r['ADX_Peak']}  end={r['ADX_End']}  Age={r['Pivot_Age_h']}h\n"
                                f"  {'':24}  BB={r['BB_TF']}  Signal={r['Signal_TF']}\n"
                                f"  {'':24}  Pine Signal Time: {r['Signal_Time']}\n"
                                f"  {'':24}  BOS/ChoCh: {r['ChoCh']}\n\n"
                            )
                    txt_bytes = txt_buf.getvalue().encode("utf-8")

                    st.session_state.update({
                        "scan_done":    True,
                        "scan_state":   state,
                        "scan_elapsed": elapsed,
                        "scan_mode":    mode_key,
                        "df_final":     df_final,
                        "buy_valid":    [(s, d) for s, d, _, _ in buy_valid],
                        "buy_wait":     [(s, d) for s, d, _, _ in buy_wait],
                        "sell_valid":   [(s, d) for s, d, _, _ in sell_valid],
                        "sell_wait":    [(s, d) for s, d, _, _ in sell_wait],
                        "csv_bytes":    csv_bytes,
                        "txt_bytes":    txt_bytes,
                        "csv_fname":    f"signals_{mode_key}_{ts_int}.csv",
                        "txt_fname":    f"signals_{mode_key}_{ts_int}.txt",
                    })
                else:
                    st.session_state.update({
                        "scan_done":    True,
                        "scan_state":   state,
                        "scan_elapsed": elapsed,
                        "scan_mode":    mode_key,
                        "df_final":     None,
                        "buy_valid": [], "buy_wait": [], "sell_valid": [], "sell_wait": [],
                    })
                st.rerun()  # clean rerender — no stale placeholders above results

        # ══════════════════════════════════════════════════════════════
        #  RESULTS — always rendered purely from session_state
        #            (fully sticky: switching tabs won't lose this)
        # ══════════════════════════════════════════════════════════════
        if st.session_state["scan_done"] and st.session_state["scan_state"] is not None:
            state      = st.session_state["scan_state"]
            elapsed    = st.session_state["scan_elapsed"]
            mode_key_r = st.session_state["scan_mode"]
            df_final   = st.session_state["df_final"]
            total      = state["total"]

            bv_list = st.session_state["buy_valid"]
            bw_list = st.session_state["buy_wait"]
            sv_list = st.session_state["sell_valid"]
            sw_list = st.session_state["sell_wait"]
            bv, bw, sv, sw = len(bv_list), len(bw_list), len(sv_list), len(sw_list)
            all_sigs = bv + bw + sv + sw

            # Persistent summary banner
            st.markdown(
                _sc_summary_html(total, elapsed, bv, bw, sv, sw, mode_key_r),
                unsafe_allow_html=True)

            if all_sigs == 0:
                st.markdown(
                    '<div class="sc-empty"><div class="ico">&#128301;</div>'
                    '<p>No signals &mdash; market conditions did not meet all 4 stage filters.</p></div>',
                    unsafe_allow_html=True)
            else:
                # Signal card tabs — sticky (backed by session_state)
                tab_labels = [
                    f"All ({all_sigs})",
                    f"BUY Confirmed  {bv}",
                    f"BUY Waiting  {bw}",
                    f"SELL Confirmed  {sv}",
                    f"SELL Waiting  {sw}",
                ]
                t_all, t_bv, t_bw, t_sv, t_sw = st.tabs(tab_labels)

                with t_all:
                    if bv_list:
                        st.markdown(_signal_cards_html(bv_list, True, True),  unsafe_allow_html=True)
                    if sv_list:
                        st.markdown(_signal_cards_html(sv_list, False, True), unsafe_allow_html=True)
                    if bw_list:
                        st.markdown(_signal_cards_html(bw_list, True, False), unsafe_allow_html=True)
                    if sw_list:
                        st.markdown(_signal_cards_html(sw_list, False, False), unsafe_allow_html=True)
                    if not any([bv_list, sv_list, bw_list, sw_list]):
                        st.markdown('<div class="sc-empty"><div class="ico">&#128269;</div><p>No signals.</p></div>',
                                    unsafe_allow_html=True)

                with t_bv:
                    st.markdown(_signal_cards_html(bv_list, True, True),  unsafe_allow_html=True)
                with t_bw:
                    st.markdown(_signal_cards_html(bw_list, True, False), unsafe_allow_html=True)
                with t_sv:
                    st.markdown(_signal_cards_html(sv_list, False, True), unsafe_allow_html=True)
                with t_sw:
                    st.markdown(_signal_cards_html(sw_list, False, False), unsafe_allow_html=True)

                # Full table + export in collapsible
                if df_final is not None and not df_final.empty:
                    with st.expander("&#128203; Full Data Table + Export", expanded=False):
                        display_cols = [
                            "Direction", "Symbol", "ChoCh", "Signal_Price",
                            "ADX_Peak", "ADX_End", "BB_TF", "Signal_TF",
                            "Signal_Time", "Pivot_Age_h",
                        ]
                        col_cfg = {
                            "Direction":    st.column_config.TextColumn("Dir",       width=85),
                            "Symbol":       st.column_config.TextColumn("Symbol",    width=150),
                            "ChoCh":        st.column_config.TextColumn("BOS/ChoCh", width=100),
                            "Signal_Price": st.column_config.TextColumn("Price",     width=100),
                            "ADX_Peak":     st.column_config.NumberColumn("ADX Pk",  format="%.1f", width=75),
                            "ADX_End":      st.column_config.NumberColumn("ADX End", format="%.1f", width=75),
                            "BB_TF":        st.column_config.TextColumn("BB TF",     width=58),
                            "Signal_TF":    st.column_config.TextColumn("Sig TF",    width=62),
                            "Signal_Time":  st.column_config.TextColumn("Signal Time", width=160),
                            "Pivot_Age_h":  st.column_config.NumberColumn("Age h",   format="%.1f", width=62),
                        }
                        st.dataframe(
                            df_final[display_cols], use_container_width=True, hide_index=True,
                            height=min(540, 50 + 36 * len(df_final)), column_config=col_cfg)

                        ec1, ec2, _sp2 = st.columns([1, 1, 2])
                        ec1.download_button(
                            "&#128196; Export CSV", data=st.session_state["csv_bytes"],
                            file_name=st.session_state["csv_fname"],
                            mime="text/csv", use_container_width=True)
                        ec2.download_button(
                            "&#128221; Export TXT", data=st.session_state["txt_bytes"],
                            file_name=st.session_state["txt_fname"],
                            mime="text/plain", use_container_width=True)

    # ══ TAB 2: DEBUG SYMBOL ═══════════════════════════════════════════
    with tab_debug:
        st.markdown("#### &#128027; Debug a Single Symbol")
        st.caption("Runs every pipeline stage verbosely — see exactly where and why a pair passes or fails.")

        d_col1, d_col2 = st.columns([2, 3])
        with d_col1:
            dbg_mode = st.radio(
                "**RULESET**",
                ["15M  (Daily → 4H → 1H → 15M)", "5M  (4H → 1H → 15M → 5M)"],
                index=0, key="dbg_mode"
            )
            dbg_cfg = MODES["15m" if dbg_mode.startswith("15M") else "5m"]
            sym_input = st.text_input(
                "Symbol",
                placeholder="BTC  or  BTCUSDT  or  BTC/USDT:USDT",
                value="BTC", key="sym_input"
            )
            dbg_go = st.button("&#128269;  Run Debug", type="primary", key="debug_btn",
                               use_container_width=True)

        with d_col2:
            st.markdown(
                "<div class='sc-pipeline-info'>"
                "<b>Pipeline stages checked:</b><br>"
                "<span class='sc-stage-dot dot-1'>S1</span> HLC3 pivot chain pattern<br>"
                "<span class='sc-stage-dot dot-1'>S1</span> ADX momentum in pivot window<br>"
                "<span class='sc-stage-dot dot-1'>S1</span> Pivot age gate (8h / 48h)<br>"
                "<span class='sc-stage-dot dot-2'>S2</span> TDI RSI fast/slow direction<br>"
                "<span class='sc-stage-dot dot-2'>S2</span> Keltner Channel band position<br>"
                "<span class='sc-stage-dot dot-2'>S2</span> Last-15-bar band cleanness<br>"
                "<span class='sc-stage-dot dot-3'>S3</span> BB continuation pullback<br>"
                "<span class='sc-stage-dot dot-3'>S3</span> Pine Final Signal + timestamps<br>"
                "<span class='sc-stage-dot dot-4'>S4</span> BOS/ChoCh validation (lower TF)"
                "</div>",
                unsafe_allow_html=True,
            )

        if dbg_go:
            with st.spinner(f"Running pipeline on {sym_input.strip().upper()}…"):
                try:
                    logs = _run_async(debug_single(sym_input, dbg_cfg))
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
                    logs = []

            if logs:
                rows = [{"Stage": lbl, "Status": stat, "Detail": detail}
                        for lbl, stat, detail in logs]
                df_dbg = pd.DataFrame(rows)

                def _color(val):
                    if "PASS" in val or "VALID" in val:   return "color:#00e676;font-weight:700"
                    if "FAIL" in val or "INVALID" in val: return "color:#ff4060;font-weight:700"
                    if "WAIT" in val: return "color:#ffca28;font-weight:700"
                    return ""

                st.dataframe(
                    df_dbg.style.map(_color, subset=["Status"]),
                    use_container_width=True, hide_index=True,
                    height=50 + 38 * len(rows),
                )

                last = logs[-1]
                if "INVALID" in last[1]:
                    st.error(f"Signal INVALIDATED at {last[0]} — opposite BOS appeared after signal")
                elif "WAIT" in last[1]:
                    st.warning(f"Signal WAITING — {last[0]}: no decisive BOS/ChoCh event yet")
                    st.caption(f"Detail: {last[2]}")
                elif "PASS" in last[1] or "VALID" in last[1]:
                    st.success("All stages passed — Signal confirmed with BOS/ChoCh validation")
                else:
                    st.error(f"Failed at {last[0]}")
                    st.caption(f"Detail: {last[2]}")


if __name__ == "__main__":
    main()
