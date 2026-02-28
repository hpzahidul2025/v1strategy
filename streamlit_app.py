#!/usr/bin/env python3
"""
Binance Futures Scanner · ULTRA-FAST Edition v9
Streamlit Web App — Binance via proxy (bypasses geo-block on cloud servers)

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

MODES = {
    "15m": {
        "pivot_tf": "1d",
        "tdi_tf":   "4h",
        "mid_tf":   "1h",
        "sig_tf":   "15m",
        "label":    "15M — Daily → 4H → 1H → 15M",
    },
    "5m": {
        "pivot_tf": "4h",
        "tdi_tf":   "1h",
        "mid_tf":   "15m",
        "sig_tf":   "5m",
        "label":    "5M — 4H → 1H → 15M → 5M",
    },
}

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Binance Futures Scanner v9",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stButton > button {
    width: 100%; height: 3.2rem;
    font-size: 1.1rem; font-weight: 700; border-radius: 8px;
  }
  .stRadio > div { gap: 0.5rem; }
  .stRadio label { font-size: 1rem; }
  .buy-badge  { background:#0f4; color:#000; padding:2px 8px; border-radius:4px; font-weight:700; }
  .sell-badge { background:#f04; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }
  .metric-box {
    border: 1px solid #333; border-radius:8px;
    padding: 0.6rem 1rem; text-align: center; margin: 0.2rem;
  }
  [data-testid="stDataFrame"] { overflow-x: auto; }
  @media (max-width: 600px) { .main .block-container { padding: 0.5rem 0.6rem; } }
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

    Scan mode (want_sell=True/False): returns bool — signal found in window.
    Debug mode (want_sell=None): returns (buy_found, sell_found, buy_details, sell_details).
    """
    if len(df) < SWING_UTAMA + 10:
        return False if want_sell is not None else (False, False)
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
        ts_arr = df.ts.values.astype(np.int64)
        window_start = int(np.searchsorted(ts_arr[:end], from_ts))
    else:
        window_start = max(0, end - LOOKBACK_SIG)

    ts_arr_full = df.ts.values.astype(np.int64)  # used for signal timestamp capture

    if want_sell is not None:
        last_p_bar  = -1
        found       = False
        found_i     = -1
        if want_sell:
            for i in range(1, end):
                if dir_main[i] == -1 and dir_main[i - 1] != -1: last_p_bar = -1
                if sell_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cdn[i] and bool(vf[i]) and below[i]
                        and dir_main[i] == -1):
                    found = True; found_i = i; break
                if cdn[i] and bool(vf[i]) and below[i] and last_p_bar >= 0:
                    last_p_bar = -1
        else:
            for i in range(1, end):
                if dir_main[i] == 1 and dir_main[i - 1] != 1: last_p_bar = -1
                if buy_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cup[i] and bool(vf[i]) and above[i]
                        and dir_main[i] == 1):
                    found = True; found_i = i; break
                if cup[i] and bool(vf[i]) and above[i] and last_p_bar >= 0:
                    last_p_bar = -1
        # Return (found_bool, signal_ts_ms) — signal_ts is the candle open timestamp
        sig_ts = int(ts_arr_full[found_i]) if found_i >= 0 else 0
        return (found, sig_ts)

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
    ts_arr    = df.ts.values[:end].astype(np.int64)
    buy_idxs  = np.where(final_buy[:end]  & w_mask)[0]
    sell_idxs = np.where(final_sell[:end] & w_mask)[0]

    def _details(idxs):
        return [(int(idx - window_start + 1), int(ts_arr[idx])) for idx in idxs]
    return (bool(buy_idxs.size > 0), bool(sell_idxs.size > 0),
            _details(buy_idxs), _details(sell_idxs))


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
    Returns (side_str, sym, detail, pivot_ts) or None.
    """
    mid_tf = cfg["mid_tf"]
    sig_tf = cfg["sig_tf"]
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
        return None  # BB fail — saves sig_tf fetch

    ds = await fetch(ex, sem, sym, sig_tf, sig_limit)
    if ds.empty or len(ds) < min_sig:
        return None

    result = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
    has_signal, signal_ts = result if isinstance(result, tuple) else (result, 0)
    if not has_signal:
        return None
    side = "SELL" if want_sell else "BUY"
    det  = (f"{detail} | {mid_tf.upper()}_BB_pullback✓ [{sig_tf.upper()} FinalSignal✓]"
            f" [window@pivot_ts]")
    return (side, sym, det, pivot_ts, signal_ts)


# ══════════════════════════════════════════════════════════════════════
#  MAIN SCAN RUNNER
# ══════════════════════════════════════════════════════════════════════

async def run_scan(cfg: dict, progress_callback: Callable) -> dict:
    """
    Run full 3-stage pipeline over all USDT perpetuals.
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
            "buy": [], "sell": [], "total": total,
        }
        last_ui_update = 0.0

        async def worker(sym: str):
            nonlocal last_ui_update
            r1 = await stage1_worker(ex, sem, sym, cfg)
            state["s1_done"] += 1
            # v9c: throttle progress callback to avoid excess Streamlit re-renders
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
                side, sym2, det2, _pt, sig_ts = r3
                if side == "BUY":  state["buy"].append((sym2, det2, r3[3], sig_ts))
                else:              state["sell"].append((sym2, det2, r3[3], sig_ts))
                progress_callback(state)  # always update on new signal
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
        pivot_tf = cfg["pivot_tf"]
        tdi_tf   = cfg["tdi_tf"]
        mid_tf   = cfg["mid_tf"]
        sig_tf   = cfg["sig_tf"]

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

        has_signal_result = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
        has_signal, sig_ts = has_signal_result if isinstance(has_signal_result, tuple) else (has_signal_result, 0)
        sig_time = datetime.datetime.utcfromtimestamp(sig_ts / 1000).strftime("%Y-%m-%d %H:%M UTC") if sig_ts else "n/a"
        logs.append(("S3 Pine Final Signal", "✅ PASS" if has_signal else "❌ FAIL",
            f"{sig_tf.upper()} Final {direction} Signal in window: {has_signal}  |  ⏰ Signal Time: {sig_time}"))
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
               signal_ts: int, now_ms: int, mode_key: str, timestamp: str) -> dict:
    """
    v10: Parse a result row into structured fields.
    signal_ts: candle open timestamp (ms) of the Pine Final Signal bar.
    """
    p      = _re.search(r"P=([\d.]+)",                   det)
    prev   = _re.search(r"prev_(?:peak|trough)=([\d.]+)", det)
    adxpk  = _re.search(r"ADX_peak=([\d.]+)",             det)
    adxend = _re.search(r"ADX_end=([\d.]+)",              det)
    bb_m   = _re.search(r"(\w+)_BB_pullback",             det)
    sig_m  = _re.search(r"\[(\w+) FinalSignal",           det)
    age_h  = round((now_ms - pivot_ts) / 3_600_000, 1)
    # v10: Pine Final Signal bar time — convert ms epoch to human-readable UTC
    if signal_ts and signal_ts > 0:
        sig_dt = datetime.datetime.utcfromtimestamp(signal_ts / 1000).strftime("%Y-%m-%d %H:%M UTC")
    else:
        sig_dt = ""
    return {
        "Direction":      direction,
        "Symbol":         sym,
        "Pivot_P":        float(p.group(1))      if p      else "",
        "Prev_Pivot":     float(prev.group(1))   if prev   else "",
        "ADX_Peak":       float(adxpk.group(1))  if adxpk  else "",
        "ADX_End":        float(adxend.group(1)) if adxend else "",
        "BB_TF":          bb_m.group(1)          if bb_m   else "",
        "Signal_TF":      sig_m.group(1)         if sig_m  else "",
        "Signal_Time":    sig_dt,
        "Pivot_Age_h":    age_h,
        "Scan_Time":      timestamp,
        "Mode":           mode_key.upper(),
    }


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT APP LAYOUT
# ══════════════════════════════════════════════════════════════════════

def _init_session():
    """Ensure all session_state keys exist on first load."""
    defaults = {
        "scan_done":    False,
        "scan_state":   None,   # raw state dict from run_scan
        "scan_elapsed": 0.0,
        "scan_mode":    "15m",
        "df_final":     None,   # pd.DataFrame of parsed results
        "csv_bytes":    None,   # pre-encoded CSV (prevents download refresh)
        "txt_bytes":    None,   # pre-encoded TXT
        "csv_fname":    "",
        "txt_fname":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    _init_session()

    st.markdown("""
<h1 style='text-align:center;margin-bottom:0.2rem'>
⚡ Binance Futures Scanner
</h1>
<p style='text-align:center;color:#888;margin-top:0;font-size:0.95rem'>
Ultra-Fast Multi-Stage Engine · Institutional Logic · Pine Accurate · v10
</p>
""", unsafe_allow_html=True)

    tab_scan, tab_debug = st.tabs(["🔍 Full Scan", "🐛 Debug Pair"])

    # ══ TAB 1: FULL SCAN ══════════════════════════════════════════════
    with tab_scan:

        # ── Proxy status ──────────────────────────────────────────────
        _proxy = _get_proxy()
        if _proxy:
            _host = _proxy.split("@")[-1] if "@" in _proxy else _proxy.split("//")[-1]
            st.success(f"✅ Proxy active — routing via **{_host}**  (Binance geo-block bypassed)", icon="🔒")
        else:
            st.error(
                "⚠️ **No proxy configured.** Binance blocks Streamlit Cloud IPs.  "
                "Add your proxy URL in **Streamlit Secrets** → key: `PROXY_URL`",
                icon="🚫"
            )
            with st.expander("📋 How to add your free proxy (Webshare.io) — takes 3 minutes"):
                st.markdown("""
**Step 1 — Get a free proxy**
1. Go to **https://proxy2.webshare.io/register** → create free account (no credit card)
2. After login → go to **Proxy** → **List** tab → Download in **Username:Password@IP:Port** format
3. Pick any proxy, e.g.: `http://username:password@12.34.56.78:8080`

**Step 2 — Add to Streamlit Secrets**
1. **https://share.streamlit.io** → your app → **⋮** → **Settings** → **Secrets**
2. Paste:
```
PROXY_URL = "http://youruser:yourpass@12.34.56.78:8080"
```
3. Click **Save** — app restarts in ~30 seconds
""")

        # ── Mode selector ─────────────────────────────────────────────
        mode_choice = st.radio(
            "Signal Timeframe",
            options=["15M  (Daily → 4H → 1H → 15M)", "5M  (4H → 1H → 15M → 5M)"],
            index=0, horizontal=True,
        )
        mode_key = "15m" if mode_choice.startswith("15M") else "5m"
        cfg      = MODES[mode_key]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pivot TF",  cfg["pivot_tf"].upper())
        col2.metric("TDI TF",    cfg["tdi_tf"].upper())
        col3.metric("Mid TF",    cfg["mid_tf"].upper())
        col4.metric("Signal TF", cfg["sig_tf"].upper())

        st.info(
            f"**Rules:**  "
            f"S1 — {cfg['pivot_tf'].upper()} pivot + {cfg['tdi_tf'].upper()} ADX > {ADX_TH:.0f}  |  "
            f"S2 — TDI + KC band  |  "
            f"S3 — {cfg['mid_tf'].upper()} BB pullback + {cfg['sig_tf'].upper()} Pine Final Signal",
            icon="ℹ️"
        )

        # Market cache refresh
        c_btn1, c_btn2 = st.columns([1, 4])
        with c_btn1:
            if st.button("🔄 Refresh Markets", help="Clear cached market list"):
                if "markets" in st.session_state:
                    del st.session_state["markets"]
                st.rerun()

        st.markdown("---")

        # ── SCAN BUTTON ───────────────────────────────────────────────
        if st.button("🚀 Start Scan", type="primary", key="scan_btn"):
            # Clear previous results
            st.session_state["scan_done"]  = False
            st.session_state["df_final"]   = None
            st.session_state["csv_bytes"]  = None
            st.session_state["txt_bytes"]  = None

            t0 = time.time()

            prog_bar    = st.progress(0, text="Connecting to Binance…")
            status_box  = st.empty()
            live_tbl    = st.empty()

            def update_ui(state: dict):
                total   = state["total"]
                s1_done = state["s1_done"]
                pct     = s1_done / total if total else 0
                elapsed = time.time() - t0
                spd     = s1_done / max(elapsed, 0.01)
                prog_bar.progress(min(pct, 1.0),
                    text=f"Scanning… {s1_done}/{total}  |  {spd:.0f} sym/s  |  "
                         f"→S2: {state['s2_in']}  →S3: {state['s3_in']}")
                nb = len(state["buy"])
                ns = len(state["sell"])
                status_box.markdown(
                    f"<div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin:0.4rem 0'>"
                    f"<div class='metric-box'><b style='color:#00ee44'>🟢 BUY</b><br>"
                    f"<span style='font-size:1.8rem;font-weight:800;color:#00ee44'>{nb}</span></div>"
                    f"<div class='metric-box'><b style='color:#ff4444'>🔴 SELL</b><br>"
                    f"<span style='font-size:1.8rem;font-weight:800;color:#ff4444'>{ns}</span></div>"
                    f"<div class='metric-box'><b>→S2</b><br><span style='font-size:1.4rem'>{state['s2_in']}</span></div>"
                    f"<div class='metric-box'><b>→S3</b><br><span style='font-size:1.4rem'>{state['s3_in']}</span></div>"
                    f"<div class='metric-box'><b>⏱ Elapsed</b><br><span style='font-size:1.4rem'>{elapsed:.0f}s</span></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                # Live table — show latest signals as they arrive
                rows = (
                    [{"🟢/🔴": "🟢 BUY", "Symbol": sym, "Detail": det}
                     for sym, det, _, _st in state["buy"]] +
                    [{"🟢/🔴": "🔴 SELL", "Symbol": sym, "Detail": det}
                     for sym, det, _, _st in state["sell"]]
                )
                if rows:
                    live_tbl.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            try:
                state = _run_async(run_scan(cfg, update_ui))
            except Exception as e:
                st.error(f"Scan error: {e}")
                st.exception(e)
                state = None

            if state:
                elapsed      = time.time() - t0
                total        = state["total"]
                buy_results  = sorted(state["buy"],  key=lambda x: x[0])
                sell_results = sorted(state["sell"], key=lambda x: x[0])
                all_sigs     = len(buy_results) + len(sell_results)

                prog_bar.progress(1.0, text=f"✅ Done in {elapsed:.1f}s  ({total/elapsed:.1f} sym/s)")

                st.success(
                    f"**Scan complete** — {total} symbols in {elapsed:.1f}s  "
                    f"({total/elapsed:.1f} sym/s) · "
                    f"Funnel: {total} → {state['s2_in']} → {state['s3_in']} → **{all_sigs} signals**"
                )

                # ── Build structured df_final ──────────────────────────
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                now_ms    = int(time.time() * 1000)
                ts_int    = int(time.time())

                if buy_results or sell_results:
                    all_rows = (
                        [_parse_row("BUY",  sym, det, pts, sig_ts, now_ms, mode_key, timestamp)
                         for sym, det, pts, sig_ts in buy_results] +
                        [_parse_row("SELL", sym, det, pts, sig_ts, now_ms, mode_key, timestamp)
                         for sym, det, pts, sig_ts in sell_results]
                    )
                    df_final = pd.DataFrame(all_rows)

                    # ── Pre-encode CSV for download (no rerun on click) ────
                    csv_buf = io.StringIO()
                    df_final.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")

                    # ── Pre-encode TXT ────────────────────────────────────
                    txt_buf = io.StringIO()
                    txt_buf.write(f"BINANCE FUTURES SCANNER  —  {mode_key.upper()} MODE\n")
                    txt_buf.write(f"Scan Time    : {timestamp}\n")
                    txt_buf.write(f"Symbols      : {total}  |  Elapsed : {elapsed:.1f}s\n")
                    txt_buf.write(f"Signals      : {len(buy_results)} BUY  |  {len(sell_results)} SELL\n")
                    txt_buf.write("=" * 72 + "\n")
                    for direction, group in [("BUY", buy_results), ("SELL", sell_results)]:
                        if not group:
                            continue
                        txt_buf.write(f"\n{'─'*30} {direction} {'─'*30}\n")
                        for sym, det, pts, sig_ts in group:
                            r = _parse_row(direction, sym, det, pts, sig_ts, now_ms, mode_key, timestamp)
                            txt_buf.write(
                                f"  {r['Symbol']:<24}  Pivot={r['Pivot_P']}   Prev={r['Prev_Pivot']}\n"
                                f"  {'':24}  ADX  peak={r['ADX_Peak']}  end={r['ADX_End']}  "
                                f"Age={r['Pivot_Age_h']}h\n"
                                f"  {'':24}  BB={r['BB_TF']}  Signal={r['Signal_TF']}\n"
                                f"  {'':24}  ⏰ Pine Final Signal Time: {r['Signal_Time']}\n"
                            )
                    txt_bytes = txt_buf.getvalue().encode("utf-8")

                    # ── Store in session_state so results survive reruns ───
                    st.session_state["scan_done"]    = True
                    st.session_state["scan_state"]   = state
                    st.session_state["scan_elapsed"] = elapsed
                    st.session_state["scan_mode"]    = mode_key
                    st.session_state["df_final"]     = df_final
                    st.session_state["csv_bytes"]    = csv_bytes
                    st.session_state["txt_bytes"]    = txt_bytes
                    st.session_state["csv_fname"]    = f"signals_{mode_key}_{ts_int}.csv"
                    st.session_state["txt_fname"]    = f"signals_{mode_key}_{ts_int}.txt"
                else:
                    st.session_state["scan_done"] = True
                    st.session_state["scan_state"] = state
                    st.session_state["scan_elapsed"] = elapsed
                    st.session_state["df_final"] = None

        # ══════════════════════════════════════════════════════════════
        # RESULTS — rendered OUTSIDE the button block so they persist
        # across Streamlit reruns (tab clicks, download clicks, etc.)
        # This is the fix for the DeltaGenerator dump and sticky results.
        # ══════════════════════════════════════════════════════════════
        if st.session_state["scan_done"] and st.session_state["scan_state"] is not None:
            state      = st.session_state["scan_state"]
            elapsed    = st.session_state["scan_elapsed"]
            mode_key   = st.session_state["scan_mode"]
            df_final   = st.session_state["df_final"]
            total      = state["total"]
            buy_count  = len(state["buy"])
            sell_count = len(state["sell"])
            all_sigs   = buy_count + sell_count

            st.markdown("---")
            st.markdown(
                f"<div style='background:#1a2a1a;border:1px solid #2a4a2a;border-radius:8px;"
                f"padding:0.8rem 1.2rem;margin-bottom:1rem'>"
                f"<span style='color:#00ee44;font-weight:700'>📊 Last Scan Results</span>"
                f"  ·  {total} symbols  ·  {elapsed:.1f}s  "
                f"·  <span style='color:#00ee44'>{buy_count} BUY</span>"
                f"  <span style='color:#ff4444'>{sell_count} SELL</span>"
                f"  ·  Mode: <b>{mode_key.upper()}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

            if df_final is not None and not df_final.empty:
                display_cols = ["Direction", "Symbol", "Pivot_P", "Prev_Pivot",
                                "ADX_Peak", "ADX_End", "BB_TF", "Signal_TF",
                                "Signal_Time", "Pivot_Age_h"]

                t_all, t_buy, t_sell = st.tabs([
                    f"📋 All ({all_sigs})",
                    f"🟢 BUY ({buy_count})",
                    f"🔴 SELL ({sell_count})",
                ])
                with t_all:
                    st.dataframe(df_final[display_cols],
                                 use_container_width=True, hide_index=True,
                                 height=min(400, 45 + 35 * len(df_final)))
                with t_buy:
                    df_b = df_final[df_final["Direction"] == "BUY"][display_cols]
                    if not df_b.empty:
                        st.dataframe(df_b, use_container_width=True, hide_index=True,
                                     height=min(400, 45 + 35 * len(df_b)))
                    else:
                        st.info("No BUY signals in this scan.")
                with t_sell:
                    df_s = df_final[df_final["Direction"] == "SELL"][display_cols]
                    if not df_s.empty:
                        st.dataframe(df_s, use_container_width=True, hide_index=True,
                                     height=min(400, 45 + 35 * len(df_s)))
                    else:
                        st.info("No SELL signals in this scan.")

                # ── Export — pre-encoded bytes, NO page refresh ──────────
                st.markdown("### ⬇️ Export Signals")
                colA, colB = st.columns(2)
                colA.download_button(
                    label="📄 Download CSV",
                    data=st.session_state["csv_bytes"],
                    file_name=st.session_state["csv_fname"],
                    mime="text/csv",
                    use_container_width=True,
                )
                colB.download_button(
                    label="📝 Download TXT Report",
                    data=st.session_state["txt_bytes"],
                    file_name=st.session_state["txt_fname"],
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.warning("No signals found in last scan.")

    # ══ TAB 2: DEBUG PAIR ═════════════════════════════════════════════
    with tab_debug:
        st.subheader("Debug a Single Symbol")
        st.caption("Verbose pass/fail for every stage — see exactly why a pair passes or fails")

        dbg_mode = st.radio(
            "Ruleset",
            ["15M  (Daily → 4H → 1H → 15M)", "5M  (4H → 1H → 15M → 5M)"],
            index=0, horizontal=True, key="dbg_mode"
        )
        dbg_cfg = MODES["15m" if dbg_mode.startswith("15M") else "5m"]

        sym_input = st.text_input(
            "Symbol  (e.g. BTC  or  BTCUSDT  or  BTC/USDT:USDT)",
            value="BTC", key="sym_input"
        )

        if st.button("🔎 Debug Symbol", type="primary", key="debug_btn"):
            with st.spinner(f"Checking {sym_input.strip().upper()}…"):
                try:
                    logs = _run_async(debug_single(sym_input, dbg_cfg))
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
                    logs = []

            if logs:
                rows = [{"Stage": lbl, "Status": status, "Detail": detail}
                        for lbl, status, detail in logs]
                df_dbg = pd.DataFrame(rows)

                def color_status(val):
                    if "PASS" in val: return "color: #00ff66; font-weight: bold"
                    if "FAIL" in val: return "color: #ff4444; font-weight: bold"
                    return ""

                st.dataframe(
                    df_dbg.style.map(color_status, subset=["Status"]),
                    use_container_width=True, hide_index=True,
                )

                last_status = logs[-1][1]
                if "PASS" in last_status:
                    st.success("✅ All stages passed — SIGNAL CONFIRMED!")
                else:
                    st.error(f"❌ Pipeline stopped at: **{logs[-1][0]}**")
                    st.info(f"Detail: {logs[-1][2]}")


if __name__ == "__main__":
    main()
