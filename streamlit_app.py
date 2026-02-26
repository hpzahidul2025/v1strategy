#!/usr/bin/env python3
"""
Binance Futures Scanner · ULTRA-FAST Edition v5
Streamlit Web App — Binance via proxy (bypasses geo-block on cloud servers)

Fix v5: Pine Final Signal — added dir_main trend filter.
  SELL signal only accepted when dir_main == -1 (bearish swing utama).
  BUY  signal only accepted when dir_main ==  1 (bullish swing utama).
  Matches TradingView: dots are hidden when price is on the wrong side
  of the Swing Utama (TSL-50) main trend line.
"""

import streamlit as st
import asyncio
import time
import sys
import os

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

# ── Proxy helper ──────────────────────────────────────────────────────
# Reads PROXY_URL from Streamlit secrets or env var.
# Format expected:  http://username:password@host:port
def _get_proxy() -> str:
    try:
        return st.secrets["PROXY_URL"]
    except Exception:
        return os.environ.get("PROXY_URL", "")

def _make_exchange():
    """Return a configured binanceusdm exchange, with proxy if available."""
    proxy = _get_proxy()
    cfg = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    return ccxt_async.binanceusdm(cfg)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — wide layout, dark theme, mobile-friendly
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Binance Futures Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mobile-friendly CSS ──────────────────────────────────────────────
st.markdown("""
<style>
  /* Larger tap targets and readable font sizes on small screens */
  .stButton > button {
    width: 100%;
    height: 3.2rem;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 8px;
  }
  .stRadio > div { gap: 0.5rem; }
  .stRadio label { font-size: 1rem; }
  .buy-badge  { background:#0f4; color:#000; padding:2px 8px; border-radius:4px; font-weight:700; }
  .sell-badge { background:#f04; color:#fff; padding:2px 8px; border-radius:4px; font-weight:700; }
  .metric-box {
    border: 1px solid #333;
    border-radius:8px;
    padding: 0.6rem 1rem;
    text-align: center;
    margin: 0.2rem;
  }
  /* Make dataframes scroll on mobile */
  [data-testid="stDataFrame"] { overflow-x: auto; }
  /* Tighter spacing on mobile */
  @media (max-width: 600px) {
    .main .block-container { padding: 0.5rem 0.6rem; }
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
MAX_CONCURRENT = 150

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
#  INDICATOR MATH  — NumPy-vectorized (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════

def _rma(a: np.ndarray, p: int) -> np.ndarray:
    if len(a) < p:
        return np.full(len(a), np.nan)
    return pd.Series(a).ewm(alpha=1.0 / p, adjust=False, ignore_na=False).mean().values

def _sma(a, p): return pd.Series(a).rolling(p, min_periods=p).mean().values
def _ema(a, p): return pd.Series(a).ewm(span=p, adjust=False).mean().values

def calc_rsi(c, p):
    d = np.diff(c, prepend=c[0])
    g = _rma(np.where(d > 0,  d,  0.0), p)
    l = _rma(np.where(d < 0, -d,  0.0), p)
    l = np.where(l == 0, 1e-9, l)
    return 100.0 - 100.0 / (1.0 + g / l)

def calc_atr(h, l, c, p):
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return _rma(tr, p)

def calc_kc(h, l, c):
    b  = _sma(c, KC_LEN)
    at = calc_atr(h, l, c, KC_ATR_LEN)
    return b + KC_MULT * at, b - KC_MULT * at

def calc_adx(h, l, c, p=ADX_LEN):
    h_ = np.array(h, dtype=float)
    l_ = np.array(l, dtype=float)
    c_ = np.array(c, dtype=float)
    tr  = np.maximum(h_[1:] - l_[1:],
          np.maximum(np.abs(h_[1:] - c_[:-1]),
                     np.abs(l_[1:] - c_[:-1])))
    tr  = np.concatenate([[h_[0] - l_[0]], tr])
    dmp = np.where(
        (h_[1:] - h_[:-1]) > (l_[:-1] - l_[1:]),
        np.maximum(h_[1:] - h_[:-1], 0.0), 0.0)
    dmp = np.concatenate([[0.0], dmp])
    dmm = np.where(
        (l_[:-1] - l_[1:]) > (h_[1:] - h_[:-1]),
        np.maximum(l_[:-1] - l_[1:], 0.0), 0.0)
    dmm = np.concatenate([[0.0], dmm])
    s_tr  = _rma(tr,  p)
    s_dmp = _rma(dmp, p)
    s_dmm = _rma(dmm, p)
    s_tr  = np.where(s_tr == 0, 1e-9, s_tr)
    dip   = s_dmp / s_tr * 100.0
    dim   = s_dmm / s_tr * 100.0
    denom = np.where((dip + dim) == 0, 1e-9, dip + dim)
    dx    = np.abs(dip - dim) / denom * 100.0
    return _sma(dx, p)

def tdi_state(c):
    r  = calc_rsi(c, TDI_RSI_P)
    fm = _sma(r, TDI_FAST)
    sm = _sma(r, TDI_SLOW)
    return bool(fm[-1] < sm[-1]), bool(fm[-1] > sm[-1])

def pivot_chain(df):
    def _pivot(row):
        return float((row.high + row.low + row.close) / 3.0)
    return _pivot(df.iloc[-2]), _pivot(df.iloc[-3]), _pivot(df.iloc[-4]), _pivot(df.iloc[-5])

def f_swing(h, l, c, no):
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

def calc_wt2(h, l, c, v):
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
                         length: int = BB_LEN, mult: float = BB_MULT):
    n     = len(c)
    basis = _sma(c, length)
    dev   = mult * pd.Series(c).rolling(length, min_periods=length).std(ddof=0).values
    upper = basis + dev
    lower = basis - dev
    sig   = np.zeros(n, dtype=bool)
    rule1_met   = False
    band_broken = False
    armed       = False
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

def signals_tf(df, from_ts: int = 0, want_sell: bool = None):
    if len(df) < max(SWING_UTAMA + 10, 120):
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

    if want_sell is not None:
        last_p_bar = -1
        found      = False
        if want_sell:
            for i in range(1, end):
                if dir_main[i] == -1 and dir_main[i - 1] != -1: last_p_bar = -1
                if sell_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cdn[i] and bool(vf[i]) and below[i]
                        and dir_main[i] == -1):  # Pine filter: only show SELL when swing main trend is bearish
                    found = True; break
                if (cdn[i] and bool(vf[i]) and below[i] and last_p_bar >= 0):
                    last_p_bar = -1
        else:
            for i in range(1, end):
                if dir_main[i] == 1 and dir_main[i - 1] != 1: last_p_bar = -1
                if buy_pressure[i]: last_p_bar = i
                if (i >= window_start and last_p_bar >= 0 and cup[i] and bool(vf[i]) and above[i]
                        and dir_main[i] == 1):   # Pine filter: only show BUY when swing main trend is bullish
                    found = True; break
                if (cup[i] and bool(vf[i]) and above[i] and last_p_bar >= 0):
                    last_p_bar = -1
        return found

    # Debug mode (both sides)
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
    w_mask = np.zeros(end, bool)
    w_mask[window_start:] = True
    ts_arr    = df.ts.values[:end].astype(np.int64)
    buy_idxs  = np.where(final_buy[:end]  & w_mask)[0]
    sell_idxs = np.where(final_sell[:end] & w_mask)[0]
    def _details(idxs):
        return [(int(idx - window_start + 1), int(ts_arr[idx])) for idx in idxs]
    return (bool(buy_idxs.size > 0), bool(sell_idxs.size > 0),
            _details(buy_idxs), _details(sell_idxs))


# ══════════════════════════════════════════════════════════════════════
#  ASYNC FETCH
# ══════════════════════════════════════════════════════════════════════

async def fetch(ex, sem, sym, tf, limit):
    async with sem:
        try:
            raw = await ex.fetch_ohlcv(sym, tf, limit=limit)
            if not raw:
                return pd.DataFrame()
            arr = np.array(raw, dtype=float)
            df  = pd.DataFrame({
                "ts":     arr[:, 0].astype(np.int64),
                "high":   arr[:, 2],
                "low":    arr[:, 3],
                "close":  arr[:, 4],
                "volume": arr[:, 5],
            })
            return df
        except Exception:
            return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
#  SCAN STAGES
# ══════════════════════════════════════════════════════════════════════

async def stage1_worker(ex, sem, sym, cfg):
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]
    dp = await fetch(ex, sem, sym, pivot_tf, 7)
    if dp.empty or len(dp) < 5: return None
    pivot_ts = int(dp.iloc[-2]["ts"])
    cur_P, prev_P, pp_P, ppp_P = pivot_chain(dp)
    sell_pivot = cur_P < prev_P and prev_P > max(pp_P, ppp_P)
    buy_pivot  = cur_P > prev_P and prev_P < min(pp_P, ppp_P)
    if not sell_pivot and not buy_pivot: return None
    da = await fetch(ex, sem, sym, tdi_tf, 150)
    if da.empty or len(da) < ADX_LEN * 2: return None
    adx_arr = calc_adx(da.high.values, da.low.values, da.close.values)
    pp_P_ts  = int(dp.iloc[-4]["ts"])
    piv4_ts  = int(dp.iloc[-1]["ts"])
    ts_vals      = da["ts"].values.astype(np.int64)
    window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= piv4_ts)
    adx_window   = adx_arr[window_mask]
    valid_window = adx_window[~np.isnan(adx_window)]
    if len(valid_window) == 0: return None
    adx_ever_above    = bool(np.any(valid_window > ADX_TH))
    adx_at_window_end = float(valid_window[-1])
    adx_end_above     = adx_at_window_end > ADX_TH
    if not (adx_ever_above and adx_end_above): return None
    adx_peak  = float(np.nanmax(valid_window))
    direction = "SELL_S1" if sell_pivot else "BUY_S1"
    det = (f"P={cur_P:.5f} "
           f"{'prev_peak' if sell_pivot else 'prev_trough'}={prev_P:.5f} "
           f"ADX_peak={adx_peak:.1f} ADX_end={adx_at_window_end:.1f}")
    return (direction, sym, det, pivot_ts, da)


def stage2_worker(sym, direction, detail, pivot_ts, da):
    if da.empty or len(da) < 60: return None
    bear_tdi, bull_tdi = tdi_state(da.close.values)
    u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
    c_t                = float(da.close.iloc[-1])
    n_t = len(da)
    s15 = max(0, n_t - 16)
    e15 = n_t - 1
    sell_band_clean = bool(np.all(da.low.values[s15:e15]  > l_t[s15:e15]))
    buy_band_clean  = bool(np.all(da.high.values[s15:e15] < u_t[s15:e15]))
    if direction == "SELL_S1" and bear_tdi and c_t > l_t[-1] and sell_band_clean:
        return (direction, sym, detail, pivot_ts)
    if direction == "BUY_S1"  and bull_tdi and c_t < u_t[-1] and buy_band_clean:
        return (direction, sym, detail, pivot_ts)
    return None


async def stage3_worker(ex, sem, sym, direction, detail, pivot_ts, cfg):
    mid_tf = cfg["mid_tf"]
    sig_tf = cfg["sig_tf"]
    is_5m_mode = sig_tf == "5m"
    sig_limit  = 350 if is_5m_mode else 120
    mid_limit  = 120 if is_5m_mode else 80
    min_sig    = 300 if is_5m_mode else 120
    dm, ds = await asyncio.gather(
        fetch(ex, sem, sym, mid_tf, mid_limit),
        fetch(ex, sem, sym, sig_tf, sig_limit),
    )
    if dm.empty or len(dm) < BB_LEN + 10: return None
    if ds.empty or len(ds) < min_sig: return None
    want_sell = (direction == "SELL_S1")
    end      = len(dm) - 1
    bb_sig   = calc_bb_continuation(
        dm.close.values[:end], dm.high.values[:end], dm.low.values[:end],
        want_sell=want_sell)
    ts_mid   = dm.ts.values[:end].astype(np.int64)
    win_mask = ts_mid >= pivot_ts
    if not bb_sig[win_mask].any(): return None
    has_signal = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
    if not has_signal: return None
    side = "SELL" if want_sell else "BUY"
    det  = (f"{detail} | {mid_tf.upper()}_BB_pullback✓ [{sig_tf.upper()} FinalSignal✓]"
            f" [window@pivot_ts]")
    return (side, sym, det, pivot_ts)


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT UI — main scan with live progress
# ══════════════════════════════════════════════════════════════════════

async def run_scan(cfg, progress_callback):
    """Run full pipeline; calls progress_callback(s1_done, total, s2_in, s3_in, results_so_far)."""
    ex = _make_exchange()
    try:
        await ex.load_markets()
        symbols = sorted([
            s for s, m in ex.markets.items()
            if m.get("type") == "swap" and m.get("active")
            and m.get("quote") == "USDT" and ":USDT" in s
        ])
        total = len(symbols)
        sem   = asyncio.Semaphore(MAX_CONCURRENT)

        # Shared state
        state = {
            "s1_done": 0, "s2_in": 0, "s3_in": 0,
            "buy": [], "sell": [], "total": total,
        }

        async def worker(sym):
            # Stage 1
            r1 = await stage1_worker(ex, sem, sym, cfg)
            state["s1_done"] += 1
            if r1 is None:
                return None
            direction, sym, detail, pivot_ts, da = r1
            state["s2_in"] += 1
            # Stage 2
            r2 = stage2_worker(sym, direction, detail, pivot_ts, da)
            if r2 is None:
                return None
            direction, sym, detail, pivot_ts = r2
            state["s3_in"] += 1
            # Stage 3
            r3 = await stage3_worker(ex, sem, sym, direction, detail, pivot_ts, cfg)
            if r3:
                side, sym2, det2, _ = r3
                if side == "BUY":  state["buy"].append((sym2, det2))
                else:              state["sell"].append((sym2, det2))
            return r3

        # Batch into chunks so we can update progress
        CHUNK = 50
        for i in range(0, total, CHUNK):
            batch = symbols[i:i + CHUNK]
            await asyncio.gather(*[worker(s) for s in batch])
            progress_callback(state)

        return state
    finally:
        await ex.close()


async def debug_single(sym_raw, cfg):
    """Debug a single symbol; returns list of (label, status, detail) tuples."""
    raw = sym_raw.strip().upper().replace(" ", "")
    raw_clean = raw.replace("/", "").replace(":", "")
    base = raw_clean.replace("USDT", "") or raw_clean
    sym  = f"{base}/USDT:USDT"

    logs = []

    ex = _make_exchange()
    try:
        await ex.load_markets()
        if sym not in ex.markets:
            logs.append(("Symbol", "❌ FAIL", f"'{sym}' not found on Binance Futures"))
            return logs

        pivot_tf = cfg["pivot_tf"]
        tdi_tf   = cfg["tdi_tf"]
        mid_tf   = cfg["mid_tf"]
        sig_tf   = cfg["sig_tf"]
        sem = asyncio.Semaphore(10)

        dp, da = await asyncio.gather(
            fetch(ex, sem, sym, pivot_tf, 7),
            fetch(ex, sem, sym, tdi_tf,  150),
        )

        if dp.empty or len(dp) < 5:
            logs.append(("S1 Pivot data", "❌ FAIL", f"Not enough {pivot_tf} candles")); return logs
        if da.empty or len(da) < ADX_LEN * 2:
            logs.append(("S1 ADX data",   "❌ FAIL", f"Not enough {tdi_tf} candles"));  return logs

        pivot_ts = int(dp.iloc[-2]["ts"])
        cur_P, prev_P, pp_P, ppp_P = pivot_chain(dp)
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

        adx_arr = calc_adx(da.high.values, da.low.values, da.close.values)
        pp_P_ts      = int(dp.iloc[-4]["ts"])
        piv4_ts      = int(dp.iloc[-1]["ts"])
        ts_vals      = da["ts"].values.astype(np.int64)
        window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= piv4_ts)
        adx_window   = adx_arr[window_mask]
        valid_window = adx_window[~np.isnan(adx_window)]

        if len(valid_window) == 0:
            logs.append(("S1 ADX", "❌ FAIL", "No ADX candles in pivot window")); return logs

        adx_ever_above    = bool(np.any(valid_window > ADX_TH))
        adx_at_window_end = float(valid_window[-1])
        adx_end_above     = adx_at_window_end > ADX_TH
        adx_peak          = float(np.nanmax(valid_window))

        if adx_ever_above and adx_end_above:
            logs.append(("S1 ADX", "✅ PASS",
                f"peak={adx_peak:.1f} end={adx_at_window_end:.1f} > {ADX_TH}"))
        elif not adx_ever_above:
            logs.append(("S1 ADX", "❌ FAIL",
                f"Never above {ADX_TH} | peak={adx_peak:.1f}")); return logs
        else:
            logs.append(("S1 ADX", "❌ FAIL",
                f"Was above {ADX_TH} but dropped | end={adx_at_window_end:.1f}")); return logs

        # Stage 2
        bear_tdi, bull_tdi = tdi_state(da.close.values)
        u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
        c_t                = float(da.close.iloc[-1])
        n_t = len(da); s15 = max(0, n_t - 16); e15 = n_t - 1
        sell_band_clean = bool(np.all(da.low.values[s15:e15]  > l_t[s15:e15]))
        buy_band_clean  = bool(np.all(da.high.values[s15:e15] < u_t[s15:e15]))

        tdi_ok   = (direction == "SELL" and bear_tdi) or (direction == "BUY" and bull_tdi)
        kc_ok    = (direction == "SELL" and c_t > l_t[-1]) or (direction == "BUY" and c_t < u_t[-1])
        band_ok  = sell_band_clean if direction == "SELL" else buy_band_clean

        logs.append(("S2 TDI", "✅ PASS" if tdi_ok else "❌ FAIL",
            f"bear={bear_tdi} bull={bull_tdi} → need {'bear' if direction=='SELL' else 'bull'}"))
        if not tdi_ok: return logs

        logs.append(("S2 KC Band", "✅ PASS" if kc_ok else "❌ FAIL",
            f"close={c_t:.5f} {'>' if direction=='SELL' else '<'} KC {'lower' if direction=='SELL' else 'upper'}"))
        if not kc_ok: return logs

        logs.append(("S2 Band Clean", "✅ PASS" if band_ok else "❌ FAIL",
            f"Last 15 {'lows > KC lower' if direction=='SELL' else 'highs < KC upper'}: {band_ok}"))
        if not band_ok: return logs

        # Stage 3
        is_5m_mode = sig_tf == "5m"
        sig_limit  = 350 if is_5m_mode else 120
        mid_limit  = 120 if is_5m_mode else 80
        min_sig    = 300 if is_5m_mode else 120

        dm, ds = await asyncio.gather(
            fetch(ex, sem, sym, mid_tf, mid_limit),
            fetch(ex, sem, sym, sig_tf, sig_limit),
        )

        if dm.empty or len(dm) < BB_LEN + 10:
            logs.append(("S3 BB data", "❌ FAIL", f"Not enough {mid_tf} candles")); return logs
        if ds.empty or len(ds) < min_sig:
            logs.append(("S3 Sig data", "❌ FAIL", f"Not enough {sig_tf} candles")); return logs

        want_sell = direction == "SELL"
        end       = len(dm) - 1
        bb_sig    = calc_bb_continuation(
            dm.close.values[:end], dm.high.values[:end], dm.low.values[:end],
            want_sell=want_sell)
        ts_mid    = dm.ts.values[:end].astype(np.int64)
        win_mask  = ts_mid >= pivot_ts
        bb_ok     = bb_sig[win_mask].any()
        logs.append(("S3 BB Pullback", "✅ PASS" if bb_ok else "❌ FAIL",
            f"{int(bb_sig[win_mask].sum())} BB {direction} signal(s) in pivot window"))
        if not bb_ok: return logs

        has_signal = signals_tf(ds, from_ts=pivot_ts, want_sell=want_sell)
        logs.append(("S3 Pine Final Signal", "✅ PASS" if has_signal else "❌ FAIL",
            f"{sig_tf.upper()} Final {direction} Signal in pivot_ts window: {has_signal}"))

        return logs
    finally:
        await ex.close()


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT APP LAYOUT
# ══════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
<h1 style='text-align:center;margin-bottom:0'>
⚡ Binance Futures Scanner
</h1>
<p style='text-align:center;color:gray;margin-top:0'>
Ultra-Fast Multi-Stage Engine · Institutional Logic · Pine Accurate
</p>
""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_scan, tab_debug = st.tabs(["🔍 Full Scan", "🐛 Debug Pair"])

    # ══ TAB 1: FULL SCAN ══════════════════════════════════════════════
    with tab_scan:
        st.subheader("Scan all USDT Perpetuals")

        # ── Proxy status banner ───────────────────────────────────────
        _proxy = _get_proxy()
        if _proxy:
            _host = _proxy.split("@")[-1] if "@" in _proxy else _proxy.split("//")[-1]
            st.success(f"✅ Proxy active — routing via **{_host}**  (Binance geo-block bypassed)", icon="🔒")
        else:
            st.error(
                "⚠️ **No proxy configured.** Binance blocks Streamlit Cloud IPs.  "
                "Add your proxy URL in **Streamlit Secrets** → key: `PROXY_URL`  "
                "See the setup guide below ↓",
                icon="🚫"
            )
            with st.expander("📋 How to add your free proxy (Webshare.io) — takes 3 minutes"):
                st.markdown("""
**Step 1 — Get a free proxy**
1. Go to **https://proxy2.webshare.io/register** → create free account (no credit card)
2. After login → go to **Proxy** → **List** tab
3. You'll see 10 free proxies. Click **Download** → choose **Username:Password@IP:Port** format
4. Pick any one proxy from the list — it looks like:  
   `http://username:password@12.34.56.78:8080`

**Step 2 — Add to Streamlit Secrets**
1. Go to **https://share.streamlit.io** → click your app → **⋮ menu** → **Settings**
2. Click **Secrets** tab
3. Paste exactly this (replace with your real proxy values):
```
PROXY_URL = "http://youruser:yourpass@12.34.56.78:8080"
```
4. Click **Save** — app auto-restarts in ~30 seconds

**Step 3 — Scan!**
Click **Start Scan** — it will now connect to Binance through your proxy ✅
""")

        mode_choice = st.radio(
            "Signal Timeframe",
            options=["15M  (Daily → 4H → 1H → 15M)", "5M  (4H → 1H → 15M → 5M)"],
            index=0,
            horizontal=True,
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

        if st.button("🚀 Start Scan", type="primary", key="scan_btn"):
            t0 = time.time()

            # Progress area
            prog_bar    = st.progress(0, text="Connecting to Binance…")
            status_row  = st.empty()
            results_ph  = st.empty()
            summary_ph  = st.empty()

            buy_results  = []
            sell_results = []

            def update_ui(state):
                total    = state["total"]
                s1_done  = state["s1_done"]
                pct      = s1_done / total if total else 0
                elapsed  = time.time() - t0
                spd      = s1_done / max(elapsed, 0.01)
                prog_bar.progress(pct,
                    text=f"Scanning… {s1_done}/{total} symbols  |  "
                         f"{spd:.0f}/s  |  →S2: {state['s2_in']}  →S3: {state['s3_in']}")

                b = len(state["buy"])
                s = len(state["sell"])
                status_row.markdown(
                    f"<div style='display:flex;gap:1rem;margin:0.5rem 0'>"
                    f"<div class='metric-box'><b style='color:#0f4'>BUY</b><br>"
                    f"<span style='font-size:1.6rem;font-weight:700'>{b}</span></div>"
                    f"<div class='metric-box'><b style='color:#f44'>SELL</b><br>"
                    f"<span style='font-size:1.6rem;font-weight:700'>{s}</span></div>"
                    f"<div class='metric-box'><b>→S2</b><br>{state['s2_in']}</div>"
                    f"<div class='metric-box'><b>→S3</b><br>{state['s3_in']}</div>"
                    f"<div class='metric-box'><b>Elapsed</b><br>{elapsed:.0f}s</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Live partial results table
                rows = (
                    [{"Direction": "🟢 BUY", "Symbol": sym, "Detail": det}
                     for sym, det in state["buy"]] +
                    [{"Direction": "🔴 SELL", "Symbol": sym, "Detail": det}
                     for sym, det in state["sell"]]
                )
                if rows:
                    results_ph.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True, hide_index=True,
                    )

            try:
                state = asyncio.run(run_scan(cfg, update_ui))
            except Exception as e:
                st.error(f"Scan error: {e}")
                return

            elapsed = time.time() - t0
            total   = state["total"]
            buy_results  = sorted(state["buy"],  key=lambda x: x[0])
            sell_results = sorted(state["sell"], key=lambda x: x[0])
            all_sigs     = len(buy_results) + len(sell_results)

            prog_bar.progress(1.0, text=f"✅ Done in {elapsed:.1f}s  ({total/elapsed:.1f} sym/s)")

            summary_ph.success(
                f"**Scan complete** — {total} symbols in {elapsed:.1f}s  "
                f"({total/elapsed:.1f} sym/s) · "
                f"Funnel: {total} → {state['s2_in']} → {state['s3_in']} → **{all_sigs} signals**"
            )

            
            # ─────────────────────────────────────────
            # Final Results (Modern + Export)
            # ─────────────────────────────────────────
            import datetime
            import io

            st.markdown("---")
            st.subheader("📊 Final Signals")

            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            if buy_results or sell_results:

                all_rows = (
                    [{"Direction": "BUY", "Symbol": sym, "Detail": det}
                     for sym, det in buy_results] +
                    [{"Direction": "SELL", "Symbol": sym, "Detail": det}
                     for sym, det in sell_results]
                )

                df_final = pd.DataFrame(all_rows)
                df_final.insert(0, "Scan Time", timestamp)
                df_final.insert(1, "Mode", mode_key.upper())

                st.dataframe(
                    df_final,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### ⬇️ Export Signals")
                colA, colB = st.columns(2)

                # CSV
                csv_buffer = io.StringIO()
                df_final.to_csv(csv_buffer, index=False)

                colA.download_button(
                    label="📄 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"signals_{mode_key}_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # TXT
                txt_buffer = io.StringIO()
                for _, row in df_final.iterrows():
                    txt_buffer.write(
                        f"{row['Scan Time']} | {row['Mode']} | "
                        f"{row['Direction']} | {row['Symbol']} | {row['Detail']}\\n"
                    )

                colB.download_button(
                    label="📝 Download TXT",
                    data=txt_buffer.getvalue(),
                    file_name=f"signals_{mode_key}_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            else:
                st.warning("No signals found this scan.")


    # ══ TAB 2: DEBUG PAIR ═════════════════════════════════════════════
    with tab_debug:
        st.subheader("Debug a Single Symbol")
        st.caption("Verbose pass/fail for every stage check")

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
                    logs = asyncio.run(debug_single(sym_input, dbg_cfg))
                except Exception as e:
                    st.error(f"Error: {e}")
                    logs = []

            if logs:
                rows = [{"Stage": lbl, "Status": status, "Detail": detail}
                        for lbl, status, detail in logs]
                df = pd.DataFrame(rows)

                # Colour coding via dataframe styles
                def color_status(val):
                    if "PASS" in val: return "color: #00ff66; font-weight: bold"
                    if "FAIL" in val: return "color: #ff4444; font-weight: bold"
                    return ""

                st.dataframe(
                    df.style.map(color_status, subset=["Status"]),
                    use_container_width=True, hide_index=True,
                )

                last_status = logs[-1][1]
                if "PASS" in last_status:
                    st.success("✅ All stages passed — SIGNAL CONFIRMED!")
                else:
                    st.error(f"❌ Stopped at: **{logs[-1][0]}**")


if __name__ == "__main__":
    main()
