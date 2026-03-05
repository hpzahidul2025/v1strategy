#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Binance Futures Scanner · ULTRA-FAST Edition  v38           ║
║  Dual Mode: 15M signals OR 5M signals (choose at startup)    ║
║                                                              ║
║  ── SIGNAL LOGIC ─────────────────────────────────────────  ║
║  Entry: Pine Script FINAL COMBINED SIGNAL (4 Steps)          ║
║   Step 1  Swing Utama trend direction (close vs TSL-50)      ║
║   Step 2  Pressure memory: wt2 < 20 / > 80 dot in cur run   ║
║   Step 3  Swing Alt (TSL-5) crossover                        ║
║   Step 4  Volume filter (volFast > volSlow)                  ║
║                                                              ║
║  15M MODE: Daily pivot → 4H ADX/TDI/KC → 1H CloudBS → 15M  ║
║   5M MODE:   4H pivot → 1H ADX/TDI/KC → 15M CloudBS → 5M   ║
║                                                              ║
║  ── ALL FIXES & OPTIMISATIONS (chronological) ────────────  ║
║                                                              ║
║  v4  Stage 2 zero API calls — reuses tdi_tf from S1          ║
║      Stage 3 mid_tf + sig_tf fetched concurrently            ║
║      _rma() vectorized via pd.ewm (no Python for-loop)       ║
║      stage2_worker made sync (no async overhead)             ║
║      fetch() np.array slice instead of list→DataFrame        ║
║      calc_bb_continuation direction-aware (one side only)    ║
║      signals_tf direction-aware + early exit on 1st signal   ║
║      f_swing fully NumPy-vectorized                          ║
║      calc_wt2 single pd.Series(v*s) with .where() masks      ║
║      MAX_CONCURRENT=150  REQUEST_DELAY=0                     ║
║                                                              ║
║  v6  FIX: signals_tf Pine display filter — SELL only when    ║
║      dirMain==-1, BUY only when dirMain==1 (matches TV)      ║
║      FIX: debug _fmt_pine_details Rich markup for decimals   ║
║                                                              ║
║  v7  FIX: sig_limit 15M 120→250 bars (30h→window coverage)  ║
║      False negatives fixed — signals beyond 30h were missed  ║
║                                                              ║
║  v8  FIX: pivot age gate added in S3 (48h/8h max window)     ║
║      FIX: sig_limit resized to exact window + warmup         ║
║           15M: 252 bars (48h×4+60wup)                        ║
║            5M: 156 bars  (8h×12+60wup)                       ║
║      FIX: mid_tf 5M 120→60 bars (exact window fit)           ║
║      FIX: min_sig aligned to signals_tf guard                ║
║      FIX: debug pivot age message f-string SyntaxError       ║
║                                                              ║
║  v9  FIX: tdi_tf limit 150→80 bars (ADX needs 74 minimum)   ║
║      FIX: signals_tf guard max(60,120)→60 (5M false-neg)    ║
║                                                              ║
║  v11 SPEED: pivot age gate moved S3→S1 (saves tdi_tf fetch   ║
║       + S2 CPU for stale pivots before any heavy work)       ║
║       SPEED: S3 split mid_tf→BB check→sig_tf only on pass    ║
║       (~60% of S3 entries skip sig_tf fetch entirely)        ║
║                                                              ║
║  v12 SPEED: aiohttp TCP pool limit=200 keepalive=30 (5-15%)  ║
║       SPEED: fetch_raw for pivot_tf — 554 DataFrame skipped  ║
║       SPEED: ScanUI asyncio.Lock removed (2770 lock ops)     ║
║       SPEED: entering_s2/s3 merged into tick_s2/s3           ║
║       SPEED: signals_tf skips unused pressure side arrays    ║
║       SPEED: calc_adx redundant np.array() casts removed     ║
║       CLEAN: want_sell bool replaces "SELL_S1"/"BUY_S1" str  ║
║                                                              ║
║  v13 FEAT: BOS/ChoCh validation on lower TF (Stage 4)        ║
║       15M mode → validates on 5M chart  (L/R = 10/10)        ║
║        5M mode → validates on 1M chart  (L/R = 10/10)        ║
║       SELL signal valid if:                                   ║
║         · last event BEFORE signal_ts = bearish ChoCh, OR    ║
║         · 1st event AFTER  signal_ts = bearish ChoCh         ║
║       SELL signal invalid if:                                 ║
║         · 1st event AFTER signal_ts = bullish BOS             ║
║       Otherwise → WAIT (shown separately in output)          ║
║       BUY: opposite rules apply                               ║
║       INVALID signals filtered out; VALID/WAIT shown          ║
║       signals_tf now returns (found, signal_ts_ms)            ║
║       Stage 4 added to debug mode with full event list        ║
║                                                              ║
║  v14 FIX: Stage 4 checks each final signal separately        ║
║       signals_tf single-side now collects ALL signal ts in   ║
║       window (not just first); last_p_bar resets after fire  ║
║       stage3_worker validates each ts, picks best outcome    ║
║       (valid > wait > invalid) — first valid wins early exit ║
║       debug Stage 4 prints per-signal result + overall best  ║
║                                                              ║
║  v15 UI:  Output redesigned for quick-glance readability     ║
║       Live panel: 4 side-by-side quadrants (BUY✅ SELL✅ ⏳⏳) ║
║       Live rows: Symbol / ADX / Sigs / ChoCh (no blob)      ║
║       Counter bar: funnel S1→S2→S3→Found + % progress       ║
║       Final tables: # / Symbol / ADX / Sig / ChoCh / Detail ║
║       Summary banner: big counts grid + funnel in one panel  ║
║                                                              ║
║  v16 FIX: Table.add_column() duplicate 'header' kwarg        ║
║       removed redundant header= from ADX and Sigs columns   ║
║       (TypeError on Python 3.14 / Rich latest)               ║
║                                                              ║
║  v17 FIX: st variable shadowing — validate_choch result      ║
║       loop variable renamed from `st` → `choch_result` in   ║
║       stage3_worker and debug_pair, eliminating the          ║
║       latent shadowing risk for consistency with the         ║
║       shared codebase (mirrors futures_scanner v19 fix)      ║
║       Updated: 2026-03-01                                    ║
║                                                              ║
║  v23 FEAT: Stage 3 mid TF KC range validity gate (windows)   ║
║       1st BB signal in pivot window opens a window tracked   ║
║       by valid_from_ts. KC violation closes the window.      ║
║       1st BB after a violation opens a fresh window —        ║
║       no opposite BB required.                               ║
║       sig_tf signals filtered to those >= valid_from_ts.     ║
║       Signals from closed (violated) windows discarded.      ║
║       NOTE: v23 docstring incorrectly said "each BB signal   ║
║       opens a window" — code was correct, doc was wrong.     ║
║       v24 added window_open_bar to fix same-bar KC check     ║
║       and corrected the docstring.                           ║
║                                                              ║
║  v24 FIX: check_bb_kc_range — KC band is mid_tf KC          ║
║       (same h/l/c as BB; explicitly NOT tdi_tf KC).          ║
║       FIX: KC violation check now starts from the candle     ║
║       AFTER the BB signal bar (was incorrectly also checking ║
║       the signal bar itself — same-bar check removed).       ║
║       FIX: docstring corrected — window opens on 1st BB      ║
║       only; consecutive BBs in a clean window do NOT open    ║
║       a new window (code was already right, doc was wrong).  ║
║       window_open_bar added to track signal bar index.       ║
║                                                              ║
║  v25 FIX: v23 header entry corrected — "Each BB signal       ║
║       opens a window" was misleading; clarified 1st BB       ║
║       opens, consecutive BBs (no violation) do not.          ║
║       FIX: stage3_worker docstring updated — stale "v13:     ║
║       Rule 5" replaced with accurate v24 KC gate summary.    ║
║       FIX: 3 inline # v23 comments updated to # v24.         ║
║       FIX: det string updated: _BB✓ → _BB+KC✓ to reflect     ║
║       that both BB continuation AND KC range gate passed.     ║
║                                                              ║
║  v26 SPEED: choch_tf fetch moved out of concurrent gather    ║
║  v27 FIX:  choch_limit floor restored on dynamic fetch       ║
║       v26's +30 warmup too shallow → "last before" ChoCh     ║
║       events missed → confirmed signals demoted to waiting   ║
║       and delayed until AFTER sig_tf is fetched and the KC   ║
║       filter has run. bars_needed computed dynamically from  ║
║       oldest surviving signal timestamp — only fetches what  ║
║       is actually required for BOS/ChoCh pivot detection.    ║
║       Formula: ceil((now - oldest_sig) / tf_ms) + 30 warmup ║
║       Floor: BOS_LR * 2 + 5 (minimum for pivot detection).  ║
║       Symbols with recent signals fetch ~30 bars instead of  ║
║       650/550 — significant bandwidth + latency reduction    ║
║       across 300+ symbol scans.                              ║
║                                                              ║
║  v28 REFACTOR: Stage 3 mid TF — BB+KC range gate replaced    ║
║       with Pine "SMA Cloud BS Signals + Bayesian Filter"     ║
║       pullback check.                                        ║
║       · Checks if a Cloud BS buy/sell signal fired on mid_tf ║
║         inside the Stage 1 pivot window.                     ║
║       · valid_from_ts = first Cloud BS signal in window.     ║
║       · sig_tf signals still filtered to >= valid_from_ts.   ║
║       · fetch() now includes open price (needed for wick/    ║
║         body candle anatomy used by Cloud BS logic).         ║
║       · check_bb_kc_range() + calc_bb_continuation() kept   ║
║         for debug stats; main gate is now Cloud BS.          ║
║                                                              ║
║  v38 FIX: Pivot age gate restored (removed in v35).          ║
║       stage1_worker rejects pairs where pivot_confirmed_ts    ║
║       is older than mode threshold (48h / 8h). Threshold      ║
║       stored in MODES as pivot_max_age_ms. debug_pair         ║
║       prints pivot age, threshold, and pass/fail line.        ║
║       FIX: Stage 3b exit KC breach check now anchors from     ║
║       the FIRST/OLDEST signal in the pivot window (was:       ║
║       last signal). Any tdi_tf KC breach from first_sig_ts    ║
║       → scan time drops the pair. debug_pair shows            ║
║       first_sig age + per-breach bar details.                 ║
║                                                              ║
║  v37 FIX: Debug mode (choice=3) was crashing with              ║
║       AttributeError: 'NoneType' object has no attribute 'get'║
║       because _http_session was never initialised in the debug ║
║       path. fetch_klines() relies on the module-level          ║
║       _http_session set by main() in scan mode, but the debug  ║
║       branch skipped that setup entirely.                      ║
║       Fix: debug mode now creates its own aiohttp              ║
║       TCPConnector + ClientSession (limit=20), assigns it to   ║
║       _http_session, and passes it to the ccxt instance        ║
║       (mirrors scan-mode setup). Session is properly closed    ║
║       in the finally block alongside ex.close().               ║
║       Connection retry loop + "Connecting…" banner also added  ║
║       to debug path for consistency with scan mode.            ║
║                                                                ║
║  v36 SPEED: Candle fetch overhaul — 4 independent gains:    ║
║       (1) Direct HTTP fetch: fetch_klines() hits Binance     ║
║           /fapi/v1/klines directly via shared aiohttp        ║
║           session, bypassing all ccxt overhead (JSON schema  ║
║           validation, market normalisation, rate-limit       ║
║           bookkeeping) on every single candle request.       ║
║       (2) Stage 1 concurrent: pivot_tf + tdi_tf now fetched  ║
║           in one asyncio.gather() instead of sequentially.   ║
║           Saves one full round-trip latency for every symbol ║
║           that passes pivot — typically the slowest symbols. ║
║       (3) Stage 3 concurrent 3-way: mid_tf, sig_tf, choch_tf ║
║           all fetched in a single gather(). Cloud BS + QM    ║
║           gates applied in-memory — eliminates the mid_tf    ║
║           sequential stall before sig_tf/choch_tf fetch.     ║
║       (4) Dynamic choch_tf limit: bars computed from actual  ║
║           pivot_win_ts age instead of worst-case 550/650.    ║
║           Recent pivots fetch ~30 bars instead of hundreds.  ║
║       fetch() / fetch_raw() kept as thin shims for           ║
║       debug_pair compatibility — same signatures, zero       ║
║       behaviour change for debug mode.                       ║
║       BUG FIX (audit): sig_limit / mid_limit were still      ║
║       hardcoded to old age-gate sizes (165/270 bars for      ║
║       sig_tf, 80/95 for mid_tf). With the age gate removed   ║
║       a pivot from 2 days ago on 15m sig_tf needs 192 bars   ║
║       just to reach it — leaving zero warmup for TSL/wt2.    ║
║       Both limits are now dynamic:                           ║
║         bars = ceil(pivot_span_ms / tf_ms) + 60 warmup       ║
║       capped at 1500 (Binance API max). debug_pair gets the  ║
║       same fix with a visible "Bar limits" info line.        ║
║       Also fixed: fetch_klines per-row Python loop replaced  ║
║       with vectorised np.array()[:,:6].astype(float) cast;  ║
║       429/5xx responses now retry with backoff (were         ║
║       silently returning None); dynamic choch floor was      ║
║       set to the cap (always returned the cap).              ║
║                                                              ║
║  v35 FEAT: Pivot age gate REMOVED — pairs no longer         ║
║       rejected for having an old pivot. Once a valid pair   ║
║       passes all stages it is dropped only if:              ║
║         · TSL dirMain flips against direction (Stage 3b),   ║
║         · OR sig_tf KC band is breached at scan time.       ║
║       Pair re-appears when a new signal fires AND both       ║
║       TSL + KC are clean at scan time.                      ║
║       Stage 3b exit gate added to stage3_worker AND         ║
║       debug_pair (full pass/fail verbose output).           ║
║                                                              ║
║  v34 FIX: 5M mode MTF QM — 1M lower-TF zigzag length and    ║
║       pivot period raised from 5 → 10 each.                  ║
║       signals_pine_only gains ltf_zz_len / ltf_s2_pp params  ║
║       (default None → inherits sig_tf values).                ║
║       stage3_worker and debug_pair pass ltf_zz_len=10,        ║
║       ltf_s2_pp=10 when sig_tf == "5m" (1M choch_tf path).   ║
║       15M mode unchanged (5M choch_tf keeps zz_len=5, pp=5). ║
║                                                              ║
║  v32 FIX: ltf_limit (choch_tf) was hardcoded to 200 in      ║
║       stage3_worker AND debug_pair — cfg["choch_limit"]      ║
║       (650/550) was defined but never wired in.              ║
║       15M: 200 bars × 5m = 16.7h covered only 35% of the    ║
║       48h pivot window → ~380 bars of 5M QM signals missed. ║
║       5M:  200 bars × 1m =  3.3h covered only 42% of the    ║
║       8h  pivot window → ~280 bars of 1M QM signals missed. ║
║       FIX: ltf_limit = cfg["choch_limit"] in both paths.    ║
║       FIX: mid_limit 60→80 (5M) / 80→95 (15M) — Cloud BS   ║
║       Bayesian warmup (40 bars) + full pivot window left     ║
║       only 5h/40h usable; near-limit-age pivots had NaN     ║
║       signals at window open silently dropped.               ║
║       FIX: sig_limit 156→165 (5M) / 252→270 (15M) —        ║
║       safety margin was only 10 bars; bumped to 75/78.      ║
║       All four limits fixed in stage3_worker and debug_pair. ║
║       FIX: ADX window bounds wrong in stage1_worker and      ║
║       debug_pair. Start was arr_p[-4] (pp_P) — missing the  ║
║       ppp_P bar that anchors the pivot rule. End was         ║
║       arr_p[-1] (live/forming bar) — incomplete candle       ║
║       distorted adx_at_window_end pass/fail check.          ║
║       Fixed: start=arr_p[-5] (ppp_P), end=arr_p[-2] (cur_P) ║
║       — the exact bars the pivot condition spans.            ║
║       FIX: Stage 3 pivot window had no explicit upper bound. ║
║       signals_pine_only and calc_sma_cloud_bs_signals        ║
║       scanned from pivot_ts to end of fetched data — QM and  ║
║       Cloud BS signals beyond cur_P close were valid.        ║
║       pivot_end_ts = arr_p[-1, 0] (open of live bar =        ║
║       close of cur_P) now threaded through the full          ║
║       pipeline: stage1→stage2→stage3→signals_pine_only and  ║
║       both calc_sma_cloud_bs variants + debug_pair.          ║
║       FIX: Stage 1 age gate used pivot_ts (arr_p[-3, 0] =   ║
║       open of peak/trough bar) as the anchor. This is        ║
║       structurally always 2 full bars before now:            ║
║       15M: ≥ 48h old → always > 48h threshold → always fail ║
║        5M: ≥  8h old → always >  8h threshold → always fail ║
║       Result: Stage 1 rejected EVERY symbol in both modes.  ║
║       Fix: age now measured from pivot_confirmed_ts (= arr_p[-1,0] ║
║       = cur_P close = moment the pivot was confirmed).       ║
║       pivot_confirmed_ts computed before the age gate.       ║
║       pivot_end_ts = now (time.time()*1000) — window stays  ║
║       open until the next pivot fires on pivot_tf. Since     ║
║       searchsorted(ts, now) = len(ts), signal search covers  ║
║       all closed bars from pivot_win_ts to present.          ║
║       Same fix applied in debug_pair age gate + display.     ║
║       FIX: Stage 3 signal window was anchored to pivot_ts    ║
║       (arr_p[-3] open = peak/trough bar open). Correct       ║
║       anchor is arr_p[-2] open (cur_P opens = pivot fires).  ║
║       New variable pivot_win_ts = arr_p[-2, 0] threaded      ║
║       through all signal search functions: stage3_worker,    ║
║       signals_pine_only, calc_sma_cloud_bs_signals (both),   ║
║       debug_pair. pivot_ts retained for ADX window + detail. ║
║       FIX: ADX window shifted one pivot forward.             ║
║       Was: ppp_P (arr_p[-5]) → cur_P open (arr_p[-2]).       ║
║       Now: pp_P  (arr_p[-4]) → cur_P close (arr_p[-1]).      ║
║       Start moves from ppp_P to pp_P; end moves from         ║
║       cur_P open to cur_P close (= pivot_confirmed_ts).      ║
╚══════════════════════════════════════════════════════════════╝

  pip install ccxt rich numpy pandas
  python binance_scanner_ULTRAFASTv37.py
"""

import asyncio, time, sys, re
import numpy  as np
import pandas as pd

for _pkg in ("ccxt", "rich", "numpy", "pandas"):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"\n  Missing: pip install {_pkg}\n")
        sys.exit(1)

import ccxt.async_support as ccxt_async
import aiohttp

from rich.console  import Console
from rich.live     import Live
from rich.layout   import Layout
from rich.panel    import Panel
from rich.table    import Table
from rich.text     import Text
from rich.prompt   import Prompt
from rich.progress import (Progress, BarColumn, TextColumn,
                           TimeRemainingColumn, MofNCompleteColumn,
                           SpinnerColumn)
from rich          import box

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
MAX_CONCURRENT = 150   # ⚡ raised from 100

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

# ══════════════════════════════════════════════════════════════════════
#  TIMEFRAME CONFIGS
# ══════════════════════════════════════════════════════════════════════
MODES = {
    "15m": {
        "pivot_tf":        "1d",
        "tdi_tf":          "4h",
        "mid_tf":          "1h",
        "sig_tf":          "15m",
        # v13: BOS/ChoCh validated on 5m (L/R=10/10, Auto mode ≤5m)
        "choch_tf":        "5m",
        # 48h pivot window × 12 bars/h  +  60 warmup bars (L+R+buffer)
        "choch_limit":     650,
        # v38: pivot age gate — pivot_confirmed_ts must be within this window
        "pivot_max_age_ms": 48 * 3_600_000,   # 48 hours
        "label":           "15M Signals  (Daily -> 4H -> 1H -> 15M)",
    },
    "5m": {
        "pivot_tf":        "4h",
        "tdi_tf":          "1h",
        "mid_tf":          "15m",
        "sig_tf":          "5m",
        # v13: BOS/ChoCh validated on 1m (L/R=10/10, Auto mode ≤5m)
        "choch_tf":        "1m",
        # 8h pivot window × 60 bars/h  +  60 warmup bars
        "choch_limit":     550,
        # v38: pivot age gate — pivot_confirmed_ts must be within this window
        "pivot_max_age_ms": 8 * 3_600_000,    # 8 hours
        "label":           "5M Signals   (4H -> 1H -> 15M -> 5M)",
    },
}

# ══════════════════════════════════════════════════════════════════════
#  INDICATOR MATH  — Pine replicas, NumPy-vectorized
# ══════════════════════════════════════════════════════════════════════

def _rma(a: np.ndarray, p: int) -> np.ndarray:
    # ⚡ IMPROVEMENT 1: fully vectorized via pandas ewm.
    # RMA recurrence: out[i] = (1/p)*a[i] + (1-1/p)*out[i-1]
    # is identical to EWM with alpha=1/p, adjust=False.
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
    # v12 [H]: callers pass .values (already float64) — no np.array() cast needed
    tr  = np.maximum(h[1:] - l[1:],
          np.maximum(np.abs(h[1:] - c[:-1]),
                     np.abs(l[1:] - c[:-1])))
    tr  = np.concatenate([[h[0] - l[0]], tr])
    dmp = np.where(
        (h[1:] - h[:-1]) > (l[:-1] - l[1:]),
        np.maximum(h[1:] - h[:-1], 0.0), 0.0)
    dmp = np.concatenate([[0.0], dmp])
    dmm = np.where(
        (l[:-1] - l[1:]) > (h[1:] - h[:-1]),
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


# ⚡ VECTORIZED f_swing — no Python for-loops
def f_swing(h, l, c, no):
    """
    Pine f_swing() — fully NumPy-vectorized.
    Returns (tsl, avn) exactly like Pine's [_tsl, _avn].

    IMPORTANT: avn (direction) is the forward-filled cross signal, NOT
    derived from close > tsl.  Pine uses avn for dirMain trend-flip
    detection, and tsl for price comparison (priceBelowMain etc.).
    These diverge at turning-point bars and must stay separate.
    """
    n   = len(c)
    res = pd.Series(h).rolling(no, min_periods=no).max().values
    sup = pd.Series(l).rolling(no, min_periods=no).min().values

    # avd: +1 if close crosses above rolling high, -1 if below rolling low
    above_res = np.zeros(n)
    above_res[no:] = np.where(c[no:] > res[no - 1:-1],  1.0, 0.0)
    below_sup = np.zeros(n)
    below_sup[no:] = np.where(c[no:] < sup[no - 1:-1], -1.0, 0.0)

    # combine: where both fire on same bar, +1 wins (bearish→bullish flip)
    avd = np.where(above_res != 0, above_res, below_sup)

    # forward-fill the last non-zero value (avn in original)
    nonzero_mask = avd != 0
    idx = np.where(nonzero_mask, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    avn = avd[idx]
    avn[:no] = 0  # before enough data, treat as neutral

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



# ══════════════════════════════════════════════════════════════════════════════
#  v32: QM STRATEGY HELPERS  — Pine Script exact replicas
# ══════════════════════════════════════════════════════════════════════════════

def _calc_qm_strat1(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    zz_len: int = 5):
    """
    ZigZag-based QM (Strategy 1).

    Pine replica:
      to_up   = high >= ta.highest(high, zz_len)
      to_down = low  <= ta.lowest(low,  zz_len)
      trend   : 1 → -1 on to_down ; -1 → 1 on to_up
      On trend flip to  1: record swing low  (lowest low since last to_down)
      On trend flip to -1: record swing high (highest high since last to_up)

    Bull QM structure: trend==-1, 3 swing highs where H2>H1 and H0>H1,
                       2 swing lows where L1>L0, close>L1
    Bear QM structure: trend== 1, 3 swing lows where L2<L1 and L0<L1,
                       2 swing highs where H1<H0, close<H1

    Returns (bull_qm, bear_qm) — bool arrays, rising-edge only.
    """
    n   = len(c)
    h_s = pd.Series(h)
    l_s = pd.Series(l)
    roll_h = h_s.rolling(zz_len, min_periods=1).max().values
    roll_l = l_s.rolling(zz_len, min_periods=1).min().values

    to_up   = (h >= roll_h)
    to_down = (l <= roll_l)

    trend = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        t = trend[i - 1]
        if t == 1 and to_down[i]:
            trend[i] = -1
        elif t == -1 and to_up[i]:
            trend[i] = 1
        else:
            trend[i] = t

    high_pts: list[tuple[float, int]] = []
    low_pts:  list[tuple[float, int]] = []

    bull_raw = np.zeros(n, bool)
    bear_raw = np.zeros(n, bool)

    last_to_up_bar   = 0
    last_to_down_bar = 0

    for i in range(n):
        if to_up[i]:
            last_to_up_bar = i
        if to_down[i]:
            last_to_down_bar = i

        if i > 0 and trend[i] != trend[i - 1]:
            if trend[i] == 1:
                # Flipped bullish: swing low = lowest since last to_down
                since = max(1, i - last_to_down_bar)
                start = max(0, i - since)
                seg   = l[start : i + 1]
                lv    = float(seg.min())
                li    = start + int(np.argmin(seg))
                low_pts.append((lv, li))
            else:
                # Flipped bearish: swing high = highest since last to_up
                since = max(1, i - last_to_up_bar)
                start = max(0, i - since)
                seg   = h[start : i + 1]
                hv    = float(seg.max())
                hi    = start + int(np.argmax(seg))
                high_pts.append((hv, hi))

        if len(high_pts) >= 3 and len(low_pts) >= 2:
            h2v = high_pts[-3][0]; h1v = high_pts[-2][0]; h0v = high_pts[-1][0]
            l1v = low_pts[-2][0];  l0v = low_pts[-1][0]
            bull_raw[i] = (trend[i] == -1 and
                           h2v > h1v and l1v > l0v and h0v > h1v and c[i] > l1v)

        if len(low_pts) >= 3 and len(high_pts) >= 2:
            l2v = low_pts[-3][0]; l1v = low_pts[-2][0]; l0v = low_pts[-1][0]
            h1v = high_pts[-2][0]; h0v = high_pts[-1][0]
            bear_raw[i] = (trend[i] == 1 and
                           l2v < l1v and h1v < h0v and l0v < l1v and c[i] < h1v)

    bull_qm = np.zeros(n, bool)
    bear_qm = np.zeros(n, bool)
    bull_qm[1:] = bull_raw[1:] & ~bull_raw[:-1]
    bear_qm[1:] = bear_raw[1:] & ~bear_raw[:-1]
    return bull_qm, bear_qm


def _calc_qm_strat2(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    pp: int = 5):
    """
    Pivot-array-based QM (Strategy 2).

    Pine replica:
      Pivots confirmed pp bars after occurrence (ta.pivothigh(pp,pp)).
      Labels: LL / HH / LH / HL via array-building state machine.
      Bear QM: last-4 types HH→HL→HH→LL, v5<v1, newest pivot=bar-pp.
      Bull QM: last-4 types LL→LH→LL→HH, v5>v1, newest pivot=bar-pp.

    Returns (bull_qm, bear_qm) — bool arrays, one True per pattern.
    """
    n = len(c)
    bull_qm = np.zeros(n, bool)
    bear_qm = np.zeros(n, bool)

    piv_h = np.full(n, np.nan)
    piv_l = np.full(n, np.nan)
    for i in range(2 * pp, n):
        window_h = h[i - 2 * pp : i + 1]
        window_l = l[i - 2 * pp : i + 1]
        if h[i - pp] == window_h.max():
            piv_h[i] = h[i - pp]
        if l[i - pp] == window_l.min():
            piv_l[i] = l[i - pp]

    piv_h_bool = ~np.isnan(piv_h)
    piv_l_bool = ~np.isnan(piv_l)

    h_val = np.full(n, np.nan); l_val = np.full(n, np.nan)
    h_idx = np.full(n, -1, dtype=np.int64)
    l_idx = np.full(n, -1, dtype=np.int64)
    _hv = np.nan; _lv = np.nan; _hi = -1; _li = -1
    for i in range(n):
        if piv_h_bool[i]: _hv = float(h[i - pp]); _hi = i - pp
        if piv_l_bool[i]: _lv = float(l[i - pp]); _li = i - pp
        h_val[i] = _hv; h_idx[i] = _hi
        l_val[i] = _lv; l_idx[i] = _li

    a_type: list[str]   = []
    a_val:  list[float] = []
    a_idx:  list[int]   = []

    bear_start = 0.0; check_be = 0
    bull_start = 0.0; check_bu = 0

    def push_low(i):
        t = ("HL" if len(a_type) > 1 and a_val[-2] < l_val[i] else "LL") if len(a_type) > 1 else "L"
        a_type.append(t); a_val.append(float(l_val[i])); a_idx.append(int(l_idx[i]))

    def push_high(i):
        t = ("HH" if len(a_type) > 1 and a_val[-2] < h_val[i] else "LH") if len(a_type) > 1 else "H"
        a_type.append(t); a_val.append(float(h_val[i])); a_idx.append(int(h_idx[i]))

    def pop_last():
        a_type.pop(); a_val.pop(); a_idx.pop()

    for i in range(n):
        hb = piv_h_bool[i]; lb = piv_l_bool[i]
        hv = h_val[i]; lv = l_val[i]
        hi_ = h_idx[i]; li_ = l_idx[i]

        if np.isnan(hv) or np.isnan(lv):
            hb_eff = hb and not np.isnan(hv)
            lb_eff = lb and not np.isnan(lv)
        else:
            hb_eff = hb; lb_eff = lb

        if hb_eff and lb_eff:
            if len(a_type) == 0:
                pass
            else:
                lt = a_type[-1]; lval = a_val[-1]
                is_ll = lt in ("L", "LL"); is_hh = lt in ("H", "HH")
                is_lh = lt == "LH";        is_hl = lt == "HL"
                if is_ll:
                    if float(piv_l[i]) < lval: pop_last(); push_low(i)
                    else: push_high(i)
                elif is_hh:
                    if float(piv_h[i]) > lval: pop_last(); push_high(i)
                    else: push_low(i)
                elif is_lh:
                    if float(piv_h[i]) < lval: push_low(i)
                    elif float(piv_h[i]) > lval:
                        if c[i] < lval: pop_last(); push_high(i)
                        elif c[i] > lval: push_low(i)
                elif is_hl:
                    if float(piv_l[i]) > lval: push_high(i)
                    elif float(piv_l[i]) < lval:
                        if c[i] > lval: pop_last(); push_low(i)
                        elif c[i] < lval: push_high(i)
        elif hb_eff:
            if len(a_type) == 0:
                a_type.append("H"); a_val.append(float(hv)); a_idx.append(int(hi_))
            else:
                lt = a_type[-1]; lval = a_val[-1]
                is_lo = lt in ("L", "HL", "LL"); is_hi = lt in ("H", "HH", "LH")
                if is_lo:
                    if float(piv_h[i]) > lval: push_high(i)
                    elif float(piv_h[i]) < lval: pop_last(); push_low(i)
                elif is_hi:
                    if lval < float(hv): pop_last(); push_high(i)
        elif lb_eff:
            if len(a_type) == 0:
                a_type.append("L"); a_val.append(float(lv)); a_idx.append(int(li_))
            else:
                lt = a_type[-1]; lval = a_val[-1]
                is_lo = lt in ("L", "HL", "LL"); is_hi = lt in ("H", "HH", "LH")
                if is_hi:
                    if float(piv_l[i]) < lval: push_low(i)
                    elif float(piv_l[i]) > lval: pop_last(); push_high(i)
                elif is_lo:
                    if lval > float(lv): pop_last(); push_low(i)

        if len(a_type) > 5:
            t1 = a_type[-1]; t2 = a_type[-2]; t3 = a_type[-3]; t4 = a_type[-4]
            v1 = a_val[-1];  v2 = a_val[-2];  v5 = a_val[-5]
            i1 = a_idx[-1]

            bear_cond = (t1 == "LL" and t2 == "HH" and t3 == "HL" and t4 == "HH"
                         and v5 < v1 and i1 == i - pp and check_be == 0)
            if bear_cond:
                bear_start = v2; check_be = 1; bear_qm[i] = True
            if bear_start != (a_val[-2] if len(a_val) >= 2 else bear_start):
                check_be = 0

            bull_cond = (t1 == "HH" and t2 == "LL" and t3 == "LH" and t4 == "LL"
                         and v5 > v1 and i1 == i - pp and check_bu == 0)
            if bull_cond:
                bull_start = v2; check_bu = 1; bull_qm[i] = True
            if bull_start != (a_val[-2] if len(a_val) >= 2 else bull_start):
                check_bu = 0

    return bull_qm, bear_qm


def signals_pine_only(ds_sig, ds_lower, pivot_win_ts: int, pivot_end_ts: int,
                      want_sell: bool,
                      zz_len: int = 5, s2_pp: int = 5,
                      ltf_zz_len: int | None = None, ltf_s2_pp: int | None = None):
    """
    v33: Exact Pine Script replica of QM + Pressure gate.
    Window: pivot_win_ts (cur_P open = pivot fires) → pivot_end_ts (now).

    Pressure dot (rising edge):
      SELL: wt2 > 80  AND  close < TSL-50
      BUY:  wt2 < 20  AND  close > TSL-50

    Latch (had_pressure): arms on pressure dot.
    Resets on TSL dirMain flip in the wrong direction.

    Chart-TF QM (Strat1 ZigZag OR Strat2 Pivot) fires while latch
    armed AND TSL filter passes → valid signal, latch consumed.

    Lower-TF QM fires while latch armed AND TSL filter passes
    → valid signal, latch consumed.

    Whichever fires first (chart-TF or lower-TF) is the valid signal.

    Returns (found: bool, sig_ts_list: list[int])
    """
    h  = ds_sig.high.values
    l  = ds_sig.low.values
    c  = ds_sig.close.values
    v  = ds_sig.volume.values
    ts = ds_sig.ts.values.astype(np.int64)
    n  = len(c)

    tsl_main, dir_main = f_swing(h, l, c, SWING_UTAMA)

    above_tsl = c > tsl_main
    below_tsl = c < tsl_main

    wt2 = calc_wt2(h, l, c, v)

    if want_sell:
        raw_p = (wt2 > 80) & below_tsl
    else:
        raw_p = (wt2 < 20) & above_tsl

    pressure = np.zeros(n, bool)
    pressure[1:] = raw_p[1:] & ~raw_p[:-1]

    s1_bull, s1_bear = _calc_qm_strat1(h, l, c, zz_len=zz_len)
    s2_bull, s2_bear = _calc_qm_strat2(h, l, c, pp=s2_pp)
    qm_bull_sig = s1_bull | s2_bull
    qm_bear_sig = s1_bear | s2_bear
    qm_sig      = qm_bear_sig if want_sell else qm_bull_sig
    qm_sig_filtered = qm_sig & (below_tsl if want_sell else above_tsl)

    ltf_bull_qm = np.empty(0, bool)
    ltf_bear_qm = np.empty(0, bool)
    ltf_ts      = np.empty(0, dtype=np.int64)
    # ltf_zz_len / ltf_s2_pp default to the same values as the sig_tf params
    _ltf_zz = ltf_zz_len if ltf_zz_len is not None else zz_len
    _ltf_pp = ltf_s2_pp  if ltf_s2_pp  is not None else s2_pp

    if ds_lower is not None and not ds_lower.empty and len(ds_lower) >= 20:
        lh = ds_lower.high.values; ll = ds_lower.low.values; lc = ds_lower.close.values
        l1b, l1s = _calc_qm_strat1(lh, ll, lc, zz_len=_ltf_zz)
        l2b, l2s = _calc_qm_strat2(lh, ll, lc, pp=_ltf_pp)
        ltf_bull_qm = l1b | l2b
        ltf_bear_qm = l1s | l2s
        ltf_ts      = ds_lower.ts.values.astype(np.int64)

    ltf_qm = ltf_bear_qm if want_sell else ltf_bull_qm

    win_start = int(np.searchsorted(ts, pivot_win_ts))
    win_end   = int(np.searchsorted(ts, pivot_end_ts))   # pivot_end_ts=now → covers all closed bars

    had_pressure  = False
    sig_ts_list:   list[int] = []
    sig_kind_list: list[str] = []   # "QM" or "MTF QM" — parallel to sig_ts_list

    for i in range(win_start, min(win_end, n - 1)):   # bounded by pivot_end_ts AND skip live bar
        # TSL direction flip → reset latch AND purge all collected signals.
        # Signals formed before a wrong-direction flip are invalidated — a new
        # pressure dot + QM must form after TSL recovers to the correct side.
        if i > 0 and dir_main[i] != dir_main[i - 1]:
            if want_sell and dir_main[i] > 0:
                had_pressure = False
                sig_ts_list.clear()
                sig_kind_list.clear()
            if not want_sell and dir_main[i] < 0:
                had_pressure = False
                sig_ts_list.clear()
                sig_kind_list.clear()

        if pressure[i]:
            had_pressure = True

        # Chart-TF QM fires
        if had_pressure and qm_sig_filtered[i]:
            sig_ts_list.append(int(ts[i]))
            sig_kind_list.append("QM")
            had_pressure = False

        # Lower-TF QM fires (MTF path) — consumes latch
        if had_pressure and ltf_ts.size > 0:
            tsl_ok = below_tsl[i] if want_sell else above_tsl[i]
            if tsl_ok:
                t_lo = int(ts[i])
                t_hi = (int(ts[i + 1]) if i + 1 < n
                        else t_lo + (int(ts[i]) - int(ts[i - 1])))
                mask = (ltf_ts >= t_lo) & (ltf_ts < t_hi) & ltf_qm[:len(ltf_ts)]
                if mask.any():
                    first_ltf = int(ltf_ts[np.where(mask)[0][0]])
                    sig_ts_list.append(first_ltf)
                    sig_kind_list.append("MTF QM")
                    had_pressure = False

    return len(sig_ts_list) > 0, sig_ts_list, sig_kind_list



def calc_sma_cloud_bs_signals(h: np.ndarray, l: np.ndarray,
                               c: np.ndarray, o: np.ndarray,
                               ts_arr: np.ndarray, pivot_win_ts: int,
                               pivot_end_ts: int,
                               want_sell: bool,
                               sma_len: int   = 20,
                               bb_sma_p: int  = 20,
                               bb_std_m: float = 2.5,
                               sma_b_p: int   = 20,
                               bayes_n: int   = 20,
                               thresh: float  = 15.0):
    """
    v28: Pine Script "SMA Cloud BS Signals + Bayesian Filter" — NumPy replica.

    Returns (found, valid_from_ts, n_signals, details):
      · found          — True if at least one Cloud BS signal (matching want_sell)
                         fired on mid_tf inside the Stage 1 pivot window.
      · valid_from_ts  — timestamp (ms) of the FIRST such signal.
                         None if found=False.
      · n_signals      — total count of Cloud BS signals in the pivot window.
      · details        — list of (candle_offset_in_window, ts_ms) tuples.

    ── SMA Cloud ──────────────────────────────────────────────────────────────
    smaHigh = SMA(high, 20),  smaLow = SMA(low, 20),  smaMid = (H+L)/2
    bullCloud = close >= smaMid

    ── Bayesian BBSMA ─────────────────────────────────────────────────────────
    Bayesian combination of three binary indicators:
      P_bbUpper — fraction of last N bars where close was above bbUpper
      P_bbBasis — fraction of last N bars where close was above BB basis
      P_sma     — fraction of last N bars where close was above SMA
    Each is normalized: p_up = p_up / (p_up + p_down)

    Pine formula (left-to-right operator precedence, nz() wraps NaN → 0):
      sigmaProbsUp   = A*B*C / A * B * C + (1-A)*(1-B)*(1-C)   <- tracks UP-momentum
                     = B2*C2 + (1-A)*(1-B)*(1-C)   [when A != 0]
    greenLine = sigmaProbsUp   * 100   (up-momentum proxy;   Pine formerly mislabelled this sigmaProbsDown)
    redLine   = sigmaProbsDown * 100   (down-momentum proxy; Pine formerly mislabelled this sigmaProbsUp)

    ── Buy signal ─────────────────────────────────────────────────────────────
    bullCloud  AND  touchedCloudBot  AND  (buyCondA OR buyCondB)  AND  bayesBuyOk
      touchedCloudBot = low <= smaHigh  AND  close >= smaLow
      buyCondA        = bullish candle  AND  close > smaLow
      buyCondB        = bearish candle  AND  lowerWick >= body×2  AND  close > smaLow
      bayesBuyOk      = greenLine > redLine  AND  greenLine > thresh

    ── Sell signal ────────────────────────────────────────────────────────────
    bearCloud  AND  touchedCloudTop  AND  (sellCondA OR sellCondB)  AND  bayesSellOk
      touchedCloudTop = high >= smaLow  AND  close <= smaHigh
      sellCondA       = bearish candle  AND  close < smaHigh
      sellCondB       = bullish candle  AND  upperWick >= body×2  AND  close < smaHigh
      bayesSellOk     = redLine > greenLine  AND  redLine > thresh
    """
    n = len(c)

    # ── SMA Cloud ────────────────────────────────────────────────────────
    sma_h   = _sma(h, sma_len)
    sma_l   = _sma(l, sma_len)
    sma_mid = (sma_h + sma_l) / 2.0
    bull_cloud = c >= sma_mid
    bear_cloud = ~bull_cloud

    # ── Bayesian BBSMA ───────────────────────────────────────────────────
    bb_basis   = _sma(c, bb_sma_p)
    bb_std_arr = pd.Series(c).rolling(bb_sma_p, min_periods=bb_sma_p).std(ddof=0).values
    bb_upper   = bb_basis + bb_std_m * bb_std_arr
    sma_b_arr  = _sma(c, sma_b_p)

    c_s = pd.Series(c)
    N   = bayes_n

    raw_bu_up   = (c_s > pd.Series(bb_upper)).rolling(N, min_periods=N).mean().values
    raw_bu_dn   = (c_s < pd.Series(bb_upper)).rolling(N, min_periods=N).mean().values
    raw_bb_up   = (c_s > pd.Series(bb_basis)).rolling(N, min_periods=N).mean().values
    raw_bb_dn   = (c_s < pd.Series(bb_basis)).rolling(N, min_periods=N).mean().values
    raw_sm_up   = (c_s > pd.Series(sma_b_arr)).rolling(N, min_periods=N).mean().values
    raw_sm_dn   = (c_s < pd.Series(sma_b_arr)).rolling(N, min_periods=N).mean().values

    eps = 1e-9
    # Normalized conditional probabilities (Pine probUpBbUpper etc.)
    A_up = raw_bu_up / np.maximum(raw_bu_up + raw_bu_dn, eps)
    B_up = raw_bb_up / np.maximum(raw_bb_up + raw_bb_dn, eps)
    C_up = raw_sm_up / np.maximum(raw_sm_up + raw_sm_dn, eps)

    A_dn = raw_bu_dn / np.maximum(raw_bu_dn + raw_bu_up, eps)
    B_dn = raw_bb_dn / np.maximum(raw_bb_dn + raw_bb_up, eps)
    C_dn = raw_sm_dn / np.maximum(raw_sm_dn + raw_sm_up, eps)

    # Pine left-to-right: A*B*C / A * B * C + (1-A)*(1-B)*(1-C)
    # = B²·C² + (1-A)·(1-B)·(1-C)  [when A≠0, else nz=0]
    # sigma_probs_up_raw   uses UP   probabilities → sigmaProbsUp   → greenLine
    # sigma_probs_down_raw uses DOWN probabilities → sigmaProbsDown → redLine
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_probs_up_raw   = np.where(
            A_up != 0,
            B_up ** 2 * C_up ** 2 + (1 - A_up) * (1 - B_up) * (1 - C_up),
            np.nan)
        sigma_probs_down_raw = np.where(
            A_dn != 0,
            B_dn ** 2 * C_dn ** 2 + (1 - A_dn) * (1 - B_dn) * (1 - C_dn),
            np.nan)

    green_line = np.nan_to_num(sigma_probs_up_raw,   nan=0.0) * 100.0  # sigmaProbsUp   → up-momentum
    red_line   = np.nan_to_num(sigma_probs_down_raw, nan=0.0) * 100.0  # sigmaProbsDown → down-momentum

    bayes_buy_ok  = (green_line > red_line)  & (green_line > thresh)
    bayes_sell_ok = (red_line > green_line)  & (red_line   > thresh)

    # ── Candle anatomy ───────────────────────────────────────────────────
    is_bull    = c >= o
    is_bear    = ~is_bull
    body       = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    atr_vals   = calc_atr(h, l, c, 14)
    valid_body = body > atr_vals * 0.03
    upper_wick_dom = (upper_wick >= body * 2) & valid_body
    lower_wick_dom = (lower_wick >= body * 2) & valid_body

    # ── Signal conditions ────────────────────────────────────────────────
    touched_cloud_top = (h >= sma_l) & (c <= sma_h)
    touched_cloud_bot = (l <= sma_h) & (c >= sma_l)

    sell_cond_a = is_bear & (c < sma_h)
    sell_cond_b = is_bull & upper_wick_dom & (c < sma_h)
    buy_cond_a  = is_bull & (c > sma_l)
    buy_cond_b  = is_bear & lower_wick_dom & (c > sma_l)

    sell_signal = bear_cloud & touched_cloud_top & (sell_cond_a | sell_cond_b) & bayes_sell_ok
    buy_signal  = bull_cloud & touched_cloud_bot & (buy_cond_a  | buy_cond_b)  & bayes_buy_ok

    # ── Find first signal in pivot window ────────────────────────────────
    win_start  = int(np.searchsorted(ts_arr, pivot_win_ts))
    win_end    = int(np.searchsorted(ts_arr, pivot_end_ts))   # pivot_end_ts=now → covers all closed bars
    sig_arr    = sell_signal if want_sell else buy_signal
    sig_idxs   = np.where(sig_arr[win_start:win_end])[0]   # relative to win_start

    if sig_idxs.size == 0:
        return False, None, 0, []

    first_abs  = sig_idxs[0] + win_start
    valid_from = int(ts_arr[first_abs])
    details    = [(int(i + 1), int(ts_arr[i + win_start])) for i in sig_idxs]
    return True, valid_from, int(sig_idxs.size), details


def calc_sma_cloud_bs_debug(h: np.ndarray, l: np.ndarray,
                             c: np.ndarray, o: np.ndarray,
                             ts_arr: np.ndarray, pivot_win_ts: int,
                             pivot_end_ts: int,
                             want_sell: bool):
    """
    Extended version of calc_sma_cloud_bs_signals for debug_pair output.
    Returns (found, valid_from_ts, n_total_signals, signal_details_list)
    where signal_details_list = [(candle_offset_in_window, ts_ms), ...]
    """
    n = len(c)

    sma_h   = _sma(h, 20)
    sma_l   = _sma(l, 20)
    sma_mid = (sma_h + sma_l) / 2.0
    bull_cloud = c >= sma_mid
    bear_cloud = ~bull_cloud

    bb_basis   = _sma(c, 20)
    bb_std_arr = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).values
    bb_upper   = bb_basis + 2.5 * bb_std_arr
    sma_b_arr  = _sma(c, 20)

    c_s = pd.Series(c); N = 20
    raw_bu_up = (c_s > pd.Series(bb_upper)).rolling(N, min_periods=N).mean().values
    raw_bu_dn = (c_s < pd.Series(bb_upper)).rolling(N, min_periods=N).mean().values
    raw_bb_up = (c_s > pd.Series(bb_basis)).rolling(N, min_periods=N).mean().values
    raw_bb_dn = (c_s < pd.Series(bb_basis)).rolling(N, min_periods=N).mean().values
    raw_sm_up = (c_s > pd.Series(sma_b_arr)).rolling(N, min_periods=N).mean().values
    raw_sm_dn = (c_s < pd.Series(sma_b_arr)).rolling(N, min_periods=N).mean().values

    eps = 1e-9
    A_up = raw_bu_up / np.maximum(raw_bu_up + raw_bu_dn, eps)
    B_up = raw_bb_up / np.maximum(raw_bb_up + raw_bb_dn, eps)
    C_up = raw_sm_up / np.maximum(raw_sm_up + raw_sm_dn, eps)
    A_dn = raw_bu_dn / np.maximum(raw_bu_dn + raw_bu_up, eps)
    B_dn = raw_bb_dn / np.maximum(raw_bb_dn + raw_bb_up, eps)
    C_dn = raw_sm_dn / np.maximum(raw_sm_dn + raw_sm_up, eps)

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_probs_up   = np.nan_to_num(np.where(A_up != 0,
            B_up**2 * C_up**2 + (1-A_up)*(1-B_up)*(1-C_up), np.nan), nan=0.0) * 100  # sigmaProbsUp   → green
        sigma_probs_down = np.nan_to_num(np.where(A_dn != 0,
            B_dn**2 * C_dn**2 + (1-A_dn)*(1-B_dn)*(1-C_dn), np.nan), nan=0.0) * 100  # sigmaProbsDown → red

    bayes_buy_ok  = (sigma_probs_up > sigma_probs_down)  & (sigma_probs_up   > 15.0)
    bayes_sell_ok = (sigma_probs_down > sigma_probs_up)  & (sigma_probs_down > 15.0)

    is_bull    = c >= o; is_bear = ~is_bull
    body       = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    atr_vals   = calc_atr(h, l, c, 14)
    valid_body = body > atr_vals * 0.03
    upper_wick_dom = (upper_wick >= body * 2) & valid_body
    lower_wick_dom = (lower_wick >= body * 2) & valid_body

    sell_signal = (bear_cloud & (h >= sma_l) & (c <= sma_h)
                   & (  (is_bear & (c < sma_h))
                      | (is_bull & upper_wick_dom & (c < sma_h)))
                   & bayes_sell_ok)
    buy_signal  = (bull_cloud & (l <= sma_h) & (c >= sma_l)
                   & (  (is_bull & (c > sma_l))
                      | (is_bear & lower_wick_dom & (c > sma_l)))
                   & bayes_buy_ok)

    win_start = int(np.searchsorted(ts_arr, pivot_win_ts))
    win_end   = int(np.searchsorted(ts_arr, pivot_end_ts))   # pivot_end_ts=now → covers all closed bars
    sig_arr   = sell_signal if want_sell else buy_signal
    sig_idxs  = np.where(sig_arr[win_start:win_end])[0]

    if sig_idxs.size == 0:
        return False, None, 0, []

    first_abs  = sig_idxs[0] + win_start
    valid_from = int(ts_arr[first_abs])
    details    = [(int(i + 1), int(ts_arr[i + win_start])) for i in sig_idxs]
    return True, valid_from, len(sig_idxs), details


# ── Binance Futures klines endpoint ──────────────────────────────────────────
# Map ccxt-style TF strings → Binance API interval strings
_TF_TO_BINANCE = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
}
_FAPI_URL = "https://fapi.binance.com/fapi/v1/klines"

# Module-level aiohttp session — set once in main(), shared by all fetchers.
# Using a direct session bypasses all ccxt overhead (JSON schema validation,
# market normalisation, rate-limit bookkeeping) for candle fetches.
_http_session: aiohttp.ClientSession | None = None


async def fetch_klines(sem, sym: str, tf: str, limit: int) -> np.ndarray | None:
    """
    v36 ⚡ Direct Binance Futures klines fetch — bypasses ccxt entirely.

    Returns float64 ndarray shape (N, 6): [ts_ms, open, high, low, close, volume]
    Returns None on error / empty response.

    Converts ccxt symbol "BTC/USDT:USDT" → Binance symbol "BTCUSDT".
    Uses module-level _http_session (set in main()) — zero session-creation
    overhead per call, persistent TCP keep-alive across all concurrent fetches.

    Binance returns each row as a mixed list [int, str, str, str, str, str, ...].
    We slice cols 0–5 via np.array(..., dtype=object)[:, :6].astype(float) — one
    vectorised cast, ~10× faster than the per-row float() loop.

    Retries up to 3× on network errors or 429/5xx HTTP status codes.
    """
    global _http_session
    # ccxt "BTC/USDT:USDT" → "BTCUSDT"
    base_sym = sym.split(":")[0].replace("/", "")
    interval = _TF_TO_BINANCE.get(tf, tf)
    params   = {"symbol": base_sym, "interval": interval, "limit": limit}

    async with sem:
        for _att in range(3):
            try:
                async with _http_session.get(_FAPI_URL, params=params) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        # Rate-limited or server error — back off and retry
                        await asyncio.sleep(1.0 * (_att + 1))
                        continue
                    if resp.status != 200:
                        return None   # 4xx client error (bad symbol etc.) — don't retry
                    data = await resp.json(content_type=None)
                    if not data:
                        return None
                    # Vectorised parse: object array slice → float64 in one cast
                    arr = np.array(data, dtype=object)[:, :6].astype(np.float64)
                    return arr
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if _att < 2:
                    await asyncio.sleep(0.5 * (_att + 1))
            except Exception:
                break
        return None


def _arr_to_df(arr: np.ndarray) -> pd.DataFrame:
    """Convert fetch_klines ndarray → labelled DataFrame (used by stages that need column access)."""
    return pd.DataFrame({
        "ts":     arr[:, 0].astype(np.int64),
        "open":   arr[:, 1],
        "high":   arr[:, 2],
        "low":    arr[:, 3],
        "close":  arr[:, 4],
        "volume": arr[:, 5],
    })


# ── Compatibility shims — kept so debug_pair and stage workers stay readable ─
async def fetch(ex, sem, sym, tf, limit) -> pd.DataFrame:
    arr = await fetch_klines(sem, sym, tf, limit)
    if arr is None or len(arr) == 0:
        return pd.DataFrame()
    return _arr_to_df(arr)


async def fetch_raw(ex, sem, sym, tf, limit) -> np.ndarray | None:
    arr = await fetch_klines(sem, sym, tf, limit)
    if arr is None or len(arr) < 5:
        return None
    return arr


# ══════════════════════════════════════════════════════════════════════
#  SCAN STAGES  ⚡ optimized — concurrent fetches, zero redundant calls
# ══════════════════════════════════════════════════════════════════════

async def stage1_worker(ex, sem, sym, cfg):
    """
    Rule 1  – Pivot structure  (pivot_tf, 7 candles)
    Rule 2a – ADX pre-filter   (tdi_tf, 80 candles)

    v36 ⚡ Both fetches fire concurrently.  Pivot direction is validated on the
    already-fetched array before ADX math runs — bail early if pivot fails.
    """
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]

    # ⚡ Concurrent fetch — pivot_tf and tdi_tf in parallel
    arr_p_raw, da = await asyncio.gather(
        fetch_raw(ex, sem, sym, pivot_tf, 7),
        fetch    (ex, sem, sym, tdi_tf,   80),
    )
    if arr_p_raw is None:
        return None

    arr_p = arr_p_raw
    pivot_ts          = int(arr_p[-3, 0])   # bar[-3] = prev_P = the peak/trough itself
    pivot_win_ts      = int(arr_p[-2, 0])   # bar[-2] = cur_P open = pivot FIRES = Stage 3 window start
    pivot_confirmed_ts = int(arr_p[-1, 0])  # bar[-1] open = close of cur_P = when pivot was confirmed
    pivot_end_ts       = int(time.time() * 1000)  # window stays open until next pivot fires; use now
    def _hlc3(row): return (row[2] + row[3] + row[4]) / 3.0
    cur_P  = _hlc3(arr_p[-2])
    prev_P = _hlc3(arr_p[-3])
    pp_P   = _hlc3(arr_p[-4])
    ppp_P  = _hlc3(arr_p[-5])

    if   cur_P < prev_P and prev_P > max(pp_P, ppp_P): want_sell = True
    elif cur_P > prev_P and prev_P < min(pp_P, ppp_P): want_sell = False
    else: return None

    # v38: pivot age gate — reject if pivot_confirmed_ts is older than the mode threshold.
    # Measured from pivot_confirmed_ts (= close of cur_P bar = moment pivot was confirmed),
    # NOT from pivot_ts (the peak/trough bar itself, which is always 2 bars older).
    pivot_max_age_ms = cfg["pivot_max_age_ms"]
    now_ms           = int(time.time() * 1000)
    if now_ms - pivot_confirmed_ts > pivot_max_age_ms:
        return None

    # ADX was already fetched concurrently above
    if da.empty or len(da) < ADX_LEN * 2:
        return None

    adx_arr = calc_adx(da.high.values, da.low.values, da.close.values)

    pp_P_ts  = int(arr_p[-4, 0])   # ADX window starts at pp_P (one pivot forward from ppp_P)
    adx_end_ts = int(arr_p[-1, 0]) # ADX window ends at cur_P close (= pivot_confirmed_ts = arr_p[-1] open)
    ts_vals      = da["ts"].values.astype(np.int64)
    window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= adx_end_ts)
    adx_window   = adx_arr[window_mask]
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
           f"ADX_cur={adx_at_window_end:.1f} "
           f"pivot_confirmed_ts_ms={pivot_confirmed_ts} "
           f"pivot_ts_ms={pivot_ts}")
    return (want_sell, sym, det, pivot_ts, pivot_win_ts, pivot_end_ts, da)


def stage2_worker(want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da):
    """
    Rule 2b – TDI direction + KC band + last-15 band-clean check.
    """
    if da.empty or len(da) < 60:
        return None

    bear_tdi, bull_tdi = tdi_state(da.close.values[:-1])   # exclude live forming bar
    u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
    c_t                = float(da.close.iloc[-2])           # last confirmed closed bar

    n_t = len(da)
    s15 = max(0, n_t - 16)
    e15 = n_t - 1

    if want_sell:
        if bear_tdi and c_t > l_t[-1] and bool(np.all(da.low.values[s15:e15] > l_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
    else:
        if bull_tdi and c_t < u_t[-1] and bool(np.all(da.high.values[s15:e15] < u_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
    return None



async def stage3_worker(ex, sem, sym, want_sell, detail, pivot_ts, pivot_win_ts, pivot_end_ts, cfg, da):
    """
    v36 Stage 3:
      3a — Cloud BS pullback gate on mid_tf (pure pass/fail).
           Window: pivot_win_ts (cur_P open = pivot fires) → pivot_end_ts (now).
      3b — QM pressure-dot → latch → QM structure on sig_tf or lower_tf.

    v36 ⚡ All three TF fetches (mid_tf, sig_tf, choch_tf) fire concurrently.
         Cloud BS gate and QM logic are applied in-memory — no stall between them.
         choch_tf limit is computed dynamically from pivot_win_ts age instead of
         the fixed worst-case 550/650 — recent pivots fetch only what they need.

    Pressure dot (wt2 extreme + TSL) arms the latch.
    First QM (Strat1 ZigZag OR Strat2 Pivot) to fire on either TF
    while latch is armed = valid signal.  Latch consumed on fire.
    TSL dirMain flip in wrong direction resets the latch.
    Window: pivot_win_ts → pivot_end_ts (= now; open until next pivot fires on pivot_tf).
    Returns 5-tuple (side, sym, det, pivot_ts, "valid") or None.
    """
    mid_tf   = cfg["mid_tf"]
    sig_tf   = cfg["sig_tf"]
    choch_tf = cfg["choch_tf"]   # lower_tf for MTF QM path

    is_5m_mode = sig_tf == "5m"

    # ── Dynamic bar limits based on actual pivot age ───────────────────────────
    # v38 age gate ensures pivot_win_ts is within 48h (15M) / 8h (5M).  These
    # limits still need to cover (pivot_win_ts → now) plus enough warmup bars for
    # indicators to stabilise (TSL-50 needs 50 bars, Bayesian warmup needs 40,
    # KC needs 20 — 60 bars covers all).  Binance klines API hard cap is 1500.
    _WARMUP   = 60
    _API_CAP  = 1500

    _tf_ms = {
        "1m":  60_000,   "3m":  180_000,  "5m":  300_000,
        "15m": 900_000,  "30m": 1_800_000,"1h":  3_600_000,
        "4h":  14_400_000, "1d": 86_400_000,
    }
    _sig_ms   = _tf_ms.get(sig_tf,   900_000)
    _mid_ms   = _tf_ms.get(mid_tf,   3_600_000)
    _choch_ms = _tf_ms.get(choch_tf, 300_000)

    _pivot_span_ms = pivot_end_ts - pivot_win_ts   # ms from pivot fire → now

    sig_limit  = min(_API_CAP, int(_pivot_span_ms / _sig_ms)   + _WARMUP + 10)
    mid_limit  = min(_API_CAP, int(_pivot_span_ms / _mid_ms)   + _WARMUP + 10)
    min_sig    = min(sig_limit, 80)   # validation floor scales with available data

    # Dynamic choch_tf limit — same formula, capped at configured ceiling
    _span_bars  = int(_pivot_span_ms / _choch_ms) + 1
    _floor      = BOS_LR * 2 + 30
    _cap        = cfg["choch_limit"]
    ltf_limit   = max(_floor, min(_span_bars + _floor, _cap))

    # ⚡ Concurrent 3-way fetch — all three TFs in one gather
    dm, ds, dl = await asyncio.gather(
        fetch(ex, sem, sym, mid_tf,   mid_limit),
        fetch(ex, sem, sym, sig_tf,   sig_limit),
        fetch(ex, sem, sym, choch_tf, ltf_limit),
    )

    # ── Stage 3a: Cloud BS pullback gate (mid_tf) — applied in-memory ────────
    if dm.empty or len(dm) < max(BB_LEN, 20) + 10:
        return None

    end    = len(dm) - 1
    ts_mid = dm.ts.values[:end].astype(np.int64)

    cloud_ok, _valid_from_ts, n_cloud, _ = calc_sma_cloud_bs_signals(
        dm.high.values[:end],  dm.low.values[:end],
        dm.close.values[:end], dm.open.values[:end],
        ts_mid, pivot_win_ts, pivot_end_ts, want_sell)

    if not cloud_ok:
        return None   # no Cloud BS signal in pivot window → skip

    # ── Stage 3b: QM pressure gate — applied in-memory (ds/dl already fetched) ─
    if ds.empty or len(ds) < min_sig:
        return None

    ds_lower = dl if (not dl.empty and len(dl) >= 20) else pd.DataFrame()

    found, sig_ts_list, sig_kind_list = signals_pine_only(
        ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
        ltf_zz_len=10 if is_5m_mode else None,
        ltf_s2_pp =10 if is_5m_mode else None)

    if not found:
        return None

    # ── Stage 3b exit validation ──────────────────────────────────────────────
    # A valid pair is DROPPED (returns None) at scan time if either:
    #   (a) TSL dirMain has flipped against the signal direction since the last signal bar, OR
    #   (b) tdi_tf KC band is breached at ANY bar from the oldest signal inside the
    #       valid pivot window (pivot_win_ts) up to scan time.
    #       Anchor = max(sig_ts_list[0], pivot_win_ts) — oldest surviving post-TSL-purge
    #       signal, clamped to the valid pivot age window (48h / 8h).

    ts_sig_arr = ds.ts.values.astype(np.int64)
    last_sig_ts = sig_ts_list[-1]
    sig_bar_idx = int(np.searchsorted(ts_sig_arr, last_sig_ts, side="left"))
    sig_bar_idx = min(sig_bar_idx, len(ds) - 1)

    h_s = ds.high.values
    l_s = ds.low.values
    c_s = ds.close.values

    # (a) TSL flip check: has dirMain flipped against direction at scan time?
    _tsl_s, _dir_s = f_swing(h_s, l_s, c_s, SWING_UTAMA)
    dir_now    = int(_dir_s[-2])   # -2 = last closed bar (skip live bar)
    expected_dir = -1 if want_sell else 1
    tsl_flipped  = (dir_now != expected_dir)

    if tsl_flipped:
        return None   # TSL trend flipped — drop the pair

    # (b) KC clean check — uses tdi_tf (da), same TF as Stage 2 KC gate.
    # Anchor: oldest signal that survived TSL purges within the pivot window.
    # Since signals only form inside [pivot_win_ts, pivot_end_ts], sig_ts_list[0]
    # is always >= pivot_win_ts — no clamping needed.
    h_t = da.high.values
    l_t = da.low.values
    c_t = da.close.values
    u_tdi, l_tdi    = calc_kc(h_t, l_t, c_t)
    ts_tdi          = da.ts.values.astype(np.int64)
    kc_anchor_ts    = sig_ts_list[0]   # oldest signal in window (post-TSL-purge)
    kc_anchor_idx   = int(np.searchsorted(ts_tdi, kc_anchor_ts, side="left"))
    # Slice from anchor bar to last closed bar (exclude live bar at [-1])
    c_range = c_t[kc_anchor_idx:-1]
    u_range = u_tdi[kc_anchor_idx:-1]
    l_range = l_tdi[kc_anchor_idx:-1]
    if want_sell:
        kc_clean = bool(np.all(c_range > l_range))   # never touched/crossed KC lower
    else:
        kc_clean = bool(np.all(c_range < u_range))   # never touched/crossed KC upper

    if not kc_clean:
        return None   # KC band breached since oldest signal in valid window → drop

    side      = "SELL" if want_sell else "BUY"
    n_sigs    = len(sig_ts_list)
    sig_label = f"{n_sigs} sig" + ("s" if n_sigs > 1 else "")
    last_sig_price = float(ds.close.iloc[sig_bar_idx])
    # sig_kind: "QM" or "MTF" (shortened from "MTF QM" to avoid spaces in key=value)
    last_sig_kind = "MTF" if sig_kind_list[-1] == "MTF QM" else "QM"

    det = (f"{detail} | {mid_tf.upper()}_CloudBS✓({n_cloud}) {sig_tf.upper()}_QM✓ ({sig_label})"
           f" sig_kind={last_sig_kind}"
           f" sig_ts_ms={last_sig_ts} sig_price={last_sig_price:.8g}")
    return (side, sym, det, pivot_ts, "valid")


async def pipeline_worker(ex, sem, sym, cfg, ui):
    """
    ⚡ ULTRA-FAST PIPELINE (3 stages, v32):
      Stage 1 — pivot_tf + tdi_tf   (concurrent)
      Stage 2 — zero new API calls  (reuses tdi_tf DataFrame)
      Stage 3 — sig_tf + choch_tf (lower_tf) concurrent fetch
                signals_pine_only: pressure dot → latch → QM (Strat1|Strat2)
    """
    r1 = await stage1_worker(ex, sem, sym, cfg)
    ui.tick_s1(r1)
    if r1 is None:
        return None

    want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r1
    r2 = stage2_worker(want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
    ui.tick_s2(r2)
    if r2 is None:
        return None

    want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r2
    r3 = await stage3_worker(ex, sem, sym, want_sell, detail, pivot_ts, pivot_win_ts, pivot_end_ts, cfg, da)
    ui.tick_s3(r3)
    return r3


# ══════════════════════════════════════════════════════════════════════
#  DEBUG SCANNER  (single pair — verbose pass/fail for every check)
# ══════════════════════════════════════════════════════════════════════

async def debug_pair(ex, sem, sym, cfg, console):
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]
    mid_tf   = cfg["mid_tf"]
    sig_tf   = cfg["sig_tf"]
    choch_tf  = cfg["choch_tf"]    # v13
    choch_lim = cfg["choch_limit"] # v13

    ok   = lambda v: f"[bold green]✓  PASS[/]  {v}"
    fail = lambda v: f"[bold red]✗  FAIL[/]  {v}"
    info = lambda v: f"[dim]       {v}[/]"

    console.print(Panel(
        f"[bold white]Debug: [cyan]{sym}[/]  |  {cfg['label']}",
        border_style="yellow", expand=False))

    console.print("\n[bold cyan]━━  STAGE 1  ━━  Pivot Structure + ADX  (⚡ concurrent fetch)[/]")

    # ⚡ concurrent fetch
    dp, da = await asyncio.gather(
        fetch(ex, sem, sym, pivot_tf, 7),
        fetch(ex, sem, sym, tdi_tf,   80),
    )

    if dp.empty or len(dp) < 5:
        console.print(fail(f"Could not fetch {pivot_tf} data")); return
    if da.empty or len(da) < ADX_LEN * 2:
        console.print(fail(f"Could not fetch {tdi_tf} ADX data")); return

    pivot_ts           = int(dp.iloc[-3]["ts"])  # bar[-3] = prev_P = the peak/trough itself
    pivot_win_ts       = int(dp.iloc[-2]["ts"])  # bar[-2] = cur_P open = pivot fires = Stage 3 window start
    pivot_confirmed_ts = int(dp.iloc[-1]["ts"])  # bar[-1] open = close of cur_P = when pivot was confirmed
    pivot_end_ts       = int(time.time() * 1000) # window open until next pivot fires; use now
    cur_P, prev_P, pp_P, ppp_P = pivot_chain(dp)

    console.print(info(f"cur_P={cur_P:.6f}  prev_P={prev_P:.6f}  pp_P={pp_P:.6f}  ppp_P={ppp_P:.6f}"))

    sell_pivot = cur_P < prev_P and prev_P > max(pp_P, ppp_P)
    buy_pivot  = cur_P > prev_P and prev_P < min(pp_P, ppp_P)

    if sell_pivot:
        console.print(ok(f"SELL pivot  |  cur_P < prev_P: {cur_P:.6f} < {prev_P:.6f}  "
                         f"|  prev_P > max(pp,ppp): {prev_P:.6f} > {max(pp_P,ppp_P):.6f}"))
        direction = "SELL"
    elif buy_pivot:
        console.print(ok(f"BUY  pivot  |  cur_P > prev_P: {cur_P:.6f} > {prev_P:.6f}  "
                         f"|  prev_P < min(pp,ppp): {prev_P:.6f} < {min(pp_P,ppp_P):.6f}"))
        direction = "BUY"
    else:
        cur_vs_prev = f"cur_P {'<' if cur_P < prev_P else '>'} prev_P"
        console.print(fail(f"No valid pivot  |  {cur_vs_prev}: {cur_P:.6f} vs {prev_P:.6f}"))
        if cur_P < prev_P:
            console.print(info(f"SELL peak check: prev_P({prev_P:.6f}) > max(pp,ppp)({max(pp_P,ppp_P):.6f})? "
                               f"{'YES' if prev_P > max(pp_P,ppp_P) else 'NO — peak rule failed'}"))
        else:
            console.print(info(f"BUY trough check: prev_P({prev_P:.6f}) < min(pp,ppp)({min(pp_P,ppp_P):.6f})? "
                               f"{'YES' if prev_P < min(pp_P,ppp_P) else 'NO — trough rule failed'}"))
        console.print("\n[yellow]⛔  Stopped at Stage 1 — pivot failed[/]"); return

    # v38: pivot age gate
    pivot_max_age_ms = cfg["pivot_max_age_ms"]
    now_ms_dbg       = int(time.time() * 1000)
    pivot_age_ms     = now_ms_dbg - pivot_confirmed_ts
    pivot_age_h      = pivot_age_ms / 3_600_000
    max_age_h        = pivot_max_age_ms / 3_600_000
    console.print(info(
        f"Pivot age: {pivot_age_h:.2f}h  |  max allowed: {max_age_h:.0f}h  "
        f"(pivot_confirmed_ts={pivot_confirmed_ts})"))
    if pivot_age_ms > pivot_max_age_ms:
        console.print(fail(
            f"Pivot TOO OLD — {pivot_age_h:.2f}h > {max_age_h:.0f}h limit"))
        console.print("\n[yellow]⛔  Stopped at Stage 1 — pivot age gate failed[/]"); return
    else:
        console.print(ok(
            f"Pivot age OK — {pivot_age_h:.2f}h ≤ {max_age_h:.0f}h limit"))

    adx_arr = calc_adx(da.high.values, da.low.values, da.close.values)

    pp_P_ts    = int(dp.iloc[-4]["ts"])  # ADX window starts at pp_P (one pivot forward from ppp_P)
    adx_end_ts = int(dp.iloc[-1]["ts"])  # ADX window ends at cur_P close (= pivot_confirmed_ts)
    ts_vals      = da["ts"].values.astype(np.int64)
    window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= adx_end_ts)
    adx_window   = adx_arr[window_mask]
    valid_window = adx_window[~np.isnan(adx_window)]

    if len(valid_window) == 0:
        console.print(fail(f"ADX: no valid candles in pivot1→pivot4 window for {tdi_tf}")); return

    n_valid           = len(valid_window)
    adx_first         = float(valid_window[0])
    adx_at_window_end = float(valid_window[-1])
    adx_peak          = float(np.nanmax(valid_window))
    adx_low           = float(np.nanmin(valid_window))
    adx_peak_idx      = int(np.nanargmax(valid_window))
    adx_ever_above    = bool(np.any(valid_window > ADX_TH))
    adx_ever_below    = bool(np.any(valid_window <= ADX_TH))
    adx_end_above     = adx_at_window_end > ADX_TH
    candles_above     = int(np.sum(valid_window > ADX_TH))
    candles_below     = int(np.sum(valid_window <= ADX_TH))

    cross_above_idx = int(np.where(valid_window > ADX_TH)[0][0]) if adx_ever_above else None
    last_above_idx  = int(np.where(valid_window > ADX_TH)[0][-1]) if (adx_ever_above and not adx_end_above) else None

    console.print(info(f"ADX window: pp_P bar opens → cur_P bar closes  "
                       f"({n_valid} {tdi_tf} candles)  threshold={ADX_TH}"))
    console.print(info(f"  [bold]candle #1 (pp_P opens)[/] = ADX {adx_first:.2f}  "
                       f"{'[green]above 25[/]' if adx_first > ADX_TH else '[red]below 25[/]'}"))
    console.print(info(f"  [bold]candle #{n_valid} (cur_P closes)[/] = ADX {adx_at_window_end:.2f}  "
                       f"{'[green]above 25[/]' if adx_end_above else '[red]below 25[/]'}"))
    console.print(info(f"  peak={adx_peak:.2f} at candle #{adx_peak_idx+1}/{n_valid}  "
                       f"low={adx_low:.2f}  "
                       f"above_25={candles_above}/{n_valid}  below_25={candles_below}/{n_valid}"))

    if not adx_ever_above:
        console.print(fail(f"ADX Scenario 1 — NO TREND STRENGTH: "
                           f"ADX never closed above {ADX_TH} in entire pivot1→pivot4 window"))
        console.print(info(f"  candle #1(pivot1)={adx_first:.2f}  "
                           f"peak={adx_peak:.2f}  "
                           f"candle #{n_valid}(pivot4)={adx_at_window_end:.2f}  "
                           f"all {n_valid} candles stayed ≤ {ADX_TH}"))
        console.print("\n[yellow]⛔  Stopped at Stage 1 — Scenario 1: no trend strength[/]"); return

    elif adx_ever_above and not adx_end_above:
        console.print(fail(f"ADX Scenario 2 — LOST STRENGTH: "
                           f"ADX rose above {ADX_TH} but dropped below before pivot4 fires"))
        console.print(info(f"  candle #1(pivot1)={adx_first:.2f}  "
                           f"first crossed >{ADX_TH} at candle #{cross_above_idx+1}  "
                           f"(ADX={valid_window[cross_above_idx]:.2f})"))
        console.print(info(f"  peak={adx_peak:.2f} at candle #{adx_peak_idx+1}/{n_valid}  "
                           f"last above {ADX_TH} at candle #{last_above_idx+1}  "
                           f"(ADX={valid_window[last_above_idx]:.2f})"))
        console.print(info(f"  candle #{n_valid}(pivot4)={adx_at_window_end:.2f} — "
                           f"below {ADX_TH}  "
                           f"(above={candles_above}  below={candles_below})"))
        console.print("\n[yellow]⛔  Stopped at Stage 1 — Scenario 2: lost strength before pivot4 fires[/]"); return

    elif adx_ever_above and adx_end_above and adx_ever_below:
        console.print(ok(f"ADX Scenario 3 — GAINING STRENGTH: "
                         f"ADX crossed above {ADX_TH} and held through pivot4 fires"))
        console.print(info(f"  candle #1(pivot1)={adx_first:.2f} (below {ADX_TH})  "
                           f"first crossed >{ADX_TH} at candle #{cross_above_idx+1}  "
                           f"(ADX={valid_window[cross_above_idx]:.2f})"))
        console.print(info(f"  peak={adx_peak:.2f} at candle #{adx_peak_idx+1}/{n_valid}  "
                           f"candle #{n_valid}(pivot4)={adx_at_window_end:.2f}  "
                           f"(above={candles_above}  below={candles_below})"))
    else:
        console.print(ok(f"ADX Scenario 4 — CONSISTENT STRENGTH: "
                         f"ADX stayed above {ADX_TH} throughout entire pivot1→pivot4 window"))
        console.print(info(f"  candle #1(pivot1)={adx_first:.2f}  "
                           f"peak={adx_peak:.2f} at candle #{adx_peak_idx+1}/{n_valid}  "
                           f"low={adx_low:.2f}  "
                           f"candle #{n_valid}(pivot4)={adx_at_window_end:.2f}  "
                           f"all {n_valid} candles above {ADX_TH}"))

    # ── STAGE 2 (reuses `da`) ─────────────────────────────────────────
    console.print("\n[bold yellow]━━  STAGE 2  ━━  TDI Direction + KC Band  (⚡ zero new API calls)[/]")

    bear_tdi, bull_tdi = tdi_state(da.close.values[:-1])   # exclude live forming bar
    u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
    c_t                = float(da.close.iloc[-2])           # last confirmed closed bar

    rsi_arr = calc_rsi(da.close.values, TDI_RSI_P)
    fast_ma = float(_sma(rsi_arr, TDI_FAST)[-1])
    slow_ma = float(_sma(rsi_arr, TDI_SLOW)[-1])

    console.print(info(f"TDI RSI={rsi_arr[-1]:.2f}  fast_MA={fast_ma:.2f}  slow_MA={slow_ma:.2f}"))

    tdi_ok = (direction == "SELL" and bear_tdi) or (direction == "BUY" and bull_tdi)
    tdi_lbl = "bearish" if direction == "SELL" else "bullish"
    if tdi_ok:
        console.print(ok(f"TDI {tdi_lbl}: fast({fast_ma:.2f}) {'<' if direction=='SELL' else '>'} slow({slow_ma:.2f})"))
    else:
        console.print(fail(f"TDI not {tdi_lbl}: fast={fast_ma:.2f}  slow={slow_ma:.2f}"))
        console.print("\n[yellow]⛔  Stopped at Stage 2 — TDI failed[/]"); return

    console.print(info(f"close={c_t:.6f}  KC_upper={u_t[-1]:.6f}  KC_lower={l_t[-1]:.6f}"))
    pos_ok = (direction == "SELL" and c_t > l_t[-1]) or (direction == "BUY" and c_t < u_t[-1])
    if pos_ok:
        ref = l_t[-1] if direction == "SELL" else u_t[-1]
        sym_str = ">" if direction == "SELL" else "<"
        console.print(ok(f"Price {sym_str} KC {'lower' if direction=='SELL' else 'upper'}: "
                         f"{c_t:.6f} {sym_str} {ref:.6f}"))
    else:
        console.print(fail(f"Price outside KC band for {direction} direction"))
        console.print("\n[yellow]⛔  Stopped at Stage 2 — KC position failed[/]"); return

    n_t = len(da); s15 = max(0, n_t - 16); e15 = n_t - 1
    if direction == "SELL":
        touches  = int(np.sum(da.low.values[s15:e15] <= l_t[s15:e15]))
        band_ok  = (touches == 0)
        band_lbl = f"last 15 lows vs KC lower: {touches} touch(es)"
    else:
        touches  = int(np.sum(da.high.values[s15:e15] >= u_t[s15:e15]))
        band_ok  = (touches == 0)
        band_lbl = f"last 15 highs vs KC upper: {touches} touch(es)"

    if band_ok:
        console.print(ok(f"Band clean  |  {band_lbl}"))
    else:
        console.print(fail(f"Band NOT clean  |  {band_lbl}"))
        console.print("\n[yellow]⛔  Stopped at Stage 2 — band clean failed[/]"); return

    # ── STAGE 3a: Cloud BS Pullback Gate (mid_tf) ───────────────────────
    want_sell = (direction == "SELL")   # FIX: was NameError — debug_pair uses direction, not want_sell
    console.print(f"\n[bold magenta]\u2501\u2501  STAGE 3a  \u2501\u2501  Cloud BS Pullback Gate  [{mid_tf.upper()}][/]")
    console.print(info(
        f"Pine \'SMA Cloud BS Signals + Bayesian Filter\' must fire on {mid_tf}\n"
        f"       in pivot window (pivot_ts={pivot_ts}).  Pure gate — no window filtering.\n"
        f"       QM window always anchors to pivot_ts regardless of valid_from_ts."))

    is_5m_mode = sig_tf == "5m"

    # ── Dynamic bar limits (mirrors stage3_worker — no hardcoded sizes) ───────
    _WARMUP  = 60
    _API_CAP = 1500
    _tf_ms = {
        "1m":  60_000,   "3m":  180_000,  "5m":  300_000,
        "15m": 900_000,  "30m": 1_800_000,"1h":  3_600_000,
        "4h":  14_400_000, "1d": 86_400_000,
    }
    _sig_ms  = _tf_ms.get(sig_tf,   900_000)
    _mid_ms  = _tf_ms.get(mid_tf,   3_600_000)
    _choch_ms_val = _tf_ms.get(choch_tf, 300_000)
    _pivot_span_ms = pivot_end_ts - pivot_win_ts

    mid_limit  = min(_API_CAP, int(_pivot_span_ms / _mid_ms)   + _WARMUP + 10)
    sig_limit  = min(_API_CAP, int(_pivot_span_ms / _sig_ms)   + _WARMUP + 10)
    min_sig    = min(sig_limit, 80)
    _span_bars = int(_pivot_span_ms / _choch_ms_val) + 1
    _floor     = BOS_LR * 2 + 30
    _cap       = cfg["choch_limit"]
    ltf_limit  = max(_floor, min(_span_bars + _floor, _cap))

    console.print(info(
        f"Bar limits (dynamic from pivot age {_pivot_span_ms/3_600_000:.1f}h):  "
        f"mid={mid_limit}  sig={sig_limit}  ltf={ltf_limit}"))

    dm = await fetch(ex, sem, sym, mid_tf, mid_limit)
    if dm.empty or len(dm) < max(BB_LEN, 20) + 10:
        console.print(fail(f"Could not fetch {mid_tf} data")); return

    end    = len(dm) - 1
    ts_mid = dm.ts.values[:end].astype(np.int64)
    win_mask = ts_mid >= pivot_win_ts
    console.print(info(f"  → {int(win_mask.sum())} {mid_tf} candles in pivot window (from cur_P open)"))

    cloud_found, valid_from_ts, n_cloud_sigs, cloud_details = calc_sma_cloud_bs_debug(
        dm.high.values[:end],  dm.low.values[:end],
        dm.close.values[:end], dm.open.values[:end],
        ts_mid, pivot_win_ts, pivot_end_ts, want_sell)

    side_label = "SELL" if want_sell else "BUY"
    now_ms = time.time() * 1000

    def _age(ts_ms):
        m = (now_ms - ts_ms) / 60_000
        return (f"{m:.0f}m ago" if m < 60
                else f"{m/60:.1f}h ago" if m < 1440
                else f"{m/1440:.1f}d ago")

    if cloud_found:
        console.print(ok(
            f"Cloud BS {side_label} PASSED  |  {n_cloud_sigs} signal(s) in window  |  "
            f"first at ts={valid_from_ts}  ({_age(valid_from_ts)})"))
        for candle_num, ts_ms in cloud_details:
            console.print(info(f"  \u21b3 candle #{candle_num} in window  ({_age(ts_ms)})"))
    else:
        console.print(fail(f"No Cloud BS {side_label} signal on {mid_tf} in pivot window"))
        console.print("\n[yellow]\u26d4  Stopped at Stage 3a — Cloud BS gate failed[/]"); return

    # ── STAGE 3b: QM Pressure Gate (sig_tf + lower_tf) ──────────────────
    console.print(f"\n[bold magenta]\u2501\u2501  STAGE 3b  \u2501\u2501  QM Pressure Gate  (\u26a1 concurrent fetch)[/]")
    console.print(info(
        f"Fetching {sig_tf} (sig_tf) and {choch_tf} (lower_tf) concurrently.\n"
        f"       Window anchors to pivot_ts={pivot_ts} (NOT valid_from_ts).\n"
        f"       Logic: pressure dot (wt2+TSL) → latch → QM Strat1/Strat2 on either TF."))

    ds, dl = await asyncio.gather(
        fetch(ex, sem, sym, sig_tf,   sig_limit),
        fetch(ex, sem, sym, choch_tf, ltf_limit),
    )

    if ds.empty or len(ds) < min_sig:
        console.print(fail(f"Could not fetch {sig_tf} data (need ≥ {min_sig} bars)")); return

    ds_lower = dl if (not dl.empty and len(dl) >= 20) else pd.DataFrame()
    ltf_label = f"{choch_tf} ({len(dl)} bars)" if not dl.empty else f"{choch_tf} (unavailable)"
    console.print(info(f"Fetched: {sig_tf}={len(ds)} bars  |  lower_tf={ltf_label}"))

    # ── Show pressure dots ────────────────────────────────────────────────
    h = ds.high.values; l = ds.low.values; c = ds.close.values; v = ds.volume.values
    ts_arr = ds.ts.values.astype(np.int64)
    tsl_main, dir_main = f_swing(h, l, c, SWING_UTAMA)
    above_tsl = c > tsl_main; below_tsl = c < tsl_main
    wt2 = calc_wt2(h, l, c, v)

    raw_p = (wt2 > 80) & below_tsl if want_sell else (wt2 < 20) & above_tsl
    pressure = np.zeros(len(c), bool)
    pressure[1:] = raw_p[1:] & ~raw_p[:-1]

    win_start = int(np.searchsorted(ts_arr, pivot_win_ts))
    win_end   = int(np.searchsorted(ts_arr, pivot_end_ts))
    p_bars = np.where(pressure[win_start:win_end])[0] + win_start

    if len(p_bars) > 0:
        console.print(info(
            f"Pressure dots in pivot window: {len(p_bars)} "
            f"({'wt2>80 & below TSL' if want_sell else 'wt2<20 & above TSL'})"))
        for bi in p_bars[-5:]:
            console.print(info(
                f"  \u21b3 bar[{bi}]  wt2={float(wt2[bi]):.1f}  "
                f"ts={ts_arr[bi]}  ({_age(ts_arr[bi])})"))
        if len(p_bars) > 5:
            console.print(info(f"  ... ({len(p_bars)-5} earlier dot(s) not shown)"))
    else:
        console.print(info(f"No pressure dots found in pivot window on {sig_tf}"))

    # ── Run full signals_pine_only ────────────────────────────────────────
    _is_5m = sig_tf == "5m"
    found, sig_ts_list, sig_kind_list = signals_pine_only(
        ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
        ltf_zz_len=10 if _is_5m else None,
        ltf_s2_pp =10 if _is_5m else None)

    if not found:
        console.print(fail(
            f"No QM signals found in pivot window on {sig_tf}/{choch_tf}"))
        console.print("\n[yellow]\u26d4  Stopped at Stage 3b — no QM signals[/]"); return

    n_sigs   = len(sig_ts_list)
    n_qm     = sig_kind_list.count("QM")
    n_mtf    = sig_kind_list.count("MTF QM")
    kind_sum = (f"QM×{n_qm}" if n_qm else "") + (" MTF QM×" + str(n_mtf) if n_mtf else "")
    console.print(ok(f"{n_sigs} signal(s) found  [{kind_sum.strip()}]:"))
    for i, (sig_ts_ms, kind) in enumerate(zip(sig_ts_list, sig_kind_list), start=1):
        tf_label = choch_tf if kind == "MTF QM" else sig_tf
        si  = min(int(np.searchsorted(ds.ts.values.astype(np.int64), sig_ts_ms, side="left")), len(ds) - 1)
        console.print(info(
            f"  \u21b3 Signal #{i}  [{kind}  {tf_label}]  ts={sig_ts_ms}  ({_age(sig_ts_ms)})  "
            f"price={float(ds.close.iloc[si]):.8g}"))

    last_sig_ts    = sig_ts_list[-1]
    last_sig_kind  = sig_kind_list[-1]
    sig_bar_idx    = min(int(np.searchsorted(ds.ts.values.astype(np.int64), last_sig_ts, side="left")), len(ds) - 1)
    last_sig_price = float(ds.close.iloc[sig_bar_idx])

    first_sig_ts   = sig_ts_list[0]   # oldest surviving signal after TSL purges
    # KC anchor = first_sig_ts directly — signals only form inside [pivot_win_ts, now]
    # so first_sig_ts >= pivot_win_ts always; no clamping needed.
    kc_anchor_ts   = first_sig_ts

    # ── Stage 3b exit validation ──────────────────────────────────────────────
    console.print(f"\n[bold magenta]━━  STAGE 3b EXIT  ━━  TSL Flip + KC Clean  (scan-time gate)[/]")
    console.print(info(
        "Pair is DROPPED if TSL dirMain has flipped against direction since last signal,\n"
        f"       or if {tdi_tf} KC band was breached at ANY bar from OLDEST signal in valid\n"
        f"       pivot window (max(first_sig_ts, pivot_win_ts)) → scan time.\n"
        "       Re-appears only after new pressure dot + QM form on the correct TSL side."))

    h_dbg = ds.high.values; l_dbg = ds.low.values; c_dbg = ds.close.values
    _tsl_dbg, _dir_dbg = f_swing(h_dbg, l_dbg, c_dbg, SWING_UTAMA)
    dir_at_sig_dbg = int(_dir_dbg[sig_bar_idx])
    dir_now_dbg    = int(_dir_dbg[-2])
    expected_dir_dbg = -1 if want_sell else 1

    console.print(info(
        f"TSL dirMain at last signal bar: {dir_at_sig_dbg}  |  "
        f"TSL dirMain now (last closed bar): {dir_now_dbg}  |  "
        f"Expected for {direction}: {expected_dir_dbg}"))

    tsl_flipped_dbg = (dir_now_dbg != expected_dir_dbg)
    if tsl_flipped_dbg:
        console.print(fail(
            f"TSL trend FLIPPED — dirMain is {dir_now_dbg} but {direction} requires {expected_dir_dbg}"))
        console.print("\n[yellow]⛔  Stopped at Stage 3b exit — TSL trend flip detected[/]"); return
    else:
        console.print(ok(
            f"TSL trend intact — dirMain={dir_now_dbg} matches expected {expected_dir_dbg} for {direction}"))

    # KC range check on tdi_tf (da).
    # Anchor = max(first_sig_ts, pivot_win_ts) — oldest signal within the valid
    # pivot age window (48h for 15M mode, 8h for 5M mode).
    h_tdi = da.high.values; l_tdi = da.low.values; c_tdi = da.close.values
    u_tdi_dbg, l_tdi_dbg = calc_kc(h_tdi, l_tdi, c_tdi)
    ts_tdi_dbg    = da.ts.values.astype(np.int64)
    kc_anchor_idx = int(np.searchsorted(ts_tdi_dbg, kc_anchor_ts, side="left"))
    c_range_dbg   = c_tdi[kc_anchor_idx:-1]
    u_range_dbg   = u_tdi_dbg[kc_anchor_idx:-1]
    l_range_dbg   = l_tdi_dbg[kc_anchor_idx:-1]
    n_checked     = len(c_range_dbg)

    pivot_win_age_h  = (int(time.time() * 1000) - pivot_win_ts) / 3_600_000
    kc_anchor_age_h  = (int(time.time() * 1000) - kc_anchor_ts) / 3_600_000
    console.print(info(
        f"KC anchor: max(first_sig, pivot_win_ts)  |  "
        f"first_sig={_age(first_sig_ts)}  pivot_win={pivot_win_age_h:.1f}h ago  "
        f"→  anchor={kc_anchor_age_h:.1f}h ago  ({n_checked} {tdi_tf} bars checked)"))

    if want_sell:
        breach_mask  = c_range_dbg <= l_range_dbg
        kc_clean_dbg = bool(np.all(c_range_dbg > l_range_dbg))
        band_label   = "KC_lower"
        breach_op    = "<="
    else:
        breach_mask  = c_range_dbg >= u_range_dbg
        kc_clean_dbg = bool(np.all(c_range_dbg < u_range_dbg))
        band_label   = "KC_upper"
        breach_op    = ">="

    n_breaches = int(np.sum(breach_mask))

    if kc_clean_dbg:
        cur_close_tdi_dbg = float(c_tdi[-2])
        kc_ref = float(l_tdi_dbg[-2]) if want_sell else float(u_tdi_dbg[-2])
        console.print(ok(
            f"KC band clean throughout  |  {n_checked} {tdi_tf} bars checked  |  "
            f"no close {breach_op} {band_label}  |  "
            f"current: close({cur_close_tdi_dbg:.6f}) vs {band_label}({kc_ref:.6f})"))
    else:
        # Show up to 3 breach bars for diagnostics
        breach_idxs = np.where(breach_mask)[0]
        for bi in breach_idxs[:3]:
            abs_i  = sig_tdi_idx + bi
            b_ts   = int(ts_tdi_dbg[abs_i])
            b_c    = float(c_tdi[abs_i])
            b_band = float(l_tdi_dbg[abs_i]) if want_sell else float(u_tdi_dbg[abs_i])
            console.print(info(
                f"  ↳ breach bar[{abs_i}]  ts={b_ts}  ({_age(b_ts)})  "
                f"close={b_c:.6f}  {band_label}={b_band:.6f}"))
        if n_breaches > 3:
            console.print(info(f"  ... ({n_breaches - 3} more breach bar(s) not shown)"))
        console.print(fail(
            f"KC band BREACHED  |  {n_breaches}/{n_checked} {tdi_tf} bars had close {breach_op} {band_label}"))
        console.print("\n[yellow]⛔  Stopped at Stage 3b exit — KC band breached since signal[/]"); return

    color = "green" if direction == "BUY" else "red"
    console.print(Panel(
        f"[bold {color}]\u2705  {direction} SIGNAL CONFIRMED  -  {sym}[/]\n"
        f"[dim]Stages 1-3 passed  |  {cfg['label']}[/]\n"
        f"[dim]S3a: Cloud BS pullback \u2713  ({n_cloud_sigs} signal(s) on {mid_tf})[/]\n"
        f"[dim]S3b: QM pressure gate \u2713  ({n_sigs} signal(s) on {sig_tf}/{choch_tf}"
        f"  [{kind_sum.strip()}])[/]\n"
        f"[dim]S3b exit: TSL trend intact \u2713  |  KC band clean \u2713[/]\n"
        f"[dim]Last signal: {last_sig_kind}  |  {_age(last_sig_ts)}  |  price={last_sig_price:.8g}[/]",
        border_style=f"bright_{color}", expand=False))



# ══════════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS  — parse detail string into structured fields
# ══════════════════════════════════════════════════════════════════════

def _parse_det(det: str) -> dict:
    """Extract key fields from the detail string for structured column display."""
    now_ms = int(time.time() * 1000)

    adx      = re.search(r"ADX_cur=([\d.]+)",              det)
    sigs     = re.search(r"\((\d+) sig",                   det)
    sig_ts   = re.search(r"sig_ts_ms=(\d+)",               det)
    sig_px   = re.search(r"sig_price=([\d.eE+\-]+)",       det)
    piv_ts   = re.search(r"pivot_confirmed_ts_ms=(\d+)",   det)
    kind_m   = re.search(r"sig_kind=(\w+)",                det)
    cloud_m  = re.search(r"CloudBS✓\((\d+)\)",            det)

    # ── Signal age ────────────────────────────────────────────────────
    if sig_ts:
        age_ms  = now_ms - int(sig_ts.group(1))
        age_h   = age_ms / 3_600_000
        age_str = f"{age_h:.1f}h"
    else:
        age_str = "—"

    # ── Pivot age (how old is the confirmed pivot at scan time) ───────
    if piv_ts:
        piv_ms  = now_ms - int(piv_ts.group(1))
        piv_h   = piv_ms / 3_600_000
        piv_str = f"{piv_h:.1f}h"
    else:
        piv_str = "—"

    # ── Signal bar close price — auto-format ──────────────────────────
    if sig_px:
        pval = float(sig_px.group(1))
        if pval >= 1000:
            price_str = f"{pval:,.2f}"
        elif pval >= 1:
            price_str = f"{pval:.4f}"
        elif pval >= 0.0001:
            price_str = f"{pval:.6f}"
        else:
            price_str = f"{pval:.4e}"
    else:
        price_str = "—"

    return {
        "adx":     f"{float(adx.group(1)):.0f}"    if adx     else "—",
        "sigs":    sigs.group(1)                    if sigs    else "1",
        "age":     age_str,
        "piv_age": piv_str,
        "price":   price_str,
        "kind":    kind_m.group(1)                  if kind_m  else "QM",
        "cloud":   cloud_m.group(1)                 if cloud_m else "—",
    }

def _sym_short(sym: str) -> str:
    """BTC/USDT:USDT  →  BTC"""
    return sym.split("/")[0]


# ══════════════════════════════════════════════════════════════════════
#  ZERO-FLASH LIVE UI  — v21
# ══════════════════════════════════════════════════════════════════════

class ScanUI:
    def __init__(self, total: int, cfg: dict):
        self.total    = total
        self.cfg      = cfg
        self.t0       = time.time()

        self.s1_done = 0
        self.s2_in   = 0
        self.s2_done = 0
        self.s3_in   = 0
        self.s3_done = 0
        self.skip    = 0
        self.buy_valid : list[tuple[str, str]] = []
        self.sell_valid: list[tuple[str, str]] = []

        p_tf = cfg["pivot_tf"].upper()
        t_tf = cfg["tdi_tf"].upper()
        m_tf = cfg["mid_tf"].upper()
        s_tf = cfg["sig_tf"].upper()
        c_tf = cfg["choch_tf"].upper()

        self.prog = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=40, complete_style="bright_green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn(" ETA "),
            TimeRemainingColumn(),
            expand=False, refresh_per_second=12,
        )
        self.t_s1 = self.prog.add_task(
            f"[cyan]S1[/] {p_tf} pivot + ADX          ", total=total)
        self.t_s2 = self.prog.add_task(
            f"[yellow]S2[/] {t_tf} TDI + KC  [dim]⚡ 0 API[/] ", total=1, visible=True)
        self.t_s3 = self.prog.add_task(
            f"[magenta]S3[/] {m_tf} CloudBS → {s_tf} QM → {c_tf} MTF", total=1, visible=True)

    @property
    def buy_res(self):  return self.buy_valid
    @property
    def sell_res(self): return self.sell_valid
    @property
    def buy_wait(self):  return []
    @property
    def sell_wait(self): return []

    def tick_s1(self, result):
        self.s1_done += 1
        if result is None: self.skip += 1
        self.prog.update(self.t_s1, completed=self.s1_done)

    def tick_s2(self, result):
        self.s2_in   += 1
        self.s2_done += 1
        if result is None: self.skip += 1
        self.prog.update(self.t_s2, total=max(self.s2_in, 1), completed=self.s2_done)

    def tick_s3(self, result):
        self.s3_in   += 1
        self.s3_done += 1
        if result is None:
            self.skip += 1
        else:
            sig, sym, det, _pts, _st = result
            if sig == "BUY":
                self.buy_valid.append((sym, det))
            elif sig == "SELL":
                self.sell_valid.append((sym, det))
        self.prog.update(self.t_s3, total=max(self.s3_in, 1), completed=self.s3_done)

    # ── Funnel counter row ────────────────────────────────────────────
    def _counters(self):
        el   = max(time.time() - self.t0, 0.001)
        spd  = self.s1_done / el
        pct  = self.s1_done / max(self.total, 1)

        bv = len(self.buy_valid)
        sv = len(self.sell_valid)

        g = Table.grid(expand=True, padding=(0, 3))
        for _ in range(9): g.add_column(justify="center")
        g.add_row(
            Text(f"🟢 BUY\n✅ {bv}", style="bold green",  justify="center"),
            Text("│",                style="dim",          justify="center"),
            Text(f"🔴 SELL\n✅ {sv}", style="bold red",   justify="center"),
            Text("│",                style="dim",          justify="center"),
            Text(f"Funnel\n{self.total}→{self.s2_in}→{self.s3_in}→{bv+sv}",
                                     style="yellow",       justify="center"),
            Text("│",                style="dim",          justify="center"),
            Text(f"Speed\n{spd:.1f} sym/s", style="cyan", justify="center"),
            Text("│",                style="dim",          justify="center"),
            Text(f"Elapsed\n{el:.0f}s  ({pct*100:.0f}%)",
                                     style="dim",          justify="center"),
        )
        mode_name = "15M" if self.cfg["sig_tf"] == "15m" else "5M"
        return Panel(g,
                     border_style="blue",
                     title=f"[bold white]{mode_name} Mode  [dim]·  {self.s1_done}/{self.total} scanned[/]",
                     height=5)

    # ── Live signal panel — side-by-side BUY | SELL ───────────────────
    def _signals(self):
        MAX_EACH = 6   # rows per quadrant

        def _mini_table(entries, valid: bool, is_buy: bool) -> Table:
            """Compact table for one quadrant (e.g. BUY VALID)."""
            col   = "green" if is_buy else "red"
            dim   = "dark_green" if is_buy else "dark_red"
            icon  = "✅" if valid else "⏳"
            label = ("BUY " if is_buy else "SELL") + (" CONFIRMED" if valid else "  WAITING ")
            t = Table(
                box=box.SIMPLE_HEAD,
                header_style=f"bold {col}" if valid else col,
                border_style=col if valid else dim,
                title=f"{icon} [{col}]{label}[/]  [dim]({len(entries)})[/]",
                title_style="bold",
                expand=True,
                show_footer=False,
                padding=(0, 1),
            )
            t.add_column("Symbol",  style=f"bold {col}", no_wrap=True, width=10)
            t.add_column("Price",   style="yellow",       no_wrap=True, width=10, justify="right")
            t.add_column("Sig Age", style="magenta",      no_wrap=True, width=7,  justify="right")
            t.add_column("PivAge",  style="cyan",         no_wrap=True, width=6,  justify="right")
            t.add_column("ADX",     style="cyan",         no_wrap=True, width=4,  justify="right")
            t.add_column("Sigs",    style="white",        no_wrap=True, width=4,  justify="right")
            t.add_column("Kind",    style="bright_white", no_wrap=True, width=4,  justify="center")

            shown = entries[-MAX_EACH:]
            for sym, det in shown:
                p = _parse_det(det)
                t.add_row(_sym_short(sym), p["price"], p["age"], p["piv_age"], p["adx"], p["sigs"], p["kind"])

            if not shown:
                t.add_row("[dim]—", "", "", "", "")
            elif len(entries) > MAX_EACH:
                t.add_row(f"[dim]+{len(entries)-MAX_EACH} more", "", "", "", "")
            return t

        lo = Layout()
        lo.split_row(
            Layout(_mini_table(self.buy_valid,  True, True),  name="bv"),
            Layout(_mini_table(self.sell_valid, True, False), name="sv"),
        )
        return Panel(lo,
                     title="[bold white]Live Signals",
                     border_style="bright_green",
                     padding=(0, 0))

    def __rich__(self):
        lo = Layout(name="root")
        lo.split_column(
            Layout(self.prog,        name="bars",   size=5),
            Layout(self._counters(), name="counts", size=5),
            Layout(self._signals(),  name="signals"),
        )
        return lo


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

async def main():
    global _http_session
    console = Console()

    console.print(Panel(
        "[bold cyan]Binance Futures Multi-TF Scanner[/]  "
        "[dim](async . pipelined . zero-flash . [bold yellow]** ULTRA-FAST[/][dim])[/]\n\n"
        "[bold white]Select mode:[/]\n\n"
        "  [bold yellow]1[/]  [green]15M Signals[/]  -  Daily pivot  ->  4H ADX/TDI/KC  ->  1H Cloud BS gate  ->  15M QM pressure (+ 5M MTF QM)\n"
        "  [bold yellow]2[/]  [cyan]5M  Signals[/]  -  4H pivot    ->  1H ADX/TDI/KC  ->  15M Cloud BS gate ->  5M  QM pressure (+ 1M MTF QM)\n"
        "  [bold yellow]3[/]  [yellow]Debug Pair[/]  -  Verbose pass/fail for every check on one symbol\n\n"
        "[dim]** Speed: concurrent fetches in S1+S3, zero redundant S2 fetch, vectorized indicators[/]\n"
        "[dim]** v13: BOS/ChoCh validation on lower TF (10/10 bars, Pine Auto mode)[/]",
        title="[bold white]Mode Select", border_style="cyan", expand=False,
    ))

    choice = Prompt.ask("\n[bold yellow]Enter mode[/]", choices=["1", "2", "3"], default="1")

    if choice == "3":
        mode_choice = Prompt.ask(
            "[bold yellow]Ruleset?[/]  1=15M  2=5M", choices=["1", "2"], default="1")
        cfg = MODES["15m" if mode_choice == "1" else "5m"]

        raw_sym = Prompt.ask("[bold yellow]Symbol[/]  (btc / btcusdt / btc/usdt / btc/usdt:usdt)").strip()
        raw = raw_sym.strip().upper().replace(" ", "")
        raw_clean = raw.replace("/", "").replace(":", "")
        base = raw_clean.replace("USDT", "") or raw_clean
        sym = f"{base}/USDT:USDT"

        # v37 FIX: debug mode must create its own aiohttp session and wire it
        # into the global _http_session so fetch_klines() works.  Previously
        # _http_session was None in debug mode → AttributeError on first fetch.
        _dbg_connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30, ttl_dns_cache=300)
        _dbg_session   = aiohttp.ClientSession(
            connector=_dbg_connector,
            timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=30),
        )
        global _http_session
        _http_session = _dbg_session

        ex = ccxt_async.binanceusdm({
            "enableRateLimit": True,
            "options":         {"defaultType": "future"},
            "session":         _dbg_session,
            "timeout":         30000,
        })
        try:
            console.print("\n[yellow]Connecting to Binance...[/]")
            await ex.load_markets()
            if sym not in ex.markets:
                console.print(f"[red]'{sym}' not found. Try format like BTC/USDT:USDT[/]")
                return
            sem = asyncio.Semaphore(10)
            await debug_pair(ex, sem, sym, cfg, console)
        finally:
            await ex.close()
            if not _dbg_session.closed:
                await _dbg_session.close()
            await _dbg_connector.close()
        return

    # ── SCAN MODE ─────────────────────────────────────────────────────
    mode_key = "15m" if choice == "1" else "5m"
    cfg      = MODES[mode_key]
    p_tf = cfg["pivot_tf"].upper(); t_tf = cfg["tdi_tf"].upper()
    m_tf = cfg["mid_tf"].upper();   s_tf = cfg["sig_tf"].upper()
    c_tf = cfg["choch_tf"].upper()

    console.print(f"\n[bold green]Mode:[/] {cfg['label']}\n")
    console.print(Panel(
        "[bold yellow]** Ultra-Fast optimizations active:[/]\n"
        f"  * S1: {p_tf} pivot checked first -> {t_tf} ADX only on pass (saves ~95% of tdi fetches)\n"
        f"  * S2: zero new API calls - reuses {t_tf} data from S1\n"
        f"  * S3a: {m_tf} Cloud BS gate (pivot_ts window)\n"
        f"  * S3b: {s_tf} + {c_tf} concurrent; QM Strat1(ZigZag)+Strat2(Pivot) on both TFs\n"
        f"  * sig_tf limit: {'156 bars (8h+wup)' if cfg['sig_tf']=='5m' else '252 bars (48h+wup)'}  "
        f"mid_tf: {'60 bars' if cfg['mid_tf']=='15m' else '80 bars'}  tdi_tf: 80 bars\n"
        f"  * lower_tf (MTF QM): {c_tf}  limit: 200 bars\n"
        f"  * f_swing + pressure dots fully NumPy-vectorized\n"
        f"  * MAX_CONCURRENT=150  REQUEST_DELAY=0\n\n"
        "[red]SELL[/]\n"
        f"  S1  {p_tf} pivot (peak rule)   +  {t_tf} ADX > {ADX_TH:.0f}\n"
        f"  S2  {t_tf} TDI bearish  +  above KC lower  +  last 15 lows clean\n"
        f"  S3a {m_tf} Cloud BS SELL pullback (pivot_ts window) — gate\n"
        f"  S3b {s_tf} pressure dot (wt2>80 & below TSL) → latch\n"
        f"       → QM Strat1/Strat2 on {s_tf} or {c_tf} while armed = valid SELL signal\n\n"
        "[green]BUY[/]\n"
        f"  S1  {p_tf} pivot (trough rule) +  {t_tf} ADX > {ADX_TH:.0f}\n"
        f"  S2  {t_tf} TDI bullish  +  below KC upper  +  last 15 highs clean\n"
        f"  S3a {m_tf} Cloud BS BUY  pullback (pivot_ts window) — gate\n"
        f"  S3b {s_tf} pressure dot (wt2<20 & above TSL) → latch\n"
        f"       → QM Strat1/Strat2 on {s_tf} or {c_tf} while armed = valid BUY  signal",
        title=f"[bold white]Rules  [{cfg['label']}]", border_style="cyan", expand=False,
    ))

    _connector = aiohttp.TCPConnector(limit=200, keepalive_timeout=30, ttl_dns_cache=300)
    _session   = aiohttp.ClientSession(
        connector=_connector,
        timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=30),
    )

    # v36 ⚡ Share the same aiohttp session for direct klines fetches —
    # fetch_klines() uses this global instead of going through ccxt.
    _http_session = _session

    ex = ccxt_async.binanceusdm({
        "enableRateLimit": True,
        "options":         {"defaultType": "future"},
        "session":         _session,
        "timeout":         30000,   # ms — ccxt request timeout
    })

    try:
        console.print("\n[yellow]Connecting to Binance...[/]")
        # Retry load_markets up to 3 times on timeout/network errors
        for _attempt in range(1, 4):
            try:
                await ex.load_markets()
                break
            except Exception as _e:
                if _attempt == 3:
                    raise
                console.print(f"[yellow]  Connection attempt {_attempt} failed ({type(_e).__name__}), retrying...[/]")
                await asyncio.sleep(3 * _attempt)

        symbols = sorted([
            s for s, m in ex.markets.items()
            if m.get("type") == "swap" and m.get("active")
            and m.get("quote") == "USDT" and ":USDT" in s
        ])
        total = len(symbols)
        console.print(f"[green]✓ {total} USDT perpetuals[/]\n")

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        ui  = ScanUI(total, cfg)

        with Live(ui, console=console, screen=True,
                  refresh_per_second=12, vertical_overflow="visible"):
            results = await asyncio.gather(*[
                pipeline_worker(ex, sem, sym, cfg, ui) for sym in symbols
            ])

        el = time.time() - ui.t0

        # v32: no valid/wait split — all results are confirmed QM signals
        def _collect(results, sig_type):
            return sorted(
                [(r[1], r[2]) for r in results if r and r[0] == sig_type],
                key=lambda x: x[0])

        buy_valid   = _collect(results, "BUY")
        sell_valid  = _collect(results, "SELL")
        buy_wait    = []
        sell_wait   = []
        total_found = len(buy_valid) + len(sell_valid)

        console.print()

        # ── Summary banner ────────────────────────────────────────────
        bv = len(buy_valid)
        sv = len(sell_valid)
        funnel = f"{total} → {ui.s2_in} → {ui.s3_in} → {total_found}"

        g = Table.grid(expand=False, padding=(0, 4))
        for _ in range(3): g.add_column(justify="center")
        g.add_row(
            Text(f"🟢 BUY\n      {bv}",   style="bold green",  justify="center"),
            Text(f"🔴 SELL\n      {sv}",  style="bold red",    justify="center"),
            Text(f"Funnel\n{funnel}",      style="yellow",      justify="center"),
        )
        console.print(Panel(
            g,
            title=f"[bold white]✅  Scan Complete  ·  {el:.1f}s  ({total/el:.1f} sym/s)  ·  {cfg['label']}",
            border_style="bright_green",
            expand=False,
            padding=(1, 4),
        ))

        # ── Result tables — structured columns ────────────────────────
        def _print_table(signals, title_str, col, bdr, wait=False):
            if not signals: return
            console.print()
            icon = "⏳" if wait else "✅"

            t = Table(
                title=f"{icon}  {title_str}",
                box=box.ROUNDED,
                header_style=f"bold {col}" if not wait else col,
                border_style=bdr,
                title_style=f"bold {col}",
                show_lines=False,
                padding=(0, 1),
            )
            t.add_column("#",        style="dim",           width=3,  justify="right")
            t.add_column("Symbol",   style=f"bold {col}",  width=10, no_wrap=True)
            t.add_column("Price",    style="yellow",        width=11, justify="right", no_wrap=True)
            t.add_column("Sig Age",  style="magenta",       width=7,  justify="right", no_wrap=True)
            t.add_column("Piv Age",  style="cyan",          width=7,  justify="right", no_wrap=True)
            t.add_column("ADX",      style="cyan",          width=4,  justify="right")
            t.add_column("Sigs",     style="white",         width=4,  justify="right")
            t.add_column("Kind",     style="bright_white",  width=4,  justify="center")
            t.add_column("Cloud",    style="blue",          width=5,  justify="right")
            t.add_column("Levels",   style="dim",           overflow="fold")

            for i, (sym, det) in enumerate(signals, start=1):
                p = _parse_det(det)
                # Extract pivot levels for the Levels column — P= and prev_peak/trough= only
                lv_p    = re.search(r"(P=[\d.]+)",                         det)
                lv_prev = re.search(r"(prev_(?:peak|trough)=[\d.]+)",      det)
                levels  = "  ".join(x.group(1) for x in [lv_p, lv_prev] if x)
                t.add_row(
                    str(i), _sym_short(sym),
                    p["price"], p["age"], p["piv_age"],
                    p["adx"],
                    p["sigs"], p["kind"], p["cloud"],
                    levels,
                )

            console.print(t)

        _print_table(buy_valid,  f"BUY  QM Signals  [{cfg['label']}]",  "green", "green")
        _print_table(sell_valid, f"SELL QM Signals  [{cfg['label']}]",  "red",   "red")

        if total_found == 0:
            console.print("\n[yellow]  No signals found.[/]")

        console.print(f"\n[dim]  {total} symbols  ·  {el:.1f}s  ·  {total/el:.1f} sym/s  ·  {mode_key.upper()} mode[/]\n")

    finally:
        await ex.close()
        if not _session.closed:
            await _session.close()
        await _connector.close()


if __name__ == "__main__":
    asyncio.run(main())
