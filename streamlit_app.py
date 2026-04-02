"""
Binance Futures Scanner - ULTRA-FAST Edition v57
Streamlit Web App — Binance via proxy (bypasses geo-block on cloud servers)

v57 24/7 BACKGROUND SCHEDULER + TELEGRAM (no browser needed):
  - Background daemon thread runs 15M + 5M scans every 15 min, clock-aligned
      to :00/:15/:30/:45 UTC marks, completely independent of any browser session.
  - Uses @st.cache_resource to start exactly once per process; survives
      page reloads and session disconnects.
  - Module-level market cache (_bg_cache) and signal dedup set (_bg_seen)
      ensure no duplicate Telegram alerts across restarts.
  - Scheduler status (last run, next run, errors) shown in Settings panel.
  - Compatible with Streamlit Cloud + UptimeRobot for fully free 24/7 hosting.

v56 AUTO-LOOP + SIGNAL HISTORY + DNS FIX:
  - Auto-Loop mode: runs both 15M and 5M scans every 15 minutes, clock-aligned
      to :00/:15/:30/:45 marks.  Toggle with the "🔄 Auto Scan" button.
      Mode checkboxes let you choose 15M-only, 5M-only, or both.
      A live countdown shows time until next scan; the page self-refreshes.
  - Signal History: every confirmed and waiting signal from every scan in the
      current session is accumulated in session_state["signal_history"] and
      written to signals_history.csv on disk (appended, not overwritten).
      A "📋 History" tab shows the full log, sortable by time or symbol, with
      a one-click CSV download of all accumulated signals.
      Duplicate suppression: same symbol+signal_ts+mode combo not added twice.
  - DNS fix (ported from CLI v56):
      TCPConnector now uses aiohttp.ThreadedResolver instead of aiodns —
      prevents "Could not contact DNS servers" errors on VPN/cloud networks.
      ttl_dns_cache raised 300 → 600s.

v55 PERF-ONLY (zero logic/accuracy changes, ported from CLI v55):
  - _np_ffill helper: NumPy forward-fill — replaces pd.Series.ffill() everywhere.
  - calc_kvo: 2 Python for-loops eliminated.
      k_trend: forward-fill via np.maximum.accumulate.
      cm:      segmented cumsum — O(n) arithmetic; no loop.
  - calc_weis_wave:
      weis_trend: forward-fill via accumulate (same trick).
      is_trending: sliding_window_view replaces shift-AND loop.
  - _calc_qm_strat2:
      Pivot rolling max/min: sliding_window_view replaces pd.Series.rolling(center=True).
      4× pd.Series.ffill() replaced with _np_ffill.
  - calc_sma_cloud_bs_signals + calc_sma_cloud_bs_debug:
      bb_std: E[X²]−E[X]² via _sma — replaces pd.Series.rolling.std.
      6× pd.Series.rolling.mean → 6× _sma (np.convolve, 7-16× faster).
      sma_b_arr eliminated — sma_b_p==bb_sma_p==20 → reuse bb_basis directly.
      Pivot detection: sliding_window_view replaces O(n) Python for-loop.
  - _calc_qm_strat1:
      rolling max/min: pd.Series.rolling(min_periods=1) → sliding_window_view
      with -inf/+inf pad, preserving min_periods=1 semantics. ~2.8× faster.

v54 S3a PIVOT HI/LO FILTER — per-signal, two rules:
  Each Cloud BS signal is validated against its own most-recent confirmed pivot:
  1. Backward scan from the bar BEFORE the signal → find latest confirmed pivot LOW (sell) or HIGH (buy).
     Pine close-based pivot, leftBars=5, rightBars=5.
     If the pivot and signal are on the same candle, the previous pivot is used instead.
  2. Breach check from signal bar to scan time:
       SELL: any close < pivot_low  → invalid.
       BUY:  any close > pivot_high → invalid.
  No pivot found → signal accepted unconditionally.
  calc_sma_cloud_bs_signals / calc_sma_cloud_bs_debug now return a 5-tuple:
    (found, valid_from_ts, n_signals, details, rejected_detail)
  stage3_worker and debug_single callers updated to unpack the 5th element.

v49 fixes:
  - Proxy: auto-rotate to next slot on 407/403 or hard connection failure mid-scan
  - Signals: waiting QMs promoted to confirmed when Pine KWV R3 fires

v46 WARMUP FIX (KWV indicator convergence):
  - _WARMUP raised 60 → 200 bars in stage3_worker dynamic limit calculation.
  - KVO uses EMA(slow=55): needs ~200 bars to converge (span×3.6).
    With only 60 warmup bars, KVO was unconverged for fresh pivots (1–4h old),
    causing the KWV R1/R2/R3 state machine to fire on biased KVO values.
  - Old pivots (24–48h) were unaffected — sig_limit was already 166–262 bars.
  - Fix: sig_limit and mid_limit now request 200 warmup bars before pivot_win_ts.
    Max sig_limit = 192 + 210 = 402 bars (15M mode, 48h pivot) — well under 1500 cap.
    All other indicators (TSL-50, KC-20, MFI-14, Weis-2) get extra headroom for free.

v45 SIGNAL WINDOW LOGIC (one signal per KWV window cycle):
  - Each KWV window cycle (R1->R2->R3 sequence) now contributes at most ONE signal.
  - A second signal requires the window to close and a brand-new cycle to open.
  - Rising edge of allow[] detected per bar to reset signal_fired_this_window.
  - TSL purge also resets signal_fired_this_window.

v44 SIGNAL ENGINE REPLACEMENT (Pine Script QM+KWV exact replica):
  - Pressure dot (wt2 > 80/< 20 + TSL) + latch REMOVED.
  - Replaced with KWV (Klinger Volume Oscillator + Weis Wave + MFI) state machines.
  New indicators: calc_mfi, calc_kvo, calc_weis_wave, calc_kwv_windows
  signals_kwv_qm replaces signals_pine_only (same call-site signature)

v43 PERF — CPU indicator math (no logic changes):
  - _sma(): pd.Series.rolling → np.convolve — 7-16x faster at all bar counts.
    _sma is called in calc_adx, tdi_state, calc_kc, calc_wt2, calc_sma_cloud_bs
    — affects every symbol in every stage.
  - calc_wt2: pd.Series.where().rolling(3).sum() → np.cumsum trick — 29x faster.
    Was the single most expensive per-symbol CPU operation.
  - f_swing: pd.Series.rolling max/min → sliding_window_view — 2-3.5x faster.
    Called twice per symbol in stage3 (sig_tf + exit validation).
  - _calc_qm_strat2: vectorized pivot detection (center=True rolling replaces
    O(n²) per-bar window loop); forward-fill via pandas ffill replaces manual loop.
  - stage2_worker offloaded to _CPU_POOL (ThreadPoolExecutor).
  - calc_sma_cloud_bs_signals + signals_pine_only in stage3_worker offloaded
    to _CPU_POOL — releases GIL for numpy, keeps event loop free for I/O.
  - MAX_CONCURRENT lowered 150 → 75: fewer 429 retries → higher real throughput.
  - fetch_klines: semaphore released before retry sleep; jittered delays to
    avoid thundering-herd on 429s.
  - asyncio.gather(return_exceptions=True) in run_scan: one crashing symbol
    no longer stops the whole scan.

v39 FIX (3 bugs — aligned with CLI v39):
  FIX 1: Pressure dot gates on dir_main instead of above/below_tsl.
          At turning-point bars dir_main (Pine's authoritative dirMain)
          and price-vs-tsl diverge: dir_main can be +1 (bullish) while
          close < tsl_main, producing a SELL dot inside a bullish TSL run.
          SELL: wt2 > 80 AND dir_main < 0
          BUY:  wt2 < 20 AND dir_main > 0
  FIX 2: stage2_worker — tdi_state() now receives da.close.values[:-1]
          (excludes live forming bar). Previously passed all bars including
          the forming candle whose mid-tick close skews RSI → fast SMA,
          flipping TDI direction vs the CLI (bear=False when CLI=True).
          c_t for KC band check changed from iloc[-1] (live) to iloc[-2]
          (last confirmed closed bar) — matches CLI stage2_worker exactly.
  FIX 3: debug_single Stage 2 — same two fixes as stage2_worker applied
          to the debug path so it shows the correct bear/bull result.

v36 UPDATES over v35 (aligned with CLI v28):
  FEAT: Stage 3 mid-TF gate replaced — BB+KC range gate removed; replaced by
        Pine "SMA Cloud BS Signals + Bayesian Filter" pullback check.
        · calc_sma_cloud_bs_signals() checks for a Cloud BS buy/sell signal
          on mid_tf inside the Stage 1 pivot window.
        · valid_from_ts = first Cloud BS signal timestamp (used to gate sig_tf
          signals; only those >= valid_from_ts survive).
        · calc_sma_cloud_bs_debug() provides extended output for debug_single.
        · check_bb_kc_range() / calc_bb_continuation() retained for reference
          but no longer called from the scan pipeline or debug path.
  FEAT: fetch() now includes open price column (arr[:, 1]) required by the
        Cloud BS candle anatomy (body, upper/lower wick calculations).
  FIX:  stage3_worker det string — "BB+KC✓" → "CloudBS✓".
  FIX:  debug_single Stage 3 — "S3 BB+KC Range/Filter" log labels →
        "S3 Cloud BS Range/Filter"; detail message updated accordingly.
  FIX:  _parse_row / _parse_det_card BB regex: r"(\\w+)_BB\\+KC" →
        r"(\\w+)_CloudBS" to match the updated det string format.
  FIX:  UI pipeline label — "BB+KC →" → "CloudBS →".
  FIX:  NoSessionContext crash (from v35) — Streamlit widget calls now execute
        exclusively on the main thread via queue-based state handoff.
  CHORE: Version bump to v36; file renamed binance_futures_scanner_v36.py.

v35 UPDATES over v34 (NoSessionContext fix):
  FIX:  NoSessionContext crash — update_ui (progress bar / counter widgets)
        was called from a background thread spawned by _run_async, but
        Streamlit session context is thread-local. Fixed by passing a
        lightweight _queue_callback into run_scan (no Streamlit calls), then
        polling the queue on the main thread where the session context exists.
  CHORE: Version bump to v35; file renamed binance_futures_scanner_v35.py.

v33 UPDATES over v32:
  UI:  Full mobile-first CSS rewrite with safe-area insets (iPhone notch/home bar).
       Bottom nav bar added — sticky scan/debug tabs pinned at bottom on mobile.
       Touch targets enforced at ≥48px everywhere (raised from 44px).
       Viewport meta tag injected via st.markdown to fix iOS zoom-on-focus.
       Horizontal padding reduced to 0.3rem on ≤390px (full-bleed cards).
       Signal cards on mobile now show 2-per-row with larger price font.
       Card touch feedback improved — :active scale + brightness flash.
       Counters on ≤640px show 3-column; labels abbreviated to 1 line max.
       Tab-list made sticky with backdrop-blur so it stays in view while scrolling.
       Sort bar on ≤640px collapses to 2×2 pill grid (no horizontal overflow).
       Mode selector cards on mobile show full-width stacked at ≥52px height.
       Settings panel uses accordion-style expand (no layout shift on mobile).
       Export buttons always full-width on mobile (stacked, no truncation).
       Proxy banner text wraps cleanly on narrow widths.
       TF pipeline flow scrolls horizontally on all sizes ≤900px.
       Debug layout: radio + input stack vertically on mobile.
       Very small (≤380px): badges hidden, further size reductions.
       Safe-area padding applied to bottom of main container (notch phones).
  UI:  Header compresses to single-line title on ≤390px; subtitle hidden.
  UI:  All st.columns([...]) in main() wrapped with mobile-override CSS.
  CHORE: Version bump to v33; file renamed binance_futures_scanner_v33.py.

v28 UPDATES over v27:
  FIX:  Export CSV and TXT now respect the active sort order — previously both
        files were always written A→Z regardless of the selected sort.
        Exports are now generated dynamically at render time from the sorted
        full-tuple lists (buy/sell valid/wait with pts and choch fields).
        The TXT header includes a "Sort: <label>" line showing which order was used.
        Export filenames include the sort label, e.g.:
        signals_15m_Newest_first_1711234567.csv
  CHORE: Version bump to v28; file renamed binance_futures_scanner_v28.py.

v27 UPDATES over v26:
  FEAT: Sort bar expanded to 4 options:
        "🕐 Newest" — sig_ts_ms descending (most recent signal first)
        "🕛 Oldest" — sig_ts_ms ascending  (oldest signal first)
        "🔤 A → Z"  — symbol name ascending
        "🔡 Z → A"  — symbol name descending
        Sort persists in session_state across tab switches.
  CHORE: Version bump to v27; file renamed binance_futures_scanner_v27.py.

v26 UPDATES over v25:
  FEAT: Results sort control — two toggle buttons above signal tabs:
        "🕐 Newest first" sorts all four groups by sig_ts_ms descending.
        "🔤 Name A→Z"    sorts all four groups alphabetically by symbol.
        Sort choice persists in session_state across tab switches.
        Applies to All tab, BUY/SELL Confirmed, and BUY/SELL Waiting.
  CHORE: Version bump to v26; file renamed binance_futures_scanner_v26.py.

v25 UPDATES over v24:
  FEAT: Multi-proxy fallback — up to 4 proxy slots (PROXY_URL … PROXY_URL_4) in
        Streamlit Secrets. Slots tried in order on each scan/debug run; first
        successful connection is used and remembered in session_state.
        If the active slot fails mid-session, next scan auto-retries all slots.
  FEAT: Proxy status banner shows all configured slots with ACTIVE / STANDBY chips
        and a green indicator on the currently connected slot.
  FEAT: debug_single logs proxy connection slot as first step for transparency.
  CHORE: Version bump to v25; file renamed binance_futures_scanner_v25.py.

v24 UPDATES over v23:
  UI:   Settings panel (TZ + time format) hidden behind gear icon (⚙️) — click to reveal.
  UI:   All-tab signals now rendered as stacked sections (Confirmed above, Waiting below)
        instead of side-by-side two-column layout — easier to scan each group.
  UI:   Debug S4 detail row now includes last-before event and first-after event
        (icon + timestamp) for instant BOS/ChoCh context without scrolling logs.
  CHORE: Version bump to v24; file renamed binance_futures_scanner_v24.py.

v23 UPDATES over v22 (aligned with CLI v27):
  FIX:  debug_single Stage 4 AND stage3_worker — both used
        max(_bars_needed, BOS_LR * 2 + 5) as the floor, omitting
        cfg["choch_limit"] (650 / 550 bars).  The +30-only floor was far
        too shallow to reach "last before" ChoCh events, causing confirmed
        signals to appear as WAIT in both the live scan and debug output.
        Root cause: same as CLI v26→v27 fix.
        Now uses the correct three-way floor matching CLI v27:
          bars_needed = ceil((now - oldest_sig) / tf_ms) + 30 warmup
          floor: max(bars_needed, cfg["choch_limit"], BOS_LR * 2 + 5)
        cfg["choch_limit"] (650 / 550) is kept as the minimum so that
        deep enough history is always fetched for pivot detection.
        Symbols with recent signals still fetch only ~30 bars over the
        floor, preserving v26's bandwidth reduction for old-edge signals.

v22 UPDATES over v21:
  FIX:  debug_single S4 detail_msg — n_sigs was stale (pre-KC-filter count);
        now uses len(sig_ts_list) (post-filter) matching actual signals checked.
  FIX:  _parse_det_card docstring — "ADX_end" → "ADX_cur" (stale label).
  FIX:  _parse_det_card comment — "prefer ADX_end" → "prefer ADX_cur".

v21 UPDATES over v20 (aligned with CLI v23–v26):
  FEAT: check_bb_kc_range() — Stage 3 mid-TF KC range validity gate (v23/v24).
        The 1st BB signal in the pivot window opens a clean window (valid_from_ts).
        Consecutive BB signals in an unviolated window do NOT open a new window.
        A close outside the mid-TF KC band closes the current window.
        The 1st BB signal after a violation opens a fresh window.
        sig_tf signals are filtered: only those >= valid_from_ts survive.
        Signals from closed (violated) windows are silently discarded.
  FEAT: stage3_worker — BB+KC gate replaces plain BB check.
        choch_tf fetch is now DYNAMIC (v26): bars_needed computed from the
        oldest surviving signal timestamp rather than a fixed ceiling.
        Formula: ceil((now - oldest_sig) / tf_ms) + 30 warmup bars.
        Floor: BOS_LR * 2 + 5 (minimum for pivot detection).
        Symbols with recent signals fetch ~30 bars instead of 550/650 —
        significant bandwidth + latency reduction across 300+ symbol scans.
  FEAT: debug_single — BB+KC check replaces plain BB pass/fail log entry.
        sig_ts_list filtered by valid_from_ts before Stage 4 ChoCh check.
  FIX:  stage1_worker det string — aligned with CLI v26 format:
        "ADX_peak=X.X ADX_end=Y.Y" → "ADX_cur=Y.Y ADX_peak=X.X"
        (ADX_cur = end-of-window value; same field as CLI ADX_cur).
  FIX:  stage3_worker det string — "BB_pullback✓" → "BB+KC✓",
        "FinalSignal✓" → "FinalSig✓" (matches CLI v26 det format).
  FIX:  _parse_det_card — ADX regex updated: "ADX_end=" → "ADX_cur=".
  FIX:  _parse_row — ADX_End column parsed from "ADX_cur="; ADX_Peak from "ADX_peak=".
  FIX:  _parse_det_card / _parse_row BB-TF regex: r"(\\w+)_BB_pullback" →
        r"(\\w+)_BB\\+KC" and FinalSignal → FinalSig to match new det format.
  FIX:  calc_bb_continuation — replaced v9a's partially-vectorized hybrid with
        the canonical v26 direction-aware loop (simpler, Pine-accurate).
        _calc_bb_loop fallback removed (no longer needed).

v20 FIXES over v19 (aligned with CLI v21):
  FIX: _parse_det_card — ADX regex changed from r"ADX_(?:cur|peak|end)=([\\d.]+)"
       to r"ADX_end=([\\d.]+)" with fallback to r"ADX_peak=([\\d.]+)" so that
       signal cards display the CURRENT ADX (end-of-window value) rather than
       the historical peak, matching v21 CLI _parse_det() behavior. The full
       data table still shows both ADX_Peak and ADX_End columns via _parse_row().
  FIX: stage3_worker det string — sig count now pluralized correctly:
       "(1 sig)" vs "(2 sigs)" matching v21 CLI sig_label format. The
       _parse_det_card n_sigs regex r"[(][\\d+) sig" matches both forms.
  FIX: _parse_det_card — simplified redundant return expression:
       "adx": adx_v if adx_v != "—" else "—" → "adx": adx_v
       (condition was always True; no behavioral change)

v19 FIXES over v18:
  FIX: st variable shadowing — validate_choch result loop variable renamed from
       `st` → `choch_result` in stage3_worker and debug_single, eliminating
       silent shadowing of the `import streamlit as st` module reference
  FIX: _parse_det_card — removed duplicate `import re as _re2` inside function
       body; now uses module-level `_re` import consistently throughout
  FIX: _parse_det_card — removed dead-code else branch that searched ADX with
       the identical regex pattern as the primary match (would never produce
       a different result, causing confusing unreachable code)
  FEAT: 12h / 24h time format toggle — persists via URL query param ?tf=
  FEAT: Active time format shown as badge in header (🕐 12H / 24H)
  FEAT: time_fmt threaded through all timestamp contexts (cards, table, CSV, TXT, Debug tab)
  FEAT: Time Fmt line added to TXT export header

v17 UPDATES over v16:
  CHORE: Version bump — all identifiers, page title, header badge, docstrings updated to v17

v16 UPDATES over v15:
  FEAT: Persistent timezone selector (32 zones, URL query-param storage)
  FEAT: All timestamps (cards, table, CSV, TXT export) respect chosen timezone
  FEAT: Redesigned header with gradient accent, glow effects, active TZ badge
  FEAT: _fmt_ts() helper centralises all epoch→local-time formatting

v15 UPDATES over v14 (UI overhaul):
  FEAT: Hover/touch highlight on all signal chips and summary banner chips
  FEAT: All-tab split into two columns — Confirmed (left) vs Waiting (right)
  FEAT: Confirmed cards: vibrant gradient, glow box-shadow, pulse dot animation
  FEAT: Wait cards: amber dashed border, dotted left stripe, muted amber palette

v12 UPDATES over v11 (ported from CLI v13/v14/v15):
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
import random
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from numpy.lib.stride_tricks import sliding_window_view

import queue
import threading
import nest_asyncio
nest_asyncio.apply()

# ══════════════════════════════════════════════════════════════════════
#  CPU THREAD POOL — offloads numpy-heavy indicator math so the event
#  loop stays free for I/O.  Workers = 4× CPU count, capped at 32.
# ══════════════════════════════════════════════════════════════════════
_CPU_POOL = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4))

import numpy as np
import pandas as pd
import aiohttp

# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM ALERTS  — v57: ported from CLI v56, extended for 24/7 bg scheduler
#  Token / chat-ID are read from Streamlit Secrets first, then fall back
#  to the hard-coded defaults below so the CLI credentials just work.
#  To override, add to .streamlit/secrets.toml:
#      TG_TOKEN   = "your-bot-token"
#      TG_CHAT_ID = "your-chat-id"
# ══════════════════════════════════════════════════════════════════════
import json
from urllib.request import urlopen, Request as _UrlRequest
from urllib.error   import URLError, HTTPError

def _tg_creds():
    """Return (token, chat_id) from Streamlit Secrets or environment variables."""
    import os
    try:
        tok = st.secrets.get("TG_TOKEN",   os.environ.get("TG_TOKEN",   ""))
        cid = st.secrets.get("TG_CHAT_ID", os.environ.get("TG_CHAT_ID", ""))
    except Exception:
        tok = os.environ.get("TG_TOKEN",   "")
        cid = os.environ.get("TG_CHAT_ID", "")
    if not tok or not cid:
        raise RuntimeError("TG_TOKEN / TG_CHAT_ID not set in Secrets.")
    return tok, cid

def _tg_send_sync(text: str) -> bool:
    """Blocking Telegram sendMessage — safe to call from any thread."""
    try:
        tok, cid = _tg_creds()
    except RuntimeError as e:
        print(f"[Telegram] ❌ {e}")
        return False
    api = f"https://api.telegram.org/bot{tok}/sendMessage"
    payload = json.dumps({
        "chat_id":                  cid,
        "text":                     text,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = _UrlRequest(api, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return bool(result.get("ok"))
    except HTTPError as e:
        print(f"[Telegram] ❌ HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        return False
    except URLError as e:
        print(f"[Telegram] ❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"[Telegram] ❌ Unexpected error: {e}")
        return False

def _tg_fmt_signal_group(signals: list, direction: str, wait: bool = False) -> list:
    """Format one group of signals into Telegram message lines."""
    if not signals:
        return []
    icon   = "🟢" if direction == "BUY" else "🔴"
    status = "⏳ WAITING" if wait else "✅ CONFIRMED"
    lines  = [f"\n{icon} <b>{direction} — {status}</b>", "─────────────────────"]
    for sym, det in signals:
        # Extract price and signal age from det string
        px_m  = _re.search(r"sig_price=([\d.eE+\-]+)", det)
        ts_m  = _re.search(r"sig_ts_ms=(\d+)",         det)
        kind_m = _re.search(r"sig_kind=(\S+)",          det)
        base  = sym.split("/")[0].replace("USDT", "").replace(":USDT", "") or sym
        price = ""
        if px_m:
            pv = float(px_m.group(1))
            price = f"{pv:,.2f}" if pv >= 1000 else (f"{pv:.4f}" if pv >= 1 else f"{pv:.6f}")
        age_str = ""
        if ts_m:
            age_s = (time.time() * 1000 - int(ts_m.group(1))) / 60_000
            age_str = f"{age_s:.0f}m" if age_s < 60 else f"{age_s/60:.1f}h"
        kind_raw = kind_m.group(1) if kind_m else ""
        kind_lbl = "MTF" if "MTF" in kind_raw else "QM"
        lines.append(
            f"{icon} <b>{base}</b>   💰 <code>{price}</code>\n"
            f"    ⏱ {age_str}   │   🔷 {kind_lbl}"
        )
    lines.append("─────────────────────")
    return lines

def _tg_send_signals(
    buy_valid: list, sell_valid: list,
    buy_wait: list,  sell_wait: list,
    label: str, elapsed: float, total: int,
) -> bool:
    """
    Send Telegram alert only when at least one signal exists.
    buy_valid / sell_valid are lists of (sym, det) tuples.
    Returns True if at least one chunk was sent successfully.
    """
    bv, sv = len(buy_valid), len(sell_valid)
    bw, sw = len(buy_wait),  len(sell_wait)
    if bv + sv + bw + sw == 0:
        return False

    ts   = time.strftime("%d %b %Y  %H:%M UTC", time.gmtime())
    body = [
        "📡 <b>BINANCE FUTURES SIGNALS</b>",
        f"🕐 {ts}",
        f"📊 {label}",
        "━━━━━━━━━━━━━━━━━━━━━",
        f"🟢 BUY   ✅ {bv}  ⏳ {bw}",
        f"🔴 SELL  ✅ {sv}  ⏳ {sw}",
        f"⚡ {total} symbols · {elapsed:.1f}s",
    ]
    body += _tg_fmt_signal_group(buy_valid,  "BUY",  wait=False)
    body += _tg_fmt_signal_group(sell_valid, "SELL", wait=False)
    body += _tg_fmt_signal_group(buy_wait,   "BUY",  wait=True)
    body += _tg_fmt_signal_group(sell_wait,  "SELL", wait=True)

    msg    = "\n".join(body)
    chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
    sent   = sum(1 for c in chunks if _tg_send_sync(c))
    return sent == len(chunks)


# ══════════════════════════════════════════════════════════════════════
#  BACKGROUND SCHEDULER  v57
#
#  Runs completely independently of any browser/Streamlit session.
#  • Started once per process via _start_bg_scheduler() (called from main()).
#  • Fires 15M + 5M scans every 15 min, clock-aligned to :00/:15/:30/:45 UTC.
#  • Uses a module-level market cache (_bg_cache) — no st.session_state.
#  • Deduplicates signals globally (_bg_seen) so each signal fires once.
#  • Sends Telegram via the existing _tg_send_signals() helper.
#  • Status readable by the UI via _bg_status dict (thread-safe via _bg_lock).
# ══════════════════════════════════════════════════════════════════════
import threading as _threading
import datetime  as _dt

_bg_cache: dict = {}          # module-level market/proxy cache
_bg_seen:  set  = set()       # dedup: (symbol, sig_ts_ms_str, mode)
_bg_lock   = _threading.Lock()
_bg_status: dict = {
    "last_run":     "Never",
    "last_mode":    "—",
    "last_signals": 0,
    "next_run":     "—",
    "running":      False,
    "error":        "",
}
_bg_thread_started = False


def _bg_next_quarter(now: float) -> float:
    """Return UTC unix timestamp of the next :00/:15/:30/:45 boundary."""
    import math
    return math.ceil(now / 900) * 900   # 900 s = 15 min


async def _bg_run_one_async(mode_key: str) -> dict:
    """
    Self-contained scan coroutine for the background thread.
    v58: Uses CryptoCompare for markets + klines — no proxy needed.
    """
    global _http_session

    cfg = MODES[mode_key]

    # Module-level market cache — survives across scans
    if "markets" not in _bg_cache:
        _bg_cache["markets"] = await _load_binance_futures_markets()

    ex = _FakeExchange(_bg_cache["markets"])

    # Shared aiohttp session
    _scan_connector = aiohttp.TCPConnector(
        limit=200, keepalive_timeout=30, ttl_dns_cache=600,
        resolver=aiohttp.ThreadedResolver(),
    )
    _scan_session = aiohttp.ClientSession(
        connector=_scan_connector,
        timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=30),
    )
    _http_session = _scan_session

    try:
        symbols = sorted([
            s for s, m in ex.markets.items()
            if m.get("type") == "swap" and m.get("active")
            and m.get("quote") == "USDT" and ":USDT" in s
        ])
        total = len(symbols)
        sem   = asyncio.Semaphore(MAX_CONCURRENT)
        state = {
            "s1_done": 0, "s2_in": 0, "s3_in": 0,
            "buy_valid": [], "sell_valid": [],
            "buy_wait":  [], "sell_wait":  [],
            "total": total,
        }

        async def worker(sym: str):
            r1 = await stage1_worker(ex, sem, sym, cfg)
            state["s1_done"] += 1
            if r1 is None:
                return
            want_sell, sym2, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r1
            state["s2_in"] += 1
            _loop = asyncio.get_running_loop()
            r2 = await _loop.run_in_executor(
                _CPU_POOL, stage2_worker,
                want_sell, sym2, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
            if r2 is None:
                return
            want_sell, sym2, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r2
            state["s3_in"] += 1
            r3 = await stage3_worker(
                ex, sem, sym2, want_sell, detail,
                pivot_ts, pivot_win_ts, pivot_end_ts, cfg, da)
            if r3:
                side, s3_sym, det2, pt = r3
                entry = (s3_sym, det2, pt)
                if   side == "BUY":       state["buy_valid"].append(entry)
                elif side == "SELL":      state["sell_valid"].append(entry)
                elif side == "WAIT_BUY":  state["buy_wait"].append(entry)
                elif side == "WAIT_SELL": state["sell_wait"].append(entry)

        await asyncio.gather(*[worker(s) for s in symbols], return_exceptions=True)
        return state

    finally:
        await ex.close()
        if not _scan_session.closed:
            await _scan_session.close()
        await _scan_connector.close()


def _bg_is_new(sym: str, det: str, mode: str) -> bool:
    """Return True and register the signal if it hasn't been sent before."""
    ts_m = _re.search(r"sig_ts_ms=(\d+)", det)
    key  = (sym, ts_m.group(1) if ts_m else det[:60], mode)
    with _bg_lock:
        if key in _bg_seen:
            return False
        _bg_seen.add(key)
        return True


def _bg_scheduler_loop() -> None:
    """
    Background thread main loop.
    Sleeps until the next :00/:15/:30/:45 UTC boundary, then runs
    15M and 5M scans back-to-back, sending Telegram for new signals.
    Runs forever; restarts after any error with a 60-s back-off.
    """
    global _bg_status

    print("[BG] Background scanner started — 24/7 Telegram alerts enabled.")

    # ── Startup ping ──────────────────────────────────────────────────
    try:
        _ping_ts = _dt.datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
        _next_ts = _dt.datetime.utcfromtimestamp(_bg_next_quarter(time.time())).strftime("%H:%M UTC")
        _tg_send_sync(
            f"🟢 <b>Binance Futures Scanner v58 — ONLINE</b>\n"
            f"🕐 {_ping_ts}\n"
            f"🤖 24/7 background scheduler started\n"
            f"📡 Data: CryptoCompare → BinanceFutures\n"
            f"⏱ First scan at {_next_ts} UTC"
        )
    except Exception as _pe:
        print(f"[BG] Startup ping failed (non-fatal): {_pe}")


    while True:
        try:
            # ── Sleep until next quarter-hour mark ──────────────────
            now    = time.time()
            next_t = _bg_next_quarter(now)
            wait   = max(next_t - now, 1.0)

            next_str = _dt.datetime.utcfromtimestamp(next_t).strftime("%H:%M UTC")
            with _bg_lock:
                _bg_status["next_run"] = next_str
                _bg_status["error"]    = ""

            print(f"[BG] Next scan at {next_str} — sleeping {wait:.0f}s")
            time.sleep(wait)

            # ── Run 15M then 5M ─────────────────────────────────────
            for mode_key in ("15m", "5m"):
                with _bg_lock:
                    _bg_status["running"]   = True
                    _bg_status["last_mode"] = mode_key.upper()

                print(f"[BG] Starting {mode_key.upper()} scan …")
                t0    = time.time()
                state = asyncio.run(_bg_run_one_async(mode_key))
                elapsed = time.time() - t0

                buy_valid  = [(s, d) for s, d, _ in state["buy_valid"]]
                sell_valid = [(s, d) for s, d, _ in state["sell_valid"]]
                buy_wait   = [(s, d) for s, d, _ in state["buy_wait"]]
                sell_wait  = [(s, d) for s, d, _ in state["sell_wait"]]
                total      = state["total"]

                # Keep only signals not already sent this session
                bv_new = [(s, d) for s, d in buy_valid  if _bg_is_new(s, d, mode_key)]
                sv_new = [(s, d) for s, d in sell_valid if _bg_is_new(s, d, mode_key)]
                bw_new = [(s, d) for s, d in buy_wait   if _bg_is_new(s, d, mode_key)]
                sw_new = [(s, d) for s, d in sell_wait  if _bg_is_new(s, d, mode_key)]
                n_new  = len(bv_new) + len(sv_new) + len(bw_new) + len(sw_new)

                print(f"[BG] {mode_key.upper()} done in {elapsed:.1f}s — "
                      f"{total} symbols · {n_new} new signal(s)")

                if n_new > 0:
                    _tg_send_signals(bv_new, sv_new, bw_new, sw_new,
                                     f"BG {mode_key.upper()}", elapsed, total)

                now_str = _dt.datetime.utcnow().strftime("%d %b %H:%M UTC")
                with _bg_lock:
                    _bg_status["last_run"]     = now_str
                    _bg_status["last_signals"] = n_new
                    _bg_status["running"]      = False

        except Exception as exc:
            err_str = str(exc)[:150]
            print(f"[BG] ⚠ Error: {err_str}")
            with _bg_lock:
                _bg_status["error"]   = err_str
                _bg_status["running"] = False
            # Invalidate market cache so next scan re-fetches
            _bg_cache.pop("markets", None)
            time.sleep(60)   # back off, then retry on next loop


@st.cache_resource
def _start_bg_scheduler() -> str:
    """
    Start the background scheduler thread exactly once per process.
    @st.cache_resource ensures this is called only once even across
    Streamlit's multiple-rerun model.
    Returns a status string for debug purposes.
    """
    t = _threading.Thread(
        target=_bg_scheduler_loop,
        name="bg-scanner",
        daemon=True,   # dies automatically if the process exits
    )
    t.start()
    return f"started pid={t.ident}"


# ══════════════════════════════════════════════════════════════════════
MAX_CONCURRENT   = 60     # CryptoCompare rate-limit friendly
RETRY_ATTEMPTS   = 3
RETRY_BASE_DELAY = 0.5
UI_THROTTLE_S    = 0.25

# ── CryptoCompare data aggregator (v58: replaces Binance FAPI) ───────────────
# CryptoCompare is UK/EU-based — NOT geo-blocked from Streamlit Cloud US servers.
# OHLCV is sourced with e=BinanceFutures, so prices match Binance exactly.
# Optional: add CC_API_KEY to Streamlit Secrets for higher rate-limits (free signup).
_CC_BASE    = "https://min-api.cryptocompare.com/data/v2"   # OHLCV endpoints (histominute/histohour/histoday)
_CC_BASE_V1 = "https://min-api.cryptocompare.com/data"       # non-v2 endpoints (exchange pairs list)

# ccxt TF string → (CryptoCompare endpoint, aggregate value)
_TF_TO_CC: dict = {
    "1m":  ("histominute",  1),
    "3m":  ("histominute",  3),
    "5m":  ("histominute",  5),
    "15m": ("histominute", 15),
    "30m": ("histominute", 30),
    "1h":  ("histohour",    1),
    "2h":  ("histohour",    2),
    "4h":  ("histohour",    4),
    "6h":  ("histohour",    6),
    "8h":  ("histohour",    8),
    "12h": ("histohour",   12),
    "1d":  ("histoday",     1),
}

def _get_cc_api_key() -> str:
    """Optional CryptoCompare API key from Streamlit Secrets / env."""
    try:
        return st.secrets.get("CC_API_KEY", "") or ""
    except Exception:
        return os.environ.get("CC_API_KEY", "") or ""


class _FakeExchange:
    """Minimal exchange wrapper holding the markets dict — no ccxt needed."""
    def __init__(self, markets: dict):
        self.markets = markets
        self.markets_by_id = {m["id"]: m for m in markets.values()}
    async def close(self):
        pass


async def _load_binance_futures_markets() -> dict:
    """
    Fetch all active Binance USDT perpetuals from CryptoCompare pair mapping.
    Returns a ccxt-style {symbol: market_info} dict.
    CryptoCompare endpoint: /data/all/exchanges/pairs?e=BinanceFutures&tsym=USDT
    """
    api_key = _get_cc_api_key()
    params: dict = {"e": "BinanceFutures", "tsym": "USDT"}
    if api_key:
        params["api_key"] = api_key

    connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        async with session.get(f"{_CC_BASE_V1}/all/exchanges/pairs", params=params) as resp:
            data = await resp.json(content_type=None)

    if data.get("Response") != "Success":
        raise RuntimeError(
            f"CryptoCompare pair list failed: {data.get('Message', 'unknown')}"
        )

    markets: dict = {}
    exchange_data = data.get("Data", {}).get("BinanceFutures", {})
    for base_sym, quote_dict in exchange_data.items():
        if "USDT" not in quote_dict:
            continue
        sym = f"{base_sym}/USDT:USDT"
        markets[sym] = {
            "type":   "swap",
            "active": True,
            "quote":  "USDT",
            "settle": "USDT",
            "id":     f"{base_sym}USDT",
            "base":   base_sym,
        }

    if not markets:
        raise RuntimeError("No USDT perpetual pairs returned by CryptoCompare")
    return markets


# Module-level aiohttp session — created once per scan/debug, shared by all fetchers.
_http_session: Optional[aiohttp.ClientSession] = None

KC_LEN        = 20
KC_MULT       = 2.0
KC_ATR_LEN    = 10
TDI_RSI_P     = 11
TDI_FAST      = 2
TDI_SLOW      = 11
SWING_ALT     = 5
SWING_UTAMA   = 50
LOOKBACK_SIG  = 100
# ── KWV (KVO + Weis Wave + MFI) — Pine Script exact replica ──────────────
KVO_FAST     = 21      # Klinger fast EMA length
KVO_SLOW     = 55      # Klinger slow EMA length
WEIS_LEN     = 2       # Weis Wave trend detection length
MFI_LEN      = 14      # Money Flow Index period
MFI_OB       = 70      # MFI overbought threshold
MFI_OS       = 30      # MFI oversold  threshold
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
        "choch_limit":      650,
        # v38: pivot age gate — pivot_confirmed_ts must be within this window
        "pivot_max_age_ms": 48 * 3_600_000,   # 48 hours
        "label":       "15M — Daily → 4H → 1H → 15M",
    },
    "5m": {
        "pivot_tf":    "4h",
        "tdi_tf":      "1h",
        "mid_tf":      "15m",
        "sig_tf":      "5m",
        # v13: BOS/ChoCh validated on 1m
        "choch_tf":    "1m",
        "choch_limit":      550,
        # v38: pivot age gate — pivot_confirmed_ts must be within this window
        "pivot_max_age_ms": 8 * 3_600_000,    # 8 hours
        "label":       "5M — 4H → 1H → 15M → 5M",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  TIMEZONES  — label → UTC offset in fractional hours
# ══════════════════════════════════════════════════════════════════════
TIMEZONES: dict[str, float] = {
    "UTC+0  — UTC / GMT":          0.0,
    "UTC+1  — London DST / CET":   1.0,
    "UTC+2  — EET / CEST":         2.0,
    "UTC+3  — Moscow / Istanbul":  3.0,
    "UTC+3:30 — Tehran":           3.5,
    "UTC+4  — Dubai / Baku":       4.0,
    "UTC+4:30 — Kabul":            4.5,
    "UTC+5  — Karachi / PKT":      5.0,
    "UTC+5:30 — India / IST":      5.5,
    "UTC+5:45 — Kathmandu / NPT":  5.75,
    "UTC+6  — Dhaka / BST":        6.0,
    "UTC+6:30 — Yangon / MMT":     6.5,
    "UTC+7  — Bangkok / WIB":      7.0,
    "UTC+8  — Singapore / HKT":    8.0,
    "UTC+9  — Tokyo / KST":        9.0,
    "UTC+9:30 — Adelaide / ACST":  9.5,
    "UTC+10 — Sydney / AEST":     10.0,
    "UTC+11 — Magadan / AEDT":    11.0,
    "UTC+12 — Auckland / NZST":   12.0,
    "UTC-1  — Azores / CVT":      -1.0,
    "UTC-2  — South Georgia":     -2.0,
    "UTC-3  — Brasília / ART":    -3.0,
    "UTC-3:30 — Newfoundland":    -3.5,
    "UTC-4  — EDT / AST":         -4.0,
    "UTC-5  — CDT / EST":         -5.0,
    "UTC-6  — MDT / CST":         -6.0,
    "UTC-7  — PDT / MST":         -7.0,
    "UTC-8  — PST / AKDT":        -8.0,
    "UTC-9  — AKST / GIT":        -9.0,
    "UTC-10 — Hawaii / HST":     -10.0,
    "UTC-11 — Samoa / NUT":      -11.0,
    "UTC-12 — IDLW / BIT":       -12.0,
}
TZ_LABELS  = list(TIMEZONES.keys())
TZ_DEFAULT = "UTC+0  — UTC / GMT"


TIME_FMTS   = ["24h", "12h"]
TIME_FMT_DEFAULT = "24h"


def _fmt_ts(ms: int, tz_h: float, tz_label: str, time_fmt: str = "24h") -> str:
    """Convert a UTC epoch-millisecond timestamp to a local time string.
    time_fmt: '24h' → HH:MM  |  '12h' → H:MM AM/PM
    """
    total_min = int(tz_h * 60)
    delta = datetime.timedelta(minutes=total_min)
    dt    = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).replace(tzinfo=None) + delta
    sign  = "+" if tz_h >= 0 else "-"
    ah    = int(abs(tz_h))
    am    = int(round((abs(tz_h) - ah) * 60))
    tz_str = f"UTC{sign}{ah:02d}:{am:02d}" if am else f"UTC{sign}{ah}"
    if time_fmt == "12h":
        return dt.strftime(f"%Y-%m-%d %I:%M %p {tz_str}").replace(" 0", " ")
    return dt.strftime(f"%Y-%m-%d %H:%M {tz_str}")

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Binance Futures Scanner v58",
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
    -webkit-text-size-adjust: 100%;
  }
  [data-testid="stHeader"],
  [data-testid="stToolbar"]         { display: none !important; }
  section[data-testid="stSidebar"]  { display: none !important; }
  .main .block-container,
  [data-testid="stMainBlockContainer"] {
    padding: 0.85rem 1.2rem 5rem !important;
    padding-bottom: max(5rem, calc(5rem + env(safe-area-inset-bottom))) !important;
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
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0a0a18 0%, #0c0c1e 45%, #07070f 100%);
    border: 1px solid rgba(0,180,216,0.18);
    border-radius: 14px;
    padding: 1.5rem 1.8rem 1.3rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.9rem;
    box-shadow: 0 4px 40px rgba(0,0,0,0.6), 0 0 60px rgba(0,100,180,0.06) inset;
  }
  /* Top-edge accent line */
  .sc-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
      transparent 0%,
      rgba(0,180,216,0.6) 20%,
      rgba(0,230,118,0.5) 50%,
      rgba(0,180,216,0.6) 80%,
      transparent 100%);
    border-radius: 14px 14px 0 0;
  }
  /* Subtle radial glow behind logo */
  .sc-header::after {
    content: '';
    position: absolute;
    top: -40px; left: -40px;
    width: 240px; height: 160px;
    background: radial-gradient(ellipse, rgba(0,144,200,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .sc-header-left { display: flex; flex-direction: column; gap: 5px; z-index: 1; }
  .sc-header h1 {
    font-family: var(--mono);
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: -0.04em;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,180,216,0.25);
  }
  .sc-header h1 .ico { font-style: normal; margin-right: 6px; }
  .sc-header h1 .brand { color: #e8f4ff; }
  .sc-header h1 .accent { color: var(--blue); }
  .sc-header .sub {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    line-height: 1;
  }
  .sc-header .sub .dot { margin: 0 5px; color: rgba(0,180,216,0.35); }
  .sc-header-right {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    z-index: 1;
  }
  .sc-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--mono);
    letter-spacing: 0.05em;
    border: 1px solid;
    white-space: nowrap;
  }
  .sc-badge.blue  {
    background: linear-gradient(135deg, rgba(0,140,200,0.15), rgba(0,100,160,0.08));
    color: var(--blue);
    border-color: rgba(0,180,216,0.35);
    box-shadow: 0 0 10px rgba(0,180,216,0.1);
  }
  .sc-badge.green {
    background: linear-gradient(135deg, rgba(0,200,100,0.13), rgba(0,160,80,0.06));
    color: var(--green);
    border-color: rgba(0,230,118,0.3);
    box-shadow: 0 0 10px rgba(0,230,118,0.08);
  }
  .sc-badge.gold  {
    background: linear-gradient(135deg, rgba(220,170,0,0.15), rgba(180,130,0,0.07));
    color: var(--gold);
    border-color: rgba(255,202,40,0.3);
    box-shadow: 0 0 10px rgba(255,202,40,0.08);
  }
  /* Live clock badge */
  .sc-badge.clock {
    background: rgba(255,255,255,0.03);
    color: var(--text2);
    border-color: var(--border2);
    font-size: 0.7rem;
    cursor: default;
  }
  /* Timezone badge — shows active TZ in header */
  .sc-tz-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--mono);
    letter-spacing: 0.05em;
    background: rgba(255,202,40,0.07);
    color: var(--gold);
    border: 1px solid rgba(255,202,40,0.28);
    white-space: nowrap;
  }
  /* Streamlit selectbox within TZ control row */
  .sc-tz-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.2rem;
    flex-wrap: wrap;
  }
  .sc-tz-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    white-space: nowrap;
  }

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
    padding: 0.65rem 1.1rem;
    margin: 0.5rem 0 0.7rem;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem 0.8rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }
  .sc-summary .ss-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text);
    white-space: nowrap;
  }
  .sc-summary .ss-title span { color: var(--green); }
  .sc-summary .ss-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: var(--mono);
    white-space: nowrap;
    border: 1px solid;
  }
  .ss-chip.g  { background: var(--green-bg);  color: var(--green); border-color: var(--green-border); }
  .ss-chip.gd { background: rgba(0,230,118,0.04); color: #50c878; border-color: rgba(80,200,120,0.2); }
  .ss-chip.r  { background: var(--red-bg);    color: var(--red);   border-color: var(--red-border); }
  .ss-chip.rd { background: rgba(255,64,96,0.04); color: #e05060; border-color: rgba(200,80,96,0.2); }
  .sc-summary .ss-meta {
    margin-left: auto;
    font-size: 0.7rem;
    color: var(--muted);
    font-family: var(--mono);
    white-space: nowrap;
  }
  .sc-summary .ss-meta b { color: var(--gold); }

  /* ── Signal cards ────────────────────────────────────────────────── */
  .sc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 9px;
    margin: 0.3rem 0 0.5rem;
  }

  /* ── Two-column All layout ──────────────────────────────────────── */
  .sc-all-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 0.3rem 0 0.5rem;
  }
  .sc-col-header {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    padding: 5px 10px;
    border-radius: 6px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sc-col-header.confirmed {
    background: rgba(0,230,118,0.07);
    color: var(--green);
    border: 1px solid var(--green-border);
  }
  .sc-col-header.waiting {
    background: rgba(255,180,0,0.07);
    color: #ffaa00;
    border: 1px dashed rgba(255,170,0,0.35);
  }
  .sc-col-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 7px;
  }

  /* ═══════ CONFIRMED cards — vibrant, glowing ══════════════════════ */
  .sc-card {
    border-radius: 10px;
    border: 1px solid var(--border2);
    background: var(--surface);
    padding: 0.6rem 0.72rem 0.55rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
    position: relative;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
    overflow: hidden;
  }
  .sc-card:hover, .sc-card:active {
    transform: translateY(-3px) scale(1.02);
  }

  /* BUY confirmed — vivid green glow + top stripe */
  .sc-card.buy {
    border-left: 3px solid var(--green);
    background: linear-gradient(135deg, rgba(0,230,118,0.06) 0%, rgba(15,15,21,1) 60%);
    box-shadow: 0 0 0 0 rgba(0,230,118,0);
  }
  .sc-card.buy::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--green), transparent);
    border-radius: 10px 10px 0 0;
  }
  .sc-card.buy:hover, .sc-card.buy:active {
    border-color: var(--green);
    background: linear-gradient(135deg, rgba(0,230,118,0.13) 0%, rgba(15,15,21,1) 65%);
    box-shadow: 0 6px 28px rgba(0,230,118,0.22), 0 2px 8px rgba(0,0,0,0.4);
  }

  /* SELL confirmed — vivid red glow + top stripe */
  .sc-card.sell {
    border-left: 3px solid var(--red);
    background: linear-gradient(135deg, rgba(255,64,96,0.06) 0%, rgba(15,15,21,1) 60%);
    box-shadow: 0 0 0 0 rgba(255,64,96,0);
  }
  .sc-card.sell::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--red), transparent);
    border-radius: 10px 10px 0 0;
  }
  .sc-card.sell:hover, .sc-card.sell:active {
    border-color: var(--red);
    background: linear-gradient(135deg, rgba(255,64,96,0.13) 0%, rgba(15,15,21,1) 65%);
    box-shadow: 0 6px 28px rgba(255,64,96,0.22), 0 2px 8px rgba(0,0,0,0.4);
  }

  /* ═══════ WAIT cards — amber dashed, clearly "pending" ════════════ */
  .sc-card.wait {
    border: 1px dashed rgba(255,170,0,0.3) !important;
    border-left: none !important;
    border-left-width: 0 !important;
    background: rgba(20,18,10,0.9);
    opacity: 1;
    position: relative;
    overflow: hidden;
  }
  .sc-card.wait::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: repeating-linear-gradient(
      to bottom,
      #ffaa00 0px, #ffaa00 5px,
      transparent 5px, transparent 9px
    );
    border-radius: 2px 0 0 2px;
  }
  .sc-card.wait:hover, .sc-card.wait:active {
    border-color: rgba(255,170,0,0.55) !important;
    background: rgba(30,25,8,0.95);
    box-shadow: 0 6px 22px rgba(255,160,0,0.14), 0 2px 8px rgba(0,0,0,0.4);
    transform: translateY(-2px) scale(1.015);
  }
  .sc-card.wait .sc-card-sym { color: #c8b070; }
  .sc-card.wait .sc-card-price { color: #c89a30; }
  .sc-card.wait .sc-card-info { color: #7a6840; }
  .sc-card.wait .sc-card-info b { color: #9a8855; }

  /* Hover effect on summary banner chips */
  .ss-chip {
    cursor: default;
    transition: transform 0.12s, box-shadow 0.12s, filter 0.12s;
  }
  .ss-chip:hover { transform: translateY(-1px); filter: brightness(1.2); box-shadow: 0 3px 10px rgba(0,0,0,0.3); }

  .sc-card-row1 {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
  }
  .sc-card-sym {
    font-family: var(--mono);
    font-size: 0.97rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1;
  }
  .sc-card-dir {
    font-size: 0.65rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    white-space: nowrap;
  }
  /* Confirmed direction badges — bright */
  .dir-buy  {
    background: linear-gradient(135deg, rgba(0,230,118,0.18), rgba(0,200,100,0.1));
    color: var(--green-hi);
    border: 1px solid rgba(0,230,118,0.4);
    text-shadow: 0 0 8px rgba(0,230,118,0.5);
  }
  .dir-sell {
    background: linear-gradient(135deg, rgba(255,64,96,0.18), rgba(220,40,70,0.1));
    color: var(--red-hi);
    border: 1px solid rgba(255,64,96,0.4);
    text-shadow: 0 0 8px rgba(255,64,96,0.5);
  }
  /* Wait direction badges — muted amber */
  .dir-buy-w  {
    background: rgba(255,170,0,0.08);
    color: #c8902a;
    border: 1px dashed rgba(200,145,40,0.35);
    letter-spacing: 0.04em;
  }
  .dir-sell-w {
    background: rgba(255,120,0,0.08);
    color: #c87030;
    border: 1px dashed rgba(200,110,45,0.35);
    letter-spacing: 0.04em;
  }

  /* Confirmed price — glowing gold */
  .sc-card.buy  .sc-card-price,
  .sc-card.sell .sc-card-price {
    font-family: var(--mono);
    font-size: 1.0rem;
    font-weight: 700;
    color: #ffd760;
    text-shadow: 0 0 12px rgba(255,210,60,0.35);
    line-height: 1.1;
  }
  .sc-card-price {
    font-family: var(--mono);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1.1;
  }
  .sc-card-info {
    font-family: var(--mono);
    font-size: 0.67rem;
    color: var(--muted);
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sc-card-info b { color: var(--text2); font-weight: 600; }

  /* Row 4 — ADX · age · kind */
  .sc-card-meta {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-top: 2px;
    flex-wrap: nowrap;
    overflow: hidden;
  }
  .sc-adx {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    white-space: nowrap;
    background: rgba(255,255,255,0.04);
    color: var(--muted);
  }
  .sc-adx.adx-hi  { background: rgba(0,230,118,0.1);  color: var(--green); }
  .sc-adx.adx-med { background: rgba(255,202,40,0.1); color: var(--gold); }
  .sc-adx.adx-lo  { background: rgba(255,255,255,0.04); color: var(--muted); }
  .sc-age {
    font-family: var(--mono);
    font-size: 0.60rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
  }
  .sc-kind-badge {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    white-space: nowrap;
    letter-spacing: 0.04em;
    flex-shrink: 0;
  }
  .sc-kind-badge.qm  { background: rgba(0,180,216,0.1); color: var(--blue); border: 1px solid rgba(0,180,216,0.2); }
  .sc-kind-badge.mtf { background: rgba(167,139,250,0.1); color: #a78bfa; border: 1px solid rgba(167,139,250,0.25); }

  /* Summary funnel display */
  .ss-funnel {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    padding: 2px 9px;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    white-space: nowrap;
  }

  /* Pulse dot for confirmed cards */
  .sc-card-pulse {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-left: 4px;
    vertical-align: middle;
    animation: pulse-ring 1.8s ease-out infinite;
  }
  .sc-card.buy  .sc-card-pulse { background: var(--green); box-shadow: 0 0 4px var(--green); }
  .sc-card.sell .sc-card-pulse { background: var(--red);   box-shadow: 0 0 4px var(--red); }
  @keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 1; }
    60%  { transform: scale(1.5); opacity: 0.4; }
    100% { transform: scale(1);   opacity: 1; }
  }

  /* ── Wait section label ─────────────────────────────────────────── */
  .sc-wait-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7a6840;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(255,170,0,0.06);
    border: 1px dashed rgba(255,170,0,0.2);
    margin: 0.6rem 0 0.4rem;
    display: inline-block;
  }

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

  /* ════════════════════════════════════════════════════════════════
     MOBILE  —  comprehensive mobile-first responsive overrides
  ════════════════════════════════════════════════════════════════ */

  /* ── Tablet (≤ 900px) ───────────────────────────────────────── */
  @media (max-width: 900px) {
    .sc-tf-flow {
      flex-wrap: nowrap;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      scroll-snap-type: x mandatory;
    }
    .sc-tf-node {
      min-width: 72px;
      scroll-snap-align: start;
    }
    .sc-grid {
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important;
    }
    .sc-all-layout { grid-template-columns: 1fr !important; }
  }

  /* ── Mobile (≤ 640px) ───────────────────────────────────────── */
  @media (max-width: 640px) {
    /* ── Base padding ─────────────────────────── */
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
      padding: 0.4rem 0.45rem !important;
      padding-bottom: max(5.5rem, calc(5.5rem + env(safe-area-inset-bottom))) !important;
    }

    /* ── Ensure all Streamlit columns stack ───── */
    [data-testid="stHorizontalBlock"] {
      flex-direction: column !important;
      gap: 0.4rem !important;
    }
    [data-testid="stColumn"] {
      width: 100% !important;
      flex: 1 1 100% !important;
      min-width: 0 !important;
    }

    /* ── Touch targets — 48px min (raised from 44px) ─ */
    .stButton > button,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"],
    [data-testid="stDownloadButton"] > button {
      min-height: 48px !important;
      font-size: 0.88rem !important;
      padding: 0.65rem 0.8rem !important;
    }

    /* ── Header ───────────────────────────────── */
    .sc-header {
      padding: 0.8rem 0.9rem 0.75rem !important;
      flex-direction: column !important;
      align-items: flex-start !important;
      gap: 0.45rem !important;
      border-radius: 10px !important;
    }
    .sc-header h1 {
      font-size: 1.1rem !important;
      line-height: 1.2 !important;
    }
    .sc-header .sub { font-size: 0.56rem !important; letter-spacing: 0.07em !important; }
    .sc-header-right {
      gap: 4px !important;
      flex-wrap: wrap !important;
      width: 100% !important;
    }
    .sc-badge        { font-size: 0.6rem !important; padding: 3px 7px !important; }
    .sc-tz-badge     { font-size: 0.6rem !important; padding: 3px 7px !important; }

    /* ── Tabs — sticky at top with blur ──────── */
    .stTabs [data-baseweb="tab-list"] {
      position: sticky !important;
      top: 0 !important;
      z-index: 100 !important;
      backdrop-filter: blur(12px) !important;
      -webkit-backdrop-filter: blur(12px) !important;
      overflow-x: auto !important;
      flex-wrap: nowrap !important;
      -webkit-overflow-scrolling: touch !important;
      padding: 4px !important;
      scrollbar-width: none !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
    .stTabs [data-baseweb="tab"] {
      font-size: 0.8rem !important;
      padding: 0.5rem 0.85rem !important;
      white-space: nowrap !important;
      min-height: 40px !important;
    }

    /* ── Mode selector cards ──────────────────── */
    .sc-mode-selector {
      flex-direction: column !important;
      gap: 6px !important;
    }
    .sc-mode-selector div[data-testid="stButton"] > button {
      height: auto !important;
      min-height: 54px !important;
      font-size: 0.84rem !important;
    }

    /* ── TF pipeline flow ────────────────────── */
    .sc-tf-flow {
      overflow-x: auto !important;
      -webkit-overflow-scrolling: touch !important;
      scroll-snap-type: x mandatory !important;
      border-radius: 8px !important;
    }
    .sc-tf-node {
      min-width: 62px !important;
      padding: 0.55rem 0.3rem 0.5rem !important;
      scroll-snap-align: start;
    }
    .sc-tf-node .tf-val   { font-size: 0.85rem !important; }
    .sc-tf-node .tf-stage { font-size: 0.48rem !important; }
    .sc-tf-node .tf-role  { font-size: 0.5rem !important; }

    /* ── Rule pills ───────────────────────────── */
    .sc-pills-v2 { gap: 4px !important; }
    .sc-pill-v2  { font-size: 0.68rem !important; padding: 4px 8px 4px 5px !important; }
    .pill-num-v2 { width: 15px !important; height: 15px !important; font-size: 0.58rem !important; }

    /* ── Sort bar — now a selectbox, no override needed ──────── */

    /* ── Signal cards — 2-per-row ────────────── */
    .sc-grid {
      grid-template-columns: repeat(2, 1fr) !important;
      gap: 6px !important;
    }
    .sc-col-grid {
      grid-template-columns: repeat(2, 1fr) !important;
      gap: 5px !important;
    }
    .sc-card {
      padding: 0.5rem 0.55rem 0.45rem !important;
      border-radius: 8px !important;
    }
    /* Stronger touch feedback on mobile */
    .sc-card:active {
      transform: scale(0.97) !important;
      filter: brightness(1.15) !important;
      transition: transform 0.08s, filter 0.08s !important;
    }
    .sc-card-sym   { font-size: 0.88rem !important; }
    .sc-card-price { font-size: 0.92rem !important; }
    .sc-card-info  { font-size: 0.6rem !important; }
    .sc-card-dir   { font-size: 0.6rem !important; padding: 2px 5px !important; }
    .sc-card-meta  { gap: 3px !important; }
    .sc-adx        { font-size: 0.56rem !important; }
    .sc-age        { font-size: 0.54rem !important; }
    .sc-kind-badge { font-size: 0.52rem !important; }

    /* ── Summary banner ───────────────────────── */
    .sc-summary {
      padding: 0.55rem 0.7rem !important;
      gap: 0.3rem 0.5rem !important;
    }
    .sc-summary .ss-title { font-size: 0.75rem !important; }
    .sc-summary .ss-chip  { font-size: 0.65rem !important; padding: 2px 7px !important; }
    .sc-summary .ss-meta  { margin-left: 0 !important; width: 100% !important; }

    /* ── Live counters — 3-column ─────────────── */
    .sc-counters {
      grid-template-columns: repeat(3, 1fr) !important;
      gap: 5px !important;
    }
    .sc-cnt        { padding: 0.55rem 0.3rem 0.45rem !important; }
    .sc-cnt .cnt-val { font-size: 1.45rem !important; }
    .sc-cnt .cnt-lbl { font-size: 0.55rem !important; line-height: 1.1 !important; }
    .sc-cnt .cnt-sub { font-size: 0.52rem !important; }

    /* ── Settings panel ───────────────────────── */
    .sc-settings-panel { padding: 0.7rem 0.7rem 0.55rem !important; }

    /* ── Proxy banner ─────────────────────────── */
    .sc-proxy-ok, .sc-proxy-err {
      font-size: 0.75rem !important;
      padding: 0.5rem 0.7rem !important;
      flex-wrap: wrap !important;
      word-break: break-word !important;
    }

    /* ── Section label ────────────────────────── */
    .sc-section-label { font-size: 0.55rem !important; margin: 0.15rem 0 0.4rem !important; }

    /* ── All-section stacked headers ─────────── */
    .sc-col-header { font-size: 0.6rem !important; padding: 4px 8px !important; }

    /* ── Debug pipeline info ──────────────────── */
    .sc-pipeline-info { font-size: 0.75rem !important; padding: 0.7rem 0.8rem !important; line-height: 1.9 !important; }
    .sc-stage-dot     { width: 17px !important; height: 17px !important; font-size: 0.58rem !important; line-height: 17px !important; }

    /* ── Expander ─────────────────────────────── */
    [data-testid="stExpander"] summary {
      font-size: 0.82rem !important;
      padding: 0.5rem 0.7rem !important;
      min-height: 44px !important;
    }

    /* ── Export buttons always full-width + stacked ── */
    [data-testid="stDownloadButton"] {
      width: 100% !important;
    }
    [data-testid="stDownloadButton"] > button {
      width: 100% !important;
      min-height: 48px !important;
    }

    /* ── DataFrames — prevent horizontal overflow ─ */
    [data-testid="stDataFrame"] {
      max-width: 100% !important;
      overflow-x: auto !important;
    }

    /* ── Text inputs — 16px prevents iOS auto-zoom on focus ────── */
    .stTextInput input {
      font-size: 16px !important;
      min-height: 44px !important;
    }
  }

  /* ── Very small phones (≤ 390px) ────────────────────────────── */
  @media (max-width: 390px) {
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
      padding: 0.3rem 0.3rem !important;
      padding-bottom: max(5.5rem, calc(5.5rem + env(safe-area-inset-bottom))) !important;
    }
    .sc-header h1 { font-size: 0.98rem !important; }
    .sc-header .sub { display: none !important; }
    .sc-header-right { display: none !important; }
    .sc-badge { display: none !important; }
    .sc-tz-badge { font-size: 0.56rem !important; }
    .sc-tf-node .tf-stage,
    .sc-tf-node .tf-role  { display: none !important; }
    .sc-tf-node .tf-val   { font-size: 0.78rem !important; }
    .sc-grid  { grid-template-columns: 1fr 1fr !important; gap: 5px !important; }
    .sc-card-sym   { font-size: 0.82rem !important; }
    .sc-card-price { font-size: 0.84rem !important; }
    .sc-counters { grid-template-columns: repeat(3, 1fr) !important; }
    .stTabs [data-baseweb="tab"] {
      font-size: 0.72rem !important;
      padding: 0.4rem 0.6rem !important;
    }
  }

  /* ── Scan config section label ───────────────────────────────────── */
  .sc-section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.62rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--muted);
    margin: 0.25rem 0 0.55rem;
  }
  .sc-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border2), transparent);
  }

  /* ── Mode selector — Streamlit buttons styled as cards ──────────── */
  :root {
    --c15: #00d4ff;   /* 15M — electric cyan  */
    --c15b: rgba(0,212,255,0.12);
    --c15g: rgba(0,212,255,0.22);
    --c5:  #ff6b35;   /* 5M  — fire orange    */
    --c5b:  rgba(255,107,53,0.12);
    --c5g:  rgba(255,107,53,0.25);
  }
  .sc-mode-selector {
    display: flex;
    gap: 10px;
    margin-bottom: 0.65rem;
    align-items: stretch;
  }
  .sc-mode-selector > div[data-testid="stColumn"] {
    flex: 1;
    min-width: 0;
  }
  /* Shared base card look */
  .sc-mode-selector div[data-testid="stButton"] > button {
    width: 100% !important;
    height: 90px !important;
    padding: 0.75rem 1rem !important;
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--border2) !important;
    background: var(--surface) !important;
    color: var(--text2) !important;
    font-family: var(--body) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    white-space: pre-wrap !important;
    line-height: 1.5 !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease,
                background 0.18s ease, transform 0.12s ease !important;
    -webkit-tap-highlight-color: transparent !important;
  }
  /* Top accent bar */
  .sc-mode-selector div[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
    opacity: 0;
    transition: opacity 0.18s;
  }

  /* ── 15M card — cyan theme ── */
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button::before {
    background: linear-gradient(90deg, #00d4ff, #00ffb3);
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button:hover,
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button:focus {
    border-color: var(--c15g) !important;
    background: linear-gradient(135deg, var(--c15b) 0%, var(--surface) 70%) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.2),
                0 6px 28px rgba(0,212,255,0.2),
                0 2px 8px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) scale(1.015) !important;
    color: var(--c15) !important;
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button:hover::before,
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button:focus::before { opacity: 1; }
  /* 15M active */
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button[data-testid="baseButton-primary"] {
    border-color: rgba(0,212,255,0.5) !important;
    background: linear-gradient(135deg, var(--c15b) 0%, var(--surface) 70%) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.15),
                0 4px 22px rgba(0,212,255,0.18) !important;
    color: var(--c15) !important;
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(1) button[data-testid="baseButton-primary"]::before { opacity: 1; }

  /* ── 5M card — orange theme ── */
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button::before {
    background: linear-gradient(90deg, #ff6b35, #ffca28);
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button:hover,
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button:focus {
    border-color: var(--c5g) !important;
    background: linear-gradient(135deg, var(--c5b) 0%, var(--surface) 70%) !important;
    box-shadow: 0 0 0 1px rgba(255,107,53,0.2),
                0 6px 28px rgba(255,107,53,0.22),
                0 2px 8px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) scale(1.015) !important;
    color: var(--c5) !important;
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button:hover::before,
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button:focus::before { opacity: 1; }
  /* 5M active */
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button[data-testid="baseButton-primary"] {
    border-color: rgba(255,107,53,0.5) !important;
    background: linear-gradient(135deg, var(--c5b) 0%, var(--surface) 70%) !important;
    box-shadow: 0 0 0 1px rgba(255,107,53,0.15),
                0 4px 22px rgba(255,107,53,0.2) !important;
    color: var(--c5) !important;
  }
  .sc-mode-selector [data-testid="stColumn"]:nth-child(2) button[data-testid="baseButton-primary"]::before { opacity: 1; }

  /* Touch press flash — shared */
  .sc-mode-selector div[data-testid="stButton"] > button:active {
    transform: translateY(0px) scale(0.97) !important;
    transition: transform 0.06s ease, box-shadow 0.06s ease !important;
  }

  /* ── TF pipeline flow ────────────────────────────────────────────── */
  .sc-tf-flow {
    display: flex;
    align-items: stretch;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    overflow: hidden;
    position: relative;
    margin-bottom: 0.6rem;
  }
  .sc-tf-flow::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
      #a78bfa 0%, #60a5fa 25%, #00b4d8 50%, #34d399 75%, #f87171 100%);
  }
  .sc-tf-node {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0.7rem 0.4rem 0.65rem;
    gap: 3px;
    border-right: 1px solid var(--border);
    position: relative;
    transition: background 0.15s;
  }
  .sc-tf-node:last-child { border-right: none; }
  .sc-tf-node:hover { background: rgba(255,255,255,0.02); }
  .sc-tf-node .tf-stage {
    font-size: 0.55rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    line-height: 1;
  }
  .sc-tf-node .tf-val {
    font-family: var(--mono);
    font-size: 1.0rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.02em;
  }
  .sc-tf-node .tf-role {
    font-size: 0.58rem;
    color: var(--muted);
    line-height: 1;
    white-space: nowrap;
  }
  .sc-tf-node:nth-child(1) .tf-val { color: #a78bfa; }
  .sc-tf-node:nth-child(2) .tf-val { color: #60a5fa; }
  .sc-tf-node:nth-child(3) .tf-val { color: #00b4d8; }
  .sc-tf-node:nth-child(4) .tf-val { color: #34d399; }
  .sc-tf-node:nth-child(5) .tf-val { color: #f87171; }
  /* Arrow connector between nodes */
  .sc-tf-arrow {
    display: flex;
    align-items: center;
    padding: 0 0;
    color: var(--border2);
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    width: 0;
    overflow: visible;
    position: relative;
    z-index: 2;
  }

  /* ── Enhanced rule pills ─────────────────────────────────────────── */
  .sc-pills-v2 {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0 0 0.75rem;
  }
  .sc-pill-v2 {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 8px;
    padding: 5px 12px 5px 6px;
    font-size: 0.76rem;
    color: var(--text2);
    font-family: var(--mono);
    white-space: nowrap;
    border: 1px solid var(--border2);
    background: var(--surface);
    position: relative;
    overflow: hidden;
    transition: border-color 0.15s, background 0.15s;
  }
  .sc-pill-v2::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
  }
  .sc-pill-v2.s1::before { background: #a78bfa; }
  .sc-pill-v2.s2::before { background: #60a5fa; }
  .sc-pill-v2.s3::before { background: #34d399; }
  .sc-pill-v2.s4::before { background: #f87171; }
  .sc-pill-v2:hover { background: var(--surface2); }
  .pill-num-v2 {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 5px;
    font-size: 0.65rem;
    font-weight: 800;
    flex-shrink: 0;
  }
  .sc-pill-v2.s1 .pill-num-v2 { background: rgba(167,139,250,0.15); color: #a78bfa; }
  .sc-pill-v2.s2 .pill-num-v2 { background: rgba(96,165,250,0.15);  color: #60a5fa; }
  .sc-pill-v2.s3 .pill-num-v2 { background: rgba(52,211,153,0.15);  color: #34d399; }
  .sc-pill-v2.s4 .pill-num-v2 { background: rgba(248,113,113,0.15); color: #f87171; }
  .pill-arr-v2 { color: var(--border2); font-weight: 700; }

  /* ── Settings panel (gear toggle) ───────────────────────────────── */  .sc-settings-panel {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem 0.7rem;
    margin-bottom: 0.65rem;
    position: relative;
    animation: settings-slide 0.18s ease;
  }
  .sc-settings-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, rgba(0,180,216,0.5), rgba(167,139,250,0.4), transparent);
    border-radius: 12px 12px 0 0;
  }
  @keyframes settings-slide {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Sort pill buttons ───────────────────────────────────────────── */
  .sc-sort-wrap [data-testid="stButton"] > button {
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.7rem !important;
    min-height: 34px !important;
    height: 34px !important;
    font-family: var(--mono) !important;
  }
  @media (max-width: 640px) {
    .sc-sort-wrap [data-testid="stButton"] > button {
      font-size: 0.7rem !important;
      padding: 0.3rem 0.4rem !important;
      min-height: 38px !important;
      height: 38px !important;
    }
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  API KEY HELPER  — CryptoCompare (optional, improves rate limits)
# ══════════════════════════════════════════════════════════════════════
#  To add a free API key:
#    1. Register at https://www.cryptocompare.com/cryptopian/api-keys
#    2. In Streamlit → app → ⋮ → Settings → Secrets, add:
#       CC_API_KEY = "your-key-here"
#  Without a key, the free public tier still works but at lower rate limits.


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR MATH  — NumPy-vectorized
# ══════════════════════════════════════════════════════════════════════

def _rma(a: np.ndarray, p: int) -> np.ndarray:
    if len(a) < p:
        return np.full(len(a), np.nan)
    return pd.Series(a).ewm(alpha=1.0 / p, adjust=False, ignore_na=False).mean().values

def _sma(a: np.ndarray, p: int) -> np.ndarray:
    # ⚡ v43: np.convolve replaces pd.Series.rolling — 7-16x faster at all bar counts.
    # convolve(valid) gives the sum of each p-window; divide by p for mean.
    # Prepend NaN for the first p-1 bars to preserve the min_periods=p semantics.
    out = np.full(len(a), np.nan)
    if len(a) >= p:
        out[p - 1:] = np.convolve(a, np.ones(p) / p, mode='valid')
    return out

def _ema(a: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(a).ewm(span=p, adjust=False).mean().values

def _np_ffill(a: np.ndarray, leading_fill=np.nan) -> np.ndarray:
    """
    ⚡ v55: NumPy forward-fill — replaces pd.Series.ffill().
    Propagates the last non-NaN value forward; elements before the first
    non-NaN are set to leading_fill (default: NaN, pass a float for fillna).
    """
    not_nan = ~np.isnan(a)
    if not not_nan.any():
        out = a.copy(); out[:] = leading_fill; return out
    idx = np.where(not_nan, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx].copy()
    first = int(np.argmax(not_nan))
    if first:
        out[:first] = leading_fill
    return out


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
    """
    Pine f_swing() — fully NumPy-vectorized.
    Returns (tsl, avn) exactly like Pine's [_tsl, _avn].

    IMPORTANT: avn (direction) is the forward-filled cross signal, NOT
    derived from close > tsl.  Pine uses avn for dirMain trend-flip
    detection, and tsl for price comparison (priceBelowMain etc.).
    These diverge at turning-point bars and must stay separate.

    ⚡ v43: rolling max/min replaced with sliding_window_view — 2-3.5x faster
    than pd.Series.rolling at real scanner bar counts.
    """
    n   = len(c)
    # ⚡ sliding_window_view avoids pandas Series construction + rolling overhead
    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    if n >= no:
        res[no - 1:] = sliding_window_view(h, no).max(axis=1)
        sup[no - 1:] = sliding_window_view(l, no).min(axis=1)

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


def calc_mfi(h: np.ndarray, l: np.ndarray, c: np.ndarray,
             v: np.ndarray, length: int = MFI_LEN) -> np.ndarray:
    """Money Flow Index (Pine replica)."""
    hlc3   = (h + l + c) / 3.0
    raw_mf = hlc3 * v
    hlc3_p = np.concatenate([[hlc3[0]], hlc3[:-1]])
    pos_mf = _sma(np.where(hlc3 > hlc3_p, raw_mf, 0.0), length)
    neg_mf = _sma(np.where(hlc3 < hlc3_p, raw_mf, 0.0), length)
    ratio  = np.where(neg_mf == 0, 1.0, pos_mf / np.where(neg_mf == 0, 1.0, neg_mf))
    return 100.0 - 100.0 / (1.0 + ratio)


def calc_kvo(h: np.ndarray, l: np.ndarray, c: np.ndarray,
             v: np.ndarray, fast: int = KVO_FAST, slow: int = KVO_SLOW) -> np.ndarray:
    """
    Klinger Volume Oscillator (Pine replica).
    ⚡ v55: 2 Python for-loops eliminated.
      k_trend: forward-fill via np.maximum.accumulate (same trick as f_swing avn).
      cm:      segmented cumsum — O(n) arithmetic; no loop.
    """
    n    = len(c)
    hlc3 = (h + l + c) / 3.0
    mom  = np.diff(hlc3, prepend=hlc3[0])
    dm   = h - l

    # ⚡ Vectorized k_trend — forward-fill np.sign(mom), skipping zeros.
    sign_mom = np.sign(mom)
    nonzero  = sign_mom != 0
    ff_idx   = np.where(nonzero, np.arange(n), 0)
    np.maximum.accumulate(ff_idx, out=ff_idx)
    k_trend  = sign_mom[ff_idx]   # 0 until first non-zero mom

    # ⚡ Vectorized cm — segmented cumsum (resets on k_trend direction change).
    dm_cumsum   = np.cumsum(dm)
    transitions = np.concatenate([[False], k_trend[1:] != k_trend[:-1]])
    offset_arr  = np.zeros(n)
    offset_arr[0] = dm[0]          # ensures cm[0] = 0
    s_idx = np.where(transitions)[0]
    if s_idx.size:
        offset_arr[s_idx] = dm_cumsum[s_idx] - dm[s_idx] - dm[s_idx - 1]
    set_mask = transitions.copy(); set_mask[0] = True
    fof_idx  = np.where(set_mask, np.arange(n), 0)
    np.maximum.accumulate(fof_idx, out=fof_idx)
    cm    = dm_cumsum - offset_arr[fof_idx]
    cm[0] = 0.0   # guard: exactly zero at bar 0

    safe_cm = np.where(cm != 0, cm, 1.0)
    vf = np.where(cm != 0,
                  100.0 * v * k_trend * np.abs(2.0 * dm / safe_cm - 1.0),
                  0.0)
    return _ema(vf, fast) - _ema(vf, slow)


def calc_weis_wave(c: np.ndarray, trend_len: int = WEIS_LEN) -> np.ndarray:
    """
    Weis Wave direction array (Pine replica). Returns +1=green, -1=red, 0=unset.
    ⚡ v55: weis_trend forward-fill via accumulate; is_trending via sliding_window_view.
    """
    n   = len(c)
    mov = np.sign(np.diff(c, prepend=c[0])).astype(np.int8)

    # ⚡ Vectorized weis_trend — forward-fill on direction-change bars.
    mov_prev    = np.concatenate([[np.int8(0)], mov[:-1]])
    change_mask = (mov != 0) & (mov != mov_prev)
    wt_raw      = np.where(change_mask, mov, np.int8(0)).astype(np.int8)
    wt_idx      = np.where(change_mask, np.arange(n), 0)
    np.maximum.accumulate(wt_idx, out=wt_idx)
    weis_trend  = wt_raw[wt_idx].astype(np.int8)
    weis_trend[0] = 0

    # ⚡ isTrending — sliding_window_view replaces shift-AND loop
    diffs = np.diff(c, prepend=c[0])
    pos   = diffs > 0
    neg   = diffs < 0
    if trend_len == 1:
        is_trending = pos | neg
    else:
        pos_wins    = sliding_window_view(pos.astype(np.uint8), trend_len)
        neg_wins    = sliding_window_view(neg.astype(np.uint8), trend_len)
        is_rising   = np.concatenate([np.zeros(trend_len - 1, bool), pos_wins.all(axis=1)])
        is_falling  = np.concatenate([np.zeros(trend_len - 1, bool), neg_wins.all(axis=1)])
        is_trending = is_rising | is_falling

    wave = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if weis_trend[i] != wave[i - 1] and is_trending[i]:
            wave[i] = weis_trend[i]
        else:
            wave[i] = wave[i - 1]
    return wave


def calc_kwv_windows(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                     v: np.ndarray, dir_main: np.ndarray) -> tuple:
    """
    KWV state machines — Pine Script exact replica.

    BUY  window:
      R1: kvo>0 & mfi>MFI_OB  → bStep=1
      R2: kvo<0 & mfi<MFI_OS  → bStep=2  (window OPENS)
      R3: kvo>0 & green wave   → fires, bullR3Fired=True, window stays open
      CLOSE : next new green wave after R3 fired
      FORCE : dir_main != +1

    SELL window (symmetric).

    Returns (kwv_bull_win, kwv_bear_win, kwv_bull_sig, kwv_bear_sig):
      kwv_bull_win / kwv_bear_win — True from R2 onward (window open, for display).
      kwv_bull_sig / kwv_bear_sig — True only after R3 fires within the open window
                                    (used as the signal-collection gate).
    """
    n = len(c)
    kvo  = calc_kvo(h, l, c, v)
    mfi  = calc_mfi(h, l, c, v)
    wave = calc_weis_wave(c)
    kvo_above = kvo > 0;  kvo_below = kvo < 0
    green_bar = wave == 1; red_bar   = wave == -1
    wave_prev = np.concatenate([[np.int8(0)], wave[:-1]])
    new_green = (wave == 1)  & (wave_prev != 1)
    new_red   = (wave == -1) & (wave_prev != -1)
    kwv_bull_win = np.zeros(n, bool)
    kwv_bear_win = np.zeros(n, bool)
    # v47: separate signal-gate arrays — only True after R3 fires
    kwv_bull_sig = np.zeros(n, bool)
    kwv_bear_sig = np.zeros(n, bool)
    s_step = 0;  b_step = 0
    kwv_bull = False;  kwv_bear = False
    bull_r3  = False;  bear_r3  = False
    for i in range(n):
        sell_fired = False; buy_fired  = False
        # ── SELL state machine (dir == -1) ──────────────────────────────
        if dir_main[i] == -1:
            if s_step == 0:
                if kvo_below[i] and mfi[i] < MFI_OS: s_step = 1
            elif s_step == 1:
                if kvo_above[i] and mfi[i] > MFI_OB: s_step = 2
            elif s_step == 2:
                if kvo_below[i] and red_bar[i]:
                    sell_fired = True; bear_r3 = True   # R3 fired — signal gate opens
                    s_step = 0
        else:
            s_step = 0; bear_r3 = False
        # ── BUY state machine (dir == 1) ─────────────────────────────────
        if dir_main[i] == 1:
            if b_step == 0:
                if kvo_above[i] and mfi[i] > MFI_OB: b_step = 1
            elif b_step == 1:
                if kvo_below[i] and mfi[i] < MFI_OS: b_step = 2
            elif b_step == 2:
                if kvo_above[i] and green_bar[i]:
                    buy_fired = True; bull_r3 = True    # R3 fired — signal gate opens
                    b_step = 0
        else:
            b_step = 0; bull_r3 = False
        # ── Open window at R2 (display only) ──────────────────────────────
        if b_step == 2: kwv_bull = True
        if s_step == 2: kwv_bear = True
        # ── Close BULL: next new green wave after R3 ──────────────────────
        if kwv_bull and bull_r3 and not buy_fired  and new_green[i]: kwv_bull = False; bull_r3 = False
        # ── Close BEAR: next new red wave after R3 ────────────────────────
        if kwv_bear and bear_r3 and not sell_fired and new_red[i]:   kwv_bear = False; bear_r3 = False
        # ── Force-close ───────────────────────────────────────────────────
        if dir_main[i] != 1:  kwv_bull = False; bull_r3 = False
        if dir_main[i] != -1: kwv_bear = False; bear_r3 = False
        kwv_bull_win[i] = kwv_bull
        kwv_bear_win[i] = kwv_bear
        # v47: signal gate = window open AND R3 has fired
        kwv_bull_sig[i] = kwv_bull and bull_r3
        kwv_bear_sig[i] = kwv_bear and bear_r3
    return kwv_bull_win, kwv_bear_win, kwv_bull_sig, kwv_bear_sig


# ══════════════════════════════════════════════════════════════════════════════
#  v34: QM STRATEGY HELPERS  — Pine Script exact replicas
# ══════════════════════════════════════════════════════════════════════════════

def _calc_qm_strat1(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    zz_len: int = 5):
    """
    ZigZag-based QM (Strategy 1).
    Bull QM: trend==-1, 3 swing highs where H2>H1 and H0>H1, 2 swing lows where L1>L0, close>L1
    Bear QM: trend== 1, 3 swing lows where L2<L1 and L0<L1, 2 swing highs where H1<H0, close<H1
    Returns (bull_qm, bear_qm) — bool arrays, rising-edge only.
    """
    n   = len(c)
    # ⚡ v55: sliding_window_view + -inf/+inf pad replaces pd.Series.rolling(min_periods=1).
    #    Equivalent: pad left with -inf (highs) / +inf (lows) then take full windows.
    _pad_h = np.concatenate([np.full(zz_len - 1, -np.inf), h])
    _pad_l = np.concatenate([np.full(zz_len - 1,  np.inf), l])
    roll_h = sliding_window_view(_pad_h, zz_len).max(axis=1)
    roll_l = sliding_window_view(_pad_l, zz_len).min(axis=1)
    to_up   = (h >= roll_h); to_down = (l <= roll_l)

    trend = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        t = trend[i - 1]
        if t == 1 and to_down[i]:   trend[i] = -1
        elif t == -1 and to_up[i]:  trend[i] = 1
        else:                        trend[i] = t

    high_pts: list = []; low_pts: list = []
    bull_raw = np.zeros(n, bool); bear_raw = np.zeros(n, bool)
    last_to_up_bar = 0; last_to_down_bar = 0

    for i in range(n):
        if to_up[i]:   last_to_up_bar   = i
        if to_down[i]: last_to_down_bar = i
        if i > 0 and trend[i] != trend[i - 1]:
            if trend[i] == 1:
                since = max(1, i - last_to_down_bar); start = max(0, i - since)
                seg = l[start : i + 1]; lv = float(seg.min()); li = start + int(np.argmin(seg))
                low_pts.append((lv, li))
            else:
                since = max(1, i - last_to_up_bar); start = max(0, i - since)
                seg = h[start : i + 1]; hv = float(seg.max()); hi = start + int(np.argmax(seg))
                high_pts.append((hv, hi))
        if len(high_pts) >= 3 and len(low_pts) >= 2:
            h2v = high_pts[-3][0]; h1v = high_pts[-2][0]; h0v = high_pts[-1][0]
            l1v = low_pts[-2][0];  l0v = low_pts[-1][0]
            bull_raw[i] = (trend[i] == -1 and h2v > h1v and l1v > l0v and h0v > h1v and c[i] > l1v)
        if len(low_pts) >= 3 and len(high_pts) >= 2:
            l2v = low_pts[-3][0]; l1v = low_pts[-2][0]; l0v = low_pts[-1][0]
            h1v = high_pts[-2][0]; h0v = high_pts[-1][0]
            bear_raw[i] = (trend[i] == 1 and l2v < l1v and h1v < h0v and l0v < l1v and c[i] < h1v)

    bull_qm = np.zeros(n, bool); bear_qm = np.zeros(n, bool)
    bull_qm[1:] = bull_raw[1:] & ~bull_raw[:-1]
    bear_qm[1:] = bear_raw[1:] & ~bear_raw[:-1]
    return bull_qm, bear_qm


def _calc_qm_strat2(h: np.ndarray, l: np.ndarray, c: np.ndarray, pp: int = 5):
    """
    Pivot-array-based QM (Strategy 2).
    Bear QM: last-4 types HH→HL→HH→LL, v5<v1, newest pivot=bar-pp.
    Bull QM: last-4 types LL→LH→LL→HH, v5>v1, newest pivot=bar-pp.
    Returns (bull_qm, bear_qm) — bool arrays, one True per pattern.
    """
    n = len(c)
    bull_qm = np.zeros(n, bool)
    bear_qm = np.zeros(n, bool)

    # ── Vectorised pivot detection ─────────────────────────────────────────
    # h[j] is a pivot high if it's the max of h[j-pp : j+pp+1].
    # Confirmed at detection bar i = j + pp (i.e. pp bars after the pivot).
    # ⚡ v55: sliding_window_view replaces pd.Series.rolling(center=True) — no pandas overhead
    _w        = 2 * pp + 1
    roll_max_h = np.full(n, np.nan)
    roll_min_l = np.full(n, np.nan)
    if n >= _w:
        wins_h              = sliding_window_view(h, _w)   # (n-_w+1, _w)
        wins_l              = sliding_window_view(l, _w)
        roll_max_h[pp:n-pp] = wins_h.max(axis=1)
        roll_min_l[pp:n-pp] = wins_l.min(axis=1)
    piv_h_at_j = np.where(h == roll_max_h, h, np.nan)   # value at pivot bar j
    piv_l_at_j = np.where(l == roll_min_l, l, np.nan)

    # Store at detection bar i = j + pp
    piv_h = np.full(n, np.nan)
    piv_l = np.full(n, np.nan)
    pj_h = np.where(~np.isnan(piv_h_at_j))[0]
    pj_l = np.where(~np.isnan(piv_l_at_j))[0]
    pi_h = pj_h + pp; pi_h = pi_h[pi_h < n]
    pi_l = pj_l + pp; pi_l = pi_l[pi_l < n]
    piv_h[pi_h] = h[pj_h[:len(pi_h)]]
    piv_l[pi_l] = l[pj_l[:len(pi_l)]]

    piv_h_bool = ~np.isnan(piv_h)
    piv_l_bool = ~np.isnan(piv_l)

    # ⚡ v55: _np_ffill replaces 4× pd.Series.ffill() — avoids pandas Series construction
    h_val = _np_ffill(piv_h)
    l_val = _np_ffill(piv_l)
    h_idx = _np_ffill(np.where(piv_h_bool, np.arange(n, dtype=float) - pp, np.nan),
                      leading_fill=-1.0).astype(np.int64)
    l_idx = _np_ffill(np.where(piv_l_bool, np.arange(n, dtype=float) - pp, np.nan),
                      leading_fill=-1.0).astype(np.int64)

    a_type: list = []; a_val: list = []; a_idx: list = []
    bear_start = 0.0; check_be = 0; bull_start = 0.0; check_bu = 0

    def push_low(i):
        t = (("HL" if len(a_type) > 1 and a_val[-2] < l_val[i] else "LL") if len(a_type) > 1 else "L")
        a_type.append(t); a_val.append(float(l_val[i])); a_idx.append(int(l_idx[i]))

    def push_high(i):
        t = (("HH" if len(a_type) > 1 and a_val[-2] < h_val[i] else "LH") if len(a_type) > 1 else "H")
        a_type.append(t); a_val.append(float(h_val[i])); a_idx.append(int(h_idx[i]))

    def pop_last():
        a_type.pop(); a_val.pop(); a_idx.pop()

    for i in range(n):
        hb = piv_h_bool[i]; lb = piv_l_bool[i]
        hv = h_val[i]; lv = l_val[i]; hi_ = h_idx[i]; li_ = l_idx[i]
        if np.isnan(hv) or np.isnan(lv):
            hb_eff = hb and not np.isnan(hv); lb_eff = lb and not np.isnan(lv)
        else:
            hb_eff = hb; lb_eff = lb

        if hb_eff and lb_eff:
            # Pine f_piv_bull/f_piv_bear both-fire exact replica:
            #   empty  → push H
            #   last Low-type  (L / LL / HL) → push High unconditionally
            #   last High-type (H / HH / LH) → push Low  unconditionally
            if len(a_type) == 0:
                a_type.append("H"); a_val.append(float(hv)); a_idx.append(int(hi_))
            else:
                lt = a_type[-1]
                if lt in ("L", "LL", "HL"):
                    push_high(i)
                elif lt in ("H", "HH", "LH"):
                    push_low(i)
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
            v1 = a_val[-1];  v2 = a_val[-2];  v5 = a_val[-5]; i1 = a_idx[-1]
            bear_cond = (t1 == "LL" and t2 == "HH" and t3 == "HL" and t4 == "HH"
                         and v5 < v1 and i1 == i - pp and check_be == 0)
            if bear_cond:
                bear_start = v2; check_be = 1; bear_qm[i] = True
            if bear_start != (a_val[-2] if len(a_val) >= 2 else bear_start): check_be = 0
            bull_cond = (t1 == "HH" and t2 == "LL" and t3 == "LH" and t4 == "LL"
                         and v5 > v1 and i1 == i - pp and check_bu == 0)
            if bull_cond:
                bull_start = v2; check_bu = 1; bull_qm[i] = True
            if bull_start != (a_val[-2] if len(a_val) >= 2 else bull_start): check_bu = 0

    return bull_qm, bear_qm


def signals_kwv_qm(ds_sig, ds_lower, pivot_win_ts: int, pivot_end_ts: int,
                   want_sell: bool,
                   zz_len: int = 5, s2_pp: int = 5,
                   ltf_zz_len=None, ltf_s2_pp=None):
    """
    v48: Pine Script QM + KWV Window Filter exact replica.

    v47 FIX: bull_allow/bear_allow now use kwv_bull_sig/kwv_bear_sig (R3-gated)
    instead of kwv_bull_win/kwv_bear_win (R2-opened). QM signals are only valid
    AFTER R3 fires within the open window.

    v48: Also returns waiting signals — QMs that fired while window was open (R2)
    but R3 had not yet fired.

    Returns 6-tuple:
      (found, sig_ts_list, sig_kind_list,
       wait_found, wait_ts_list, wait_kind_list)
    """
    h  = ds_sig.high.values;  l  = ds_sig.low.values
    c  = ds_sig.close.values; v  = ds_sig.volume.values
    ts = ds_sig.ts.values.astype(np.int64)
    n  = len(c)

    tsl_main, dir_main = f_swing(h, l, c, SWING_UTAMA)
    above_tsl = c > tsl_main; below_tsl = c < tsl_main

    # v47: 4-array return — *_sig are R3-gated, *_win open at R2 (for wait tracking)
    kwv_bull_win, kwv_bear_win, kwv_bull_sig, kwv_bear_sig = calc_kwv_windows(h, l, c, v, dir_main)

    # Signal gate uses R3-gated arrays (*_sig)
    bull_allow = (dir_main == 1)  & kwv_bull_sig
    bear_allow = (dir_main == -1) & kwv_bear_sig
    allow      = bear_allow if want_sell else bull_allow

    # Wait gate: window open at R2 but R3 not yet fired
    kwv_win_side = kwv_bear_win if want_sell else kwv_bull_win
    kwv_sig_side = kwv_bear_sig if want_sell else kwv_bull_sig
    wait_gate    = (dir_main == (-1 if want_sell else 1)) & kwv_win_side & ~kwv_sig_side

    s1_bull, s1_bear = _calc_qm_strat1(h, l, c, zz_len=zz_len)
    s2_bull, s2_bear = _calc_qm_strat2(h, l, c, pp=s2_pp)
    qm_bull = s1_bull | s2_bull; qm_bear = s1_bear | s2_bear
    qm_sig  = qm_bear if want_sell else qm_bull
    qm_sig_filtered  = qm_sig & allow    & (below_tsl if want_sell else above_tsl)
    qm_wait_filtered = qm_sig & wait_gate & (below_tsl if want_sell else above_tsl)

    ltf_bull_qm = np.empty(0, bool); ltf_bear_qm = np.empty(0, bool)
    ltf_ts      = np.empty(0, dtype=np.int64)
    _ltf_zz = ltf_zz_len if ltf_zz_len is not None else zz_len
    _ltf_pp = ltf_s2_pp  if ltf_s2_pp  is not None else s2_pp

    if ds_lower is not None and not ds_lower.empty and len(ds_lower) >= 20:
        lh = ds_lower.high.values; ll = ds_lower.low.values; lc = ds_lower.close.values
        l1b, l1s = _calc_qm_strat1(lh, ll, lc, zz_len=_ltf_zz)
        l2b, l2s = _calc_qm_strat2(lh, ll, lc, pp=_ltf_pp)
        ltf_bull_qm = l1b | l2b; ltf_bear_qm = l1s | l2s
        ltf_ts = ds_lower.ts.values.astype(np.int64)

    ltf_qm = ltf_bear_qm if want_sell else ltf_bull_qm

    win_start = int(np.searchsorted(ts, pivot_win_ts))
    win_end   = int(np.searchsorted(ts, pivot_end_ts))

    sig_ts_list:   list = []
    sig_kind_list: list = []
    wait_ts_list:  list = []
    wait_kind_list: list = []

    signal_fired_this_window = False
    wait_fired_this_window   = False
    prev_allow = bool(allow[win_start - 1]) if win_start > 0 else False
    prev_wait  = bool(wait_gate[win_start - 1]) if win_start > 0 else False

    for i in range(win_start, min(win_end, n - 1)):
        # TSL direction flip → purge all signals
        if i > 0 and dir_main[i] != dir_main[i - 1]:
            if want_sell and dir_main[i] > 0:
                sig_ts_list.clear(); sig_kind_list.clear()
                wait_ts_list.clear(); wait_kind_list.clear()
                signal_fired_this_window = False
                wait_fired_this_window   = False
            if not want_sell and dir_main[i] < 0:
                sig_ts_list.clear(); sig_kind_list.clear()
                wait_ts_list.clear(); wait_kind_list.clear()
                signal_fired_this_window = False
                wait_fired_this_window   = False

        # Rising edge of confirmed allow → new R3-confirmed cycle
        cur_allow = bool(allow[i])
        if cur_allow and not prev_allow:
            signal_fired_this_window = False
        prev_allow = cur_allow

        # Rising edge of wait gate → new pre-R3 window cycle
        cur_wait = bool(wait_gate[i])
        if cur_wait and not prev_wait:
            wait_fired_this_window = False
        prev_wait = cur_wait

        # ── R3 fires — promote all waiting QMs to confirmed ─────────────
        # Pine's sellSignal/buySignal fires on this bar: any QMs collected
        # in the R2→R3 waiting window are now confirmed signals.
        if allow[i] and wait_ts_list:
            sig_ts_list.extend(wait_ts_list)
            sig_kind_list.extend(k + " (promoted)" for k in wait_kind_list)
            wait_ts_list.clear()
            wait_kind_list.clear()
            signal_fired_this_window = True
            wait_fired_this_window   = False

        # ── Confirmed signals (R3 passed, fresh QM on R3 bar itself) ─────
        if qm_sig_filtered[i] and not signal_fired_this_window:
            sig_ts_list.append(int(ts[i])); sig_kind_list.append("QM")
            signal_fired_this_window = True

        if ltf_ts.size > 0 and allow[i] and not signal_fired_this_window:
            tsl_ok = below_tsl[i] if want_sell else above_tsl[i]
            if tsl_ok:
                t_lo = int(ts[i])
                t_hi = (int(ts[i + 1]) if i + 1 < n else t_lo + (int(ts[i]) - int(ts[i - 1])))
                mask = (ltf_ts >= t_lo) & (ltf_ts < t_hi) & ltf_qm[:len(ltf_ts)]
                if mask.any():
                    first_ltf = int(ltf_ts[np.where(mask)[0][0]])
                    sig_ts_list.append(first_ltf); sig_kind_list.append("MTF QM")
                    signal_fired_this_window = True

        # ── Waiting signals (window open at R2, R3 not yet fired) ────────
        if qm_wait_filtered[i] and not wait_fired_this_window:
            wait_ts_list.append(int(ts[i])); wait_kind_list.append("QM")
            wait_fired_this_window = True

        if ltf_ts.size > 0 and wait_gate[i] and not wait_fired_this_window:
            tsl_ok = below_tsl[i] if want_sell else above_tsl[i]
            if tsl_ok:
                t_lo = int(ts[i])
                t_hi = (int(ts[i + 1]) if i + 1 < n else t_lo + (int(ts[i]) - int(ts[i - 1])))
                mask = (ltf_ts >= t_lo) & (ltf_ts < t_hi) & ltf_qm[:len(ltf_ts)]
                if mask.any():
                    first_ltf = int(ltf_ts[np.where(mask)[0][0]])
                    wait_ts_list.append(first_ltf); wait_kind_list.append("MTF QM")
                    wait_fired_this_window = True

    return (len(sig_ts_list)  > 0, sig_ts_list,  sig_kind_list,
            len(wait_ts_list) > 0, wait_ts_list, wait_kind_list)


def calc_bb_continuation(c: np.ndarray, h: np.ndarray, l: np.ndarray,
                          want_sell: bool,
                          length: int = BB_LEN, mult: float = BB_MULT) -> np.ndarray:
    """
    Direction-aware BB continuation signal — canonical v26 loop version.
    Only tracks the state machine for the needed side (Pine-accurate).
    """
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
            if np.isnan(basis[i]):
                continue
            if h[i] < lower[i]:
                rule1_met   = True
            if l[i] > upper[i]:
                rule1_met = band_broken = armed = False
            if rule1_met and c[i] < lower[i]:
                band_broken = True
                armed       = False
            if band_broken and h[i] >= lower[i]:
                armed = True
            if armed and c[i] < basis[i]:
                sig[i]      = True
                band_broken = armed = False
    else:
        for i in range(n):
            if np.isnan(basis[i]):
                continue
            if l[i] > upper[i]:
                rule1_met   = True
            if h[i] < lower[i]:
                rule1_met = band_broken = armed = False
            if rule1_met and c[i] > upper[i]:
                band_broken = True
                armed       = False
            if band_broken and l[i] <= upper[i]:
                armed = True
            if armed and c[i] > basis[i]:
                sig[i]      = True
                band_broken = armed = False

    return sig



def check_bb_kc_range(c: np.ndarray, h: np.ndarray, l: np.ndarray,
                      ts_arr: np.ndarray, pivot_ts: int,
                      want_sell: bool):
    """
    v24: Stage 3 mid TF — KC range validity gate (window-based).

    KC band used: mid_tf KC (same h/l/c as the BB calculation).
    KC_LEN=20, KC_MULT=2.0, KC_ATR_LEN=10 applied to mid_tf candles.

    Window rules:
      · The 1st BB signal in the pivot window opens a window (valid_from_ts).
      · Consecutive BB signals while the window is still clean do NOT open a
        new window — they are all covered by the original window start ts.
      · After a KC violation, only the 1st new BB signal opens a fresh window.
        Further consecutive BBs while that new window is clean again do nothing.

    KC violation check:
      · Starts from the candle AFTER the BB signal fires (not the signal bar).
      · A close outside KC (close > kc_upper OR close < kc_lower) closes the
        current window — price left the tradeable range.

    sig_tf signals are later filtered: only those >= valid_from_ts survive,
    meaning only signals emitted inside the current open (clean) window.

    Returns:
        valid          (bool)      — True if an open window exists
        valid_from_ts  (int|None)  — ts of the BB that opened the current
                                     clean window; None if no open window
        detail         (str)       — human-readable summary for debug output
    """
    n = len(c)

    bb_main = calc_bb_continuation(c, h, l, want_sell=want_sell)
    # KC computed on mid_tf (same h/l/c passed in) — intentionally NOT tdi_tf KC
    kc_upper, kc_lower = calc_kc(h, l, c)

    win_start = int(np.searchsorted(ts_arr, pivot_ts))

    # ── Tracking ──────────────────────────────────────────────────────
    valid_from_ts  = None   # ts of the BB that opened the current clean window
    kc_violated    = False  # KC violation since valid_from_ts?
    window_open_bar = -1    # bar index where current window was opened
                            # KC check is skipped on this exact bar (starts next)

    # Debug stats
    windows: list = []   # (start_ts, end_ts_or_None) per window
    n_windows  = 0   # count of windows opened
    n_kc_viols = 0

    for i in range(win_start, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]):
            continue

        # ── BB signal fires ───────────────────────────────────────────
        if bb_main[i]:
            if valid_from_ts is None or kc_violated:
                # 1st BB in pivot window, OR 1st BB after a KC violation
                # → open a new window from this bar's timestamp
                valid_from_ts   = int(ts_arr[i])
                kc_violated     = False
                window_open_bar = i
                windows.append((valid_from_ts, None))
                n_windows      += 1
            # else: consecutive BB inside a still-clean window — no change

        # ── KC violation check ────────────────────────────────────────
        # Only while a clean window is open AND past the bar that opened it
        if (valid_from_ts is not None
                and not kc_violated
                and i > window_open_bar):
            if c[i] > kc_upper[i] or c[i] < kc_lower[i]:
                kc_violated = True
                n_kc_viols += 1
                windows[-1] = (windows[-1][0], int(ts_arr[i]))

    # Window is open if a BB was seen and no KC violation has closed it
    valid = (valid_from_ts is not None) and (not kc_violated)

    side   = "SELL" if want_sell else "BUY"
    closed = [(s, e) for s, e in windows if e is not None]
    detail = (
        f"{n_windows} BB {side} window(s) opened  |  "
        f"{n_kc_viols} KC violation(s) closed {len(closed)} window(s)"
        + (f"  |  open window from ts={valid_from_ts}" if valid else "  |  no open window")
    )
    return valid, (valid_from_ts if valid else None), detail


def calc_sma_cloud_bs_signals(h: np.ndarray, l: np.ndarray,
                               c: np.ndarray, o: np.ndarray,
                               ts_arr: np.ndarray, pivot_win_ts: int,
                               pivot_end_ts: int,
                               want_sell: bool,
                               sma_len: int    = 20,
                               bb_sma_p: int   = 20,
                               bb_std_m: float = 2.5,
                               sma_b_p: int    = 20,
                               bayes_n: int    = 20,
                               thresh: float   = 15.0):
    """
    v34 (CLI) / v37 (Streamlit): Pine Script "SMA Cloud BS Signals + Bayesian Filter" — NumPy replica.

    Returns (found, valid_from_ts):
      · found          — True if at least one Cloud BS signal (matching want_sell)
                         fired on mid_tf inside the Stage 1 pivot window.
      · valid_from_ts  — timestamp (ms) of the FIRST such signal (used to gate
                         sig_tf signals: only those >= valid_from_ts survive).
                         None if found=False.

    ── SMA Cloud ─────────────────────────────────────────────────────────────
    smaHigh = SMA(high, 20),  smaLow = SMA(low, 20),  smaMid = (H+L)/2
    bullCloud = close >= smaMid

    ── Bayesian BBSMA ────────────────────────────────────────────────────────
    Bayesian combination of three binary indicators:
      P_bbUpper — fraction of last N bars where close was above bbUpper
      P_bbBasis — fraction of last N bars where close was above BB basis
      P_sma     — fraction of last N bars where close was above SMA
    Each normalized: p_up = p_up / (p_up + p_down)

    ── Buy signal ────────────────────────────────────────────────────────────
    bullCloud AND touchedCloudBot AND (buyCondA OR buyCondB) AND bayesBuyOk
    ── Sell signal ───────────────────────────────────────────────────────────
    bearCloud AND touchedCloudTop AND (sellCondA OR sellCondB) AND bayesSellOk
    """
    n = len(c)
    sma_h   = _sma(h, sma_len)
    sma_l   = _sma(l, sma_len)
    sma_mid = (sma_h + sma_l) / 2.0
    bull_cloud = c >= sma_mid
    bear_cloud = ~bull_cloud

    bb_basis   = _sma(c, bb_sma_p)
    # ⚡ v55: Population std via E[X²]−E[X]² — avoids pd.Series.rolling.std (ddof=0 exact match)
    bb_std_arr = np.sqrt(np.maximum(0.0, _sma(c ** 2, bb_sma_p) - bb_basis ** 2))
    bb_upper   = bb_basis + bb_std_m * bb_std_arr
    # ⚡ v55: sma_b_p == bb_sma_p (both 20) → sma_b_arr ≡ bb_basis; reuse directly.
    sma_b_arr  = bb_basis

    N   = bayes_n
    # ⚡ v55: Replace 6× pd.Series.rolling.mean with _sma (np.convolve — 7-16× faster)
    raw_bu_up = _sma((c > bb_upper).astype(np.float64), N)
    raw_bu_dn = _sma((c < bb_upper).astype(np.float64), N)
    raw_bb_up = _sma((c > bb_basis).astype(np.float64), N)
    raw_bb_dn = _sma((c < bb_basis).astype(np.float64), N)
    raw_sm_up = _sma((c > sma_b_arr).astype(np.float64), N)
    raw_sm_dn = _sma((c < sma_b_arr).astype(np.float64), N)

    eps   = 1e-9
    A_up  = raw_bu_up / np.maximum(raw_bu_up + raw_bu_dn, eps)
    B_up  = raw_bb_up / np.maximum(raw_bb_up + raw_bb_dn, eps)
    C_up  = raw_sm_up / np.maximum(raw_sm_up + raw_sm_dn, eps)
    A_dn  = raw_bu_dn / np.maximum(raw_bu_dn + raw_bu_up, eps)
    B_dn  = raw_bb_dn / np.maximum(raw_bb_dn + raw_bb_up, eps)
    C_dn  = raw_sm_dn / np.maximum(raw_sm_dn + raw_sm_up, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_down = np.where(A_up != 0,
            B_up ** 2 * C_up ** 2 + (1 - A_up) * (1 - B_up) * (1 - C_up), np.nan)
        sigma_up   = np.where(A_dn != 0,
            B_dn ** 2 * C_dn ** 2 + (1 - A_dn) * (1 - B_dn) * (1 - C_dn), np.nan)

    green_line    = np.nan_to_num(sigma_down, nan=0.0) * 100.0
    red_line      = np.nan_to_num(sigma_up,   nan=0.0) * 100.0
    bayes_buy_ok  = (green_line > red_line) & (green_line > thresh)
    bayes_sell_ok = (red_line > green_line) & (red_line   > thresh)

    is_bull      = c >= o
    is_bear      = ~is_bull
    body         = np.abs(c - o)
    upper_wick   = h - np.maximum(c, o)
    lower_wick   = np.minimum(c, o) - l
    atr_vals     = calc_atr(h, l, c, 14)
    valid_body   = body > atr_vals * 0.03
    upper_wick_dom = (upper_wick >= body * 2) & valid_body
    lower_wick_dom = (lower_wick >= body * 2) & valid_body

    touched_cloud_top = (h >= sma_l) & (c <= sma_h)
    touched_cloud_bot = (l <= sma_h) & (c >= sma_l)
    sell_cond_a = is_bear & (c < sma_h)
    sell_cond_b = is_bull & upper_wick_dom & (c < sma_h)
    buy_cond_a  = is_bull & (c > sma_l)
    buy_cond_b  = is_bear & lower_wick_dom & (c > sma_l)

    sell_signal = bear_cloud & touched_cloud_top & (sell_cond_a | sell_cond_b) & bayes_sell_ok
    buy_signal  = bull_cloud & touched_cloud_bot & (buy_cond_a  | buy_cond_b)  & bayes_buy_ok

    win_start = int(np.searchsorted(ts_arr, pivot_win_ts))
    win_end   = int(np.searchsorted(ts_arr, pivot_end_ts))
    sig_arr   = sell_signal if want_sell else buy_signal
    sig_idxs  = np.where(sig_arr[win_start:win_end])[0]

    if sig_idxs.size == 0:
        return False, None, 0, [], []

    # ── v54 Pivot Hi/Lo invalidation filter (Pine close-based, leftBars=5 rightBars=5) ──
    # For each Cloud BS signal independently:
    #   1. Backward scan from the bar BEFORE the signal → most recent confirmed pivot
    #      LOW (sell) / HIGH (buy).  leftBars=5, rightBars=5 — c[i] must be the
    #      min/max of its 11-bar window (confirmed).
    #      If the first pivot found is on the same candle as the signal, use the
    #      previous pivot instead (not the signal itself).
    #   2. Breach check from signal bar (inclusive) → scan time:
    #        SELL: any close < pivot_low  → invalid.
    #        BUY:  any close > pivot_high → invalid.
    #   No pivot found → accept unconditionally.
    _PL = 5   # leftBars  — matches Pine indicator
    _PR = 5   # rightBars — matches Pine indicator
    n_c = len(c)

    pivot_low_vals  = np.full(n_c, np.nan)
    pivot_high_vals = np.full(n_c, np.nan)
    _pw = _PL + _PR + 1
    if n_c >= _pw:
        # ⚡ v55: sliding_window_view replaces O(n) Python for-loop
        _wins = sliding_window_view(c, _pw)                # (n_c-_pw+1, _pw)
        _ci   = np.arange(_PL, _PL + _wins.shape[0])      # center indices
        _cv   = c[_ci]
        pivot_low_vals[_ci]  = np.where(_cv == _wins.min(axis=1), _cv, np.nan)
        pivot_high_vals[_ci] = np.where(_cv == _wins.max(axis=1), _cv, np.nan)

    valid_sig_idxs  = []
    rejected_detail = []
    pv_arr = pivot_low_vals if want_sell else pivot_high_vals

    for rel in sig_idxs:
        abs_i = rel + win_start

        # 1. Backward scan — most recent confirmed pivot strictly before signal bar
        ref_level = np.nan
        for pivot_i in range(abs_i - 1, -1, -1):
            if not np.isnan(pv_arr[pivot_i]):
                ref_level = pv_arr[pivot_i]
                break

        if np.isnan(ref_level):
            # No pivot found → accept unconditionally
            valid_sig_idxs.append(rel)
            continue

        # 2. Breach check — signal bar (inclusive) → scan time
        scan_closes = c[abs_i : win_end]
        breach_mask = scan_closes < ref_level if want_sell else scan_closes > ref_level
        breach_offs = np.where(breach_mask)[0]
        if breach_offs.size == 0:
            valid_sig_idxs.append(rel)
        else:
            breach_abs = abs_i + int(breach_offs[0])
            breach_ts  = int(ts_arr[breach_abs]) if breach_abs < len(ts_arr) else -1
            rejected_detail.append((
                int(rel + 1),
                int(ts_arr[abs_i]),
                float(ref_level),
                breach_ts,
                float(c[breach_abs]) if breach_abs < len(c) else float('nan'),
            ))

    if not valid_sig_idxs:
        return False, None, 0, [], rejected_detail

    sig_idxs   = np.array(valid_sig_idxs, dtype=np.intp)
    first_abs  = int(sig_idxs[0]) + win_start
    valid_from = int(ts_arr[first_abs])
    details    = [(int(i + 1), int(ts_arr[i + win_start])) for i in sig_idxs]
    return True, valid_from, int(sig_idxs.size), details, rejected_detail


def calc_sma_cloud_bs_debug(h: np.ndarray, l: np.ndarray,
                             c: np.ndarray, o: np.ndarray,
                             ts_arr: np.ndarray, pivot_win_ts: int,
                             pivot_end_ts: int,
                             want_sell: bool):
    """
    Extended version of calc_sma_cloud_bs_signals for debug_single output.
    v54: same pivot Hi/Lo filter as calc_sma_cloud_bs_signals.
    Returns (found, valid_from_ts, n_total_signals, signal_details_list, rejected_detail)
    where signal_details_list = [(candle_offset_in_window, ts_ms), ...]
    and   rejected_detail     = [(offset, ts, ref_level, breach_ts, breach_close), ...]
    """
    n = len(c)
    sma_h   = _sma(h, 20); sma_l = _sma(l, 20)
    sma_mid = (sma_h + sma_l) / 2.0
    bull_cloud = c >= sma_mid; bear_cloud = ~bull_cloud

    bb_basis   = _sma(c, 20)
    # ⚡ v55: Population std via E[X²]−E[X]² — avoids pd.Series.rolling.std
    bb_std_arr = np.sqrt(np.maximum(0.0, _sma(c ** 2, 20) - bb_basis ** 2))
    bb_upper   = bb_basis + 2.5 * bb_std_arr
    # ⚡ v55: sma_b_arr ≡ bb_basis when both periods == 20; reuse directly.
    sma_b_arr  = bb_basis

    N = 20
    # ⚡ v55: 6× pd.Series.rolling.mean → 6× _sma (np.convolve — 7-16× faster)
    raw_bu_up = _sma((c > bb_upper).astype(np.float64), N)
    raw_bu_dn = _sma((c < bb_upper).astype(np.float64), N)
    raw_bb_up = _sma((c > bb_basis).astype(np.float64), N)
    raw_bb_dn = _sma((c < bb_basis).astype(np.float64), N)
    raw_sm_up = _sma((c > sma_b_arr).astype(np.float64), N)
    raw_sm_dn = _sma((c < sma_b_arr).astype(np.float64), N)

    eps = 1e-9
    A_up = raw_bu_up / np.maximum(raw_bu_up + raw_bu_dn, eps)
    B_up = raw_bb_up / np.maximum(raw_bb_up + raw_bb_dn, eps)
    C_up = raw_sm_up / np.maximum(raw_sm_up + raw_sm_dn, eps)
    A_dn = raw_bu_dn / np.maximum(raw_bu_dn + raw_bu_up, eps)
    B_dn = raw_bb_dn / np.maximum(raw_bb_dn + raw_bb_up, eps)
    C_dn = raw_sm_dn / np.maximum(raw_sm_dn + raw_sm_up, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_down = np.nan_to_num(np.where(A_up != 0,
            B_up**2 * C_up**2 + (1-A_up)*(1-B_up)*(1-C_up), np.nan), nan=0.0) * 100
        sigma_up   = np.nan_to_num(np.where(A_dn != 0,
            B_dn**2 * C_dn**2 + (1-A_dn)*(1-B_dn)*(1-C_dn), np.nan), nan=0.0) * 100

    bayes_buy_ok  = (sigma_down > sigma_up)  & (sigma_down > 15.0)
    bayes_sell_ok = (sigma_up > sigma_down)  & (sigma_up   > 15.0)

    is_bull = c >= o; is_bear = ~is_bull
    body    = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    atr_vals   = calc_atr(h, l, c, 14)
    valid_body = body > atr_vals * 0.03
    upper_wick_dom = (upper_wick >= body * 2) & valid_body
    lower_wick_dom = (lower_wick >= body * 2) & valid_body

    sell_signal = (bear_cloud & (h >= sma_l) & (c <= sma_h)
                   & ((is_bear & (c < sma_h)) | (is_bull & upper_wick_dom & (c < sma_h)))
                   & bayes_sell_ok)
    buy_signal  = (bull_cloud & (l <= sma_h) & (c >= sma_l)
                   & ((is_bull & (c > sma_l)) | (is_bear & lower_wick_dom & (c > sma_l)))
                   & bayes_buy_ok)

    win_start = int(np.searchsorted(ts_arr, pivot_win_ts))
    win_end   = int(np.searchsorted(ts_arr, pivot_end_ts))
    sig_arr   = sell_signal if want_sell else buy_signal
    sig_idxs  = np.where(sig_arr[win_start:win_end])[0]

    if sig_idxs.size == 0:
        return False, None, 0, [], []

    # ── v54/v55 Pivot Hi/Lo invalidation filter ──────────────────────────────
    _PL = 5; _PR = 5
    n_c = len(c)
    pivot_low_vals  = np.full(n_c, np.nan)
    pivot_high_vals = np.full(n_c, np.nan)
    _pw = _PL + _PR + 1
    if n_c >= _pw:
        # ⚡ v55: sliding_window_view replaces O(n) Python for-loop
        _wins = sliding_window_view(c, _pw)
        _ci   = np.arange(_PL, _PL + _wins.shape[0])
        _cv   = c[_ci]
        pivot_low_vals[_ci]  = np.where(_cv == _wins.min(axis=1), _cv, np.nan)
        pivot_high_vals[_ci] = np.where(_cv == _wins.max(axis=1), _cv, np.nan)

    valid_sig_idxs  = []
    rejected_detail = []
    pv_arr = pivot_low_vals if want_sell else pivot_high_vals

    for rel in sig_idxs:
        abs_i = rel + win_start
        ref_level = np.nan
        for pivot_i in range(abs_i - 1, -1, -1):
            if not np.isnan(pv_arr[pivot_i]):
                ref_level = pv_arr[pivot_i]; break
        if np.isnan(ref_level):
            valid_sig_idxs.append(rel); continue
        scan_closes = c[abs_i : win_end]
        breach_mask = scan_closes < ref_level if want_sell else scan_closes > ref_level
        breach_offs = np.where(breach_mask)[0]
        if breach_offs.size == 0:
            valid_sig_idxs.append(rel)
        else:
            breach_abs = abs_i + int(breach_offs[0])
            breach_ts  = int(ts_arr[breach_abs]) if breach_abs < len(ts_arr) else -1
            rejected_detail.append((
                int(rel + 1), int(ts_arr[abs_i]), float(ref_level),
                breach_ts,
                float(c[breach_abs]) if breach_abs < len(c) else float('nan'),
            ))

    if not valid_sig_idxs:
        return False, None, 0, [], rejected_detail

    sig_idxs   = np.array(valid_sig_idxs, dtype=np.intp)
    first_abs  = int(sig_idxs[0]) + win_start
    valid_from = int(ts_arr[first_abs])
    details    = [(int(i + 1), int(ts_arr[i + win_start])) for i in sig_idxs]
    return True, valid_from, int(sig_idxs.size), details, rejected_detail


# ══════════════════════════════════════════════════════════════════════
#  ASYNC FETCH WITH RETRY
# ══════════════════════════════════════════════════════════════════════
#  ASYNC FETCH WITH RETRY
# ══════════════════════════════════════════════════════════════════════

async def fetch_klines(sem, sym: str, tf: str, limit: int) -> Optional[np.ndarray]:
    """
    v58: CryptoCompare data aggregator fetch — replaces Binance FAPI.

    CryptoCompare (UK/EU-based) is NOT geo-blocked from Streamlit Cloud US servers.
    Data is sourced with e=BinanceFutures so prices match Binance futures exactly.

    Returns float64 ndarray shape (N, 6): [ts_ms, open, high, low, close, volume]
    Returns None on error / empty response.

    CryptoCompare response format:
      Data.Data[] = [{time(s), open, high, low, close, volumefrom, ...}, ...]
      Ordered OLDEST → NEWEST (no reversal needed).
      time is UNIX seconds → multiply by 1000 for ms.
    """
    global _http_session

    base_sym = sym.split("/")[0]   # "BTC" from "BTC/USDT:USDT"

    if tf not in _TF_TO_CC:
        return None
    endpoint, aggregate = _TF_TO_CC[tf]

    api_key = _get_cc_api_key()
    params: dict = {
        "fsym":      base_sym,
        "tsym":      "USDT",
        "e":         "BinanceFutures",
        "limit":     min(limit, 2000),
        "aggregate": aggregate,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{_CC_BASE}/{endpoint}"

    for _att in range(3):
        try:
            async with sem:
                async with _http_session.get(url, params=params) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        pass   # fall through to jittered back-off
                    elif resp.status != 200:
                        return None
                    else:
                        data = await resp.json(content_type=None)
                        if data.get("Response") != "Success":
                            return None
                        candles = data.get("Data", {}).get("Data", [])
                        if not candles:
                            return None
                        # Filter out zero/empty placeholder candles, build ndarray
                        rows = [
                            [
                                c["time"] * 1000,   # seconds → milliseconds
                                c["open"],
                                c["high"],
                                c["low"],
                                c["close"],
                                c["volumefrom"],    # volume in base currency
                            ]
                            for c in candles
                            if c.get("open", 0) != 0 or c.get("close", 0) != 0
                        ]
                        if not rows:
                            return None
                        return np.array(rows, dtype=np.float64)
            # Jittered back-off outside sem so other coroutines can proceed
            await asyncio.sleep(1.0 * (_att + 1) + random.random() * 0.5)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if _att < 2:
                await asyncio.sleep(0.5 * (_att + 1) + random.random() * 0.3)
        except (ValueError, KeyError):
            break
        except Exception:
            break
    return None


def _arr_to_df(arr: np.ndarray) -> pd.DataFrame:
    """Convert fetch_klines ndarray → labelled DataFrame."""
    return pd.DataFrame({
        "ts":     arr[:, 0].astype(np.int64),
        "open":   arr[:, 1],
        "high":   arr[:, 2],
        "low":    arr[:, 3],
        "close":  arr[:, 4],
        "volume": arr[:, 5],
    })


async def fetch(ex, sem, sym: str, tf: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV as DataFrame — uses direct HTTP (v38 ⚡)."""
    arr = await fetch_klines(sem, sym, tf, limit)
    if arr is None or len(arr) == 0:
        return pd.DataFrame()
    return _arr_to_df(arr)


async def fetch_raw(ex, sem, sym: str, tf: str, limit: int) -> Optional[np.ndarray]:
    """Fetch OHLCV as raw numpy array — uses direct HTTP (v38 ⚡)."""
    arr = await fetch_klines(sem, sym, tf, limit)
    if arr is None or len(arr) < 5:
        return None
    return arr


# ══════════════════════════════════════════════════════════════════════
#  SCAN STAGES
# ══════════════════════════════════════════════════════════════════════

async def stage1_worker(ex, sem, sym: str, cfg: dict):
    """
    Stage 1: Pivot pattern detection + ADX momentum filter.
    v44 ⚡ BANDWIDTH FIX: pivot_tf fetched first (7 bars); tdi_tf (80 bars) only
    fetched after the pivot pattern + age gate pass — avoids downloading 80 bars
    of tdi_tf for the ~95%+ of symbols that fail the pivot check immediately.
    Returns (want_sell, sym, detail_str, pivot_ts, pivot_win_ts, pivot_end_ts, tdi_df) or None.
    """
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]

    # ⚡ Step 1: fetch only the 7-bar pivot candles (tiny — ~560 bytes per symbol)
    arr_p = await fetch_raw(ex, sem, sym, pivot_tf, 7)
    if arr_p is None or len(arr_p) < 6:
        return None

    # v34 FIX: correct pivot timestamp anchoring
    pivot_ts           = int(arr_p[-3, 0])  # bar[-3] = prev_P = the peak/trough itself
    pivot_win_ts       = int(arr_p[-2, 0])  # bar[-2] = cur_P open = pivot FIRES = Stage 3 window start
    pivot_confirmed_ts = int(arr_p[-1, 0])  # bar[-1] open = close of cur_P = when pivot was confirmed
    pivot_end_ts       = int(time.time() * 1000)  # window open until next pivot fires; use now

    def _hlc3(row): return (row[2] + row[3] + row[4]) / 3.0
    cur_P  = _hlc3(arr_p[-2]);  prev_P = _hlc3(arr_p[-3])
    pp_P   = _hlc3(arr_p[-4]);  ppp_P  = _hlc3(arr_p[-5])

    # ⚡ Pivot pattern check BEFORE fetching tdi_tf — eliminates ~95% of symbols here
    if   cur_P < prev_P and prev_P > max(pp_P, ppp_P): want_sell = True
    elif cur_P > prev_P and prev_P < min(pp_P, ppp_P): want_sell = False
    else: return None

    # v38 FIX: pivot age gate — use per-mode threshold from cfg (48h/15M, 8h/5M)
    # ⚡ Age gate also runs before tdi_tf fetch — avoids fetch for stale pivots
    pivot_max_age_ms = cfg["pivot_max_age_ms"]
    if int(time.time() * 1000) - pivot_confirmed_ts > pivot_max_age_ms:
        return None

    # ⚡ Step 2: pivot + age gate passed — NOW fetch the 80-bar tdi_tf (only for survivors)
    da = await fetch(ex, sem, sym, tdi_tf, 80)

    if da.empty or len(da) < ADX_LEN * 2:
        return None

    adx_arr = calc_adx(da.high.values, da.low.values, da.close.values)

    # v34 FIX: ADX window pp_P (arr_p[-4]) → cur_P close (arr_p[-1])
    pp_P_ts    = int(arr_p[-4, 0])   # ADX window starts at pp_P
    adx_end_ts = int(arr_p[-1, 0])   # ADX window ends at cur_P close = pivot_confirmed_ts
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
           f"ADX_peak={adx_peak:.1f} "
           f"pivot_confirmed_ts_ms={pivot_confirmed_ts} "
           f"pivot_ts_ms={pivot_ts}")
    return (want_sell, sym, det, pivot_ts, pivot_win_ts, pivot_end_ts, da)


def stage2_worker(want_sell: bool, sym: str, detail: str, pivot_ts: int,
                  pivot_win_ts: int, pivot_end_ts: int, da: pd.DataFrame):
    """
    Stage 2: TDI direction + Keltner Channel band filter.
    Returns (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts) or None.
    """
    if da.empty or len(da) < 60:
        return None
    bear_tdi, bull_tdi = tdi_state(da.close.values[:-1])   # exclude live forming bar
    u_t, l_t           = calc_kc(da.high.values, da.low.values, da.close.values)
    c_t                = float(da.close.iloc[-2])           # last confirmed closed bar
    n_t = len(da); s15 = max(0, n_t - 16); e15 = n_t - 1
    if want_sell:
        if bear_tdi and c_t > l_t[-1] and bool(np.all(da.low.values[s15:e15] > l_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
    else:
        if bull_tdi and c_t < u_t[-1] and bool(np.all(da.high.values[s15:e15] < u_t[s15:e15])):
            return (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
    return None


async def stage3_worker(ex, sem, sym: str, want_sell: bool, detail: str,
                         pivot_ts: int, pivot_win_ts: int, pivot_end_ts: int,
                         cfg: dict, da: pd.DataFrame):
    """
    v38 Stage 3:
      3a — Cloud BS pullback gate on mid_tf (pure pass/fail).
           Window: pivot_win_ts (cur_P open = pivot fires) → pivot_end_ts (now).
      3b — QM signal gate on sig_tf or lower_tf.

    v38 changes vs v34:
      ⚡ All three TF fetches (mid_tf, sig_tf, choch_tf) fire concurrently.
      ⚡ Dynamic bar limits from actual pivot age (not hardcoded 165/270/80/95).
      FIX: exit validation — TSL flip check + KC anchor at oldest signal (first_sig_ts).
           Any tdi_tf KC breach from first_sig_ts to scan time drops the pair.
    Returns (side_str, sym, detail, pivot_ts) or None.
    """
    mid_tf   = cfg["mid_tf"]
    sig_tf   = cfg["sig_tf"]
    choch_tf = cfg["choch_tf"]   # lower_tf for MTF QM path

    is_5m_mode = sig_tf == "5m"

    # ── Dynamic bar limits based on actual pivot age ───────────────────────
    # v38 age gate caps pivot_win_ts to 48h (15M) / 8h (5M), so these limits
    # only need to cover (pivot_win_ts → now) + indicator warmup.
    # v46: _WARMUP raised 60 → 200 — KVO EMA(slow=55) needs ~200 bars to converge.
    _WARMUP   = 200
    _API_CAP  = 2000   # CryptoCompare allows up to 2000 candles per request
    _tf_ms = {
        "1m":  60_000,   "3m":  180_000,  "5m":   300_000,
        "15m": 900_000,  "30m": 1_800_000, "1h":  3_600_000,
        "4h":  14_400_000, "1d": 86_400_000,
    }
    _sig_ms        = _tf_ms.get(sig_tf,   900_000)
    _mid_ms        = _tf_ms.get(mid_tf,   3_600_000)
    _choch_ms      = _tf_ms.get(choch_tf, 300_000)
    _pivot_span_ms = pivot_end_ts - pivot_win_ts   # ms from pivot fire → now

    sig_limit  = min(_API_CAP, int(_pivot_span_ms / _sig_ms)   + _WARMUP + 10)
    mid_limit  = min(_API_CAP, int(_pivot_span_ms / _mid_ms)   + _WARMUP + 10)
    min_sig    = min(sig_limit, 80)   # validation floor scales with available data

    # Dynamic choch_tf limit — capped at configured ceiling
    _span_bars = int(_pivot_span_ms / _choch_ms) + 1
    _floor     = BOS_LR * 2 + 30
    ltf_limit  = max(_floor, min(_span_bars + _floor, cfg["choch_limit"]))

    # ── Stage 3a: Cloud BS pullback gate (mid_tf) ─────────────────────────
    # ⚡ v44 BANDWIDTH FIX: fetch mid_tf alone first — Cloud BS is a cheap pass/fail.
    # sig_tf and choch_tf (the heavy fetches — up to 650 bars of 1m/5m data) are only
    # fetched AFTER Cloud BS passes.  Symbols failing 3a pay zero cost for 3b data.
    dm = await fetch(ex, sem, sym, mid_tf, mid_limit)

    if dm.empty or len(dm) < max(BB_LEN, 20) + 10:
        return None

    end    = len(dm) - 1
    ts_mid = dm.ts.values[:end].astype(np.int64)

    _loop = asyncio.get_running_loop()

    # ⚡ offload to thread pool — keeps event loop free for I/O
    cloud_ok, _valid_from_ts, n_cloud, _, _rejected = await _loop.run_in_executor(
        _CPU_POOL,
        lambda: calc_sma_cloud_bs_signals(
            dm.high.values[:end],  dm.low.values[:end],
            dm.close.values[:end], dm.open.values[:end],
            ts_mid, pivot_win_ts, pivot_end_ts, want_sell))

    if not cloud_ok:
        return None

    # ── Stage 3b: KWV QM gate ────────────────────────────────────────────
    # ⚡ Cloud BS passed — now fetch sig_tf + choch_tf concurrently (only survivors reach here)
    ds, dl = await asyncio.gather(
        fetch(ex, sem, sym, sig_tf,   sig_limit),
        fetch(ex, sem, sym, choch_tf, ltf_limit),
    )

    if ds.empty or len(ds) < min_sig:
        return None

    ds_lower = dl if (not dl.empty and len(dl) >= 20) else pd.DataFrame()
    _ltf_zz  = 10 if is_5m_mode else None
    _ltf_pp  = 10 if is_5m_mode else None

    # ⚡ offload to thread pool
    found, sig_ts_list, sig_kind_list, \
    wait_found, wait_ts_list, wait_kind_list = await _loop.run_in_executor(
        _CPU_POOL,
        lambda: signals_kwv_qm(
            ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
            ltf_zz_len=_ltf_zz, ltf_s2_pp=_ltf_pp))

    if not found and not wait_found:
        return None

    # ── Shared exit validation ────────────────────────────────────────────────
    ts_sig_arr  = ds.ts.values.astype(np.int64)
    h_s = ds.high.values; l_s = ds.low.values; c_s = ds.close.values
    _tsl_s, _dir_s = f_swing(h_s, l_s, c_s, SWING_UTAMA)
    dir_now      = int(_dir_s[-2])
    expected_dir = -1 if want_sell else 1
    tsl_flipped  = (dir_now != expected_dir)

    h_t = da.high.values; l_t = da.low.values; c_t = da.close.values
    u_tdi, l_tdi = calc_kc(h_t, l_t, c_t)
    ts_tdi       = da.ts.values.astype(np.int64)

    def _kc_clean(anchor_ts):
        idx     = int(np.searchsorted(ts_tdi, anchor_ts, side="left"))
        c_range = c_t[idx:-1]; u_r = u_tdi[idx:-1]; l_r = l_tdi[idx:-1]
        return bool(np.all(c_range > l_r) if want_sell else np.all(c_range < u_r))

    # ── CONFIRMED path ────────────────────────────────────────────────────────
    if found:
        last_sig_ts    = sig_ts_list[-1]
        sig_bar_idx    = min(int(np.searchsorted(ts_sig_arr, last_sig_ts, side="left")), len(ds) - 1)
        last_sig_price = float(ds.close.iloc[sig_bar_idx])

        if tsl_flipped:
            found = False; sig_ts_list.clear(); sig_kind_list.clear()
        elif not _kc_clean(sig_ts_list[0]):
            found = False; sig_ts_list.clear(); sig_kind_list.clear()
        else:
            side           = "SELL" if want_sell else "BUY"
            n_sigs         = len(sig_ts_list)
            sig_label      = f"{n_sigs} sig" + ("s" if n_sigs > 1 else "")
            last_sig_kind  = "MTF" if sig_kind_list[-1] == "MTF QM" else "QM"
            det = (f"{detail} | {mid_tf.upper()}_CloudBS✓({n_cloud}) {sig_tf.upper()}_QM✓ ({sig_label})"
                   f" sig_kind={last_sig_kind}"
                   f" sig_ts_ms={last_sig_ts} sig_price={last_sig_price:.8g}")
            return (side, sym, det, pivot_ts)

    # ── WAITING path (QM fired pre-R3, or confirmed was dropped above) ────────
    if wait_found:
        if tsl_flipped:
            return None
        if not _kc_clean(wait_ts_list[0]):
            return None
        last_wait_ts    = wait_ts_list[-1]
        wait_bar_idx    = min(int(np.searchsorted(ts_sig_arr, last_wait_ts, side="left")), len(ds) - 1)
        last_wait_price = float(ds.close.iloc[wait_bar_idx])
        n_wait          = len(wait_ts_list)
        wait_label      = f"{n_wait} sig" + ("s" if n_wait > 1 else "")
        last_wait_kind  = "MTF" if wait_kind_list[-1] == "MTF QM" else "QM"
        side            = "WAIT_SELL" if want_sell else "WAIT_BUY"
        det = (f"{detail} | {mid_tf.upper()}_CloudBS✓({n_cloud}) {sig_tf.upper()}_QM⏳ ({wait_label})"
               f" sig_kind={last_wait_kind}"
               f" sig_ts_ms={last_wait_ts} sig_price={last_wait_price:.8g}")
        return (side, sym, det, pivot_ts)

    return None


# ══════════════════════════════════════════════════════════════════════
#  MAIN SCAN RUNNER
# ══════════════════════════════════════════════════════════════════════

async def run_scan(cfg: dict, progress_callback: Callable) -> dict:
    """
    Run full 4-stage pipeline over all Binance USDT perpetuals.
    v58: Markets and OHLCV sourced from CryptoCompare (no proxy, no geo-blocking).
    """
    # Load markets from CryptoCompare if not already cached
    if "markets" not in st.session_state:
        markets = await _load_binance_futures_markets()
        st.session_state["markets"] = markets

    ex = _FakeExchange(st.session_state["markets"])

    # Shared aiohttp session for all CryptoCompare kline fetches
    _scan_connector = aiohttp.TCPConnector(
        limit=200, keepalive_timeout=30, ttl_dns_cache=600,
        resolver=aiohttp.ThreadedResolver(),
    )
    _scan_session = aiohttp.ClientSession(
        connector=_scan_connector,
        timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=30),
    )
    global _http_session
    _http_session = _scan_session

    try:

        symbols = sorted([
            s for s, m in ex.markets.items()
            if m.get("type") == "swap" and m.get("active")
            and m.get("quote") == "USDT" and ":USDT" in s
        ])
        total = len(symbols)
        sem   = asyncio.Semaphore(MAX_CONCURRENT)

        state = {
            "s1_done": 0, "s2_in": 0, "s3_in": 0,
            "buy_valid": [], "sell_valid": [],
            "buy_wait":  [], "sell_wait":  [],
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

            want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r1
            state["s2_in"] += 1

            _loop = asyncio.get_running_loop()
            r2 = await _loop.run_in_executor(
                _CPU_POOL, stage2_worker,
                want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
            if r2 is None:
                return

            want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r2
            state["s3_in"] += 1

            r3 = await stage3_worker(ex, sem, sym, want_sell, detail, pivot_ts, pivot_win_ts, pivot_end_ts, cfg, da)
            if r3:
                side, sym2, det2, pt = r3
                entry = (sym2, det2, pt)
                if side == "BUY":
                    state["buy_valid"].append(entry)
                elif side == "SELL":
                    state["sell_valid"].append(entry)
                elif side == "WAIT_BUY":
                    state["buy_wait"].append(entry)
                elif side == "WAIT_SELL":
                    state["sell_wait"].append(entry)
                progress_callback(state)
                last_ui_update = time.time()

        await asyncio.gather(*[worker(s) for s in symbols], return_exceptions=True)
        progress_callback(state)  # final update
        return state
    finally:
        await ex.close()
        # v38 ⚡ close shared aiohttp session
        if not _scan_session.closed:
            await _scan_session.close()
        await _scan_connector.close()


# ══════════════════════════════════════════════════════════════════════
#  DEBUG SINGLE SYMBOL
# ══════════════════════════════════════════════════════════════════════

async def debug_single(sym_raw: str, cfg: dict, tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> list:
    """
    Debug a single symbol through all pipeline stages.
    v9j: delegates to shared stage workers — no duplicated logic.
    v18: adds Stage 4 BOS/ChoCh validation.
    Returns list of (label, status, detail) tuples.
    """
    raw       = sym_raw.strip().upper().replace(" ", "")
    raw_clean = raw.replace("/", "").replace(":", "")
    base      = raw_clean.replace("USDT", "") or raw_clean
    sym       = f"{base}/USDT:USDT"
    logs      = []

    # v58: no proxy needed — CryptoCompare session for kline fetches
    _dbg_connector = aiohttp.TCPConnector(
        limit=20, keepalive_timeout=30, ttl_dns_cache=600,
        resolver=aiohttp.ThreadedResolver(),
    )
    _dbg_session = aiohttp.ClientSession(
        connector=_dbg_connector,
        timeout=aiohttp.ClientTimeout(total=60, connect=15, sock_read=30),
    )
    global _http_session
    _http_session = _dbg_session

    try:
        markets = await _load_binance_futures_markets()
        ex = _FakeExchange(markets)
    except RuntimeError as _err:
        logs.append(("Market Load", "❌ FAIL", str(_err)))
        if not _dbg_session.closed: await _dbg_session.close()
        await _dbg_connector.close()
        return logs
    try:
        if sym not in ex.markets:
            logs.append(("Symbol", "❌ FAIL", f"'{sym}' not found in CryptoCompare BinanceFutures pairs"))
            return logs
        logs.append(("Data Source", "✅ PASS",
            f"CryptoCompare BinanceFutures — {len(ex.markets)} USDT pairs loaded"))

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

        # v34 FIX: correct pivot timestamp anchoring
        arr_p_ts           = dp["ts"].values.astype(np.int64)
        pivot_ts           = int(arr_p_ts[-3])   # bar[-3] = prev_P = peak/trough itself
        pivot_win_ts       = int(arr_p_ts[-2])   # bar[-2] = cur_P open = pivot FIRES
        pivot_confirmed_ts = int(arr_p_ts[-1])   # bar[-1] open = close of cur_P
        pivot_end_ts       = int(time.time() * 1000)
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
        # v34 FIX: ADX window pp_P (arr_p_ts[-4]) → cur_P close (arr_p_ts[-1])
        adx_arr      = calc_adx(da.high.values, da.low.values, da.close.values)
        pp_P_ts      = int(arr_p_ts[-4])   # ADX starts at pp_P
        adx_end_ts   = int(arr_p_ts[-1])   # ADX ends at cur_P close
        ts_vals      = da["ts"].values.astype(np.int64)
        window_mask  = (ts_vals >= pp_P_ts) & (ts_vals <= adx_end_ts)
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
        bear_tdi, bull_tdi = tdi_state(da.close.values[:-1])   # exclude live forming bar
        u_t, l_t    = calc_kc(da.high.values, da.low.values, da.close.values)
        c_t         = float(da.close.iloc[-2])           # last confirmed closed bar
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

        # ── Stage 3a: Cloud BS pullback gate (mid_tf) ────────────────
        is_5m_mode = sig_tf == "5m"

        # v38: dynamic bar limits based on actual pivot age (matches stage3_worker)
        # v46: _WARMUP raised 60 → 200 — KVO EMA(slow=55) needs ~200 bars to converge.
        _WARMUP   = 200
        _API_CAP  = 1500
        _tf_ms_d = {
            "1m":  60_000,   "3m":  180_000,  "5m":   300_000,
            "15m": 900_000,  "30m": 1_800_000, "1h":  3_600_000,
            "4h":  14_400_000, "1d": 86_400_000,
        }
        _pivot_span_ms_d = pivot_end_ts - pivot_win_ts
        sig_limit  = min(_API_CAP, int(_pivot_span_ms_d / _tf_ms_d.get(sig_tf,   900_000)) + _WARMUP + 10)
        mid_limit  = min(_API_CAP, int(_pivot_span_ms_d / _tf_ms_d.get(mid_tf,   3_600_000)) + _WARMUP + 10)
        _span_bars_d = int(_pivot_span_ms_d / _tf_ms_d.get(choch_tf, 300_000)) + 1
        _floor_d     = BOS_LR * 2 + 30
        ltf_limit    = max(_floor_d, min(_span_bars_d + _floor_d, cfg["choch_limit"]))
        min_sig      = min(sig_limit, 80)

        # v38 FIX: age gate uses cfg["pivot_max_age_ms"] (48h/15M, 8h/5M)
        pivot_max_age_ms = cfg["pivot_max_age_ms"]
        now_ms_age       = int(time.time() * 1000)
        if now_ms_age - pivot_confirmed_ts > pivot_max_age_ms:
            max_h = pivot_max_age_ms / 3_600_000
            age_h = (now_ms_age - pivot_confirmed_ts) / 3_600_000
            logs.append(("S3a Pivot age", "❌ FAIL",
                f"Pivot too old: confirmed {age_h:.1f}h ago (max {max_h:.0f}h for {sig_tf} mode)"))
            return logs

        dm = await fetch(ex, sem, sym, mid_tf, mid_limit)
        if dm.empty or len(dm) < max(BB_LEN, 20) + 10:
            logs.append(("S3a Cloud BS data", "❌ FAIL", f"Not enough {mid_tf} candles"))
            return logs

        end    = len(dm) - 1
        ts_mid = dm.ts.values[:end].astype(np.int64)
        win_bars = int((ts_mid >= pivot_win_ts).sum())
        logs.append(("S3a Window", "ℹ️ INFO",
            f"{win_bars} {mid_tf} candles in pivot window (from cur_P open = pivot fires)"))

        cloud_found, valid_from_ts, n_cloud_sigs, cloud_details, cloud_rejected = calc_sma_cloud_bs_debug(
            dm.high.values[:end], dm.low.values[:end],
            dm.close.values[:end], dm.open.values[:end],
            ts_mid, pivot_win_ts, pivot_end_ts, want_sell)

        now_ms_dbg = time.time() * 1000
        def _age(ts_ms):
            m = (now_ms_dbg - ts_ms) / 60_000
            return (f"{m:.0f}m ago" if m < 60 else f"{m/60:.1f}h ago" if m < 1440 else f"{m/1440:.1f}d ago")

        cloud_detail_str = (
            f"{n_cloud_sigs} Cloud BS {direction} signal(s) in window | "
            f"first at ts={valid_from_ts}  ({_age(valid_from_ts) if valid_from_ts else '—'})"
        ) if cloud_found else f"No Cloud BS {direction} signal in pivot window"
        logs.append(("S3a Cloud BS", "✅ PASS" if cloud_found else "❌ FAIL", cloud_detail_str))

        if cloud_rejected:
            for _off, _ts, _ref, _bts, _bc in cloud_rejected:
                _breach_age = f"breach at {_age(_bts)}" if _bts > 0 else "breach ts unknown"
                logs.append(("  S3a Piv-filter", "ℹ️ INFO",
                    f"Signal #{_off} rejected — pivot_ref={_ref:.8g} | close={_bc:.8g} breached it | {_breach_age}"))
        if not cloud_found:
            return logs

        # ── Stage 3b: KWV QM gate (sig_tf + lower_tf) ───────────
        ds, dl = await asyncio.gather(
            fetch(ex, sem, sym, sig_tf,   sig_limit),
            fetch(ex, sem, sym, choch_tf, ltf_limit),
        )
        if ds.empty or len(ds) < min_sig:
            logs.append(("S3b Sig data", "❌ FAIL", f"Not enough {sig_tf} candles (need ≥ {min_sig})"))
            return logs

        ds_lower = dl if (not dl.empty and len(dl) >= 20) else pd.DataFrame()
        ltf_label = f"{choch_tf} ({len(dl)} bars)" if not dl.empty else f"{choch_tf} (unavailable)"
        logs.append(("S3b Fetched", "ℹ️ INFO",
            f"{sig_tf}={len(ds)} bars  |  lower_tf={ltf_label}"))

        _is_5m = sig_tf == "5m"
        found, sig_ts_list, sig_kind_list, \
        wait_found, wait_ts_list, wait_kind_list = signals_kwv_qm(
            ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
            ltf_zz_len=10 if _is_5m else None,
            ltf_s2_pp =10 if _is_5m else None)

        n_sigs = len(sig_ts_list)
        n_wait = len(wait_ts_list)
        if found:
            n_qm  = sig_kind_list.count("QM")
            n_mtf = sig_kind_list.count("MTF QM")
            kind_sum = (f"QM×{n_qm}" if n_qm else "") + (" MTF QM×" + str(n_mtf) if n_mtf else "")
            sig_detail = f"{n_sigs} confirmed signal(s) [{kind_sum.strip()}] | latest: {_age(sig_ts_list[-1])}"
            if wait_found:
                sig_detail += f" | {n_wait} waiting (pre-R3)"
        else:
            sig_detail = f"No confirmed QM signals in pivot window on {sig_tf}/{choch_tf}"
            if wait_found:
                sig_detail += f" | {n_wait} waiting signal(s) pre-R3"

        logs.append(("S3b QM Signal", "✅ PASS" if found else "❌ FAIL", sig_detail))
        if not found:
            return logs

        # Show individual signals
        for i, (sig_ts_ms, kind) in enumerate(zip(sig_ts_list, sig_kind_list), start=1):
            tf_label = choch_tf if kind == "MTF QM" else sig_tf
            si = min(int(np.searchsorted(ds.ts.values.astype(np.int64), sig_ts_ms, side="left")), len(ds) - 1)
            logs.append((f"  Signal #{i}", "ℹ️ INFO",
                f"[{kind}  {tf_label}]  ts={sig_ts_ms}  ({_age(sig_ts_ms)})  "
                f"price={float(ds.close.iloc[si]):.8g}"))

        last_sig_ts    = sig_ts_list[-1]
        sig_bar_idx    = min(int(np.searchsorted(ds.ts.values.astype(np.int64), last_sig_ts, side="left")), len(ds) - 1)
        last_sig_price = float(ds.close.iloc[sig_bar_idx])
        sig_times = _fmt_ts(last_sig_ts, tz_h, tz_label, time_fmt)

        # ── Stage 3b exit validation (v38 FIX) ───────────────────────
        # (a) TSL flip check — has dirMain on sig_tf flipped against direction?
        _tsl_dbg, _dir_dbg = f_swing(ds.high.values, ds.low.values, ds.close.values, SWING_UTAMA)
        dir_now_dbg  = int(_dir_dbg[-2])   # last closed bar
        exp_dir_dbg  = -1 if want_sell else 1
        tsl_ok_dbg   = (dir_now_dbg == exp_dir_dbg)
        dir_str      = "bear (-1)" if dir_now_dbg == -1 else "bull (+1)"
        logs.append(("S3b TSL Check", "✅ PASS" if tsl_ok_dbg else "❌ FAIL",
            f"sig_tf dirMain={dir_str} | need {'bear (-1)' if want_sell else 'bull (+1)'} | "
            f"{'OK — trend intact' if tsl_ok_dbg else 'FLIPPED — trend reversed since signal'}"))
        if not tsl_ok_dbg:
            return logs

        # (b) KC clean check — tdi_tf (da), anchor = oldest signal (sig_ts_list[0])
        _h_t = da.high.values;  _l_t = da.low.values;  _c_t = da.close.values
        _u_tdi, _l_tdi  = calc_kc(_h_t, _l_t, _c_t)
        _ts_tdi         = da.ts.values.astype(np.int64)
        kc_anchor_ts    = sig_ts_list[0]   # oldest signal in window (v38 FIX: was last)
        kc_anchor_idx   = int(np.searchsorted(_ts_tdi, kc_anchor_ts, side="left"))
        _c_range        = _c_t[kc_anchor_idx:-1]
        _u_range        = _u_tdi[kc_anchor_idx:-1]
        _l_range        = _l_tdi[kc_anchor_idx:-1]
        if want_sell:
            kc_clean_dbg = bool(np.all(_c_range > _l_range))
        else:
            kc_clean_dbg = bool(np.all(_c_range < _u_range))
        kc_anchor_age = (int(time.time() * 1000) - kc_anchor_ts) / 3_600_000
        n_checked     = max(len(_c_range), 0)
        logs.append(("S3b KC Clean", "✅ PASS" if kc_clean_dbg else "❌ FAIL",
            f"tdi_tf KC check from first_sig ({kc_anchor_age:.1f}h ago) → now | "
            f"{n_checked} {tdi_tf} bars | "
            f"{'no breach — price stayed inside KC' if kc_clean_dbg else 'KC BREACHED — price crossed band since first signal'}"))
        if not kc_clean_dbg:
            return logs

        logs.append(("Signal Confirmed", "✅ VALID",
            f"{direction} | {n_sigs} QM signal(s) on {sig_tf}/{choch_tf} | "
            f"last: {sig_times} | price={last_sig_price:.8g}"))

        return logs
    finally:
        await ex.close()
        # v38 ⚡ close debug aiohttp session
        if not _dbg_session.closed:
            await _dbg_session.close()
        await _dbg_connector.close()


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _run_async(coro):
    """
    v10: Run coroutine in a dedicated background thread with its own event loop.

    This avoids two failure modes on Streamlit Cloud:
      1. 'This event loop is already running' — Streamlit runs its own async loop,
         so get_event_loop().run_until_complete() conflicts with it.
      2. 'Cannot reuse already awaited coroutine' — the old try/except approach
         called asyncio.run(coro) on a coroutine that had already been (partially)
         started by the failed run_until_complete() attempt.

    Running in a fresh thread gives us a brand-new event loop with no conflicts.
    """
    import threading

    result_box: list = [None]
    error_box:  list = [None]

    def _target():
        try:
            result_box[0] = asyncio.run(coro)
        except Exception as exc:          # noqa: BLE001
            error_box[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]


def _parse_row(direction: str, sym: str, det: str, pivot_ts: int,
               now_ms: int, mode_key: str, timestamp: str,
               tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> dict:
    """
    Parse a result row into structured fields.
    tz_h: UTC offset in fractional hours for all timestamps
    """
    p      = _re.search(r"P=([\d.]+)",                    det)
    prev   = _re.search(r"prev_(?:peak|trough)=([\d.]+)", det)
    adxpk  = _re.search(r"ADX_peak=([\d.]+)",             det)
    adxend = _re.search(r"ADX_cur=([\d.]+)",              det)
    bb_m   = _re.search(r"(\w+)_CloudBS",               det)
    sig_m  = _re.search(r"\[(\w+)_QM",                    det)
    sig_ts = _re.search(r"sig_ts_ms=(\d+)",               det)
    sig_px = _re.search(r"sig_price=([\d.eE+\-]+)",       det)
    age_h  = round((now_ms - pivot_ts) / 3_600_000, 1)

    # Signal bar time — apply user timezone
    if sig_ts:
        sig_dt = _fmt_ts(int(sig_ts.group(1)), tz_h, tz_label, time_fmt)
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
        "Pivot_Age_h":    age_h,
        "Scan_Time":      timestamp,
        "Mode":           mode_key.upper(),
    }


def _parse_det_card(det: str, tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> dict:
    """Parse detail string into card display fields.
    ADX shown is ADX_cur (current strength at window close), falling back to
    ADX_peak if cur is unavailable — matches v21 CLI _parse_det() behavior.
    """
    # v21 FIX: prefer ADX_cur (current value) over ADX_peak (historical peak)
    adx      = _re.search(r"ADX_cur=([\d.]+)",        det) or _re.search(r"ADX_peak=([\d.]+)", det)
    adx_pk   = _re.search(r"ADX_peak=([\d.]+)",       det)
    bb_m     = _re.search(r"(\w+)_CloudBS",           det)
    sig_m    = _re.search(r"\[(\w+)_QM",              det)
    sig_ts   = _re.search(r"sig_ts_ms=(\d+)",         det)
    sig_px   = _re.search(r"sig_price=([\d.eE+\-]+)", det)
    n_sigs   = _re.search(r"\((\d+) sig",             det)
    kind_m   = _re.search(r"sig_kind=(\w+)",          det)
    cloud_m  = _re.search(r"CloudBS\u2713\((\d+)\)",  det)

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
        if age_h < 1:    age_str = f"{age_h*60:.0f}m ago"
        elif age_h < 24: age_str = f"{age_h:.1f}h ago"
        else:            age_str = f"{age_h/24:.1f}d ago"
        sig_time = _fmt_ts(int(sig_ts.group(1)), tz_h, tz_label, time_fmt)
    else:
        age_h = 0.0; age_str = "—"; sig_time = "—"

    # ADX values
    adx_v    = f"{float(adx.group(1)):.0f}"    if adx    else "—"
    adx_pk_v = f"{float(adx_pk.group(1)):.0f}" if adx_pk else adx_v

    # Signal kind label — "MTF" → "MTF QM", "QM" → "QM"
    raw_kind = kind_m.group(1) if kind_m else "QM"
    sig_kind = "MTF QM" if raw_kind == "MTF" else "QM"

    return {
        "price":    price_str,
        "adx":      adx_v,
        "adx_peak": adx_pk_v,
        "bb_tf":    bb_m.group(1).upper()  if bb_m   else "—",
        "sig_tf":   sig_m.group(1).upper() if sig_m  else "—",
        "age_str":  age_str,
        "age_h":    str(age_h),
        "sig_time": sig_time,
        "n_sigs":   n_sigs.group(1) if n_sigs else "1",
        "sig_kind": sig_kind,
        "n_cloud":  cloud_m.group(1) if cloud_m else "—",
    }


def _sort_signals(lst: list, sort_key: str) -> list:
    """
    Sort a list of (sym, det) tuples.
    sort_key: "newest"  — newest signal first  (sig_ts_ms descending)
              "oldest"  — oldest signal first   (sig_ts_ms ascending)
              "name_az" — symbol name A → Z
              "name_za" — symbol name Z → A
    """
    def _sig_ts(item):
        m = _re.search(r"sig_ts_ms=(\d+)", item[1])
        return int(m.group(1)) if m else 0

    if sort_key == "oldest":
        return sorted(lst, key=_sig_ts, reverse=False)
    if sort_key == "name_az":
        return sorted(lst, key=lambda x: x[0])
    if sort_key == "name_za":
        return sorted(lst, key=lambda x: x[0], reverse=True)
    # Default: newest first
    return sorted(lst, key=_sig_ts, reverse=True)


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT APP LAYOUT
# ══════════════════════════════════════════════════════════════════════

def _init_session():
    """Ensure all session_state keys exist on first load."""
    # Persist timezone + time format across refreshes via query params
    _qp_tz = st.query_params.get("tz", None)
    _tz_default = _qp_tz if (_qp_tz and _qp_tz in TIMEZONES) else TZ_DEFAULT

    _qp_tf = st.query_params.get("tf", None)
    _tf_default = _qp_tf if (_qp_tf and _qp_tf in TIME_FMTS) else TIME_FMT_DEFAULT

    defaults = {
        "scan_done":    False,
        "scan_state":   None,
        "scan_elapsed": 0.0,
        "scan_mode":    "15m",
        "df_final":     None,
        "buy_valid":    [],
        "sell_valid":   [],
        "buy_wait":     [],
        "sell_wait":    [],
        "csv_bytes":    None,
        "txt_bytes":    None,
        "csv_fname":    "",
        "txt_fname":    "",
        "results_tab":  "all",
        "results_sort": "newest",
        "scan_ts_int":  0,
        "scan_now_ms":  0,
        "scan_timestamp": "",
        "buy_valid_full":  [],
        "sell_valid_full": [],
        "buy_wait_full":   [],
        "sell_wait_full":  [],
        "tz_key":       _tz_default,
        "time_fmt":     _tf_default,
        "show_tz_panel": False,
        "scan_mode_sel": "15m",
        # ── v57: Auto-loop ────────────────────────────────────────────
        "auto_loop":        False,
        "auto_loop_15m":    True,
        "auto_loop_5m":     True,
        "next_scan_time":   0.0,
        "auto_scan_running": False,
        "auto_scan_mode":   None,   # "15m" or "5m" — which mode is currently queued
        # ── v57: Telegram ─────────────────────────────────────────────
        "tg_enabled":       True,   # send alert after each scan with signals
        # ── v57: Signal history ───────────────────────────────────────
        "signal_history":   [],     # list of dicts — accumulated across all scans
        "history_seen":     set(),  # dedup key: (symbol, signal_ts_ms, mode)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sc_counters_html(nbv: int, nsv: int, nbw: int, nsw: int,
                      s2: int, s3: int, elapsed: float, total: int, done: int) -> str:
    spd = done / max(elapsed, 0.01)
    return f"""
<div class="sc-counters">
  <div class="sc-cnt g">
    <div class="cnt-lbl">BUY ✅</div>
    <div class="cnt-val">{nbv}</div>
  </div>
  <div class="sc-cnt r">
    <div class="cnt-lbl">SELL ✅</div>
    <div class="cnt-val">{nsv}</div>
  </div>
  <div class="sc-cnt gy">
    <div class="cnt-lbl">BUY ⏳</div>
    <div class="cnt-val">{nbw}</div>
  </div>
  <div class="sc-cnt gy">
    <div class="cnt-lbl">SELL ⏳</div>
    <div class="cnt-val">{nsw}</div>
  </div>
  <div class="sc-cnt b">
    <div class="cnt-lbl">S2 Passed</div>
    <div class="cnt-val">{s2}</div>
  </div>
  <div class="sc-cnt b">
    <div class="cnt-lbl">S3 Passed</div>
    <div class="cnt-val">{s3}</div>
  </div>
  <div class="sc-cnt gy">
    <div class="cnt-lbl">Scanned</div>
    <div class="cnt-val">{done}</div>
    <div class="cnt-sub">{spd:.0f}/s · {elapsed:.0f}s</div>
  </div>
</div>"""


def _sc_summary_html(total: int, elapsed: float, bv: int,
                     sv: int, bw: int, sw: int, mode_key: str,
                     s2: int = 0, s3: int = 0) -> str:
    all_s = bv + sv + bw + sw
    spd   = total / max(elapsed, 0.01)
    confirmed = bv + sv
    waiting   = bw + sw
    funnel = f"{total}&thinsp;→&thinsp;{s2}&thinsp;→&thinsp;{s3}&thinsp;→&thinsp;<b style='color:var(--gold)'>{all_s}</b>" if s2 or s3 else f"{total} sym"
    return (
        f'<div class="sc-summary">'
        f'<span class="ss-title">&#9989; Scan <span>Complete</span></span>'
        f'<span class="ss-chip g">&#9650; BUY {bv}</span>'
        f'<span class="ss-chip r">&#9660; SELL {sv}</span>'
        + (f'<span class="ss-chip gd">&#9650;&#8987; {bw}</span>' if bw else '')
        + (f'<span class="ss-chip rd">&#9660;&#8987; {sw}</span>' if sw else '')
        + f'<span class="ss-funnel">{funnel}</span>'
        + f'<span class="ss-meta">'
        f'{elapsed:.1f}s &middot; {spd:.0f}/s &middot; <b>{mode_key.upper()}</b>'
        f'</span>'
        f'</div>'
    )


def _signal_cards_html(entries: list, is_buy: bool, is_valid: bool, mode_key: str = "15m",
                       grid_cls: str = "sc-grid",
                       tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> str:
    """Rich signal cards: symbol | direction | price | time | ADX · age · kind."""
    if not entries:
        label = ("BUY" if is_buy else "SELL") + (" confirmed" if is_valid else " waiting")
        return f'<div class="sc-empty"><div class="ico">&#128269;</div><p>No {label} signals.</p></div>'

    card_cls = ("buy" if is_buy else "sell") + ("" if is_valid else " wait")

    if is_buy and is_valid:
        dir_cls, dir_txt = "dir-buy",    "&#9650; BUY"
    elif is_buy:
        dir_cls, dir_txt = "dir-buy-w",  "&#9650; WAIT"
    elif is_valid:
        dir_cls, dir_txt = "dir-sell",   "&#9660; SELL"
    else:
        dir_cls, dir_txt = "dir-sell-w", "&#9660; WAIT"

    pulse = '<span class="sc-card-pulse"></span>' if is_valid else ''

    cards = []
    for sym, det in entries:
        p    = _parse_det_card(det, tz_h, tz_label, time_fmt)
        base = sym.split("/")[0].replace("USDT", "").replace("BUSD", "").replace("USD", "") or sym.split("/")[0]

        # Kind badge: MTF QM gets special treatment
        kind_badge = (
            '<span class="sc-kind-badge mtf">MTF</span>'
            if p["sig_kind"] == "MTF QM"
            else '<span class="sc-kind-badge qm">QM</span>'
        )

        # ADX color class
        try:
            adx_val = float(p["adx"])
            adx_cls = "adx-hi" if adx_val >= 40 else ("adx-med" if adx_val >= 25 else "adx-lo")
        except (ValueError, TypeError):
            adx_cls = ""

        cards.append(
            f'<div class="sc-card {card_cls}">'
            # Row 1: symbol + direction badge
            f'<div class="sc-card-row1">'
            f'<span class="sc-card-sym">{base}{pulse}</span>'
            f'<span class="sc-card-dir {dir_cls}">{dir_txt}</span>'
            f'</div>'
            # Row 2: price
            f'<div class="sc-card-price">{p["price"]}</div>'
            # Row 3: signal time
            f'<div class="sc-card-info"><b>{p["sig_tf"] if p["sig_tf"] != "—" else mode_key.upper()}</b>'
            f'&nbsp;{p["sig_time"]}</div>'
            # Row 4: ADX · age · kind
            f'<div class="sc-card-meta">'
            f'<span class="sc-adx {adx_cls}">ADX&nbsp;{p["adx"]}</span>'
            f'<span class="sc-age">{p["age_str"]}</span>'
            f'{kind_badge}'
            f'</div>'
            f'</div>'
        )
    return f'<div class="{grid_cls}">{"".join(cards)}</div>'


def _all_signals_two_col_html(bv_list, sv_list, bw_list, sw_list, mode_key: str,
                              tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> str:
    """Render All tab with BUY and SELL confirmed + waiting signal cards."""
    parts = []
    if bv_list:
        parts.append(_signal_cards_html(bv_list, True,  True, mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    if sv_list:
        parts.append(_signal_cards_html(sv_list, False, True, mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    if bw_list or sw_list:
        parts.append('<div class="sc-wait-label">⏳ WAITING — KWV window open (R2 passed), awaiting R3 confirmation</div>')
    if bw_list:
        parts.append(_signal_cards_html(bw_list, True,  False, mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    if sw_list:
        parts.append(_signal_cards_html(sw_list, False, False, mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    return "".join(parts) if parts else '<div class="sc-empty"><div class="ico">&#128269;</div><p>No signals.</p></div>'


def main():
    # ── Start the 24/7 background scheduler (once per process) ───────────
    _start_bg_scheduler()

    _init_session()

    # ── Timezone + Time format — load from session / query params ──────
    tz_key   = st.session_state.get("tz_key", TZ_DEFAULT)
    tz_h     = TIMEZONES.get(tz_key, 0.0)
    sign_s   = "+" if tz_h >= 0 else "-"
    ah_s     = int(abs(tz_h)); am_s = int(round((abs(tz_h)-ah_s)*60))
    tz_short = f"UTC{sign_s}{ah_s:02d}:{am_s:02d}" if am_s else f"UTC{sign_s}{ah_s}"
    time_fmt = st.session_state.get("time_fmt", TIME_FMT_DEFAULT)

    # ── Header (full-width) ───────────────────────────────────────────
    show_tz = st.session_state.get("show_tz_panel", False)
    gear_label = "⚙️ Settings ▲" if show_tz else "⚙️ Settings"

    hdr_col, gear_col = st.columns([7, 1])
    with hdr_col:
        st.markdown(f"""
<div class="sc-header">
  <div class="sc-header-left">
    <h1><i class="ico">&#9889;</i><span class="brand">Binance Futures</span> <span class="accent">Scanner</span></h1>
    <div class="sub">
      CryptoCompare Data
      <span class="dot">&bull;</span>
      Multi-Stage Pipeline
      <span class="dot">&bull;</span>
      Pine Accurate
    </div>
  </div>
  <div class="sc-header-right">
    <span class="sc-badge blue">&#128640; v58</span>
    <span class="sc-badge green">&#10004; 3 Stages</span>
    <span class="sc-tz-badge">&#127758; {tz_short}</span>
    <span class="sc-tz-badge" style="background:rgba(0,180,216,0.07);color:var(--blue);border-color:rgba(0,180,216,0.28);">&#128336; {time_fmt.upper()}</span>
    <span class="sc-tz-badge" style="{'background:rgba(0,230,118,0.07);color:#00e676;border-color:rgba(0,230,118,0.28)' if st.session_state.get('tg_enabled', True) else 'background:rgba(255,255,255,0.04);color:#5a5a72;border-color:rgba(255,255,255,0.08)'};">&#128232; TG {'ON' if st.session_state.get('tg_enabled', True) else 'OFF'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    with gear_col:
        st.markdown('<div style="height:1.35rem"></div>', unsafe_allow_html=True)
        if st.button(gear_label, key="gear_btn", width="stretch",
                     help="Timezone & time format settings"):
            st.session_state["show_tz_panel"] = not show_tz
            st.rerun()

    # ── Settings panel (hidden until gear clicked) ────────────────────
    if st.session_state.get("show_tz_panel", False):
        st.markdown(
            "<div class='sc-settings-panel'>",
            unsafe_allow_html=True)
        s_c1, s_c2, s_c3 = st.columns([3, 2, 1])
        with s_c1:
            st.markdown(
                '<div class="sc-tz-label">&#127758;&nbsp; Display Timezone</div>',
                unsafe_allow_html=True)
            tz_sel_idx = TZ_LABELS.index(tz_key) if tz_key in TZ_LABELS else 0
            new_tz = st.selectbox(
                "tz_selector", TZ_LABELS, index=tz_sel_idx,
                key="tz_selectbox", label_visibility="collapsed",
            )
            if new_tz != tz_key:
                st.session_state["tz_key"] = new_tz
                st.query_params["tz"] = new_tz
                tz_key   = new_tz
                tz_h     = TIMEZONES.get(new_tz, 0.0)
                sign_s   = "+" if tz_h >= 0 else "-"
                ah_s     = int(abs(tz_h)); am_s = int(round((abs(tz_h)-ah_s)*60))
                tz_short = f"UTC{sign_s}{ah_s:02d}:{am_s:02d}" if am_s else f"UTC{sign_s}{ah_s}"
                st.rerun()
        with s_c2:
            st.markdown(
                '<div class="sc-tz-label">&#128336;&nbsp; Time Format</div>',
                unsafe_allow_html=True)
            tf_c1, tf_c2 = st.columns(2)
            with tf_c1:
                if st.button("24h", key="btn_24h", width="stretch",
                             type="primary" if time_fmt == "24h" else "secondary"):
                    if time_fmt != "24h":
                        st.session_state["time_fmt"] = "24h"
                        st.query_params["tf"] = "24h"
                        st.rerun()
            with tf_c2:
                if st.button("12h", key="btn_12h", width="stretch",
                             type="primary" if time_fmt == "12h" else "secondary"):
                    if time_fmt != "12h":
                        st.session_state["time_fmt"] = "12h"
                        st.query_params["tf"] = "12h"
                        st.rerun()
        with s_c3:
            st.markdown('<div style="height:1.55rem"></div>', unsafe_allow_html=True)
            if st.button("✕ Close", key="close_tz", width="stretch"):
                st.session_state["show_tz_panel"] = False
                st.rerun()
        st.markdown(
            '<div style="font-size:0.67rem;color:#5a5a72;margin-top:2px;font-family:var(--mono)">'
            'Settings persist across reloads via URL query params</div>',
            unsafe_allow_html=True)

        # ── Background scheduler status ───────────────────────────
        st.markdown(
            '<div style="border-top:1px solid #1e1e2a;margin:8px 0 6px"></div>',
            unsafe_allow_html=True)
        with _bg_lock:
            _bgs = dict(_bg_status)
        _run_col  = "#00e676" if not _bgs["error"] else "#ff5252"
        _run_icon = "🔄 Running…" if _bgs["running"] else ("⚠ Error" if _bgs["error"] else "✅ Idle")
        st.markdown(
            f'<div style="font-size:0.72rem;color:#7ecfea;margin-bottom:4px;">'
            f'<b>🤖 Background Scheduler (24/7)</b></div>'
            f'<div style="font-size:0.68rem;font-family:var(--mono);color:#9a9aaa;'
            f'background:rgba(0,180,216,0.05);border:1px solid rgba(0,180,216,0.15);'
            f'border-radius:6px;padding:5px 10px;line-height:1.7;">'
            f'Status: <b style="color:{_run_col}">{_run_icon}</b> &nbsp;·&nbsp; '
            f'Last run: <b>{_bgs["last_run"]}</b> &nbsp;·&nbsp; '
            f'Mode: <b>{_bgs["last_mode"]}</b> &nbsp;·&nbsp; '
            f'New signals: <b>{_bgs["last_signals"]}</b><br>'
            f'Next scan: <b>{_bgs["next_run"]}</b>'
            + (f'<br><span style="color:#ff5252">⚠ {_bgs["error"]}</span>' if _bgs["error"] else '')
            + f'</div>',
            unsafe_allow_html=True)

        # ── Telegram sub-section ──────────────────────────────────────
        st.markdown(
            '<div style="border-top:1px solid #1e1e2a;margin:8px 0 6px"></div>',
            unsafe_allow_html=True)
        tg_c1, tg_c2, tg_c3 = st.columns([3, 2, 2])
        with tg_c1:
            _tg_enabled = st.session_state.get("tg_enabled", True)
            if st.checkbox(
                "📨 Telegram alerts after each scan",
                value=_tg_enabled,
                key="tg_toggle",
                help="Send a Telegram message whenever signals are found",
            ):
                st.session_state["tg_enabled"] = True
            else:
                st.session_state["tg_enabled"] = False
        with tg_c2:
            _tg_tok, _tg_cid = _tg_creds()
            st.markdown(
                f'<div style="font-size:0.7rem;color:#5a5a72;padding-top:6px;font-family:var(--mono)">'
                f'Bot: …{_tg_tok[-8:]}<br>Chat: {_tg_cid}</div>',
                unsafe_allow_html=True)
        with tg_c3:
            st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
            if st.button("📨 Test Telegram", key="tg_test_btn", width="stretch"):
                import threading as _thr_tg
                _tg_ok = [False]
                def _tg_test_thread():
                    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
                    _tg_ok[0] = _tg_send_sync(
                        f"✅ <b>Binance Futures Scanner v58</b>\n"
                        f"🕐 {ts}\n"
                        f"Telegram alerts are working!"
                    )
                _t = _thr_tg.Thread(target=_tg_test_thread, daemon=True)
                _t.start(); _t.join(timeout=20)
                if _tg_ok[0]:
                    st.success("✅ Test message sent!")
                else:
                    st.error("❌ Failed — check bot token and chat ID in Secrets")
        st.markdown("</div>", unsafe_allow_html=True)

    tab_scan, tab_history, tab_debug = st.tabs(["&#128269;  Full Scan", "&#128203;  History", "&#128027;  Debug Symbol"])

    # ══ TAB 1: FULL SCAN ══════════════════════════════════════════════
    with tab_scan:

        # ── Data source banner ────────────────────────────────────────
        _n_markets = len(st.session_state.get("markets", {}))
        _cc_key_set = bool(_get_cc_api_key())
        _key_chip = (
            '<span style="background:rgba(0,230,118,0.12);color:#00e676;'
            'border:1px solid rgba(0,230,118,0.3);padding:2px 8px;border-radius:12px;'
            'font-size:0.72rem;font-family:var(--mono)">🔑 API key set</span>'
            if _cc_key_set else
            '<span style="background:rgba(255,202,40,0.1);color:#ffca28;'
            'border:1px solid rgba(255,202,40,0.25);padding:2px 8px;border-radius:12px;'
            'font-size:0.72rem;font-family:var(--mono)">⚠ No API key — free tier</span>'
        )
        _pairs_chip = (
            f'<span style="background:rgba(0,180,216,0.1);color:#00b4d8;'
            f'border:1px solid rgba(0,180,216,0.25);padding:2px 8px;border-radius:12px;'
            f'font-size:0.72rem;font-family:var(--mono)">📊 {_n_markets} pairs cached</span>'
            if _n_markets else ""
        )
        st.markdown(
            f'<div class="sc-proxy-ok" style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'📡&nbsp;<b>Data: CryptoCompare → BinanceFutures</b>'
            f'&nbsp;&mdash;&nbsp;{_key_chip}'
            + (f'&nbsp;{_pairs_chip}' if _pairs_chip else '')
            + f'&nbsp;&middot;&nbsp;<span style="color:#5a8a5a;font-size:0.78rem">'
            f'EU/UK servers · no proxy needed</span></div>',
            unsafe_allow_html=True)

        # ── Mode + Timeframes row ─────────────────────────────────────
        mode_key = st.session_state["scan_mode_sel"]
        cfg = MODES[mode_key]

        # Section label
        st.markdown(
            "<div class='sc-section-label'>&#9881;&nbsp; Scan Configuration</div>",
            unsafe_allow_html=True)

        # Mode selector — Streamlit buttons ARE the cards (clickable/tappable)
        lbl_15m = (
            f"{'✅' if mode_key == '15m' else '📊'} 15M · Swing\n"
            f"D → 4H → 1H → 15M"
        )
        lbl_5m  = (
            f"{'✅' if mode_key == '5m' else '⚡'} 5M · Scalp\n"
            f"4H → 1H → 15M → 5M"
        )
        st.markdown("<div class='sc-mode-selector'>", unsafe_allow_html=True)
        _mc1, _mc2 = st.columns(2)
        with _mc1:
            if st.button(lbl_15m, key="mode_btn_15m", width="stretch",
                         type="primary" if mode_key == "15m" else "secondary"):
                st.session_state["scan_mode_sel"] = "15m"
                st.rerun()
        with _mc2:
            if st.button(lbl_5m, key="mode_btn_5m", width="stretch",
                         type="primary" if mode_key == "5m" else "secondary"):
                st.session_state["scan_mode_sel"] = "5m"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # TF pipeline flow diagram
        st.markdown(
            f"<div class='sc-tf-flow'>"
            f"<div class='sc-tf-node'>"
            f"  <span class='tf-stage'>S1 · Pivot</span>"
            f"  <span class='tf-val'>{cfg['pivot_tf'].upper()}</span>"
            f"  <span class='tf-role'>HLC3 chain + ADX</span>"
            f"</div>"
            f"<div class='sc-tf-node'>"
            f"  <span class='tf-stage'>S2 · TDI/KC</span>"
            f"  <span class='tf-val'>{cfg['tdi_tf'].upper()}</span>"
            f"  <span class='tf-role'>RSI · Keltner</span>"
            f"</div>"
            f"<div class='sc-tf-node'>"
            f"  <span class='tf-stage'>S3a · CloudBS</span>"
            f"  <span class='tf-val'>{cfg['mid_tf'].upper()}</span>"
            f"  <span class='tf-role'>Cloud BS gate</span>"
            f"</div>"
            f"<div class='sc-tf-node'>"
            f"  <span class='tf-stage'>S3b · QM</span>"
            f"  <span class='tf-val'>{cfg['sig_tf'].upper()}</span>"
            f"  <span class='tf-role'>Pressure → QM</span>"
            f"</div>"
            f"<div class='sc-tf-node'>"
            f"  <span class='tf-stage'>S3b · MTF QM</span>"
            f"  <span class='tf-val'>{cfg['choch_tf'].upper()}</span>"
            f"  <span class='tf-role'>Lower-TF QM</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True)

        # ── Rule pills ────────────────────────────────────────────────
        st.markdown(
            f"<div class='sc-pills-v2'>"
            f"<span class='sc-pill-v2 s1'><span class='pill-num-v2'>S1</span>"
            f"{cfg['pivot_tf'].upper()} Pivot <span class='pill-arr-v2'>&#8594;</span> ADX&gt;{ADX_TH:.0f}</span>"
            f"<span class='sc-pill-v2 s2'><span class='pill-num-v2'>S2</span>"
            f"TDI direction <span class='pill-arr-v2'>&#8594;</span> KC Band</span>"
            f"<span class='sc-pill-v2 s3'><span class='pill-num-v2'>S3a</span>"
            f"{cfg['mid_tf'].upper()} CloudBS gate</span>"
            f"<span class='sc-pill-v2 s4'><span class='pill-num-v2'>S3b</span>"
            f"KWV window <span class='pill-arr-v2'>&#8594;</span> {cfg['sig_tf'].upper()}/{cfg['choch_tf'].upper()} QM</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Auto-loop controls ────────────────────────────────────────
        auto_on = st.session_state.get("auto_loop", False)
        al_c1, al_c2, al_c3, al_c4 = st.columns([3, 2, 2, 1])
        with al_c1:
            if st.button(
                "⏹ Stop Auto Scan" if auto_on else "🔄 Auto Scan (15-min)",
                key="auto_loop_btn",
                type="primary" if auto_on else "secondary",
                width="stretch",
                help="Automatically run scans every 15 minutes aligned to :00/:15/:30/:45",
            ):
                st.session_state["auto_loop"] = not auto_on
                if not auto_on:
                    # Compute first upcoming :00/:15/:30/:45 mark
                    _now = time.time()
                    _el  = (int(time.gmtime(_now).tm_min) % 15) * 60 + int(time.gmtime(_now).tm_sec)
                    st.session_state["next_scan_time"] = _now + (15 * 60) - _el
                st.rerun()
        with al_c2:
            al_15m = st.session_state.get("auto_loop_15m", True)
            if st.checkbox("15M scans", value=al_15m, key="al_15m_cb",
                           disabled=not auto_on):
                st.session_state["auto_loop_15m"] = True
            else:
                st.session_state["auto_loop_15m"] = False
        with al_c3:
            al_5m = st.session_state.get("auto_loop_5m", True)
            if st.checkbox("5M scans", value=al_5m, key="al_5m_cb",
                           disabled=not auto_on):
                st.session_state["auto_loop_5m"] = True
            else:
                st.session_state["auto_loop_5m"] = False
        with al_c4:
            if st.button("&#128260;", key="clear_mkts", width="stretch",
                         help="Refresh market list — clears cache and reloads from Binance"):
                st.session_state.pop("markets", None)
                st.rerun()

        # Auto-loop countdown + status
        if auto_on:
            nxt = st.session_state.get("next_scan_time", 0.0)
            secs_left = max(0.0, nxt - time.time())
            mins_left, s_left = divmod(int(secs_left), 60)
            _modes_active = []
            if st.session_state.get("auto_loop_15m", True): _modes_active.append("15M")
            if st.session_state.get("auto_loop_5m",  True): _modes_active.append("5M")
            _modes_str = " + ".join(_modes_active) if _modes_active else "none"
            st.markdown(
                f'<div style="background:rgba(0,180,216,0.07);border:1px solid rgba(0,180,216,0.22);'
                f'border-radius:8px;padding:6px 14px;font-size:0.82rem;color:#7ecfea;'
                f'display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px">'
                f'<span>&#128338; Next scan in <b>{mins_left}:{s_left:02d}</b></span>'
                f'<span style="color:#5a8a9a">·</span>'
                f'<span>Modes: <b>{_modes_str}</b></span>'
                f'<span style="color:#5a8a9a">·</span>'
                f'<span>History: <b>{len(st.session_state.get("signal_history",[]))} signals</b></span>'
                f'</div>',
                unsafe_allow_html=True)

        # ── Action buttons ────────────────────────────────────────────
        scan_clicked = st.button("&#128640;  Start Scan", type="primary", key="scan_btn",
                                 width="stretch")

        # ── Auto-loop trigger: fire scan when countdown reaches zero ──
        if auto_on:
            nxt = st.session_state.get("next_scan_time", 0.0)
            if time.time() >= nxt and not st.session_state.get("auto_scan_running", False):
                # Determine which mode to run next
                _al_mode = st.session_state.get("auto_scan_mode", None)
                _do_15m  = st.session_state.get("auto_loop_15m", True)
                _do_5m   = st.session_state.get("auto_loop_5m",  True)
                if _al_mode is None:
                    # Start of cycle — run 15M first (or 5M if 15M disabled)
                    _al_mode = "15m" if _do_15m else ("5m" if _do_5m else None)
                elif _al_mode == "15m":
                    # 15M just ran — next is 5M
                    _al_mode = "5m" if _do_5m else None
                elif _al_mode == "5m":
                    # 5M is queued (set by post-15M bookkeeping) — run it now
                    pass  # keep _al_mode as "5m"
                else:
                    _al_mode = None  # both done — advance to next 15-min mark

                if _al_mode is not None:
                    st.session_state["auto_scan_running"] = True
                    st.session_state["auto_scan_mode"]    = _al_mode
                    st.session_state["scan_mode_sel"]     = _al_mode
                    scan_clicked = True   # inject a virtual click for the chosen mode
                    mode_key = _al_mode   # refresh local var — assigned before this block
                    cfg      = MODES[mode_key]
                else:
                    # Both modes done — advance clock
                    _now = time.time()
                    _el  = (int(time.gmtime(_now).tm_min) % 15) * 60 + int(time.gmtime(_now).tm_sec)
                    st.session_state["next_scan_time"]    = _now + (15 * 60) - _el
                    st.session_state["auto_scan_mode"]    = None
                    st.session_state["auto_scan_running"] = False
                    st.rerun()
            elif auto_on and secs_left > 0:
                # Refresh every ~10 s so the countdown updates
                time.sleep(10)
                st.rerun()

        st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.5rem 0 0.7rem'>",
                    unsafe_allow_html=True)

        # ── SCAN EXECUTION ────────────────────────────────────────────
        if scan_clicked:
            st.session_state.update({
                "scan_done": False, "df_final": None,
                "buy_valid": [], "sell_valid": [],
                "csv_bytes": None, "txt_bytes": None,
            })
            t0 = time.time()

            prog_bar = st.progress(0.0, text="Connecting to Binance…")
            ctr_ph   = st.empty()

            def update_ui(state: dict):
                """Render progress — must only be called from the main thread."""
                total   = state["total"]
                done    = state["s1_done"]
                elapsed = time.time() - t0
                pct     = done / max(total, 1)
                all_s   = len(state["buy_valid"]) + len(state["sell_valid"])
                spd = done / max(elapsed, 0.01)
                prog_bar.progress(
                    min(pct, 1.0),
                    text=f"Scanning {done}/{total} · {spd:.0f} sym/s · "
                         f"S2:{state['s2_in']} S3:{state['s3_in']} · Signals:{all_s}"
                )
                ctr_ph.markdown(
                    _sc_counters_html(
                        len(state["buy_valid"]),
                        len(state["sell_valid"]),
                        len(state.get("buy_wait", [])),
                        len(state.get("sell_wait", [])),
                        state["s2_in"], state["s3_in"], elapsed, total, done),
                    unsafe_allow_html=True,
                )

            # ── Fix: run the async scan in a background thread and deliver
            #    state snapshots back to the main thread via a Queue so that
            #    all Streamlit widget calls (update_ui) happen here, on the
            #    main script thread, avoiding NoSessionContext errors.
            _state_q: queue.Queue = queue.Queue()
            _result_box: list = [None]
            _error_box:  list = [None]

            def _queue_callback(s: dict):
                """Called from the worker thread — only enqueues, never touches Streamlit."""
                try:
                    _state_q.put_nowait({k: list(v) if isinstance(v, list) else v
                                         for k, v in s.items()})
                except queue.Full:
                    pass  # drop if full; UI will catch the next update

            def _scan_target():
                try:
                    _result_box[0] = asyncio.run(run_scan(cfg, _queue_callback))
                except Exception as exc:  # noqa: BLE001
                    _error_box[0] = exc

            _scan_thread = threading.Thread(target=_scan_target, daemon=True)
            _scan_thread.start()

            # Poll the queue on the main thread so Streamlit UI updates are safe
            try:
                while _scan_thread.is_alive():
                    try:
                        _snap = _state_q.get(timeout=0.15)
                        update_ui(_snap)
                    except queue.Empty:
                        pass
                _scan_thread.join()
                # Drain any remaining snapshots
                while not _state_q.empty():
                    try:
                        update_ui(_state_q.get_nowait())
                    except queue.Empty:
                        break
                if _error_box[0] is not None:
                    raise _error_box[0]
                state = _result_box[0]
            except Exception as e:
                st.error(f"Scan failed: {e}")
                st.exception(e)
                state = None

            if state:
                elapsed    = time.time() - t0
                total      = state["total"]
                buy_valid  = sorted(state["buy_valid"],  key=lambda x: x[0])
                sell_valid = sorted(state["sell_valid"], key=lambda x: x[0])
                buy_wait   = sorted(state.get("buy_wait",  []), key=lambda x: x[0])
                sell_wait  = sorted(state.get("sell_wait", []), key=lambda x: x[0])

                prog_bar.progress(1.0, text=f"Done — {total} symbols in {elapsed:.1f}s")
                ctr_ph.empty()

                now_ms    = int(time.time() * 1000)
                ts_int    = int(time.time())
                timestamp = _fmt_ts(now_ms, tz_h, tz_key, time_fmt)

                all_results = (
                    [("BUY",       s, d, p) for s, d, p in buy_valid] +
                    [("SELL",      s, d, p) for s, d, p in sell_valid] +
                    [("WAIT_BUY",  s, d, p) for s, d, p in buy_wait]  +
                    [("WAIT_SELL", s, d, p) for s, d, p in sell_wait]
                )

                if all_results:
                    all_rows = [
                        _parse_row(dir_, s, d, p, now_ms, mode_key, timestamp, tz_h, tz_key, time_fmt)
                        for dir_, s, d, p in all_results
                    ]
                    df_final = pd.DataFrame(all_rows)
                    csv_buf  = io.StringIO()
                    df_final.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")

                    txt_buf = io.StringIO()
                    txt_buf.write(f"BINANCE FUTURES SCANNER  —  {mode_key.upper()} MODE\n")
                    txt_buf.write(f"Scan Time : {timestamp}\n")
                    txt_buf.write(f"Timezone  : {tz_key}\n")
                    txt_buf.write(f"Time Fmt  : {time_fmt.upper()}\n")
                    txt_buf.write(f"Symbols   : {total}  |  Elapsed : {elapsed:.1f}s\n")
                    txt_buf.write(f"BUY  : {len(buy_valid)}\n")
                    txt_buf.write(f"SELL : {len(sell_valid)}\n")
                    txt_buf.write("=" * 72 + "\n")
                    for dir_, group_label, group in [
                        ("BUY",  "BUY",  buy_valid),
                        ("SELL", "SELL", sell_valid),
                    ]:
                        if not group: continue
                        txt_buf.write(f"\n{'─'*28} {group_label} {'─'*28}\n")
                        for sym, det, pts in group:
                            r = _parse_row(dir_, sym, det, pts, now_ms, mode_key, timestamp, tz_h, tz_key, time_fmt)
                            txt_buf.write(
                                f"  {r['Symbol']:<24}  Price={r['Signal_Price']}\n"
                                f"  {'':24}  ADX peak={r['ADX_Peak']}  end={r['ADX_End']}  Age={r['Pivot_Age_h']}h\n"
                                f"  {'':24}  BB={r['BB_TF']}  Signal={r['Signal_TF']}\n"
                                f"  {'':24}  Pine Signal Time: {r['Signal_Time']}\n\n"
                            )
                    txt_bytes = txt_buf.getvalue().encode("utf-8")

                    st.session_state.update({
                        "scan_done":    True,
                        "scan_state":   state,
                        "scan_elapsed": elapsed,
                        "scan_mode":    mode_key,
                        "scan_ts_int":  ts_int,
                        "scan_now_ms":  now_ms,
                        "scan_timestamp": timestamp,
                        "df_final":     df_final,
                        "buy_valid":    [(s, d) for s, d, _ in buy_valid],
                        "sell_valid":   [(s, d) for s, d, _ in sell_valid],
                        "buy_wait":     [(s, d) for s, d, _ in buy_wait],
                        "sell_wait":    [(s, d) for s, d, _ in sell_wait],
                        "buy_valid_full":  [(s, d, p) for s, d, p in buy_valid],
                        "sell_valid_full": [(s, d, p) for s, d, p in sell_valid],
                        "buy_wait_full":   [(s, d, p) for s, d, p in buy_wait],
                        "sell_wait_full":  [(s, d, p) for s, d, p in sell_wait],
                    })

                    # ── v57: accumulate signals into history ──────────────
                    _hist    = st.session_state.setdefault("signal_history", [])
                    _seen    = st.session_state.setdefault("history_seen", set())
                    _new_rows = []
                    for row in all_rows:
                        # Dedup key: symbol + signal timestamp + mode
                        _dk = (row["Symbol"], row.get("Signal_Time", ""), row["Mode"])
                        if _dk not in _seen:
                            _seen.add(_dk)
                            _hist.append(row)
                            _new_rows.append(row)
                    # Append only new rows to CSV on disk
                    if _new_rows:
                        import os as _os
                        _hist_path = "signals_history.csv"
                        _hist_df   = pd.DataFrame(_new_rows)
                        _write_hdr = not _os.path.exists(_hist_path)
                        _hist_df.to_csv(_hist_path, mode="a", index=False, header=_write_hdr)

                    # ── v57: Telegram alert ───────────────────────────────
                    if st.session_state.get("tg_enabled", True):
                        _tg_bv = [(s, d) for s, d, _ in buy_valid]
                        _tg_sv = [(s, d) for s, d, _ in sell_valid]
                        _tg_bw = [(s, d) for s, d, _ in buy_wait]
                        _tg_sw = [(s, d) for s, d, _ in sell_wait]
                        import threading as _thr
                        _thr.Thread(
                            target=_tg_send_signals,
                            args=(_tg_bv, _tg_sv, _tg_bw, _tg_sw,
                                  mode_key.upper(), elapsed, total),
                            daemon=True,
                        ).start()

                    # ── v57: auto-loop bookkeeping after scan completes ───
                    if st.session_state.get("auto_loop", False):
                        _do_15m = st.session_state.get("auto_loop_15m", True)
                        _do_5m  = st.session_state.get("auto_loop_5m",  True)
                        _cur    = st.session_state.get("auto_scan_mode", mode_key)
                        # Decide what comes next
                        if _cur == "15m" and _do_5m:
                            st.session_state["auto_scan_mode"]    = "5m"
                            st.session_state["auto_scan_running"] = False
                        else:
                            # Advance clock to next 15-min mark
                            _now = time.time()
                            _el  = (int(time.gmtime(_now).tm_min) % 15) * 60 + int(time.gmtime(_now).tm_sec)
                            st.session_state["next_scan_time"]    = _now + (15 * 60) - _el
                            st.session_state["auto_scan_mode"]    = None
                            st.session_state["auto_scan_running"] = False
                else:
                    st.session_state.update({
                        "scan_done":    True,
                        "scan_state":   state,
                        "scan_elapsed": elapsed,
                        "scan_mode":    mode_key,
                        "df_final":     None,
                        "buy_valid": [], "sell_valid": [],
                    })
                    # ── v57: auto-loop bookkeeping (no-signal path) ───────
                    if st.session_state.get("auto_loop", False):
                        _do_15m = st.session_state.get("auto_loop_15m", True)
                        _do_5m  = st.session_state.get("auto_loop_5m",  True)
                        _cur    = st.session_state.get("auto_scan_mode", mode_key)
                        if _cur == "15m" and _do_5m:
                            st.session_state["auto_scan_mode"]    = "5m"
                            st.session_state["auto_scan_running"] = False
                        else:
                            _now = time.time()
                            _el  = (int(time.gmtime(_now).tm_min) % 15) * 60 + int(time.gmtime(_now).tm_sec)
                            st.session_state["next_scan_time"]    = _now + (15 * 60) - _el
                            st.session_state["auto_scan_mode"]    = None
                            st.session_state["auto_scan_running"] = False
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
            r_tz_key   = st.session_state.get("tz_key", TZ_DEFAULT)
            r_tz_h     = TIMEZONES.get(r_tz_key, 0.0)
            r_time_fmt = st.session_state.get("time_fmt", TIME_FMT_DEFAULT)

            bv_list_raw = st.session_state["buy_valid"]
            sv_list_raw = st.session_state["sell_valid"]
            bw_list_raw = st.session_state.get("buy_wait",  [])
            sw_list_raw = st.session_state.get("sell_wait", [])
            bv, sv = len(bv_list_raw), len(sv_list_raw)
            bw, sw = len(bw_list_raw), len(sw_list_raw)
            all_sigs = bv + sv + bw + sw

            # Persistent summary banner
            st.markdown(
                _sc_summary_html(total, elapsed, bv, sv, bw, sw, mode_key_r,
                                 s2=state.get("s2_in", 0), s3=state.get("s3_in", 0)),
                unsafe_allow_html=True)

            if all_sigs == 0:
                st.markdown(
                    '<div class="sc-empty"><div class="ico">&#128301;</div>'
                    '<p>No signals &mdash; market conditions did not meet all 3 stage filters.</p></div>',
                    unsafe_allow_html=True)
            else:
                # ── Sort control — pill buttons ───────────────────────
                cur_sort = st.session_state.get("results_sort", "newest")
                _sort_opts = [
                    ("🕐 Newest", "newest"),
                    ("🕛 Oldest", "oldest"),
                    ("🔤 A→Z",    "name_az"),
                    ("🔡 Z→A",    "name_za"),
                ]
                st.markdown("<div class='sc-sort-wrap'>", unsafe_allow_html=True)
                _scols = st.columns(len(_sort_opts))
                for (_slbl, _skey), _scol in zip(_sort_opts, _scols):
                    with _scol:
                        _is_active = cur_sort == _skey
                        if st.button(
                            _slbl, key=f"sort_btn_{_skey}", width="stretch",
                            type="primary" if _is_active else "secondary"
                        ):
                            if _skey != cur_sort:
                                st.session_state["results_sort"] = _skey
                                st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                _sort_lbl = next(l for l, k in _sort_opts if k == cur_sort)

                st.markdown("<div style='margin-bottom:0.3rem'></div>",
                            unsafe_allow_html=True)

                # Apply sort to all lists
                cur_sort = st.session_state.get("results_sort", "newest")
                bv_list = _sort_signals(bv_list_raw, cur_sort)
                sv_list = _sort_signals(sv_list_raw, cur_sort)
                bw_list = _sort_signals(bw_list_raw, cur_sort)
                sw_list = _sort_signals(sw_list_raw, cur_sort)

                # Signal card tabs
                tab_labels = [
                    f"All ({all_sigs})",
                    f"BUY ✅ {bv}",
                    f"SELL ✅ {sv}",
                    f"BUY ⏳ {bw}",
                    f"SELL ⏳ {sw}",
                ]
                t_all, t_bv, t_sv, t_bw, t_sw = st.tabs(tab_labels)

                with t_all:
                    if bv_list or sv_list or bw_list or sw_list:
                        st.markdown(
                            _all_signals_two_col_html(bv_list, sv_list, bw_list, sw_list, mode_key_r, r_tz_h, r_tz_key, r_time_fmt),
                            unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sc-empty"><div class="ico">&#128269;</div><p>No signals.</p></div>',
                                    unsafe_allow_html=True)

                with t_bv:
                    st.markdown(_signal_cards_html(bv_list, True, True, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)
                with t_sv:
                    st.markdown(_signal_cards_html(sv_list, False, True, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)
                with t_bw:
                    st.markdown('<div class="sc-wait-label">⏳ WAITING — QM setup complete, awaiting R3 confirmation</div>', unsafe_allow_html=True)
                    st.markdown(_signal_cards_html(bw_list, True, False, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)
                with t_sw:
                    st.markdown('<div class="sc-wait-label">⏳ WAITING — QM setup complete, awaiting R3 confirmation</div>', unsafe_allow_html=True)
                    st.markdown(_signal_cards_html(sw_list, False, False, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)

                # Full table + export — rebuilt dynamically with current sort ──
                _bv_full = st.session_state.get("buy_valid_full",  [])
                _sv_full = st.session_state.get("sell_valid_full", [])
                _bw_full = st.session_state.get("buy_wait_full",   [])
                _sw_full = st.session_state.get("sell_wait_full",  [])

                # Sort the full tuples the same way as the display lists
                def _sort_full(lst, sk):
                    """Sort (sym,det,pts) tuples using same keys as _sort_signals."""
                    def _ts(item):
                        m = _re.search(r"sig_ts_ms=(\d+)", item[1])
                        return int(m.group(1)) if m else 0
                    if sk == "oldest":   return sorted(lst, key=_ts)
                    if sk == "name_az":  return sorted(lst, key=lambda x: x[0])
                    if sk == "name_za":  return sorted(lst, key=lambda x: x[0], reverse=True)
                    return sorted(lst, key=_ts, reverse=True)  # newest

                _bv_s = _sort_full(_bv_full, cur_sort)
                _sv_s = _sort_full(_sv_full, cur_sort)
                _bw_s = _sort_full(_bw_full, cur_sort)
                _sw_s = _sort_full(_sw_full, cur_sort)

                _exp_now_ms    = st.session_state.get("scan_now_ms",    int(time.time()*1000))
                _exp_timestamp = st.session_state.get("scan_timestamp", "")
                _exp_ts_int    = st.session_state.get("scan_ts_int",    int(time.time()))
                _sort_labels   = {"newest": "Newest first", "oldest": "Oldest first",
                                  "name_az": "A→Z", "name_za": "Z→A"}
                _sort_lbl      = _sort_labels.get(cur_sort, cur_sort)

                # Build sorted export rows
                _all_sorted = (
                    [("BUY",       s, d, p) for s, d, p in _bv_s] +
                    [("SELL",      s, d, p) for s, d, p in _sv_s] +
                    [("WAIT_BUY",  s, d, p) for s, d, p in _bw_s] +
                    [("WAIT_SELL", s, d, p) for s, d, p in _sw_s]
                )
                if _all_sorted:
                    _exp_rows = [
                        _parse_row(dir_, s, d, p, _exp_now_ms, mode_key_r,
                                   _exp_timestamp, r_tz_h, r_tz_key, r_time_fmt)
                        for dir_, s, d, p in _all_sorted
                    ]
                    _df_sorted = pd.DataFrame(_exp_rows)

                    # CSV bytes
                    _cbuf = io.StringIO()
                    _df_sorted.to_csv(_cbuf, index=False)
                    _csv_bytes = _cbuf.getvalue().encode("utf-8")

                    # TXT bytes
                    _tbuf = io.StringIO()
                    _tbuf.write(f"BINANCE FUTURES SCANNER  —  {mode_key_r.upper()} MODE\n")
                    _tbuf.write(f"Scan Time : {_exp_timestamp}\n")
                    _tbuf.write(f"Timezone  : {r_tz_key}\n")
                    _tbuf.write(f"Time Fmt  : {r_time_fmt.upper()}\n")
                    _tbuf.write(f"Sort      : {_sort_lbl}\n")
                    _tbuf.write(f"Symbols   : {total}  |  Elapsed : {elapsed:.1f}s\n")
                    _tbuf.write(f"BUY  : {bv}\n")
                    _tbuf.write(f"SELL : {sv}\n")
                    _tbuf.write(f"BUY  (WAIT) : {bw}\n")
                    _tbuf.write(f"SELL (WAIT) : {sw}\n")
                    _tbuf.write("=" * 72 + "\n")
                    for _dir, _glbl, _grp in [
                        ("BUY",       "BUY  ✅ CONFIRMED",   _bv_s),
                        ("SELL",      "SELL ✅ CONFIRMED",   _sv_s),
                        ("WAIT_BUY",  "BUY  ⏳ WAITING R3",  _bw_s),
                        ("WAIT_SELL", "SELL ⏳ WAITING R3",  _sw_s),
                    ]:
                        if not _grp: continue
                        _tbuf.write(f"\n{'─'*28} {_glbl} {'─'*28}\n")
                        for _sym, _det, _pts in _grp:
                            _r = _parse_row(_dir, _sym, _det, _pts,
                                            _exp_now_ms, mode_key_r, _exp_timestamp,
                                            r_tz_h, r_tz_key, r_time_fmt)
                            _tbuf.write(
                                f"  {_r['Symbol']:<24}  Price={_r['Signal_Price']}\n"
                                f"  {'':24}  ADX peak={_r['ADX_Peak']}  end={_r['ADX_End']}  Age={_r['Pivot_Age_h']}h\n"
                                f"  {'':24}  BB={_r['BB_TF']}  Signal={_r['Signal_TF']}\n"
                                f"  {'':24}  Pine Signal Time: {_r['Signal_Time']}\n\n"
                            )
                    _txt_bytes = _tbuf.getvalue().encode("utf-8")

                    if df_final is not None and not df_final.empty:
                        with st.expander("&#128203; Full Data Table + Export", expanded=False):
                            display_cols = [
                                "Direction", "Symbol", "Signal_Price",
                                "ADX_Peak", "ADX_End", "BB_TF", "Signal_TF",
                                "Signal_Time", "Pivot_Age_h",
                            ]
                            col_cfg = {
                                "Direction":    st.column_config.TextColumn("Dir",       width=85),
                                "Symbol":       st.column_config.TextColumn("Symbol",    width=150),
                                "Signal_Price": st.column_config.TextColumn("Price",     width=100),
                                "ADX_Peak":     st.column_config.NumberColumn("ADX Pk",  format="%.1f", width=75),
                                "ADX_End":      st.column_config.NumberColumn("ADX End", format="%.1f", width=75),
                                "BB_TF":        st.column_config.TextColumn("BB TF",     width=58),
                                "Signal_TF":    st.column_config.TextColumn("Sig TF",    width=62),
                                "Signal_Time":  st.column_config.TextColumn("Signal Time", width=160),
                                "Pivot_Age_h":  st.column_config.NumberColumn("Age h",   format="%.1f", width=62),
                            }
                            # Show sorted df in table
                            st.dataframe(
                                _df_sorted[display_cols], width="stretch",
                                hide_index=True,
                                height=min(540, 50 + 36 * len(_df_sorted)),
                                column_config=col_cfg)
                            st.caption(f"Sort: {_sort_lbl} — export matches this order")

                            ec1, ec2, _sp2 = st.columns([1, 1, 2])
                            ec1.download_button(
                                "&#128196; Export CSV",
                                data=_csv_bytes,
                                file_name=f"signals_{mode_key_r}_{_sort_lbl.replace(' ','_').replace('→','').replace('↓','')}_{_exp_ts_int}.csv",
                                mime="text/csv", width="stretch")
                            ec2.download_button(
                                "&#128221; Export TXT",
                                data=_txt_bytes,
                                file_name=f"signals_{mode_key_r}_{_sort_lbl.replace(' ','_').replace('→','').replace('↓','')}_{_exp_ts_int}.txt",
                                mime="text/plain", width="stretch")

    # ══ TAB 2: SIGNAL HISTORY ════════════════════════════════════════
    with tab_history:
        st.markdown("#### &#128203; Signal History — All Scans This Session")

        _hist = st.session_state.get("signal_history", [])

        if not _hist:
            st.markdown(
                '<div style="padding:2rem;text-align:center;color:#5a5a72">'
                '&#128269; No signals accumulated yet. Run a scan to start building history.</div>',
                unsafe_allow_html=True)
        else:
            _df_hist = pd.DataFrame(_hist)

            # ── Summary counters ──────────────────────────────────────
            _hbv = (_df_hist["Direction"] == "BUY").sum()
            _hsv = (_df_hist["Direction"] == "SELL").sum()
            _hwb = (_df_hist["Direction"] == "WAIT_BUY").sum()
            _hws = (_df_hist["Direction"] == "WAIT_SELL").sum()
            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">'
                f'<span style="background:rgba(0,230,118,0.1);color:#00e676;border:1px solid rgba(0,230,118,0.25);'
                f'padding:4px 12px;border-radius:16px;font-size:0.82rem">&#9650; BUY {_hbv}</span>'
                f'<span style="background:rgba(255,64,96,0.1);color:#ff4060;border:1px solid rgba(255,64,96,0.25);'
                f'padding:4px 12px;border-radius:16px;font-size:0.82rem">&#9660; SELL {_hsv}</span>'
                f'<span style="background:rgba(0,230,118,0.05);color:#7ecfa0;border:1px solid rgba(0,230,118,0.12);'
                f'padding:4px 12px;border-radius:16px;font-size:0.82rem">&#8987; WAIT BUY {_hwb}</span>'
                f'<span style="background:rgba(255,64,96,0.05);color:#cf7e8a;border:1px solid rgba(255,64,96,0.12);'
                f'padding:4px 12px;border-radius:16px;font-size:0.82rem">&#8987; WAIT SELL {_hws}</span>'
                f'<span style="color:#5a5a72;padding:4px 12px;font-size:0.82rem">'
                f'Total: <b style="color:#c0c0d0">{len(_df_hist)}</b></span>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Sort control ──────────────────────────────────────────
            _hs_c1, _hs_c2 = st.columns([3, 1])
            with _hs_c1:
                _hist_sort = st.selectbox(
                    "Sort by",
                    ["Newest first", "Oldest first", "Symbol A→Z", "BUY first", "SELL first"],
                    index=0, key="hist_sort", label_visibility="collapsed"
                )
            with _hs_c2:
                if st.button("&#128465; Clear History", key="clear_hist", width="stretch",
                             help="Clear all accumulated signals from this session"):
                    st.session_state["signal_history"] = []
                    st.session_state["history_seen"]   = set()
                    st.rerun()

            # Apply sort
            if _hist_sort == "Newest first":
                _df_show = _df_hist.iloc[::-1].reset_index(drop=True)
            elif _hist_sort == "Oldest first":
                _df_show = _df_hist.reset_index(drop=True)
            elif _hist_sort == "Symbol A→Z":
                _df_show = _df_hist.sort_values("Symbol").reset_index(drop=True)
            elif _hist_sort == "BUY first":
                _df_show = _df_hist.sort_values("Direction").reset_index(drop=True)
            else:
                _df_show = _df_hist.sort_values("Direction", ascending=False).reset_index(drop=True)

            # Display table
            _hist_display_cols = [
                "Scan_Time", "Direction", "Symbol", "Signal_Price",
                "Signal_TF", "Signal_Time", "ADX_Peak", "Pivot_Age_h", "Mode",
            ]
            _hist_display_cols = [c for c in _hist_display_cols if c in _df_show.columns]
            _hist_col_cfg = {
                "Scan_Time":    st.column_config.TextColumn("Scan Time",   width=150),
                "Direction":    st.column_config.TextColumn("Dir",         width=90),
                "Symbol":       st.column_config.TextColumn("Symbol",      width=140),
                "Signal_Price": st.column_config.TextColumn("Price",       width=100),
                "Signal_TF":    st.column_config.TextColumn("TF",          width=55),
                "Signal_Time":  st.column_config.TextColumn("Signal Time", width=150),
                "ADX_Peak":     st.column_config.NumberColumn("ADX",       format="%.1f", width=60),
                "Pivot_Age_h":  st.column_config.NumberColumn("Age h",     format="%.1f", width=60),
                "Mode":         st.column_config.TextColumn("Mode",        width=55),
            }
            st.dataframe(
                _df_show[_hist_display_cols],
                width="stretch",
                hide_index=True,
                height=min(600, 50 + 36 * min(len(_df_show), 50)),
                column_config=_hist_col_cfg,
            )
            st.caption(f"{len(_df_show)} total signals · also saved to signals_history.csv on disk")

            # ── Download ──────────────────────────────────────────────
            _hcbuf = io.StringIO()
            _df_show.to_csv(_hcbuf, index=False)
            st.download_button(
                "&#128196; Download Full History CSV",
                data=_hcbuf.getvalue().encode("utf-8"),
                file_name=f"signals_history_{int(time.time())}.csv",
                mime="text/csv",
                width="stretch",
            )

    # ══ TAB 3: DEBUG SYMBOL ═══════════════════════════════════════════
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
                               width="stretch")

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
                "<span class='sc-stage-dot dot-3'>S3</span> KWV window gate (R1→R2→R3)<br>"
                "<span class='sc-stage-dot dot-3'>S3</span> QM Strat1/Strat2 + timestamps"
                "</div>",
                unsafe_allow_html=True,
            )

        if dbg_go:
            with st.spinner(f"Running pipeline on {sym_input.strip().upper()}…"):
                try:
                    _dbg_tz_key  = st.session_state.get("tz_key", TZ_DEFAULT)
                    _dbg_tz_h    = TIMEZONES.get(_dbg_tz_key, 0.0)
                    _dbg_tf      = st.session_state.get("time_fmt", TIME_FMT_DEFAULT)
                    logs = _run_async(debug_single(sym_input, dbg_cfg, _dbg_tz_h, _dbg_tz_key, _dbg_tf))
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
                    width="stretch", hide_index=True,
                    height=50 + 38 * len(rows),
                )

                last = logs[-1]
                if "PASS" in last[1] or "VALID" in last[1]:
                    st.success("All stages passed — Signal confirmed")
                else:
                    st.error(f"Failed at {last[0]}")
                    st.caption(f"Detail: {last[2]}")


if __name__ == "__main__":
    main()
