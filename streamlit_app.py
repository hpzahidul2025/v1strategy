"""
Binance Futures Scanner - ULTRA-FAST Edition v38
Streamlit Web App — Binance via proxy (bypasses geo-block on cloud servers)

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
from typing import Optional, Callable

import queue
import threading
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
# SWING_ALT removed — only used by deprecated signals_tf (dead code)
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

MODES = {
    "15m": {
        "pivot_tf":        "1d",
        "tdi_tf":          "4h",
        "mid_tf":          "1h",
        "sig_tf":          "15m",
        # v13: BOS/ChoCh validated on 5m
        "choch_tf":        "5m",
        "choch_limit":     650,
        "pivot_max_age_ms": 48 * 3_600_000,   # 48 hours
        "label":           "15M — Daily → 4H → 1H → 15M",
    },
    "5m": {
        "pivot_tf":        "4h",
        "tdi_tf":          "1h",
        "mid_tf":          "15m",
        "sig_tf":          "5m",
        # v13: BOS/ChoCh validated on 1m
        "choch_tf":        "1m",
        "choch_limit":     550,
        "pivot_max_age_ms": 8 * 3_600_000,    # 8 hours
        "label":           "5M — 4H → 1H → 15M → 5M",
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
    page_title="Binance Futures Scanner v38",
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
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
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
    padding: 0.55rem 0.7rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 3px;
    position: relative;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  .sc-card:hover, .sc-card:active {
    transform: translateY(-3px) scale(1.02);
  }

  /* BUY confirmed — vivid green glow */
  .sc-card.buy {
    border-left: 3px solid var(--green);
    background: linear-gradient(135deg, rgba(0,230,118,0.06) 0%, rgba(15,15,21,1) 60%);
    box-shadow: 0 0 0 0 rgba(0,230,118,0);
  }
  .sc-card.buy:hover, .sc-card.buy:active {
    border-color: var(--green);
    background: linear-gradient(135deg, rgba(0,230,118,0.13) 0%, rgba(15,15,21,1) 65%);
    box-shadow: 0 6px 28px rgba(0,230,118,0.22), 0 2px 8px rgba(0,0,0,0.4);
  }

  /* SELL confirmed — vivid red glow */
  .sc-card.sell {
    border-left: 3px solid var(--red);
    background: linear-gradient(135deg, rgba(255,64,96,0.06) 0%, rgba(15,15,21,1) 60%);
    box-shadow: 0 0 0 0 rgba(255,64,96,0);
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

  /* ── Settings panel (gear toggle) ───────────────────────────────── */
  .sc-settings-panel {
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

  /* ── Sort / Filter control ───────────────────────────────────────── */
  .sc-sort-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 0.7rem;
  }
  .sc-sort-icon-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 13px 5px 10px;
    border-radius: 20px;
    border: 1px solid var(--border2);
    background: var(--surface2);
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--text2);
    font-family: var(--mono);
    white-space: nowrap;
    letter-spacing: 0.04em;
  }
  .sc-sort-icon-pill.active {
    background: rgba(0,180,216,0.1);
    border-color: rgba(0,180,216,0.45);
    color: var(--blue);
    box-shadow: 0 0 10px rgba(0,180,216,0.12);
  }
  /* Make the selectbox inline and compact — flush with pill */
  .sc-sort-select [data-testid="stSelectbox"] {
    margin: 0 !important;
  }
  .sc-sort-select [data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    min-height: 34px !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
    font-family: var(--mono) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    cursor: pointer !important;
  }
  .sc-sort-select [data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(0,180,216,0.5) !important;
    color: var(--blue) !important;
  }
  /* Hide the label */
  .sc-sort-select label { display: none !important; }
  /* Mobile — make select full-touch-friendly */
  @media (max-width: 640px) {
    .sc-sort-row { gap: 6px !important; margin: 0 0 0.5rem !important; }
    .sc-sort-icon-pill { font-size: 0.7rem !important; padding: 5px 10px 5px 8px !important; }
    .sc-sort-select [data-testid="stSelectbox"] > div > div {
      font-size: 0.78rem !important;
      min-height: 40px !important;
      padding: 6px 14px !important;
      border-radius: 20px !important;
    }
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PROXY / EXCHANGE HELPERS  — multi-proxy fallback
# ══════════════════════════════════════════════════════════════════════

# Secret keys checked in priority order (add more as needed)
PROXY_KEYS = ["PROXY_URL", "PROXY_URL_2", "PROXY_URL_3", "PROXY_URL_4"]


def _get_all_proxies() -> list[str]:
    """Return all configured proxy URLs in priority order, skipping empty slots."""
    proxies = []
    for key in PROXY_KEYS:
        try:
            val = st.secrets.get(key, "") or ""
        except Exception:
            val = os.environ.get(key, "") or ""
        if val.strip():
            proxies.append(val.strip())
    return proxies


def _get_proxy() -> str:
    """Return the first available proxy URL (backwards-compatible helper)."""
    proxies = _get_all_proxies()
    return proxies[0] if proxies else ""


def _proxy_label(proxy: str) -> str:
    """Return a safe display label (host:port only, no credentials)."""
    if not proxy:
        return "none"
    return proxy.split("@")[-1] if "@" in proxy else proxy.split("//")[-1]


def _make_exchange_with_proxy(proxy: str = "") -> ccxt_async.binanceusdm:
    """Return a configured binanceusdm exchange using the given proxy URL."""
    cfg: dict = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if proxy:
        cfg["aiohttp_proxy"] = proxy
    return ccxt_async.binanceusdm(cfg)


def _make_exchange() -> ccxt_async.binanceusdm:
    """Return exchange with first available proxy (backwards-compatible)."""
    return _make_exchange_with_proxy(_get_proxy())


async def _try_load_markets(proxies: list[str]) -> tuple:
    """
    Attempt load_markets() across proxy list until one succeeds.
    Returns (exchange, active_proxy_url, active_proxy_index).
    Raises RuntimeError if all proxies fail.
    """
    errors = []
    for i, proxy in enumerate(proxies if proxies else [""]):
        ex = _make_exchange_with_proxy(proxy)
        try:
            await ex.load_markets()
            return ex, proxy, i
        except Exception as e:
            await ex.close()
            errors.append(f"Proxy {i+1} ({_proxy_label(proxy)}): {e}")
    raise RuntimeError(
        "All proxies failed to connect:\n" + "\n".join(errors)
    )


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
    h_s = pd.Series(h); l_s = pd.Series(l)
    roll_h = h_s.rolling(zz_len, min_periods=1).max().values
    roll_l = l_s.rolling(zz_len, min_periods=1).min().values
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
    bull_qm = np.zeros(n, bool); bear_qm = np.zeros(n, bool)
    piv_h = np.full(n, np.nan); piv_l = np.full(n, np.nan)
    for i in range(2 * pp, n):
        window_h = h[i - 2 * pp : i + 1]; window_l = l[i - 2 * pp : i + 1]
        if h[i - pp] == window_h.max(): piv_h[i] = h[i - pp]
        if l[i - pp] == window_l.min(): piv_l[i] = l[i - pp]

    piv_h_bool = ~np.isnan(piv_h); piv_l_bool = ~np.isnan(piv_l)
    h_val = np.full(n, np.nan); l_val = np.full(n, np.nan)
    h_idx = np.full(n, -1, dtype=np.int64); l_idx = np.full(n, -1, dtype=np.int64)
    _hv = np.nan; _lv = np.nan; _hi = -1; _li = -1
    for i in range(n):
        if piv_h_bool[i]: _hv = float(h[i - pp]); _hi = i - pp
        if piv_l_bool[i]: _lv = float(l[i - pp]); _li = i - pp
        h_val[i] = _hv; h_idx[i] = _hi; l_val[i] = _lv; l_idx[i] = _li

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
            if len(a_type) > 0:
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


def signals_pine_only(ds_sig, ds_lower, pivot_win_ts: int, pivot_end_ts: int,
                      want_sell: bool,
                      zz_len: int = 5, s2_pp: int = 5,
                      ltf_zz_len=None, ltf_s2_pp=None):
    """
    v34: QM + Pressure gate (replaces old signals_tf Pine Final Signal logic).
    Window: pivot_win_ts (cur_P open = pivot fires) → pivot_end_ts (now).

    Pressure dot arms latch. TSL dirMain flip in wrong direction resets latch.
    Chart-TF QM OR lower-TF QM fires while latch armed → valid signal, latch consumed.
    Returns (found: bool, sig_ts_list: list[int], sig_kind_list: list[str])
    """
    h  = ds_sig.high.values;  l  = ds_sig.low.values
    c  = ds_sig.close.values; v  = ds_sig.volume.values
    ts = ds_sig.ts.values.astype(np.int64)
    n  = len(c)

    tsl_main, dir_main = f_swing(h, l, c, SWING_UTAMA)
    above_tsl = c > tsl_main; below_tsl = c < tsl_main
    wt2 = calc_wt2(h, l, c, v)

    if want_sell: raw_p = (wt2 > 80) & below_tsl
    else:         raw_p = (wt2 < 20) & above_tsl
    pressure = np.zeros(n, bool)
    pressure[1:] = raw_p[1:] & ~raw_p[:-1]

    s1_bull, s1_bear = _calc_qm_strat1(h, l, c, zz_len=zz_len)
    s2_bull, s2_bear = _calc_qm_strat2(h, l, c, pp=s2_pp)
    qm_bull_sig = s1_bull | s2_bull; qm_bear_sig = s1_bear | s2_bear
    qm_sig      = qm_bear_sig if want_sell else qm_bull_sig
    qm_sig_filtered = qm_sig & (below_tsl if want_sell else above_tsl)

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

    had_pressure  = False
    sig_ts_list:   list = []
    sig_kind_list: list = []

    for i in range(win_start, min(win_end, n - 1)):
        if i > 0 and dir_main[i] != dir_main[i - 1]:
            if want_sell     and dir_main[i] > 0:
                had_pressure = False
                sig_ts_list.clear()
                sig_kind_list.clear()
            if not want_sell and dir_main[i] < 0:
                had_pressure = False
                sig_ts_list.clear()
                sig_kind_list.clear()
        if pressure[i]:
            had_pressure = True
        if had_pressure and qm_sig_filtered[i]:
            sig_ts_list.append(int(ts[i])); sig_kind_list.append("QM")
            had_pressure = False
        if had_pressure and ltf_ts.size > 0:
            tsl_ok = below_tsl[i] if want_sell else above_tsl[i]
            if tsl_ok:
                t_lo = int(ts[i])
                t_hi = (int(ts[i + 1]) if i + 1 < n else t_lo + (int(ts[i]) - int(ts[i - 1])))
                mask = (ltf_ts >= t_lo) & (ltf_ts < t_hi) & ltf_qm[:len(ltf_ts)]
                if mask.any():
                    first_ltf = int(ltf_ts[np.where(mask)[0][0]])
                    sig_ts_list.append(first_ltf); sig_kind_list.append("MTF QM")
                    had_pressure = False

    return len(sig_ts_list) > 0, sig_ts_list, sig_kind_list




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
                    "open":   arr[:, 1],
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
    v34 fixes: pivot_ts/pivot_win_ts/pivot_confirmed_ts anchoring,
               age gate uses pivot_confirmed_ts, ADX window pp_P→cur_P_close.
    Returns (want_sell, sym, detail_str, pivot_ts, pivot_win_ts, pivot_end_ts, tdi_df) or None.
    """
    pivot_tf = cfg["pivot_tf"]
    tdi_tf   = cfg["tdi_tf"]

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

    if   cur_P < prev_P and prev_P > max(pp_P, ppp_P): want_sell = True
    elif cur_P > prev_P and prev_P < min(pp_P, ppp_P): want_sell = False
    else: return None

    # v38 FIX: age measured from pivot_confirmed_ts, using cfg["pivot_max_age_ms"]
    pivot_max_age_ms = cfg["pivot_max_age_ms"]
    if int(time.time() * 1000) - pivot_confirmed_ts > pivot_max_age_ms:
        return None

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
           f"pivot_confirmed_ts_ms={pivot_confirmed_ts} "
           f"pivot_ts_ms={pivot_ts}")
    return (want_sell, sym, det, pivot_ts, pivot_win_ts, pivot_end_ts, da)


def stage2_worker(want_sell: bool, sym: str, detail: str, pivot_ts: int,
                  pivot_win_ts: int, pivot_end_ts: int, da: pd.DataFrame):
    """
    Stage 2: TDI direction + Keltner Channel band filter.
    Returns (want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da) or None.
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
      3b — QM pressure-dot → latch → QM structure on sig_tf or lower_tf.

    v38 ⚡ All three TF fetches (mid_tf, sig_tf, choch_tf) fire concurrently.
         Bar limits computed dynamically from pivot age (not hardcoded).
         Stage 3b exit validation: TSL flip check + KC clean check.
    Returns (side_str, sym, detail, pivot_ts, "valid") or None.
    """
    mid_tf   = cfg["mid_tf"]
    sig_tf   = cfg["sig_tf"]
    choch_tf = cfg["choch_tf"]   # lower_tf for MTF QM path

    is_5m_mode = sig_tf == "5m"

    # ── Dynamic bar limits based on actual pivot age ───────────────────────────
    _WARMUP  = 60
    _API_CAP = 1500

    _tf_ms = {
        "1m":  60_000,   "3m":  180_000,  "5m":  300_000,
        "15m": 900_000,  "30m": 1_800_000, "1h": 3_600_000,
        "4h":  14_400_000, "1d": 86_400_000,
    }
    _sig_ms   = _tf_ms.get(sig_tf,   900_000)
    _mid_ms   = _tf_ms.get(mid_tf,   3_600_000)
    _choch_ms = _tf_ms.get(choch_tf, 300_000)

    _pivot_span_ms = pivot_end_ts - pivot_win_ts   # ms from pivot fire → now

    sig_limit = min(_API_CAP, int(_pivot_span_ms / _sig_ms)  + _WARMUP + 10)
    mid_limit = min(_API_CAP, int(_pivot_span_ms / _mid_ms)  + _WARMUP + 10)
    min_sig   = min(sig_limit, 80)   # validation floor scales with available data

    # Dynamic choch_tf limit — same formula, capped at configured ceiling
    _span_bars = int(_pivot_span_ms / _choch_ms) + 1
    _floor     = BOS_LR * 2 + 30
    _cap       = cfg["choch_limit"]
    ltf_limit  = max(_floor, min(_span_bars + _floor, _cap))

    # ⚡ Concurrent 3-way fetch — all three TFs in one gather
    dm, ds, dl = await asyncio.gather(
        fetch(ex, sem, sym, mid_tf,   mid_limit),
        fetch(ex, sem, sym, sig_tf,   sig_limit),
        fetch(ex, sem, sym, choch_tf, ltf_limit),
    )

    # ── Stage 3a: Cloud BS pullback gate (mid_tf) ─────────────────────────
    if dm.empty or len(dm) < max(BB_LEN, 20) + 10:
        return None

    end    = len(dm) - 1
    ts_mid = dm.ts.values[:end].astype(np.int64)

    cloud_ok, _valid_from_ts, n_cloud, _ = calc_sma_cloud_bs_signals(
        dm.high.values[:end], dm.low.values[:end],
        dm.close.values[:end], dm.open.values[:end],
        ts_mid, pivot_win_ts, pivot_end_ts, want_sell)

    if not cloud_ok:
        return None

    # ── Stage 3b: QM pressure gate ────────────────────────────────────────
    if ds.empty or len(ds) < min_sig:
        return None

    ds_lower = dl if (not dl.empty and len(dl) >= 20) else pd.DataFrame()

    found, sig_ts_list, sig_kind_list = signals_pine_only(
        ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
        ltf_zz_len=10 if is_5m_mode else None,
        ltf_s2_pp =10 if is_5m_mode else None)

    if not found:
        return None

    # ── Stage 3b exit validation ──────────────────────────────────────────
    # Drop pair if TSL dirMain has flipped, or KC band breached since oldest signal.
    ts_sig_arr  = ds.ts.values.astype(np.int64)
    last_sig_ts = sig_ts_list[-1]
    sig_bar_idx = int(np.searchsorted(ts_sig_arr, last_sig_ts, side="left"))
    sig_bar_idx = min(sig_bar_idx, len(ds) - 1)

    # (a) TSL flip check
    _tsl_s, _dir_s = f_swing(ds.high.values, ds.low.values, ds.close.values, SWING_UTAMA)
    dir_now      = int(_dir_s[-2])   # -2 = last closed bar (skip live bar)
    expected_dir = -1 if want_sell else 1
    if dir_now != expected_dir:
        return None   # TSL trend flipped — drop the pair

    # (b) KC clean check — uses tdi_tf (da), same TF as Stage 2 KC gate.
    h_t = da.high.values
    l_t_kc = da.low.values
    c_t_kc = da.close.values
    u_tdi, l_tdi = calc_kc(h_t, l_t_kc, c_t_kc)
    ts_tdi       = da.ts.values.astype(np.int64)
    kc_anchor_ts  = sig_ts_list[0]   # oldest signal in window (post-TSL-purge)
    kc_anchor_idx = int(np.searchsorted(ts_tdi, kc_anchor_ts, side="left"))
    c_range = c_t_kc[kc_anchor_idx:-1]
    u_range = u_tdi[kc_anchor_idx:-1]
    l_range = l_tdi[kc_anchor_idx:-1]
    if want_sell:
        kc_clean = bool(np.all(c_range > l_range))
    else:
        kc_clean = bool(np.all(c_range < u_range))

    if not kc_clean:
        return None   # KC band breached since oldest signal → drop

    side      = "SELL" if want_sell else "BUY"
    n_sigs    = len(sig_ts_list)
    sig_label = f"{n_sigs} sig" + ("s" if n_sigs > 1 else "")
    last_sig_price = float(ds.close.iloc[sig_bar_idx])
    last_sig_kind  = "MTF" if sig_kind_list[-1] == "MTF QM" else "QM"

    det = (f"{detail} | {mid_tf.upper()}_CloudBS\u2713({n_cloud}) {sig_tf.upper()}_QM\u2713 ({sig_label})"
           f" sig_kind={last_sig_kind}"
           f" sig_ts_ms={last_sig_ts} sig_price={last_sig_price:.8g}")
    return (side, sym, det, pivot_ts, "valid")


# ══════════════════════════════════════════════════════════════════════
#  MAIN SCAN RUNNER
# ══════════════════════════════════════════════════════════════════════

async def run_scan(cfg: dict, progress_callback: Callable) -> dict:
    """
    Run full 4-stage pipeline over all USDT perpetuals.
    Calls progress_callback(state_dict) for live UI updates (throttled by caller).
    v24: multi-proxy fallback — tries each configured proxy in order.
    """
    proxies = _get_all_proxies()

    # v24: try proxies in order until one successfully loads markets
    if "markets" not in st.session_state:
        ex, active_proxy, proxy_idx = await _try_load_markets(proxies)
        st.session_state["markets"]     = ex.markets
        st.session_state["active_proxy"]     = active_proxy
        st.session_state["active_proxy_idx"] = proxy_idx
    else:
        # Reuse cached markets; reconnect with last known good proxy
        active_proxy = st.session_state.get("active_proxy", proxies[0] if proxies else "")
        proxy_idx    = st.session_state.get("active_proxy_idx", 0)
        ex = _make_exchange_with_proxy(active_proxy)
        ex.markets = st.session_state["markets"]
        ex.markets_by_id = {m["id"]: m for m in ex.markets.values()}

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

            r2 = stage2_worker(want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da)
            if r2 is None:
                return

            want_sell, sym, detail, pivot_ts, pivot_win_ts, pivot_end_ts, da = r2
            state["s3_in"] += 1

            r3 = await stage3_worker(ex, sem, sym, want_sell, detail, pivot_ts, pivot_win_ts, pivot_end_ts, cfg, da)
            if r3:
                side, sym2, det2, pt, choch_st = r3
                entry = (sym2, det2, pt, choch_st)
                if side == "BUY":
                    state["buy_valid"].append(entry)
                else:
                    state["sell_valid"].append(entry)
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

async def debug_single(sym_raw: str, cfg: dict, tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> list:
    """
    Debug a single symbol through all pipeline stages.
    v38: delegates to shared stage workers — no duplicated logic.
         Includes pressure dot visualization (restored from CLI debug_pair).
    Returns list of (label, status, detail) tuples.
    """
    raw       = sym_raw.strip().upper().replace(" ", "")
    raw_clean = raw.replace("/", "").replace(":", "")
    base      = raw_clean.replace("USDT", "") or raw_clean
    sym       = f"{base}/USDT:USDT"
    logs      = []

    # v24: multi-proxy fallback — try each proxy until load_markets succeeds
    proxies = _get_all_proxies()
    try:
        ex, active_proxy, proxy_idx = await _try_load_markets(proxies)
    except RuntimeError as _proxy_err:
        logs.append(("Proxy", "❌ FAIL", str(_proxy_err)))
        return logs
    try:
        if sym not in ex.markets:
            logs.append(("Symbol", "❌ FAIL", f"'{sym}' not found on Binance Futures"))
            return logs
        logs.append(("Proxy", "✅ PASS",
            f"Connected via proxy slot {proxy_idx + 1} ({_proxy_label(active_proxy)})"))

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
        c_t         = float(da.close.iloc[-2])   # last confirmed closed bar
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

        # v38: dynamic bar limits based on actual pivot age
        _WARMUP  = 60
        _API_CAP = 1500
        _tf_ms_dbg = {
            "1m":  60_000,   "3m":  180_000,  "5m":  300_000,
            "15m": 900_000,  "30m": 1_800_000, "1h": 3_600_000,
            "4h":  14_400_000, "1d": 86_400_000,
        }
        _sig_ms_dbg   = _tf_ms_dbg.get(sig_tf,   900_000)
        _mid_ms_dbg   = _tf_ms_dbg.get(mid_tf,   3_600_000)
        _choch_ms_dbg = _tf_ms_dbg.get(choch_tf, 300_000)
        _pivot_span_ms = pivot_end_ts - pivot_win_ts
        sig_limit = min(_API_CAP, int(_pivot_span_ms / _sig_ms_dbg)  + _WARMUP + 10)
        mid_limit = min(_API_CAP, int(_pivot_span_ms / _mid_ms_dbg)  + _WARMUP + 10)
        min_sig   = min(sig_limit, 80)
        _span_bars_dbg = int(_pivot_span_ms / _choch_ms_dbg) + 1
        _floor_dbg     = BOS_LR * 2 + 30
        ltf_limit = max(_floor_dbg, min(_span_bars_dbg + _floor_dbg, cfg["choch_limit"]))

        logs.append(("S3 Bar limits", "ℹ️ INFO",
            f"Bar limits (dynamic from pivot age {_pivot_span_ms/3_600_000:.1f}h):  "
            f"mid={mid_limit}  sig={sig_limit}  ltf={ltf_limit}"))

        # v38: age gate uses cfg["pivot_max_age_ms"]
        pivot_max_age_ms = cfg["pivot_max_age_ms"]
        if int(time.time() * 1000) - pivot_confirmed_ts > pivot_max_age_ms:
            max_h = pivot_max_age_ms / 3_600_000
            age_h = (int(time.time() * 1000) - pivot_confirmed_ts) / 3_600_000
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

        cloud_found, valid_from_ts, n_cloud_sigs, cloud_details = calc_sma_cloud_bs_debug(
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
        if not cloud_found:
            return logs

        # ── Stage 3b: QM pressure gate (sig_tf + lower_tf) ───────────
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

        # ── Pressure dots (v38: restored from CLI debug_pair) ────────
        h_p  = ds.high.values;  l_p  = ds.low.values
        c_p  = ds.close.values; v_p  = ds.volume.values
        ts_p = ds.ts.values.astype(np.int64)
        tsl_main_p, dir_main_p = f_swing(h_p, l_p, c_p, SWING_UTAMA)
        above_tsl_p = c_p > tsl_main_p; below_tsl_p = c_p < tsl_main_p
        wt2_p = calc_wt2(h_p, l_p, c_p, v_p)
        raw_p_dots = (wt2_p > 80) & below_tsl_p if want_sell else (wt2_p < 20) & above_tsl_p
        pressure_p = np.zeros(len(c_p), bool)
        pressure_p[1:] = raw_p_dots[1:] & ~raw_p_dots[:-1]
        win_start_p = int(np.searchsorted(ts_p, pivot_win_ts))
        win_end_p   = int(np.searchsorted(ts_p, pivot_end_ts))
        p_bars = np.where(pressure_p[win_start_p:win_end_p])[0] + win_start_p
        cond_lbl = "wt2>80 & below TSL" if want_sell else "wt2<20 & above TSL"
        if len(p_bars) > 0:
            dot_summary = f"{len(p_bars)} pressure dot(s) in pivot window ({cond_lbl})"
            for bi in p_bars[-5:]:
                logs.append(("  → Pressure dot", "ℹ️ INFO",
                    f"bar[{bi}]  wt2={float(wt2_p[bi]):.1f}  ts={ts_p[bi]}  ({_age(ts_p[bi])})"))
            if len(p_bars) > 5:
                logs.append(("  → Pressure dot", "ℹ️ INFO",
                    f"... ({len(p_bars)-5} earlier dot(s) not shown)"))
        else:
            dot_summary = f"No pressure dots in pivot window on {sig_tf} ({cond_lbl})"
        logs.append(("S3b Pressure dots", "ℹ️ INFO", dot_summary))

        _is_5m = sig_tf == "5m"
        found, sig_ts_list, sig_kind_list = signals_pine_only(
            ds, ds_lower, pivot_win_ts, pivot_end_ts, want_sell,
            ltf_zz_len=10 if _is_5m else None,
            ltf_s2_pp =10 if _is_5m else None)

        n_sigs = len(sig_ts_list)
        if found:
            n_qm  = sig_kind_list.count("QM")
            n_mtf = sig_kind_list.count("MTF QM")
            kind_sum = (f"QM×{n_qm}" if n_qm else "") + (" MTF QM×" + str(n_mtf) if n_mtf else "")
            sig_detail = f"{n_sigs} signal(s) [{kind_sum.strip()}] | latest: {_age(sig_ts_list[-1])}"
        else:
            sig_detail = f"No QM signals in pivot window on {sig_tf}/{choch_tf}"

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

        # ── Stage 3b exit validation ──────────────────────────────────
        # (a) TSL flip check
        _tsl_dbg, _dir_dbg = f_swing(ds.high.values, ds.low.values, ds.close.values, SWING_UTAMA)
        dir_now_dbg    = int(_dir_dbg[-2])   # last closed bar
        expected_dir_dbg = -1 if want_sell else 1
        tsl_flipped_dbg = (dir_now_dbg != expected_dir_dbg)
        logs.append(("S3b TSL flip check", "❌ FAIL" if tsl_flipped_dbg else "✅ PASS",
            f"TSL dirMain now (last closed bar): {dir_now_dbg}  |  "
            f"expected={expected_dir_dbg} for {direction}  |  "
            f"{'FLIPPED — pair would be dropped' if tsl_flipped_dbg else 'intact'}"))
        if tsl_flipped_dbg:
            return logs

        # (b) KC clean check on tdi_tf (da)
        h_t_dbg = da.high.values
        l_t_dbg = da.low.values
        c_t_dbg = da.close.values
        u_tdi_dbg, l_tdi_dbg = calc_kc(h_t_dbg, l_t_dbg, c_t_dbg)
        ts_tdi_dbg     = da.ts.values.astype(np.int64)
        kc_anchor_ts_dbg  = sig_ts_list[0]
        kc_anchor_idx_dbg = int(np.searchsorted(ts_tdi_dbg, kc_anchor_ts_dbg, side="left"))
        c_range_dbg = c_t_dbg[kc_anchor_idx_dbg:-1]
        u_range_dbg = u_tdi_dbg[kc_anchor_idx_dbg:-1]
        l_range_dbg = l_tdi_dbg[kc_anchor_idx_dbg:-1]
        if want_sell:
            kc_clean_dbg = bool(np.all(c_range_dbg > l_range_dbg))
            n_breach_dbg = int(np.sum(c_range_dbg <= l_range_dbg))
        else:
            kc_clean_dbg = bool(np.all(c_range_dbg < u_range_dbg))
            n_breach_dbg = int(np.sum(c_range_dbg >= u_range_dbg))
        logs.append(("S3b KC clean check", "✅ PASS" if kc_clean_dbg else "❌ FAIL",
            f"KC band {'clean' if kc_clean_dbg else f'BREACHED ({n_breach_dbg} bar(s))'} "
            f"from oldest signal ({_age(kc_anchor_ts_dbg)}) → scan time"))
        if not kc_clean_dbg:
            return logs

        logs.append(("Signal Confirmed", "✅ VALID",
            f"{direction} | {n_sigs} QM signal(s) on {sig_tf}/{choch_tf} | "
            f"last: {sig_times} | price={last_sig_price:.8g}"))

        return logs
    finally:
        await ex.close()


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
               choch_status: str, now_ms: int, mode_key: str, timestamp: str,
               tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> dict:
    """
    v18: Parse a result row into structured fields.
    choch_status: "valid" or "wait"
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
    adx    = _re.search(r"ADX_cur=([\d.]+)",   det) or _re.search(r"ADX_peak=([\d.]+)", det)
    bb_m   = _re.search(r"(\w+)_CloudBS",               det)
    sig_m  = _re.search(r"\[(\w+)_QM",                    det)
    sig_ts = _re.search(r"sig_ts_ms=(\d+)",               det)
    sig_px = _re.search(r"sig_price=([\d.eE+\-]+)",       det)
    n_sigs = _re.search(r"\((\d+) sig",                   det)
    kind_m = _re.search(r"sig_kind=(\w+)",                det)
    piv_m  = _re.search(r"pivot_confirmed_ts_ms=(\d+)",   det)

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
        sig_time = _fmt_ts(int(sig_ts.group(1)), tz_h, tz_label, time_fmt)
    else:
        age_h = 0.0; age_str = "—"; sig_time = "—"

    # Pivot age from pivot_confirmed_ts_ms
    if piv_m:
        piv_age_ms = int(time.time() * 1000) - int(piv_m.group(1))
        piv_age_h  = piv_age_ms / 3_600_000
        if piv_age_h < 1:    piv_str = f"{piv_age_h*60:.0f}m"
        elif piv_age_h < 24: piv_str = f"{piv_age_h:.1f}h"
        else:                 piv_str = f"{piv_age_h/24:.1f}d"
    else:
        piv_str = "—"

    # ADX value — v20 FIX: simplified redundant return expression
    adx_v = f"{float(adx.group(1)):.0f}" if adx else "—"

    return {
        "price":  price_str,
        "adx":    adx_v,
        "bb_tf":  bb_m.group(1).upper()  if bb_m  else "—",
        "sig_tf": sig_m.group(1).upper() if sig_m else "—",
        "age_str": age_str,
        "age_h":   str(age_h),
        "sig_time": sig_time,
        "n_sigs": n_sigs.group(1) if n_sigs else "1",
        "kind":   kind_m.group(1) if kind_m else "—",
        "piv_age": piv_str,
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
        "tz_key":       _tz_default,
        "time_fmt":     _tf_default,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sc_counters_html(nbv: int, nsv: int,
                      s2: int, s3: int, elapsed: float, total: int, done: int) -> str:
    spd = done / max(elapsed, 0.01)
    return f"""
<div class="sc-counters">
  <div class="sc-cnt g">
    <div class="cnt-lbl">BUY ✅ VALID</div>
    <div class="cnt-val">{nbv}</div>
  </div>
  <div class="sc-cnt r">
    <div class="cnt-lbl">SELL ✅ VALID</div>
    <div class="cnt-val">{nsv}</div>
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
                     sv: int, mode_key: str) -> str:
    all_s = bv + sv
    spd   = total / max(elapsed, 0.01)
    return (
        f'<div class="sc-summary">'
        f'<span class="ss-title">&#9989; Scan <span>Complete</span></span>'
        f'<span class="ss-chip g">&#9650; BUY {bv}</span>'
        f'<span class="ss-chip r">&#9660; SELL {sv}</span>'
        f'<span class="ss-meta">'
        f'<b>{all_s}</b> signals &middot; {total} sym &middot; '
        f'{elapsed:.1f}s &middot; {spd:.0f}/s &middot; <b>{mode_key.upper()}</b>'
        f'</span>'
        f'</div>'
    )


def _signal_cards_html(entries: list, is_buy: bool, is_valid: bool, mode_key: str = "15m",
                       grid_cls: str = "sc-grid",
                       tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> str:
    """Compact cards: symbol | price | TF | signal time | direction."""
    if not entries:
        label = ("BUY" if is_buy else "SELL") + (" confirmed" if is_valid else " waiting")
        return f'<div class="sc-empty"><div class="ico">&#128269;</div><p>No {label} signals.</p></div>'

    card_cls = ("buy" if is_buy else "sell") + ("" if is_valid else " wait")
    tf_label  = mode_key.upper()   # "15M" or "5M"

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
        p     = _parse_det_card(det, tz_h, tz_label, time_fmt)
        # Extract bare base name: "BERA/USDT:USDT" → "BERA"
        base  = sym.split("/")[0].replace("USDT", "").replace("BUSD", "").replace("USD", "")
        if not base:  # fallback if already bare
            base = sym.split("/")[0]
        cards.append(
            f'<div class="sc-card {card_cls}">'
            f'<div class="sc-card-row1">'
            f'<span class="sc-card-sym">{base}{pulse}</span>'
            f'<span class="sc-card-dir {dir_cls}">{dir_txt}</span>'
            f'</div>'
            f'<div class="sc-card-price">{p["price"]}</div>'
            f'<div class="sc-card-info"><b>{tf_label}</b> &nbsp;{p["sig_time"]}</div>'
            f'</div>'
        )
    return f'<div class="{grid_cls}">{"".join(cards)}</div>'


def _all_signals_two_col_html(bv_list, sv_list, mode_key: str,
                              tz_h: float = 0.0, tz_label: str = TZ_DEFAULT, time_fmt: str = "24h") -> str:
    """Render All tab — BUY and SELL confirmed signals."""
    confirmed_parts = []
    if bv_list:
        confirmed_parts.append(_signal_cards_html(bv_list, True,  True,  mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    if sv_list:
        confirmed_parts.append(_signal_cards_html(sv_list, False, True,  mode_key, "sc-grid", tz_h, tz_label, time_fmt))
    conf_count = len(bv_list) + len(sv_list)
    conf_section = (
        f'<div class="sc-all-section">'
        f'<div class="sc-col-header confirmed">&#9989; Confirmed &nbsp;'
        f'<span style="opacity:0.7;font-weight:600">{conf_count}</span></div>'
        + ("".join(confirmed_parts) if confirmed_parts else
           '<div class="sc-empty" style="padding:0.8rem 1rem">'
           '<div class="ico" style="font-size:1.3rem">&#128269;</div>'
           '<p style="font-size:0.8rem">No confirmed signals</p></div>')
        + '</div>'
    )
    return f'<div>{conf_section}</div>'


def main():
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
      Ultra-Fast
      <span class="dot">&bull;</span>
      Multi-Stage Pipeline
      <span class="dot">&bull;</span>
      Pine Accurate
    </div>
  </div>
  <div class="sc-header-right">
    <span class="sc-badge blue">&#128640; v38</span>
    <span class="sc-badge green">&#10004; 3 Stages</span>
    <span class="sc-tz-badge">&#127758; {tz_short}</span>
    <span class="sc-tz-badge" style="background:rgba(0,180,216,0.07);color:var(--blue);border-color:rgba(0,180,216,0.28);">&#128336; {time_fmt.upper()}</span>
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
        st.markdown("</div>", unsafe_allow_html=True)

    tab_scan, tab_debug = st.tabs(["&#128269;  Full Scan", "&#128027;  Debug Symbol"])

    # ══ TAB 1: FULL SCAN ══════════════════════════════════════════════
    with tab_scan:

        # ── Proxy status banner ───────────────────────────────────────
        _all_proxies  = _get_all_proxies()
        _active_proxy = st.session_state.get("active_proxy", "")
        _active_idx   = st.session_state.get("active_proxy_idx", -1)

        if _all_proxies:
            # Build slot chips: green = active, grey = standby, red = none
            _slot_chips = []
            for _i, _p in enumerate(_all_proxies):
                _lbl   = _proxy_label(_p)
                _is_active = (_i == _active_idx) or (_active_idx == -1 and _i == 0)
                _chip_style = (
                    "background:rgba(0,230,118,0.12);color:#00e676;"
                    "border:1px solid rgba(0,230,118,0.35);"
                ) if _is_active else (
                    "background:rgba(255,255,255,0.04);color:#6a6a88;"
                    "border:1px solid rgba(255,255,255,0.08);"
                )
                _dot = "🟢" if _is_active else "⚪"
                _slot_chips.append(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;'
                    f'padding:3px 10px;border-radius:20px;font-family:var(--mono);'
                    f'font-size:0.72rem;{_chip_style}">' 
                    f'{_dot} Slot {_i+1}: {_lbl}'
                    f'{"&nbsp;<b style='color:#00e676'>ACTIVE</b>" if _is_active else " STANDBY"}'
                    f'</span>'
                )
            _chips_html = "&nbsp;".join(_slot_chips)
            st.markdown(
                f'<div class="sc-proxy-ok" style="display:flex;align-items:center;'
                f'flex-wrap:wrap;gap:6px;">'
                f'&#128274;&nbsp;<b>{len(_all_proxies)} proxy slot(s) configured</b>'
                f'&nbsp;&mdash;&nbsp;{_chips_html}'
                f'&nbsp;&middot;&nbsp;<span style="color:#5a8a5a;font-size:0.78rem">'
                f'auto-fallback enabled</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sc-proxy-err">&#128683; No proxy &mdash; Binance blocks Streamlit Cloud IPs. '
                'Add <code>PROXY_URL</code> to Streamlit Secrets.</div>',
                unsafe_allow_html=True)
            with st.expander("How to set up a free proxy (3 min)"):
                st.markdown("""
1. Register at **https://proxy2.webshare.io** (free, no credit card)
2. Go to **Proxy List** → copy as `Username:Password@host:port`
3. In Streamlit → your app → **⋮ → Settings → Secrets**, add:
```toml
PROXY_URL   = "http://user1:pass1@p.webshare.io:80"
PROXY_URL_2 = "http://user2:pass2@p.webshare.io:80"
```
4. Save — app restarts in ~30s. Add up to 4 slots for auto-fallback.
""")

        # ── Mode + Timeframes row ─────────────────────────────────────
        # Persist mode selection across reruns
        if "scan_mode_sel" not in st.session_state:
            st.session_state["scan_mode_sel"] = "15m"
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
            f"Pressure dot <span class='pill-arr-v2'>&#8594;</span> {cfg['sig_tf'].upper()}/{cfg['choch_tf'].upper()} QM</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Action buttons ────────────────────────────────────────────
        btn_c1, btn_c2 = st.columns([4, 1])
        with btn_c1:
            scan_clicked = st.button("&#128640;  Start Scan", type="primary", key="scan_btn",
                                     width="stretch")
        with btn_c2:
            if st.button("&#128260;", key="clear_mkts", width="stretch",
                         help="Refresh market list — clears cache and reloads from Binance"):
                st.session_state.pop("markets", None)
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
                all_s   = (len(state["buy_valid"]) +
                           len(state["sell_valid"]))
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

                prog_bar.progress(1.0, text=f"Done — {total} symbols in {elapsed:.1f}s")
                ctr_ph.empty()

                now_ms    = int(time.time() * 1000)
                ts_int    = int(time.time())
                timestamp = _fmt_ts(now_ms, tz_h, tz_key, time_fmt)

                all_results = (
                    [("BUY",  s, d, p, c) for s, d, p, c in buy_valid] +
                    [("SELL", s, d, p, c) for s, d, p, c in sell_valid]
                )

                if all_results:
                    all_rows = [
                        _parse_row(dir_, s, d, p, choch_st, now_ms, mode_key, timestamp, tz_h, tz_key, time_fmt)
                        for dir_, s, d, p, choch_st in all_results
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
                        ("BUY",  "BUY CONFIRMED",  buy_valid),
                        ("SELL", "SELL CONFIRMED", sell_valid),
                    ]:
                        if not group: continue
                        txt_buf.write(f"\n{'─'*28} {group_label} {'─'*28}\n")
                        for sym, det, pts, choch_st in group:
                            r = _parse_row(dir_, sym, det, pts, choch_st, now_ms, mode_key, timestamp, tz_h, tz_key, time_fmt)
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
                        "buy_valid":    [(s, d) for s, d, _, _ in buy_valid],
                        "sell_valid":   [(s, d) for s, d, _, _ in sell_valid],
                        "buy_valid_full":  [(s, d, p, c) for s, d, p, c in buy_valid],
                        "sell_valid_full": [(s, d, p, c) for s, d, p, c in sell_valid],
                    })
                else:
                    st.session_state.update({
                        "scan_done":    True,
                        "scan_state":   state,
                        "scan_elapsed": elapsed,
                        "scan_mode":    mode_key,
                        "df_final":     None,
                        "buy_valid": [], "sell_valid": [],
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
            r_tz_key   = st.session_state.get("tz_key", TZ_DEFAULT)
            r_tz_h     = TIMEZONES.get(r_tz_key, 0.0)
            r_time_fmt = st.session_state.get("time_fmt", TIME_FMT_DEFAULT)

            bv_list_raw = st.session_state["buy_valid"]
            sv_list_raw = st.session_state["sell_valid"]
            bv, sv = len(bv_list_raw), len(sv_list_raw)
            all_sigs = bv + sv

            # Persistent summary banner
            st.markdown(
                _sc_summary_html(total, elapsed, bv, sv, mode_key_r),
                unsafe_allow_html=True)

            if all_sigs == 0:
                st.markdown(
                    '<div class="sc-empty"><div class="ico">&#128301;</div>'
                    '<p>No signals &mdash; market conditions did not meet all 3 stage filters.</p></div>',
                    unsafe_allow_html=True)
            else:
                # ── Sort / Filter control ─────────────────────────────
                cur_sort = st.session_state.get("results_sort", "newest")
                _sort_options = {
                    "🕐  Newest first": "newest",
                    "🕛  Oldest first": "oldest",
                    "🔤  Name A → Z":   "name_az",
                    "🔡  Name Z → A":   "name_za",
                }
                _sort_labels  = list(_sort_options.keys())
                _sort_cur_lbl = next(
                    (k for k, v in _sort_options.items() if v == cur_sort),
                    _sort_labels[0]
                )
                _is_non_default = cur_sort != "newest"
                pill_cls = "sc-sort-icon-pill active" if _is_non_default else "sc-sort-icon-pill"

                _scol_icon, _scol_sel = st.columns([1, 3])
                with _scol_icon:
                    st.markdown(
                        f"<div class='{pill_cls}' style='margin-top:6px'>"
                        f"&#9651; Sort</div>",
                        unsafe_allow_html=True)
                with _scol_sel:
                    st.markdown("<div class='sc-sort-select'>", unsafe_allow_html=True)
                    _new_sort_lbl = st.selectbox(
                        "sort_select_label",
                        _sort_labels,
                        index=_sort_labels.index(_sort_cur_lbl),
                        key="sort_selectbox",
                        label_visibility="collapsed",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    _new_sort_key = _sort_options[_new_sort_lbl]
                    if _new_sort_key != cur_sort:
                        st.session_state["results_sort"] = _new_sort_key
                        st.rerun()

                st.markdown("<div style='margin-bottom:0.3rem'></div>",
                            unsafe_allow_html=True)

                # Apply sort
                cur_sort = st.session_state.get("results_sort", "newest")
                bv_list = _sort_signals(bv_list_raw, cur_sort)
                sv_list = _sort_signals(sv_list_raw, cur_sort)

                # Signal card tabs
                tab_labels = [
                    f"All ({all_sigs})",
                    f"BUY  {bv}",
                    f"SELL  {sv}",
                ]
                t_all, t_bv, t_sv = st.tabs(tab_labels)

                with t_all:
                    if bv_list or sv_list:
                        st.markdown(
                            _all_signals_two_col_html(bv_list, sv_list, mode_key_r, r_tz_h, r_tz_key, r_time_fmt),
                            unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sc-empty"><div class="ico">&#128269;</div><p>No signals.</p></div>',
                                    unsafe_allow_html=True)

                with t_bv:
                    st.markdown(_signal_cards_html(bv_list, True, True, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)
                with t_sv:
                    st.markdown(_signal_cards_html(sv_list, False, True, mode_key_r, "sc-grid", r_tz_h, r_tz_key, r_time_fmt), unsafe_allow_html=True)

                # Full table + export — rebuilt dynamically with current sort ──
                _bv_full = st.session_state.get("buy_valid_full",  [])
                _sv_full = st.session_state.get("sell_valid_full", [])

                # Sort the full tuples the same way as the display lists
                def _sort_full(lst, sk):
                    """Sort (sym,det,pts,choch) tuples using same keys as _sort_signals."""
                    import re as _re2
                    def _ts(item):
                        m = _re2.search(r"sig_ts_ms=(\d+)", item[1])
                        return int(m.group(1)) if m else 0
                    if sk == "oldest":   return sorted(lst, key=_ts)
                    if sk == "name_az":  return sorted(lst, key=lambda x: x[0])
                    if sk == "name_za":  return sorted(lst, key=lambda x: x[0], reverse=True)
                    return sorted(lst, key=_ts, reverse=True)  # newest

                _bv_s = _sort_full(_bv_full, cur_sort)
                _sv_s = _sort_full(_sv_full, cur_sort)

                _exp_now_ms    = st.session_state.get("scan_now_ms",    int(time.time()*1000))
                _exp_timestamp = st.session_state.get("scan_timestamp", "")
                _exp_ts_int    = st.session_state.get("scan_ts_int",    int(time.time()))
                _sort_labels   = {"newest": "Newest first", "oldest": "Oldest first",
                                  "name_az": "A→Z", "name_za": "Z→A"}
                _sort_lbl      = _sort_labels.get(cur_sort, cur_sort)

                # Build sorted export rows
                _all_sorted = (
                    [("BUY",  s, d, p, c) for s, d, p, c in _bv_s] +
                    [("SELL", s, d, p, c) for s, d, p, c in _sv_s]
                )
                if _all_sorted:
                    _exp_rows = [
                        _parse_row(dir_, s, d, p, ch, _exp_now_ms, mode_key_r,
                                   _exp_timestamp, r_tz_h, r_tz_key, r_time_fmt)
                        for dir_, s, d, p, ch in _all_sorted
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
                    _tbuf.write("=" * 72 + "\n")
                    for _dir, _glbl, _grp in [
                        ("BUY",  "BUY CONFIRMED",  _bv_s),
                        ("SELL", "SELL CONFIRMED", _sv_s),
                    ]:
                        if not _grp: continue
                        _tbuf.write(f"\n{'─'*28} {_glbl} {'─'*28}\n")
                        for _sym, _det, _pts, _cst in _grp:
                            _r = _parse_row(_dir, _sym, _det, _pts, _cst,
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
                "<span class='sc-stage-dot dot-3'>S3</span> SMA Cloud BS pullback gate<br>"
                "<span class='sc-stage-dot dot-3'>S3</span> QM pressure dot → latch → signal"
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
