"""
Binance Futures API Host Tester
Tests which fapi hosts are reachable from your server (with optional proxy).
"""

import asyncio
import time
import streamlit as st
import aiohttp

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Binance API Host Tester",
    page_icon="📡",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark terminal background */
.stApp {
    background: #0d0f14;
    color: #c9d1d9;
}

h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #f0f6fc !important;
}

/* Result cards */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    transition: border-color 0.2s;
}

.result-card.ok    { border-left: 4px solid #3fb950; }
.result-card.warn  { border-left: 4px solid #d29922; }
.result-card.error { border-left: 4px solid #f85149; }

.badge-ok    { background:#1a3a20; color:#3fb950; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-warn  { background:#3a2f10; color:#d29922; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }
.badge-error { background:#3a1a1a; color:#f85149; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:700; }

.host-text  { flex:1; color:#8b949e; }
.host-text b { color:#f0f6fc; }
.ms-text    { color:#8b949e; min-width:60px; text-align:right; }
.ms-fast    { color:#3fb950; }
.ms-mid     { color:#d29922; }
.ms-slow    { color:#f85149; }

.tip-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #388bfd;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 10px;
}
.tip-box b { color: #79c0ff; }

.stButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 10px 28px;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043; }

.stTextInput > div > div > input {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
}

.stNumberInput > div > div > input {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
FAPI_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

ENDPOINTS = {
    "/fapi/v1/ping":           "Ping (lightest)",
    "/fapi/v1/time":           "Server time",
    "/fapi/v1/exchangeInfo":   "Exchange info (heavy)",
}


# ── Async test logic ──────────────────────────────────────────────────────────
async def test_host(session: aiohttp.ClientSession, host: str, path: str, proxy: str, timeout: int):
    url = host + path
    start = time.monotonic()
    try:
        kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout)}
        if proxy:
            kwargs["proxy"] = proxy
        async with session.get(url, **kwargs) as resp:
            await resp.read()
            ms = int((time.monotonic() - start) * 1000)
            return {"host": host, "status": resp.status, "ms": ms, "error": None}
    except asyncio.TimeoutError:
        ms = int((time.monotonic() - start) * 1000)
        return {"host": host, "status": None, "ms": ms, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        ms = int((time.monotonic() - start) * 1000)
        return {"host": host, "status": None, "ms": ms, "error": str(e)[:60]}


async def run_tests(hosts, path, proxy, timeout):
    connector = aiohttp.TCPConnector(ssl=False, limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [test_host(session, h, path, proxy, timeout) for h in hosts]
        return await asyncio.gather(*tasks)


# ── Helper: render one result card ───────────────────────────────────────────
def render_card(r, best_host):
    host   = r["host"]
    status = r["status"]
    ms     = r["ms"]
    err    = r["error"]

    if status == 200:
        cls   = "ok"
        badge = f'<span class="badge-ok">✓ {status} OK</span>'
        star  = " ⭐" if host == best_host else ""
        label = f"<b>{host}</b>{star}"
    elif status:
        cls   = "warn"
        badge = f'<span class="badge-warn">⚠ {status} BLOCKED</span>'
        label = f"<b>{host}</b>"
    else:
        cls   = "error"
        badge = f'<span class="badge-error">✗ FAILED</span>'
        label = f"<b>{host}</b>"

    # colour the ms value
    if ms < 300:
        ms_cls = "ms-fast"
    elif ms < 1000:
        ms_cls = "ms-mid"
    else:
        ms_cls = "ms-slow"

    note = f'<span class="{ms_cls}">{ms} ms</span>'
    if err:
        note += f'  <span style="color:#6e7681;font-size:0.78rem;">— {err}</span>'

    st.markdown(f"""
    <div class="result-card {cls}">
        {badge}
        <span class="host-text">{label}</span>
        <span class="ms-text">{note}</span>
    </div>""", unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Binance API Host Tester")
st.markdown("<p style='color:#8b949e;font-size:0.92rem;'>Tests which <code>fapi</code> endpoints are reachable from your server — useful when hosting on US-based cloud providers that are geo-blocked by Binance.</p>", unsafe_allow_html=True)

st.markdown("---")

# Settings
col1, col2 = st.columns([2, 1])
with col1:
    endpoint_label = st.selectbox(
        "Test endpoint",
        list(ENDPOINTS.keys()),
        format_func=lambda x: f"{x}  →  {ENDPOINTS[x]}",
    )
with col2:
    timeout = st.number_input("Timeout (sec)", min_value=2, max_value=30, value=6, step=1)

proxy_input = st.text_input(
    "Proxy URL (optional — leave blank to test direct)",
    placeholder="http://user:pass@proxy-host:port",
)

# Custom hosts
with st.expander("➕ Add custom hosts to test"):
    custom_raw = st.text_area(
        "One URL per line",
        placeholder="https://fapi5.binance.com\nhttps://my-custom-proxy.example.com",
        height=100,
    )

st.markdown("---")

# Run button
if st.button("🚀 Run Test Now"):
    custom_hosts = [h.strip() for h in custom_raw.splitlines() if h.strip()] if custom_raw else []
    all_hosts    = FAPI_HOSTS + custom_hosts
    proxy        = proxy_input.strip()

    mode_label = f"via proxy `{proxy.split('@')[-1]}`" if proxy else "direct (no proxy)"
    st.markdown(f"<p style='color:#8b949e;font-size:0.85rem;'>Testing <b>{len(all_hosts)}</b> hosts — {mode_label} — endpoint: <code>{endpoint_label}</code></p>", unsafe_allow_html=True)

    with st.spinner("Pinging all hosts in parallel…"):
        results = asyncio.run(run_tests(all_hosts, endpoint_label, proxy, timeout))

    # Sort: 200 first by ms, then blocked, then errors
    def sort_key(r):
        if r["status"] == 200:
            return (0, r["ms"])
        elif r["status"]:
            return (1, r["ms"])
        else:
            return (2, r["ms"])

    results_sorted = sorted(results, key=sort_key)

    ok_results = [r for r in results_sorted if r["status"] == 200]
    best_host  = ok_results[0]["host"] if ok_results else None

    st.markdown("### Results")
    for r in results_sorted:
        render_card(r, best_host)

    st.markdown("---")

    # Summary
    total   = len(results)
    ok      = len(ok_results)
    blocked = len([r for r in results if r["status"] and r["status"] != 200])
    failed  = len([r for r in results if not r["status"]])

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Reachable", ok)
    c2.metric("⚠️ Blocked",  blocked)
    c3.metric("❌ Failed",   failed)

    # Recommendation
    if best_host:
        klines_url = best_host + "/fapi/v1/klines"
        st.markdown(f"""
        <div class="tip-box">
            <b>✅ Recommended — update your scanner:</b><br><br>
            Line 775 in <code>binance_futures_scanner_v57_streamlit.py</code>:<br><br>
            <span style="color:#79c0ff;">_FAPI_URL = "<b>{klines_url}</b>"</span>
        </div>
        """, unsafe_allow_html=True)
    elif not proxy:
        st.markdown("""
        <div class="tip-box">
            <b>❌ All hosts blocked — direct access not possible from this server.</b><br><br>
            Enter a proxy URL above (residential or datacenter, non-US) and run the test again.<br>
            Format: <span style="color:#79c0ff;">http://user:pass@proxy-host:port</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tip-box">
            <b>❌ All hosts blocked even via proxy.</b><br><br>
            Try a different proxy provider or a non-US exit node.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="tip-box">
        <b>How to use:</b><br>
        1. Choose an endpoint (ping is fastest and lightest)<br>
        2. Optionally enter a proxy URL to test through it<br>
        3. Click <b>Run Test Now</b> — all hosts are tested in parallel<br>
        4. Copy the recommended <code>_FAPI_URL</code> into your scanner
    </div>
    """, unsafe_allow_html=True)
