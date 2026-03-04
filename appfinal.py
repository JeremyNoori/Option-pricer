import streamlit as st

# ─────────────────────────────────────────────
#  PAGE CONFIG — MUST be the first Streamlit command
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OTC Options Pricer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, gaussian_kde, skewnorm, genextreme
from scipy.optimize import brentq, minimize
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time as _time_mod
import json
from datetime import datetime, timedelta
import warnings
import math
warnings.filterwarnings('ignore')

# Supabase backend (optional — loaded lazily to avoid SessionInfo errors)
_SUPABASE_IMPORTS_OK = False

def _is_supabase_available():
    """Lazy check — imports supabase_config on first call."""
    global _SUPABASE_IMPORTS_OK
    if _SUPABASE_IMPORTS_OK is None:
        return False
    try:
        from supabase_config import get_supabase_client
        _SUPABASE_IMPORTS_OK = True
        return get_supabase_client() is not None
    except Exception:
        _SUPABASE_IMPORTS_OK = None  # don't retry
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API KEYS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COINGECKO_API_KEY = "CG-V6cHxSdrovx4eoExmoUAEcsw"
CMC_API_KEY       = "77e9e49f6c124981973dff95ec600d7e"

# CoinGecko Demo-tier header (sent with every request)
_CG_HEADERS = {
    "accept": "application/json",
    "x-cg-demo-api-key": COINGECKO_API_KEY,
}

# CoinMarketCap header
_CMC_HEADERS = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
    "Accept": "application/json",
}

# CoinGecko ID → CMC slug mapping (for fallback)
_CG_TO_CMC_SLUG = {
    "bitcoin":      "BTC",
    "ethereum":     "ETH",
    "xdc-network":  "XDC",
    "hyperliquid":  "HYPE",
    "solana":       "SOL",
}


def _cg_get(url, timeout=8):
    """CoinGecko GET with API key header. Raises on non-200."""
    r = requests.get(url, headers=_CG_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def _cmc_price(symbol):
    """Fallback: fetch latest USD price from CoinMarketCap."""
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        r = requests.get(url, headers=_CMC_HEADERS,
                         params={"symbol": symbol.upper(), "convert": "USD"},
                         timeout=8)
        data = r.json()
        return data["data"][symbol.upper()]["quote"]["USD"]["price"]
    except Exception:
        return None


def _cmc_history(symbol, days=90):
    """Fallback: fetch daily OHLCV history from CoinMarketCap."""
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
        r = requests.get(url, headers=_CMC_HEADERS,
                         params={
                             "symbol": symbol.upper(),
                             "convert": "USD",
                             "count": str(days),
                             "interval": "daily",
                         }, timeout=12)
        data = r.json()
        quotes = data["data"]["quotes"]
        return [q["quote"]["USD"]["close"] for q in quotes]
    except Exception:
        return None


def _fetch_coin_price(coin_id, ticker):
    """Fetch live price: CoinGecko first, CMC fallback."""
    # CoinGecko
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        r = _cg_get(url, timeout=5)
        price = r.json().get(coin_id, {}).get("usd", None)
        if price is not None:
            return price
    except Exception:
        pass
    # CMC fallback
    cmc_sym = _CG_TO_CMC_SLUG.get(coin_id, ticker)
    return _cmc_price(cmc_sym)


def _fetch_coin_history(coin_id, ticker, days=90):
    """Fetch price history: CoinGecko first, CMC fallback."""
    # CoinGecko
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
        r = _cg_get(url)
        prices = [p[1] for p in r.json().get("prices", [])]
        if len(prices) > 5:
            return prices
    except Exception:
        pass
    # CMC fallback
    cmc_sym = _CG_TO_CMC_SLUG.get(coin_id, ticker)
    return _cmc_history(cmc_sym, days)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CACHING LAYER
#  TTL-based cache for all external API calls.
#  Avoids redundant network requests across Streamlit reruns.
#  Cache levels:
#    1. @st.cache_data(ttl=...)  — cross-rerun, auto-expires
#    2. session_state            — per-session persistence
#    3. Supabase                 — cross-session (optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Cache TTL constants (seconds)
CACHE_TTL_SPOT        = 60       # Live price: 1 minute
CACHE_TTL_HISTORY     = 300      # Price history: 5 minutes
CACHE_TTL_MARKET_DATA = 300      # Market cap, volume: 5 minutes
CACHE_TTL_FEAR_GREED  = 600      # Fear & Greed: 10 minutes
CACHE_TTL_DERIBIT     = 120      # Vol surface: 2 minutes
CACHE_TTL_COMPUTATION = 600      # Heavy computations: 10 minutes


def _cache_get(key):
    """Read from session_state cache with TTL check."""
    entry = st.session_state.get(f"_cache_{key}")
    if entry is None:
        return None
    data, expiry = entry
    if _time_mod.time() > expiry:
        return None  # expired
    return data


def _cache_set(key, data, ttl):
    """Write to session_state cache with TTL."""
    st.session_state[f"_cache_{key}"] = (data, _time_mod.time() + ttl)


def _cache_age(key):
    """Return age of cache entry in seconds, or None if missing/expired."""
    entry = st.session_state.get(f"_cache_{key}")
    if entry is None:
        return None
    _, expiry = entry
    ttl_remaining = expiry - _time_mod.time()
    if ttl_remaining <= 0:
        return None
    # We don't store creation time, so approximate from remaining TTL
    return ttl_remaining


def _cache_clear_all():
    """Clear all cache entries from session_state."""
    keys_to_delete = [k for k in st.session_state if k.startswith("_cache_")]
    for k in keys_to_delete:
        del st.session_state[k]


def cache_stats():
    """Return summary of cached entries."""
    entries = {}
    for k, v in st.session_state.items():
        if k.startswith("_cache_"):
            clean_key = k[7:]  # strip _cache_ prefix
            _, expiry = v
            remaining = max(0, expiry - _time_mod.time())
            entries[clean_key] = {
                'remaining_s': int(remaining),
                'alive': remaining > 0,
            }
    return entries
# ─────────────────────────────────────────────
#  CHART KEY COUNTER — unique key per plotly chart per run
# ─────────────────────────────────────────────
_chart_key_counter = [0]

def _next_chart_key():
    _chart_key_counter[0] += 1
    return f"ck_{_chart_key_counter[0]}"



# ─────────────────────────────────────────────
#  CUSTOM CSS — dark industrial / quantitative
# ─────────────────────────────────────────────
# ── Dynamic theme CSS ────────────────────────────────────────────
_dark = st.session_state.get('dark_mode', True)

# Colour palette
_C = {
    'app_bg':     '#0a0c10' if _dark else '#f0f4f8',
    'card_bg':    '#0d1117' if _dark else '#ffffff',
    'card_bg2':   '#111520' if _dark else '#f8fafc',
    'sidebar_bg': '#070910' if _dark else '#e8eef5',
    'border':     '#1e2d40' if _dark else '#d0dae8',
    'border2':    '#1a2535' if _dark else '#c8d4e0',
    'text_pri':   '#c8d6e5' if _dark else '#1a2535',
    'text_sec':   '#5a7a99' if _dark else '#4a6480',
    'text_dim':   '#3d6080' if _dark else '#7a9ab8',
    'accent':     '#00d4ff',
    'tab_bg':     '#0d1117' if _dark else '#e8eef5',
    'chip_bg':    '#111a26' if _dark else '#e0eaf5',
    'chip_bdr':   '#1e3a55' if _dark else '#b8cce0',
    'rec_bg':     'linear-gradient(135deg,#0d1f10,#0a1a0d)' if _dark else 'linear-gradient(135deg,#e8f5ea,#f0faf0)',
    'rec_bdr':    '#1a4020' if _dark else '#90c8a0',
    'warn_bg':    'linear-gradient(135deg,#1f1500,#1a1000)' if _dark else 'linear-gradient(135deg,#fff8e8,#fffaf0)',
    'warn_bdr':   '#402a00' if _dark else '#e0c060',
    'btn_bg':     'linear-gradient(135deg,#003d5c,#00263d)' if _dark else 'linear-gradient(135deg,#dceef8,#c8e0f0)',
    'btn_bdr':    '#005580' if _dark else '#7ab8d8',
    'val_color':  '#e8f4fd' if _dark else '#0a1a2e',
    'sec_hdr':    '#3d6080' if _dark else '#3d6080',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Mono', monospace;
    background-color: {_C['app_bg']};
    color: {_C['text_pri']};
}}

.stApp {{ background-color: {_C['app_bg']}; }}

h1, h2, h3 {{
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
    color: {_C['text_pri']};
}}

.metric-card {{
    background: {_C['card_bg2']};
    border: 1px solid {_C['border']};
    border-left: 3px solid #00d4ff;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 6px 0;
}}

.metric-card.put {{ border-left-color: #ff4b6e; }}
.metric-card.call {{ border-left-color: #00d4ff; }}
.metric-card.neutral {{ border-left-color: #f0a500; }}

.metric-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {_C['text_sec']};
    margin-bottom: 4px;
}}

.metric-value {{
    font-size: 26px;
    font-weight: 700;
    color: {_C['val_color']};
    font-family: 'Space Mono', monospace;
}}

.metric-sub {{
    font-size: 11px;
    color: {_C['text_dim']};
    margin-top: 4px;
}}

.greek-row {{ display: flex; gap: 8px; margin: 8px 0; }}

.greek-chip {{
    background: {_C['chip_bg']};
    border: 1px solid {_C['chip_bdr']};
    border-radius: 3px;
    padding: 6px 12px;
    font-size: 12px;
    flex: 1;
    text-align: center;
}}

.greek-name {{ color: {_C['text_sec']}; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
.greek-val  {{ color: #00d4ff; font-size: 15px; font-weight: 600; }}

.stSelectbox > div, .stNumberInput > div, .stSlider {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

.section-header {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {_C['sec_hdr']};
    border-bottom: 1px solid {_C['border2']};
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
}}

.source-chip {{
    display: inline-block;
    background: {_C['chip_bg']};
    border: 1px solid {_C['chip_bdr']};
    border-radius: 2px;
    padding: 3px 10px;
    font-size: 10px;
    color: #5a9abf;
    margin: 3px 3px;
    letter-spacing: 1px;
}}

.recommendation-box {{
    background: {_C['rec_bg']};
    border: 1px solid {_C['rec_bdr']};
    border-left: 3px solid #00e676;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 10px 0;
}}

.warning-box {{
    background: {_C['warn_bg']};
    border: 1px solid {_C['warn_bdr']};
    border-left: 3px solid #f0a500;
    border-radius: 4px;
    padding: 14px 20px;
    margin: 10px 0;
    font-size: 12px;
    color: {'#c8a050' if _dark else '#806020'};
}}

.stButton > button {{
    background: {_C['btn_bg']};
    color: #00d4ff;
    border: 1px solid {_C['btn_bdr']};
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 8px 20px;
    transition: all 0.2s;
}}

.stSidebar {{ background-color: {_C['sidebar_bg']} !important; }}
.stSidebar > div {{ background-color: {_C['sidebar_bg']} !important; }}
div[data-testid="stSidebarContent"] {{ background-color: {_C['sidebar_bg']}; }}

.stTabs [data-baseweb="tab-list"] {{
    background-color: {_C['tab_bg']};
    border-bottom: 1px solid {_C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    color: {_C['text_dim']};
}}
.stTabs [aria-selected="true"] {{
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}}

footer {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  THEME HELPERS (used by all plotly charts)
# ─────────────────────────────────────────────
_PLT_BG   = '#0d1117' if _dark else '#ffffff'
_PLT_BG2  = '#0a0c10' if _dark else '#f5f7fa'
_PLT_GRID = '#1a2535' if _dark else '#e0e8f0'
_PLT_TXT  = '#5a7a99' if _dark else '#445566'
_PLT_LEG  = '#0d1117' if _dark else '#ffffff'
_PLT_BDR  = '#1e2d40' if _dark else '#d0dae8'

# ─────────────────────────────────────────────
#  CORE MODELS
# ─────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Standard Black-Scholes pricing."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0)


def merton_jump_diffusion(S, K, T, r, sigma, lam=0.5, mu_j=-0.1, sigma_j=0.2, option_type='call', n_terms=50):
    """Merton (1976) Jump-Diffusion — critical for crypto/illiquid OTC."""
    price = 0.0
    for n in range(n_terms):
        r_n = r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1) + n * (mu_j + 0.5 * sigma_j**2) / T if T > 0 else r
        sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T) if T > 0 else sigma
        lam_prime = lam * np.exp(mu_j + 0.5 * sigma_j**2)
        poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T)**n / math.factorial(min(n, 50))
        bs = black_scholes(S, K, T, r_n, sigma_n, option_type)
        price += poisson_weight * bs
    return price


def compute_greeks(S, K, T, r, sigma, option_type='call'):
    """Compute all first/second order Greeks."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega':  round(vega, 4),
        'rho':   round(rho, 4)
    }


def historical_volatility(prices, window=30):
    """Compute annualised HV from price series."""
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    if len(log_returns) < 2:
        return 0.6
    return np.std(log_returns) * np.sqrt(365)


# ─────────────────────────────────────────────
#  STRUCTURED PRODUCT PAYOFFS
# ─────────────────────────────────────────────

def snowball_payoff(S_path, K, barrier, coupon_rate, T_periods):
    """
    Autocallable Snowball:
    - Each period: if S >= barrier → autocall + accumulated coupon
    - At maturity: if S < K → loss = S/K - 1 (put exposure)
    """
    accumulated = 0
    for t, s in enumerate(S_path):
        accumulated += coupon_rate
        if s >= barrier:
            return accumulated  # Autocalled
    final = S_path[-1]
    if final < K:
        return final / K - 1  # Capital loss on put
    return coupon_rate * T_periods  # Full coupon, no autocall


def ladder_payoff(S_final, K, rungs):
    """
    Ladder: locks in gains at each rung (level above strike).
    rungs = list of (level, locked_return) pairs sorted ascending.
    """
    payoff = 0
    for level, locked in sorted(rungs):
        if S_final >= level:
            payoff = max(payoff, locked)
    if S_final > rungs[-1][0]:
        payoff = max(payoff, S_final / K - 1)
    return payoff


def monte_carlo_price(S, K, T, r, sigma, option_type='call', n_sims=50000, seed=42,
                      jump=False, lam=0.3, mu_j=-0.05, sigma_j=0.15):
    """Monte Carlo with optional jump-diffusion."""
    np.random.seed(seed)
    dt = T
    Z = np.random.standard_normal(n_sims)
    
    if jump:
        # Poisson jumps
        N_jumps = np.random.poisson(lam * T, n_sims)
        jump_sizes = np.array([
            np.sum(np.random.normal(mu_j, sigma_j, n)) if n > 0 else 0
            for n in N_jumps
        ])
        adj_r = r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        S_T = S * np.exp((adj_r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z + jump_sizes)
    else:
        S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_err, S_T


def _fetch_xdc_price_raw():
    """Raw XDC price fetch with CMC fallback."""
    return _fetch_coin_price("xdc-network", "XDC")


def fetch_xdc_price():
    """Cached XDC price (TTL: 60s)."""
    cached = _cache_get("xdc_price")
    if cached is not None:
        return cached
    result = _fetch_xdc_price_raw()
    if result is not None:
        _cache_set("xdc_price", result, CACHE_TTL_SPOT)
    return result


def _fetch_xdc_history_raw():
    """Raw 90d XDC history fetch with CMC fallback."""
    return _fetch_coin_history("xdc-network", "XDC", 90)


def fetch_xdc_history():
    """Cached 90d XDC history (TTL: 5min)."""
    cached = _cache_get("xdc_history_90d")
    if cached is not None:
        return cached
    result = _fetch_xdc_history_raw()
    if result is not None:
        _cache_set("xdc_history_90d", result, CACHE_TTL_HISTORY)
    return result


# ─────────────────────────────────────────────
#  MARKET DYNAMICS — PROBABILITY DISTRIBUTION
# ─────────────────────────────────────────────

def _fetch_eth_history_raw(days=90):
    """Raw ETH history fetch with CMC fallback."""
    return _fetch_coin_history("ethereum", "ETH", days)


def fetch_eth_history(days=90):
    """Cached ETH history (TTL: 5min)."""
    cache_key = f"eth_history_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = _fetch_eth_history_raw(days)
    if result is not None:
        _cache_set(cache_key, result, CACHE_TTL_HISTORY)
    return result

def _fetch_fear_greed_raw(days=30):
    """Raw Fear & Greed fetch."""
    try:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            return None
        data = r.json()
        entries = data.get('data', [])
        if not entries:
            return None
        result = []
        for e in entries:
            try:
                result.append((int(e['timestamp']), int(e['value']), e.get('value_classification', 'Unknown')))
            except (ValueError, KeyError):
                continue
        return result if result else None
    except Exception:
        return None


def fetch_fear_greed(days=30):
    """Cached Fear & Greed (TTL: 10min)."""
    cache_key = f"fear_greed_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = _fetch_fear_greed_raw(days)
    if result is not None:
        _cache_set(cache_key, result, CACHE_TTL_FEAR_GREED)
    return result

def _fetch_btc_history_raw(days=90):
    """Raw BTC history fetch with CMC fallback."""
    return _fetch_coin_history("bitcoin", "BTC", days)


def fetch_btc_history(days=90):
    """Cached BTC history (TTL: 5min)."""
    cache_key = f"btc_history_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = _fetch_btc_history_raw(days)
    if result is not None:
        _cache_set(cache_key, result, CACHE_TTL_HISTORY)
    return result

def _fetch_xdc_extended_raw(days=180):
    """Raw extended XDC history + volumes fetch."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/xdc-network/market_chart?vs_currency=usd&days={days}&interval=daily"
        r = _cg_get(url)
        data = r.json()
        prices = [p[1] for p in data.get('prices', [])]
        volumes = [p[1] for p in data.get('total_volumes', [])]
        if len(prices) > 5:
            return prices, volumes
    except Exception:
        pass
    # CMC fallback (prices only, synthesise flat volumes)
    prices = _cmc_history("XDC", days)
    if prices and len(prices) > 5:
        return prices, [1e6] * len(prices)
    return None, None


def fetch_xdc_extended(days=180):
    """Cached extended XDC history (TTL: 5min)."""
    cache_key = f"xdc_extended_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = _fetch_xdc_extended_raw(days)
    if result[0] is not None:
        _cache_set(cache_key, result, CACHE_TTL_HISTORY)
    return result

def _fetch_xdc_market_data_raw():
    """Raw XDC market data fetch (CoinGecko only — no CMC equivalent)."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/xdc-network?localization=false&tickers=false&community_data=true&developer_data=false"
        r = _cg_get(url, timeout=10)
        data = r.json()
        md = data.get('market_data', {})
        sentiment = data.get('sentiment_votes_up_percentage', None)
        return {
            'market_cap': md.get('market_cap', {}).get('usd', None),
            'volume_24h': md.get('total_volume', {}).get('usd', None),
            'price_change_7d': md.get('price_change_percentage_7d', None),
            'price_change_30d': md.get('price_change_percentage_30d', None),
            'ath': md.get('ath', {}).get('usd', None),
            'ath_change_pct': md.get('ath_change_percentage', {}).get('usd', None),
            'sentiment_up': sentiment,
        }
    except:
        return None


def fetch_xdc_market_data():
    """Cached XDC market data (TTL: 5min)."""
    cached = _cache_get("xdc_market_data")
    if cached is not None:
        return cached
    result = _fetch_xdc_market_data_raw()
    if result is not None:
        _cache_set("xdc_market_data", result, CACHE_TTL_MARKET_DATA)
        return result
    return {}

def compute_log_returns(prices):
    p = np.array(prices)
    return np.log(p[1:] / p[:-1])

def fit_gbm(log_rets, annualize=365):
    """Fit GBM (lognormal) — baseline."""
    mu = np.mean(log_rets) * annualize
    sigma = np.std(log_rets) * np.sqrt(annualize)
    return mu, sigma

def fit_student_t(log_rets):
    """Fit Student-t — heavy tails, better for crypto."""
    from scipy.stats import t as st_t
    df, loc, scale = st_t.fit(log_rets)
    return df, loc, scale

def fit_skew_normal(log_rets):
    """Fit skew-normal — captures asymmetric return distribution."""
    a, loc, scale = skewnorm.fit(log_rets)
    return a, loc, scale

def fit_gev(log_rets):
    """Fit Generalised Extreme Value — tail risk model."""
    c, loc, scale = genextreme.fit(log_rets)
    return c, loc, scale

def detect_regime(prices, short_window=10, long_window=30):
    """
    Simple regime detection:
    - Trend: SMA crossover
    - Volatility: rolling std vs historical
    - Momentum: RSI-like
    Returns regime label and confidence.
    """
    if len(prices) < long_window + 2:
        return "UNKNOWN", 0.5, {}
    
    p = np.array(prices)
    log_rets = np.log(p[1:] / p[:-1])
    
    # SMA trend
    sma_short = np.mean(p[-short_window:])
    sma_long  = np.mean(p[-long_window:])
    trend_signal = (sma_short - sma_long) / sma_long  # positive = bullish
    
    # Volatility regime
    recent_vol = np.std(log_rets[-10:]) * np.sqrt(365)
    hist_vol   = np.std(log_rets) * np.sqrt(365)
    vol_ratio  = recent_vol / hist_vol if hist_vol > 0 else 1.0
    
    # Momentum (simplified RSI)
    gains  = np.where(log_rets[-14:] > 0, log_rets[-14:], 0)
    losses = np.where(log_rets[-14:] < 0, -log_rets[-14:], 0)
    avg_gain = np.mean(gains) if np.mean(gains) > 0 else 1e-6
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 1e-6
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    
    # Drawdown from recent peak
    peak = np.max(p[-30:])
    drawdown = (p[-1] - peak) / peak
    
    # Regime classification
    if trend_signal > 0.03 and rsi > 55 and vol_ratio < 1.5:
        regime = "BULL TREND"
        confidence = min(0.45 + abs(trend_signal) * 2 + (rsi - 50) / 200, 0.85)
        color = "#00e676"
    elif trend_signal < -0.03 and rsi < 45 and drawdown < -0.10:
        regime = "BEAR TREND"
        confidence = min(0.45 + abs(trend_signal) * 2 + (50 - rsi) / 200, 0.85)
        color = "#ff4b6e"
    elif vol_ratio > 1.8:
        regime = "HIGH VOL / CRISIS"
        confidence = min(0.50 + (vol_ratio - 1.8) * 0.15, 0.80)
        color = "#f0a500"
    elif vol_ratio < 0.6:
        regime = "LOW VOL / ACCUMULATION"
        confidence = min(0.50 + (0.6 - vol_ratio) * 0.3, 0.75)
        color = "#b388ff"
    else:
        regime = "SIDEWAYS / CONSOLIDATION"
        confidence = 0.50 + abs(trend_signal) * 0.5
        color = "#5a9abf"
    
    stats = {
        'sma_short': sma_short,
        'sma_long': sma_long,
        'trend_signal': trend_signal,
        'vol_ratio': vol_ratio,
        'rsi': rsi,
        'drawdown': drawdown,
        'recent_vol': recent_vol,
        'hist_vol': hist_vol,
    }
    return regime, confidence, stats, color

def compute_btc_xdc_beta(xdc_prices, btc_prices):
    """Compute XDC beta vs BTC (market proxy)."""
    min_len = min(len(xdc_prices), len(btc_prices)) - 1
    if min_len < 10:
        return 1.0, 0.0
    xdc_rets = compute_log_returns(xdc_prices[-min_len-1:])
    btc_rets = compute_log_returns(btc_prices[-min_len-1:])
    cov = np.cov(xdc_rets, btc_rets)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
    corr = np.corrcoef(xdc_rets, btc_rets)[0, 1]
    return beta, corr

def project_distributions_30d(spot, log_rets, mu_hist, sigma_hist, regime, 
                               regime_confidence, btc_beta, btc_mu, n_sims=100000, seed=99):
    """
    Generate 30d price distributions using FOUR models:
    1. Historical GBM (from XDC own returns)
    2. Regime-adjusted GBM (drift tilted by regime)
    3. Student-t (fat tails)
    4. Market Proxy (BTC-implied drift + XDC beta)
    """
    np.random.seed(seed)
    T = 30 / 365
    
    # ── Model 1: Historical GBM
    Z = np.random.standard_normal(n_sims)
    S_hist = spot * np.exp((mu_hist - 0.5 * sigma_hist**2) * T + sigma_hist * np.sqrt(T) * Z)
    
    # ── Model 2: Regime-Adjusted GBM
    regime_drift_adj = {
        "BULL TREND":           +0.30,
        "BEAR TREND":           -0.30,
        "HIGH VOL / CRISIS":    -0.15,
        "LOW VOL / ACCUMULATION": +0.10,
        "SIDEWAYS / CONSOLIDATION": 0.0,
        "UNKNOWN":              0.0,
    }.get(regime, 0.0)
    
    # Blend historical drift with regime signal weighted by confidence
    mu_regime = mu_hist * (1 - regime_confidence) + regime_drift_adj * regime_confidence
    S_regime = spot * np.exp((mu_regime - 0.5 * sigma_hist**2) * T + sigma_hist * np.sqrt(T) * Z)
    
    # ── Model 3: Student-t (fat tails)
    df_t, loc_t, scale_t = fit_student_t(log_rets)
    from scipy.stats import t as st_t
    Z_t = st_t.rvs(df=df_t, loc=0, scale=1, size=n_sims)
    daily_sigma = sigma_hist / np.sqrt(365)
    # Scale to 30d via sum of daily t-draws
    # Approximate 30d terminal: use scaled t
    sigma_30 = sigma_hist * np.sqrt(T)
    mu_30 = (mu_hist - 0.5 * sigma_hist**2) * T
    S_student = spot * np.exp(mu_30 + sigma_30 * Z_t)
    
    # ── Model 4: Market Proxy (BTC-implied)
    # BTC expected 30d drift (annualised) applied via beta
    mu_btc_implied = btc_mu * btc_beta  # beta-scaled BTC drift
    # Blend with own history
    mu_proxy = 0.5 * mu_hist + 0.5 * mu_btc_implied
    S_proxy = spot * np.exp((mu_proxy - 0.5 * sigma_hist**2) * T + sigma_hist * np.sqrt(T) * Z)
    
    return {
        'Historical GBM':     S_hist,
        'Regime-Adjusted':    S_regime,
        'Fat-Tail (Student-t)': S_student,
        'Market Proxy (BTC β)': S_proxy,
    }

def ensemble_distribution(model_results, weights=None):
    """Blend all models into a single ensemble distribution."""
    if weights is None:
        weights = [0.25, 0.30, 0.25, 0.20]  # regime gets slightly more
    
    models = list(model_results.values())
    n = len(models[0])
    
    # Sample from each model according to weights
    ensemble = np.zeros(n)
    idx_start = 0
    for i, (model_samples, w) in enumerate(zip(models, weights)):
        n_take = int(w * n)
        perm = np.random.permutation(n)[:n_take]
        start = idx_start
        ensemble[start:start+n_take] = model_samples[perm]
        idx_start += n_take
    # Fill remainder from ensemble average
    if idx_start < n:
        avg = np.mean([m[:n-idx_start] for m in models], axis=0)
        ensemble[idx_start:] = avg
    
    return ensemble

def compute_price_scenarios(spot, distributions):
    """Return bull/base/bear scenario probabilities for key price levels."""
    scenarios = {}
    for name, sims in distributions.items():
        scenarios[name] = {
            'P(>+50%)': np.mean(sims > spot * 1.50),
            'P(>+30%)': np.mean(sims > spot * 1.30),
            'P(>+10%)': np.mean(sims > spot * 1.10),
            'P(flat±5%)': np.mean((sims > spot * 0.95) & (sims < spot * 1.05)),
            'P(<-10%)': np.mean(sims < spot * 0.90),
            'P(<-30%)': np.mean(sims < spot * 0.70),
            'P(<-50%)': np.mean(sims < spot * 0.50),
            'Median': np.median(sims),
            'Mean':   np.mean(sims),
            'P5':     np.percentile(sims, 5),
            'P25':    np.percentile(sims, 25),
            'P75':    np.percentile(sims, 75),
            'P95':    np.percentile(sims, 95),
        }
    return scenarios


def fit_garch11(log_rets, horizon=30):
    """
    GARCH(1,1) vol forecast with MLE parameter estimation.
    σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
    Uses scipy.optimize.minimize to find optimal (ω, α, β).
    Returns daily vol forecast for each of `horizon` days.
    """
    n = len(log_rets)
    var_baseline = np.var(log_rets)

    def garch_loglik(params, returns):
        omega, alpha, beta = params
        n_ = len(returns)
        sigma2 = np.zeros(n_)
        sigma2[0] = var_baseline
        for i in range(1, n_):
            sigma2[i] = omega + alpha * returns[i-1]**2 + beta * sigma2[i-1]
            sigma2[i] = max(sigma2[i], 1e-12)
        # Negative log-likelihood (Gaussian)
        ll = -0.5 * np.sum(np.log(sigma2[1:]) + returns[1:]**2 / sigma2[1:])
        return -ll  # minimize negative LL

    # Starting values
    x0 = [var_baseline * 0.05, 0.10, 0.85]
    bounds = [(1e-10, var_baseline * 5), (1e-6, 0.5), (0.3, 0.999)]

    # Constraint: alpha + beta < 1 (stationarity)
    constraints = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}]

    try:
        result = minimize(
            garch_loglik, x0, args=(log_rets,),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if result.success:
            omega, alpha, beta = result.x
        else:
            omega, alpha, beta = x0
    except Exception:
        omega, alpha, beta = x0

    # Filter: compute conditional variance series with estimated params
    sigma2 = np.zeros(n)
    sigma2[0] = var_baseline
    for i in range(1, n):
        sigma2[i] = omega + alpha * log_rets[i-1]**2 + beta * sigma2[i-1]
        sigma2[i] = max(sigma2[i], 1e-12)

    # Forecast h steps ahead
    last_eps2  = log_rets[-1]**2
    last_sig2  = sigma2[-1]
    persist    = alpha + beta
    long_run   = omega / max(1 - persist, 1e-6)

    forecasts = []
    sig2_fwd = last_sig2
    for h in range(1, horizon + 1):
        sig2_fwd = omega + persist * sig2_fwd
        forecasts.append(np.sqrt(sig2_fwd * 365))  # annualised

    return {
        'omega': omega, 'alpha': alpha, 'beta': beta,
        'persistence': persist,
        'long_run_vol': np.sqrt(long_run * 365),
        'current_vol': np.sqrt(last_sig2 * 365),
        'forecasts': forecasts,  # daily annualised vol for 30 days
        'sigma2_series': sigma2,
    }


def fit_skew_normal_params(log_rets):
    """Fit skew-normal — captures left/right skew asymmetry in returns."""
    a, loc, scale = skewnorm.fit(log_rets)
    return a, loc, scale


def fit_gev_params(log_rets):
    """Fit GEV — generalised extreme value distribution for tail analysis."""
    c, loc, scale = genextreme.fit(log_rets)
    return c, loc, scale


def project_distributions_30d_enhanced(
    spot, log_rets, mu_hist, sigma_hist, regime, regime_confidence,
    btc_beta, btc_mu, eth_beta, eth_mu, fg_score,
    garch_result, n_sims=100_000, seed=99
):
    """
    Generate 30d price distributions using SIX models:
    1. Historical GBM          — XDC own returns, plain GBM
    2. Regime-Adjusted GBM     — drift tilted by detected regime
    3. Student-t               — fat tails fitted to XDC returns
    4. BTC Market Proxy        — BTC drift × XDC beta
    5. ETH Market Proxy        — ETH drift × XDC eta-beta (second independent proxy)
    6. GARCH(1,1) + Skew       — vol clustering + skew-normal shocks
    """
    np.random.seed(seed)
    T_ann = 30 / 365
    n_days = 30
    
    # Shared Gaussian innovations
    Z = np.random.standard_normal(n_sims)

    # ── 1. Historical GBM
    S_hist = spot * np.exp((mu_hist - 0.5 * sigma_hist**2) * T_ann + sigma_hist * np.sqrt(T_ann) * Z)

    # ── 2. Regime-Adjusted GBM
    regime_adj = {
        "BULL TREND":             +0.30,
        "BEAR TREND":             -0.30,
        "HIGH VOL / CRISIS":      -0.15,
        "LOW VOL / ACCUMULATION": +0.10,
        "SIDEWAYS / CONSOLIDATION": 0.0,
        "UNKNOWN":                0.0,
    }.get(regime, 0.0)
    
    # Fear & Greed overlay: shift drift by ±5% based on F&G
    fg_adj = 0.0
    if fg_score is not None:
        fg_adj = (fg_score - 50) / 50 * 0.08  # ±8% annualised drift at extremes
    
    mu_regime = (mu_hist * (1 - regime_confidence) + regime_adj * regime_confidence) + fg_adj
    S_regime  = spot * np.exp((mu_regime - 0.5 * sigma_hist**2) * T_ann + sigma_hist * np.sqrt(T_ann) * Z)

    # ── 3. Student-t (fat tails)
    from scipy.stats import t as st_t
    df_t, loc_t, scale_t = fit_student_t(log_rets)
    Z_t      = st_t.rvs(df=df_t, loc=0, scale=1, size=n_sims)
    sigma_30 = sigma_hist * np.sqrt(T_ann)
    mu_30    = (mu_hist - 0.5 * sigma_hist**2) * T_ann
    S_student = spot * np.exp(mu_30 + sigma_30 * Z_t)

    # ── 4. BTC Market Proxy
    mu_btc_proxy = 0.50 * mu_hist + 0.50 * (btc_mu * btc_beta)
    S_btc = spot * np.exp((mu_btc_proxy - 0.5 * sigma_hist**2) * T_ann + sigma_hist * np.sqrt(T_ann) * Z)

    # ── 5. ETH Market Proxy (independent second proxy)
    mu_eth_proxy = 0.50 * mu_hist + 0.50 * (eth_mu * eth_beta)
    # Use different random seed for independence
    np.random.seed(seed + 1)
    Z_eth = np.random.standard_normal(n_sims)
    S_eth = spot * np.exp((mu_eth_proxy - 0.5 * sigma_hist**2) * T_ann + sigma_hist * np.sqrt(T_ann) * Z_eth)

    # ── 6. GARCH(1,1) path simulation with skew-normal shocks
    np.random.seed(seed + 2)
    a_sn, loc_sn, scale_sn = fit_skew_normal_params(log_rets)
    
    omega_g = garch_result['omega']
    alpha_g = garch_result['alpha']
    beta_g  = garch_result['beta']
    
    # Simulate n_sims paths of 30 days
    sig2_t  = np.full(n_sims, garch_result['sigma2_series'][-1])  # start from last observed
    S_paths = np.ones(n_sims) * spot
    daily_mu_garch = mu_hist / 365
    prev_eps = np.zeros(n_sims)  # previous innovation for GARCH update

    for d in range(n_days):
        # Skew-normal innovations
        eps = skewnorm.rvs(a=a_sn, loc=0, scale=scale_sn, size=n_sims)
        # GARCH vol update: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
        sig2_t = omega_g + alpha_g * prev_eps**2 + beta_g * sig2_t
        sig2_t = np.maximum(sig2_t, 1e-12)
        sig_t  = np.sqrt(sig2_t)
        prev_eps = sig_t * eps  # store scaled innovation for next step
        S_paths = S_paths * np.exp(daily_mu_garch - 0.5 * sig_t**2 + sig_t * eps)

    S_garch = S_paths

    return {
        'Historical GBM':        S_hist,
        'Regime + Fear/Greed':   S_regime,
        'Fat-Tail (Student-t)':  S_student,
        'BTC Market Proxy':      S_btc,
        'ETH Market Proxy':      S_eth,
        'GARCH(1,1) + Skew':     S_garch,
    }


def compute_model_conviction(distributions, spot):
    """
    Measures agreement/disagreement across models.
    Returns a conviction score (0-100) and direction agreement.
    """
    medians = {k: np.median(v) for k, v in distributions.items()}
    bull_count = sum(1 for m in medians.values() if m > spot * 1.05)
    bear_count = sum(1 for m in medians.values() if m < spot * 0.95)
    flat_count = len(medians) - bull_count - bear_count
    
    total = len(medians)
    bull_pct = bull_count / total
    bear_pct = bear_count / total
    
    # Conviction = how aligned the models are
    max_agree = max(bull_count, bear_count, flat_count)
    conviction = (max_agree / total) * 100
    
    if bull_count >= bear_count and bull_count >= flat_count:
        direction = "BULLISH"
        dir_color = "#00e676"
    elif bear_count > bull_count and bear_count >= flat_count:
        direction = "BEARISH"
        dir_color = "#ff4b6e"
    else:
        direction = "NEUTRAL"
        dir_color = "#f0a500"
    
    # Median spread (IQR of all model medians as % of spot)
    median_vals = list(medians.values())
    spread = (max(median_vals) - min(median_vals)) / spot * 100
    
    return {
        'conviction': conviction,
        'direction': direction,
        'dir_color': dir_color,
        'bull_models': bull_count,
        'bear_models': bear_count,
        'flat_models': flat_count,
        'median_spread_pct': spread,
        'medians': medians,
    }


def compute_conditional_distributions(spot, log_rets_xdc, btc_rets, eth_rets, 
                                       sigma_hist, n_sims=50_000, seed=77):
    """
    Conditional distributions: 
    - IF BTC +20% in 30d → what does XDC distribution look like?
    - IF BTC -20% in 30d → what does XDC distribution look like?
    Uses historical correlation structure.
    """
    np.random.seed(seed)
    min_len = min(len(log_rets_xdc), len(btc_rets) if btc_rets is not None else len(log_rets_xdc)) - 1
    min_len = max(min_len, 10)
    
    xdc_r = log_rets_xdc[-min_len:]
    if btc_rets is not None and len(btc_rets) > min_len:
        btc_r = btc_rets[-min_len:]
    else:
        btc_r = xdc_r * 0.7 + np.random.normal(0, 0.01, len(xdc_r))
    
    if len(xdc_r) != len(btc_r):
        min_n = min(len(xdc_r), len(btc_r))
        xdc_r, btc_r = xdc_r[:min_n], btc_r[:min_n]
    
    # Estimate correlation
    rho = np.corrcoef(xdc_r, btc_r)[0, 1] if len(xdc_r) > 3 else 0.65
    rho = np.clip(rho, -0.95, 0.95)
    
    mu_xdc_daily = np.mean(log_rets_xdc) 
    sigma_xdc_daily = np.std(log_rets_xdc)
    mu_btc_daily    = np.mean(btc_r)
    sigma_btc_daily = np.std(btc_r)
    
    n_days = 30
    
    results = {}
    scenarios_cond = {
        'BTC Bull (+20%)': mu_btc_daily + 0.20 / n_days,
        'BTC Bear (-20%)': mu_btc_daily - 0.20 / n_days,
        'BTC Flat (±2%)':  mu_btc_daily,
        'BTC Crash (-40%)': mu_btc_daily - 0.40 / n_days,
    }
    
    for label, mu_btc_cond in scenarios_cond.items():
        # Simulate correlated BTC/XDC paths
        Z1 = np.random.standard_normal((n_sims, n_days))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal((n_sims, n_days))
        
        # Condition BTC on scenario by adjusting drift
        btc_drift_adj = mu_btc_cond
        xdc_drift_adj = mu_xdc_daily + rho * (sigma_xdc_daily / sigma_btc_daily) * (mu_btc_cond - mu_btc_daily)
        
        # Simulate XDC paths using conditional drift
        xdc_paths = spot * np.exp(np.cumsum(
            xdc_drift_adj + sigma_xdc_daily * Z2, axis=1
        ))
        results[label] = xdc_paths[:, -1]  # terminal price
    
    return results, rho


# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend_structure(hv, spot, call_prices, put_prices, strikes, T):
    """Heuristic recommendation based on vol and pricing."""
    
    put_call_ratios = [p/c if c > 0.001 else 1 for p, c in zip(put_prices, call_prices)]
    avg_pcr = np.mean(put_call_ratios)
    skew = put_call_ratios[-1] - put_call_ratios[0]  # OTM put vs call
    
    recs = []

    # High vol → sell structured vol (snowball earns coupon)
    if hv > 0.6:
        recs.append({
            'name': 'Snowball (Autocallable)',
            'score': 90 if hv > 0.8 else 75,
            'rationale': f'High HV ({hv:.0%}) → elevated coupon. Snowball benefits from vol premium. Sell vol, earn carry.',
            'risk': 'Downside put exposure if price gaps below strike.',
            'color': '#00e676'
        })
    
    # Directional upside with protection → Ladder
    if hv > 0.4:
        recs.append({
            'name': 'Ladder (Call Spread + Locks)',
            'score': 80,
            'rationale': f'Captures upside at +30/+40/+50% while locking in gains at each rung. Good for asymmetric upside plays.',
            'risk': 'Capped upside beyond highest rung unless you own the digital.',
            'color': '#00d4ff'
        })

    # High put/call skew → collar or risk reversal
    if avg_pcr > 1.3:
        recs.append({
            'name': 'Risk Reversal (Sell Put / Buy Call)',
            'score': 70,
            'rationale': f'Put skew is elevated (PCR avg: {avg_pcr:.2f}). Selling OTM put to finance OTM call is attractive.',
            'risk': 'Short put exposure; requires margin or cash collateral.',
            'color': '#f0a500'
        })

    # Low vol → buy optionality cheap
    if hv < 0.4:
        recs.append({
            'name': 'Long Straddle / Strangle',
            'score': 85,
            'rationale': f'Low HV ({hv:.0%}) means options are cheap. Buy vol before it mean-reverts.',
            'risk': 'Decay (theta) if vol stays low.',
            'color': '#b388ff'
        })

    # Long T → DCD (Dual Currency Deposit equivalent)
    if T > 180/365:
        recs.append({
            'name': 'Dual Currency Deposit (DCD)',
            'score': 65,
            'rationale': f'Tenor > 6 months. Deposit earns enhanced yield by embedding a short OTM option. Common in Islamic finance structures.',
            'risk': 'Settlement in weaker currency if option triggers.',
            'color': '#ff8a65'
        })

    return sorted(recs, key=lambda x: -x['score'])


# ─────────────────────────────────────────────
#  SIDEBAR — INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div style="font-family:Space Mono;font-size:18px;color:#00d4ff;letter-spacing:2px;padding:10px 0;">⬡ OTC PRICER</div>', unsafe_allow_html=True)

    # ── Theme toggle + Refresh ──────────────────────────────
    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = True
    _tb1, _tb2 = st.columns(2)
    with _tb1:
        _mode_label = "🌙 Dark" if st.session_state['dark_mode'] else "☀️ Light"
        if st.button(_mode_label, key="theme_toggle", use_container_width=True):
            st.session_state['dark_mode'] = not st.session_state['dark_mode']
            st.rerun()
    with _tb2:
        if st.button("🔄 Refresh Data", key="global_refresh", use_container_width=True):
            _cache_clear_all()
            # Clear vol-surface session keys too
            for k in list(st.session_state.keys()):
                if k.startswith("vs_"):
                    del st.session_state[k]
            st.rerun()
    st.markdown(f'<div style="font-size:10px;color:#3d6080;letter-spacing:2px;margin-bottom:20px;">TRADE FINTECH · {st.session_state.get("active_token_ticker","XDC")} OPTIONS</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">UNDERLYING</div>', unsafe_allow_html=True)

    # ── Token selector ──────────────────────────────────
    TOKENS = {
        "XDC  — XDC Network":    ("xdc-network",        "XDC",   0.035),
        "BTC  — Bitcoin":        ("bitcoin",                "BTC",   65000.0),
        "ETH  — Ethereum":       ("ethereum",               "ETH",   3500.0),
        "HYPE — Hyperliquid":    ("hyperliquid",            "HYPE",  20.0),
        "Custom / Manual Entry": (None,                     "CUSTOM", 1.0),
    }

    token_choice = st.selectbox("Token", list(TOKENS.keys()), index=0, key="token_choice")
    coin_id, token_ticker, token_default_price = TOKENS[token_choice]

    if coin_id is not None:
        with st.spinner(f"Fetching {token_ticker} price..."):
            live_price = _fetch_coin_price(coin_id, token_ticker)
        if live_price:
            st.success(f"Live {token_ticker}: ${live_price:,.4f}")
            spot = st.number_input("Spot Override ($)", value=float(live_price),
                                   format="%.6f" if live_price < 1 else "%.2f")
        else:
            st.warning("Price API unavailable — enter manually")
            spot = st.number_input("Spot Price ($)", value=float(token_default_price),
                                   format="%.6f" if token_default_price < 1 else "%.2f")
    else:
        spot = st.number_input("Spot Price ($)", value=1.0, format="%.6f")

    # Store coin_id in session for tabs that need history (vol, market dynamics)
    st.session_state["active_coin_id"]     = coin_id or "custom"
    st.session_state["active_token_ticker"] = token_ticker

    st.markdown('<div class="section-header">VOLATILITY</div>', unsafe_allow_html=True)
    
    vol_mode = st.selectbox("Vol Source", ["Manual Input", f'Auto from {st.session_state.get("active_token_ticker","Token")} History', "GARCH (EWMA)"])
    
    if vol_mode == "Manual Input":
        hv = st.slider("Historical Vol (annual)", 0.10, 2.50, 0.80, 0.05)
        iv_spread = st.slider("OTC IV Spread (add to HV)", 0.0, 0.30, 0.10, 0.01)
        sigma = hv + iv_spread
    elif "Auto from" in vol_mode:
        _cid_vol = st.session_state.get("active_coin_id", "xdc-network")
        _cid_vol = _cid_vol if _cid_vol != "custom" else "xdc-network"
        _ticker_vol = st.session_state.get("active_token_ticker", "XDC")
        with st.spinner("Fetching 90d history..."):
            price_hist = _fetch_coin_history(_cid_vol, _ticker_vol, 90)
        if price_hist:
            hv_30 = historical_volatility(price_hist[-31:], 30)
            hv_90 = historical_volatility(price_hist, 90)
            st.metric("30d HV", f"{hv_30:.1%}")
            st.metric("90d HV", f"{hv_90:.1%}")
            hv = hv_30
        else:
            st.warning("Could not fetch history")
            hv = 0.80
        iv_spread = st.slider("OTC Spread", 0.0, 0.30, 0.10, 0.01)
        sigma = hv + iv_spread
    else:  # EWMA
        _cid_vol = st.session_state.get("active_coin_id", "xdc-network")
        _cid_vol = _cid_vol if _cid_vol != "custom" else "xdc-network"
        _ticker_vol = st.session_state.get("active_token_ticker", "XDC")
        with st.spinner("Fetching history..."):
            price_hist = _fetch_coin_history(_cid_vol, _ticker_vol, 90)
        if price_hist and len(price_hist) > 5:
            log_rets = np.log(np.array(price_hist[1:]) / np.array(price_hist[:-1]))
            lam_ewma = 0.94
            ewma_var = np.var(log_rets[:5])
            for r in log_rets[5:]:
                ewma_var = lam_ewma * ewma_var + (1 - lam_ewma) * r**2
            hv = np.sqrt(ewma_var * 365)
            st.metric("EWMA Vol (λ=0.94)", f"{hv:.1%}")
        else:
            hv = 0.80
        iv_spread = st.slider("OTC Spread", 0.0, 0.30, 0.10, 0.01)
        sigma = hv + iv_spread

    st.markdown('<div class="section-header">CONTRACT TERMS</div>', unsafe_allow_html=True)
    
    T_days = st.slider("Expiry (days)", 7, 365, 90)
    T = T_days / 365
    r_pct = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 5.25, 0.25)
    r = r_pct / 100
    notional = st.number_input("Notional (USD)", value=100_000, step=10_000)

    st.markdown('<div class="section-header">MODEL</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Pricing Model", ["Black-Scholes", "Jump-Diffusion (Merton)", "Monte Carlo", "Monte Carlo + Jumps"])
    
    use_jump = model_choice in ["Jump-Diffusion (Merton)", "Monte Carlo + Jumps"]
    
    if use_jump:
        with st.expander("Jump Parameters"):
            lam_j  = st.slider("Jump Intensity (λ)", 0.0, 5.0, 0.5, 0.1)
            mu_j   = st.slider("Mean Jump Size", -0.5, 0.2, -0.10, 0.01)
            sig_j  = st.slider("Jump Vol", 0.01, 0.80, 0.20, 0.01)
    else:
        lam_j, mu_j, sig_j = 0.5, -0.10, 0.20

    # Snowball params moved to Structures tab (tab4)

    # ── CACHE STATUS ──────────────────────────────────────
    st.markdown('<div class="section-header">CACHE STATUS</div>', unsafe_allow_html=True)
    _stats = cache_stats()
    _active_caches = sum(1 for v in _stats.values() if v['alive'])
    _total_caches = len(_stats)

    if _total_caches > 0:
        st.markdown(
            f'<div style="font-size:10px;color:#5a7a99;line-height:1.8;">'
            f'Active: <span style="color:#00e676">{_active_caches}</span> / {_total_caches} entries<br>'
            + '<br>'.join(
                f'<span style="color:{"#00e676" if v["alive"] else "#ff4b6e"}">{"●" if v["alive"] else "○"}</span> '
                f'{k}: <span style="color:#5a9abf">{v["remaining_s"]}s</span>'
                for k, v in sorted(_stats.items())[:8]
            )
            + '</div>',
            unsafe_allow_html=True
        )
    else:
        st.caption("No cached data yet")

    if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
        _cache_clear_all()
        st.rerun()


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

st.markdown(f'<h1 style="font-family:Space Mono;color:#e8f4fd;font-size:24px;letter-spacing:-1px;">OTC STRUCTURED OPTIONS PRICER — {st.session_state.get("active_token_ticker","XDC")}</h1>', unsafe_allow_html=True)
st.markdown(f'<div style="font-size:11px;color:#3d6080;letter-spacing:2px;margin-bottom:24px;">SPOT: <span style="color:#00d4ff">${spot:.5f}</span> &nbsp;|&nbsp; σ: <span style="color:#f0a500">{sigma:.1%}</span> &nbsp;|&nbsp; T: <span style="color:#00e676">{T_days}d</span> &nbsp;|&nbsp; MODEL: <span style="color:#b388ff">{model_choice.upper()}</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STRIKE GRID: 30 / 40 / 50% OTM
# ─────────────────────────────────────────────

strike_offsets = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]
strike_labels  = ["-30%", "-20%", "-10%", "ATM", "+10%", "+20%", "+30%", "+40%", "+50%"]
strikes        = [spot * x for x in strike_offsets]

call_prices, put_prices, call_greeks_list, put_greeks_list = [], [], [], []

for K in strikes:
    if model_choice == "Black-Scholes":
        cp = black_scholes(spot, K, T, r, sigma, 'call')
        pp = black_scholes(spot, K, T, r, sigma, 'put')
    elif model_choice == "Jump-Diffusion (Merton)":
        cp = merton_jump_diffusion(spot, K, T, r, sigma, lam_j, mu_j, sig_j, 'call')
        pp = merton_jump_diffusion(spot, K, T, r, sigma, lam_j, mu_j, sig_j, 'put')
    elif model_choice == "Monte Carlo":
        cp, _, _ = monte_carlo_price(spot, K, T, r, sigma, 'call')
        pp, _, _ = monte_carlo_price(spot, K, T, r, sigma, 'put')
    else:  # MC + Jumps
        cp, _, _ = monte_carlo_price(spot, K, T, r, sigma, 'call', jump=True, lam=lam_j, mu_j=mu_j, sigma_j=sig_j)
        pp, _, _ = monte_carlo_price(spot, K, T, r, sigma, 'put',  jump=True, lam=lam_j, mu_j=mu_j, sigma_j=sig_j)
    
    call_prices.append(cp)
    put_prices.append(pp)
    call_greeks_list.append(compute_greeks(spot, K, T, r, sigma, 'call'))
    put_greeks_list.append(compute_greeks(spot, K, T, r, sigma, 'put'))



# ─────────────────────────────────────────────────────────
#  DERIBIT & VOL SURFACE FUNCTIONS
# ─────────────────────────────────────────────────────────

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

def deribit_get(method, params=None):
    """Generic Deribit public API — no auth needed for market data."""
    try:
        url = f"{DERIBIT_BASE}/{method}"
        r = requests.get(url, params=params or {}, timeout=10)
        data = r.json()
        if data.get("result") is not None:
            return data["result"]
        return None
    except Exception:
        return None

def fetch_deribit_index(currency):
    """Cached Deribit index price (TTL: 2min)."""
    cache_key = f"deribit_idx_{currency}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = deribit_get("get_index_price", {"index_name": f"{currency.lower()}_usd"})
    val = result.get("index_price") if result else None
    if val is not None:
        _cache_set(cache_key, val, CACHE_TTL_DERIBIT)
    return val

def fetch_deribit_book_summary(currency):
    """Cached Deribit book summary (TTL: 2min)."""
    cache_key = f"deribit_book_{currency}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = deribit_get("get_book_summary_by_currency", {
        "currency": currency.upper(), "kind": "option"
    })
    val = result or []
    if val:
        _cache_set(cache_key, val, CACHE_TTL_DERIBIT)
    return val

def fetch_deribit_dvol_history(currency, days=30):
    """Cached DVOL history (TTL: 10min)."""
    cache_key = f"deribit_dvol_{currency}_{days}d"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = deribit_get("get_volatility_index_data", {
            "currency":        currency.upper(),
            "start_timestamp": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "end_timestamp":   int(datetime.now().timestamp() * 1000),
            "resolution":      "3600"
        })
        if result and result.get("data"):
            val = [(row[0], row[4]) for row in result["data"]]  # (timestamp, close)
            _cache_set(cache_key, val, CACHE_TTL_COMPUTATION)
            return val
        return None
    except Exception:
        return None

def fetch_deribit_ticker(instrument_name):
    """Cached Deribit ticker (TTL: 2min)."""
    cache_key = f"deribit_tick_{instrument_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    result = deribit_get("get_ticker", {"instrument_name": instrument_name})
    if result is not None:
        _cache_set(cache_key, result, CACHE_TTL_DERIBIT)
    return result

def parse_book_summary(book_data, index_price):
    """Parse raw book summary into structured vol surface DataFrame."""
    if not book_data:
        return pd.DataFrame()
    rows = []
    for item in book_data:
        name   = item.get("instrument_name", "")
        parts  = name.split("-")
        if len(parts) != 4:
            continue
        currency   = parts[0]
        expiry_str = parts[1]
        try:
            strike = float(parts[2])
        except ValueError:
            continue
        opt_type = "call" if parts[3] == "C" else "put"
        try:
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y")
        except ValueError:
            try:
                expiry_dt = datetime.strptime(expiry_str, "%d%b%Y")
            except ValueError:
                continue

        T_days    = max((expiry_dt - datetime.now()).days, 0)
        T_years   = T_days / 365
        moneyness = strike / index_price if index_price and index_price > 0 else 1.0
        greeks    = item.get("greeks", {}) or {}
        bid_iv    = item.get("bid_iv")
        ask_iv    = item.get("ask_iv")
        mark_iv   = item.get("mark_iv")
        mid_iv    = (bid_iv + ask_iv) / 2 if bid_iv and ask_iv else mark_iv

        rows.append({
            "instrument":    name,
            "currency":      currency,
            "expiry":        expiry_dt,
            "expiry_str":    expiry_str,
            "T_days":        T_days,
            "T_years":       T_years,
            "strike":        strike,
            "option_type":   opt_type,
            "moneyness":     moneyness,
            "log_moneyness": np.log(moneyness) if moneyness > 0 else 0,
            "mark_iv":       mark_iv,
            "bid_iv":        bid_iv,
            "ask_iv":        ask_iv,
            "mid_iv":        mid_iv,
            "mark_price":    item.get("mark_price"),
            "volume_usd":    item.get("volume_usd") or 0,
            "open_interest": item.get("open_interest") or 0,
            "delta":         greeks.get("delta"),
            "gamma":         greeks.get("gamma"),
            "vega":          greeks.get("vega"),
            "theta":         greeks.get("theta"),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["mark_iv"].notna() & (df["T_days"] > 0) &
            (df["moneyness"] > 0.25) & (df["moneyness"] < 4.0)]
    return df.sort_values(["T_days", "strike"])

def compute_skew_metrics(df, iv_col="mark_iv"):
    """Compute ATM vol, 25d skew, wings, and vol-of-vol per expiry."""
    metrics = []
    if df.empty:
        return pd.DataFrame()
    for T_days, grp in df[df[iv_col].notna()].groupby("T_days"):
        grp = grp.sort_values("moneyness")
        if len(grp) < 3:
            continue
        atm_idx  = (grp["moneyness"] - 1.0).abs().idxmin()
        atm_iv   = grp.loc[atm_idx, iv_col]
        otm_put  = grp[grp["moneyness"].between(0.85, 0.95)][iv_col]
        otm_call = grp[grp["moneyness"].between(1.05, 1.15)][iv_col]
        deep_put  = grp[grp["moneyness"].between(0.70, 0.80)][iv_col]
        deep_call = grp[grp["moneyness"].between(1.20, 1.30)][iv_col]
        put_iv   = otm_put.mean()  if not otm_put.empty  else np.nan
        call_iv  = otm_call.mean() if not otm_call.empty else np.nan
        skew_25  = put_iv - call_iv
        metrics.append({
            "T_days":     T_days,
            "Expiry":     grp["expiry_str"].iloc[0],
            "ATM IV %":   round(atm_iv, 2) if not np.isnan(atm_iv) else None,
            "25d Skew":   round(skew_25, 2) if not np.isnan(skew_25) else None,
            "Put Wing":   round(deep_put.mean(),  2) if not deep_put.empty  else None,
            "Call Wing":  round(deep_call.mean(), 2) if not deep_call.empty else None,
            "Vol-of-Vol": round(grp[iv_col].std(), 2),
            "# Strikes":  len(grp),
        })
    return pd.DataFrame(metrics).sort_values("T_days")

def interpolate_iv_at(df, target_moneyness, target_T_days, iv_col="mark_iv"):
    """Interpolate IV from a surface at given (moneyness, tenor)."""
    if df.empty:
        return None
    df_clean = df[df[iv_col].notna() & (df[iv_col] > 0)]
    if df_clean.empty:
        return None
    tenors     = df_clean["T_days"].unique()
    nearest_T  = tenors[np.argmin(np.abs(tenors - target_T_days))]
    slice_df   = df_clean[df_clean["T_days"] == nearest_T].sort_values("moneyness")
    if len(slice_df) < 2:
        return None
    return float(np.interp(
        target_moneyness,
        slice_df["moneyness"].values,
        slice_df[iv_col].values
    ))

def estimate_xdc_iv(df_btc, df_eth, target_moneyness, target_T_days):
    """
    Estimate XDC implied vol from BTC/ETH surfaces + micro-cap premium.
    Returns (btc_iv, eth_iv, xdc_iv_estimate).
    """
    btc_iv = interpolate_iv_at(df_btc, target_moneyness, target_T_days)
    eth_iv = interpolate_iv_at(df_eth, target_moneyness, target_T_days)
    available = [v for v in [btc_iv, eth_iv] if v is not None]
    if not available:
        return None, None, None
    proxy_iv   = np.mean(available)
    wing_factor = 1.0 + 0.4 * abs(target_moneyness - 1.0) * 2
    xdc_iv_est  = proxy_iv * 1.5 * wing_factor   # 1.5x liquidity premium + wing steepening
    return btc_iv, eth_iv, xdc_iv_est

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊  PRICE MATRIX",
    "📈  PAYOFF CHARTS",
    "⚙️  GREEKS",
    "🏗️  STRUCTURES",
    "🧠  MARKET DYNAMICS",
    "📡  DATA SOURCES",
    "🎯  CUSTOM PRICER",
    "🔬  REVERSE ENGINEER",
    "🌐  VOL SURFACE",
    "🧬  STRATEGY LAB",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 — PRICE MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    cols = st.columns(3)
    
    with cols[0]:
        atm_idx = 3
        st.markdown(f'''
        <div class="metric-card call">
            <div class="metric-label">ATM CALL</div>
            <div class="metric-value">${call_prices[atm_idx]:.5f}</div>
            <div class="metric-sub">{call_prices[atm_idx]/spot*100:.2f}% of spot &nbsp;|&nbsp; {notional*call_prices[atm_idx]/spot:,.0f} USD notional premium</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f'''
        <div class="metric-card put">
            <div class="metric-label">ATM PUT</div>
            <div class="metric-value">${put_prices[atm_idx]:.5f}</div>
            <div class="metric-sub">{put_prices[atm_idx]/spot*100:.2f}% of spot &nbsp;|&nbsp; {notional*put_prices[atm_idx]/spot:,.0f} USD notional premium</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        pcr = put_prices[atm_idx] / call_prices[atm_idx] if call_prices[atm_idx] > 0 else 0
        st.markdown(f'''
        <div class="metric-card neutral">
            <div class="metric-label">PUT/CALL RATIO (ATM)</div>
            <div class="metric-value">{pcr:.3f}</div>
            <div class="metric-sub">σ (effective) = {sigma:.1%} &nbsp;|&nbsp; {model_choice}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">FULL STRIKE GRID</div>', unsafe_allow_html=True)
    
    df = pd.DataFrame({
        'Strike Level': strike_labels,
        'Strike ($)': [f"${k:.6f}" for k in strikes],
        'Call Price ($)': [f"${p:.6f}" for p in call_prices],
        'Call % Spot': [f"{p/spot*100:.3f}%" for p in call_prices],
        'Put Price ($)': [f"${p:.6f}" for p in put_prices],
        'Put % Spot': [f"{p/spot*100:.3f}%" for p in put_prices],
        'P/C Ratio': [f"{p/c:.3f}" if c > 0.0001 else "∞" for p, c in zip(put_prices, call_prices)]
    })
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('<div class="section-header" style="margin-top:24px;">NOTIONAL PREMIUM BREAKDOWN</div>', unsafe_allow_html=True)
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Call Premium', x=strike_labels,
        y=[p * notional / spot for p in call_prices],
        marker_color='#00d4ff', marker_line_width=0, opacity=0.85
    ))
    fig_bar.add_trace(go.Bar(
        name='Put Premium', x=strike_labels,
        y=[p * notional / spot for p in put_prices],
        marker_color='#ff4b6e', marker_line_width=0, opacity=0.85
    ))
    fig_bar.update_layout(
        barmode='group', plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=11),
        xaxis=dict(gridcolor=_PLT_GRID, title='Strike Level'),
        yaxis=dict(gridcolor=_PLT_GRID, title='Premium (USD)'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        margin=dict(t=20, b=40),
        height=320
    )
    st.plotly_chart(fig_bar, use_container_width=True, key=_next_chart_key())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2 — PAYOFF CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    selected_strikes_idx = st.multiselect(
        "Select strikes to chart",
        options=list(range(len(strike_labels))),
        default=[0, 3, 6, 8],  # -30%, ATM, +30%, +50%
        format_func=lambda i: f"{strike_labels[i]} (${strikes[i]:.5f})"
    )
    
    if not selected_strikes_idx:
        selected_strikes_idx = [3]
    
    S_range = np.linspace(spot * 0.3, spot * 2.5, 500)
    
    fig_payoff = make_subplots(rows=1, cols=2, subplot_titles=["CALL PAYOFFS AT EXPIRY", "PUT PAYOFFS AT EXPIRY"])
    
    colors_call = ['#00d4ff', '#00e676', '#f0a500', '#b388ff', '#ff8a65']
    colors_put  = ['#ff4b6e', '#ff7043', '#ff8a65', '#ffb74d', '#ffd54f']
    
    for idx_pos, i in enumerate(selected_strikes_idx):
        K_sel = strikes[i]
        premium_c = call_prices[i]
        premium_p = put_prices[i]
        
        pnl_call = np.maximum(S_range - K_sel, 0) - premium_c
        pnl_put  = np.maximum(K_sel - S_range, 0) - premium_p
        
        fig_payoff.add_trace(go.Scatter(
            x=S_range, y=pnl_call, name=f"Call {strike_labels[i]}",
            line=dict(color=colors_call[idx_pos % 5], width=2),
            hovertemplate="Spot: $%{x:.5f}<br>P&L: $%{y:.5f}"
        ), row=1, col=1)
        
        fig_payoff.add_trace(go.Scatter(
            x=S_range, y=pnl_put, name=f"Put {strike_labels[i]}",
            line=dict(color=colors_put[idx_pos % 5], width=2),
            hovertemplate="Spot: $%{x:.5f}<br>P&L: $%{y:.5f}"
        ), row=1, col=2)
    
    for col in [1, 2]:
        fig_payoff.add_hline(y=0, line_dash="dot", line_color="#3d6080", row=1, col=col)
        fig_payoff.add_vline(x=spot, line_dash="dash", line_color="#f0a500",
                             annotation_text="SPOT", annotation_font_color="#f0a500", row=1, col=col)
    
    fig_payoff.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='Spot at Expiry'),
        yaxis=dict(gridcolor=_PLT_GRID, title='P&L ($)'),
        xaxis2=dict(gridcolor=_PLT_GRID, title='Spot at Expiry'),
        yaxis2=dict(gridcolor=_PLT_GRID, title='P&L ($)'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        height=440,
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_payoff, use_container_width=True, key=_next_chart_key())

    # Vol smile — scrollable container to avoid oversized chart at top bracket
    st.markdown('<div class="section-header">IMPLIED VOL SMILE (OTC PRICING)</div>', unsafe_allow_html=True)

    # Expanded strike range for higher-resolution smile
    smile_offsets = np.arange(0.50, 1.81, 0.05)
    smile_labels  = [f"{(o-1)*100:+.0f}%" for o in smile_offsets]
    iv_smile = [sigma * (1 + 0.08 * abs(o - 1)**1.2 + 0.04 * (o - 1)) for o in smile_offsets]

    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=smile_labels, y=[v * 100 for v in iv_smile],
        line=dict(color='#00d4ff', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
        name='OTC IV',
        hovertemplate="Strike: %{x}<br>IV: %{y:.1f}%<extra></extra>"
    ))
    fig_smile.add_hline(y=hv * 100, line_dash="dot", line_color="#f0a500",
                         annotation_text=f"HV {hv:.0%}", annotation_font_color="#f0a500")
    fig_smile.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=11),
        xaxis=dict(gridcolor=_PLT_GRID, title='Strike (% from ATM)',
                   rangeslider=dict(visible=True, thickness=0.08)),
        yaxis=dict(gridcolor=_PLT_GRID, title='Implied Vol (%)',
                   fixedrange=False),
        height=360, margin=dict(t=10, b=40),
        dragmode='pan',
    )
    config_smile = {'scrollZoom': True, 'displayModeBar': True}
    st.plotly_chart(fig_smile, use_container_width=True, key=_next_chart_key(),
                    config=config_smile)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3 — GREEKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    greek_df = pd.DataFrame([
        {
            'Strike': strike_labels[i],
            'Strike ($)': f"${strikes[i]:.6f}",
            'C Delta': call_greeks_list[i]['delta'],
            'C Gamma': call_greeks_list[i]['gamma'],
            'C Theta': call_greeks_list[i]['theta'],
            'C Vega':  call_greeks_list[i]['vega'],
            'P Delta': put_greeks_list[i]['delta'],
            'P Gamma': put_greeks_list[i]['gamma'],
            'P Theta': put_greeks_list[i]['theta'],
            'P Vega':  put_greeks_list[i]['vega'],
        }
        for i in range(len(strikes))
    ])
    st.dataframe(greek_df, use_container_width=True, hide_index=True)

    fig_greeks = make_subplots(rows=2, cols=2,
        subplot_titles=["DELTA", "GAMMA", "THETA (daily)", "VEGA (per 1% vol)"])
    
    greek_keys = ['delta', 'gamma', 'theta', 'vega']
    positions = [(1,1),(1,2),(2,1),(2,2)]
    
    for (row, col), gk in zip(positions, greek_keys):
        fig_greeks.add_trace(go.Scatter(
            x=strike_labels,
            y=[g[gk] for g in call_greeks_list],
            name=f'Call {gk}', line=dict(color='#00d4ff', width=2)
        ), row=row, col=col)
        fig_greeks.add_trace(go.Scatter(
            x=strike_labels,
            y=[g[gk] for g in put_greeks_list],
            name=f'Put {gk}', line=dict(color='#ff4b6e', width=2)
        ), row=row, col=col)
    
    fig_greeks.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        height=500, showlegend=False,
        margin=dict(t=40, b=20)
    )
    for i in range(1, 5):
        fig_greeks.update_xaxes(gridcolor=_PLT_GRID, row=(i-1)//2+1, col=(i-1)%2+1)
        fig_greeks.update_yaxes(gridcolor=_PLT_GRID, row=(i-1)//2+1, col=(i-1)%2+1)
    
    st.plotly_chart(fig_greeks, use_container_width=True, key=_next_chart_key())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4 — STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    recs = recommend_structure(hv, spot, call_prices, put_prices, strikes, T)

    st.markdown('<div class="section-header">STRATEGY RECOMMENDATION ENGINE</div>', unsafe_allow_html=True)

    for rec in recs:
        st.markdown(f'''
        <div class="recommendation-box" style="border-left-color:{rec["color"]}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div style="font-family:Space Mono;font-size:15px;color:{rec["color"]}">{rec["name"]}</div>
                <div style="font-size:11px;color:#3d6080;">SCORE: <span style="color:{rec["color"]}">{rec["score"]}/100</span></div>
            </div>
            <div style="font-size:12px;color:#8fb3d0;margin-bottom:6px;">{rec["rationale"]}</div>
            <div style="font-size:11px;color:#5a7a99;">⚠ Risk: {rec["risk"]}</div>
        </div>
        ''', unsafe_allow_html=True)

    # ── SNOWBALL AUTOCALLABLE — parameters now in Structures tab
    st.markdown('<div class="section-header" style="margin-top:28px;">SNOWBALL AUTOCALLABLE</div>', unsafe_allow_html=True)

    sb_param_cols = st.columns(3)
    with sb_param_cols[0]:
        barrier_pct = st.slider("Snowball Barrier (%)", 80, 130, 100, key="sb_barrier")
    with sb_param_cols[1]:
        coupon_rate = st.slider("Coupon / Period (%)", 1.0, 20.0, 8.0, 0.5, key="sb_coupon") / 100
    with sb_param_cols[2]:
        T_periods = st.slider("Snowball Periods", 1, 12, 4, key="sb_periods")

    barrier = spot * barrier_pct / 100

    snowball_cols = st.columns(3)
    with snowball_cols[0]:
        st.markdown(f'''<div class="metric-card neutral">
            <div class="metric-label">COUPON / PERIOD</div>
            <div class="metric-value">{coupon_rate*100:.1f}%</div>
            <div class="metric-sub">Max return if autocalled period 1: {coupon_rate*100:.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with snowball_cols[1]:
        max_coupon = coupon_rate * T_periods * 100
        st.markdown(f'''<div class="metric-card call">
            <div class="metric-label">MAX COUPON ({T_periods}P)</div>
            <div class="metric-value">{max_coupon:.1f}%</div>
            <div class="metric-sub">Annualised: {max_coupon/(T_days/365):.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with snowball_cols[2]:
        put_k = strikes[3]  # ATM put exposure
        worst_case = put_prices[3] / spot * 100
        st.markdown(f'''<div class="metric-card put">
            <div class="metric-label">ATM PUT EXPOSURE</div>
            <div class="metric-value">{worst_case:.2f}%</div>
            <div class="metric-sub">Cost of embedded put = max downside</div>
        </div>''', unsafe_allow_html=True)

    # Snowball MC simulation — 10,000 paths
    st.markdown("**Monte Carlo Snowball Outcomes (10,000 paths)**")
    np.random.seed(42)
    n_sims_sb = 10_000
    outcomes = []
    period_len = T / T_periods

    for _ in range(n_sims_sb):
        path_spots = [spot]
        for p in range(T_periods):
            z = np.random.normal()
            next_s = path_spots[-1] * np.exp((r - 0.5*sigma**2)*period_len + sigma*np.sqrt(period_len)*z)
            path_spots.append(next_s)
        payout = snowball_payoff(path_spots[1:], spot, barrier, coupon_rate, T_periods)
        outcomes.append(payout * 100)

    outcomes = np.array(outcomes)

    fig_sb = go.Figure()
    fig_sb.add_trace(go.Histogram(
        x=outcomes, nbinsx=60,
        marker_color='#00d4ff', marker_line_width=0,
        opacity=0.7, name='Snowball P&L'
    ))
    fig_sb.add_vline(x=0, line_color='#ff4b6e', line_dash='dash', annotation_text='Breakeven')
    fig_sb.add_vline(x=np.mean(outcomes), line_color='#00e676', annotation_text=f'Mean: {np.mean(outcomes):.1f}%')
    fig_sb.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='Return (%)'),
        yaxis=dict(gridcolor=_PLT_GRID, title='Frequency'),
        height=300, margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_sb, use_container_width=True, key=_next_chart_key())

    prob_pos = np.mean(outcomes > 0) * 100
    st.markdown(f'''
    <div class="warning-box">
        Snowball stats: Mean return = <strong>{np.mean(outcomes):.1f}%</strong>
        &nbsp;|&nbsp; P(positive) = <strong>{prob_pos:.0f}%</strong>
        &nbsp;|&nbsp; Worst 5% = <strong>{np.percentile(outcomes, 5):.1f}%</strong>
        &nbsp;|&nbsp; Best 5% = <strong>{np.percentile(outcomes, 95):.1f}%</strong>
    </div>
    ''', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  MONTE CARLO SNOWBALL STRUCTURE EXPLORER
    #  Simulates different barrier/coupon/period combinations
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header" style="margin-top:28px;">SNOWBALL STRUCTURE EXPLORER — MONTE CARLO COMPARISON</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:14px;line-height:1.8;">'
        'Compare snowball autocallable structures with different barriers, coupons, and periods. '
        'Each configuration runs 10,000 Monte Carlo paths to compute P&L statistics.</div>',
        unsafe_allow_html=True
    )

    # Configuration controls
    mc_exp_cols = st.columns(3)
    with mc_exp_cols[0]:
        barriers_input = st.multiselect(
            "Barriers to test (%)",
            options=[80, 85, 90, 95, 100, 105, 110, 120, 130],
            default=[85, 95, 100, 110],
            key="mc_barriers"
        )
    with mc_exp_cols[1]:
        coupons_input = st.multiselect(
            "Coupons to test (%/period)",
            options=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
            default=[4.0, 8.0, 12.0],
            key="mc_coupons"
        )
    with mc_exp_cols[2]:
        periods_input = st.multiselect(
            "Periods to test",
            options=[2, 3, 4, 6, 8, 12],
            default=[3, 4, 6],
            key="mc_periods"
        )

    if barriers_input and coupons_input and periods_input:
        mc_explorer_results = []
        np.random.seed(123)
        n_mc_exp = 10_000

        with st.spinner(f"Running {len(barriers_input)*len(coupons_input)*len(periods_input)} snowball structures × {n_mc_exp:,} paths..."):
            for b_pct in barriers_input:
                for c_pct in coupons_input:
                    for n_p in periods_input:
                        b_level = spot * b_pct / 100
                        c_rate  = c_pct / 100
                        pl_len  = T / n_p
                        outs = []
                        for _ in range(n_mc_exp):
                            path = [spot]
                            for _ in range(n_p):
                                z = np.random.normal()
                                ns = path[-1] * np.exp((r - 0.5*sigma**2)*pl_len + sigma*np.sqrt(pl_len)*z)
                                path.append(ns)
                            pay = snowball_payoff(path[1:], spot, b_level, c_rate, n_p)
                            outs.append(pay * 100)
                        outs = np.array(outs)
                        mc_explorer_results.append({
                            'Barrier (%)':   b_pct,
                            'Coupon (%/P)':  c_pct,
                            'Periods':       n_p,
                            'Mean Return':   f"{np.mean(outs):.1f}%",
                            'P(Positive)':   f"{np.mean(outs > 0)*100:.0f}%",
                            'Worst 5%':      f"{np.percentile(outs, 5):.1f}%",
                            'Best 5%':       f"{np.percentile(outs, 95):.1f}%",
                            'Sharpe-like':   f"{np.mean(outs)/max(np.std(outs),0.01):.2f}",
                            'P(Autocall)':   f"{np.mean(outs >= c_pct)*100:.0f}%",
                            '_mean_raw':     np.mean(outs),
                        })

        mc_exp_df = pd.DataFrame(mc_explorer_results)

        # Results table
        mc_exp_df = mc_exp_df.sort_values('_mean_raw', ascending=False)
        display_cols_mc = [c for c in mc_exp_df.columns if not c.startswith('_')]
        st.dataframe(mc_exp_df[display_cols_mc],
                      use_container_width=True, hide_index=True)

        # Heatmap of mean returns: barrier vs coupon
        if len(barriers_input) >= 2 and len(coupons_input) >= 2:
            # Use first period setting for heatmap
            p_hm = periods_input[0]
            hm_data = mc_exp_df[mc_exp_df['Periods'] == p_hm]
            if not hm_data.empty:
                pivot = hm_data.pivot_table(
                    values='_mean_raw', index='Barrier (%)', columns='Coupon (%/P)'
                )
                fig_hm = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[f"{c:.0f}%" for c in pivot.columns],
                    y=[f"{b}%" for b in pivot.index],
                    colorscale=[
                        [0.0, '#ff4b6e'], [0.3, '#f0a500'],
                        [0.5, '#1a2535'], [0.7, '#00d4ff'], [1.0, '#00e676']
                    ],
                    colorbar=dict(title='Mean Return %', tickfont=dict(color='#5a7a99')),
                    hovertemplate='Barrier: %{y}<br>Coupon: %{x}<br>Mean: %{z:.1f}%<extra></extra>'
                ))
                fig_hm.update_layout(
                    title=dict(text=f'Mean Return Heatmap ({p_hm} periods)', font=dict(size=11, color='#5a9abf')),
                    plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                    font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
                    xaxis=dict(title='Coupon / Period'), yaxis=dict(title='Barrier'),
                    height=320, margin=dict(t=40, b=40)
                )
                st.plotly_chart(fig_hm, use_container_width=True, key=_next_chart_key())

        # Best structure highlight
        best_idx = mc_exp_df['_mean_raw'].idxmax()
        best = mc_exp_df.iloc[best_idx]
        st.markdown(f'''
        <div class="recommendation-box" style="border-left-color:#00e676">
            <div style="font-family:Space Mono;font-size:14px;color:#00e676;margin-bottom:6px;">OPTIMAL SNOWBALL STRUCTURE</div>
            <div style="font-size:12px;color:#8fb3d0;">
                Barrier: <strong>{best["Barrier (%)"]:.0f}%</strong> ·
                Coupon: <strong>{best["Coupon (%/P)"]:.0f}%/period</strong> ·
                Periods: <strong>{best["Periods"]}</strong> ·
                Mean Return: <strong>{best["Mean Return"]}</strong> ·
                P(Positive): <strong>{best["P(Positive)"]}</strong>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.info("Select at least one barrier, coupon, and period to run the explorer.")
    
    # ── LADDER
    st.markdown('<div class="section-header" style="margin-top:24px;">CALL LADDER (UPSIDE LOCKS)</div>', unsafe_allow_html=True)
    
    rungs = [
        (spot * 1.30, 0.15),  # +30% → lock 15%
        (spot * 1.40, 0.25),  # +40% → lock 25%
        (spot * 1.50, 0.35),  # +50% → lock 35%
    ]
    
    S_ladder = np.linspace(spot * 0.5, spot * 2.0, 400)
    ladder_payoffs = [ladder_payoff(s, spot, rungs) * 100 for s in S_ladder]
    
    fig_lad = go.Figure()
    fig_lad.add_trace(go.Scatter(
        x=S_ladder, y=ladder_payoffs,
        line=dict(color='#00e676', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,230,118,0.05)',
        name='Ladder Payoff'
    ))
    
    # Vanilla call for comparison
    vanilla_call_payoff = [max(s - spot, 0) / spot * 100 for s in S_ladder]
    fig_lad.add_trace(go.Scatter(
        x=S_ladder, y=vanilla_call_payoff,
        line=dict(color='#3d6080', width=1.5, dash='dot'),
        name='Vanilla Call'
    ))
    
    for level, locked in rungs:
        fig_lad.add_vline(x=level, line_color='#f0a500', line_dash='dash',
                           annotation_text=f"+{locked*100:.0f}% lock", annotation_font_color='#f0a500')
    
    fig_lad.add_vline(x=spot, line_color=_PLT_TXT, line_dash='dot', annotation_text='SPOT')
    
    fig_lad.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='Spot at Expiry'),
        yaxis=dict(gridcolor=_PLT_GRID, title='Return (%)'),
        legend=dict(bgcolor=_PLT_LEG),
        height=320, margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig_lad, use_container_width=True, key=_next_chart_key())
    
    # Ladder premium cost
    cost_30c = call_prices[6]  # +30% call
    cost_40c = call_prices[7]  # +40% call
    cost_50c = call_prices[8]  # +50% call
    ladder_cost = cost_30c - cost_40c + cost_40c - cost_50c  # simplified
    st.markdown(f'''
    <div class="warning-box">
        Ladder indicative cost (call spread 30%→50%): <strong>${cost_30c - cost_50c:.6f}</strong> per unit 
        &nbsp;|&nbsp; USD: <strong>${(cost_30c - cost_50c) * notional / spot:,.0f}</strong>
    </div>
    ''', unsafe_allow_html=True)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 5 — MARKET DYNAMICS & PROBABILITY DISTRIBUTIONS (v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown(f'<div class="section-header">{st.session_state.get("active_token_ticker","XDC")} 30-DAY PROBABILITY ENGINE — SIX INDEPENDENT MODELS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:16px;line-height:1.8;">'
        'Six models — three historical (GBM, Student-t, GARCH+Skew) and three market-proxy '
        '(Regime+FearGreed, BTC β, ETH β) — are fitted independently and blended. '
        f'Conditional distributions show {st.session_state.get("active_token_ticker","XDC")} behaviour under different BTC scenarios. '
        'Never rely on a single model.</div>',
        unsafe_allow_html=True
    )

    # ── DATA LOADING ──
    with st.spinner(f"Loading {st.session_state.get('active_token_ticker','XDC')} 180d + BTC + ETH + Fear & Greed..."):
        _cid_md = st.session_state.get("active_coin_id", "xdc-network")
        _cid_md = _cid_md if _cid_md != "custom" else "xdc-network"
        _ticker_md = st.session_state.get("active_token_ticker", "XDC")
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{_cid_md}/market_chart?vs_currency=usd&days=180&interval=daily"
            r = _cg_get(url)
            _mddata = r.json()
            xdc_prices_ext = [p[1] for p in _mddata.get("prices", [])]
            xdc_volumes    = [p[1] for p in _mddata.get("total_volumes", [])]
            if len(xdc_prices_ext) < 5:
                xdc_prices_ext, xdc_volumes = None, None
        except Exception:
            xdc_prices_ext, xdc_volumes = None, None
        # CMC fallback for 180d history
        if xdc_prices_ext is None:
            _cmc_sym = _CG_TO_CMC_SLUG.get(_cid_md, _ticker_md)
            _cmc_h = _cmc_history(_cmc_sym, 180)
            if _cmc_h and len(_cmc_h) > 5:
                xdc_prices_ext = _cmc_h
                xdc_volumes = [1e6] * len(_cmc_h)
        btc_prices  = fetch_btc_history(90)
        eth_prices  = fetch_eth_history(90)
        _cid_mkd = st.session_state.get("active_coin_id", "xdc-network")
        _cid_mkd = _cid_mkd if _cid_mkd != "custom" else "xdc-network"
        try:
            _mkdurl = f"https://api.coingecko.com/api/v3/coins/{_cid_mkd}?localization=false&tickers=false&community_data=true&developer_data=false"
            _mkdr   = _cg_get(_mkdurl)
            _mkdd   = _mkdr.json()
            _mkdmd  = _mkdd.get("market_data", {})
            mkt_data = {
                "market_cap":     _mkdmd.get("market_cap", {}).get("usd"),
                "volume_24h":     _mkdmd.get("total_volume", {}).get("usd"),
                "price_change_7d":  _mkdmd.get("price_change_percentage_7d"),
                "price_change_30d": _mkdmd.get("price_change_percentage_30d"),
                "ath":            _mkdmd.get("ath", {}).get("usd"),
                "ath_change_pct": _mkdmd.get("ath_change_percentage", {}).get("usd"),
                "sentiment_up":   _mkdd.get("sentiment_votes_up_percentage"),
            }
        except Exception:
            mkt_data = {}
        fg_data     = fetch_fear_greed(30)

    data_ok = xdc_prices_ext is not None and len(xdc_prices_ext) > 30

    if not data_ok:
        st.warning("Live data unavailable — using synthetic demo data")
        np.random.seed(1)
        xdc_prices_ext = list(spot * np.cumprod(np.exp(np.random.normal(0.0005, 0.04, 180))))
        xdc_volumes    = [1e6] * 180
        btc_prices     = list(60000 * np.cumprod(np.exp(np.random.normal(0.0008, 0.025, 90))))
        eth_prices     = list(3000  * np.cumprod(np.exp(np.random.normal(0.0006, 0.030, 90))))

    log_rets_xdc = compute_log_returns(xdc_prices_ext)
    mu_hist, sigma_hist_dyn = fit_gbm(log_rets_xdc)

    # BTC stats
    if btc_prices and len(btc_prices) > 10:
        log_rets_btc = compute_log_returns(btc_prices)
        mu_btc, _    = fit_gbm(log_rets_btc)
        btc_beta, btc_corr = compute_btc_xdc_beta(xdc_prices_ext, btc_prices)
    else:
        log_rets_btc = None
        mu_btc, btc_beta, btc_corr = 0.20, 1.20, 0.65

    # ETH stats (second independent proxy)
    if eth_prices and len(eth_prices) > 10:
        log_rets_eth = compute_log_returns(eth_prices)
        mu_eth, _    = fit_gbm(log_rets_eth)
        eth_beta, eth_corr = compute_btc_xdc_beta(xdc_prices_ext, eth_prices)
    else:
        log_rets_eth = None
        mu_eth, eth_beta, eth_corr = 0.18, 1.10, 0.55

    # Fear & Greed
    fg_current = None
    fg_label   = "N/A"
    fg_trend   = None
    if fg_data and len(fg_data) > 0:
        fg_current = fg_data[0][1]   # most recent
        fg_label   = fg_data[0][2]
        fg_7d_avg  = np.mean([x[1] for x in fg_data[:7]])
        fg_trend   = "↑ Improving" if fg_current > fg_7d_avg else "↓ Deteriorating"

    # GARCH(1,1) vol model
    garch_result = fit_garch11(log_rets_xdc, horizon=30)

    # Regime detection
    regime, regime_conf, regime_stats, regime_color = detect_regime(xdc_prices_ext)

    # ─── REGIME + FEAR/GREED BANNER ───
    fg_color = (
        '#ff4b6e' if fg_current and fg_current < 25 else
        '#f0a500' if fg_current and fg_current < 50 else
        '#b3ff00' if fg_current and fg_current < 75 else '#00e676'
    ) if fg_current else '#5a7a99'

    banner_cols = st.columns([3, 1])
    with banner_cols[0]:
        st.markdown(f'''
        <div style="background:linear-gradient(135deg,#0d1a14,#091008);border:1px solid #1a3020;
                    border-left:4px solid {regime_color};border-radius:4px;padding:16px 22px;">
            <div style="font-size:10px;letter-spacing:3px;color:#3d6080;text-transform:uppercase;margin-bottom:4px;">Market Regime</div>
            <div style="font-family:Space Mono;font-size:20px;color:{regime_color};margin-bottom:6px;">{regime}</div>
            <div style="font-size:11px;color:#5a7a99;line-height:1.9;">
                Confidence: <span style="color:{regime_color}">{regime_conf:.0%}</span>
                &nbsp;|&nbsp; HV(180d): <span style="color:#f0a500">{sigma_hist_dyn:.0%}</span>
                &nbsp;|&nbsp; GARCH Current Vol: <span style="color:#b388ff">{garch_result["current_vol"]:.0%}</span>
                &nbsp;|&nbsp; GARCH LR Vol: <span style="color:#b388ff">{garch_result["long_run_vol"]:.0%}</span>
                &nbsp;|&nbsp; RSI(14): <span style="color:#b388ff">{regime_stats.get("rsi", 50):.0f}</span><br>
                BTC β: <span style="color:#00d4ff">{btc_beta:.2f}</span>
                &nbsp;|&nbsp; BTC ρ: <span style="color:#00d4ff">{btc_corr:.2f}</span>
                &nbsp;|&nbsp; ETH β: <span style="color:#f0a500">{eth_beta:.2f}</span>
                &nbsp;|&nbsp; ETH ρ: <span style="color:#f0a500">{eth_corr:.2f}</span>
                &nbsp;|&nbsp; GARCH Persist: <span style="color:#b388ff">{garch_result["persistence"]:.3f}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with banner_cols[1]:
        fg_display = fg_current if fg_current is not None else "N/A"
        fg_display_label = fg_label if fg_current is not None else "API Unavailable"
        st.markdown(f'''
        <div style="background:#0d1117;border:1px solid {fg_color};border-radius:4px;
                    padding:16px;text-align:center;height:100%;">
            <div style="font-size:10px;letter-spacing:2px;color:#3d6080;margin-bottom:8px;">FEAR & GREED</div>
            <div style="font-family:Space Mono;font-size:36px;color:{fg_color};">{fg_display}</div>
            <div style="font-size:11px;color:{fg_color};margin-top:4px;">{fg_display_label}</div>
            <div style="font-size:10px;color:#3d6080;margin-top:6px;">{fg_trend or "Source: alternative.me"}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Fear & Greed 30-day trend chart
    if fg_data and len(fg_data) > 1:
        fg_dates = [datetime.fromtimestamp(e[0]) for e in fg_data]
        fg_vals  = [e[1] for e in fg_data]
        fig_fg = go.Figure()
        fig_fg.add_trace(go.Scatter(
            x=fg_dates, y=fg_vals, name='Fear & Greed',
            line=dict(width=2),
            marker=dict(
                color=[
                    '#ff4b6e' if v < 25 else '#f0a500' if v < 50 else '#b3ff00' if v < 75 else '#00e676'
                    for v in fg_vals
                ],
                size=6
            ),
            mode='lines+markers',
            fill='tozeroy', fillcolor='rgba(0,212,255,0.03)',
            hovertemplate='Date: %{x}<br>Score: %{y}<extra></extra>'
        ))
        fig_fg.add_hline(y=50, line_color='#3d6080', line_dash='dot')
        fig_fg.add_hline(y=25, line_color='#ff4b6e', line_dash='dot',
                          annotation_text='Extreme Fear', annotation_font_size=9)
        fig_fg.add_hline(y=75, line_color='#00e676', line_dash='dot',
                          annotation_text='Extreme Greed', annotation_font_size=9)
        fig_fg.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
            xaxis=dict(gridcolor=_PLT_GRID), yaxis=dict(gridcolor=_PLT_GRID, title='F&G Score', range=[0, 100]),
            height=200, margin=dict(t=10, b=30), showlegend=False,
            title=dict(text='Fear & Greed — 30 Day Trend', font=dict(size=10, color='#5a9abf'))
        )
        st.plotly_chart(fig_fg, use_container_width=True, key=_next_chart_key())

    # ── GARCH VOL FORECAST ──
    st.markdown('<div class="section-header" style="margin-top:20px;">GARCH(1,1) VOL FORECAST — NEXT 30 DAYS</div>', unsafe_allow_html=True)

    fig_garch = make_subplots(rows=1, cols=2, 
        subplot_titles=["CONDITIONAL VOL FORECAST (annualised)", "GARCH σ² TIME SERIES"])

    # Forward vol forecast
    forecast_days = list(range(1, 31))
    fig_garch.add_trace(go.Scatter(
        x=forecast_days, y=[v * 100 for v in garch_result['forecasts']],
        name='GARCH Forecast', line=dict(color='#b388ff', width=2.5)
    ), row=1, col=1)
    fig_garch.add_hline(y=garch_result['long_run_vol'] * 100, line_color='#f0a500', 
                         line_dash='dash', annotation_text='Long-run vol', row=1, col=1)
    fig_garch.add_hline(y=sigma_hist_dyn * 100, line_color='#5a9abf', 
                         line_dash='dot', annotation_text='HV baseline', row=1, col=1)

    # Historical conditional variance
    sig_series = np.sqrt(garch_result['sigma2_series'] * 365) * 100
    fig_garch.add_trace(go.Scatter(
        x=list(range(len(sig_series))), y=sig_series,
        name='Historical σ', line=dict(color='#5a9abf', width=1.5), opacity=0.7
    ), row=1, col=2)

    fig_garch.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        height=280, showlegend=False, margin=dict(t=40, b=40)
    )
    for c in [1, 2]:
        fig_garch.update_xaxes(gridcolor=_PLT_GRID, row=1, col=c)
        fig_garch.update_yaxes(gridcolor=_PLT_GRID, title_text='Vol (%)', row=1, col=c)
    st.plotly_chart(fig_garch, use_container_width=True, key=_next_chart_key())

    # ── MODEL WEIGHTS ──
    with st.expander("⚙️  Adjust Model Weights & Parameters"):
        wcols = st.columns(6)
        w1 = wcols[0].slider("Hist GBM",    0.0, 1.0, 0.15, 0.05, key='w1')
        w2 = wcols[1].slider("Regime+F&G",  0.0, 1.0, 0.20, 0.05, key='w2')
        w3 = wcols[2].slider("Student-t",   0.0, 1.0, 0.20, 0.05, key='w3')
        w4 = wcols[3].slider("BTC Proxy",   0.0, 1.0, 0.20, 0.05, key='w4')
        w5 = wcols[4].slider("ETH Proxy",   0.0, 1.0, 0.15, 0.05, key='w5')
        w6 = wcols[5].slider("GARCH+Skew",  0.0, 1.0, 0.10, 0.05, key='w6')
        total_w6 = w1 + w2 + w3 + w4 + w5 + w6
        if abs(total_w6 - 1.0) > 0.01:
            st.info(f"Weights sum to {total_w6:.2f} — auto-normalised")
        raw_w = [w1, w2, w3, w4, w5, w6]
        weights = [w / total_w6 for w in raw_w]

    # ── RUN 6 DISTRIBUTIONS ──
    with st.spinner("Running 100k simulations across 6 models..."):
        distributions = project_distributions_30d_enhanced(
            spot, log_rets_xdc, mu_hist, sigma_hist_dyn,
            regime, regime_conf,
            btc_beta, mu_btc, eth_beta, mu_eth,
            fg_score=fg_current,
            garch_result=garch_result,
            n_sims=100_000
        )
        ensemble = ensemble_distribution(distributions, weights)
        distributions['Ensemble (Blended)'] = ensemble
        scenarios  = compute_price_scenarios(spot, distributions)
        conviction = compute_model_conviction(
            {k: v for k, v in distributions.items() if k != 'Ensemble (Blended)'}, spot
        )

    model_colors = {
        'Historical GBM':        '#5a9abf',
        'Regime + Fear/Greed':   regime_color,
        'Fat-Tail (Student-t)':  '#b388ff',
        'BTC Market Proxy':      '#f0a500',
        'ETH Market Proxy':      '#ff8a65',
        'GARCH(1,1) + Skew':     '#e040fb',
        'Ensemble (Blended)':    '#00d4ff',
    }

    # ── CONVICTION METER ──
    st.markdown('<div class="section-header">MODEL CONVICTION DASHBOARD</div>', unsafe_allow_html=True)

    conv_cols = st.columns(5)
    conv_c = conviction
    with conv_cols[0]:
        st.markdown(f'''<div class="metric-card" style="border-left-color:{conviction["dir_color"]}">
            <div class="metric-label">DIRECTION</div>
            <div class="metric-value" style="font-size:20px;color:{conviction["dir_color"]};">{conviction["direction"]}</div>
            <div class="metric-sub">Conviction: {conviction["conviction"]:.0f}/100</div>
        </div>''', unsafe_allow_html=True)
    with conv_cols[1]:
        st.markdown(f'''<div class="metric-card" style="border-left-color:#00e676">
            <div class="metric-label">BULL MODELS</div>
            <div class="metric-value" style="font-size:20px;color:#00e676;">{conviction["bull_models"]}/6</div>
            <div class="metric-sub">Median &gt; spot +5%</div>
        </div>''', unsafe_allow_html=True)
    with conv_cols[2]:
        st.markdown(f'''<div class="metric-card" style="border-left-color:#ff4b6e">
            <div class="metric-label">BEAR MODELS</div>
            <div class="metric-value" style="font-size:20px;color:#ff4b6e;">{conviction["bear_models"]}/6</div>
            <div class="metric-sub">Median &lt; spot -5%</div>
        </div>''', unsafe_allow_html=True)
    with conv_cols[3]:
        st.markdown(f'''<div class="metric-card" style="border-left-color:#f0a500">
            <div class="metric-label">MEDIAN SPREAD</div>
            <div class="metric-value" style="font-size:20px;color:#f0a500;">{conviction["median_spread_pct"]:.1f}%</div>
            <div class="metric-sub">Range of model medians</div>
        </div>''', unsafe_allow_html=True)
    with conv_cols[4]:
        fg_str = f"{fg_current:.0f} ({fg_label})" if fg_current else "N/A"
        st.markdown(f'''<div class="metric-card" style="border-left-color:{fg_color}">
            <div class="metric-label">FEAR & GREED</div>
            <div class="metric-value" style="font-size:20px;color:{fg_color};">{fg_str}</div>
            <div class="metric-sub">{fg_trend or "alternative.me index"}</div>
        </div>''', unsafe_allow_html=True)

    # Model medians bar chart
    med_names = list(conviction['medians'].keys())
    med_vals  = [(v/spot - 1)*100 for v in conviction['medians'].values()]
    med_colors_bar = [model_colors.get(n, '#888') for n in med_names]
    fig_conv = go.Figure(go.Bar(
        x=med_names, y=med_vals,
        marker_color=med_colors_bar, marker_line_width=0,
        text=[f"{v:+.1f}%" for v in med_vals],
        textposition='outside', textfont=dict(size=10, color=_PLT_TXT)
    ))
    fig_conv.add_hline(y=0, line_color='#3d6080', line_dash='dot')
    fig_conv.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID),
        yaxis=dict(gridcolor=_PLT_GRID, title='30d Median Return (%)'),
        height=260, margin=dict(t=30, b=60),
        title=dict(text='Model Median 30-Day Return Forecasts', font=dict(size=11, color='#5a9abf'))
    )
    st.plotly_chart(fig_conv, use_container_width=True, key=_next_chart_key())

    # ── MAIN PDF CHART ──
    st.markdown('<div class="section-header">30-DAY PRICE PROBABILITY DENSITY — ALL MODELS</div>', unsafe_allow_html=True)

    fig_dist = go.Figure()
    price_range = np.linspace(spot * 0.10, spot * 3.5, 600)

    for model_name, sims in distributions.items():
        try:
            kde = gaussian_kde(sims, bw_method='scott')
            density = kde(price_range)
            is_ens = model_name == 'Ensemble (Blended)'
            fig_dist.add_trace(go.Scatter(
                x=price_range, y=density, name=model_name,
                line=dict(
                    color=model_colors.get(model_name, '#aaa'),
                    width=3.5 if is_ens else 1.5,
                    dash='solid' if is_ens else 'dot'
                ),
                fill='tozeroy' if is_ens else 'none',
                fillcolor='rgba(0,212,255,0.06)' if is_ens else None,
                opacity=1.0 if is_ens else 0.75,
                hovertemplate=f"<b>{model_name}</b><br>Price: $%{{x:.5f}}<br>Density: %{{y:.4f}}<extra></extra>"
            ))
        except Exception:
            pass

    for label, price, color in [
        ("SPOT", spot, '#ffffff'), ("+50%", spot*1.50, '#00e676'),
        ("+30%", spot*1.30, '#44cc77'), ("-30%", spot*0.70, '#ff6688'),
        ("-50%", spot*0.50, '#ff4b6e'),
    ]:
        fig_dist.add_vline(x=price, line_color=color, line_dash='dash', line_width=1,
                           annotation_text=label, annotation_font_color=color, annotation_font_size=10)

    fig_dist.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='XDC Price at Day 30 ($)', tickformat='.5f'),
        yaxis=dict(gridcolor=_PLT_GRID, title='Probability Density'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR, font=dict(size=10)),
        height=440, margin=dict(t=20, b=50), hovermode='x unified'
    )
    st.plotly_chart(fig_dist, use_container_width=True, key=_next_chart_key())

    # ── SCENARIO PROBABILITY TABLE ──
    st.markdown('<div class="section-header">SCENARIO PROBABILITY TABLE — 6 MODELS + ENSEMBLE</div>', unsafe_allow_html=True)

    scenario_keys   = ['P(>+50%)', 'P(>+30%)', 'P(>+10%)', 'P(flat±5%)', 'P(<-10%)', 'P(<-30%)', 'P(<-50%)']
    scenario_labels = ['+50% 🚀', '+30% 📈', '+10% ↗', '±5% ➡', '-10% ↘', '-30% 📉', '-50% 💥']

    rows = []
    for key, label in zip(scenario_keys, scenario_labels):
        row = {'Scenario': label}
        for model_name in distributions:
            val = scenarios[model_name][key]
            row[model_name] = f"{val:.1%}"
        rows.append(row)
    scen_df = pd.DataFrame(rows)
    st.dataframe(scen_df, use_container_width=True, hide_index=True)

    # ── ENSEMBLE SUMMARY CARDS ──
    st.markdown('<div class="section-header">ENSEMBLE DISTRIBUTION SUMMARY</div>', unsafe_allow_html=True)

    ens_sc = scenarios['Ensemble (Blended)']

    def stat_card(label, value, sub, color='#00d4ff'):
        return f'''<div class="metric-card" style="border-left-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:18px;">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>'''

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(stat_card("MEDIAN 30D", f"${ens_sc['Median']:.5f}",
            f"vs spot: {(ens_sc['Median']/spot-1)*100:+.1f}%", '#00d4ff'), unsafe_allow_html=True)
    with sc2:
        st.markdown(stat_card("P(BULL >+30%)", f"{ens_sc['P(>+30%)']:.1%}",
            f"P(>+50%): {ens_sc['P(>+50%)']:.1%}", '#00e676'), unsafe_allow_html=True)
    with sc3:
        st.markdown(stat_card("P(BEAR <-30%)", f"{ens_sc['P(<-30%)']:.1%}",
            f"P(<-50%): {ens_sc['P(<-50%)']:.1%}", '#ff4b6e'), unsafe_allow_html=True)
    with sc4:
        st.markdown(stat_card("90% BAND",
            f"${ens_sc['P5']:.4f}–${ens_sc['P95']:.4f}",
            f"IQR: ${ens_sc['P25']:.4f}–${ens_sc['P75']:.4f}", '#f0a500'), unsafe_allow_html=True)

    # ── FAN CHART ──
    st.markdown('<div class="section-header">ENSEMBLE FAN CHART — 30-DAY PATHS</div>', unsafe_allow_html=True)

    days_fwd = np.arange(1, 31)
    n_paths  = 8000
    np.random.seed(55)

    regime_drift_map = {
        "BULL TREND": 0.30, "BEAR TREND": -0.30, "HIGH VOL / CRISIS": -0.15,
        "LOW VOL / ACCUMULATION": 0.10, "SIDEWAYS / CONSOLIDATION": 0.0, "UNKNOWN": 0.0
    }
    daily_mu_fan    = (mu_hist * (1 - regime_conf) + regime_drift_map.get(regime, 0) * regime_conf) / 365
    # Use GARCH day-1 vol for fan chart
    garch_day1_vol  = garch_result['forecasts'][0] / np.sqrt(365)
    daily_sigma_fan = garch_day1_vol

    paths = np.zeros((n_paths, 30))
    Z_fan = np.where(
        np.random.random(n_paths) < 0.25,
        np.random.standard_t(df=4, size=n_paths),
        np.random.standard_normal(n_paths)
    )
    paths[:, 0] = spot * np.exp(daily_mu_fan + daily_sigma_fan * Z_fan)
    for d in range(1, 30):
        # GARCH vol decays toward long-run
        garch_vol_d = garch_result['forecasts'][d] / np.sqrt(365)
        Z_day = np.where(
            np.random.random(n_paths) < 0.25,
            np.random.standard_t(df=4, size=n_paths),
            np.random.standard_normal(n_paths)
        )
        paths[:, d] = paths[:, d-1] * np.exp(daily_mu_fan + garch_vol_d * Z_day)

    pcts = {pct: np.percentile(paths, pct, axis=0) for pct in [5, 10, 25, 50, 75, 90, 95]}

    _active_tok_fan = st.session_state.get("active_token_ticker", "XDC")
    _is_dark_fan    = st.session_state.get("dark_mode", True)
    _bg_fan  = "#0d1117" if _is_dark_fan else "#ffffff"
    _bg2_fan = "#0a0c10" if _is_dark_fan else "#f5f7fa"
    _grid_fan = "#1a2535" if _is_dark_fan else "#e0e6ed"
    _txt_fan  = "#5a7a99" if _is_dark_fan else "#444455"

    fig_fan = go.Figure()

    # ── All individual MC paths (thin, semi-transparent) ──
    # Colour-code by terminal return: green = above spot, red = below
    for i in range(n_paths):
        terminal = paths[i, -1]
        if terminal > spot * 1.10:
            path_col = 'rgba(0,230,118,0.08)'
        elif terminal < spot * 0.90:
            path_col = 'rgba(255,75,110,0.08)'
        else:
            path_col = 'rgba(0,212,255,0.05)'
        fig_fan.add_trace(go.Scatter(
            x=days_fwd, y=paths[i], mode='lines',
            line=dict(color=path_col, width=0.6),
            showlegend=False, hoverinfo='skip'
        ))

    # ── Percentile bands (subtle fills) ──
    for lo, hi, col, name in [
        (pcts[5],  pcts[95], 'rgba(0,212,255,0.03)', '5–95th pct'),
        (pcts[10], pcts[90], 'rgba(0,212,255,0.05)', '10–90th pct'),
        (pcts[25], pcts[75], 'rgba(0,212,255,0.09)', '25–75th pct'),
    ]:
        fig_fan.add_trace(go.Scatter(
            x=np.concatenate([days_fwd, days_fwd[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill='toself', fillcolor=col, line=dict(width=0), name=name,
            hoverinfo='skip'
        ))

    # ── Orbital median: large glowing markers on key days ──
    orbital_days  = [0, 5, 10, 15, 20, 25, 29]
    orbital_x     = [days_fwd[d] for d in orbital_days]
    orbital_y     = [pcts[50][d] for d in orbital_days]
    # Outer glow ring
    fig_fan.add_trace(go.Scatter(
        x=orbital_x, y=orbital_y,
        mode='markers', name='Median path',
        marker=dict(
            color='rgba(0,212,255,0.15)',
            size=22,
            line=dict(color='rgba(0,212,255,0.5)', width=2)
        ),
        showlegend=False, hoverinfo='skip'
    ))
    # Inner bright dot
    fig_fan.add_trace(go.Scatter(
        x=orbital_x, y=orbital_y,
        mode='markers+lines',
        name=f'Median ({_active_tok_fan})',
        marker=dict(color='#00d4ff', size=10,
                    line=dict(color='#ffffff', width=1.5)),
        line=dict(color='#00d4ff', width=2.5)
    ))

    # ── Reference levels ──
    for label, price, color in [
        ('+50%', spot*1.50, '#00e676'),
        ('+30%', spot*1.30, '#44cc77'),
        ('-30%', spot*0.70, '#ff6688'),
        ('Spot', spot,      '#ffffff'),
    ]:
        fig_fan.add_hline(y=price, line_color=color, line_dash='dash', line_width=1,
                          annotation_text=label, annotation_font_color=color,
                          annotation_font_size=10)

    fig_fan.update_layout(
        plot_bgcolor=_bg_fan, paper_bgcolor=_bg_fan,
        font=dict(family='IBM Plex Mono', color=_txt_fan, size=10),
        xaxis=dict(gridcolor=_grid_fan, title='Days Forward'),
        yaxis=dict(gridcolor=_grid_fan, title=f'{_active_tok_fan} Price ($)', tickformat='.5f'),
        legend=dict(bgcolor=_bg_fan, bordercolor=_grid_fan),
        height=480, margin=dict(t=20, b=50)
    )
    st.plotly_chart(fig_fan, use_container_width=True, key=_next_chart_key())

    # ── CONDITIONAL DISTRIBUTIONS (BTC SCENARIO ANALYSIS) ──
    st.markdown('<div class="section-header">CONDITIONAL DISTRIBUTIONS — BTC SCENARIO ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#5a7a99;margin-bottom:12px;">How XDC price distribution shifts under different BTC outcomes in 30 days. Based on empirical BTC↔XDC correlation.</div>', unsafe_allow_html=True)

    with st.spinner("Computing conditional distributions..."):
        cond_dists, rho_btc = compute_conditional_distributions(
            spot, log_rets_xdc,
            log_rets_btc if log_rets_btc is not None else None,
            log_rets_eth if log_rets_eth is not None else None,
            sigma_hist_dyn, n_sims=50_000
        )

    cond_colors = {
        'BTC Bull (+20%)':  '#00e676',
        'BTC Flat (±2%)':   '#5a9abf',
        'BTC Bear (-20%)':  '#f0a500',
        'BTC Crash (-40%)': '#ff4b6e',
    }

    fig_cond = go.Figure()
    for scenario_label, sims in cond_dists.items():
        try:
            kde = gaussian_kde(sims, bw_method='scott')
            density = kde(price_range)
            fig_cond.add_trace(go.Scatter(
                x=price_range, y=density,
                name=scenario_label,
                line=dict(color=cond_colors.get(scenario_label, '#888'), width=2.5),
                fill='tozeroy',
                fillcolor=f"rgba({','.join(str(int(cond_colors.get(scenario_label,'#888').lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.04)",
                hovertemplate=f"<b>{scenario_label}</b><br>${{x:.5f}}<extra></extra>"
            ))
        except Exception:
            pass

    fig_cond.add_vline(x=spot, line_color='#ffffff', line_dash='dash',
                       annotation_text='SPOT', annotation_font_color='#ffffff')

    fig_cond.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='XDC Price at Day 30 ($)', tickformat='.5f'),
        yaxis=dict(gridcolor=_PLT_GRID, title='Probability Density'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR, font=dict(size=11)),
        height=360, margin=dict(t=20, b=50),
        hovermode='x unified'
    )
    st.plotly_chart(fig_cond, use_container_width=True, key=_next_chart_key())

    # Conditional scenario probability table
    cond_rows = []
    for sc_label, sc_sims in cond_dists.items():
        cond_rows.append({
            'BTC Scenario': sc_label,
            f'P({st.session_state.get("active_token_ticker","TOKEN")} >+30%)': f"{np.mean(sc_sims > spot*1.30):.1%}",
            f'P({st.session_state.get("active_token_ticker","TOKEN")} >+10%)': f"{np.mean(sc_sims > spot*1.10):.1%}",
            f'P({st.session_state.get("active_token_ticker","TOKEN")} flat±5%)': f"{np.mean((sc_sims > spot*0.95) & (sc_sims < spot*1.05)):.1%}",
            f'P({st.session_state.get("active_token_ticker","TOKEN")} <-10%)': f"{np.mean(sc_sims < spot*0.90):.1%}",
            f'P({st.session_state.get("active_token_ticker","TOKEN")} <-30%)': f"{np.mean(sc_sims < spot*0.70):.1%}",
            f'{st.session_state.get("active_token_ticker","TOKEN")} Median': f"${np.median(sc_sims):.5f}",
            f'{st.session_state.get("active_token_ticker","TOKEN")} P5': f"${np.percentile(sc_sims, 5):.5f}",
            f'{st.session_state.get("active_token_ticker","TOKEN")} P95': f"${np.percentile(sc_sims, 95):.5f}",
        })
    st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

    st.markdown(f'''
    <div class="warning-box">
        BTC↔{st.session_state.get('active_token_ticker','TOKEN')} empirical correlation (ρ): <strong>{rho_btc:.2f}</strong>
        &nbsp;|&nbsp; BTC β: <strong>{btc_beta:.2f}</strong>
        &nbsp;|&nbsp; ETH β: <strong>{eth_beta:.2f}</strong>
        &nbsp;|&nbsp; F&G today: <strong>{fg_current if fg_current else "N/A"} ({fg_label})</strong>
        &nbsp;|&nbsp; Use BTC scenario → {st.session_state.get('active_token_ticker','TOKEN')} distribution to price barrier/knock-in options.
    </div>
    ''', unsafe_allow_html=True)

    # ── RETURN HISTOGRAM ──
    st.markdown('<div class="section-header">30D RETURN HISTOGRAM — ALL MODELS OVERLAID</div>', unsafe_allow_html=True)
    fig_ret = go.Figure()
    for model_name, sims in distributions.items():
        returns_pct = (sims / spot - 1) * 100
        fig_ret.add_trace(go.Histogram(
            x=returns_pct, nbinsx=100, name=model_name,
            marker_color=model_colors.get(model_name, '#888'),
            opacity=0.45 if model_name != 'Ensemble (Blended)' else 0.80,
            histnorm='probability density'
        ))
    fig_ret.add_vline(x=0,    line_color='#ffffff', line_dash='dot', annotation_text='0%')
    fig_ret.add_vline(x=30,   line_color='#00e676', line_dash='dash', annotation_text='+30%')
    fig_ret.add_vline(x=-30,  line_color='#ff4b6e', line_dash='dash', annotation_text='-30%')
    fig_ret.update_layout(
        barmode='overlay', plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='30-Day Return (%)', range=[-100, 250]),
        yaxis=dict(gridcolor=_PLT_GRID, title='Probability Density'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        height=360, margin=dict(t=20, b=50)
    )
    st.plotly_chart(fig_ret, use_container_width=True, key=_next_chart_key())

    # ── LIVE MARKET CONTEXT ──
    if mkt_data:
        st.markdown('<div class="section-header">LIVE MARKET CONTEXT (COINGECKO)</div>', unsafe_allow_html=True)
        mkt_cols2 = st.columns(4)
        for i, (lbl, val, col) in enumerate([
            ("7D CHANGE",  f"{(mkt_data.get('price_change_7d') or 0):.1f}%",  '#00d4ff' if (mkt_data.get('price_change_7d') or 0) > 0 else '#ff4b6e'),
            ("30D CHANGE", f"{(mkt_data.get('price_change_30d') or 0):.1f}%", '#00d4ff' if (mkt_data.get('price_change_30d') or 0) > 0 else '#ff4b6e'),
            ("ATH DD",     f"{(mkt_data.get('ath_change_pct') or 0):.1f}%",   '#ff4b6e'),
            ("SENTIMENT",  f"{float(mkt_data.get('sentiment_up') or 50):.0f}%", '#00e676'),
        ]):
            with mkt_cols2[i]:
                st.markdown(stat_card(lbl, val, "CoinGecko live", col), unsafe_allow_html=True)

    # ── MODEL METHODOLOGY ──
    st.markdown(f'''
    <div style="background:#0a0e14;border:1px solid #1a2535;border-radius:3px;padding:16px 20px;
                margin-top:20px;font-size:11px;color:#5a7a99;line-height:1.9;">
    <div style="color:#00d4ff;font-family:Space Mono;font-size:12px;margin-bottom:10px;">
        SIX-MODEL METHODOLOGY
    </div>
    <strong style="color:#5a9abf;">① Historical GBM</strong> — GBM fitted to XDC's own 180d log-returns. Backward-looking baseline only.<br>
    <strong style="color:{regime_color}">② Regime + Fear/Greed</strong> — Regime via SMA crossover + RSI + vol ratio. Drift tilted by regime × confidence. Fear & Greed Index (alternative.me) shifts drift ±8% at extremes (0=extreme fear → bear tilt; 100=extreme greed → bull tilt).<br>
    <strong style="color:#b388ff">③ Fat-Tail (Student-t)</strong> — Gaussian replaced with Student-t (fitted degrees-of-freedom). Correct weighting on crypto crashes and moonshots that GBM systematically underestimates.<br>
    <strong style="color:#f0a500">④ BTC Market Proxy</strong> — BTC's 90d realised drift × XDC's historical beta to BTC. Captures macro crypto risk-on/risk-off independent of XDC-specific dynamics.<br>
    <strong style="color:#ff8a65">⑤ ETH Market Proxy</strong> — ETH's 90d realised drift × XDC's beta to ETH. Second independent market proxy — ETH and BTC often diverge; this provides non-redundant information.<br>
    <strong style="color:#e040fb">⑥ GARCH(1,1) + Skew-Normal</strong> — Vol clustering model. Captures the phenomenon where high-vol days cluster together (persistence={garch_result["persistence"]:.3f}). Shocks drawn from fitted skew-normal to match the actual asymmetric return distribution.<br>
    <strong style="color:#00d4ff">Ensemble</strong> — Weighted blend. Adjustable weights above. Default: {w1:.0%}/{w2:.0%}/{w3:.0%}/{w4:.0%}/{w5:.0%}/{w6:.0%}.<br>
    <strong style="color:#44cc77">Conditional</strong> — BTC scenario → XDC using empirical correlation structure. Critical for pricing barrier, knock-in, or binary options where BTC direction matters.
    </div>
    ''', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 6 — DATA SOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown('<div class="section-header">LIVE DATA SOURCES FOR OTC PRICING</div>', unsafe_allow_html=True)
    
    sources = [
        ("SPOT PRICE", [
            ("CoinGecko API", "Demo key integrated. /simple/price endpoint. Primary source.", "https://api.coingecko.com/api/v3/"),
            ("CoinMarketCap", "Pro key integrated. Automatic fallback when CoinGecko fails.", "https://pro-api.coinmarketcap.com/v1/"),
            ("GeckoTerminal", "DEX pool prices in real-time. Good for XDC/USDT pairs.", "https://api.geckoterminal.com/api/v2/"),
            ("XDC Network RPC", "Direct on-chain. Query validator node for DEX AMM prices.", "https://rpc.xinfin.network"),
        ]),
        ("VOLATILITY", [
            ("Deribit IV Index", "BTC/ETH implied vol as benchmark. Use as proxy + premium.", "https://deribit.com/api/v2/"),
            ("DVOL Index", "Deribit's 30d forward-looking vol index. API free.", "https://deribit.com/api/v2/public/get_volatility_index_data"),
            ("CoinGecko History", "Pull 90d prices, compute HV yourself. Used in this tool.", "https://api.coingecko.com/api/v3/coins/{id}/market_chart"),
            ("Kaiko", "Professional grade OHLCV + vol data. Paid, institutional.", "https://www.kaiko.com/"),
        ]),
        ("RISK-FREE RATE", [
            ("FRED API", "SOFR, Fed Funds, T-Bill rates. Free, no key for basic.", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"),
            ("UAE EIBOR", "UAE Interbank rate from CBUAE. Use for AED-denominated deals.", "https://www.centralbank.ae/en/forex-eibor/eibor/"),
            ("AAVE / Compound", "DeFi lending rates if pricing crypto-collateralised. On-chain.", "https://aave.com/"),
        ]),
        ("STRUCTURED PRODUCT BENCHMARKS", [
            ("Bloomberg (OVDV)", "Gold standard for vol surfaces. Requires terminal.", "Bloomberg terminal"),
            ("Markit / ICE", "OTC derivatives pricing benchmarks. Institutional.", "https://ihsmarkit.com/"),
            ("CME FedWatch", "Rate expectations curve. Free.", "https://www.cmegroup.com/markets/interest-rates/"),
        ]),
    ]
    
    for category, items in sources:
        st.markdown(f'<div class="section-header">{category}</div>', unsafe_allow_html=True)
        for name, desc, url in items:
            st.markdown(f'''
            <div style="background:#0d1117;border:1px solid #1a2535;border-radius:3px;padding:12px 16px;margin:6px 0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <span style="font-family:Space Mono;font-size:13px;color:#00d4ff;">{name}</span>
                        <div style="font-size:11px;color:#5a7a99;margin-top:4px;">{desc}</div>
                    </div>
                    <div style="font-size:10px;color:#3d6080;text-align:right;font-family:IBM Plex Mono;max-width:300px;word-break:break-all;">{url}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">OTC PRICING METHODOLOGY USED</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2535;border-radius:3px;padding:16px 20px;font-size:12px;color:#8fb3d0;line-height:1.8;">
    <strong style="color:#00d4ff;">1. Black-Scholes</strong> — base case for liquid conditions. Analytical closed-form.<br>
    <strong style="color:#00d4ff;">2. Merton Jump-Diffusion</strong> — adds Poisson jumps to GBM. Essential for crypto's fat tails.<br>
    <strong style="color:#00d4ff;">3. Monte Carlo</strong> — simulation-based. Flexible for path-dependent payoffs (barriers, Asians).<br>
    <strong style="color:#00d4ff;">4. Monte Carlo + Jumps</strong> — most accurate for illiquid tokens like XDC. Most compute-heavy.<br><br>
    <strong style="color:#f0a500;">OTC Spread:</strong> Added on top of HV to account for: illiquidity premium, model risk, 
    counterparty credit risk, and cost of hedging. Negotiate your mid-market and apply your vol bid/offer.<br><br>
    <strong style="color:#f0a500;">Settlement Reference:</strong> Always agree VWAP window (e.g., 1hr VWAP at expiry from CoinGecko or agreed CEX).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px;font-size:10px;color:#2a3d55;text-align:center;letter-spacing:2px;">TRADE FINTECH LTD · XDC NETWORK · OTC DESK · NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 7 — CUSTOM PRICER (manual date + strike, all models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab7:
    st.markdown('<div class="section-header">CUSTOM OPTION PRICER — ALL MODELS SIDE BY SIDE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#5a7a99;margin-bottom:16px;">Enter any strike and expiry date. Every model prices it independently so you can see the spread of fair values.</div>', unsafe_allow_html=True)

    # ── Inputs ──────────────────────────────────────────
    cp_col1, cp_col2, cp_col3 = st.columns(3)

    with cp_col1:
        st.markdown('<div class="section-header">CONTRACT</div>', unsafe_allow_html=True)
        cp_option_type = st.selectbox("Option Type", ["Call", "Put", "Both"], key='cp_type')
        cp_strike_mode = st.selectbox("Strike Input", ["Price ($)", "% of Spot"], key='cp_smode')
        if cp_strike_mode == "Price ($)":
            cp_strike = st.number_input("Strike Price ($)", value=float(spot), format="%.6f", key='cp_strike_abs')
        else:
            cp_strike_pct = st.number_input("Strike (% of Spot)", value=100.0, min_value=1.0, max_value=500.0, key='cp_strike_pct')
            cp_strike = spot * cp_strike_pct / 100
            st.caption(f"= ${cp_strike:.6f}")
        cp_expiry = st.date_input("Expiry Date", value=(datetime.today() + timedelta(days=90)).date(), key='cp_expiry')
        cp_T = max((cp_expiry - datetime.today().date()).days, 1) / 365
        st.caption(f"T = {(cp_expiry - datetime.today().date()).days} days = {cp_T:.4f} years")

    with cp_col2:
        st.markdown('<div class="section-header">MARKET INPUTS</div>', unsafe_allow_html=True)
        cp_spot    = st.number_input("Spot Price ($)", value=float(spot), format="%.6f", key='cp_spot')
        cp_r       = st.number_input("Risk-Free Rate (%)", value=5.25, key='cp_r') / 100
        cp_sigma   = st.number_input("Base Vol / HV (%)", value=float(sigma*100), key='cp_sigma') / 100
        cp_notional = st.number_input("Notional (USD)", value=100_000, step=10_000, key='cp_notional')

    with cp_col3:
        st.markdown('<div class="section-header">JUMP PARAMETERS</div>', unsafe_allow_html=True)
        cp_lam   = st.slider("Jump Intensity (λ)", 0.0, 5.0, 0.5, 0.1, key='cp_lam')
        cp_mu_j  = st.slider("Mean Jump Size",     -0.5, 0.2, -0.10, 0.01, key='cp_muj')
        cp_sig_j = st.slider("Jump Vol (σⱼ)",      0.01, 0.80, 0.20, 0.01, key='cp_sigj')
        cp_sims  = st.select_slider("MC Simulations", options=[10_000, 50_000, 100_000, 200_000], value=50_000, key='cp_sims')

    st.markdown("---")

    # ── Run all models ──────────────────────────────────
    K  = cp_strike
    S  = cp_spot
    T_ = cp_T
    r_ = cp_r
    v_ = cp_sigma

    def run_all_models(S, K, T, r, sigma, otype, lam, mu_j, sig_j, n_sims):
        results = {}

        # 1. Black-Scholes
        results['Black-Scholes'] = {
            'price': black_scholes(S, K, T, r, sigma, otype),
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#00d4ff', 'desc': 'Analytical closed-form. Assumes lognormal, no jumps.'
        }

        # 2. Merton Jump-Diffusion
        results['Merton Jump-Diffusion'] = {
            'price': merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sig_j, otype),
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#f0a500', 'desc': f'GBM + Poisson jumps (λ={lam}, μⱼ={mu_j}, σⱼ={sig_j}). Better for crypto fat tails.'
        }

        # 3. Monte Carlo (GBM)
        mc_price, mc_se, _ = monte_carlo_price(S, K, T, r, sigma, otype, n_sims=n_sims)
        results['Monte Carlo (GBM)'] = {
            'price': mc_price, 'std_err': mc_se,
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#00e676', 'desc': f'Simulation ({n_sims:,} paths). GBM dynamics. Std err: ±${mc_se:.6f}'
        }

        # 4. Monte Carlo + Jumps
        mcj_price, mcj_se, _ = monte_carlo_price(S, K, T, r, sigma, otype, n_sims=n_sims,
                                                   jump=True, lam=lam, mu_j=mu_j, sigma_j=sig_j)
        results['Monte Carlo + Jumps'] = {
            'price': mcj_price, 'std_err': mcj_se,
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#b388ff', 'desc': f'Simulation ({n_sims:,} paths) + jump-diffusion. Most realistic for illiquid tokens.'
        }

        # 5. Binomial Tree (American-style capable)
        def binomial_tree(S, K, T, r, sigma, otype, N=500):
            dt   = T / N
            u    = np.exp(sigma * np.sqrt(dt))
            d    = 1 / u
            p    = (np.exp(r * dt) - d) / (u - d)
            disc = np.exp(-r * dt)
            # Terminal payoffs
            ST = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N+1, 1))
            if otype == 'call':
                V = np.maximum(ST - K, 0)
            else:
                V = np.maximum(K - ST, 0)
            # Backward induction
            for i in range(N - 1, -1, -1):
                V = disc * (p * V[:-1] + (1 - p) * V[1:])
            return V[0]

        results['Binomial Tree (500 steps)'] = {
            'price': binomial_tree(S, K, T, r, sigma, otype),
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#ff8a65', 'desc': 'CRR binomial tree, 500 steps. Handles early exercise (American options).'
        }

        # 6. Bachelier (Normal model — useful when S close to 0)
        def bachelier(S, K, T, r, sigma_n, otype):
            # sigma_n here = normal vol = sigma * S approximately
            sigma_abs = sigma_n * S
            if sigma_abs <= 0 or T <= 0:
                return max(S-K, 0) if otype == 'call' else max(K-S, 0)
            d = (S - K) / (sigma_abs * np.sqrt(T))
            if otype == 'call':
                return np.exp(-r*T) * ((S-K)*norm.cdf(d) + sigma_abs*np.sqrt(T)*norm.pdf(d))
            else:
                return np.exp(-r*T) * ((K-S)*norm.cdf(-d) + sigma_abs*np.sqrt(T)*norm.pdf(d))

        results['Bachelier (Normal)'] = {
            'price': bachelier(S, K, T, r, sigma, otype),
            'greeks': compute_greeks(S, K, T, r, sigma, otype),
            'color': '#e040fb', 'desc': 'Normal model. Better when spot is near zero (micro-cap tokens). Allows negative prices.'
        }

        return results

    types_to_run = []
    if cp_option_type in ['Call', 'Both']:
        types_to_run.append('call')
    if cp_option_type in ['Put', 'Both']:
        types_to_run.append('put')

    all_results = {}
    for ot in types_to_run:
        all_results[ot] = run_all_models(S, K, T_, r_, v_, ot, cp_lam, cp_mu_j, cp_sig_j, cp_sims)

    # ── Summary cards ────────────────────────────────────
    moneyness = (S / K - 1) * 100
    moneyness_label = "ITM" if moneyness > 0 else ("OTM" if moneyness < 0 else "ATM")
    st.markdown(f'''
    <div style="background:#0d1117;border:1px solid #1e2d40;border-radius:4px;padding:12px 20px;
                margin-bottom:18px;font-size:11px;color:#5a7a99;display:flex;gap:40px;">
        <span>Spot: <strong style="color:#e8f4fd">${S:.6f}</strong></span>
        <span>Strike: <strong style="color:#e8f4fd">${K:.6f}</strong></span>
        <span>Moneyness: <strong style="color:{"#00e676" if moneyness > 0 else "#ff4b6e"}">{moneyness:+.2f}% {moneyness_label}</strong></span>
        <span>T: <strong style="color:#e8f4fd">{T_*365:.0f}d ({T_:.4f}y)</strong></span>
        <span>σ: <strong style="color:#f0a500">{v_:.1%}</strong></span>
        <span>r: <strong style="color:#e8f4fd">{r_:.2%}</strong></span>
        <span>Notional: <strong style="color:#e8f4fd">${cp_notional:,.0f}</strong></span>
    </div>
    ''', unsafe_allow_html=True)

    for otype_label, ot_key in [('CALL', 'call'), ('PUT', 'put')]:
        if ot_key not in all_results:
            continue
        st.markdown(f'<div class="section-header">{otype_label} — ALL MODEL PRICES</div>', unsafe_allow_html=True)

        res = all_results[ot_key]
        model_names = list(res.keys())
        prices = [res[m]['price'] for m in model_names]
        colors = [res[m]['color'] for m in model_names]

        # Price cards
        card_cols = st.columns(len(model_names))
        for i, (mname, col) in enumerate(zip(model_names, card_cols)):
            p = res[mname]['price']
            pct_spot = p / S * 100 if S > 0 else 0
            usd_prem = p * cp_notional / S if S > 0 else 0
            se_str   = f"±${res[mname].get('std_err', 0):.6f}" if 'std_err' in res[mname] else ""
            col.markdown(f'''
            <div class="metric-card {'call' if ot_key=='call' else 'put'}" style="border-left-color:{res[mname]["color"]}">
                <div class="metric-label" style="font-size:9px;">{mname.upper()}</div>
                <div class="metric-value" style="font-size:18px;">${p:.6f}</div>
                <div class="metric-sub">{pct_spot:.3f}% spot &nbsp;|&nbsp; ${usd_prem:,.0f} USD {se_str}</div>
            </div>
            ''', unsafe_allow_html=True)

        # Model comparison bar chart
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Bar(
            x=model_names, y=prices,
            marker_color=colors, marker_line_width=0,
            text=[f"${p:.6f}" for p in prices],
            textposition='outside', textfont=dict(size=10, color='#8fb3d0')
        ))
        # Spread band
        if len(prices) > 1:
            fig_cp.add_hrect(
                y0=min(prices), y1=max(prices),
                fillcolor='rgba(0,212,255,0.04)',
                line_width=0, annotation_text=f"Model spread: ${max(prices)-min(prices):.6f}",
                annotation_font_color=_PLT_TXT, annotation_font_size=10
            )
        fig_cp.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
            xaxis=dict(gridcolor=_PLT_GRID),
            yaxis=dict(gridcolor=_PLT_GRID, title='Price ($)'),
            height=280, margin=dict(t=20, b=60, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_cp, use_container_width=True, key=_next_chart_key())

        # Greeks table
        st.markdown(f'<div class="section-header">{otype_label} GREEKS</div>', unsafe_allow_html=True)
        greek_rows = []
        for mname in model_names:
            g = res[mname]['greeks']
            greek_rows.append({
                'Model': mname,
                'Delta': f"{g['delta']:.4f}",
                'Gamma': f"{g['gamma']:.6f}",
                'Theta ($/day)': f"{g['theta']:.6f}",
                'Vega (per 1% vol)': f"{g['vega']:.6f}",
                'Rho': f"{g['rho']:.6f}",
            })
        st.dataframe(pd.DataFrame(greek_rows), use_container_width=True, hide_index=True)

        # Model descriptions
        with st.expander(f"ℹ️ Model descriptions — {otype_label}"):
            for mname in model_names:
                color = res[mname]['color']
                desc  = res[mname]['desc']
                st.markdown(f'<span style="color:{color};font-family:Space Mono;font-size:12px;">▸ {mname}</span><br>'
                            f'<span style="font-size:11px;color:#5a7a99;">{desc}</span><br><br>', unsafe_allow_html=True)

    # ── Payoff diagram ───────────────────────────────────
    st.markdown('<div class="section-header">PAYOFF AT EXPIRY — ALL MODELS</div>', unsafe_allow_html=True)
    S_range = np.linspace(S * 0.30, S * 2.50, 500)

    fig_cpay = go.Figure()
    for ot_key, otype_label in [('call', 'Call'), ('put', 'Put')]:
        if ot_key not in all_results:
            continue
        res = all_results[ot_key]
        for mname in list(res.keys())[:3]:  # top 3 models only for clarity
            p    = res[mname]['price']
            pnl  = (np.maximum(S_range - K, 0) if ot_key == 'call' else np.maximum(K - S_range, 0)) - p
            fig_cpay.add_trace(go.Scatter(
                x=S_range, y=pnl,
                name=f"{otype_label} / {mname}",
                line=dict(color=res[mname]['color'], width=1.8,
                          dash='solid' if ot_key == 'call' else 'dot')
            ))
    fig_cpay.add_hline(y=0, line_color='#3d6080', line_dash='dot')
    fig_cpay.add_vline(x=S, line_color='#ffffff', line_dash='dash', annotation_text='SPOT')
    fig_cpay.add_vline(x=K, line_color='#f0a500', line_dash='dash', annotation_text='STRIKE')
    fig_cpay.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title='Spot at Expiry ($)'),
        yaxis=dict(gridcolor=_PLT_GRID, title='P&L ($)'),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR, font=dict(size=9)),
        height=360, margin=dict(t=20, b=50)
    )
    st.plotly_chart(fig_cpay, use_container_width=True, key=_next_chart_key())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 8 — REVERSE ENGINEER (counterparty IV & premium)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab8:
    st.markdown('<div class="section-header">REVERSE ENGINEER — COUNTERPARTY QUOTE ANALYSER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:16px;line-height:1.8;">'
        'Enter quotes you received from counterparties. The tool back-solves the implied vol '
        'and implied premium they used, compares against your fair value, and analyses all '
        'quotes together to detect patterns in counterparty pricing.</div>',
        unsafe_allow_html=True
    )

    # ── Session state memory ─────────────────────────────
    if 'quotes' not in st.session_state:
        st.session_state.quotes = []

    # ── Add new quote form ───────────────────────────────
    st.markdown('<div class="section-header">ADD A COUNTERPARTY QUOTE</div>', unsafe_allow_html=True)

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    with q_col1:
        q_label       = st.text_input("Counterparty / Label", value=f"CP-{len(st.session_state.quotes)+1}", key='q_label')
        q_option_type = st.selectbox("Option Type", ["call", "put"], key='q_type')
        q_strike_mode = st.selectbox("Strike Input", ["Price ($)", "% of Spot"], key='q_smode')
        if q_strike_mode == "Price ($)":
            q_strike = st.number_input("Strike ($)", value=float(spot), format="%.6f", key='q_strike_abs')
        else:
            q_spct   = st.number_input("Strike (% of Spot)", value=100.0, key='q_spct')
            q_strike = spot * q_spct / 100
            st.caption(f"= ${q_strike:.6f}")

    with q_col2:
        q_expiry     = st.date_input("Expiry Date", value=(datetime.today() + timedelta(days=90)).date(), key='q_expiry')
        q_T          = max((q_expiry - datetime.today().date()).days, 1) / 365
        q_T_days     = max((q_expiry - datetime.today().date()).days, 1)
        st.caption(f"T = {q_T_days} days")
        q_spot       = st.number_input("Spot at Quote Time ($)", value=float(spot), format="%.6f", key='q_spot')
        q_r          = st.number_input("Risk-Free Rate (%)", value=5.25, key='q_r') / 100

    with q_col3:
        q_quoted_price = st.number_input("Quoted Premium ($)", value=0.0, format="%.6f", key='q_qprice',
                                          help="The dollar premium the counterparty quoted")
        q_quoted_pct   = st.number_input("OR Quoted as % of Spot", value=0.0, format="%.4f", key='q_qpct',
                                          help="If quoted as %, leave dollar field as 0")
        if q_quoted_pct > 0 and q_quoted_price == 0:
            q_quoted_price = q_spot * q_quoted_pct / 100
            st.caption(f"= ${q_quoted_price:.6f}")
        q_notional = st.number_input("Notional (USD)", value=100_000, step=10_000, key='q_notional')

    with q_col4:
        q_notes = st.text_area("Notes (optional)", key='q_notes', height=80,
                                placeholder="e.g. 'Spread includes credit risk', 'negotiated verbally'")
        q_counterparty_claimed_iv = st.number_input("Counterparty Claimed IV % (if disclosed)", 
                                                      value=0.0, key='q_cp_iv',
                                                      help="Leave 0 if they didn't disclose their vol")

    add_col, clear_col = st.columns([1, 4])
    with add_col:
        if st.button("➕  Add Quote", key='q_add'):
            if q_quoted_price > 0:
                # Back-solve IV using Black-Scholes
                def bs_price_for_iv(iv, S, K, T, r, otype):
                    return black_scholes(S, K, T, r, iv, otype)

                try:
                    implied_vol = brentq(
                        lambda iv: bs_price_for_iv(iv, q_spot, q_strike, q_T, q_r, q_option_type) - q_quoted_price,
                        1e-6, 20.0, xtol=1e-6, maxiter=500
                    )
                except Exception:
                    implied_vol = None

                # Fair values from all models
                fv_bs  = black_scholes(q_spot, q_strike, q_T, q_r, sigma, q_option_type)
                fv_mjd = merton_jump_diffusion(q_spot, q_strike, q_T, q_r, sigma, 0.5, -0.1, 0.2, q_option_type)
                fv_mc, _, _ = monte_carlo_price(q_spot, q_strike, q_T, q_r, sigma, q_option_type)
                fv_mcj, _, _ = monte_carlo_price(q_spot, q_strike, q_T, q_r, sigma, q_option_type,
                                                  jump=True, lam=0.5, mu_j=-0.1, sigma_j=0.2)
                fv_avg = np.mean([fv_bs, fv_mjd, fv_mc, fv_mcj])

                moneyness_pct = (q_spot / q_strike - 1) * 100

                quote_entry = {
                    'label':           q_label,
                    'type':            q_option_type,
                    'strike':          q_strike,
                    'expiry':          str(q_expiry),
                    'T_days':          q_T_days,
                    'T':               q_T,
                    'spot':            q_spot,
                    'r':               q_r,
                    'quoted_price':    q_quoted_price,
                    'notional':        q_notional,
                    'implied_vol':     implied_vol,
                    'claimed_iv':      q_counterparty_claimed_iv if q_counterparty_claimed_iv > 0 else None,
                    'fv_bs':           fv_bs,
                    'fv_mjd':          fv_mjd,
                    'fv_mc':           fv_mc,
                    'fv_mcj':          fv_mcj,
                    'fv_avg':          fv_avg,
                    'our_sigma':       sigma,
                    'overcharge_abs':  q_quoted_price - fv_avg,
                    'overcharge_pct':  (q_quoted_price / fv_avg - 1) * 100 if fv_avg > 0 else 0,
                    'moneyness_pct':   moneyness_pct,
                    'notes':           q_notes,
                    'usd_premium':     q_quoted_price * q_notional / q_spot if q_spot > 0 else 0,
                }
                st.session_state.quotes.append(quote_entry)
                st.success(f"✅ Quote from [{q_label}] added. Total quotes: {len(st.session_state.quotes)}")
                st.rerun()
            else:
                st.error("Enter a quoted premium greater than 0")

    with clear_col:
        if st.button("🗑️  Clear All Quotes", key='q_clear'):
            st.session_state.quotes = []
            st.rerun()

    # ── Display stored quotes ────────────────────────────
    if not st.session_state.quotes:
        st.markdown('''
        <div class="warning-box" style="margin-top:20px;">
            No quotes yet. Add your first counterparty quote above.
            You can add up to as many as you want — the analysis below updates automatically.
        </div>
        ''', unsafe_allow_html=True)
    else:
        n_quotes = len(st.session_state.quotes)
        st.markdown(f'<div class="section-header" style="margin-top:24px;">STORED QUOTES ({n_quotes})</div>', unsafe_allow_html=True)

        # ── Individual quote cards ───────────────────────
        for i, q in enumerate(st.session_state.quotes):
            iv_str      = f"{q['implied_vol']:.1%}" if q['implied_vol'] else "N/A (check premium)"
            iv_vs_ours  = ((q['implied_vol'] or 0) - q['our_sigma']) * 100
            overcharge_color = '#ff4b6e' if q['overcharge_abs'] > 0 else '#00e676'
            iv_color    = '#ff4b6e' if iv_vs_ours > 5 else ('#f0a500' if iv_vs_ours > 0 else '#00e676')
            type_color  = '#00d4ff' if q['type'] == 'call' else '#ff4b6e'

            c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 0.5])

            with c1:
                st.markdown(f'''
                <div class="metric-card" style="border-left-color:{type_color}">
                    <div class="metric-label">QUOTE #{i+1} — {q["label"]}</div>
                    <div style="font-size:13px;color:{type_color};font-family:Space Mono;">{q["type"].upper()} @ ${q["strike"]:.6f}</div>
                    <div class="metric-sub">Expiry: {q["expiry"]} ({q["T_days"]}d) &nbsp;|&nbsp; Spot: ${q["spot"]:.6f}</div>
                    <div class="metric-sub" style="margin-top:4px;color:#3d6080;font-size:10px;">{q["notes"][:60] if q["notes"] else ""}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
                <div class="metric-card" style="border-left-color:{overcharge_color}">
                    <div class="metric-label">QUOTED PRICE</div>
                    <div class="metric-value" style="font-size:18px;">${q["quoted_price"]:.6f}</div>
                    <div class="metric-sub">${q["usd_premium"]:,.0f} USD notional</div>
                </div>
                ''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''
                <div class="metric-card" style="border-left-color:{iv_color}">
                    <div class="metric-label">IMPLIED VOL (BACK-SOLVED)</div>
                    <div class="metric-value" style="font-size:18px;color:{iv_color};">{iv_str}</div>
                    <div class="metric-sub">Our σ: {q["our_sigma"]:.1%} &nbsp;|&nbsp; 
                    Spread: <span style="color:{iv_color}">{iv_vs_ours:+.1f}pp</span></div>
                </div>
                ''', unsafe_allow_html=True)
            with c4:
                st.markdown(f'''
                <div class="metric-card" style="border-left-color:{overcharge_color}">
                    <div class="metric-label">VS OUR FAIR VALUE (AVG)</div>
                    <div class="metric-value" style="font-size:18px;color:{overcharge_color};">{q["overcharge_pct"]:+.1f}%</div>
                    <div class="metric-sub">${q["overcharge_abs"]:+.6f} &nbsp;|&nbsp; Fair: ${q["fv_avg"]:.6f}</div>
                </div>
                ''', unsafe_allow_html=True)
            with c5:
                if st.button("❌", key=f'del_q_{i}', help="Remove this quote"):
                    st.session_state.quotes.pop(i)
                    st.rerun()

        # ── Full breakdown table ─────────────────────────
        st.markdown('<div class="section-header" style="margin-top:20px;">FULL BREAKDOWN TABLE</div>', unsafe_allow_html=True)

        table_rows = []
        for i, q in enumerate(st.session_state.quotes):
            iv_str = f"{q['implied_vol']:.1%}" if q['implied_vol'] else "N/A"
            table_rows.append({
                '#':           i + 1,
                'Label':       q['label'],
                'Type':        q['type'].upper(),
                'Strike':      f"${q['strike']:.6f}",
                'Expiry':      q['expiry'],
                'T (days)':    q['T_days'],
                'Quoted ($)':  f"${q['quoted_price']:.6f}",
                'Impl. Vol':   iv_str,
                'Our σ':       f"{q['our_sigma']:.1%}",
                'IV Spread':   f"{((q['implied_vol'] or 0) - q['our_sigma'])*100:+.1f}pp",
                'FV BS':       f"${q['fv_bs']:.6f}",
                'FV Jump':     f"${q['fv_mjd']:.6f}",
                'FV MC':       f"${q['fv_mc']:.6f}",
                'FV Avg':      f"${q['fv_avg']:.6f}",
                'Overcharge':  f"{q['overcharge_pct']:+.1f}%",
                'USD Delta':   f"${q['overcharge_abs'] * q['notional'] / q['spot']:+,.0f}",
                'Moneyness':   f"{q['moneyness_pct']:+.2f}%",
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # ── Multi-quote analysis (only when >1 quote) ───
        if n_quotes >= 2:
            st.markdown('<div class="section-header" style="margin-top:24px;">MULTI-QUOTE ANALYSIS</div>', unsafe_allow_html=True)

            # Implied vol comparison chart
            fig_iv = go.Figure()
            labels_q   = [q['label'] for q in st.session_state.quotes]
            ivs        = [(q['implied_vol'] or 0)*100 for q in st.session_state.quotes]
            fv_avgs    = [q['fv_avg'] for q in st.session_state.quotes]
            quoted_ps  = [q['quoted_price'] for q in st.session_state.quotes]
            overcharges= [q['overcharge_pct'] for q in st.session_state.quotes]

            fig_iv = make_subplots(rows=1, cols=2,
                subplot_titles=["IMPLIED VOL BY QUOTE", "QUOTED vs FAIR VALUE"])

            bar_colors = ['#ff4b6e' if iv > sigma*100 + 5 else '#f0a500' if iv > sigma*100 else '#00e676'
                          for iv in ivs]

            fig_iv.add_trace(go.Bar(
                x=labels_q, y=ivs, name='Implied Vol (%)',
                marker_color=bar_colors, marker_line_width=0,
                text=[f"{v:.1f}%" for v in ivs], textposition='outside'
            ), row=1, col=1)
            fig_iv.add_hline(y=sigma*100, line_color='#00d4ff', line_dash='dash',
                             annotation_text=f"Our σ={sigma:.0%}", row=1, col=1)

            fig_iv.add_trace(go.Bar(
                x=labels_q, y=quoted_ps, name='Quoted Price',
                marker_color='#f0a500', marker_line_width=0, opacity=0.8
            ), row=1, col=2)
            fig_iv.add_trace(go.Bar(
                x=labels_q, y=fv_avgs, name='Our Fair Value (avg)',
                marker_color='#00d4ff', marker_line_width=0, opacity=0.8
            ), row=1, col=2)

            fig_iv.update_layout(
                plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
                barmode='group', height=320,
                legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
                margin=dict(t=40, b=60)
            )
            for c in [1, 2]:
                fig_iv.update_xaxes(gridcolor=_PLT_GRID, row=1, col=c)
                fig_iv.update_yaxes(gridcolor=_PLT_GRID, row=1, col=c)
            st.plotly_chart(fig_iv, use_container_width=True, key=_next_chart_key())

            # Overcharge analysis
            st.markdown('<div class="section-header">OVERCHARGE ANALYSIS</div>', unsafe_allow_html=True)

            fig_oc = go.Figure(go.Bar(
                x=labels_q, y=overcharges,
                marker_color=['#ff4b6e' if o > 0 else '#00e676' for o in overcharges],
                marker_line_width=0,
                text=[f"{o:+.1f}%" for o in overcharges], textposition='outside'
            ))
            fig_oc.add_hline(y=0, line_color='#3d6080', line_dash='dot')
            fig_oc.update_layout(
                plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
                xaxis=dict(gridcolor=_PLT_GRID),
                yaxis=dict(gridcolor=_PLT_GRID, title='Overcharge vs Fair Value (%)'),
                height=260, margin=dict(t=20, b=60),
                title=dict(text='Quote overcharge vs our average fair value', font=dict(size=11, color='#5a9abf'))
            )
            st.plotly_chart(fig_oc, use_container_width=True, key=_next_chart_key())

            # Summary statistics across all quotes
            valid_ivs = [q['implied_vol']*100 for q in st.session_state.quotes if q['implied_vol']]
            st.markdown('<div class="section-header">AGGREGATE STATISTICS</div>', unsafe_allow_html=True)

            ag_cols = st.columns(5)
            def ag_card(label, value, sub, color='#00d4ff'):
                return f'''<div class="metric-card" style="border-left-color:{color}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:18px;color:{color};">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>'''

            with ag_cols[0]:
                avg_iv = np.mean(valid_ivs) if valid_ivs else 0
                st.markdown(ag_card("AVG IMPLIED VOL", f"{avg_iv:.1f}%",
                    f"vs our {sigma:.0%} → {avg_iv - sigma*100:+.1f}pp", '#f0a500'), unsafe_allow_html=True)
            with ag_cols[1]:
                iv_range = max(valid_ivs) - min(valid_ivs) if len(valid_ivs) > 1 else 0
                st.markdown(ag_card("IV RANGE", f"{iv_range:.1f}pp",
                    f"Min: {min(valid_ivs):.1f}% | Max: {max(valid_ivs):.1f}%", '#b388ff'), unsafe_allow_html=True)
            with ag_cols[2]:
                avg_overcharge = np.mean(overcharges)
                st.markdown(ag_card("AVG OVERCHARGE", f"{avg_overcharge:+.1f}%",
                    f"Worst: {max(overcharges):+.1f}% | Best: {min(overcharges):+.1f}%",
                    '#ff4b6e' if avg_overcharge > 0 else '#00e676'), unsafe_allow_html=True)
            with ag_cols[3]:
                total_usd_overcharge = sum(
                    q['overcharge_abs'] * q['notional'] / q['spot']
                    for q in st.session_state.quotes if q['spot'] > 0
                )
                st.markdown(ag_card("TOTAL USD DELTA", f"${total_usd_overcharge:+,.0f}",
                    "Sum across all quotes vs fair value",
                    '#ff4b6e' if total_usd_overcharge > 0 else '#00e676'), unsafe_allow_html=True)
            with ag_cols[4]:
                # Best quote = closest to our fair value
                best_idx = int(np.argmin([abs(q['overcharge_pct']) for q in st.session_state.quotes]))
                best_q   = st.session_state.quotes[best_idx]
                st.markdown(ag_card("BEST QUOTE", best_q['label'],
                    f"Overcharge: {best_q['overcharge_pct']:+.1f}%", '#00e676'), unsafe_allow_html=True)

            # Narrative verdict
            verdict_color = '#ff4b6e' if avg_overcharge > 15 else '#f0a500' if avg_overcharge > 5 else '#00e676'
            verdict_text  = (
                f"Counterparties are pricing significantly above fair value on average (+{avg_overcharge:.1f}%). "
                f"The implied vol spread over your model vol is {np.mean(valid_ivs) - sigma*100:+.1f}pp — "
                "push back on the vol, not the premium. Negotiate IV down to at most "
                f"{sigma*100 + 5:.0f}% to get fair pricing."
                if avg_overcharge > 10 else
                f"Quotes are broadly in line with fair value (avg overcharge: {avg_overcharge:+.1f}%). "
                f"The IV spread of {np.mean(valid_ivs) - sigma*100:+.1f}pp is within normal OTC bid/ask. "
                "Pick the quote with lowest implied vol."
                if avg_overcharge > 0 else
                f"Counterparties are quoting below your fair value ({avg_overcharge:.1f}%). "
                "Double-check your vol input — or this may be a favourable entry point."
            )
            st.markdown(f'''
            <div class="recommendation-box" style="border-left-color:{verdict_color};margin-top:16px;">
                <div style="font-family:Space Mono;font-size:13px;color:{verdict_color};margin-bottom:8px;">
                    VERDICT
                </div>
                <div style="font-size:12px;color:#8fb3d0;line-height:1.8;">{verdict_text}</div>
            </div>
            ''', unsafe_allow_html=True)

        # ── Single quote case ────────────────────────────
        elif n_quotes == 1:
            q = st.session_state.quotes[0]
            iv_str = f"{q['implied_vol']:.1%}" if q['implied_vol'] else "N/A"
            iv_vs = ((q['implied_vol'] or 0) - q['our_sigma']) * 100
            iv_col = '#ff4b6e' if iv_vs > 5 else '#f0a500' if iv_vs > 0 else '#00e676'
            oc_col = '#ff4b6e' if q['overcharge_pct'] > 0 else '#00e676'

            st.markdown(f'''
            <div class="recommendation-box" style="border-left-color:{oc_col};margin-top:16px;">
                <div style="font-family:Space Mono;font-size:13px;color:{oc_col};margin-bottom:10px;">
                    SINGLE QUOTE ANALYSIS — {q["label"]}
                </div>
                <div style="font-size:12px;color:#8fb3d0;line-height:2.0;">
                    Quoted premium: <strong style="color:#e8f4fd">${q["quoted_price"]:.6f}</strong>
                    &nbsp;|&nbsp; Our fair value (avg 4 models): <strong style="color:#e8f4fd">${q["fv_avg"]:.6f}</strong><br>
                    Back-solved IV: <strong style="color:{iv_col}">{iv_str}</strong>
                    &nbsp;|&nbsp; Our σ: <strong style="color:#e8f4fd">{q["our_sigma"]:.1%}</strong>
                    &nbsp;|&nbsp; IV spread: <strong style="color:{iv_col}">{iv_vs:+.1f}pp</strong><br>
                    Overcharge vs fair: <strong style="color:{oc_col}">{q["overcharge_pct"]:+.1f}%</strong>
                    &nbsp;|&nbsp; USD impact on ${q["notional"]:,} notional: 
                    <strong style="color:{oc_col}">${q["overcharge_abs"] * q["notional"] / q["spot"]:+,.0f}</strong><br><br>
                    <span style="color:#5a7a99;">Add more quotes from different counterparties to get a full competitive analysis.</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Model breakdown for this quote
            st.markdown('<div class="section-header">FAIR VALUE — ALL MODELS vs QUOTED</div>', unsafe_allow_html=True)
            fig_single = go.Figure()
            model_fvs = {
                'BS': q['fv_bs'], 'Jump-Diff': q['fv_mjd'],
                'MC': q['fv_mc'], 'MC+Jumps': q['fv_mcj']
            }
            fv_colors = ['#5a9abf', '#f0a500', '#00e676', '#b388ff']
            fig_single.add_trace(go.Bar(
                x=list(model_fvs.keys()), y=list(model_fvs.values()),
                marker_color=fv_colors, name='Fair Value', opacity=0.85
            ))
            fig_single.add_hline(y=q['quoted_price'], line_color='#ff4b6e', line_dash='dash',
                                  annotation_text=f"Quoted: ${q['quoted_price']:.6f}",
                                  annotation_font_color='#ff4b6e')
            fig_single.update_layout(
                plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
                xaxis=dict(gridcolor=_PLT_GRID),
                yaxis=dict(gridcolor=_PLT_GRID, title='Price ($)'),
                height=280, margin=dict(t=30, b=40), showlegend=False
            )
            st.plotly_chart(fig_single, use_container_width=True, key=_next_chart_key())



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 9 — LIVE VOL SURFACE (Deribit: BTC / ETH / SOL + XDC proxy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab9:
    st.markdown('<div class="section-header">LIVE VOLATILITY SURFACE — DERIBIT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:16px;line-height:1.8;">'
        'Live implied vol surfaces from Deribit (BTC, ETH, SOL). '
        'All data is public — no API key required. '
        'XDC has no listed options: its vol is estimated as a proxy from BTC/ETH surface '
        'with a micro-cap illiquidity premium applied.</div>',
        unsafe_allow_html=True
    )

    # ── Controls ────────────────────────────────────────────────────
    ctrl_cols = st.columns([2, 2, 2, 3])
    with ctrl_cols[0]:
        # Default to sidebar token if it has a Deribit surface (BTC/ETH/SOL)
        _sidebar_tok = st.session_state.get("active_token_ticker", "BTC")
        _deribit_tokens = ["BTC", "ETH", "SOL"]
        _default_vs = _sidebar_tok if _sidebar_tok in _deribit_tokens else "BTC"
        surface_asset = st.selectbox("Primary Asset", _deribit_tokens,
                                     index=_deribit_tokens.index(_default_vs), key="vs_asset")
    with ctrl_cols[1]:
        iv_col_choice = st.selectbox("IV Type", ["mark_iv", "mid_iv", "bid_iv", "ask_iv"], key="vs_ivcol")
    with ctrl_cols[2]:
        opt_type_filter = st.selectbox("Option Type", ["Both", "call", "put"], key="vs_opttype")
    with ctrl_cols[3]:
        xdc_T_days_vs   = st.slider("XDC Tenor for IV Proxy (days)", 7, 180, 30, key="vs_xdcT")
        xdc_moneyness_vs = st.slider("XDC Moneyness for IV Proxy", 0.5, 2.0, 1.0, 0.05, key="vs_xdcmon")

    refresh_btn = st.button("🔄  Refresh Live Data", key="vs_refresh")

    # ── Cache data in session state ─────────────────────────────────
    cache_key_book  = f"vs_book_{surface_asset}"
    cache_key_index = f"vs_index_{surface_asset}"
    cache_key_btc   = "vs_book_BTC"
    cache_key_eth   = "vs_book_ETH"
    cache_key_time  = "vs_last_fetch"

    needs_fetch = (
        refresh_btn or
        cache_key_book  not in st.session_state or
        cache_key_index not in st.session_state
    )

    if needs_fetch:
        with st.spinner(f"Fetching live Deribit data for {surface_asset}, BTC, ETH..."):
            # Fetch primary asset
            idx_price = fetch_deribit_index(surface_asset)
            book_raw  = fetch_deribit_book_summary(surface_asset)
            st.session_state[cache_key_index] = idx_price
            st.session_state[cache_key_book]  = book_raw

            # Always fetch BTC + ETH for XDC proxy
            if surface_asset != "BTC":
                btc_idx  = fetch_deribit_index("BTC")
                btc_book = fetch_deribit_book_summary("BTC")
                st.session_state["vs_index_BTC"] = btc_idx
                st.session_state["vs_book_BTC"]  = btc_book
            if surface_asset != "ETH":
                eth_idx  = fetch_deribit_index("ETH")
                eth_book = fetch_deribit_book_summary("ETH")
                st.session_state["vs_index_ETH"] = eth_idx
                st.session_state["vs_book_ETH"]  = eth_book

            # DVOL history for BTC + ETH
            st.session_state["vs_dvol_BTC"] = fetch_deribit_dvol_history("BTC", days=30)
            st.session_state["vs_dvol_ETH"] = fetch_deribit_dvol_history("ETH", days=30)
            st.session_state[cache_key_time] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    idx_price = st.session_state.get(cache_key_index)
    book_raw  = st.session_state.get(cache_key_book, [])
    last_time = st.session_state.get(cache_key_time, "Not yet fetched")

    # Parse
    df_surface = parse_book_summary(book_raw, idx_price)

    # BTC/ETH for XDC proxy
    btc_idx  = st.session_state.get("vs_index_BTC") or fetch_deribit_index("BTC")
    eth_idx  = st.session_state.get("vs_index_ETH") or fetch_deribit_index("ETH")
    df_btc   = parse_book_summary(st.session_state.get("vs_book_BTC", []), btc_idx)
    df_eth   = parse_book_summary(st.session_state.get("vs_book_ETH", []), eth_idx)

    # ── Live index prices banner ─────────────────────────────────────
    price_cols = st.columns(5)
    prices_map = {
        "BTC": (btc_idx, "#f0a500"),
        "ETH": (eth_idx, "#b388ff"),
        "XDC": (spot,    "#00d4ff"),
    }
    if surface_asset == "SOL":
        sol_idx = st.session_state.get("vs_index_SOL") or fetch_deribit_index("SOL")
        prices_map["SOL"] = (sol_idx, "#00e676")

    for i, (asset, (px_val, col)) in enumerate(prices_map.items()):
        with price_cols[i]:
            st.markdown(f'''
            <div class="metric-card" style="border-left-color:{col};padding:12px 16px;">
                <div class="metric-label">{asset} INDEX</div>
                <div class="metric-value" style="font-size:20px;color:{col};">
                    {"${:,.2f}".format(px_val) if px_val else "N/A"}
                </div>
                <div class="metric-sub">Deribit live</div>
            </div>''', unsafe_allow_html=True)

    with price_cols[-1]:
        st.markdown(f'''
        <div style="font-size:10px;color:#3d6080;padding:12px 0;">
            Last fetch:<br><span style="color:#5a9abf">{last_time}</span><br>
            Instruments: <span style="color:#5a9abf">{len(df_surface)}</span>
        </div>''', unsafe_allow_html=True)

    if df_surface.empty:
        st.warning("⚠️  No data returned from Deribit. Check your internet connection or click Refresh.")
        st.stop()

    # Apply option type filter
    if opt_type_filter != "Both":
        df_surface = df_surface[df_surface["option_type"] == opt_type_filter]

    # ── DVOL History ─────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:20px;">DVOL INDEX — 30-DAY HISTORY (BTC & ETH)</div>', unsafe_allow_html=True)

    dvol_btc = st.session_state.get("vs_dvol_BTC")
    dvol_eth = st.session_state.get("vs_dvol_ETH")

    fig_dvol = go.Figure()
    if dvol_btc:
        ts_btc = [datetime.fromtimestamp(t/1000) for t, v in dvol_btc]
        vl_btc = [v for t, v in dvol_btc]
        fig_dvol.add_trace(go.Scatter(
            x=ts_btc, y=vl_btc, name="BTC DVOL",
            line=dict(color="#f0a500", width=2),
            fill="tozeroy", fillcolor="rgba(240,165,0,0.06)"
        ))
    if dvol_eth:
        ts_eth = [datetime.fromtimestamp(t/1000) for t, v in dvol_eth]
        vl_eth = [v for t, v in dvol_eth]
        fig_dvol.add_trace(go.Scatter(
            x=ts_eth, y=vl_eth, name="ETH DVOL",
            line=dict(color="#b388ff", width=2),
            fill="tozeroy", fillcolor="rgba(179,136,255,0.05)"
        ))

    if dvol_btc or dvol_eth:
        fig_dvol.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
            xaxis=dict(gridcolor=_PLT_GRID),
            yaxis=dict(gridcolor=_PLT_GRID, title="DVOL (%)"),
            legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
            height=240, margin=dict(t=10, b=40)
        )
        st.plotly_chart(fig_dvol, use_container_width=True, key=_next_chart_key())
    else:
        st.info("DVOL history unavailable — Deribit may be rate-limiting. Try Refresh.")

    # ── 3D VOL SURFACE ────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">{surface_asset} VOLATILITY SURFACE — 3D</div>', unsafe_allow_html=True)

    # Build surface grid
    df_for_surface = df_surface[df_surface[iv_col_choice].notna()].copy()
    expiries_uniq  = sorted(df_for_surface["T_days"].unique())
    mon_grid       = np.linspace(0.55, 1.65, 45)

    if len(expiries_uniq) >= 2:
        surface_z = np.full((len(expiries_uniq), len(mon_grid)), np.nan)
        for i, T in enumerate(expiries_uniq):
            sl = df_for_surface[df_for_surface["T_days"] == T].sort_values("moneyness")
            if len(sl) >= 2:
                try:
                    surface_z[i, :] = np.interp(
                        mon_grid, sl["moneyness"].values,
                        sl[iv_col_choice].values, left=np.nan, right=np.nan
                    )
                except Exception:
                    pass

        fig_3d = go.Figure(data=[go.Surface(
            x=mon_grid,
            y=expiries_uniq,
            z=surface_z,
            colorscale=[
                [0.0,  "#0d1117"],
                [0.2,  "#003d5c"],
                [0.4,  "#005580"],
                [0.6,  "#00d4ff"],
                [0.8,  "#f0a500"],
                [1.0,  "#ff4b6e"],
            ],
            opacity=0.92,
            colorbar=dict(
                title=dict(text="IV (%)", font=dict(color="#5a7a99", size=10)),
                tickfont=dict(color="#5a7a99", size=9),
                len=0.6
            ),
            hovertemplate=(
                "Moneyness: %{x:.2f}<br>"
                "Tenor: %{y}d<br>"
                "IV: %{z:.1f}%<extra></extra>"
            )
        )])

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Moneyness (K/S)", gridcolor=_PLT_GRID,
                           backgroundcolor=_PLT_BG2, color="#5a7a99"),
                yaxis=dict(title="Tenor (days)", gridcolor=_PLT_GRID,
                           backgroundcolor=_PLT_BG2, color="#5a7a99"),
                zaxis=dict(title="IV (%)", gridcolor=_PLT_GRID,
                           backgroundcolor=_PLT_BG2, color="#5a7a99"),
                bgcolor="#0a0c10",
            ),
            paper_bgcolor=_PLT_BG,
            font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
            height=520,
            margin=dict(t=20, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True, key=_next_chart_key())
    else:
        st.info("Need at least 2 expiries to render 3D surface. Data may be loading.")

    # ── 2D SMILE SLICES ───────────────────────────────────────────────
    st.markdown(f'<div class="section-header">{surface_asset} VOL SMILE — BY EXPIRY</div>', unsafe_allow_html=True)

    # Let user pick which expiries to show
    expiry_labels = {}
    for T in expiries_uniq:
        grp = df_for_surface[df_for_surface["T_days"] == T]
        if not grp.empty:
            expiry_labels[T] = f"{grp['expiry_str'].iloc[0]} ({T}d)"

    selected_expiries = st.multiselect(
        "Select expiries to plot",
        options=list(expiry_labels.keys()),
        default=list(expiry_labels.keys())[:min(5, len(expiry_labels))],
        format_func=lambda t: expiry_labels.get(t, str(t)),
        key="vs_expiry_sel"
    )

    smile_palette = ["#00d4ff", "#f0a500", "#00e676", "#b388ff",
                     "#ff4b6e", "#ff8a65", "#e040fb", "#44cc77"]

    fig_smile2d = go.Figure()
    for i, T in enumerate(selected_expiries):
        sl = df_for_surface[df_for_surface["T_days"] == T].sort_values("moneyness")
        col_s = smile_palette[i % len(smile_palette)]
        lbl   = expiry_labels.get(T, f"{T}d")

        # Mark IV line
        sl_iv = sl[sl[iv_col_choice].notna()]
        if not sl_iv.empty:
            fig_smile2d.add_trace(go.Scatter(
                x=sl_iv["moneyness"], y=sl_iv[iv_col_choice],
                name=lbl, line=dict(color=col_s, width=2),
                hovertemplate=f"<b>{lbl}</b><br>Mon: %{{x:.3f}}<br>IV: %{{y:.1f}}%<extra></extra>"
            ))

        # Bid/ask shading
        sl_ba = sl[sl["bid_iv"].notna() & sl["ask_iv"].notna()]
        if not sl_ba.empty:
            fig_smile2d.add_trace(go.Scatter(
                x=pd.concat([sl_ba["moneyness"], sl_ba["moneyness"].iloc[::-1]]),
                y=pd.concat([sl_ba["ask_iv"], sl_ba["bid_iv"].iloc[::-1]]),
                fill="toself",
                fillcolor=col_s.replace(")", ",0.07)").replace("rgb", "rgba") if "rgb" in col_s
                          else f"rgba({int(col_s[1:3],16)},{int(col_s[3:5],16)},{int(col_s[5:7],16)},0.07)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))

    fig_smile2d.add_vline(x=1.0, line_color="#ffffff", line_dash="dot",
                          annotation_text="ATM", annotation_font_color="#ffffff")
    fig_smile2d.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title="Moneyness (K/S)"),
        yaxis=dict(gridcolor=_PLT_GRID, title=f"{iv_col_choice} (%)"),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        height=400, margin=dict(t=10, b=50),
        hovermode="x unified"
    )
    st.plotly_chart(fig_smile2d, use_container_width=True, key=_next_chart_key())

    # ── TERM STRUCTURE ────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">ATM TERM STRUCTURE — {surface_asset} vs BTC vs ETH</div>', unsafe_allow_html=True)

    def get_atm_term_structure(df, iv_col):
        """Extract ATM IV (moneyness closest to 1.0) per expiry."""
        pts = []
        for T, grp in df[df[iv_col].notna()].groupby("T_days"):
            atm_idx = (grp["moneyness"] - 1.0).abs().idxmin()
            pts.append((T, grp.loc[atm_idx, iv_col]))
        return sorted(pts)

    fig_ts = go.Figure()
    for asset_name, df_ts, col_ts in [
        (surface_asset, df_surface, "#00d4ff"),
        ("BTC",         df_btc,     "#f0a500"),
        ("ETH",         df_eth,     "#b388ff"),
    ]:
        if df_ts.empty:
            continue
        pts = get_atm_term_structure(df_ts, iv_col_choice if iv_col_choice in df_ts.columns else "mark_iv")
        if pts:
            xs, ys = zip(*pts)
            fig_ts.add_trace(go.Scatter(
                x=xs, y=ys, name=f"{asset_name} ATM",
                line=dict(color=col_ts, width=2.5),
                mode="lines+markers",
                marker=dict(size=5, color=col_ts),
                hovertemplate=f"<b>{asset_name}</b> %{{x}}d: %{{y:.1f}}%<extra></extra>"
            ))

    fig_ts.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title="Tenor (days)"),
        yaxis=dict(gridcolor=_PLT_GRID, title="ATM IV (%)"),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        height=300, margin=dict(t=10, b=50)
    )
    st.plotly_chart(fig_ts, use_container_width=True, key=_next_chart_key())

    # ── SKEW & METRICS TABLE ─────────────────────────────────────────
    st.markdown(f'<div class="section-header">{surface_asset} SKEW METRICS BY EXPIRY</div>', unsafe_allow_html=True)

    skew_df = compute_skew_metrics(df_surface, iv_col_choice)
    if not skew_df.empty:
        st.dataframe(skew_df, use_container_width=True, hide_index=True)

        # Skew term structure plot
        fig_skew = make_subplots(rows=1, cols=2,
            subplot_titles=["ATM IV Term Structure", "25-Delta Skew Term Structure"])

        valid_atm  = skew_df[skew_df["ATM IV %"].notna()]
        valid_skew = skew_df[skew_df["25d Skew"].notna()]

        if not valid_atm.empty:
            fig_skew.add_trace(go.Scatter(
                x=valid_atm["T_days"], y=valid_atm["ATM IV %"],
                line=dict(color="#00d4ff", width=2.5),
                mode="lines+markers", name="ATM IV"
            ), row=1, col=1)

        if not valid_skew.empty:
            colors_skew = ["#ff4b6e" if v > 0 else "#00e676"
                           for v in valid_skew["25d Skew"]]
            fig_skew.add_trace(go.Bar(
                x=valid_skew["T_days"], y=valid_skew["25d Skew"],
                marker_color=colors_skew, name="25d Skew"
            ), row=1, col=2)
            fig_skew.add_hline(y=0, line_color="#3d6080",
                                line_dash="dot", row=1, col=2)

        fig_skew.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
            height=280, showlegend=False, margin=dict(t=40, b=40)
        )
        for c in [1, 2]:
            fig_skew.update_xaxes(gridcolor=_PLT_GRID, title_text="Tenor (days)", row=1, col=c)
            fig_skew.update_yaxes(gridcolor=_PLT_GRID, title_text="IV (%)", row=1, col=c)
        st.plotly_chart(fig_skew, use_container_width=True, key=_next_chart_key())

    # ── XDC HISTORICAL VOLATILITY SURFACE ─────────────────────────────
    _active_tok_vs = st.session_state.get("active_token_ticker", "XDC")
    st.markdown(f'<div class="section-header">{_active_tok_vs} HISTORICAL VOL ANALYSIS</div>', unsafe_allow_html=True)

    # Fetch XDC historical data for vol computation
    _cid_vs = st.session_state.get("active_coin_id", "xdc-network")
    _cid_vs = _cid_vs if _cid_vs != "custom" else "xdc-network"
    _ticker_vs = st.session_state.get("active_token_ticker", "XDC")
    _vs_hist_key = f"vs_hist_{_cid_vs}"
    if _vs_hist_key not in st.session_state or refresh_btn:
        _hist = _fetch_coin_history(_cid_vs, _ticker_vs, 365)
        st.session_state[_vs_hist_key] = _hist if (_hist and len(_hist) > 5) else None

    xdc_hist_prices = st.session_state.get(_vs_hist_key)

    if xdc_hist_prices and len(xdc_hist_prices) > 30:
        xdc_log_rets_vs = np.log(np.array(xdc_hist_prices[1:]) / np.array(xdc_hist_prices[:-1]))

        # Compute rolling HV for multiple windows
        windows = [7, 14, 30, 60, 90, 180]
        hv_series = {}
        for w in windows:
            if len(xdc_log_rets_vs) >= w:
                rolling_hv = []
                for i in range(w, len(xdc_log_rets_vs)):
                    window_rets = xdc_log_rets_vs[i-w:i]
                    rolling_hv.append(np.std(window_rets) * np.sqrt(365) * 100)
                hv_series[w] = rolling_hv

        # Plot rolling HV
        hv_palette = ['#00d4ff', '#00e676', '#f0a500', '#b388ff', '#ff8a65', '#e040fb']
        fig_xdc_hv = go.Figure()
        for i, (w, series) in enumerate(hv_series.items()):
            fig_xdc_hv.add_trace(go.Scatter(
                x=list(range(len(series))), y=series,
                name=f'{w}d HV', line=dict(color=hv_palette[i % len(hv_palette)], width=2 if w == 30 else 1.2),
                opacity=1.0 if w == 30 else 0.7,
                hovertemplate=f'{w}d HV: %{{y:.1f}}%<extra></extra>'
            ))
        fig_xdc_hv.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
            xaxis=dict(gridcolor=_PLT_GRID, title='Days (most recent right)'),
            yaxis=dict(gridcolor=_PLT_GRID, title='Annualised HV (%)'),
            legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
            height=320, margin=dict(t=10, b=40),
            title=dict(text=f'{_active_tok_vs} Rolling Historical Volatility', font=dict(size=11, color='#5a9abf'))
        )
        st.plotly_chart(fig_xdc_hv, use_container_width=True, key=_next_chart_key())

        # HV metrics cards
        hv_cards = st.columns(len(windows))
        for i, w in enumerate(windows):
            if w in hv_series and len(hv_series[w]) > 0:
                current_hv = hv_series[w][-1]
                with hv_cards[i]:
                    st.markdown(f'''<div class="metric-card" style="border-left-color:{hv_palette[i % len(hv_palette)]}">
                        <div class="metric-label">{w}D HV</div>
                        <div class="metric-value" style="font-size:18px;color:{hv_palette[i % len(hv_palette)]};">{current_hv:.1f}%</div>
                        <div class="metric-sub">Annualised</div>
                    </div>''', unsafe_allow_html=True)

        # Synthetic XDC vol term structure from historical windows
        st.markdown(f'<div class="section-header">{_active_tok_vs} SYNTHETIC VOL TERM STRUCTURE</div>', unsafe_allow_html=True)
        tenor_days_ts = [7, 14, 30, 60, 90, 180]
        xdc_hv_ts = []
        for w in tenor_days_ts:
            if w in hv_series and len(hv_series[w]) > 0:
                xdc_hv_ts.append(hv_series[w][-1])
            else:
                xdc_hv_ts.append(None)

        fig_xdc_ts = go.Figure()
        valid_tenors = [t for t, v in zip(tenor_days_ts, xdc_hv_ts) if v is not None]
        valid_hvs = [v for v in xdc_hv_ts if v is not None]
        if valid_tenors:
            fig_xdc_ts.add_trace(go.Scatter(
                x=valid_tenors, y=valid_hvs,
                name=f'{_active_tok_vs} HV Term Structure',
                line=dict(color='#00d4ff', width=3),
                mode='lines+markers', marker=dict(size=8, color='#00d4ff')
            ))
        fig_xdc_ts.update_layout(
            plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
            font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=10),
            xaxis=dict(gridcolor=_PLT_GRID, title='Tenor (days)'),
            yaxis=dict(gridcolor=_PLT_GRID, title='HV (%)'),
            height=260, margin=dict(t=10, b=40), showlegend=False
        )
        st.plotly_chart(fig_xdc_ts, use_container_width=True, key=_next_chart_key())
    else:
        st.info(f"Could not fetch {_active_tok_vs} price history for vol calculation.")

    # ── XDC VOL PROXY ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">{st.session_state.get("active_token_ticker","XDC")} IMPLIED VOL PROXY — FROM BTC/ETH SURFACE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:14px;line-height:1.8;">'
        'XDC has no listed options. This estimates XDC IV by interpolating the BTC and ETH '
        'surfaces at the same moneyness/tenor, then applying a <strong style="color:#f0a500">1.5× '
        'liquidity premium</strong> + wing steepening. Use this as your OTC negotiation anchor.</div>',
        unsafe_allow_html=True
    )

    # Compute proxy across a range of moneynesses for the chosen tenor
    mon_range    = np.arange(0.60, 1.61, 0.05)
    xdc_proxy_ivs = []
    btc_ref_ivs   = []
    eth_ref_ivs   = []

    for mon in mon_range:
        b_iv, e_iv, x_iv = estimate_xdc_iv(df_btc, df_eth, mon, xdc_T_days_vs)
        btc_ref_ivs.append(b_iv)
        eth_ref_ivs.append(e_iv)
        xdc_proxy_ivs.append(x_iv)

    fig_xdc_proxy = go.Figure()
    if any(v is not None for v in btc_ref_ivs):
        btc_clean = [v if v is not None else np.nan for v in btc_ref_ivs]
        fig_xdc_proxy.add_trace(go.Scatter(
            x=mon_range, y=btc_clean, name="BTC Surface",
            line=dict(color="#f0a500", width=1.5, dash="dot")
        ))
    if any(v is not None for v in eth_ref_ivs):
        eth_clean = [v if v is not None else np.nan for v in eth_ref_ivs]
        fig_xdc_proxy.add_trace(go.Scatter(
            x=mon_range, y=eth_clean, name="ETH Surface",
            line=dict(color="#b388ff", width=1.5, dash="dot")
        ))
    if any(v is not None for v in xdc_proxy_ivs):
        xdc_clean = [v if v is not None else np.nan for v in xdc_proxy_ivs]
        fig_xdc_proxy.add_trace(go.Scatter(
            x=mon_range, y=xdc_clean, name="XDC Proxy (1.5× premium)",
            line=dict(color="#00d4ff", width=3),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"
        ))

    # Mark ATM + user's chosen moneyness
    fig_xdc_proxy.add_vline(x=1.0, line_color="#ffffff", line_dash="dot",
                             annotation_text="ATM", annotation_font_color="#ffffff")
    fig_xdc_proxy.add_vline(x=xdc_moneyness_vs, line_color="#00e676", line_dash="dash",
                             annotation_text=f"Your strike ({xdc_moneyness_vs:.2f}×)",
                             annotation_font_color="#00e676")

    fig_xdc_proxy.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title="Moneyness (Strike / Spot)"),
        yaxis=dict(gridcolor=_PLT_GRID, title="Implied Vol (%)"),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR),
        height=340, margin=dict(t=10, b=50)
    )
    st.plotly_chart(fig_xdc_proxy, use_container_width=True, key=_next_chart_key())

    # Point estimate for user's chosen tenor + moneyness
    btc_pt, eth_pt, xdc_pt = estimate_xdc_iv(df_btc, df_eth, xdc_moneyness_vs, xdc_T_days_vs)

    proxy_cols = st.columns(4)
    def vs_card(label, val, sub, color):
        v = f"{val:.1f}%" if val is not None else "N/A"
        return f'''<div class="metric-card" style="border-left-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:20px;color:{color};">{v}</div>
            <div class="metric-sub">{sub}</div>
        </div>'''

    with proxy_cols[0]:
        st.markdown(vs_card("BTC IV (ref)", btc_pt,
            f"Mon={xdc_moneyness_vs:.2f}, T={xdc_T_days_vs}d", "#f0a500"), unsafe_allow_html=True)
    with proxy_cols[1]:
        st.markdown(vs_card("ETH IV (ref)", eth_pt,
            f"Mon={xdc_moneyness_vs:.2f}, T={xdc_T_days_vs}d", "#b388ff"), unsafe_allow_html=True)
    with proxy_cols[2]:
        st.markdown(vs_card("XDC IV PROXY", xdc_pt,
            "1.5× liquidity premium applied", "#00d4ff"), unsafe_allow_html=True)
    with proxy_cols[3]:
        # Price the option at proxy IV
        if xdc_pt is not None:
            proxy_sigma_frac = xdc_pt / 100
            K_proxy  = spot * xdc_moneyness_vs
            T_proxy  = xdc_T_days_vs / 365
            call_proxy = black_scholes(spot, K_proxy, T_proxy, r, proxy_sigma_frac, "call")
            put_proxy  = black_scholes(spot, K_proxy, T_proxy, r, proxy_sigma_frac, "put")
            st.markdown(f'''<div class="metric-card" style="border-left-color:#00e676">
                <div class="metric-label">XDC BS PRICE @ PROXY IV</div>
                <div style="font-size:13px;color:#00e676;margin-top:6px;">
                    Call: ${call_proxy:.6f} ({call_proxy/spot*100:.2f}% spot)<br>
                    Put:  ${put_proxy:.6f}  ({put_proxy/spot*100:.2f}% spot)
                </div>
                <div class="metric-sub">K={xdc_moneyness_vs:.2f}× spot, T={xdc_T_days_vs}d</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(vs_card("XDC BS PRICE", None, "Need BTC/ETH data", "#3d6080"),
                        unsafe_allow_html=True)

    # ── RAW DATA TABLE ────────────────────────────────────────────────
    with st.expander(f"📋  Raw {surface_asset} Option Data ({len(df_surface)} instruments)"):
        display_cols = ["instrument", "expiry_str", "T_days", "strike",
                        "option_type", "moneyness", "mark_iv", "bid_iv",
                        "ask_iv", "mark_price", "volume_usd", "open_interest",
                        "delta", "gamma", "vega", "theta"]
        display_cols = [c for c in display_cols if c in df_surface.columns]
        st.dataframe(
            df_surface[display_cols].sort_values(["T_days", "moneyness"]),
            use_container_width=True, hide_index=True
        )

    # ── METHODOLOGY NOTE ─────────────────────────────────────────────
    st.markdown('''
    <div style="background:#0a0e14;border:1px solid #1a2535;border-radius:3px;
                padding:16px 20px;margin-top:20px;font-size:11px;color:#5a7a99;line-height:1.9;">
        <div style="color:#00d4ff;font-family:Space Mono;font-size:12px;margin-bottom:10px;">
            DATA SOURCES & METHODOLOGY
        </div>
        <strong style="color:#f0a500;">Deribit API</strong> — Public endpoint, no API key required.
        All prices are mark prices (Deribit's fair value model). bid/ask IVs reflect the live order book.
        Data refreshes on demand (click Refresh).<br>
        <strong style="color:#b388ff;">DVOL Index</strong> — Deribit's proprietary 30-day forward-looking
        implied vol index for BTC and ETH. Analogous to VIX for crypto. Computed from the full options surface.<br>
        <strong style="color:#00d4ff;">XDC Proxy</strong> — No listed options exist for XDC.
        Proxy method: (1) interpolate BTC and ETH IV at same moneyness/tenor,
        (2) average the two, (3) apply 1.5× liquidity premium for micro-cap illiquidity,
        (4) apply wing steepening (OTM options get proportionally higher premium).
        Use the proxy as your OTC anchor — expect counterparties to quote 1.2–2.0× BTC ATM vol.<br>
        <strong style="color:#00e676;">Skew</strong> — 25-delta skew = OTM put IV minus OTM call IV.
        Positive skew = put premium (fear of downside). Crypto typically shows positive skew during risk-off.
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGY DEPOSITORY — FULL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

STRATEGY_DEPOSITORY = {

    # ── BULLISH ─────────────────────────────────────────────────────
    "Long Call": {
        "view": "bullish", "vol_bias": "long",
        "complexity": 1, "max_loss": "limited", "max_gain": "unlimited",
        "legs": [("call", "buy", "ATM", 1)],
        "description": "Buy a call option. Profits if asset rises above strike + premium paid. Classic directional bet.",
        "best_conditions": "Strong upside conviction. Low IV (cheap premium). Sufficient time.",
        "risks": "Full premium lost if price stays below strike at expiry.",
        "tags": ["directional", "simple", "debit"],
        "breakeven_note": "Strike + premium paid",
    },
    "Long Call Spread (Bull Call Spread)": {
        "view": "bullish", "vol_bias": "neutral",
        "complexity": 2, "max_loss": "limited", "max_gain": "limited",
        "legs": [("call", "buy", "ATM", 1), ("call", "sell", "+30%", 1)],
        "description": "Buy lower-strike call, sell higher-strike call. Reduces cost vs long call. Capped upside.",
        "best_conditions": "Moderately bullish. High IV environment (reduces net debit). Clear target price.",
        "risks": "Capped profit. Net debit still at risk.",
        "tags": ["directional", "debit", "defined-risk"],
        "breakeven_note": "Lower strike + net premium",
    },
    "Bull Put Spread": {
        "view": "bullish", "vol_bias": "short",
        "complexity": 2, "max_loss": "limited", "max_gain": "limited",
        "legs": [("put", "sell", "ATM", 1), ("put", "buy", "-20%", 1)],
        "description": "Sell higher-strike put, buy lower-strike put for protection. Net credit received.",
        "best_conditions": "Mildly bullish. High IV (sell rich premium). Comfortable owning asset at lower strike.",
        "risks": "Max loss = spread width minus credit. Assigned if price falls below short put.",
        "tags": ["income", "credit", "defined-risk"],
        "breakeven_note": "Short put strike - net credit",
    },
    "Cash-Secured Put": {
        "view": "bullish", "vol_bias": "short",
        "complexity": 1, "max_loss": "large", "max_gain": "limited",
        "legs": [("put", "sell", "ATM", 1)],
        "description": "Sell a put backed by cash to buy at strike. Earn premium; acquire asset at discount if assigned.",
        "best_conditions": "Willing to own token at strike price. High IV to maximise premium income.",
        "risks": "Must buy at strike even if price falls far below.",
        "tags": ["income", "acquisition", "credit"],
        "breakeven_note": "Strike - premium received",
    },
    "Covered Call": {
        "view": "mildly bullish", "vol_bias": "short",
        "complexity": 1, "max_loss": "large", "max_gain": "limited",
        "legs": [("call", "sell", "+20%", 1)],
        "description": "Own the token, sell a call above current price. Generate income. Cap upside at strike.",
        "best_conditions": "Already long the token. Neutral to mildly bullish. High IV to earn more.",
        "risks": "Token called away if price rises above strike. Loses all upside above strike.",
        "tags": ["income", "hedge", "credit"],
        "breakeven_note": "Purchase price - premium received",
    },
    "Call Ladder (Bull Call Ladder)": {
        "view": "bullish", "vol_bias": "short",
        "complexity": 3, "max_loss": "unlimited above top strike", "max_gain": "limited",
        "legs": [("call", "buy", "ATM", 1), ("call", "sell", "+30%", 1), ("call", "sell", "+50%", 1)],
        "description": "Buy ATM call, sell two higher calls. Net credit or low debit. Profits in range, risks at extreme rally.",
        "best_conditions": "Moderately bullish with price target. High IV. Want to reduce premium cost.",
        "risks": "Exposed to unlimited loss if price rallies past highest short strike.",
        "tags": ["income", "range", "advanced"],
        "breakeven_note": "Complex — two breakevens",
    },
    "LEAP Call": {
        "view": "bullish", "vol_bias": "long",
        "complexity": 1, "max_loss": "limited", "max_gain": "unlimited",
        "legs": [("call", "buy", "ATM", 1)],
        "description": "Long-dated call (6–18 months). Leveraged long exposure with time for thesis to play out.",
        "best_conditions": "High conviction long-term bull. Want leverage without margin.",
        "risks": "Significant premium decay if thesis is delayed. IV collapse risk.",
        "tags": ["directional", "long-dated", "leverage"],
        "breakeven_note": "Strike + premium",
    },

    # ── BEARISH ─────────────────────────────────────────────────────
    "Long Put": {
        "view": "bearish", "vol_bias": "long",
        "complexity": 1, "max_loss": "limited", "max_gain": "large",
        "legs": [("put", "buy", "ATM", 1)],
        "description": "Buy a put option. Profits if asset falls below strike. Direct downside bet or portfolio hedge.",
        "best_conditions": "Bearish conviction. Low IV (cheap puts). Catalyst expected.",
        "risks": "Full premium lost if price stays above strike.",
        "tags": ["directional", "hedge", "debit"],
        "breakeven_note": "Strike - premium paid",
    },
    "Bear Put Spread": {
        "view": "bearish", "vol_bias": "neutral",
        "complexity": 2, "max_loss": "limited", "max_gain": "limited",
        "legs": [("put", "buy", "ATM", 1), ("put", "sell", "-30%", 1)],
        "description": "Buy higher-strike put, sell lower-strike put. Reduced cost, capped profit.",
        "best_conditions": "Moderately bearish with price target. Cost-effective downside.",
        "risks": "Capped downside capture. Net debit at risk.",
        "tags": ["directional", "debit", "defined-risk"],
        "breakeven_note": "Higher strike - net premium",
    },
    "Bear Call Spread": {
        "view": "bearish", "vol_bias": "short",
        "complexity": 2, "max_loss": "limited", "max_gain": "limited",
        "legs": [("call", "sell", "ATM", 1), ("call", "buy", "+30%", 1)],
        "description": "Sell lower-strike call, buy higher-strike call. Net credit. Profit if asset stays below short strike.",
        "best_conditions": "Mildly bearish. High IV to sell premium. Expect sideways-to-down.",
        "risks": "Max loss = spread width minus credit. Loss if price rallies past long strike.",
        "tags": ["income", "credit", "defined-risk"],
        "breakeven_note": "Short strike + net credit",
    },
    "Put Backspread": {
        "view": "bearish", "vol_bias": "long",
        "complexity": 3, "max_loss": "limited", "max_gain": "unlimited",
        "legs": [("put", "sell", "ATM", 1), ("put", "buy", "-20%", 2)],
        "description": "Sell 1 ATM put, buy 2 OTM puts. Profits from sharp drop. Small loss in middle zone.",
        "best_conditions": "Expecting large downside move. Crash hedge. Low IV on OTM puts.",
        "risks": "Maximum loss if price finishes near short put strike.",
        "tags": ["crash-hedge", "advanced", "long-vol"],
        "breakeven_note": "Two breakevens",
    },
    "Put Ladder (Bear Put Ladder)": {
        "view": "bearish", "vol_bias": "short",
        "complexity": 3, "max_loss": "unlimited below bottom strike", "max_gain": "limited",
        "legs": [("put", "buy", "ATM", 1), ("put", "sell", "-20%", 1), ("put", "sell", "-40%", 1)],
        "description": "Buy ATM put, sell two lower puts. Low/no-cost entry. Profits in range, risk at extreme crash.",
        "best_conditions": "Moderate downside expected. High put skew makes sold puts expensive.",
        "risks": "Unlimited loss if price crashes through lowest short strike.",
        "tags": ["income", "range", "advanced"],
        "breakeven_note": "Complex — two breakevens",
    },

    # ── NEUTRAL ─────────────────────────────────────────────────────
    "Short Straddle": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 2, "max_loss": "unlimited", "max_gain": "limited",
        "legs": [("call", "sell", "ATM", 1), ("put", "sell", "ATM", 1)],
        "description": "Sell ATM call + put. Profit if price stays near strike. Earn maximum theta decay.",
        "best_conditions": "Very high IV (overpriced options). Expect price to stay flat. Post-event.",
        "risks": "Unlimited loss in either direction. Best for experienced traders only.",
        "tags": ["income", "short-vol", "advanced"],
        "breakeven_note": "Strike ± total premium",
    },
    "Short Strangle": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 2, "max_loss": "unlimited", "max_gain": "limited",
        "legs": [("call", "sell", "+20%", 1), ("put", "sell", "-20%", 1)],
        "description": "Sell OTM call + OTM put. Wider profit zone than straddle. Lower premium collected.",
        "best_conditions": "High IV. Expect moderate range-bound action. More room for error than straddle.",
        "risks": "Unlimited loss if price moves sharply in either direction.",
        "tags": ["income", "short-vol", "advanced"],
        "breakeven_note": "Short call + premium received / Short put - premium received",
    },
    "Iron Condor": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 4, "max_loss": "limited", "max_gain": "limited",
        "legs": [
            ("put", "buy", "-40%", 1), ("put", "sell", "-20%", 1),
            ("call", "sell", "+20%", 1), ("call", "buy", "+40%", 1)
        ],
        "description": "Sell OTM put spread + sell OTM call spread. Profit in range. Defined risk on both sides.",
        "best_conditions": "High IV. Expect tight range. Classic earnings-style structure.",
        "risks": "Max loss if price moves past either long wing.",
        "tags": ["income", "range", "defined-risk", "classic"],
        "breakeven_note": "Four breakevens — one at each wing",
    },
    "Iron Butterfly": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 4, "max_loss": "limited", "max_gain": "limited",
        "legs": [
            ("put", "buy", "-30%", 1), ("put", "sell", "ATM", 1),
            ("call", "sell", "ATM", 1), ("call", "buy", "+30%", 1)
        ],
        "description": "Short straddle + protective wings. Higher premium than condor, narrower profit zone.",
        "best_conditions": "Very high IV. Very tight price prediction. Short-dated.",
        "risks": "Narrow profit zone. Loss if price moves significantly in either direction.",
        "tags": ["income", "range", "defined-risk"],
        "breakeven_note": "ATM ± net credit",
    },
    "Long Butterfly": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 3, "max_loss": "limited", "max_gain": "limited",
        "legs": [
            ("call", "buy", "-20%", 1), ("call", "sell", "ATM", 2), ("call", "buy", "+20%", 1)
        ],
        "description": "Buy wings, sell 2 ATM calls. Cheap structure. Maximum profit exactly at middle strike.",
        "best_conditions": "Very low debit, high precision on price target. Low IV.",
        "risks": "Very narrow profit zone. Full debit at risk.",
        "tags": ["debit", "range", "low-cost"],
        "breakeven_note": "Lower strike + debit / Upper strike - debit",
    },
    "Calendar Spread": {
        "view": "neutral", "vol_bias": "long near / short far",
        "complexity": 3, "max_loss": "limited", "max_gain": "limited",
        "legs": [("call", "sell", "ATM", 1), ("call", "buy", "ATM farther expiry", 1)],
        "description": "Sell near-dated option, buy far-dated same strike. Profit from faster time decay of near leg.",
        "best_conditions": "Expect muted near-term move but vol to rise longer-term. Vol term structure steep.",
        "risks": "Large move before near expiry hurts. Vol term structure flattening.",
        "tags": ["time-decay", "vol-term-structure", "advanced"],
        "breakeven_note": "Depends on vol at near expiry",
    },
    "Ratio Spread": {
        "view": "neutral-to-bullish", "vol_bias": "short",
        "complexity": 3, "max_loss": "unlimited above top strike", "max_gain": "limited",
        "legs": [("call", "buy", "ATM", 1), ("call", "sell", "+20%", 2)],
        "description": "Buy 1 call, sell 2 higher calls. Net credit or low cost. Risk above top short strike.",
        "best_conditions": "Mildly bullish. High IV (sell expensive). Clear ceiling price target.",
        "risks": "Unlimited upside risk above top short strike if not managed.",
        "tags": ["income", "range", "advanced"],
        "breakeven_note": "Lower strike + premium / Upper strike complex",
    },

    # ── LONG VOLATILITY ─────────────────────────────────────────────
    "Long Straddle": {
        "view": "neutral / high vol", "vol_bias": "long",
        "complexity": 2, "max_loss": "limited", "max_gain": "unlimited",
        "legs": [("call", "buy", "ATM", 1), ("put", "buy", "ATM", 1)],
        "description": "Buy ATM call + put. Profit from large move in either direction. Classic vol play.",
        "best_conditions": "Very low IV (cheap). Pre-catalyst (listing, partnership announcement). Expect big move.",
        "risks": "Lose both premiums if price stays flat (theta decay).",
        "tags": ["long-vol", "event", "debit"],
        "breakeven_note": "Strike ± total premium",
    },
    "Long Strangle": {
        "view": "neutral / high vol", "vol_bias": "long",
        "complexity": 2, "max_loss": "limited", "max_gain": "unlimited",
        "legs": [("call", "buy", "+20%", 1), ("put", "buy", "-20%", 1)],
        "description": "Buy OTM call + OTM put. Cheaper than straddle. Needs bigger move to profit.",
        "best_conditions": "Very low IV. Cheap OTM options. Expect explosive move but direction unknown.",
        "risks": "Both premiums at risk. Needs larger move than straddle to profit.",
        "tags": ["long-vol", "event", "debit"],
        "breakeven_note": "Call strike + total premium / Put strike - total premium",
    },
    "Volatility Collar": {
        "view": "neutral / risk management", "vol_bias": "neutral",
        "complexity": 2, "max_loss": "limited", "max_gain": "limited",
        "legs": [("put", "buy", "-20%", 1), ("call", "sell", "+20%", 1)],
        "description": "Buy protective put, sell covered call. Zero or low cost hedge. Cap gains, floor losses.",
        "best_conditions": "Own the token. Want downside protection. Willing to sacrifice upside above call.",
        "risks": "Misses upside rally above short call. Net debit if collar costs money.",
        "tags": ["hedge", "risk-management", "zero-cost"],
        "breakeven_note": "Entry price +/- net premium",
    },

    # ── STRUCTURED / OTC-SPECIFIC ────────────────────────────────────
    "Snowball (Autocallable)": {
        "view": "mildly bullish / neutral", "vol_bias": "short",
        "complexity": 5, "max_loss": "large (put exposure)", "max_gain": "limited (coupons)",
        "legs": [("custom", "structured", "barrier", 1)],
        "description": "Earn periodic coupons if asset stays above barrier. Auto-calls when price exceeds trigger. Embedded short put at barrier.",
        "best_conditions": "High IV (earn large coupons). Expect mild upward drift or sideways. OTC deal.",
        "risks": "If price breaches barrier, full downside exposure like owning asset at strike.",
        "tags": ["structured", "OTC", "income", "crypto-native"],
        "breakeven_note": "Depends on coupon vs drawdown",
    },
    "Dual Currency Deposit (DCD)": {
        "view": "neutral / mildly bearish on one asset", "vol_bias": "short",
        "complexity": 3, "max_loss": "large", "max_gain": "limited (yield)",
        "legs": [("put", "sell", "ATM", 1)],
        "description": "Deposit currency X, earn enhanced yield. If asset falls below strike, receive asset Y instead of X. Embedded short put.",
        "best_conditions": "High IV (high yield). Comfortable receiving XDC if price falls. Short tenor (7–30d).",
        "risks": "Receive depreciating asset if price falls. Opportunity cost if price rallies.",
        "tags": ["structured", "OTC", "yield", "crypto-native"],
        "breakeven_note": "Deposit × (1 + yield) vs asset price at maturity",
    },
    "Range Accrual": {
        "view": "neutral", "vol_bias": "short",
        "complexity": 5, "max_loss": "premium", "max_gain": "limited",
        "legs": [("custom", "structured", "range-accrual", 1)],
        "description": "Earn daily coupon for each day asset stays within predefined range. Accrues like daily binary.",
        "best_conditions": "Very high realized vol but expect consolidation. OTC deal. Post-rally settling.",
        "risks": "Zero payout if price exits range for most of the period.",
        "tags": ["structured", "OTC", "range", "advanced"],
        "breakeven_note": "N/A — depends on accrual fraction",
    },
    "Accumulator": {
        "view": "bullish", "vol_bias": "short",
        "complexity": 5, "max_loss": "unlimited", "max_gain": "limited",
        "legs": [("custom", "structured", "accumulator", 1)],
        "description": "Contractually buy fixed amount of asset below market daily. Doubles if price drops below knock-in. OTC deal.",
        "best_conditions": "Very bullish. Want to accumulate at discount. High IV environment.",
        "risks": "Forced to buy large amounts if price crashes. Can be margin-intensive.",
        "tags": ["structured", "OTC", "accumulation", "advanced"],
        "breakeven_note": "Average of daily forward price vs spot",
    },
    "Knock-In Put": {
        "view": "bearish / crash hedge", "vol_bias": "long",
        "complexity": 4, "max_loss": "premium", "max_gain": "large",
        "legs": [("put", "buy", "ATM", 1)],
        "description": "Put that only activates if asset first falls to knock-in barrier. Cheaper than vanilla put.",
        "best_conditions": "Cheap crash protection. Expect either stability or large drop — not middle.",
        "risks": "Worthless if price falls moderately but never hits barrier.",
        "tags": ["exotic", "OTC", "hedge", "barrier"],
        "breakeven_note": "Strike - premium (only if activated)",
    },
    "Risk Reversal": {
        "view": "bullish (OTM call) or bearish (OTM put)", "vol_bias": "neutral",
        "complexity": 2, "max_loss": "moderate", "max_gain": "large",
        "legs": [("put", "sell", "-20%", 1), ("call", "buy", "+20%", 1)],
        "description": "Sell OTM put to finance OTM call. Zero-cost structure. Net long upside, net short downside.",
        "best_conditions": "Bullish but want zero-cost. Put skew elevated (sell expensive put).",
        "risks": "Exposed to full downside below short put if price crashes.",
        "tags": ["zero-cost", "directional", "OTC"],
        "breakeven_note": "Call strike (upside) / Put strike (downside)",
    },
}


def score_strategies(view, confidence, hv, pcr, T_days, spot, sigma, is_otc_ok=True):
    """
    Score all strategies 0–100 based on:
    - View alignment (bullish/bearish/neutral)
    - Confidence level
    - Implied vol regime (HV vs normal)
    - Put/call ratio (skew signal)
    - Time to expiry
    - OTC suitability
    Returns sorted list of (strategy_name, score, reasoning).
    """
    results = []
    # confidence is already 0-1 (pre-scaled by caller with conviction × scalar)
    conf = max(0.0, min(1.0, float(confidence)))

    # Vol regime
    vol_high  = hv > 0.70   # high IV → prefer short-vol structures
    vol_low   = hv < 0.40   # low IV  → prefer long-vol / debit
    vol_med   = not vol_high and not vol_low

    # Skew signal
    put_skew_high  = pcr > 1.3
    put_skew_low   = pcr < 0.8

    # Tenor
    short_dated = T_days <= 30
    long_dated  = T_days >= 90

    for name, strat in STRATEGY_DEPOSITORY.items():
        score  = 0
        reasons = []

        # 1. VIEW ALIGNMENT (0–40 pts)
        sview = strat["view"].lower()
        if view == "bullish":
            if "bullish" in sview:
                score += 35 * conf
                reasons.append(f"View aligned ({strat['view']})")
            elif "neutral" in sview:
                score += 15 * conf
                reasons.append("Neutral strategy — partial fit for bullish view")
            elif "bearish" in sview:
                score -= 30 * conf
                reasons.append("View misaligned (bearish strategy)")
            if "high vol" in sview:
                score += 10 * conf
        elif view == "bearish":
            if "bearish" in sview:
                score += 35 * conf
                reasons.append(f"View aligned ({strat['view']})")
            elif "neutral" in sview:
                score += 15 * conf
                reasons.append("Neutral strategy — partial fit for bearish view")
            elif "bullish" in sview:
                score -= 30 * conf
                reasons.append("View misaligned (bullish strategy)")
        elif view == "neutral":
            if "neutral" in sview:
                score += 35 * conf
                reasons.append(f"View aligned ({strat['view']})")
            elif "bullish" in sview or "bearish" in sview:
                score -= 10 * conf
                reasons.append("Directional strategy — suboptimal for neutral view")
            if "high vol" in sview:
                score += 10 * conf

        # 2. VOL REGIME (0–25 pts)
        vbias = strat["vol_bias"].lower()
        if vol_high:
            if "short" in vbias:
                score += 25
                reasons.append("✅ High IV — ideal for short-vol strategies")
            elif "long" in vbias:
                score -= 15
                reasons.append("⚠️ High IV makes long-vol expensive")
            else:
                score += 10
        elif vol_low:
            if "long" in vbias:
                score += 25
                reasons.append("✅ Low IV — ideal for long-vol / debit strategies")
            elif "short" in vbias:
                score -= 10
                reasons.append("⚠️ Low IV reduces income from short-vol")
            else:
                score += 10
        else:
            score += 12  # neutral vol — most strategies ok
            reasons.append("Moderate IV — most strategies viable")

        # 3. PUT/CALL RATIO / SKEW (0–15 pts)
        if put_skew_high:
            if "zero-cost" in strat["tags"] or "risk reversal" in name.lower():
                score += 15
                reasons.append("✅ High put skew — sell expensive puts to finance calls")
            elif "income" in strat["tags"] and "put" in str(strat["legs"]):
                score += 10
                reasons.append("High put skew — good time to sell puts")
        if put_skew_low:
            if "long" in vbias and "call" in str(strat["legs"]):
                score += 10
                reasons.append("Low put skew — calls relatively cheap")

        # 4. TENOR FIT (0–10 pts)
        if short_dated:
            if "short-vol" in strat["tags"] or "income" in strat["tags"]:
                score += 10
                reasons.append("✅ Short tenor favours theta-positive strategies")
            elif "long-dated" in strat["tags"]:
                score -= 8
                reasons.append("⚠️ LEAP / long-dated structure not suited to short expiry")
        elif long_dated:
            if "long-dated" in strat["tags"] or "long-vol" in strat["tags"]:
                score += 10
                reasons.append("✅ Long tenor — long-vol and LEAP strategies benefit")
            elif "short-vol" in strat["tags"]:
                score += 5

        # 5. COMPLEXITY PENALTY FOR LOW CONFIDENCE (–5 to 0)
        if conf < 0.5 and strat["complexity"] >= 4:
            score -= 8
            reasons.append("Complexity penalty: low confidence → avoid exotic structures")

        # 6. OTC SUITABILITY
        if not is_otc_ok and "OTC" in strat["tags"]:
            score -= 20
            reasons.append("⚠️ OTC structure requires bilateral agreement")
        elif is_otc_ok and "OTC" in strat["tags"]:
            score += 8
            reasons.append("OTC deal viable — structured product applicable")

        # 7. CRYPTO-SPECIFIC BOOST
        if "crypto-native" in strat["tags"]:
            score += 5
            reasons.append("Crypto-native structure — well-suited to token markets")

        # Clamp 0–100
        score = max(0, min(100, int(score)))
        results.append((name, score, reasons, strat))

    return sorted(results, key=lambda x: -x[1])


def build_strategy_payoff(strat_name, strat_def, S, sigma, T, r, n_points=300):
    """
    Build payoff curve for a strategy at expiry.
    Returns (price_range, payoff, net_premium, legs_priced).
    """
    price_range = np.linspace(S * 0.30, S * 2.50, n_points)

    # Map strike labels to actual prices
    strike_map = {
        "ATM":   S,
        "+10%":  S * 1.10, "+20%": S * 1.20, "+30%": S * 1.30,
        "+40%":  S * 1.40, "+50%": S * 1.50,
        "-10%":  S * 0.90, "-20%": S * 0.80, "-30%": S * 0.70,
        "-40%":  S * 0.60, "-50%": S * 0.50,
        "barrier": S * 0.85, "range-accrual": S, "accumulator": S,
        "ATM farther expiry": S,
    }

    legs         = strat_def["legs"]
    payoff       = np.zeros(n_points)
    net_premium  = 0.0
    legs_priced  = []

    for opt_type, direction, strike_label, qty in legs:
        if opt_type == "custom":
            continue  # structured products — skip payoff calc
        K = strike_map.get(strike_label, S)
        prem = black_scholes(S, K, T, r, sigma, opt_type)
        sign = 1 if direction == "buy" else -1
        net_premium += sign * prem * qty

        if opt_type == "call":
            intrinsic = np.maximum(price_range - K, 0)
        else:
            intrinsic = np.maximum(K - price_range, 0)

        payoff += sign * qty * (intrinsic - prem)
        legs_priced.append({
            "type": opt_type, "direction": direction,
            "strike_label": strike_label, "K": K,
            "premium": prem, "qty": qty
        })

    return price_range, payoff, net_premium, legs_priced


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 10 — STRATEGY LAB
#  Full depository + view/confidence/style toggles + scored recs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab10:

    # ── Section header ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🧬  STRATEGY LAB — DEPOSITORY & SCORED RECOMMENDATIONS</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:11px;color:#5a7a99;margin-bottom:18px;line-height:1.9;">'
        f'{len(STRATEGY_DEPOSITORY)} strategies catalogued across directional, neutral, long-vol, short-vol, '
        f'and structured/OTC categories. Set your view, conviction, and token below — the scoring engine '
        f'ranks every strategy in real time.</div>',
        unsafe_allow_html=True
    )

    # ── CONTROL PANEL ───────────────────────────────────────────────
    st.markdown('<div class="section-header">YOUR PARAMETERS</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])

    with ctrl1:
        sl_token = st.selectbox(
            "Token / Asset",
            ["XDC", "BTC", "ETH", "SOL", "HYPE", "Custom"],
            key="sl_token_lab"
        )
        if sl_token == "Custom":
            sl_token = st.text_input("Ticker", value="XDC", key="sl_tok_custom").upper()

        sl_view = st.select_slider(
            "Market View",
            options=["Strongly Bearish", "Mildly Bearish", "Neutral", "Mildly Bullish", "Strongly Bullish"],
            value="Mildly Bullish",
            key="sl_view_lab"
        )

    with ctrl2:
        sl_conf = st.slider(
            "Conviction (%)", 10, 100, 65, 5, key="sl_conf_lab",
            help="How sure are you? Scores high-complexity strategies down when conviction is low."
        )
        sl_style = st.radio(
            "Strategy Style",
            ["All", "Vanilla Only", "Exotic / OTC Only"],
            horizontal=True, key="sl_style_lab"
        )

    with ctrl3:
        sl_hv_lab  = st.number_input(
            "Current HV / Implied Vol (%)",
            value=float(sigma * 100), min_value=1.0, max_value=500.0,
            key="sl_hv_lab"
        ) / 100
        sl_T_lab   = st.slider("Desired Tenor (days)", 7, 365, 30, key="sl_T_lab")
        sl_notional_lab = st.number_input(
            "Notional (USD)", value=100_000, step=10_000, key="sl_notional_lab"
        )

    with ctrl4:
        sl_pcr_lab = st.number_input(
            "Put/Call Ratio",
            value=round(float(put_prices[3] / call_prices[3]), 3) if call_prices[3] > 0 else 1.0,
            format="%.3f", key="sl_pcr_lab",
            help="ATM P/C ratio. >1.2 = bearish skew, <0.8 = bullish skew"
        )
        sl_otc_lab = st.checkbox("Include OTC / Structured Products", value=True, key="sl_otc_lab")
        sl_topn    = st.slider("Top N to show", 3, len(STRATEGY_DEPOSITORY), 8, key="sl_topn_lab")
        sl_min_sc  = st.slider("Min score threshold", 0, 80, 0, 5, key="sl_minscore_lab")

    # ── View-to-score-engine mapping ─────────────────────────────────
    view_engine_map = {
        "Strongly Bullish": "bullish",
        "Mildly Bullish":   "bullish",
        "Neutral":          "neutral",
        "Mildly Bearish":   "bearish",
        "Strongly Bearish": "bearish",
    }
    # Confidence scalar: Strongly = 1.0, Mildly = 0.75
    conf_scalar_map = {
        "Strongly Bullish": 1.0,
        "Mildly Bullish":   0.75,
        "Neutral":          0.85,
        "Mildly Bearish":   0.75,
        "Strongly Bearish": 1.0,
    }
    view_engine  = view_engine_map[sl_view]
    conf_engine  = (sl_conf / 100) * conf_scalar_map[sl_view]

    # Colour palette for view
    view_colors = {
        "Strongly Bullish": "#00e676",
        "Mildly Bullish":   "#44cc88",
        "Neutral":          "#f0a500",
        "Mildly Bearish":   "#ff8a65",
        "Strongly Bearish": "#ff4b6e",
    }
    vc = view_colors[sl_view]

    # ── View summary banner ──────────────────────────────────────────
    vol_label   = "HIGH" if sl_hv_lab > 0.70 else "LOW" if sl_hv_lab < 0.40 else "MODERATE"
    vol_color   = "#ff4b6e" if sl_hv_lab > 0.70 else "#00e676" if sl_hv_lab < 0.40 else "#f0a500"
    pcr_label   = "BEARISH SKEW" if sl_pcr_lab > 1.2 else "BULLISH SKEW" if sl_pcr_lab < 0.8 else "BALANCED"
    pcr_color   = "#ff4b6e" if sl_pcr_lab > 1.2 else "#00e676" if sl_pcr_lab < 0.8 else "#f0a500"

    st.markdown(f'''
    <div style="background:linear-gradient(135deg,#0d1a14,#091008);
                border:1px solid #1e2d40;border-left:5px solid {vc};
                border-radius:5px;padding:16px 24px;margin:16px 0;
                display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:10px;letter-spacing:3px;color:#3d6080;">CURRENT VIEW</div>
            <div style="font-family:Space Mono;font-size:26px;color:{vc};margin-top:4px;">{sl_view.upper()}</div>
            <div style="font-size:11px;color:#5a7a99;margin-top:6px;">
                Token: <span style="color:#00d4ff;font-weight:bold;">{sl_token}</span>&nbsp;·&nbsp;
                Conviction: <span style="color:{vc};font-weight:bold;">{sl_conf}%</span>&nbsp;·&nbsp;
                Vol: <span style="color:{vol_color};font-weight:bold;">{sl_hv_lab:.0%} ({vol_label})</span>&nbsp;·&nbsp;
                Tenor: <span style="color:#b388ff;font-weight:bold;">{sl_T_lab}d</span>&nbsp;·&nbsp;
                PCR: <span style="color:{pcr_color};font-weight:bold;">{sl_pcr_lab:.2f} ({pcr_label})</span>
            </div>
        </div>
        <div style="text-align:right;font-size:11px;color:#3d6080;line-height:2.0;">
            Style: <span style="color:#00d4ff">{sl_style}</span><br>
            Notional: <span style="color:#5a9abf">${sl_notional_lab:,}</span><br>
            OTC: <span style="color:{'#00e676' if sl_otc_lab else '#ff4b6e'}">{'Yes' if sl_otc_lab else 'No'}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── RUN SCORING ENGINE ───────────────────────────────────────────
    raw_results = score_strategies(
        view_engine, conf_engine, sl_hv_lab, sl_pcr_lab,
        sl_T_lab, spot, sl_hv_lab, is_otc_ok=sl_otc_lab
    )

    # Apply style filter
    def passes_style(name, sdef):
        is_exotic = "OTC" in sdef.get("tags", []) or \
                    "structured" in sdef.get("tags", []) or \
                    "exotic" in sdef.get("tags", []) or \
                    "barrier" in sdef.get("tags", [])
        if sl_style == "Vanilla Only" and is_exotic:
            return False
        if sl_style == "Exotic / OTC Only" and not is_exotic:
            return False
        return True

    filtered = [
        (name, sc, reasons, sdef)
        for name, sc, reasons, sdef in raw_results
        if passes_style(name, sdef) and sc >= sl_min_sc
    ]

    top_results = filtered[:sl_topn]

    if not top_results:
        st.warning("No strategies pass the current filters. Try lowering the minimum score or changing the style filter.")
        st.stop()

    # ── SCORE LEADERBOARD ────────────────────────────────────────────
    st.markdown('<div class="section-header">SCORE LEADERBOARD</div>', unsafe_allow_html=True)

    bar_names   = [r[0] for r in top_results]
    bar_scores  = [r[1] for r in top_results]
    bar_views   = [r[3]["view"] for r in top_results]

    def bar_color(sc):
        if sc >= 70: return "#00e676"
        if sc >= 45: return "#f0a500"
        return "#ff4b6e"

    fig_board = go.Figure(go.Bar(
        x=bar_scores[::-1],
        y=bar_names[::-1],
        orientation="h",
        marker_color=[bar_color(s) for s in bar_scores[::-1]],
        marker_line_width=0,
        text=[f"  {s}  |  {v}" for s, v in zip(bar_scores[::-1], bar_views[::-1])],
        textposition="outside",
        textfont=dict(size=10, color="#8fb3d0"),
    ))
    fig_board.add_vline(x=70, line_color="#00e676", line_dash="dot",
                        annotation_text="Strong ≥70", annotation_font_color="#00e676",
                        annotation_font_size=9)
    fig_board.add_vline(x=45, line_color="#f0a500", line_dash="dot",
                        annotation_text="OK ≥45", annotation_font_color="#f0a500",
                        annotation_font_size=9)
    fig_board.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, range=[0, 120], title="Score / 100"),
        yaxis=dict(gridcolor=_PLT_GRID),
        height=max(280, len(top_results) * 40),
        margin=dict(t=10, b=40, l=10, r=130),
        showlegend=False
    )
    st.plotly_chart(fig_board, use_container_width=True, key=_next_chart_key())

    # ── TOP RECOMMENDATION SPOTLIGHT ────────────────────────────────
    top_name, top_sc, top_reasons, top_def = top_results[0]
    top_color = bar_color(top_sc)

    # Compute indicative prices for top strategy
    T_yrs = sl_T_lab / 365
    c_atm = black_scholes(spot, spot, T_yrs, r, sl_hv_lab, "call")
    p_atm = black_scholes(spot, spot, T_yrs, r, sl_hv_lab, "put")

    price_range_top, payoff_top, net_prem_top, legs_top = build_strategy_payoff(
        top_name, top_def, spot, sl_hv_lab, T_yrs, r
    )

    rec_cols = st.columns([3, 2])
    with rec_cols[0]:
        st.markdown(f'''
        <div style="background:linear-gradient(135deg,#091408,#060e06);
                    border:1px solid #1a3020;border-left:5px solid {top_color};
                    border-radius:6px;padding:20px 26px;">
            <div style="font-size:10px;letter-spacing:3px;color:#3d6080;margin-bottom:8px;">
                🏆  TOP RECOMMENDATION FOR {sl_token}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div style="font-family:Space Mono;font-size:24px;color:{top_color};">{top_name}</div>
                    <div style="font-size:11px;color:#5a7a99;margin-top:4px;">
                        View: <span style="color:#f0a500">{top_def["view"]}</span> ·
                        Vol bias: <span style="color:#b388ff">{top_def["vol_bias"]}</span> ·
                        Complexity: <span style="color:#00d4ff">{"●" * top_def["complexity"]}{"○" * (5 - top_def["complexity"])}</span>
                    </div>
                </div>
                <div style="font-family:Space Mono;font-size:48px;color:{top_color};line-height:1.0;">
                    {top_sc}<span style="font-size:16px;color:#3d6080;">/100</span>
                </div>
            </div>
            <div style="font-size:12px;color:#8fb3d0;margin-top:14px;line-height:1.9;">
                {top_def["description"]}
            </div>
            <div style="margin-top:12px;font-size:11px;color:#5a7a99;line-height:2.1;
                        border-top:1px solid #1a2535;padding-top:10px;">
                <strong style="color:#3d6080;">Best conditions:</strong> {top_def["best_conditions"]}<br>
                <strong style="color:#3d6080;">Key risks:</strong> {top_def["risks"]}<br>
                <strong style="color:#3d6080;">Breakeven:</strong> <span style="color:#f0a500">{top_def["breakeven_note"]}</span><br>
                <strong style="color:#3d6080;">Max profit:</strong> <span style="color:#00e676">{top_def["max_gain"]}</span> ·
                <strong style="color:#3d6080;">Max loss:</strong> <span style="color:#ff4b6e">{top_def["max_loss"]}</span>
            </div>
            <div style="margin-top:12px;font-size:10px;color:#3d6080;border-top:1px solid #1a2535;padding-top:8px;">
                WHY NOW: &nbsp;{'&nbsp; · &nbsp;'.join(f'<span style="color:#5a9abf">{rr}</span>' for rr in top_reasons[:4])}
            </div>
            <div style="margin-top:10px;font-size:10px;">
                {''.join(f'<span style="background:#1a2535;color:#5a9abf;padding:2px 8px;margin-right:6px;border-radius:2px;">{t}</span>' for t in top_def.get("tags", []))}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with rec_cols[1]:
        # Payoff diagram for top recommendation
        if np.any(payoff_top != 0):
            fig_top_pay = go.Figure()
            fig_top_pay.add_trace(go.Scatter(
                x=price_range_top, y=payoff_top,
                fill="tozeroy",
                fillcolor=f"rgba({int(top_color[1:3],16)},{int(top_color[3:5],16)},{int(top_color[5:7],16)},0.07)",
                line=dict(color=top_color, width=2.5),
                hovertemplate="Spot: $%{x:.5f}<br>P&L: $%{y:.6f}<extra></extra>"
            ))
            fig_top_pay.add_hline(y=0, line_color="#3d6080", line_dash="dot")
            fig_top_pay.add_vline(x=spot, line_color="#ffffff", line_dash="dash",
                                  annotation_text="SPOT", annotation_font_color="#ffffff",
                                  annotation_font_size=9)
            if net_prem_top != 0:
                fig_top_pay.add_hline(y=-abs(net_prem_top), line_color="#ff4b6e",
                                      line_dash="dot", opacity=0.5)
            fig_top_pay.update_layout(
                plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=9),
                xaxis=dict(gridcolor=_PLT_GRID, title="Spot at Expiry ($)", tickformat=".5f"),
                yaxis=dict(gridcolor=_PLT_GRID, title="P&L per unit ($)"),
                height=320, showlegend=False,
                margin=dict(t=20, b=50, l=10, r=10)
            )
            st.plotly_chart(fig_top_pay, use_container_width=True, key=_next_chart_key())

            # Net premium card
            if legs_top:
                st.markdown(f'''
                <div style="background:#0a0c10;border:1px solid #1a2535;border-radius:3px;padding:12px 16px;font-size:11px;">
                    <div style="color:#3d6080;letter-spacing:2px;font-size:10px;margin-bottom:8px;">INDICATIVE COST</div>
                    {''.join(
                        f'<div style="color:#5a7a99;">'
                        f'{lg["direction"].upper()} {lg["qty"]}× {lg["type"].upper()} @ {lg["strike_label"]} '
                        f'= <span style="color:{("#ff4b6e" if lg["direction"]=="buy" else "#00e676")}">'
                        f'{"−" if lg["direction"]=="buy" else "+"}'
                        f'${lg["premium"]:.6f}</span></div>'
                        for lg in legs_top
                    )}
                    <div style="border-top:1px solid #1a2535;margin-top:8px;padding-top:8px;">
                        Net: <span style="color:{("#ff4b6e" if net_prem_top < 0 else "#00e676")};font-size:13px;">
                        {"−" if net_prem_top < 0 else "+"}'${abs(net_prem_top):.6f}</span>
                        <span style="color:#3d6080;"> per unit</span><br>
                        USD impact on ${sl_notional_lab:,}:
                        <span style="color:{("#ff4b6e" if net_prem_top < 0 else "#00e676")};">
                        {"−" if net_prem_top < 0 else "+"}${abs(net_prem_top) * sl_notional_lab / spot:,.0f}
                        </span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

    # ── TOP N STRATEGY CARDS ─────────────────────────────────────────
    st.markdown('<div class="section-header">ALL RANKED STRATEGIES — DETAILED BREAKDOWN</div>',
                unsafe_allow_html=True)

    medals = ["🥇", "🥈", "🥉"] + [f"#{i+1}" for i in range(3, 20)]

    for idx, (name, sc, reasons, sdef) in enumerate(top_results):
        sc_c = bar_color(sc)
        is_exotic_tag = any(t in sdef.get("tags", [])
                            for t in ["OTC", "structured", "exotic", "barrier"])
        cat_badge = (
            f'<span style="background:#1e1230;color:#e040fb;padding:1px 7px;'
            f'border-radius:2px;font-size:9px;">EXOTIC/OTC</span>'
            if is_exotic_tag else
            f'<span style="background:#0d1a28;color:#00d4ff;padding:1px 7px;'
            f'border-radius:2px;font-size:9px;">VANILLA</span>'
        )

        with st.expander(
            f"{medals[idx]}  {name}   —   {sc}/100   [{sdef['view']}]",
            expanded=(idx < 2)
        ):
            card_c1, card_c2, card_c3 = st.columns([3, 2, 2])

            with card_c1:
                # Reason bullets
                reason_html = "".join(
                    f'<div style="margin:2px 0;">'
                    f'<span style="color:{"#ff4b6e" if "⚠" in rr else "#00e676"}">{"⚠" if "⚠" in rr else "✓"}</span> '
                    f'<span style="color:#8fb3d0">{rr.replace("⚠️ ", "").replace("✅ ", "")}</span></div>'
                    for rr in reasons[:6]
                )
                st.markdown(f'''
                <div style="font-size:11px;line-height:1.9;">
                    {cat_badge}
                    <div style="font-size:12px;color:#8fb3d0;margin-top:10px;">{sdef["description"]}</div>
                    <div style="margin-top:10px;border-top:1px solid #1a2535;padding-top:8px;">
                        <div style="color:#3d6080;font-size:10px;letter-spacing:2px;margin-bottom:4px;">SCORING REASONS</div>
                        {reason_html}
                    </div>
                    <div style="margin-top:10px;font-size:10px;color:#5a7a99;line-height:2.0;">
                        <strong style="color:#3d6080;">Best when:</strong> {sdef["best_conditions"]}<br>
                        <strong style="color:#3d6080;">Risks:</strong> {sdef["risks"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

            with card_c2:
                st.markdown(f'''
                <div style="font-size:11px;color:#5a7a99;line-height:2.1;">
                    <div style="color:#3d6080;font-size:10px;letter-spacing:2px;margin-bottom:6px;">PROFILE</div>
                    Max gain: <span style="color:#00e676">{sdef["max_gain"]}</span><br>
                    Max loss: <span style="color:#ff4b6e">{sdef["max_loss"]}</span><br>
                    Breakeven: <span style="color:#f0a500">{sdef["breakeven_note"]}</span><br>
                    Complexity: <span style="color:#00d4ff">{"●" * sdef["complexity"]}{"○" * (5-sdef["complexity"])}</span><br>
                    Vol bias: <span style="color:#b388ff">{sdef["vol_bias"]}</span><br>
                </div>
                ''', unsafe_allow_html=True)

                # Legs table
                non_custom_legs = [lg for lg in sdef["legs"] if lg[0] != "custom"]
                if non_custom_legs:
                    T_yrs_c = sl_T_lab / 365
                    leg_rows = []
                    for leg in non_custom_legs:
                        opt_t, direction, strike_lbl, qty = leg
                        strike_map_c = {
                            "ATM": spot, "+10%": spot*1.10, "+20%": spot*1.20,
                            "+30%": spot*1.30, "+40%": spot*1.40, "+50%": spot*1.50,
                            "-10%": spot*0.90, "-20%": spot*0.80, "-30%": spot*0.70,
                            "-40%": spot*0.60, "-50%": spot*0.50,
                            "ATM farther expiry": spot
                        }
                        K_leg  = strike_map_c.get(strike_lbl, spot)
                        prem_c = black_scholes(spot, K_leg, T_yrs_c, r, sl_hv_lab, opt_t)
                        sign   = "+" if direction == "buy" else "−"
                        leg_rows.append({
                            "Leg":     f"{direction.upper()} {qty}× {opt_t.upper()}",
                            "Strike":  strike_lbl,
                            "K ($)":   f"${K_leg:.5f}",
                            "Premium": f"{sign}${prem_c:.6f}",
                        })
                    st.dataframe(pd.DataFrame(leg_rows), hide_index=True,
                                 use_container_width=True)

            with card_c3:
                # Payoff chart
                pr, pay, np_leg, _ = build_strategy_payoff(
                    name, sdef, spot, sl_hv_lab, sl_T_lab / 365, r
                )
                if np.any(pay != 0):
                    fig_card = go.Figure()
                    fig_card.add_trace(go.Scatter(
                        x=pr, y=pay, fill="tozeroy",
                        fillcolor=f"rgba({int(sc_c[1:3],16)},{int(sc_c[3:5],16)},{int(sc_c[5:7],16)},0.06)",
                        line=dict(color=sc_c, width=2),
                        hovertemplate="Spot: $%{x:.5f}<br>P&L: $%{y:.6f}<extra></extra>"
                    ))
                    fig_card.add_hline(y=0, line_color="#3d6080", line_dash="dot")
                    fig_card.add_vline(x=spot, line_color="#ffffff", line_dash="dash",
                                       annotation_text="S", annotation_font_size=9,
                                       annotation_font_color="#ffffff")
                    fig_card.update_layout(
                        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=9),
                        xaxis=dict(gridcolor=_PLT_GRID, tickformat=".5f"),
                        yaxis=dict(gridcolor=_PLT_GRID),
                        height=200, showlegend=False,
                        margin=dict(t=10, b=40, l=0, r=0)
                    )
                    st.plotly_chart(fig_card, use_container_width=True, key=_next_chart_key())
                else:
                    st.caption("Schematic payoff — exotic structure")

                # USD notional impact
                if np_leg != 0:
                    usd_cost = abs(np_leg) * sl_notional_lab / spot
                    st.markdown(
                        f'<div style="font-size:10px;color:#3d6080;text-align:center;">'
                        f'Net: <span style="color:{("#ff4b6e" if np_leg < 0 else "#00e676")};">'
                        f'{"−" if np_leg < 0 else "+"}${abs(np_leg):.6f}/unit · '
                        f'${usd_cost:,.0f} on ${sl_notional_lab:,}</span></div>',
                        unsafe_allow_html=True
                    )

    # ── OVERLAY: TOP 3 PAYOFFS COMPARED ─────────────────────────────
    st.markdown('<div class="section-header">TOP 3 STRATEGIES — PAYOFF OVERLAY</div>',
                unsafe_allow_html=True)

    overlay_palette = ["#00d4ff", "#00e676", "#f0a500", "#b388ff", "#ff8a65"]
    S_rng_ov = np.linspace(spot * 0.35, spot * 2.60, 400)

    fig_ov = go.Figure()
    for oi, (oname, osc, _, odef) in enumerate(top_results[:3]):
        pr_ov, pay_ov, _, _ = build_strategy_payoff(
            oname, odef, spot, sl_hv_lab, sl_T_lab / 365, r
        )
        if np.any(pay_ov != 0):
            # Interpolate to common range
            pay_ov_interp = np.interp(S_rng_ov, pr_ov, pay_ov)
            fig_ov.add_trace(go.Scatter(
                x=S_rng_ov, y=pay_ov_interp,
                name=f"#{oi+1} {oname} ({osc})",
                line=dict(color=overlay_palette[oi], width=2.5),
                hovertemplate=f"<b>{oname}</b><br>Spot: $%{{x:.5f}}<br>P&L: $%{{y:.6f}}<extra></extra>"
            ))

    fig_ov.add_hline(y=0, line_color="#3d6080", line_dash="dot")
    fig_ov.add_vline(x=spot, line_color="#ffffff", line_dash="dash",
                     annotation_text="CURRENT SPOT", annotation_font_color="#ffffff")
    fig_ov.update_layout(
        plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
        font=dict(family="IBM Plex Mono", color=_PLT_TXT, size=10),
        xaxis=dict(gridcolor=_PLT_GRID, title="Spot at Expiry ($)", tickformat=".5f"),
        yaxis=dict(gridcolor=_PLT_GRID, title="P&L per unit ($)"),
        legend=dict(bgcolor=_PLT_LEG, bordercolor=_PLT_BDR, font=dict(size=10)),
        height=380, margin=dict(t=10, b=50),
        hovermode="x unified"
    )
    st.plotly_chart(fig_ov, use_container_width=True, key=_next_chart_key())

    # ── MONTE CARLO EXPECTED P&L — TOP 3 STRATEGIES ────────────────
    st.markdown('<div class="section-header">MONTE CARLO EXPECTED P&L — TOP 3 STRATEGIES (10,000 PATHS)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#5a7a99;margin-bottom:12px;line-height:1.8;">'
        'Simulates 10,000 GBM paths to compute expected P&L distributions for the top 3 strategies. '
        'Uses the current vol and drift assumptions.</div>',
        unsafe_allow_html=True
    )

    np.random.seed(2024)
    n_mc_strat = 10_000
    T_mc_strat = sl_T_lab / 365
    Z_mc_strat = np.random.standard_normal(n_mc_strat)
    S_mc_terminal = spot * np.exp((r - 0.5 * sl_hv_lab**2) * T_mc_strat + sl_hv_lab * np.sqrt(T_mc_strat) * Z_mc_strat)

    mc_strat_cols = st.columns(min(3, len(top_results)))
    for si, (sname, ssc, _, sdef) in enumerate(top_results[:3]):
        pr_mc, pay_mc, nprem_mc, legs_mc = build_strategy_payoff(
            sname, sdef, spot, sl_hv_lab, T_mc_strat, r
        )
        if np.any(pay_mc != 0) and len(pr_mc) > 0:
            # Interpolate payoff at simulated terminal prices
            pnl_sims = np.interp(S_mc_terminal, pr_mc, pay_mc)
            mean_pnl = np.mean(pnl_sims)
            p_profit = np.mean(pnl_sims > 0) * 100
            var_95 = np.percentile(pnl_sims, 5)
            best_95 = np.percentile(pnl_sims, 95)

            with mc_strat_cols[si]:
                pnl_color = '#00e676' if mean_pnl > 0 else '#ff4b6e'
                st.markdown(f'''
                <div class="metric-card" style="border-left-color:{pnl_color}">
                    <div class="metric-label">#{si+1} {sname}</div>
                    <div class="metric-value" style="font-size:16px;color:{pnl_color};">E[P&L]: ${mean_pnl:.6f}</div>
                    <div class="metric-sub">
                        P(profit): {p_profit:.0f}% · VaR(5%): ${var_95:.6f} · Best(95%): ${best_95:.6f}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # Mini histogram
                fig_mc_mini = go.Figure(go.Histogram(
                    x=pnl_sims, nbinsx=40,
                    marker_color=pnl_color, marker_line_width=0, opacity=0.7
                ))
                fig_mc_mini.add_vline(x=0, line_color='#ffffff', line_dash='dot')
                fig_mc_mini.add_vline(x=mean_pnl, line_color='#f0a500', line_dash='dash')
                fig_mc_mini.update_layout(
                    plot_bgcolor=_PLT_BG, paper_bgcolor=_PLT_BG,
                    font=dict(family='IBM Plex Mono', color=_PLT_TXT, size=9),
                    xaxis=dict(gridcolor=_PLT_GRID, title='P&L ($)'),
                    yaxis=dict(gridcolor=_PLT_GRID), showlegend=False,
                    height=180, margin=dict(t=5, b=30, l=10, r=10)
                )
                st.plotly_chart(fig_mc_mini, use_container_width=True, key=_next_chart_key())

    # ── FULL DEPOSITORY BROWSER ──────────────────────────────────────
    st.markdown(
        f'<div class="section-header">FULL STRATEGY DEPOSITORY — ALL {len(STRATEGY_DEPOSITORY)} STRATEGIES</div>',
        unsafe_allow_html=True
    )

    dep_filter_col1, dep_filter_col2 = st.columns(2)
    with dep_filter_col1:
        dep_view_filter = st.multiselect(
            "Filter by view",
            ["bullish", "mildly bullish", "neutral", "bearish",
             "high vol", "structured", "all"],
            default=["bullish", "mildly bullish", "neutral", "bearish",
                     "high vol", "structured", "all"],
            key="dep_view_filter"
        )
    with dep_filter_col2:
        dep_tag_filter = st.multiselect(
            "Filter by tag",
            sorted(set(t for s in STRATEGY_DEPOSITORY.values() for t in s.get("tags", []))),
            default=[],
            key="dep_tag_filter"
        )

    # Build depository table
    score_lookup = {name: sc for name, sc, _, _ in raw_results}
    dep_rows = []
    for strat_name, sdef in STRATEGY_DEPOSITORY.items():
        view_match = any(vf in sdef["view"].lower() for vf in dep_view_filter) or not dep_view_filter
        tag_match  = (not dep_tag_filter) or any(t in sdef.get("tags", []) for t in dep_tag_filter)
        if not view_match or not tag_match:
            continue

        sc_val = score_lookup.get(strat_name, 0)
        dep_rows.append({
            "Strategy":    strat_name,
            "View":        sdef["view"],
            "Vol Bias":    sdef["vol_bias"],
            "Max Gain":    sdef["max_gain"],
            "Max Loss":    sdef["max_loss"],
            "Complexity":  "●" * sdef["complexity"] + "○" * (5 - sdef["complexity"]),
            "Tags":        ", ".join(sdef.get("tags", [])),
            "Your Score":  sc_val,
        })

    dep_df = pd.DataFrame(dep_rows).sort_values("Your Score", ascending=False)
    st.dataframe(dep_df, use_container_width=True, hide_index=True, height=460)

    # Download button
    csv_data = dep_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download full depository as CSV",
        data=csv_data,
        file_name=f"strategy_depository_{sl_token}_{sl_view.replace(' ','_')}.csv",
        mime="text/csv",
        key="dep_download"
    )

    # ── SCORING METHODOLOGY ──────────────────────────────────────────
    with st.expander("ℹ️  How the scoring engine works"):
        st.markdown(f'''
        <div style="font-size:11px;color:#5a7a99;line-height:2.1;">
            <strong style="color:#00d4ff;">View Alignment (0–35 pts × conviction scalar)</strong><br>
            Perfect match = 35 × confidence. Neutral strategy in directional view = 15 pts.
            Opposite view = −30 pts (hard penalty).<br><br>
            <strong style="color:#00d4ff;">Vol Regime (0–25 pts)</strong><br>
            Current HV = <strong style="color:{vol_color}">{sl_hv_lab:.0%} ({vol_label})</strong>.
            Short-vol strategies score 25pts in high-IV, −15 in low-IV.
            Long-vol strategies score 25pts in low-IV, −15 in high-IV.<br><br>
            <strong style="color:#00d4ff;">Put/Call Ratio / Skew (0–15 pts)</strong><br>
            PCR = <strong style="color:{pcr_color}">{sl_pcr_lab:.2f} ({pcr_label})</strong>.
            High put skew boosts risk reversals and cash-secured puts.
            Low PCR boosts debit call strategies.<br><br>
            <strong style="color:#00d4ff;">Tenor Fit (0–10 pts)</strong><br>
            Short tenor ({sl_T_lab}d) → income/theta strategies boosted.
            Long tenor → long-vol and LEAP structures boosted.<br><br>
            <strong style="color:#00d4ff;">Complexity Penalty (−8 pts)</strong><br>
            Applied when conviction &lt; 50% and strategy complexity ≥ 4.
            Don't run complex structures on weak conviction.<br><br>
            <strong style="color:#00d4ff;">OTC Gate</strong><br>
            Structured/OTC strategies score −20 if OTC toggle is off.
        </div>
        ''', unsafe_allow_html=True)
