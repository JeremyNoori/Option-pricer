"""Configuration constants, API keys, and token definitions."""
import os

# ── API Keys ──────────────────────────────────────────────────────────
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "CG-V6cHxSdrovx4eoExmoUAEcsw")
CMC_API_KEY = os.environ.get("CMC_API_KEY", "77e9e49f6c124981973dff95ec600d7e")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
CMC_BASE = "https://pro-api.coinmarketcap.com/v1"
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
FNG_BASE = "https://api.alternative.me/fng"

# ── TTL Constants (seconds) ──────────────────────────────────────────
TTL_SPOT = 60
TTL_HISTORY = 300
TTL_MARKET = 300
TTL_FNG = 600
TTL_DERIBIT = 120
TTL_DVOL = 600

# DB cache max age (seconds)
DB_MAX_AGE_SPOT = 7200       # 2 hours
DB_MAX_AGE_HISTORY = 7200
DB_MAX_AGE_DVOL = 600

# ── Token Definitions ────────────────────────────────────────────────
TOKENS = {
    "XDC":  {"coin_id": "xdce-crowd-sale", "ticker": "XDC",  "cmc_id": 2634},
    "BTC":  {"coin_id": "bitcoin",         "ticker": "BTC",  "cmc_id": 1},
    "ETH":  {"coin_id": "ethereum",        "ticker": "ETH",  "cmc_id": 1027},
    "SOL":  {"coin_id": "solana",          "ticker": "SOL",  "cmc_id": 5426},
    "HYPE": {"coin_id": "hyperliquid",     "ticker": "HYPE", "cmc_id": 0},
}

# ── Plotly Theme Colors ──────────────────────────────────────────────
PLT_BG = "#0a0c10"
PLT_BG2 = "#0d1117"
PLT_TXT = "#8fb3d0"
PLT_GRID = "#111922"
PLT_LEG = "rgba(10,12,16,0.85)"
PLT_BDR = "#1e2d40"

# ── App Color Palette ────────────────────────────────────────────────
COLORS = {
    "accent": "#00d4ff",
    "call": "#00d4ff",
    "put": "#ff4b6e",
    "green": "#00e676",
    "red": "#ff4b6e",
    "orange": "#f0a500",
    "purple": "#b388ff",
    "text_primary": "#c8d6e5",
    "text_secondary": "#5a7a99",
    "text_dim": "#3d6080",
    "card_bg": "#0d1117",
    "app_bg": "#0a0c10",
    "border": "#1e2d40",
}

# ── Chart Layout Template ────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor=PLT_BG,
    paper_bgcolor=PLT_BG,
    font=dict(family="IBM Plex Mono, monospace", color=PLT_TXT, size=10),
    margin=dict(t=20, b=50, l=20, r=20),
)

def chart_axes():
    return dict(gridcolor=PLT_GRID), dict(gridcolor=PLT_GRID)

def chart_legend():
    return dict(bgcolor=PLT_LEG, bordercolor=PLT_BDR, font=dict(size=9))
