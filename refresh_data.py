#!/usr/bin/env python3
"""
Background data refresh for the OTC Options Pricer.

Fetches all market data from external APIs and stores it in Supabase
so that the Streamlit app loads instantly from the DB cache.

Usage:
  # One-shot refresh
  python refresh_data.py

  # Cron (every 2 hours)
  0 */2 * * * cd /path/to/Option-pricer && python refresh_data.py >> refresh.log 2>&1

  # GitHub Actions (see .github/workflows/refresh.yml)

Environment variables required:
  SUPABASE_URL   — your Supabase project URL
  SUPABASE_KEY   — your Supabase anon/public key
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timezone, timedelta

# ── API Keys ──────────────────────────────────────────────────────
COINGECKO_API_KEY = "CG-V6cHxSdrovx4eoExmoUAEcsw"
CMC_API_KEY       = "77e9e49f6c124981973dff95ec600d7e"

CG_HEADERS = {
    "accept": "application/json",
    "x-cg-demo-api-key": COINGECKO_API_KEY,
}
CMC_HEADERS = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
    "Accept": "application/json",
}
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# ── Tokens to refresh ────────────────────────────────────────────
TOKENS = {
    "xdc-network":  "XDC",
    "bitcoin":      "BTC",
    "ethereum":     "ETH",
    "hyperliquid":  "HYPE",
    "solana":       "SOL",
}

DERIBIT_CURRENCIES = ["BTC", "ETH", "SOL"]

# ── Supabase client ──────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)
    from supabase import create_client
    _client = create_client(url, key)
    return _client


def db_set(cache_key, data):
    """Upsert into market_data_cache."""
    try:
        client = get_client()
        client.table("market_data_cache").upsert(
            {
                "cache_key":  cache_key,
                "data":       data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="cache_key",
        ).execute()
    except Exception as e:
        log(f"  DB write failed for {cache_key}: {e}")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ── CoinGecko helpers ────────────────────────────────────────────

def cg_get(url, timeout=10):
    r = requests.get(url, headers=CG_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def cmc_price(symbol):
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        r = requests.get(url, headers=CMC_HEADERS,
                         params={"symbol": symbol, "convert": "USD"}, timeout=8)
        return r.json()["data"][symbol]["quote"]["USD"]["price"]
    except Exception:
        return None


# ── Fetch + store functions ──────────────────────────────────────

def refresh_spot_prices():
    """Fetch live spot prices for all tokens."""
    log("Refreshing spot prices...")
    for coin_id, ticker in TOKENS.items():
        price = None
        try:
            data = cg_get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
                timeout=5
            )
            price = data.get(coin_id, {}).get("usd")
        except Exception as e:
            log(f"  CoinGecko failed for {ticker}: {e}")

        if price is None:
            price = cmc_price(ticker)
            if price:
                log(f"  {ticker}: ${price:.6f} (CMC fallback)")
            else:
                log(f"  {ticker}: FAILED — both APIs down")
                continue
        else:
            log(f"  {ticker}: ${price:.6f}")

        try:
            db_set(f"price:{coin_id}", price)
        except Exception as e:
            log(f"  {ticker}: DB write failed — {e}")
        time.sleep(1.5)  # respect CoinGecko rate limit


def refresh_price_history():
    """Fetch price history (90d, 180d, 365d) for all tokens."""
    log("Refreshing price history...")
    for coin_id, ticker in TOKENS.items():
        for days in [90, 180, 365]:
            key = f"history:{coin_id}:{days}"
            prices = None
            try:
                data = cg_get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                    f"?vs_currency=usd&days={days}&interval=daily"
                )
                prices = [p[1] for p in data.get("prices", [])]
                if len(prices) < 5:
                    prices = None
            except Exception as e:
                log(f"  CoinGecko history {ticker}/{days}d failed: {e}")

            if prices is None:
                try:
                    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
                    r = requests.get(url, headers=CMC_HEADERS,
                                     params={"symbol": ticker, "convert": "USD",
                                             "count": str(days), "interval": "daily"},
                                     timeout=12)
                    quotes = r.json()["data"]["quotes"]
                    prices = [q["quote"]["USD"]["close"] for q in quotes]
                    if len(prices) < 5:
                        prices = None
                    else:
                        log(f"  {ticker}/{days}d: {len(prices)} pts (CMC fallback)")
                except Exception:
                    pass

            if prices:
                try:
                    db_set(key, prices)
                    log(f"  {ticker}/{days}d: {len(prices)} data points")
                except Exception as e:
                    log(f"  {ticker}/{days}d: DB write failed — {e}")
            else:
                log(f"  {ticker}/{days}d: FAILED")

            time.sleep(2)  # rate limit


def refresh_market_data():
    """Fetch detailed market data for all tokens."""
    log("Refreshing market data...")
    for coin_id, ticker in TOKENS.items():
        try:
            data = cg_get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                f"?localization=false&tickers=false&community_data=true&developer_data=false",
                timeout=10
            )
            md = data.get("market_data", {})
            result = {
                "market_cap":       md.get("market_cap", {}).get("usd"),
                "volume_24h":       md.get("total_volume", {}).get("usd"),
                "price_change_7d":  md.get("price_change_percentage_7d"),
                "price_change_30d": md.get("price_change_percentage_30d"),
                "ath":              md.get("ath", {}).get("usd"),
                "ath_change_pct":   md.get("ath_change_percentage", {}).get("usd"),
                "sentiment_up":     data.get("sentiment_votes_up_percentage"),
            }
            db_set(f"market_data:{coin_id}", result)
            log(f"  {ticker}: OK")
        except Exception as e:
            log(f"  {ticker}: FAILED — {e}")
        time.sleep(2)


def refresh_fear_greed():
    """Fetch Fear & Greed index."""
    log("Refreshing Fear & Greed...")
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=30&format=json",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        entries = r.json().get("data", [])
        result = []
        for e in entries:
            try:
                result.append((int(e["timestamp"]), int(e["value"]),
                               e.get("value_classification", "Unknown")))
            except (ValueError, KeyError):
                continue
        if result:
            db_set("fear_greed:30", result)
            log(f"  {len(result)} entries")
        else:
            log("  FAILED — no entries parsed")
    except Exception as e:
        log(f"  FAILED — {e}")


def refresh_deribit():
    """Fetch Deribit index prices, book summaries, and DVOL history."""
    log("Refreshing Deribit data...")

    for currency in DERIBIT_CURRENCIES:
        # Index price
        try:
            r = requests.get(f"{DERIBIT_BASE}/get_index_price",
                             params={"index_name": f"{currency.lower()}_usd"},
                             timeout=10)
            result = r.json().get("result", {})
            idx = result.get("index_price")
            if idx:
                db_set(f"deribit_index:{currency}", idx)
                log(f"  {currency} index: ${idx:,.2f}")
        except Exception as e:
            log(f"  {currency} index FAILED: {e}")

        # Book summary
        try:
            r = requests.get(f"{DERIBIT_BASE}/get_book_summary_by_currency",
                             params={"currency": currency, "kind": "option"},
                             timeout=10)
            result = r.json().get("result", [])
            if result:
                db_set(f"deribit_book:{currency}", result)
                log(f"  {currency} book: {len(result)} instruments")
        except Exception as e:
            log(f"  {currency} book FAILED: {e}")

        # DVOL history
        try:
            now_ms  = int(datetime.now().timestamp() * 1000)
            ago_ms  = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            r = requests.get(f"{DERIBIT_BASE}/get_volatility_index_data",
                             params={"currency": currency,
                                     "start_timestamp": ago_ms,
                                     "end_timestamp": now_ms,
                                     "resolution": "3600"},
                             timeout=10)
            result = r.json().get("result", {})
            if result and result.get("data"):
                dvol = [(row[0], row[4]) for row in result["data"]]
                db_set(f"deribit_dvol:{currency}:30", dvol)
                log(f"  {currency} DVOL: {len(dvol)} hours")
        except Exception as e:
            log(f"  {currency} DVOL FAILED: {e}")

        time.sleep(0.5)


# ── Main ─────────────────────────────────────────────────────────

def refresh_all():
    """Run a full data refresh cycle."""
    start = time.time()
    log("=" * 50)
    log("Starting full data refresh")
    log("=" * 50)

    refresh_spot_prices()
    refresh_price_history()
    refresh_market_data()
    refresh_fear_greed()
    refresh_deribit()

    elapsed = time.time() - start
    log("=" * 50)
    log(f"Done in {elapsed:.1f}s")
    log("=" * 50)


if __name__ == "__main__":
    try:
        refresh_all()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
