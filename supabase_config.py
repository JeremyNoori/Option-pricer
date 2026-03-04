"""
Supabase backend for emeris.co deployment.

Provides:
  1. market_data_cache — persistent cross-session cache for ALL external API data
  2. saved_quotes      — counterparty quote comparison storage
  3. vol_snapshots     — historical vol tracking
  4. strategy_results  — strategy lab output log

Setup:
  1. Create a Supabase project at https://supabase.com
  2. Set env vars or Streamlit secrets:  SUPABASE_URL, SUPABASE_KEY
  3. Run the SQL below in the Supabase SQL editor

SQL MIGRATIONS:
--------------

-- Core market data cache (the big win — shared across all users)
CREATE TABLE IF NOT EXISTS market_data_cache (
    cache_key  TEXT PRIMARY KEY,
    data       JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mdc_updated ON market_data_cache(updated_at);

-- User / session tables (unchanged)
CREATE TABLE IF NOT EXISTS saved_quotes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL,
    label TEXT,
    option_type TEXT,
    strike FLOAT,
    expiry TEXT,
    spot FLOAT,
    quoted_price FLOAT,
    implied_vol FLOAT,
    fair_value_avg FLOAT,
    overcharge_pct FLOAT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vol_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    token TEXT NOT NULL,
    hv_7d FLOAT, hv_30d FLOAT, hv_90d FLOAT,
    garch_current FLOAT, garch_forecast_30d FLOAT,
    snapshot_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL,
    token TEXT, view TEXT, conviction FLOAT,
    top_strategy TEXT, top_score INT,
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE market_data_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_quotes      ENABLE ROW LEVEL SECURITY;
ALTER TABLE vol_snapshots     ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_results  ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all" ON market_data_cache FOR ALL USING (true);
CREATE POLICY "Allow all" ON saved_quotes      FOR ALL USING (true);
CREATE POLICY "Allow all" ON vol_snapshots     FOR ALL USING (true);
CREATE POLICY "Allow all" ON strategy_results  FOR ALL USING (true);
"""

import os
import json
from datetime import datetime, timezone

# ── Supabase client (lazy singleton) ──────────────────────────────
_supabase_client = None


def get_supabase_client():
    """Get or create the Supabase client. Returns None if not configured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        try:
            import streamlit as _st
            url = url or _st.secrets.get("SUPABASE_URL", "")
            key = key or _st.secrets.get("SUPABASE_KEY", "")
        except Exception:
            pass

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MARKET DATA CACHE — persistent, cross-session, cross-user
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def db_cache_get(cache_key, max_age_s=7200):
    """
    Read from Supabase market_data_cache.
    Returns the cached data if it exists and is fresher than max_age_s seconds.
    Returns None otherwise.
    """
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("market_data_cache")
            .select("data, updated_at")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data[0]
        updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        if age > max_age_s:
            return None  # stale
        return row["data"]
    except Exception:
        return None


def db_cache_set(cache_key, data):
    """
    Write to Supabase market_data_cache (upsert).
    `data` must be JSON-serialisable.
    """
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("market_data_cache").upsert(
            {
                "cache_key":  cache_key,
                "data":       data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="cache_key",
        ).execute()
        return True
    except Exception:
        return False


def db_cache_get_age(cache_key):
    """Return age in seconds of a cache entry, or None if missing."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = (
            client.table("market_data_cache")
            .select("updated_at")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        updated = datetime.fromisoformat(resp.data[0]["updated_at"].replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated).total_seconds()
    except Exception:
        return None


def db_cache_keys():
    """List all cache keys with their age. For diagnostics."""
    client = get_supabase_client()
    if client is None:
        return {}
    try:
        resp = (
            client.table("market_data_cache")
            .select("cache_key, updated_at")
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
        )
        now = datetime.now(timezone.utc)
        result = {}
        for row in resp.data:
            updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            result[row["cache_key"]] = int((now - updated).total_seconds())
        return result
    except Exception:
        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  USER DATA (quotes, vol snapshots, strategy results)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_quote(session_id, quote_data):
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("saved_quotes").insert({
            "session_id":    session_id,
            "label":         quote_data.get("label"),
            "option_type":   quote_data.get("type"),
            "strike":        quote_data.get("strike"),
            "expiry":        quote_data.get("expiry"),
            "spot":          quote_data.get("spot"),
            "quoted_price":  quote_data.get("quoted_price"),
            "implied_vol":   quote_data.get("implied_vol"),
            "fair_value_avg":quote_data.get("fv_avg"),
            "overcharge_pct":quote_data.get("overcharge_pct"),
            "notes":         quote_data.get("notes"),
        }).execute()
        return True
    except Exception:
        return False


def save_vol_snapshot(token, hv_7d, hv_30d, hv_90d, garch_current, garch_30d):
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("vol_snapshots").insert({
            "token": token, "hv_7d": hv_7d, "hv_30d": hv_30d,
            "hv_90d": hv_90d, "garch_current": garch_current,
            "garch_forecast_30d": garch_30d,
        }).execute()
        return True
    except Exception:
        return False


def save_strategy_result(session_id, token, view, conviction,
                         top_strategy, top_score, params):
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("strategy_results").insert({
            "session_id": session_id, "token": token, "view": view,
            "conviction": conviction, "top_strategy": top_strategy,
            "top_score": top_score, "parameters": params,
        }).execute()
        return True
    except Exception:
        return False


def load_quotes(session_id):
    client = get_supabase_client()
    if client is None:
        return []
    try:
        resp = (client.table("saved_quotes")
                .select("*").eq("session_id", session_id)
                .order("created_at", desc=True).execute())
        return resp.data
    except Exception:
        return []


def load_vol_history(token, days=30):
    client = get_supabase_client()
    if client is None:
        return []
    try:
        resp = (client.table("vol_snapshots")
                .select("*").eq("token", token)
                .order("snapshot_date", desc=True).limit(days).execute())
        return resp.data
    except Exception:
        return []
