"""
Supabase backend configuration for emeris.co deployment.

Setup instructions:
1. Create a Supabase project at https://supabase.com
2. Set environment variables:
   - SUPABASE_URL: Your project URL (e.g., https://xxxx.supabase.co)
   - SUPABASE_KEY: Your anon/public key
3. Run the SQL migrations below in the Supabase SQL editor

SQL MIGRATIONS:
--------------
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL,
    token TEXT DEFAULT 'XDC',
    spot_price FLOAT,
    volatility FLOAT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
    hv_7d FLOAT,
    hv_30d FLOAT,
    hv_90d FLOAT,
    garch_current FLOAT,
    garch_forecast_30d FLOAT,
    snapshot_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL,
    token TEXT,
    view TEXT,
    conviction FLOAT,
    top_strategy TEXT,
    top_score INT,
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE vol_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_results ENABLE ROW LEVEL SECURITY;

-- Allow anon access (for Streamlit public app)
CREATE POLICY "Allow all" ON user_sessions FOR ALL USING (true);
CREATE POLICY "Allow all" ON saved_quotes FOR ALL USING (true);
CREATE POLICY "Allow all" ON vol_snapshots FOR ALL USING (true);
CREATE POLICY "Allow all" ON strategy_results FOR ALL USING (true);
"""

import os
import streamlit as st

# Supabase client initialization
_supabase_client = None


def get_supabase_client():
    """Get or create the Supabase client. Returns None if not configured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    # Try Streamlit secrets (may not exist if not configured)
    if not url or not key:
        try:
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_KEY", "")
        except Exception:
            pass

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except ImportError:
        st.warning("supabase package not installed. Run: pip install supabase")
        return None
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        return None


def save_quote(session_id, quote_data):
    """Save a counterparty quote to Supabase."""
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("saved_quotes").insert({
            "session_id": session_id,
            "label": quote_data.get("label"),
            "option_type": quote_data.get("type"),
            "strike": quote_data.get("strike"),
            "expiry": quote_data.get("expiry"),
            "spot": quote_data.get("spot"),
            "quoted_price": quote_data.get("quoted_price"),
            "implied_vol": quote_data.get("implied_vol"),
            "fair_value_avg": quote_data.get("fv_avg"),
            "overcharge_pct": quote_data.get("overcharge_pct"),
            "notes": quote_data.get("notes"),
        }).execute()
        return True
    except Exception:
        return False


def save_vol_snapshot(token, hv_7d, hv_30d, hv_90d, garch_current, garch_30d):
    """Save a volatility snapshot to Supabase."""
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("vol_snapshots").insert({
            "token": token,
            "hv_7d": hv_7d,
            "hv_30d": hv_30d,
            "hv_90d": hv_90d,
            "garch_current": garch_current,
            "garch_forecast_30d": garch_30d,
        }).execute()
        return True
    except Exception:
        return False


def save_strategy_result(session_id, token, view, conviction, top_strategy, top_score, params):
    """Save strategy lab results to Supabase."""
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("strategy_results").insert({
            "session_id": session_id,
            "token": token,
            "view": view,
            "conviction": conviction,
            "top_strategy": top_strategy,
            "top_score": top_score,
            "parameters": params,
        }).execute()
        return True
    except Exception:
        return False


def load_quotes(session_id):
    """Load saved quotes from Supabase."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        response = client.table("saved_quotes") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .execute()
        return response.data
    except Exception:
        return []


def load_vol_history(token, days=30):
    """Load historical vol snapshots from Supabase."""
    client = get_supabase_client()
    if client is None:
        return []
    try:
        response = client.table("vol_snapshots") \
            .select("*") \
            .eq("token", token) \
            .order("snapshot_date", desc=True) \
            .limit(days) \
            .execute()
        return response.data
    except Exception:
        return []
