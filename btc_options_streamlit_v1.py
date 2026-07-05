# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║ Shantanu's BTC Options Dashboard — Streamlit v1                      ║
║ Data: Deribit Public API (www.deribit.com/api/v2) — FREE, no key    ║
║ Logic: identical to nifty_streamlit_v5 / crypto_dashboard_v3 engine ║
║ BTC & ETH · Full Chain · Native Greeks · IV · OI · Bias · Strategy  ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install -r requirements.txt
    streamlit run btc_options_streamlit_v1.py

API:  Deribit public REST (JSON-RPC over GET) — no API key needed.
      public/get_instruments              → strikes + expiries
      public/get_book_summary_by_currency → OI, bid/ask, mark_iv, volume (one call)
      public/get_index_price              → BTC/ETH spot index
      public/ticker                       → native Greeks (ATM band, parallel)
      public/get_volatility_index_data    → DVOL (crypto's "VIX")
      public/ticker BTC-PERPETUAL         → basis + funding (futures triangulation)

SELF-TEST (no network, no streamlit UI):
    python btc_options_streamlit_v1.py --selftest
"""

import os, sys, json, time, tempfile, threading
from datetime import datetime, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import plotly.graph_objs as go

# ─── Colours (identical to crypto v3 / options dashboard v13) ────────────────
BG, CARD, TEXT   = "#F5F6FA", "#FFFFFF", "#1A1A2E"
ACCENT, MUTED    = "#5C35CC", "#6B7280"
GOLD, GREEN, RED = "#B45309", "#059669", "#DC2626"
AMBER, BLUE      = "#D97706", "#2563EB"
CYAN, PINK       = "#0891B2", "#9333EA"
BORDER           = "#E5E7EB"
SECTION_BG       = "#EEF2FF"
BTC_COLOR, ETH_COLOR = "#F7931A", "#627EEA"

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_REST       = "https://www.deribit.com/api/v2"
HEADERS         = {"Content-Type": "application/json"}
SYMBOLS         = ["BTC", "ETH"]
DEFAULT_SYMBOL  = "BTC"
STRUCTURAL_BAND = 10
SIGNAL_BAND     = 5
GREEK_BAND      = 15          # strikes each side of ATM for native-Greeks fetch
REFRESH_SECS    = 60          # page + data refresh cadence
HISTORY_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "crypto_streamlit_history.json")
MAX_HISTORY     = 500

# Bias weights — IDENTICAL to crypto v3 (which mirrors NIFTY dashboard)
BIAS_WEIGHTS = {
    "net_delta": 20, "momentum": 22, "ev_ratio": 12, "atm_pressure": 10,
    "skew_slope": 8,
    "ev_ratio_bull": 1.15, "ev_ratio_bear": 0.87,
    "skew_slope_threshold": 0.3,
    "regime_range": 25, "regime_trend": 15, "regime_transition": 10,
    "near_oi_concentration": 12, "near_oichg_concentration": 15,
    "near_oi_min": 0.35, "near_oichg_min": 0.40,
    "wall_proximity": 10, "wall_proximity_pts": 500,
    "wall_shift": 8, "max_pain_drift": 5,
    "range_compression": 8, "expansion_building": 5,
    "persistence": 10,
    "bias_bull_threshold": 12, "bias_bear_threshold": -12,
    "confidence_min_strategy": 20,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def safe_num(v, default=0.0):
    try:
        f = float(v)
        return default if (f != f) else f
    except Exception:
        return default


def utc_str(fmt="%d-%m-%Y  %H:%M:%S UTC"):
    return datetime.now(timezone.utc).strftime(fmt)


def hex_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def chart_layout(**kw):
    base = dict(
        paper_bgcolor=CARD, plot_bgcolor="#F9FAFB",
        font=dict(color=TEXT, size=12),
        margin=dict(l=42, r=18, t=50, b=40), height=290,
        xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#D1D5DB"),
        yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#D1D5DB"),
    )
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# DERIBIT PUBLIC API — FETCHERS  (identical strategy to crypto v3)
# ─────────────────────────────────────────────────────────────────────────────
_session = requests.Session()


def _deribit_get(method: str, params: dict, timeout: int = 12) -> dict:
    """Single helper for all Deribit public REST calls (JSON-RPC over GET)."""
    try:
        r = _session.get(f"{BASE_REST}/{method}", params=params,
                         headers=HEADERS, timeout=timeout)
        if not r.text.strip():
            return {}
        j = r.json()
        if "error" in j:
            return {}
        return j
    except Exception:
        return {}


def fetch_expiries(symbol="BTC") -> list:
    """Sorted upcoming expiry dates via public/get_instruments."""
    j = _deribit_get("public/get_instruments",
                     {"currency": symbol, "kind": "option", "expired": "false"})
    expiries = set()
    for inst in j.get("result", []):
        ts_ms = inst.get("expiration_timestamp", 0)
        if ts_ms:
            exp_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            if exp_dt >= date.today():
                expiries.add(exp_dt)
    return sorted(expiries)


def fetch_spot(symbol="BTC") -> float:
    """Index price via public/get_index_price; perpetual ticker as fallback."""
    j  = _deribit_get("public/get_index_price", {"index_name": f"{symbol.lower()}_usd"})
    sp = safe_num(j.get("result", {}).get("index_price", 0))
    if sp > 0:
        return round(sp, 2)
    j2 = _deribit_get("public/ticker", {"instrument_name": f"{symbol}-PERPETUAL"})
    return safe_num(j2.get("result", {}).get("index_price", 0))


def _fetch_one_ticker(name: str) -> tuple:
    j      = _deribit_get("public/ticker", {"instrument_name": name}, timeout=6)
    res    = j.get("result", {})
    greeks = res.get("greeks", {}) or {}
    stats  = res.get("stats",  {}) or {}
    return name, {
        "delta":  safe_num(greeks.get("delta", 0)),
        "gamma":  safe_num(greeks.get("gamma", 0)),
        "theta":  safe_num(greeks.get("theta", 0)),
        "vega":   safe_num(greeks.get("vega",  0)),
        "iv":     safe_num(res.get("mark_iv", 0)),        # already in %
        "bid":    safe_num(res.get("best_bid_price", 0)),
        "ask":    safe_num(res.get("best_ask_price", 0)),
        "ltp":    safe_num(res.get("last_price", 0) or res.get("mark_price", 0)),
        "mark":   safe_num(res.get("mark_price", 0)),
        "oi":     safe_num(res.get("open_interest", 0)),
        "oi_chg": safe_num(stats.get("volume", 0)),       # 24h volume as proxy
    }


def _fetch_greeks_batch(instrument_names: list) -> dict:
    """Native Greeks via public/ticker — parallel (8 workers) to keep the
    Streamlit rerun fast while staying inside Deribit's 20 req/s public limit."""
    results = {}
    if not instrument_names:
        return results
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch_one_ticker, n) for n in instrument_names]
        for f in as_completed(futs):
            try:
                name, data = f.result()
                results[name] = data
            except Exception:
                pass
    return results


def get_strike_step(symbol="BTC"):
    return 1000 if symbol == "BTC" else 50


def fetch_option_chain(symbol="BTC", expiry_date=None):
    """Full option chain → (DataFrame, spot, expiry). Same 6-step strategy as
    crypto v3: book summary in ONE call, Greeks sampled for ATM±GREEK_BAND."""
    spot = fetch_spot(symbol)
    if spot == 0:
        return pd.DataFrame(), 0.0, None

    j_inst = _deribit_get("public/get_instruments",
                          {"currency": symbol, "kind": "option", "expired": "false"})
    instruments = j_inst.get("result", [])
    if not instruments:
        return pd.DataFrame(), spot, None

    inst_map, all_expiries = {}, set()
    for inst in instruments:
        name   = inst.get("instrument_name", "")
        strike = safe_num(inst.get("strike", 0))
        ts_ms  = inst.get("expiration_timestamp", 0)
        opt_t  = inst.get("option_type", "")
        if not name or strike == 0 or not opt_t:
            continue
        exp_dt = (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
                  if ts_ms else None)
        if exp_dt and exp_dt >= date.today():
            all_expiries.add(exp_dt)
            inst_map[name] = {"strike": strike, "expiry": exp_dt, "option_type": opt_t}

    if expiry_date and isinstance(expiry_date, str):
        try:
            expiry_date = date.fromisoformat(expiry_date[:10])
        except Exception:
            expiry_date = None
    if not expiry_date:
        upcoming = sorted(all_expiries)
        expiry_date = upcoming[0] if upcoming else None
    if not expiry_date:
        return pd.DataFrame(), spot, None

    j_book   = _deribit_get("public/get_book_summary_by_currency",
                            {"currency": symbol, "kind": "option"})
    book_map = {it.get("instrument_name", ""): it for it in j_book.get("result", [])}

    calls_raw, puts_raw = {}, {}
    for name, meta in inst_map.items():
        if meta["expiry"] != expiry_date:
            continue
        bk = book_map.get(name, {})
        entry = {
            "oi":     safe_num(bk.get("open_interest", 0)),
            "oi_chg": safe_num(bk.get("volume", 0)),      # 24h volume
            "iv":     safe_num(bk.get("mark_iv", 0)),     # already in %
            "bid":    safe_num(bk.get("bid_price", 0)),
            "ask":    safe_num(bk.get("ask_price", 0)),
            "ltp":    safe_num(bk.get("last", 0) or bk.get("mark_price", 0)),
            "mark":   safe_num(bk.get("mark_price", 0)),
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "name":  name,
        }
        (calls_raw if meta["option_type"] == "call" else puts_raw)[meta["strike"]] = entry

    if not calls_raw and not puts_raw:
        return pd.DataFrame(), spot, expiry_date

    all_strikes = sorted(set(calls_raw) | set(puts_raw))
    atm  = min(all_strikes, key=lambda x: abs(x - spot))
    step = get_strike_step(symbol)
    greek_strikes = {K for K in all_strikes if abs(K - atm) <= GREEK_BAND * step}

    greek_names = []
    for K in greek_strikes:
        if K in calls_raw: greek_names.append(calls_raw[K]["name"])
        if K in puts_raw:  greek_names.append(puts_raw[K]["name"])
    greeks_data = _fetch_greeks_batch(greek_names)

    for side in (calls_raw, puts_raw):
        for K, entry in side.items():
            g = greeks_data.get(entry["name"], {})
            for fld in ("delta", "gamma", "theta", "vega"):
                if g.get(fld, 0) != 0:
                    entry[fld] = g[fld]
            for fld in ("iv", "oi", "bid", "ask", "ltp"):
                if g.get(fld, 0) > 0:
                    entry[fld] = g[fld]

    rows = []
    for K in all_strikes:
        c, p = calls_raw.get(K, {}), puts_raw.get(K, {})
        rows.append({
            "strike":      K,
            "call_oi":     c.get("oi", 0),     "put_oi":     p.get("oi", 0),
            "call_oi_chg": c.get("oi_chg", 0), "put_oi_chg": p.get("oi_chg", 0),
            "call_iv":     c.get("iv", 0),     "put_iv":     p.get("iv", 0),
            "call_delta":  c.get("delta", 0),  "put_delta":  p.get("delta", 0),
            "call_gamma":  c.get("gamma", 0),  "put_gamma":  p.get("gamma", 0),
            "call_theta":  c.get("theta", 0),  "put_theta":  p.get("theta", 0),
            "call_vega":   c.get("vega", 0),   "put_vega":   p.get("vega", 0),
            "call_ltp":    c.get("ltp", 0),    "put_ltp":    p.get("ltp", 0),
            "call_bid":    c.get("bid", 0),    "put_bid":    p.get("bid", 0),
            "call_ask":    c.get("ask", 0),    "put_ask":    p.get("ask", 0),
        })
    return pd.DataFrame(rows), round(spot, 2), expiry_date


def fetch_dvol(symbol="BTC"):
    """DVOL — Deribit's implied-vol index (crypto analogue of India VIX)."""
    now_ms = int(time.time() * 1000)
    j = _deribit_get("public/get_volatility_index_data",
                     {"currency": symbol, "resolution": "3600",
                      "start_timestamp": now_ms - 48 * 3600 * 1000,
                      "end_timestamp": now_ms})
    data = j.get("result", {}).get("data", [])
    if not data:
        return None, []
    closes = [safe_num(row[4]) for row in data if len(row) >= 5]
    return (closes[-1] if closes else None), closes


def fetch_perpetual(symbol="BTC"):
    """Perpetual ticker — basis + funding (analogue of futures triangulation)."""
    j   = _deribit_get("public/ticker", {"instrument_name": f"{symbol}-PERPETUAL"})
    res = j.get("result", {})
    if not res:
        return {}
    mark  = safe_num(res.get("mark_price", 0))
    index = safe_num(res.get("index_price", 0))
    return {
        "mark": mark, "index": index,
        "basis": round(mark - index, 2),
        "basis_pct": round((mark - index) / index * 100, 4) if index else 0.0,
        "funding_8h": safe_num(res.get("funding_8h", 0)),
        "current_funding": safe_num(res.get("current_funding", 0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-EXCHANGE FREE OPTIONS APIs  (v2 addition)
#   • OKX  public REST    — BTC/ETH options chain, no key, no rate-limit issues
#   • Bybit public v5     — BTC/ETH options chain, no key
#   • Binance public      — spot + futures funding (no real options API, used for
#                            cross-exchange spot triangulation and basis)
#   • CoinGecko public    — aggregate spot price (cross-venue reference)
#   • CryptoCompare public — historical OHLC for DVOL-style IV proxy if needed
# All endpoints are FREE, public, no API key required.
# ═════════════════════════════════════════════════════════════════════════════
OKX_BASE    = "https://www.okx.com/api/v5/public"
BYBIT_BASE  = "https://api.bybit.com/v5/market"
BINANCE_BASE= "https://api.binance.com"
COINGECKO   = "https://api.coingecko.com/api/v3"

EXCHANGE_LIST = ["Deribit", "OKX", "Bybit"]   # user-selectable chain source


def _okx_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    """OKX public GET helper. Returns {} on any error."""
    try:
        r = _session.get(f"{OKX_BASE}{path}", params=params or {},
                         headers=HEADERS, timeout=timeout)
        if not r.text.strip():
            return {}
        j = r.json()
        if j.get("code") != "0":
            return {}
        return j
    except Exception:
        return {}


def _bybit_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    """Bybit public v5 GET helper. Returns {} on any error."""
    try:
        r = _session.get(f"{BYBIT_BASE}{path}", params=params or {},
                         headers=HEADERS, timeout=timeout)
        if not r.text.strip():
            return {}
        j = r.json()
        if j.get("retCode") != 0:
            return {}
        return j
    except Exception:
        return {}


def _binance_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    """Binance public GET helper (spot + futures + funding)."""
    try:
        r = _session.get(f"{BINANCE_BASE}{path}", params=params or {},
                         headers=HEADERS, timeout=timeout)
        if not r.text.strip():
            return {}
        return r.json()
    except Exception:
        return {}


def _coingecko_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    """CoinGecko public GET helper (aggregate crypto spot reference)."""
    try:
        r = _session.get(f"{COINGECKO}{path}", params=params or {},
                         headers={"Accept": "application/json"}, timeout=timeout)
        if not r.text.strip():
            return {}
        return r.json()
    except Exception:
        return {}


# ── OKX options chain ─────────────────────────────────────────────────────────
def fetch_okx_expiries(symbol="BTC") -> list:
    """OKX option expiries — returns sorted list of date objects."""
    inst = symbol + "-USD"
    j = _okx_get("/public/instruments", {"instType": "OPTION", "uly": inst})
    exps = set()
    for row in j.get("data", []) or []:
        exp_str = row.get("expTime")     # ms epoch
        if exp_str:
            try:
                dt = datetime.fromtimestamp(int(exp_str) / 1000, tz=timezone.utc).date()
                if dt >= date.today():
                    exps.add(dt)
            except Exception:
                pass
    return sorted(exps)


def fetch_okx_option_chain(symbol="BTC", expiry_date=None):
    """
    Returns DataFrame with the same schema as the Deribit fetcher
    (strike, call_oi, put_oi, call_iv, put_iv, call_delta, call_gamma,
     call_theta, call_vega, put_delta, put_gamma, put_theta, put_vega,
     call_ltp, put_ltp, call_oi_chg, put_oi_chg, call_bid, call_ask,
     put_bid, put_ask) + spot float + expiry date.
    All numeric fields coerced via safe_num.
    """
    uly = symbol + "-USD"
    j   = _okx_get("/public/instruments", {"instType": "OPTION", "uly": uly})
    rows = []
    spot = 0.0
    # OKX spot via index tickers endpoint
    sp_j = _okx_get("/market/index-tickers", {"quoteCcyy": "USD"})
    for it in sp_j.get("data", []) or []:
        if it.get("instId") == f"{uly}-INDEX":
            spot = safe_num(it.get("idxPx"))
            break
    if not spot:
        # Fallback: derive from BTC-USDT spot
        sp_j2 = _okx_get("/market/ticker", {"instId": f"{symbol}-USDT"})
        spot = safe_num((sp_j2.get("data", [{}])[0] or {}).get("last"))

    instruments = j.get("data", []) or []
    if not instruments:
        return pd.DataFrame(), spot, expiry_date

    # Filter to chosen expiry if given
    if expiry_date is None:
        # nearest expiry
        def _exp(inst):
            try:
                return datetime.fromtimestamp(int(inst["expTime"]) / 1000, tz=timezone.utc).date()
            except Exception:
                return date.today() + timedelta(days=999)
        instruments = sorted(instruments, key=_exp)
        if instruments:
            expiry_date = _exp(instruments[0])

    # Filter instruments by expiry
    filt = []
    for inst in instruments:
        try:
            d = datetime.fromtimestamp(int(inst["expTime"]) / 1000, tz=timezone.utc).date()
        except Exception:
            continue
        if d == expiry_date:
            filt.append(inst)
    instruments = filt

    # OKX public ticker batch — gives OI + last + bid/ask + mark IV
    # Use /public/market-data with instId family — but easier: per-ticker
    # We'll batch-call tickers; OKX has a 10-per-call limit via /market/tickers
    inst_ids = [i["instId"] for i in instruments if i.get("instId")]
    # Build a map instId → instrument metadata
    meta = {i["instId"]: i for i in instruments if i.get("instId")}

    # /market/tickers supports OPTION instType with uly
    tk_j = _okx_get("/market/tickers", {"instType": "OPTION", "uly": uly})
    tickers = {t["instId"]: t for t in (tk_j.get("data", []) or []) if t.get("instId")}

    by_strike = {}
    for iid, meta_i in meta.items():
        # OKX OPTION instId: BTC-USD-241227-95000-C  (date yymmdd, strike, C/P)
        parts = iid.split("-")
        if len(parts) < 5:
            continue
        try:
            strike = safe_num(parts[-2])
            cp     = parts[-1].upper()
        except Exception:
            continue
        if strike <= 0:
            continue
        t = tickers.get(iid, {})
        d = by_strike.setdefault(strike, {})
        oi  = safe_num(t.get("oi"))
        oic = safe_num(t.get("oiC"))    # OI change (24h)
        iv  = safe_num(t.get("markIV")) / 100.0 if t.get("markIV") else 0.0
        ltp = safe_num(t.get("last"))
        bid = safe_num(t.get("bidPx"))
        ask = safe_num(t.get("askPx"))
        # OKX exposes greeks via /public/opt-summary (one call per instrument — expensive)
        # For v2 we leave greeks = 0; compute_metrics will fall back to BS solver.
        greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        if cp == "C":
            d.update({
                "call_oi": oi, "call_oi_chg": oic, "call_iv": iv,
                "call_delta": greeks["delta"], "call_gamma": greeks["gamma"],
                "call_theta": greeks["theta"], "call_vega": greeks["vega"],
                "call_ltp": ltp, "call_bid": bid, "call_ask": ask,
            })
        else:
            d.update({
                "put_oi": oi, "put_oi_chg": oic, "put_iv": iv,
                "put_delta": greeks["delta"], "put_gamma": greeks["gamma"],
                "put_theta": greeks["theta"], "put_vega": greeks["vega"],
                "put_ltp": ltp, "put_bid": bid, "put_ask": ask,
            })

    for strike, d in by_strike.items():
        d.setdefault("call_oi", 0); d.setdefault("put_oi", 0)
        d.setdefault("call_oi_chg", 0); d.setdefault("put_oi_chg", 0)
        d.setdefault("call_iv", 0); d.setdefault("put_iv", 0)
        d.setdefault("call_ltp", 0); d.setdefault("put_ltp", 0)
        d.setdefault("call_delta", 0); d.setdefault("put_delta", 0)
        d.setdefault("call_gamma", 0); d.setdefault("put_gamma", 0)
        d.setdefault("call_theta", 0); d.setdefault("put_theta", 0)
        d.setdefault("call_vega", 0); d.setdefault("put_vega", 0)
        d.setdefault("call_bid", 0); d.setdefault("call_ask", 0)
        d.setdefault("put_bid", 0); d.setdefault("put_ask", 0)
        d["strike"] = strike
        rows.append(d)

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df, round(spot, 2), expiry_date


# ── Bybit options chain (v5) ──────────────────────────────────────────────────
def fetch_bybit_expiries(symbol="BTC") -> list:
    """Bybit v5 option expiries — returns sorted list of date objects."""
    j = _bybit_get("/tickers", {"category": "option", "baseCoin": symbol})
    exps = set()
    for row in j.get("result", {}).get("list", []) or []:
        sym = row.get("symbol", "")   # e.g. BTC-27DEC24-95000-C
        parts = sym.split("-")
        if len(parts) < 4:
            continue
        try:
            dt = datetime.strptime(parts[1], "%d%b%y").date()
            if dt >= date.today():
                exps.add(dt)
        except Exception:
            pass
    return sorted(exps)


def fetch_bybit_option_chain(symbol="BTC", expiry_date=None):
    """
    Returns DataFrame with the same schema as the Deribit fetcher.
    Bybit's /market/tickers endpoint for options returns OI, mark price, IV,
    bid/ask — but NO greeks. We compute greeks via BS solver downstream.
    """
    j = _bybit_get("/tickers", {"category": "option", "baseCoin": symbol})
    rows = j.get("result", {}).get("list", []) or []
    if not rows:
        return pd.DataFrame(), 0.0, expiry_date

    # Spot via Bybit index price
    sp_j = _bybit_get("/tickers", {"category": "linear", "symbol": f"{symbol}USDT"})
    spot = safe_num((sp_j.get("result", {}).get("list", [{}])[0] or {}).get("lastPrice"))

    # Filter by expiry
    parsed = []
    for row in rows:
        sym = row.get("symbol", "")
        parts = sym.split("-")
        if len(parts) < 4:
            continue
        try:
            dt = datetime.strptime(parts[1], "%d%b%y").date()
        except Exception:
            continue
        if expiry_date and dt != expiry_date:
            continue
        try:
            strike = safe_num(parts[2])
        except Exception:
            continue
        cp = parts[3].upper()
        parsed.append((dt, strike, cp, row))

    if not parsed:
        return pd.DataFrame(), spot, expiry_date

    if expiry_date is None:
        # nearest expiry
        parsed.sort(key=lambda x: x[0])
        expiry_date = parsed[0][0]
        parsed = [p for p in parsed if p[0] == expiry_date]

    by_strike = {}
    for dt, strike, cp, row in parsed:
        d = by_strike.setdefault(strike, {"strike": strike,
                                          "call_oi": 0, "put_oi": 0,
                                          "call_oi_chg": 0, "put_oi_chg": 0,
                                          "call_iv": 0, "put_iv": 0,
                                          "call_ltp": 0, "put_ltp": 0,
                                          "call_delta": 0, "put_delta": 0,
                                          "call_gamma": 0, "put_gamma": 0,
                                          "call_theta": 0, "put_theta": 0,
                                          "call_vega": 0, "put_vega": 0,
                                          "call_bid": 0, "call_ask": 0,
                                          "put_bid": 0, "put_ask": 0})
        oi  = safe_num(row.get("openInterest"))
        iv  = safe_num(row.get("markIv")) / 100.0 if row.get("markIv") else 0.0
        ltp = safe_num(row.get("markPrice"))
        bid = safe_num(row.get("bid1Price"))
        ask = safe_num(row.get("ask1Price"))
        if cp == "C":
            d.update({"call_oi": oi, "call_iv": iv, "call_ltp": ltp,
                      "call_bid": bid, "call_ask": ask})
        else:
            d.update({"put_oi": oi, "put_iv": iv, "put_ltp": ltp,
                      "put_bid": bid, "put_ask": ask})

    df = pd.DataFrame(list(by_strike.values())).sort_values("strike").reset_index(drop=True)
    return df, round(spot, 2), expiry_date


# ── Binance spot + funding rate (cross-exchange reference) ────────────────────
def fetch_binance_spot(symbol="BTC") -> dict:
    """Binance spot + perpetual funding (for cross-venue triangulation)."""
    out = {"spot": 0.0, "perp_mark": 0.0, "funding_rate": 0.0, "next_funding_ms": 0}
    sp = _binance_get("/api/v3/ticker/price", {"symbol": f"{symbol}USDT"})
    out["spot"] = safe_num(sp.get("price"))
    # Perpetual futures
    fp = _binance_get("/fapi/v1/premiumIndex", {"symbol": f"{symbol}USDT"})
    out["perp_mark"]      = safe_num(fp.get("markPrice"))
    out["funding_rate"]   = safe_num(fp.get("lastFundingRate"))
    out["next_funding_ms"]= safe_num(fp.get("nextFundingTime"))
    return out


# ── CoinGecko aggregate spot ──────────────────────────────────────────────────
def fetch_coingecko_spot(coin="bitcoin") -> float:
    """CoinGecko aggregate spot — cross-venue reference price."""
    j = _coingecko_get(f"/simple/price", {"ids": coin, "vs_currencies": "usd"})
    return safe_num(j.get(coin, {}).get("usd"))


# ── Unified chain fetcher (multi-exchange) ────────────────────────────────────
def fetch_chain_multi(symbol="BTC", expiry_arg=None, exchange="Deribit"):
    """
    Routes the chain fetch to the chosen exchange.
    Returns (df, spot, expiry_date).
    Falls back to Deribit if the chosen exchange fails or returns empty.
    """
    if exchange == "OKX":
        try:
            df, spot, exp = fetch_okx_option_chain(symbol, expiry_arg)
            if not df.empty and spot > 0:
                return df, spot, exp
        except Exception:
            pass
    elif exchange == "Bybit":
        try:
            df, spot, exp = fetch_bybit_option_chain(symbol, expiry_arg)
            if not df.empty and spot > 0:
                return df, spot, exp
        except Exception:
            pass
    # Default / fallback: Deribit
    return fetch_option_chain(symbol, expiry_arg)


def fetch_cross_venue_spot(symbol="BTC") -> dict:
    """
    Aggregate spot across Deribit, Binance, CoinGecko — useful for
    arbitrage detection and cross-venue divergence signals.
    """
    venues = {}
    try: venues["deribit"]   = fetch_spot(symbol)
    except Exception: venues["deribit"] = 0
    try: venues["binance"]   = fetch_binance_spot(symbol)["spot"]
    except Exception: venues["binance"] = 0
    try: venues["coingecko"] = fetch_coingecko_spot("bitcoin" if symbol == "BTC" else "ethereum")
    except Exception: venues["coingecko"] = 0
    venues["mean"] = round(np.mean([v for v in venues.values() if v > 0]), 2) if any(venues.values()) else 0
    venues["max_spread_pct"] = 0.0
    pts = [v for v in [venues["deribit"], venues["binance"], venues["coingecko"]] if v > 0]
    if len(pts) >= 2 and venues["mean"] > 0:
        venues["max_spread_pct"] = round((max(pts) - min(pts)) / venues["mean"] * 100, 4)
    return venues


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENGINE — IDENTICAL to crypto v3 (mirrors NIFTY dashboard)
# ─────────────────────────────────────────────────────────────────────────────
def select_atm_band(df, spot, symbol="BTC", band=SIGNAL_BAND):
    if df.empty:
        return pd.DataFrame(), 0.0
    step    = get_strike_step(symbol)
    strikes = sorted(df["strike"].dropna().unique())
    if not strikes:
        return pd.DataFrame(), 0.0
    atm    = min(strikes, key=lambda x: abs(x - spot))
    lo, hi = atm - band * step, atm + band * step
    out    = df[df["strike"].between(lo, hi)].copy()
    return out.sort_values("strike").reset_index(drop=True), float(atm)


def compute_max_pain(df):
    if df.empty:
        return 0.0
    results = {}
    for K in df["strike"].values:
        lower = df[df["strike"] < K]
        upper = df[df["strike"] > K]
        results[K] = ((lower["call_oi"] * (K - lower["strike"])).sum() +
                      (upper["put_oi"] * (upper["strike"] - K)).sum())
    return float(min(results, key=results.get)) if results else 0.0


def compute_metrics(df, spot, symbol="BTC"):
    if df.empty or spot == 0:
        return {}
    wide_df, atm = select_atm_band(df, spot, symbol, STRUCTURAL_BAND)
    tight_df, _  = select_atm_band(df, spot, symbol, SIGNAL_BAND)
    if wide_df.empty or tight_df.empty:
        return {}

    step = get_strike_step(symbol)
    t, w = tight_df.copy(), wide_df.copy()

    net_delta = float((t["call_oi"] * t["call_delta"]).sum() + (t["put_oi"] * t["put_delta"]).sum())
    net_gamma = float((t["call_oi"] * t["call_gamma"]).sum() + (t["put_oi"] * t["put_gamma"]).sum())
    net_theta = float((t["call_oi"] * t["call_theta"]).sum() + (t["put_oi"] * t["put_theta"]).sum())
    net_vega  = float((t["call_oi"] * t["call_vega"]).sum()  + (t["put_oi"] * t["put_vega"]).sum())
    momentum  = float((t["call_oi_chg"] * t["call_delta"]).sum() +
                      (t["put_oi_chg"]  * t["put_delta"]).sum())

    gex = float((w["call_oi"] * w["call_gamma"] - w["put_oi"] * w["put_gamma"]).sum()) * spot

    total_coi = float(w["call_oi"].sum())
    total_poi = float(w["put_oi"].sum())
    pcr = total_poi / total_coi if total_coi > 0 else 1.0

    atm_row = w[w["strike"] == atm]
    atm_iv  = 0.0
    if not atm_row.empty:
        ci, pi = safe_num(atm_row["call_iv"].iloc[0]), safe_num(atm_row["put_iv"].iloc[0])
        atm_iv = float((ci + pi) / max((ci > 0) + (pi > 0), 1))

    iv_vals = w["call_iv"][w["call_iv"] > 0.1].values
    iv_rank = float(np.percentile(iv_vals, 50)) if len(iv_vals) >= 3 else 50.0

    support    = float(w.loc[w["put_oi"].idxmax(),  "strike"])
    resistance = float(w.loc[w["call_oi"].idxmax(), "strike"])
    max_pain   = compute_max_pain(df)
    wall_width = resistance - support

    near = w[w["strike"].between(atm - 2 * step, atm + 2 * step)]
    atm_pressure = float(near["put_oi_chg"].sum() - near["call_oi_chg"].sum())

    total_oi = float((w["call_oi"] + w["put_oi"]).sum())
    near_oi  = float((near["call_oi"] + near["put_oi"]).sum())
    near_oi_concentration = near_oi / total_oi if total_oi > 0 else 0.0

    total_oichg = float((w["call_oi_chg"].abs() + w["put_oi_chg"].abs()).sum())
    near_oichg  = float((near["call_oi_chg"].abs() + near["put_oi_chg"].abs()).sum())
    near_oichg_concentration = near_oichg / total_oichg if total_oichg > 0 else 0.0

    put_side  = w[w["strike"] <= atm][["strike", "put_iv"]].dropna()
    put_side  = put_side[put_side["put_iv"] > 0.1]
    call_side = w[w["strike"] >= atm][["strike", "call_iv"]].dropna()
    call_side = call_side[call_side["call_iv"] > 0.1]
    put_slope = call_slope = 0.0
    if len(put_side) >= 2:
        px = (put_side["strike"] - atm) / max(step, 1)
        put_slope = float(np.polyfit(px, put_side["put_iv"], 1)[0])
    if len(call_side) >= 2:
        cx = (call_side["strike"] - atm) / max(step, 1)
        call_slope = float(np.polyfit(cx, call_side["call_iv"], 1)[0])
    skew_slope = put_slope - call_slope

    gt_ratio = abs(net_gamma) / max(abs(net_theta), 1e-6)

    t["intr_c"] = np.maximum(0, spot - t["strike"])
    t["ev_c"]   = np.maximum(0, t["call_ltp"] - t["intr_c"])
    t["intr_p"] = np.maximum(0, t["strike"] - spot)
    t["ev_p"]   = np.maximum(0, t["put_ltp"] - t["intr_p"])
    ev_sum_c, ev_sum_p = float(t["ev_c"].sum()), float(t["ev_p"].sum())
    ev_ratio = ev_sum_c / ev_sum_p if ev_sum_p > 0 else 1.0

    gex_series = (w["call_oi"] * w["call_gamma"] - w["put_oi"] * w["put_gamma"]).values
    s_arr      = w["strike"].values
    gamma_flip = None
    for i in range(len(gex_series) - 1):
        if gex_series[i] * gex_series[i + 1] < 0:
            gamma_flip = float(s_arr[i])
            break

    return {
        "atm": float(atm), "spot": round(spot, 2),
        "support": support, "resistance": resistance,
        "max_pain": max_pain, "wall_width": round(wall_width, 2),
        "atm_pressure": round(atm_pressure, 0),
        "dist_to_support":    round(spot - support, 2),
        "dist_to_resistance": round(resistance - spot, 2),
        "net_delta":  round(net_delta, 2), "net_gamma": round(net_gamma, 6),
        "net_theta":  round(net_theta, 2), "net_vega":  round(net_vega, 2),
        "momentum":   round(momentum, 2),  "gex":       round(gex, 0),
        "gamma_flip": round(gamma_flip, 0) if gamma_flip else None,
        "pcr":        round(pcr, 3),       "atm_iv":    round(atm_iv, 2),
        "iv_rank":    round(iv_rank, 1),   "ev_ratio":  round(ev_ratio, 3),
        "gt_ratio":   round(gt_ratio, 4),  "skew_slope": round(skew_slope, 3),
        "call_oi_total": round(total_coi, 0),
        "put_oi_total":  round(total_poi, 0),
        "near_oi_concentration":    round(near_oi_concentration, 3),
        "near_oichg_concentration": round(near_oichg_concentration, 3),
        "df_band":   wide_df,
        "df_signal": tight_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BIAS ENGINE — IDENTICAL to crypto v3
# ─────────────────────────────────────────────────────────────────────────────
def classify_gamma_regime(gex, wall_width, momentum, atm_iv, iv_rank, spot, gamma_flip):
    near_flip = (gamma_flip is not None and wall_width > 0 and
                 abs(spot - gamma_flip) < wall_width * 0.15)
    if near_flip:
        regime = "GAMMA FLIP ZONE"
    elif gex > 0 and abs(momentum) < 500:
        regime = "PINNED / RANGE"
    elif gex > 0:
        regime = "RANGE / CHOPPY"
    elif gex < 0 and abs(momentum) > 1000:
        regime = "TREND DAY"
    else:
        regime = "TRANSITION"
    vol_regime = ("LOW VOL" if atm_iv < 40 else
                  ("MID VOL" if atm_iv < 70 else "HIGH VOL"))
    return regime, vol_regime, near_flip


def compute_bias(m, history=None):
    if not m:
        return {"bias_score": 0.0, "confidence": 0.0, "regime": "NO DATA",
                "direction": "NEUTRAL", "factors": [], "vol_regime": "—",
                "near_flip": False}
    history = history or []
    BW = BIAS_WEIGHTS
    direction = confidence = 0.0
    factors = []

    for key, weight, pos_lbl, neg_lbl in [
        ("net_delta", BW["net_delta"], "Net delta bullish", "Net delta bearish"),
        ("momentum",  BW["momentum"],  "OI momentum bullish", "OI momentum bearish"),
    ]:
        v = m.get(key, 0)
        if v > 0:   direction += weight; factors.append(pos_lbl)
        elif v < 0: direction -= weight; factors.append(neg_lbl)

    er = m.get("ev_ratio", 1.0)
    if er >= BW["ev_ratio_bull"]:
        direction += BW["ev_ratio"]; factors.append("Call premium stronger")
    elif er <= BW["ev_ratio_bear"]:
        direction -= BW["ev_ratio"]; factors.append("Put premium stronger")

    ap = m.get("atm_pressure", 0)
    if ap > 0:   direction += BW["atm_pressure"]; factors.append("ATM put support stronger")
    elif ap < 0: direction -= BW["atm_pressure"]; factors.append("ATM call pressure stronger")

    ss = m.get("skew_slope", 0)
    if ss > BW["skew_slope_threshold"]:
        direction -= BW["skew_slope"]; factors.append("Downside IV skew stronger")
    elif ss < -BW["skew_slope_threshold"]:
        direction += BW["skew_slope"]; factors.append("Upside call skew improving")

    regime, vol_regime, near_flip = classify_gamma_regime(
        gex=m["gex"], wall_width=m["wall_width"], momentum=m["momentum"],
        atm_iv=m["atm_iv"], iv_rank=m["iv_rank"],
        spot=m["atm"], gamma_flip=m.get("gamma_flip"))

    if "PINNED" in regime or "RANGE" in regime:
        confidence += BW["regime_range"];      factors.append(f"Regime: {regime}")
    elif "TREND" in regime:
        confidence += BW["regime_trend"];      factors.append(f"Regime: {regime}")
    elif "FLIP" in regime:
        confidence += BW["regime_transition"]; factors.append("⚠ Gamma Flip Zone")
    else:
        confidence += BW["regime_transition"]

    if near_flip:
        factors.append("⚠ Near Gamma Flip — breakout risk")
    if m["near_oi_concentration"] >= BW["near_oi_min"]:
        confidence += BW["near_oi_concentration"]; factors.append("Near-ATM OI concentrated")
    if m["near_oichg_concentration"] >= BW["near_oichg_min"]:
        confidence += BW["near_oichg_concentration"]; factors.append("Fresh OI active near ATM")
    if (abs(m["dist_to_support"]) < BW["wall_proximity_pts"] or
            abs(m["dist_to_resistance"]) < BW["wall_proximity_pts"]):
        confidence += BW["wall_proximity"]; factors.append("Spot close to active wall")

    if history:
        prev = history[-1]
        if m["support"] > safe_num(prev.get("support", m["support"])):
            direction += BW["wall_shift"]; factors.append("Walls shifting higher")
        elif m["support"] < safe_num(prev.get("support", m["support"])):
            direction -= BW["wall_shift"]; factors.append("Walls shifting lower")

    bias_score = max(-100, min(100, round(direction, 1)))
    confidence = max(0, min(100, round(confidence, 1)))
    direction_label = ("BULLISH"  if bias_score >= BW["bias_bull_threshold"] else
                       "BEARISH"  if bias_score <= BW["bias_bear_threshold"] else "NEUTRAL")
    return {"bias_score": bias_score, "confidence": confidence,
            "regime": regime, "vol_regime": vol_regime,
            "near_flip": near_flip, "direction": direction_label,
            "factors": factors[:6]}


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ENGINE — IDENTICAL to crypto v3
# ─────────────────────────────────────────────────────────────────────────────
def strategy_recommendation(bias, m, symbol="BTC"):
    atm        = int(m.get("atm", 0))
    support    = int(m.get("support", 0))
    resistance = int(m.get("resistance", 0))
    step       = get_strike_step(symbol)
    gamma_flip = m.get("gamma_flip")
    iv_rank    = m.get("iv_rank", 50)
    gex        = m.get("gex", 0)
    direction  = bias.get("direction", "NEUTRAL")
    regime     = bias.get("regime", "TRANSITION")
    confidence = bias.get("confidence", 0)
    near_flip  = bias.get("near_flip", False)
    iv_ctx     = f"IV {iv_rank:.0f} ({bias.get('vol_regime','—')})"

    if not atm or not support or not resistance:
        return {"name": "WAIT", "legs": "ATM/wall data unavailable",
                "color": BLUE, "market_mode": regime, "iv_context": iv_ctx}
    if confidence < BIAS_WEIGHTS["confidence_min_strategy"]:
        return {"name": "WAIT", "legs": "No clear edge — await regime confirmation",
                "color": BLUE, "market_mode": regime, "iv_context": iv_ctx}

    if near_flip or "FLIP" in regime:
        flip_str = f"${int(gamma_flip):,}" if gamma_flip else "N/A"
        return {"name": "⚠ Long Straddle / Strangle",
                "legs": (f"EXIT short vega. Buy ${atm:,} C + ${atm:,} P (Straddle) "
                         f"OR Buy ${atm+step:,} C + ${atm-step:,} P (Strangle). Flip@{flip_str}"),
                "color": PINK, "market_mode": regime, "iv_context": f"FLIP ZONE — {iv_ctx}"}

    if gex < 0:
        if direction == "BEARISH":
            return {"name": "Bear Put Spread (GEX Trend Day)",
                    "legs": f"Buy ${atm:,} P | Sell ${atm-2*step:,} P",
                    "color": RED, "market_mode": regime, "iv_context": f"GEX−ve — {iv_ctx}"}
        elif direction == "BULLISH":
            return {"name": "Bull Call Spread (GEX Trend Day)",
                    "legs": f"Buy ${atm:,} C | Sell ${atm+2*step:,} C",
                    "color": GREEN, "market_mode": regime, "iv_context": f"GEX−ve — {iv_ctx}"}
        return {"name": "WAIT — GEX Negative",
                "legs": "No short-vega allowed. Await directional confirmation.",
                "color": BLUE, "market_mode": regime, "iv_context": f"GEX−ve — {iv_ctx}"}

    if "RANGE" in regime or "PINNED" in regime:
        if iv_rank >= 60:
            return {"name": "Iron Condor — High IV",
                    "legs": (f"Sell ${support+step:,} P / Buy ${support-step:,} P  "
                             f"+ Sell ${resistance-step:,} C / Buy ${resistance+step:,} C"),
                    "color": AMBER, "market_mode": regime,
                    "iv_context": f"IV {iv_rank:.0f} — ideal IC"}
        elif iv_rank <= 35:
            return {"name": "Iron Fly — Low IV",
                    "legs": (f"Sell ${atm:,} C + ${atm:,} P  "
                             f"| Buy ${atm+2*step:,} C + ${atm-2*step:,} P"),
                    "color": GOLD, "market_mode": regime,
                    "iv_context": f"IV {iv_rank:.0f} — Iron Fly preferred"}
        return {"name": "Iron Condor",
                "legs": (f"Sell ${support+step:,} P / Buy ${support-step:,} P  "
                         f"+ Sell ${resistance-step:,} C / Buy ${resistance+step:,} C"),
                "color": AMBER, "market_mode": regime, "iv_context": iv_ctx}

    if direction == "BULLISH":
        if iv_rank <= 40:
            return {"name": "Bull Call Spread (Debit)",
                    "legs": f"Buy ${atm:,} C | Sell ${atm+2*step:,} C",
                    "color": GREEN, "market_mode": regime,
                    "iv_context": f"IV {iv_rank:.0f} — cheap debit"}
        return {"name": "Bull Put Spread (Credit)",
                "legs": f"Sell ${support:,} P | Buy ${support-step:,} P",
                "color": GREEN, "market_mode": regime, "iv_context": iv_ctx}

    if direction == "BEARISH":
        if iv_rank <= 40:
            return {"name": "Bear Put Spread (Debit)",
                    "legs": f"Buy ${atm:,} P | Sell ${atm-2*step:,} P",
                    "color": RED, "market_mode": regime,
                    "iv_context": f"IV {iv_rank:.0f} — cheap debit"}
        return {"name": "Bear Call Spread (Credit)",
                "legs": f"Sell ${resistance:,} C | Buy ${resistance+step:,} C",
                "color": RED, "market_mode": regime, "iv_context": iv_ctx}

    return {"name": "WAIT / WATCH",
            "legs": f"Watch ${support:,} support / ${resistance:,} resistance",
            "color": BLUE, "market_mode": regime, "iv_context": iv_ctx}


# ─────────────────────────────────────────────────────────────────────────────
# MARKET SENTIMENTS + OI VELOCITY — IDENTICAL to crypto v3
# ─────────────────────────────────────────────────────────────────────────────
def compute_market_sentiments(history, n=15):
    if not history or len(history) < 3:
        return {}
    h, n_ticks = history[-min(n, len(history)):], len(history[-min(n, len(history)):])

    def _z(arr):
        a = np.array(arr, dtype=float)
        return float((a[-1] - a.mean()) / a.std()) if a.std() > 1e-9 else 0.0

    iv_arr  = [safe_num(x.get("atm_iv", 0))   for x in h]
    gex_arr = [safe_num(x.get("gex", 0))       for x in h]
    pcr_arr = [safe_num(x.get("pcr", 1))       for x in h]
    nd_arr  = [safe_num(x.get("net_delta", 0)) for x in h]

    warming = n_ticks < 5
    vega_z, theta_z = _z(iv_arr), _z(gex_arr)
    oi_z, pos_z     = -_z(pcr_arr), _z(nd_arr)

    def _lbl(z):
        if z >  1.5: return "STRONG", GREEN
        if z >  0.5: return "STRONG", CYAN
        if z < -1.5: return "WEAK",   RED
        if z < -0.5: return "WEAK",   AMBER
        return "NEUTRAL", MUTED

    vl, vc = _lbl(vega_z); tl, tc = _lbl(theta_z); ol, oc = _lbl(oi_z)
    pos_score = round(pos_z, 2)
    pos_dot   = GREEN if pos_score > 0.3 else (RED if pos_score < -0.3 else AMBER)
    abs_ps    = abs(pos_score)
    if abs_ps >= 1.5:
        overall, ov_col = ("STRONGLY BULLISH", GREEN) if pos_score > 0 else ("STRONGLY BEARISH", RED)
    elif abs_ps >= 0.5:
        overall, ov_col = ("BULLISH", CYAN) if pos_score > 0 else ("BEARISH", AMBER)
    else:
        overall, ov_col = "NEUTRAL", MUTED

    return {"vega_label": vl, "vega_color": vc, "theta_label": tl, "theta_color": tc,
            "oi_label": ol, "oi_color": oc, "pos_score": pos_score, "pos_dot": pos_dot,
            "strength_color": pos_dot, "overall": overall, "overall_color": ov_col,
            "pos_caption": (f"Net-delta Z = {pos_score:+.2f} — "
                            f"{'Calls dominating' if pos_score > 0 else 'Puts dominating'}"),
            "n_ticks": n_ticks, "warming": warming}


def compute_oi_velocity(history):
    if not history or len(history) < 3:
        return {"call_vel_zscore": 0.0, "put_vel_zscore": 0.0,
                "alert_level": "NONE", "alert_text": "Insufficient history"}
    call_oi = np.array([safe_num(x.get("call_oi_total", 0)) for x in history], dtype=float)
    put_oi  = np.array([safe_num(x.get("put_oi_total",  0)) for x in history], dtype=float)
    if call_oi.max() == 0 and put_oi.max() == 0:
        nd = np.array([safe_num(x.get("net_delta", 0)) for x in history], dtype=float)
        call_oi, put_oi = np.maximum(nd, 0), np.maximum(-nd, 0)

    def _zl(arr, w=10):
        if len(arr) < 3: return 0.0
        a = arr[-min(w, len(arr)):]
        return float((a[-1] - a.mean()) / a.std()) if a.std() > 1e-9 else 0.0

    cz, pz = _zl(np.diff(call_oi)), _zl(np.diff(put_oi))
    al = "DANGER" if (abs(cz) >= 2 or abs(pz) >= 2) else \
         "WATCH"  if (abs(cz) >= 1 or abs(pz) >= 1) else "NONE"
    return {"call_vel_zscore": round(cz, 2), "put_vel_zscore": round(pz, 2),
            "alert_level": al, "alert_text": f"Call Z={cz:+.2f}σ  Put Z={pz:+.2f}σ"}


# ─────────────────────────────────────────────────────────────────────────────
# Δ-WEIGHTED OI FLOW ENGINE — IDENTICAL to crypto v3
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ts_to_bucket(ts: str):
    try:
        t  = ts.split("T")[-1] if "T" in ts else ts
        hh, mm = int(t[:2]), int(t[3:5])
        return f"{hh:02d}:{(mm // 15) * 15:02d}"
    except Exception:
        return None


def compute_dw_flow_buckets(sym_history: list) -> dict:
    if len(sym_history) < 2:
        return {}
    bucket_data = {}
    for tick in sym_history:
        bkt = _parse_ts_to_bucket(tick.get("ts", ""))
        if bkt is None:
            continue
        if bkt not in bucket_data:
            bucket_data[bkt] = {"call_dw": [], "put_dw": [], "spot": [], "gex": [],
                                "gamma_flip": [], "max_pain": [], "support": [],
                                "resistance": []}
        mom = safe_num(tick.get("momentum", 0))
        bucket_data[bkt]["call_dw"].append(max(-mom, 0))
        bucket_data[bkt]["put_dw"].append(max(mom, 0))
        bucket_data[bkt]["spot"].append(safe_num(tick.get("spot", 0)))
        bucket_data[bkt]["gex"].append(safe_num(tick.get("gex", 0)))
        gf = tick.get("gamma_flip")
        if gf is not None:
            bucket_data[bkt]["gamma_flip"].append(safe_num(gf))
        bucket_data[bkt]["max_pain"].append(safe_num(tick.get("max_pain", 0)))
        bucket_data[bkt]["support"].append(safe_num(tick.get("support", 0)))
        bucket_data[bkt]["resistance"].append(safe_num(tick.get("resistance", 0)))

    labels    = sorted(bucket_data.keys())
    call_flow = [float(np.sum(bucket_data[b]["call_dw"])) for b in labels]
    put_flow  = [float(np.sum(bucket_data[b]["put_dw"]))  for b in labels]
    net_flow  = [p - c for c, p in zip(call_flow, put_flow)]

    def _last_valid(lst): return lst[-1] if lst else None
    return {
        "labels": labels, "call_flow": call_flow, "put_flow": put_flow,
        "net_flow": net_flow,
        "spot":       [float(np.mean(bucket_data[b]["spot"])) for b in labels],
        "gex":        [float(np.mean(bucket_data[b]["gex"]))  for b in labels],
        "gamma_flip": [_last_valid(bucket_data[b]["gamma_flip"])    for b in labels],
        "max_pain":   [float(np.mean(bucket_data[b]["max_pain"]))   for b in labels],
        "support":    [float(np.mean(bucket_data[b]["support"]))    for b in labels],
        "resistance": [float(np.mean(bucket_data[b]["resistance"])) for b in labels],
        "delta_active": any(abs(v) > 10 for v in call_flow + put_flow),
    }


def compute_dw_composite_bias(bkt: dict, expiry_str=None) -> dict:
    if not bkt or len(bkt.get("labels", [])) < 2:
        return {"score": 0, "direction": "NEUTRAL", "confidence": 0,
                "components": {}, "narrative": "Need 2+ buckets — collecting data."}
    net_flow, gex_arr = bkt["net_flow"], bkt["gex"]
    gf_arr, spot_arr  = bkt["gamma_flip"], bkt["spot"]

    recent_flow   = net_flow[-1]
    flow_3        = float(np.mean(net_flow[-3:])) if len(net_flow) >= 3 else recent_flow
    session_range = max(abs(f) for f in net_flow) if any(f != 0 for f in net_flow) else 1.0
    flow_norm     = max(-1.0, min(1.0, flow_3 / max(session_range, 1.0)))
    c1_score      = round(flow_norm * 35, 1)
    c1_label      = (f"Net Δ-flow: {flow_3:+,.0f} "
                     f"({'bullish' if flow_3 > 0 else 'bearish'}) "
                     f"[session max: {session_range:,.0f}]")

    if len(net_flow) >= 2:
        accel      = net_flow[-1] - net_flow[-2]
        accel_norm = max(-1.0, min(1.0, accel / max(session_range, 1.0)))
        c2_score   = round(accel_norm * 25, 1)
        c2_label   = (f"Flow accel: {accel:+,.0f} "
                      f"({'accelerating ↑' if accel_norm > 0.1 else 'decelerating ↓' if accel_norm < -0.1 else 'steady →'})")
    else:
        c2_score, c2_label = 0, "Acceleration: need 2+ buckets"

    gex_now = gex_arr[-1] if gex_arr else 0
    if gex_now > 0:
        c3_score, c3_label = +10, f"GEX +{gex_now:,.0f} (long-gamma — pinning tendency)"
    elif gex_now < 0:
        c3_score, c3_label = -20, f"GEX {gex_now:,.0f} (short-gamma — trending/amplifying regime)"
    else:
        c3_score, c3_label = 0, "GEX near zero — unstable transition"

    spot_now = spot_arr[-1] if spot_arr else 0
    gf_now   = next((g for g in reversed(gf_arr) if g is not None), None)
    if gf_now and spot_now > 0:
        dist      = spot_now - gf_now
        dist_norm = max(-1.0, min(1.0, dist / max(abs(dist) + 1e-9, 500)))
        c4_score  = round(dist_norm * 20, 1)
        side      = "above flip (stable)" if dist > 0 else "BELOW flip (short-gamma danger)"
        c4_label  = f"Spot {dist:+.0f} from flip @ ${gf_now:,.0f} — {side}"
    else:
        c4_score, c4_label = 0, "Gamma flip not available"

    c5_score, c5_label = 0, "Max pain: not expiry week (no weight)"
    if expiry_str:
        try:
            exp_date    = date.fromisoformat(str(expiry_str)[:10])
            days_to_exp = (exp_date - date.today()).days
            if days_to_exp <= 2:
                mp_now = bkt["max_pain"][-1] if bkt["max_pain"] else 0
                if mp_now > 0 and spot_now > 0:
                    pull      = mp_now - spot_now
                    pull_norm = max(-1.0, min(1.0, pull / max(spot_now * 0.01, 500)))
                    c5_score  = round(pull_norm * 10, 1)
                    c5_label  = (f"Max pain gravity: {pull:+.0f} toward "
                                 f"${mp_now:,.0f} ({days_to_exp}d to exp)")
        except Exception:
            pass

    total = max(-100, min(100, c1_score + c2_score + c3_score + c4_score + c5_score))
    if total >= 45:
        direction, confidence = "BULLISH", min(100, int(abs(total) * 1.2))
    elif total >= 15:
        direction, confidence = "MILD BULLISH", int(abs(total) * 0.9)
    elif total <= -45:
        direction, confidence = "BEARISH", min(100, int(abs(total) * 1.2))
    elif total <= -15:
        direction, confidence = "MILD BEARISH", int(abs(total) * 0.9)
    else:
        direction, confidence = "NEUTRAL", max(0, 100 - int(abs(total) * 3))

    _delta_ok  = bkt.get("delta_active", False)
    _flow_note = (
        f"Net Δ-flow {'positive — put-writing dominates' if flow_3 > 0 else 'negative — call-writing dominates'}"
        if _delta_ok else
        "⚠️ Δ-flow proxy — bias reflects GEX + flip, NOT direct delta flow")
    narrative = (
        f"{_flow_note} | "
        f"GEX {'long-gamma (range-bound)' if gex_now > 0 else 'short-gamma (amplifying)'} | "
        f"{'Above' if (gf_now and spot_now > gf_now) else 'Below'} gamma flip"
        + (f" | {c5_label}" if c5_score != 0 else ""))

    return {"score": round(total, 1), "direction": direction, "confidence": confidence,
            "components": {
                "net_flow_dir": (c1_score, c1_label),
                "flow_accel":   (c2_score, c2_label),
                "gex_regime":   (c3_score, c3_label),
                "flip_side":    (c4_score, c4_label),
                "max_pain":     (c5_score, c5_label)},
            "narrative": narrative, "delta_active": _delta_ok}


# ═════════════════════════════════════════════════════════════════════════════
# NEW v2 ANALYSIS LAYERS  (ported & adapted from nifty_streamlit_v5_fixed.py)
#   • classify_iv_smile_scenario  — 9-state smile classifier (Sc01..Sc09)
#   • get_s34_band / get_smile_bucket / classify_quadrant
#   • detect_divergence_type + _compute_divergence_proximity
#   • generate_combined_decision  — the Combined Bias Decision engine
#   • _compute_grf  — Greek Risk Framework (0–10 conviction scorer)
#   • compute_leading_signals  — divergence / velocity / exhaustion / inter-expiry
#   • compute_shantanu_view    — ND/NDM Decision Matrix + Enhanced NDM
# All operate on the existing `m` metrics dict + `df_band` and need NO new
# API calls. Crypto-adapted (no India VIX, uses DVOL + perp basis instead).
# ═════════════════════════════════════════════════════════════════════════════

# ── Smile scenario metadata ──────────────────────────────────────────────────
_SMILE_SCENARIOS = {
    1:  {"name": "Put-Skew (Hedge Build)",       "bucket": "BEARISH_FEAR",      "color": "#DC2626"},
    2:  {"name": "Crash Fear (Wing Bid)",        "bucket": "BEARISH_FEAR",      "color": "#7F1D1D"},
    3:  {"name": "Put-Skew + IV Expansion",      "bucket": "BEARISH_FEAR",      "color": "#B91C1C"},
    4:  {"name": "Smile Flattening (Relief)",    "bucket": "BULLISH_NEUTRAL",   "color": "#10B981"},
    5:  {"name": "Call-Skew (Risk-On)",          "bucket": "BULLISH_NEUTRAL",   "color": "#059669"},
    6:  {"name": "Call Wing Lift",               "bucket": "BULLISH_NEUTRAL",   "color": "#34D399"},
    7:  {"name": "Symmetric Smile (Stable)",     "bucket": "BULLISH_NEUTRAL",   "color": "#6B7280"},
    8:  {"name": "Coiled Spring (IV Compression)","bucket": "NEUTRAL_COMPRESSED","color": "#9333EA"},
    9:  {"name": "Term-Structure Inversion",     "bucket": "NEUTRAL_COMPRESSED","color": "#B45309"},
}

_QUADRANT_META = {
    "Q1": {"name": "Bullish · Confirmed", "short": "Q1 BULL",
           "color": "#059669", "badge_bg": "#ECFDF5",
           "description": "S3/4 bullish + smile risk-on.",
           "action": "Look for long entries on dips; favour call spreads and put selling."},
    "Q2": {"name": "Bearish · Confirmed", "short": "Q2 BEAR",
           "color": "#DC2626", "badge_bg": "#FEE2E2",
           "description": "S3/4 bullish + smile fear.",
           "action": "Bull-bear divergence — reduce longs, hedge with put spreads."},
    "Q3": {"name": "Reversal Watch (Bear→Bull)", "short": "Q3 REV",
           "color": "#D97706", "badge_bg": "#FFFBEB",
           "description": "S3/4 bearish + smile relief.",
           "action": "Bear trap risk — cover shorts, watch for upside confirmation."},
    "Q4": {"name": "Bearish · Confirmed", "short": "Q4 BEAR",
           "color": "#7F1D1D", "badge_bg": "#FECACA",
           "description": "S3/4 bearish + smile fear.",
           "action": "Look for short entries on rallies; favour put spreads and call selling."},
    "CN": {"name": "Neutral · Compressed", "short": "CN NEUTRAL",
           "color": "#6B7280", "badge_bg": "#F3F4F6",
           "description": "S3/4 in dead band; vol squeeze likely.",
           "action": "Favour long-vol structures (straddles); avoid directional bets."},
}


def classify_iv_smile_scenario(df_band, m, spot, iv_smile_history=None):
    """
    Classify the current IV-smile shape into one of 9 scenarios (Sc01..Sc09).
    Adapted from nifty v5 — uses crypto-appropriate thresholds.
    Reads put/call IV at OTM wings relative to ATM.
    Returns dict: scenario_id, name, bucket, color, iv_rank, skew_now.
    """
    if df_band is None or df_band.empty:
        return {"scenario_id": 0, "name": "Insufficient data", "bucket": "NEUTRAL",
                "color": "#6B7280", "iv_rank": m.get("iv_rank", 50), "skew_now": 0.0}

    atm        = safe_num(m.get("atm", spot))
    step       = get_strike_step(m.get("_symbol", "BTC"))
    df         = df_band.copy()
    for c in ("strike", "call_iv", "put_iv"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    atm_rows   = df[df["strike"].between(atm - step, atm + step)]
    otm_puts   = df[df["strike"] < atm - 3 * step]
    otm_calls  = df[df["strike"] > atm + 3 * step]

    atm_iv     = float(atm_rows["call_iv"].mean() or atm_rows["put_iv"].mean() or 0) if not atm_rows.empty else 0
    put_wing   = float(otm_puts["put_iv"].mean())  if not otm_puts.empty  else 0
    call_wing  = float(otm_calls["call_iv"].mean()) if not otm_calls.empty else 0

    if atm_iv <= 0:
        return {"scenario_id": 0, "name": "ATM IV missing", "bucket": "NEUTRAL",
                "color": "#6B7280", "iv_rank": m.get("iv_rank", 50), "skew_now": 0.0}

    put_skew   = (put_wing  - atm_iv) if put_wing  > 0 else 0
    call_skew  = (call_wing - atm_iv) if call_wing > 0 else 0
    skew_now   = round(put_skew - call_skew, 2)
    iv_rank    = safe_num(m.get("iv_rank", 50))

    # Decision tree
    if put_skew > 12 and iv_rank >= 65:
        sid = 2  # Crash Fear
    elif put_skew > 8 and iv_rank >= 50:
        sid = 3  # Put-Skew + IV Expansion
    elif put_skew > 6:
        sid = 1  # Put-Skew (Hedge Build)
    elif call_skew > 6 and iv_rank <= 45:
        sid = 5  # Call-Skew (Risk-On)
    elif call_skew > 4:
        sid = 6  # Call Wing Lift
    elif abs(put_skew) < 3 and abs(call_skew) < 3 and iv_rank >= 40:
        sid = 7  # Symmetric Smile (Stable)
    elif abs(put_skew) < 3 and abs(call_skew) < 3 and iv_rank <= 20:
        sid = 8  # Coiled Spring
    elif put_skew < -2 and call_skew < -2:
        sid = 4  # Smile Flattening (Relief)
    else:
        sid = 9  # Term-Structure Inversion / catchall

    meta = _SMILE_SCENARIOS.get(sid, _SMILE_SCENARIOS[9])
    return {"scenario_id": sid, "name": meta["name"], "bucket": meta["bucket"],
            "color": meta["color"], "iv_rank": iv_rank, "skew_now": skew_now,
            "put_skew": put_skew, "call_skew": call_skew}


def get_s34_band(score: float) -> str:
    """Map S3/4 bias score to a band A..E."""
    if score >= 51:   return "A"
    if score >= 16:   return "B"
    if score > -16:   return "C"
    if score > -51:   return "D"
    return "E"


def get_smile_bucket(scenario_id: int) -> str:
    return _SMILE_SCENARIOS.get(scenario_id, {}).get("bucket", "NEUTRAL")


def classify_quadrant(s34_score: float, scenario_id: int) -> dict:
    """4-Quadrant classification (Q1..Q4 + CN)."""
    band         = get_s34_band(s34_score)
    smile_bucket = get_smile_bucket(scenario_id)
    if band == "C":
        q = "CN"
    elif band in ("A", "B"):
        q = "Q1" if smile_bucket != "BEARISH_FEAR" else "Q2"
    else:
        q = "Q3" if smile_bucket == "BULLISH_NEUTRAL" else "Q4"
    meta = _QUADRANT_META[q]
    return {
        "quadrant": q, "name": meta["name"], "short": meta["short"],
        "color": meta["color"], "badge_bg": meta["badge_bg"],
        "description": meta["description"], "action": meta["action"],
        "smile_bucket": smile_bucket, "s34_band": band,
    }


# ── Divergence metadata ──────────────────────────────────────────────────────
_DIVERGENCE_META = {
    1: {"type": "Type 1 — Capitulation Bottom", "color": "#059669", "badge_bg": "#ECFDF5",
        "warning": "POTENTIAL CAPITULATION BOTTOM — do not initiate new shorts at these levels",
        "detail": "S3/4 at extreme bear AND IV smile shows Crash Fear. Hedgers already in — they are sellers of the rally."},
    2: {"type": "Type 2 — Structural Ceiling", "color": "#D97706", "badge_bg": "#FFFBEB",
        "warning": "STRUCTURAL CEILING — smart money hedging longs; reduce long exposure",
        "detail": "S3/4 strong bull BUT smile is building put-skew. Smart money reducing risk at the top."},
    3: {"type": "Type 3 — Squeeze Warning", "color": "#9333EA", "badge_bg": "#FAF5FF",
        "warning": "SQUEEZE WARNING — breakout imminent; direction unknown; favour long vol",
        "detail": "S3/4 neutral AND smile compressed (Coiled Spring). Directional biases unreliable; favour straddles."},
    4: {"type": "Type 4 — Bear Trap", "color": "#2563EB", "badge_bg": "#EFF6FF",
        "warning": "BEAR TRAP RISK — call buying despite bearish OI structure; cover shorts",
        "detail": "S3/4 negative BUT smile shows call buying / relief. Sentiment recovering — squeeze risk."},
    5: {"type": "Type 5 — Pre-Move Setup", "color": "#B45309", "badge_bg": "#FEF3C7",
        "warning": "PRE-MOVE SETUP — smart money positioning; wait for OI velocity confirmation",
        "detail": "PCR extreme AND S3/4 moderately biased AND smile aligned. Watch OI velocity + gamma flip."},
}


def detect_divergence_type(s34_score, scenario_id, pcr=1.0, div_proximity=0.0):
    """Returns divergence dict or None if no divergence is active."""
    band         = get_s34_band(s34_score)
    smile_bucket = get_smile_bucket(scenario_id)
    # Hard triggers
    if band == "E" and scenario_id == 2:
        d = _DIVERGENCE_META[1].copy()
        d["strength"] = "HARD"; return d
    if band == "A" and scenario_id in (1, 3):
        d = _DIVERGENCE_META[2].copy()
        d["strength"] = "HARD"; return d
    if band == "C" and scenario_id == 8:
        d = _DIVERGENCE_META[3].copy()
        d["strength"] = "HARD"; return d
    if band in ("D", "E") and scenario_id in (4, 5, 6):
        d = _DIVERGENCE_META[4].copy()
        d["strength"] = "HARD"; return d
    if (pcr < 0.70 or pcr > 1.55) and abs(s34_score) > 15 and scenario_id not in (7, 8):
        d = _DIVERGENCE_META[5].copy()
        d["strength"] = "HARD"; return d
    # Soft / approaching
    if div_proximity >= 60:
        # find the closest hard trigger by re-evaluating proximity components
        if s34_score < -40:   d = _DIVERGENCE_META[1].copy()
        elif s34_score > 40:  d = _DIVERGENCE_META[2].copy()
        elif abs(s34_score) < 15: d = _DIVERGENCE_META[3].copy()
        elif s34_score < -16: d = _DIVERGENCE_META[4].copy()
        else:                 d = _DIVERGENCE_META[5].copy()
        d["strength"] = "SOFT"; return d
    return None


def _compute_divergence_proximity(s34_score, scenario_id, pcr, iv_rank):
    """0–100. >= 60 triggers an APPROACHING DIVERGENCE alert."""
    scores = []
    t1_s34 = max(0, min(100, (abs(s34_score) - 40) / 11 * 100)) if s34_score < -40 else 0
    t1_smile = max(0, min(100, (iv_rank - 50) / 15 * 100)) if iv_rank > 50 else 0
    if scenario_id in (1, 2, 3): t1_smile = max(t1_smile, 60)
    scores.append((t1_s34 + t1_smile) / 2)
    t2_s34 = max(0, min(100, (s34_score - 40) / 11 * 100)) if s34_score > 40 else 0
    t2_smile = 70 if scenario_id in (1, 3) else (30 if scenario_id in (2, 9) else 0)
    scores.append((t2_s34 + t2_smile) / 2)
    t3_neutral = max(0, min(100, (15 - abs(s34_score)) / 15 * 100)) if abs(s34_score) < 15 else 0
    t3_compress = max(0, min(100, (25 - iv_rank) / 5 * 100)) if iv_rank < 25 else 0
    scores.append((t3_neutral + t3_compress) / 2)
    t5_pcr = 0
    if pcr < 0.70:    t5_pcr = max(0, min(100, (0.70 - pcr) / 0.20 * 100))
    elif pcr > 1.55:  t5_pcr = max(0, min(100, (pcr - 1.55) / 0.25 * 100))
    t5_s34 = max(0, min(100, (abs(s34_score) - 15) / 15 * 100)) if abs(s34_score) > 15 else 0
    scores.append((t5_pcr + t5_s34) / 2)
    return round(max(scores), 1) if scores else 0.0


def generate_combined_decision(m, spot, bias, smile, history=None,
                                vwap_or=None, ts_signal=None, vix_signal=None,
                                enhanced_price_bias=None):
    """
    Combine S3/4 options bias + IV smile + PCR + price confirmation layers
    into a single Quadrant + Action + Confidence panel.
    Adapted from nifty v5 generate_combined_decision — for crypto we use
    DVOL/perp-basis as the VIX surrogate.
    """
    s34_score      = safe_num(bias.get("bias_score", 0))
    s34_dir        = bias.get("direction", "NEUTRAL")
    s34_breakdown  = bias.get("factors", [])
    scenario_id    = smile.get("scenario_id", 0)
    pcr            = safe_num(m.get("pcr", 1.0))
    iv_rank        = safe_num(m.get("iv_rank", 50))

    quad = classify_quadrant(s34_score, scenario_id)
    div_prox = _compute_divergence_proximity(s34_score, scenario_id, pcr, iv_rank)
    div      = detect_divergence_type(s34_score, scenario_id, pcr, div_prox)

    # Confidence: how far score is from neutral, plus smile alignment
    base_conf = min(100, abs(s34_score) * 1.1)
    smile_aligned = (s34_score > 15 and smile["bucket"] == "BULLISH_NEUTRAL") or \
                     (s34_score < -15 and smile["bucket"] == "BEARISH_FEAR")
    if smile_aligned: base_conf = min(100, base_conf + 15)
    elif smile["bucket"] in ("BEARISH_FEAR", "BULLISH_NEUTRAL"):
        base_conf = max(0, base_conf - 10)
    if div and div.get("strength") == "HARD":
        base_conf = max(0, base_conf - 20)

    if   base_conf >= 75: conf_label, conf_color = "HIGH",      "#059669"
    elif base_conf >= 50: conf_label, conf_color = "MODERATE",  "#D97706"
    else:                 conf_label, conf_color = "LOW",       "#DC2626"

    # Explanation lines
    lines = [
        f"S3/4 bias: <strong>{s34_dir}</strong> at {s34_score:+.0f}/100 ({quad['s34_band']}-band)",
        f"IV Smile: <strong>{smile.get('name','—')}</strong> ({smile.get('bucket','—')})",
        f"PCR {pcr:.2f} · IV rank {iv_rank:.0f} · Max-pain pull ${safe_num(m.get('max_pain',spot)) - spot:+,.0f}",
    ]
    if div:
        lines.append(f"<strong style='color:{div['color']}'>{div['type']}</strong> — {div['warning']}")

    # Enhanced price layer override (if supplied)
    overridden = False
    enhanced_score = 0
    if enhanced_price_bias and enhanced_price_bias.get("enhanced_score", 0) != 0:
        enhanced_score = enhanced_price_bias["enhanced_score"]
        # If enhanced layer strongly disagrees with S3/4, override quadrant
        if (s34_score > 15 and enhanced_score < -25) or (s34_score < -15 and enhanced_score > 25):
            overridden = True
            quad = classify_quadrant(enhanced_score, scenario_id)

    return {
        "quadrant":         quad["quadrant"],
        "quadrant_short":   quad["short"],
        "quadrant_color":   quad["color"],
        "badge_bg":         quad["badge_bg"],
        "action":           quad["action"],
        "confidence_label": conf_label,
        "confidence_color": conf_color,
        "confidence_pct":   base_conf,
        "explanation_lines": lines,
        "divergence":       div,
        "s34_score":        s34_score,
        "s34_direction":    s34_dir,
        "smile_scenario":   smile.get("name", "—"),
        "pcr":              pcr,
        "quadrant_overridden": overridden,
        "enhanced_score":   enhanced_score,
    }


def _compute_grf(m_dict, spot_px):
    """
    Greek Risk Framework scorer — adapted from nifty v5 _compute_grf.
    Returns 0–10 conviction score split as Gamma(0-3) + Delta(0-3) + Momentum(0-4).
    For crypto, we use the existing m['gex'], m['net_delta'], m['momentum'] etc.
    """
    nd      = safe_num(m_dict.get("net_delta", 0))
    mom     = safe_num(m_dict.get("momentum", 0))
    gex     = safe_num(m_dict.get("gex", 0))
    d_res   = safe_num(m_dict.get("dist_to_resistance", 0))
    d_sup   = safe_num(m_dict.get("dist_to_support", 0))
    mp_val  = safe_num(m_dict.get("max_pain", spot_px))
    pcr     = safe_num(m_dict.get("pcr", 1.0))
    iv_r    = safe_num(m_dict.get("iv_rank", 50))
    gflip   = m_dict.get("gamma_flip")
    sup_w   = safe_num(m_dict.get("support", 0))
    res_w   = safe_num(m_dict.get("resistance", 0))
    fac     = []

    _oi_scale = max(1.0, safe_num(m_dict.get("call_oi_total", 0)) + safe_num(m_dict.get("put_oi_total", 0)))
    nd_sig    = abs(nd)  > max(100, _oi_scale * 0.001)
    mom_sig   = abs(mom) > max(50,  _oi_scale * 0.0005)

    # 1. Gamma Score (0-3)
    g = 0
    inside_band = (d_res > 0) and (d_sup > 0)
    if inside_band:
        min_buf = (min(d_res, d_sup) / spot_px * 100) if spot_px > 0 else 0
        if   min_buf > 1.0: g += 2; fac.append(f"Walls {min_buf:.1f}% from spot — safe sell range")
        elif min_buf > 0.5: g += 1; fac.append(f"Moderate wall buffer ({min_buf:.1f}%)")
        else:                       fac.append(f"Walls very close ({min_buf:.1f}%) — elevated gamma risk")
        if gex > 0: g += 1
    else:
        broke_dir = "above resistance" if d_res <= 0 else "below support"
        fac.append(f"⚠ Spot {broke_dir} — gamma range breached, avoid selling")
    g = min(g, 3)

    # 2. Delta Score (0-3)
    d = 0
    nd_bull = nd > 0
    if nd_sig:
        d += 1
        fac.append(f"Net delta {'bullish' if nd_bull else 'bearish'} ({nd:+,.0f})")
    if nd_sig and mp_val > 0 and spot_px > 0:
        mp_bull = mp_val > spot_px
        if nd_bull == mp_bull and abs(mp_val - spot_px) > 20:
            d += 1
            fac.append(f"Max pain (${int(mp_val)}) confirms {'upside' if mp_bull else 'downside'} pull")
    if nd_sig and gflip is not None:
        gf = safe_num(gflip)
        if gf > 0 and (spot_px > gf) == nd_bull:
            d += 1
            fac.append(f"Spot {'above' if spot_px > gf else 'below'} gamma flip (${int(gf)}) — regime aligned")
    d = min(d, 3)

    # 3. Momentum Score (0-4)
    ms       = 0
    mom_bull = mom > 0
    if not mom_sig:
        ms = 1
    elif nd_sig and mom_bull == nd_bull:
        ms = 3
        fac.append(f"OI momentum confirms {'bullish' if mom_bull else 'bearish'} flow ({mom:+,.0f})")
    elif nd_sig and mom_bull != nd_bull:
        ms = 0
        fac.append("⚠ Momentum contradicts net delta — divergence, cut size")
    else:
        ms = 2
    if nd_sig:
        if nd_bull and pcr >= 1.2:
            ms = min(ms + 1, 4); fac.append(f"PCR {pcr:.2f} confirms bullish support")
        elif not nd_bull and pcr <= 0.8:
            ms = min(ms + 1, 4); fac.append(f"PCR {pcr:.2f} confirms bearish pressure")
    if   iv_r <= 35: ms = min(ms + 1, 4)   # calm IV = ideal sell environment
    elif iv_r >= 70: ms = max(ms - 1, 0)   # high IV = elevated risk
    ms = min(ms, 4)

    total = g + d + ms

    # Bias label
    if   nd > 0 and mom > 0:   bias_s = "BULLISH"
    elif nd < 0 and mom < 0:   bias_s = "BEARISH"
    elif nd > 0 and mom < 0:   bias_s = "MIXED — delta bull / momentum fading"
    elif nd < 0 and mom > 0:   bias_s = "MIXED — delta bear / momentum recovering"
    elif nd > 0:               bias_s = "BULLISH (flat momentum)"
    elif nd < 0:               bias_s = "BEARISH (flat momentum)"
    elif mom > 0:              bias_s = "NEUTRAL — flow tilting bullish"
    elif mom < 0:              bias_s = "NEUTRAL — flow tilting bearish"
    else:                      bias_s = "NEUTRAL"

    if   total >= 8: conv, cc, sl, rtxt = "HIGH CONVICTION", "#059669", "Full size",  "All Greeks aligned. Deploy full planned size within the gamma range."
    elif total >= 6: conv, cc, sl, rtxt = "GOOD SETUP",      "#10B981", "Standard",   "Most signals confirm. Trade standard size; monitor the weakest Greek."
    elif total >= 4: conv, cc, sl, rtxt = "MODERATE",        "#D97706", "Half size",  "Mixed signals. Half size only, or wait 30–60 min for clarity."
    elif total >= 2: conv, cc, sl, rtxt = "LOW",             "#F59E0B", "Avoid",      "Greeks not aligned. Watch only — do not deploy capital now."
    else:            conv, cc, sl, rtxt = "NO TRADE",        "#DC2626", "Stay out",   "Conflicting signals. Protect capital and wait for a cleaner setup."

    iv_env = "Low IV — ideal" if iv_r <= 35 else ("High IV — caution" if iv_r >= 70 else "Mid IV — ok")
    return dict(
        total=total, g=g, d=d, ms=ms,
        bias_s=bias_s, conv=conv, cc=cc, sl=sl, rtxt=rtxt,
        fac=fac[:4],
        gamma_range=f"${int(sup_w):,}–${int(res_w):,}" if sup_w and res_w else "—",
        iv_env=iv_env, iv_r=iv_r,
    )


def compute_leading_signals(m, bias, spot, history, smile, expiry_list=None):
    """
    Leading Signals / Early Warning panel data — adapted from nifty v5.
    Computes:
      • divergence_proximity (0–100)
      • bias_velocity + acceleration
      • gamma_flip_proximity
      • OI momentum exhaustion
      • inter-expiry OI flow (only if multiple expiries available)
    """
    out = {
        "div_proximity":     0.0,
        "velocity":          0.0,
        "acceleration":      0.0,
        "gamma_flip_proximity": None,
        "oi_exhaustion":     None,
        "inter_expiry":      None,
    }

    s34_score = safe_num(bias.get("bias_score", 0))
    scen_id   = smile.get("scenario_id", 0)
    pcr       = safe_num(m.get("pcr", 1.0))
    iv_rank   = safe_num(m.get("iv_rank", 50))

    out["div_proximity"] = _compute_divergence_proximity(s34_score, scen_id, pcr, iv_rank)

    # Bias velocity from history
    if history and len(history) >= 3:
        scores = [safe_num(h.get("bias_score", 0)) for h in history[-5:]]
        scores.append(s34_score)
        if len(scores) >= 3:
            vels = [scores[i] - scores[i-1] for i in range(1, len(scores))]
            out["velocity"]     = round(vels[-1], 1) if vels else 0.0
            if len(vels) >= 2:
                out["acceleration"] = round(vels[-1] - vels[-2], 1)

    # Gamma flip proximity
    gflip = m.get("gamma_flip")
    if gflip and gflip > 0 and spot > 0:
        flip_dist = abs(spot - gflip)
        wall_w    = safe_num(m.get("wall_width", 500))
        step      = max(wall_w / 20, 50)
        thresh    = max(2.0 * step, 100)
        out["gamma_flip_proximity"] = {
            "flip_strike":         round(gflip, 0),
            "distance_pts":        round(flip_dist, 1),
            "threshold_pts":       round(thresh, 1),
            "pct_of_threshold":    round(flip_dist / thresh * 100, 1) if thresh > 0 else 0,
            "side":                "ABOVE" if spot > gflip else "BELOW",
            "zone":                "FLIP_ZONE" if flip_dist < thresh else "SAFE",
            "regime_risk":         "HIGH" if flip_dist < step else ("ELEVATED" if flip_dist < thresh else "LOW"),
        }

    # OI momentum exhaustion (needs 5+ history points)
    if history and len(history) >= 5:
        moms = [safe_num(h.get("momentum", 0)) for h in history[-6:]]
        moms.append(safe_num(m.get("momentum", 0)))
        if len(moms) >= 5:
            sign = 1 if moms[-1] > 0 else (-1 if moms[-1] < 0 else 0)
            if sign != 0:
                signed = [v * sign for v in moms]
                if all(v > 0 for v in signed):
                    mags = [abs(v) for v in moms]
                    recent  = float(np.mean(mags[-2:]))
                    earlier = float(np.mean(mags[:3])) if len(mags) >= 3 else recent
                    ratio   = recent / earlier if earlier > 0.5 else 1.0
                    out["oi_exhaustion"] = {
                        "direction":     "BULL" if sign > 0 else "BEAR",
                        "exhaust_ratio": round(ratio, 2),
                        "exhausting":    ratio < 0.50,
                        "label":         ("EXHAUSTING" if ratio < 0.50 else
                                          "FADING"     if ratio < 0.75 else "STRONG"),
                        "color":         ("#DC2626" if ratio < 0.50 else
                                          "#F59E0B" if ratio < 0.75 else "#059669"),
                    }
    return out


def compute_shantanu_view(df_band, m, spot, symbol="BTC"):
    """
    Shantanu's ND/NDM Decision Matrix — adapted from nifty v5.
    Per-strike Net Delta (ND) and NDM (Net Delta Momentum):
      ND  = (Call OI × |Call Δ|) − (Put OI × |Put Δ|)
      NDM = (Call OI Chg × |Call Δ|) − (Put OI Chg × |Put Δ|)
    Enhanced NDM corrects for buyer/writer stance using EV ratio.
    Returns dict with ND/NDM totals + Enhanced NDM breakdown.
    """
    if df_band is None or df_band.empty:
        return {"available": False}

    df = df_band.copy()
    for c in ("strike","call_oi","put_oi","call_oi_chg","put_oi_chg",
              "call_delta","put_delta","call_ltp","put_ltp","call_iv","put_iv"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["_cd_abs"] = df["call_delta"].abs()
    df["_pd_abs"] = df["put_delta"].abs()
    df["_nd"]     = (df["call_oi"]     * df["_cd_abs"]) - (df["put_oi"]     * df["_pd_abs"])
    df["_ndm"]    = (df["call_oi_chg"] * df["_cd_abs"]) - (df["put_oi_chg"] * df["_pd_abs"])

    atm      = safe_num(m.get("atm", spot))
    step     = get_strike_step(symbol)
    calls    = df[df["strike"] > atm + step]
    puts     = df[df["strike"] < atm - step]
    atm_band = df[df["strike"].between(atm - step, atm + step)]

    total_nd  = float(df["_nd"].sum())
    total_ndm = float(df["_ndm"].sum())

    # Enhanced NDM using EV ratio
    ev_ratio = safe_num(m.get("ev_ratio", 1.0))
    if ev_ratio > 1:    c_prem_dir, p_prem_dir = 1, -1     # Call buyer / Put writer
    elif ev_ratio < 1:  c_prem_dir, p_prem_dir = -1, 1     # Put buyer / Call writer
    else:               c_prem_dir, p_prem_dir = 0, 0

    enhanced_rows = []
    for _, r in df.iterrows():
        c_oi_chg = float(r.get("call_oi_chg", 0))
        p_oi_chg = float(r.get("put_oi_chg",  0))
        c_delta  = abs(float(r.get("call_delta", 0)))
        p_delta  = abs(float(r.get("put_delta", 0)))
        raw_ndm  = (c_oi_chg * c_delta) - (p_oi_chg * p_delta)
        if c_prem_dir == 0 and p_prem_dir == 0:
            enh_val = 0
        else:
            c_contrib = c_oi_chg * c_delta * c_prem_dir
            p_contrib = p_oi_chg * p_delta * (-p_prem_dir)
            enh_val   = c_contrib + p_contrib
        enhanced_rows.append({
            "Strike":     int(r["strike"]),
            "C OI Chg":   int(c_oi_chg),
            "C Prem Dir": "~ Neutral" if c_prem_dir == 0 else ("↑ Buyer" if c_prem_dir == 1 else "↓ Writer"),
            "P OI Chg":   int(p_oi_chg),
            "P Prem Dir": "~ Neutral" if p_prem_dir == 0 else ("↑ Buyer" if p_prem_dir == 1 else "↓ Writer"),
            "Enhanced NDM": round(enh_val),
            "Raw NDM":      round(raw_ndm),
        })

    enh_total = sum(r["Enhanced NDM"] for r in enhanced_rows)
    raw_total = sum(r["Raw NDM"]      for r in enhanced_rows)

    # Signal classification
    if enh_total > 0 and raw_total > 0:
        signal, sc, sbg = ("✅ CONFIRMED BULLISH — Buyer-driven call pressure. MM hedge = buy spot.",
                           "#059669", "#D1FAE5")
    elif enh_total < 0 and raw_total < 0:
        signal, sc, sbg = ("✅ CONFIRMED BEARISH — Buyer-driven put pressure. MM hedge = sell spot.",
                           "#DC2626", "#FEE2E2")
    elif enh_total > 0 and raw_total < 0:
        signal, sc, sbg = ("⚠️ DIVERGENCE — Writer puts reversing raw signal → Lean BULLISH. Verify DVOL + PCR.",
                           "#D97706", "#FFFBEB")
    elif enh_total < 0 and raw_total > 0:
        signal, sc, sbg = ("⚠️ DIVERGENCE — Writer calls reversing raw signal → Lean BEARISH. Verify DVOL + PCR.",
                           "#D97706", "#FFFBEB")
    else:
        signal, sc, sbg = ("➖ NEUTRAL / MIXED — No dominant aggressor side.",
                           "#6B7280", "#F9FAFB")

    return {
        "available":       True,
        "otm_call_nd":     float(calls["_nd"].sum())   if not calls.empty   else 0,
        "otm_put_nd":      float(puts["_nd"].sum())    if not puts.empty    else 0,
        "atm_nd":          float(atm_band["_nd"].sum()) if not atm_band.empty else 0,
        "otm_call_ndm":    float(calls["_ndm"].sum())  if not calls.empty   else 0,
        "otm_put_ndm":     float(puts["_ndm"].sum())   if not puts.empty    else 0,
        "atm_ndm":         float(atm_band["_ndm"].sum()) if not atm_band.empty else 0,
        "total_nd":        total_nd,
        "total_ndm":       total_ndm,
        "enh_total":       enh_total,
        "raw_total":       raw_total,
        "ev_ratio":        ev_ratio,
        "signal":          signal,
        "signal_color":    sc,
        "signal_bg":       sbg,
        "enhanced_rows":   enhanced_rows,
        "df":              df,
    }


def compute_enhanced_price_bias(vwap_or, ts_signal, vix_signal,
                                 s34_score, spot):
    """
    Top-of-dashboard Enhanced Price Bias — adapted from nifty v5.
    Combines S3/4 options bias with three price-action confirmation layers:
      • VWAP / Opening Range (crypto: rolling VWAP from intraday candles)
      • Term Structure (front vs back expiry IV — if available)
      • VIX surrogate = DVOL (Deribit's 30-day IV index)
    For v2 crypto: VWAP/OR may be None (no intraday source yet); ts_signal
    comes from comparing two expiries; vix_signal comes from DVOL.
    """
    layers = []
    s34_col = "#059669" if s34_score > 10 else ("#DC2626" if s34_score < -10 else "#6B7280")
    layers.append(("S3/4", s34_score, s34_col))

    price_score = 0
    if vwap_or and vwap_or.get("available"):
        price_score = safe_num(vwap_or.get("price_score", 0))
        price_col   = vwap_or.get("price_color", "#6B7280")
        layers.append(("VWAP/OR", price_score, price_col))

    ts_score = 0
    if ts_signal and ts_signal.get("available"):
        ts_score = safe_num(ts_signal.get("ts_score", 0))
        ts_col   = ts_signal.get("ts_color", "#6B7280")
        layers.append(("Term Struct", ts_score, ts_col))

    vix_score = 0
    if vix_signal and vix_signal.get("available"):
        vix_score = safe_num(vix_signal.get("vix_score", 0))
        vix_col   = vix_signal.get("vix_color", "#6B7280")
        layers.append(("VIX", vix_score, vix_col))

    if not layers:
        return None

    # Weighted average — S3/4 always weighted; new signals weighted by availability
    weights = {"S3/4": 0.5, "VWAP/OR": 0.20, "Term Struct": 0.15, "VIX": 0.15}
    num = sum(score * weights.get(name, 0) for name, score, _ in layers)
    den = sum(weights.get(name, 0) for name, _, _ in layers)
    enhanced_score = round(num / den, 1) if den > 0 else 0

    # Direction + color
    if   enhanced_score >  15: edir, ecol = "BULLISH", "#059669"
    elif enhanced_score < -15: edir, ecol = "BEARISH", "#DC2626"
    else:                      edir, ecol = "NEUTRAL", "#6B7280"

    # Agreement
    signs = [1 if s > 5 else (-1 if s < -5 else 0) for _, s, _ in layers]
    non_zero = [s for s in signs if s != 0]
    if non_zero:
        agreement = round(sum(1 for s in non_zero if s == non_zero[0]) / len(non_zero) * 100, 0)
    else:
        agreement = 0

    # Confidence
    conf = min(100, abs(enhanced_score) * 1.1 + (agreement - 50) * 0.4)

    new_sig_avail = sum(1 for n, _, _ in layers if n != "S3/4")

    return {
        "enhanced_score":   enhanced_score,
        "direction":        edir,
        "color":            ecol,
        "enhanced_conf":    conf,
        "agreement_pct":    agreement,
        "s34_score":        s34_score,
        "price_score":      price_score,
        "ts_score":         ts_score,
        "vix_score":        vix_score,
        "new_signals_available": new_sig_avail,
        "vwap_or":          vwap_or,
        "ts_signal":        ts_signal,
        "vix_signal":       vix_signal,
    }


def fetch_vix_signal(dvol_now, dvol_hist):
    """
    Crypto VIX surrogate using Deribit DVOL.
    Returns dict with vix, change, label, color, score, available.
    """
    if not dvol_now or dvol_now <= 0:
        return {"available": False}
    # Determine if high/low relative to history
    if dvol_hist and len(dvol_hist) >= 10:
        hist_vals = [v for v in dvol_hist[-30:] if v > 0]
        if hist_vals:
            mn, mx = min(hist_vals), max(hist_vals)
            pct = (dvol_now - mn) / (mx - mn) * 100 if mx > mn else 50
        else:
            pct = 50
    else:
        pct = 50

    # Score: high DVOL = bearish VIX signal (fear), low DVOL = complacent/bullish
    # Crypto convention: DVOL > 80 → fear; DVOL < 50 → complacency
    if   dvol_now >= 90:  vlabel, vcolor, vscore = "EXTREME FEAR", "#7F1D1D", -45
    elif dvol_now >= 75:  vlabel, vcolor, vscore = "HIGH FEAR",    "#DC2626", -25
    elif dvol_now >= 60:  vlabel, vcolor, vscore = "ELEVATED",     "#D97706", -10
    elif dvol_now >= 45:  vlabel, vcolor, vscore = "NEUTRAL",      "#6B7280",   0
    elif dvol_now >= 35:  vlabel, vcolor, vscore = "COMPLACENT",   "#059669",  10
    else:                 vlabel, vcolor, vscore = "OVER-COMPLACENT","#7F1D1D", 20  # too low = surprise risk

    chg = None
    if dvol_hist and len(dvol_hist) >= 2:
        prev = dvol_hist[-2] if dvol_hist[-2] > 0 else 0
        if prev > 0:
            chg = dvol_now - prev

    return {
        "available":   True,
        "vix":         dvol_now,
        "vix_change":  chg,
        "vix_label":   vlabel,
        "vix_color":   vcolor,
        "vix_score":   vscore,
        "vix_pctile":  pct,
    }


def fetch_term_structure_signal(symbol="BTC", front_expiry=None, back_expiry=None):
    """
    Compare front-month ATM IV vs back-month ATM IV.
    Returns dict with ts_score, ts_label, ts_color, available.
    Contango (front < back) = normal/complacent; Inversion (front > back) = stress.
    """
    if not front_expiry or not back_expiry or front_expiry == back_expiry:
        return {"available": False}
    try:
        front_df, front_spot, _ = fetch_option_chain(symbol, front_expiry)
        back_df,  back_spot,  _ = fetch_option_chain(symbol, back_expiry)
        if front_df.empty or back_df.empty:
            return {"available": False}
        # ATM IV from each
        f_atm = min(front_df["strike"], key=lambda x: abs(x - front_spot))
        b_atm = min(back_df["strike"],  key=lambda x: abs(x - back_spot))
        f_iv  = float(front_df[front_df["strike"] == f_atm][["call_iv","put_iv"]].mean().mean() or 0)
        b_iv  = float(back_df[back_df["strike"]  == b_atm][["call_iv","put_iv"]].mean().mean() or 0)
        if f_iv <= 0 or b_iv <= 0:
            return {"available": False}
        diff = f_iv - b_iv   # positive = inversion
        if   diff >  3:  ts_label, ts_color, ts_score = "INVERSION (stress)",    "#DC2626", -30
        elif diff >  1:  ts_label, ts_color, ts_score = "Mild Inversion",        "#D97706", -10
        elif diff > -1:  ts_label, ts_color, ts_score = "Flat",                  "#6B7280",   0
        elif diff > -3:  ts_label, ts_color, ts_score = "Normal Contango",       "#059669",  10
        else:            ts_label, ts_color, ts_score = "Steep Contango (calm)", "#10B981",  20
        return {"available": True, "front_iv": f_iv, "back_iv": b_iv, "diff": diff,
                "ts_label": ts_label, "ts_color": ts_color, "ts_score": ts_score}
    except Exception:
        return {"available": False}


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY PERSISTENCE  (atomic JSON write — same pattern as nifty streamlit v5)
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_json_write(path, obj):
    try:
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_history():
    try:
        with open(HISTORY_FILE) as f:
            h = json.load(f)
        return h if isinstance(h, dict) else {}
    except Exception:
        return {}


def _save_history(h):
    _atomic_json_write(HISTORY_FILE, h)


# ─────────────────────────────────────────────────────────────────────────────
# OWNER SETTINGS PERSISTENCE  (refresh interval, vega band width, Z-score TF/lookback)
# Adapted from nifty_streamlit_v5_fixed.py — atomic JSON write, TTL cache.
# For crypto v2 we keep the same shape but the FILE path is project-local.
# ─────────────────────────────────────────────────────────────────────────────
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "crypto_owner_settings.json")
_persist_lock   = threading.Lock()
_owner_settings_cache = {"data": None, "ts": 0.0}
_OWNER_SETTINGS_TTL = 5.0   # seconds — coalesce multiple calls within 5s

# Owner refresh-interval presets (data refresh, NOT page refresh)
_REFRESH_OPTIONS = {
    "30 sec":  30,
    "1 min":   60,
    "2 min":   120,
    "5 min":   300,
    "15 min":  900,
    "30 min":  1800,
}

# Vega-band-width presets — governs all 4 Z-Score charts (raw & OI-wtd Vega
# Ratio, raw & OI-wtd CE/PE EV Ratio). Number of strikes either side of ATM.
_VEGA_BAND_OPTIONS = {"±3 strikes (default)": 3,
                      "±5 strikes":           5,
                      "±7 strikes":           7,
                      "±10 strikes":          10}

# Z-Score TF (time-bucket width) — governs all 4 Z-Score charts
_ZS_TF_OPTIONS = {
    "5 min":           5,
    "15 min (default)": 15,
    "30 min":          30,
    "60 min":          60,
}

# Z-Score look-back period — number of TF bars the rolling Z-score spans
_ZS_LOOKBACK_OPTIONS = {f"{n} bars": n for n in range(5, 11)}


def _load_owner_settings():
    """TTL-cached read of owner_settings.json (5s coalesce window)."""
    now = time.time()
    if (_owner_settings_cache["data"] is not None and
        now - _owner_settings_cache["ts"] < _OWNER_SETTINGS_TTL):
        return _owner_settings_cache["data"]
    with _persist_lock:
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"refresh_interval": REFRESH_SECS, "vega_band_strikes": 3,
                    "zscore_tf_minutes": 15, "zscore_lookback_buckets": 6,
                    "selected_expiry": None}
    _owner_settings_cache["data"] = data
    _owner_settings_cache["ts"]   = now
    return data


def _save_owner_settings(settings):
    with _persist_lock:
        try:
            _atomic_json_write(SETTINGS_FILE, settings)
        except Exception:
            pass
    # invalidate TTL cache so the next read picks up the new value
    _owner_settings_cache["data"] = None
    _owner_settings_cache["ts"]   = 0.0


# ── ATM-band vega + extrinsic-value helpers (used by build_history_entry + Z-Score charts) ──
def _compute_atm_band_vegas(df_band, atm, step, band_n):
    """
    Sum call_vega / put_vega across ATM ± band_n strikes.
    Returns (raw_call_vega, raw_put_vega, oiw_call_vega, oiw_put_vega).
    OI-weighted versions multiply each strike's vega by its OI before summing.
    """
    out = (0.0, 0.0, 0.0, 0.0)
    if df_band is None or df_band.empty or step <= 0:
        return out
    lo, hi = atm - band_n * step, atm + band_n * step
    vb = df_band[df_band["strike"].between(lo, hi)].copy()
    if vb.empty:
        return out
    for c in ("call_vega", "put_vega", "call_oi", "put_oi"):
        if c in vb.columns:
            vb[c] = pd.to_numeric(vb[c], errors="coerce").fillna(0.0)
    raw_c = float(vb["call_vega"].sum())
    raw_p = float(vb["put_vega"].sum())
    oiw_c = float((vb["call_oi"] * vb["call_vega"]).sum())
    oiw_p = float((vb["put_oi"]  * vb["put_vega"]).sum())
    return (raw_c, raw_p, oiw_c, oiw_p)


def _extrinsic_value(ltp, intrinsic):
    """EV = max(0, LTP − intrinsic). Returns 0 if LTP missing."""
    if ltp is None or ltp <= 0:
        return 0.0
    return max(0.0, float(ltp) - max(0.0, float(intrinsic)))


def _compute_atm_band_ev_ratios(df_band, spot, atm, step, band_n):
    """
    Strike-wise CE/PE extrinsic-value ratio across ATM ± band_n strikes.
    Returns (raw_avg, oiw_avg).
      raw_avg = mean over strikes of (call_ev / put_ev)
      oiw_avg = mean over strikes of (call_ev × call_oi) / (put_ev × put_oi)
    Skips strikes where put_ev == 0 (NaN-safe).
    """
    if df_band is None or df_band.empty or step <= 0:
        return (1.0, 1.0)
    lo, hi = atm - band_n * step, atm + band_n * step
    vb = df_band[df_band["strike"].between(lo, hi)].copy()
    if vb.empty:
        return (1.0, 1.0)
    for c in ("call_ltp", "put_ltp", "call_oi", "put_oi", "strike"):
        if c in vb.columns:
            vb[c] = pd.to_numeric(vb[c], errors="coerce").fillna(0.0)
    # Per-strike extrinsic values
    vb["call_ev"] = vb.apply(
        lambda r: _extrinsic_value(r.get("call_ltp", 0), spot - r["strike"]), axis=1)
    vb["put_ev"]  = vb.apply(
        lambda r: _extrinsic_value(r.get("put_ltp", 0),  r["strike"] - spot), axis=1)
    # Raw per-strike ratio
    raw_ratio = vb["call_ev"] / vb["put_ev"].replace(0, np.nan)
    raw_avg = float(raw_ratio.mean(skipna=True)) if raw_ratio.notna().any() else 1.0
    # OI-weighted per-strike ratio
    oiw_num = vb["call_ev"] * vb["call_oi"]
    oiw_den = vb["put_ev"]  * vb["put_oi"]
    oiw_ratio = oiw_num / oiw_den.replace(0, np.nan)
    oiw_avg = float(oiw_ratio.mean(skipna=True)) if oiw_ratio.notna().any() else 1.0
    return (raw_avg, oiw_avg)


# ── Z-Score engine helpers (used by Section 18) ──────────────────────────────
def _make_tf_buckets(ts_list, cols, freq_min=15):
    """Floor tick timestamps into freq_min-minute bars, keep the LAST value per bar.
    The most recent (in-progress) bar reflects the latest tick and updates every
    rerun; once a new bar starts, the prior one is frozen for good."""
    _idx = pd.to_datetime(ts_list, errors="coerce")
    _df = pd.DataFrame(cols, index=_idx)
    _df = _df[~_df.index.isna()]
    if _df.empty:
        return _df
    _df["bucket"] = _df.index.floor(f"{freq_min}min")
    return _df.groupby("bucket", as_index=True).last().sort_index()


def _bucket_zscore(series, window_buckets=6):
    """Z-score on an already-bucketed series using an N-bar rolling window."""
    _roll = series.rolling(window_buckets, min_periods=2)
    _mean = _roll.mean()
    _std  = _roll.std(ddof=0)
    _z = (series - _mean) / _std.replace(0, np.nan)
    return _z.fillna(0.0).round(3)


def build_history_entry(m, spot, symbol, ts):
    # ── ATM-band vega + EV ratios — captured at fetch time so the Z-Score
    # charts can read them straight from history without re-fetching. Band
    # width is the owner-configurable vega_band_strikes (default ±3). ──────
    try:
        _vb_n = int(_load_owner_settings().get("vega_band_strikes", 3))
    except Exception:
        _vb_n = 3
    _step = get_strike_step(symbol)
    _atm  = safe_num(m.get("atm", spot))
    # df_band is popped off m before build_history_entry is called for the
    # main fetch path, so we re-extract a tight band from df_signal if available.
    _df_for_vega = m.get("df_signal")
    if _df_for_vega is None or getattr(_df_for_vega, "empty", True):
        _df_for_vega = m.get("df_band")
    _raw_c, _raw_p, _oiw_c, _oiw_p = _compute_atm_band_vegas(_df_for_vega, _atm, _step, _vb_n)
    _ev_raw, _ev_oiw = _compute_atm_band_ev_ratios(_df_for_vega, spot, _atm, _step, _vb_n)

    return {
        "ts": ts, "symbol": symbol, "spot": spot,
        "atm": m.get("atm", 0), "support": m.get("support", 0),
        "resistance": m.get("resistance", 0), "max_pain": m.get("max_pain", 0),
        "wall_width": m.get("wall_width", 0), "atm_iv": m.get("atm_iv", 0),
        "iv_rank": m.get("iv_rank", 0), "gex": m.get("gex", 0),
        "gamma_flip": m.get("gamma_flip"), "pcr": m.get("pcr", 1),
        "net_delta": m.get("net_delta", 0), "momentum": m.get("momentum", 0),
        "atm_pressure": m.get("atm_pressure", 0), "gt_ratio": m.get("gt_ratio", 0),
        "skew_slope": m.get("skew_slope", 0),
        "call_oi_total": m.get("call_oi_total", 0),
        "put_oi_total": m.get("put_oi_total", 0),
        "near_oi_concentration": m.get("near_oi_concentration", 0),
        "near_oichg_concentration": m.get("near_oichg_concentration", 0),
        "ev_ratio": m.get("ev_ratio", 1), "net_vega": m.get("net_vega", 0),
        "net_theta": m.get("net_theta", 0),
        # ── v2 NEW: ATM-band vega + EV-ratio fields for Z-Score charts ──
        "atm_call_vega_raw":  round(_raw_c, 6),    # raw Σcall_vega (no OI weighting)
        "atm_put_vega_raw":   round(_raw_p, 6),    # raw Σput_vega
        "atm_call_vega":      round(_oiw_c, 6),    # OI-weighted Σcall_oi×call_vega
        "atm_put_vega":       round(_oiw_p, 6),    # OI-weighted Σput_oi×put_vega
        "ev_ratio_avg_strikewise":     round(_ev_raw, 4),  # strike-wise avg of CE/PE EV ratio
        "ev_ratio_oiw_avg_strikewise": round(_ev_oiw, 4),  # OI-weighted strike-wise avg
        "vega_band_strikes": _vb_n,                # band half-width used this tick
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS — IDENTICAL to crypto v3 (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def bias_gauge_fig(score, symbol="BTC"):
    color = GREEN if score >= 12 else (RED if score <= -12 else AMBER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": f"Bias Score — {symbol}", "font": {"color": TEXT, "size": 12}},
        number={"font": {"color": color, "size": 34}},
        gauge={"axis": {"range": [-100, 100], "tickcolor": "#444"},
               "bar": {"color": color}, "bgcolor": CARD,
               "steps": [
                   {"range": [-100, -40], "color": "#FEE2E2"},
                   {"range": [-40, -12],  "color": "#FEF2F2"},
                   {"range": [-12, 12],   "color": "#FEF9C3"},
                   {"range": [12, 40],    "color": "#F0FDF4"},
                   {"range": [40, 100],   "color": "#DCFCE7"}],
               "threshold": {"line": {"color": color, "width": 3},
                             "thickness": 0.8, "value": score}}))
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD,
                      margin=dict(l=20, r=20, t=30, b=5), height=200)
    return fig


def build_oi_chart(df, spot, symbol):
    if df.empty:
        f = go.Figure(); f.update_layout(**chart_layout(title="OI Chart — no data")); return f
    coin_color = BTC_COLOR if symbol == "BTC" else ETH_COLOR
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=df["call_oi"], name="Call OI",
                         marker_color=RED, opacity=0.75))
    fig.add_trace(go.Bar(x=df["strike"], y=df["put_oi"], name="Put OI",
                         marker_color=GREEN, opacity=0.75))
    fig.add_vline(x=spot, line_dash="dash", line_color=coin_color,
                  annotation_text=f"Spot ${spot:,.0f}", annotation_font_color=coin_color)
    fig.update_layout(**chart_layout(title=f"OI by Strike — {symbol} (±{STRUCTURAL_BAND})"),
                      barmode="group")
    return fig


def build_delta_chart(df, spot, symbol):
    if df.empty:
        f = go.Figure(); f.update_layout(**chart_layout(title="Net Delta — no data")); return f
    vals   = df["call_oi"] * df["call_delta"] + df["put_oi"] * df["put_delta"]
    colors = [GREEN if v >= 0 else RED for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=vals, name="Net δ",
                         marker_color=colors, opacity=0.8))
    fig.add_vline(x=spot, line_dash="dash", line_color=CYAN)
    fig.update_layout(**chart_layout(title=f"Net Delta by Strike — {symbol}"))
    return fig


def build_momentum_chart(df, spot, symbol):
    if df.empty:
        f = go.Figure(); f.update_layout(**chart_layout(title="OI Change — no data")); return f
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=df["call_oi_chg"], name="Call OI Δ (24h vol)",
                         marker_color=RED, opacity=0.75))
    fig.add_trace(go.Bar(x=df["strike"], y=df["put_oi_chg"], name="Put OI Δ (24h vol)",
                         marker_color=GREEN, opacity=0.75))
    fig.add_vline(x=spot, line_dash="dash", line_color=CYAN)
    fig.update_layout(**chart_layout(title=f"OI Change by Strike — {symbol}"),
                      barmode="group")
    return fig


def build_gex_chart(df, spot, symbol):
    if df.empty:
        f = go.Figure(); f.update_layout(**chart_layout(title="GEX — no data")); return f
    gex    = (df["call_oi"] * df["call_gamma"] - df["put_oi"] * df["put_gamma"]) * spot
    colors = [GREEN if v >= 0 else RED for v in gex]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=gex, name="GEX",
                         marker_color=colors, opacity=0.8))
    fig.add_vline(x=spot, line_dash="dash", line_color=CYAN)
    fig.add_hline(y=0, line_dash="dot", line_color=AMBER, opacity=0.5)
    fig.update_layout(**chart_layout(title=f"Gamma Exposure (GEX) — {symbol}"))
    return fig


def build_iv_smile_chart(df, spot, symbol):
    if df.empty:
        f = go.Figure(); f.update_layout(**chart_layout(title="IV Smile — no data")); return f
    dc, dp = df[df["call_iv"] > 0.1], df[df["put_iv"] > 0.1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dc["strike"], y=dc["call_iv"], name="Call IV",
                             mode="lines+markers", line=dict(color=RED, width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=dp["strike"], y=dp["put_iv"], name="Put IV",
                             mode="lines+markers", line=dict(color=GREEN, width=2),
                             marker=dict(size=5)))
    fig.add_vline(x=spot, line_dash="dash", line_color=CYAN,
                  annotation_text="ATM", annotation_font_color=CYAN)
    fig.update_layout(**chart_layout(title=f"IV Smile — {symbol}"), yaxis_title="IV %")
    return fig


def build_dw_flow_chart(bkt, symbol="BTC"):
    fig, labels = go.Figure(), bkt.get("labels", [])
    if not labels:
        fig.update_layout(**chart_layout(title="Δ-Weighted Flow — need 2+ buckets"))
        return fig
    fig.add_trace(go.Scatter(x=labels, y=bkt["call_flow"], name="Call Δ-Flow (ceiling)",
                             mode="lines+markers", line=dict(color=RED, width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=labels, y=bkt["put_flow"], name="Put Δ-Flow (floor)",
                             mode="lines+markers", line=dict(color=GREEN, width=2),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=labels, y=bkt["net_flow"], name="Net Sentiment (Put−Call)",
                             mode="lines", line=dict(color=ACCENT, width=2.5, dash="dot")))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)
    delta_note = "" if bkt.get("delta_active") else "  ⚠ delta proxy"
    fig.update_layout(**chart_layout(
        title=f"Δ-Weighted OI Flow — Call vs Put (15-min){delta_note}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def build_spot_reference_chart(bkt, symbol="BTC"):
    fig, labels = go.Figure(), bkt.get("labels", [])
    if not labels:
        fig.update_layout(**chart_layout(title=f"{symbol} Spot (15-min) — need data"))
        return fig
    coin_color = BTC_COLOR if symbol == "BTC" else ETH_COLOR
    fig.add_trace(go.Scatter(x=labels, y=bkt["spot"], name=f"{symbol} Spot",
                             mode="lines+markers", line=dict(color=coin_color, width=2.5),
                             marker=dict(size=5)))
    gf = next((g for g in reversed(bkt.get("gamma_flip", [])) if g is not None), None)
    if gf:
        fig.add_hline(y=gf, line_dash="solid", line_color=AMBER, line_width=2,
                      annotation_text=f"Gamma Flip ${gf:,.0f}",
                      annotation_position="top right", annotation_font_color=AMBER)
    mp = bkt["max_pain"][-1] if bkt.get("max_pain") else None
    if mp and mp > 0:
        fig.add_hline(y=mp, line_dash="dot", line_color=MUTED, line_width=1.5,
                      annotation_text=f"Max Pain ${mp:,.0f}",
                      annotation_position="bottom right", annotation_font_color=MUTED)
    sup = bkt["support"][-1] if bkt.get("support") else None
    res = bkt["resistance"][-1] if bkt.get("resistance") else None
    if sup and sup > 0:
        fig.add_hline(y=sup, line_dash="dash", line_color=GREEN, line_width=1.5,
                      annotation_text=f"Support ${sup:,.0f}",
                      annotation_position="top left", annotation_font_color=GREEN)
    if res and res > 0:
        fig.add_hline(y=res, line_dash="dash", line_color=RED, line_width=1.5,
                      annotation_text=f"Resistance ${res:,.0f}",
                      annotation_position="bottom left", annotation_font_color=RED)
    fig.update_layout(**chart_layout(title=f"{symbol} Spot (15-min) with Key Levels"))
    return fig


def build_intraday_charts(history, symbol):
    ef = go.Figure()
    ef.update_layout(**chart_layout(title="Collecting data — need 2+ ticks", height=250))
    if not history or len(history) < 2:
        return ef, ef, ef, ef

    biv, bnd, boid, bmp = {}, {}, {}, {}
    for x in history:
        b = _parse_ts_to_bucket(x.get("ts", "")) or x.get("ts", "")
        biv[b]  = safe_num(x.get("atm_iv", 0))
        bnd[b]  = safe_num(x.get("net_delta", 0))
        boid[b] = safe_num(x.get("momentum", 0))
        bmp[b]  = safe_num(x.get("max_pain", 0))

    def _cum(buckets, title, color, ylab=""):
        labs = sorted(buckets.keys())
        vals = [buckets[l] for l in labs]
        cum  = [v - vals[0] for v in vals]
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=labs, y=cum, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=5),
                                 fill="tozeroy", fillcolor=hex_rgba(color)))
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED, opacity=0.5)
        lk = chart_layout(title=title, height=250)
        if ylab:
            lk["yaxis"]["title"] = ylab
        fig.update_layout(**lk)
        return fig

    return (_cum(biv,  f"Cumulative Δ ATM IV — {symbol}",   CYAN,  "IV %"),
            _cum(bnd,  f"Cumulative Net Delta — {symbol}",   BLUE,  "Δ"),
            _cum(boid, f"Cumulative OI Momentum — {symbol}", GREEN, "Momentum"),
            _cum(bmp,  f"Max Pain Drift — {symbol}",         GOLD,  "$"))


def build_dvol_chart(closes, symbol):
    fig = go.Figure()
    if not closes:
        fig.update_layout(**chart_layout(title="DVOL — no data"))
        return fig
    fig.add_trace(go.Scatter(y=closes, mode="lines+markers",
                             line=dict(color=PINK, width=2), marker=dict(size=4),
                             name="DVOL"))
    fig.update_layout(**chart_layout(title=f"{symbol} DVOL — 48h hourly (crypto VIX)",
                                     height=250), yaxis_title="DVOL")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (no network / no UI):  python btc_options_streamlit_v1.py --selftest
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(42)
    spot, step = 62500.0, 1000
    strikes = np.arange(spot - 15 * step, spot + 16 * step, step)
    rows = []
    for K in strikes:
        d = (spot - K) / (8 * step)
        cd = float(np.clip(0.5 + d * 0.4, 0.02, 0.98))
        rows.append({
            "strike": float(K),
            "call_oi": float(rng.uniform(50, 800)), "put_oi": float(rng.uniform(50, 800)),
            "call_oi_chg": float(rng.uniform(0, 120)), "put_oi_chg": float(rng.uniform(0, 120)),
            "call_iv": float(45 + abs(K - spot) / spot * 60 + rng.uniform(-1, 1)),
            "put_iv":  float(46 + abs(K - spot) / spot * 65 + rng.uniform(-1, 1)),
            "call_delta": cd, "put_delta": cd - 1.0,
            "call_gamma": float(np.exp(-((K - spot) / (4 * step)) ** 2) * 1e-4),
            "put_gamma":  float(np.exp(-((K - spot) / (4 * step)) ** 2) * 1e-4),
            "call_theta": -float(rng.uniform(5, 40)), "put_theta": -float(rng.uniform(5, 40)),
            "call_vega": float(rng.uniform(10, 90)), "put_vega": float(rng.uniform(10, 90)),
            "call_ltp": float(max(spot - K, 0) / spot + 0.01),
            "put_ltp":  float(max(K - spot, 0) / spot + 0.011),
            "call_bid": 0.01, "put_bid": 0.01, "call_ask": 0.012, "put_ask": 0.012,
        })
    df = pd.DataFrame(rows)

    m = compute_metrics(df, spot, "BTC")
    assert m, "compute_metrics returned empty"
    for k in ("atm", "support", "resistance", "max_pain", "gex", "pcr",
              "atm_iv", "iv_rank", "ev_ratio", "skew_slope", "net_delta"):
        assert k in m, f"missing metric {k}"
    assert abs(m["atm"] - spot) <= step / 2 + 1e-9

    # max pain sanity: within strike range
    assert strikes.min() <= m["max_pain"] <= strikes.max()

    hist = []
    for i in range(8):
        e = build_history_entry(m, spot + i * 20, "BTC", f"10:{i*7:02d}:00")
        e["momentum"] = float((-1) ** i * (200 + 60 * i))
        hist.append(e)

    bias  = compute_bias(m, hist)
    assert -100 <= bias["bias_score"] <= 100 and 0 <= bias["confidence"] <= 100
    strat = strategy_recommendation(bias, m, "BTC")
    assert "name" in strat and "legs" in strat

    sent = compute_market_sentiments(hist)
    assert "overall" in sent
    vel = compute_oi_velocity(hist)
    assert vel["alert_level"] in ("NONE", "WATCH", "DANGER")

    bkt = compute_dw_flow_buckets(hist)
    assert bkt.get("labels"), "dw buckets empty"
    cb = compute_dw_composite_bias(bkt, str(date.today()))
    assert -100 <= cb["score"] <= 100

    for fig in (bias_gauge_fig(bias["bias_score"]),
                build_oi_chart(m["df_band"], spot, "BTC"),
                build_delta_chart(m["df_band"], spot, "BTC"),
                build_momentum_chart(m["df_band"], spot, "BTC"),
                build_gex_chart(m["df_band"], spot, "BTC"),
                build_iv_smile_chart(m["df_band"], spot, "BTC"),
                build_dw_flow_chart(bkt), build_spot_reference_chart(bkt),
                build_dvol_chart([50, 52, 49], "BTC")):
        assert isinstance(fig, go.Figure)
    i1, i2, i3, i4 = build_intraday_charts(hist, "BTC")
    assert all(isinstance(f, go.Figure) for f in (i1, i2, i3, i4))

    # ── v2 NEW: test the new analysis layers ─────────────────────────────────
    m["_symbol"] = "BTC"
    smile   = classify_iv_smile_scenario(m["df_band"], m, spot)
    assert "scenario_id" in smile and 0 <= smile["scenario_id"] <= 9

    grf     = _compute_grf(m, spot)
    assert 0 <= grf["total"] <= 10
    assert grf["conv"] in ("HIGH CONVICTION", "GOOD SETUP", "MODERATE", "LOW", "NO TRADE")

    ls      = compute_leading_signals(m, bias, spot, hist, smile)
    assert "div_proximity" in ls and 0 <= ls["div_proximity"] <= 100

    sv      = compute_shantanu_view(m["df_band"], m, spot, "BTC")
    assert sv.get("available") is True
    assert "enhanced_rows" in sv and len(sv["enhanced_rows"]) > 0

    vix_sig = fetch_vix_signal(72.0, [60, 65, 70, 72])
    assert vix_sig.get("available") is True
    assert vix_sig["vix_label"] in ("EXTREME FEAR", "HIGH FEAR", "ELEVATED",
                                     "NEUTRAL", "COMPLACENT", "OVER-COMPLACENT")

    cd      = generate_combined_decision(m, spot, bias, smile, hist)
    assert "quadrant" in cd and cd["quadrant"] in ("Q1", "Q2", "Q3", "Q4", "CN")
    assert "action" in cd and "explanation_lines" in cd

    eb      = compute_enhanced_price_bias(None, {"available": False}, vix_sig,
                                          bias["bias_score"], spot)
    assert eb is None or -100 <= eb["enhanced_score"] <= 100

    # ── v2 NEW: owner settings + Z-Score engine + ATM-band vega/EV helpers ──
    # Persist + read back owner settings
    _test_settings = {"refresh_interval": 120, "vega_band_strikes": 5,
                      "zscore_tf_minutes": 15, "zscore_lookback_buckets": 6,
                      "selected_expiry": None}
    _save_owner_settings(_test_settings)
    _read_back = _load_owner_settings()
    assert _read_back.get("refresh_interval") == 120
    assert _read_back.get("vega_band_strikes") == 5
    # Restore defaults
    _save_owner_settings({"refresh_interval": REFRESH_SECS, "vega_band_strikes": 3,
                          "zscore_tf_minutes": 15, "zscore_lookback_buckets": 6,
                          "selected_expiry": None})

    # ATM-band vega + EV ratio helpers
    _rc, _rp, _oc, _op = _compute_atm_band_vegas(m["df_band"], m["atm"], 1000, 3)
    assert _rc >= 0 and _rp >= 0
    _ev_r, _ev_o = _compute_atm_band_ev_ratios(m["df_band"], spot, m["atm"], 1000, 3)
    assert _ev_r > 0 and _ev_o > 0

    # Z-Score engine helpers
    _ts_list = ["2024-01-01 10:00:00", "2024-01-01 10:05:00", "2024-01-01 10:10:00",
                "2024-01-01 10:15:00", "2024-01-01 10:20:00", "2024-01-01 10:25:00"]
    _bkt = _make_tf_buckets(_ts_list, {"val": [1, 2, 3, 4, 5, 6]}, freq_min=5)
    assert len(_bkt) >= 5
    _z = _bucket_zscore(_bkt["val"], window_buckets=3)
    assert len(_z) == len(_bkt)

    # build_history_entry should now include the new ATM-band fields
    _test_entry = build_history_entry(m, spot, "BTC", "10:30:00")
    assert "atm_call_vega_raw" in _test_entry
    assert "atm_put_vega_raw" in _test_entry
    assert "atm_call_vega" in _test_entry
    assert "atm_put_vega" in _test_entry
    assert "ev_ratio_avg_strikewise" in _test_entry
    assert "ev_ratio_oiw_avg_strikewise" in _test_entry
    assert "vega_band_strikes" in _test_entry

    print("SELFTEST OK —",
          f"atm={m['atm']:,.0f} mp={m['max_pain']:,.0f} pcr={m['pcr']}",
          f"gex={m['gex']:,.0f} bias={bias['bias_score']:+.0f} {bias['direction']}",
          f"strat={strat['name']} dw={cb['score']:+.1f} {cb['direction']}",
          f"smile=Sc{smile['scenario_id']:02d} grf={grf['total']}/10",
          f"quad={cd['quadrant']} div_prox={ls['div_proximity']:.0f}",
          f"shantanu_nd={sv['total_nd']:+,.0f} enh_ndm={sv['enh_total']:+,.0f}")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI  (mirrors nifty_streamlit_v5 section structure)
# ═════════════════════════════════════════════════════════════════════════════
import streamlit as st

st.set_page_config(page_title="Shantanu's BTC Options Dashboard — Deribit",
                   page_icon="📊", layout="wide")

# Auto-refresh registered FIRST (CI #5 lesson from nifty v5: register before any
# st.stop() can fire, so the page always self-recovers from API failures).
# Page refresh is always 60s; data refresh interval is owner-controlled.
# We use 60s for the page auto-refresh (data TTL is governed by cached_chain's
# ttl parameter, which is set from owner settings).
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 1000, key="auto_refresh")
except Exception:
    st.markdown(f"<meta http-equiv='refresh' content='60'>",
                unsafe_allow_html=True)

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
  .sec {{ background:{SECTION_BG}; color:{ACCENT}; font-weight:700; font-size:14px;
          padding:8px 16px; border-radius:8px; margin:14px 0 10px 0;
          border-left:4px solid {ACCENT}; }}
  .kcard {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
            padding:10px 14px; }}
  .klab {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.5px; }}
  .kval {{ font-size:19px; font-weight:700; }}
  .ksub {{ font-size:12px; color:{MUTED}; }}
</style>""", unsafe_allow_html=True)


def sec(title):
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def kcard(col, label, value, color=TEXT, sub=""):
    col.markdown(f"<div class='kcard'><div class='klab'>{label}</div>"
                 f"<div class='kval' style='color:{color}'>{value}</div>"
                 f"<div class='ksub'>{sub}</div></div>", unsafe_allow_html=True)


# ─── Owner-mode sidebar (PIN-protected advanced settings) ────────────────────
# Adapted from nifty_streamlit_v5_fixed.py. For crypto v2 we read OWNER_PIN
# from environment variable (no Streamlit secrets dependency). If OWNER_PIN is
# not set, owner controls are exposed UNLOCKED by default (since all data
# sources are public and there are no trading credentials to protect). Set
# OWNER_PIN env var to require PIN entry.
def _get_owner_pin():
    """Returns the OWNER_PIN env var or None if not set."""
    return os.environ.get("OWNER_PIN") or None


def _render_owner_sidebar(expiry_list):
    """
    Renders the sidebar with two modes:
      - Locked (OWNER_PIN env var is set): PIN entry form only.
      - Unlocked (OWNER_PIN env var is NOT set): all controls exposed openly.

    Owner-mode controls:
      • Expiry selector (persisted)
      • Data refresh interval (persisted — governs cached_chain TTL)
      • ATM Vega band width (persisted — governs all 4 Z-Score charts)
      • Z-Score TF (time bucket) — persisted, governs all 4 Z-Score charts
      • Z-Score look-back period — persisted, governs all 4 Z-Score charts
      • Manual "Refresh Now" button

    Returns: (sel_expiry, manual_refresh_clicked, effective_refresh_seconds)
    """
    import hmac
    correct_pin = _get_owner_pin()

    if "owner_unlocked" not in st.session_state:
        # Default to unlocked if no PIN is configured
        st.session_state.owner_unlocked = (correct_pin is None)

    if "owner_pin_fail_count" not in st.session_state:
        st.session_state.owner_pin_fail_count = 0
    if "owner_pin_lock_until" not in st.session_state:
        st.session_state.owner_pin_lock_until = 0.0

    # Smart default for refresh (60s during typical active hours, 1800s otherwise)
    if "refresh_seconds" not in st.session_state:
        st.session_state.refresh_seconds = REFRESH_SECS

    sel_expiry             = None
    manual_refresh_clicked = False
    effective_refresh      = REFRESH_SECS

    with st.sidebar:
        st.markdown("### ⚙️ Owner Controls")

        # ── Locked path: PIN entry only ──────────────────────────────────────
        if correct_pin is not None and not st.session_state.owner_unlocked:
            _now_ts = time.time()
            if _now_ts < st.session_state.owner_pin_lock_until:
                _mins_left = int((st.session_state.owner_pin_lock_until - _now_ts) / 60) + 1
                st.warning(f"⏱️ Too many failed attempts. Try again in {_mins_left} min.")
            else:
                st.caption("Enter PIN to access advanced settings.")
                pin = st.text_input("Owner PIN", type="password",
                                    key="owner_pin_input", placeholder="Enter owner PIN")
                if st.button("Unlock", key="owner_unlock_btn", type="primary"):
                    if pin and hmac.compare_digest(pin, correct_pin):
                        st.session_state.owner_unlocked = True
                        st.session_state.owner_pin_fail_count = 0
                        st.rerun()
                    else:
                        st.session_state.owner_pin_fail_count += 1
                        if st.session_state.owner_pin_fail_count >= 5:
                            st.session_state.owner_pin_lock_until = _now_ts + 300
                            st.session_state.owner_pin_fail_count = 0
                            st.error("❌ Too many failed attempts. Locked for 5 minutes.")
                        else:
                            st.error(f"❌ Incorrect PIN (attempt "
                                     f"{st.session_state.owner_pin_fail_count}/5)")
            st.divider()
            st.caption("🔒 Dashboard is in **read-only** mode for guests.")
            return sel_expiry, manual_refresh_clicked, effective_refresh

        # ── Unlocked path: full controls ─────────────────────────────────────
        if correct_pin is not None:
            st.success("🔓 Owner mode")

        _cur = _load_owner_settings()

        # Expiry selector (persisted)
        if expiry_list:
            _saved_exp = _cur.get("selected_expiry") or expiry_list[0]
            _exp_idx = expiry_list.index(_saved_exp) if _saved_exp in expiry_list else 0
            sel_expiry = st.selectbox("📅 Expiry", expiry_list,
                                       index=_exp_idx, key="owner_expiry")
            if sel_expiry != _cur.get("selected_expiry"):
                _cur["selected_expiry"] = sel_expiry
                _save_owner_settings(_cur)
                st.cache_data.clear()
        else:
            st.caption("Expiry: Auto (nearest)")

        st.divider()

        # Refresh interval (persisted)
        _saved_int = _cur.get("refresh_interval", REFRESH_SECS)
        st.session_state.refresh_seconds = _saved_int
        _cur_label = next((k for k, v in _REFRESH_OPTIONS.items() if v == _saved_int),
                          list(_REFRESH_OPTIONS.keys())[1])
        _chosen = st.selectbox(
            "🔄 Data refresh interval",
            list(_REFRESH_OPTIONS.keys()),
            index=list(_REFRESH_OPTIONS.keys()).index(_cur_label),
            key="refresh_selector",
        )
        _new_int = _REFRESH_OPTIONS[_chosen]
        if _new_int != _saved_int:
            _cur["refresh_interval"] = _new_int
            _save_owner_settings(_cur)
            st.session_state.refresh_seconds = _new_int
            st.cache_data.clear()  # force re-fetch with new TTL
        effective_refresh = _new_int
        _mins = _new_int // 60
        _secs = _new_int % 60
        st.caption(f"Data refresh: **{_new_int}s** ({_mins}m {_secs}s) · "
                   f"Page refresh: 60s (always)")

        st.divider()

        # ATM Vega band width — governs all 4 Z-Score charts
        _saved_vb = _cur.get("vega_band_strikes", 3)
        _vb_label = next((k for k, v in _VEGA_BAND_OPTIONS.items() if v == _saved_vb),
                         "±3 strikes (default)")
        _chosen_vb = st.selectbox(
            "📐 ATM Vega band width",
            list(_VEGA_BAND_OPTIONS.keys()),
            index=list(_VEGA_BAND_OPTIONS.keys()).index(_vb_label),
            key="vega_band_selector",
            help="Number of strikes either side of ATM. Governs all 4 Z-Score "
                 "charts (Raw & OI-Wtd Vega Ratio, Raw & OI-Wtd CE/PE EV Ratio). "
                 "Takes effect on the next data fetch (forced immediately on change).",
        )
        _new_vb = _VEGA_BAND_OPTIONS[_chosen_vb]
        if _new_vb != _saved_vb:
            _cur["vega_band_strikes"] = _new_vb
            _save_owner_settings(_cur)
            st.cache_data.clear()  # recompute history entry with new band

        st.divider()

        # Z-Score TF — governs all 4 Z-Score charts
        _saved_zs_tf = _cur.get("zscore_tf_minutes", 15)
        _zs_tf_label = next((k for k, v in _ZS_TF_OPTIONS.items() if v == _saved_zs_tf),
                            "15 min (default)")
        _chosen_zs_tf = st.selectbox(
            "📊 Z-Score chart TF (time bucket)",
            list(_ZS_TF_OPTIONS.keys()),
            index=list(_ZS_TF_OPTIONS.keys()).index(_zs_tf_label),
            key="zscore_tf_selector",
            help="Applies to all 4 Z-Score charts. Ticks are resampled into bars "
                 "of this width (last value per bar); the in-progress bar updates "
                 "every refresh and locks once its window closes.",
        )
        _new_zs_tf = _ZS_TF_OPTIONS[_chosen_zs_tf]
        if _new_zs_tf != _saved_zs_tf:
            _cur["zscore_tf_minutes"] = _new_zs_tf
            _save_owner_settings(_cur)

        # Z-Score look-back period
        _saved_zs_lb = _cur.get("zscore_lookback_buckets", 6)
        _zs_lb_label = next((k for k, v in _ZS_LOOKBACK_OPTIONS.items() if v == _saved_zs_lb),
                            "6 bars")
        _chosen_zs_lb = st.selectbox(
            "📊 Z-Score look-back period",
            list(_ZS_LOOKBACK_OPTIONS.keys()),
            index=list(_ZS_LOOKBACK_OPTIONS.keys()).index(_zs_lb_label),
            key="zscore_lookback_selector",
            help="Number of TF bars the rolling Z-score window spans (5-10), "
                 "e.g. 6 bars × 15-min TF = 90-min lookback.",
        )
        _new_zs_lb = _ZS_LOOKBACK_OPTIONS[_chosen_zs_lb]
        if _new_zs_lb != _saved_zs_lb:
            _cur["zscore_lookback_buckets"] = _new_zs_lb
            _save_owner_settings(_cur)

        st.divider()

        # Manual refresh — owner only
        if st.button("⟳ Refresh Now", key="owner_refresh_btn", type="primary"):
            manual_refresh_clicked = True
            st.cache_data.clear()

        if correct_pin is not None:
            st.divider()
            if st.button("🔒 Lock", key="owner_lock_btn"):
                st.session_state.owner_unlocked = False
                if "owner_pin_input" in st.session_state:
                    del st.session_state["owner_pin_input"]
                st.rerun()

    return sel_expiry, manual_refresh_clicked, effective_refresh


# ─── Sidebar controls ─────────────────────────────────────────────────────────
# Pick expiry list based on chosen exchange — done BEFORE the owner sidebar so
# the owner expiry selector can use it.
def _fetch_expiry_list_for(symbol, exchange):
    if exchange == "OKX":
        try: return fetch_okx_expiries(symbol) or fetch_expiries(symbol)
        except Exception: return fetch_expiries(symbol)
    elif exchange == "Bybit":
        try: return fetch_bybit_expiries(symbol) or fetch_expiries(symbol)
        except Exception: return fetch_expiries(symbol)
    return fetch_expiries(symbol)


# First sidebar pass: symbol + exchange (always visible)
with st.sidebar:
    st.markdown("### 📊 Symbol & Exchange")
    symbol = st.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(DEFAULT_SYMBOL))
    exchange = st.selectbox("Options Exchange (chain source)", EXCHANGE_LIST, index=0,
                            help="Deribit = native Greeks + DVOL · OKX = larger OI book · "
                                 "Bybit = alt venue. Falls back to Deribit if the chosen venue fails.")
    if st.button("🔄 Force reload"):
        st.cache_data.clear()
        st.rerun()

# Build expiry list for the chosen symbol/exchange
_exp_list_raw = _fetch_expiry_list_for(symbol, exchange)
_exp_opts     = [str(e) for e in _exp_list_raw] or ["(auto — nearest)"]

# Second sidebar pass: owner-mode controls (PIN-protected if OWNER_PIN env var is set)
_owner_expiry, _manual_refresh, _effective_refresh = _render_owner_sidebar(_exp_opts)

# Resolve effective expiry: owner selection wins, else auto-nearest
if _owner_expiry and not _owner_expiry.startswith("("):
    expiry_sel = _owner_expiry
elif expiry_sel_default := next((e for e in _exp_opts if not e.startswith("(")), None):
    expiry_sel = expiry_sel_default
else:
    expiry_sel = _exp_opts[0]
expiry_arg = None if expiry_sel.startswith("(") else expiry_sel

# Use owner-controlled refresh interval (defaults to REFRESH_SECS)
REFRESH_SECS_EFFECTIVE = _effective_refresh or REFRESH_SECS

with st.sidebar:
    st.markdown("---")
    st.markdown("#### 🌐 Free Data Sources")
    st.caption(f"• **{exchange}** — options chain + OI + IV (free, no key)")
    st.caption("• **Binance** — spot + perp funding (cross-venue triangulation)")
    st.caption("• **CoinGecko** — aggregate spot reference (cross-venue)")
    st.caption("• **Deribit DVOL** — crypto VIX (when available)")
    st.caption(f"• Active refresh: {REFRESH_SECS_EFFECTIVE}s")


# ─── Cached fetch (one exchange pull per refresh window, shared by visitors) ──
@st.cache_data(ttl=REFRESH_SECS_EFFECTIVE, show_spinner=f"Fetching option chain…")
def cached_chain(symbol, expiry_arg, exchange):
    df, spot, exp = fetch_chain_multi(symbol, expiry_arg, exchange)
    return df, spot, (str(exp) if exp else None), datetime.now(timezone.utc).strftime("%H:%M:%S")


@st.cache_data(ttl=300)
def cached_dvol(symbol):
    return fetch_dvol(symbol)


@st.cache_data(ttl=REFRESH_SECS_EFFECTIVE)
def cached_perp(symbol):
    return fetch_perpetual(symbol)


@st.cache_data(ttl=REFRESH_SECS_EFFECTIVE)
def cached_cross_venue(symbol):
    return fetch_cross_venue_spot(symbol)


@st.cache_data(ttl=600)
def cached_term_structure(symbol, front_exp, back_exp):
    return fetch_term_structure_signal(symbol, front_exp, back_exp)


df, spot, expiry_str, tick_ts = cached_chain(symbol, expiry_arg, exchange)

if df.empty or spot == 0:
    st.error(f"{exchange} returned no data — will retry automatically on next refresh. "
             f"(Falling back to Deribit on next tick if the issue persists.)")
    st.stop()

m = compute_metrics(df, spot, symbol)
if not m:
    st.error("Not enough strikes near ATM to compute metrics.")
    st.stop()

# Tag the symbol onto m so smile classifier can pick up the right strike step
m["_symbol"] = symbol

# History (persisted across reruns + sessions; appended once per new tick)
# NOTE: build_history_entry() reads m["df_signal"]/m["df_band"] to compute the
# ATM-band vega + EV-ratio fields that feed Section 18's Z-Score charts. It
# must run BEFORE these keys are popped off m below — otherwise it always
# sees None and every Z-Score chart is stuck at zero/constant forever.
all_hist = _load_history()
hist     = all_hist.get(symbol, [])
if not hist or hist[-1].get("ts") != tick_ts:
    hist.append(build_history_entry(m, spot, symbol, tick_ts))
    hist = hist[-MAX_HISTORY:]
    all_hist[symbol] = hist
    _save_history(all_hist)

df_band   = m.pop("df_band")
df_signal = m.pop("df_signal")

bias  = compute_bias(m, hist[:-1])
strat = strategy_recommendation(bias, m, symbol)

# ── v2 NEW: pre-compute all extra layers used by the new sections ────────────
smile_state      = classify_iv_smile_scenario(df_band, m, spot)
dvol_now, dvol_hist = cached_dvol(symbol)
vix_signal       = fetch_vix_signal(dvol_now, dvol_hist)
# Term structure: compare front expiry vs next one if available
back_expiry_str  = None
if len(_exp_list_raw) >= 2:
    try:
        back_expiry_str = str(_exp_list_raw[1] if (not expiry_arg or str(_exp_list_raw[0]) == expiry_arg) else _exp_list_raw[0])
    except Exception:
        back_expiry_str = None
ts_signal        = cached_term_structure(symbol, expiry_arg, back_expiry_str) if back_expiry_str else {"available": False}
# VWAP/OR for crypto: not implemented in v2 (no intraday candle source yet) — leave as None
vwap_or          = None
enhanced_bias    = compute_enhanced_price_bias(vwap_or, ts_signal, vix_signal,
                                                safe_num(bias.get("bias_score", 0)), spot)
combined_decision= generate_combined_decision(m, spot, bias, smile_state, hist,
                                              vwap_or, ts_signal, vix_signal, enhanced_bias)
grf              = _compute_grf(m, spot)
leading_signals  = compute_leading_signals(m, bias, spot, hist, smile_state, _exp_list_raw)
shantanu_view    = compute_shantanu_view(df_band, m, spot, symbol)
cross_venue      = cached_cross_venue(symbol)

# ─── Header ───────────────────────────────────────────────────────────────────
coin_color = BTC_COLOR if symbol == "BTC" else ETH_COLOR
st.markdown(f"""
<div style="background:{ACCENT};border-radius:12px;padding:14px 22px;margin-bottom:8px;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;color:white;">
  <div style="font-size:20px;font-weight:800;">📊 Shantanu's {symbol} Options Dashboard
      <span style="font-weight:500;font-size:13px;opacity:.85;">Streamlit v2 · {exchange} chain · same engine as NIFTY v5 + new sections</span></div>
  <div style="font-size:14px;font-weight:600;">
      Spot <span style="color:{coin_color};font-size:18px;font-weight:800;">${spot:,.0f}</span>
      &nbsp;·&nbsp; Exp {expiry_str} &nbsp;·&nbsp; {utc_str()} &nbsp;·&nbsp; tick {tick_ts}</div>
</div>""", unsafe_allow_html=True)

# ── v2 NEW: cross-venue spot triangulation strip (right under header) ────────
_cv = cross_venue or {}
if _cv.get("mean", 0) > 0:
    _cv_spread = _cv.get("max_spread_pct", 0)
    _cv_col   = GREEN if _cv_spread < 0.10 else (AMBER if _cv_spread < 0.30 else RED)
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:8px;
                padding:8px 14px;margin-bottom:10px;display:flex;gap:18px;
                align-items:center;flex-wrap:wrap;font-size:12px;'>
      <span style='font-weight:800;color:{ACCENT};'>🌐 Cross-Venue Spot</span>
      <span>Deribit <strong>${_cv.get('deribit',0):,.0f}</strong></span>
      <span>Binance <strong>${_cv.get('binance',0):,.0f}</strong></span>
      <span>CoinGecko <strong>${_cv.get('coingecko',0):,.0f}</strong></span>
      <span>Mean <strong style='color:{coin_color};'>${_cv.get('mean',0):,.0f}</strong></span>
      <span style='color:{_cv_col};font-weight:700;'>Max spread {_cv_spread:.4f}%</span>
    </div>""", unsafe_allow_html=True)

# ─── Section 1 — Market Sentiments ───────────────────────────────────────────
sec("📡 Section 1 — Market Sentiments")
sent = compute_market_sentiments(hist)
if not sent:
    st.info("Warming up — need 3+ ticks of history (auto-collects every refresh).")
else:
    c1, c2, c3, c4 = st.columns(4)
    kcard(c1, "IV Impulse (Vega)", sent["vega_label"], sent["vega_color"],
          "Z-score of ATM IV trail")
    kcard(c2, "GEX Impulse", sent["theta_label"], sent["theta_color"],
          "Z-score of GEX trail")
    kcard(c3, "OI Sentiment (PCR)", sent["oi_label"], sent["oi_color"],
          "Inverted PCR Z-score")
    kcard(c4, "Overall", sent["overall"], sent["overall_color"], sent["pos_caption"])
    if sent["warming"]:
        st.caption(f"⚠ Warming up — {sent['n_ticks']} ticks collected")

# ─── Section 2 — Bias Engine · Strategy · Key Metrics ────────────────────────
sec("🧠 Section 2 — Bias Engine · Strategy · Key Metrics")
g1, g2 = st.columns([1, 2])
with g1:
    st.plotly_chart(bias_gauge_fig(bias["bias_score"], symbol),
                    use_container_width=True, key="gauge")
    st.markdown(f"**{bias['direction']}** · Confidence **{bias['confidence']:.0f}%** · "
                f"Regime **{bias['regime']}** · {bias['vol_regime']}")
with g2:
    st.markdown(f"""
<div class='kcard' style='border-left:5px solid {strat['color']};'>
  <div class='klab'>Strategy Recommendation</div>
  <div class='kval' style='color:{strat['color']}'>{strat['name']}</div>
  <div class='ksub'>{strat['legs']}</div>
  <div class='ksub'>Mode: {strat['market_mode']} · {strat['iv_context']}</div>
</div>""", unsafe_allow_html=True)
    if bias["factors"]:
        st.markdown("**Factors:** " + " · ".join(bias["factors"]))

c = st.columns(6)
kcard(c[0], "ATM", f"${m['atm']:,.0f}")
kcard(c[1], "ATM IV", f"{m['atm_iv']:.1f}%",
      CYAN, f"IV rank {m['iv_rank']:.0f}")
kcard(c[2], "PCR", f"{m['pcr']:.3f}",
      GREEN if m["pcr"] > 1 else RED)
kcard(c[3], "GEX", f"{m['gex']:,.0f}",
      GREEN if m["gex"] >= 0 else RED,
      "long-gamma" if m["gex"] >= 0 else "short-gamma")
kcard(c[4], "Net Δ", f"{m['net_delta']:,.0f}",
      GREEN if m["net_delta"] >= 0 else RED)
kcard(c[5], "OI Momentum", f"{m['momentum']:,.0f}",
      GREEN if m["momentum"] >= 0 else RED, "Δ-weighted 24h vol")

c = st.columns(6)
kcard(c[0], "Net Vega", f"{m['net_vega']:,.0f}")
kcard(c[1], "Net Theta", f"{m['net_theta']:,.0f}")
kcard(c[2], "Γ/Θ ratio", f"{m['gt_ratio']:.4f}")
kcard(c[3], "EV ratio (C/P)", f"{m['ev_ratio']:.3f}",
      GREEN if m["ev_ratio"] > 1.15 else (RED if m["ev_ratio"] < 0.87 else TEXT))
kcard(c[4], "Skew slope", f"{m['skew_slope']:+.3f}",
      RED if m["skew_slope"] > 0.3 else (GREEN if m["skew_slope"] < -0.3 else TEXT))
kcard(c[5], "ATM pressure", f"{m['atm_pressure']:,.0f}",
      GREEN if m["atm_pressure"] >= 0 else RED)

# ─── Section 3 — Key Price Levels ─────────────────────────────────────────────
sec("🎯 Section 3 — Key Price Levels")
c = st.columns(5)
kcard(c[0], "Support (max Put OI)", f"${m['support']:,.0f}", GREEN,
      f"{m['dist_to_support']:+,.0f} from spot")
kcard(c[1], "Resistance (max Call OI)", f"${m['resistance']:,.0f}", RED,
      f"{m['dist_to_resistance']:+,.0f} to spot")
kcard(c[2], "Max Pain", f"${m['max_pain']:,.0f}", GOLD,
      f"{m['max_pain']-spot:+,.0f} pull")
kcard(c[3], "Gamma Flip",
      f"${m['gamma_flip']:,.0f}" if m["gamma_flip"] else "—", AMBER,
      "⚠ near flip" if bias["near_flip"] else "")
kcard(c[4], "Wall Width", f"${m['wall_width']:,.0f}",
      TEXT, f"OI walls S↔R")

# ─── Section 9 — Δ-Weighted OI Flow · Composite Bias ─────────────────────────
sec("🧲 Section 9 — Δ-Weighted OI Flow · Sentiment · Composite Bias")
bkt = compute_dw_flow_buckets(hist)
cb  = compute_dw_composite_bias(bkt, expiry_str)
f1, f2 = st.columns(2)
with f1:
    st.plotly_chart(build_dw_flow_chart(bkt, symbol), use_container_width=True, key="dwf")
with f2:
    st.plotly_chart(build_spot_reference_chart(bkt, symbol), use_container_width=True, key="spr")

bar_col = (GREEN if cb["score"] >= 45 else "#10B981" if cb["score"] >= 15 else
           RED if cb["score"] <= -45 else "#F87171" if cb["score"] <= -15 else AMBER)
pct = int((cb["score"] + 100) / 2)
comp_labels = {"net_flow_dir": "Net Δ-Flow Direction", "flow_accel": "Flow Acceleration",
               "gex_regime": "GEX Regime", "flip_side": "vs Gamma Flip",
               "max_pain": "Max Pain Gravity"}
rows_html = "".join(
    f"<div style='display:flex;justify-content:space-between;gap:6px;margin-bottom:4px;'>"
    f"<span style='font-size:12px;font-weight:600;min-width:160px;'>{comp_labels.get(k,k)}</span>"
    f"<span style='font-size:12px;font-weight:700;color:{GREEN if p>0 else RED if p<0 else MUTED};white-space:nowrap;'>{p:+.1f} pts</span>"
    f"<span style='font-size:11px;color:{MUTED};flex:1;text-align:right;'>{l}</span></div>"
    for k, (p, l) in cb.get("components", {}).items())
badge = ("✅ Δ-weighted active" if cb.get("delta_active")
         else "⚠️ Δ proxy (raw OI)")
st.markdown(f"""
<div class='kcard'>
  <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;'>
    <span style='font-weight:800;font-size:14px;color:{ACCENT};'>Composite Δ-Flow Bias</span>
    <span style='background:{bar_col}22;color:{bar_col};border:1px solid {bar_col};
                 border-radius:6px;padding:2px 10px;font-size:13px;font-weight:800;'>
      {cb['direction']} {cb['score']:+.1f}/100</span>
    <span style='font-size:12px;color:{MUTED};font-weight:600;'>Conf {cb['confidence']}%</span>
    <span style='font-size:11px;font-weight:700;'>{badge}</span>
  </div>
  <div style='background:{BORDER};border-radius:6px;height:10px;margin-bottom:10px;position:relative;'>
    <div style='width:{pct}%;height:100%;background:{bar_col};border-radius:6px;'></div>
    <div style='position:absolute;top:-1px;left:50%;width:2px;height:12px;background:{MUTED};'></div>
  </div>
  {rows_html}
  <div style='font-size:12px;background:{SECTION_BG};padding:7px 10px;border-radius:6px;
              line-height:1.6;margin-top:8px;'>{cb['narrative']}</div>
</div>""", unsafe_allow_html=True)

# ─── Section 4 — Strike-wise Charts ──────────────────────────────────────────
sec(f"📊 Section 4 — Strike-wise Charts (Structural Band ±{STRUCTURAL_BAND})")
r1c1, r1c2 = st.columns(2)
with r1c1:
    st.plotly_chart(build_oi_chart(df_band, spot, symbol), use_container_width=True, key="oi")
with r1c2:
    st.plotly_chart(build_gex_chart(df_band, spot, symbol), use_container_width=True, key="gex")
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.plotly_chart(build_delta_chart(df_band, spot, symbol), use_container_width=True, key="dlt")
with r2c2:
    st.plotly_chart(build_momentum_chart(df_band, spot, symbol), use_container_width=True, key="mom")
st.plotly_chart(build_iv_smile_chart(df_band, spot, symbol), use_container_width=True, key="smile")

# ─── Section 5 — Intraday Cumulative Metrics ─────────────────────────────────
sec("📈 Section 5 — Intraday Cumulative Metrics (15-min buckets)")
i1, i2, i3, i4 = build_intraday_charts(hist, symbol)
a, b = st.columns(2)
with a: st.plotly_chart(i1, use_container_width=True, key="iv15")
with b: st.plotly_chart(i2, use_container_width=True, key="nd15")
a, b = st.columns(2)
with a: st.plotly_chart(i3, use_container_width=True, key="mo15")
with b: st.plotly_chart(i4, use_container_width=True, key="mp15")

# ─── Section 6 — OI Velocity ─────────────────────────────────────────────────
sec("⚡ Section 6 — OI Velocity Z-Score")
vel = compute_oi_velocity(hist)
vcol = RED if vel["alert_level"] == "DANGER" else \
       AMBER if vel["alert_level"] == "WATCH" else GREEN
c = st.columns(3)
kcard(c[0], "Call OI velocity", f"{vel['call_vel_zscore']:+.2f}σ")
kcard(c[1], "Put OI velocity", f"{vel['put_vel_zscore']:+.2f}σ")
kcard(c[2], "Alert", vel["alert_level"], vcol, vel["alert_text"])

# ─── Section 8 — DVOL + Perpetual Basis (VIX & futures-basis analogues) ──────
sec("🌡️ Section 8 — DVOL (crypto VIX) · Perpetual Basis Triangulation")
# dvol_now, dvol_hist already pre-computed above (used by Enhanced Bias layer)
perp = cached_perp(symbol)
c = st.columns(4)
kcard(c[0], f"{symbol} DVOL", f"{dvol_now:.1f}" if dvol_now else "—", PINK,
      "Deribit 30-day IV index")
if perp:
    bcol = GREEN if perp["basis"] >= 0 else RED
    kcard(c[1], "Perp basis", f"${perp['basis']:+,.2f}", bcol,
          f"{perp['basis_pct']:+.4f}% (mark − index)")
    kcard(c[2], "Funding (8h)", f"{perp['funding_8h']*100:+.4f}%",
          GREEN if perp["funding_8h"] >= 0 else RED,
          "Longs pay shorts" if perp["funding_8h"] > 0 else "Shorts pay longs")
    kcard(c[3], "Perp mark / index", f"${perp['mark']:,.0f}",
          TEXT, f"index ${perp['index']:,.0f}")
st.plotly_chart(build_dvol_chart(dvol_hist, symbol), use_container_width=True, key="dvol")

# ─── Section 7 — Raw Option Chain ────────────────────────────────────────────
sec("📋 Section 7 — Raw Option Chain · Greeks · IV (ATM Band)")
col_map = {
    "call_oi": "Call OI", "call_oi_chg": "Call OI Δ", "call_iv": "Call IV%",
    "call_delta": "Cδ", "call_gamma": "Cγ", "call_theta": "Cθ", "call_vega": "Cν",
    "call_ltp": "Call LTP", "strike": "STRIKE", "put_ltp": "Put LTP",
    "put_vega": "Pν", "put_theta": "Pθ", "put_gamma": "Pγ", "put_delta": "Pδ",
    "put_iv": "Put IV%", "put_oi_chg": "Put OI Δ", "put_oi": "Put OI"}
avail = [cn for cn in col_map if cn in df_band.columns]
dfs = df_band[avail].copy().round(4)
dfs.columns = [col_map[cn] for cn in avail]
st.dataframe(
    dfs.style
       .map(lambda v: f"color:{GREEN};font-weight:700" if isinstance(v, (int, float)) and v > 0 else
                      (f"color:{RED};font-weight:700" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=[cc for cc in ("Call OI Δ", "Put OI Δ") if cc in dfs.columns])
       .format(precision=4, thousands=","),
    use_container_width=True, height=520)

# ═════════════════════════════════════════════════════════════════════════════
# v2 NEW SECTIONS — ported & adapted from nifty_streamlit_v5_fixed.py
# All sections below this point are additions; existing Sections 1–9 above are
# unchanged in behaviour. New sections:
#   • Section 10 — ⚡ Enhanced Market Bias (4-layer composite)
#   • Section 11 — 🎯 Combined Bias Decision (Quadrant + Action + Confidence)
#   • Section 12 — 🔬 Greek Risk Framework (0–10 conviction)
#   • Section 13 — 📈 S3/4 Bias vs Spot (5-min history chart)
#   • Section 14 — ⚡ Gamma Data (Option C: GEX + Net Vega per strike)
#   • Section 15 — 🔮 Leading Signals / Early Warning
#   • Section 16 — 🎯 Shantanu's View (ND/NDM Decision Matrix + Enhanced NDM)
#   • Section 17 — 🌐 Multi-Exchange OI Comparison (when chain from OKX/Bybit)
# ═════════════════════════════════════════════════════════════════════════════

# ─── Section 10 — ⚡ Enhanced Market Bias ────────────────────────────────────
sec("⚡ Section 10 — Enhanced Market Bias (4-Layer Composite)")
if enhanced_bias:
    eb = enhanced_bias
    esc   = eb["enhanced_score"]
    ecol  = eb["color"]
    edir  = eb["direction"]
    econf = eb["enhanced_conf"]
    eagr  = eb["agreement_pct"]
    nsig  = eb["new_signals_available"]

    def _chip(label, value, color, bg=None):
        bg = bg or f"{color}18"
        return (f'<span style="background:{bg};color:{color};border:1px solid {color};'
                f'border-radius:5px;padding:2px 9px;font-size:11px;font-weight:700;'
                f'white-space:nowrap;">{label}: {value}</span>')

    bar_pct = int(abs(esc))
    score_bar = (f'<div style="background:#F3F4F6;border-radius:4px;height:8px;margin:6px 0 4px 0;">'
                 f'<div style="width:{bar_pct}%;background:{ecol};height:8px;border-radius:4px;'
                 f'transition:width 0.4s;"></div></div>')

    s34_col = "#059669" if eb["s34_score"] > 10 else ("#DC2626" if eb["s34_score"] < -10 else "#6B7280")
    chips = " ".join([
        _chip("S3/4",      f"{eb['s34_score']:+.0f}", s34_col),
        _chip("VWAP/OR",   f"{eb['price_score']:+.0f}" if vwap_or else "—",
              vwap_or.get("price_color","#6B7280") if vwap_or else "#6B7280"),
        _chip("Term Struct", ts_signal.get("ts_label","—").replace(" ","_") if ts_signal.get("available") else "—",
              ts_signal.get("ts_color","#6B7280") if ts_signal.get("available") else "#6B7280"),
        _chip("DVOL/VIX",  f"{vix_signal.get('vix',0):.1f}" if vix_signal.get("available") else "—",
              vix_signal.get("vix_color","#6B7280") if vix_signal.get("available") else "#6B7280"),
    ])

    detail_lines = []
    if vwap_or and vwap_or.get("available"):
        detail_lines.append(f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
                            f'<strong>VWAP</strong> {vwap_or.get("vwap",0):,.1f} &nbsp;·&nbsp; '
                            f'<span style="color:{vwap_or.get("price_color","#6B7280")};font-weight:700;">'
                            f'{vwap_or.get("price_label","—")}</span></div>')
    else:
        detail_lines.append('<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
                            '&#9642; VWAP/OR: not yet implemented for crypto (no intraday candle source)</div>')

    if ts_signal.get("available"):
        detail_lines.append(f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
                            f'<strong>Term Structure</strong> — front IV {ts_signal["front_iv"]:.1f}% vs '
                            f'back {ts_signal["back_iv"]:.1f}% (diff {ts_signal["diff"]:+.1f}) · '
                            f'<span style="color:{ts_signal["ts_color"]};font-weight:700;">'
                            f'{ts_signal["ts_label"]}</span></div>')
    else:
        detail_lines.append('<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
                            '&#9642; Term Structure: single expiry only (back-month data unavailable)</div>')

    if vix_signal.get("available"):
        chg_str = (f' &nbsp;·&nbsp; Δ {vix_signal["vix_change"]:+.2f} pts this tick'
                   if vix_signal.get("vix_change") is not None else "")
        detail_lines.append(f'<div style="font-size:11px;color:#374151;padding:2px 0;">&#9642; '
                            f'<strong>DVOL (crypto VIX)</strong> — {vix_signal["vix"]:.1f} · '
                            f'<span style="color:{vix_signal["vix_color"]};font-weight:700;">'
                            f'{vix_signal["vix_label"]}</span>{chg_str}</div>')
    else:
        detail_lines.append('<div style="font-size:11px;color:#9CA3AF;padding:2px 0;">'
                            '&#9642; DVOL: unavailable</div>')

    details_html = "\n".join(detail_lines)
    agr_color = "#059669" if eagr >= 75 else ("#F59E0B" if eagr >= 50 else "#DC2626")
    agr_label = "High agreement" if eagr >= 75 else ("Partial agreement" if eagr >= 50 else "Mixed signals")
    conf_bar = (f'<div style="background:#F3F4F6;border-radius:4px;height:5px;margin-top:4px;">'
                f'<div style="width:{int(econf)}%;background:{ecol};height:5px;border-radius:4px;"></div></div>')

    st.markdown(f"""
    <div style="background:#fff;border:2px solid {ecol};border-radius:12px;
                padding:14px 20px 12px 24px;margin-bottom:14px;position:relative;
                box-shadow:0 2px 8px rgba(0,0,0,0.07);">
      <div style="position:absolute;left:0;top:0;bottom:0;width:6px;
                  background:{ecol};border-radius:12px 0 0 12px;"></div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
        <span style="font-size:16px;font-weight:900;color:#1A1A2E;">⚡ Enhanced Bias</span>
        <span style="background:{ecol};color:#fff;border-radius:6px;padding:4px 14px;
                     font-size:14px;font-weight:800;letter-spacing:0.5px;">{edir}</span>
        <span style="background:{ecol}22;color:{ecol};border:1px solid {ecol};
                     border-radius:6px;padding:2px 10px;font-size:13px;font-weight:800;">
              {esc:+.0f} / 100</span>
        <span style="background:{agr_color}22;color:{agr_color};border:1px solid {agr_color};
                     border-radius:6px;padding:2px 9px;font-size:11px;font-weight:700;">
              {agr_label} ({int(eagr)}%)</span>
        <span style="margin-left:auto;font-size:10px;color:#9CA3AF;">{nsig}/3 new signals live</span>
      </div>
      {score_bar}
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 10px 0;">{chips}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-bottom:2px;">Composite confidence: {econf:.0f}%</div>
      {conf_bar}
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid #F3F4F6;">{details_html}</div>
    </div>""", unsafe_allow_html=True)
else:
    st.info("Enhanced bias layer not yet available — needs DVOL/term-structure signals.")


# ─── Section 11 — 🎯 Combined Bias Decision ──────────────────────────────────
sec("🎯 Section 11 — Combined Bias Decision (Quadrant · Action · Confidence)")
cd = combined_decision
qcolor  = cd["quadrant_color"]
qbg     = cd["badge_bg"]
qshort  = cd["quadrant_short"]
action  = cd["action"]
conf_l  = cd["confidence_label"]
conf_c  = cd["confidence_color"]
lines   = cd["explanation_lines"]
div     = cd["divergence"]

div_html = ""
if div:
    _div_strength = div.get("strength", "HARD")
    _div_icon = "⚠" if _div_strength == "HARD" else "🔮"
    _div_border = "solid" if _div_strength == "HARD" else "dashed"
    _div_opacity = "1.0" if _div_strength == "HARD" else "0.75"
    div_html = (f'<div style="background:{div["badge_bg"]};border:1.5px {_div_border} {div["color"]};'
                f'border-radius:8px;padding:8px 14px;margin-top:10px;opacity:{_div_opacity};">'
                f'<span style="font-size:12px;font-weight:800;color:{div["color"]};">'
                f'{_div_icon} {_div_strength}: {div["type"]}</span>'
                f'<div style="font-size:11.5px;color:#374151;margin-top:4px;">{div["warning"]}</div>'
                f'<div style="font-size:11px;color:#6B7280;margin-top:3px;">{div["detail"]}</div>'
                f'</div>')

lines_html = "".join(
    f'<div style="font-size:11.5px;color:#374151;padding:2px 0;line-height:1.5;">&#9656; {ln}</div>'
    for ln in lines)

_override_badge = ""
if cd.get("quadrant_overridden"):
    _override_badge = ("<span style='background:#FEF3C7;color:#B45309;"
                       "border:1px dashed #B45309;border-radius:6px;"
                       "padding:2px 9px;font-size:10px;font-weight:700;'>"
                       "OVERRIDDEN by Price Layer</span>")
_enh_html = ""
if cd.get("enhanced_score", 0) != 0:
    _enh_html = (f"<span>Enhanced Score: <strong style='color:#1A1A2E;'>"
                 f"{cd['enhanced_score']:+.0f}</strong></span>")

colour_bar = f"""<div style="position:absolute;left:0;top:0;bottom:0;width:5px;
    background:{qcolor};border-radius:10px 0 0 10px;"></div>"""

st.markdown(f"""
<div style="background:{qbg};border:1.5px solid {qcolor};border-radius:10px;
            padding:12px 18px 12px 22px;margin-bottom:12px;position:relative;">
  {colour_bar}
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px;">
    <span style="font-size:15px;font-weight:900;color:#1A1A2E;">🎯 Combined Bias Decision</span>
    <span style="background:{qcolor};color:#fff;border-radius:6px;padding:3px 12px;
                 font-size:13px;font-weight:800;letter-spacing:0.5px;">{qshort}</span>
    <span style="background:{conf_c}33;color:{conf_c};border:1px solid {conf_c};
                 border-radius:6px;padding:2px 9px;font-size:11px;font-weight:700;">
          Confidence: {conf_l}</span>
    {_override_badge}
  </div>
  <div style="font-size:12.5px;font-weight:700;color:{qcolor};margin-bottom:8px;letter-spacing:0.2px;">
      {action}</div>
  {lines_html}
  <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;padding-top:8px;
              border-top:1px solid #E5E7EB;font-size:11px;color:#6B7280;">
    <span>S3/4 Score: <strong style="color:{qcolor};">{cd['s34_score']:+.0f}</strong> ({cd['s34_direction']})</span>
    <span>IV Smile: <strong style="color:#374151;">{cd['smile_scenario']}</strong></span>
    <span>PCR: <strong style="color:#374151;">{cd['pcr']:.2f}</strong></span>
    {_enh_html}
    <span style="margin-left:auto;font-size:10px;color:#9CA3AF;">
          Combined Bias Engine (v2 inline)</span>
  </div>
  {div_html}
</div>""", unsafe_allow_html=True)


# ─── Section 12 — 🔬 Greek Risk Framework ────────────────────────────────────
sec("🔬 Section 12 — Greek Risk Framework (0–10 Conviction Score)")
_grf_dc = GREEN if grf["bias_s"] == "BULLISH" else (RED if grf["bias_s"] == "BEARISH" else AMBER)
_grf_fac_html = "".join(
    f'<div style="font-size:11px;color:#374151;padding:2px 0;line-height:1.4;">&#9656; {f}</div>'
    for f in grf["fac"]) or '<div style="font-size:11px;color:#9CA3AF;">Collecting signals…</div>'

def _gbar(v, mx, clr):
    pct = int(v / mx * 100) if mx > 0 else 0
    return (f'<div style="background:#F3F4F6;border-radius:4px;height:7px;margin-top:4px;">'
            f'<div style="width:{pct}%;background:{clr};height:7px;border-radius:4px;"></div></div>')

st.markdown(f"""
<div style="background:#fff;border:1.5px solid {grf['cc']};border-radius:10px;
            padding:14px 18px 12px 22px;margin-bottom:14px;position:relative;">
  <div style="position:absolute;left:0;top:0;bottom:0;width:5px;
              background:{grf['cc']};border-radius:10px 0 0 10px;"></div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
    <span style="background:{_grf_dc}22;color:{_grf_dc};border:1px solid {_grf_dc};
                 border-radius:6px;padding:3px 12px;font-size:13px;font-weight:800;">
          {grf['bias_s']}</span>
    <span style="background:{grf['cc']}22;color:{grf['cc']};border:1px solid {grf['cc']};
                 border-radius:6px;padding:2px 10px;font-size:12px;font-weight:700;">
          {grf['conv']} &nbsp;·&nbsp; {grf['total']}/10</span>
    <span style="font-size:11px;color:#6B7280;margin-left:auto;">
          Gamma range: <strong>{grf['gamma_range']}</strong>
          &nbsp;·&nbsp; Position size: <strong style="color:{grf['cc']};">{grf['sl']}</strong></span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:10px;">
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
          Gamma · Range quality &nbsp;<strong style="color:#1A1A2E;">{grf['g']}/3</strong></div>
      {_gbar(grf['g'], 3, '#5DCAA5')}
    </div>
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
          Delta · Equilibrium &nbsp;<strong style="color:#1A1A2E;">{grf['d']}/3</strong></div>
      {_gbar(grf['d'], 3, '#378ADD')}
    </div>
    <div>
      <div style="font-size:11px;font-weight:600;color:#6B7280;">
          Momentum · Flow &nbsp;<strong style="color:#1A1A2E;">{grf['ms']}/4</strong></div>
      {_gbar(grf['ms'], 4, '#7F77DD')}
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;
              padding-top:10px;border-top:1px solid #F3F4F6;">
    <div>
      <div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:4px;">Key signals</div>
      {_grf_fac_html}
    </div>
    <div style="background:{grf['cc']}22;border-radius:8px;padding:10px 12px;">
      <div style="font-size:11px;font-weight:700;color:{grf['cc']};margin-bottom:4px;">Recommendation</div>
      <div style="font-size:12px;color:#374151;line-height:1.55;">{grf['rtxt']}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:6px;">
          {grf['iv_env']} (IV rank {grf['iv_r']:.0f}) &nbsp;·&nbsp;
          Sources: net delta · OI momentum · GEX · PCR · max pain</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─── Section 13 — 📈 S3/4 Bias vs Spot History ───────────────────────────────
sec("📈 Section 13 — S3/4 Bias vs Spot (5-min snapshots)")
if len(hist) >= 2:
    _B_GREEN = "#22C55E"; _B_RED = "#EF4444"; _B_CYAN = "#22D3EE"
    _bh_ts    = [r.get("ts","") for r in hist]
    _bh_spot  = [safe_num(r.get("spot", 0)) for r in hist]
    _bh_score = [safe_num(r.get("bias_score", 0)) for r in hist]

    _bh_hover = []
    for r in hist:
        _hlines = [
            f"<b>{r.get('ts','')}</b>",
            f"Bias: <b>{safe_num(r.get('bias_score',0)):+.0f}</b>",
            f"Spot: ${safe_num(r.get('spot',0)):,.0f}",
        ]
        for _sk, _lbl in [("momentum","S2 Mom"),("pcr","PCR"),("atm_iv","ATM IV")]:
            if _sk in r:
                _hlines.append(f"{_lbl}: {r[_sk]}")
        _bh_hover.append("<br>".join(_hlines))

    _bf = go.Figure()
    _bf.add_hrect(y0=40,  y1=100,  fillcolor="#DCFCE7", opacity=0.25, layer="below", line_width=0)
    _bf.add_hrect(y0=-15, y1=15,   fillcolor="#E5E7EB", opacity=0.35, layer="below", line_width=0)
    _bf.add_hrect(y0=-100, y1=-40, fillcolor="#FEE2E2", opacity=0.25, layer="below", line_width=0)
    for _rl, _rd, _rc in [(0,"dot","#9CA3AF"),(40,"dash","#86EFAC"),
                          (-40,"dash","#FCA5A5"),(15,"dot","#D1D5DB"),(-15,"dot","#D1D5DB")]:
        _bf.add_hline(y=_rl, line_width=1.2, line_dash=_rd, line_color=_rc)
    _bf.add_trace(go.Scatter(
        x=_bh_ts, y=_bh_spot, name=f"{symbol} Spot",
        mode="lines+markers",
        line=dict(color=_B_CYAN, width=2.2),
        marker=dict(size=4, color=_B_CYAN),
        yaxis="y2",
        hovertemplate="%{x}<br>Spot: $%{y:,.0f}<extra>Spot</extra>",
    ))
    for _bi in range(len(_bh_score)):
        _bc = _B_GREEN if _bh_score[_bi] >= 0 else _B_RED
        if _bi < len(_bh_score) - 1:
            _bf.add_trace(go.Scatter(
                x=[_bh_ts[_bi], _bh_ts[_bi + 1]],
                y=[_bh_score[_bi], _bh_score[_bi + 1]],
                mode="lines",
                line=dict(color=_bc, width=2.8),
                showlegend=False, yaxis="y1", hoverinfo="skip",
            ))
        _bf.add_trace(go.Scatter(
            x=[_bh_ts[_bi]], y=[_bh_score[_bi]],
            mode="markers",
            marker=dict(color=_bc, size=8, line=dict(color="#fff", width=1.5), symbol="circle"),
            name="S3/4 Bias" if _bi == 0 else None,
            showlegend=(_bi == 0),
            customdata=[_bh_hover[_bi]],
            hovertemplate="%{customdata}<extra></extra>",
            yaxis="y1",
        ))

    _spot_vals  = [v for v in _bh_spot if v > 0]
    _spot_pad   = (max(_spot_vals) - min(_spot_vals)) * 0.15 if len(_spot_vals) > 1 else 50
    _spot_range = [min(_spot_vals) - _spot_pad, max(_spot_vals) + _spot_pad] if _spot_vals else None

    _s34_bc = "#22C55E" if safe_num(bias.get("bias_score",0)) > 15 else \
              ("#EF4444" if safe_num(bias.get("bias_score",0)) < -15 else "#F59E0B")
    _bf.update_layout(
        title=dict(
            text=(f"S3/4 Market Bias vs {symbol} Spot — "
                  f"<span style='color:{_s34_bc}'>{bias['direction']} {bias['bias_score']:+.0f}</span>"
                  f"  <span style='font-size:11px;color:#9CA3AF'>· snapshots every {REFRESH_SECS}s</span>"),
            font=dict(size=13, color="#1A1A2E"),
        ),
        height=270,
        paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
        margin=dict(l=52, r=66, t=44, b=30),
        font=dict(color="#1A1A2E", size=11),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        hovermode="closest",
        yaxis=dict(
            title=dict(text="Bias Score", font=dict(size=10, color="#1A1A2E")),
            range=[-105, 105], zeroline=False,
            gridcolor="#F3F4F6",
            tickvals=[-100, -60, -40, -15, 0, 15, 40, 60, 100],
            tickfont=dict(size=9),
        ),
        yaxis2=dict(
            title=dict(text=f"{symbol} Spot", font=dict(size=10, color=_B_CYAN)),
            overlaying="y", side="right",
            showgrid=False, zeroline=False,
            range=_spot_range,
            tickfont=dict(size=9, color=_B_CYAN),
        ),
    )
    st.plotly_chart(_bf, use_container_width=True, key="s34_hist")
elif len(hist) == 1:
    st.caption("⏳ Chart will appear after the second snapshot is recorded.")
else:
    st.caption("⏳ No history yet.")


# ─── Section 14 — ⚡ Gamma Data (Option C: GEX + Net Vega per strike) ────────
sec("⚡ Section 14 — Gamma Data (Option C: Standard GEX + Net Vega per Strike)")
if df_band is not None and not df_band.empty:
    _gd_src = df_band.copy()
    for _gc in ["strike", "call_oi", "put_oi", "call_gamma", "put_gamma",
                "call_vega", "put_vega"]:
        if _gc in _gd_src.columns:
            _gd_src[_gc] = pd.to_numeric(_gd_src[_gc], errors="coerce").fillna(0.0)
    _gd_src = _gd_src.sort_values("strike").reset_index(drop=True)

    _spot2 = spot ** 2
    # Standard GEX: OI × Gamma × Spot² × 0.01  (notional-scaled)
    # For crypto we don't have a lot size — Deribit OI is already in coin contracts
    _gd_src["call_gex"] = _gd_src["call_oi"] * _gd_src["call_gamma"] * _spot2 * 0.01
    _gd_src["put_gex"]  = _gd_src["put_oi"]  * _gd_src["put_gamma"]  * _spot2 * 0.01
    _gd_src["net_gex"]  = _gd_src["call_gex"] - _gd_src["put_gex"]

    _gd_src["call_vega_exp"] = _gd_src["call_oi"] * _gd_src["call_vega"]
    _gd_src["put_vega_exp"]  = _gd_src["put_oi"]  * _gd_src["put_vega"]
    _gd_src["net_vega"]      = _gd_src["call_vega_exp"] - _gd_src["put_vega_exp"]

    _chart_gamma_flip = m.get("gamma_flip")
    _gc_col1, _gc_col2 = st.columns(2)

    with _gc_col1:
        _gc1_fig = go.Figure()
        _gc1_fig.add_trace(go.Bar(
            x=_gd_src["strike"], y=_gd_src["call_gex"],
            name="Call GEX (Dealer Buy — Pinning)",
            marker_color="#EF4444", opacity=0.75,
            hovertemplate="Strike %{x:,.0f}<br>Call GEX: %{y:,.2f}<extra>Dealer Buy / Pinning</extra>",
        ))
        _gc1_fig.add_trace(go.Bar(
            x=_gd_src["strike"], y=-_gd_src["put_gex"],
            name="Put GEX (Dealer Sell — Amplifying)",
            marker_color="#22C55E", opacity=0.75,
            hovertemplate="Strike %{x:,.0f}<br>Put GEX: %{y:,.2f}<extra>Dealer Sell / Amplifying</extra>",
        ))
        _gc1_fig.add_trace(go.Scatter(
            x=_gd_src["strike"], y=_gd_src["net_gex"],
            name="Net GEX", mode="lines+markers",
            line=dict(color="#7C3AED", width=2.2),
            marker=dict(size=5, color="#7C3AED"),
            hovertemplate="Strike %{x:,.0f}<br>Net GEX: %{y:,.2f}<extra>Net GEX</extra>",
        ))
        _gc1_fig.add_vline(x=spot, line_dash="dash", line_color="#F59E0B", line_width=2,
                           annotation_text=f"Spot ${spot:,.0f}",
                           annotation_font=dict(size=10, color="#F59E0B"),
                           annotation_position="top right")
        if _chart_gamma_flip:
            _gc1_fig.add_vline(x=_chart_gamma_flip, line_dash="dot",
                               line_color="#10B981", line_width=1.8,
                               annotation_text=f"Flip ${int(_chart_gamma_flip):,}",
                               annotation_font=dict(size=9, color="#10B981"),
                               annotation_position="top left")
        _gc1_fig.update_layout(
            title=dict(
                text="Option C — Standard GEX per Strike  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Red=Call (Pinning) · Green=Put (Amplifying) · Purple=Net</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            height=310,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=55, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            yaxis=dict(
                title="GEX  (OI × Γ × Spot² × 0.01)",
                gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF", zerolinewidth=1.2,
                tickfont=dict(size=9),
            ),
            xaxis=dict(title="Strike", tickfont=dict(size=9)),
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_gc1_fig, use_container_width=True, key="gex_per_strike")

    with _gc_col2:
        _gv_fig = go.Figure()
        _gv_fig.add_trace(go.Bar(
            x=_gd_src["strike"], y=_gd_src["call_vega_exp"],
            name="Call Vega Exposure (ΣOI×Vega)",
            marker_color="#2563EB", opacity=0.70,
            hovertemplate="Strike %{x:,.0f}<br>Call Vega Exp: %{y:,.2f}<extra>Call-side Vega</extra>",
        ))
        _gv_fig.add_trace(go.Bar(
            x=_gd_src["strike"], y=-_gd_src["put_vega_exp"],
            name="Put Vega Exposure (shown negative)",
            marker_color="#F59E0B", opacity=0.70,
            hovertemplate="Strike %{x:,.0f}<br>Put Vega Exp: %{y:,.2f}<extra>Put-side Vega</extra>",
        ))
        _gv_fig.add_trace(go.Scatter(
            x=_gd_src["strike"], y=_gd_src["net_vega"],
            name="Net Vega", mode="lines+markers",
            line=dict(color="#0891B2", width=2.2),
            marker=dict(size=5, color="#0891B2"),
            hovertemplate="Strike %{x:,.0f}<br>Net Vega: %{y:,.2f}<extra>Net Vega</extra>",
        ))
        _gv_fig.add_vline(x=spot, line_dash="dash", line_color="#F59E0B", line_width=2,
                          annotation_text=f"Spot ${spot:,.0f}",
                          annotation_font=dict(size=10, color="#F59E0B"),
                          annotation_position="top right")
        _gv_fig.update_layout(
            title=dict(
                text="Net Vega per Strike  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Blue=Call · Orange=Put · Teal=Net · +ve=IV expansion zone</span>",
                font=dict(size=13),
            ),
            barmode="overlay",
            height=310,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=55, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.18, font=dict(size=10)),
            yaxis=dict(
                title="Vega Exposure  (OI × Vega)",
                gridcolor="#F3F4F6",
                zeroline=True, zerolinecolor="#9CA3AF", zerolinewidth=1.2,
                tickfont=dict(size=9),
            ),
            xaxis=dict(title="Strike", tickfont=dict(size=9)),
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_gv_fig, use_container_width=True, key="vega_per_strike")
else:
    st.info("Option chain band is empty — cannot compute Gamma Data.")


# ─── Section 15 — 🔮 Leading Signals / Early Warning ─────────────────────────
sec("🔮 Section 15 — Leading Signals / Early Warning")
_ls_col1, _ls_col2, _ls_col3 = st.columns(3)

with _ls_col1:
    _dp = leading_signals.get("div_proximity", 0)
    _dp_color = "#DC2626" if _dp >= 60 else ("#F59E0B" if _dp >= 35 else "#059669")
    _dp_label = "APPROACHING DIVERGENCE" if _dp >= 60 else ("WATCHING" if _dp >= 35 else "CLEAR")
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                padding:12px 14px;margin-bottom:10px;'>
      <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Divergence Proximity</div>
      <div style="font-size:28px;font-weight:900;color:{_dp_color};margin:6px 0;">{_dp:.0f} / 100</div>
      <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
        <div style="background:{_dp_color};height:6px;border-radius:4px;width:{_dp}%;"></div>
      </div>
      <div style="font-size:12px;font-weight:700;color:{_dp_color};">{_dp_label}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Fires at 60+ — early warning before divergences trigger</div>
    </div>""", unsafe_allow_html=True)

    _vel = leading_signals.get("velocity", 0)
    _acc = leading_signals.get("acceleration", 0)
    _vel_color = "#059669" if _vel > 2 else ("#DC2626" if _vel < -2 else "#6B7280")
    _vel_label = "ACCELERATING BULL" if _vel > 5 else ("ACCELERATING BEAR" if _vel < -5 else "STEADY")
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;'>
      <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Bias Velocity</div>
      <div style="font-size:28px;font-weight:900;color:{_vel_color};margin:6px 0;">{_vel:+.1f} pts/tick</div>
      <div style="font-size:12px;font-weight:700;color:{_vel_color};">{_vel_label}</div>
      <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Acceleration: {_acc:+.1f} pts/tick²</div>
    </div>""", unsafe_allow_html=True)

with _ls_col2:
    _gfp = leading_signals.get("gamma_flip_proximity")
    if _gfp:
        _gfp_color = "#DC2626" if _gfp["regime_risk"] == "HIGH" else ("#F59E0B" if _gfp["regime_risk"] == "ELEVATED" else "#059669")
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
          <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Gamma Flip Proximity</div>
          <div style="font-size:16px;font-weight:800;color:#1A1A2E;">Flip @ ${_gfp["flip_strike"]:,.0f}</div>
          <div style="font-size:13px;color:{_gfp_color};font-weight:700;">Spot {_gfp["side"]} by ${_gfp["distance_pts"]:,.0f} ({_gfp["pct_of_threshold"]:.0f}% of threshold)</div>
          <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
            <div style="background:{_gfp_color};height:6px;border-radius:4px;width:{min(100, _gfp["pct_of_threshold"])}%;"></div>
          </div>
          <div style="font-size:12px;font-weight:700;color:{_gfp_color};">Risk: {_gfp["regime_risk"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
                    f'<div style="font-size:11px;font-weight:700;color:{MUTED};">Gamma Flip Proximity</div>'
                    f'<div style="font-size:12px;color:#9CA3AF;margin-top:6px;">No gamma flip detected</div></div>',
                    unsafe_allow_html=True)

    _oe = leading_signals.get("oi_exhaustion")
    if _oe:
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;'>
          <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">OI Momentum Exhaustion</div>
          <div style="font-size:16px;font-weight:800;color:{_oe["color"]};">{_oe["label"]}</div>
          <div style="font-size:12px;color:#374151;">{_oe["direction"]} flow exhaust ratio: {_oe["exhaust_ratio"]:.2f}</div>
          <div style="background:#E5E7EB;border-radius:4px;height:6px;margin:4px 0;">
            <div style="background:{_oe["color"]};height:6px;border-radius:4px;width:{min(100, _oe["exhaust_ratio"] * 100)}%;"></div>
          </div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Ratio <0.50 = exhaustion (reversal risk)</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;">'
                    f'<div style="font-size:11px;font-weight:700;color:{MUTED};">OI Momentum Exhaustion</div>'
                    f'<div style="font-size:12px;color:#9CA3AF;margin-top:6px;">Need 5+ ticks of history</div></div>',
                    unsafe_allow_html=True)

with _ls_col3:
    # IV smile scenario card
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {smile_state.get("color",BORDER)};border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
      <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">IV Smile Scenario</div>
      <div style="font-size:16px;font-weight:800;color:{smile_state.get("color","#1A1A2E")};">Sc{smile_state.get("scenario_id",0):02d} — {smile_state.get("name","—")}</div>
      <div style="font-size:12px;color:#374151;">Bucket: <strong>{smile_state.get("bucket","—")}</strong></div>
      <div style="font-size:11px;color:#6B7280;margin-top:4px;">
          Put skew: {smile_state.get("put_skew",0):+.1f} · Call skew: {smile_state.get("call_skew",0):+.1f}
      </div>
    </div>""", unsafe_allow_html=True)

    # DVOL/VIX signal card
    if vix_signal.get("available"):
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {vix_signal["vix_color"]};border-radius:10px;padding:12px 14px;'>
          <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">DVOL / Crypto VIX</div>
          <div style="font-size:24px;font-weight:900;color:{vix_signal["vix_color"]};">{vix_signal["vix"]:.1f}</div>
          <div style="font-size:12px;font-weight:700;color:{vix_signal["vix_color"]};">{vix_signal["vix_label"]}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Percentile: {vix_signal["vix_pctile"]:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;">'
                    f'<div style="font-size:11px;font-weight:700;color:{MUTED};">DVOL / Crypto VIX</div>'
                    f'<div style="font-size:12px;color:#9CA3AF;margin-top:6px;">DVOL unavailable</div></div>',
                    unsafe_allow_html=True)


# ─── Section 16 — 🎯 Shantanu's View (ND/NDM Decision Matrix) ───────────────
sec("🎯 Section 16 — Shantanu's View (ND/NDM Decision Matrix + Enhanced NDM)")
if shantanu_view.get("available"):
    sv = shantanu_view
    ec1, ec2, ec3 = st.columns([1, 1, 2])
    with ec1:
        st.markdown(f"""
        <div style="background:#F8F7FF;border:1.5px solid #7C3AED;border-radius:10px;padding:14px 16px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Total ND</div>
          <div style="font-size:24px;font-weight:900;color:{'#059669' if sv['total_nd'] > 0 else '#DC2626' if sv['total_nd'] < 0 else '#6B7280'};">{sv['total_nd']:+,.0f}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Net Delta (OI-weighted)</div>
        </div>""", unsafe_allow_html=True)
    with ec2:
        st.markdown(f"""
        <div style="background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:10px;padding:14px 16px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Total NDM</div>
          <div style="font-size:24px;font-weight:900;color:{'#059669' if sv['total_ndm'] > 0 else '#DC2626' if sv['total_ndm'] < 0 else '#6B7280'};">{sv['total_ndm']:+,.0f}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Net Delta Momentum (Δ-weighted OI chg)</div>
        </div>""", unsafe_allow_html=True)
    with ec3:
        st.markdown(f"""
        <div style="background:{sv['signal_bg']};border:1.5px solid {sv['signal_color']};border-radius:10px;padding:14px 16px;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Raw NDM Signal Interpretation</div>
          <div style="font-size:13px;font-weight:800;color:{sv['signal_color']};line-height:1.5;">{sv['signal']}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">EV ratio this tick: {sv['ev_ratio']:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Sub-section: Enhanced NDM (Buyer/Writer Adjusted) ────────────────────
    st.markdown(
        '<div style="font-size:16px;font-weight:900;color:#7C3AED;letter-spacing:0.4px;'
        'padding:14px 0 6px 0;border-top:2px solid #E5E7EB;margin-top:16px;margin-bottom:8px;">'
        '🔬 Enhanced NDM — Buyer / Writer Adjusted</div>',
        unsafe_allow_html=True)
    st.caption(
        "Raw NDM assumes ALL OI addition is buyer-driven. Enhanced NDM corrects this using the "
        "raw Call/Put Extrinsic Value ratio: EV ratio > 1 → Call buyer / Put writer; "
        "EV ratio < 1 → Put buyer / Call writer. The MM takes the opposite side of the buyer, "
        "reversing the hedge direction. If Enhanced NDM diverges from Raw NDM, the raw signal is unreliable.")

    _ec1, _ec2, _ec3 = st.columns([1, 1, 2])
    with _ec1:
        st.markdown(f"""
        <div style="background:#F8F7FF;border:1.5px solid #7C3AED;border-radius:10px;padding:14px 16px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Enhanced NDM</div>
          <div style="font-size:24px;font-weight:900;color:{'#059669' if sv['enh_total'] > 0 else '#DC2626' if sv['enh_total'] < 0 else '#6B7280'};">{sv['enh_total']:+,.0f}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Buyer/Writer Adjusted</div>
        </div>""", unsafe_allow_html=True)
    with _ec2:
        _endm_rc = "#059669" if sv["raw_total"] > 0 else "#DC2626" if sv["raw_total"] < 0 else "#6B7280"
        st.markdown(f"""
        <div style="background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:10px;padding:14px 16px;text-align:center;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Raw NDM</div>
          <div style="font-size:24px;font-weight:900;color:{_endm_rc};">{sv['raw_total']:+,.0f}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:3px;">Standard Formula</div>
        </div>""", unsafe_allow_html=True)
    with _ec3:
        # Reuse the signal card (already computed)
        st.markdown(f"""
        <div style="background:{sv['signal_bg']};border:1.5px solid {sv['signal_color']};border-radius:10px;padding:14px 16px;">
          <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Signal Interpretation</div>
          <div style="font-size:13px;font-weight:800;color:{sv['signal_color']};line-height:1.5;">{sv['signal']}</div>
          <div style="font-size:10px;color:#9CA3AF;margin-top:4px;">Divergence = Raw NDM unreliable. Trust Enhanced NDM + cross-check DVOL &amp; PCR.</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 Strike-by-Strike Enhanced NDM Breakdown", expanded=False):
        st.caption(
            f"Raw Call/Put EV ratio this tick: **{sv['ev_ratio']:.3f}**. "
            "↑ Buyer = this side is buyer-dominated per the EV ratio (MM hedges WITH the move). "
            "↓ Writer = this side is writer-dominated (MM hedges AGAINST the move, flipping sign).")
        _endm_df = pd.DataFrame(sv["enhanced_rows"]).sort_values("Strike", ascending=False)
        def _endm_style(val):
            if isinstance(val, (int, float)):
                if val > 0:   return "color:#059669;font-weight:700"
                elif val < 0: return "color:#DC2626;font-weight:700"
            return ""
        st.dataframe(
            _endm_df.style.map(_endm_style, subset=["Enhanced NDM", "Raw NDM"]),
            use_container_width=True, hide_index=True)
else:
    st.info("⏳ Shantanu's View: Waiting for option chain data to initialise.")


# ─── Section 17 — 🌐 Multi-Exchange Snapshot ────────────────────────────────
sec("🌐 Section 17 — Multi-Exchange Snapshot (Cross-Venue Reference)")
me_c1, me_c2, me_c3 = st.columns(3)
with me_c1:
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;'>
      <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Chain Source</div>
      <div style="font-size:18px;font-weight:800;color:{ACCENT};">{exchange}</div>
      <div style="font-size:11px;color:#6B7280;margin-top:4px;">
          Strikes in band: {len(df_band)} · ATM ${m.get('atm', 0):,.0f}</div>
    </div>""", unsafe_allow_html=True)
with me_c2:
    binance_data = fetch_binance_spot(symbol)
    if binance_data.get("spot", 0) > 0:
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:12px 14px;'>
          <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Binance Spot + Perp</div>
          <div style="font-size:18px;font-weight:800;color:{TEXT};">${binance_data['spot']:,.0f}</div>
          <div style="font-size:11px;color:#6B7280;margin-top:4px;">
              Perp mark ${binance_data['perp_mark']:,.0f} · Funding {binance_data['funding_rate']*100:+.4f}%</div>
        </div>""", unsafe_allow_html=True)
with me_c3:
    if vix_signal.get("available"):
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {vix_signal["vix_color"]};border-radius:10px;padding:12px 14px;'>
          <div style="font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;">Deribit DVOL (crypto VIX)</div>
          <div style="font-size:18px;font-weight:800;color:{vix_signal["vix_color"]};">{vix_signal["vix"]:.1f}</div>
          <div style="font-size:11px;color:#6B7280;margin-top:4px;">{vix_signal["vix_label"]} · pctile {vix_signal["vix_pctile"]:.0f}%</div>
        </div>""", unsafe_allow_html=True)

st.caption(f"{exchange} chain · Binance + CoinGecko cross-venue · Deribit DVOL · "
           f"history {len(hist)} ticks · ⚠ Educational analytics — not financial advice.")


# ─── Section 18 — 📊 CE/PE Vega + Extrinsic-Value Z-Score Charts (4-panel) ───
# Adapted from nifty_streamlit_v5_fixed.py. Four charts, all driven by the
# owner-configurable Z-Score engine (TF + look-back) and ATM Vega band width:
#   A. Raw Vega Ratio Z-Score          — Σcall_vega / Σput_vega (no OI weighting)
#   B. OI-Weighted Vega Ratio Z-Score  — Σ(call_oi×call_vega) / Σ(put_oi×put_vega)
#   C. Raw CE/PE EV Ratio Z-Score      — strike-wise avg of (call_ev / put_ev)
#   D. OI-Wtd CE/PE EV Ratio Z-Score   — strike-wise avg of (call_ev×call_oi / put_ev×put_oi)
# Each chart is a dual-axis time-series: amber line = spot (left), coloured
# line = Z-Score (right) with ±1σ / ±2σ reference lines.
# ─────────────────────────────────────────────────────────────────────────────
sec("📊 Section 18 — CE/PE Vega & Extrinsic-Value Ratio Z-Scores (4-Panel)")

# Read owner-controlled Z-Score settings
_zs_settings        = _load_owner_settings()
_ZS_BUCKET_MIN      = int(_zs_settings.get("zscore_tf_minutes", 15))
_ZS_BUCKET_LOOKBACK = int(_zs_settings.get("zscore_lookback_buckets", 6))
_VEGA_BAND_N        = int(_zs_settings.get("vega_band_strikes", 3))
_lookback_label     = f"{_ZS_BUCKET_LOOKBACK}×{_ZS_BUCKET_MIN}m bars ({_ZS_BUCKET_LOOKBACK * _ZS_BUCKET_MIN}min lookback)"

st.caption(f"**Active Z-Score engine:** TF = {_ZS_BUCKET_MIN} min · look-back = {_ZS_BUCKET_LOOKBACK} bars · "
           f"ATM band = ±{_VEGA_BAND_N} strikes  →  {_lookback_label}. "
           "Adjust these via the owner-mode sidebar (⚙️ Owner Controls → Z-Score settings).")


def _add_atm_change_annotations(fig, times, atm_ks):
    """Draw grey dashed vlines + ATM-shift labels (avoids mean() crash on string x-axis)."""
    _prev = None
    for _ti, _ak in zip(times, atm_ks):
        if _ak and _ak != _prev and _prev is not None:
            fig.add_vline(x=_ti, line_dash="dash", line_color="#6B7280",
                          line_width=1, opacity=0.5)
            fig.add_annotation(x=_ti, y=0.95, xref="x", yref="paper",
                               text=f"ATM→{_ak:,.0f}", font=dict(size=8, color="#6B7280"),
                               showarrow=False, xanchor="left")
        _prev = _ak


def _add_sigma_lines(fig, color, yref="y2"):
    """Mean (0σ) line + ±1σ / ±2σ level markers, all on the right Z-Score axis."""
    fig.add_hline(y=0, yref=yref, line_dash="dot",
                  line_color=color, line_width=1.5, opacity=0.6)
    fig.add_annotation(x=1, y=0, xref="paper", yref=yref,
                       text="Mean (0σ)", font=dict(size=9, color=color),
                       showarrow=False, xanchor="left")
    for _zlvl, _zcol, _zdash in [(1, color, "dash"), (-1, color, "dash"),
                                  (2, "#DC2626", "dashdot"), (-2, "#DC2626", "dashdot")]:
        fig.add_hline(y=_zlvl, yref=yref, line_dash=_zdash,
                       line_color=_zcol, line_width=1, opacity=0.5)
        fig.add_annotation(x=1, y=_zlvl, xref="paper", yref=yref,
                           text=f"{_zlvl:+d}σ", font=dict(size=8, color=_zcol),
                           showarrow=False, xanchor="left")


# ── Build time-series arrays from history (only ticks with non-zero vega) ──
_vd_ts_full, _vd_spot, _vd_atm_k = [], [], []
_vd_raw_ratio, _vd_oiw_ratio     = [], []
_evz_ts_full, _evz_spot, _evz_atm_k = [], [], []
_evz_raw_ratio, _evz_oiw_ratio   = [], []

for _h in hist:
    _cv_raw = _h.get("atm_call_vega_raw")
    _pv_raw = _h.get("atm_put_vega_raw")
    _cv_oiw = _h.get("atm_call_vega")
    _pv_oiw = _h.get("atm_put_vega")
    if (_cv_raw is not None and _pv_raw is not None and float(_pv_raw) != 0 and
        _cv_oiw is not None and _pv_oiw is not None and float(_pv_oiw) != 0 and
        _h.get("spot")):
        _vd_ts_full.append(_h["ts"])
        _vd_spot.append(float(_h["spot"]))
        _vd_raw_ratio.append(round(float(_cv_raw) / float(_pv_raw), 4))
        _vd_oiw_ratio.append(round(float(_cv_oiw) / float(_pv_oiw), 4))
        _vd_atm_k.append(int(_h.get("atm", 0)))

    _evr     = _h.get("ev_ratio_avg_strikewise")
    _evr_oiw = _h.get("ev_ratio_oiw_avg_strikewise")
    if _evr is not None and _evr_oiw is not None and _h.get("spot"):
        _evz_ts_full.append(_h["ts"])
        _evz_spot.append(float(_h["spot"]))
        _evz_raw_ratio.append(float(_evr))
        _evz_oiw_ratio.append(float(_evr_oiw))
        _evz_atm_k.append(int(_h.get("atm", 0)))

# ── Bucket + Z-score for Vega Ratio ─────────────────────────────────────────
_vd_times, _vd_raw_z, _vd_oiw_z = [], [], []
if len(_vd_ts_full) >= 2:
    _vd_bkt = _make_tf_buckets(
        _vd_ts_full,
        {"spot": _vd_spot, "atm": _vd_atm_k, "raw_ratio": _vd_raw_ratio, "oiw_ratio": _vd_oiw_ratio},
        freq_min=_ZS_BUCKET_MIN,
    )
    if not _vd_bkt.empty:
        _vd_times     = _vd_bkt.index.strftime("%H:%M").tolist()
        _vd_spot      = _vd_bkt["spot"].tolist()
        _vd_atm_k     = _vd_bkt["atm"].astype(int).tolist()
        _vd_raw_ratio = _vd_bkt["raw_ratio"].tolist()
        _vd_oiw_ratio = _vd_bkt["oiw_ratio"].tolist()
        _vd_raw_z     = _bucket_zscore(_vd_bkt["raw_ratio"], _ZS_BUCKET_LOOKBACK).tolist()
        _vd_oiw_z     = _bucket_zscore(_vd_bkt["oiw_ratio"], _ZS_BUCKET_LOOKBACK).tolist()

# ── Bucket + Z-score for EV Ratio ───────────────────────────────────────────
_evz_times, _evz_raw_z, _evz_oiw_z = [], [], []
if len(_evz_ts_full) >= 2:
    _evz_bkt = _make_tf_buckets(
        _evz_ts_full,
        {"spot": _evz_spot, "atm": _evz_atm_k, "raw_ratio": _evz_raw_ratio, "oiw_ratio": _evz_oiw_ratio},
        freq_min=_ZS_BUCKET_MIN,
    )
    if not _evz_bkt.empty:
        _evz_times     = _evz_bkt.index.strftime("%H:%M").tolist()
        _evz_spot      = _evz_bkt["spot"].tolist()
        _evz_atm_k     = _evz_bkt["atm"].astype(int).tolist()
        _evz_raw_ratio = _evz_bkt["raw_ratio"].tolist()
        _evz_oiw_ratio = _evz_bkt["oiw_ratio"].tolist()
        _evz_raw_z     = _bucket_zscore(_evz_bkt["raw_ratio"], _ZS_BUCKET_LOOKBACK).tolist()
        _evz_oiw_z     = _bucket_zscore(_evz_bkt["oiw_ratio"], _ZS_BUCKET_LOOKBACK).tolist()


# ── Chart A: Raw Vega Ratio Z-Score ─────────────────────────────────────────
if len(_vd_times) >= 2:
    _vd_col1, _vd_col2 = st.columns(2)

    with _vd_col1:
        _vr_fig = go.Figure()
        _vr_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_spot,
            name=f"{symbol} Spot",
            mode="lines",
            line=dict(color="#F59E0B", width=2.5),
            yaxis="y1",
            hovertemplate="%{x}<br>Spot: <b>$%{y:,.0f}</b><extra>Spot</extra>",
        ))
        _vr_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_raw_z,
            name=f"Raw Vega Ratio Z-Score ({_lookback_label}, ±{_VEGA_BAND_N} strikes)",
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.0),
            marker=dict(size=4, color="#7C3AED"),
            yaxis="y2",
            customdata=_vd_raw_ratio,
            hovertemplate="%{x}<br>Z-Score: <b>%{y:.2f}σ</b><br>Raw Ratio: %{customdata:.4f}"
                          "<extra>Σcall_vega / Σput_vega</extra>",
        ))
        _add_sigma_lines(_vr_fig, "#7C3AED")
        _add_atm_change_annotations(_vr_fig, _vd_times, _vd_atm_k)
        _vr_fig.update_layout(
            title=dict(
                text=f"Raw Vega Ratio Z-Score — {_lookback_label}  (±{_VEGA_BAND_N} strikes)  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Amber=Spot (left) · Purple=Z-Score (right) · "
                     "&gt;+1σ/+2σ = Call vega dominant · &lt;−1σ/−2σ = Put vega dominant · "
                     "Grey dash=ATM shift</span>",
                font=dict(size=13),
            ),
            height=290,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=65, r=65, t=55, b=30),
            legend=dict(orientation="h", y=1.22, font=dict(size=10)),
            yaxis=dict(
                title=dict(text=f"{symbol} Spot", font=dict(color="#F59E0B")),
                tickfont=dict(color="#F59E0B", size=9),
                gridcolor="#F3F4F6", autorange=True, showgrid=True,
            ),
            yaxis2=dict(
                title=dict(text=f"Raw Vega Ratio Z-Score ({_lookback_label})", font=dict(color="#7C3AED")),
                tickfont=dict(color="#7C3AED", size=9),
                overlaying="y", side="right",
                zeroline=False, autorange=True, showgrid=False,
            ),
            xaxis=dict(tickfont=dict(size=9), title="Time (UTC)",
                       showgrid=True, gridcolor="#F3F4F6"),
            hovermode="x unified",
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_vr_fig, use_container_width=True, key="raw_vega_z")

    # ── Chart B: OI-Weighted Vega Ratio Z-Score ──────────────────────────────
    with _vd_col2:
        _vo_fig = go.Figure()
        _vo_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_spot,
            name=f"{symbol} Spot",
            mode="lines",
            line=dict(color="#F59E0B", width=2.5),
            yaxis="y1",
            hovertemplate="%{x}<br>Spot: <b>$%{y:,.0f}</b><extra>Spot</extra>",
        ))
        _vo_fig.add_trace(go.Scatter(
            x=_vd_times, y=_vd_oiw_z,
            name=f"OI-Wtd Vega Ratio Z-Score ({_lookback_label}, ±{_VEGA_BAND_N} strikes)",
            mode="lines+markers",
            line=dict(color="#0891B2", width=2.0),
            marker=dict(size=4, color="#0891B2"),
            yaxis="y2",
            customdata=_vd_oiw_ratio,
            hovertemplate="%{x}<br>Z-Score: <b>%{y:.2f}σ</b><br>OI-Wtd Ratio: %{customdata:.4f}"
                          "<extra>ΣOI×call_vega / ΣOI×put_vega</extra>",
        ))
        _add_sigma_lines(_vo_fig, "#0891B2")
        _add_atm_change_annotations(_vo_fig, _vd_times, _vd_atm_k)
        _vo_fig.update_layout(
            title=dict(
                text=f"OI-Weighted Vega Ratio Z-Score — {_lookback_label}  (±{_VEGA_BAND_N} strikes)  "
                     "<span style='font-size:11px;color:#6B7280'>"
                     "Amber=Spot (left) · Cyan=Z-Score (right) · "
                     "&gt;+1σ/+2σ = Call exposure dominant · &lt;−1σ/−2σ = Put / hedge demand · "
                     "Grey dash=ATM shift</span>",
                font=dict(size=13),
            ),
            height=290,
            paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
            margin=dict(l=65, r=65, t=55, b=30),
            legend=dict(orientation="h", y=1.22, font=dict(size=10)),
            yaxis=dict(
                title=dict(text=f"{symbol} Spot", font=dict(color="#F59E0B")),
                tickfont=dict(color="#F59E0B", size=9),
                gridcolor="#F3F4F6", autorange=True, showgrid=True,
            ),
            yaxis2=dict(
                title=dict(text=f"OI-Wtd Vega Ratio Z-Score ({_lookback_label})", font=dict(color="#0891B2")),
                tickfont=dict(color="#0891B2", size=9),
                overlaying="y", side="right",
                zeroline=False, autorange=True, showgrid=False,
            ),
            xaxis=dict(tickfont=dict(size=9), title="Time (UTC)",
                       showgrid=True, gridcolor="#F3F4F6"),
            hovermode="x unified",
            font=dict(color="#1A1A2E", size=11),
        )
        st.plotly_chart(_vo_fig, use_container_width=True, key="oiw_vega_z")

    # ── Chart C: Raw CE/PE EV Ratio Z-Score ──────────────────────────────────
    if len(_evz_times) >= 2:
        _evz_col1, _evz_col2 = st.columns(2)
        with _evz_col1:
            _ev_fig = go.Figure()
            _ev_fig.add_trace(go.Scatter(
                x=_evz_times, y=_evz_spot,
                name=f"{symbol} Spot",
                mode="lines",
                line=dict(color="#F59E0B", width=2.5),
                yaxis="y1",
                hovertemplate="%{x}<br>Spot: <b>$%{y:,.0f}</b><extra>Spot</extra>",
            ))
            _ev_fig.add_trace(go.Scatter(
                x=_evz_times, y=_evz_raw_z,
                name=f"Raw CE/PE EV Ratio Z-Score ({_lookback_label}, ±{_VEGA_BAND_N} strikes)",
                mode="lines+markers",
                line=dict(color="#059669", width=2.0),
                marker=dict(size=4, color="#059669"),
                yaxis="y2",
                customdata=_evz_raw_ratio,
                hovertemplate="%{x}<br>Z-Score: <b>%{y:.2f}σ</b><br>Avg CE/PE EV Ratio: %{customdata:.4f}"
                              f"<extra>strike-wise avg, ±{_VEGA_BAND_N}</extra>",
            ))
            _add_sigma_lines(_ev_fig, "#059669")
            _add_atm_change_annotations(_ev_fig, _evz_times, _evz_atm_k)
            _ev_fig.update_layout(
                title=dict(
                    text=f"Raw CE/PE EV Ratio Z-Score — {_lookback_label}  (±{_VEGA_BAND_N} strikes)  "
                         "<span style='font-size:11px;color:#6B7280'>"
                         "Amber=Spot (left) · Green=Z-Score (right) · "
                         "&gt;+1σ/+2σ = Call premium relatively rich · &lt;−1σ/−2σ = Put premium relatively rich · "
                         "Grey dash=ATM shift</span>",
                    font=dict(size=13),
                ),
                height=290,
                paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=65, r=65, t=55, b=30),
                legend=dict(orientation="h", y=1.22, font=dict(size=10)),
                yaxis=dict(
                    title=dict(text=f"{symbol} Spot", font=dict(color="#F59E0B")),
                    tickfont=dict(color="#F59E0B", size=9),
                    gridcolor="#F3F4F6", autorange=True, showgrid=True,
                ),
                yaxis2=dict(
                    title=dict(text=f"Raw CE/PE EV Ratio Z-Score ({_lookback_label})", font=dict(color="#059669")),
                    tickfont=dict(color="#059669", size=9),
                    overlaying="y", side="right",
                    zeroline=False, autorange=True, showgrid=False,
                ),
                xaxis=dict(tickfont=dict(size=9), title="Time (UTC)",
                           showgrid=True, gridcolor="#F3F4F6"),
                hovermode="x unified",
                font=dict(color="#1A1A2E", size=11),
            )
            st.plotly_chart(_ev_fig, use_container_width=True, key="raw_ev_z")

        # ── Chart D: OI-Weighted CE/PE EV Ratio Z-Score ───────────────────────
        with _evz_col2:
            _evo_fig = go.Figure()
            _evo_fig.add_trace(go.Scatter(
                x=_evz_times, y=_evz_spot,
                name=f"{symbol} Spot",
                mode="lines",
                line=dict(color="#F59E0B", width=2.5),
                yaxis="y1",
                hovertemplate="%{x}<br>Spot: <b>$%{y:,.0f}</b><extra>Spot</extra>",
            ))
            _evo_fig.add_trace(go.Scatter(
                x=_evz_times, y=_evz_oiw_z,
                name=f"OI-Wtd CE/PE EV Ratio Z-Score ({_lookback_label}, ±{_VEGA_BAND_N} strikes)",
                mode="lines+markers",
                line=dict(color="#DB2777", width=2.0),
                marker=dict(size=4, color="#DB2777"),
                yaxis="y2",
                customdata=_evz_oiw_ratio,
                hovertemplate="%{x}<br>Z-Score: <b>%{y:.2f}σ</b><br>OI-Wtd CE/PE EV Ratio: %{customdata:.4f}"
                              f"<extra>strike-wise, OI-weighted, ±{_VEGA_BAND_N}</extra>",
            ))
            _add_sigma_lines(_evo_fig, "#DB2777")
            _add_atm_change_annotations(_evo_fig, _evz_times, _evz_atm_k)
            _evo_fig.update_layout(
                title=dict(
                    text=f"OI-Weighted CE/PE EV Ratio Z-Score — {_lookback_label}  (±{_VEGA_BAND_N} strikes)  "
                         "<span style='font-size:11px;color:#6B7280'>"
                         "Amber=Spot (left) · Pink=Z-Score (right) · "
                         "&gt;+1σ/+2σ = Call premium×OI relatively rich · &lt;−1σ/−2σ = Put premium×OI relatively rich · "
                         "Grey dash=ATM shift</span>",
                    font=dict(size=13),
                ),
                height=290,
                paper_bgcolor="#fff", plot_bgcolor="#F9FAFB",
                margin=dict(l=65, r=65, t=55, b=30),
                legend=dict(orientation="h", y=1.22, font=dict(size=10)),
                yaxis=dict(
                    title=dict(text=f"{symbol} Spot", font=dict(color="#F59E0B")),
                    tickfont=dict(color="#F59E0B", size=9),
                    gridcolor="#F3F4F6", autorange=True, showgrid=True,
                ),
                yaxis2=dict(
                    title=dict(text=f"OI-Wtd CE/PE EV Ratio Z-Score ({_lookback_label})", font=dict(color="#DB2777")),
                    tickfont=dict(color="#DB2777", size=9),
                    overlaying="y", side="right",
                    zeroline=False, autorange=True, showgrid=False,
                ),
                xaxis=dict(tickfont=dict(size=9), title="Time (UTC)",
                           showgrid=True, gridcolor="#F3F4F6"),
                hovermode="x unified",
                font=dict(color="#1A1A2E", size=11),
            )
            st.plotly_chart(_evo_fig, use_container_width=True, key="oiw_ev_z")
    else:
        st.info("⏳ CE/PE EV Ratio Z-Score charts — accumulating ticks (needs ≥2 data refreshes to plot). "
                "EV ratio fields are populated as history builds up.")
else:
    st.info("⏳ All 4 Z-Score charts — accumulating ticks (needs ≥2 data refreshes to plot). "
            "Once the second tick is recorded, the charts will populate here automatically.")

st.caption(f"Owner settings · refresh {REFRESH_SECS_EFFECTIVE}s · vega band ±{_VEGA_BAND_N} strikes · "
           f"Z-Score TF {_ZS_BUCKET_MIN}m · lookback {_ZS_BUCKET_LOOKBACK} bars · "
           f"history {len(hist)} ticks · ⚠ Educational analytics — not financial advice.")
