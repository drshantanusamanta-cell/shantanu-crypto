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

import os, sys, json, time, tempfile
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


def build_history_entry(m, spot, symbol, ts):
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

    print("SELFTEST OK —",
          f"atm={m['atm']:,.0f} mp={m['max_pain']:,.0f} pcr={m['pcr']}",
          f"gex={m['gex']:,.0f} bias={bias['bias_score']:+.0f} {bias['direction']}",
          f"strat={strat['name']} dw={cb['score']:+.1f} {cb['direction']}")


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
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SECS * 1000, key="auto_refresh")
except Exception:
    st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_SECS}'>",
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


# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    symbol = st.selectbox("Symbol", SYMBOLS, index=SYMBOLS.index(DEFAULT_SYMBOL))
    exp_list = fetch_expiries(symbol)
    exp_opts = [str(e) for e in exp_list] or ["(auto — nearest)"]
    expiry_sel = st.selectbox("Expiry", exp_opts, index=0)
    expiry_arg = None if expiry_sel.startswith("(") else expiry_sel
    st.caption(f"Data: Deribit public API · refresh {REFRESH_SECS}s · 24/7 market")
    if st.button("🔄 Force reload"):
        st.cache_data.clear()
        st.rerun()


# ─── Cached fetch (one Deribit pull per refresh window, shared by visitors) ──
@st.cache_data(ttl=REFRESH_SECS, show_spinner="Fetching Deribit option chain…")
def cached_chain(symbol, expiry_arg):
    df, spot, exp = fetch_option_chain(symbol, expiry_arg)
    return df, spot, (str(exp) if exp else None), datetime.now(timezone.utc).strftime("%H:%M:%S")


@st.cache_data(ttl=300)
def cached_dvol(symbol):
    return fetch_dvol(symbol)


@st.cache_data(ttl=REFRESH_SECS)
def cached_perp(symbol):
    return fetch_perpetual(symbol)


df, spot, expiry_str, tick_ts = cached_chain(symbol, expiry_arg)

if df.empty or spot == 0:
    st.error("Deribit returned no data — will retry automatically on next refresh.")
    st.stop()

m = compute_metrics(df, spot, symbol)
if not m:
    st.error("Not enough strikes near ATM to compute metrics.")
    st.stop()

df_band   = m.pop("df_band")
df_signal = m.pop("df_signal")

# History (persisted across reruns + sessions; appended once per new tick)
all_hist = _load_history()
hist     = all_hist.get(symbol, [])
if not hist or hist[-1].get("ts") != tick_ts:
    hist.append(build_history_entry(m, spot, symbol, tick_ts))
    hist = hist[-MAX_HISTORY:]
    all_hist[symbol] = hist
    _save_history(all_hist)

bias  = compute_bias(m, hist[:-1])
strat = strategy_recommendation(bias, m, symbol)

# ─── Header ───────────────────────────────────────────────────────────────────
coin_color = BTC_COLOR if symbol == "BTC" else ETH_COLOR
st.markdown(f"""
<div style="background:{ACCENT};border-radius:12px;padding:14px 22px;margin-bottom:8px;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;color:white;">
  <div style="font-size:20px;font-weight:800;">📊 Shantanu's {symbol} Options Dashboard
      <span style="font-weight:500;font-size:13px;opacity:.85;">Streamlit v1 · Deribit · same engine as NIFTY v5</span></div>
  <div style="font-size:14px;font-weight:600;">
      Spot <span style="color:{coin_color};font-size:18px;font-weight:800;">${spot:,.0f}</span>
      &nbsp;·&nbsp; Exp {expiry_str} &nbsp;·&nbsp; {utc_str()} &nbsp;·&nbsp; tick {tick_ts}</div>
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
dvol_now, dvol_hist = cached_dvol(symbol)
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

st.caption(f"Deribit public API · no key required · OI in {symbol} contracts · "
           f"option prices quoted in {symbol} · history {len(hist)} ticks · "
           f"⚠ Educational analytics — not financial advice.")
