# Shantanu's BTC Options Dashboard — Streamlit v1 (Deribit)

Streamlit port of the NIFTY v5 analytics engine for **BTC/ETH options**, wired to the
**Deribit public API** (free, no API key, ~85% of global crypto options volume —
same API as crypto_dashboard_v3).

## Run
```bash
pip install -r requirements.txt
streamlit run btc_options_streamlit_v1.py
```

Self-test (no network / no UI):
```bash
python btc_options_streamlit_v1.py --selftest
```

## Data (all free, live, no key)
| Endpoint | Used for |
|---|---|
| `public/get_instruments` | strikes + expiries |
| `public/get_book_summary_by_currency` | full chain OI / bid-ask / mark IV in ONE call |
| `public/get_index_price` | BTC/ETH spot index |
| `public/ticker` (ATM±15, parallel) | native exchange Greeks |
| `public/get_volatility_index_data` | DVOL — crypto VIX analogue |
| `public/ticker BTC-PERPETUAL` | basis + funding (futures triangulation analogue) |

## Sections (mirror NIFTY v5 / crypto v3)
1. Market Sentiments (Z-score trail) · 2. Bias Engine + Strategy + Key Metrics ·
3. Key Price Levels · 9. Δ-Weighted OI Flow + Composite Bias · 4. Strike-wise charts
(OI, GEX, Net Δ, OI change, IV smile) · 5. Intraday cumulative (15-min) ·
6. OI Velocity Z-score · 8. DVOL + Perp basis/funding · 7. Raw chain table.

## Engine (identical weights/logic)
Max Pain, PCR, GEX + gamma-flip strike, ATM IV + IV rank, skew slope, EV ratio,
Γ/Θ ratio, near-ATM OI concentration, −100…+100 bias score with the same
`BIAS_WEIGHTS`, regime classifier (PINNED/RANGE/TREND/FLIP), strategy engine
(IC / Iron Fly / verticals / straddle on flip), Δ-flow composite (35/25/20/20/10).

## Crypto-specific notes
- Market is 24/7 — no market-hours gating; timestamps in UTC.
- OI and option prices are quoted **in BTC (or ETH)**, not USD.
- Deribit has no per-tick OI-change field → 24h volume is the OI-Δ proxy
  (flagged "Δ proxy" in the UI, exactly as crypto v3 does).
- Strike step: $1,000 BTC / $50 ETH. History persists to
  `crypto_streamlit_history.json` (last 500 ticks, atomic writes).

⚠ Educational analytics — not financial advice.
