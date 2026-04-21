# Weak Coin 20-Day High Short

A standalone Streamlit backtest scanner for a counter-trend SHORT strategy on Bybit USDT perpetuals.

---

## Strategy Logic

**Core idea:** Thinly traded coins that pump to a new 20-day high on a volume spike are more likely being manipulated (MM / whale dump into retail FOMO) than breaking out genuinely. Fade the pump — short at the top, ride the reversal.

**Entry conditions (all must pass):**
1. Coin is in the **bottom X% of universe by average historical turnover** — weak/thin coins are easier to pump
2. Price makes a **fresh 20-day high** this candle
3. **Volume surge** — current volume > N× the 7-day average (confirms the pump is real, not a drift)
4. Historical 24h turnover is within the configured min/max band at that timestamp
5. *(Optional)* **EMA guard** — skip if EMA fast > EMA slow (established uptrend may be a real breakout, not a pump)

**Exit conditions (first to trigger):**
- **Hard stop** — price moves +7.5% above entry (pump continues, get out)
- **ATR trail stop** — once profitable, trail locks in gains as price drops
- **EMA cross exit** — if EMA turns bullish, uptrend confirmed, close the short

---

## Universe

Downloads **all active Bybit USDT perpetuals** (~500+ pairs) on first run. Caches candle data locally in `data/raw/`.

The turnover filter is applied **historically** using `(Volume × Close).rolling(24h).sum()` — not today's live API. This means:
- A coin that was thin in January 2026 is only eligible then, even if it's large now
- A coin that pumped its way into the band is included from that moment onwards
- No look-ahead bias in universe selection

The **weak coin set** (eligible shorts) is ranked by average historical turnover over the full backtest period — the least-traded coins in the universe.

---

## Installation

```bash
pip install -r requirements.txt
streamlit run streamlit_high_fade.py
```

Opens at `http://localhost:8501`

---

## Key Parameters

| Parameter | Default | What it does |
|---|---|---|
| **Weak coin %** | 30% | Bottom X% of universe by avg turnover — most pump-susceptible |
| **High lookback** | 20 days | How far back to check for the "new high" |
| **Volume surge** | 2× | Volume must exceed N× the 7-day average |
| **EMA guard** | OFF | Skip entry if EMA is already bullish (established trend ≠ pump) |
| **Hard stop** | 7.5% | Exit if price goes 7.5% above entry |
| **Trail activate** | 2% | ATR trail kicks in once position is 2% profitable |
| **Min hold** | 6h | Prevents whipsaw exits if EMA flips back within hours of entry |

---

## Results (Jan–Apr 2026, standalone)

Tested on Bybit universe, $1,000 starting capital, $10/trade at 10x leverage:

- **43.2% win rate** on high-fade signal entries
- Profit factor > 1 on pure signal trades
- Strategy conflicts with momentum-long strategies on the same coins — run independently

---

## Data

Candle data is downloaded from Bybit's **public API** — no API keys required. All data is stored locally in `data/raw/` (not included in this repo). First run downloads the full universe and takes ~15–25 minutes. Subsequent runs are fast (incremental updates only).

---

## Related

Main EMA breakout scanner (long + short momentum): [Trading repo](https://github.com/MacNg11247/Trading)
