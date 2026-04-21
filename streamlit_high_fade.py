"""
High-Fade Scanner — Standalone Streamlit App
Run: streamlit run scripts/streamlit_high_fade.py

Strategy: SHORT weak coins (bottom X% by turnover) that make a fresh 20-day high
on a volume spike, but ONLY when EMA 10 is NOT above EMA 20.

Logic:
  - Weak coins breaking to new 20-day highs on volume = MM/whale pump into FOMO
  - EMA guard: if EMA is already bullish the move may be real — skip it
  - Exit: hard stop, ATR trail, or EMA crosses BUY (uptrend confirmed = get out)
"""

import time, warnings, json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent.parent
RAW_DIR = BASE / "data" / "raw"
RES_DIR = BASE / "data" / "results"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_MS_MAP = {"60": 3_600_000, "30": 1_800_000, "240": 14_400_000}

st.set_page_config(page_title="High-Fade Scanner", page_icon="📉", layout="wide")
st.title("📉 High-Fade Scanner — Weak Coin 20-Day High Short")
st.caption(
    "SHORT weak coins making fresh 20-day highs on volume. "
    "EMA guard blocks entries when the uptrend is already established. "
    "Separate from the EMA breakout scanner."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ High-Fade Config")

    st.subheader("Time Window")
    from datetime import date as _date
    _today     = _date.today()
    _default_s = _date(2026, 1, 1)
    date_start = st.date_input("From", value=_default_s, max_value=_today)
    date_end   = st.date_input("To",   value=_today,     max_value=_today)
    if date_start >= date_end:
        st.error("Start must be before End")

    st.subheader("Universe")
    st.caption(
        "Downloads ALL active Bybit USDT perps. "
        "Turnover filter applied using **historical** Volume × Close — "
        "not today's live API. Universe dynamically changes as coins enter/leave the band."
    )
    min_turn_m = st.slider("Min turnover (M$/day)", 0.1, 10.0, 0.5, step=0.1,
                           help="Applied historically per timestamp from candle data.")
    max_turn_m = st.slider("Max turnover (M$/day)", 10.0, 500.0, 500.0, step=10.0,
                           help="Applied historically per timestamp from candle data.")

    st.subheader("Weak Coin Filter")
    st.caption(
        "Weak coins = bottom X% of universe by **average** historical turnover over the period. "
        "Only these are eligible for high-fade short entries."
    )
    weak_coin_pct = st.slider("Bottom % of universe (by turnover)", 10, 60, 30,
                              help="30 = bottom 30% by avg historical turnover over the backtest period.")

    st.subheader("Entry Signal")
    high_lookback  = st.slider("New high lookback (days)", 10, 40, 20,
                                help="Price must be at its highest point in this many days.")
    vol_surge_mult = st.slider("Volume surge multiplier", 1.0, 5.0, 2.0, step=0.5,
                                help="Current volume must exceed X × 7-day average.")
    use_ema_guard  = st.checkbox("Enable EMA guard", value=False,
                                 help="When ON: skip entry if EMA fast > EMA slow (established uptrend). "
                                      "Tested Jan–Apr 2026 — too restrictive in bull markets (only 2 trades fired). "
                                      "Turn ON for bear/sideways periods.")
    ema_fast       = st.slider("EMA Fast (guard)", 5, 20, 10, disabled=not use_ema_guard)
    ema_slow       = st.slider("EMA Slow (guard)", 10, 40, 20, disabled=not use_ema_guard)
    if use_ema_guard:
        st.caption(f"Guard ON: skip entry if EMA {ema_fast} > EMA {ema_slow}.")
    else:
        st.caption("Guard OFF: entries fire on any weak coin making a new 20d high + volume.")
    weekend_filter = st.checkbox("Skip weekend entries", value=True)

    st.subheader("Risk Management")
    stop_loss_pct      = st.slider("Hard stop %", 3.0, 15.0, 7.5, step=0.5)
    trail_activate_pct = st.slider("Trail activate %", 1.0, 5.0, 2.0, step=0.5)
    trail_atr_mult     = st.slider("Trail ATR multiplier", 1.5, 6.0, 3.5, step=0.5)
    min_hold_hours     = st.slider("Min hold before signal exit (h)", 0, 24, 6,
                                   help="Prevents whipsaw exits within the first N hours.")
    consec_sl_limit    = st.slider("Consecutive stops before pause", 2, 8, 4)

    st.subheader("Position Sizing")
    max_positions    = st.slider("Max open positions", 3, 20, 10)
    trade_usdt       = st.slider("Trade size (USDT)", 5.0, 50.0, 10.0, step=5.0)
    fixed_leverage   = st.slider("Leverage", 5, 20, 10)
    starting_capital = st.number_input("Starting capital ($)", 100, 10000, 1000, step=100)

    run_btn  = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    save_btn = st.button("💾 Save Results", use_container_width=True)


# ── Shared helpers (inline so this file is self-contained) ───────────────────
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_all_symbols():
    """Return ALL active Bybit USDT perps — no turnover pre-filter.
    Turnover filtering happens historically inside the scanner."""
    resp = requests.get("https://api.bybit.com/v5/market/tickers",
                        params={"category": "linear"}, timeout=10)
    resp.raise_for_status()
    tickers = pd.DataFrame(resp.json()["result"]["list"])
    tickers = tickers[tickers["symbol"].str.endswith("USDT")].copy()
    tickers["turnover24h"] = pd.to_numeric(tickers["turnover24h"], errors="coerce").fillna(0)
    tickers = tickers[tickers["turnover24h"] >= 10_000]  # exclude completely dead/delisted
    return tickers.sort_values("turnover24h", ascending=False)["symbol"].tolist()


def bybit_get_candles(symbol, interval, start_ms, end_ms):
    iv_ms    = INTERVAL_MS_MAP[interval]
    url      = "https://api.bybit.com/v5/market/kline"
    rows_all, cursor = [], start_ms
    while cursor < end_ms:
        params = {"category":"linear","symbol":symbol,"interval":interval,
                  "start":cursor,"end":min(cursor+1000*iv_ms,end_ms),"limit":1000}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                rows = r.json()["result"]["list"]
                break
            except Exception:
                time.sleep(3*(attempt+1)); rows=[]
        if not rows: break
        rows_all.extend(rows)
        cursor = int(rows[0][0]) + iv_ms
        time.sleep(0.15)
    if not rows_all: return pd.DataFrame()
    df = pd.DataFrame(rows_all, columns=["ts","Open","High","Low","Close","Volume","Turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    return df.set_index("ts").astype(float).sort_index().drop(columns=["Turnover"])


@st.cache_data(show_spinner=False, ttl=3600)
def load_all_data(date_start_iso, date_end_iso):
    """Download ALL active Bybit perps — turnover filter applied historically inside scanner."""
    candidates = fetch_all_symbols()
    ds   = datetime.fromisoformat(date_start_iso).replace(tzinfo=timezone.utc)
    de   = (datetime.fromisoformat(date_end_iso).replace(tzinfo=timezone.utc)
            + timedelta(days=1))
    cutoff    = pd.Timestamp(ds)
    days_back = (de - ds).days
    expected  = days_back * 24
    data      = {}
    progress  = st.progress(0, text="Loading candle data...")
    for i, sym in enumerate(candidates):
        progress.progress((i+1)/len(candidates), text=f"[{i+1}/{len(candidates)}] {sym}")
        csv_path = RAW_DIR / f"{sym}_60m.csv"
        try:
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
                last_ts = df.index.max()
                from_ms = int((last_ts + pd.Timedelta(minutes=60)).timestamp() * 1000)
                to_ms   = int(de.timestamp() * 1000)
                if from_ms < to_ms:
                    fwd = bybit_get_candles(sym, "60", from_ms, to_ms)
                    if not fwd.empty:
                        df = pd.concat([df, fwd])
                        df = df[~df.index.duplicated(keep="last")].sort_index()
                        df.to_csv(csv_path)
                first_ts = df.index.min()
                if first_ts > cutoff + pd.Timedelta(hours=1):
                    back = bybit_get_candles(sym, "60",
                                             int(cutoff.timestamp()*1000),
                                             int(first_ts.timestamp()*1000))
                    if not back.empty:
                        df = pd.concat([back, df])
                        df = df[~df.index.duplicated(keep="last")].sort_index()
                        df.to_csv(csv_path)
            else:
                df = bybit_get_candles(sym, "60",
                                       int(ds.timestamp()*1000),
                                       int(de.timestamp()*1000))
                if not df.empty:
                    df.to_csv(csv_path)
            if df.empty: continue
            df  = df[(df.index >= cutoff) & (df.index <= de)]
            cov = len(df) / expected * 100
            if cov >= 20:
                data[sym] = df
        except Exception:
            pass
    progress.empty()
    return data


# ── Signal computation ────────────────────────────────────────────────────────
def compute_signals(df, ema_fast, ema_slow, vol_surge_mult, high_lookback_days, atr_period=14):
    df = df.copy()
    # EMA (used as guard + exit signal, not for entry)
    df["ema_fast"]  = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"]  = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["ema_bull"]  = df["ema_fast"] > df["ema_slow"]   # True = uptrend
    ema_prev        = df["ema_bull"].shift(1).fillna(False)
    df["ema_cross_buy"] = df["ema_bull"] & ~ema_prev    # just turned bullish

    # Volume
    df["vol_sma"]   = df["Volume"].rolling(168, min_periods=24).mean()
    df["vol_surge"] = df["Volume"] > vol_surge_mult * df["vol_sma"]

    # 20-day rolling high (lookback in 1h candles)
    lbk             = high_lookback_days * 24
    df["high_Nd"]   = df["Close"].rolling(lbk, min_periods=lbk // 2).max()
    # New high = current close ≥ previous rolling max (fresh breakout)
    df["new_Nd_high"] = (df["Close"] >= df["high_Nd"].shift(1)) & df["high_Nd"].shift(1).notna()

    # ATR for trail
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["atr"]     = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=atr_period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # Historical 24h turnover — used for dynamic universe + weak coin ranking
    df["turnover_24h"] = (df["Volume"] * df["Close"]).rolling(24, min_periods=6).sum()
    return df


# ── Scanner ───────────────────────────────────────────────────────────────────
def run_high_fade(data, weak_coin_set, cfg):
    signals = {sym: compute_signals(df,
                                    cfg["EMA_FAST"], cfg["EMA_SLOW"],
                                    cfg["VOL_SURGE_MULT"],
                                    cfg["HIGH_LOOKBACK_DAYS"])
               for sym, df in data.items()}

    all_timestamps   = sorted(set(ts for sd in signals.values() for ts in sd.index))
    open_positions   = {}
    pair_pause_until = {}
    pair_consec_sl   = {}
    pair_total_loss  = {}
    pair_killed      = set()
    all_trades       = []
    account          = cfg["STARTING_CAPITAL"]
    equity_curve     = []

    for ts in all_timestamps:
        is_wknd = ts.weekday() >= 5

        # ── Manage open positions ─────────────────────────────────────────────
        for sym in list(open_positions.keys()):
            if sym not in signals or ts not in signals[sym].index:
                continue
            row       = signals[sym].loc[ts]
            st_       = open_positions[sym]
            entry_px  = st_["entry_px"]
            entry_ts  = st_["entry_ts"]
            peak_fav  = st_["peak_fav"]
            trail_act = st_["trail_act"]
            px, low_px, high_px = row["Close"], row["Low"], row["High"]
            atr       = row["atr"] if not pd.isna(row.get("atr")) else 0
            pnl_pct   = (entry_px - px) / entry_px * 100   # SHORT: profit when price drops
            peak_fav  = min(peak_fav, px)                  # track lowest price reached
            open_positions[sym]["peak_fav"] = peak_fav
            closed    = False
            stop_px   = entry_px * (1 + cfg["STOP_LOSS_PCT"] / 100)   # stop above entry
            trade_size = st_.get("trade_size", cfg["TRADE_USDT"])

            # Hard stop (price went UP past stop)
            if high_px >= stop_px:
                pnl = (entry_px - stop_px) / entry_px * trade_size * cfg["FIXED_LEVERAGE"]
                all_trades.append(dict(
                    Symbol=sym, Dir="SHORT", Open_Time=entry_ts, Close_Time=ts,
                    Open_Price=round(entry_px,6), Close_Price=round(stop_px,6),
                    PnL=round(pnl,3), Note=f"stop-{cfg['STOP_LOSS_PCT']:.0f}%",
                    Trade_Size=trade_size,
                ))
                account += pnl
                pair_total_loss[sym] = pair_total_loss.get(sym, 0) + pnl
                pair_consec_sl[sym]  = pair_consec_sl.get(sym, 0) + 1
                if pair_consec_sl[sym] >= cfg["CONSEC_SL_LIMIT"]:
                    pair_pause_until[sym] = ts + pd.Timedelta(hours=24)
                del open_positions[sym]; closed = True

            # ATR trail stop
            if not closed:
                if pnl_pct >= cfg["TRAIL_ACTIVATE_PCT"]:
                    trail_act = True
                open_positions[sym]["trail_act"] = trail_act
                if trail_act and atr > 0:
                    # Trail above lowest price reached
                    tl = peak_fav + cfg["TRAIL_ATR_MULT"] * atr
                    if px >= tl:
                        pnl = (entry_px - px) / entry_px * trade_size * cfg["FIXED_LEVERAGE"]
                        all_trades.append(dict(
                            Symbol=sym, Dir="SHORT", Open_Time=entry_ts, Close_Time=ts,
                            Open_Price=round(entry_px,6), Close_Price=round(px,6),
                            PnL=round(pnl,3), Note="trail-stop", Trade_Size=trade_size,
                        ))
                        account += pnl
                        pair_total_loss[sym] = pair_total_loss.get(sym, 0) + pnl
                        pair_consec_sl[sym]  = 0
                        del open_positions[sym]; closed = True

            # EMA cross BUY = uptrend confirmed, exit short
            if not closed and row.get("ema_cross_buy", False):
                hours_open = (ts - entry_ts).total_seconds() / 3600
                if hours_open >= cfg["MIN_HOLD_HOURS"]:
                    pnl = (entry_px - px) / entry_px * trade_size * cfg["FIXED_LEVERAGE"]
                    all_trades.append(dict(
                        Symbol=sym, Dir="SHORT", Open_Time=entry_ts, Close_Time=ts,
                        Open_Price=round(entry_px,6), Close_Price=round(px,6),
                        PnL=round(pnl,3), Note="ema-exit", Trade_Size=trade_size,
                    ))
                    account += pnl
                    pair_total_loss[sym] = pair_total_loss.get(sym, 0) + pnl
                    pair_consec_sl[sym]  = 0
                    del open_positions[sym]

        # ── New entries ───────────────────────────────────────────────────────
        if cfg["WEEKEND_FILTER"] and is_wknd:
            if ts.hour == 0: equity_curve.append((ts, account, len(open_positions)))
            continue

        min_turn_usd = cfg["MIN_TURN_M"] * 1_000_000
        max_turn_usd = cfg["MAX_TURN_M"] * 1_000_000

        candidates = []
        for sym in weak_coin_set:
            if sym not in signals or sym in open_positions: continue
            if sym in pair_killed: continue
            if sym in pair_pause_until and ts < pair_pause_until[sym]: continue
            if pair_total_loss.get(sym, 0) <= -abs(cfg["TRADE_USDT"] * cfg["FIXED_LEVERAGE"] * 3):
                pair_killed.add(sym); continue
            if ts not in signals[sym].index: continue
            row = signals[sym].loc[ts]

            # Dynamic historical universe check
            t24 = row.get("turnover_24h", np.nan)
            if pd.isna(t24) or t24 < min_turn_usd or t24 > max_turn_usd: continue

            # Entry conditions
            if not row.get("new_Nd_high", False): continue   # must be fresh 20d high
            if not row.get("vol_surge", False): continue      # must have volume
            if cfg.get("USE_EMA_GUARD") and row.get("ema_bull", False): continue  # skip uptrends
            if pd.isna(row.get("atr_pct")) or row["atr_pct"] < 1.0: continue  # min volatility

            candidates.append((sym, row))

        # Sort by ATR% descending (highest momentum pumps first)
        candidates.sort(key=lambda x: x[1].get("atr_pct", 0), reverse=True)

        for sym, row in candidates:
            if len(open_positions) >= cfg["MAX_POSITIONS"]: break
            if account - len(open_positions) * cfg["TRADE_USDT"] < cfg["TRADE_USDT"]: break
            px = row["Close"]
            open_positions[sym] = dict(
                entry_px=px, entry_ts=ts,
                peak_fav=px, trail_act=False,
                trade_size=cfg["TRADE_USDT"],
            )
            pair_consec_sl[sym] = 0

        if ts.hour == 0:
            equity_curve.append((ts, account, len(open_positions)))

    # Close remaining positions at end of data
    for sym, st_ in list(open_positions.items()):
        sig_df = signals[sym]
        px     = sig_df.iloc[-1]["Close"]
        ts_    = sig_df.index[-1]
        ts_entry = st_["entry_ts"]
        pnl    = (st_["entry_px"] - px) / st_["entry_px"] * st_.get("trade_size", cfg["TRADE_USDT"]) * cfg["FIXED_LEVERAGE"]
        all_trades.append(dict(
            Symbol=sym, Dir="SHORT", Open_Time=ts_entry, Close_Time=ts_,
            Open_Price=round(st_["entry_px"],6), Close_Price=round(px,6),
            PnL=round(pnl,3), Note="end-of-data", Trade_Size=st_.get("trade_size", cfg["TRADE_USDT"]),
        ))
        account += pnl

    trades_df = pd.DataFrame(all_trades)
    eq_df     = pd.DataFrame(equity_curve, columns=["Date","Account","Open_Positions"])
    if not trades_df.empty:
        trades_df["Open_Time"]  = pd.to_datetime(trades_df["Open_Time"],  utc=True)
        trades_df["Close_Time"] = pd.to_datetime(trades_df["Close_Time"], utc=True)
        trades_df["duration_h"] = (trades_df["Close_Time"] - trades_df["Open_Time"]).dt.total_seconds() / 3600
    return trades_df, eq_df


# ── Dark chart helper ─────────────────────────────────────────────────────────
def dark_ax(fig, ax):
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if date_start >= date_end:
        st.error("Fix date range."); st.stop()

    cfg = dict(
        EMA_FAST=ema_fast, EMA_SLOW=ema_slow, USE_EMA_GUARD=use_ema_guard,
        VOL_SURGE_MULT=vol_surge_mult,
        HIGH_LOOKBACK_DAYS=high_lookback,
        WEAK_COIN_PCT=weak_coin_pct,
        STOP_LOSS_PCT=stop_loss_pct,
        TRAIL_ACTIVATE_PCT=trail_activate_pct,
        TRAIL_ATR_MULT=trail_atr_mult,
        MIN_HOLD_HOURS=min_hold_hours,
        CONSEC_SL_LIMIT=consec_sl_limit,
        MAX_POSITIONS=max_positions,
        TRADE_USDT=float(trade_usdt),
        FIXED_LEVERAGE=fixed_leverage,
        STARTING_CAPITAL=float(starting_capital),
        WEEKEND_FILTER=weekend_filter,
        MIN_TURN_M=min_turn_m,
        MAX_TURN_M=max_turn_m,
    )

    with st.status("📡 Loading data...", expanded=True) as status:
        st.write("Fetching full Bybit universe (all active USDT perps)...")
        data    = load_all_data(date_start.isoformat(), date_end.isoformat())
        n_pairs = len(data)

        # Weak coin set: bottom X% by HISTORICAL avg turnover (Volume × Close) over the period
        _avg_t  = {s: (df["Volume"] * df["Close"]).mean() for s, df in data.items()}
        _sorted = sorted(_avg_t, key=lambda s: _avg_t[s])
        _n      = max(1, int(len(_sorted) * weak_coin_pct / 100))
        weak_coin_set = set(_sorted[:_n])
        st.write(f"Loaded {n_pairs} pairs  |  Weak coin set: {len(weak_coin_set)} (bottom {weak_coin_pct}% by historical turnover)")

        status.update(label="Running high-fade backtest...", state="running")
        trades_df, eq_df = run_high_fade(data, weak_coin_set, cfg)
        status.update(label=f"✅ Done — {len(trades_df)} trades", state="complete")

    if trades_df.empty:
        st.warning("No trades fired. Try lowering the weak coin % or extending the date range.")
        st.stop()

    trades_df["month"] = trades_df["Open_Time"].dt.to_period("M").astype(str)
    trades_df["dow"]   = trades_df["Open_Time"].dt.day_name()
    st.session_state["hf_trades"] = trades_df
    st.session_state["hf_eq"]     = eq_df
    st.session_state["hf_cfg"]    = cfg
    st.session_state["hf_n"]      = n_pairs


# ── Save ──────────────────────────────────────────────────────────────────────
if save_btn:
    if "hf_trades" not in st.session_state:
        st.warning("Run a backtest first."); st.stop()
    ts_  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid  = f"highfade_{ts_}"
    st.session_state["hf_trades"].to_csv(RES_DIR / f"{rid}_trades.csv", index=False)
    st.session_state["hf_eq"].to_csv(RES_DIR / f"{rid}_equity.csv",     index=False)
    (RES_DIR / f"{rid}_config.json").write_text(
        json.dumps(st.session_state["hf_cfg"], indent=2))
    st.sidebar.success(f"✅ Saved: {rid}")


# ── Display ───────────────────────────────────────────────────────────────────
if "hf_trades" not in st.session_state:
    st.info("👈 Configure parameters and click **Run Backtest**.")
    st.markdown("""
| Parameter | What it does |
|---|---|
| **Weak coin %** | Only short the bottom X% of universe by avg daily turnover — thinly traded coins most likely to be pumped |
| **High lookback** | How many days back to check for the "new high" — 20d = fresh 3-week high |
| **EMA guard** | Skips entry if EMA fast > EMA slow — prevents shorting into a real established trend |
| **Vol surge** | Requires volume spike on the pump — confirms it's a real FOMO move, not a drift |
| **Min hold** | Holds position even if EMA crosses back quickly — prevents immediate whipsaw exits |
""")
    st.stop()

trades_df = st.session_state["hf_trades"]
eq_df     = st.session_state["hf_eq"]
cfg       = st.session_state["hf_cfg"]
n_pairs   = st.session_state["hf_n"]
sc        = cfg["STARTING_CAPITAL"]

# ── Metrics ───────────────────────────────────────────────────────────────────
n    = len(trades_df); wins = (trades_df["PnL"]>0).sum(); tot = trades_df["PnL"].sum()
final = sc + tot; ret_pct = tot / sc * 100
winners = trades_df[trades_df["PnL"]>0]; losers = trades_df[trades_df["PnL"]<0]
eq_vals = eq_df["Account"].values if not eq_df.empty else np.array([sc])
peak_eq = np.maximum.accumulate(eq_vals)
mdd     = (eq_vals - peak_eq).min()
pf      = winners["PnL"].sum() / abs(losers["PnL"].sum()) if len(losers) > 0 else float("inf")
stops   = trades_df[trades_df["Note"].str.contains("stop", na=False)]

st.info(
    f"**Downloaded:** {n_pairs} pairs (all active perps)  |  "
    f"**Weak coin set:** bottom {cfg['WEAK_COIN_PCT']}% ({len(weak_coin_set) if 'weak_coin_set' in dir() else '?'} pairs) by historical turnover  |  "
    f"**High lookback:** {cfg['HIGH_LOOKBACK_DAYS']}d  |  "
    f"**Turnover band:** ${cfg.get('MIN_TURN_M',0.5):.1f}M–${cfg.get('MAX_TURN_M',500):.0f}M (historical)  |  "
    f"**EMA guard:** {'ON ' + str(cfg['EMA_FAST']) + '/' + str(cfg['EMA_SLOW']) if cfg.get('USE_EMA_GUARD') else 'OFF'}  |  "
    f"**Stop:** {cfg['STOP_LOSS_PCT']}%  |  "
    f"**Min hold:** {cfg['MIN_HOLD_HOURS']}h"
)

# ── Risk metrics (matches main scanner for comparison) ────────────────────────
_rf        = 0.05 / 365
_daily_pnl = trades_df.copy()
_daily_pnl["_date"] = _daily_pnl["Close_Time"].dt.date
_dpnl      = _daily_pnl.groupby("_date")["PnL"].sum()
_drange    = pd.date_range(trades_df["Open_Time"].min().date(),
                           trades_df["Close_Time"].max().date(), freq="D")
_dpnl      = _dpnl.reindex(_drange.date, fill_value=0)
_dret      = _dpnl / sc
_excess    = _dret - _rf
sharpe     = float((_excess.mean() / _excess.std()) * np.sqrt(365)) if _excess.std() > 0 else 0
_down      = _dret[_dret < 0]
sortino    = float((_dret.mean() - _rf) / _down.std() * np.sqrt(365)) if len(_down) and _down.std() > 0 else 0
_total_days= len(_dpnl)
_ann_ret   = ret_pct / _total_days * 365
_mdd_pct   = mdd / sc * 100
calmar     = float(_ann_ret / abs(_mdd_pct)) if _mdd_pct != 0 else 0
_monthly   = trades_df.groupby(trades_df["Close_Time"].dt.to_period("M").astype(str))["PnL"].sum()
m_sharpe   = float((_monthly.mean() / _monthly.std()) * np.sqrt(12)) if _monthly.std() > 0 else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Return",      f"${tot:+,.2f}", f"{ret_pct:+.1f}%")
c2.metric("Final Account",     f"${final:,.2f}", f"Ann. {_ann_ret:+.0f}%")
c3.metric("Win Rate",          f"{wins/n*100:.1f}%", f"{wins}W / {n-wins}L")
c4.metric("Profit Factor",     f"{pf:.2f}")
c5.metric("Avg Win / Avg Loss",f"${winners['PnL'].mean():.2f} / ${losers['PnL'].mean():.2f}" if len(losers) else "—")
c6.metric("Max Drawdown",      f"${mdd:,.2f}", f"{_mdd_pct:.1f}% of peak")

r1,r2,r3,r4 = st.columns(4)
r1.metric("Sharpe Ratio",  f"{sharpe:.3f}",
          "excellent >2" if sharpe>2 else ("good >1" if sharpe>1 else "weak <1"))
r2.metric("Sortino Ratio", f"{sortino:.3f}", help="Downside-only risk.")
r3.metric("Calmar Ratio",  f"{calmar:.3f}",  help="Annual return ÷ max drawdown %.")
r4.metric("Monthly Sharpe",f"{m_sharpe:.3f}",help="Consistency month-to-month.")

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")
if not eq_df.empty:
    fig, ax = plt.subplots(figsize=(14, 4))
    color   = "#e74c3c" if final < sc else "#2ecc71"
    dates   = pd.to_datetime(eq_df["Date"])
    ax.plot(dates, eq_vals, lw=1.5, color=color)
    ax.fill_between(dates, eq_vals, sc, alpha=0.15, color=color)
    ax.axhline(sc, color="white", lw=0.6, ls="--", alpha=0.4)
    ax.set_ylabel("Account ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    dark_ax(fig, ax); st.pyplot(fig); plt.close(fig)

# ── Monthly breakdown ─────────────────────────────────────────────────────────
st.subheader("Monthly Breakdown")
monthly_tbl = trades_df.groupby("month").agg(
    Trades    =("PnL","count"),
    Win_Rate  =("PnL", lambda x: round((x>0).mean()*100,1)),
    Total_PnL =("PnL","sum"),
    Stops     =("Note", lambda x: x.str.contains("stop").sum()),
    Trail_wins=("Note", lambda x: (x=="trail-stop").sum()),
    EMA_exits =("Note", lambda x: (x=="ema-exit").sum()),
    Avg_hold  =("duration_h","mean"),
).reset_index().rename(columns={"month":"Month"})
monthly_tbl["Total_PnL"] = monthly_tbl["Total_PnL"].round(2)
monthly_tbl["Avg_hold"]  = monthly_tbl["Avg_hold"].round(1)
st.dataframe(monthly_tbl, use_container_width=True, hide_index=True)

# ── Exit breakdown ────────────────────────────────────────────────────────────
st.subheader("Exit Breakdown")
exit_tbl = trades_df.groupby("Note")["PnL"].agg(["count","sum","mean"]).round(2)
exit_tbl.columns = ["Count","Total_PnL","Avg_PnL"]
st.dataframe(exit_tbl, use_container_width=True)

# ── Top pairs ─────────────────────────────────────────────────────────────────
st.subheader("All Pairs — P&L Summary")
pair_tbl = trades_df.groupby("Symbol").agg(
    Trades   =("PnL","count"),
    Win_Rate =("PnL", lambda x: round((x>0).mean()*100,1)),
    Total_PnL=("PnL","sum"),
    Best     =("PnL","max"),
    Worst    =("PnL","min"),
).round(2).sort_values("Total_PnL", ascending=False).reset_index()
st.dataframe(pair_tbl, use_container_width=True, hide_index=True)

with st.expander("📋 All Trades"):
    st.dataframe(
        trades_df[["Symbol","Open_Time","Close_Time","Open_Price","Close_Price",
                   "PnL","Note","duration_h","dow","month"]].sort_values("Close_Time", ascending=False),
        use_container_width=True, hide_index=True,
    )
