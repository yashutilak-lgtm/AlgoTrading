import ccxt
import pandas as pd
import numpy as np
import requests
import time

pd.set_option('future.no_silent_downcasting', True)

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
SYMBOL        = "BTC/USDT"
TIMEFRAME     = "2h"         
START_DATE    = "2023-01-01T00:00:00Z"

INITIAL_CAPITAL   = 10_000.0
RISK_PER_TRADE    = 0.01
FEE_RATE          = 0.0005
SLIPPAGE          = 0.0003

# ── KEY GEOMETRY CHANGE ──────────────────────────
STOP_ATR_MULT      = 1.0    # tighter: 1.0×ATR (was 1.5) → cuts avg loss
RR_RATIO           = 2.5    # bigger:  2.5R target (was 2.0) → raises avg win
TRAIL_ATR_MULT     = 1.5    # trail width unchanged
TRAIL_ACTIVATION_R = 2.0    # activate trail LATER (was 1.5) → let target hit
MAX_STOP_PCT       = 0.05
MAX_NOTIONAL_FRAC  = 0.30

ATR_P     = 14
EMA_FAST  = 21
EMA_MED   = 50
RSI_P     = 14
ADX_P     = 14
SWING_W   = 6
HTF_EMA_P = 50
HTF_RSI_P = 14

PB_RSI_LOW    = 35
PB_RSI_HIGH   = 52
ENTRY_RSI_MAX = 58
PB_ATR_DIST   = 2.5
ADX_MIN       = 20
HTF_RSI_MIN   = 50

USE_SESSION = True;  SESSION_S = 6;  SESSION_E = 21
USE_FG      = True;  FG_MIN = 25;   FG_MAX = 80

# ═══════════════════════════════════════════════════
# ON-CHAIN
# ═══════════════════════════════════════════════════

def fetch_fear_greed(limit=1000) -> pd.Series:
    print("Fetching Fear & Greed Index...")
    try:
        r = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json",
            timeout=15)
        r.raise_for_status()
        s = pd.Series({
            pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"): int(d["value"])
            for d in r.json()["data"]
        }).sort_index()
        print(f"  F&G: {len(s)} days ({s.index[0].date()} → {s.index[-1].date()})")
        return s
    except Exception as e:
        print(f"  [WARN] F&G unavailable: {e}")
        return pd.Series(dtype=float)


def align_fg(fg: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if fg.empty:
        return pd.Series(50, index=idx)
    return (fg.reindex(fg.index.union(idx))
              .ffill().infer_objects(copy=False)
              .reindex(idx))
    
# ═══════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════

def download_ohlcv() -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True,
                       "options": {"defaultType": "spot"}})
    since = ex.parse8601(START_DATE)
    rows  = []
    print(f"Downloading {SYMBOL} {TIMEFRAME}...")
    while True:
        batch = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME,
                               since=since, limit=1000)
        if not batch: break
        rows.extend(batch)
        print(f"  {len(rows):,} candles", end="\r")
        if len(batch) < 1000: break
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    print(f"\n  Done — {len(rows):,} candles")
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df.drop(columns=["ts"])