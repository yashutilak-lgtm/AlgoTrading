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

