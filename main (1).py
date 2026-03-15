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

# ═══════════════════════════════════════════════════
# HTF
# ═══════════════════════════════════════════════════

def build_htf(df: pd.DataFrame) -> pd.DataFrame:
    tf = "1h" if TIMEFRAME == "15m" else "4h"
    h_close = df["close"].resample(tf).last().dropna()
    ema = h_close.ewm(span=HTF_EMA_P, adjust=False).mean()
    d   = h_close.diff()
    g   = d.clip(lower=0).rolling(HTF_RSI_P).mean()
    lo  = (-d.clip(upper=0)).rolling(HTF_RSI_P).mean()
    rsi = 100 - (100 / (1 + g / lo.replace(0, np.nan)))
    htf = pd.DataFrame({
        "htf_trend_up": (h_close > ema).astype(bool),
        "htf_slope_up": (ema > ema.shift(3)).astype(bool),
        "htf_rsi_ok":   (rsi >= HTF_RSI_MIN).astype(bool),
    }, index=h_close.index)
    return (htf.reindex(htf.index.union(df.index))
               .ffill().infer_objects(copy=False)
               .reindex(df.index))
    
# ═══════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    pc = df["close"].shift(1)
    df["tr"]  = np.maximum(df["high"] - df["low"],
                np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    df["atr"] = df["tr"].rolling(ATR_P).mean()
    df["ema21"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_MED,  adjust=False).mean()

    d    = df["close"].diff()
    gain = d.clip(lower=0).rolling(RSI_P).mean()
    loss = (-d.clip(upper=0)).rolling(RSI_P).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ADX
    up   = df["high"] - df["high"].shift(1)
    dn   = df["low"].shift(1) - df["low"]
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    mdm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    a    = 1.0 / ADX_P
    atrw = pd.Series(df["tr"].values, index=df.index).ewm(alpha=a, adjust=False).mean()
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atrw.replace(0, np.nan)
    mdi  = 100 * pd.Series(mdm, index=df.index).ewm(alpha=a, adjust=False).mean() / atrw.replace(0, np.nan)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=a, adjust=False).mean()

    rmin = df["low"].rolling(SWING_W * 2 + 1, center=False).min()
    df["swing_low"] = df["low"].where(df["low"] == rmin).shift(SWING_W)

    df["atr_pct"]    = df["atr"] / df["close"]
    df["atr_pct_ma"] = df["atr_pct"].rolling(50).mean()
    df["calm"]       = df["atr_pct"] < df["atr_pct_ma"] * 1.8
    df["hour_utc"]   = df.index.hour

    print(f"Indicators done ({TIMEFRAME}).")
    return df

# ═══════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════

def run_backtest(df, htf, fga):
    equity = INITIAL_CAPITAL
    pos    = None
    eqc    = []
    tlog   = []
    wins = losses = 0
    WARMUP = max(200 + EMA_MED, SWING_W * 4, ADX_P * 3)

    cl  = df["close"].values;   op = df["open"].values
    hi  = df["high"].values;    lo = df["low"].values
    atr = df["atr"].values;     e21= df["ema21"].values
    e50 = df["ema50"].values;   rs = df["rsi"].values
    adx = df["adx"].values;     sw = df["swing_low"].values
    calm= df["calm"].values;    hr = df["hour_utc"].values
    tup = htf["htf_trend_up"].values
    tsl = htf["htf_slope_up"].values
    tri = htf["htf_rsi_ok"].values
    fg  = fga.values

    for i in range(WARMUP, len(df)):
        px = cl[i]; av = atr[i]; ts = df.index[i]

        if np.isnan(av) or np.isnan(e50[i]) or np.isnan(adx[i]):
            eqc.append(equity); continue

        fg_ok  = not USE_FG or (FG_MIN <= float(fg[i]) <= FG_MAX)
        sess   = not USE_SESSION or (SESSION_S <= int(hr[i]) < SESSION_E)
        htf_ok = bool(tup[i]) and bool(tsl[i]) and bool(tri[i])

        # ── Entry ────────────────────────────────────
        if pos is None and i > WARMUP:
            slo = sw[i]; pr = rs[i - 1]
            ok = [
                not np.isnan(slo), not np.isnan(pr),
                px > e50[i],                              # above EMA-50
                px > slo,                                 # above swing low
                PB_RSI_LOW <= pr <= PB_RSI_HIGH,          # prior RSI dipped
                op[i] < cl[i],                            # green bar
                cl[i] > cl[i - 1],                       # recovering
                rs[i] < ENTRY_RSI_MAX,                   # not extended
                abs(px - e21[i]) < PB_ATR_DIST * av,     # near EMA-21
                bool(calm[i]),                            # calm ATR
                adx[i] >= ADX_MIN,                       # trend strength
                htf_ok, sess, fg_ok,
            ]
            if all(ok):
                entry = px * (1 + SLIPPAGE)
                stop  = slo - STOP_ATR_MULT * av         # TIGHTER: 1.0×ATR
                rdist = entry - stop
                if rdist <= 0 or rdist / entry > MAX_STOP_PCT:
                    eqc.append(equity); continue
                tgt  = entry + RR_RATIO * rdist           # BIGGER: 2.5R
                size = (equity * RISK_PER_TRADE) / rdist
                if size * entry > equity * MAX_NOTIONAL_FRAC:
                    size = (equity * MAX_NOTIONAL_FRAC) / entry
                equity -= entry * size * FEE_RATE
                pos = {"entry": entry, "stop": stop, "target": tgt,
                       "size": size, "highest": entry, "idx": i,
                       "ts": ts, "rdist": rdist, "fg": float(fg[i]),
                       "adx": round(float(adx[i]), 1)}

        # ── Management ───────────────────────────────
        if pos is not None:
            h = hi[i]; l = lo[i]
            if h > pos["highest"]: pos["highest"] = h

            # Trail — activates at 2.0R (LATER than before)
            pr_r = (pos["highest"] - pos["entry"]) / pos["rdist"]
            if pr_r >= TRAIL_ACTIVATION_R:
                trail = pos["highest"] - TRAIL_ATR_MULT * av
                pos["stop"] = max(pos["stop"], trail)

            # Target checked before stop
            epx = ewhy = None
            if h >= pos["target"]:
                epx = pos["target"] * (1 - SLIPPAGE); ewhy = "target"
            elif l <= pos["stop"]:
                epx  = pos["stop"] * (1 - SLIPPAGE)
                ewhy = "trailing" if pos["stop"] > pos["entry"] else "stop"

            if epx is not None:
                pnl    = (epx - pos["entry"]) * pos["size"]
                pnl   -= epx * pos["size"] * FEE_RATE
                equity += pnl
                wins   += 1 if pnl > 0 else 0
                losses += 1 if pnl <= 0 else 0
                tlog.append({
                    "entry_time":  pos["ts"], "exit_time": ts,
                    "entry": round(pos["entry"],2), "exit": round(epx,2),
                    "stop":  round(pos["stop"],2),  "target": round(pos["target"],2),
                    "size_btc": round(pos["size"],6), "risk_dist": round(pos["rdist"],2),
                    "pnl_usd":   round(pnl,2),
                    "r_multiple": round(pnl/(pos["rdist"]*pos["size"]),3),
                    "bars_held": i - pos["idx"], "reason": ewhy,
                    "adx_entry": pos["adx"], "fg_entry": round(pos["fg"],1),
                    "equity_after": round(equity,2),
                })
                pos = None
        eqc.append(equity)

    if pos is not None:
        fp   = cl[-1] * (1 - SLIPPAGE)
        pnl  = (fp - pos["entry"]) * pos["size"]
        pnl -= fp * pos["size"] * FEE_RATE
        equity += pnl
        wins   += 1 if pnl > 0 else 0
        losses += 1 if pnl <= 0 else 0
        tlog.append({
            "entry_time": pos["ts"], "exit_time": df.index[-1],
            "entry": round(pos["entry"],2), "exit": round(fp,2),
            "stop": round(pos["stop"],2), "target": round(pos["target"],2),
            "size_btc": round(pos["size"],6), "risk_dist": round(pos["rdist"],2),
            "pnl_usd": round(pnl,2),
            "r_multiple": round(pnl/(pos["rdist"]*pos["size"]),3),
            "bars_held": len(df)-1-pos["idx"], "reason": "final_close",
            "adx_entry": pos["adx"], "fg_entry": pos["fg"],
            "equity_after": round(equity,2),
        })
    return {"equity": equity, "eqc": eqc, "tlog": tlog,
            "wins": wins, "losses": losses, "warmup": WARMUP}
    
# ═══════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════

def report(res, df, fg_series, fga):
    eq   = res["equity"]; eqc = res["eqc"]
    tlog = res["tlog"];   W   = res["warmup"]
    n    = len(eqc)

    eq_s = pd.Series(eqc, index=df.index[W: W+n])
    rets = eq_s.pct_change().dropna()
    tot  = res["wins"] + res["losses"]

    pnls = [t["pnl_usd"] for t in tlog]
    wp   = [p for p in pnls if p > 0]
    lp   = [p for p in pnls if p <= 0]
    pf   = sum(wp)/abs(sum(lp)) if lp else float("inf")
    wr   = res["wins"]/tot*100 if tot else 0
    aw   = np.mean(wp) if wp else 0
    al   = np.mean(lp) if lp else 0
    exp  = (wr/100)*aw + (1-wr/100)*al
    avg_r= np.mean([t["r_multiple"] for t in tlog]) if tlog else 0
    avg_b= np.mean([t["bars_held"]  for t in tlog]) if tlog else 0

    dd   = (eq_s.cummax()-eq_s)/eq_s.cummax()
    mdd  = dd.max()*100
    tr   = (eq/INITIAL_CAPITAL-1)*100
    ppy  = 365.25*24*(4 if TIMEFRAME=="15m" else 1)
    sh   = (rets.mean()/rets.std())*np.sqrt(ppy) if rets.std() else 0
    dsd  = rets[rets<0].std()
    so   = (rets.mean()/dsd)*np.sqrt(ppy) if dsd else 0
    cal  = (tr/100)/(mdd/100) if mdd else 0

    cw=cl=mw=ml=0
    for p in pnls:
        if p>0: cw+=1; cl=0
        else:   cl+=1; cw=0
        mw=max(mw,cw); ml=max(ml,cl)

    by_r={}
    for t in tlog: by_r.setdefault(t["reason"],[]).append(t["pnl_usd"])

    pf_s = f"{pf:.2f}" if pf!=float("inf") else "∞"
    wl   = f"{abs(aw/al):.2f}×" if al else "N/A"
    bep  = (1/(1+(aw/abs(al))))*100 if al else 0
    W_   = 67

    print("\n"+"═"*W_)
    print(f"  BTC/USDT {TIMEFRAME} — v7  (geometry fix: stop 1.0×ATR, target 2.5R)")
    print("═"*W_)
    print(f"  Final equity         : ${eq_s.iloc[-1]:>12,.2f}  (started ${INITIAL_CAPITAL:,.0f})")
    print(f"  Total return         : {tr:>+10.2f}%")
    print(f"  Max drawdown         : {mdd:>10.2f}%")
    print("─"*W_)
    print(f"  Total trades         : {tot:>10d}")
    print(f"  Win rate             : {wr:>10.1f}%  (break-even at this W/L: {bep:.1f}%)")
    print(f"  Profit factor        : {pf_s:>10}  (viable ≥ 1.3)")
    print(f"  Avg win              : ${aw:>10,.2f}")
    print(f"  Avg loss             : ${al:>10,.2f}")
    print(f"  Win / Loss ratio     : {wl:>10}")
    print(f"  Expectancy / trade   : ${exp:>10,.2f}")
    print(f"  Avg R multiple       : {avg_r:>10.3f}R")
    print(f"  Avg hold             : {avg_b:>10.1f} bars  ({avg_b*15/60:.1f}h)" if TIMEFRAME=="15m"
          else f"  Avg hold             : {avg_b:>10.1f} bars  ({avg_b:.1f}h)")
    print("─"*W_)
    print(f"  Sharpe ratio         : {sh:>10.2f}")
    print(f"  Sortino ratio        : {so:>10.2f}")
    print(f"  Calmar ratio         : {cal:>10.2f}")
    print("─"*W_)
    print(f"  Max consec. wins     : {mw:>10d}")
    print(f"  Max consec. losses   : {ml:>10d}")
    print("─"*W_)
    print("  Exit breakdown:")
    n_hard=0
    for reason, ps in sorted(by_r.items()):
        n_=len(ps); tot_=sum(ps); avg_=np.mean(ps)
        wr_=sum(1 for p in ps if p>0)/n_*100
        print(f"    {reason:<14}: {n_:>4} trades  avg ${avg_:>7.2f}  "
              f"total ${tot_:>9,.2f}  WR {wr_:>5.1f}%")
        if reason=="stop": n_hard=n_

    stop_pct = (n_hard+len(by_r.get("trailing",[]))) / tot*100 if tot else 0
    tgt_n    = len(by_r.get("target",[]))
    tgt_pct  = tgt_n/tot*100 if tot else 0
    flag = "" if stop_pct<60 else (" " if stop_pct<75 else "")
    print("─"*W_)
    print(f"  Combined stop rate   : {stop_pct:>10.1f}%  {flag}")
    print(f"  Target hit rate      : {tgt_pct:>10.1f}%")

    # Theory vs actual check
    exp_wl = abs(aw/al) if al else 0
    bep_wl = (1-wr/100)/(wr/100) if wr else 0
    gap    = exp_wl - bep_wl
    print("─"*W_)
    print(f"  Break-even W/L needed: {bep_wl:>10.3f}×")
    print(f"  Actual W/L           : {exp_wl:>10.3f}×")
    gf = " profitable geometry" if gap>0 else f" need {bep_wl-exp_wl:.3f}× more"
    print(f"  Gap                  : {gap:>+10.3f}×  {gf}")
    print("═"*W_)

    if not fg_series.empty:
        lv = int(fg_series.iloc[-1])
        label=("Extreme Fear" if lv<25 else "Fear" if lv<45 else
               "Neutral" if lv<55 else "Greed" if lv<80 else "Extreme Greed")
        blk=sum(1 for v in fga if not (FG_MIN<=v<=FG_MAX))
        print(f"\n  ON-CHAIN:")
        print(f"  Fear & Greed today   : {lv} — {label}  ({fg_series.index[-1].date()})")
        print(f"  Bars gated by F&G    : {blk:,}/{len(fga):,} ({blk/len(fga)*100:.1f}%)")

    pd.DataFrame(tlog).to_csv(f"trades_v7_{TIMEFRAME}.csv", index=False)
    eq_s.to_csv(f"equity_curve_v7_{TIMEFRAME}.csv", header=["equity"])
    print(f"\n  Saved: trades_v7_{TIMEFRAME}.csv | equity_curve_v7_{TIMEFRAME}.csv")
    print("═"*W_)