# -*- coding: utf-8 -*-
"""
sr_price_action.py
------------------
进攻版 S/R + Price Action（防重绘 + zone + 更强趋势/突破回踩逻辑）
输出：按日 0/1 持仓信号（每资产最多一仓）

目标：大幅提高年化
- 更早开跑（更短 warmup）
- 更偏趋势跟随（减少过早结构性止盈）
- 突破采用“突破回踩确认”降低假突破
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..backtester import backtest_spot_strategy, BacktestResult


@dataclass
class SRConfig:
    # --- S/R ---
    pivot_order: int = 6              # 更快 pivot（原 10 偏慢）
    atr_window: int = 14
    zone_atr_mult: float = 1.2        # zone 稍微收紧，减少“太宽导致信号迟钝”
    min_bars: int = 30                # 原 200 太晚，年化被拖累

    # --- 趋势过滤（更偏进攻） ---
    use_trend_filter: bool = True
    ema_fast_window: int = 50
    ema_slow_window: int = 120        # 原 200 太慢，缩短更早参与趋势
    ema_slope_window: int = 10        # 用 slow EMA 斜率确认趋势“在上升”

    # --- 入场增强 ---
    use_breakout_retest: bool = True  # 突破后回踩确认再进，减少假突破
    retest_max_atr: float = 1.0       # 回踩幅度限制：low 不可比阻力上沿低太多（按 ATR 衡量）

    # --- 退出逻辑（让利润奔跑） ---
    use_fast_ema_exit: bool = True    # 跌破 EMA_fast 就离场（比“碰阻回落”更趋势）
    min_hold_bars: int = 3            # 避免刚进就被小抖动洗掉

    # --- 可选：成交量过滤（默认关；日线 volume 有时噪声大）---
    use_volume_filter: bool = False
    vol_ma_window: int = 20
    vol_mult: float = 1.2             # vol > MA*mult 才允许 breakout 进场（更干净但更少交易）


def _calc_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _confirmed_pivots(close: pd.Series, order: int) -> tuple[pd.Series, pd.Series]:
    """
    center rolling max/min + 右移确认 order 天，避免重绘。
    """
    x = pd.to_numeric(close, errors="coerce")
    win = 2 * order + 1
    rmin = x.rolling(win, center=True, min_periods=win).min()
    rmax = x.rolling(win, center=True, min_periods=win).max()
    piv_low = (x == rmin)
    piv_high = (x == rmax)

    sup = pd.Series(np.nan, index=x.index)
    res = pd.Series(np.nan, index=x.index)
    sup[piv_low] = x[piv_low]
    res[piv_high] = x[piv_high]
    return sup.shift(order), res.shift(order)


def _build_zones(level_series: pd.Series, atr: pd.Series, mult: float) -> pd.DataFrame:
    """
    把确认 pivot 点变成 zone（上下界）。
    非 pivot 行为 NaN；之后上层会 ffill() 延续最近有效 zone。
    """
    lvl = pd.to_numeric(level_series, errors="coerce")
    a = pd.to_numeric(atr, errors="coerce")
    z = pd.DataFrame(index=lvl.index)
    z["level"] = lvl
    z["z_low"] = lvl - mult * a
    z["z_high"] = lvl + mult * a
    m = lvl.notna()
    z.loc[~m, ["level", "z_low", "z_high"]] = np.nan
    return z


def _make_signals(spot: pd.DataFrame, cfg: SRConfig) -> pd.Series:
    df = spot.copy().sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["atr"] = _calc_atr(df, cfg.atr_window)

    # pivots on close (consistent with original)
    sup_c, res_c = _confirmed_pivots(df["close"], cfg.pivot_order)
    sup_zone = _build_zones(sup_c, df["atr"], cfg.zone_atr_mult)
    res_zone = _build_zones(res_c, df["atr"], cfg.zone_atr_mult)

    # extend last valid zones forward
    df["sup_low"] = sup_zone["z_low"].ffill()
    df["sup_high"] = sup_zone["z_high"].ffill()
    df["res_low"] = res_zone["z_low"].ffill()
    df["res_high"] = res_zone["z_high"].ffill()

    # trend filter
    if cfg.use_trend_filter:
        df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast_window, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow_window, adjust=False).mean()

        # slow EMA slope confirmation: rising over slope_window
        slow_slope = df["ema_slow"] - df["ema_slow"].shift(cfg.ema_slope_window)

        trend_ok = (df["ema_fast"] > df["ema_slow"]) & (slow_slope > 0)
    else:
        df["ema_fast"] = np.nan
        df["ema_slow"] = np.nan
        trend_ok = pd.Series(True, index=df.index)

    # volume filter (optional)
    if cfg.use_volume_filter and "volume" in df.columns:
        df["vol_ma"] = df["volume"].rolling(cfg.vol_ma_window, min_periods=cfg.vol_ma_window).mean()
        vol_ok = df["volume"] > (df["vol_ma"] * cfg.vol_mult)
    else:
        vol_ok = pd.Series(True, index=df.index)

    sig = pd.Series(0, index=df.index, dtype=int)

    in_pos = False
    hold_bars = 0

    for i in range(len(df)):
        if i < cfg.min_bars:
            sig.iloc[i] = 0
            continue

        # required prices
        c0 = df.loc[i, "close"]
        c1 = df.loc[i - 1, "close"]
        h0 = df.loc[i, "high"]
        l0 = df.loc[i, "low"]
        atr0 = df.loc[i, "atr"]

        sup_low = df.loc[i, "sup_low"]
        sup_high = df.loc[i, "sup_high"]
        res_low = df.loc[i, "res_low"]
        res_high = df.loc[i, "res_high"]

        if not in_pos:
            # --- Entry A: support bounce (more sensitive than before: uses low touch) ---
            # low touched support zone (or near) AND close reclaims above sup_high
            bounce = (
                np.isfinite(sup_high)
                and np.isfinite(l0)
                and (l0 <= sup_high)
                and (c0 > sup_high)
            )

            # --- Entry B: breakout ---
            breakout = (
                np.isfinite(res_high)
                and (c0 > res_high)
                and vol_ok.iloc[i]
            )

            # --- Entry C: breakout-retest (preferred) ---
            # yesterday closed above res_high, today intraday retests near res_high but closes back above
            if cfg.use_breakout_retest and np.isfinite(res_high) and np.isfinite(atr0) and atr0 > 0:
                retest_ok = (l0 >= (res_high - cfg.retest_max_atr * atr0))  # do not dip too deep below res_high
                breakout_retest = (c1 > res_high) and retest_ok and (c0 > res_high)
            else:
                breakout_retest = False

            enter = (bounce or breakout or breakout_retest)

            if trend_ok.iloc[i] and enter:
                in_pos = True
                hold_bars = 1
                sig.iloc[i] = 1
            else:
                sig.iloc[i] = 0

        else:
            hold_bars += 1

            # --- Exit rules (profit-runner) ---
            # 1) structural failure: close breaks below support zone
            fail_support = (np.isfinite(sup_low) and (c0 < sup_low))

            # 2) trend failure: fast EMA cross down / price below ema_fast (more responsive)
            if cfg.use_fast_ema_exit and np.isfinite(df.loc[i, "ema_fast"]):
                trend_fail = (c0 < df.loc[i, "ema_fast"]) or (df.loc[i, "ema_fast"] < df.loc[i, "ema_slow"])
            else:
                trend_fail = (not trend_ok.iloc[i])

            # 3) optional: failed breakout -> closes back under res_low (only if res_low exists)
            fail_breakout = (np.isfinite(res_low) and (c0 < res_low))

            # avoid immediate whipsaw exits
            if hold_bars < cfg.min_hold_bars:
                exit_now = False
            else:
                exit_now = bool(fail_support or trend_fail or fail_breakout)

            if exit_now:
                in_pos = False
                hold_bars = 0
                sig.iloc[i] = 0
            else:
                sig.iloc[i] = 1

    sig.index = df["date"]
    return sig


def run_sr_price_action(
    spot_by_symbol: Dict[str, pd.DataFrame],
    cfg: SRConfig = SRConfig(),
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """
    入口：生成每个 symbol 的 0/1 信号，然后调用 backtester 执行（含 ATR 风控 + 仓位）
    这里为了“提高年化”，默认把仓位风险和止盈目标调得更进攻：
      - risk_per_trade: 2%
      - TP: 9 * ATR (让趋势利润奔跑)
      - Stop: 2.5 * ATR (略收紧，提高资金周转)
    """
    signals: Dict[str, pd.Series] = {}
    for sym, df in spot_by_symbol.items():
        if "date" not in df.columns:
            raise ValueError(f"{sym}: missing date column")
        signals[sym] = _make_signals(df, cfg)

    return backtest_spot_strategy(
        spot_by_symbol=spot_by_symbol,
        signals_by_symbol=signals,
        initial_capital=initial_capital,
        fee_rate=0.0004,
        slippage_bps=5.0,
        risk_per_trade=0.02,     # ✅ 关键：放大年化（同时也会放大波动）
        atr_window=cfg.atr_window,
        atr_mult_stop=2.5,       # 更紧 stop，提高周转（风险：更容易被洗）
        atr_mult_tp=9.0,         # 更远 TP，吃趋势（风险：触发更少）
        use_trend_filter=False,  # ✅ 注意：趋势过滤已在信号里做了，这里关闭避免“双重过滤”
        ema_trend_window=2000,    #
    )


__all__ = ["SRConfig", "run_sr_price_action"]