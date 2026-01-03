# -*- coding: utf-8 -*-
"""
funding_arbitrage_baseline.py
-----------------------------
1b. Baseline Funding Rate Arbitrage Strategy (STRICT)

Requirement:
- continuously maintain delta-neutral position for EACH asset:
  long spot + short perpetual of equal USDT notional
- returns mainly from funding payments
- one open arbitrage position per asset at any time (ON/OFF)

Implementation:
- signal is ALWAYS True for each asset across all dates (continuous hold)
- execution/PNL/accounting handled by engine:
  src.backtester.run_engine_delta_neutral_pairs(..., rebalance_daily=True)
"""

from __future__ import annotations

from typing import Dict
import pandas as pd

from ..backtester import run_engine_delta_neutral_pairs, BacktestResult


def build_hold_signals_always_on(
    merged_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """
    Return boolean hold signals per asset:
    - index: date
    - value: True means hold the delta-neutral pair (spot long + perp short)
    """
    signals: Dict[str, pd.Series] = {}
    for sym, df in merged_by_symbol.items():
        if "date" not in df.columns:
            raise ValueError(f"{sym} merged_df missing 'date'")
        dates = pd.to_datetime(df["date"])
        sig = pd.Series(True, index=dates, dtype=bool)
        sig = sig[~sig.index.duplicated(keep="last")].sort_index()
        signals[sym] = sig
    return signals


def backtest_funding_arbitrage_baseline(
    merged_by_symbol: Dict[str, pd.DataFrame],
    initial_capital: float = 1_000_000.0,
    fee_spot: float = 0.0004,
    fee_perp: float = 0.0004,
    slippage_bps: float = 2.0,
    init_margin_ratio: float = 0.10,
    maint_margin_ratio: float = 0.05,
) -> BacktestResult:
    """
    Strategy entry/exit:
      - Always in position (continuous)
    Engine enforces:
      - equal USDT notional spot long and perp short
      - daily rebalance to maintain equal notional
      - market-neutral portfolio
    """
    signals = build_hold_signals_always_on(merged_by_symbol)

    return run_engine_delta_neutral_pairs(
        merged_by_symbol=merged_by_symbol,
        hold_signal_by_symbol=signals,
        initial_capital=initial_capital,
        fee_spot=fee_spot,
        fee_perp=fee_perp,
        slippage_bps=slippage_bps,
        rebalance_daily=True,  # ✅ continuous equal-notional maintenance
        init_margin_ratio=init_margin_ratio,
        maint_margin_ratio=maint_margin_ratio,
    )


# 兼容你旧 main 里用过的名字（如果你还想保留 run_baseline 调用）
def run_baseline(merged_by_symbol: Dict[str, pd.DataFrame], **kwargs) -> BacktestResult:
    return backtest_funding_arbitrage_baseline(merged_by_symbol, **kwargs)


__all__ = [
    "build_hold_signals_always_on",
    "backtest_funding_arbitrage_baseline",
    "run_baseline",
]