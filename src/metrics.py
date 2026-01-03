# -*- coding: utf-8 -*-
"""
metrics.py
----------
Performance metrics:
- annualized return
- Sharpe
- Max Drawdown
- alpha (vs benchmark)
- buy&hold CAGR

Fixes:
1) benchmark 对齐后允许 bfill，避免首日 NaN
2) CAGR 计算用 first_valid / last_valid，避免首尾 NaN 导致全链路 NaN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union

import numpy as np
import pandas as pd

AlphaMode = Literal["cagr_diff"]


@dataclass
class PerfSummary:
    annualized_return: float
    sharpe: float
    max_drawdown: float
    alpha: float
    buy_hold_cagr: float


def _to_datetime_index(x) -> pd.DatetimeIndex:
    idx = pd.to_datetime(pd.Index(x))
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    return pd.DatetimeIndex(idx)


def _infer_periods_per_year(dates: pd.Series | pd.Index) -> float:
    dt = pd.to_datetime(pd.Index(dates)).sort_values()
    if len(dt) < 3:
        return 252.0
    diffs = dt.to_series().diff().dropna()
    med_days = float(diffs.median() / np.timedelta64(1, "D"))
    if med_days <= 0:
        return 252.0
    return 365.25 / med_days


def equity_to_returns(
    equity_curve: pd.DataFrame,
    equity_col: str = "equity",
    date_col: str = "date",
) -> pd.Series:
    df = equity_curve[[date_col, equity_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce")
    df = df.sort_values(date_col)

    # 关键：先去掉 equity 的 NaN，否则 pct_change 会把 NaN 传播
    df = df.dropna(subset=[equity_col])
    if len(df) < 2:
        return pd.Series(dtype=float)

    rets = df[equity_col].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    rets.index = pd.to_datetime(df[date_col].iloc[1:]).values
    rets.name = "returns"
    return rets


def max_drawdown_from_equity(
    equity_curve: pd.DataFrame,
    equity_col: str = "equity",
) -> float:
    eq = pd.to_numeric(equity_curve[equity_col], errors="coerce").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 2:
        return float("nan")

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def cagr_from_equity(
    equity_curve: Union[pd.DataFrame, pd.Series],
    equity_col: str = "equity",
    date_col: str = "date",
) -> float:
    if isinstance(equity_curve, pd.Series):
        s = pd.to_numeric(equity_curve, errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.dropna()
        if len(s) < 2:
            return float("nan")
        first_idx = s.index[0]
        last_idx = s.index[-1]
        first_val = float(s.iloc[0])
        last_val = float(s.iloc[-1])
    else:
        df = equity_curve[[date_col, equity_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce").astype(float)
        df = df.sort_values(date_col)
        df[equity_col] = df[equity_col].replace([np.inf, -np.inf], np.nan)

        first_pos = df[equity_col].first_valid_index()
        last_pos = df[equity_col].last_valid_index()
        if first_pos is None or last_pos is None:
            return float("nan")
        if first_pos == last_pos:
            return float("nan")

        first_idx = df.loc[first_pos, date_col]
        last_idx = df.loc[last_pos, date_col]
        first_val = float(df.loc[first_pos, equity_col])
        last_val = float(df.loc[last_pos, equity_col])

    if not (np.isfinite(first_val) and np.isfinite(last_val)):
        return float("nan")
    if first_val <= 0 or last_val <= 0:
        return float("nan")

    years = (pd.to_datetime(last_idx) - pd.to_datetime(first_idx)).days / 365.25
    if years <= 0:
        return float("nan")

    return float((last_val / first_val) ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 3:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def _align_benchmark_to_dates(benchmark: pd.Series, dates: pd.Index) -> pd.Series:
    b = benchmark.copy()
    b.index = pd.to_datetime(b.index)
    b = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # 关键：ffill + bfill，保证首日也有值（否则 CAGR 会 NaN）
    b = b.reindex(pd.to_datetime(dates)).ffill().bfill()
    return b


def summarize_performance(
    equity_curve: pd.DataFrame,
    equity_col: str = "equity",
    date_col: str = "date",
    benchmark_price: Optional[pd.Series] = None,
    alpha_mode: AlphaMode = "cagr_diff",
) -> PerfSummary:
    df = equity_curve[[date_col, equity_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce").astype(float)
    df = df.sort_values(date_col)

    periods_per_year = _infer_periods_per_year(df[date_col])

    rets = equity_to_returns(df, equity_col=equity_col, date_col=date_col)
    ann_ret = cagr_from_equity(df, equity_col=equity_col, date_col=date_col)
    sharpe = sharpe_ratio(rets, periods_per_year)
    mdd = max_drawdown_from_equity(df, equity_col=equity_col)

    alpha = float("nan")
    buy_hold_cagr = float("nan")

    if benchmark_price is not None:
        bench = _align_benchmark_to_dates(benchmark_price, pd.Index(df[date_col]))
        buy_hold_cagr = cagr_from_equity(bench)  # bench 本身是 value 序列
        if alpha_mode == "cagr_diff" and np.isfinite(ann_ret) and np.isfinite(buy_hold_cagr):
            alpha = float(ann_ret - buy_hold_cagr)

    return PerfSummary(
        annualized_return=float(ann_ret),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        alpha=float(alpha),
        buy_hold_cagr=float(buy_hold_cagr),
    )