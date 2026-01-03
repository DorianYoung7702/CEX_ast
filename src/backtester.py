# backtester.py
# -*- coding: utf-8 -*-
"""
Generic execution engine ONLY (no strategy logic, no arbitrage strategy functions).

It executes "delta-neutral spot+perp pairs" given boolean hold signals per asset:
- signal at t close, execute at t+1 open
- open pair when signal True: long spot + short perp of equal USDT notional
- close pair when signal False: flatten both legs
- rebalance_daily=True => keeps equal USDT notional continuously

Constraints satisfied by construction:
- at most one open pair per asset (position is ON/OFF)
- market-neutral: each ON pair has equal notional long spot and short perp

PnL:
- open->open
- funding: funding_rate_daily * perp_notional (daily approx)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional

import numpy as np
import pandas as pd

from .metrics import summarize_performance

Side = Literal["BUY", "SELL"]
Action = Literal["OPEN", "CLOSE", "REBALANCE"]


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    perf: dict
    extra: dict

    def save(self, out_dir: str = "output", prefix: str = "result") -> Dict[str, str]:
        import os, json
        os.makedirs(out_dir, exist_ok=True)
        eq_path = os.path.join(out_dir, f"{prefix}_equity.csv")
        tr_path = os.path.join(out_dir, f"{prefix}_trades.csv")
        pf_path = os.path.join(out_dir, f"{prefix}_perf.json")
        self.equity_curve.to_csv(eq_path, index=False)
        self.trade_log.to_csv(tr_path, index=False)
        with open(pf_path, "w", encoding="utf-8") as f:
            json.dump(self.perf, f, ensure_ascii=False, indent=2, allow_nan=True)
        return {"equity": eq_path, "trades": tr_path, "perf": pf_path}


def _apply_slippage(price: float, side: Side, slippage_bps: float) -> float:
    s = float(slippage_bps) / 10_000.0
    return float(price) * (1 + s) if side == "BUY" else float(price) * (1 - s)


def _fee(notional_abs: float, fee_rate: float) -> float:
    return float(abs(notional_abs)) * float(fee_rate)


def _make_equal_weight_benchmark_open(
    aligned_by_symbol: Dict[str, pd.DataFrame],
    dates: pd.Index,
    price_col: str = "spot_open",
    initial_capital: float = 1_000_000.0,
) -> pd.Series:
    dates = pd.to_datetime(pd.Index(dates))
    px_list = []
    for s, df in aligned_by_symbol.items():
        if price_col not in df.columns:
            continue
        ser = pd.to_numeric(df.set_index("date")[price_col], errors="coerce")
        ser.index = pd.to_datetime(ser.index)
        ser = ser.reindex(dates).ffill().bfill()
        if ser.isna().all():
            continue
        px_list.append(ser.rename(s))

    if not px_list:
        return pd.Series(index=dates, dtype=float)

    px = pd.concat(px_list, axis=1).dropna(axis=1, how="any")
    if px.empty:
        return pd.Series(index=dates, dtype=float)

    first = px.index[0]
    alloc = float(initial_capital) / px.shape[1]
    qty = {c: alloc / float(px.loc[first, c]) for c in px.columns}

    value = pd.Series(index=dates, dtype=float)
    for dt in dates:
        value.loc[dt] = sum(qty[c] * float(px.loc[dt, c]) for c in px.columns)
    return value


def run_engine_delta_neutral_pairs(
    merged_by_symbol: Dict[str, pd.DataFrame],
    hold_signal_by_symbol: Dict[str, pd.Series],   # bool series indexed by date (signal at t close)
    initial_capital: float = 1_000_000.0,
    fee_spot: float = 0.0004,
    fee_perp: float = 0.0004,
    slippage_bps: float = 2.0,
    rebalance_daily: bool = True,                 # ✅ continuous equal-notional maintenance
    init_margin_ratio: float = 0.10,
    maint_margin_ratio: float = 0.05,
) -> BacktestResult:
    syms = sorted(merged_by_symbol.keys())
    if not syms:
        raise ValueError("merged_by_symbol empty")

    # date intersection
    common = pd.Index(pd.to_datetime(merged_by_symbol[syms[0]]["date"]))
    for s in syms[1:]:
        common = common.intersection(pd.Index(pd.to_datetime(merged_by_symbol[s]["date"])))
    common = common.sort_values()
    if len(common) < 30:
        raise ValueError("Common date range too short")

    # align data
    aligned: Dict[str, pd.DataFrame] = {}
    need_cols = {"spot_open", "perp_open", "funding_rate_daily"}
    for s in syms:
        df = merged_by_symbol[s].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(common).reset_index()
        missing = need_cols - set(df.columns)
        if missing:
            raise ValueError(f"{s} missing columns: {sorted(missing)}")
        for c in ["spot_open", "perp_open", "funding_rate_daily"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        aligned[s] = df

    # align signals
    sig_aligned: Dict[str, pd.Series] = {}
    for s in syms:
        sig = hold_signal_by_symbol.get(s)
        if sig is None:
            sig = pd.Series(False, index=common)
        sig = sig.copy()
        sig.index = pd.to_datetime(sig.index)
        sig_aligned[s] = sig.reindex(common).fillna(False).astype(bool)

    cash = float(initial_capital)
    margin = 0.0
    spot_qty = {s: 0.0 for s in syms}
    perp_qty = {s: 0.0 for s in syms}  # short qty

    trade_rows: List[Dict[str, Any]] = []
    eq_rows: List[Dict[str, Any]] = []

    def _topup_margin(x: float):
        nonlocal cash, margin
        if x <= 0:
            return
        y = min(cash, x)
        cash -= y
        margin += y

    def _ensure_init_margin(perp_price: float, tgt_perp_qty: float):
        req = init_margin_ratio * max(0.0, float(tgt_perp_qty) * float(perp_price))
        if margin < req:
            _topup_margin(req - margin)

    def _retarget(
        s: str, d_exec, d_sig,
        spot_price: float, perp_price: float,
        tgt_spot_qty: float, tgt_perp_qty: float,
        action: Action
    ):
        nonlocal cash, margin
        cur_sp, cur_pp = float(spot_qty[s]), float(perp_qty[s])
        dq_sp, dq_pp = float(tgt_spot_qty - cur_sp), float(tgt_perp_qty - cur_pp)

        if abs(dq_sp) < 1e-12 and abs(dq_pp) < 1e-12:
            return

        _ensure_init_margin(perp_price, tgt_perp_qty)

        # spot leg with cash constraint -> scale both legs
        if dq_sp > 0:
            px = _apply_slippage(spot_price, "BUY", slippage_bps)
            cost = dq_sp * px
            fee = _fee(cost, fee_spot)
            need = cost + fee
            if need > cash + 1e-12:
                scale = (cash / need) if need > 0 else 0.0
                dq_sp *= scale
                dq_pp *= scale
                tgt_spot_qty = cur_sp + dq_sp
                tgt_perp_qty = cur_pp + dq_pp
                cost = dq_sp * px
                fee = _fee(cost, fee_spot)

            cash -= cost
            cash -= fee
            trade_rows.append({
                "date": d_exec, "signal_date": d_sig, "symbol": s,
                "leg": "spot", "action": action, "side": "BUY",
                "qty": float(dq_sp), "price": float(px),
                "trade_value": float(cost), "fee": float(fee),
            })

        elif dq_sp < 0:
            qty_sell = -dq_sp
            px = _apply_slippage(spot_price, "SELL", slippage_bps)
            value = qty_sell * px
            fee = _fee(value, fee_spot)
            cash += value
            cash -= fee
            trade_rows.append({
                "date": d_exec, "signal_date": d_sig, "symbol": s,
                "leg": "spot", "action": action, "side": "SELL",
                "qty": float(qty_sell), "price": float(px),
                "trade_value": float(value), "fee": float(fee),
            })

        cur_sp = float(tgt_spot_qty)

        # perp leg fee from margin; topup if needed
        dq_pp = float(tgt_perp_qty - cur_pp)
        if dq_pp > 0:
            px = _apply_slippage(perp_price, "SELL", slippage_bps)
            notional_abs = dq_pp * px
            fee = _fee(notional_abs, fee_perp)
            if fee > margin:
                _topup_margin(fee - margin)
            fee = min(fee, margin)
            margin -= fee
            trade_rows.append({
                "date": d_exec, "signal_date": d_sig, "symbol": s,
                "leg": "perp", "action": action, "side": "SELL",
                "qty": float(dq_pp), "price": float(px),
                "trade_value": float(abs(notional_abs)), "fee": float(fee),
            })
            cur_pp = float(tgt_perp_qty)

        elif dq_pp < 0:
            qty_buy = -dq_pp
            px = _apply_slippage(perp_price, "BUY", slippage_bps)
            notional_abs = qty_buy * px
            fee = _fee(notional_abs, fee_perp)
            if fee > margin:
                _topup_margin(fee - margin)
            fee = min(fee, margin)
            margin -= fee
            trade_rows.append({
                "date": d_exec, "signal_date": d_sig, "symbol": s,
                "leg": "perp", "action": action, "side": "BUY",
                "qty": float(qty_buy), "price": float(px),
                "trade_value": float(abs(notional_abs)), "fee": float(fee),
            })
            cur_pp = float(tgt_perp_qty)

        spot_qty[s], perp_qty[s] = float(cur_sp), float(cur_pp)

    # initial row
    eq_rows.append({
        "date": common[0],
        "equity": float(initial_capital),
        "cash": float(cash),
        "margin": float(margin),
        "pnl_spot": 0.0,
        "pnl_perp": 0.0,
        "pnl_funding": 0.0,
        "pnl_total": 0.0,
        "n_active": 0,
    })

    for i in range(1, len(common) - 1):
        d_sig = common[i - 1]
        d_exec = common[i]
        d_next = common[i + 1]

        spot_o = {s: float(aligned[s].iloc[i]["spot_open"]) for s in syms}
        perp_o = {s: float(aligned[s].iloc[i]["perp_open"]) for s in syms}
        spot_on = {s: float(aligned[s].iloc[i + 1]["spot_open"]) for s in syms}
        perp_on = {s: float(aligned[s].iloc[i + 1]["perp_open"]) for s in syms}

        want_hold = {s: bool(sig_aligned[s].loc[d_sig]) for s in syms}
        active = [s for s in syms if want_hold[s]]
        n_active = len(active)

        # ✅ FIX: 用“总权益”做 sizing（把 spot 市值算回 available）
        # 否则 cash 被买 spot 消耗后，available 变小 => target_notional 越变越小 => 收益看起来很小
        spot_value_exec = sum(
            spot_qty[s] * spot_o[s]
            for s in syms
            if np.isfinite(spot_o[s])
        )
        available = max(0.0, cash + margin + float(spot_value_exec))  # ✅ FIX

        if n_active > 0:
            denom = (1.0 + init_margin_ratio + fee_spot + fee_perp)
            target_notional = (available / n_active) / denom
            target_notional = max(0.0, float(target_notional))
        else:
            target_notional = 0.0

        for s in syms:
            if not (np.isfinite(spot_o[s]) and np.isfinite(perp_o[s])):
                continue

            on = (spot_qty[s] != 0.0) or (perp_qty[s] != 0.0)
            want_on = want_hold[s]

            if (not rebalance_daily) and on and want_on:
                continue

            tgt_notional = target_notional if want_on else 0.0
            tgt_sp = (tgt_notional / spot_o[s]) if tgt_notional > 0 else 0.0
            tgt_pp = (tgt_notional / perp_o[s]) if tgt_notional > 0 else 0.0

            if (not on) and want_on:
                action: Action = "OPEN"
            elif on and (not want_on):
                action = "CLOSE"
            else:
                action = "REBALANCE"

            _retarget(s, d_exec, d_sig, spot_o[s], perp_o[s], tgt_sp, tgt_pp, action)

        # pnl open->open
        pnl_spot_total = 0.0
        pnl_perp_total = 0.0
        pnl_funding_total = 0.0

        for s in syms:
            if not (np.isfinite(spot_o[s]) and np.isfinite(perp_o[s]) and np.isfinite(spot_on[s]) and np.isfinite(perp_on[s])):
                continue
            pnl_spot = spot_qty[s] * (spot_on[s] - spot_o[s])
            pnl_perp = perp_qty[s] * (perp_o[s] - perp_on[s])
            fr = float(aligned[s].iloc[i].get("funding_rate_daily", 0.0) or 0.0)
            perp_notional = perp_qty[s] * perp_o[s]
            pnl_funding = fr * perp_notional

            pnl_spot_total += pnl_spot
            pnl_perp_total += pnl_perp
            pnl_funding_total += pnl_funding

        # perp+funding into margin (spot is reflected in spot_value mark-to-market)
        margin += float(pnl_perp_total + pnl_funding_total)

        # maintenance margin check (soft)
        total_perp_notional = sum(perp_qty[s] * perp_o[s] for s in syms if np.isfinite(perp_o[s]))
        maint_req = maint_margin_ratio * total_perp_notional if total_perp_notional > 0 else 0.0
        if maint_req > 0 and margin < maint_req:
            _topup_margin(maint_req - margin)

        spot_value_next = sum(spot_qty[s] * spot_on[s] for s in syms if np.isfinite(spot_on[s]))
        equity_next = cash + margin + spot_value_next

        eq_rows.append({
            "date": d_next,
            "equity": float(equity_next),
            "cash": float(cash),
            "margin": float(margin),
            "pnl_spot": float(pnl_spot_total),
            "pnl_perp": float(pnl_perp_total),
            "pnl_funding": float(pnl_funding_total),
            "pnl_total": float(pnl_spot_total + pnl_perp_total + pnl_funding_total),
            "n_active": int(n_active),
        })

        if cash < 0:
            cash = 0.0

    equity_curve = pd.DataFrame(eq_rows)
    trade_log = pd.DataFrame(trade_rows)

    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    equity_curve = (
        equity_curve.set_index("date")
        .reindex(pd.to_datetime(common))
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    bench = _make_equal_weight_benchmark_open(
        aligned_by_symbol=aligned,
        dates=pd.Index(equity_curve["date"]),
        price_col="spot_open",
        initial_capital=initial_capital,
    )

    perf_obj = summarize_performance(
        equity_curve=equity_curve,
        equity_col="equity",
        date_col="date",
        benchmark_price=bench,
        alpha_mode="cagr_diff",
    )
    perf = perf_obj.__dict__

    extra = {
        "final_equity": float(equity_curve["equity"].iloc[-1]),
        "cumulative_return": float(equity_curve["equity"].iloc[-1] / initial_capital - 1.0),
        "total_trades": int(len(trade_log)),
        "total_fees": float(trade_log["fee"].sum()) if (not trade_log.empty and "fee" in trade_log.columns) else 0.0,
    }
    return BacktestResult(equity_curve=equity_curve, trade_log=trade_log, perf=perf, extra=extra)


# -------------------------
# SPOT STRATEGY EXECUTION ENGINE (GENERIC)
# -------------------------
def backtest_spot_strategy(
    spot_by_symbol: Dict[str, pd.DataFrame],
    signals_by_symbol: Dict[str, pd.Series],
    initial_capital: float = 1_000_000.0,
    fee_rate: float = 0.0004,
    slippage_bps: float = 5.0,
    risk_per_trade: float = 0.01,
    atr_window: int = 14,
    atr_mult_stop: float = 3.0,
    atr_mult_tp: float = 6.0,
    use_trend_filter: bool = True,
    ema_trend_window: int = 200,
) -> BacktestResult:
    """
    Generic spot execution engine used by SR strategy.
    - signal at t close, execute at t+1 open
    - open->open PnL
    - ATR stop/take-profit
    - optional EMA trend filter
    - no leverage: position size constrained by cash
    """
    syms = sorted(spot_by_symbol.keys())
    if not syms:
        raise ValueError("spot_by_symbol empty")

    def _norm_dates(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "date" not in d.columns:
            raise ValueError("spot df missing 'date'")
        d["date"] = pd.to_datetime(d["date"])
        return d.sort_values("date").reset_index(drop=True)

    spot_by_symbol = {s: _norm_dates(df) for s, df in spot_by_symbol.items()}

    common = pd.Index(spot_by_symbol[syms[0]]["date"])
    for s in syms[1:]:
        common = common.intersection(pd.Index(spot_by_symbol[s]["date"]))
    common = common.sort_values()
    if len(common) < 60:
        raise ValueError("Common date range too short")

    aligned: Dict[str, pd.DataFrame] = {}
    for s in syms:
        df = spot_by_symbol[s].copy().set_index("date").reindex(common).reset_index()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # ATR (simple TR mean)
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(atr_window, min_periods=atr_window).mean()

        if use_trend_filter:
            df["ema_trend"] = df["close"].ewm(span=ema_trend_window, adjust=False).mean()

        aligned[s] = df

    sig_aligned: Dict[str, pd.Series] = {}
    for s in syms:
        sig = signals_by_symbol.get(s)
        if sig is None:
            sig = pd.Series(0, index=common)
        sig = sig.copy()
        sig.index = pd.to_datetime(sig.index)
        sig_aligned[s] = sig.reindex(common).fillna(0).astype(int).clip(0, 1)

    cash = float(initial_capital)
    qty = {s: 0.0 for s in syms}
    stop_price = {s: np.nan for s in syms}
    tp_price = {s: np.nan for s in syms}

    trade_rows: List[Dict[str, Any]] = []
    eq_rows: List[Dict[str, Any]] = []

    eq_rows.append({
        "date": common[0],
        "equity": float(initial_capital),
        "cash": float(cash),
        "pnl_spot": 0.0,
        "pnl_fees": 0.0,
        "pnl_total": 0.0,
        "n_positions": 0,
    })

    for i in range(1, len(common) - 1):
        d_sig = common[i - 1]
        d_exec = common[i]
        d_next = common[i + 1]

        # equity valued at exec open
        spot_value = 0.0
        for s in syms:
            o = aligned[s].iloc[i]["open"]
            if not pd.isna(o):
                spot_value += qty[s] * float(o)
        equity = cash + spot_value

        fees_paid = 0.0
        pnl_spot = 0.0

        for s in syms:
            row = aligned[s].iloc[i]
            row_next = aligned[s].iloc[i + 1]
            o = row["open"]
            o_next = row_next["open"]
            if pd.isna(o) or pd.isna(o_next):
                continue

            want = int(sig_aligned[s].loc[d_sig])

            if use_trend_filter:
                close_sig = aligned[s].iloc[i - 1]["close"]
                ema_sig = aligned[s].iloc[i - 1].get("ema_trend", np.nan)
                if want == 1 and (pd.isna(close_sig) or pd.isna(ema_sig) or close_sig <= ema_sig):
                    want = 0

            # risk exits at exec open
            if qty[s] > 0:
                if (not np.isnan(stop_price[s]) and float(o) <= float(stop_price[s])) or \
                   (not np.isnan(tp_price[s]) and float(o) >= float(tp_price[s])):
                    want = 0

            # OPEN
            if want == 1 and qty[s] == 0:
                atr = row.get("atr", np.nan)
                if pd.isna(atr) or float(atr) <= 1e-12:
                    continue

                stop_dist = atr_mult_stop * float(atr)
                risk_budget = equity * float(risk_per_trade)
                position_value = risk_budget * (float(o) / stop_dist)

                position_value = min(position_value, equity)
                est_fee = _fee(position_value, fee_rate)
                if position_value + est_fee > cash:
                    position_value = max(0.0, cash / (1.0 + fee_rate))
                    est_fee = _fee(position_value, fee_rate)

                if position_value <= 0:
                    continue

                px = _apply_slippage(float(o), "BUY", slippage_bps)
                tgt_qty = position_value / px

                fee = _fee(position_value, fee_rate)
                fees_paid += fee
                cash -= position_value
                cash -= fee

                qty[s] = tgt_qty
                stop_price[s] = px - stop_dist
                tp_price[s] = px + atr_mult_tp * float(atr)

                trade_rows.append({
                    "date": d_exec,
                    "signal_date": d_sig,
                    "symbol": s,
                    "action": "OPEN",
                    "side": "BUY",
                    "qty": float(tgt_qty),
                    "price": float(px),
                    "trade_value": float(position_value),
                    "fee": float(fee),
                    "reason": "signal_open_with_atr_risk",
                })

            # CLOSE
            elif want == 0 and qty[s] > 0:
                px = _apply_slippage(float(o), "SELL", slippage_bps)
                value = qty[s] * px
                fee = _fee(value, fee_rate)
                fees_paid += fee
                cash += value
                cash -= fee

                trade_rows.append({
                    "date": d_exec,
                    "signal_date": d_sig,
                    "symbol": s,
                    "action": "CLOSE",
                    "side": "SELL",
                    "qty": float(qty[s]),
                    "price": float(px),
                    "trade_value": float(value),
                    "fee": float(fee),
                    "reason": "signal_or_risk_exit",
                })

                qty[s] = 0.0
                stop_price[s] = np.nan
                tp_price[s] = np.nan

            # open->open pnl
            pnl_spot += qty[s] * (float(o_next) - float(o))

        cash += pnl_spot

        spot_value_next = 0.0
        for s in syms:
            o_next = aligned[s].iloc[i + 1]["open"]
            if not pd.isna(o_next):
                spot_value_next += qty[s] * float(o_next)
        equity_next = cash + spot_value_next

        eq_rows.append({
            "date": d_next,
            "equity": float(equity_next),
            "cash": float(cash),
            "pnl_spot": float(pnl_spot),
            "pnl_fees": float(-fees_paid),
            "pnl_total": float(pnl_spot - fees_paid),
            "n_positions": int(sum(1 for s in syms if qty[s] > 0)),
        })

    equity_curve = pd.DataFrame(eq_rows)
    trade_log = pd.DataFrame(trade_rows)

    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    equity_curve = (
        equity_curve.set_index("date")
        .reindex(pd.to_datetime(common))
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # benchmark cash
    bench_cash = pd.Series(initial_capital, index=pd.to_datetime(equity_curve["date"]))
    perf_obj = summarize_performance(
        equity_curve=equity_curve,
        equity_col="equity",
        date_col="date",
        benchmark_price=bench_cash,
        alpha_mode="cagr_diff",
    )
    perf = perf_obj.__dict__

    extra = {
        "final_equity": float(equity_curve["equity"].iloc[-1]),
        "total_trades": int(len(trade_log)),
        "total_fees": float(trade_log["fee"].sum()) if (not trade_log.empty and "fee" in trade_log.columns) else 0.0,
    }
    return BacktestResult(equity_curve=equity_curve, trade_log=trade_log, perf=perf, extra=extra)


__all__ = ["BacktestResult", "run_engine_delta_neutral_pairs", "backtest_spot_strategy"]