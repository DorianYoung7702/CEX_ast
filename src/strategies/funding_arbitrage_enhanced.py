# -*- coding: utf-8 -*-
"""
funding_arbitrage_enhanced.py
-----------------------------
1c. Enhanced Funding Arbitrage Strategy

Design goal:
- Keep most of the time IN the delta-neutral carry trade (otherwise cannot beat baseline)
- Exit ONLY under "bad carry" OR "basis blowout risk" regimes (explicit dynamic enter/exit)
- Optional Top-K (capital concentration) with anti-churn "switch band" to avoid over-trading

Key fixes vs previous version:
1) Properly USE fr_slow_span + enter/exit confirmation days (previously ignored).
2) Fix basis parameter semantics:
   - main.py users often pass basis_enter_z < 0 to mean "do NOT block entry".
     We now interpret basis_enter_z <= 0 as disabling the entry basis filter.
   - basis_exit_z < 0 means "exit only on deep backwardation"; basis_exit_z > 0 means blowout exit.
3) Add hysteresis + anti-churn: switch only if score improves by a margin (switch_threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..backtester import run_engine_delta_neutral_pairs, BacktestResult


def _common_dates(merged_by_symbol: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    syms = sorted(merged_by_symbol.keys())
    if not syms:
        return pd.DatetimeIndex([])
    common = pd.DatetimeIndex(pd.to_datetime(merged_by_symbol[syms[0]]["date"]).dt.normalize())
    for s in syms[1:]:
        common = common.intersection(pd.DatetimeIndex(pd.to_datetime(merged_by_symbol[s]["date"]).dt.normalize()))
    return common.sort_values()


def _basis_open(df: pd.DataFrame) -> pd.Series:
    # execution-consistent: perp_open - spot_open
    if {"perp_open", "spot_open"}.issubset(df.columns):
        return pd.to_numeric(df["perp_open"], errors="coerce") - pd.to_numeric(df["spot_open"], errors="coerce")
    if {"perp_close", "spot_close"}.issubset(df.columns):
        return pd.to_numeric(df["perp_close"], errors="coerce") - pd.to_numeric(df["spot_close"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


@dataclass
class EnhancedConfig:
    # ----- basis stats (Bollinger/Z) -----
    lookback: int = 60
    min_periods: int = 20
    slope_window: int = 3

    # ENTRY basis filter (block entry only when z is too high / blowout-ish)
    # If you want "basis should not block entry", set basis_enter_z<=0 in run_enhanced kwargs (mapped to +inf).
    entry_z_max: float = 3.0

    # EXIT basis filters (either blowout OR backwardation)
    blowout_z: float = 2.5         # exit if z >= blowout_z AND slope>0 (blowout widening)
    backward_z: float = -1.5       # exit if z <= backward_z (deep backwardation)

    # ----- funding regime (carry) -----
    fr_fast_span: int = 3          # faster to re-enter
    fr_slow_span: int = 20         # smoother regime anchor
    fr_entry: float = 0.0          # require carry not bad to enter/hold
    fr_exit: float = -0.0006       # force exit threshold (more negative => fewer exits)

    # confirmations (anti-noise)
    enter_confirm_days: int = 1
    exit_confirm_days: int = 2

    # ----- execution control -----
    top_k: int = 1
    min_hold_days: int = 3
    cooldown_days: int = 0
    warmup_allow_trade: bool = True

    # anti-churn switching: only switch if new score better by margin
    switch_threshold: float = 0.10   # relative margin, e.g. 0.10 = 10%

    # scoring weights for Top-K selection
    score_fr_weight: float = 15000.0
    score_z_weight: float = 0.10
    score_slope_weight: float = 0.10

    # fallback: if all off but some positive carry exists, keep one on
    require_at_least_one_if_positive: bool = True


def build_enhanced_hold_signals(
    merged_by_symbol: Dict[str, pd.DataFrame],
    cfg: EnhancedConfig,
    verbose: bool = False,
) -> Dict[str, pd.Series]:
    syms = sorted(merged_by_symbol.keys())
    if not syms:
        raise ValueError("merged_by_symbol empty")

    common = _common_dates(merged_by_symbol)
    if len(common) < 30:
        raise ValueError("Common date range too short")

    # features
    z_map: Dict[str, np.ndarray] = {}
    slope_map: Dict[str, np.ndarray] = {}
    fr_fast_map: Dict[str, np.ndarray] = {}
    fr_slow_map: Dict[str, np.ndarray] = {}

    for s in syms:
        df0 = merged_by_symbol[s].copy()
        df0["date"] = pd.to_datetime(df0["date"]).dt.normalize()
        df = df0.set_index("date").reindex(common).reset_index()

        basis = _basis_open(df).astype(float)
        fr = pd.to_numeric(df.get("funding_rate_daily", 0.0), errors="coerce").fillna(0.0).astype(float)

        fr_fast = fr.ewm(span=cfg.fr_fast_span, adjust=False).mean()
        fr_slow = fr.ewm(span=cfg.fr_slow_span, adjust=False).mean()

        mu = basis.rolling(cfg.lookback, min_periods=cfg.min_periods).mean()
        sd = basis.rolling(cfg.lookback, min_periods=cfg.min_periods).std(ddof=1).replace(0, np.nan)
        z = (basis - mu) / sd

        slope = basis - basis.shift(cfg.slope_window)

        z_map[s] = z.values
        slope_map[s] = slope.values
        fr_fast_map[s] = fr_fast.values
        fr_slow_map[s] = fr_slow.values

    # state machine
    in_pos = {s: False for s in syms}
    hold_days = {s: 0 for s in syms}
    cooldown = {s: 0 for s in syms}

    # confirmation counters
    enter_good = {s: 0 for s in syms}
    exit_bad = {s: 0 for s in syms}
    exit_blowout = {s: 0 for s in syms}
    exit_backward = {s: 0 for s in syms}

    holds = {s: np.zeros(len(common), dtype=bool) for s in syms}

    def _score(sym: str, i: int) -> float:
        frf = float(fr_fast_map[sym][i]) if np.isfinite(fr_fast_map[sym][i]) else 0.0
        frs = float(fr_slow_map[sym][i]) if np.isfinite(fr_slow_map[sym][i]) else 0.0
        z = z_map[sym][i]
        sl = slope_map[sym][i]

        # prefer higher carry and improving regime (fast - slow)
        regime = frf - frs
        z_val = 0.0 if np.isnan(z) else float(z)
        sl_pen = 0.0 if np.isnan(sl) else max(0.0, float(sl))

        return (cfg.score_fr_weight * frf) + (5000.0 * regime) - (cfg.score_z_weight * max(0.0, z_val)) - (cfg.score_slope_weight * sl_pen)

    def _entry_allowed(sym: str, i: int) -> bool:
        frf = float(fr_fast_map[sym][i]) if np.isfinite(fr_fast_map[sym][i]) else 0.0
        frs = float(fr_slow_map[sym][i]) if np.isfinite(fr_slow_map[sym][i]) else 0.0
        z = z_map[sym][i]
        sl = slope_map[sym][i]

        # funding gate: require not-bad carry
        if frf < cfg.fr_entry and frs < cfg.fr_entry:
            return False

        # basis entry filter: block only when z is very high
        if np.isnan(z):
            if not cfg.warmup_allow_trade:
                return False
        else:
            if z > cfg.entry_z_max:
                return False

        # if already blowout and widening, do not enter
        if (not np.isnan(z)) and (z >= cfg.blowout_z) and (not np.isnan(sl)) and (sl > 0):
            return False

        return True

    def _exit_conditions(sym: str, i: int) -> Tuple[bool, bool, bool]:
        """return (bad_carry, blowout, backward) for this day"""
        frf = float(fr_fast_map[sym][i]) if np.isfinite(fr_fast_map[sym][i]) else 0.0
        frs = float(fr_slow_map[sym][i]) if np.isfinite(fr_slow_map[sym][i]) else 0.0
        z = z_map[sym][i]
        sl = slope_map[sym][i]

        # Bad carry: fast < exit AND also below slow (regime deterioration)
        bad_carry = (frf <= cfg.fr_exit) and (frf < frs)

        blowout = False
        if (not np.isnan(z)) and (z >= cfg.blowout_z):
            # require widening to avoid exiting on stable high basis
            if (not np.isnan(sl)) and (sl > 0):
                blowout = True

        backward = (not np.isnan(z)) and (z <= cfg.backward_z)
        return bad_carry, blowout, backward

    # Main loop
    for i in range(len(common)):
        # tick cooldown
        for s in syms:
            if cooldown[s] > 0:
                cooldown[s] -= 1

        # ---- EXIT step (dynamic exiting rule with confirmation) ----
        for s in syms:
            if not in_pos[s]:
                continue

            # only allow exits after min_hold_days to reduce churn
            if hold_days[s] < cfg.min_hold_days:
                continue

            bad_carry, blowout, backward = _exit_conditions(s, i)

            exit_bad[s] = exit_bad[s] + 1 if bad_carry else 0
            exit_blowout[s] = exit_blowout[s] + 1 if blowout else 0
            exit_backward[s] = exit_backward[s] + 1 if backward else 0

            if (
                exit_bad[s] >= cfg.exit_confirm_days
                or exit_blowout[s] >= cfg.exit_confirm_days
                or exit_backward[s] >= cfg.exit_confirm_days
            ):
                in_pos[s] = False
                hold_days[s] = 0
                cooldown[s] = cfg.cooldown_days
                exit_bad[s] = exit_blowout[s] = exit_backward[s] = 0
                enter_good[s] = 0

        # ---- ENTRY eligibility (dynamic entering rule with confirmation) ----
        eligible: List[str] = []
        positive_carry: List[Tuple[str, float]] = []

        for s in syms:
            if cooldown[s] > 0:
                enter_good[s] = 0
                continue

            frf = float(fr_fast_map[s][i]) if np.isfinite(fr_fast_map[s][i]) else 0.0
            if frf > 0:
                positive_carry.append((s, frf))

            ok = _entry_allowed(s, i)
            enter_good[s] = enter_good[s] + 1 if ok else 0

            if enter_good[s] >= cfg.enter_confirm_days:
                eligible.append(s)

        # candidates = current holdings + eligible
        candidates = set([s for s in syms if in_pos[s]]).union(set(eligible))
        cand_list = list(candidates)

        # fallback: don't stay fully out if there is positive carry somewhere
        if (len(cand_list) == 0) and cfg.require_at_least_one_if_positive and positive_carry:
            best = sorted(positive_carry, key=lambda x: x[1], reverse=True)[0][0]
            cand_list = [best]

        # rank by score
        cand_list.sort(key=lambda s: _score(s, i), reverse=True)

        # pick Top-K with anti-churn (stickiness + switch_threshold)
        picked: List[str] = []
        current = [s for s in syms if in_pos[s]]

        # keep current first
        for s in current:
            if s in cand_list and len(picked) < cfg.top_k:
                picked.append(s)

        # fill remaining slots
        for s in cand_list:
            if len(picked) >= cfg.top_k:
                break
            if s in picked:
                continue
            picked.append(s)

        # anti-churn for top_k=1: only switch if new is sufficiently better
        if cfg.top_k == 1 and len(picked) == 1 and len(current) == 1:
            cur = current[0]
            new = picked[0]
            if cur != new:
                cur_score = _score(cur, i)
                new_score = _score(new, i)
                # relative improvement required
                denom = max(1e-12, abs(cur_score))
                if (new_score - cur_score) / denom < cfg.switch_threshold:
                    picked = [cur]  # keep current

        picked_set = set(picked)

        # turn off those not picked
        for s in syms:
            if in_pos[s] and (s not in picked_set):
                in_pos[s] = False
                hold_days[s] = 0
                cooldown[s] = cfg.cooldown_days
                exit_bad[s] = exit_blowout[s] = exit_backward[s] = 0
                enter_good[s] = 0

        # apply picked
        for s in syms:
            if s in picked_set:
                if not in_pos[s]:
                    in_pos[s] = True
                    hold_days[s] = 1
                else:
                    hold_days[s] += 1
                holds[s][i] = True
            else:
                holds[s][i] = False

    return {s: pd.Series(holds[s], index=common).astype(bool) for s in syms}


def backtest_funding_arbitrage_enhanced(
    merged_by_symbol: Dict[str, pd.DataFrame],
    cfg: EnhancedConfig = EnhancedConfig(),
    initial_capital: float = 1_000_000.0,
    fee_spot: float = 0.0004,
    fee_perp: float = 0.0004,
    slippage_bps: float = 2.0,
    init_margin_ratio: float = 0.10,
    maint_margin_ratio: float = 0.05,
    verbose: bool = False,
) -> BacktestResult:
    normalized = {}
    for sym, df in merged_by_symbol.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        normalized[sym] = d

    sig = build_enhanced_hold_signals(normalized, cfg=cfg, verbose=verbose)

    return run_engine_delta_neutral_pairs(
        merged_by_symbol=normalized,
        hold_signal_by_symbol=sig,
        initial_capital=initial_capital,
        fee_spot=fee_spot,
        fee_perp=fee_perp,
        slippage_bps=slippage_bps,
        rebalance_daily=True,
        init_margin_ratio=init_margin_ratio,
        maint_margin_ratio=maint_margin_ratio,
    )


# Backward-compatible wrapper for your existing main.py
def run_enhanced(
    merged_by_symbol: Dict[str, pd.DataFrame],
    lookback: int = 60,
    entry_z: float = 1.0,      # legacy
    exit_z: float = 0.2,       # legacy
    min_funding: float = 0.0,
    exit_funding: float = -0.0004,
    fr_fast_span: int = 3,
    fr_slow_span: int = 20,
    top_k: int = 1,
    cooldown_days: int = 0,
    min_hold_days: int = 3,
    enter_confirm_days: int = 1,
    exit_confirm_days: int = 2,
    initial_capital: float = 1_000_000.0,
    fee_spot: float = 0.0004,
    fee_perp: float = 0.0004,
    slippage_bps: float = 2.0,
    init_margin_ratio: float = 0.10,
    maint_margin_ratio: float = 0.05,
    verbose: bool = False,
    # additional user-friendly basis params
    basis_enter_z: Optional[float] = None,
    basis_exit_z: Optional[float] = None,
    switch_threshold: Optional[float] = None,
    **kwargs,
) -> BacktestResult:
    """
    Compatibility notes:
    - basis_enter_z <= 0 => DISABLE entry basis filter (set entry_z_max=+inf)
    - basis_exit_z < 0  => treat as backwardation exit threshold, disable blowout exit
    - basis_exit_z > 0  => treat as blowout_z exit threshold
    """

    cfg = EnhancedConfig(
        lookback=int(lookback),
        fr_fast_span=int(fr_fast_span),
        fr_slow_span=int(fr_slow_span),
        fr_entry=float(min_funding),
        fr_exit=float(exit_funding),
        top_k=int(top_k),
        cooldown_days=int(cooldown_days),
        min_hold_days=int(min_hold_days),
        enter_confirm_days=int(enter_confirm_days),
        exit_confirm_days=int(exit_confirm_days),
        # keep legacy mapping as fallback
        entry_z_max=float(max(1.5, entry_z)),
        backward_z=float(min(-0.8, -abs(exit_z))),
    )

    # user-friendly basis overrides
    if basis_enter_z is not None:
        be = float(basis_enter_z)
        if be <= 0:
            cfg.entry_z_max = float("inf")  # disable entry basis gate
        else:
            cfg.entry_z_max = be

    if basis_exit_z is not None:
        bx = float(basis_exit_z)
        if bx < 0:
            cfg.backward_z = bx
            cfg.blowout_z = float("inf")   # disable blowout exit
        else:
            cfg.blowout_z = bx

    if switch_threshold is not None:
        cfg.switch_threshold = float(switch_threshold)

    # allow explicit overrides via kwargs (optional)
    for k in [
        "blowout_z",
        "entry_z_max",
        "backward_z",
        "require_at_least_one_if_positive",
        "warmup_allow_trade",
        "score_fr_weight",
        "score_z_weight",
        "score_slope_weight",
    ]:
        if k in kwargs:
            setattr(cfg, k, kwargs[k])

    return backtest_funding_arbitrage_enhanced(
        merged_by_symbol=merged_by_symbol,
        cfg=cfg,
        initial_capital=initial_capital,
        fee_spot=fee_spot,
        fee_perp=fee_perp,
        slippage_bps=slippage_bps,
        init_margin_ratio=init_margin_ratio,
        maint_margin_ratio=maint_margin_ratio,
        verbose=verbose,
    )


__all__ = [
    "EnhancedConfig",
    "build_enhanced_hold_signals",
    "backtest_funding_arbitrage_enhanced",
    "run_enhanced",
]