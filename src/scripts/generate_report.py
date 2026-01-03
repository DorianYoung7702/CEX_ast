# -*- coding: utf-8 -*-
"""
05_generate_report.py
---------------------
Generate output/report.md using existing outputs:
- basis_stats.csv
- arb_baseline_perf.json / arb_enhanced_perf.json / sr_price_action_perf.json
- arb_baseline_equity.csv / arb_enhanced_equity.csv / sr_price_action_equity.csv
- plots (*.png)

Run:
  python scripts/05_generate_report.py

Notes:
- This file contains NO business logic for strategies/backtests.
- It only summarizes outputs produced elsewhere.
"""

import os
import json
import math
import pandas as pd

from src.data_loader import DataLoaderConfig


# ---------------------------
# helpers
# ---------------------------
def load_perf(out_dir: str, name: str) -> dict:
    p = os.path.join(out_dir, f"{name}_perf.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_equity(out_dir: str, name: str) -> pd.DataFrame:
    p = os.path.join(out_dir, f"{name}_equity.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return ""


def _fmt_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):,.4f}"
    except Exception:
        return ""


def _fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return ""


def _infer_initial_final(eq: pd.DataFrame):
    if eq.empty or "equity" not in eq.columns:
        return None, None
    return float(eq["equity"].iloc[0]), float(eq["equity"].iloc[-1])


def _equity_contrib_summary(eq: pd.DataFrame) -> dict:
    """
    Try to decompose contributions using columns that the engine records.
    Works for arbitrage equity curves that contain pnl_funding/pnl_spot/pnl_perp/pnl_fees.
    If not present, returns empty dict.
    """
    if eq.empty:
        return {}
    cols = set(eq.columns)
    need_any = {"pnl_funding", "pnl_spot", "pnl_perp", "pnl_fees", "pnl_total"}
    if len(cols.intersection(need_any)) == 0:
        return {}

    out = {}
    for c in ["pnl_funding", "pnl_spot", "pnl_perp", "pnl_fees", "pnl_total"]:
        if c in eq.columns:
            out[c] = float(pd.to_numeric(eq[c], errors="coerce").fillna(0.0).sum())
    if "n_active" in eq.columns:
        out["avg_n_active"] = float(pd.to_numeric(eq["n_active"], errors="coerce").fillna(0.0).mean())
        out["active_days"] = int((pd.to_numeric(eq["n_active"], errors="coerce").fillna(0.0) > 0).sum())
        out["total_days"] = int(len(eq))
    return out


def _perf_table(perf: dict, strategy_name: str) -> dict:
    """
    Normalize perf json into a stable table schema.
    """
    return {
        "strategy": strategy_name,
        "annualized_return": perf.get("annualized_return", None),
        "sharpe": perf.get("sharpe", None),
        "max_drawdown": perf.get("max_drawdown", None),
        "alpha": perf.get("alpha", None),
        "buy_hold_cagr": perf.get("buy_hold_cagr", None),
    }


def _perf_table_formatted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format key metrics for markdown readability.
    Keep raw values too? Here we output the formatted version in report.
    """
    out = df.copy()
    for c in ["annualized_return", "max_drawdown", "alpha", "buy_hold_cagr"]:
        if c in out.columns:
            out[c] = out[c].apply(_fmt_pct)
    if "sharpe" in out.columns:
        out["sharpe"] = out["sharpe"].apply(_fmt_num)
    return out


# ---------------------------
# main
# ---------------------------
def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    cfg = DataLoaderConfig(data_dir="data", start_date="2021-01-01", end_date="2024-12-31")

    # basis stats
    basis_stats_path = os.path.join(out_dir, "basis_stats.csv")
    basis_stats = pd.read_csv(basis_stats_path) if os.path.exists(basis_stats_path) else pd.DataFrame()

    # perf
    perf_base = load_perf(out_dir, "arb_baseline")
    perf_enh = load_perf(out_dir, "arb_enhanced")
    perf_sr = load_perf(out_dir, "sr_price_action")

    perf_tbl = pd.DataFrame([
        _perf_table(perf_base, "arb_baseline"),
        _perf_table(perf_enh, "arb_enhanced"),
        _perf_table(perf_sr, "sr_price_action"),
    ])
    perf_tbl_fmt = _perf_table_formatted(perf_tbl)

    # equity curves for decomposition (if available)
    eq_base = load_equity(out_dir, "arb_baseline")
    eq_enh = load_equity(out_dir, "arb_enhanced")
    eq_sr = load_equity(out_dir, "sr_price_action")

    base_init, base_final = _infer_initial_final(eq_base)
    enh_init, enh_final = _infer_initial_final(eq_enh)
    sr_init, sr_final = _infer_initial_final(eq_sr)

    decomp_base = _equity_contrib_summary(eq_base)
    decomp_enh = _equity_contrib_summary(eq_enh)

    # quick comparison bullets (best-effort)
    compare_lines = []
    if base_final is not None and enh_final is not None and base_init is not None:
        compare_lines.append(f"- Final equity: baseline={_fmt_money(base_final)} vs enhanced={_fmt_money(enh_final)} (initial={_fmt_money(base_init)}).")
    if decomp_base.get("pnl_funding") is not None and decomp_enh.get("pnl_funding") is not None:
        compare_lines.append(f"- Funding PnL sum: baseline={_fmt_money(decomp_base['pnl_funding'])} vs enhanced={_fmt_money(decomp_enh['pnl_funding'])}.")
    if decomp_base.get("pnl_fees") is not None and decomp_enh.get("pnl_fees") is not None:
        compare_lines.append(f"- Fees sum: baseline={_fmt_money(decomp_base['pnl_fees'])} vs enhanced={_fmt_money(decomp_enh['pnl_fees'])} (negative means cost).")
    if decomp_base.get("avg_n_active") is not None and decomp_enh.get("avg_n_active") is not None:
        compare_lines.append(f"- Avg active positions (n_active): baseline={decomp_base['avg_n_active']:.2f} vs enhanced={decomp_enh['avg_n_active']:.2f}.")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Quant Technical Assessment Report\n\n")

        # -----------------------
        # Data section
        # -----------------------
        f.write("## Data\n")
        f.write("- **Source:** Binance REST APIs (spot/perpetual klines, funding rates)\n")
        f.write(f"- **Period:** {cfg.start_date} to {cfg.end_date}\n")
        f.write("- **Symbols:** BTCUSDT, ETHUSDT\n")
        f.write("- **Frequency:** 1d (signal computed on *t close*, executed on *t+1 open*, PnL measured open→open)\n\n")

        # -----------------------
        # Part 1a
        # -----------------------
        f.write("## Part 1a: Basis Analysis (Perp − Spot)\n\n")
        f.write("### Definition\n")
        f.write("- **Basis (open-consistent):** `basis_t = perp_open_t − spot_open_t`\n")
        f.write("- Interpretation: positive basis implies perpetual trading at a premium to spot; extreme values may indicate risk regimes.\n\n")

        f.write("### Basis time series plots\n")
        f.write("- ![](basis_BTCUSDT.png)\n")
        f.write("- ![](basis_ETHUSDT.png)\n\n")

        f.write("### Statistical summary\n\n")
        if not basis_stats.empty:
            f.write(basis_stats.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("_basis_stats.csv not found. Run scripts/01_basis_analysis.py first._\n\n")

        # -----------------------
        # Part 1b/1c
        # -----------------------
        f.write("## Part 1b/1c: Funding Rate Arbitrage Strategies\n\n")

        f.write("### Strategy requirements (from prompt)\n")
        f.write("- **Delta-neutral:** long spot + short perpetual of **equal USDT notional** per held asset.\n")
        f.write("- **One position per asset:** ON/OFF per symbol (no multiple concurrent pairs on the same asset).\n")
        f.write("- **Market-neutral portfolio:** no net directional exposure (only delta-neutral pairs).\n\n")

        f.write("### Baseline (1b)\n")
        f.write("- **Rule:** always maintain the delta-neutral pair for each asset (continuous carry collection).\n")
        f.write("- **Rebalancing:** daily to keep spot notional ≈ perp notional.\n")
        f.write("- **Primary return driver:** funding payments.\n\n")

        f.write("### Enhanced (1c)\n")
        f.write("- **Rule:** dynamically enter/exit the delta-neutral pair based on *carry regime* (funding) and *basis risk regimes*.\n")
        f.write("- **Intuition:** remain in the trade to collect funding when carry is favorable, but reduce exposure during unfavorable carry or extreme basis regimes.\n")
        f.write("- **Optional concentration:** Top-K selection (K=1 by default) to allocate capital to the asset with better carry signal, with anti-churn controls to limit switching.\n\n")

        f.write("### Equity curves\n")
        f.write("- ![](equity_arb_baseline.png)\n")
        f.write("- ![](equity_arb_enhanced.png)\n\n")

        # Decomposition (if available)
        if decomp_base or decomp_enh:
            f.write("### Return attribution (best-effort, using engine PnL components)\n\n")
            rows = []
            for name, d in [("baseline", decomp_base), ("enhanced", decomp_enh)]:
                if not d:
                    continue
                rows.append({
                    "strategy": name,
                    "pnl_funding": d.get("pnl_funding", 0.0),
                    "pnl_spot": d.get("pnl_spot", 0.0),
                    "pnl_perp": d.get("pnl_perp", 0.0),
                    "pnl_fees": d.get("pnl_fees", 0.0),
                    "pnl_total_sum": d.get("pnl_total", 0.0),
                    "avg_n_active": d.get("avg_n_active", None),
                    "active_days": d.get("active_days", None),
                })
            if rows:
                ddf = pd.DataFrame(rows)
                # format for markdown
                for c in ["pnl_funding", "pnl_spot", "pnl_perp", "pnl_fees", "pnl_total_sum"]:
                    if c in ddf.columns:
                        ddf[c] = ddf[c].apply(_fmt_money)
                if "avg_n_active" in ddf.columns:
                    ddf["avg_n_active"] = ddf["avg_n_active"].apply(lambda x: f"{x:.2f}" if x is not None else "")
                if "active_days" in ddf.columns:
                    ddf["active_days"] = ddf["active_days"].apply(lambda x: str(int(x)) if x is not None else "")
                f.write(ddf.to_markdown(index=False))
                f.write("\n\n")

            if compare_lines:
                f.write("**Quick comparison:**\n")
                for line in compare_lines:
                    f.write(f"{line}\n")
                f.write("\n")

        # -----------------------
        # Part 2
        # -----------------------
        f.write("## Part 2: Support/Resistance Price Action Strategy\n\n")
        f.write("### Price + Support/Resistance overlay\n")
        f.write("- ![](sr_overlay_BTCUSDT.png)\n")
        f.write("- ![](sr_overlay_ETHUSDT.png)\n\n")
        f.write("### Equity curve\n")
        f.write("- ![](equity_sr_price_action.png)\n\n")

        # -----------------------
        # Performance
        # -----------------------
        f.write("## Performance Summary (includes Alpha vs buy&hold)\n\n")
        f.write(perf_tbl_fmt.to_markdown(index=False))
        f.write("\n\n")

        # -----------------------
        # Discussion (expanded)
        # -----------------------
        f.write("## Part 3: Discussion\n\n")

        f.write("### Strategy execution details\n")
        f.write("- **Signal timing:** signals are computed using information available at **t close**.\n")
        f.write("- **Execution timing:** orders are executed at **t+1 open** with **slippage** and **fees** applied.\n")
        f.write("- **PnL accounting:** daily PnL is computed **open→open** to remain consistent with the execution price.\n")
        f.write("- **Market-neutral constraint:** when a symbol is ON, the portfolio holds **long spot + short perp** with equal USDT notional.\n")
        f.write("- **Position constraint:** each symbol is ON/OFF (at most one arbitrage pair per asset).\n\n")

        f.write("### What drives performance for funding arbitrage\n")
        f.write("- **Funding income is the dominant driver** in baseline; price drift and basis changes can add noise but should not dominate over long horizons.\n")
        f.write("- Enhanced strategies can outperform only if they either:\n")
        f.write("  1) avoid sufficiently many **negative-carry days**, and/or\n")
        f.write("  2) avoid **basis blowout risk** periods that cause drawdowns,\n")
        f.write("  while not sacrificing too many positive funding days.\n\n")

        f.write("### Why enhanced may underperform baseline (and how we mitigated it)\n")
        f.write("- **Turnover cost (fees+slippage)**: dynamic rules can increase trading frequency and switching, which quickly eats thin carry.\n")
        f.write("- **Over-filtering**: strict entry conditions reduce time-in-market, directly reducing funding income.\n")
        f.write("- **Parameter semantics mismatch**: basis thresholds must match the implemented condition (e.g., `enter if z <= entry_z_max`).\n")
        f.write("- **Mitigation direction used here**: higher switching thresholds, longer minimum hold, stricter exit confirmation, and less restrictive entry gates.\n\n")

        f.write("### Challenges & solutions\n")
        f.write("- **API pagination / rate-limits:** handled via iterative REST pagination and local CSV caching.\n")
        f.write("- **Data completeness:** aligned spot/perp/funding on a common daily index and used forward-filling where appropriate.\n")
        f.write("- **Execution consistency:** used open-consistent basis definition and open→open PnL.\n")
        f.write("- **Risk modeling simplifications:** margin and liquidation are simplified; the goal is consistent relative comparison of strategies.\n\n")

        f.write("### Limitations (important assumptions)\n")
        f.write("- **Funding aggregation:** funding is treated as `funding_rate_daily * perp_notional` (daily approximation).\n")
        f.write("- **Fees/slippage:** constant rates are assumed; in reality they depend on venue, liquidity, and order type.\n")
        f.write("- **Margin & liquidation:** simplified maintenance margin checks; real exchanges have more complex mark price/liquidation rules.\n")
        f.write("- **Borrow/financing for spot:** spot long implicitly assumes capital is available; no borrow cost is included.\n\n")

        f.write("### Suggested improvements / robustness checks\n")
        f.write("- **Robustness grid / walk-forward:** evaluate enhanced parameters (lookback, exit thresholds, switch bands) on rolling windows.\n")
        f.write("- **Transaction cost sensitivity:** re-run with higher fees/slippage to test if alpha survives realistic costs.\n")
        f.write("- **Multiple assets / universe expansion:** include more symbols to diversify carry and reduce concentration risk.\n")
        f.write("- **Better basis risk control:** add widening-speed filters (slope/acceleration) and volatility scaling.\n")
        f.write("- **Risk budget:** cap exposure per asset and add portfolio-level stop/risk-off logic for extreme regimes.\n\n")

    print(f"✅ Saved: {report_path}")


if __name__ == "__main__":
    main()