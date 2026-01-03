# -*- coding: utf-8 -*-
"""
main.py
-------
One-click orchestration ONLY (no business logic).
Guarantees:
- No duplicate function definitions
- main() called exactly once
- Consistent start/end across downloader + loader + backtests
- Uses existing project modules (binance_client, data_loader, strategies, scripts)

Flow:
  1) Download dataset (spot/perp/funding) -> data/
  2) Load & align merged data (spot+perp+funding) -> in-memory
  3) Run strategies (baseline/enhanced/sr) -> output/*_equity.csv, *_trades.csv, *_perf.json
  4) Postprocess (basis plots/stats, equity plots, sr overlay, report) -> output/*.png, report.md
"""

from src.binance_client import BinanceRESTClient
from src.data_loader import DataLoaderConfig, load_assessment_data_for_symbol, load_klines_csv

from src.strategies.funding_arbitrage_baseline import run_baseline
from src.strategies.funding_arbitrage_enhanced import run_enhanced
from src.strategies.sr_price_action import run_sr_price_action


def run_backtests(
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    data_dir: str = "data",
    out_dir: str = "output",
    force_download: bool = False,
) -> None:
    # 1) Download dataset (REST -> CSV under data/)
    client = BinanceRESTClient()
    client.download_assessment_dataset(
        symbols=symbols,
        start=start,
        end=end,
        interval=interval,
        force=force_download,
    )

    # 2) Load merged (spot+perp+funding aligned by date)
    cfg = DataLoaderConfig(data_dir=data_dir, start_date=start, end_date=end)
    merged = {s: load_assessment_data_for_symbol(s, cfg=cfg)["merged"] for s in symbols}

    # 3) Funding arbitrage baseline (1b)
    res_base = run_baseline(
        merged_by_symbol=merged,
        initial_capital=1_000_000.0,
    )
    res_base.save(out_dir=out_dir, prefix="arb_baseline")

    # 4) Funding arbitrage enhanced (1c)
    res_enh = run_enhanced(
        merged_by_symbol=merged,
        lookback=60,
        min_funding=0.0,
        exit_funding=-0.0010,  # 少退出
        fr_fast_span=5,
        fr_slow_span=30,
        enter_confirm_days=2,
        exit_confirm_days=3,  # 退出更慢更稳
        min_hold_days=7,  # 降 churn
        cooldown_days=1,  # 防抖
        basis_enter_z=0,  # 不挡入场
        basis_exit_z=2.8,  # 只在更极端 blowout 才退出
        switch_threshold=0.40,  # ✅关键：大幅减少 BTC/ETH 来回切
        # 如果你的 run_enhanced 支持 kwargs：
        backward_z=-3.0,  # backwardation 极端才退（或直接禁用）
        verbose=True,
        initial_capital=1_000_000.0,
    )

    res_enh.save(out_dir=out_dir, prefix="arb_enhanced")

    # 5) S/R price action (spot-only)
    spot_by_symbol = {
        s: load_klines_csv(cfg.data_dir, s, "spot", interval, cfg.start_date, cfg.end_date)
        for s in symbols
    }
    res_sr = run_sr_price_action(spot_by_symbol)
    res_sr.save(out_dir=out_dir, prefix="sr_price_action")

    print("✅ Backtests finished. Raw outputs saved under ./output")


def run_postprocess() -> None:
    # Import locally to keep main import-time lightweight
    from src.scripts.basis_analysis import main as basis_main
    from src.scripts.plot_equity_curves import main as equity_main
    from src.scripts.plot_sr_overlay import main as sr_overlay_main
    from src.scripts.generate_report import main as report_main

    basis_main()
    equity_main()
    sr_overlay_main()
    report_main()

    print("✅ Postprocess finished. Deliverables saved under ./output")


def main() -> None:
    # Single source of truth for params
    symbols = ("BTCUSDT", "ETHUSDT")
    start = "2021-01-01"
    end = "2024-12-31"
    interval = "1d"

    run_backtests(
        symbols=symbols,
        start=start,
        end=end,
        interval=interval,
        data_dir="data",
        out_dir="output",
        force_download=False,
    )
    run_postprocess()
    print("✅ All done: backtests + required plots + report.md")


if __name__ == "__main__":
    main()