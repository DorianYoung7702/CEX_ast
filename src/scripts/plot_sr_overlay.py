# -*- coding: utf-8 -*-
"""
03_plot_sr_overlay.py
Part 2: price + last confirmed support/resistance overlay, optionally with trade markers.

Run:
  python scripts/03_plot_sr_overlay.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import DataLoaderConfig, load_klines_csv


def confirmed_pivots(series: pd.Series, order: int = 10):
    """
    简化 pivot 检测（不依赖 scipy） + 右移确认 order 天（避免重绘）
    """
    x = pd.to_numeric(series, errors="coerce")
    win = 2 * order + 1
    rolling_min = x.rolling(win, center=True, min_periods=win).min()
    rolling_max = x.rolling(win, center=True, min_periods=win).max()

    piv_low = (x == rolling_min)
    piv_high = (x == rolling_max)

    sup = pd.Series(np.nan, index=x.index)
    res = pd.Series(np.nan, index=x.index)
    sup[piv_low] = x[piv_low]
    res[piv_high] = x[piv_high]

    sup_confirm = sup.shift(order)
    res_confirm = res.shift(order)
    return sup_confirm, res_confirm


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    cfg = DataLoaderConfig(data_dir="data", start_date="2021-01-01", end_date="2024-12-31")
    symbols = ("BTCUSDT", "ETHUSDT")
    order = 10

    trades_path = os.path.join(out_dir, "sr_price_action_trades.csv")
    trades = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
        trades["symbol"] = trades["symbol"].astype(str)

    for s in symbols:
        spot = load_klines_csv(cfg.data_dir, s, "spot", "1d", cfg.start_date, cfg.end_date).copy()
        spot["date"] = pd.to_datetime(spot["date"])
        spot = spot.sort_values("date").reset_index(drop=True)

        sup_c, res_c = confirmed_pivots(spot["close"], order=order)
        last_sup = sup_c.ffill()
        last_res = res_c.ffill()

        plt.figure(figsize=(12, 5))
        plt.plot(spot["date"], spot["close"], label="Close")
        plt.plot(spot["date"], last_sup, label="Last Confirmed Support")
        plt.plot(spot["date"], last_res, label="Last Confirmed Resistance")

        # trade markers（如你的 trade log 列不同，改这里）
        if not trades.empty and "side" in trades.columns and "price" in trades.columns:
            t = trades[trades["symbol"] == s]
            buys = t[t["side"] == "BUY"]
            sells = t[t["side"] == "SELL"]
            if not buys.empty:
                plt.scatter(buys["date"], buys["price"], marker="^")
            if not sells.empty:
                plt.scatter(sells["date"], sells["price"], marker="v")

        plt.title(f"Price + Support/Resistance Overlay: {s}")
        plt.xlabel("Date")
        plt.ylabel("Price (USDT)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sr_overlay_{s}.png"))
        plt.close()

        print(f"✅ Saved: output/sr_overlay_{s}.png")


if __name__ == "__main__":
    main()