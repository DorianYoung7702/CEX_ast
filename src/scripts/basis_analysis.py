# -*- coding: utf-8 -*-
"""
01_basis_analysis.py
Part 1a: basis analysis with robust cleaning + plots

Outputs:
- output/spot_close_{symbol}.png       (Spot close price curve)
- output/basis_pct_{symbol}.png        (basis %)
- output/basis_stats.csv / basis_stats.md

Notes:
- We choose basis_pct as the SINGLE basis metric for plots/stats because it's scale-free
  and comparable across time and assets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import DataLoaderConfig, load_assessment_data_for_symbol


def _winsorize(s: pd.Series, lower_q=0.005, upper_q=0.995) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().empty:
        return x
    lo = x.quantile(lower_q)
    hi = x.quantile(upper_q)
    return x.clip(lower=lo, upper=hi)


def _plot_line(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    figsize=(12, 4),
):
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    cfg = DataLoaderConfig(data_dir="data", start_date="2021-01-01", end_date="2024-12-31")
    symbols = ("BTCUSDT", "ETHUSDT")

    stats_rows = []

    for s in symbols:
        d = load_assessment_data_for_symbol(s, cfg=cfg)
        m = d["merged"].copy()
        m["date"] = pd.to_datetime(m["date"])

        # 强制数值化
        m["spot_close"] = pd.to_numeric(m.get("spot_close"), errors="coerce")
        m["perp_close"] = pd.to_numeric(m.get("perp_close"), errors="coerce")

        # 清洗：去掉无效价格
        m = m[(m["spot_close"] > 0) & (m["perp_close"] > 0)].copy()
        m = m.sort_values("date").reset_index(drop=True)

        # ========= 新增：Spot close 价格曲线 =========
        _plot_line(
            x=m["date"],
            y=m["spot_close"],
            title=f"Spot Close Price: {s}",
            xlabel="Date",
            ylabel="Spot Close (USDT)",
            out_path=os.path.join(out_dir, f"spot_close_{s}.png"),
        )

        # ========= Basis：只保留 pct =========
        # basis_pct = perp_close / spot_close - 1
        m["basis_pct"] = (m["perp_close"] / m["spot_close"]) - 1.0

        # 可选：年化仅用于描述（不要用于交易逻辑阈值）
        m["basis_ann"] = m["basis_pct"] * 365.0

        b = m["basis_pct"].dropna()
        if b.empty:
            stats_rows.append({
                "symbol": s,
                "count": 0,
                "basis_pct_mean": np.nan,
                "basis_pct_std": np.nan,
                "basis_pct_min": np.nan,
                "basis_pct_p05": np.nan,
                "basis_pct_p50": np.nan,
                "basis_pct_p95": np.nan,
                "basis_pct_max": np.nan,
                "basis_ann_mean": np.nan,
            })
            continue

        stats_rows.append({
            "symbol": s,
            "count": int(b.shape[0]),
            "basis_pct_mean": float(b.mean()),
            "basis_pct_std": float(b.std(ddof=0)),
            "basis_pct_min": float(b.min()),
            "basis_pct_p05": float(b.quantile(0.05)),
            "basis_pct_p50": float(b.quantile(0.50)),
            "basis_pct_p95": float(b.quantile(0.95)),
            "basis_pct_max": float(b.max()),
            "basis_ann_mean": float(m["basis_ann"].dropna().mean()),
        })

        # 为了避免“纵轴被极端点拉爆”，用 winsorize 版本绘图
        y_pct = _winsorize(m["basis_pct"], 0.005, 0.995)

        _plot_line(
            x=m["date"],
            y=y_pct * 100.0,  # 转成 %
            title=f"Basis % ((Perp/Spot)-1) [winsorized 0.5%-99.5%]: {s}",
            xlabel="Date",
            ylabel="Basis (%)",
            out_path=os.path.join(out_dir, f"basis_pct_{s}.png"),
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(out_dir, "basis_stats.csv"), index=False)

    with open(os.path.join(out_dir, "basis_stats.md"), "w", encoding="utf-8") as f:
        f.write("# Basis Statistical Summary\n\n")
        f.write("Definitions:\n")
        f.write("- basis_pct = perp_close/spot_close - 1\n")
        f.write("- basis_ann = basis_pct * 365 (descriptive only)\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n")

    print("✅ Saved:")
    print("- output/spot_close_BTCUSDT.png, output/spot_close_ETHUSDT.png")
    print("- output/basis_pct_BTCUSDT.png, output/basis_pct_ETHUSDT.png")
    print("- output/basis_stats.csv, output/basis_stats.md")


if __name__ == "__main__":
    main()