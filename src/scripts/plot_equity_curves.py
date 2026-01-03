# -*- coding: utf-8 -*-
"""
02_plot_equity_curves.py
Plots equity curve charts for strategies (robust datetime parsing + diagnostics).

Run:
  python src/scripts/plot_equity_curves.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robust parse:
    - epoch seconds/ms
    - ISO strings
    - mixed formats
    """
    x = s.copy()

    # If numeric epoch
    if pd.api.types.is_numeric_dtype(x):
        # heuristic: ms if large
        unit = "ms" if x.dropna().astype(float).median() > 1e10 else "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce").dt.tz_convert(None)

    # string-like
    x_str = pd.Series(x.astype(str).values, index=x.index)
    # try ISO/mixed
    dt = pd.to_datetime(x_str, errors="coerce", utc=False)
    if dt.notna().mean() >= 0.8:
        return dt

    # try mixed explicitly (pandas>=2.0)
    dt = pd.to_datetime(x_str, errors="coerce", format="mixed")
    return dt


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    for name in ["arb_baseline", "arb_enhanced", "sr_price_action"]:
        eq_path = os.path.join(out_dir, f"{name}_equity.csv")
        if not os.path.exists(eq_path):
            print(f"⚠️ Missing: {eq_path}")
            continue

        df = pd.read_csv(eq_path)

        if "date" not in df.columns or "equity" not in df.columns:
            print(f"⚠️ Bad schema in {eq_path}. cols={df.columns.tolist()}")
            continue

        df["date_raw"] = df["date"]
        df["date"] = _parse_date_series(df["date"])
        df = df.dropna(subset=["date", "equity"]).copy()
        df = df.sort_values("date")

        if df.empty:
            print(f"⚠️ Empty after parsing: {eq_path}")
            # show a few raw samples for debugging
            print("raw date samples:", df.get("date_raw", pd.Series(dtype=str)).head(5).tolist())
            continue

        # diagnostics: THIS will tell you immediately if your equity.csv has wrong dates
        print(f"[{name}] rows={len(df)} date_min={df['date'].min()} date_max={df['date'].max()}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["date"], df["equity"])
        ax.set_title(f"Equity Curve: {name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USDT)")

        # Better date ticks
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        fig.tight_layout()
        out_png = os.path.join(out_dir, f"equity_{name}.png")
        fig.savefig(out_png)
        plt.close(fig)

        print(f"✅ Saved: {out_png}")


if __name__ == "__main__":
    main()