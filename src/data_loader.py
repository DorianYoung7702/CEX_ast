# -*- coding: utf-8 -*-
"""
data_loader.py
--------------
读取本地 CSV、对齐、清洗（Spot/Perp/Funding -> Daily）

目标：
- 读取 binance_client.py 落盘的 CSV
- Spot 与 Perp 在同一日频上对齐（UTC 日）
- Funding rate（8h）聚合到 daily（UTC 日）
- 生成用于策略/回测的标准化 DataFrame

输出典型字段：
- date (UTC date)
- spot_close, perp_close
- basis = perp_close - spot_close
- funding_rate_daily (sum of 8h rates per day)
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataLoaderConfig:
    data_dir: str = "data"
    timezone: str = "UTC"  # 推荐 UTC，避免 spot/perp 日边界不一致
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    keep_na: bool = True


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").replace("/", "").replace("-", "").upper()


def _date_index(start: str, end: str) -> pd.Index:
    # crypto 7x24，用日历日
    return pd.date_range(start=start, end=end, freq="D").date


def _find_latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _parse_datetime_utc(series: pd.Series) -> pd.Series:
    """
    强健的 UTC 时间解析：
    - 兼容混合 ISO 格式（有无微秒、有无时区）
    - 兼容字符串 / datetime / pandas Timestamp
    - 兼容毫秒时间戳（int/float）列
    """
    s = series

    # 若已经是 datetime64[ns, tz] / datetime64[ns]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        return dt

    # 若是纯数字（时间戳）
    if pd.api.types.is_numeric_dtype(s):
        # 默认当作毫秒
        dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        return dt

    # 字符串/混合类型
    # pandas>=2.0 支持 format="mixed"
    try:
        dt = pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    except TypeError:
        # 旧 pandas 不支持 mixed
        dt = pd.to_datetime(s, utc=True, errors="coerce")

    # 如果解析后大面积 NaT，再尝试 ISO8601（部分 pandas 版本支持）
    if dt.isna().mean() > 0.5:
        try:
            dt2 = pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")
            if dt2.isna().mean() < dt.isna().mean():
                dt = dt2
        except Exception:
            pass

    return dt


def load_klines_csv(
    data_dir: str,
    symbol: str,
    market: str,
    interval: str = "1d",
    start: str = "2021-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    读取落盘的 klines CSV。

    约定文件名：
    {market}_{symbol}_{interval}_{startTag}_{endTag}.csv
    例如：spot_BTCUSDT_1d_20210101_20241231.csv
    """
    sym = _normalize_symbol(symbol)
    start_tag = pd.Timestamp(start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")
    exact = os.path.join(data_dir, f"{market}_{sym}_{interval}_{start_tag}_{end_tag}.csv")

    fpath = exact if os.path.exists(exact) else _find_latest_file(
        os.path.join(data_dir, f"{market}_{sym}_{interval}_*.csv")
    )
    if not fpath or not os.path.exists(fpath):
        raise FileNotFoundError(f"klines file not found for {market} {sym}: {exact}")

    df = pd.read_csv(fpath)
    if "open_time" not in df.columns:
        raise ValueError(f"Invalid klines CSV schema: missing open_time in {fpath}")

    df["open_time"] = _parse_datetime_utc(df["open_time"])
    if df["open_time"].isna().any():
        bad = int(df["open_time"].isna().sum())
        raise ValueError(f"Failed to parse open_time for {bad} rows in {fpath}")

    df["date"] = df["open_time"].dt.date

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    df = df[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_funding_csv(
    data_dir: str,
    symbol: str,
    start: str = "2021-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    读取落盘 funding CSV。

    约定文件名：
    funding_{symbol}_{startTag}_{endTag}.csv
    例如：funding_BTCUSDT_20210101_20241231.csv
    """
    sym = _normalize_symbol(symbol)
    start_tag = pd.Timestamp(start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")
    exact = os.path.join(data_dir, f"funding_{sym}_{start_tag}_{end_tag}.csv")

    fpath = exact if os.path.exists(exact) else _find_latest_file(
        os.path.join(data_dir, f"funding_{sym}_*.csv")
    )
    if not fpath or not os.path.exists(fpath):
        raise FileNotFoundError(f"funding file not found for {sym}: {exact}")

    df = pd.read_csv(fpath)
    if "funding_time" not in df.columns:
        raise ValueError(f"Invalid funding CSV schema: missing funding_time in {fpath}")

    # ✅ 关键修复：混合格式解析
    df["funding_time"] = _parse_datetime_utc(df["funding_time"])
    if df["funding_time"].isna().any():
        # 不直接炸：先尝试丢弃无法解析的行（通常极少数）
        df = df.dropna(subset=["funding_time"]).copy()

    df["date"] = df["funding_time"].dt.date

    # 数值列清洗
    if "funding_rate" in df.columns:
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    if "mark_price" in df.columns:
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")

    # funding_rate 缺失行无意义
    if "funding_rate" in df.columns:
        df = df.dropna(subset=["funding_rate"]).copy()

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    df = df[(df["date"] >= start_d) & (df["date"] <= end_d)].copy()

    df = df.sort_values("funding_time").reset_index(drop=True)
    return df


def aggregate_funding_daily(funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    funding 每 8 小时一条，聚合成 daily（sum）。
    """
    if funding_df.empty:
        return pd.DataFrame(columns=["date", "funding_rate_daily", "mark_price_last"])

    g = funding_df.groupby("date", as_index=False)
    daily = g.agg(
        funding_rate_daily=("funding_rate", "sum"),
        mark_price_last=("mark_price", "last"),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def align_spot_perp_funding(
    spot_df: pd.DataFrame,
    perp_df: pd.DataFrame,
    funding_daily_df: pd.DataFrame,
    cfg: DataLoaderConfig,
    symbol: str,
) -> pd.DataFrame:
    """
    对齐 spot/perp/funding 到统一日频索引。
    输出字段（核心）：
      date, spot_close, perp_close, basis, funding_rate_daily
    """
    idx = pd.Index(_date_index(cfg.start_date, cfg.end_date), name="date")

    def _reindex_ohlc(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep_cols = ["date", "open", "high", "low", "close", "volume"]
        cols = [c for c in keep_cols if c in df.columns]
        out = df[cols].copy()
        out = out.set_index("date").reindex(idx)
        out = out.rename(
            columns={
                "open": f"{prefix}_open",
                "high": f"{prefix}_high",
                "low": f"{prefix}_low",
                "close": f"{prefix}_close",
                "volume": f"{prefix}_volume",
            }
        )
        return out

    spot_x = _reindex_ohlc(spot_df, "spot")
    perp_x = _reindex_ohlc(perp_df, "perp")

    fund_x = funding_daily_df.copy()
    if not fund_x.empty:
        fund_x = fund_x.set_index("date").reindex(idx)
        fund_x = fund_x.rename(columns={"funding_rate_daily": "funding_rate_daily"})
    else:
        fund_x = pd.DataFrame(index=idx, data={"funding_rate_daily": np.nan})

    merged = spot_x.join(perp_x, how="outer").join(fund_x[["funding_rate_daily"]], how="outer")
    merged = merged.reset_index()

    merged["symbol"] = _normalize_symbol(symbol)
    merged["basis"] = merged["perp_close"] - merged["spot_close"]

    for c in [
        "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume",
        "perp_open", "perp_high", "perp_low", "perp_close", "perp_volume",
        "basis", "funding_rate_daily",
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if not cfg.keep_na:
        merged = merged.dropna(subset=["spot_close", "perp_close"]).reset_index(drop=True)

    return merged


def load_assessment_data_for_symbol(
    symbol: str,
    cfg: Optional[DataLoaderConfig] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    单个 symbol 的全量载入：
    - spot klines
    - perp klines
    - funding daily
    - merged aligned
    """
    cfg = cfg or DataLoaderConfig()
    sym = _normalize_symbol(symbol)

    spot = load_klines_csv(cfg.data_dir, sym, "spot", interval, cfg.start_date, cfg.end_date)
    perp = load_klines_csv(cfg.data_dir, sym, "perp", interval, cfg.start_date, cfg.end_date)
    funding = load_funding_csv(cfg.data_dir, sym, cfg.start_date, cfg.end_date)
    funding_daily = aggregate_funding_daily(funding)

    merged = align_spot_perp_funding(spot, perp, funding_daily, cfg, sym)

    return {
        "spot": spot,
        "perp": perp,
        "funding": funding,
        "funding_daily": funding_daily,
        "merged": merged,
    }


def load_assessment_dataset(
    symbols: List[str] = ["BTCUSDT", "ETHUSDT"],
    cfg: Optional[DataLoaderConfig] = None,
    interval: str = "1d",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    cfg = cfg or DataLoaderConfig()
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for s in symbols:
        out[_normalize_symbol(s)] = load_assessment_data_for_symbol(s, cfg=cfg, interval=interval)
    return out


if __name__ == "__main__":
    cfg = DataLoaderConfig()
    dataset = load_assessment_dataset(["BTCUSDT", "ETHUSDT"], cfg=cfg)

    for sym, d in dataset.items():
        m = d["merged"]
        print(f"\n[{sym}] merged shape={m.shape}")
        print(m[["date", "spot_close", "perp_close", "basis", "funding_rate_daily"]].tail(5))