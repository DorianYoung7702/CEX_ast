# -*- coding: utf-8 -*-
"""
binance_client.py
-----------------
Binance REST 数据拉取 + 分页 + 重试 + 落盘（CSV）

覆盖评估所需数据：
- Spot OHLCV (daily):      GET https://api.binance.com/api/v3/klines
- Perp OHLCV (daily):      GET https://fapi.binance.com/fapi/v1/klines
- Perp Funding Rate hist:  GET https://fapi.binance.com/fapi/v1/fundingRate

输出落盘：
- data/spot_BTCUSDT_1d_20210101_20241231.csv
- data/perp_BTCUSDT_1d_20210101_20241231.csv
- data/funding_BTCUSDT_20210101_20241231.csv
(ETH 同理)

依赖：requests, pandas, numpy
"""

from __future__ import annotations

import os
import time
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import requests
import pandas as pd

Market = Literal["spot", "perp"]

BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUT_BASE = "https://fapi.binance.com"

KLINES_SPOT_PATH = "/api/v3/klines"
KLINES_FUT_PATH = "/fapi/v1/klines"
FUNDING_PATH = "/fapi/v1/fundingRate"


@dataclass
class BinanceClientConfig:
    data_dir: str = "data"
    timeout_sec: int = 20
    max_retries: int = 6
    base_sleep_sec: float = 0.35
    jitter_sec: float = 0.15
    user_agent: str = "quant-assessment-client/1.0"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_symbol(symbol: str) -> str:
    """Binance 用 BTCUSDT；兼容 BTC/USDT、BTC-USDT。"""
    s = (symbol or "").strip()
    if not s:
        raise ValueError("symbol is empty")
    return s.replace("/", "").replace("-", "").upper()


def _to_utc_ms(ts: Any) -> int:
    """
    - None -> 当前 UTC ms
    - int/float -> 秒/毫秒（自动判断）
    - str/datetime/Timestamp -> 解析为 UTC ms
      若无 tzinfo：默认当作 UTC
    """
    if ts is None:
        return int(pd.Timestamp.utcnow().timestamp() * 1000)

    if isinstance(ts, (int, float)):
        v = float(ts)
        return int(v if v > 1e10 else v * 1000)

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp() * 1000)


def _fmt_yyyymmdd(d: Any) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _safe_float(x: Any) -> float:
    """安全转 float：None/''/非法 -> NaN"""
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _normalize_end_time(end_time: Any) -> Any:
    """
    为了数据“完整性”：
    - 若 end_time 是 'YYYY-MM-DD' 这种日期字符串，则扩展为当日结束 23:59:59.999（UTC）
      这样 funding 的当天 08:00/16:00 也能被包含。
    """
    if isinstance(end_time, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_time.strip()):
        return end_time.strip() + " 23:59:59.999"
    return end_time


def _request_json(
    url: str,
    params: Dict[str, Any],
    cfg: BinanceClientConfig,
) -> Any:
    """
    带重试与退避的 GET 请求。
    对 429/418 做退避；网络错误重试。
    """
    headers = {"User-Agent": cfg.user_agent}
    last_err: Optional[Exception] = None

    for i in range(cfg.max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=cfg.timeout_sec)

            if r.status_code in (418, 429):
                sleep = cfg.base_sleep_sec * (2 ** i) + random.uniform(0, cfg.jitter_sec)
                time.sleep(min(sleep, 10.0))
                continue

            r.raise_for_status()
            data = r.json()

            # 极少数情况下（例如网关/代理异常）会返回 dict 的错误结构
            if isinstance(data, dict) and ("code" in data or "msg" in data):
                raise RuntimeError(f"Binance API error payload: {data}")

            return data

        except Exception as e:
            last_err = e
            sleep = cfg.base_sleep_sec * (2 ** i) + random.uniform(0, cfg.jitter_sec)
            time.sleep(min(sleep, 10.0))

    raise RuntimeError(f"GET failed after retries. url={url}, params={params}, err={last_err}")


class BinanceRESTClient:
    def __init__(self, cfg: Optional[BinanceClientConfig] = None):
        self.cfg = cfg or BinanceClientConfig()
        _ensure_dir(self.cfg.data_dir)

    # -------------------------
    # KLINES
    # -------------------------
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        market: Market,
        start_time: Any,
        end_time: Any,
        limit: int = 1000,
        sleep_sec: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        分页拉取 K 线，返回标准化 DataFrame（UTC）。
        """
        sym = _normalize_symbol(symbol)
        end_time = _normalize_end_time(end_time)

        start_ms = _to_utc_ms(start_time)
        end_ms = _to_utc_ms(end_time)
        if end_ms <= start_ms:
            raise ValueError("end_time must be after start_time")

        base = BINANCE_SPOT_BASE if market == "spot" else BINANCE_FUT_BASE
        path = KLINES_SPOT_PATH if market == "spot" else KLINES_FUT_PATH
        url = base + path

        rows: List[Dict[str, Any]] = []
        cursor = start_ms
        prev_last_open: Optional[int] = None

        while True:
            params = {
                "symbol": sym,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            }
            data = _request_json(url, params, self.cfg)
            if not data:
                break

            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected klines response type: {type(data)}; data={data}")

            last_open = int(data[-1][0])
            if prev_last_open is not None and last_open <= prev_last_open:
                break
            prev_last_open = last_open

            for k in data:
                ot = int(k[0])
                if ot > end_ms:
                    continue
                rows.append(
                    {
                        "open_time": pd.to_datetime(ot, unit="ms", utc=True),
                        "open": _safe_float(k[1]),
                        "high": _safe_float(k[2]),
                        "low": _safe_float(k[3]),
                        "close": _safe_float(k[4]),
                        "volume": _safe_float(k[5]),
                        "close_time": pd.to_datetime(int(k[6]), unit="ms", utc=True),
                        "quote_volume": _safe_float(k[7]),
                        "trades": int(k[8]) if str(k[8]).strip() != "" else 0,
                        "taker_buy_base": _safe_float(k[9]),
                        "taker_buy_quote": _safe_float(k[10]),
                    }
                )

            cursor = last_open + 1
            if len(data) < limit or cursor > end_ms:
                break

            time.sleep(sleep_sec if sleep_sec is not None else self.cfg.base_sleep_sec)

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote",
                ]
            )

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return df

    def save_klines_csv(
        self,
        symbol: str,
        interval: str,
        market: Market,
        start_time: Any,
        end_time: Any,
        force: bool = False,
    ) -> str:
        sym = _normalize_symbol(symbol)
        start_tag = _fmt_yyyymmdd(start_time)
        end_tag = _fmt_yyyymmdd(end_time)
        fname = f"{market}_{sym}_{interval}_{start_tag}_{end_tag}.csv"
        fpath = os.path.join(self.cfg.data_dir, fname)

        if os.path.exists(fpath) and not force:
            return fpath

        df = self.fetch_klines(sym, interval, market, start_time, end_time)
        if df.empty:
            raise RuntimeError(f"Empty klines for {market} {sym} {interval}")

        df.to_csv(fpath, index=False)
        return fpath

    # -------------------------
    # FUNDING RATE
    # -------------------------
    def fetch_funding_rates(
        self,
        symbol: str,
        start_time: Any,
        end_time: Any,
        limit: int = 1000,
        sleep_sec: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        分页拉取 funding rate 历史（8小时一条），返回 DataFrame（UTC）。

        注意：Binance 可能返回 markPrice=""（空字符串），必须容错解析。
        """
        sym = _normalize_symbol(symbol)
        end_time = _normalize_end_time(end_time)

        start_ms = _to_utc_ms(start_time)
        end_ms = _to_utc_ms(end_time)

        url = BINANCE_FUT_BASE + FUNDING_PATH

        rows: List[Dict[str, Any]] = []
        cursor = start_ms
        prev_last: Optional[int] = None

        while True:
            params = {"symbol": sym, "startTime": cursor, "endTime": end_ms, "limit": limit}
            data = _request_json(url, params, self.cfg)
            if not data:
                break

            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected funding response type: {type(data)}; data={data}")

            last_t = int(data[-1].get("fundingTime", 0))
            if last_t <= 0:
                break
            if prev_last is not None and last_t <= prev_last:
                break
            prev_last = last_t

            for item in data:
                ft = int(item.get("fundingTime", 0) or 0)
                if ft <= 0 or ft < start_ms or ft > end_ms:
                    continue

                fr = _safe_float(item.get("fundingRate"))
                mp = _safe_float(item.get("markPrice"))

                rows.append(
                    {
                        "symbol": item.get("symbol", sym),
                        "funding_time": pd.to_datetime(ft, unit="ms", utc=True),
                        "funding_rate": fr,
                        "mark_price": mp,  # 允许 NaN
                    }
                )

            cursor = last_t + 1
            if len(data) < limit or cursor > end_ms:
                break

            time.sleep(sleep_sec if sleep_sec is not None else self.cfg.base_sleep_sec)

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "funding_time", "funding_rate", "mark_price"])

        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
        df = df.dropna(subset=["funding_rate"]).copy()
        df = df.drop_duplicates(subset=["funding_time"]).sort_values("funding_time").reset_index(drop=True)
        return df

    def save_funding_csv(
        self,
        symbol: str,
        start_time: Any,
        end_time: Any,
        force: bool = False,
    ) -> str:
        sym = _normalize_symbol(symbol)
        start_tag = _fmt_yyyymmdd(start_time)
        end_tag = _fmt_yyyymmdd(end_time)
        fname = f"funding_{sym}_{start_tag}_{end_tag}.csv"
        fpath = os.path.join(self.cfg.data_dir, fname)

        if os.path.exists(fpath) and not force:
            return fpath

        df = self.fetch_funding_rates(sym, start_time, end_time)
        if df.empty:
            raise RuntimeError(f"Empty funding for {sym}")

        df.to_csv(fpath, index=False)
        return fpath

    # -------------------------
    # Convenience for assessment
    # -------------------------
    def download_assessment_dataset(
        self,
        symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT"),
        start: str = "2021-01-01",
        end: str = "2024-12-31",
        interval: str = "1d",
        force: bool = False,
    ) -> Dict[str, List[str]]:
        """
        一次性拉取评估需要的数据并落盘。
        """
        # 关键：保证 end 是“当日结束”，确保 funding 最后一日完整
        end_norm = _normalize_end_time(end)

        outputs: Dict[str, List[str]] = {"spot": [], "perp": [], "funding": []}
        for s in symbols:
            outputs["spot"].append(self.save_klines_csv(s, interval, "spot", start, end_norm, force=force))
            outputs["perp"].append(self.save_klines_csv(s, interval, "perp", start, end_norm, force=force))
            outputs["funding"].append(self.save_funding_csv(s, start, end_norm, force=force))
        return outputs


if __name__ == "__main__":
    client = BinanceRESTClient()
    out = client.download_assessment_dataset(
        symbols=("BTCUSDT", "ETHUSDT"),
        start=os.getenv("START_DATE", "2021-01-01"),
        end=os.getenv("END_DATE", "2024-12-31"),
        interval=os.getenv("INTERVAL", "1d"),
        force=os.getenv("FORCE", "0") == "1",
    )
    print(json.dumps(out, indent=2))