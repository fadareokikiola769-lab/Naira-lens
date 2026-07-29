"""
Data Collection Module (3.5.1) + Price Data Preprocessing (3.5.2)

Pulls historical/real-time price data from the CoinGecko public API
(no key required), persists it to the market_data table, and applies
the cleaning steps described in the chapter: forward-fill for missing
values, IQR-based outlier removal, and min-max normalisation ahead of
modelling.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import Config
from extensions import db
from models import MarketData

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"


def _fetch_from_coingecko(coin: str, days: int) -> pd.DataFrame:
    coin_id = Config.COINGECKO_IDS[coin]
    resp = requests.get(
        COINGECKO_URL.format(id=coin_id),
        params={"vs_currency": "usd", "days": days},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()

    prices = payload.get("prices", [])
    volumes = dict(payload.get("total_volumes", []))
    if not prices:
        raise RuntimeError(f"CoinGecko returned no price data for {coin}")

    rows = []
    for ts_ms, price in prices:
        ts = datetime.utcfromtimestamp(ts_ms / 1000)
        rows.append({"timestamp": ts, "close": price, "volume": volumes.get(ts_ms)})

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # CoinGecko's market_chart endpoint only returns a closing price per
    # sample point. We derive open/high/low from neighbouring closes so the
    # market_data table matches the schema in Section 3.3.3.
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values; remove outliers via the IQR method."""
    df = df.copy()
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].ffill().bfill()

    q1, q3 = df["close"].quantile(0.25), df["close"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = df["close"].between(lower, upper)
    # Only drop outliers if it doesn't gut the dataset (keeps demo robust
    # against thin history where IQR bounds can be noisy).
    if mask.sum() >= max(20, int(len(df) * 0.7)):
        df = df[mask].reset_index(drop=True)
    return df


def _save(coin: str, df: pd.DataFrame) -> None:
    existing = {
        row.timestamp
        for row in MarketData.query.filter_by(coin=coin).with_entities(MarketData.timestamp)
    }
    new_rows = []
    for _, r in df.iterrows():
        ts = r["timestamp"].to_pydatetime() if hasattr(r["timestamp"], "to_pydatetime") else r["timestamp"]
        if ts in existing:
            continue
        new_rows.append(
            MarketData(
                coin=coin,
                timestamp=ts,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]) if pd.notna(r["volume"]) else None,
            )
        )
    if new_rows:
        db.session.bulk_save_objects(new_rows)
        db.session.commit()


def _is_fresh(coin: str) -> bool:
    latest = (
        MarketData.query.filter_by(coin=coin)
        .order_by(MarketData.timestamp.desc())
        .first()
    )
    if not latest:
        return False
    return latest.timestamp >= datetime.utcnow() - timedelta(minutes=Config.MARKET_CACHE_MINUTES)


def get_market_data(coin: str, force: bool = False) -> pd.DataFrame:
    """
    Returns a cleaned DataFrame of [timestamp, open, high, low, close, volume]
    for the requested coin, fetching fresh data from CoinGecko when the
    cached copy is stale.
    """
    if coin not in Config.COINGECKO_IDS:
        raise ValueError(f"Unsupported coin: {coin}")

    if force or not _is_fresh(coin):
        raw = _fetch_from_coingecko(coin, Config.MARKET_HISTORY_DAYS)
        cleaned = _clean(raw)
        _save(coin, cleaned)

    rows = (
        MarketData.query.filter_by(coin=coin)
        .order_by(MarketData.timestamp.asc())
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )
    return _clean(df)


def minmax_normalise(series: pd.Series) -> tuple[np.ndarray, float, float]:
    """Simple min-max scaling used ahead of feeding the network (3.5.2)."""
    lo, hi = series.min(), series.max()
    span = (hi - lo) or 1.0
    return ((series - lo) / span).to_numpy(), lo, hi
