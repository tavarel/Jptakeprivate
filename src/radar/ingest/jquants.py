"""J-Quants API V2 client (JPX official market data): listed companies and daily prices."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from radar.ingest.http import HttpClient
from radar.storage.paths import (
    ensure_parent,
    jquants_bars_path,
    jquants_master_path,
    jquants_topix_path,
)

logger = logging.getLogger(__name__)

EQUITIES_MASTER_PATH = "/equities/master"  # V1: /listed/info
DAILY_BARS_PATH = "/equities/bars/daily"  # V1: /prices/daily_quotes
TOPIX_PATH = "/indices/bars/daily/topix"  # Light plan and above
PAGINATION_KEY = "pagination_key"

PRICE_COLUMNS = ["code", "date", "close", "adj_factor", "trading_value"]
LISTING_COLUMNS = ["code", "date", "company_name", "market_segment", "sector33"]


class JQuantsError(RuntimeError):
    pass


class JQuantsClient:
    def __init__(
        self,
        api_key: str,
        raw_dir: Path,
        base_url: str = "https://api.jquants.com/v2",
        http: HttpClient | None = None,
        requests_per_minute: float = 4,
    ) -> None:
        if not api_key:
            raise JQuantsError("Missing J-Quants API key (set RADAR_JQUANTS_API_KEY in .env)")
        self._raw_dir = raw_dir
        self._base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            requests_per_second=requests_per_minute / 60, headers={"x-api-key": api_key}
        )

    def _pages(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        query = dict(params)
        while True:
            try:
                payload: dict[str, Any] = self._http.get(f"{self._base_url}{path}", query).json()
            except httpx.HTTPStatusError as exc:
                # The key travels in a header, never in the URL, so the message is safe to show.
                message = _error_message(exc.response)
                raise JQuantsError(
                    f"HTTP {exc.response.status_code} on {path}: {message}"
                ) from None
            except httpx.HTTPError as exc:
                raise JQuantsError(f"{type(exc).__name__} on {path}") from None
            yield from payload.get("data", [])
            next_key = payload.get(PAGINATION_KEY)
            if not next_key:
                return
            query[PAGINATION_KEY] = next_key

    def _cached(self, path: Path, endpoint: str, day: date) -> list[dict[str, Any]]:
        """Rows for one day: from data/raw if already fetched, else from the API (then cached)."""
        if path.exists():
            rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
            return rows
        rows = list(self._pages(endpoint, {"date": day.isoformat()}))
        ensure_parent(path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        logger.info("fetched %s for %s: %d rows", endpoint, day, len(rows))
        return rows

    def daily_bars(self, day: date) -> list[dict[str, Any]]:
        """Daily bars of every listed stock on `day` (empty on market holidays)."""
        return self._cached(jquants_bars_path(self._raw_dir, day), DAILY_BARS_PATH, day)

    def equities_master(self, day: date) -> list[dict[str, Any]]:
        """Companies listed on `day`, with market segment (next business day if `day` is not)."""
        return self._cached(jquants_master_path(self._raw_dir, day), EQUITIES_MASTER_PATH, day)

    def topix(self, start: date, end: date, *, refresh: bool = False) -> list[dict[str, Any]]:
        """TOPIX daily bars for [start, end] within ONE calendar year, cached per year."""
        if start.year != end.year:
            raise ValueError("one calendar year per call")
        path = jquants_topix_path(self._raw_dir, start.year)
        if path.exists() and not refresh:
            rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
            return rows
        params = {"from": start.isoformat(), "to": end.isoformat()}
        rows = list(self._pages(TOPIX_PATH, params))
        ensure_parent(path).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("message", ""))[:300]
    except ValueError:
        return ""


def bars_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw bars → code, date, close (UNADJUSTED `C`), adj_factor, trading_value (`Va`, yen)."""
    frame = pd.DataFrame(rows, columns=["Code", "Date", "C", "AdjFactor", "Va"])
    frame.columns = PRICE_COLUMNS
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    frame["trading_value"] = pd.to_numeric(frame["trading_value"], errors="coerce")
    return frame


def master_frame(rows: list[dict[str, Any]], segment_names: Mapping[str, str]) -> pd.DataFrame:
    """Raw master → code, date, company_name, market_segment (Prime / Standard / Growth / …),
    sector33 (TSE 33-sector name, 業種).
    """
    frame = pd.DataFrame(rows, columns=["Code", "Date", "CoName", "MktNm", "S33Nm"])
    frame.columns = LISTING_COLUMNS
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["market_segment"] = frame["market_segment"].map(lambda s: segment_names.get(s, s))
    return frame


def topix_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw TOPIX bars → date, close."""
    frame = pd.DataFrame(rows, columns=["Date", "C"])
    frame.columns = ["date", "close"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame
