"""J-Quants V2 client with the network mocked (respx)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from radar import pipeline
from radar.config import Settings
from radar.ingest.http import HttpClient
from radar.ingest.jquants import (
    JQuantsClient,
    JQuantsError,
    bars_frame,
    master_frame,
    topix_frame,
)

BASE = "https://api.jquants.com/v2"
DAY = date(2025, 6, 30)
BAR = {
    "Date": "2025-06-30", "Code": "99990", "O": 1000.0, "H": 1010.0, "L": 990.0, "C": 1005.0,
    "UL": "0", "LL": "0", "Vo": 1000.0, "Va": 1005000.0, "AdjFactor": 1.0, "AdjO": 1000.0,
    "AdjH": 1010.0, "AdjL": 990.0, "AdjC": 1005.0, "AdjVo": 1000.0, "MktCap": 100.0, "ExRT": None,
}  # fmt: skip
LISTING = {
    "Date": "2025-06-30", "Code": "99990", "CoName": "サンプル工業", "CoNameEn": "SAMPLE",
    "S17": "1", "S17Nm": "食品", "S33": "0050", "S33Nm": "水産・農林業",
    "ScaleCat": "TOPIX Small 2", "Mkt": "0112", "MktNm": "スタンダード", "Mrgn": "2",
    "MrgnNm": "貸借", "ProdCat": "011",
}  # fmt: skip


@pytest.fixture
def client(tmp_path: Path) -> JQuantsClient:
    http = HttpClient(requests_per_second=1000, headers={"x-api-key": "secret-test-key"})
    return JQuantsClient("secret-test-key", tmp_path, BASE, http)


@respx.mock
def test_daily_bars_follows_pagination_and_caches(client: JQuantsClient) -> None:
    route = respx.get(f"{BASE}/equities/bars/daily").mock(
        side_effect=[
            httpx.Response(200, json={"data": [BAR], "pagination_key": "next"}),
            httpx.Response(200, json={"data": [{**BAR, "Code": "99980"}]}),
        ]
    )
    rows = client.daily_bars(DAY)
    assert [r["Code"] for r in rows] == ["99990", "99980"]
    assert route.calls[0].request.headers["x-api-key"] == "secret-test-key"
    assert route.calls[1].request.url.params["pagination_key"] == "next"

    assert client.daily_bars(DAY) == rows  # second call: from data/raw, no request
    assert route.call_count == 2


@respx.mock
def test_out_of_plan_date_raises_with_the_api_message(client: JQuantsClient) -> None:
    message = "Your subscription covers the following dates: 2024-07-16 ~ 2026-07-16."
    respx.get(f"{BASE}/equities/bars/daily").mock(
        return_value=httpx.Response(400, json={"message": message})
    )
    with pytest.raises(JQuantsError, match="2024-07-16 ~ 2026-07-16") as info:
        client.daily_bars(date(2026, 9, 30))
    assert "secret-test-key" not in str(info.value)


@respx.mock
def test_errors_are_not_cached(client: JQuantsClient, tmp_path: Path) -> None:
    respx.get(f"{BASE}/equities/master").mock(return_value=httpx.Response(400, json={}))
    with pytest.raises(JQuantsError):
        client.equities_master(DAY)
    assert not list(tmp_path.rglob("*.json"))


def test_missing_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(JQuantsError, match="RADAR_JQUANTS_API_KEY"):
        JQuantsClient("", tmp_path)


def test_bars_frame_keeps_unadjusted_close_and_factor() -> None:
    split_day = {**BAR, "C": 500.0, "AdjFactor": 0.5, "AdjC": 500.0}
    no_trade = {**BAR, "Code": "99980", "C": None}
    frame = bars_frame([BAR, split_day, no_trade])
    assert list(frame.columns) == ["code", "date", "close", "adj_factor", "trading_value"]
    assert frame.loc[0, "trading_value"] == 1005000.0
    assert frame.loc[0, "date"] == DAY
    assert frame["close"].tolist()[:2] == [1005.0, 500.0]
    assert frame.loc[1, "adj_factor"] == 0.5
    assert pd.isna(frame.loc[2, "close"])  # no trade → missing, not 0


def test_master_frame_translates_segment_names() -> None:
    other = {**LISTING, "Code": "99980", "MktNm": "TOKYO PRO MARKET"}
    frame = master_frame([LISTING, other], {"スタンダード": "Standard"})
    assert frame["market_segment"].tolist() == ["Standard", "TOKYO PRO MARKET"]
    assert frame.loc[0, "company_name"] == "サンプル工業"
    assert frame.loc[0, "date"] == DAY
    assert frame.loc[0, "sector33"] == "水産・農林業"


# ------------------------------------------------------------------ pipeline: staging tables
@respx.mock
def test_sync_prices_one_call_per_weekday_and_merges(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    http = HttpClient(requests_per_second=1000)
    client = JQuantsClient("k", settings.raw_dir, BASE, http)
    route = respx.get(f"{BASE}/equities/bars/daily").mock(
        side_effect=lambda request: httpx.Response(
            200, json={"data": [{**BAR, "Date": request.url.params["date"]}]}
        )
    )
    assert pipeline.sync_prices(client, settings, date(2025, 6, 27), date(2025, 6, 30)) == 2
    assert route.call_count == 2
    pipeline.sync_prices(client, settings, date(2025, 6, 30), date(2025, 6, 30))  # overlap
    prices = pipeline.load_staging(settings, "prices")
    assert prices["date"].tolist() == [date(2025, 6, 27), date(2025, 6, 30)]  # no duplicates


@respx.mock
def test_sync_listings_month_ends(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    client = JQuantsClient("k", settings.raw_dir, BASE, HttpClient(requests_per_second=1000))
    route = respx.get(f"{BASE}/equities/master").mock(
        return_value=httpx.Response(200, json={"data": [LISTING]})
    )
    assert pipeline.sync_listings(client, settings, date(2025, 5, 15), date(2025, 7, 31)) == 3
    assert [c.request.url.params["date"] for c in route.calls] == [
        "2025-05-31",
        "2025-06-30",
        "2025-07-31",
    ]
    listings = pipeline.load_staging(settings, "listings")
    assert listings["market_segment"].unique().tolist() == ["Standard"]


@respx.mock
def test_sync_prices_keeps_days_fetched_before_an_error(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    client = JQuantsClient("k", settings.raw_dir, BASE, HttpClient(requests_per_second=1000))
    respx.get(f"{BASE}/equities/bars/daily").mock(
        side_effect=[
            httpx.Response(200, json={"data": [{**BAR, "Date": "2025-06-27"}]}),
            httpx.Response(403, json={"message": "The incoming api key is invalid or expired."}),
        ]
    )
    with pytest.raises(JQuantsError, match="invalid or expired"):
        pipeline.sync_prices(client, settings, date(2025, 6, 27), date(2025, 6, 30))
    assert pipeline.load_staging(settings, "prices")["date"].tolist() == [date(2025, 6, 27)]


@respx.mock
def test_topix_cached_per_year(client: JQuantsClient) -> None:
    route = respx.get(f"{BASE}/indices/bars/daily/topix").mock(
        return_value=httpx.Response(
            200, json={"data": [{"Date": "2025-06-30", "O": 1, "H": 1, "L": 1, "C": 2850.5}]}
        )
    )
    rows = client.topix(date(2025, 3, 3), date(2025, 12, 31))
    assert route.calls[0].request.url.params["from"] == "2025-03-03"
    assert client.topix(date(2025, 3, 3), date(2025, 12, 31)) == rows and route.call_count == 1
    with pytest.raises(ValueError, match="one calendar year"):
        client.topix(date(2024, 12, 1), date(2025, 1, 31))
    frame = topix_frame(rows)
    assert frame.loc[0, "date"] == DAY and frame.loc[0, "close"] == 2850.5
