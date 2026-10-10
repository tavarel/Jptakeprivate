"""EDINET client behaviour with the network mocked (respx)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from radar.ingest.edinet import (
    DocumentNotAvailableError,
    EdinetClient,
    EdinetError,
    parse_document_list,
)
from radar.ingest.http import HttpClient

BASE = "https://api.edinet-fsa.go.jp/api/v2"
KEY = "secret-test-key"


@pytest.fixture
def client(tmp_path: Path) -> EdinetClient:
    # Very high rate so tests do not sleep.
    return EdinetClient(KEY, tmp_path, BASE, HttpClient(requests_per_second=1000))


def test_parse_document_list(doclist_payload: dict[str, Any]) -> None:
    docs = parse_document_list(doclist_payload)
    assert len(docs) == 4
    first = docs[0]
    assert first.doc_id == "S100TEST1"
    assert first.csv_flag is True
    assert first.period_end == date(2025, 3, 31)
    assert docs[3].is_withdrawn


def test_parse_document_list_skips_masked_rows(doclist_payload: dict[str, Any]) -> None:
    masked = {
        "seqNumber": 99,
        "docID": "S100MASK1",
        "edinetCode": None,
        "docTypeCode": None,
        "submitDateTime": None,
        "withdrawalStatus": "0",
        "xbrlFlag": "0",
        "csvFlag": "0",
        "legalStatus": "0",
    }
    payload = {**doclist_payload, "results": [*doclist_payload["results"], masked]}
    docs = parse_document_list(payload)
    assert [d.doc_id for d in docs] == [r["docID"] for r in doclist_payload["results"]]


def test_parse_document_list_raises_on_api_error() -> None:
    with pytest.raises(EdinetError, match="401"):
        parse_document_list({"metadata": {"status": "401", "message": "Access denied"}})


@respx.mock
def test_list_documents_sends_key_and_caches_past_days(
    client: EdinetClient, doclist_payload: dict[str, Any]
) -> None:
    route = respx.get(f"{BASE}/documents.json").mock(
        return_value=httpx.Response(200, json=doclist_payload)
    )
    day = date(2025, 6, 27)
    assert len(client.list_documents(day)) == 4
    assert route.calls.last.request.url.params["Subscription-Key"] == KEY
    assert route.calls.last.request.url.params["type"] == "2"

    client.list_documents(day)  # second call must come from the raw cache
    assert route.call_count == 1


@respx.mock
def test_download_detects_json_error_body(client: EdinetClient) -> None:
    respx.get(f"{BASE}/documents/S100MISSING").mock(
        return_value=httpx.Response(
            200, json={"metadata": {"status": "404", "message": "Not Found"}}
        )
    )
    with pytest.raises(DocumentNotAvailableError, match="404"):
        client.download("S100MISSING")


@respx.mock
def test_errors_never_leak_the_api_key(client: EdinetClient) -> None:
    respx.get(f"{BASE}/documents.json").mock(return_value=httpx.Response(403))
    with pytest.raises(EdinetError) as excinfo:
        client.list_documents(date(2025, 6, 27), refresh=True)
    assert KEY not in str(excinfo.value)


def test_missing_key_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EdinetError, match="RADAR_EDINET_API_KEY"):
        EdinetClient("", tmp_path)
