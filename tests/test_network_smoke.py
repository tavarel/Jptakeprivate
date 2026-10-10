"""Smoke test against the REAL EDINET API."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from radar.config import get_settings
from radar.ingest.edinet import EdinetClient
from radar.ingest.http import HttpClient

KNOWN_DAY = date(2025, 6, 2)


@pytest.mark.network
def test_edinet_lists_a_known_day(tmp_path: Path) -> None:
    settings = get_settings()
    if settings.edinet_api_key is None:
        pytest.skip("RADAR_EDINET_API_KEY not set")
    client = EdinetClient(
        settings.edinet_api_key.get_secret_value(),
        tmp_path,
        settings.edinet_base_url,
        HttpClient(settings.requests_per_second, settings.http_timeout_s),
    )

    docs = client.list_documents(KNOWN_DAY)

    assert len(docs) > 50
    # A day's list also repeats OLDER documents whose metadata was edited that day
    # (docInfoEditStatus "1"), so submit dates are <= the list day, not always equal to it.
    assert all(doc.submit_datetime.date() <= KNOWN_DAY for doc in docs)
    assert any(doc.doc_type_code == "120" for doc in docs)  # annual reports exist that day
