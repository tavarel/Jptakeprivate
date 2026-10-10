"""EDINET API v2 client: daily document lists and document downloads."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from radar.ingest.http import HttpClient
from radar.storage.paths import edinet_doc_path, edinet_doclist_path, ensure_parent

logger = logging.getLogger(__name__)


class DocType(StrEnum):
    """`docTypeCode` values used by this project (full list in the EDINET spec)."""

    ANNUAL_REPORT = "120"  # 有価証券報告書
    ANNUAL_REPORT_AMENDED = "130"  # 訂正有価証券報告書
    SEMI_ANNUAL_REPORT = "160"  # 半期報告書
    EXTRAORDINARY_REPORT = "180"  # 臨時報告書
    TENDER_OFFER = "240"  # 公開買付届出書
    TENDER_OFFER_AMENDED = "250"  # 訂正公開買付届出書
    TENDER_OFFER_WITHDRAWN = "260"  # 公開買付撤回届出書
    TENDER_OFFER_RESULT = "270"  # 公開買付報告書
    OPINION_REPORT = "290"  # 意見表明報告書 (filed by the target)
    LARGE_SHAREHOLDING = "350"  # 大量保有報告書 (new AND change reports)
    LARGE_SHAREHOLDING_AMENDED = "360"  # 訂正大量保有報告書


class FetchType(IntEnum):
    """`type` parameter of the document download endpoint."""

    XBRL = 1
    PDF = 2
    ATTACHMENTS = 3
    ENGLISH = 4
    CSV = 5


class EdinetError(RuntimeError):
    """Any EDINET failure."""


class DocumentNotAvailableError(EdinetError):
    """The API answered with a JSON error (e.g. status 404) instead of the file."""


class DocumentMeta(BaseModel):
    """One row of the document list (only the fields this project uses)."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    doc_id: str = Field(alias="docID")
    edinet_code: str | None = Field(default=None, alias="edinetCode")
    sec_code: str | None = Field(default=None, alias="secCode")
    filer_name: str | None = Field(default=None, alias="filerName")
    doc_type_code: str | None = Field(default=None, alias="docTypeCode")
    doc_description: str | None = Field(default=None, alias="docDescription")
    submit_datetime: datetime = Field(alias="submitDateTime")
    period_end: date | None = Field(default=None, alias="periodEnd")
    issuer_edinet_code: str | None = Field(default=None, alias="issuerEdinetCode")
    subject_edinet_code: str | None = Field(default=None, alias="subjectEdinetCode")
    parent_doc_id: str | None = Field(default=None, alias="parentDocID")
    withdrawal_status: str = Field(default="0", alias="withdrawalStatus")
    xbrl_flag: bool = Field(default=False, alias="xbrlFlag")
    csv_flag: bool = Field(default=False, alias="csvFlag")

    @field_validator("submit_datetime", mode="before")
    @classmethod
    def _parse_submit(cls, value: Any) -> Any:
        # The API uses "YYYY-MM-DD hh:mm" (no seconds, no timezone; times are JST).
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d %H:%M")
        return value

    @field_validator("xbrl_flag", "csv_flag", mode="before")
    @classmethod
    def _flag(cls, value: Any) -> bool:
        return str(value) == "1"

    @property
    def is_withdrawn(self) -> bool:
        return self.withdrawal_status != "0"


def parse_document_list(payload: dict[str, Any]) -> list[DocumentMeta]:
    """Turn a `documents.json?type=2` payload into typed rows."""
    status = str(payload.get("metadata", {}).get("status", ""))
    if status and status != "200":
        message = payload.get("metadata", {}).get("message", "unknown error")
        raise EdinetError(f"EDINET document list returned status {status}: {message}")
    results = payload.get("results") or []
    usable = [row for row in results if row.get("submitDateTime") is not None]
    if len(usable) < len(results):
        logger.info("skipped %d masked documents (no metadata)", len(results) - len(usable))
    return [DocumentMeta.model_validate(row) for row in usable]


def iter_days(start: date, end: date) -> Iterator[date]:
    """Inclusive day range."""
    if end < start:
        raise ValueError("end must be on or after start")
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


class EdinetClient:
    def __init__(
        self,
        api_key: str,
        raw_dir: Path,
        base_url: str = "https://api.edinet-fsa.go.jp/api/v2",
        http: HttpClient | None = None,
    ) -> None:
        if not api_key:
            raise EdinetError("Missing EDINET API key (set RADAR_EDINET_API_KEY in .env)")
        self._api_key = api_key
        self._raw_dir = raw_dir
        self._base_url = base_url.rstrip("/")
        self._http = http or HttpClient()

    # ------------------------------------------------------------------ internals
    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        try:
            return self._http.get(url, params={**params, "Subscription-Key": self._api_key})
        except httpx.HTTPError as exc:
            raise EdinetError(self._redact(f"{type(exc).__name__}: {exc}")) from None

    # ------------------------------------------------------------------ public API
    def list_documents(self, day: date, *, refresh: bool = False) -> list[DocumentMeta]:
        """Document list for one submission day, cached as raw JSON."""
        path = edinet_doclist_path(self._raw_dir, day)
        is_settled = day < date.today() - timedelta(days=1)
        if path.exists() and is_settled and not refresh:
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            response = self._get(
                f"{self._base_url}/documents.json", {"date": day.isoformat(), "type": 2}
            )
            payload = response.json()
            ensure_parent(path).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        return parse_document_list(payload)

    def download(self, doc_id: str, fetch_type: int = FetchType.CSV) -> Path:
        """Download one document and return the cached file path."""
        path = edinet_doc_path(self._raw_dir, doc_id, fetch_type)
        if path.exists():
            return path
        response = self._get(f"{self._base_url}/documents/{doc_id}", {"type": fetch_type})
        content_type = response.headers.get("content-type", "")
        # Known pitfall: a missing document comes back as HTTP 200 with a JSON error body.
        if "application/json" in content_type:
            meta = response.json().get("metadata", {})
            raise DocumentNotAvailableError(
                f"{doc_id} type={fetch_type}: status {meta.get('status')} {meta.get('message')}"
            )
        ensure_parent(path).write_bytes(response.content)
        logger.info("downloaded %s (type=%s, %d bytes)", doc_id, fetch_type, len(response.content))
        return path
