"""Polite HTTP client: rate limiting + retries with exponential backoff."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    """Enforces a minimum interval between calls (thread-safe)."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


class HttpClient:
    """Thin wrapper around `httpx.Client` used by every ingest module."""

    def __init__(
        self,
        requests_per_second: float = 0.5,
        timeout_s: float = 60.0,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._limiter = RateLimiter(requests_per_second)
        self._client = httpx.Client(
            timeout=timeout_s,
            headers=dict(headers or {}),
            follow_redirects=True,
            transport=transport,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def get(self, url: str, params: Mapping[str, Any] | None = None) -> httpx.Response:
        self._limiter.wait()
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
