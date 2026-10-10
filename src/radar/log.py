"""Logging setup."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # httpx logs every request URL at INFO, and EDINET URLs contain the API key as a query
    # parameter. Keep httpx quiet so the key never lands in logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
