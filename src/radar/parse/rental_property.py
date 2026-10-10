"""Rental real estate at book and fair value (賃貸等不動産関係 note) — for the adjusted PBR."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from radar.text_ja import normalize

_UNITS = {"百万円": 1_000_000, "千円": 1_000, "円": 1}
_UNIT = re.compile(r"単位\s*[:：]?\s*(百万円|千円|円)")
_BOOK = re.compile(r"(?:期末残高|年度末残高|末残高)")
_FAIR = re.compile(r"(?:期末時価|期末の時価|末の時価|末時価)")
_NUMBER_RUN = re.compile(r"[△\-−]?[\d,]+")
_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})*")
_OMITTED = ("記載を省略", "重要性が乏しい", "重要性が乏しく", "僅少")
_NOT_APPLICABLE = "該当事項はありません"
_FOLLOWED_BY_NUMBER = r"(?=[△\-−]?\d)"


@dataclass(frozen=True)
class RentalConfig:
    ratio_bounds: tuple[float, float]

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> RentalConfig:
        low, high = raw["rental_ratio_bounds"]
        return cls(ratio_bounds=(float(low), float(high)))


@dataclass(frozen=True)
class RentalNote:
    status: str
    book_value: float | None = None
    fair_value: float | None = None


def parse_rental_note(text: str | None, config: RentalConfig) -> RentalNote:
    if not text or not text.strip():
        return RentalNote("absent")
    flat = re.sub(r"\s+", "", normalize(text))
    if _NOT_APPLICABLE in flat and not re.search(r"\d{1,3},\d{3}", flat):
        return RentalNote("none")
    book, fair = _last_value_after(_BOOK, flat), _last_value_after(_FAIR, flat)
    if book is None and fair is None:
        return RentalNote("immaterial" if any(o in flat for o in _OMITTED) else "unreadable")
    unit = _UNIT.search(flat)
    if book is None or fair is None or unit is None or book <= 0:
        return RentalNote("unreadable")
    low, high = config.ratio_bounds
    if not low <= fair / book <= high:
        return RentalNote("unreadable")
    scale = _UNITS[unit.group(1)]
    return RentalNote("disclosed", book * scale, fair * scale)


def _last_value_after(label: re.Pattern[str], text: str) -> float | None:
    """The LAST number glued after the first `label` that is directly followed by a number (the
    label also appears in prose): the current-year column.
    """
    match = re.search(label.pattern + _FOLLOWED_BY_NUMBER, text)
    if match is None:
        return None
    run = _NUMBER_RUN.match(text, match.end())
    if run is None:
        return None
    digits = run.group(0).lstrip("△-−")
    # Formatted numbers ≥ 1,000 carry separators, so glued columns split greedily after each
    # full 3-digit group («3,075,5051,987,739» → 3,075,505 | 1,987,739). A run of ≥ 4 digits
    # with no separator at all cannot be split: refuse to guess.
    if "," not in digits and len(digits) > 3:
        return None
    tokens = _TOKEN.findall(digits)
    if not tokens or not _tiles(tokens, digits):
        return None
    return float(tokens[-1].replace(",", ""))


def _tiles(tokens: list[str], digits: str) -> bool:
    """True if the tokens rebuild the run exactly (no digit left over or merged)."""
    return "".join(tokens) == digits
