"""Typed access to `config/xbrl_elements.yaml` and helpers to pick facts out of a facts table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from radar.config import CONFIG_DIR, load_yaml


@dataclass(frozen=True)
class ElementMap:
    officers: dict[str, list[str]]
    executive_officers: dict[str, list[str]]
    officers_proposal: dict[str, list[str]]
    major_shareholders: dict[str, list[str]]
    financials: dict[str, list[str]]
    shares: dict[str, list[str]]
    contexts: dict[str, str]
    dei: dict[str, str]
    excerpts: dict[str, str]
    rental_note: list[str]
    per_share: dict[str, list[str]]

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> ElementMap:
        raw: dict[str, Any] = load_yaml("xbrl_elements", config_dir)
        return cls(
            officers=raw["officers"],
            executive_officers=raw["executive_officers"],
            officers_proposal=raw["officers_proposal"],
            major_shareholders=raw["major_shareholders"],
            financials=raw["financials"],
            shares=raw["shares"],
            contexts=raw["contexts"],
            dei=raw["dei"],
            excerpts=raw.get("excerpts") or {},
            rental_note=raw.get("rental_note") or [],
            per_share=raw.get("per_share") or {},
        )


def first_present(facts: pd.DataFrame, candidates: list[str]) -> str | None:
    """First candidate element ID that actually appears in this filing."""
    present = set(facts["element_id"].unique())
    for element_id in candidates:
        if element_id in present:
            return element_id
    return None


def to_number(value: object) -> float | None:
    """'1,234' / '△1,234' (Japanese negative sign) / '' → float or None."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    negative = text.startswith(("△", "▲", "-"))
    text = text.lstrip("△▲-")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number
