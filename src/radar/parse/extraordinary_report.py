"""Why a 臨時報告書 (extraordinary report, docTypeCode 180) was filed.

Data contract: facts table of one report (element_id, value) → reasons (keys, in report order),
labels (Japanese, joined), alert (any M&A-relevant reason), tender_offer, excerpt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

PARENT_OR_SUBSIDIARY = "parent_or_subsidiary"


@dataclass(frozen=True)
class Reason:
    match: str  # fragment of the element's local name
    reason: str
    label: str
    alert: bool


@dataclass(frozen=True)
class ExtraordinaryConfig:
    reason_text_element: str
    excerpt_chars: int
    reasons: tuple[Reason, ...]
    parent_keyword: str
    parent_label: str
    subsidiary_label: str
    tender_offer_keyword: str

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> ExtraordinaryConfig:
        return cls(
            reason_text_element=str(raw["reason_text_element"]),
            excerpt_chars=int(raw["excerpt_chars"]),
            reasons=tuple(Reason(**r) for r in raw["reasons"]),
            parent_keyword=str(raw["parent_keyword"]),
            parent_label=str(raw["parent_label"]),
            subsidiary_label=str(raw["subsidiary_label"]),
            tender_offer_keyword=str(raw["tender_offer_keyword"]),
        )


def classify_extraordinary(facts: pd.DataFrame, config: ExtraordinaryConfig) -> dict[str, object]:
    """Reasons of one report, from its text-block elements (cover page and 提出理由 excluded)."""
    text = _reason_text(facts, config)
    found: list[tuple[str, str, bool]] = []
    for element in dict.fromkeys(facts["element_id"].astype(str)):
        local = element.split(":")[-1]
        if (
            not local.endswith("TextBlock")
            or "CoverPage" in local
            or element == config.reason_text_element
        ):
            continue
        found.append(_reason_of(local, text, config))
    unique = list(dict.fromkeys(found))
    return {
        "reasons": [key for key, _, _ in unique],
        "labels": " / ".join(label for _, label, _ in unique),
        "alert": any(alert for _, _, alert in unique),
        "tender_offer": config.tender_offer_keyword in text,
        "excerpt": _excerpt(text, config.excerpt_chars),
    }


def _reason_of(local: str, text: str, config: ExtraordinaryConfig) -> tuple[str, str, bool]:
    for rule in config.reasons:
        if rule.match not in local:
            continue
        if rule.reason == PARENT_OR_SUBSIDIARY:  # control changed, or the company bought?
            if config.parent_keyword in text:
                return "parent_change", config.parent_label, True
            return "specified_subsidiary", config.subsidiary_label, False
        return rule.reason, rule.label, rule.alert
    return "other", "その他", False


def _reason_text(facts: pd.DataFrame, config: ExtraordinaryConfig) -> str:
    rows = facts.loc[facts["element_id"] == config.reason_text_element, "value"]
    return "" if rows.empty or pd.isna(rows.iloc[0]) else str(rows.iloc[0])


def _excerpt(text: str, limit: int) -> str:
    clean = " ".join(text.replace("１【提出理由】", "").split())
    return clean if len(clean) <= limit else clean[:limit] + "…"
