"""Balance-sheet / P&L items needed for valuation features, from an annual-report facts table."""

from __future__ import annotations

import pandas as pd

from radar.parse.elements import ElementMap, first_present, to_number
from radar.parse.rental_property import RentalConfig, parse_rental_note

DURATION_ITEMS = frozenset({"operating_income"})


def _value(facts: pd.DataFrame, element_id: str, contexts: list[str]) -> float | None:
    """Value of an element in the first context (in preference order) where it exists."""
    rows = facts[facts["element_id"] == element_id]
    for context in contexts:
        match = rows[rows["context_id"] == context]
        if not match.empty:
            return to_number(match.iloc[0]["value"])
    return None


def extract_financials(
    facts: pd.DataFrame, elements: ElementMap, rental: RentalConfig | None = None
) -> dict[str, float | bool | str | None]:
    """One flat record of financial items for the current fiscal year."""
    is_jgaap = _is_jgaap(facts, elements)
    ctx = elements.contexts
    instant = [ctx["instant_consolidated"], ctx["instant_non_consolidated"]]
    duration = [ctx["duration_consolidated"], ctx["duration_non_consolidated"]]

    record: dict[str, float | bool | str | None] = {}
    for key, candidates in {**elements.financials, **elements.shares}.items():
        element_id = first_present(facts, candidates)
        if element_id is None or (is_jgaap is False and key in elements.financials):
            record[key] = None
            continue
        contexts = duration if key in DURATION_ITEMS else instant
        # Share counts are tagged on the filing-date or year-end context; try both families.
        # The common-share context comes first: with preferred shares outstanding, the bare
        # FilingDateInstant row adds all classes, but only common shares trade (きらぼしFG).
        if key in elements.shares:
            contexts = [ctx["filing_date_common_shares"], *instant, "FilingDateInstant", *duration]
        record[key] = _value(facts, element_id, contexts)

    # Per-share figures are the filer's own, for every standard: non-consolidated first.
    for key, candidates in elements.per_share.items():
        element_id = first_present(facts, candidates)
        record[key] = None if element_id is None else _value(facts, element_id, duration[::-1])

    consolidated = _dei_value(facts, elements.dei["consolidated"])
    record["is_consolidated"] = None if consolidated is None else consolidated.lower() == "true"
    record["is_jgaap"] = is_jgaap
    # Head-office address (本店の所在の場所): tells apart listed companies from homonyms.
    record["hq_address"] = _dei_value(facts, elements.dei["hq_address"])
    record.update(_rental(facts, elements, rental, is_jgaap))
    return record


def _rental(
    facts: pd.DataFrame, elements: ElementMap, config: RentalConfig | None, is_jgaap: bool | None
) -> dict[str, float | str | None]:
    """Rental real estate at book and fair value (賃貸等不動産, adjusted PBR)."""
    empty: dict[str, float | str | None] = dict.fromkeys(
        ("rental_book_value", "rental_fair_value", "rental_note_status")
    )
    if config is None or is_jgaap is False:
        return empty
    element_id = first_present(facts, elements.rental_note)
    text = None
    if element_id is not None:
        rows = facts[facts["element_id"] == element_id]
        current = rows[rows["context_id"].astype(str).str.startswith("CurrentYear")]
        text = str((current if not current.empty else rows).iloc[0]["value"])
    note = parse_rental_note(text, config)
    return {
        "rental_book_value": note.book_value,
        "rental_fair_value": note.fair_value,
        "rental_note_status": note.status,
    }


def _is_jgaap(facts: pd.DataFrame, elements: ElementMap) -> bool | None:
    """True/False from the filing's own accounting-standard tag (会計基準), None if absent."""
    standard = _dei_value(facts, elements.dei["accounting_standard"])
    return None if standard is None else standard == elements.dei["jgaap_value"]


def _dei_value(facts: pd.DataFrame, element_id: str) -> str | None:
    """A cover-page (DEI) value as stripped text; None if the element is absent or empty."""
    rows = facts[facts["element_id"] == element_id]
    if rows.empty:
        return None
    text = str(rows.iloc[0]["value"]).strip()
    return text or None
