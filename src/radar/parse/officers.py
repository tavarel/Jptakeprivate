"""Officers (役員の状況) from an annual-report facts table."""

from __future__ import annotations

import re

import pandas as pd

from radar.parse.elements import ElementMap, first_present, to_number
from radar.text_ja import join_spaced_characters, normalize, parse_japanese_date

OFFICER_COLUMNS = [
    "person_key",
    "name",
    "title",
    "date_of_birth",
    "shares_held",
    "is_representative",
    "role",
]


# 副社長 / 副会長 (vice president / vice chair) contain 社長 / 会長: not the top role.
_CHAIR = re.compile(r"(?<!副)会長")
_PRESIDENT = re.compile(r"(?<!副)社長")


def _compact(title: str) -> str:
    """Title without any whitespace: some filings space out words (「代 表取締役社 長」)."""
    return re.sub(r"\s+", "", normalize(title))


def classify_role(title: object) -> str:
    """'chair' / 'president' / 'other' from a Japanese job title (NaN/None → 'other')."""
    if not isinstance(title, str) or not title:
        return "other"
    text = _compact(title)
    if _CHAIR.search(text):
        return "chair"
    if _PRESIDENT.search(text) or "CEO" in text.upper():
        return "president"
    return "other"


def is_representative(title: object) -> bool:
    """Representative director / executive officer (代表取締役 / 代表執行役)."""
    if not isinstance(title, str) or not title:
        return False
    text = _compact(title)
    return "代表取締役" in text or "代表執行役" in text


def extract_officers(facts: pd.DataFrame, elements: ElementMap) -> pd.DataFrame:
    """One row per officer."""
    blocks = (elements.officers, elements.executive_officers)
    current = _merge([_officer_table(facts, b) for b in blocks])
    if current["is_representative"].any():
        return current
    # No representative among current officers: the re-elected ones are AGM nominees (quirk
    # documented in config/xbrl_elements.yaml). Add nominees not already listed, by name.
    nominees = _merge([_officer_table(facts, elements.officers_proposal)])
    new = nominees[~nominees["name"].isin(current["name"])]
    if new.empty:
        return current
    return pd.concat([current, new], ignore_index=True)[OFFICER_COLUMNS]


def _merge(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Rows of the same person (one context) merged; titles joined so both tables are seen."""
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame(columns=OFFICER_COLUMNS).astype({"is_representative": bool})
    merged = (
        pd.concat(tables, ignore_index=True)
        .groupby("person_key", as_index=False, sort=True)
        .agg(
            name=("name", "first"),
            title=("title", _join_titles),
            date_of_birth=("date_of_birth", "first"),
            shares_held=("shares_held", "first"),
        )
    )
    merged["is_representative"] = merged["title"].map(is_representative)
    merged["role"] = merged["title"].map(classify_role)
    return merged[OFFICER_COLUMNS].reset_index(drop=True)


def _officer_table(facts: pd.DataFrame, block: dict[str, list[str]]) -> pd.DataFrame:
    """One officer table (directors or executive officers) pivoted to one row per context."""
    keys = ("name", "title", "date_of_birth", "shares_held")
    fields = {key: first_present(facts, block.get(key, [])) for key in keys}
    if fields["name"] is None:
        return pd.DataFrame(columns=["person_key", *keys])

    wanted = {element_id: key for key, element_id in fields.items() if element_id}
    subset = facts[facts["element_id"].isin(wanted)].copy()
    subset["field"] = subset["element_id"].map(wanted)
    wide = (
        subset.pivot_table(index="context_id", columns="field", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"context_id": "person_key"})
    )
    for key in keys:
        if key not in wide.columns:
            wide[key] = None
    wide = wide[wide["name"].map(lambda v: isinstance(v, str) and v.strip() != "")].copy()
    wide["name"] = wide["name"].map(join_spaced_characters)
    wide["date_of_birth"] = wide["date_of_birth"].map(parse_japanese_date)
    wide["shares_held"] = wide["shares_held"].map(to_number)
    return wide[["person_key", *keys]]


def _join_titles(titles: pd.Series) -> str | None:
    unique = [t for t in dict.fromkeys(titles) if isinstance(t, str) and t]
    return " / ".join(unique) if unique else None
