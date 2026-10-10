"""Top-10 major shareholders (大株主の状況) from an annual-report facts table."""

from __future__ import annotations

import re

import pandas as pd

from radar.parse.elements import ElementMap, first_present, to_number
from radar.text_ja import is_organization, normalize

SHAREHOLDER_COLUMNS = ["rank", "name", "shares_held", "ratio", "is_organization", "address"]
_RANK = re.compile(r"No(\d+)MajorShareholdersMember")
_TOTAL_ROW_NAMES = frozenset({"計", "合計", "総計"})


def rank_from_context(context_id: str) -> int | None:
    match = _RANK.search(context_id)
    return int(match.group(1)) if match else None


def normalize_ratio(value: float | None) -> float | None:
    """Return a fraction in [0, 1]."""
    if value is None:
        return None
    return value / 100 if value > 1 else value


def extract_major_shareholders(facts: pd.DataFrame, elements: ElementMap) -> pd.DataFrame:
    fields: dict[str, str | None] = {
        key: first_present(facts, elements.major_shareholders.get(key, []))
        for key in ("name", "shares_held", "ratio", "address")
    }
    if fields["name"] is None:
        return pd.DataFrame(columns=SHAREHOLDER_COLUMNS)

    wanted = {element_id: key for key, element_id in fields.items() if element_id}
    subset = facts[facts["element_id"].isin(wanted)].copy()
    subset["field"] = subset["element_id"].map(wanted)
    wide = subset.pivot_table(
        index="context_id", columns="field", values="value", aggfunc="first"
    ).reset_index()
    for key in ("name", "shares_held", "ratio", "address"):
        if key not in wide.columns:
            wide[key] = None

    wide["rank"] = wide["context_id"].map(rank_from_context)
    wide = wide[wide["name"].map(lambda v: isinstance(v, str) and v.strip() != "")]
    wide["name"] = wide["name"].map(normalize)
    # Some filings tag the total row (計 / 合計) like a shareholder, with a name and a rank.
    wide = wide[~wide["name"].isin(_TOTAL_ROW_NAMES)]
    wide["shares_held"] = wide["shares_held"].map(to_number)
    wide["ratio"] = wide["ratio"].map(to_number).map(normalize_ratio)
    wide["is_organization"] = wide["name"].map(is_organization)
    # Address of the holder: tells apart companies with the same name (features/ownership).
    wide["address"] = wide["address"].map(lambda v: normalize(v) if isinstance(v, str) else None)
    wide = wide.sort_values("rank", na_position="last")
    return wide[SHAREHOLDER_COLUMNS].reset_index(drop=True)
