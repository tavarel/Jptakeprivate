"""Ownership-structure features from the top-10 shareholders.

Data contract: holders: name, ratio (0–1 or None), is_organization[, address] (one filing's top-10)
documents: filer_name, sec_code, submit_datetime (the EDINET index) Out (`ownership_features`):
largest_listed_holder_stake, largest_listed_holder_name, top10_concentration_ex_custodian. None when
unknown (no shareholder table / no ratios).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from radar.text_ja import normalize

_LEGAL_FORMS = re.compile(r"株式会社|\(株\)|有限会社|合同会社|\s")


@dataclass(frozen=True)
class HolderPatterns:
    """Ordered (holder_type, name substrings); the first matching type wins."""

    ordered: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> HolderPatterns:
        """From config/ownership.yaml (`holder_types` mapping, in file order)."""
        types = raw["holder_types"]
        return cls(tuple((name, tuple(str(p) for p in types[name])) for name in types))


def classify_holder(name: str, is_organization: bool, patterns: HolderPatterns) -> str:
    """treasury / employee_plan / custodian / foundation / financial / company / person."""
    text = normalize(name).upper()
    for holder_type, substrings in patterns.ordered:
        if any(normalize(s).upper() in text for s in substrings):
            return holder_type
    return "company" if is_organization else "person"


def company_key(name: str) -> str:
    """Name without legal form, spaces or width differences: 'トヨタ自動車(株)' → 'トヨタ自動車'."""
    return _LEGAL_FORMS.sub("", normalize(name))


def listed_company_codes(documents: pd.DataFrame, as_of: date) -> dict[str, str]:
    """company_key → EDINET code of every filer known to be listed before `as_of` (the latest filing
    wins).
    """
    cutoff = datetime.combine(as_of, time.min)
    known = documents[
        documents["sec_code"].notna() & (pd.to_datetime(documents["submit_datetime"]) < cutoff)
    ].sort_values("submit_datetime")
    return {
        company_key(str(n)): str(c)
        for n, c in zip(known["filer_name"], known["edinet_code"], strict=True)
    }


def parent_prior_offers(documents: pd.DataFrame, parent_code: str, as_of: date) -> int:
    """Third-party tender offers (240) FILED BY the parent itself before `as_of`."""
    cutoff = datetime.combine(as_of, time.min)
    offers = documents[
        (documents["doc_type_code"] == "240")
        & (documents["edinet_code"] == parent_code)
        & (documents["subject_edinet_code"] != parent_code)  # not a self-tender
        & documents["subject_edinet_code"].notna()
        & (documents["withdrawal_status"].astype(str) == "0")
        & (pd.to_datetime(documents["submit_datetime"]) < cutoff)
    ]
    return int(offers["subject_edinet_code"].nunique())


def holder_portfolio_counts(holders: pd.DataFrame, min_stake: float) -> dict[str, int]:
    """company_key of a holder → number of companies in which it holds ≥ `min_stake`."""
    big = holders[holders["ratio"] >= min_stake]
    keys = big["name"].map(lambda n: company_key(str(n)))
    return {str(k): int(v) for k, v in big.groupby(keys)["edinet_code"].nunique().items()}


def listed_company_keys(documents: pd.DataFrame, as_of: date) -> frozenset[str]:
    """Keys of companies known to be listed before `as_of`: filers of any document with a securities
    code submitted strictly before `as_of` (point-in-time, no external list needed).
    """
    cutoff = datetime.combine(as_of, time.min)
    known = documents[
        documents["sec_code"].notna() & (pd.to_datetime(documents["submit_datetime"]) < cutoff)
    ]
    return frozenset(company_key(str(n)) for n in known["filer_name"].dropna())


def _is_known_homonym(name: str, address: object, homonyms: Mapping[str, Sequence[str]]) -> bool:
    """True if this holder is a known homonym of a listed company (matched by address)."""
    fragments = homonyms.get(company_key(name), ())
    text = normalize(address) if isinstance(address, str) else ""
    return bool(text) and any(normalize(f) in text for f in fragments)


def ownership_features(
    holders: pd.DataFrame,
    listed_keys: frozenset[str],
    patterns: HolderPatterns,
    homonyms: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Listed-parent stake and top-10 concentration for one filing."""
    if holders.empty:
        return dict.fromkeys(
            [
                "largest_listed_holder_stake",
                "largest_listed_holder_name",
                "top10_concentration_ex_custodian",
            ]
        )
    types = [
        classify_holder(str(n), bool(o), patterns)
        for n, o in zip(holders["name"], holders["is_organization"], strict=True)
    ]
    frame = holders.assign(holder_type=types)
    addresses = frame["address"] if "address" in frame.columns else pd.Series(None, frame.index)
    is_listed = pd.Series(
        [
            t == "company"
            and company_key(str(n)) in listed_keys
            and not _is_known_homonym(str(n), a, homonyms or {})
            for t, n, a in zip(frame["holder_type"], frame["name"], addresses, strict=True)
        ],
        index=frame.index,
    )
    listed = frame[is_listed]
    known_listed = listed[listed["ratio"].notna()].sort_values("ratio", ascending=False)
    if listed.empty:
        stake, parent = 0.0, None
    elif known_listed.empty:
        stake, parent = None, None
    else:
        stake, parent = float(known_listed["ratio"].iloc[0]), str(known_listed["name"].iloc[0])

    beneficial = frame[
        ~frame["holder_type"].isin(["custodian", "treasury"]) & frame["ratio"].notna()
    ]
    concentration = float(beneficial["ratio"].sum()) if frame["ratio"].notna().any() else None
    return {
        "largest_listed_holder_stake": stake,
        "largest_listed_holder_name": parent,
        "top10_concentration_ex_custodian": concentration,
    }


def _person_key(name: str) -> str:
    return re.sub(r"\s", "", normalize(name))


def tradable_share_ratio(
    holders: pd.DataFrame,
    officers: pd.DataFrame,
    *,
    outstanding: float | None,
    issued: float | None,
    major_holder: float,
    patterns: HolderPatterns,
) -> float | None:
    """Approximate tradable share ratio (流通株式比率) of listed shares, TSE-style."""
    if holders.empty or not outstanding or not issued:
        return None
    is_custodian = pd.Series(
        [
            classify_holder(str(n), bool(o), patterns) == "custodian"
            for n, o in zip(holders["name"], holders["is_organization"], strict=True)
        ],
        index=holders.index,
    )
    major = holders[(holders["ratio"] >= major_holder) & ~is_custodian]
    major_names = {_person_key(str(n)) for n in major["name"]}
    officer_shares = 0.0
    if "shares_held" in officers.columns:
        is_major = officers["name"].map(lambda n: _person_key(str(n)) in major_names).astype(bool)
        others = officers.loc[~is_major]  # .loc + bool dtype: an empty mask still filters rows
        officer_shares = float(others["shares_held"].fillna(0.0).sum())
    not_tradable = float(major["ratio"].sum()) + officer_shares / outstanding
    return max(0.0, 1.0 - not_tradable) * outstanding / issued


def listing_criteria_margin(
    ratio: float | None,
    market_cap: float | None,
    segment: str | None,
    criteria: Mapping[str, Mapping[str, float]],
) -> float | None:
    """How far above the tighter of the two TSE tradable-share minimums the company sits: min(ratio
    / minimum ratio, tradable value / minimum value) − 1.
    """
    rule = criteria.get(segment or "")
    if rule is None or ratio is None or market_cap is None:
        return None
    return min(ratio / rule["ratio"], ratio * market_cap / rule["value"]) - 1


PARENT_COLUMNS = {
    "pbr": "parent_pbr",
    "net_cash_to_market_cap": "parent_net_cash_to_market_cap",
    "activist_filers_365d": "parent_activist_filers_365d",
}


def parent_features(table: pd.DataFrame, listed_codes: Mapping[str, str]) -> pd.DataFrame:
    """The listed parent's own valuation and pressure, from its row in the SAME feature table."""
    by_code = table.set_index("edinet_code")
    rows = []
    for name in table["largest_listed_holder_name"]:
        code = listed_codes.get(company_key(str(name))) if isinstance(name, str) else None
        parent = by_code.loc[code] if code is not None and code in by_code.index else None
        rows.append(
            {
                target: (None if parent is None or pd.isna(parent[source]) else parent[source])
                for source, target in PARENT_COLUMNS.items()
            }
        )
    return pd.DataFrame(rows, index=table.index, columns=list(PARENT_COLUMNS.values()))


def ownership_changes(
    now: pd.DataFrame,
    before: pd.DataFrame | None,
    listed_keys: frozenset[str],
    patterns: HolderPatterns,
    homonyms: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float | None]:
    """Change since the previous annual report: the largest listed holder's stake (same holder name
    in both reports) and the top-10 concentration ex custodians.
    """
    empty: dict[str, float | None] = {"listed_holder_stake_change": None, "top10_change": None}
    if before is None or before.empty or now.empty:
        return empty
    current = ownership_features(now, listed_keys, patterns, homonyms)
    previous = ownership_features(before, listed_keys, patterns, homonyms)
    top10 = None
    if current["top10_concentration_ex_custodian"] is not None and (
        previous["top10_concentration_ex_custodian"] is not None
    ):
        top10 = (
            current["top10_concentration_ex_custodian"]
            - previous["top10_concentration_ex_custodian"]
        )
    stake = None
    name, value = current["largest_listed_holder_name"], current["largest_listed_holder_stake"]
    if name is not None and value is not None:
        same = before[before["name"].map(company_key) == company_key(name)]["ratio"].dropna()
        stake = value - float(same.iloc[0]) if not same.empty else None
    return {"listed_holder_stake_change": stake, "top10_change": top10}
