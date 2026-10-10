"""Activist-investor signal from large-shareholding reports.

Data contract: documents: doc_type_code, filer_name, issuer_edinet_code (the company held),
submit_datetime Out: one row per company with ≥1 activist report in [as_of − 365 days, as_of):
edinet_code, activist_filers_365d (distinct activists), activist_names (' / '-joined)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from radar.text_ja import normalize

LARGE_SHAREHOLDING = "350"
WINDOW_DAYS = 365
_LEGAL_FORM = re.compile(r"株式会社|\(株\)")


ActivistEntry = str | Mapping[str, Any]


def _name(entry: ActivistEntry) -> str:
    return str(entry["name"]) if isinstance(entry, Mapping) else str(entry)


def active_patterns(entries: Sequence[ActivistEntry], as_of: date) -> tuple[str, ...]:
    """Names known as activists on `as_of`."""
    names = []
    for entry in entries:
        since = entry.get("since") if isinstance(entry, Mapping) else None
        if since is None or as_of >= date(int(since), 1, 1):
            names.append(_name(entry))
    return tuple(names)


def all_patterns(entries: Sequence[ActivistEntry]) -> tuple[str, ...]:
    """Every name, whatever its date: for today's alerts and for discovery."""
    return tuple(_name(e) for e in entries)


def is_activist(filer_name: str, patterns: Sequence[str]) -> bool:
    text = normalize(filer_name).upper()
    return any(normalize(p).upper() in text for p in patterns)


def activism_window_covered(
    index_start: date | None, as_of: date, window_days: int = WINDOW_DAYS
) -> bool:
    """True if the daily document lists were fetched from the first day of the window on."""
    return index_start is not None and index_start <= as_of - timedelta(days=window_days)


def activist_features(
    documents: pd.DataFrame, as_of: date, patterns: Sequence[str]
) -> pd.DataFrame:
    """Distinct activist filers per held company in the year strictly before `as_of`."""
    columns = ["edinet_code", "activist_filers_365d", "activist_names"]
    submitted = pd.to_datetime(documents["submit_datetime"])
    end = datetime.combine(as_of, time.min)
    reports = documents[
        (documents["doc_type_code"] == LARGE_SHAREHOLDING)
        & documents["issuer_edinet_code"].notna()
        & (submitted >= end - timedelta(days=WINDOW_DAYS))
        & (submitted < end)
    ]
    activists = reports[reports["filer_name"].map(lambda n: is_activist(str(n), patterns))]
    if activists.empty:
        return pd.DataFrame(columns=columns)
    names = activists["filer_name"].map(lambda n: _LEGAL_FORM.sub("", normalize(str(n))).strip())
    grouped = names.groupby(activists["issuer_edinet_code"]).agg(lambda s: sorted(set(s)))
    return pd.DataFrame(
        {
            "edinet_code": grouped.index,
            "activist_filers_365d": grouped.map(len).to_numpy(),
            "activist_names": grouped.map(" / ".join).to_numpy(),
        }
    )


INITIAL_REPORT = "大量保有報告書"  # change reports are 変更報告書
SPECIAL_REGIME = "特例"  # 特例対象株券等: institutional investors, passive by regime


@dataclass(frozen=True)
class DiscoveryConfig:
    purpose_element: str
    proposal_keyword: str
    negations: tuple[str, ...]
    excluded_markers: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> DiscoveryConfig:
        return cls(
            purpose_element=str(raw["purpose_element"]),
            proposal_keyword=normalize(str(raw["proposal_keyword"])),
            negations=tuple(normalize(str(n)) for n in raw["negations"]),
            excluded_markers=tuple(normalize(str(m)) for m in raw.get("excluded_markers", [])),
        )


def initial_reports_per_filer(documents: pd.DataFrame) -> pd.DataFrame:
    """The most recent initial, non-特例 large-shareholding report of each filer.

    Data contract: documents with doc_id, edinet_code (the filer), doc_type_code, doc_description,
    submit_datetime, withdrawal_status. One row per filer.
    """
    description = documents["doc_description"].astype(str)
    reports = documents[
        (documents["doc_type_code"] == LARGE_SHAREHOLDING)
        & description.str.startswith(INITIAL_REPORT)
        & ~description.str.contains(SPECIAL_REGIME)
        & (documents["withdrawal_status"].astype(str) == "0")
    ]
    latest = reports.sort_values("submit_datetime", ascending=False)
    return latest.drop_duplicates("edinet_code").reset_index(drop=True)


def declares_proposals(purpose: str, config: DiscoveryConfig) -> bool:
    """True if a sentence mentions 重要提案行為 without negating it, in a purpose that is not a
    squeeze-out by a tender-offer vehicle or a strategic partnership (`excluded_markers`).
    """
    text = normalize(purpose)
    if any(marker in text for marker in config.excluded_markers):
        return False
    start = text.find(config.proposal_keyword)
    while start != -1:
        sentence_start = text.rfind("。", 0, start) + 1
        end = text.find("。", start)
        sentence = text[sentence_start : end if end != -1 else len(text)]
        if not any(negation in sentence for negation in config.negations):
            return True
        start = text.find(config.proposal_keyword, start + 1)
    return False


def activist_candidates(
    purposes: pd.DataFrame, known_patterns: Sequence[str], config: DiscoveryConfig
) -> pd.DataFrame:
    """Filers declaring proposals that the curated list does not know yet."""
    declaring = purposes[purposes["purpose"].map(lambda p: declares_proposals(str(p), config))]
    unknown = declaring[~declaring["filer_name"].map(lambda n: is_activist(str(n), known_patterns))]
    return unknown.drop_duplicates("filer_edinet_code").reset_index(drop=True)


def new_large_holders(documents: pd.DataFrame, as_of: date, window_days: int) -> dict[str, int]:
    """Distinct filers of an initial non-特例 5% report per company in the window before as_of."""
    end = datetime.combine(as_of, time.min)
    submitted = pd.to_datetime(documents["submit_datetime"])
    description = documents["doc_description"].astype(str)
    reports = documents[
        (documents["doc_type_code"] == LARGE_SHAREHOLDING)
        & description.str.startswith(INITIAL_REPORT)
        & ~description.str.contains(SPECIAL_REGIME)
        & documents["issuer_edinet_code"].notna()
        & (submitted >= end - timedelta(days=window_days))
        & (submitted < end)
    ]
    counts = reports.groupby("issuer_edinet_code")["edinet_code"].nunique()
    return {str(k): int(v) for k, v in counts.items()}


TENDER_OFFER = "240"


def recent_offers(documents: pd.DataFrame, as_of: date, window_days: int) -> dict[str, date]:
    """Companies that were the subject of a third-party tender offer (240) filed in [as_of −
    window_days, as_of): already in play, so ranking them is not a prediction.
    """
    end = datetime.combine(as_of, time.min)
    submitted = pd.to_datetime(documents["submit_datetime"])
    offers = documents[
        (documents["doc_type_code"] == TENDER_OFFER)
        & documents["subject_edinet_code"].notna()
        & (documents["subject_edinet_code"] != documents["edinet_code"])
        & (documents["withdrawal_status"].astype(str) == "0")
        & (submitted >= end - timedelta(days=window_days))
        & (submitted < end)
    ]
    latest = pd.to_datetime(offers["submit_datetime"]).groupby(offers["subject_edinet_code"]).max()
    return {str(k): v.date() for k, v in latest.items()}
