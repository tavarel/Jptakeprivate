"""Data-quality report for the parsed staging tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    filings: int
    with_representative: float
    representative_with_birth_date: float
    with_shareholder_ratios: float
    jgaap: float


def quality_report(
    officers: pd.DataFrame, shareholders: pd.DataFrame, financials: pd.DataFrame
) -> QualityReport:
    """Share of parsed filings (0–1) that have each item."""
    filings = financials["doc_id"].nunique()
    if filings == 0:
        return QualityReport(0, 0.0, 0.0, 0.0, 0.0)

    reps = officers[officers["is_representative"].astype(bool)]
    reps_with_dob = reps[reps["date_of_birth"].notna()]
    with_ratio = shareholders[shareholders["ratio"].notna()]
    jgaap = financials[financials["is_jgaap"].eq(True)]  # None/NaN → not J-GAAP

    def share(frame: pd.DataFrame) -> float:
        return float(frame["doc_id"].nunique() / filings)

    return QualityReport(
        filings=int(filings),
        with_representative=share(reps),
        representative_with_birth_date=share(reps_with_dob),
        with_shareholder_ratios=share(with_ratio),
        jgaap=share(jgaap),
    )


# Director ages outside this range are parsing errors (young heirs and centenarian chairs exist).
_MIN_AGE, _MAX_AGE = 18, 110
_RATIO_TOLERANCE = 0.005  # rounding in filings (each ratio is rounded to 0.01%)


def sanity_checks(
    officers: pd.DataFrame, shareholders: pd.DataFrame, financials: pd.DataFrame
) -> pd.DataFrame:
    """Values that cannot be true → one row per (doc_id, check, detail)."""
    issues: list[tuple[str, str, str]] = []
    totals = shareholders.groupby("doc_id")["ratio"].sum()
    over_100 = totals[totals > 1 + _RATIO_TOLERANCE]
    issues += [(str(d), "top10_ratio_over_100pct", f"{v:.1%}") for d, v in over_100.items()]
    bad = shareholders[(shareholders["ratio"] < 0) | (shareholders["ratio"] > 1)]
    issues += [
        (str(d), "ratio_out_of_range", str(v))
        for d, v in zip(bad["doc_id"], bad["ratio"], strict=True)
    ]
    ranked = shareholders[shareholders["rank"].notna()]
    dupes = ranked[ranked.duplicated(["doc_id", "rank"], keep=False)]
    issues += [(str(d), "duplicate_rank", "") for d in dupes["doc_id"].unique()]

    with_rep = set(officers.loc[officers["is_representative"].astype(bool), "doc_id"])
    issues += [(str(d), "no_representative", "") for d in financials["doc_id"] if d not in with_rep]
    period = dict(zip(financials["doc_id"], financials["period_end"], strict=True))
    born = officers[officers["date_of_birth"].notna()]
    for doc_id, birth in zip(born["doc_id"], born["date_of_birth"], strict=True):
        end = period.get(doc_id)
        if isinstance(end, date) and isinstance(birth, date):
            age = end.year - birth.year
            if not _MIN_AGE <= age <= _MAX_AGE:
                issues.append((str(doc_id), "implausible_age", str(age)))

    over = financials[financials["treasury"] > financials["issued"]]
    issues += [(str(d), "treasury_exceeds_issued", "") for d in over["doc_id"]]
    no_equity = financials[financials["is_jgaap"].eq(True) & financials["net_assets"].isna()]
    issues += [(str(d), "jgaap_without_net_assets", "") for d in no_equity["doc_id"]]
    return pd.DataFrame(issues, columns=["doc_id", "check", "detail"]).drop_duplicates()
