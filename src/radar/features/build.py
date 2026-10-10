"""Assemble the feature table: one row per company for a given `as_of` date."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from radar.features.activism import (
    WINDOW_DAYS,
    ActivistEntry,
    active_patterns,
    activism_window_covered,
    activist_features,
    new_large_holders,
    recent_offers,
)
from radar.features.family import family_stake, family_surnames
from radar.features.market import (
    median_trading_value,
    momentum_12_1,
    officer_stake,
    price_change_12m,
)
from radar.features.ownership import (
    HolderPatterns,
    company_key,
    holder_portfolio_counts,
    listed_company_codes,
    listed_company_keys,
    listing_criteria_margin,
    ownership_changes,
    ownership_features,
    parent_features,
    parent_prior_offers,
    tradable_share_ratio,
)
from radar.features.point_in_time import annual_filings_history, price_as_of
from radar.features.sector import sector_percentiles
from radar.features.trends import (
    CloseHistory,
    consecutive_below_book,
    net_cash_change,
    year_end_pbr,
)
from radar.features.valuation import (
    adjusted_net_assets,
    market_cap,
    net_cash,
    outstanding_at_filing,
    price_to_book,
    ratio,
    rental_gain_after_tax,
    split_multiplier,
)
from radar.text_ja import age_on, surname_dictionary

FEATURE_COLUMNS = [
    "as_of",
    "edinet_code",
    "sec_code",
    "company_name",
    "source_doc_id",
    "filing_submitted_at",
    "oldest_representative_age",
    "ceo_age",
    "chair_age",
    "founder_in_control",
    "family_stake",
    "market_cap_jpy",
    "pbr",
    "adjusted_pbr",
    "rental_unrealized_gain_to_market_cap",
    "net_cash_jpy",
    "net_cash_to_market_cap",
    "operating_income_to_market_cap",
    "investment_securities_to_market_cap",
    "officer_stake",
    "price_change_12m",
    "momentum_12_1",
    "dividend_yield",
    "median_trading_value_jpy",
    "turnover",
    "tradable_share_ratio",
    "listing_criteria_margin",
    "largest_listed_holder_stake",
    "largest_listed_holder_name",
    "parent_prior_offers",
    "parent_other_holdings",
    "parent_pbr",
    "parent_net_cash_to_market_cap",
    "parent_activist_filers_365d",
    "document_index_start",
    "top10_concentration_ex_custodian",
    "top10_change",
    "listed_holder_stake_change",
    "new_5pct_holders_90d",
    "offer_filed_365d",
    "years_below_book",
    "net_cash_change_to_market_cap",
    "activist_filers_365d",
    "activist_names",
    "market_segment",
    "sector33",
    "pbr_sector_percentile",
    "net_cash_sector_percentile",
    "capital_plan_disclosed",
    "is_jgaap",
]


def build_feature_table(
    as_of: date,
    *,
    documents: pd.DataFrame,
    officers: pd.DataFrame,
    shareholders: pd.DataFrame,
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    listings: pd.DataFrame | None = None,
    capital_plans: pd.DataFrame | None = None,
    price_lag_days: int = 1,
    holder_patterns: HolderPatterns | None = None,
    activist_patterns: Sequence[ActivistEntry] | None = None,
    index_start: date | None = None,
    capital_plan_segments: Sequence[str] | None = None,
    portfolio_min_stake: float | None = None,
    tradable: TradableConfig | None = None,
    trend_years: int = 3,
    sector_min_peers: int = 5,
    homonym_list: Sequence[Mapping[str, str]] | None = None,
    rental_tax_rate: float | None = None,
    new_holder_window_days: int = 90,
) -> pd.DataFrame:
    """Join the point-in-time selections and compute every feature."""
    # Only PARSED annual reports compete for "latest": if the newest one is not downloaded yet,
    # the previous one (still public before as_of) is used instead of dropping the company.
    parsed = documents["doc_id"].isin(financials["doc_id"])
    history = annual_filings_history(documents[parsed], as_of, trend_years)
    filings = history.drop_duplicates("edinet_code")  # newest period first: the latest report
    # Universe: listed companies (funds also file 有報, without sec_code).
    if not filings.empty:
        filings = filings[filings["sec_code"].notna()]
    if filings.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    officers_by_doc = dict(tuple(officers.groupby("doc_id")))
    holders_by_doc = dict(tuple(shareholders.groupby("doc_id")))
    financials_by_doc: dict[str, dict[Hashable, Any]] = {
        str(r["doc_id"]): r for r in financials.to_dict("records")
    }
    latest = price_as_of(prices, as_of, price_lag_days).set_index("code")
    closes = CloseHistory(prices)
    periods_by_code = {
        str(code): group.to_dict("records") for code, group in history.groupby("edinet_code")
    }
    splits_by_code = _splits(prices)
    adjusted = "adj_factor" in prices.columns
    change_12m = price_change_12m(prices, as_of) if adjusted else {}
    momentum = momentum_12_1(prices, as_of) if adjusted else {}
    liquidity = median_trading_value(prices, as_of)
    segments = _market_segments(listings, as_of)
    sectors = _latest_listing_field(listings, as_of, "sector33")
    plans = _capital_plan_codes(capital_plans, as_of)
    listed_keys = listed_company_keys(documents, as_of)
    homonyms = _homonyms(holder_patterns, homonym_list)
    excluded = _employee_plan_patterns(holder_patterns)
    new_holders = (
        new_large_holders(documents, as_of, new_holder_window_days)
        if activism_window_covered(index_start, as_of, new_holder_window_days)
        and "doc_description" in documents.columns  # tells initial reports from changes
        else None
    )
    in_play = (
        recent_offers(documents, as_of, WINDOW_DAYS)
        if "subject_edinet_code" in documents.columns
        else {}
    )
    known = active_patterns(activist_patterns, as_of) if activist_patterns is not None else None
    activism = _activism(documents, as_of, known, index_start)
    track = _ParentTrack(documents, filings, shareholders, as_of, portfolio_min_stake)
    known_surnames = surname_dictionary(
        officers[officers["doc_id"].isin(filings["doc_id"])]["name"]
    )

    rows = []
    for filing in filings.to_dict("records"):
        doc_id, sec_code = filing["doc_id"], filing["sec_code"]
        doc_officers = officers_by_doc.get(doc_id, officers.iloc[0:0])
        doc_holders = holders_by_doc.get(doc_id, shareholders.iloc[0:0])
        fin = financials_by_doc.get(doc_id, {})
        ownership = _ownership(doc_holders, listed_keys, holder_patterns, homonyms)
        close = _split_adjusted_close(latest, splits_by_code, sec_code, filing, fin)
        valuation = _valuation_features(fin, close, rental_tax_rate)
        rows.append(
            {
                "as_of": as_of,
                "edinet_code": filing["edinet_code"],
                "sec_code": sec_code,
                "company_name": filing["filer_name"],
                "source_doc_id": doc_id,
                "filing_submitted_at": filing["submit_datetime"],
                **_age_features(doc_officers, as_of),
                **_family_features(doc_officers, doc_holders, known_surnames, excluded),
                **valuation,
                **ownership,
                **_changes(
                    doc_holders,
                    periods_by_code[str(filing["edinet_code"])],
                    holders_by_doc,
                    listed_keys=listed_keys,
                    patterns=holder_patterns,
                    homonyms=homonyms,
                ),
                # Already the subject of a tender offer this year: in play, not a prediction.
                "offer_filed_365d": in_play.get(str(filing["edinet_code"])),
                "new_5pct_holders_90d": (
                    None if new_holders is None else new_holders.get(str(filing["edinet_code"]), 0)
                ),
                **_trend_features(
                    periods_by_code[str(filing["edinet_code"])],
                    financials_by_doc,
                    closes,
                    splits_by_code.get(str(sec_code)),
                    valuation["market_cap_jpy"],
                ),
                **track.lookup(ownership["largest_listed_holder_name"]),
                "document_index_start": index_start,
                "officer_stake": officer_stake(doc_officers, outstanding_at_filing(fin)),
                "price_change_12m": change_12m.get(str(sec_code)),
                "momentum_12_1": momentum.get(str(sec_code)),
                "median_trading_value_jpy": liquidity.get(str(sec_code)),
                **_tradable(
                    doc_holders,
                    doc_officers,
                    fin,
                    market_cap=valuation["market_cap_jpy"],
                    segment=segments.get(sec_code) if segments is not None else None,
                    trading_value=liquidity.get(str(sec_code)),
                    config=tradable,
                ),
                **_activism_row(activism, filing["edinet_code"]),
                "market_segment": segments.get(sec_code) if segments is not None else None,
                "sector33": sectors.get(sec_code) if sectors is not None else None,
                "capital_plan_disclosed": _capital_plan(
                    sec_code, plans, segments, capital_plan_segments
                ),
                "is_jgaap": _clean(fin.get("is_jgaap")),
            }
        )
    table = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    table["pbr_sector_percentile"] = sector_percentiles(table, "pbr", sector_min_peers)
    table["net_cash_sector_percentile"] = sector_percentiles(
        table, "net_cash_to_market_cap", sector_min_peers
    )
    parents = parent_features(table, listed_company_codes(documents, as_of))
    table[parents.columns] = parents
    return table


def _age_features(officers: pd.DataFrame, as_of: date) -> dict[str, int | None]:
    """Ages on `as_of` of representative directors only (succession pressure sits with them)."""
    reps = officers[officers["is_representative"].astype(bool)]
    ceos = reps[reps["role"] == "president"]
    if ceos.empty and len(reps) == 1:
        ceos = reps

    def oldest(frame: pd.DataFrame) -> int | None:
        ages = [age_on(b, as_of) for b in frame["date_of_birth"] if isinstance(b, date)]
        return max((a for a in ages if a is not None), default=None)

    return {
        "oldest_representative_age": oldest(reps),
        "ceo_age": oldest(ceos),
        "chair_age": oldest(reps[reps["role"] == "chair"]),
    }


def _family_features(
    officers: pd.DataFrame,
    holders: pd.DataFrame,
    known_surnames: frozenset[str],
    excluded: tuple[str, ...] = (),
) -> dict[str, Any]:
    surnames = family_surnames(officers, known_surnames)
    result = family_stake(holders, surnames, known_surnames, excluded)
    known = result.stake is not None
    return {
        "founder_in_control": result.founder_in_control if known else None,
        "family_stake": result.stake,
    }


def _changes(
    holders: pd.DataFrame,
    periods: list[dict[Hashable, Any]],
    holders_by_doc: Mapping[Any, pd.DataFrame],
    *,
    listed_keys: frozenset[str],
    patterns: HolderPatterns | None,
    homonyms: Mapping[str, Sequence[str]],
) -> dict[str, float | None]:
    """Ownership changes vs the previous report public before as_of."""
    if patterns is None:
        return {"listed_holder_stake_change": None, "top10_change": None}
    previous = holders_by_doc.get(str(periods[1]["doc_id"])) if len(periods) > 1 else None
    return ownership_changes(holders, previous, listed_keys, patterns, homonyms)


def _trend_features(
    periods: list[dict[Hashable, Any]],
    financials_by_doc: Mapping[str, Mapping[Hashable, Any]],
    closes: CloseHistory,
    splits: pd.DataFrame | None,
    mcap: float | None,
) -> dict[str, float | int | None]:
    """years_below_book and net_cash_change_to_market_cap from the reports public before as_of,
    newest first (`periods`).
    """
    fins = [
        {str(k): _clean(v) for k, v in financials_by_doc.get(str(p["doc_id"]), {}).items()}
        for p in periods
    ]
    pbrs = []
    for period, fin in zip(periods, fins, strict=True):
        raw_end = period["period_end"]
        period_end = None if pd.isna(raw_end) else pd.Timestamp(raw_end).date()
        close = closes.close_on_or_before(str(period["sec_code"]), period_end)
        pbrs.append(year_end_pbr(fin, close, splits, period_end))
    return {
        "years_below_book": consecutive_below_book(pbrs),
        "net_cash_change_to_market_cap": net_cash_change(
            fins[0], fins[1] if len(fins) > 1 else None, mcap
        ),
    }


def _splits(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows with a split / consolidation (adj_factor ≠ 1) per code; none if the column is absent."""
    if "adj_factor" not in prices.columns:
        return {}
    events = prices[prices["adj_factor"].notna() & (prices["adj_factor"] != 1.0)].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return {str(code): frame for code, frame in events.groupby("code")}


def _split_adjusted_close(
    latest: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    code: Any,
    filing: Mapping[Hashable, Any],
    fin: Mapping[Hashable, Any],
) -> float | None:
    """Close × splits since the date of the share count, so close × shares is the market cap (ADR
    0005).
    """
    if code not in latest.index:
        return None
    row = latest.loc[code]
    if _clean(fin.get("issued_at_filing")):
        since = pd.Timestamp(filing["submit_datetime"]).date()
    else:
        since = filing.get("period_end") or pd.Timestamp(filing["submit_datetime"]).date()
    return float(row["close"]) * split_multiplier(splits.get(str(code)), since, row["date"])


def _valuation_features(
    fin: Mapping[Hashable, Any], price: float | None, rental_tax_rate: float | None = None
) -> dict[str, float | None]:
    """Market cap from the lagged close × shares outstanding in the latest annual report."""
    clean = {str(key): _clean(value) for key, value in fin.items()}
    mcap = market_cap(_clean(price), outstanding_at_filing(fin))
    cash = net_cash(clean)
    adjusted = None if rental_tax_rate is None else adjusted_net_assets(clean, rental_tax_rate)
    gain = None if rental_tax_rate is None else rental_gain_after_tax(clean, rental_tax_rate)
    return {
        "market_cap_jpy": mcap,
        "pbr": price_to_book(mcap, clean.get("net_assets")),
        # Added, not replacing pbr: book value with rental real estate at fair value (after tax).
        "adjusted_pbr": price_to_book(mcap, adjusted),
        "rental_unrealized_gain_to_market_cap": ratio(gain, mcap),
        "net_cash_jpy": cash,
        "net_cash_to_market_cap": ratio(cash, mcap),
        # 営業利益 / market cap (operating yield): is the business itself earning?
        "operating_income_to_market_cap": ratio(clean.get("operating_income"), mcap),
        # 1株当たり配当額 × year-end shares outstanding = the year's dividends, on that year's
        # share basis (split-safe). A proxy for the next 12 months' cash yield.
        "dividend_yield": ratio(_dividends(clean), mcap),
        # 投資有価証券 (largely cross-holdings, 政策保有株): "hidden" value that can fund a buyout.
        "investment_securities_to_market_cap": ratio(clean.get("investment_securities"), mcap),
    }


def _dividends(clean: Mapping[str, float | None]) -> float | None:
    """Total dividends of the last fiscal year; None when the per-share figure is unknown."""
    per_share, issued = clean.get("dividend_per_share"), clean.get("issued")
    if per_share is None or issued is None:
        return None
    return per_share * (issued - (clean.get("treasury") or 0.0))


@dataclass(frozen=True)
class TradableConfig:
    """TSE tradable-share settings (config/jpx.yaml) plus the holder-type patterns."""

    criteria: Mapping[str, Mapping[str, float]]
    major_holder: float
    patterns: HolderPatterns


def _tradable(
    holders: pd.DataFrame,
    officers: pd.DataFrame,
    fin: Mapping[Hashable, Any],
    *,
    market_cap: float | None,
    segment: str | None,
    trading_value: float | None,
    config: TradableConfig | None,
) -> dict[str, float | None]:
    """Turnover and TSE tradable-share features (流通株式), None when unknown."""
    turnover = ratio(trading_value, market_cap)
    if config is None:
        return {"turnover": turnover, "tradable_share_ratio": None, "listing_criteria_margin": None}
    issued = _clean(fin.get("issued_at_filing")) or _clean(fin.get("issued"))
    share_ratio = tradable_share_ratio(
        holders,
        officers,
        outstanding=outstanding_at_filing(fin),
        issued=issued,
        major_holder=config.major_holder,
        patterns=config.patterns,
    )
    margin = listing_criteria_margin(share_ratio, market_cap, segment, config.criteria)
    return {
        "turnover": turnover,
        "tradable_share_ratio": share_ratio,
        "listing_criteria_margin": margin,
    }


def _ownership(
    holders: pd.DataFrame,
    listed_keys: frozenset[str],
    patterns: HolderPatterns | None,
    homonyms: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Without holder patterns (config/ownership.yaml) the ownership features are missing."""
    if patterns is None:
        return ownership_features(holders.iloc[0:0], listed_keys, HolderPatterns(()))
    return ownership_features(holders, listed_keys, patterns, homonyms)


def _homonyms(
    patterns: HolderPatterns | None, entries: Sequence[Mapping[str, str]] | None
) -> dict[str, tuple[str, ...]]:
    """company_key → address fragments of checked homonyms (config/ownership.yaml)."""
    if patterns is None or not entries:
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        key = company_key(str(entry["name"]))
        result[key] = (*result.get(key, ()), str(entry["address_contains"]))
    return result


def _employee_plan_patterns(patterns: HolderPatterns | None) -> tuple[str, ...]:
    """Name patterns of employee share plans (持株会): never family (今村証券社員持株会)."""
    if patterns is None:
        return ()
    return next((subs for kind, subs in patterns.ordered if kind == "employee_plan"), ())


def _activism(
    documents: pd.DataFrame, as_of: date, patterns: Sequence[str] | None, index_start: date | None
) -> dict[str, tuple[int, str]] | None:
    """edinet_code → (activist count, names); None when unknown (no patterns, or the index does not
    cover the whole 365-day window, so an activist could be missing from it).
    """
    if patterns is None or not activism_window_covered(index_start, as_of):
        return None
    found = activist_features(documents, as_of, patterns)
    return {
        str(r["edinet_code"]): (int(r["activist_filers_365d"]), str(r["activist_names"]))
        for r in found.to_dict("records")
    }


def _activism_row(activism: dict[str, tuple[int, str]] | None, code: Any) -> dict[str, Any]:
    if activism is None:
        return {"activist_filers_365d": None, "activist_names": None}
    count, names = activism.get(str(code), (0, None))
    return {"activist_filers_365d": count, "activist_names": names}


class _ParentTrack:
    """Track record of the largest listed holder ("does this parent buy out subsidiaries?")."""

    def __init__(
        self,
        documents: pd.DataFrame,
        filings: pd.DataFrame,
        shareholders: pd.DataFrame,
        as_of: date,
        min_stake: float | None,
    ) -> None:
        self.as_of = as_of
        self.enabled = min_stake is not None
        self.codes = listed_company_codes(documents, as_of) if self.enabled else {}
        self.offers = documents[documents["doc_type_code"] == "240"]
        latest = shareholders[shareholders["doc_id"].isin(filings["doc_id"])]
        owner = dict(zip(filings["doc_id"], filings["edinet_code"], strict=True))
        latest = latest.assign(edinet_code=latest["doc_id"].map(owner))
        self.portfolio = holder_portfolio_counts(latest, min_stake) if min_stake is not None else {}

    def lookup(self, parent_name: str | None) -> dict[str, int | None]:
        if not self.enabled or not parent_name:
            return {"parent_prior_offers": None, "parent_other_holdings": None}
        key = company_key(parent_name)
        code = self.codes.get(key)
        offers = None if code is None else parent_prior_offers(self.offers, code, self.as_of)
        return {
            "parent_prior_offers": offers,
            "parent_other_holdings": max(self.portfolio.get(key, 1) - 1, 0),
        }


def _market_segments(listings: pd.DataFrame | None, as_of: date) -> dict[str, str] | None:
    """Latest known segment per code from rows dated strictly before `as_of`."""
    return _latest_listing_field(listings, as_of, "market_segment")


def _latest_listing_field(
    listings: pd.DataFrame | None, as_of: date, field: str
) -> dict[str, str] | None:
    """Latest known `field` per code from rows dated strictly before `as_of`; None if the listings
    (or that column, e.g. sector33 in an old staging table) are absent.
    """
    if listings is None or field not in listings.columns:
        return None
    known = listings[pd.to_datetime(listings["date"]) < _midnight(as_of)]
    latest = known.sort_values("date").groupby("code").tail(1)
    return dict(zip(latest["code"], latest[field], strict=True))


def _capital_plan_codes(capital_plans: pd.DataFrame | None, as_of: date) -> set[str] | None:
    """Codes on any JPX capital-plan list published strictly before `as_of`."""
    if capital_plans is None:
        return None
    published = capital_plans[pd.to_datetime(capital_plans["published_on"]) < _midnight(as_of)]
    if published.empty:
        return None  # no list public yet: unknown, not "did not disclose"
    return set(published["sec_code"])


def _capital_plan(
    code: Any,
    plans: set[str] | None,
    segments: dict[str, str] | None,
    requested: Sequence[str] | None,
) -> bool | None:
    """On a published JPX list?"""
    if plans is None:
        return None
    if requested is not None:
        segment = segments.get(code) if segments is not None else None
        if segment not in requested:
            return None
    return code in plans


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min)


def _clean(value: Any) -> Any:
    """NaN (from Parquet/pandas) → None, so the pure helpers see 'missing' consistently."""
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


_STAKES = (
    "family_stake",
    "largest_listed_holder_stake",
    "top10_concentration_ex_custodian",
    "officer_stake",
)


def audit_features(features: pd.DataFrame, requested_segments: Sequence[str]) -> pd.DataFrame:
    """Invariants every feature row must satisfy → (edinet_code, check, detail); empty = clean."""
    issues: list[tuple[str, str, str]] = []
    for row in features.to_dict("records"):
        code = str(row["edinet_code"])
        as_of = _midnight(row["as_of"])
        if pd.Timestamp(row["filing_submitted_at"]) >= as_of:
            issues.append((code, "filing_not_before_as_of", str(row["filing_submitted_at"])))
        plan = row.get("capital_plan_disclosed")
        if not _missing(plan) and row.get("market_segment") not in requested_segments:
            issues.append((code, "capital_plan_outside_request", str(row.get("market_segment"))))
        for column in _STAKES:
            value = row.get(column)
            if isinstance(value, int | float) and not pd.isna(value) and not 0 <= value <= 1 + 1e-6:
                issues.append((code, "stake_out_of_range", f"{column}={value}"))
        age = row.get("oldest_representative_age")
        if isinstance(age, int | float) and not pd.isna(age) and not 18 <= age <= 110:
            issues.append((code, "implausible_age", str(age)))
    return pd.DataFrame(issues, columns=["edinet_code", "check", "detail"])


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))
