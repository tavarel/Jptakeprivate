"""Tables for the dashboard: Tender offers tab and company card."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from radar.features.ownership import HolderPatterns, classify_holder
from radar.model.offer_price import OfferRange, OfferRangeConfig, likely_offer_range
from radar.model.rules import RuleSet
from radar.report.explain import edinet_url
from radar.text_ja import age_on


def premium_summary(study: pd.DataFrame) -> pd.DataFrame:
    """Median premium and run-up per event type, take-privates only."""
    usable = study[study["pre_announced"].eq(False) & study["is_take_private"].eq(True)]
    return (
        usable.groupby("event_type")
        .agg(
            offers=("doc_id", "count"),
            median_premium=("premium", "median"),
            median_runup=("runup_abnormal", "median"),
        )
        .reset_index()
    )


def public_reason(row: Mapping[Hashable, object]) -> str:
    """A public explanation for a run-up, if any; '—' means unexplained (possible leakage)."""
    if row.get("earlier_offer") is True:
        return "earlier offer"
    activist = row.get("activist_before")
    return f"activist: {activist}" if isinstance(activist, str) else "—"


def runup_cases(review: pd.DataFrame, unexplained_only: bool) -> pd.DataFrame:
    """Flagged run-ups for reading one by one, largest first, with a link to the 240 filing."""
    flagged = review[review["flagged"].astype(bool)].copy()
    flagged["why_public"] = [public_reason(r) for r in flagged.to_dict("records")]
    if unexplained_only:
        flagged = flagged[flagged["why_public"] == "—"]
    flagged["filing"] = flagged["doc_id"].map(lambda d: edinet_url(str(d)))
    columns = [
        "event_date", "company", "event_type", "runup_abnormal", "runup_short", "premium",
        "volume_ratio", "why_public", "filing",
    ]  # fmt: skip
    return flagged.sort_values("runup_abnormal", ascending=False)[columns].reset_index(drop=True)


def price_history(prices: pd.DataFrame, code: str, as_of: date, days: int) -> pd.DataFrame:
    """date, close for `code` over `days` before `as_of` (strictly before: point-in-time), restated
    to the latest share basis so a split does not look like a crash.
    """
    stock = prices[prices["code"] == code].copy()
    stock["date"] = pd.to_datetime(stock["date"]).dt.date
    stock = stock[(stock["date"] < as_of) & (stock["date"] >= as_of - timedelta(days=days))]
    stock = stock.sort_values("date")
    factors = stock["adj_factor"].fillna(1.0).astype(float)
    # A close is multiplied by every factor dated AFTER it (reverse cumulative product, shifted).
    later = factors[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return pd.DataFrame({"date": stock["date"], "close": stock["close"] * later}).reset_index(
        drop=True
    )


def holders_table(holders: pd.DataFrame, doc_id: str, patterns: HolderPatterns) -> pd.DataFrame:
    """The top-10 shareholders of one report with their type (custodian, person, company, …)."""
    rows = holders[holders["doc_id"] == doc_id].sort_values("rank")
    return pd.DataFrame(
        {
            "rank": rows["rank"],
            "holder": rows["name"],
            "type": [
                classify_holder(str(n), bool(o), patterns)
                for n, o in zip(rows["name"], rows["is_organization"], strict=True)
            ],
            "stake": rows["ratio"],
        }
    ).reset_index(drop=True)


def officers_table(officers: pd.DataFrame, doc_id: str, as_of: date) -> pd.DataFrame:
    """Directors and officers of one report with their age on the screening date."""
    rows = officers[officers["doc_id"] == doc_id]
    ages = [age_on(b, as_of) if isinstance(b, date) else None for b in rows["date_of_birth"]]
    return pd.DataFrame(
        {
            "name": rows["name"],
            "title": rows["title"],
            "age": pd.array(ages, dtype="Int64"),
            "shares": rows["shares_held"],
            "representative": rows["is_representative"].astype(bool),
        }
    ).reset_index(drop=True)


def hit_rate_by_band(panel: pd.DataFrame, max_score: float, band_points: float) -> pd.DataFrame:
    """Share of companies that became targets within the horizon, per band of points."""
    earned = panel["score"].astype(float) * max_score
    floors = earned.map(lambda p: _band_floor(p, max_score, band_points))
    frame = pd.DataFrame({"floor": floors, "label": panel["label"].astype(int)})
    bands = frame.groupby("floor").agg(companies=("label", "size"), hits=("label", "sum"))
    bands["hit_rate"] = bands["hits"] / bands["companies"]
    bands = bands.sort_index(ascending=False).reset_index()
    labels = [_band_label(f, max_score, band_points) for f in bands.pop("floor")]
    bands.insert(0, "points", labels)
    return bands


def points_band(earned: float, max_score: float, width: float) -> str:
    """Label of the band holding `earned` points ('10–12'); the card and the table share it."""
    return _band_label(_band_floor(earned, max_score, width), max_score, width)


def _band_floor(earned: float, max_score: float, width: float) -> float:
    # The maximum itself joins the band below when it is a multiple (10 of 10 → 8–10).
    top = (max_score - 1e-9) // width * width
    return float(min(earned // width * width, top))


def _band_label(floor: float, max_score: float, width: float) -> str:
    return f"{floor:g}–{min(floor + width, max_score):g}"


def likely_buyer(row: Mapping[str, object], ruleset: RuleSet) -> str:
    """The most plausible kind of buyer, from the same thresholds the rules use."""
    thresholds = {rule.id: rule.threshold for rule in ruleset.rules}

    def at_least(feature: str, rule_id: str) -> bool:
        value, threshold = row.get(feature), thresholds[rule_id]
        if not isinstance(value, int | float) or not isinstance(threshold, int | float):
            return False
        return not pd.isna(value) and float(value) >= float(threshold)

    if at_least("largest_listed_holder_stake", "listed_parent"):
        return "parent"
    if at_least("family_stake", "family_control"):
        return "mbo"
    return "third_party"


def track_record(panel: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Companies in the top `top_n` of a past screening that later received an offer."""
    ranked = panel.assign(
        rank=panel.groupby("as_of")["score"].rank(ascending=False, method="first").astype(int)
    )
    hits = ranked[(ranked["rank"] <= top_n) & (ranked["label"] == 1)]
    typed = [c for c in panel.columns if c.startswith("label_") and c != "label_complete"]
    kinds = [
        ", ".join(c.removeprefix("label_") for c in typed if row[c] == 1)
        for row in hits.to_dict("records")
    ]
    columns = ["as_of", "rank", "company_name", "sec_code"]
    return (
        hits[columns]
        .assign(event_types=kinds)
        .sort_values(["as_of", "rank"])
        .reset_index(drop=True)
    )


TSE_SESSIONS_PER_YEAR = 245  # trading days on the Tokyo Stock Exchange (weekdays − holidays)


def liquidity_context(row: Mapping[str, object], scores: pd.DataFrame) -> dict[str, str]:
    """Plain-language comparison of a company's liquidity with the other companies of the same
    screening date: all of them, and the same market-cap decile.
    """
    context: dict[str, str] = {}
    value, turnover = _number(row.get("median_trading_value_jpy")), _number(row.get("turnover"))
    if value is not None:
        below = _share_below(scores["median_trading_value_jpy"], value)
        text = f"Higher than {below:.0%} of listed companies on this date."
        peers = _size_peers(scores, _number(row.get("market_cap_jpy")))
        if peers is not None and peers["median_trading_value_jpy"].notna().any():
            low, high = peers["market_cap_jpy"].min(), peers["market_cap_jpy"].max()
            median = peers["median_trading_value_jpy"].median()
            text += (
                f" Median of companies of the same size (¥{low / 1e9:.1f}bn–¥{high / 1e9:.1f}bn):"
                f" ¥{median / 1e6:.1f}m."
            )
        context["median_trading_value_jpy"] = text
    if turnover is not None and turnover > 0:
        years = 1 / turnover / TSE_SESSIONS_PER_YEAR
        context["turnover"] = (
            f"Higher than {_share_below(scores['turnover'], turnover):.0%} of listed companies. "
            f"At this pace the whole market value changes hands every ~{years:.1f} years."
        )
    return context


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and not pd.isna(value):
        return float(value)
    return None


def _share_below(values: pd.Series, value: float) -> float:
    known = values.dropna().astype(float)
    return float((known < value).mean()) if len(known) else float("nan")


def _size_peers(scores: pd.DataFrame, mcap: float | None) -> pd.DataFrame | None:
    """Companies in the same market-cap decile as `mcap` on this date."""
    sized = scores[scores["market_cap_jpy"].notna()]
    if mcap is None or len(sized) < 10:
        return None
    ranked = sized.sort_values("market_cap_jpy").reset_index(drop=True)
    ranked["decile"] = [i * 10 // len(ranked) for i in range(len(ranked))]
    position = max(int((ranked["market_cap_jpy"] <= mcap).sum()) - 1, 0)
    peers: pd.DataFrame = ranked[ranked["decile"] == ranked.at[position, "decile"]]
    return peers


def price_axis_domain(history: pd.DataFrame, padding: float = 0.05) -> tuple[float, float]:
    """y-axis limits that hug the prices (a price line need not start at 0; starting there left most
    of the chart empty).
    """
    low, high = float(history["close"].min()), float(history["close"].max())
    span = high - low if high > low else max(abs(high), 1.0)
    return low - span * padding, high + span * padding


def offer_range_for(
    row: Mapping[str, object],
    study: pd.DataFrame,
    price: float | None,
    ruleset: RuleSet,
    config: OfferRangeConfig,
) -> OfferRange | None:
    """Likely offer price for one company: its likely buyer type, its last price and its book value
    per share (= price ÷ PBR, the same share basis as the market cap).
    """
    pbr = _number(row.get("pbr"))
    book = price / pbr if price is not None and pbr else None
    return likely_offer_range(study, likely_buyer(row, ruleset), price, book, config, pbr=pbr)


def offer_range_table(found: OfferRange, price: float | None) -> pd.DataFrame:
    """One row per anchor (premium paid, PBR paid) with low / median / high in yen and the median's
    distance to the last close — easier to read at a glance than one long sentence.
    """
    rows = []
    anchors = (
        (f"Premium paid ({found.offers} past {found.event_type} offers)", found.by_premium),
        (_book_label(found), found.by_book),
    )
    for label, values in anchors:
        if values is None:
            continue
        low, *_, high = values
        middle = values[len(values) // 2]
        upside = f"{middle / price - 1:+.0%}" if price else "—"
        rows.append([label, f"¥{low:,.0f}", f"¥{middle:,.0f}", f"¥{high:,.0f}", upside])
    columns = ["anchor", "low (25%)", "median", "high (75%)", "median vs last close"]
    return pd.DataFrame(rows, columns=columns)


def _book_label(found: OfferRange) -> str:
    if found.book_basis.startswith("PBR"):
        return f"PBR paid ({found.book_offers} offers for targets at {found.book_basis})"
    return f"PBR paid ({found.book_offers} past {found.event_type} offers)"


def help_texts(columns: list[str], texts: Mapping[str, str]) -> dict[str, str]:
    """Help text per column: the exact name first, else the longest key ending in "_" that prefixes
    it (lift_top_mbo → "lift_top_").
    """
    found: dict[str, str] = {}
    for column in columns:
        if column in texts:
            found[column] = texts[column]
            continue
        prefixes = [k for k in texts if k.endswith("_") and column.startswith(k)]
        if prefixes:
            found[column] = texts[max(prefixes, key=len)]
    return found


def universe_share_label(n: int, companies: float | None) -> str:
    """ "top 50 (≈1.2%)": a portfolio size with its share of the ranked list (~4,100 companies)."""
    if companies is None or companies <= 0:
        return f"top {n}"
    return f"top {n} (≈{n / companies:.1%})"


def live_column(
    model: str, mapping: Mapping[str, str | None], columns: Sequence[str]
) -> str | None:
    """The column of today's screen that ranks like walk-forward `model` (config/report.yaml), or
    None when that ranking has no live version or this screen predates it.
    """
    column = mapping.get(model)
    return column if column is not None and column in columns else None


def skill_verdict(probability: float | None) -> str:
    """Plain reading of a deflated Sharpe ratio (Portfolio tab)."""
    if probability is None or pd.isna(probability):
        return "not computed yet"
    if probability >= 0.95:
        return "strong evidence of a real edge"
    if probability >= 0.5:
        return "a hint, not proof — watch it on the live scoreboard"
    return "luck alone can explain this result, given how many strategies were tried"


def data_freshness(
    last_document: date | None, last_price: date | None, *, today: date, max_days: int
) -> dict[str, object]:
    """Banner text: when were filings and prices last updated, and is that too old to act on?"""
    parts, stale = [], False
    for name, day in (("filings", last_document), ("prices", last_price)):
        if day is None:
            parts.append(f"no {name} yet")
            stale = True
            continue
        age = int(np.busday_count(day, today))
        stale |= age > max_days
        parts.append(f"{name} up to {day:%Y-%m-%d} ({age} business day{'s' if age != 1 else ''})")
    return {"stale": stale, "text": " · ".join(parts)}


def ordering_options(
    orderings: Mapping[str, str], fallbacks: Mapping[str, str], columns: Sequence[str]
) -> dict[str, str]:
    """Orderings available in a screen: label → column, using an older equivalent column
    (`fallbacks`) for screens written before the preferred one existed.
    """
    present = set(columns)
    options = {}
    for label, column in orderings.items():
        chosen = column if column in present else fallbacks.get(column)
        if chosen in present:
            options[label] = str(chosen)
    return options
