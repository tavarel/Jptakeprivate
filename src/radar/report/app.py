"""Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from radar import pipeline
from radar.config import get_settings, load_yaml
from radar.features.ownership import HolderPatterns
from radar.features.point_in_time import price_as_of
from radar.model.offer_price import OfferRangeConfig
from radar.model.rules import RuleSet
from radar.report.explain import (
    edinet_url,
    explain,
    format_value,
    points,
    top_percent,
    unavailable_rules,
)
from radar.report.portfolio_tab import portfolio_tab
from radar.report.views import (
    data_freshness,
    help_texts,
    hit_rate_by_band,
    holders_table,
    likely_buyer,
    liquidity_context,
    offer_range_for,
    offer_range_table,
    officers_table,
    ordering_options,
    points_band,
    premium_summary,
    price_axis_domain,
    price_history,
    runup_cases,
    track_record,
)
from radar.storage.paths import journal_path, staging_table_path

DISCLAIMER = (
    "Research tool. Not investment advice. Scores are statistical signals from public filings; "
    "most companies near the top will never be taken private."
)
ABOUT = """
**What it is.** A daily screen of ~4,200 Japanese listed companies, ranked by how closely they
resemble companies that were later taken private — parents buying out listed subsidiaries,
management buyouts (MBO) and third-party tender offers. About 1.8% of listed companies receive
such an offer in a year.

**Data.** EDINET filings (有価証券報告書, 5% reports, tender offers, 臨時報告書), J-Quants daily
prices, JPX lists. Point-in-time: a fact is used only if it was public before the screening
date; prices lag one day.

**Rankings.** *Offer likelihood* (default) — nine rules with weights learned from 13 quarters;
strongest on parent buyouts. *MBO likelihood* — a per-type model for management and family
buyouts. Others answer a portfolio question (what made money), not an M&A one.

**Evidence.** Walk-forward tests on dates each model never saw, with the decision rules written
before running, and a live scoreboard of rankings frozen each day. Seven overlapping test dates
are weak evidence: the live record is the test that counts. Missing data is shown as unknown,
never as zero.
"""
OFFERS_HELP = """
Every take-private tender offer since 2021-09 (公開買付け by someone other than the company
itself; partial offers and buybacks excluded). **Buyer types**: *mbo* (management), *parent*
(already held ≥ 20%), *third_party* (a company or a fund). **Premium** = offer price ÷ last close
before the announcement − 1. **Run-up** = the stock's return minus TOPIX over the 60 (and last 10)
sessions before. Offers already priced in at the filing date (announced earlier, or contested)
are left out: the event date is the filing (EDINET), not the first announcement (TDnet).
A large run-up on heavy volume with no public reason is consistent with a leak — not proof.
"""
BACKTEST_HELP = """
Were the top-ranked companies taken private more often than average in the following year?
**Lift** = offer rate in the top 10% ÷ the average rate (1.0 = chance). Rows whose 12 months are
not over yet (`label_complete = False`) are shown but excluded from the averages.
"""
# "?" on every Backtest column. Keys ending with "_" apply to the per-type columns
# (lift_top_mbo, positives_parent, …).
BACKTEST_COLUMN_HELP = {
    "as_of": "Screening date: the ranking uses only data public before this day.",
    "label_complete": (
        "True when the 12 months after the screening date are over. False: later offers are not "
        "known yet, so hits are understated — do not judge the model on those rows."
    ),
    "companies": "Listed companies ranked on that date.",
    "positives": (
        "Companies that received a take-private offer within 12 months after the date "
        "(the 'targets'). Base rate = positives ÷ companies, about 2%."
    ),
    "precision_at_20": (
        "Share of the 20 highest-ranked companies that became targets. 0.05 = 1 of 20. Compare "
        "with the base rate (~0.02): above it, the very top of the list is useful."
    ),
    "precision_at_50": "Same for the 50 highest-ranked companies.",
    "recall_top": (
        "Share of ALL targets that were in the top 10% of the ranking. 0.16 = 16% of the "
        "companies that got an offer had been flagged in the top 10%. Chance would be 10%."
    ),
    "lift_top": (
        "Take-private rate in the top 10% ÷ rate of all companies. 1.0 = no better than "
        "chance; 1.6 = the top 10% got offers 1.6 times as often as an average company. "
        "Reading: below 1 worse than chance; 1–1.5 weak; 1.5–2.5 useful; above 2.5 strong."
    ),
    "positives_": "Targets of this deal type (mbo / parent / third_party) in the 12 months.",
    "recall_top_": "Share of this type's targets that were in the top 10%.",
    "lift_top_": (
        "Lift for this deal type only. Tells whether the ranking finds MBOs, parent buyouts "
        "or third-party deals — a ranking can be good at one and blind to another."
    ),
    "without": "The rule removed for this row; '(none)' is the full rule set.",
    "mean_metric": (
        "Mean of the decision metric over the walk-forward test dates: lift@top10% for the "
        "offer choice, top-25 excess return for the portfolio choice."
    ),
    "wins_vs_baseline": "Test dates on which this ranking beat the hand-set v0 on lift.",
    "eligible": "Meets the pre-registered rule (beats v0 on at least 5 of the 7 dates).",
    "lift_top_delta": (
        "Lift without the rule − lift with it. NEGATIVE = the ranking got worse without the rule, "
        "so the rule helps. Positive = the rule lowers the result."
    ),
    "model": "Which ranking: the rule set, or a model trained only on earlier dates.",
    "rank": "Position in that date's ranking (1 = highest score).",
    "event_types": "Type of the offer the company received.",
    "ranking": (
        "Which ordering of the frozen list: score = the rules; p_any = model probability; "
        "expected_value = probability × premium. The one that keeps winning here becomes the "
        "default — decided on future offers, not on past ones."
    ),
    "days_elapsed": "Days since the ranking was frozen.",
    "window_complete": "True once 12 months have passed; before that, offers are still to come.",
    "targets_so_far": "Companies that received a take-private offer since the ranking was frozen.",
    "hits_top_fraction": "How many of those targets were in the top 10% of the frozen ranking.",
    "hits_top_n": "How many were in the top 50.",
    "lift_so_far": (
        "Lift with the offers known so far. Early on it jumps around (a handful of offers); "
        "only a complete 12-month window is a real result."
    ),
}


def _column_help(columns: list[str]) -> dict[str, Any]:
    """st.column_config with a "?" for every known column."""
    texts = help_texts(columns, BACKTEST_COLUMN_HELP)
    return {column: st.column_config.Column(help=text) for column, text in texts.items()}


def _show(frame: pd.DataFrame, *, index: bool = False) -> None:
    columns = [
        *([str(frame.index.name)] if index and frame.index.name else []),
        *map(str, frame.columns),
    ]
    st.dataframe(frame, hide_index=not index, width="stretch", column_config=_column_help(columns))


CHANGES_HELP = (
    "Followed companies = the top 50 of the later date + the watchlist. The same note is written "
    "every morning to data/marts/alerts_<date>.md."
)
ORDERINGS = {
    "Offer likelihood": "score",
    "MBO likelihood": "p_mbo_prior",
    "Offer + MBO combined": "combo_v2_mbo",
    "Model probability (any offer)": "p_any_prior",
    "Value basket": "value_basket",
    "Value basket, profitable": "value_basket_quality",
    "Return model": "return_model",
    "Return model, sector-neutral": "return_model_sector_neutral",
    "Rule baseline (v0)": "score_v0",
}
ORDERING_FALLBACKS = {"p_mbo_prior": "p_mbo", "p_any_prior": "p_any"}
ORDERING_HELP = (
    "M&A rankings — Offer likelihood (default): 2.2x chance in the top 10%, 4.4x on parent "
    "buyouts, nearly blind to MBOs. MBO likelihood: 1.9x on MBOs. Combined: the top 7 of each "
    "in turn. Portfolio rankings (what made money, not who gets an offer) — Value basket: small, "
    "below book, cash-rich; the portfolio default. Return models: predicted 12-month excess "
    "return; challengers tracked live. Walk-forward results, 7 test dates; not advice."
)
SERIES_COLOR = "#2a78d6"  # single series: categorical slot 1 of the dataviz default palette
FACTS = [
    ("offer_filed_365d", "Tender offer already filed on it (last 12 months)"),
    ("p_any", "Model probability of a take-private offer within 12 months"),
    ("expected_value", "Expected 12-month value (research estimate)"),
    ("oldest_representative_age", "Oldest representative director"),
    ("family_stake", "Founding-family stake (top-10 holders)"),
    ("largest_listed_holder_stake", "Largest listed corporate holder"),
    ("largest_listed_holder_name", "… which is"),
    ("parent_prior_offers", "… tender offers it filed (see coverage below)"),
    ("parent_other_holdings", "… OTHER listed companies where it holds ≥20% (this one excluded)"),
    ("parent_pbr", "… its own price-to-book"),
    ("parent_net_cash_to_market_cap", "… its own net cash / market cap"),
    ("parent_activist_filers_365d", "… activists on the parent (last 12 months)"),
    ("activist_filers_365d", "Activists with a 5% filing (last 12 months)"),
    ("top10_concentration_ex_custodian", "Top-10 holders (excl. custodians)"),
    ("top10_change", "… change since the previous report"),
    ("listed_holder_stake_change", "Listed parent's stake, change since the previous report"),
    ("new_5pct_holders_90d", "New 5% holders in the last 90 days"),
    ("market_cap_jpy", "Market cap"),
    ("pbr", "Price-to-book"),
    ("pbr_sector_percentile", "… vs its sector"),
    ("adjusted_pbr", "Adjusted PBR (rental property at fair value)"),
    ("rental_unrealized_gain_to_market_cap", "… unrealized gain on rental property, after tax"),
    ("years_below_book", "Fiscal year ends below book"),
    ("net_cash_to_market_cap", "Net cash / market cap"),
    ("net_cash_sector_percentile", "… vs its sector"),
    ("net_cash_change_to_market_cap", "Net cash change since the previous report"),
    ("market_segment", "Market segment"),
    ("sector33", "Sector (TSE 33)"),
    ("capital_plan_disclosed", "JPX capital plan disclosed"),
    ("officer_stake", "Directors' and officers' stake"),
    ("price_change_12m", "Price change, 12 months"),
    ("median_trading_value_jpy", "Median daily trading value"),
    ("turnover", "Turnover (trading value relative to size)"),
    ("tradable_share_ratio", "Tradable shares 流通株式 (approx., upper bound)"),
    ("listing_criteria_margin", "Distance to the TSE tradable-share minimum"),
]

# "?" next to a key fact: what it means and how to read it. Company-specific comparisons
# (`liquidity_context`) are appended at display time.
FACT_HELP = {
    "offer_filed_365d": (
        "A tender offer was already filed on this company in the last 12 months: the deal is "
        "public, the price already sits near the offer. Not a prediction any more — hidden from "
        "the default list (toggle in the sidebar)."
    ),
    "oldest_representative_age": (
        "Oldest 代表取締役 at the report date. Under 60: no succession question. 60–70: watch. "
        "70+: succession pressure — the classic MBO / sale trigger in family companies. Across "
        "all deals the learned weight is slightly NEGATIVE (big groups also have old chairs), so "
        "read it together with the family stake."
    ),
    "family_stake": (
        "Shares of the founder, relatives and their asset companies in the top 10. Under 10%: no "
        "family anchor. 10–30%: influence. 30–50%: an anchor that can lead an MBO and block a "
        "hostile bid. Over 50%: family control — an MBO only needs financing. Unknown when no "
        "surname could be read (not the same as 0%)."
    ),
    "largest_listed_holder_stake": (
        "Biggest LISTED company in the top 10 (親子上場 when ≥ 50%). 20–33%: equity-method "
        "affiliate (持分法) — a natural buyer, cannot force a deal. 33.4%+: blocks special "
        "resolutions. 50%+: parent — the most frequent buyer since the TSE pushed groups to "
        "resolve parent–subsidiary listings. The strongest single signal (learned weight +1.4)."
    ),
    "largest_listed_holder_name": "Name as written in the 大株主の状況 table.",
    "parent_prior_offers": (
        "Tender offers this holder filed since the index starts: 0 = never bought anyone out "
        "here; 1+ = it has done it before (a 'parent that buys'). Bids through SPCs or other "
        "group companies are missed, so 0 is a floor."
    ),
    "parent_other_holdings": (
        "Other listed companies where this holder owns ≥ 20%. Several = a group with many "
        "listed subsidiaries, usually under pressure to tidy them up; 0 = a single stake."
    ),
    "parent_net_cash_to_market_cap": (
        "The PARENT's net cash ÷ its own market cap. Above ~20% it can buy out the minority "
        "with its own cash; negative = it would need debt or shares."
    ),
    "activist_filers_365d": (
        "Distinct funds on config/activists.yaml that filed a 5% report in the last 12 months. "
        "0: none. 1: pressure for buybacks, dividends or a sale. 2+: a crowded register — often "
        "precedes a deal or a big capital-policy change. Learned weight +0.8 (stable)."
    ),
    "top10_concentration_ex_custodian": (
        "Shares held by the top 10 holders, custodian banks excluded. Under 30%: widely held, a "
        "buyout needs many sellers. 30–60%: a few holders decide. Over 60%: a handful of "
        "holders can agree a deal between themselves."
    ),
    "top10_change": (
        "Change in that concentration since the previous annual report. A rise of 5+ points = "
        "someone is building a stake; a fall = holders leaving."
    ),
    "market_cap_jpy": (
        "Price before the screening date × shares outstanding (treasury excluded). Under ¥10bn: "
        "micro cap, very easy to finance. ¥10–30bn: small, the usual MBO / PE size. ¥30–100bn: "
        "mid, needs a larger fund or a strategic buyer. Over ¥100bn: large deals, mostly parents."
    ),
    "pbr": (
        "Market cap ÷ net assets (J-GAAP, latest report). Under 0.5: deeply below book (assets "
        "or cash the market ignores, or losses). 0.5–1: below book — under TSE pressure since "
        "2023. 1–2: normal. Over 2: priced for growth; a buyout is expensive."
    ),
    "net_cash_to_market_cap": (
        "(Cash + short-term securities − interest-bearing debt) ÷ market cap. Negative: net "
        "debt. 0–30%: normal. 30–100%: cash-rich, a buyout partly pays for itself. Over 100%: "
        "the market values the business itself at less than zero."
    ),
    "net_cash_change_to_market_cap": (
        "Change in net cash since the previous report, in % of market cap. Rising = cash piling "
        "up (a target for activists); falling = spending or losses."
    ),
    "market_segment": (
        "Prime / Standard / Growth. Standard companies face the strongest pressure to justify "
        "their listing (small, low float); Prime companies must meet tighter float and value "
        "rules."
    ),
    "sector33": "TSE 33-sector classification (J-Quants). Compare valuation with the same sector.",
    "capital_plan_disclosed": (
        "Whether the company is on the JPX list of companies that disclosed a plan to improve "
        "capital efficiency (資本コストや株価を意識した経営). Not disclosed = the listing may "
        "not be a priority. Known only from 2025-10: the rule is untested."
    ),
    "officer_stake": (
        "Shares held by directors and officers ÷ shares outstanding. Under 5%: management has "
        "little skin in the game. 5–20%: meaningful. Over 20%: management can lead an MBO."
    ),
    "price_change_12m": (
        "Split-adjusted change over 12 months. Below −20%: a fall — MBOs are often timed after "
        "one (cheaper for management). Above +50%: a re-rating or deal rumours already priced."
    ),
    "median_trading_value_jpy": (
        "Median yen value traded per day over the recent sessions. Below ~¥5m a day the market "
        "is very thin: building or selling a ¥100m position takes weeks, so most institutions "
        "cannot own the stock. The TSE asks Prime companies for an average daily trading value "
        "of at least ¥20m; Standard and Growth only need a minimum monthly volume. A thin market "
        "gives the owners little from the listing while its costs remain — a reason often given "
        "in MBO announcements (上場維持コスト). It mostly tracks size, so compare with companies "
        "of the same size."
    ),
    "turnover": (
        "Median daily trading value ÷ market cap: liquidity relative to size, so small and "
        "large companies can be compared. Around 0.2% a day is typical on the TSE; 0.05% or "
        "less (the least-traded ~15%) is the `low_turnover` candidate rule. Very low turnover "
        "usually means a family, a parent or cross-holders keep most shares. On its own it "
        "does not predict a deal; with a controlling holder it makes a buyout simpler."
    ),
    "tradable_share_ratio": (
        "Approximate 流通株式: shares not held by ≥10% holders (custodians excepted) or by "
        "officers. An upper bound — bank and corporate cross-holdings the TSE also excludes "
        "cannot be identified from the top-10 list."
    ),
    "listing_criteria_margin": (
        "How far the tradable shares are above the TSE minimum for the segment (ratio and "
        "value). Close to or below zero, the company must fix its float or leave the market."
    ),
    "years_below_book": (
        "Consecutive fiscal year ends with PBR < 1. Prices start in Oct 2021, so older years are "
        "unknown and the count is a floor."
    ),
    "pbr_sector_percentile": "PBR ranked within the TSE 33 sector (needs ≥ 5 peers with a value).",
    "p_any": (
        "From the per-type logistic models frozen by `radar model`, calibrated by isotonic "
        "regression on the last training year. About 2% of listed companies receive an offer "
        "in a year: under 2% = below average; 2–5% = up to 2.5× the average; 5%+ = the top of "
        "the list (the calibration is flat up there, so 6% vs 8% is not a real difference). "
        "Judged live by the scoreboard — not yet proven."
    ),
    "expected_value": (
        "Σ P(offer of each type) × median premium paid for that type + (1 − Σ P) × the typical "
        "12-month return of companies that got no offer. A research estimate to compare "
        "companies, not a forecast and not a recommendation."
    ),
    "new_5pct_holders_90d": (
        "Holders that filed an INITIAL large-shareholding report (大量保有報告書) on this company "
        "in the last 90 days — funds, companies or people. Institutional 特例 filers excluded. A "
        "new 5% holder often precedes pressure, a sale of the stake, or an offer."
    ),
    "listed_holder_stake_change": (
        "Percentage points gained or lost by the largest listed holder since the previous annual "
        "report. A parent buying more is a classic prelude to a full buyout."
    ),
    "parent_pbr": (
        "The listed parent's own PBR on the same date. A parent trading below book is itself "
        "under TSE pressure, and buying out a cheap subsidiary can raise its own returns."
    ),
    "parent_activist_filers_365d": (
        "Activists with a 5% filing on the PARENT in the last year: pressure to simplify the "
        "group (親子上場の解消) often lands on its listed subsidiaries."
    ),
    "adjusted_pbr": (
        "Market cap ÷ (net assets + after-tax unrealized gain on rental and idle real estate). "
        "Japanese companies carry land at historical cost; the 賃貸等不動産 note gives its fair "
        "value. The gain is taxed at the effective rate (~30.6%). Equal to PBR when the note is "
        "absent or omitted as immaterial; unknown when the note could not be read. A rough floor "
        "for valuation, not a target price. Listed cross-holdings need no fix: they are already "
        "at market value on a J-GAAP balance sheet."
    ),
    "rental_unrealized_gain_to_market_cap": (
        "Fair value − book value of rental / idle real estate, after tax, as a share of the "
        "market cap. 0 when the company reports no material rental property."
    ),
    "net_cash_sector_percentile": (
        "Net cash / market cap ranked within the TSE 33 sector (needs ≥ 5 peers with a value)."
    ),
}


def score_files(marts_dir: Path) -> list[Path]:
    return sorted(marts_dir.glob("scores_*.parquet"), reverse=True)


def _signal(rule_id: str) -> str:
    """listed_parent → Listed parent: rule ids read as words on screen."""
    return rule_id.replace("_", " ").capitalize()


def filters(scores: pd.DataFrame, ruleset: RuleSet) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    name = st.sidebar.text_input("Search company or code")
    # Learned weights can be negative (v2): start at the lowest possible score, else companies
    # whose only rule lowers the odds would silently vanish from the list.
    low, step = ruleset.min_score, round(ruleset.max_score / 25, 2) or 0.1
    min_points = st.sidebar.slider("Minimum score", low, ruleset.max_score, low, step)
    segments = sorted(scores["market_segment"].dropna().unique())
    chosen = st.sidebar.multiselect("Market segment", segments, placeholder="All segments")
    required = st.sidebar.multiselect(
        "Required signals", [r.id for r in ruleset.rules], format_func=_signal
    )
    watch_only = st.sidebar.toggle("Watchlist only", value=False,
                                   help="Companies in config/watchlist.yaml.")  # fmt: skip
    in_play = st.sidebar.toggle(
        "Include companies already in play",
        value=False,
        help="A tender offer was filed on the company in the last 12 months: the deal is "
        "public, so it is no longer a prediction.",
    )

    keep = scores["score"] * ruleset.max_score >= min_points - 1e-9
    if name:
        text = scores["company_name"].astype(str) + " " + scores["sec_code"].astype(str)
        keep &= text.str.contains(name, case=False, regex=False)
    if chosen:  # nothing chosen = every segment
        keep &= scores["market_segment"].isin(chosen)
    for rule_id in required:
        keep &= scores["triggered_rules"].map(lambda rules, r=rule_id: r in list(rules))
    if not in_play and "offer_filed_365d" in scores.columns:
        keep &= scores["offer_filed_365d"].isna()
    if watch_only:
        codes = {
            f"{str(e['code']).strip()}0" for e in load_yaml("watchlist").get("watchlist") or []
        }
        keep &= scores["sec_code"].isin(codes)
    return scores[keep]


def ranked_table(shown: pd.DataFrame, ruleset: RuleSet) -> None:
    table = pd.DataFrame(
        {
            "#": shown["rank"],
            "company": shown["company_name"],
            "code": shown["sec_code"].astype(str).str[:4],
            "segment": shown["market_segment"],
            "score": shown["triggered_rules"].map(
                lambda r: points({"triggered_rules": r}, ruleset)[0]
            ),
            "signals": shown["triggered_rules"].map(
                lambda r: ", ".join(_signal(x) for x in r) or "—"
            ),
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "code": st.column_config.TextColumn(width="small"),
            "score": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=ruleset.max_score,
                format="%.1f",
                help="Sum of the weights of the signals that fired (log-odds). NOT a "
                "probability. The bar is relative to the maximum any company could get.",
            ),
            "signals": st.column_config.Column(
                help="Rules that fired. Listed parent and Activist holder carry most of the "
                "signal; Family control and Below book are weak on their own."
            ),
        },
    )


def company_card(shown: pd.DataFrame, ruleset: RuleSet, scores: pd.DataFrame) -> None:
    if shown.empty:
        st.info("No company matches the filters.")
        return
    labels = shown["company_name"] + " (" + shown["sec_code"].astype(str).str[:4] + ")"
    choice = st.selectbox(
        "Company", labels.tolist(),
        help="Everything the ranking used for this company, from filings public BEFORE the "
        "screening date. Each fact has a ? with how to read its value.",
    )  # fmt: skip
    row = {str(k): v for k, v in shown[labels == choice].iloc[0].to_dict().items()}
    with st.container(border=True):
        _card_header(row, ruleset, len(scores))
        tabs = st.tabs(["Signals", "Fundamentals", "Ownership", "Price", "Filings", "Offer range"])
        with tabs[0]:
            _signals(row, ruleset)
        with tabs[1]:
            _fundamentals(row, scores)
        with tabs[2]:
            _ownership(row)
        with tabs[3]:
            _price(row)
        with tabs[4]:
            _filings(row)
        with tabs[5]:
            _offer_range(row, ruleset)


def _card_header(row: Mapping[str, Any], ruleset: RuleSet, universe: int) -> None:
    """Name, identity line and five headline numbers."""
    st.markdown(f"#### {row['company_name']}")
    identity = [str(row["sec_code"])[:4], row.get("market_segment"), row.get("sector33")]
    st.caption(" · ".join(str(x) for x in identity if x is not None and not pd.isna(x)))
    earned, maximum = points(row, ruleset)
    cells = st.columns(5)
    headline = [
        ("Score", f"{earned:.1f} / {maximum:.1f}",
         "Sum of the weights of the signals that fired; not a probability."),
        ("Rank", f"#{int(row['rank'])} of {universe:,}",
         f"Top {row['top_percent']:.1f}% of companies in this ordering."),
        ("P(offer, 12 m)", format_value("p_any", row.get("p_any")), FACT_HELP["p_any"]),
        ("Market cap", format_value("market_cap_jpy", row.get("market_cap_jpy")),
         FACT_HELP["market_cap_jpy"]),
        ("PBR", format_value("pbr", row.get("pbr")), FACT_HELP["pbr"]),
    ]  # fmt: skip
    for cell, (label, value, text) in zip(cells, headline, strict=True):
        cell.metric(label, value, help=text)


def _signals(row: Mapping[str, Any], ruleset: RuleSet) -> None:
    reasons = explain(row, ruleset)
    if not reasons:
        st.markdown("No signal fired for this company.")
    facts = dict(FACTS)
    for reason in reasons:
        detail = f" — {reason.detail}" if reason.detail else ""
        field = facts.get(reason.feature, _signal(reason.feature)).lstrip("… ")
        st.markdown(
            f"- **{_signal(reason.rule_id)}** — {field}: **{reason.value}**{detail}",
            help=reason.why,
        )
    earned, _ = points(row, ruleset)
    _history_note(earned, row, ruleset)


def _fundamentals(row: Mapping[str, Any], scores: pd.DataFrame) -> None:
    st.caption(
        "From the latest annual report before the screening date and prices up to the day "
        "before. 'unknown' = not reported or not readable — never zero."
    )
    left, right = st.columns(2)
    context = liquidity_context(row, scores)
    for i, (feature, label) in enumerate(FACTS):
        if feature == "largest_listed_holder_name" and not row.get("largest_listed_holder_stake"):
            continue  # no listed holder: the name is absent, not unknown
        notes = [n for n in (FACT_HELP.get(feature), context.get(feature)) if n]
        (left if i % 2 == 0 else right).markdown(
            f"{label}: **{format_value(feature, row.get(feature))}**",
            help="\n\n".join(notes) if notes else None,
        )


def _filings(row: Mapping[str, Any]) -> None:
    _extraordinary_card(row)
    start = row.get("document_index_start")
    if isinstance(start, date):
        st.caption(
            f"Tender offers are counted from {start} only: EDINET hides filings older than ~5 years"
            " (inspection period), so earlier offers by this holder are invisible here."
        )
    st.markdown(
        f"[Open the source annual report on EDINET]({edinet_url(str(row['source_doc_id']))})"
    )


def _history_note(earned: float, row: Mapping[str, Any], ruleset: RuleSet) -> None:
    """How often similar scores were followed by an offer, and what the likely buyer paid."""
    marts_dir, config = get_settings().marts_dir, load_yaml("backtest")
    name = f"scoring_v{ruleset.version}"
    panel = _complete_panel(marts_dir, name)
    if panel is not None:
        width = ruleset.max_score / float(config["points_bands"])
        bands = hit_rate_by_band(panel, ruleset.max_score, width)
        band = bands[bands["points"] == points_band(earned, ruleset.max_score, width)]
        metrics = pd.read_parquet(marts_dir / f"backtest_metrics_{name}.parquet")
        in_sample = "in_sample" in metrics.columns and bool(metrics["in_sample"].any())
        if not band.empty:
            b = band.iloc[0]
            st.markdown(
                f"**Track record of this score band ({b['points']} points):** "
                f"{b['hit_rate']:.1%} of {int(b['companies']):,} past company-dates received a "
                f"take-private offer within 12 months ({int(b['hits'])} offers)."
                + (" _In-sample: the weights were learned on these dates._" if in_sample else "")
            )
    buyer = likely_buyer(row, ruleset)
    study_path = marts_dir / "event_study.parquet"
    if study_path.exists():
        study = pd.read_parquet(study_path)
        summary = premium_summary(study).set_index("event_type")
        if buyer in summary.index:
            premium = float(summary["median_premium"].get(buyer, float("nan")))
            offers = int(summary["offers"].get(buyer, 0))
            st.markdown(
                f"**Most likely buyer type: {buyer}** — median premium paid in past "
                f"{buyer} take-privates: {premium:.0%} ({offers} offers)."
            )


def _offer_range(row: Mapping[str, Any], ruleset: RuleSet) -> None:
    study_path = get_settings().marts_dir / "event_study.parquet"
    if not study_path.exists():
        st.info("No tender-offer study yet: run `uv run radar event-study`.")
        return
    _offer_range_note(row, pd.read_parquet(study_path), ruleset)


def _offer_range_note(row: Mapping[str, Any], study: pd.DataFrame, ruleset: RuleSet) -> None:
    """Likely offer price: premium and PBR paid in past take-privates of the likely buyer type."""
    prices = _stock_prices(row["sec_code"])
    if prices is None:
        return
    latest = price_as_of(prices, row["as_of"])
    price = None if latest.empty else float(latest["close"].iloc[0])
    config = OfferRangeConfig.from_config(load_yaml("event_study"))
    found = offer_range_for(row, study, price, ruleset, config)
    if found is None or price is None:
        return
    st.markdown(
        f"**Likely offer price** — last close **¥{price:,.0f}**",
        help=(
            "Not a forecast. What buyers of the same type paid in past take-privates, applied to"
            " this company. Premium paid: last close × (1 + premium of past offers of the likely"
            " buyer type). PBR paid: book value per share × the price-to-book paid for targets"
            " valued like this one (offer PBR follows the target's own PBR). Low / high = 25th"
            " and 75th percentiles. Contested and pre-announced offers are excluded."
        ),
    )
    st.dataframe(offer_range_table(found, price), hide_index=True, width="stretch")


def _complete_panel(marts_dir: Path, ruleset: str) -> pd.DataFrame | None:
    """The backtest panel restricted to dates whose 12-month label window is over."""
    panel_path = marts_dir / f"backtest_panel_{ruleset}.parquet"
    return _complete_panel_cached(marts_dir, ruleset, _mtime(panel_path))


@st.cache_resource(show_spinner=False, max_entries=4)
def _complete_panel_cached(marts_dir: Path, ruleset: str, mtime: float) -> pd.DataFrame | None:
    panel_path = marts_dir / f"backtest_panel_{ruleset}.parquet"
    metrics_path = marts_dir / f"backtest_metrics_{ruleset}.parquet"
    if not panel_path.exists() or not metrics_path.exists():
        return None
    metrics = pd.read_parquet(metrics_path)
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    panel = pd.read_parquet(panel_path)
    return panel[panel["as_of"].isin(complete)]


def _price(row: Mapping[str, Any]) -> None:
    prices = _stock_prices(row["sec_code"])
    history = (
        price_history(prices, str(row["sec_code"]), row["as_of"], days=365)
        if prices is not None
        else None
    )
    if history is None or history.empty:
        st.info("No TSE price for this company.")
        return
    st.markdown(
        "**Price, last 12 months** (yen, split-adjusted)",
        help="Closes before the screening date, restated for splits. A steady fall can "
        "precede an MBO; a sharp rise on volume may be a leak or a rumour.",
    )
    st.altair_chart(_price_chart(history), width="stretch")


def _ownership(row: Mapping[str, Any]) -> None:
    doc_id = str(row["source_doc_id"])
    holders, officers = _staging("shareholders"), _staging("officers")
    left, right = st.columns(2)
    if holders is not None:
        left.markdown(
            "**Top-10 shareholders**",
            help="From the 大株主の状況 table of the report, each with its holder type "
            "(family, listed company, custodian, fund…).",
        )
        left.dataframe(
            holders_table(holders, doc_id, _patterns()),
            hide_index=True,
            column_config={"stake": st.column_config.NumberColumn(format="percent")},
        )
    if officers is not None:
        right.markdown(
            "**Directors and officers**", help="From 役員の状況, with age and shares held."
        )
        right.dataframe(officers_table(officers, doc_id, row["as_of"]), hide_index=True)


def _price_chart(history: pd.DataFrame) -> alt.LayerChart:
    """Single-series price line: y-axis fitted to the prices (not from 0), 2px line, and a crosshair
    that snaps to the nearest date with the close in the tooltip.
    """
    low, high = price_axis_domain(history)
    nearest = alt.selection_point(
        nearest=True, on="pointermove", fields=["date"], empty=False, clear="pointerout"
    )
    base = alt.Chart(history).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False))
    )
    line = base.mark_line(color=SERIES_COLOR, strokeWidth=2).encode(
        y=alt.Y(
            "close:Q",
            title="yen",
            scale=alt.Scale(domain=[low, high], zero=False, nice=False),
            axis=alt.Axis(format=",.0f", tickCount=5),
        )
    )
    shown = alt.condition(nearest, alt.value(1), alt.value(0))
    rule = (
        base.mark_rule(color="#808495", strokeWidth=1)
        .encode(
            opacity=shown,
            tooltip=[
                alt.Tooltip("date:T", title="date", format="%Y-%m-%d"),
                alt.Tooltip("close:Q", title="close (yen)", format=",.0f"),
            ],
        )
        .add_params(nearest)
    )
    dot = line.mark_point(filled=True, size=64, color=SERIES_COLOR).encode(opacity=shown)
    chart: alt.LayerChart = (line + rule + dot).properties(height=220)
    return chart


def radar_tab(marts_dir: Path) -> None:
    files = score_files(marts_dir)
    if not files:
        st.info("No scores yet. Run `uv run radar features --as-of YYYY-MM-DD` and `radar score`.")
        return
    st.sidebar.markdown("### Screen")
    path = st.sidebar.selectbox("Screening date", files, format_func=lambda p: p.stem[7:])
    scores = pd.read_parquet(path)
    options = ordering_options(ORDERINGS, ORDERING_FALLBACKS, list(scores.columns))
    order_label = st.sidebar.selectbox("Ranking", list(options), help=ORDERING_HELP, index=0)
    column = options[order_label]
    # Rule scores are discrete (24 companies once tied): break ties by the offer probability
    # instead of leaving the order to chance.
    order = [column] + (["p_any_prior"] if column == "score" and "p_any_prior" in scores else [])
    scores = scores.sort_values(order, ascending=False, kind="mergesort", na_position="last")
    scores = scores.reset_index(drop=True)
    scores["top_percent"] = top_percent(scores[column].fillna(scores[column].min() - 1))
    scores["rank"] = scores.index + 1  # position in the chosen ordering (ties broken above)
    ruleset = RuleSet.default()
    missing = unavailable_rules(scores, ruleset)
    if missing:
        lost = sum(r.weight for r in ruleset.rules if r.id in missing)
        st.warning(
            f"No data yet on this date for: {', '.join(missing)} — {lost:.1f} of "
            f"{ruleset.max_score:.1f} points cannot be earned by anyone, so all points look low."
        )
    shown = filters(scores, ruleset)
    _screen_kpis(scores, shown, path.stem[7:])
    title, download = st.columns([4, 1], vertical_alignment="bottom")
    title.markdown(f"**{order_label}** — {len(shown):,} companies")
    export = shown.drop(columns=["triggered_rules"]).assign(
        rules=shown["triggered_rules"].map(", ".join)
    )
    download.download_button(
        "Export CSV",
        _csv_bytes(export),
        file_name=f"radar_{path.stem[7:]}.csv",
        mime="text/csv",
        width="stretch",
    )
    ranked_table(shown, ruleset)
    company_card(shown, ruleset, scores)


def _screen_kpis(scores: pd.DataFrame, shown: pd.DataFrame, as_of: str) -> None:
    """Four numbers that frame the list: how many, how many shown, base rate, followed."""
    cells = st.columns(4)
    cells[0].metric("Companies screened", f"{len(scores):,}", help=f"Screening date {as_of}.")
    cells[1].metric("Matching filters", f"{len(shown):,}")
    rate, _ = _base_rate(get_settings().marts_dir)
    cells[2].metric(
        "Base rate",
        f"{rate:.1%} / yr",
        help="Share of listed companies that receive a take-private offer within 12 months "
        "(13 complete backtest dates). A ranking is useful only if its top beats this.",
    )
    in_play = int(scores["offer_filed_365d"].notna().sum()) if "offer_filed_365d" in scores else 0
    cells[3].metric(
        "Already in play",
        f"{in_play:,}",
        help="Companies with a tender offer filed in the last 12 months — hidden by default.",
    )


@st.fragment
def offers_tab(marts_dir: Path) -> None:
    study_path = marts_dir / "event_study.parquet"
    review_path = marts_dir / "runup_review.parquet"
    if not study_path.exists() or not review_path.exists():
        st.info("No tender-offer study yet. Run `uv run radar event-study --runups`.")
        return
    summary = premium_summary(pd.read_parquet(study_path))
    cells = st.columns(len(summary) + 1)
    cells[0].metric("Take-private offers studied", f"{int(summary['offers'].sum()):,}",
                    help="Tender offers aimed at delisting, 2021-09 onwards.")  # fmt: skip
    names = {"mbo": "MBO", "parent": "Parent buyout", "third_party": "Third party"}
    for cell, row in zip(cells[1:], summary.to_dict("records"), strict=True):
        cell.metric(
            f"{names.get(str(row['event_type']), row['event_type'])} · median premium",
            f"{row['median_premium']:.0%}",
            delta=f"run-up {row['median_runup'] + 0.0:+.1%} before",
            delta_color="off",
            delta_arrow="off",
            help=f"{int(row['offers'])} offers. Run-up = return vs TOPIX in the ~60 sessions "
            "before the announcement: near 0 means the premium was still there to collect.",
        )
    with st.expander("Method"):
        st.markdown(OFFERS_HELP)
    st.subheader(
        "Premium and run-up by type of buyer",
        help="Premium = offer price vs the close before the announcement. Run-up = how much the "
        "stock beat TOPIX in the ~60 days before. A high run-up means the market guessed early "
        "(less premium left to collect); parent and MBO deals show little run-up.",
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Buyer type": summary["event_type"].map(lambda t: names.get(str(t), t)),
                "Offers": summary["offers"],
                "Median premium": summary["median_premium"] * 100,
                "Median run-up (60 d vs TOPIX)": summary["median_runup"] * 100,
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Median premium": st.column_config.NumberColumn(format="%.0f%%"),
            "Median run-up (60 d vs TOPIX)": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.subheader(
        "Large run-ups before the announcement",
        help="Offers whose target rose far more than TOPIX before the filing: a possible leak, "
        "a pre-announced deal, or news. Listed to check by hand, not as proof of anything.",
    )
    review = pd.read_parquet(review_path)
    unexplained = st.toggle("Only run-ups with no public explanation", value=True)
    cases = runup_cases(review, unexplained_only=unexplained)
    st.caption(f"{len(cases)} of {len(review)} offers")
    shown = cases.rename(
        columns={
            "event_date": "Filed",
            "company": "Company",
            "event_type": "Buyer type",
            "why_public": "Public reason",
        }
    ).assign(**{"Buyer type": lambda d: d["Buyer type"].map(lambda t: names.get(str(t), t))})
    for column in ("runup_abnormal", "runup_short", "premium"):
        shown[column] = shown[column] * 100
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "runup_abnormal": st.column_config.NumberColumn(
                "Run-up 60 d vs TOPIX", format="%+.0f%%"
            ),
            "runup_short": st.column_config.NumberColumn("Run-up 10 d", format="%+.0f%%"),
            "premium": st.column_config.NumberColumn("Premium", format="%.0f%%"),
            "volume_ratio": st.column_config.NumberColumn("Volume × normal", format="%.1f"),
            "filing": st.column_config.LinkColumn("Filing", display_text="open"),
        },
    )  # fmt: skip


@st.fragment
def backtest_tab(marts_dir: Path) -> None:
    runs = sorted(marts_dir.glob("backtest_metrics_*.parquet"))
    if not runs:
        st.info("No backtest yet: run `uv run radar backtest --start … --end …`.")
        return
    walk = marts_dir / "walk_forward_scoring_v0.parquet"
    live = marts_dir / "live_scoreboard.parquet"
    _evidence_kpis(walk, live, marts_dir)
    if walk.exists():
        _lift_chart(pd.read_parquet(walk))
    st.caption(BACKTEST_HELP)
    choice_path = marts_dir / "walk_forward_choice_scoring_v0.parquet"
    if choice_path.exists():
        with st.expander("Walk-forward comparison of all rankings"):
            st.caption(
                "Each test date (2024-03 → 2025-09) predicted by models trained only on data a "
                "year older. Rule fixed before running: replace the hand-set v0 only if it wins "
                "on ≥ 5 of 7 dates; among those within 0.1x of the best, the simplest wins."
            )
            _show(pd.read_parquet(choice_path))
    if live.exists():
        with st.expander("Live track record (rankings frozen each day)"):
            st.caption(
                "Judged on the offers that came after each freeze — no model choice has seen "
                "these outcomes. Windows shorter than 12 months are still incomplete."
            )
            _show(pd.read_parquet(live))
    for path in runs:
        _ruleset_evidence(marts_dir, path)


def _base_rate(marts_dir: Path) -> tuple[float, int]:
    """Share of companies taken private within 12 months, over complete backtest dates."""
    path = marts_dir / "backtest_metrics_scoring_v0.parquet"
    if not path.exists():
        return float("nan"), 0
    metrics = pd.read_parquet(path)
    done = metrics[metrics["label_complete"].astype(bool)]
    if done.empty:
        return float("nan"), 0
    return float((done["positives"] / done["companies"]).mean()), len(done)


def _evidence_kpis(walk: Path, live: Path, marts_dir: Path) -> None:
    cells = st.columns(4)
    cells[0].metric(
        "Default ranking",
        "Offer likelihood (v2)",
        help="Chosen by the pre-registered walk-forward rule (docs/results.md).",
    )
    if walk.exists():
        lift = pd.read_parquet(walk).groupby("model")["lift_top"].mean()
        cells[1].metric(
            "Walk-forward lift (top 10%)",
            f"{lift.get('v2', float('nan')):.2f}x",
            delta=f"v0 baseline {lift.get('v0', float('nan')):.2f}x",
            delta_color="off",
            delta_arrow="off",
            help="Offer rate in the top 10% ÷ average rate, mean of 7 unseen test dates. "
            "Below 1 worse than chance; 1.5–2.5 useful; above 2.5 strong.",
        )
    snapshots = len(list((marts_dir / "live").glob("scores_*.parquet")))
    cells[2].metric("Live screens frozen", f"{snapshots}",
                    help="One per daily update; judged later on real offers.")  # fmt: skip
    rate, dates = _base_rate(marts_dir)
    cells[3].metric(
        "Complete backtest dates",
        f"{dates}",
        help=f"Quarter-ends whose 12 months are over. Base rate on them: {rate:.1%} a year.",
    )


def _lift_chart(walk: pd.DataFrame) -> None:
    """Lift of the default ranking vs the hand-set baseline on each unseen test date."""
    names = {"v2": "Offer likelihood (v2)", "v0": "Rule baseline (v0)"}
    frame = walk[walk["model"].isin(names)].assign(
        ranking=lambda d: d["model"].map(names), as_of=lambda d: pd.to_datetime(d["as_of"])
    )
    st.markdown(
        "**Lift on each unseen test date**",
        help="Each point: a quarter-end ranked with a model trained only on data a year older, "
        "judged on the offers of the next 12 months. The dashed line is chance (1.0).",
    )
    lines = alt.Chart(frame).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("as_of:T", title=None, axis=alt.Axis(format="%Y-%m", tickCount=7, grid=False)),
        y=alt.Y("lift_top:Q", title="lift (top 10%)"),
        color=alt.Color("ranking:N", scale=alt.Scale(domain=list(names.values()),
                        range=[SERIES_COLOR, "#808495"]), legend=alt.Legend(orient="bottom",
                        title=None)),
        tooltip=[alt.Tooltip("ranking:N"), alt.Tooltip("as_of:T", format="%Y-%m-%d"),
                 alt.Tooltip("lift_top:Q", format=".2f")],
    )  # fmt: skip
    chance = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[4, 4], color="#c3c2b7")
        .encode(y="y:Q")
    )
    st.altair_chart((chance + lines).properties(height=240), width="stretch")


def _ruleset_evidence(marts_dir: Path, path: Path) -> None:
    """Backtest by date, ablation, model comparison and track record of one rule set."""
    ruleset = path.stem.removeprefix("backtest_metrics_")
    metrics = pd.read_parquet(path)
    if "lift_top" not in metrics.columns:
        return  # learned weights re-scored in-sample: no metrics by design
    done = metrics[metrics["label_complete"].astype(bool)]
    title = f"Backtest by date — {ruleset}"
    if not done.empty:
        title += f" · mean lift {done['lift_top'].mean():.2f}x on {len(done)} complete dates"
    with st.expander(title):
        _show(metrics)
    ablation_path = marts_dir / f"ablation_{ruleset}.parquet"
    if ablation_path.exists():
        with st.expander(f"Signal ablation — {ruleset}"):
            st.caption(
                "Each row re-scores the same dates without one rule. lift_top_delta < 0: the "
                "rule helps. Per-type columns show WHICH deals a rule finds."
            )
            _show(pd.read_parquet(ablation_path))
    model_path = marts_dir / f"model_comparison_{ruleset}.parquet"
    if model_path.exists():
        with st.expander(f"Rules vs models on 2025 test dates — {ruleset}"):
            comparison = pd.read_parquet(model_path).groupby("model").mean(numeric_only=True)
            _show(comparison, index=True)
    panel = _complete_panel(marts_dir, ruleset)
    if panel is not None:
        top_n = int(load_yaml("backtest")["track_record_top_n"])
        record = track_record(panel, top_n)
        with st.expander(f"Track record — top {top_n} names later taken private "
                         f"({len(record)} company-dates)"):  # fmt: skip
            _show(record)


JOURNAL_HELP = """
**Decision journal** — write what you decided and why in `data/manual/journal.yaml` (personal:
never committed), one entry per decision, then see here what happened since:

```yaml
entries:
  - {date: 2026-10-10, code: "7122", action: watch, price: 1234, reason: "parent holds 45%"}
  - {date: 2026-10-12, code: "3710", action: pass, reason: "too illiquid"}
```

`action`: watch · buy · sell · pass. `price` is optional. Returns are split-adjusted, from the
last close on or before the entry date to the last close before today.
"""


@st.fragment
def journal_tab() -> None:
    path = journal_path(get_settings().manual_dir)
    with st.expander("How to record a decision", expanded=not path.exists()):
        st.markdown(JOURNAL_HELP)
    if not path.exists():
        st.info(f"No journal yet: create `{path}` with entries like the example above.")
        return
    try:
        prices = staging_table_path(get_settings().staging_dir, "prices")
        table = _journal((_mtime(path), _mtime(prices)), date.today())
    except ValueError as exc:  # a typo in the YAML: say which entry, do not crash the page
        st.error(str(exc))
        return
    if table is None or table.empty:
        st.info("The journal has no entries yet.")
        return
    shown = table.sort_values("date", ascending=False)
    cells = st.columns(4)
    cells[0].metric("Decisions", f"{len(shown)}")
    offers = int(shown["offer_after"].notna().sum())
    cells[1].metric(
        "Later received an offer",
        f"{offers}",
        help="Compare with the 1.8% base rate and with the top of the ranking.",
    )
    median = shown["return_since"].median()
    cells[2].metric("Median return since", "—" if pd.isna(median) else f"{median:+.1%}")
    cells[3].metric("Watching / bought", f"{int(shown['action'].isin(['watch', 'buy']).sum())}")
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={"return_since": st.column_config.NumberColumn(format="percent")},
    )  # fmt: skip


@st.cache_data(show_spinner=False, max_entries=2)
def _journal(stamps: tuple[float, float], today: date) -> pd.DataFrame | None:
    """`stamps` (journal and price file times) only key the cache: edits show at once."""
    return pipeline.journal_table(get_settings(), today)


@st.fragment
def changes_tab(marts_dir: Path) -> None:
    dates = sorted(p.stem[7:] for p in score_files(marts_dir))
    if len(dates) < 2:
        st.info("Needs two screening dates: run `uv run radar update` on two different days.")
        return
    st.caption(CHANGES_HELP)
    left, right = st.columns(2)
    since = left.selectbox("From", dates, index=len(dates) - 2)
    until = right.selectbox("To", dates, index=len(dates) - 1)
    if since >= until:
        st.warning("Pick a 'From' date earlier than 'To'.")
        return
    changed, events = _changes(date.fromisoformat(since), date.fromisoformat(until))
    status = changed["status"].astype(str) if not changed.empty else pd.Series(dtype=str)
    cells = st.columns(4)
    cells[0].metric(
        "New filings",
        f"{len(events)}",
        help="5% reports, tender offers, opinions, control-related 臨時報告書.",
    )
    cells[1].metric("Tender offers", f"{int((events['kind'] == 'tender offer filed').sum())}")
    cells[2].metric("Entered the top 50", f"{int(status.str.startswith('entered').sum())}")
    cells[3].metric("Left the top 50", f"{int(status.str.startswith('left').sum())}")
    st.subheader(
        f"New filings on followed companies ({len(events)})",
        help="Top 50 + watchlist. 5% reports (activists flagged), tender offers, targets' "
        "opinions and control-related 臨時報告書 (new parent, major shareholder, representative, "
        "squeeze-out). Each is a fact, not a forecast.",
    )
    names = _company_names(until)
    st.dataframe(
        pd.DataFrame(
            {
                "Filed": pd.to_datetime(events["submit_datetime"]).dt.strftime("%Y-%m-%d %H:%M"),
                "Company": events["company_code"].map(lambda c: names.get(str(c), c)),
                "Event": events["kind"],
                "Filer": events["filer_name"],
                "Filing": events["doc_id"].map(lambda d: edinet_url(str(d))),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Filing": st.column_config.LinkColumn(display_text="open")},
    )
    st.subheader(
        f"Ranking changes ({len(changed)})",
        help="Companies that entered or left the top 50, or whose rules changed, between the "
        "two dates. Ties count together, so a tie never looks like a move.",
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Company": changed["company_name"],
                "Code": changed["sec_code"].astype(str).str[:4],
                "Change": changed["status"],
                "Rank before": changed["rank_old"],
                "Rank now": changed["rank_new"],
                "Score change": changed["points_change"],
                "Signals added": changed["rules_added"].map(
                    lambda r: ", ".join(_signal(x) for x in r)
                ),
                "Signals removed": changed["rules_removed"].map(
                    lambda r: ", ".join(_signal(x) for x in r)
                ),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Score change": st.column_config.NumberColumn(format="%+.2f")},
    )


@st.cache_data(show_spinner=False, max_entries=4)
def _company_names(as_of: str) -> dict[str, str]:
    """EDINET code → company name from the screen of `as_of` (for readable alert tables)."""
    path = get_settings().marts_dir / f"scores_{as_of}.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["edinet_code", "company_name"])
    return dict(zip(frame["edinet_code"].astype(str), frame["company_name"], strict=False))


@st.cache_data(show_spinner=False)
def _changes(since: date, until: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pipeline.run_changes(get_settings(), since, until)


def _staging(table: str) -> pd.DataFrame | None:
    """A staging table, or None before it exists (fresh install: no prices downloaded yet)."""
    path = staging_table_path(get_settings().staging_dir, table)
    return _read_table(path, _mtime(path)) if path.exists() else None


@st.cache_data(show_spinner=False, max_entries=4)
def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """The CSV export, built once per filtered list instead of on every click (~0.4 s)."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def _freshness_banner() -> None:
    """When were filings and prices last updated?"""
    staging = get_settings().staging_dir
    stamps = (
        _mtime(staging_table_path(staging, "documents")),
        _mtime(staging_table_path(staging, "prices")),
    )
    last_doc, last_price = _last_dates(stamps)
    status = data_freshness(last_doc, last_price, today=date.today(), max_days=3)
    if status["stale"]:
        st.warning(f"Data may be stale — {status['text']}. Run `uv run radar update`.")
    else:
        st.caption(f"✓ Data current · filings and prices to {last_price or last_doc}")


@st.cache_data(show_spinner=False, max_entries=2)
def _last_dates(stamps: tuple[float, float]) -> tuple[date | None, date | None]:
    """Latest filing and price dates; `stamps` (file times) only key the cache."""
    documents, prices = _staging("documents"), _staging("prices")
    doc = None if documents is None else pd.Timestamp(documents["submit_datetime"].max()).date()
    price = None if prices is None else pd.Timestamp(prices["date"].max()).date()
    return doc, price


def _extraordinary_card(row: Mapping[str, Any]) -> None:
    """臨時報告書 of the last 12 months: control changes are often the first trace of a deal."""
    documents = _staging("documents")
    if documents is None:
        return
    as_of = pd.Timestamp(row["as_of"])
    mine = documents[
        (documents["doc_type_code"] == "180")
        & (documents["edinet_code"] == row["edinet_code"])
        & (pd.to_datetime(documents["submit_datetime"]) < as_of)
        & (pd.to_datetime(documents["submit_datetime"]) >= as_of - pd.Timedelta(days=365))
    ]
    if mine.empty:
        return
    found = _classified(tuple(mine["doc_id"].astype(str)))
    table = pd.DataFrame(
        {
            "filed": pd.to_datetime(mine["submit_datetime"]).dt.date,
            "reason": [
                str(found[d]["labels"]) if d in found else "not downloaded" for d in mine["doc_id"]
            ],
            "link": [edinet_url(str(d)) for d in mine["doc_id"]],
        }
    ).sort_values("filed", ascending=False)
    st.markdown(
        "**臨時報告書 (last 12 months)**",
        help="Extraordinary reports: changes of parent, major shareholder or representative "
        "director, squeeze-out meetings … Only followed companies (top 50 + watchlist) are "
        "downloaded by `radar update`; others show 'not downloaded'.",
    )
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={"link": st.column_config.LinkColumn(display_text="EDINET")},
    )  # fmt: skip


@st.cache_data(show_spinner=False, max_entries=64)
def _classified(doc_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return pipeline.classify_extraordinary_reports(get_settings(), doc_ids)


def _mtime(path: Path) -> float:
    """Part of a cache key: a rewritten file (radar update / walkforward) is read again."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner="Loading data…", max_entries=8)
def _read_table(path: Path, mtime: float) -> pd.DataFrame:
    """Shared, NOT copied (cache_resource): copying the 5-million-row price table on every click
    cost ~2 s.
    """
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Indexing prices…", max_entries=2)
def _prices_by_code(mtime: float) -> dict[str, pd.DataFrame]:
    """One company's rows in O(1) instead of scanning 5 million rows per card."""
    prices = _staging("prices")
    if prices is None:
        return {}
    return {str(code): group for code, group in prices.groupby("code", sort=False)}


def _stock_prices(code: object) -> pd.DataFrame | None:
    path = staging_table_path(get_settings().staging_dir, "prices")
    if not path.exists():
        return None
    return _prices_by_code(_mtime(path)).get(str(code))  # None: no TSE price for it


@st.cache_resource
def _patterns() -> HolderPatterns:
    return HolderPatterns.from_config(load_yaml("ownership"))


STYLE = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px;}
  h1 {font-size: 1.9rem !important; font-weight: 700; margin-bottom: 0 !important;}
  h3 {font-size: 1.15rem !important; font-weight: 600 !important; padding-top: 0.8rem !important;}
  [data-testid="stMetric"] {background: #F7F8FA; border: 1px solid #E6E9EF; border-radius: 8px;
                            padding: 0.7rem 1rem;}
  [data-testid="stMetricValue"] {font-size: 1.45rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #808495;}
  .stTabs [data-baseweb="tab-list"] {gap: 1.5rem;}
  .stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def _header() -> None:
    """Title, one line on what the tool does, and how fresh the data is."""
    st.markdown(STYLE, unsafe_allow_html=True)
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    left.title("Japan Take-Private Radar")
    left.caption(
        "Daily screen of Japanese listed companies for take-private likelihood "
        "(parent buyouts · MBOs · tender offers) — EDINET · J-Quants · JPX"
    )
    with right:
        _freshness_banner()


def _footer() -> None:
    st.divider()
    with st.expander("About & methodology"):
        st.markdown(ABOUT)
    st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="Japan Take-Private Radar", layout="wide")
    _header()
    marts_dir = get_settings().marts_dir
    screen, alerts, journal, deals, evidence, portfolio = st.tabs(
        ["Screen", "Alerts", "Journal", "Deal study", "Evidence", "Portfolio"]
    )
    with screen:
        radar_tab(marts_dir)
    with alerts:
        changes_tab(marts_dir)
    with journal:
        journal_tab()
    with deals:
        offers_tab(marts_dir)
    with evidence:
        backtest_tab(marts_dir)
    with portfolio:
        portfolio_tab(marts_dir)
    _footer()


main()
