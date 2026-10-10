"""Application layer: orchestrates ingest → parse → features → model, and reads/writes storage."""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

from radar.config import CONFIG_DIR, Settings, load_yaml
from radar.features.activism import (
    DiscoveryConfig,
    activist_candidates,
    activist_features,
    all_patterns,
    initial_reports_per_filer,
)
from radar.features.build import TradableConfig, audit_features, build_feature_table
from radar.features.market import unusual_trading
from radar.features.ownership import HolderPatterns
from radar.features.point_in_time import latest_annual_filings, price_as_of
from radar.features.valuation import book_value_per_share, split_multiplier
from radar.ingest.edinet import (
    DocumentMeta,
    DocumentNotAvailableError,
    EdinetClient,
    EdinetError,
    FetchType,
    iter_days,
)
from radar.ingest.jpx import load_capital_plan_lists
from radar.ingest.jquants import JQuantsClient, bars_frame, master_frame, topix_frame
from radar.labels.tender_offers import (
    already_public,
    label_frame,
    take_private_events,
    tender_offer_events,
)
from radar.model.allocation import AllocationConfig, allocate, kelly_fraction
from radar.model.backtest import (
    Choice,
    ablation,
    choose_default,
    compare_rankings,
    evaluate_by_type,
    evaluate_ranking,
    live_scoreboard,
    quarter_ends,
    time_split,
    walk_forward_splits,
)
from radar.model.event_study import (
    EventStudyConfig,
    earlier_offer_within,
    event_study,
    volume_ratio,
)
from radar.model.expected_value import (
    apply_isotonic,
    brier,
    expected_value,
    fit_isotonic,
    reliability_table,
)
from radar.model.inference import style_benchmark
from radar.model.logistic import (
    LogisticConfig,
    coefficients,
    combine,
    design_matrix,
    fit,
    fit_by_type,
    predict,
    prior_correct,
)
from radar.model.offer_price import OfferRangeConfig, offer_pbr
from radar.model.portfolio import (
    AdjustedPrices,
    TradingCosts,
    ValueBasketConfig,
    holding_return,
    live_returns,
    pick_returns,
    sector_neutral,
    summarize_picks,
    top_k_breakdown,
    union_rank_score,
    value_basket_score,
    value_catalyst_score,
    value_momentum_score,
)
from radar.model.return_model import fit_return_model, predict_return
from radar.model.rules import (
    RuleSet,
    audit_scores,
    bootstrap_rule_weights,
    default_ruleset_name,
    fit_rule_model,
    rule_flags,
    score,
    with_weights,
)
from radar.parse.elements import ElementMap
from radar.parse.excerpts import report_excerpts
from radar.parse.extraordinary_report import ExtraordinaryConfig, classify_extraordinary
from radar.parse.financials import extract_financials
from radar.parse.large_holding import FORM_PREFIX as LARGE_HOLDING_PREFIX
from radar.parse.large_holding import holding_purposes
from radar.parse.officers import OFFICER_COLUMNS, extract_officers
from radar.parse.rental_property import RentalConfig
from radar.parse.shareholders import SHAREHOLDER_COLUMNS, extract_major_shareholders
from radar.parse.tender_offer import TenderOfferConfig, classify_event, extract_tender_offer
from radar.parse.xbrl_csv import read_facts_zip
from radar.report.changes import (
    alerts_markdown,
    compare_scores,
    coverage_drift,
    new_events,
    watch_codes,
)
from radar.report.factpack import FactPackInputs, factpack_markdown
from radar.report.journal import journal_outcomes, parse_journal
from radar.report.views import (
    hit_rate_by_band,
    holders_table,
    likely_buyer,
    offer_range_for,
    offer_range_table,
    officers_table,
    premium_summary,
)
from radar.storage.files import DAILY_COMPRESSION, expired_dated_files, write_parquet_atomic
from radar.storage.paths import (
    edinet_doc_path,
    edinet_doclist_path,
    ensure_parent,
    factpack_path,
    journal_path,
    live_model_path,
    live_scores_path,
    mart_table_path,
    return_model_path,
    staging_table_path,
)

logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = "documents"
TENDER_OFFER_EVENTS_TABLE = "events_tender_offers"
OFFICERS_TABLE = "officers"
SHAREHOLDERS_TABLE = "shareholders"
FINANCIALS_TABLE = "financials"
PRICES_TABLE = "prices"
LISTINGS_TABLE = "listings"
CAPITAL_PLANS_TABLE = "capital_plans"
TOPIX_TABLE = "topix"
TENDER_OFFERS_TABLE = "tender_offers"
ACTIVIST_CANDIDATES_TABLE = "activist_candidates"
DISCLOSED = "開示済"  # JPX list status: plan disclosed (vs 検討中, under consideration)
ANNUAL_REPORT_TYPES = ("120", "130")  # 有価証券報告書 and its amendment
LINEAGE_COLUMNS = ["doc_id", "edinet_code"]


def documents_frame(metas: Iterable[DocumentMeta]) -> pd.DataFrame:
    rows = [m.model_dump() for m in metas]
    columns = list(DocumentMeta.model_fields)
    return pd.DataFrame(rows, columns=columns)


def load_documents(settings: Settings) -> pd.DataFrame:
    path = staging_table_path(settings.staging_dir, DOCUMENTS_TABLE)
    if not path.exists():
        return documents_frame([])
    return pd.read_parquet(path)


def merge_documents(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union by doc_id; the newest copy of a row wins (withdrawal status can change)."""
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset="doc_id", keep="last").reset_index(drop=True)


def sync_document_index(client: EdinetClient, settings: Settings, start: date, end: date) -> int:
    """Fetch daily document lists for [start, end] and merge them into staging/documents.parquet."""
    new_rows = [meta for day in iter_days(start, end) for meta in client.list_documents(day)]
    merged = merge_documents(load_documents(settings), documents_frame(new_rows))
    # Atomic: the hours-long backfill may be reading this file while `radar update` writes it.
    write_parquet_atomic(merged, staging_table_path(settings.staging_dir, DOCUMENTS_TABLE))
    logger.info("document index: %d new rows, %d total", len(new_rows), len(merged))
    return len(new_rows)


def select_documents(
    documents: pd.DataFrame,
    doc_types: Iterable[str],
    start: date | None = None,
    end: date | None = None,
    only_listed: bool = True,
) -> pd.DataFrame:
    """Filter the index: given types, submission window, not withdrawn, CSV available."""
    submitted = pd.to_datetime(documents["submit_datetime"]).dt.date
    mask = documents["doc_type_code"].isin(list(doc_types)) & (
        documents["withdrawal_status"].astype(str) == "0"
    )
    mask &= documents["csv_flag"].astype(bool)
    if start:
        mask &= submitted >= start
    if end:
        mask &= submitted <= end
    if only_listed:
        mask &= documents["sec_code"].notna()
    return documents[mask]


MAX_CONSECUTIVE_FAILURES = 10


def download_documents(
    client: EdinetClient, documents: pd.DataFrame, limit: int | None = None
) -> list[Path]:
    """Download (or find in the cache) each document; returns the paths available."""
    paths: list[Path] = []
    consecutive = 0
    for doc_id in documents["doc_id"].head(limit) if limit else documents["doc_id"]:
        try:
            paths.append(client.download(str(doc_id), FetchType.CSV))
            consecutive = 0
        except DocumentNotAvailableError as exc:
            logger.warning("skipped: %s", exc)
        except EdinetError as exc:  # already redacted: never contains the API key
            consecutive += 1
            logger.warning("network error on %s (%d in a row): %s", doc_id, consecutive, exc)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise EdinetError(f"{consecutive} consecutive failures: {exc}") from None
    return paths


def build_tender_offer_events(settings: Settings) -> pd.DataFrame:
    events = tender_offer_events(load_documents(settings))
    path = ensure_parent(mart_table_path(settings.marts_dir, TENDER_OFFER_EVENTS_TABLE))
    events.to_parquet(path, index=False)
    return events


@dataclass(frozen=True)
class ParseResult:
    parsed: int
    failed: list[str]  # doc_ids whose file could not be read


def parse_downloaded_filings(
    settings: Settings, elements: ElementMap | None = None, *, only_new: bool = False
) -> ParseResult:
    """Parse every downloaded annual report into staging officers / shareholders / financials."""
    elements = elements or ElementMap.load()
    rental = RentalConfig.from_config(load_yaml("trends"))
    documents = load_documents(settings)
    annual = documents[documents["doc_type_code"].isin(ANNUAL_REPORT_TYPES)]
    existing = {
        t: _optional_staging(settings, t)
        for t in (OFFICERS_TABLE, SHAREHOLDERS_TABLE, FINANCIALS_TABLE)
    }
    parsed = existing[FINANCIALS_TABLE]
    known: set[str] = set(parsed["doc_id"]) if only_new and parsed is not None else set()
    officers: list[pd.DataFrame] = []
    holders: list[pd.DataFrame] = []
    financials: list[dict[str, object]] = []
    failed: list[str] = []
    for doc in annual.sort_values("doc_id").to_dict("records"):
        path = edinet_doc_path(settings.raw_dir, str(doc["doc_id"]), FetchType.CSV)
        if not path.exists() or doc["doc_id"] in known:
            continue
        try:
            facts = read_facts_zip(path)
        except (zipfile.BadZipFile, ValueError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s: %s", doc["doc_id"], type(exc).__name__)
            failed.append(str(doc["doc_id"]))
            continue
        lineage = {key: doc[key] for key in LINEAGE_COLUMNS}
        officers.append(extract_officers(facts, elements).assign(**lineage))
        holders.append(extract_major_shareholders(facts, elements).assign(**lineage))
        financials.append(
            {
                **lineage,
                "period_end": doc["period_end"],
                **extract_financials(facts, elements, rental),
            }
        )

    new_financials = _financials_frame(financials, elements)
    if known:  # keep what is already parsed; the new reports are appended
        officers.insert(0, _require(existing[OFFICERS_TABLE]))
        holders.insert(0, _require(existing[SHAREHOLDERS_TABLE]))
        new_financials = pd.concat([_require(existing[FINANCIALS_TABLE]), new_financials])
    _write_staging(settings, OFFICERS_TABLE, officers, OFFICER_COLUMNS)
    _write_staging(settings, SHAREHOLDERS_TABLE, holders, SHAREHOLDER_COLUMNS)
    _write_staging(settings, FINANCIALS_TABLE, [new_financials], [])
    logger.info("parsed %d annual reports, %d unreadable", len(financials), len(failed))
    return ParseResult(parsed=len(financials), failed=failed)


def _require(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        raise FileNotFoundError("staging table missing: run a full `radar parse` first")
    return frame


def _financials_frame(records: list[dict[str, object]], elements: ElementMap) -> pd.DataFrame:
    """Amounts as float64 even when a column is all None (else Parquet infers a null type)."""
    amounts = [
        *elements.financials,
        *elements.shares,
        *elements.per_share,
        "rental_book_value",
        "rental_fair_value",
    ]
    columns = [
        *LINEAGE_COLUMNS,
        "period_end",
        *amounts,
        "is_consolidated",
        "is_jgaap",
        "hq_address",
        "rental_note_status",
    ]
    return pd.DataFrame(records, columns=columns).astype({c: "float64" for c in amounts})


def _write_staging(
    settings: Settings, table: str, frames: list[pd.DataFrame], columns: list[str]
) -> None:
    """Concatenate per-filing frames, lineage first; an empty run still writes the columns."""
    non_empty = [f for f in frames if not f.empty]
    frame = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    ordered = [*LINEAGE_COLUMNS, *[c for c in columns if c not in LINEAGE_COLUMNS]]
    ordered += [c for c in frame.columns if c not in ordered]
    frame = frame.reindex(columns=ordered)
    if not frame.empty:  # same row order for a full and an incremental parse
        frame = frame.sort_values("doc_id", kind="stable").reset_index(drop=True)
    frame.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, table)), index=False)


def load_staging(settings: Settings, table: str) -> pd.DataFrame:
    return pd.read_parquet(staging_table_path(settings.staging_dir, table))


def _optional_staging(settings: Settings, table: str) -> pd.DataFrame | None:
    path = staging_table_path(settings.staging_dir, table)
    return pd.read_parquet(path) if path.exists() else None


def _feature_inputs(settings: Settings) -> dict[str, Any]:
    """Every staging input of `build_feature_table`, loaded once (the backtest reuses them)."""
    prices = _optional_staging(settings, PRICES_TABLE)
    if prices is None:
        logger.warning("no staging/prices yet: valuation features will be missing")
        prices = pd.DataFrame(columns=["code", "date", "close"])
    return {
        "documents": load_documents(settings),
        "officers": load_staging(settings, OFFICERS_TABLE),
        "shareholders": load_staging(settings, SHAREHOLDERS_TABLE),
        "financials": load_staging(settings, FINANCIALS_TABLE),
        "prices": prices,
        "listings": _optional_staging(settings, LISTINGS_TABLE),
        "capital_plans": _disclosed_capital_plans(settings),
        "holder_patterns": HolderPatterns.from_config(load_yaml("ownership")),
        "activist_patterns": tuple(load_yaml("activists")["activist_filers"]),  # dated in build
        "index_start": index_start(settings),
        "capital_plan_segments": tuple(load_yaml("jpx")["requested_segments"]),
        "portfolio_min_stake": float(load_yaml("ownership")["portfolio_min_stake"]),
        "trend_years": int(load_yaml("trends")["years"]),
        "sector_min_peers": int(load_yaml("trends")["sector_min_peers"]),
        "homonym_list": load_yaml("ownership").get("homonyms") or [],
        "rental_tax_rate": float(load_yaml("trends")["rental_tax_rate"]),
        "new_holder_window_days": int(load_yaml("trends")["new_holder_window_days"]),
        "tradable": TradableConfig(
            criteria=load_yaml("jpx")["tradable_criteria"],
            major_holder=float(load_yaml("jpx")["tradable_major_holder"]),
            patterns=HolderPatterns.from_config(load_yaml("ownership")),
        ),
    }


def index_start(settings: Settings) -> date | None:
    """First day of the contiguous run of fetched EDINET daily lists that ends at the latest one."""
    days = sorted(
        date.fromisoformat(p.stem)
        for p in edinet_doclist_path(settings.raw_dir, date.today()).parent.glob("*.json")
    )
    if not days:
        return None
    start = days[-1]
    for earlier in reversed(days[:-1]):
        if (start - earlier).days != 1:
            break
        start = earlier
    return start


def build_features(settings: Settings, as_of: date) -> pd.DataFrame:
    """Feature table for `as_of` → marts/features_<as_of>.parquet."""
    features = build_feature_table(as_of, **_feature_inputs(settings))
    path = ensure_parent(mart_table_path(settings.marts_dir, f"features_{as_of.isoformat()}"))
    features.to_parquet(path, index=False, compression=DAILY_COMPRESSION)
    return features


def score_companies(settings: Settings, as_of: date, ruleset: str | None = None) -> pd.DataFrame:
    """Rule-based score of marts/features_<as_of> → marts/scores_<as_of>.parquet (ranked)."""
    source = mart_table_path(settings.marts_dir, f"features_{as_of.isoformat()}")
    if not source.exists():
        raise FileNotFoundError(f"{source.name} not found; run `radar features` first")
    name = ruleset or default_ruleset_name()
    features = pd.read_parquet(source)
    scores = score(features, RuleSet.load(name))
    if name != BASELINE_RULES:
        baseline = score(features, RuleSet.load(BASELINE_RULES)).set_index("edinet_code")["score"]
        scores["score_v0"] = scores["edinet_code"].map(baseline)
    scores = _add_portfolio_columns(settings, _add_live_models(settings, scores))
    path = ensure_parent(mart_table_path(settings.marts_dir, f"scores_{as_of.isoformat()}"))
    scores.to_parquet(path, index=False, compression=DAILY_COMPRESSION)
    return scores


BASELINE_RULES = "scoring_v0"  # hand-set weights: the research baseline
COMBO = "combo_v2_mbo"  # top 7 of the offer ranking (v2) + top 7 of MBO likelihood, in turn
LIVE_MODEL_COLUMNS = [
    COMBO,
    "score_v0",
    "value_basket",
    "value_basket_quality",
    "value_momentum",
    "value_catalyst",
    "return_model",
    "return_model_sector_neutral",
    "p_any",
    "p_any_prior",
    "p_mbo_prior",  # the MBO list (dashboard ordering), judged live on future offers
    "expected_value",
]


def _basket_config(raw: Mapping[str, Any], *, quality: bool) -> ValueBasketConfig:
    """The value basket of config/model.yaml; `quality` adds the round-2 operating-profit filter."""
    basket = raw["value_basket"]
    return ValueBasketConfig(
        float(basket["max_market_cap"]),
        float(basket["max_pbr"]),
        float(basket["min_net_cash"]),
        float(raw["returns"]["round2"]["min_operating_yield"]) if quality else None,
    )


def _add_portfolio_columns(settings: Settings, scores: pd.DataFrame) -> pd.DataFrame:
    """`value_basket`, its quality variant and, when saved by `radar walkforward`, `return_model`
    (the challenger's predicted excess return) and its sector-neutral variant — all frozen daily so
    the live scoreboard can compare them.
    """
    raw = load_yaml("model")
    scores["value_basket"] = value_basket_score(scores, _basket_config(raw, quality=False))
    if "operating_income_to_market_cap" in scores.columns:
        scores["value_basket_quality"] = value_basket_score(
            scores, _basket_config(raw, quality=True)
        )
    basket = _basket_config(raw, quality=False)
    if "momentum_12_1" in scores.columns:
        worst = float(raw["returns"]["round3"]["momentum_worst_share"])
        scores["value_momentum"] = value_momentum_score(scores, basket, worst)
    if {"p_parent_prior", "p_mbo_prior"} <= set(scores.columns):
        # Ranked on the prior-corrected P, as in the walk-forward: isotonic would tie the top.
        prior = scores.assign(p_parent=scores["p_parent_prior"], p_mbo=scores["p_mbo_prior"])
        scores["value_catalyst"] = value_catalyst_score(prior, basket)
    mbo = "p_mbo_prior" if "p_mbo_prior" in scores.columns else "p_mbo"
    if mbo in scores.columns and "p_any_prior" in scores.columns:
        scores[COMBO] = union_rank_score(scores, {"score": "p_any_prior", mbo: None})
    path = return_model_path(settings.marts_dir)
    if path.exists() and not scores.empty:
        bundle = joblib.load(path)
        x = design_matrix(scores, bundle["config"])
        scores["return_model"] = predict_return(bundle["model"], x, bundle["columns"])
        scores["return_model_sector_neutral"] = sector_neutral(
            scores["return_model"], scores["sector33"]
        )
    return scores


def _add_live_models(settings: Settings, scores: pd.DataFrame) -> pd.DataFrame:
    """Calibrated P per deal type, P(any offer) and expected value from the frozen live models
    (`radar model`), next to the rule score.
    """
    path = live_model_path(settings.marts_dir)
    if not path.exists() or scores.empty:
        return scores
    bundle = joblib.load(path)
    models, config, rates = bundle["models"], bundle["config"], bundle["base_rates"]
    if not models or not all(k in rates for k in models):
        return scores
    calibrated = _calibrated(scores, models, config, rates)
    prior_any = combine({k: calibrated[k] for k in calibrated.columns})
    # Isotonic is a step function: it ties companies at the top. The ORDER the walk-forward
    # validated is the prior-corrected one, so it is kept for ranking; p_any is for reading.
    scores["p_any_prior"] = prior_any
    for kind in calibrated.columns:  # the per-type order the walk-forward validated (round 3)
        scores[f"p_{kind}_prior"] = calibrated[kind]
    if bundle.get("calibrator") is not None:
        isotonic_any = apply_isotonic(bundle["calibrator"], prior_any)
        scale = (isotonic_any / prior_any).where(prior_any > 0)
        calibrated = calibrated.mul(scale, axis=0)  # per type: same share of the new level
        prior_any = isotonic_any
    for kind in calibrated.columns:
        scores[f"p_{kind}"] = calibrated[kind]
    scores["p_any"] = prior_any
    scores["expected_value"] = expected_value(
        calibrated, bundle["premiums"], bundle["no_offer_return"]
    )
    return scores


def run_backtest(
    settings: Settings, start: date, end: date, ruleset: str = "scoring_v0"
) -> pd.DataFrame:
    """Score every quarter-end in [start, end] and compare with real tender offers."""
    config = load_yaml("backtest")
    horizon = int(config["horizon_days"])
    inputs = _feature_inputs(settings)
    offers = _optional_staging(settings, TENDER_OFFERS_TABLE)
    events = take_private_events(tender_offer_events(inputs["documents"]), offers)
    known_until = pd.to_datetime(inputs["documents"]["submit_datetime"]).max().date()
    rules = RuleSet.load(ruleset or default_ruleset_name())
    adjusted = AdjustedPrices(inputs["prices"]) if "adj_factor" in inputs["prices"] else None
    tolerance = float(load_yaml("event_study")["pre_announced_tolerance"])

    panels, rows = [], []
    for as_of in quarter_ends(start, end):
        scored = score(build_feature_table(as_of, **inputs), rules)
        labels = label_frame(scored["edinet_code"], events, offers, as_of, horizon)
        scored = pd.concat([scored, labels], axis=1)
        public = _already_public_codes(
            scored, events, offers, adjusted, as_of=as_of, horizon=horizon, tolerance=tolerance
        )
        scored["excluded_already_public"] = scored["edinet_code"].isin(public)
        panels.append(scored)
        judged = scored[~scored["excluded_already_public"]]
        metrics = evaluate_ranking(
            judged["score"], judged["label"], config["precision_at"], config["top_fraction"]
        )
        metrics["excluded_already_public"] = len(public)
        typed = evaluate_by_type(
            judged["score"], judged[list(labels.columns.drop("label"))], config["top_fraction"]
        )
        complete = as_of + timedelta(days=horizon) <= known_until
        rows.append({"as_of": as_of, "label_complete": complete, **metrics, **typed})

    if panels:
        panel = pd.concat(panels, ignore_index=True)
        path = mart_table_path(settings.marts_dir, f"backtest_panel_{ruleset}")
        panel.to_parquet(ensure_parent(path))
    metrics_frame = pd.DataFrame(rows)
    path = ensure_parent(mart_table_path(settings.marts_dir, f"backtest_metrics_{ruleset}"))
    metrics_frame.to_parquet(path, index=False)
    return metrics_frame


def _already_public_codes(
    scored: pd.DataFrame,
    events: pd.DataFrame,
    offers: pd.DataFrame | None,
    adjusted: AdjustedPrices | None,
    *,
    as_of: date,
    horizon: int,
    tolerance: float,
) -> set[str]:
    """Targets in this date's label window whose offer was already public on as_of."""
    if adjusted is None or offers is None or offers.empty:
        return set()
    end = as_of + timedelta(days=horizon)
    window = events[(events["event_date"] > as_of) & (events["event_date"] <= end)]
    window = window.merge(offers[["doc_id", "offer_price"]], on="doc_id", how="left")
    codes = dict(zip(scored["edinet_code"], scored["sec_code"], strict=True))
    closes: dict[str, float] = {}
    for target in window["subject_edinet_code"].unique():
        sec_code = codes.get(target)
        close = adjusted.close_on_or_before(str(sec_code), as_of - timedelta(days=1))
        if sec_code is not None and close is not None:
            closes[str(target)] = close[1]
    return already_public(window, closes, tolerance)


def _evaluable(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows the backtest judges: offers already public on as_of are left out."""
    if "excluded_already_public" not in panel.columns:
        return panel
    kept: pd.DataFrame = panel[~panel["excluded_already_public"].astype(bool)]
    return kept


def _merge_staging(settings: Settings, table: str, new: pd.DataFrame, key: list[str]) -> int:
    """Union `new` into a staging table by `key` (the newest copy wins); returns rows added."""
    existing = _optional_staging(settings, table)
    combined = new if existing is None else pd.concat([existing, new], ignore_index=True)
    merged = combined.drop_duplicates(subset=key, keep="last").sort_values(key)
    path = ensure_parent(staging_table_path(settings.staging_dir, table))
    merged.reset_index(drop=True).to_parquet(path, index=False)
    return len(merged) - (0 if existing is None else len(existing))


def sync_prices(client: JQuantsClient, settings: Settings, start: date, end: date) -> int:
    """Daily bars of every stock for each weekday in [start, end] → staging/prices.parquet."""
    days = [d.date() for d in pd.bdate_range(start, end)]
    frames: list[pd.DataFrame] = []
    try:
        for day in days:
            frames.append(bars_frame(client.daily_bars(day)))
    finally:  # an API error (e.g. an expired key) must not discard the days already fetched
        non_empty = [f for f in frames if not f.empty]
        if non_empty:
            added = _merge_staging(settings, PRICES_TABLE, pd.concat(non_empty), ["code", "date"])
            logger.info("prices: %d of %d days, %d new rows", len(frames), len(days), added)
    return len(days)


def sync_listings(client: JQuantsClient, settings: Settings, start: date, end: date) -> int:
    """Listed-company master at each month-end in [start, end] → staging/listings.parquet."""
    names = load_yaml("jquants")["segment_names"]
    days = [d.date() for d in pd.date_range(start, end, freq="ME")]
    frames = [master_frame(client.equities_master(day), names) for day in days]
    non_empty = [f for f in frames if not f.empty]
    if non_empty:
        _merge_staging(settings, LISTINGS_TABLE, pd.concat(non_empty), ["code", "date"])
    return len(days)


def _disclosed_capital_plans(settings: Settings) -> pd.DataFrame | None:
    """JPX lists restricted to 開示済 (disclosed); 検討中 (under consideration) is not a plan."""
    lists = _optional_staging(settings, CAPITAL_PLANS_TABLE)
    if lists is None:
        return None
    disclosed: pd.DataFrame = lists[lists["status"] == DISCLOSED]
    return disclosed


def sync_capital_plans(settings: Settings) -> pd.DataFrame:
    """Excel lists in data/manual/capital_plans → staging/capital_plans.parquet (rebuilt)."""
    lists = load_capital_plan_lists(settings.manual_dir / "capital_plans")
    path = ensure_parent(staging_table_path(settings.staging_dir, CAPITAL_PLANS_TABLE))
    lists.to_parquet(path, index=False)
    return lists


def run_ablation(settings: Settings, ruleset: str = "scoring_v0") -> pd.DataFrame:
    """Rule ablation on the last backtest → marts/ablation.parquet."""
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{ruleset}"))
    metrics_path = mart_table_path(settings.marts_dir, f"backtest_metrics_{ruleset}")
    metrics = pd.read_parquet(metrics_path)
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    usable = _evaluable(panel[panel["as_of"].isin(complete)])
    if usable.empty:
        logger.warning("no as_of date with a complete label window: nothing to ablate")
        return pd.DataFrame()
    config = load_yaml("backtest")
    result = ablation(usable, RuleSet.load(ruleset), config["precision_at"], config["top_fraction"])
    path = ensure_parent(mart_table_path(settings.marts_dir, f"ablation_{ruleset}"))
    result.to_parquet(path, index=False)
    return result


def run_model(settings: Settings, ruleset: str = "scoring_v0") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logistic regression (pooled and per deal type) vs the rule set on the same test dates."""
    config = LogisticConfig.from_config(load_yaml("model"))
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{ruleset}"))
    metrics = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{ruleset}"))
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    train, test = time_split(
        _evaluable(panel[panel["as_of"].isin(complete)]), config.train_end, config.test_start
    )
    if train.empty or test.empty:
        raise ValueError("the backtest panel has no complete dates on one side of the split")
    model = fit(train, config)
    test = test.assign(logistic=predict(model, test, config)).rename(columns={"score": ruleset})
    names = [ruleset, "logistic"]
    # One model per deal type (rules find MBOs, the pooled model finds parent deals); a type
    # without a positive in training cannot be fitted and is skipped, not faked.
    kinds = [
        k for k in load_yaml("model")["event_types"]
        if f"label_{k}" in train.columns and train[f"label_{k}"].nunique() == 2
    ]  # fmt: skip
    skipped = set(load_yaml("model")["event_types"]) - set(kinds)
    if skipped:
        logger.warning("no positive in training for %s: no per-type model", sorted(skipped))
    models = fit_by_type(train, config, kinds)
    per_type = {k: predict(m, test, config) for k, m in models.items()}
    for kind, probability in per_type.items():
        test[f"logistic_{kind}"] = probability
        names.append(f"logistic_{kind}")
    if per_type:
        test["logistic_by_type"] = combine(per_type)
        names.append("logistic_by_type")
        # Hybrid: the rules' rank (MBOs) plus the parent model's rank, within each date.
        partner = per_type.get("parent", test["logistic_by_type"])
        test["hybrid"] = test.groupby("as_of")[ruleset].rank(pct=True) + partner.groupby(
            test["as_of"]
        ).rank(pct=True)
        names.append("hybrid")
    backtest = load_yaml("backtest")
    ev_inputs = _ev_inputs(settings, train, kinds)
    if ev_inputs is not None and models:
        premiums, no_offer, base_rates = ev_inputs
        calibrated = _calibrated(test, models, config, base_rates)
        test["expected_value"] = expected_value(calibrated, premiums, no_offer)
        names.append("expected_value")
        _evaluate_expected_value(settings, ruleset, test, calibrated)
        _save_live_models(settings, panel[panel["as_of"].isin(complete)], config, kinds, premiums)
    comparison = compare_rankings(test, names, backtest["precision_at"], backtest["top_fraction"])
    coefs = coefficients(model)
    comparison.to_parquet(
        ensure_parent(mart_table_path(settings.marts_dir, f"model_comparison_{ruleset}")),
        index=False,
    )
    coefs.to_parquet(
        mart_table_path(settings.marts_dir, f"logistic_coefficients_{ruleset}"), index=False
    )
    for kind, fitted in models.items():
        coefficients(fitted).to_parquet(
            mart_table_path(settings.marts_dir, f"logistic_coefficients_{ruleset}_{kind}"),
            index=False,
        )
    logger.info("model: %d training rows, %d test rows", len(train), len(test))
    return comparison, coefs


def _ev_inputs(
    settings: Settings, train: pd.DataFrame, kinds: list[str]
) -> tuple[dict[str, float], float, dict[str, float]] | None:
    """(median premium per type, no-offer return, training base rate per type) — the inputs of the
    expected value.
    """
    study_path = mart_table_path(settings.marts_dir, "event_study")
    prices = _optional_staging(settings, PRICES_TABLE)
    if not study_path.exists() or prices is None or "sec_code" not in train.columns:
        return None
    summary = premium_summary(pd.read_parquet(study_path)).set_index("event_type")
    medians = summary["median_premium"]
    premiums = {k: float(medians.get(k, float("nan"))) for k in kinds if k in summary.index}
    adjusted = _adjusted_prices(prices)
    config = load_yaml("backtest")
    horizon, stale = int(config["horizon_days"]), int(config["pick_stale_days"])
    returns = []
    for row in train[train["label"] == 0][["sec_code", "as_of"]].to_dict("records"):
        as_of = pd.Timestamp(str(row["as_of"])).date()
        held = holding_return(
            adjusted, str(row["sec_code"]), as_of, as_of + timedelta(days=horizon), stale
        )
        if held is not None:
            returns.append(held[0])
    if not returns or not premiums:
        return None
    base_rates = {k: float(train[f"label_{k}"].mean()) for k in kinds}
    return premiums, float(pd.Series(returns).median()), base_rates


def _calibrated(
    frame: pd.DataFrame,
    models: Mapping[str, Any],
    config: LogisticConfig,
    rates: Mapping[str, float],
) -> pd.DataFrame:
    """Prior-corrected P per deal type (columns = types)."""
    return pd.DataFrame(
        {k: prior_correct(predict(m, frame, config), rates[k]) for k, m in models.items()},
        index=frame.index,
    )


def _evaluate_expected_value(
    settings: Settings, ruleset: str, test: pd.DataFrame, calibrated: pd.DataFrame
) -> None:
    """Calibration table and top-N pick returns per ranking on the test dates."""
    p_any = combine({k: calibrated[k] for k in calibrated.columns})
    bins = int(load_yaml("model")["reliability_bins"])
    reliability_table(p_any, test["label"], bins).to_parquet(
        ensure_parent(mart_table_path(settings.marts_dir, f"model_reliability_{ruleset}")),
        index=False,
    )
    backtest = load_yaml("backtest")
    adjusted = _adjusted_prices(load_staging(settings, PRICES_TABLE))
    topix = load_staging(settings, TOPIX_TABLE)
    frames = []
    for column in (ruleset, "logistic_by_type", "expected_value"):
        if column not in test.columns:
            continue
        ranked = test.assign(score=test[column])
        picks = pick_returns(
            ranked,
            adjusted,
            topix,
            horizon_days=int(backtest["horizon_days"]),
            sizes=[int(n) for n in backtest["pick_sizes"]],
            stale_days=int(backtest["pick_stale_days"]),
        )
        frames.append(picks.assign(model=column))
    pd.concat(frames, ignore_index=True).to_parquet(
        mart_table_path(settings.marts_dir, f"pick_returns_by_model_{ruleset}"), index=False
    )


def fit_with_isotonic(
    panel: pd.DataFrame, config: LogisticConfig, kinds: list[str]
) -> tuple[dict[str, Any], dict[str, float], IsotonicRegression] | None:
    """Per-type models fitted on the dates ≥ 1 year before the panel's last date, and an isotonic
    map of their P(any offer) fitted on that last year — the scheme the walk-forward validated
    (Brier 0.0227 vs 0.0256).
    """
    as_of = panel["as_of"].map(lambda d: pd.Timestamp(str(d)).date())
    cut = as_of.max() - timedelta(days=365)
    fit_part, calib_part = panel[as_of <= cut], panel[as_of > cut]
    usable = [k for k in kinds if fit_part[f"label_{k}"].nunique() == 2]
    if fit_part.empty or calib_part.empty or not usable:
        return None
    models = fit_by_type(fit_part, config, usable)
    rates = {k: float(fit_part[f"label_{k}"].mean()) for k in usable}
    p_calib = _p_any(_calibrated(calib_part, models, config, rates))
    return models, rates, fit_isotonic(p_calib, calib_part["label"])


def _save_live_models(
    settings: Settings,
    complete_panel: pd.DataFrame,
    config: LogisticConfig,
    kinds: list[str],
    premiums: Mapping[str, float],
) -> None:
    """Per-type models frozen for daily scoring, judged later by the live scoreboard."""
    everything = _evaluable(complete_panel)
    _, no_offer, rates = _ev_inputs(settings, everything, kinds) or ({}, float("nan"), {})
    fitted = fit_with_isotonic(everything, config, kinds)
    models, calibrator = fit_by_type(everything, config, kinds), None
    if fitted is not None:
        models, rates, calibrator = fitted
    else:
        logger.warning("live models: no year to hold out, P stays prior-corrected")
    bundle = {
        "models": models,
        "config": config,
        "base_rates": rates,
        "premiums": dict(premiums),
        "no_offer_return": no_offer,
        "calibrator": calibrator,
        "fitted_on": sorted({str(d) for d in everything["as_of"]}),
    }
    joblib.dump(bundle, ensure_parent(live_model_path(settings.marts_dir)))


def run_walk_forward(
    settings: Settings, ruleset: str = "scoring_v0"
) -> tuple[pd.DataFrame, Choice, Choice, dict[str, Choice]]:
    """Walk-forward choice of the default ranking."""
    raw, config = load_yaml("model"), LogisticConfig.from_config(load_yaml("model"))
    wf = raw["walk_forward"]
    backtest = load_yaml("backtest")
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{ruleset}"))
    metrics = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{ruleset}"))
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    usable = _evaluable(panel[panel["as_of"].isin(complete)])
    splits = walk_forward_splits(
        sorted({pd.Timestamp(str(d)).date() for d in usable["as_of"]}),
        int(backtest["horizon_days"]),
        wf["first_test"],
    )
    context = _WalkForwardContext.load(settings, usable, ruleset, raw)
    candidates = [
        *wf["simplicity"],
        *raw["returns"]["simplicity"],
        *raw["returns"]["round2"]["simplicity"],
        *raw["returns"]["round3"]["simplicity"],
        "v2_no_activist",
        COMBO,  # tracked, not a candidate of any pre-registered rule
    ]
    rows, pick_frames, scored_frames = [], [], []
    for train_dates, test_date in splits:
        train = usable[usable["as_of"].map(lambda d: pd.Timestamp(str(d)).date()).isin(train_dates)]
        test = usable[usable["as_of"].map(lambda d: pd.Timestamp(str(d)).date()) == test_date]
        scored = context.score(train, test.copy(), config)
        scored_frames.append(scored)
        names = [n for n in dict.fromkeys(candidates) if n in scored.columns]
        rows.append(
            compare_rankings(scored, names, backtest["precision_at"], backtest["top_fraction"])
        )
        pick_frames += [
            context.picks(scored.assign(score=scored[name])).assign(model=name) for name in names
        ]
    result = pd.concat(rows, ignore_index=True)
    picks = pd.concat(pick_frames, ignore_index=True)
    scored_all = pd.concat(scored_frames, ignore_index=True)
    _save_calibration(settings, ruleset, scored_all, raw)
    _save_top_k(settings, ruleset, context, scored_all, list(dict.fromkeys(candidates)))
    lift_models = result[result["model"].isin(wf["simplicity"])]  # the pre-registered six
    choice = choose_default(
        lift_models, "v0", int(wf["min_wins"]), float(wf["tie"]), wf["simplicity"]
    )
    returns = raw["returns"]
    by_return = _return_choice(picks, returns, returns["simplicity"])
    rounds = {
        name: _return_choice(picks, returns, returns[name]["simplicity"])
        for name in ("round2", "round3")
    }
    result.to_parquet(
        ensure_parent(mart_table_path(settings.marts_dir, f"walk_forward_{ruleset}")), index=False
    )
    picks.to_parquet(
        mart_table_path(settings.marts_dir, f"walk_forward_picks_{ruleset}"), index=False
    )
    summarize_picks(picks).to_parquet(
        mart_table_path(settings.marts_dir, f"walk_forward_picks_summary_{ruleset}"), index=False
    )
    by_return.table.to_parquet(
        mart_table_path(settings.marts_dir, f"walk_forward_return_choice_{ruleset}"), index=False
    )
    for name, decided in rounds.items():
        decided.table.to_parquet(
            mart_table_path(settings.marts_dir, f"walk_forward_return_{name}_{ruleset}"),
            index=False,
        )
    choice.table.to_parquet(
        mart_table_path(settings.marts_dir, f"walk_forward_choice_{ruleset}"), index=False
    )
    context.save_live_return_model(usable, config, return_model_path(settings.marts_dir))
    return result, choice, by_return, rounds


def trading_costs() -> TradingCosts:
    """Cost tiers (config/backtest.yaml) and the capacity rule (config/allocation.yaml)."""
    tiers = tuple(
        (
            None if t["max_trading_value"] is None else float(t["max_trading_value"]),
            float(t["cost"]),
        )
        for t in load_yaml("backtest")["cost_tiers"]
    )
    allocation = load_yaml("allocation")
    return TradingCosts(tiers, float(allocation["participation"]), int(allocation["max_days"]))


def _save_top_k(
    settings: Settings,
    ruleset: str,
    context: _WalkForwardContext,
    scored: pd.DataFrame,
    rankings: list[str],
) -> None:
    """marts/walk_forward_top_k_<ruleset>: every ranking × size, with the style benchmark,
    dividends, costs and capacity (model/portfolio.top_k_breakdown).
    """
    frame = scored.assign(realized_return=context.realized(scored))
    frame["style_return"] = style_benchmark(frame, int(load_yaml("backtest")["style_buckets"]))
    sizes = [int(n) for n in load_yaml("model")["returns"]["small_portfolios"]]
    top_k_breakdown(frame, rankings, sizes, trading_costs()).to_parquet(
        mart_table_path(settings.marts_dir, f"walk_forward_top_k_{ruleset}"), index=False
    )


def _return_choice(
    picks: pd.DataFrame, returns: Mapping[str, Any], simplicity: list[str]
) -> Choice:
    """The pre-registered return rule: beat the value basket's top-N excess return on enough dates
    and by enough on the mean, else the simpler candidate stays.
    """
    top = picks[(picks["top_n"] == int(returns["top_n"])) & picks["model"].isin(simplicity)]
    return choose_default(
        top.assign(excess=top["mean_return"] - top["universe_mean_return"]),
        "value_basket",
        int(returns["min_wins"]),
        float(returns["tie"]),
        simplicity,
        metric="excess",
    )


def _p_any(calibrated: pd.DataFrame) -> pd.Series:
    """P(any offer) from per-type columns (see `combine`)."""
    return combine({str(k): calibrated[k] for k in calibrated.columns})


def _save_calibration(
    settings: Settings, ruleset: str, scored: pd.DataFrame, raw: Mapping[str, Any]
) -> None:
    """Prior correction vs isotonic on the walk-forward test predictions: Brier score and predicted
    vs observed rate per decile → marts/walk_forward_calibration_<ruleset>.
    """
    methods = [m for m in ("p_prior", "p_isotonic") if m in scored.columns]
    known = scored.dropna(subset=methods)  # same rows for both, or the Brier is not comparable
    bins = int(raw["reliability_bins"])
    frames = []
    for method in methods if not known.empty else []:
        table = reliability_table(known[method], known["label"], bins)
        frames.append(table.assign(method=method, brier=brier(known[method], known["label"])))
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(
            mart_table_path(settings.marts_dir, f"walk_forward_calibration_{ruleset}"), index=False
        )


@dataclass
class _WalkForwardContext:
    """What every walk-forward split shares: the rule set, prices, past premiums and the 12-month
    return of every (company, date), computed once.
    """

    rules: RuleSet
    raw: dict[str, Any]
    adjusted: AdjustedPrices
    topix: pd.DataFrame
    premiums: dict[str, float]
    returns: dict[tuple[str, str], float]

    @classmethod
    def load(
        cls, settings: Settings, usable: pd.DataFrame, ruleset: str, raw: dict[str, Any]
    ) -> _WalkForwardContext:
        adjusted = _adjusted_prices(load_staging(settings, PRICES_TABLE))
        backtest = load_yaml("backtest")
        horizon, stale = int(backtest["horizon_days"]), int(backtest["pick_stale_days"])
        returns: dict[tuple[str, str], float] = {}
        for code, as_of in zip(usable["sec_code"], usable["as_of"], strict=True):
            day = pd.Timestamp(str(as_of)).date()
            held = holding_return(adjusted, str(code), day, day + timedelta(days=horizon), stale)
            if held is not None:
                returns[(str(code), str(day))] = held[0]
        study = pd.read_parquet(mart_table_path(settings.marts_dir, "event_study"))
        medians = premium_summary(study).set_index("event_type")["median_premium"]
        return cls(
            rules=RuleSet.load(ruleset),
            raw=raw,
            adjusted=adjusted,
            topix=load_staging(settings, TOPIX_TABLE),
            premiums={str(k): float(v) for k, v in medians.items()},
            returns=returns,
        )

    def score(
        self, train: pd.DataFrame, test: pd.DataFrame, config: LogisticConfig
    ) -> pd.DataFrame:
        """Add one column per candidate to `test`, every model fitted on `train` only."""
        kinds = [k for k in self.raw["event_types"] if train[f"label_{k}"].nunique() == 2]
        c = float(self.raw["walk_forward"]["rule_weights_C"])
        test["v0"] = test["score"]
        test["value_basket"] = value_basket_score(test, _basket_config(self.raw, quality=False))
        test["value_basket_quality"] = value_basket_score(
            test, _basket_config(self.raw, quality=True)
        )
        test["return_model"] = self._return_model(train, test, config)
        test["return_model_sector_neutral"] = sector_neutral(test["return_model"], test["sector33"])
        # Sensitivity: v2 without the activist rule, whose curated list looks ahead.
        no_activist = RuleSet(
            version=self.rules.version,
            rules=[r for r in self.rules.rules if r.id != "activist_holder"],
        )
        test["v2_no_activist"] = fit_rule_model(train, no_activist, "label", c).predict_proba(
            rule_flags(test, no_activist).to_numpy()
        )[:, 1]
        flags = rule_flags(test, self.rules).to_numpy()
        test["v2"] = fit_rule_model(train, self.rules, "label", c).predict_proba(flags)[:, 1]
        by_type = {
            k: pd.Series(
                fit_rule_model(train, self.rules, f"label_{k}", c).predict_proba(flags)[:, 1],
                index=test.index,
            )
            for k in kinds
        }
        if by_type:
            test["v2_by_type"] = combine(by_type)
        models = fit_by_type(train, config, kinds)
        if not models:
            return test
        per_type = {k: predict(m, test, config) for k, m in models.items()}
        test["logistic_by_type"] = combine(per_type)
        partner = per_type.get("parent", test["logistic_by_type"])
        test["hybrid"] = test["v0"].rank(pct=True) + partner.rank(pct=True)
        rates = {k: float(train[f"label_{k}"].mean()) for k in kinds}
        calibrated = _calibrated(test, models, config, rates)
        test["p_prior"] = _p_any(calibrated)
        self._round3(test, calibrated)
        test[COMBO] = union_rank_score(test, {"v2": "logistic_by_type", "p_mbo": None})
        test["p_isotonic"] = self._isotonic_p_any(train, test, config, kinds)
        no_offer = self._no_offer_return(train)
        if no_offer is not None and all(k in self.premiums for k in kinds):
            test["expected_value"] = expected_value(calibrated, self.premiums, no_offer)
        return test

    def _round3(self, test: pd.DataFrame, calibrated: pd.DataFrame) -> None:
        """Round-3 candidates: value + catalyst, value ex worst momentum."""
        basket = _basket_config(self.raw, quality=False)
        for kind in ("parent", "mbo"):
            test[f"p_{kind}"] = calibrated[kind] if kind in calibrated.columns else np.nan
        test["value_catalyst"] = value_catalyst_score(test, basket)
        if "momentum_12_1" in test.columns:
            worst = float(self.raw["returns"]["round3"]["momentum_worst_share"])
            test["value_momentum"] = value_momentum_score(test, basket, worst)

    def _isotonic_p_any(
        self, train: pd.DataFrame, test: pd.DataFrame, config: LogisticConfig, kinds: list[str]
    ) -> pd.Series:
        """P(any offer) calibrated by isotonic regression on the LAST training year: models fitted
        on the training dates ≥ 1 year before it, calibrated on it, applied to the test date.
        """
        fitted = fit_with_isotonic(train, config, kinds)
        if fitted is None:
            return pd.Series(np.nan, index=test.index)
        models, rates, calibrator = fitted
        return apply_isotonic(calibrator, _p_any(_calibrated(test, models, config, rates)))

    def realized(self, frame: pd.DataFrame) -> pd.Series:
        """Realized 12-month holding return of each (company, date) row; NaN when unpriced."""
        keys = zip(
            frame["sec_code"].astype(str),
            frame["as_of"].map(lambda d: str(pd.Timestamp(str(d)).date())),
            strict=True,
        )
        return pd.Series([self.returns.get(k, np.nan) for k in keys], index=frame.index)

    def _excess(self, frame: pd.DataFrame) -> pd.Series:
        """Realized 12-month return minus the same date's all-companies mean (NaN if unknown)."""
        realized = self.realized(frame)
        return realized - realized.groupby(frame["as_of"]).transform("mean")

    def _return_model(
        self, train: pd.DataFrame, test: pd.DataFrame, config: LogisticConfig
    ) -> pd.Series:
        settings = self.raw["returns"]
        excess = self._excess(train)
        known = excess.notna()
        x_train, x_test = design_matrix(train[known], config), design_matrix(test, config)
        model = fit_return_model(
            x_train, excess[known].reset_index(drop=True), list(x_train.columns),
            alpha=float(settings["ridge_alpha"]), clip=float(settings["target_clip"]),
        )  # fmt: skip
        return pd.Series(predict_return(model, x_test, list(x_test.columns)), index=test.index)

    def save_live_return_model(
        self, usable: pd.DataFrame, config: LogisticConfig, path: Path
    ) -> None:
        """The return model refitted on EVERY complete date, frozen for the daily screen."""
        settings = self.raw["returns"]
        excess = self._excess(usable)
        known = excess.notna()
        x = design_matrix(usable[known], config)
        model = fit_return_model(
            x, excess[known].reset_index(drop=True), list(x.columns),
            alpha=float(settings["ridge_alpha"]), clip=float(settings["target_clip"]),
        )  # fmt: skip
        bundle = {"model": model, "columns": list(x.columns), "config": config}
        joblib.dump(bundle, ensure_parent(path))

    def _no_offer_return(self, train: pd.DataFrame) -> float | None:
        quiet = train[train["label"] == 0]
        values = [
            self.returns[key]
            for key in zip(
                quiet["sec_code"].astype(str),
                quiet["as_of"].map(lambda d: str(pd.Timestamp(str(d)).date())),
                strict=True,
            )
            if key in self.returns
        ]
        return float(pd.Series(values).median()) if values else None

    def picks(self, ranked: pd.DataFrame) -> pd.DataFrame:
        backtest = load_yaml("backtest")
        return pick_returns(
            ranked,
            self.adjusted,
            self.topix,
            horizon_days=int(backtest["horizon_days"]),
            sizes=[int(n) for n in backtest["pick_sizes"]],
            stale_days=int(backtest["pick_stale_days"]),
        )


def learn_weights(
    settings: Settings, base: str, target: str, config_dir: Path = CONFIG_DIR
) -> RuleSet:
    """Learn the weights of `base`'s rules on every complete backtest date → config/<target>.yaml
    (same rules and thresholds), and re-score the backtest panel with them (marked in-sample).
    """
    base_rules = RuleSet.load(base, config_dir)
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{base}"))
    metrics = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{base}"))
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    usable = _evaluable(panel[panel["as_of"].isin(complete)])
    c = float(load_yaml("model")["walk_forward"]["rule_weights_C"])
    model = fit_rule_model(usable, base_rules, "label", c)
    weights = [round(float(w), 2) for w in model.coef_[0]]
    learned = with_weights(
        base_rules, dict(zip([r.id for r in base_rules.rules], weights, strict=True))
    )
    suffix = target.rsplit("_v", 1)[-1]  # scoring_v2 → version 2
    learned = learned.model_copy(
        update={"version": int(suffix) if suffix.isdigit() else base_rules.version}
    )
    _write_rules(learned, base_rules, target, config_dir, sorted(str(d) for d in complete))
    _rescore_backtest(settings, panel, metrics, learned, target)
    return learned


def weight_stability(settings: Settings, base: str) -> pd.DataFrame:
    """Bootstrap of the learned rule weights over complete dates (model/rules.py) →
    marts/rule_weights_bootstrap_<base>.
    """
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{base}"))
    metrics = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{base}"))
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    wf = load_yaml("model")["walk_forward"]
    table = bootstrap_rule_weights(
        _evaluable(panel[panel["as_of"].isin(complete)]), RuleSet.load(base), "label",
        c=float(wf["rule_weights_C"]), draws=int(wf["bootstrap_draws"]),
        seed=int(wf["bootstrap_seed"]),
    )  # fmt: skip
    table.to_parquet(mart_table_path(settings.marts_dir, f"rule_weights_bootstrap_{base}"))
    return table


def _write_rules(
    learned: RuleSet, base: RuleSet, target: str, config_dir: Path, dates: list[str]
) -> None:
    header = (
        f"# Rule set {target}: the rules of v{base.version} with weights LEARNED from data\n"
        f"# (`radar learn-weights`, L2 logistic on the 0/1 rule flags, balanced classes) on the\n"
        f"# {len(dates)} complete backtest dates {dates[0]} → {dates[-1]}. Weights are log-odds:\n"
        "# positive raises the odds of a take-private offer, negative lowers them. Chosen as\n"
        "# the default by the pre-registered walk-forward rule (docs/results.md). Do not edit\n"
        "# by hand: regenerate, and judge it on the live scoreboard.\n"
    )
    rules = []
    for rule, old in zip(learned.rules, base.rules, strict=True):
        entry = rule.model_dump()
        entry["rationale"] = (
            f"{old.rationale} [Weight learned from data: {rule.weight:+.2f} "
            f"(hand-set v{base.version}: {old.weight:g}).]"
        )
        rules.append(entry)
    body = yaml.safe_dump(
        {"version": learned.version, "rules": rules}, allow_unicode=True, sort_keys=False, width=100
    )
    (config_dir / f"{target}.yaml").write_text(header + body, encoding="utf-8")


def _rescore_backtest(
    settings: Settings, panel: pd.DataFrame, metrics: pd.DataFrame, rules: RuleSet, target: str
) -> None:
    """The base panel re-scored with `rules` (features and labels unchanged) → backtest marts of
    `target`, flagged in_sample.
    """
    features = panel.drop(columns=["score", "triggered_rules"], errors="ignore")
    rescored = pd.concat([score(g, rules) for _, g in features.groupby("as_of")], ignore_index=True)
    rescored.to_parquet(
        ensure_parent(mart_table_path(settings.marts_dir, f"backtest_panel_{target}"))
    )
    out = metrics[["as_of", "label_complete"]].assign(in_sample=True)
    out.to_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{target}"), index=False)


def discover_activists(settings: Settings, edinet: EdinetClient | None) -> pd.DataFrame:
    """Candidate activists from 保有目的 → marts/activist_candidates."""
    config = load_yaml("activists")
    discovery = DiscoveryConfig.from_config(config["discovery"])
    reports = initial_reports_per_filer(load_documents(settings))
    if edinet is not None:
        download_documents(edinet, reports)
    rows: list[dict[str, Any]] = []
    for report in reports.to_dict("records"):
        path = edinet_doc_path(settings.raw_dir, str(report["doc_id"]), FetchType.CSV)
        if not path.exists():
            continue
        try:
            facts = read_facts_zip(path, prefix=LARGE_HOLDING_PREFIX)
        except (zipfile.BadZipFile, ValueError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s: %s", report["doc_id"], type(exc).__name__)
            continue
        for purpose in holding_purposes(facts, discovery.purpose_element):
            rows.append(
                {
                    "filer_edinet_code": report["edinet_code"],
                    "filer_name": report["filer_name"],
                    "doc_id": report["doc_id"],
                    "submit_datetime": report["submit_datetime"],
                    "purpose": purpose,
                }
            )
    purposes = pd.DataFrame(
        rows, columns=["filer_edinet_code", "filer_name", "doc_id", "submit_datetime", "purpose"]
    )
    candidates = activist_candidates(purposes, all_patterns(config["activist_filers"]), discovery)
    path = ensure_parent(mart_table_path(settings.marts_dir, ACTIVIST_CANDIDATES_TABLE))
    candidates.to_parquet(path, index=False)
    logger.info("%d reports read, %d candidates", purposes["doc_id"].nunique(), len(candidates))
    return candidates


def _adjusted_prices(prices: pd.DataFrame) -> AdjustedPrices:
    """Split-restated prices with impossible one-day quotes flagged (config/backtest.yaml)."""
    ratio = float(load_yaml("backtest")["max_daily_price_ratio"])
    return AdjustedPrices(prices, max_daily_ratio=ratio)


def run_picks(settings: Settings, ruleset: str = "scoring_v0") -> pd.DataFrame:
    """Top-N pick returns on the dates of the last backtest with a complete window →
    marts/pick_returns_<ruleset>.
    """
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_panel_{ruleset}"))
    metrics = pd.read_parquet(mart_table_path(settings.marts_dir, f"backtest_metrics_{ruleset}"))
    complete = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
    config = load_yaml("backtest")
    result = pick_returns(
        panel[panel["as_of"].isin(complete)],
        _adjusted_prices(load_staging(settings, PRICES_TABLE)),
        load_staging(settings, TOPIX_TABLE),
        horizon_days=int(config["horizon_days"]),
        sizes=[int(n) for n in config["pick_sizes"]],
        stale_days=int(config["pick_stale_days"]),
    )
    path = ensure_parent(mart_table_path(settings.marts_dir, f"pick_returns_{ruleset}"))
    result.to_parquet(path, index=False)
    return result


def snapshot_live_scores(settings: Settings, as_of: date) -> bool:
    """Freeze scores_<as_of> into data/marts/live — only if no snapshot exists for that day."""
    target = live_scores_path(settings.marts_dir, as_of)
    if target.exists():
        return False
    scores = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{as_of.isoformat()}"))
    scores.to_parquet(ensure_parent(target), index=False, compression=DAILY_COMPRESSION)
    return True


def run_scoreboard(settings: Settings, today: date) -> pd.DataFrame:
    """Live track record of every frozen ranking → marts/live_scoreboard."""
    folder = live_scores_path(settings.marts_dir, today).parent
    snapshots = {
        date.fromisoformat(p.stem.removeprefix("scores_")): pd.read_parquet(p)
        for p in sorted(folder.glob("scores_*.parquet"))
    }
    documents = load_documents(settings)
    offers = _optional_staging(settings, TENDER_OFFERS_TABLE)
    events = take_private_events(tender_offer_events(documents), offers)
    config = load_yaml("backtest")
    board = live_scoreboard(
        snapshots,
        events,
        today=today,
        horizon_days=int(config["horizon_days"]),
        top_fraction=float(config["top_fraction"]),
        top_n=int(config["track_record_top_n"]),
        rankings=["score", *LIVE_MODEL_COLUMNS],
    )
    if snapshots and "sec_code" in next(iter(snapshots.values())).columns:
        returns = live_returns(
            snapshots,
            ["score", *LIVE_MODEL_COLUMNS],
            _adjusted_prices(load_staging(settings, PRICES_TABLE)),
            today=today,
            top_n=int(load_yaml("model")["returns"]["top_n"]),
            stale_days=int(config["pick_stale_days"]),
        )
        if not board.empty and not returns.empty:
            board = board.merge(returns.drop(columns="top_n"), on=["as_of", "ranking"], how="left")
    board.to_parquet(ensure_parent(mart_table_path(settings.marts_dir, "live_scoreboard")))
    return board


def build_factpacks(
    settings: Settings, as_of: date, codes: Sequence[str] = (), top: int = 15
) -> list[Path]:
    """Fact packs for the top `top` companies of scores_<as_of> plus `codes` (5-character securities
    codes) → data/marts/factpack_<as_of>_<code>.md.
    """
    ruleset = RuleSet.default()
    scores = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{as_of.isoformat()}"))
    candidates = scores
    if "offer_filed_365d" in scores.columns:  # already in play: not a prediction
        candidates = scores[scores["offer_filed_365d"].isna()]
    top_index = candidates.head(top).index
    chosen = scores[scores.index.isin(top_index) | scores["sec_code"].isin(codes)]
    documents = load_documents(settings)
    context = _FactPackContext.load(settings, ruleset, as_of)
    paths = []
    for row in chosen.to_dict("records"):
        inputs = context.inputs({str(k): v for k, v in row.items()}, documents)
        path = ensure_parent(factpack_path(settings.marts_dir, as_of, str(row["sec_code"])))
        path.write_text(factpack_markdown(inputs), encoding="utf-8")
        paths.append(path)
    return paths


@dataclass
class _FactPackContext:
    """Tables shared by every fact pack of one date, loaded once."""

    settings: Settings
    ruleset: RuleSet
    as_of: date
    study: pd.DataFrame | None
    hit_rates: pd.DataFrame
    prices: pd.DataFrame | None

    @classmethod
    def load(cls, settings: Settings, ruleset: RuleSet, as_of: date) -> _FactPackContext:
        study_path = mart_table_path(settings.marts_dir, "event_study")
        study = pd.read_parquet(study_path) if study_path.exists() else None
        name = f"scoring_v{ruleset.version}"
        panel_path = mart_table_path(settings.marts_dir, f"backtest_panel_{name}")
        hit_rates = pd.DataFrame(columns=["points", "companies", "hits", "hit_rate"])
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            metrics = pd.read_parquet(
                mart_table_path(settings.marts_dir, f"backtest_metrics_{name}")
            )
            done = set(metrics.loc[metrics["label_complete"].astype(bool), "as_of"])
            band = ruleset.max_score / float(load_yaml("backtest")["points_bands"])
            hit_rates = hit_rate_by_band(panel[panel["as_of"].isin(done)], ruleset.max_score, band)
        prices = _optional_staging(settings, PRICES_TABLE)
        return cls(settings, ruleset, as_of, study, hit_rates, prices)

    def inputs(self, row: dict[str, Any], documents: pd.DataFrame) -> FactPackInputs:
        doc_id = str(row["source_doc_id"])
        buyer = likely_buyer(row, self.ruleset)
        price, table = self._price(row), None
        if self.study is not None:
            config = OfferRangeConfig.from_config(load_yaml("event_study"))
            found = offer_range_for(row, self.study, price, self.ruleset, config)
            table = offer_range_table(found, price) if found else None
        return FactPackInputs(
            as_of=self.as_of,
            row=row,
            ruleset=self.ruleset,
            likely_buyer=buyer,
            last_close=price,
            offer_table=table,
            hit_rates=self.hit_rates,
            holders=holders_table(
                load_staging(self.settings, SHAREHOLDERS_TABLE),
                doc_id,
                HolderPatterns.from_config(load_yaml("ownership")),
            ),
            officers=officers_table(
                load_staging(self.settings, OFFICERS_TABLE), doc_id, self.as_of
            ),
            excerpts=self._excerpts(doc_id),
            recent_filings=self._recent(documents, str(row["edinet_code"])),
        )

    def _price(self, row: Mapping[str, Any]) -> float | None:
        if self.prices is None:
            return None
        latest = price_as_of(self.prices[self.prices["code"] == row["sec_code"]], self.as_of)
        return None if latest.empty else float(latest["close"].iloc[0])

    def _excerpts(self, doc_id: str) -> dict[str, str]:
        path = edinet_doc_path(self.settings.raw_dir, doc_id, FetchType.CSV)
        if not path.exists():
            return {}
        limit = int(load_yaml("report")["excerpt_max_chars"])
        return report_excerpts(read_facts_zip(path), ElementMap.load().excerpts, limit)

    def _recent(self, documents: pd.DataFrame, edinet_code: str) -> pd.DataFrame:
        about = documents[
            documents["doc_type_code"].isin(["240", "290", "350"])
            & (
                (documents["issuer_edinet_code"] == edinet_code)
                | (documents["subject_edinet_code"] == edinet_code)
            )
            & (pd.to_datetime(documents["submit_datetime"]) < pd.Timestamp(self.as_of))
        ].sort_values("submit_datetime")
        count = int(load_yaml("report")["recent_filings"])
        columns = ["submit_datetime", "filer_name", "doc_description"]
        return about[columns].tail(count).reset_index(drop=True)


def sync_topix(client: JQuantsClient, settings: Settings, start: date, end: date) -> int:
    """TOPIX daily closes for [start, end] → staging/topix.parquet (rebuilt), one call per year."""
    frames = []
    for year in range(start.year, end.year + 1):
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        rows = client.topix(first, last, refresh=year == date.today().year)
        frames.append(topix_frame(rows))
    topix = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    topix.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, TOPIX_TABLE)))
    return len(topix)


def parse_tender_offers(settings: Settings) -> pd.DataFrame:
    """Every downloaded third-party tender offer (240, form `jptoo`) → staging/tender_offers."""
    config = TenderOfferConfig.from_config(load_yaml("tender_offers"))
    rows: list[dict[str, Any]] = []
    for event in tender_offer_events(load_documents(settings)).to_dict("records"):
        path = edinet_doc_path(settings.raw_dir, str(event["doc_id"]), FetchType.CSV)
        if not path.exists():
            continue
        try:
            terms = extract_tender_offer(read_facts_zip(path, prefix="jptoo"), config)
        except (zipfile.BadZipFile, ValueError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s: %s", event["doc_id"], type(exc).__name__)
            continue
        row: dict[str, Any] = {str(k): v for k, v in event.items()}
        rows.append({**row, **terms, "event_type": classify_event(terms, config)})
    offers = pd.DataFrame(rows)
    path = ensure_parent(staging_table_path(settings.staging_dir, TENDER_OFFERS_TABLE))
    offers.to_parquet(path, index=False)
    return offers


def run_event_study(settings: Settings) -> pd.DataFrame:
    """Premium and run-up per offer → marts/event_study.parquet (see model/event_study.py)."""
    documents = load_documents(settings)
    listed = documents[documents["sec_code"].notna()].sort_values("submit_datetime")
    sec_codes: dict[str, str] = {
        str(e): str(s) for e, s in zip(listed["edinet_code"], listed["sec_code"], strict=True)
    }
    result = event_study(
        load_staging(settings, TENDER_OFFERS_TABLE),
        load_staging(settings, PRICES_TABLE),
        load_staging(settings, TOPIX_TABLE),
        sec_codes,
        EventStudyConfig.from_config(load_yaml("event_study")),
    )
    books = _offer_book_values(result, documents, settings, sec_codes)
    result["book_per_share"] = books
    result["offer_pbr"] = [
        offer_pbr(price, book) for price, book in zip(result["offer_price"], books, strict=True)
    ]
    # The target's own PBR just before the offer: the book anchor of the likely offer price
    # compares a candidate with targets priced like it (model/offer_price.py).
    result["pre_pbr"] = [
        offer_pbr(close, book) for close, book in zip(result["pre_close"], books, strict=True)
    ]
    result.to_parquet(ensure_parent(mart_table_path(settings.marts_dir, "event_study")))
    return result


def _offer_book_values(
    offers: pd.DataFrame, documents: pd.DataFrame, settings: Settings, sec_codes: dict[str, str]
) -> list[float | None]:
    """Target's book value per share at each offer, on the offer's share basis."""
    financials = load_staging(settings, FINANCIALS_TABLE)
    by_doc = {str(r["doc_id"]): r for r in financials.to_dict("records")}
    parsed = documents[documents["doc_id"].isin(by_doc)]
    prices = _optional_staging(settings, PRICES_TABLE)
    splits = _split_rows(prices)
    books: list[float | None] = []
    for offer in offers.to_dict("records"):
        target = str(offer["subject_edinet_code"])
        filings = latest_annual_filings(
            parsed[parsed["edinet_code"] == target], offer["event_date"]
        )
        if filings.empty:
            books.append(None)
            continue
        filing = filings.iloc[0]
        book = book_value_per_share(by_doc[str(filing["doc_id"])])
        since = pd.Timestamp(filing["submit_datetime"]).date()
        multiplier = split_multiplier(
            splits.get(sec_codes.get(target, "")), since, offer["event_date"]
        )
        books.append(None if book is None else book / multiplier)
    return books


def _split_rows(prices: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Rows with a split / consolidation (adj_factor ≠ 1) per code."""
    if prices is None or "adj_factor" not in prices.columns:
        return {}
    events = prices[prices["adj_factor"].notna() & (prices["adj_factor"] != 1.0)].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return {str(code): frame for code, frame in events.groupby("code")}


def review_runups(settings: Settings) -> pd.DataFrame:
    """Every priced take-private offer not already at/above its price, with leakage indicators →
    marts/runup_review.
    """
    raw = load_yaml("event_study")
    study = pd.read_parquet(mart_table_path(settings.marts_dir, "event_study"))
    usable = study[study["pre_announced"].eq(False) & study["is_take_private"].eq(True)].copy()
    documents = load_documents(settings)
    names = documents.drop_duplicates("edinet_code", keep="last").set_index("edinet_code")
    listed = documents[documents["sec_code"].notna()].drop_duplicates("edinet_code", keep="last")
    sec_codes = dict(zip(listed["edinet_code"], listed["sec_code"], strict=True))
    prices = load_staging(settings, PRICES_TABLE)
    by_code = {
        str(c): g for c, g in prices[prices["code"].isin(set(sec_codes.values()))].groupby("code")
    }
    patterns = all_patterns(load_yaml("activists")["activist_filers"])
    offers = load_staging(settings, TENDER_OFFERS_TABLE).set_index("doc_id")

    def extra(row: pd.Series) -> pd.Series:
        code = sec_codes.get(row["subject_edinet_code"])
        bars = by_code.get(str(code)) if code else None
        ratio = (
            None
            if bars is None
            else volume_ratio(
                bars,
                row["event_date"],
                raw["volume_recent_sessions"],
                raw["volume_baseline_sessions"],
            )
        )
        found = activist_features(documents, row["event_date"], patterns)
        hit = found[found["edinet_code"] == row["subject_edinet_code"]]
        return pd.Series(
            {
                "company": names["filer_name"].get(row["subject_edinet_code"]),
                "acquirer": offers["acquirer_name"].get(row["doc_id"]),
                "volume_ratio": ratio,
                "activist_before": None if hit.empty else str(hit["activist_names"].iloc[0]),
            }
        )

    if usable.empty:
        logger.warning("no priced take-private offer to review")
        return usable
    review = pd.concat([usable, usable.apply(extra, axis=1)], axis=1)
    competing = earlier_offer_within(study, int(raw["competing_bid_days"]))
    review["earlier_offer"] = competing.loc[review.index]
    review["flagged"] = review["runup_abnormal"] > float(raw["runup_review_threshold"])
    review = review.sort_values("runup_abnormal", ascending=False)
    review.to_parquet(ensure_parent(mart_table_path(settings.marts_dir, "runup_review")))
    return review


def run_audit(settings: Settings, as_of: date, ruleset: str | None = None) -> pd.DataFrame:
    """Re-check marts/features_<as_of> and scores_<as_of> (see `audit_features`, `audit_scores`).

    Returns every issue found (source, edinet_code, check, detail); empty = all consistent.
    """
    day = as_of.isoformat()
    features = pd.read_parquet(mart_table_path(settings.marts_dir, f"features_{day}"))
    scores = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{day}"))
    segments = tuple(load_yaml("jpx")["requested_segments"])
    return pd.concat(
        [
            audit_features(features, segments).assign(source="features"),
            audit_scores(scores, RuleSet.load(ruleset or default_ruleset_name())).assign(
                source="scores"
            ),
        ],
        ignore_index=True,
    )


def score_dates(settings: Settings) -> list[date]:
    """Screening dates that have a scores file, oldest first."""
    stems = mart_table_path(settings.marts_dir, "scores_x").parent.glob("scores_*.parquet")
    return sorted(date.fromisoformat(p.stem.removeprefix("scores_")) for p in stems)


def run_changes(
    settings: Settings, since: date, until: date, ruleset: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranking changes between two screening dates and new filings on followed companies (top N of
    `until` + config/watchlist.yaml) → marts/alerts_<until>.md.
    """
    watch = load_yaml("watchlist")
    old = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{since.isoformat()}"))
    new = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{until.isoformat()}"))
    rules = RuleSet.load(ruleset or default_ruleset_name())
    top_n = int(watch["alert_top_n"])
    changes = compare_scores(old, new, top_n, rules.max_score)
    documents = load_documents(settings)
    followed = followed_companies(new, documents)
    window = (datetime.combine(since, time.min), datetime.combine(until, time.min))
    events = new_events(
        documents,
        followed,
        *window,
        all_patterns(load_yaml("activists")["activist_filers"]),
        extraordinary=classify_extraordinary_reports(
            settings, _extraordinary_docs(documents, followed, *window)["doc_id"]
        ),
    )
    names = dict(zip(new["edinet_code"], new["company_name"], strict=True))
    factpacks = {
        str(code): path
        for code in changes["sec_code"].astype(str)
        if (path := factpack_path(settings.marts_dir, until, code)).exists()
    }
    report = load_yaml("report")
    drift = coverage_drift(
        old, new, report["coverage_drift"]["columns"], float(report["coverage_drift"]["max_drop"])
    )
    for line in drift:
        logger.warning("data coverage dropped — %s", line)
    unusual = _unusual_trading(settings, new, followed, until, report["unusual_trading"])
    note = alerts_markdown(
        changes, events, names, since, until, factpacks=factpacks, unusual=unusual, drift=drift
    )
    path = mart_table_path(settings.marts_dir, f"alerts_{until.isoformat()}").with_suffix(".md")
    ensure_parent(path).write_text(note, encoding="utf-8")
    return changes, events


def journal_table(settings: Settings, today: date) -> pd.DataFrame | None:
    """The decision journal (data/manual/journal.yaml) with what happened since each entry: return,
    a later take-private offer, today's rank.
    """
    path = journal_path(settings.manual_dir)
    if not path.exists():
        return None
    entries = parse_journal(yaml.safe_load(path.read_text(encoding="utf-8")))
    documents = load_documents(settings)
    events = take_private_events(
        tender_offer_events(documents), _optional_staging(settings, TENDER_OFFERS_TABLE)
    )
    codes = documents.dropna(subset=["sec_code"]).drop_duplicates("edinet_code")
    to_sec = dict(zip(codes["edinet_code"], codes["sec_code"].astype(str), strict=True))
    offers = pd.DataFrame(
        {"sec_code": events["subject_edinet_code"].map(to_sec), "event_date": events["event_date"]}
    )
    dates = score_dates(settings)
    ranks = pd.Series(dtype=float)
    if dates:
        latest = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{dates[-1]}"))
        ordered = latest.sort_values("score", ascending=False).reset_index(drop=True)
        ranks = pd.Series(ordered.index + 1, index=ordered["sec_code"].astype(str))
    prices = _optional_staging(settings, PRICES_TABLE)
    if prices is None:  # fresh install: returns unknown, the rest still shown
        prices = pd.DataFrame(columns=["code", "date", "close", "adj_factor"])
    return journal_outcomes(entries, prices, offers, ranks, today=today)


def _unusual_trading(
    settings: Settings,
    scores: pd.DataFrame,
    followed: set[str],
    until: date,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Unusual volume / price in the last week for followed companies (features/market.py)."""
    prices = _optional_staging(settings, PRICES_TABLE)
    topix = _optional_staging(settings, TOPIX_TABLE)
    if prices is None or topix is None or "trading_value" not in prices.columns:
        return pd.DataFrame(columns=["sec_code", "volume_ratio", "return_5d", "excess_5d"])
    codes = set(scores.loc[scores["edinet_code"].isin(followed), "sec_code"].astype(str))
    found = unusual_trading(
        prices, topix, codes, until,
        short=int(config["short_sessions"]), long=int(config["long_sessions"]),
        volume_ratio=float(config["volume_ratio"]), excess_move=float(config["excess_move"]),
    )  # fmt: skip
    names = dict(zip(scores["sec_code"].astype(str), scores["company_name"], strict=False))
    return found.assign(company_name=found["sec_code"].map(names))


def followed_companies(scores: pd.DataFrame, documents: pd.DataFrame) -> set[str]:
    """EDINET codes followed by the alerts: the top N of the default ranking + the watchlist."""
    watch = load_yaml("watchlist")
    top = scores[_in_top(scores, int(watch["alert_top_n"]))]
    return set(top["edinet_code"]) | watch_codes(watch.get("watchlist") or [], documents)


def _in_top(scores: pd.DataFrame, top: int) -> pd.Series:
    """In the top N with ties kept together (rules give discrete scores: 24 companies once shared
    #13).
    """
    return scores["score"].rank(ascending=False, method="min") <= top


def _extraordinary_docs(
    documents: pd.DataFrame, followed: set[str], since: datetime, until: datetime
) -> pd.DataFrame:
    """臨時報告書 (180) filed by a followed company in (since, until] (point-in-time)."""
    submitted = pd.to_datetime(documents["submit_datetime"])
    return documents[
        (documents["doc_type_code"] == "180")
        & documents["edinet_code"].isin(followed)
        & (documents["withdrawal_status"].astype(str) == "0")
        & (submitted > since)
        & (submitted <= until)
    ]


def classify_extraordinary_reports(
    settings: Settings, doc_ids: Iterable[str]
) -> dict[str, dict[str, object]]:
    """Reasons of each DOWNLOADED 臨時報告書 (parse/extraordinary_report.py); others absent."""
    config = ExtraordinaryConfig.from_config(load_yaml("extraordinary_reports"))
    found = {}
    for doc_id in doc_ids:
        path = edinet_doc_path(settings.raw_dir, str(doc_id), FetchType.CSV)
        if not path.exists():
            continue
        try:
            found[str(doc_id)] = classify_extraordinary(read_facts_zip(path), config)
        except (zipfile.BadZipFile, ValueError, UnicodeDecodeError) as exc:
            logger.warning("could not read %s: %s", doc_id, type(exc).__name__)
    return found


@dataclass(frozen=True)
class UpdateResult:
    index_from: date
    index_to: date
    downloaded: int
    parsed: int
    prices_days: int
    screening_date: date
    companies: int
    audit_issues: int
    previous_screening: date | None
    extraordinary_reports: int = 0  # 臨時報告書 of followed companies fetched
    new_factpacks: tuple[Path, ...] = ()  # fact packs of companies new in the top


def _last_doclist_day(settings: Settings) -> date | None:
    folder = edinet_doclist_path(settings.raw_dir, date.today()).parent
    days = [date.fromisoformat(p.stem) for p in folder.glob("*.json")]
    return max(days) if days else None


def update(
    settings: Settings, edinet: EdinetClient, jquants: JQuantsClient, today: date
) -> UpdateResult:
    """Bring everything up to date since the last run, then screen `today`."""
    last_day = _last_doclist_day(settings)
    if last_day is None:
        raise FileNotFoundError("no EDINET index yet: run the backfill first (README)")
    yesterday = today - timedelta(days=1)
    start = min(last_day, yesterday)  # the last fetched day may have been incomplete
    sync_document_index(edinet, settings, start, yesterday)
    documents = load_documents(settings)
    new_docs = pd.concat(
        [
            select_documents(documents, ANNUAL_REPORT_TYPES, start, yesterday),
            select_documents(documents, ["240", "290"], start, yesterday, only_listed=False),
        ]
    )
    downloaded = download_documents(edinet, new_docs)
    parsed = parse_downloaded_filings(settings, only_new=True)
    parse_tender_offers(settings)
    prices = _optional_staging(settings, PRICES_TABLE)
    price_days = 0
    if prices is not None:
        first = pd.to_datetime(prices["date"]).max().date() + timedelta(days=1)
        if first <= yesterday:
            price_days = sync_prices(jquants, settings, first, yesterday)
        sync_listings(
            jquants, settings, date(today.year, today.month, 1) - timedelta(days=1), today
        )
        # sync_topix REBUILDS the table, so start where prices start: past years come from the
        # raw cache, only the current year is re-fetched.
        sync_topix(jquants, settings, pd.to_datetime(prices["date"]).min().date(), yesterday)
    previous = [d for d in score_dates(settings) if d < today]
    features = build_features(settings, today)
    scores = score_companies(settings, today)
    snapshot_live_scores(settings, today)  # write-once out-of-sample record
    run_scoreboard(settings, today)  # the live record moves every day, not only on demand
    issues = run_audit(settings, today)
    # 臨時報告書 of followed companies only: ~14,000 a year market-wide, a handful for these.
    lookback = int(load_yaml("extraordinary_reports")["lookback_days"])
    window = (datetime.combine(today - timedelta(days=lookback), time.min),
              datetime.combine(today, time.min))  # fmt: skip
    extraordinary = _extraordinary_docs(documents, followed_companies(scores, documents), *window)
    fetched = download_documents(edinet, extraordinary)
    factpacks = _factpacks_for_entrants(settings, previous[-1] if previous else None, today)
    prune_marts(settings, today)
    return UpdateResult(
        start,
        yesterday,
        len(downloaded),
        parsed.parsed,
        price_days,
        today,
        len(features),
        len(issues),
        previous[-1] if previous else None,
        len(fetched),
        tuple(factpacks),
    )


def prune_marts(settings: Settings, today: date) -> list[Path]:
    """Delete old DERIVED files (config/report.yaml `retention`): features_<date> (the same columns
    are kept in scores_<date>) and old fact packs.
    """
    retention = load_yaml("report")["retention"]
    folder = mart_table_path(settings.marts_dir, "x").parent
    old = expired_dated_files(folder, "features_", today, int(retention["features_days"]))
    old += expired_dated_files(folder, "factpack_", today, int(retention["factpack_days"]))
    for path in old:
        path.unlink()
    if old:
        logger.info("pruned %d old derived files from marts", len(old))
    return old


def _factpacks_for_entrants(settings: Settings, previous: date | None, today: date) -> list[Path]:
    """Fact packs for companies that entered the top `factpack_new_top` since the previous
    screening: the memo for a new name is one file away the morning it appears.
    """
    if previous is None:
        return []
    top = int(load_yaml("report")["factpack_new_top"])
    old = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{previous.isoformat()}"))
    new = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{today.isoformat()}"))
    before = set(old.loc[_in_top(old, top), "sec_code"].astype(str))
    now = new.loc[_in_top(new, top)].sort_values("score", ascending=False)["sec_code"].astype(str)
    entrants = [code for code in now if code not in before]
    return build_factpacks(settings, today, entrants, top=0) if entrants else []


def run_allocation(
    settings: Settings,
    as_of: date,
    *,
    ranking: str,
    top: int,
    capital: float,
    ruleset: str = "scoring_v0",
    model: str | None = None,
    size_by_kelly: bool = True,
) -> tuple[pd.DataFrame, float | None, pd.Series]:
    """1/3-Kelly research allocation for the top `top` of `ranking` in marts/scores_<as_of>."""
    raw = load_yaml("allocation")
    config = AllocationConfig(**{k: raw[k] for k in AllocationConfig.__dataclass_fields__})
    history = pd.read_parquet(mart_table_path(settings.marts_dir, f"walk_forward_picks_{ruleset}"))
    name = model or ranking
    past = history[(history["model"] == name) & (history["top_n"] == top)]
    if past.empty:
        raise ValueError(f"no walk-forward history for {name} top {top}; see config pick_sizes")
    returns = past.set_index("as_of")["mean_return"]
    benchmark = (
        past.set_index("as_of")["universe_mean_return"] if raw.get("kelly_on") == "excess" else None
    )
    invested = kelly_fraction(returns, config, benchmark)
    scores = pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{as_of.isoformat()}"))
    if ranking not in scores.columns:
        raise ValueError(f"{ranking} not in scores_{as_of.isoformat()}; run `radar score`")
    chosen = scores[scores[ranking].notna()].sort_values(ranking, ascending=False).head(top)
    prices = AdjustedPrices(pd.read_parquet(staging_table_path(settings.staging_dir, "prices")))
    closes = [
        prices.close_on_or_before(str(c), as_of - timedelta(days=1)) for c in chosen["sec_code"]
    ]
    picks = chosen[["sec_code", "company_name", "median_trading_value_jpy"]].assign(
        close=[c[1] if c else np.nan for c in closes]
    )
    share = (invested or 0.0) if size_by_kelly else 1.0
    plan = allocate(picks, capital=capital, invested=share, config=config)
    return plan, invested, returns
