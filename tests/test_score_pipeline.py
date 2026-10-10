"""`radar features` / `radar score`: staging → marts, end to end on a synthetic filing."""

from __future__ import annotations

import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from radar import pipeline
from radar.config import Settings, load_yaml
from radar.model.expected_value import fit_isotonic
from radar.model.logistic import LogisticConfig, fit_by_type
from radar.model.rules import RuleSet
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
from test_parse_pipeline import GOOD_FILING, _index, _write_zip

AS_OF = date(2025, 7, 1)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    _index(s, [("S100AAA1", "E99001", "120")])
    _write_zip(s, "S100AAA1", GOOD_FILING)
    pipeline.parse_downloaded_filings(s)
    return s


def test_build_features_writes_mart_without_prices(settings: Settings) -> None:
    features = pipeline.build_features(settings, AS_OF)
    assert features["edinet_code"].tolist() == ["E99001"]
    assert features.loc[0, "oldest_representative_age"] == 75
    assert pd.isna(features.loc[0, "market_cap_jpy"])  # no prices yet → missing, not 0
    stored = pd.read_parquet(mart_table_path(settings.marts_dir, "features_2025-07-01"))
    assert len(stored) == 1


def test_build_features_uses_staging_prices(settings: Settings) -> None:
    prices = pd.DataFrame({"code": ["99990"], "date": [date(2025, 6, 30)], "close": [100.0]})
    prices.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "prices")))
    features = pipeline.build_features(settings, AS_OF)
    assert pd.isna(features.loc[0, "market_cap_jpy"])  # no share count in the fixture
    assert features.loc[0, "is_jgaap"]


def test_score_ranks_and_explains(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    scores = pipeline.score_companies(settings, AS_OF)  # default: config/scoring.yaml
    assert "aging_leader" in scores.loc[0, "triggered_rules"]  # 75 ≥ 70
    assert "score_v0" in scores.columns  # the hand-set score is kept for the live comparison
    assert "value_basket" in scores.columns
    assert mart_table_path(settings.marts_dir, "scores_2025-07-01").exists()
    v0 = pipeline.score_companies(settings, AS_OF, "scoring_v0")
    assert 0 < v0.loc[0, "score"] <= 1  # hand-set weights are all positive
    assert v0.loc[0, "score"] == pytest.approx(scores.loc[0, "score_v0"])


def test_score_requires_features_first(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="radar features"):
        pipeline.score_companies(settings, date(2030, 1, 1))


def _add_tender_offer(settings: Settings, submitted: datetime) -> None:
    """A third-party tender offer (240) for E99001, appended to the document index."""
    path = staging_table_path(settings.staging_dir, "documents")
    docs = pd.read_parquet(path)
    offer = docs.iloc[[0]].assign(
        doc_id="S100TOB1",
        edinet_code="E99100",
        sec_code=None,
        filer_name="サンプルSPC株式会社",
        doc_type_code="240",
        submit_datetime=submitted,
        subject_edinet_code="E99001",
    )
    pd.concat([docs, offer], ignore_index=True).to_parquet(path)


def test_backtest_labels_each_quarter_end(settings: Settings) -> None:
    _add_tender_offer(settings, datetime(2025, 9, 1, 15))
    metrics = pipeline.run_backtest(settings, date(2025, 6, 30), date(2025, 9, 30))

    assert metrics["as_of"].tolist() == [date(2025, 6, 30), date(2025, 9, 30)]
    assert metrics["positives"].tolist() == [1, 0]  # the offer is in the first window only
    assert metrics.loc[0, "precision_at_20"] == 1.0
    assert not metrics["label_complete"].any()
    panel = pd.read_parquet(mart_table_path(settings.marts_dir, "backtest_panel_scoring_v0"))
    assert set(panel.columns) >= {"as_of", "edinet_code", "score", "label"}


def test_backtest_without_quarter_ends_is_empty(settings: Settings) -> None:
    assert pipeline.run_backtest(settings, date(2025, 7, 1), date(2025, 9, 29)).empty


def test_capital_plans_only_count_disclosed(settings: Settings, tmp_path: Path) -> None:
    lists = pd.DataFrame(
        {
            "list_month": [date(2025, 4, 1), date(2025, 4, 1)],
            "published_on": [date(2025, 6, 1), date(2025, 6, 1)],
            "sec_code": ["99990", "99980"],
            "company_name": ["サンプル工業", "サンプル商事"],
            "market_segment": ["スタンダード", "グロース"],
            "status": ["検討中", "開示済"],
        }
    )
    lists.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "capital_plans")))
    features = pipeline.build_features(settings, AS_OF)
    assert not features.loc[0, "capital_plan_disclosed"]  # 99990 is only 検討中


def test_index_start_is_the_contiguous_run_of_fetched_days(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    assert pipeline.index_start(s) is None
    for day in [date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7)]:
        ensure_parent(edinet_doclist_path(s.raw_dir, day)).write_text("{}")
    assert pipeline.index_start(s) == date(2024, 1, 5)  # gap after 01-01: coverage restarts


def _synthetic_walk_forward(marts: Path) -> None:
    """Walk-forward marts for two rankings × two sizes × two dates (Portfolio tab)."""
    rows = [
        {"model": m, "top_n": n, "as_of": d, "mean_return": r, "universe_mean_return": 0.10,
         "topix_return": 0.08, "hit_rate": 0.1, "share_positive": 0.6, "worst_return": -0.3}
        for m, r in [("value_basket", 0.15), ("v2", 0.05)]
        for n in (10, 25)
        for d in (date(2024, 3, 31), date(2024, 6, 30))
    ]  # fmt: skip
    pd.DataFrame(rows).to_parquet(
        ensure_parent(mart_table_path(marts, "walk_forward_picks_scoring_v0"))
    )
    pd.DataFrame(
        {"model": ["v2"], "top_n": [10], "p_prior_mean": [0.1], "p_isotonic_mean": [0.05],
         "from_offers": [0.02], "from_others": [0.03]}
    ).to_parquet(mart_table_path(marts, "walk_forward_top_k_scoring_v0"))  # fmt: skip
    pd.DataFrame({"as_of": [date(2024, 3, 31)], "model": ["v2"], "companies": [4000]}).to_parquet(
        mart_table_path(marts, "walk_forward_scoring_v0")
    )


def test_portfolio_tab_renders_and_switches_ranking(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    _synthetic_walk_forward(settings.marts_dir)
    monkeypatch.setenv("RADAR_DATA_DIR", str(settings.data_dir))
    script = Path(__file__).parents[1] / "src" / "radar" / "report" / "app.py"

    app = testing.AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    ranking = app.selectbox(key="pf_model")
    assert ranking.value == "value_basket"  # the return lens' pick starts selected
    ranking.select("v2").run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Excl. best date" in labels and "Chance it's skill" in labels
    assert all(len(label) <= 22 for label in labels)  # long labels were cut off with "…"


def test_dashboard_renders_scores(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    monkeypatch.setenv("RADAR_DATA_DIR", str(settings.data_dir))

    script = Path(__file__).parents[1] / "src" / "radar" / "report" / "app.py"
    app = testing.AppTest.from_file(str(script), default_timeout=30).run()

    assert not app.exception
    reasons = [m.value for m in app.markdown if m.value.startswith("- **")]
    assert any("Aging leader" in r and "75 years" in r for r in reasons)


def test_ablation_uses_only_complete_label_windows(settings: Settings) -> None:
    _add_tender_offer(settings, datetime(2025, 9, 1, 15))
    pipeline.run_backtest(settings, date(2025, 6, 30), date(2025, 6, 30))
    assert pipeline.run_ablation(settings).empty

    _add_tender_offer(settings, datetime(2026, 7, 1, 9))  # index now covers the whole year
    pipeline.run_backtest(settings, date(2025, 6, 30), date(2025, 6, 30))
    result = pipeline.run_ablation(settings).set_index("without")
    assert "(none)" in result.index and "aging_leader" in result.index
    assert mart_table_path(settings.marts_dir, "ablation_scoring_v0").exists()


def test_rule_sets_v0_and_v1_compare_on_the_same_dates(settings: Settings) -> None:
    v0 = RuleSet.load("scoring_v0")
    v1 = RuleSet.load("scoring_v1")
    assert [r.id for r in v1.rules[: len(v0.rules)]] == [r.id for r in v0.rules]  # v0 unchanged
    assert "insider_stake" not in {r.id for r in v1.rules}
    assert {"price_slump", "low_turnover", "near_listing_criteria"} <= {r.id for r in v1.rules}
    metrics = pipeline.run_backtest(settings, date(2025, 6, 30), date(2025, 6, 30), "scoring_v1")
    assert len(metrics) == 1
    assert mart_table_path(settings.marts_dir, "backtest_panel_scoring_v1").exists()


def test_tender_offers_and_event_study_end_to_end(settings: Settings) -> None:
    _add_tender_offer(settings, datetime(2025, 9, 1, 15))  # bidder E99100 → target E99001
    e = load_yaml("tender_offers")["elements"]
    rows = [
        (e["price_text"], "株券１株につき金1,500円"),
        (e["purpose_text"], "本公開買付けは、いわゆるマネジメント・バイアウト（ＭＢＯ）…上場廃止"),
        (e["owned_by_bidder"], ""),
        (e["owned_by_related"], "300"),
        (e["all_voting_rights"], "1000"),
        (e["ratio_after"], "1.0000"),  # aims at all shares: a take-private
    ]
    header = (
        "要素ID\t項目名\tコンテキストID\t相対年度\t連結・個別\t期間・時点\tユニットID\t単位\t値"
    )
    lines = [header] + [f"{el}\t\tFilingDateInstant\t\t\t\t\t\t{v}" for el, v in rows]
    path = ensure_parent(edinet_doc_path(settings.raw_dir, "S100TOB1", 5))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("XBRL_TO_CSV/jptoo-ton-001.csv", "\n".join(lines).encode("utf-16"))

    offers = pipeline.parse_tender_offers(settings)
    assert offers.loc[0, "event_type"] == "mbo"
    assert offers.loc[0, "prior_stake"] == pytest.approx(0.3)

    days = [d.date() for d in pd.bdate_range("2025-01-01", "2025-08-29")]  # ≥ 140 sessions
    prices = pd.DataFrame(
        {"code": "99990", "date": days, "close": 1000.0, "adj_factor": 1.0, "trading_value": 1e6}
    )
    prices.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "prices")))
    topix = pd.DataFrame({"date": days, "close": 2800.0})
    topix.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "topix")))

    # The target's report filed before the offer: 1bn net assets, 1M shares → ¥1,000 per share.
    fin_path = staging_table_path(settings.staging_dir, "financials")
    fin = pd.read_parquet(fin_path).assign(
        net_assets=1e9, issued=1e6, issued_at_filing=1e6, treasury=0.0
    )
    fin.to_parquet(fin_path, index=False)

    study = pipeline.run_event_study(settings)
    assert study.loc[0, "book_per_share"] == pytest.approx(1000.0)
    assert study.loc[0, "offer_pbr"] == pytest.approx(1.5)  # offer ¥1,500 / book ¥1,000
    assert study.loc[0, "pre_pbr"] == pytest.approx(1.0)  # close ¥1,000 / book ¥1,000
    assert study.loc[0, "premium"] == pytest.approx(0.5)
    assert study.loc[0, "runup_abnormal"] == pytest.approx(0.0)
    assert mart_table_path(settings.marts_dir, "event_study").exists()

    review = pipeline.review_runups(settings)
    row = review.iloc[0]
    assert not row["flagged"]  # flat price: no run-up
    assert row["volume_ratio"] == pytest.approx(1.0)  # flat volume: normal trading
    assert row["activist_before"] is None
    assert row["acquirer"] == "サンプルSPC株式会社"


def test_audit_passes_on_freshly_scored_data(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    assert pipeline.run_audit(settings, AS_OF).empty


class _FakeEdinet:
    """Stands in for EdinetClient: records the days asked for, returns no new documents."""

    def __init__(self) -> None:
        self.days: list[date] = []

    def list_documents(self, day: date) -> list[object]:
        self.days.append(day)
        return []

    def download(self, doc_id: str, fetch_type: int = 5) -> Path:
        raise AssertionError("nothing new to download")


class _FakeJQuants:
    def __init__(self) -> None:
        self.bar_days: list[date] = []

    def daily_bars(self, day: date) -> list[dict[str, object]]:
        self.bar_days.append(day)
        return []

    def equities_master(self, day: date) -> list[dict[str, object]]:
        return []

    def topix(self, start: date, end: date, *, refresh: bool = False) -> list[dict[str, object]]:
        return [{"Date": start.isoformat(), "C": 2800.0}]


def test_update_keeps_topix_history_of_earlier_years(settings: Settings) -> None:
    # Update rebuilt staging/topix from 1 January only, wiping 2021–2025.
    ensure_parent(edinet_doclist_path(settings.raw_dir, date(2025, 7, 2))).write_text("{}")
    prices = pd.DataFrame(
        {"code": ["99990"], "date": [date(2024, 3, 1)], "close": [100.0], "adj_factor": [1.0]}
    )
    prices.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "prices")))

    pipeline.update(settings, _FakeEdinet(), _FakeJQuants(), date(2025, 7, 3))  # type: ignore[arg-type]

    topix = pd.read_parquet(staging_table_path(settings.staging_dir, "topix"))
    assert sorted(topix["date"].astype(str)) == ["2024-03-01", "2025-01-01"]  # one row per year


def test_update_resumes_from_the_last_indexed_day(settings: Settings) -> None:
    edinet, jquants = _FakeEdinet(), _FakeJQuants()
    with pytest.raises(FileNotFoundError, match="backfill"):
        pipeline.update(settings, edinet, jquants, date(2025, 7, 3))  # type: ignore[arg-type]

    ensure_parent(edinet_doclist_path(settings.raw_dir, date(2025, 6, 30))).write_text("{}")
    prices = pd.DataFrame(
        {"code": ["99990"], "date": [date(2025, 6, 30)], "close": [100.0], "adj_factor": [1.0]}
    )
    prices.to_parquet(ensure_parent(staging_table_path(settings.staging_dir, "prices")))

    result = pipeline.update(settings, edinet, jquants, date(2025, 7, 3))  # type: ignore[arg-type]

    assert edinet.days == [date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 2)]  # last day again
    assert jquants.bar_days == [date(2025, 7, 1), date(2025, 7, 2)]  # weekdays after the last price
    assert result.screening_date == date(2025, 7, 3)
    assert result.companies == 1 and result.audit_issues == 0
    assert mart_table_path(settings.marts_dir, "scores_2025-07-03").exists()
    assert mart_table_path(settings.marts_dir, "live_scoreboard").exists()  # updated daily


def test_run_changes_writes_an_alerts_note(settings: Settings) -> None:
    for day in (date(2025, 7, 1), date(2025, 9, 1)):
        pipeline.build_features(settings, day)
        pipeline.score_companies(settings, day)
    changes, _events = pipeline.run_changes(settings, date(2025, 7, 1), date(2025, 9, 1))
    assert changes.empty  # same filing, same score: nothing moved
    note = (settings.marts_dir / "alerts_2025-09-01.md").read_text(encoding="utf-8")
    assert note.startswith("# Radar alerts 2025-07-01 → 2025-09-01")


def test_run_model_trains_before_and_tests_after_the_gap(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    config = LogisticConfig.from_config(load_yaml("model"))
    rng = np.random.default_rng(0)
    dates = [date(2023, 3, 31), date(2023, 6, 30), date(2025, 3, 31), date(2025, 6, 30)]
    rows = []
    for as_of in dates:
        frame = pd.DataFrame({f: rng.normal(size=200) for f in config.features})
        frame["market_cap_jpy"] = rng.uniform(1e9, 1e11, 200)
        frame["median_trading_value_jpy"] = rng.uniform(1e6, 1e8, 200)
        frame["market_segment"] = "Standard"
        frame["label"] = (frame["pbr"] + rng.normal(scale=0.5, size=200) > 1.5).astype(int)
        frame["label_mbo"] = frame["label"]
        frame["label_parent"] = 0  # no parent deal at all: that model cannot be trained
        frame["score"] = -frame["pbr"]  # a deliberately wrong rule score
        frame["as_of"] = as_of
        frame["sec_code"] = [f"C{i:03d}" for i in range(200)]
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    # Flat prices (buy the day before each date, sell a year later) and past offer premiums.
    price_days = sorted(
        {d - pd.Timedelta(days=1) for d in dates} | {d + pd.Timedelta(days=365) for d in dates}
    )
    prices = pd.DataFrame(
        [(f"C{i:03d}", day, 1000.0, 1.0) for i in range(200) for day in price_days],
        columns=["code", "date", "close", "adj_factor"],
    )
    prices.to_parquet(ensure_parent(staging_table_path(s.staging_dir, "prices")))
    pd.DataFrame({"date": price_days, "close": 2000.0}).to_parquet(
        staging_table_path(s.staging_dir, "topix")
    )
    study = pd.DataFrame(
        {"doc_id": ["A", "B"], "event_type": ["mbo", "parent"], "premium": [0.4, 0.3],
         "runup_abnormal": [0.0, 0.0], "is_take_private": [True, True],
         "pre_announced": [False, False]}
    )  # fmt: skip
    study.to_parquet(ensure_parent(mart_table_path(s.marts_dir, "event_study")))
    panel.to_parquet(ensure_parent(mart_table_path(s.marts_dir, "backtest_panel_scoring_v0")))
    metrics = pd.DataFrame({"as_of": dates, "label_complete": True})
    metrics.to_parquet(mart_table_path(s.marts_dir, "backtest_metrics_scoring_v0"))

    comparison, coefs = pipeline.run_model(s, "scoring_v0")

    assert sorted(comparison["as_of"].unique()) == dates[2:]  # only dates after test_start
    mean = comparison.groupby("model")["lift_top"].mean()
    assert mean["logistic"] > mean["scoring_v0"]
    assert coefs.loc[0, "feature"] == "pbr"  # the generating feature dominates
    # Per-type models: MBO trained; parent skipped (no positive in training); combined + hybrid.
    models = set(comparison["model"])
    assert {"logistic_mbo", "logistic_by_type", "hybrid"} <= models
    assert "logistic_parent" not in models
    assert mart_table_path(s.marts_dir, "model_comparison_scoring_v0").exists()
    assert "expected_value" in models
    picks = pd.read_parquet(mart_table_path(s.marts_dir, "pick_returns_by_model_scoring_v0"))
    assert {"scoring_v0", "expected_value"} <= set(picks["model"])
    assert mart_table_path(s.marts_dir, "model_reliability_scoring_v0").exists()
    assert live_model_path(s.marts_dir).exists()


def test_build_factpacks_writes_one_markdown_per_company(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)

    paths = pipeline.build_factpacks(settings, AS_OF, codes=["99990"], top=0)

    assert [p.name for p in paths] == ["factpack_2025-07-01_9999.md"]
    text = paths[0].read_text(encoding="utf-8")
    assert "## Why it screens" in text and "aging_leader" in text
    assert "_Too few comparable offers for a price range._" in text  # no event study yet
    assert "_Not in this filing._" in text  # the synthetic filing has no 事業の内容


def test_live_snapshot_is_written_once_and_never_rewritten(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    assert pipeline.snapshot_live_scores(settings, AS_OF) is True
    path = live_scores_path(settings.marts_dir, AS_OF)
    first = pd.read_parquet(path)

    scores_path = mart_table_path(settings.marts_dir, "scores_2025-07-01")
    pd.read_parquet(scores_path).assign(score=0.0).to_parquet(scores_path)  # re-scored later
    assert pipeline.snapshot_live_scores(settings, AS_OF) is False
    pd.testing.assert_frame_equal(pd.read_parquet(path), first)  # the record did not change


def test_run_picks_writes_top_n_returns(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    days = [d.date() for d in pd.bdate_range("2024-03-01", "2025-04-30")]
    prices = pd.DataFrame(
        {"code": "99990", "date": days, "close": 1000.0, "adj_factor": 1.0, "trading_value": 1e6}
    )
    prices.to_parquet(ensure_parent(staging_table_path(s.staging_dir, "prices")))
    pd.DataFrame({"date": days, "close": 2000.0}).to_parquet(
        staging_table_path(s.staging_dir, "topix")
    )
    panel = pd.DataFrame(
        {"as_of": [date(2024, 3, 31)], "sec_code": ["99990"], "score": [0.5], "label": [0]}
    )
    panel.to_parquet(ensure_parent(mart_table_path(s.marts_dir, "backtest_panel_scoring_v0")))
    metrics = pd.DataFrame({"as_of": [date(2024, 3, 31)], "label_complete": [True]})
    metrics.to_parquet(mart_table_path(s.marts_dir, "backtest_metrics_scoring_v0"))

    result = pipeline.run_picks(s, "scoring_v0")

    assert sorted(result["top_n"].unique()) == sorted(load_yaml("backtest")["pick_sizes"])
    assert result["mean_return"].iloc[0] == pytest.approx(0.0)  # flat price
    assert mart_table_path(s.marts_dir, "pick_returns_scoring_v0").exists()


def test_scores_gain_live_model_columns_when_models_are_saved(settings: Settings) -> None:
    config = LogisticConfig.from_config(load_yaml("model"))
    rng = np.random.default_rng(1)
    train = pd.DataFrame({f: rng.normal(size=300) for f in config.features})
    train["market_cap_jpy"] = rng.uniform(1e9, 1e11, 300)
    train["median_trading_value_jpy"] = rng.uniform(1e6, 1e8, 300)
    train["market_segment"] = "Standard"
    train["label_mbo"] = (train["pbr"] > 1.2).astype(int)
    bundle = {
        "models": fit_by_type(train, config, ["mbo"]),
        "config": config,
        "base_rates": {"mbo": 0.01},
        "premiums": {"mbo": 0.4},
        "no_offer_return": 0.05,
        "fitted_on": ["2025-03-31"],
    }
    joblib.dump(bundle, ensure_parent(live_model_path(settings.marts_dir)))
    pipeline.build_features(settings, AS_OF)

    scores = pipeline.score_companies(settings, AS_OF)

    assert {"p_mbo", "p_any", "expected_value"} <= set(scores.columns)
    assert scores["p_mbo"].between(0, 1).all()
    assert scores["p_any"].iloc[0] == pytest.approx(scores["p_mbo"].iloc[0])  # one type only

    observed = pd.Series([0, 0, 0, 1] * 25)
    bundle["calibrator"] = fit_isotonic(pd.Series(np.linspace(0.0, 1.0, 100)), observed)
    joblib.dump(bundle, live_model_path(settings.marts_dir))
    calibrated = pipeline.score_companies(settings, AS_OF)
    assert calibrated["p_any"].rank().tolist() == pytest.approx(scores["p_any"].rank().tolist())
    assert calibrated["p_any"].between(0, 1).all()
    assert calibrated["p_mbo"].tolist() == pytest.approx(calibrated["p_any"].tolist())
    assert calibrated["p_mbo_prior"].tolist() == pytest.approx(scores["p_mbo"].tolist())
    assert calibrated["combo_v2_mbo"].notna().all()  # live: the combined ranking is frozen too


def test_run_walk_forward_scores_every_candidate_on_seven_dates(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    config = LogisticConfig.from_config(load_yaml("model"))
    rng = np.random.default_rng(2)
    dates = [d.date() for d in pd.date_range("2022-09-30", "2025-09-30", freq="QE")]
    rows = []
    for as_of in dates:
        frame = pd.DataFrame({f: rng.normal(size=150) for f in config.features})
        frame["market_cap_jpy"] = rng.uniform(1e9, 1e11, 150)
        frame["median_trading_value_jpy"] = rng.uniform(1e6, 1e8, 150)
        frame["market_segment"] = "Standard"
        frame["sector33"] = rng.choice(["銀行業", "化学", "小売業"], 150)
        frame["operating_income_to_market_cap"] = rng.normal(0.03, 0.05, 150)
        frame["momentum_12_1"] = rng.normal(0.0, 0.3, 150)
        frame["dividend_yield"] = rng.uniform(0.0, 0.04, 150)
        frame["oldest_representative_age"] = rng.uniform(40, 90, 150)
        frame["label"] = (frame["oldest_representative_age"] > 80).astype(int)
        frame["label_mbo"] = frame["label"]
        frame["label_parent"] = (frame["pbr"] > 1.8).astype(int)
        frame["label_third_party"] = 0
        frame["score"] = rng.random(150)
        frame["as_of"] = as_of
        frame["sec_code"] = [f"C{i:03d}" for i in range(150)]
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(ensure_parent(mart_table_path(s.marts_dir, "backtest_panel_scoring_v0")))
    pd.DataFrame({"as_of": dates, "label_complete": True}).to_parquet(
        mart_table_path(s.marts_dir, "backtest_metrics_scoring_v0")
    )
    days = sorted(
        {d - pd.Timedelta(days=1) for d in dates} | {d + pd.Timedelta(days=365) for d in dates}
    )
    prices = pd.DataFrame(
        [(f"C{i:03d}", d, 1000.0, 1.0) for i in range(150) for d in days],
        columns=["code", "date", "close", "adj_factor"],
    )
    prices.to_parquet(ensure_parent(staging_table_path(s.staging_dir, "prices")))
    pd.DataFrame({"date": days, "close": 2000.0}).to_parquet(
        staging_table_path(s.staging_dir, "topix")
    )
    pd.DataFrame(
        {"doc_id": ["A", "B"], "event_type": ["mbo", "parent"], "premium": [0.4, 0.3],
         "runup_abnormal": [0.0, 0.0], "is_take_private": [True, True],
         "pre_announced": [False, False]}
    ).to_parquet(ensure_parent(mart_table_path(s.marts_dir, "event_study")))  # fmt: skip

    metrics, choice, by_return, rounds = pipeline.run_walk_forward(s, "scoring_v0")

    assert metrics["as_of"].nunique() == 7
    assert {"v0", "v2", "v2_by_type", "logistic_by_type", "hybrid", "expected_value"} <= set(
        metrics["model"]
    )
    # Age drives the targets here: the learned rule weights beat a random v0 score.
    mean = metrics.groupby("model")["lift_top"].mean()
    assert mean["v2"] > mean["v0"]
    assert choice.winner != "v0"
    assert mart_table_path(s.marts_dir, "walk_forward_scoring_v0").exists()
    assert {"value_basket", "return_model", "v2_no_activist"} <= set(metrics["model"])
    assert {"value_basket_quality", "return_model_sector_neutral"} <= set(metrics["model"])
    assert by_return.winner in {"value_basket", "v0", "hybrid", "v2", "return_model"}
    assert rounds["round2"].winner in {
        "value_basket",
        "value_basket_quality",
        "return_model_sector_neutral",
    }
    # Round 3 and the tracked "top 7 v2 + top 7 MBO" combination.
    assert {"value_catalyst", "value_momentum", "combo_v2_mbo"} <= set(metrics["model"])
    assert rounds["round3"].winner in {"value_basket", "value_momentum", "value_catalyst"}
    top_k = pd.read_parquet(mart_table_path(s.marts_dir, "walk_forward_top_k_scoring_v0"))
    assert top_k[["net_return", "capacity_jpy", "style_excess"]].notna().all().all()
    assert mart_table_path(s.marts_dir, "walk_forward_picks_summary_scoring_v0").exists()
    assert return_model_path(s.marts_dir).exists()  # the challenger, refitted on every date
    calibration = pd.read_parquet(
        mart_table_path(s.marts_dir, "walk_forward_calibration_scoring_v0")
    )
    # Isotonic needs a year held out of training: it exists for the later test dates.
    assert set(calibration["method"]) == {"p_prior", "p_isotonic"}
    assert calibration["brier"].between(0, 1).all()


def test_learn_weights_writes_a_rule_file_and_rescores_the_backtest(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path / "data", _env_file=None)  # type: ignore[call-arg]
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "scoring_vbase.yaml").write_text(
        "version: 0\nrules:\n"
        "  - {id: old, feature: age, op: '>=', threshold: 70, weight: 2.0, rationale: Old.}\n"
        "  - {id: noise, feature: noise, op: '>=', threshold: 1, weight: 2.0, rationale: N.}\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(4)
    dates = [date(2024, 3, 31), date(2024, 6, 30)]
    rows = []
    for as_of in dates:
        frame = pd.DataFrame({"age": rng.uniform(40, 90, 400), "noise": rng.integers(0, 2, 400)})
        frame["label"] = ((frame["age"] >= 70) & (rng.random(400) < 0.3)).astype(int)
        frame["as_of"], frame["edinet_code"], frame["score"] = as_of, range(400), 0.0
        frame["excluded_already_public"] = False
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(ensure_parent(mart_table_path(s.marts_dir, "backtest_panel_scoring_vbase")))
    pd.DataFrame({"as_of": dates, "label_complete": True}).to_parquet(
        mart_table_path(s.marts_dir, "backtest_metrics_scoring_vbase")
    )

    learned = pipeline.learn_weights(s, "scoring_vbase", "scoring_vlearned", config_dir=config_dir)

    weights = {r.id: r.weight for r in learned.rules}
    assert weights["old"] > weights["noise"]
    written = RuleSet.load("scoring_vlearned", config_dir=config_dir)
    assert written.rules[0].threshold == 70  # thresholds unchanged, only weights learned
    assert "learned" in (config_dir / "scoring_vlearned.yaml").read_text(encoding="utf-8")
    metrics = pd.read_parquet(mart_table_path(s.marts_dir, "backtest_metrics_scoring_vlearned"))
    assert metrics["in_sample"].all()  # weights fitted on these very dates: flagged


def test_run_allocation_sizes_the_top_picks_with_kelly_from_their_own_history(
    tmp_path: Path,
) -> None:
    s = Settings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    as_of = date(2026, 10, 9)
    pd.DataFrame(
        {"model": "value_basket", "top_n": 25, "as_of": pd.date_range("2024-03-31", periods=7),
         "mean_return": [0.10, 0.25, 0.05, 0.20, 0.15, 0.30, 0.00],
         "universe_mean_return": 0.0}
    ).to_parquet(
        ensure_parent(mart_table_path(s.marts_dir, "walk_forward_picks_scoring_v0"))
    )  # fmt: skip
    pd.DataFrame(
        {"sec_code": ["1001", "1002", "1003"], "company_name": ["サンプルA", "B", "C"],
         "value_basket": [0.9, None, 0.5], "median_trading_value_jpy": [1e9, 1e9, 1e5]}
    ).to_parquet(mart_table_path(s.marts_dir, "scores_2026-10-09"))  # fmt: skip
    pd.DataFrame(
        {"code": ["1001", "1003", "1001"], "date": [date(2026, 10, 8), date(2026, 10, 8), as_of],
         "close": [1000.0, 100.0, 9999.0], "adj_factor": 1.0}
    ).to_parquet(ensure_parent(staging_table_path(s.staging_dir, "prices")))  # fmt: skip

    plan, invested, past = pipeline.run_allocation(
        s, as_of, ranking="value_basket", top=25, capital=10_000_000
    )

    assert len(past) == 7 and invested is not None and 0 < invested <= 1
    assert plan["sec_code"].tolist() == ["1001", "1003"]  # not eligible (NaN) is not a pick
    assert plan.loc[0, "close"] == 1000.0  # the close BEFORE as_of, never the as_of close
    assert plan.loc[1, "reason"] == "liquidity cap"

    # The live column can differ from the walk-forward name (v2 ranks by `score` today).
    scores = pd.read_parquet(mart_table_path(s.marts_dir, "scores_2026-10-09"))
    scores.rename(columns={"value_basket": "score"}).to_parquet(
        mart_table_path(s.marts_dir, "scores_2026-10-09")
    )
    plan, _, past = pipeline.run_allocation(
        s, as_of, ranking="score", model="value_basket", top=25, capital=10_000_000
    )
    assert len(past) == 7 and plan["sec_code"].tolist() == ["1001", "1003"]


def test_update_writes_fact_packs_for_companies_new_in_the_top(settings: Settings) -> None:
    for day in (date(2025, 7, 1), date(2025, 9, 1)):
        pipeline.build_features(settings, day)
        pipeline.score_companies(settings, day)
    old_path = mart_table_path(settings.marts_dir, "scores_2025-07-01")
    old = pd.read_parquet(old_path)
    newcomer = str(old.iloc[0]["sec_code"])
    old.iloc[1:].to_parquet(old_path)  # the leader was not in the July list

    packs = pipeline._factpacks_for_entrants(settings, date(2025, 7, 1), date(2025, 9, 1))

    expected = factpack_path(settings.marts_dir, date(2025, 9, 1), newcomer).name
    assert [p.name for p in packs] == [expected]
    assert pipeline._factpacks_for_entrants(settings, None, date(2025, 9, 1)) == []
    _changes, _events = pipeline.run_changes(settings, date(2025, 7, 1), date(2025, 9, 1))
    note = (settings.marts_dir / "alerts_2025-09-01.md").read_text(encoding="utf-8")
    assert expected in note  # the alert points to the new file


def test_journal_table_reads_the_owners_file(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    assert pipeline.journal_table(settings, AS_OF) is None  # no journal yet: nothing to show
    code = str(pd.read_parquet(mart_table_path(settings.marts_dir, f"scores_{AS_OF}"))
               .iloc[0]["sec_code"])  # fmt: skip
    ensure_parent(journal_path(settings.manual_dir)).write_text(
        f"entries:\n  - {{date: 2025-01-06, code: '{code[:4]}', action: watch, reason: test}}\n",
        encoding="utf-8",
    )
    table = pipeline.journal_table(settings, AS_OF)
    assert table is not None and table.loc[0, "rank_now"] == 1
    assert table.loc[0, "action"] == "watch"


def test_ties_at_the_cutoff_are_not_new_entrants(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for day in (date(2025, 7, 1), date(2025, 9, 1)):
        pipeline.build_features(settings, day)
        pipeline.score_companies(settings, day)
        path = mart_table_path(settings.marts_dir, f"scores_{day.isoformat()}")
        frame = pd.read_parquet(path).assign(score=0.5)  # everyone tied
        frame = frame.iloc[::-1] if day.month == 9 else frame  # a different arbitrary order
        frame.to_parquet(path)
    real = pipeline.load_yaml
    monkeypatch.setattr(  # top 1 with three tied companies: head(1) would pick a different one
        pipeline, "load_yaml",
        lambda name, *a: {"factpack_new_top": 1} if name == "report" else real(name, *a),
    )  # fmt: skip
    assert pipeline._factpacks_for_entrants(settings, date(2025, 7, 1), date(2025, 9, 1)) == []


def test_prune_marts_keeps_scores_and_the_live_record(settings: Settings) -> None:
    pipeline.build_features(settings, AS_OF)
    pipeline.score_companies(settings, AS_OF)
    pipeline.snapshot_live_scores(settings, AS_OF)
    deleted = pipeline.prune_marts(settings, AS_OF + timedelta(days=400))
    assert [p.name for p in deleted] == [f"features_{AS_OF.isoformat()}.parquet"]
    assert mart_table_path(settings.marts_dir, f"scores_{AS_OF.isoformat()}").exists()
    assert live_scores_path(settings.marts_dir, AS_OF).exists()  # write-once, never pruned
