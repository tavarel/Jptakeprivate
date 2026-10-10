from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from radar.ingest.edinet import parse_document_list
from radar.labels.tender_offers import (
    already_public,
    label_companies,
    label_frame,
    take_private_events,
    tender_offer_events,
)
from radar.model.backtest import (
    ablation,
    compare_rankings,
    evaluate_by_type,
    evaluate_ranking,
    lift_at_fraction,
    live_scoreboard,
    precision_at_k,
    quarter_ends,
    recall_at_fraction,
    time_split,
)
from radar.model.rules import Rule, RuleSet, audit_scores, score
from radar.pipeline import documents_frame


# ------------------------------------------------------------------ labels
def test_tender_offer_events_exclude_self_tenders(doclist_payload: dict[str, Any]) -> None:
    documents = documents_frame(parse_document_list(doclist_payload))
    events = tender_offer_events(documents)
    assert list(events["subject_edinet_code"]) == ["E99002"]  # E99003 is a self-tender
    assert events.loc[0, "event_date"] == date(2025, 6, 27)


def test_tender_offer_events_drop_withdrawn_offers() -> None:
    def doc(doc_id: str, type_: str, subject: str, parent: str | None, day: str) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "doc_type_code": type_,
            "withdrawal_status": "0",
            "edinet_code": "E99100",
            "filer_name": "サンプル買付者",
            "subject_edinet_code": subject,
            "parent_doc_id": parent,
            "submit_datetime": f"{day} 09:00",
        }

    documents = pd.DataFrame(
        [
            doc("WITHDRAWN", "240", "E99001", None, "2025-04-04"),
            doc("NOTICE", "260", "E99001", "WITHDRAWN", "2025-05-09"),
            doc("NO_PARENT", "240", "E99002", None, "2025-04-04"),
            doc("NOTICE2", "260", "E99002", None, "2025-05-09"),  # no link: matched by parties
            doc("KEPT", "240", "E99003", None, "2025-04-04"),
        ]
    )
    assert tender_offer_events(documents)["doc_id"].tolist() == ["KEPT"]


def test_label_window_starts_after_as_of() -> None:
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1", "E2", "E3"],
            "event_date": [date(2025, 3, 31), date(2025, 4, 1), date(2026, 4, 1)],
            "doc_id": ["a", "b", "c"],
            "acquirer_name": ["x", "y", "z"],
        }
    )
    codes = pd.Series(["E1", "E2", "E3", "E4"])
    labels = label_companies(codes, events, as_of=date(2025, 3, 31), horizon_days=365)
    assert list(labels) == [0, 1, 0, 0]  # same-day and >365-day events are not labels


# ------------------------------------------------------------------ rules
def test_real_ruleset_loads_and_scores() -> None:
    ruleset = RuleSet.load("scoring_v0")
    features = pd.DataFrame(
        {
            "edinet_code": ["HIT", "MISS", "MISSING"],
            "oldest_representative_age": [76, 50, None],
            "family_stake": [0.45, 0.02, None],
            "pbr": [0.6, 2.5, None],
            "net_cash_to_market_cap": [0.5, 0.0, None],
            "market_cap_jpy": [8e9, 9e11, None],
            "market_segment": ["Standard", "Prime", None],
            "capital_plan_disclosed": [False, True, None],
            "largest_listed_holder_stake": [0.25, 0.0, None],
            "activist_filers_365d": [1, 0, None],
        }
    )
    scored = score(features, ruleset).set_index("edinet_code")
    assert scored.loc["HIT", "score"] == pytest.approx(1.0)
    assert scored.loc["MISSING", "score"] == 0.0  # missing data never adds points
    assert "aging_leader" in scored.loc["HIT", "triggered_rules"]


# ------------------------------------------------------------------ backtest metrics
def test_ranking_metrics() -> None:
    scores = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
    labels = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    assert precision_at_k(scores, labels, 2) == 0.5
    assert recall_at_fraction(scores, labels, 0.3) == pytest.approx(2 / 3)
    assert lift_at_fraction(scores, labels, 0.3) == pytest.approx((2 / 3) / 0.3)


def test_time_split_requires_a_gap() -> None:
    frame = pd.DataFrame({"as_of": [date(2023, 3, 31), date(2024, 3, 31), date(2025, 3, 31)]})
    train, test = time_split(frame, train_end=date(2023, 3, 31), test_start=date(2024, 4, 1))
    assert len(train) == 1
    assert len(test) == 1
    with pytest.raises(ValueError):
        time_split(frame, train_end=date(2024, 1, 1), test_start=date(2023, 1, 1))


def test_quarter_ends() -> None:
    assert quarter_ends(date(2024, 3, 31), date(2024, 12, 1)) == [
        date(2024, 3, 31),
        date(2024, 6, 30),
        date(2024, 9, 30),
    ]
    assert quarter_ends(date(2024, 4, 1), date(2024, 6, 29)) == []


def test_evaluate_ranking() -> None:
    scores = pd.Series([0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
    labels = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    result = evaluate_ranking(scores, labels, precision_at=[2], top_fraction=0.3)
    assert result == {
        "companies": 10,
        "positives": 3,
        "precision_at_2": 0.5,
        "recall_top": pytest.approx(2 / 3),
        "lift_top": pytest.approx((2 / 3) / 0.3),
    }


def test_ablation_measures_each_rule() -> None:
    ruleset = RuleSet(
        version=0,
        rules=[
            Rule(id="old", feature="age", op=">=", threshold=70, weight=2.0),
            Rule(id="noise", feature="noise", op=">=", threshold=1, weight=1.0),
        ],
    )
    # One as_of: the target is old; "noise" fires on two non-targets and pushes them up.
    panel = pd.DataFrame(
        {
            "as_of": [date(2025, 6, 30)] * 4,
            "age": [75, 50, 50, 50],
            "noise": [0, 1, 1, 0],
            "label": [1, 0, 0, 0],
        }
    )
    result = ablation(panel, ruleset, precision_at=[1], top_fraction=0.25).set_index("without")
    assert result.loc["(none)", "precision_at_1"] == 1.0  # full rule set: target ranked first
    assert result.loc["old", "precision_at_1"] == 0.0  # without the age rule the target sinks
    assert result.loc["noise", "precision_at_1"] == 1.0  # removing noise changes nothing
    assert result.loc["old", "lift_top_delta"] < 0  # the rule carries the lift


def test_audit_scores_recomputes_every_rule_and_the_score() -> None:
    ruleset = RuleSet.load()
    features = pd.DataFrame(
        {
            "edinet_code": ["OK", "TAMPERED"],
            "oldest_representative_age": [76, 76],
            "capital_plan_disclosed": [False, True],
            "market_segment": ["Standard", "Standard"],
        }
    )
    scored = score(features, ruleset).set_index("edinet_code")
    assert audit_scores(scored.reset_index(), ruleset).empty  # consistent as produced

    # Simulate a stale/edited scores file: TAMPERED claims a rule its data does not support.
    broken = scored.reset_index()
    broken["triggered_rules"] = [
        list(scored.loc["OK", "triggered_rules"]),
        ["aging_leader", "standard_market", "no_capital_plan"],
    ]
    issues = audit_scores(broken, ruleset)
    assert set(issues["edinet_code"]) == {"TAMPERED"}
    assert "no_capital_plan" in issues.iloc[0]["detail"]


def test_take_private_events_drop_known_partial_offers() -> None:
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1", "E2", "E3"],
            "event_date": [date(2025, 5, 1)] * 3,
            "doc_id": ["FULL", "PARTIAL", "NOT_READ"],
            "acquirer_name": ["A", "牧 寛之", "C"],
        }
    )
    offers = pd.DataFrame({"doc_id": ["FULL", "PARTIAL"], "is_take_private": [True, False]})
    kept = take_private_events(events, offers)
    # BASE-like partial offer removed; an offer whose 240 was not read yet is kept (unknown).
    assert kept["doc_id"].tolist() == ["FULL", "NOT_READ"]
    assert take_private_events(events, None)["doc_id"].tolist() == ["FULL", "PARTIAL", "NOT_READ"]


def test_label_frame_has_one_label_per_event_type() -> None:
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1", "E2", "E3"],
            "event_date": [date(2025, 5, 1)] * 3,
            "doc_id": ["A", "B", "NOT_READ"],
            "acquirer_name": ["x", "y", "z"],
        }
    )
    offers = pd.DataFrame({"doc_id": ["A", "B"], "event_type": ["mbo", "parent"]})
    codes = pd.Series(["E1", "E2", "E3", "E4"])
    labels = label_frame(codes, events, offers, date(2025, 3, 31), horizon_days=365)
    assert labels["label"].tolist() == [1, 1, 1, 0]  # every offer, typed or not
    assert labels["label_mbo"].tolist() == [1, 0, 0, 0]
    assert labels["label_parent"].tolist() == [0, 1, 0, 0]
    assert labels["label_third_party"].tolist() == [0, 0, 0, 0]  # column exists even if empty
    assert labels["label_unknown"].tolist() == [0, 0, 1, 0]  # offer not read yet


def test_label_frame_without_offer_terms_types_everything_unknown() -> None:
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1"],
            "event_date": [date(2025, 5, 1)],
            "doc_id": ["A"],
            "acquirer_name": ["x"],
        }
    )
    labels = label_frame(pd.Series(["E1"]), events, None, date(2025, 3, 31), 365)
    assert labels["label_unknown"].tolist() == [1]


def test_evaluate_by_type_reports_recall_and_lift_per_type() -> None:
    scores = pd.Series([4.0, 3.0, 2.0, 1.0])
    labels = pd.DataFrame({"label_mbo": [1, 0, 0, 0], "label_parent": [0, 0, 0, 1]})
    result = evaluate_by_type(scores, labels, top_fraction=0.25)
    assert result["positives_mbo"] == 1 and result["recall_top_mbo"] == 1.0
    assert result["lift_top_mbo"] == 4.0  # precision 1 in the top 25% / base rate 1/4
    assert result["recall_top_parent"] == 0.0


def test_ablation_reports_lift_per_event_type() -> None:
    ruleset = RuleSet(
        version=0, rules=[Rule(id="old", feature="age", op=">=", threshold=70, weight=2.0)]
    )
    panel = pd.DataFrame(
        {
            "as_of": [date(2025, 6, 30)] * 4,
            "age": [50, 75, 50, 50],  # target second: ties keep input order
            "label": [0, 1, 0, 0],
            "label_mbo": [0, 1, 0, 0],
        }
    )
    result = ablation(panel, ruleset, precision_at=[1], top_fraction=0.25).set_index("without")
    assert result.loc["(none)", "lift_top_mbo"] == 4.0
    assert result.loc["old", "lift_top_mbo"] < 4.0  # without the rule the MBO target is not on top


def test_compare_rankings_judges_each_model_on_the_same_dates() -> None:
    panel = pd.DataFrame(
        {
            "as_of": [date(2025, 3, 31)] * 4 + [date(2025, 6, 30)] * 4,
            "good": [4.0, 3.0, 2.0, 1.0] * 2,
            "bad": [1.0, 2.0, 3.0, 4.0] * 2,
            "label": [1, 0, 0, 0] * 2,
            "label_mbo": [1, 0, 0, 0] * 2,
        }
    )
    result = compare_rankings(panel, ["good", "bad"], precision_at=[1], top_fraction=0.25)
    assert result["as_of"].nunique() == 2
    mean = result.groupby("model")[["precision_at_1", "lift_top_mbo"]].mean()
    assert mean.loc["good", "precision_at_1"] == 1.0 and mean.loc["bad", "precision_at_1"] == 0.0
    assert mean.loc["good", "lift_top_mbo"] == 4.0


def test_label_frame_tolerates_a_duplicated_offer_row() -> None:
    events = pd.DataFrame(
        {"subject_edinet_code": ["E1"], "event_date": [date(2025, 5, 1)], "doc_id": ["A"],
         "acquirer_name": ["x"]}
    )  # fmt: skip
    offers = pd.DataFrame({"doc_id": ["A", "A"], "event_type": ["mbo", "mbo"]})
    labels = label_frame(pd.Series(["E1"]), events, offers, date(2025, 3, 31), 365)
    assert labels["label_mbo"].tolist() == [1]


def test_live_scoreboard_counts_hits_so_far_and_flags_partial_windows() -> None:
    snapshot = pd.DataFrame(
        {"edinet_code": ["E1", "E2", "E3", "E4"] * 5, "score": [0.9, 0.5, 0.2, 0.1] * 5}
    ).drop_duplicates("edinet_code")
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1", "E4", "E2"],
            "event_date": [date(2026, 10, 9), date(2026, 12, 1), date(2027, 3, 1)],
            "doc_id": ["SAME_DAY", "B", "C"],
            "acquirer_name": ["x", "y", "z"],
        }
    )
    board = live_scoreboard(
        {date(2026, 10, 9): snapshot}, events, today=date(2027, 1, 15), horizon_days=365,
        top_fraction=0.25, top_n=1,
    )  # fmt: skip
    row = board.iloc[0]
    assert row["days_elapsed"] == 98 and not row["window_complete"]
    assert row["targets_so_far"] == 1  # E1's offer is ON the snapshot day; E2's is in the future
    assert row["hits_top_fraction"] == 0  # the only target so far (E4) ranked last
    assert row["lift_so_far"] == 0.0


def test_already_public_offers_leave_the_evaluation() -> None:
    # A 予告TOB is announced weeks before its 240 is filed; on the screening date
    # the price already sits at the offer, so "predicting" it would be cheating.
    events = pd.DataFrame(
        {
            "subject_edinet_code": ["E1", "E2", "E3", "E4"],
            "offer_price": [1000.0, 1000.0, None, 1000.0],
        }
    )
    closes = {"E1": 985.0, "E2": 700.0, "E3": 990.0}  # E4: no price
    public = already_public(events, closes, tolerance=0.03)
    assert public == {"E1"}  # E2 far below; E3 offer price unknown; E4 price unknown


def test_live_scoreboard_compares_several_rankings_side_by_side() -> None:
    snapshot = pd.DataFrame(
        {"edinet_code": ["E1", "E2", "E3", "E4"], "score": [0.9, 0.5, 0.2, 0.1],
         "expected_value": [0.0, 0.1, 0.2, 0.9]}
    )  # fmt: skip
    events = pd.DataFrame(
        {"subject_edinet_code": ["E4"], "event_date": [date(2026, 12, 1)], "doc_id": ["B"],
         "acquirer_name": ["y"]}
    )  # fmt: skip
    board = live_scoreboard(
        {date(2026, 10, 9): snapshot}, events, today=date(2027, 1, 15), horizon_days=365,
        top_fraction=0.25, top_n=1, rankings=["score", "expected_value"],
    ).set_index("ranking")  # fmt: skip
    assert board.loc["score", "hits_top_fraction"] == 0
    assert board.loc["expected_value", "hits_top_fraction"] == 1  # E4 ranked first by EV
