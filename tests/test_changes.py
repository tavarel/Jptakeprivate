"""'What changed' between two screening dates, and event alerts (report/changes.py)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from radar.report.changes import (
    alerts_markdown,
    compare_scores,
    coverage_drift,
    new_events,
    watch_codes,
)


def _scores(rows: list[tuple[str, float, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edinet_code": [r[0] for r in rows],
            "company_name": [f"サンプル{r[0]}" for r in rows],
            "sec_code": [f"9{r[0][-3:]}0" for r in rows],
            "score": [r[1] for r in rows],
            "triggered_rules": [r[2] for r in rows],
        }
    )


def test_compare_scores_entries_exits_and_new_rules() -> None:
    old = _scores([("E001", 0.6, ["a", "b"]), ("E002", 0.5, ["a"]), ("E003", 0.1, [])])
    new = _scores([("E001", 0.7, ["a", "b", "c"]), ("E002", 0.1, []), ("E003", 0.55, ["d"])])
    changes = compare_scores(old, new, top_n=2, max_score=10.0).set_index("edinet_code")
    assert changes.loc["E003", "status"] == "entered top 2"
    assert changes.loc["E002", "status"] == "left top 2"
    assert changes.loc["E001", "status"] == "rules changed"
    assert changes.loc["E001", "rules_added"] == ["c"]
    assert changes.loc["E002", "rules_removed"] == ["a"]
    assert changes.loc["E003", "points_change"] == 4.5  # 0.1 → 0.55 of 10 points


def test_compare_scores_new_company_and_quiet_companies() -> None:
    old = _scores([("E001", 0.6, ["a"]), ("E009", 0.0, [])])
    new = _scores([("E001", 0.6, ["a"]), ("E009", 0.0, []), ("E010", 0.9, ["a", "b"])])
    changes = compare_scores(old, new, top_n=1, max_score=10.0)
    # E010 (first report) enters the top and pushes E001 out; nothing happened to E009.
    assert set(changes["edinet_code"]) == {"E010", "E001"}


def _documents() -> pd.DataFrame:
    rows = [
        (
            "A1",
            "350",
            "株式会社ストラテジックキャピタル",
            None,
            "E001",
            None,
            datetime(2026, 9, 10, 15),
        ),
        ("A2", "350", "野村證券株式会社", None, "E001", None, datetime(2026, 9, 11, 15)),
        ("T1", "240", "サンプルSPC株式会社", "E001", None, "E99", datetime(2026, 9, 12, 9)),
        ("O1", "290", "サンプル001", None, None, "E001", datetime(2026, 9, 13, 9)),
        ("X1", "350", "エリオット", None, "E777", None, datetime(2026, 9, 14, 9)),  # not followed
        (
            "S0",
            "240",
            "サンプルSPC株式会社",
            "E001",
            None,
            "E99",
            datetime(2026, 9, 1, 9),
        ),  # = since
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "doc_id", "doc_type_code", "filer_name", "subject_edinet_code",
            "issuer_edinet_code", "edinet_code", "submit_datetime",
        ],
    ).assign(withdrawal_status="0")  # fmt: skip


def test_new_events_for_followed_companies_after_since() -> None:
    events = new_events(
        _documents(),
        followed={"E001"},
        since=datetime(2026, 9, 1, 9),
        until=datetime(2026, 9, 30, 23, 59),
        activist_patterns=("ストラテジックキャピタル", "エリオット"),
    )
    assert events["doc_id"].tolist() == ["O1", "T1", "A2", "A1"]  # newest first
    kinds = dict(zip(events["doc_id"], events["kind"], strict=True))
    assert kinds["A1"] == "activist 5% filing"
    assert kinds["A2"] == "5% filing"
    assert kinds["T1"] == "tender offer filed"
    assert kinds["O1"] == "target's opinion on an offer"
    assert set(events["company_code"]) == {"E001"}


def test_watch_codes_maps_exchange_codes_to_edinet_codes() -> None:
    documents = pd.DataFrame(
        {
            "edinet_code": ["E001", "E002"],
            "sec_code": ["96470", "130A0"],
            "submit_datetime": [date(2026, 1, 1)] * 2,
        }
    )
    entries = [{"code": "9647", "note": "x"}, {"code": "130A"}, {"code": "0000"}]
    assert watch_codes(entries, documents) == {"E001", "E002"}  # unknown codes are ignored


def test_alerts_markdown_lists_changes_and_events() -> None:
    changes = compare_scores(
        _scores([("E001", 0.1, [])]), _scores([("E001", 0.6, ["a"])]), top_n=1, max_score=10.0
    )
    events = new_events(
        _documents(),
        followed={"E001"},
        since=datetime(2026, 9, 1, 9),
        until=datetime(2026, 9, 30),
        activist_patterns=("ストラテジックキャピタル",),
    )
    text = alerts_markdown(
        changes, events, {"E001": "サンプル001"}, date(2026, 9, 1), date(2026, 10, 1)
    )
    assert text.startswith("# Radar alerts 2026-09-01 → 2026-10-01")
    # The only company is #1 on both dates: its rules changed (+5 points, rule a added).
    assert "rules changed" in text and "+5.0 points" in text and "+a" in text
    assert "activist 5% filing" in text and "サンプル001" in text
    assert "disclosure2.edinet-fsa.go.jp" in text  # each event links to its filing


def test_alerts_markdown_when_nothing_happened() -> None:
    empty = compare_scores(_scores([]), _scores([]), top_n=5, max_score=10.0)
    events = new_events(
        _documents().iloc[0:0], set(), datetime(2026, 9, 1), datetime(2026, 9, 2), ()
    )
    assert "Nothing new" in alerts_markdown(empty, events, {}, date(2026, 9, 1), date(2026, 9, 2))


def test_extraordinary_reports_alert_only_control_related_reasons() -> None:
    def report(doc_id: str, day: int) -> tuple[object, ...]:
        return (doc_id, "180", "サンプル001", None, None, "E001", datetime(2026, 9, day, 15))

    documents = pd.DataFrame(
        [report("K1", 10), report("K2", 11), report("K3", 12),
         ("K4", "180", "サンプル777", None, None, "E777", datetime(2026, 9, 12, 15))],
        columns=["doc_id", "doc_type_code", "filer_name", "subject_edinet_code",
                 "issuer_edinet_code", "edinet_code", "submit_datetime"],
    ).assign(withdrawal_status="0")  # fmt: skip
    classified = {
        "K1": {"alert": True, "labels": "親会社の異動 / 主要株主の異動", "tender_offer": True},
        "K2": {"alert": False, "labels": "株主総会の決議", "tender_offer": False},
    }  # K3 not downloaded yet; K4 is about a company nobody follows
    events = new_events(
        documents, followed={"E001"}, since=datetime(2026, 9, 1), until=datetime(2026, 9, 30),
        activist_patterns=(), extraordinary=classified,
    )  # fmt: skip
    kinds = dict(zip(events["doc_id"], events["kind"], strict=True))
    assert kinds["K1"] == "臨時報告書: 親会社の異動 / 主要株主の異動 (after a tender offer)"
    assert "K2" not in kinds  # AGM vote results: routine, not alerted
    assert kinds["K3"] == "臨時報告書 (reason not read yet)"
    assert "K4" not in kinds


def test_coverage_drift_flags_a_feature_that_suddenly_goes_missing() -> None:
    old = pd.DataFrame({"pbr": [1.0, 0.5, 0.8, 1.2], "family_stake": [0.1, None, 0.3, 0.0]})
    new = pd.DataFrame({"pbr": [1.0, None, None, None], "family_stake": [0.1, None, 0.3, 0.0]})
    drift = coverage_drift(old, new, ["pbr", "family_stake", "absent"], max_drop=0.05)
    assert drift == ["pbr: known for 100% → 25% of companies"]


def test_alerts_note_puts_broken_data_first_and_lists_unusual_trading() -> None:
    empty = compare_scores(_scores([]), _scores([]), top_n=5, max_score=10.0)
    events = pd.DataFrame(
        columns=["submit_datetime", "company_code", "kind", "filer_name", "doc_id"]
    )
    unusual = pd.DataFrame({"sec_code": ["99990"], "volume_ratio": [4.0], "return_5d": [0.15],
                            "excess_5d": [0.15], "company_name": ["サンプル工業"]})  # fmt: skip
    text = alerts_markdown(
        empty, events, {}, date(2026, 9, 1), date(2026, 10, 1),
        unusual=unusual, drift=["pbr: known for 98% → 40% of companies"],
    )  # fmt: skip
    assert text.index("Data coverage dropped") < text.index("Unusual trading")
    assert "**サンプル工業** (9999): volume ×4.0 vs usual, +15% vs TOPIX" in text
