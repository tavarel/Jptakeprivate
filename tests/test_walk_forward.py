"""Walk-forward validation, learned rule weights and the pre-registered decision rule."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.model.backtest import choose_default, quarter_ends, walk_forward_splits
from radar.model.return_model import fit_return_model, predict_return
from radar.model.rules import (
    Rule,
    RuleSet,
    bootstrap_rule_weights,
    default_ruleset_name,
    fit_rule_model,
    rule_flags,
    with_weights,
)


def test_walk_forward_splits_keep_a_full_horizon_between_train_and_test() -> None:
    dates = quarter_ends(date(2022, 9, 30), date(2025, 9, 30))  # 13 complete dates
    splits = walk_forward_splits(dates, horizon_days=365, first_test=date(2024, 3, 31))
    tests = [test for _, test in splits]
    assert tests[0] == date(2024, 3, 31) and tests[-1] == date(2025, 9, 30)
    assert len(splits) == 7
    for train, test in splits:
        assert train and max(train) <= date(test.year - 1, test.month, test.day)
        assert test not in train
    assert splits[0][0] == [date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]


RULES = RuleSet(
    version=2,
    rules=[
        Rule(id="old", feature="age", op=">=", threshold=70, weight=1.0),
        Rule(id="noise", feature="noise", op=">=", threshold=1, weight=1.0),
    ],
)


def test_rule_flags_are_zero_one_and_missing_is_zero() -> None:
    frame = pd.DataFrame({"age": [75.0, 50.0, None], "noise": [0, 1, 1]})
    flags = rule_flags(frame, RULES)
    assert flags["old"].tolist() == [1, 0, 0]  # a missing feature never fires a rule
    assert flags["noise"].tolist() == [0, 1, 1]


def test_learned_weights_favour_the_rule_that_generates_targets() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame({"age": rng.uniform(40, 90, n), "noise": rng.integers(0, 2, n)})
    frame["label"] = ((frame["age"] >= 70) & (rng.random(n) < 0.3)).astype(int)
    model = fit_rule_model(frame, RULES, "label", c=1.0)
    weights = dict(zip([r.id for r in RULES.rules], model.coef_[0], strict=True))
    assert weights["old"] > 1.0 and abs(weights["noise"]) < 0.3
    learned = with_weights(RULES, weights)
    assert learned.rules[0].weight == pytest.approx(weights["old"])
    assert learned.max_score == pytest.approx(max(weights["old"], 0) + max(weights["noise"], 0))
    assert learned.min_score == pytest.approx(min(weights["old"], 0) + min(weights["noise"], 0))


def test_choose_default_needs_wins_on_most_dates_and_prefers_simple() -> None:
    dates = [date(2024, 3, 31) + pd.Timedelta(days=91 * i) for i in range(7)]
    rows = []
    for i, d in enumerate(dates):
        rows += [
            {"as_of": d, "model": "v0", "lift_top": 1.5},
            {"as_of": d, "model": "v2", "lift_top": 2.0 if i < 6 else 1.0},  # wins 6 of 7
            {"as_of": d, "model": "logistic_by_type", "lift_top": 2.05 if i < 4 else 1.4},
        ]  # fmt: skip
    metrics = pd.DataFrame(rows)
    order = ["v0", "v2", "v2_by_type", "logistic_by_type"]
    result = choose_default(metrics, baseline="v0", min_wins=5, tie=0.1, simplicity=order)
    assert result.winner == "v2"  # logistic wins only 4 of 7 dates: not eligible
    assert result.table.set_index("model").loc["v2", "wins_vs_baseline"] == 6
    weak = metrics.assign(lift_top=np.where(metrics["model"] == "v0", 3.0, metrics["lift_top"]))
    assert choose_default(weak, "v0", 5, 0.1, order).winner == "v0"  # nobody beats it


def test_default_ruleset_comes_from_config(tmp_path: Path) -> None:
    (tmp_path / "scoring.yaml").write_text("default_ruleset: scoring_vtest\n", encoding="utf-8")
    (tmp_path / "scoring_vtest.yaml").write_text(
        "version: 9\nrules:\n  - {id: old, feature: age, op: '>=', threshold: 70, weight: 1.5}\n",
        encoding="utf-8",
    )
    ruleset = RuleSet.default(config_dir=tmp_path)
    assert ruleset.version == 9 and ruleset.rules[0].weight == 1.5
    assert default_ruleset_name(config_dir=tmp_path) == "scoring_vtest"


def test_choose_default_works_on_any_metric() -> None:
    rows = [
        {"as_of": d, "model": m, "excess": v}
        for d in range(7)
        for m, v in (("value_basket", 0.03), ("v0", 0.05 if d < 5 else 0.0))
    ]
    order = ["value_basket", "v0"]
    choice = choose_default(pd.DataFrame(rows), "value_basket", 5, 0.01, order, metric="excess")
    # v0 wins 5 of 7 dates, but its mean (3.57%) is not 1 pp above the basket (3%): basket stays.
    assert choice.winner == "value_basket"
    assert choice.table.set_index("model").loc["v0", "wins_vs_baseline"] == 5


def test_return_model_learns_which_feature_predicts_excess_returns() -> None:
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"pbr": rng.uniform(0.3, 3, 800), "noise": rng.normal(size=800)})
    train["excess"] = -0.1 * train["pbr"] + rng.normal(scale=0.05, size=800)
    model = fit_return_model(train, train["excess"], ["pbr", "noise"], alpha=1.0, clip=0.01)
    predicted = predict_return(model, train, ["pbr", "noise"])
    assert np.corrcoef(predicted, train["excess"])[0, 1] > 0.5


def test_bootstrap_over_dates_shows_which_weights_are_stable() -> None:
    rng = np.random.default_rng(3)
    frames = []
    for i in range(8):
        frame = pd.DataFrame({"age": rng.uniform(40, 90, 300), "noise": rng.integers(0, 2, 300)})
        frame["label"] = ((frame["age"] >= 70) & (rng.random(300) < 0.3)).astype(int)
        frame["as_of"] = date(2023, 3, 31) + pd.Timedelta(days=91 * i)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    table = bootstrap_rule_weights(panel, RULES, "label", c=1.0, draws=30, seed=7)
    table = table.set_index("rule")
    assert table.loc["old", "p5"] > 0.5  # the generating rule is positive in almost every draw
    assert table.loc["noise", "p5"] < 0 < table.loc["noise", "p95"]  # noise straddles zero
    assert table.loc["old", "share_positive"] == pytest.approx(1.0)
    again = bootstrap_rule_weights(panel, RULES, "label", c=1.0, draws=30, seed=7)
    assert again.set_index("rule").equals(table)  # same seed → same result
