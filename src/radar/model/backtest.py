"""Backtest metrics for a ranking problem with rare positives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from radar.model.rules import RuleSet, score


def _ranked(scores: pd.Series, labels: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"score": scores.to_numpy(), "label": labels.to_numpy()})
    # Stable sort so ties keep input order (deterministic results).
    return frame.sort_values("score", ascending=False, kind="mergesort")["label"]


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = _ranked(scores, labels).head(k)
    return float(top.mean()) if len(top) else 0.0


def recall_at_fraction(scores: pd.Series, labels: pd.Series, fraction: float = 0.1) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    k = max(1, math.ceil(len(labels) * fraction))
    return float(_ranked(scores, labels).head(k).sum() / positives)


def lift_at_fraction(scores: pd.Series, labels: pd.Series, fraction: float = 0.1) -> float:
    base_rate = float(labels.mean())
    if base_rate == 0:
        return float("nan")
    k = max(1, math.ceil(len(labels) * fraction))
    return precision_at_k(scores, labels, k) / base_rate


def time_split(
    frame: pd.DataFrame, train_end: date, test_start: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a panel with an `as_of` column."""
    if test_start <= train_end:
        raise ValueError("test_start must be after train_end")
    as_of = pd.to_datetime(frame["as_of"]).dt.date
    return frame[as_of <= train_end], frame[as_of >= test_start]


def quarter_ends(start: date, end: date) -> list[date]:
    """Calendar quarter-ends (03-31, 06-30, 09-30, 12-31) within [start, end]."""
    ends = pd.date_range(start, end, freq="QE")
    return [d.date() for d in ends]


def evaluate_ranking(
    scores: pd.Series, labels: pd.Series, precision_at: list[int], top_fraction: float
) -> dict[str, float | int]:
    """All ranking metrics for one as_of date (see module docstring)."""
    result: dict[str, float | int] = {"companies": len(labels), "positives": int(labels.sum())}
    for k in precision_at:
        result[f"precision_at_{k}"] = precision_at_k(scores, labels, k)
    result["recall_top"] = recall_at_fraction(scores, labels, top_fraction)
    result["lift_top"] = lift_at_fraction(scores, labels, top_fraction)
    return result


def evaluate_by_type(
    scores: pd.Series, labels: pd.DataFrame, top_fraction: float
) -> dict[str, float | int]:
    """positives / recall / lift in the top fraction for each `label_<type>` column."""
    result: dict[str, float | int] = {}
    for column in labels.columns:
        kind = column.removeprefix("label_")
        result[f"positives_{kind}"] = int(labels[column].sum())
        result[f"recall_top_{kind}"] = recall_at_fraction(scores, labels[column], top_fraction)
        result[f"lift_top_{kind}"] = lift_at_fraction(scores, labels[column], top_fraction)
    return result


def compare_rankings(
    panel: pd.DataFrame, score_columns: list[str], precision_at: list[int], top_fraction: float
) -> pd.DataFrame:
    """Judge several scores (e.g. rules v0 vs logistic) on the SAME companies and dates."""
    typed = _typed_label_columns(panel)
    rows = []
    for as_of, group in panel.groupby("as_of"):
        for column in score_columns:
            metrics = evaluate_ranking(group[column], group["label"], precision_at, top_fraction)
            by_type = evaluate_by_type(group[column], group[typed], top_fraction)
            rows.append({"as_of": as_of, "model": column, **metrics, **by_type})
    return pd.DataFrame(rows)


def _typed_label_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("label_") and c != "label_complete"]


def ablation(
    panel: pd.DataFrame, ruleset: RuleSet, precision_at: list[int], top_fraction: float
) -> pd.DataFrame:
    """Which rules carry the result?"""
    variants = [("(none)", ruleset)] + [
        (
            rule.id,
            RuleSet(version=ruleset.version, rules=[r for r in ruleset.rules if r is not rule]),
        )
        for rule in ruleset.rules
    ]
    features = panel.drop(columns=["score", "triggered_rules"], errors="ignore")
    typed = _typed_label_columns(features)
    rows = []
    for name, variant in variants:
        per_date = []
        for _, group in features.groupby("as_of"):
            scored = score(group, variant)
            metrics = evaluate_ranking(scored["score"], scored["label"], precision_at, top_fraction)
            by_type = evaluate_by_type(scored["score"], scored[typed], top_fraction)
            per_date.append(metrics | {k: v for k, v in by_type.items() if k.startswith("lift")})
        rows.append({"without": name, **pd.DataFrame(per_date).mean(numeric_only=True)})
    result = pd.DataFrame(rows)
    baseline = float(result["lift_top"].iloc[0])  # row 0 = full rule set
    result["lift_top_delta"] = result["lift_top"] - baseline
    return result


def live_scoreboard(
    snapshots: Mapping[date, pd.DataFrame],
    events: pd.DataFrame,
    *,
    today: date,
    horizon_days: int,
    top_fraction: float,
    top_n: int,
    rankings: Sequence[str] = ("score",),
) -> pd.DataFrame:
    """Out-of-sample track record of frozen rankings (data/marts/live): the only test that no model
    choice has seen.
    """
    rows = []
    for as_of, ranking in sorted(snapshots.items()):
        end = min(as_of + timedelta(days=horizon_days), today)
        in_window = events[(events["event_date"] > as_of) & (events["event_date"] <= end)]
        labels = ranking["edinet_code"].isin(set(in_window["subject_edinet_code"])).astype(int)
        base = float(labels.mean()) if len(labels) else 0.0
        k = max(1, math.ceil(len(ranking) * top_fraction))
        for column in [c for c in rankings if c in ranking.columns]:
            ranked = _ranked(ranking[column].fillna(-float("inf")), labels)
            hits_top = int(ranked.head(k).sum())
            rows.append(
                {
                    "as_of": as_of,
                    "ranking": column,
                    "days_elapsed": (today - as_of).days,
                    "window_complete": (today - as_of).days >= horizon_days,
                    "companies": len(ranking),
                    "targets_so_far": int(labels.sum()),
                    "hits_top_fraction": hits_top,
                    "hits_top_n": int(ranked.head(top_n).sum()),
                    "lift_so_far": (hits_top / k) / base if base else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def walk_forward_splits(
    dates: Sequence[date], horizon_days: int, first_test: date
) -> list[tuple[list[date], date]]:
    """Expanding-window splits: each test date T ≥ first_test is predicted by a model trained only
    on dates ≤ T − horizon, so no training label (which looks `horizon` ahead) overlaps the test
    period.
    """
    ordered = sorted(dates)
    splits = []
    for test in ordered:
        if test < first_test:
            continue
        train = [d for d in ordered if d <= test - timedelta(days=horizon_days)]
        if train:
            splits.append((train, test))
    return splits


@dataclass(frozen=True)
class Choice:
    winner: str
    table: pd.DataFrame  # per model: mean lift, dates won vs the baseline, eligible


def choose_default(
    metrics: pd.DataFrame,
    baseline: str,
    min_wins: int,
    tie: float,
    simplicity: Sequence[str],
    *,
    metric: str = "lift_top",
) -> Choice:
    """The pre-registered decision rule: a candidate replaces the baseline only if it beats it on
    `metric` in at least `min_wins` test dates; among eligible candidates within `tie` of the best
    mean, the simplest (earliest in `simplicity`) wins.
    """
    wide = metrics.pivot_table(index="as_of", columns="model", values=metric)
    rows = []
    for model in wide.columns:
        wins = int((wide[model] > wide[baseline]).sum()) if model != baseline else 0
        rows.append(
            {
                "model": model,
                "mean_metric": float(wide[model].mean()),
                "wins_vs_baseline": wins,
                "eligible": model == baseline or wins >= min_wins,
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_metric", ascending=False).reset_index(drop=True)
    eligible = table[table["eligible"]]
    best = float(eligible["mean_metric"].max())
    close = eligible[eligible["mean_metric"] >= best - tie]["model"].tolist()
    rank = {name: i for i, name in enumerate(simplicity)}
    winner = min(close, key=lambda m: rank.get(m, len(rank)))
    return Choice(winner=winner, table=table)
