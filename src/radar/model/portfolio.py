"""Top-N pick returns: what if you had bought the N highest scores?

Data contract: panel with as_of, sec_code, score, label; prices with code, date, close, adj_factor;
topix with date, close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from radar.model.inference import deflated_sharpe, newey_west_se

PRICE_LAG = timedelta(days=1)


class AdjustedPrices:
    """Split-restated closes per code (every close on the latest share basis)."""

    def __init__(self, prices: pd.DataFrame, max_daily_ratio: float | None = None) -> None:
        frame = prices[["code", "date", "close", "adj_factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.sort_values(["code", "date"], kind="mergesort")
        factors = frame["adj_factor"].fillna(1.0).astype(float)
        # A close is multiplied by every factor dated AFTER it (reverse cumulative product).
        later = factors.iloc[::-1].groupby(frame["code"].iloc[::-1]).cumprod().iloc[::-1] / factors
        frame["adjusted"] = frame["close"] * later
        known = frame[frame["close"].notna()]
        self._suspect: dict[str, np.ndarray[Any, Any]] = {}
        if max_daily_ratio is not None:
            previous = known.groupby("code")["adjusted"].shift()
            ratio = known["adjusted"] / previous
            bad = known[(ratio > max_daily_ratio) | (ratio < 1 / max_daily_ratio)]
            self._suspect = {
                str(code): group["date"].to_numpy(dtype="datetime64[D]")
                for code, group in bad.groupby("code")
            }
        self._by_code: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {
            str(code): (
                group["date"].to_numpy(dtype="datetime64[D]"),
                group["adjusted"].to_numpy(dtype=float),
            )
            for code, group in known.groupby("code")
        }

    @property
    def suspect_codes(self) -> set[str]:
        return set(self._suspect)

    def suspect_between(self, code: str, start: date, end: date) -> bool:
        """True if a suspect quote of `code` falls in (start, end]."""
        days = self._suspect.get(str(code))
        if days is None:
            return False
        low, high = np.datetime64(start, "D"), np.datetime64(end, "D")
        return bool(((days > low) & (days <= high)).any())

    def close_on_or_before(self, code: str, day: date) -> tuple[date, float] | None:
        series = self._by_code.get(str(code))
        if series is None:
            return None
        days, closes = series
        i = int(np.searchsorted(days, np.datetime64(day, "D"), side="right")) - 1
        return None if i < 0 else (pd.Timestamp(days[i]).date(), float(closes[i]))


def holding_return(
    prices: AdjustedPrices, code: str, start: date, end: date, stale_days: int
) -> tuple[float, bool] | None:
    """(return, exited early) for a stock bought in the morning of `start` and held to `end`; None
    when there is no recent price to buy at.
    """
    buy = prices.close_on_or_before(code, start - PRICE_LAG)
    if buy is None or (start - buy[0]).days > stale_days or buy[1] <= 0:
        return None
    sell = prices.close_on_or_before(code, end)
    if sell is None or sell[0] < buy[0]:
        return None
    if prices.suspect_between(code, buy[0], sell[0]):
        return None  # an impossible one-day move inside the holding: data error, not a return
    exited = (end - sell[0]).days > stale_days
    return sell[1] / buy[1] - 1, exited


def pick_returns(
    panel: pd.DataFrame,
    prices: AdjustedPrices,
    topix: pd.DataFrame,
    *,
    horizon_days: int,
    sizes: Sequence[int],
    stale_days: int = 10,
) -> pd.DataFrame:
    """One row per (as_of, top N): hit rate, equal-weight returns and benchmarks."""
    index = AdjustedPrices(topix.assign(code="TOPIX", adj_factor=1.0))
    rows = []
    for key, group in panel.groupby("as_of", sort=True):
        as_of = pd.Timestamp(str(key)).date()
        end = as_of + timedelta(days=horizon_days)
        ranked = group.sort_values("score", ascending=False, kind="mergesort")
        held = {
            str(code): holding_return(prices, str(code), as_of, end, stale_days)
            for code in ranked["sec_code"]
        }
        universe = [r[0] for r in held.values() if r is not None]
        bench = holding_return(index, "TOPIX", as_of, end, stale_days)
        for n in sizes:
            top = ranked.head(n)
            results: list[tuple[float, bool]] = [
                r for r in (held[str(c)] for c in top["sec_code"]) if r is not None
            ]
            returns = pd.Series([value for value, _ in results], dtype=float)
            rows.append(
                {
                    "as_of": as_of,
                    "top_n": n,
                    "picks": len(top),
                    "priced": len(results),
                    "hit_rate": float(top["label"].mean()) if len(top) else float("nan"),
                    "mean_return": float(returns.mean()),
                    "median_return": float(returns.median()),
                    "share_positive": float((returns > 0).mean()) if len(returns) else np.nan,
                    "worst_return": float(returns.min()),
                    "exited_early": sum(1 for _, exited in results if exited),
                    "topix_return": bench[0] if bench else float("nan"),
                    "universe_mean_return": float(np.mean(universe)) if universe else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["excess_vs_universe"] = result["mean_return"] - result["universe_mean_return"]
    return result


@dataclass(frozen=True)
class ValueBasketConfig:
    max_market_cap: float
    max_pbr: float
    min_net_cash: float
    # Quality filter (round-2 candidate): operating income / market cap above this. None = off.
    min_operating_yield: float | None = None


def value_basket_score(frame: pd.DataFrame, config: ValueBasketConfig) -> pd.Series:
    """The simple yardstick: small AND below book AND net cash ≥ threshold, ranked by net cash /
    market cap.
    """
    mcap = pd.to_numeric(frame["market_cap_jpy"], errors="coerce")
    pbr = pd.to_numeric(frame["pbr"], errors="coerce")
    cash = pd.to_numeric(frame["net_cash_to_market_cap"], errors="coerce")
    eligible = (
        (mcap <= config.max_market_cap) & (pbr < config.max_pbr) & (cash >= config.min_net_cash)
    )
    if config.min_operating_yield is not None:
        yield_ = pd.to_numeric(frame["operating_income_to_market_cap"], errors="coerce")
        eligible &= yield_ > config.min_operating_yield
    return cash.where(eligible).rename("value_basket")


def value_catalyst_score(frame: pd.DataFrame, config: ValueBasketConfig) -> pd.Series:
    """Round 3 "value + catalyst": the value basket (a base that beat the market on most dates),
    ranked by P(parent buyout) + P(MBO) instead of by cash.
    """
    eligible = value_basket_score(frame, config).notna()
    catalyst = frame[["p_parent", "p_mbo"]].astype(float).sum(axis=1, min_count=1)
    return catalyst.where(eligible).rename("value_catalyst")


def value_momentum_score(
    frame: pd.DataFrame, config: ValueBasketConfig, worst_share: float
) -> pd.Series:
    """Round 3 "value ex worst momentum": the value basket without the companies whose 12-1 momentum
    is in the worst `worst_share` of the whole date's list — cheap stocks that keep falling (value
    traps).
    """
    basket = value_basket_score(frame, config)
    rank = pd.to_numeric(frame["momentum_12_1"], errors="coerce").rank(pct=True)
    falling = rank <= worst_share
    return basket.where(~falling).rename("value_momentum")


def union_rank_score(frame: pd.DataFrame, columns: Mapping[str, str | None]) -> pd.Series:
    """One ranking that takes the top of several lists in turn ("top 7 of v2 + top 7 of MBO"): each
    company gets its BEST percentile rank among `columns`, so the top 2k are the top k of each list
    (fewer if they overlap — the next names fill in).
    """
    ranks = []
    for column, tie_breaker in columns.items():
        value = pd.to_numeric(frame[column], errors="coerce")
        if tie_breaker is not None:  # lexicographic: the tie-breaker only orders equal values
            extra = pd.to_numeric(frame[tie_breaker], errors="coerce").rank(pct=True).fillna(0)
            value = value.rank(method="dense") + extra * 0.5
        ranks.append(value.rank(pct=True))
    return pd.concat(ranks, axis=1).max(axis=1).rename("union_rank")


def sector_neutral(score: pd.Series, sector: pd.Series) -> pd.Series:
    """Score minus its sector median (one screening date)."""
    groups = sector.astype("object").where(sector.notna(), "unknown")
    return score - score.groupby(groups).transform("median")


def summarize_picks(picks: pd.DataFrame) -> pd.DataFrame:
    """Per (model, top_n): the distribution of the portfolio's excess return ACROSS dates — mean,
    median, worst, mean without the best date, dates beating the universe.
    """
    frame = picks.assign(excess=picks["mean_return"] - picks["universe_mean_return"])
    rows = []
    for (model, top_n), group in frame.groupby(["model", "top_n"]):
        excess = group["excess"].dropna().sort_values()
        rows.append(
            {
                "model": model,
                "top_n": top_n,
                "dates": len(excess),
                "mean_excess": float(excess.mean()),
                "median_excess": float(excess.median()),
                "worst_excess": float(excess.min()),
                "mean_excess_without_best": float(excess.iloc[:-1].mean())
                if len(excess) > 1
                else np.nan,
                "dates_beating_universe": int((excess > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def strategy_table(
    picks: pd.DataFrame, top_k: pd.DataFrame, *, nw_lags: int = 3, z: float = 1.645
) -> pd.DataFrame:
    """One row per strategy (model, top_n) for the Portfolio tab: level, spread and luck of its
    12-month return across the walk-forward dates, plus where the return came from.
    """
    frame = picks.assign(
        excess=picks["mean_return"] - picks["universe_mean_return"],
        excess_topix=picks["mean_return"] - picks["topix_return"],
    )
    extra = (
        frame.groupby(["model", "top_n"])
        .agg(
            mean_return=("mean_return", "mean"),
            best_excess=("excess", "max"),
            std_excess=("excess", "std"),
            mean_excess_vs_topix=("excess_topix", "mean"),
            dates_positive=("mean_return", lambda r: int((r > 0).sum())),
            hit_rate=("hit_rate", "mean"),
            share_positive=("share_positive", "mean"),
            worst_stock=("worst_return", "min"),
        )
        .reset_index()
    )
    optional = ["style_excess", "dividend_estimate", "dividend_coverage", "cost", "net_return",
                "net_excess", "capacity_jpy"]  # fmt: skip
    breakdown = ["p_prior_mean", "p_isotonic_mean", "from_offers", "from_others"]
    breakdown += [c for c in optional if c in top_k.columns]
    table = (
        summarize_picks(picks)
        .merge(extra, on=["model", "top_n"])
        .merge(top_k[["model", "top_n", *breakdown]], on=["model", "top_n"], how="left")
    )
    return table.merge(_evidence(frame, nw_lags, z), on=["model", "top_n"])


def _moment(values: pd.Series, name: str, min_len: int, default: float) -> float:
    """Sample skew / excess kurtosis; `default` (the normal's) when too few points to estimate."""
    if len(values) <= min_len:
        return default
    value = pd.to_numeric(getattr(values, name)(), errors="coerce")
    return default if pd.isna(value) else float(value)


def _evidence(frame: pd.DataFrame, nw_lags: int, z: float) -> pd.DataFrame:
    """Per strategy: Newey-West error, t, confidence interval and deflated Sharpe ratio."""
    rows = []
    for (model, top_n), group in frame.groupby(["model", "top_n"]):
        excess = group.sort_values("as_of")["excess"].dropna()
        se = newey_west_se(excess, nw_lags)
        mean, std = float(excess.mean()), float(excess.std(ddof=1))
        rows.append(
            {
                "model": model,
                "top_n": top_n,
                "excess_se_nw": se,
                "t_nw": mean / se if se > 0 else np.nan,
                "ci_low": mean - z * se,
                "ci_high": mean + z * se,
                "sharpe": mean / std if std > 0 else np.nan,
                "skew": _moment(excess, "skew", 2, 0.0),
                "kurtosis": _moment(excess, "kurt", 3, 0.0) + 3,  # pandas gives excess kurtosis
                "n_obs": len(excess),
            }
        )
    table = pd.DataFrame(rows)
    sharpes = table["sharpe"].dropna()
    variance = float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0
    table["deflated_sharpe"] = [
        deflated_sharpe(
            r["sharpe"], n_obs=r["n_obs"], n_trials=len(sharpes), trials_variance=variance,
            skew=0.0 if pd.isna(r["skew"]) else r["skew"],
            kurtosis=3.0 if pd.isna(r["kurtosis"]) else r["kurtosis"],
        )
        for r in table.to_dict("records")
    ]  # fmt: skip
    return table.drop(columns=["skew", "kurtosis", "n_obs"])


def pick_lenses(
    table: pd.DataFrame, lenses: Mapping[str, Mapping[str, str]]
) -> dict[str, tuple[str, int]]:
    """The strategy each lens prefers: the highest `metric`, ties to the lowest `tie_break`."""
    chosen = {}
    for name, lens in lenses.items():
        known = table.dropna(subset=[lens["metric"]])
        if known.empty:
            continue
        best = known.sort_values(
            [lens["metric"], lens["tie_break"]], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        chosen[name] = (str(best["model"]), int(best["top_n"]))
    return chosen


@dataclass(frozen=True)
class TradingCosts:
    """Round-trip trading cost by liquidity, and the capacity rule (config/backtest.yaml
    `cost_tiers`, config/allocation.yaml participation / max_days).
    """

    tiers: tuple[tuple[float | None, float], ...]  # (max median ¥/day or None = above, cost)
    participation: float  # max share of a day's median trading value we may be
    days: int  # days allowed to build (or exit) a position

    def cost(self, trading_value: float | None) -> float:
        """Unknown liquidity pays the highest tier: an untradeable name is not free to trade."""
        if trading_value is None or pd.isna(trading_value):
            return max(cost for _, cost in self.tiers)
        for limit, cost in self.tiers:
            if limit is None or trading_value < limit:
                return cost
        return self.tiers[-1][1]


def top_k_breakdown(
    scored: pd.DataFrame,
    rankings: Sequence[str],
    sizes: Sequence[int],
    costs: TradingCosts | None = None,
) -> pd.DataFrame:
    """Small portfolios per ranking, pooled over the walk-forward test dates: is the predicted P
    right AT THE TOP, where does the return come from, and what is left after costs?

    Columns in: as_of, label, realized_return (NaN = no price), one column per ranking; optional
    p_prior / p_isotonic, style_return, dividend_yield, median_trading_value_jpy. Unknown scores
    rank last.
    """
    priced_rows = scored[scored["realized_return"].notna()]
    universe = priced_rows.groupby("as_of")["realized_return"].mean()
    dividends = (
        priced_rows["dividend_yield"].fillna(0.0).groupby(priced_rows["as_of"]).mean()
        if "dividend_yield" in scored.columns
        else None
    )
    rows = []
    for model in [r for r in rankings if r in scored.columns]:
        for n in sizes:
            top = pd.concat(
                group.sort_values(
                    model, ascending=False, kind="mergesort", na_position="last"
                ).head(n)
                for _, group in scored.groupby("as_of", sort=True)
            )
            row = {"model": model, "top_n": n, **_top_k_row(top, universe)}
            rows.append({**row, **_net_row(top, n, row, dividends, costs)})
    return pd.DataFrame(rows)


def _top_k_row(top: pd.DataFrame, universe: pd.Series) -> dict[str, Any]:
    priced = top[top["realized_return"].notna()]
    hits = priced[priced["label"] == 1]["realized_return"]
    misses = priced[priced["label"] != 1]["realized_return"]
    share = len(hits) / len(priced) if len(priced) else np.nan
    from_offers = share * float(hits.mean()) if len(hits) else 0.0
    from_others = (1 - share) * float(misses.mean()) if len(misses) else 0.0
    base = float(universe.reindex(top["as_of"].unique()).mean())
    mean = from_offers + from_others if len(priced) else np.nan

    def average(column: str) -> float:
        return float(top[column].mean()) if column in top.columns else np.nan

    style = (
        float((priced["realized_return"] - priced["style_return"]).mean())
        if "style_return" in priced.columns
        else np.nan
    )
    return {
        "dates": top["as_of"].nunique(),
        "picks": len(top),
        "priced": len(priced),
        "hits": int(top["label"].sum()),
        "hit_rate": float(top["label"].mean()),
        "p_prior_mean": average("p_prior"),
        "p_isotonic_mean": average("p_isotonic"),
        "return_hits": float(hits.mean()) if len(hits) else np.nan,
        "return_misses": float(misses.mean()) if len(misses) else np.nan,
        "from_offers": from_offers,
        "from_others": from_others,
        "mean_return": mean,
        "universe_return": base,
        "excess": mean - base,
        "style_excess": style,
    }


def _net_row(
    top: pd.DataFrame,
    n: int,
    gross: Mapping[str, Any],
    dividends: pd.Series | None,
    costs: TradingCosts | None,
) -> dict[str, float]:
    """Dividends, costs, net return and capacity of one strategy (NaN without `costs`)."""
    empty = dict.fromkeys(
        ("dividend_estimate", "dividend_coverage", "cost", "net_return", "net_excess",
         "capacity_jpy"),
        np.nan,
    )  # fmt: skip
    if costs is None or "median_trading_value_jpy" not in top.columns:
        return empty
    priced = top[top["realized_return"].notna()]
    known = priced["dividend_yield"] if "dividend_yield" in priced.columns else pd.Series()
    dividend = float(known.fillna(0.0).mean()) if len(priced) else np.nan
    cost = float(priced["median_trading_value_jpy"].map(costs.cost).mean())
    net = float(gross["mean_return"]) + (0.0 if pd.isna(dividend) else dividend) - cost
    base = gross["universe_return"] + (
        float(dividends.reindex(top["as_of"].unique()).mean()) if dividends is not None else 0.0
    )
    per_date = top.groupby("as_of")["median_trading_value_jpy"].min()
    return {
        "dividend_estimate": dividend,
        "dividend_coverage": float(known.notna().mean()) if len(known) else np.nan,
        "cost": cost,
        "net_return": net,
        "net_excess": net - base,
        "capacity_jpy": float((per_date * costs.participation * costs.days * n).median()),
    }


def live_returns(
    snapshots: Mapping[date, pd.DataFrame],
    rankings: Sequence[str],
    prices: AdjustedPrices,
    *,
    today: date,
    top_n: int,
    stale_days: int,
) -> pd.DataFrame:
    """Return so far of each frozen ranking's top N vs all ranked companies: bought in the morning
    of the snapshot day, valued at the last close ≤ today.
    """
    rows = []
    for as_of, snapshot in sorted(snapshots.items()):
        held = {
            str(c): holding_return(prices, str(c), as_of, today, stale_days)
            for c in snapshot["sec_code"]
        }
        universe = [r[0] for r in held.values() if r is not None]
        base = float(np.mean(universe)) if universe else float("nan")
        for column in [c for c in rankings if c in snapshot.columns]:
            ranked = snapshot.sort_values(
                column, ascending=False, kind="mergesort", na_position="last"
            )
            top = [held[str(c)] for c in ranked["sec_code"].head(top_n)]
            values = [r[0] for r in top if r is not None]
            mean = float(np.mean(values)) if values else float("nan")
            rows.append(
                {
                    "as_of": as_of,
                    "ranking": column,
                    "top_n": top_n,
                    "top_return_so_far": mean,
                    "universe_return_so_far": base,
                    "excess_so_far": mean - base,
                }
            )
    return pd.DataFrame(rows)
