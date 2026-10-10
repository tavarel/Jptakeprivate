"""Dashboard tab "Portfolio construction": which ranking and how many names, judged on the
walk-forward history, then today's list sized by liquidity.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from radar import pipeline
from radar.config import get_settings, load_yaml
from radar.model.portfolio import pick_lenses, strategy_table
from radar.report.views import help_texts, live_column, skill_verdict, universe_share_label

# dataviz default palette: categorical slots 1–3, diverging blue ↔ red with a gray midpoint.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, MID_GRAY, MUTED = "#e34948", "#f0efec", "#898781"

INTRO = """
**Which ranking, and how many names?** Each row of the history is a *strategy*: buy the top N
of one ranking on a screening date, equal weights, hold 12 months. Seven walk-forward dates
(2024-03 → 2025-09), every ranking trained only on data at least a year older. **Excess** =
the portfolio's return minus the average of all listed companies on the same dates.
Two lenses, fixed in `config/report.yaml` before looking: **return** = best mean excess
*without the luckiest date* (a mean carried by one penny stock is not a strategy); **risk** =
best *worst date*. Seven overlapping dates in a rising market are a hint, not proof; the live
scoreboard is the honest test. Price return only: dividends (+2–3%/yr) and costs (−1 to −1.5
pp/yr) are not included. Not investment advice.
"""
COLUMN_HELP = {
    "model": "The ranking: rules (v0, v2), models, or the simple value basket.",
    "top_n": "Number of names bought on each date (equal weights).",
    "dates": "Walk-forward test dates with a priced portfolio.",
    "mean_excess": "Mean over the dates of (portfolio return − all-companies return).",
    "median_excess": "Middle date: less sensitive to one extreme date than the mean.",
    "worst_excess": "The worst date's excess: the risk lens. Above −5 pp = mild; −5 to −15 = a "
    "bad year you must be able to sit through; below −15 = painful.",
    "best_excess": "The best date's excess.",
    "mean_excess_without_best": "Mean excess with the best date removed: the return lens. "
    "Positive = the edge does not hang on one lucky year; near 0 or negative = it did.",
    "std_excess": "Spread of the excess across dates (lower = steadier).",
    "dates_beating_universe": "Dates on which the portfolio beat all listed companies. 6–7 of 7 "
    "= consistent; 4–5 = coin-flip territory; 3 or fewer = worse than holding everything.",
    "dates_positive": "Dates on which the portfolio made money (return > 0).",
    "mean_return": "Mean 12-month return of the portfolio (not excess).",
    "mean_excess_vs_topix": "Mean return minus TOPIX (large-cap weighted, so a harder bar).",
    "hit_rate": "Share of the picks that received a take-private offer within 12 months.",
    "share_positive": "Average share of the picks that went up.",
    "worst_stock": "The worst single holding over all dates.",
    "p_prior_mean": "Mean model P(offer) of the picks, prior-corrected. Compare with hit_rate.",
    "p_isotonic_mean": "Mean model P(offer) of the picks, isotonic-calibrated.",
    "from_offers": "Part of the mean return earned by picks that received an offer.",
    "from_others": "Part earned by the other picks. Usually most of the result.",
    "excess_se_nw": "Newey-West error of the mean excess: allows for the overlapping 12-month "
    "windows (each date shares 9 months with its neighbours), so it is wider than a naive one.",
    "t_nw": "Mean excess ÷ its Newey-West error. Below ~1.7 the mean is within noise.",
    "ci_low": "Lower end of the 90% interval of the mean excess.",
    "ci_high": "Upper end of the 90% interval of the mean excess.",
    "sharpe": "Mean excess ÷ its spread across dates (per date, not annualized).",
    "deflated_sharpe": "Probability that this strategy beats luck, AFTER counting every "
    "strategy in this table as a try (Bailey & López de Prado). Above 0.95 is strong evidence.",
    "style_excess": "Return minus the average of companies of the same size × PBR quintile: "
    "what remains beyond 'small and cheap' (size + value).",
    "dividend_estimate": "Last year's dividend yield of the picks (有報 1株当たり配当額); unknown "
    "counts as 0.",
    "dividend_coverage": "Share of picks with a known dividend.",
    "cost": "Estimated round-trip cost (spread, commission, impact) by liquidity.",
    "net_return": "Mean return + dividends − costs (estimate). Compare with all companies' "
    "return plus their dividends (net_excess), not with zero.",
    "net_excess": "Net return minus all companies' return with their dividends. Above +3 pp a "
    "year = worth the effort; 0–3 = marginal after taxes and your time; negative = just buy all.",
    "capacity_jpy": "Largest equal-weight portfolio every pick can absorb at 10% of its daily "
    "trading value over 5 days (median over dates). Above it, the backtest is not executable.",
}


@st.fragment  # a click here re-runs only this tab, not the whole dashboard
def portfolio_tab(marts_dir: Path) -> None:
    config = load_yaml("report")["portfolio"]
    ruleset = str(config["ruleset"])
    picks_path = marts_dir / f"walk_forward_picks_{ruleset}.parquet"
    if not picks_path.exists():
        st.info("No walk-forward yet: run `uv run radar walkforward`.")
        return
    st.caption(
        "Buy the top N of a ranking, equal weights, hold 12 months — judged on 7 walk-forward "
        "dates against the average of all listed companies. Research, not advice."
    )
    with st.expander("Method"):
        st.markdown(INTRO)
    top_k_path = marts_dir / f"walk_forward_top_k_{ruleset}.parquet"
    version = (_mtime(picks_path), _mtime(top_k_path))  # new walk-forward → fresh numbers
    table = _strategies(marts_dir, ruleset, list(config["models"]), version)
    companies = _mean_companies(marts_dir, ruleset, version)
    lenses = pick_lenses(table, config["lenses"])
    _lens_summary(lenses, table, companies)
    model, top_n = _selection(table, lenses, companies)
    chosen = table[(table["model"] == model) & (table["top_n"] == top_n)].iloc[0]
    _metrics(chosen)
    left, right = st.columns(2)
    with left:
        _risk_return_chart(table, (model, top_n), lenses, companies)
    with right:
        _heatmap(table, companies)
    left, right = st.columns(2)
    with left:
        _per_date_chart(pd.read_parquet(picks_path), model, top_n)
    with right:
        _decomposition_chart(table, top_n)
    shown = table.sort_values("mean_excess_without_best", ascending=False)
    with st.expander(f"All {len(shown)} strategies (hover a column name for how to read it)"):
        st.dataframe(shown, hide_index=True, width="stretch", column_config=_column_help(shown))
    _live(marts_dir, config, model)
    _todays_portfolio(marts_dir, config, model, top_n, ruleset)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=4)
def _columns(path: Path, mtime: float) -> list[str]:
    """Column names of a screen, read once per file version (not on every click)."""
    return [str(c) for c in pd.read_parquet(path).columns]


@st.cache_data(show_spinner=False, max_entries=4)
def _strategies(
    marts_dir: Path, ruleset: str, models: list[str], version: tuple[float, float]
) -> pd.DataFrame:
    """`version` (file times) is only a cache key: a rerun walk-forward is read again."""
    picks = pd.read_parquet(marts_dir / f"walk_forward_picks_{ruleset}.parquet")
    top_k_path = marts_dir / f"walk_forward_top_k_{ruleset}.parquet"
    columns = ["model", "top_n", "p_prior_mean", "p_isotonic_mean", "from_offers", "from_others"]
    top_k = pd.read_parquet(top_k_path) if top_k_path.exists() else pd.DataFrame(columns=columns)
    inference = load_yaml("backtest")["inference"]
    table = strategy_table(
        picks[picks["model"].isin(models)], top_k,
        nw_lags=int(inference["nw_lags"]), z=float(inference["ci_z"]),
    )  # fmt: skip
    order = {name: i for i, name in enumerate(models)}
    return table.sort_values(
        ["model", "top_n"], key=lambda c: c.map(order) if c.name == "model" else c
    )


@st.cache_data(show_spinner=False, max_entries=4)
def _mean_companies(marts_dir: Path, ruleset: str, version: tuple[float, float]) -> float | None:
    path = marts_dir / f"walk_forward_{ruleset}.parquet"
    if not path.exists():
        return None
    return float(pd.read_parquet(path, columns=["companies"])["companies"].mean())


DISPLAY = {
    "value_basket": "Value basket",
    "value_basket_quality": "Value basket, profitable",
    "value_momentum": "Value + momentum",
    "value_catalyst": "Value + catalyst",
    "v0": "Rule baseline (v0)",
    "v2": "Offer likelihood (v2)",
    "v2_by_type": "Offer likelihood per type",
    "hybrid": "Hybrid (rules + parent model)",
    "logistic_by_type": "Logistic per type",
    "expected_value": "Expected value",
    "return_model": "Return model",
    "return_model_sector_neutral": "Return model, sector-neutral",
    "combo_v2_mbo": "Offer + MBO combined",
}


def _name(model: str) -> str:
    """Readable name of a ranking key (the keys stay in the data and the config)."""
    return DISPLAY.get(model, model)


def _label(model: str, top_n: int, companies: float | None) -> str:
    return f"{_name(model)} · {universe_share_label(top_n, companies)}"


def _lens_summary(
    lenses: dict[str, tuple[str, int]], table: pd.DataFrame, companies: float | None
) -> None:
    cards = st.columns(len(lenses) or 1)
    titles = {"return": "Best robust return", "risk": "Least bad worst date"}
    for card, (name, (model, top_n)) in zip(cards, lenses.items(), strict=False):
        row = table[(table["model"] == model) & (table["top_n"] == top_n)].iloc[0]
        card.metric(
            titles.get(name, name),
            _label(model, top_n, companies),
            f"{row['mean_excess_without_best']:+.1%} without best · "
            f"worst {row['worst_excess']:+.1%}",
            delta_color="off",
            delta_arrow="off",
            help=f"Lens `{name}` of config/report.yaml, fixed before looking at this table.",
        )


def _selection(
    table: pd.DataFrame, lenses: dict[str, tuple[str, int]], companies: float | None
) -> tuple[str, int]:
    models = list(dict.fromkeys(table["model"]))
    start_model, start_n = lenses.get("return", (models[0], int(table["top_n"].min())))
    left, right = st.columns(2)
    model = left.selectbox(
        "Ranking", models, index=models.index(start_model), key="pf_model", format_func=_name
    )
    sizes = sorted(int(n) for n in table.loc[table["model"] == model, "top_n"].unique())
    top_n = right.select_slider(
        "Portfolio size",
        options=sizes,
        value=start_n if start_n in sizes else sizes[0],
        format_func=lambda n: universe_share_label(int(n), companies),
        key="pf_size",
    )
    if isinstance(top_n, tuple):  # a range slider returns (low, high); this one is single
        top_n = top_n[0]
    return model, int(top_n)


SKILL_HELP = (
    "How likely is it that this edge is real and not luck? 60 strategies (10 rankings × 6 sizes) "
    "were tried on only 7 dates, and the best of many tries always looks good by chance — like "
    "the luckiest of 60 coin-flippers. This number (the deflated Sharpe ratio, Bailey & López de "
    "Prado 2014) is the probability that the true average excess is above zero AFTER allowing "
    "for the number of tries, the few dates and the extreme dates. Read it as: 95% or more = "
    "strong evidence; 50–95% = a hint worth watching live; below 50% = what luck alone produces."
)


COLUMN_HELP["deflated_sharpe"] = SKILL_HELP  # same words in the table as on the card


def _metrics(row: pd.Series) -> None:
    """Three rows of four short labels (long labels were cut off with "…" in narrow columns)."""
    dates = int(row["dates"])
    st.caption("**Return** — 12 months, minus the average of all listed companies")
    cells = st.columns(4)
    cells[0].metric("Mean excess", _signed(row["mean_excess"]), help=COLUMN_HELP["mean_excess"])
    cells[1].metric(
        "Excl. best date",
        _signed(row["mean_excess_without_best"]),
        help=COLUMN_HELP["mean_excess_without_best"],
    )
    cells[2].metric("Worst date", _signed(row["worst_excess"]), help=COLUMN_HELP["worst_excess"])
    cells[3].metric(
        "Avg 12-month return",
        _signed(row["mean_return"]),
        help=f"Not excess. vs TOPIX {row['mean_excess_vs_topix']:+.1%}.",
    )
    st.caption("**Evidence** — is it more than luck, and more than 'small and cheap'?")
    cells = st.columns(4)
    cells[0].metric(
        "Dates beating market",
        f"{int(row['dates_beating_universe'])} of {dates}",
        help=f"Dates on which it beat all listed companies. Made money on "
        f"{int(row['dates_positive'])} of {dates} dates.",
    )
    cells[1].metric(
        "90% range of excess",
        f"{row['ci_low']:+.0%} to {row['ci_high']:+.0%}",
        help=COLUMN_HELP["excess_se_nw"] + " A range that contains 0 = no proof of an edge.",
    )
    skill = row.get("deflated_sharpe")
    cells[2].metric("Chance it's skill", _percent(skill), help=SKILL_HELP)
    cells[3].metric(
        "Beyond size + value", _signed(row.get("style_excess")), help=COLUMN_HELP["style_excess"]
    )
    st.caption(f"Chance it's skill: **{_percent(skill)}** → {skill_verdict(skill)}")
    st.caption("**Practice** — after costs, how much money fits, how often an offer came")
    cells = st.columns(4)
    cells[0].metric(
        "Net return (est.)",
        _signed(row.get("net_return")),
        help=COLUMN_HELP["net_return"]
        + f" Dividends {_signed(row.get('dividend_estimate'))},"
        + f" costs −{_percent(row.get('cost'))}.",
    )
    capacity = row.get("capacity_jpy")
    cells[1].metric(
        "Capacity",
        "—" if capacity is None or pd.isna(capacity) else f"¥{capacity / 1e6:,.0f}m",
        help=COLUMN_HELP["capacity_jpy"],
    )
    predicted = row.get("p_isotonic_mean")
    cells[2].metric(
        "Got an offer",
        f"{row['hit_rate']:.1%}",
        help="Share of picks with a take-private offer within 12 months (base rate ≈ 2.2%)."
        + ("" if pd.isna(predicted) else f" The model predicted {predicted:.1%} (isotonic)."),
    )
    cells[3].metric("Cost (round trip)", _percent(row.get("cost")), help=COLUMN_HELP["cost"])
    if pd.isna(row.get("net_return")):
        st.caption(
            "Net return, capacity and 'beyond size + value' appear after the next "
            "`uv run radar walkforward`."
        )


def _signed(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):+.1%}"


def _percent(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}".replace(".0%", "%")


def _risk_return_chart(
    table: pd.DataFrame,
    chosen: tuple[str, int],
    lenses: dict[str, tuple[str, int]],
    companies: float | None,
) -> None:
    st.markdown(
        "**Risk × return** — up = better robust return, right = safer (or bigger)",
        help="One dot per strategy (ranking × size). Blue = your choice; orange = the lenses' "
        "picks. Hover a dot for its numbers.",
    )
    axis = st.radio(
        "Horizontal axis", ["Worst date", "Capacity"], horizontal=True, key="pf_axis",
        help="Capacity: how much money the strategy could hold without moving the prices.",
    )  # fmt: skip
    x = (
        alt.X("worst_excess:Q", title="Worst date (excess)", axis=alt.Axis(format="+%"))
        if axis == "Worst date"
        else alt.X("capacity_jpy:Q", title="Capacity (yen, log)", scale=alt.Scale(type="log"))
    )
    frame = table.assign(
        strategy=[
            _label(m, int(n), companies)
            for m, n in zip(table["model"], table["top_n"], strict=True)
        ],
        role=[
            _role((m, int(n)), chosen, lenses)
            for m, n in zip(table["model"], table["top_n"], strict=True)
        ],
    )
    if axis == "Capacity":
        frame = frame[frame["capacity_jpy"] > 0]
    base = alt.Chart(frame).encode(
        x=x,
        y=alt.Y(
            "mean_excess_without_best:Q",
            title="Mean excess without best date",
            axis=alt.Axis(format="+%"),
        ),
    )
    dots = base.mark_circle(stroke="white", strokeWidth=2, opacity=0.9).encode(
        color=alt.Color(
            "role:N",
            scale=alt.Scale(
                domain=["your choice", "lens pick", "other"], range=[BLUE, ORANGE, MUTED]
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        size=alt.condition(alt.datum.role == "other", alt.value(70), alt.value(180)),
        tooltip=_tooltip(),
    )
    labels = (
        base.transform_filter(alt.datum.role != "other")
        .mark_text(align="left", dx=9, dy=-6, fontSize=11, color="#52514e")
        .encode(text="strategy:N")
    )
    zero = alt.Chart(pd.DataFrame({"v": [0.0]}))
    rules = zero.mark_rule(color="#c3c2b7").encode(y="v:Q")
    if axis == "Worst date":
        rules += zero.mark_rule(color="#c3c2b7").encode(x="v:Q")
    st.altair_chart((rules + dots + labels).properties(height=360), width="stretch")


def _role(key: tuple[str, int], chosen: tuple[str, int], lenses: dict[str, tuple[str, int]]) -> str:
    if key == chosen:
        return "your choice"
    return "lens pick" if key in lenses.values() else "other"


def _tooltip() -> list[alt.Tooltip]:
    return [
        alt.Tooltip("strategy:N", title="Strategy"),
        alt.Tooltip("mean_excess:Q", title="Mean excess", format="+.1%"),
        alt.Tooltip("mean_excess_without_best:Q", title="Without best date", format="+.1%"),
        alt.Tooltip("worst_excess:Q", title="Worst date", format="+.1%"),
        alt.Tooltip("dates_beating_universe:Q", title="Dates beating all companies"),
        alt.Tooltip("hit_rate:Q", title="Got an offer", format=".1%"),
        alt.Tooltip("deflated_sharpe:Q", title="Better than luck", format=".0%"),
        alt.Tooltip("capacity_jpy:Q", title="Capacity (yen)", format=",.0f"),
    ]


def _heatmap(table: pd.DataFrame, companies: float | None) -> None:
    st.markdown(
        "**Where the edge is** — mean excess without the best date",
        help="Blue = beat all listed companies even without its luckiest date; red = lost to "
        "them; gray ≈ no difference.",
    )
    frame = table.assign(
        size=[universe_share_label(int(n), companies) for n in table["top_n"]],
        ranking=table["model"].map(_name),
    )
    sizes = [universe_share_label(int(n), companies) for n in sorted(table["top_n"].unique())]
    limit = float(table["mean_excess_without_best"].abs().max() or 0.01)
    base = alt.Chart(frame).encode(
        x=alt.X("size:N", sort=sizes, title="Portfolio size"),
        y=alt.Y("ranking:N", sort=[_name(m) for m in dict.fromkeys(table["model"])], title=None),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color(
            "mean_excess_without_best:Q",
            scale=alt.Scale(domain=[-limit, 0, limit], range=[RED, MID_GRAY, BLUE]),
            legend=alt.Legend(title="Excess", format="+%", orient="bottom"),
        ),
        tooltip=[alt.Tooltip("ranking:N"), alt.Tooltip("size:N"), *_tooltip()[1:]],
    )
    text = base.mark_text(fontSize=11, color="#0b0b0b").encode(
        text=alt.Text("mean_excess_without_best:Q", format="+.0%")
    )
    st.altair_chart((cells + text).properties(height=360), width="stretch")


def _per_date_chart(picks: pd.DataFrame, model: str, top_n: int) -> None:
    st.markdown(
        f"**Each test date** — {_name(model)} top {top_n} vs all companies vs TOPIX",
        help="12-month return bought on each date. A strategy you can live with wins most dates, "
        "not one.",
    )
    rows = picks[(picks["model"] == model) & (picks["top_n"] == top_n)]
    frame = rows.melt(
        id_vars="as_of",
        value_vars=["mean_return", "universe_mean_return", "topix_return"],
        var_name="series",
        value_name="return",
    ).replace({"series": {"mean_return": "Portfolio", "universe_mean_return": "All companies",
                          "topix_return": "TOPIX"}})  # fmt: skip
    frame["as_of"] = frame["as_of"].astype(str)
    order = ["Portfolio", "All companies", "TOPIX"]
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("as_of:N", title="Bought on"),
        xOffset=alt.XOffset("series:N", sort=order),
        y=alt.Y("return:Q", title="12-month return", axis=alt.Axis(format="+%")),
        color=alt.Color(
            "series:N", sort=order, scale=alt.Scale(domain=order, range=[BLUE, ORANGE, AQUA]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[alt.Tooltip("as_of:N", title="Bought on"), alt.Tooltip("series:N"),
                 alt.Tooltip("return:Q", format="+.1%")],
    )  # fmt: skip
    st.altair_chart(chart.properties(height=320), width="stretch")


def _decomposition_chart(table: pd.DataFrame, top_n: int) -> None:
    st.markdown(
        f"**Where the return comes from** — top {top_n} of each ranking",
        help="Mean 12-month return of the picks = part from those that got an offer + part from "
        "the rest. Finding offers is not enough if the other picks lose.",
    )
    rows = table[(table["top_n"] == top_n) & table["from_offers"].notna()]
    if rows.empty:
        st.caption("No breakdown for this size yet: rerun `radar walkforward`.")
        return
    frame = (
        rows.assign(ranking=rows["model"].map(_name))
        .melt(
            id_vars="ranking",
            value_vars=["from_offers", "from_others"],
            var_name="part",
            value_name="v",
        )
        .replace({"part": {"from_offers": "Picks that got an offer", "from_others": "Other picks"}})
    )
    order = ["Picks that got an offer", "Other picks"]
    chart = alt.Chart(frame).mark_bar(stroke="white", strokeWidth=2).encode(
        y=alt.Y("ranking:N", sort=[_name(m) for m in dict.fromkeys(rows["model"])], title=None),
        x=alt.X(
            "v:Q", stack="zero", title="Contribution to mean return", axis=alt.Axis(format="+%")
        ),
        color=alt.Color(
            "part:N", sort=order, scale=alt.Scale(domain=order, range=[ORANGE, BLUE]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("ranking:N"), alt.Tooltip("part:N"), alt.Tooltip("v:Q", format="+.1%")
        ],
    )  # fmt: skip
    st.altair_chart(chart.properties(height=320), width="stretch")


def _column_help(frame: pd.DataFrame) -> dict[str, Any]:
    texts = help_texts([str(c) for c in frame.columns], COLUMN_HELP)
    return {column: st.column_config.Column(help=text) for column, text in texts.items()}


def _live(marts_dir: Path, config: dict[str, Any], model: str) -> None:
    path = marts_dir / "live_scoreboard.parquet"
    column = config["models"].get(model)
    st.subheader(
        "Live record of this ranking",
        help="Rankings frozen each day by `radar update` and judged on what came after: the "
        "only result no choice has seen. Refresh with `radar scoreboard`.",
    )
    if column is None:
        st.caption(f"{_name(model)} has no live version (backtest history only).")
        return
    board = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    rows = board[board["ranking"] == column] if "ranking" in board.columns else board
    if rows.empty:
        st.caption(f"No frozen ranking of `{column}` yet: it starts with the next `radar update`.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")


def _todays_portfolio(
    marts_dir: Path, config: dict[str, Any], model: str, top_n: int, ruleset: str
) -> None:
    st.subheader(
        "Today's portfolio for this strategy",
        help="Equal weights, each position capped by liquidity (10% of a day's median trading "
        "value over 5 days) and rounded down to 100-share lots (config/allocation.yaml).",
    )
    files = sorted(marts_dir.glob("scores_*.parquet"))
    if not files:
        st.caption("No screen yet: run `radar update`.")
        return
    as_of = date.fromisoformat(files[-1].stem.removeprefix("scores_"))
    columns = _columns(files[-1], _mtime(files[-1]))
    column = live_column(model, config["models"], columns)
    if column is None:
        st.caption(f"{_name(model)} is not in the screen of {as_of} (no live column).")
        return
    capital = st.number_input(
        "Capital (yen)", min_value=100_000, step=1_000_000,
        value=int(config["default_capital"]), key="pf_capital",
    )  # fmt: skip
    if not st.button(f"Build {_name(model)} · top {top_n} for {as_of}", key="pf_build"):
        st.caption("Loads the price history (a few seconds).")
        return
    plan, invested, _ = _plan(
        as_of, column, model, top_n, capital=float(capital), ruleset=ruleset,
        version=(_mtime(files[-1]), _mtime(marts_dir / f"walk_forward_picks_{ruleset}.parquet")),
    )  # fmt: skip
    _plan_view(plan, invested, as_of, model, top_n)


@st.cache_data(show_spinner="Sizing positions…")
def _plan(
    as_of: date,
    column: str,
    model: str,
    top_n: int,
    *,
    capital: float,
    ruleset: str,
    version: tuple[float, float],
) -> tuple[pd.DataFrame, float | None, pd.Series]:
    return pipeline.run_allocation(
        get_settings(), as_of, ranking=column, model=model, top=top_n, capital=capital,
        ruleset=ruleset, size_by_kelly=False,
    )  # fmt: skip


def _plan_view(
    plan: pd.DataFrame, invested: float | None, as_of: date, model: str, top_n: int
) -> None:
    kelly = "unknown (too few dates)" if invested is None else f"{invested:.0%}"
    st.caption(
        f"Kelly fraction (1/3 Kelly on the EXCESS over all companies, mean lowered by one "
        f"Newey-West error): {kelly} of capital. 0% = the history does not prove an edge over "
        "simply holding the market. The plan below is sized with equal weights on the whole "
        "capital, capped by liquidity; caps left in cash are not redistributed."
    )
    shown = plan.rename(columns={"position_jpy": "position (yen)", "reason": "binding cap"})
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(
        f"Invested ¥{plan['position_jpy'].sum():,.0f} in {int((plan['shares'] > 0).sum())} "
        "names; the rest stays in cash."
    )
    st.download_button(
        "Download this portfolio (CSV)",
        plan.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"portfolio_{model}_top{top_n}_{as_of}.csv",
        mime="text/csv",
        key="pf_download",
    )
