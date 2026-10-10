"""Command-line interface."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from radar import pipeline
from radar.config import get_settings, load_yaml
from radar.ingest.edinet import DocType, EdinetClient, FetchType
from radar.ingest.http import HttpClient
from radar.ingest.jquants import JQuantsClient
from radar.log import setup_logging
from radar.parse.quality import quality_report, sanity_checks
from radar.parse.xbrl_csv import read_facts_zip
from radar.report.views import public_reason
from radar.storage.paths import edinet_doc_path

app = typer.Typer(help="Japan take-private radar", no_args_is_help=True)
console = Console()

DateOpt = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])]
RulesOpt = Annotated[str, typer.Option(help="rule file in config/, e.g. scoring_v1")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


def _edinet_client() -> EdinetClient:
    settings = get_settings()
    key = settings.edinet_api_key.get_secret_value() if settings.edinet_api_key else ""
    http = HttpClient(settings.requests_per_second, settings.http_timeout_s)
    return EdinetClient(key, settings.raw_dir, settings.edinet_base_url, http)


def _jquants_client() -> JQuantsClient:
    settings = get_settings()
    key = settings.jquants_api_key.get_secret_value() if settings.jquants_api_key else ""
    rate = float(load_yaml("jquants")["requests_per_minute"])
    return JQuantsClient(key, settings.raw_dir, settings.jquants_base_url, requests_per_minute=rate)


@app.command()
def doclists(start: DateOpt, end: DateOpt) -> None:
    """Fetch EDINET daily document lists and update the document index."""
    settings = get_settings()
    n = pipeline.sync_document_index(_edinet_client(), settings, start.date(), end.date())
    console.print(f"[green]{n} documents indexed[/green] ({start.date()} → {end.date()})")


@app.command()
def download(
    doc_type: Annotated[list[str], typer.Option(help="docTypeCode, repeatable")] = [  # noqa: B006
        DocType.ANNUAL_REPORT.value
    ],
    start: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    end: Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
    limit: Annotated[
        int | None, typer.Option(help="max documents (use small numbers first)")
    ] = None,
    include_unlisted: Annotated[
        bool, typer.Option(help="also filers without a securities code (funds filing 350s)")
    ] = False,
) -> None:
    """Download type=5 CSV zips for indexed documents (listed filers only, unless told)."""
    settings = get_settings()
    docs = pipeline.select_documents(
        pipeline.load_documents(settings),
        doc_type,
        start.date() if start else None,
        end.date() if end else None,
        only_listed=not include_unlisted,
    )
    console.print(f"{len(docs)} documents match; downloading {limit or len(docs)}")
    paths = pipeline.download_documents(_edinet_client(), docs, limit)
    console.print(f"[green]{len(paths)} files available[/green] under {settings.raw_dir}")


@app.command()
def prices(start: DateOpt, end: DateOpt) -> None:
    """Fetch J-Quants daily bars (all stocks, one call per weekday) into staging/prices."""
    days = pipeline.sync_prices(_jquants_client(), get_settings(), start.date(), end.date())
    console.print(f"[green]{days} weekdays synced[/green] ({start.date()} → {end.date()})")


@app.command()
def listings(start: DateOpt, end: DateOpt) -> None:
    """Fetch the J-Quants listed-company master at each month-end into staging/listings."""
    months = pipeline.sync_listings(_jquants_client(), get_settings(), start.date(), end.date())
    console.print(f"[green]{months} month-end snapshots synced[/green]")


@app.command("capital-plans")
def capital_plans() -> None:
    """Load the JPX capital-plan Excel lists saved in data/manual/capital_plans."""
    lists = pipeline.sync_capital_plans(get_settings())
    months = lists["list_month"].nunique()
    if lists.empty:
        console.print(
            "[yellow]No list found — save the JPX Excel in data/manual/capital_plans[/yellow]"
        )
    else:
        console.print(f"[green]{months} monthly lists, {len(lists)} rows[/green]")


@app.command()
def topix(start: DateOpt, end: DateOpt) -> None:
    """Fetch TOPIX daily closes (J-Quants Light plan and above)."""
    n = pipeline.sync_topix(_jquants_client(), get_settings(), start.date(), end.date())
    console.print(f"[green]{n} TOPIX sessions[/green]")


@app.command("tender-offers")
def tender_offers() -> None:
    """Read downloaded tender-offer filings (240): price, prior stake, event type."""
    offers = pipeline.parse_tender_offers(get_settings())
    if offers.empty:
        console.print("[yellow]No 240 filing downloaded yet[/yellow]")
        return
    console.print(f"[green]{len(offers)} offers[/green]")
    console.print(offers["event_type"].value_counts().to_string())


@app.command("event-study")
def event_study_cmd(
    runups: Annotated[
        bool, typer.Option(help="list large run-ups with leakage indicators")
    ] = False,
) -> None:
    """Premium and pre-announcement run-up per tender offer (median by year and type)."""
    result = pipeline.run_event_study(get_settings())
    if runups:
        _print_runups(pipeline.review_runups(get_settings()))
        return
    usable = result[result["pre_announced"].eq(False) & result["is_take_private"].eq(True)].copy()
    usable["year"] = pd.to_datetime(usable["event_date"]).dt.year
    summary = usable.groupby(["year", "event_type"]).agg(
        offers=("doc_id", "count"),
        median_premium=("premium", "median"),
        median_runup=("runup_abnormal", "median"),
    )
    console.print(
        f"{len(result)} offers, {len(usable)} usable "
        "(take-private, priced, price not already at/above the offer)"
    )
    console.print(summary.round(3).to_string())


def _print_runups(review: pd.DataFrame) -> None:
    flagged = review[review["flagged"]]
    calm = review[~review["flagged"]]
    console.print(
        f"{len(flagged)} of {len(review)} offers ran up more than the review threshold. "
        f"Median volume ratio: flagged {flagged['volume_ratio'].median():.2f} vs "
        f"others {calm['volume_ratio'].median():.2f} (1.0 = normal trading)"
    )
    table = Table(title="Large pre-announcement run-ups (read these filings one by one)")
    for column in (
        "date",
        "target",
        "type",
        "run-up",
        "premium",
        "volume×",
        "activist before",
        "240",
    ):
        table.add_column(column)
    for row in flagged.to_dict("records"):
        volume = row["volume_ratio"]
        table.add_row(
            str(row["event_date"]),
            str(row["company"])[:22],
            str(row["event_type"]),
            f"{row['runup_abnormal']:+.0%}",
            f"{row['premium']:.0%}",
            "—" if pd.isna(volume) else f"{volume:.1f}",
            public_reason(row),
            str(row["doc_id"]),
        )
    console.print(table)


@app.command()
def audit(
    as_of: DateOpt,
    rules: Annotated[
        str | None, typer.Option(help="rule file in config/ (default: config/scoring.yaml)")
    ] = None,
) -> None:
    """Recompute every rule and score, and check feature invariants, for one screening date."""
    issues = pipeline.run_audit(get_settings(), as_of.date(), rules)
    if issues.empty:
        console.print(f"[green]Audit {as_of.date()}: features and scores are consistent[/green]")
        return
    console.print(f"[red]{len(issues)} issue(s)[/red]")
    console.print(issues.groupby(["source", "check"]).size().to_string())
    console.print(issues.head(10).to_string(index=False))


@app.command()
def update(
    today: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="screening date (default: today)")
    ] = None,
) -> None:
    """Daily refresh: new filings, prices and lists since the last run, then screen today."""
    settings = get_settings()
    day = (today or datetime.now()).date()
    result = pipeline.update(settings, _edinet_client(), _jquants_client(), day)
    console.print(
        f"[green]Updated[/green] index {result.index_from} → {result.index_to} · "
        f"{result.downloaded} documents · {result.parsed} new reports parsed · "
        f"{result.prices_days} price days · {result.companies} companies screened on {day}"
    )
    if result.audit_issues:
        console.print(f"[red]Audit: {result.audit_issues} issue(s) — run `radar audit`[/red]")
    console.print(
        f"臨時報告書 of followed companies fetched: {result.extraordinary_reports} · "
        f"live scoreboard refreshed"
    )
    for path in result.new_factpacks:
        console.print(f"New in the top — fact pack: {path}")
    if result.previous_screening is not None:
        _print_changes(*pipeline.run_changes(settings, result.previous_screening, day))


@app.command()
def changes(
    since: Annotated[datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    until: Annotated[datetime, typer.Option("--to", formats=["%Y-%m-%d"])],
) -> None:
    """What changed between two screening dates (+ new filings on followed companies)."""
    _print_changes(*pipeline.run_changes(get_settings(), since.date(), until.date()))


def _print_changes(changed: pd.DataFrame, events: pd.DataFrame) -> None:
    if changed.empty and events.empty:
        console.print("Nothing new on followed companies.")
        return
    for row in events.to_dict("records"):
        console.print(f"[bold]{row['kind']}[/bold] · {row['company_code']} · {row['filer_name']}")
    for row in changed.to_dict("records"):
        rules = [f"+{r}" for r in row["rules_added"]] + [f"-{r}" for r in row["rules_removed"]]
        console.print(f"{row['company_name']}: {row['status']} {' '.join(rules)}")
    console.print("Full note: data/marts/alerts_<date>.md")


@app.command("inspect-doc")
def inspect_doc(
    doc_id: str,
    grep: Annotated[str, typer.Option(help="regex on element ID or label")] = ".",
    max_rows: int = 60,
) -> None:
    """Show element IDs in a downloaded filing — used to verify config/xbrl_elements.yaml."""
    path: Path = edinet_doc_path(get_settings().raw_dir, doc_id, FetchType.CSV)
    if not path.exists():
        raise typer.BadParameter(f"{path} not found; run `radar download` first")
    facts = read_facts_zip(path)
    pattern = re.compile(grep)
    hits = facts[
        facts["element_id"].str.contains(pattern) | facts["label"].str.contains(pattern)
    ].head(max_rows)
    table = Table()
    for column in ("element_id", "label", "context_id", "value"):
        table.add_column(column, overflow="fold")  # never truncate IDs: we copy them into config
    for row in hits.itertuples(index=False):
        table.add_row(str(row.element_id), str(row.label), str(row.context_id), str(row.value)[:40])
    console.print(table)
    console.print(f"{len(hits)} rows shown (of {len(facts)} facts)")


@app.command()
def events() -> None:
    """Build the tender-offer event table from the document index (backtest ground truth)."""
    result = pipeline.build_tender_offer_events(get_settings())
    console.print(f"[green]{len(result)} tender-offer events[/green]")


@app.command()
def parse() -> None:
    """Parse downloaded annual reports into staging tables and print a data-quality report."""
    settings = get_settings()
    result = pipeline.parse_downloaded_filings(settings)
    staging = (
        pipeline.load_staging(settings, pipeline.OFFICERS_TABLE),
        pipeline.load_staging(settings, pipeline.SHAREHOLDERS_TABLE),
        pipeline.load_staging(settings, pipeline.FINANCIALS_TABLE),
    )
    report = quality_report(*staging)
    table = Table(title=f"Data quality — {report.filings} annual reports parsed")
    table.add_column("filings with …")
    table.add_column("share", justify="right")
    table.add_row("≥1 representative director", f"{report.with_representative:.1%}")
    table.add_row("representative's birth date", f"{report.representative_with_birth_date:.1%}")
    table.add_row("shareholder ratios", f"{report.with_shareholder_ratios:.1%}")
    table.add_row("J-GAAP financials", f"{report.jgaap:.1%}")
    console.print(table)
    if result.failed:
        console.print(f"[red]{len(result.failed)} unreadable:[/red] {', '.join(result.failed)}")
    issues = sanity_checks(*staging)
    if issues.empty:
        console.print("[green]Sanity checks: no impossible values[/green]")
    else:
        counts = issues["check"].value_counts()
        console.print("[yellow]Sanity checks (open these filings by hand):[/yellow]")
        for check, n in counts.items():
            sample = ", ".join(issues.loc[issues["check"] == check, "doc_id"].head(3))
            console.print(f"  {check}: {n} filing(s), e.g. {sample}")


@app.command()
def features(as_of: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])]) -> None:
    """Build the point-in-time feature table for one date (marts/features_<as_of>.parquet)."""
    result = pipeline.build_features(get_settings(), as_of.date())
    console.print(f"[green]{len(result)} companies[/green] with features as of {as_of.date()}")


@app.command()
def score(
    as_of: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    top: Annotated[int, typer.Option(help="rows to print")] = 30,
) -> None:
    """Score companies with the rule set and write data/marts/scores_<as_of>.parquet."""
    try:
        scores = pipeline.score_companies(get_settings(), as_of.date())
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from None
    table = Table(title=f"Top {min(top, len(scores))} as of {as_of.date()}")
    for column in ("rank", "company", "score", "rules triggered"):
        table.add_column(column)
    for rank, row in enumerate(scores.head(top).to_dict("records"), start=1):
        rules = ", ".join(str(rule) for rule in row["triggered_rules"]) or "—"
        table.add_row(str(rank), str(row["company_name"]), f"{row['score']:.2f}", rules)
    console.print(table)


@app.command()
def backtest(
    start: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    end: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])],
    rules: RulesOpt = "scoring_v0",
) -> None:
    """Score every quarter-end between --start and --end and compare with real tender offers."""
    metrics = pipeline.run_backtest(get_settings(), start.date(), end.date(), rules)
    if metrics.empty:
        console.print("[yellow]No quarter-end between --start and --end[/yellow]")
        return
    table = Table(title=f"Backtest ({rules})")
    for column in metrics.columns:
        table.add_column(str(column), justify="right")
    for row in metrics.to_dict("records"):
        table.add_row(*(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()))
    console.print(table)
    if not metrics["label_complete"].all():
        console.print("[yellow]label_complete=False: the window is not fully observed yet[/yellow]")


@app.command("ablation")
def ablation_cmd(rules: RulesOpt = "scoring_v0") -> None:
    """Which rules carry the backtest result?"""
    result = pipeline.run_ablation(get_settings(), rules)
    if result.empty:
        console.print(
            "[yellow]No complete label window in the last backtest — nothing to ablate[/yellow]"
        )
        return
    table = Table(title="Ablation (mean over complete as_of dates; delta < 0 → the rule helps)")
    for column in result.columns:
        table.add_column(str(column), justify="right")
    for row in result.to_dict("records"):
        table.add_row(*(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()))
    console.print(table)


@app.command("discover-activists")
def discover_activists_cmd(
    download: Annotated[
        bool, typer.Option(help="download missing reports first (~3,000, hours; cached)")
    ] = False,
) -> None:
    """Filers whose 5% report declares 重要提案行為 but are not in config/activists.yaml."""
    client = _edinet_client() if download else None
    candidates = pipeline.discover_activists(get_settings(), client)
    table = Table(title=f"{len(candidates)} activist candidates — review before adding any")
    for column in ("filer_name", "submit_datetime", "purpose"):
        table.add_column(column)
    for row in candidates.to_dict("records"):
        table.add_row(str(row["filer_name"]), str(row["submit_datetime"])[:10], str(row["purpose"]))
    console.print(table)


@app.command("factpack")
def factpack_cmd(
    as_of: Annotated[datetime, typer.Option("--as-of", formats=["%Y-%m-%d"])],
    top: Annotated[int, typer.Option(help="the N highest-scored companies")] = 15,
    code: Annotated[
        list[str], typer.Option(help="extra securities code, repeatable (7122 or 71220)")
    ] = [],  # noqa: B006
) -> None:
    """Fact packs: the facts and sources behind a company memo."""
    codes = [c if len(c) == 5 else f"{c}0" for c in code]
    paths = pipeline.build_factpacks(get_settings(), as_of.date(), codes, top)
    for path in paths:
        console.print(str(path))
    console.print(f"[green]{len(paths)} fact packs[/green]")


@app.command("picks")
def picks_cmd(rules: RulesOpt = "scoring_v0") -> None:
    """What if you had bought the top N?"""
    result = pipeline.run_picks(get_settings(), rules)
    columns = [
        "hit_rate", "mean_return", "median_return", "share_positive", "worst_return",
        "topix_return", "universe_mean_return", "excess_vs_universe", "exited_early",
    ]  # fmt: skip
    mean = result.groupby("top_n")[columns].mean()
    table = Table(title=f"Top-N picks, mean over {result['as_of'].nunique()} complete dates")
    table.add_column("top_n")
    for column in columns:
        table.add_column(column, justify="right")
    for n, row in mean.iterrows():
        table.add_row(str(n), *(f"{v:.3f}" for v in row))
    console.print(table)


@app.command("walkforward")
def walkforward_cmd(rules: RulesOpt = "scoring_v0") -> None:
    """Pick the default ranking walk-forward with the pre-registered rule."""
    metrics, choice, by_return, rounds = pipeline.run_walk_forward(get_settings(), rules)
    columns = ["lift_top", "recall_top", "precision_at_20", "precision_at_50"]
    columns += [c for c in metrics.columns if c in ("lift_top_mbo", "lift_top_parent")]
    mean = metrics.groupby("model")[columns].mean()
    dates = metrics["as_of"].nunique()
    table = Table(title=f"Walk-forward: mean over {dates} test dates, each trained on the past")
    table.add_column("model")
    for column in [*columns, "wins vs v0"]:
        table.add_column(column, justify="right")
    wins = choice.table.set_index("model")["wins_vs_baseline"]
    for name, row in mean.iterrows():
        table.add_row(str(name), *(f"{v:.3f}" for v in row), str(wins.get(name, "—")))
    console.print(table)
    console.print(f"[bold]Decision (pre-registered rule): {choice.winner}[/bold]")
    console.print("Portfolio question — top-25 return minus all companies, mean over the dates:")
    for entry in by_return.table.to_dict("records"):
        console.print(
            f"  {entry['model']:<14} {entry['mean_metric']:+.3f}  wins vs basket: "
            f"{entry['wins_vs_baseline']}  eligible: {entry['eligible']}"
        )
    console.print(
        f"[bold]Portfolio default (pre-registered return rule): {by_return.winner}[/bold]"
    )
    for name, decided in rounds.items():
        console.print(f"{name} (rule written before running, docs/results.md):")
        for entry in decided.table.to_dict("records"):
            console.print(
                f"  {entry['model']:<28} {entry['mean_metric']:+.3f}  wins vs basket: "
                f"{entry['wins_vs_baseline']}  eligible: {entry['eligible']}"
            )
        console.print(f"[bold]{name} decision: {decided.winner}[/bold]")


@app.command("allocate")
def allocate_cmd(
    as_of: Annotated[datetime, typer.Option("--as-of", formats=["%Y-%m-%d"])],
    capital: Annotated[float, typer.Option(help="capital in yen")],
    ranking: Annotated[str, typer.Option(help="score column to rank by")] = "value_basket",
    top: Annotated[int, typer.Option(help="number of picks (a walk-forward pick size)")] = 25,
    rules: RulesOpt = "scoring_v0",
) -> None:
    """Research allocation: 1/3-Kelly invested fraction, equal weights capped by liquidity in
    100-share lots.
    """
    plan, invested, past = pipeline.run_allocation(
        get_settings(), as_of.date(), ranking=ranking, top=top, capital=capital, ruleset=rules
    )
    console.print(
        f"Past {ranking} top {top} 12-month returns on {len(past)} walk-forward dates: "
        f"mean {past.mean():+.1%}, st. dev. {past.std():.1%}"
    )
    if invested is None:
        console.print("[yellow]Too few dates (or no spread) for a Kelly estimate: 0% invested.")
    else:
        console.print(f"[bold]1/3-Kelly invested fraction: {invested:.0%} of capital[/bold]")
    table = Table(title=f"Allocation of ¥{capital:,.0f} — {ranking} top {top}, {as_of:%Y-%m-%d}")
    for column in ["code", "company", "close", "shares", "position ¥", "days", "binding cap"]:
        table.add_column(
            column, justify="left" if column in ("company", "binding cap") else "right"
        )
    for row in plan.to_dict("records"):
        table.add_row(
            str(row["sec_code"]), str(row["company_name"])[:18], f"{row['close']:,.0f}",
            f"{row['shares']:,}", f"{row['position_jpy']:,.0f}",
            "—" if pd.isna(row["days_to_trade"]) else f"{row['days_to_trade']:.1f}",
            str(row["reason"]),
        )  # fmt: skip
    console.print(table)
    console.print(
        f"Invested ¥{plan['position_jpy'].sum():,.0f}; the rest stays in cash. "
        "Caps left in cash are not redistributed (that would concentrate the basket)."
    )


@app.command("learn-weights")
def learn_weights_cmd(
    base: Annotated[str, typer.Option(help="rule set whose rules are re-weighted")] = "scoring_v0",
    target: Annotated[str, typer.Option(help="new rule file in config/")] = "scoring_v2",
) -> None:
    """Learn the rule weights from every complete backtest date → config/<target>.yaml."""
    learned = pipeline.learn_weights(get_settings(), base, target)
    for rule in learned.rules:
        console.print(f"  {rule.id:<22} {rule.weight:+.2f}")
    console.print(
        f"[green]config/{target}.yaml written[/green] (max {learned.max_score:.2f} points)"
    )


@app.command("weight-stability")
def weight_stability_cmd(
    base: Annotated[str, typer.Option(help="rule set whose rules are re-weighted")] = "scoring_v0",
) -> None:
    """Bootstrap the learned rule weights over whole dates: which weights are stable?"""
    table = pipeline.weight_stability(get_settings(), base)
    console.print(f"Learned weights over {int(table['draws'].iloc[0])} resamples of the dates:")
    for row in table.to_dict("records"):
        console.print(
            f"  {row['rule']:<22} p5 {row['p5']:+.2f}  median {row['p50']:+.2f}  "
            f"p95 {row['p95']:+.2f}  positive in {row['share_positive']:.0%}"
        )


@app.command("scoreboard")
def scoreboard_cmd(
    freeze: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="freeze this day's scores first (write once)"),
    ] = None,
) -> None:
    """Live, out-of-sample track record of the frozen rankings (`radar update` freezes daily)."""
    settings = get_settings()
    if freeze is not None:
        written = pipeline.snapshot_live_scores(settings, freeze.date())
        console.print(f"snapshot {freeze.date()}: {'written' if written else 'already frozen'}")
    board = pipeline.run_scoreboard(settings, date.today())
    if board.empty:
        console.print("[yellow]No frozen ranking yet: run `radar update` (or --freeze).[/yellow]")
        return
    table = Table(title="Live track record (rankings frozen on the day, judged afterwards)")
    for column in board.columns:
        table.add_column(str(column), justify="right")
    for row in board.to_dict("records"):
        table.add_row(*(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()))
    console.print(table)


@app.command("model")
def model_cmd(rules: RulesOpt = "scoring_v0") -> None:
    """Logistic regression vs the rule set on the same later dates (needs the last backtest)."""
    comparison, coefs = pipeline.run_model(get_settings(), rules)
    columns = ["precision_at_20", "precision_at_50", "recall_top", "lift_top"]
    columns += [c for c in comparison.columns if c.startswith("lift_top_")]
    mean = comparison.groupby("model")[columns].mean()
    dates = ", ".join(str(d) for d in sorted(comparison["as_of"].unique()))
    table = Table(title=f"Test dates {dates} (mean)")
    table.add_column("model")
    for column in columns:
        table.add_column(column, justify="right")
    for name, row in mean.iterrows():
        table.add_row(str(name), *(f"{v:.3f}" for v in row))
    console.print(table)
    console.print("Largest coefficients (standardised inputs):")
    for feature, coef in zip(coefs["feature"].head(12), coefs["coefficient"], strict=False):
        console.print(f"  {feature:<40} {coef:+.3f}")


if __name__ == "__main__":
    app()
