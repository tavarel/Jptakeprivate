"""What changed between two screening dates, and new events on followed companies."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from radar.features.activism import is_activist
from radar.report.explain import edinet_url

CHANGE_COLUMNS = [
    "edinet_code",
    "company_name",
    "sec_code",
    "status",
    "rank_old",
    "rank_new",
    "points_old",
    "points_new",
    "points_change",
    "rules_added",
    "rules_removed",
]
EVENT_COLUMNS = ["submit_datetime", "company_code", "kind", "filer_name", "doc_id"]
_STATUS_ORDER = {"entered": 0, "rules": 1, "left": 2}


def _rules(value: Any) -> set[str]:
    """Rule ids of one row; a missing row (company absent on that date) has none."""
    if isinstance(value, Iterable) and not isinstance(value, str):
        return {str(r) for r in value}
    return set()


def _ranked(scores: pd.DataFrame) -> pd.DataFrame:
    """edinet_code → name, code, score, rank (1 = best; ties share the better rank), rules."""
    ranked = scores.assign(rank=scores["score"].rank(ascending=False, method="min"))
    columns = ["edinet_code", "company_name", "sec_code", "score", "rank", "triggered_rules"]
    return ranked[columns].set_index("edinet_code")


def _status(rank_old: float, rank_new: float, rules_moved: bool, top_n: int) -> str | None:
    in_old = bool(pd.notna(rank_old) and rank_old <= top_n)
    in_new = bool(pd.notna(rank_new) and rank_new <= top_n)
    if in_new and not in_old:
        return f"entered top {top_n}"
    if in_old and not in_new:
        return f"left top {top_n}"
    if (in_old or in_new) and rules_moved:
        return "rules changed"
    return None


def compare_scores(
    old: pd.DataFrame, new: pd.DataFrame, top_n: int, max_score: float
) -> pd.DataFrame:
    """Companies whose position or reasons changed (scores tables: edinet_code, company_name,
    sec_code, score, triggered_rules).
    """
    before, after = _ranked(old), _ranked(new)
    rows = []
    for code in before.index.union(after.index):
        was = before.loc[code] if code in before.index else None
        now = after.loc[code] if code in after.index else None
        rules_old = _rules(None if was is None else was["triggered_rules"])
        rules_new = _rules(None if now is None else now["triggered_rules"])
        rank_old = float("nan") if was is None else float(was["rank"])
        rank_new = float("nan") if now is None else float(now["rank"])
        status = _status(rank_old, rank_new, rules_old != rules_new, top_n)
        if status is None:
            continue
        points_old = None if was is None else float(was["score"]) * max_score
        points_new = None if now is None else float(now["score"]) * max_score
        latest = now if now is not None else was
        assert latest is not None  # the code comes from one of the two tables
        rows.append(
            {
                "edinet_code": code,
                "company_name": latest["company_name"],
                "sec_code": latest["sec_code"],
                "status": status,
                "rank_old": rank_old,
                "rank_new": rank_new,
                "points_old": points_old,
                "points_new": points_new,
                "points_change": None
                if points_old is None or points_new is None
                else round(points_new - points_old, 6),
                "rules_added": sorted(rules_new - rules_old),
                "rules_removed": sorted(rules_old - rules_new),
            }
        )
    result = pd.DataFrame(rows, columns=CHANGE_COLUMNS)
    order = result["status"].str.split().str[0].map(_STATUS_ORDER)
    result = result.assign(_order=order).sort_values(["_order", "rank_new"])
    return result.drop(columns="_order").reset_index(drop=True)


def _concerned_company(row: Mapping[Hashable, Any]) -> Any:
    """The listed company a filing is ABOUT: the issuer of a 5% report, the target of an offer, the
    filer of an opinion (the target itself).
    """
    kind = row["doc_type_code"]
    if kind == "350":
        return row.get("issuer_edinet_code")
    if kind == "240":
        return row.get("subject_edinet_code")
    return row.get("edinet_code")


def new_events(
    documents: pd.DataFrame,
    followed: set[str],
    since: datetime,
    until: datetime,
    activist_patterns: Sequence[str],
    *,
    extraordinary: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Filings submitted in (since, until] about a followed company, newest first."""
    extraordinary = extraordinary or {}
    submitted = pd.to_datetime(documents["submit_datetime"])
    window = documents[
        documents["doc_type_code"].isin(["350", "240", "290", "180"])
        & (documents["withdrawal_status"].astype(str) == "0")
        & (submitted > since)
        & (submitted <= until)
    ]
    rows = []
    for row in window.to_dict("records"):
        company = _concerned_company(row)
        if company not in followed:
            continue
        if row["doc_type_code"] == "350":
            activist = is_activist(str(row["filer_name"]), activist_patterns)
            kind = "activist 5% filing" if activist else "5% filing"
        elif row["doc_type_code"] == "240":
            kind = "tender offer filed"
        elif row["doc_type_code"] == "180":
            found = extraordinary.get(str(row["doc_id"]))
            if found is None:
                kind = "臨時報告書 (reason not read yet)"
            elif not found["alert"]:
                continue
            else:
                after_offer = " (after a tender offer)" if found["tender_offer"] else ""
                kind = f"臨時報告書: {found['labels']}{after_offer}"
        else:
            kind = "target's opinion on an offer"
        rows.append(
            {
                "submit_datetime": row["submit_datetime"],
                "company_code": company,
                "kind": kind,
                "filer_name": row["filer_name"],
                "doc_id": row["doc_id"],
            }
        )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events.sort_values("submit_datetime", ascending=False).reset_index(drop=True)


def watch_codes(entries: Iterable[Mapping[str, Any]], documents: pd.DataFrame) -> set[str]:
    """EDINET codes of watchlist entries (4-character exchange codes → 5-character sec_code)."""
    wanted = {f"{str(e['code']).strip()}0" for e in entries}
    listed = documents[documents["sec_code"].isin(wanted)]
    return {str(c) for c in listed["edinet_code"].unique()}


def alerts_markdown(
    changes: pd.DataFrame,
    events: pd.DataFrame,
    names: Mapping[str, str],
    since: date,
    until: date,
    *,
    factpacks: Mapping[str, Path] | None = None,
    unusual: pd.DataFrame | None = None,
    drift: Sequence[str] = (),
) -> str:
    """A short markdown note (to read, mail or paste in Slack) of changes and events."""
    factpacks = factpacks or {}
    unusual = unusual if unusual is not None else pd.DataFrame(columns=["sec_code"])
    lines = [f"# Radar alerts {since} → {until}", ""]
    if drift:  # first: when the data broke, the rest of the note cannot be trusted
        lines += ["## ⚠ Data coverage dropped — check before trusting this screen", ""]
        lines += [f"- {line}" for line in drift] + [""]
    if not unusual.empty:
        lines += ["## Unusual trading on followed companies (check by hand)", ""]
        for row in unusual.to_dict("records"):
            name = row.get("company_name") or str(row["sec_code"])[:4]
            lines.append(
                f"- **{name}** ({str(row['sec_code'])[:4]}): volume ×{row['volume_ratio']:.1f} "
                f"vs usual, {row['excess_5d']:+.0%} vs TOPIX in the last week"
            )
        lines.append("")
    if changes.empty and events.empty:
        return "\n".join([*lines, "Nothing new on followed companies."]) + "\n"
    if not events.empty:
        lines += ["## New filings on followed companies", ""]
        for row in events.to_dict("records"):
            company = names.get(str(row["company_code"]), str(row["company_code"]))
            when = pd.Timestamp(row["submit_datetime"]).date()
            url = edinet_url(str(row["doc_id"]))
            lines.append(
                f"- {when} **{company}** — {row['kind']} by {row['filer_name']} ([filing]({url}))"
            )
        lines.append("")
    if not changes.empty:
        lines += ["## Ranking changes", ""]
        for row in changes.to_dict("records"):
            delta = row["points_change"]
            points = "" if delta is None or pd.isna(delta) else f" ({delta:+.1f} points)"
            rules = [f"+{r}" for r in row["rules_added"]] + [f"−{r}" for r in row["rules_removed"]]
            reasons = f" — {', '.join(rules)}" if rules else ""
            pack = factpacks.get(str(row["sec_code"]))
            link = f" — fact pack: `{pack.name}`" if pack else ""
            lines.append(f"- **{row['company_name']}**: {row['status']}{points}{reasons}{link}")
    return "\n".join(lines) + "\n"


def coverage_drift(
    old: pd.DataFrame, new: pd.DataFrame, columns: Sequence[str], max_drop: float
) -> list[str]:
    """Key features whose share of known values fell by more than `max_drop` between two screens."""
    lines = []
    for column in columns:
        if column not in old.columns or column not in new.columns or old.empty or new.empty:
            continue
        before, after = float(old[column].notna().mean()), float(new[column].notna().mean())
        if before - after > max_drop:
            lines.append(f"{column}: known for {before:.0%} → {after:.0%} of companies")
    return lines
