"""Decision journal: what was decided, why, and what happened since."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

from radar.model.portfolio import AdjustedPrices

ACTIONS = ("watch", "buy", "sell", "pass")
COLUMNS = ["date", "code", "action", "price", "reason"]


def parse_journal(raw: Mapping[str, Any] | None) -> pd.DataFrame:
    """Validated entries; a bad entry raises ValueError naming the entry and the field."""
    rows = []
    for number, entry in enumerate((raw or {}).get("entries") or [], start=1):
        rows.append(_entry(number, entry))
    return pd.DataFrame(rows, columns=COLUMNS)


def _entry(number: int, entry: Mapping[str, Any]) -> dict[str, Any]:
    where = f"journal entry {number}"
    try:
        day = pd.Timestamp(str(entry["date"])).date()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{where}: `date` missing or not YYYY-MM-DD") from exc
    action = str(entry.get("action", "")).lower()
    if action not in ACTIONS:
        raise ValueError(f"{where}: `action` must be one of {', '.join(ACTIONS)}")
    if "code" not in entry:
        raise ValueError(f"{where}: `code` missing")
    price = entry.get("price")
    return {
        "date": day,
        "code": str(entry["code"]).strip(),
        "action": action,
        "price": float(price) if price is not None else float("nan"),
        "reason": str(entry.get("reason") or ""),
    }


def journal_outcomes(
    entries: pd.DataFrame,
    prices: pd.DataFrame,
    offers: pd.DataFrame,
    ranks: pd.Series,
    *,
    today: date,
) -> pd.DataFrame:
    """Each entry with: split-adjusted return from the last close on or before its date to the last
    close before today, the first take-private offer filed AFTER it, and today's rank.
    """
    adjusted = AdjustedPrices(prices)
    after: list[object] = []
    returns = []
    for row in entries.to_dict("records"):
        code = sec_code(str(row["code"]))
        returns.append(_return(adjusted, code, row["date"], today))
        later = offers.loc[
            (offers["sec_code"].astype(str) == code)
            & (pd.to_datetime(offers["event_date"]).dt.date > row["date"]),
            "event_date",
        ]
        after.append(pd.Timestamp(later.min()).date() if not later.empty else pd.NaT)
    return entries.assign(
        return_since=returns,
        offer_after=pd.Series(after, index=entries.index, dtype=object),
        rank_now=[ranks.get(sec_code(str(code)), float("nan")) for code in entries["code"]],
    )


def sec_code(code: str) -> str:
    """People write 4-character exchange codes (7122, 130A); the data uses 5 (71220, 130A0)."""
    return f"{code}0" if len(code) == 4 else code


def _return(prices: AdjustedPrices, code: str, start: date, today: date) -> float:
    bought = prices.close_on_or_before(code, start)
    now = prices.close_on_or_before(code, today - timedelta(days=1))
    if bought is None or now is None or bought[1] <= 0:
        return float("nan")
    return now[1] / bought[1] - 1
