"""Tender-offer events from the EDINET document index, and point-in-time labels."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta

import pandas as pd

TENDER_OFFER = "240"
WITHDRAWAL = "260"  # 公開買付撤回届出書
EVENT_COLUMNS = ["subject_edinet_code", "event_date", "doc_id", "acquirer_name"]
# parse/tender_offer.classify_event; "unknown" = 240 not read or stake unreadable.
EVENT_TYPES = ("mbo", "parent", "third_party", "unknown")


def tender_offer_events(documents: pd.DataFrame) -> pd.DataFrame:
    """One row per (target company, tender offer filing)."""
    offers = documents[
        (documents["doc_type_code"] == TENDER_OFFER)
        & (documents["withdrawal_status"].astype(str) == "0")
        & documents["subject_edinet_code"].notna()
        & (documents["subject_edinet_code"] != documents["edinet_code"])
    ].copy()
    offers = offers[~offers["doc_id"].isin(_withdrawn_offer_ids(documents, offers))]
    offers["event_date"] = pd.to_datetime(offers["submit_datetime"]).dt.date
    offers = offers.rename(columns={"filer_name": "acquirer_name"})
    return offers[EVENT_COLUMNS].sort_values("event_date").reset_index(drop=True)


def _withdrawn_offer_ids(documents: pd.DataFrame, offers: pd.DataFrame) -> set[str]:
    """240s cancelled by a 260: linked by `parent_doc_id`, else the same bidder and target with the
    240 filed before the 260 (the link is not always filled in).
    """
    notices = documents[documents["doc_type_code"] == WITHDRAWAL]
    withdrawn = set(notices["parent_doc_id"].dropna())
    unlinked = notices[notices["parent_doc_id"].isna()]
    submitted = pd.to_datetime(offers["submit_datetime"])
    for notice in unlinked.to_dict("records"):
        same_parties = (offers["edinet_code"] == notice["edinet_code"]) & (
            offers["subject_edinet_code"] == notice["subject_edinet_code"]
        )
        earlier = submitted < pd.Timestamp(str(notice["submit_datetime"]))
        withdrawn |= set(offers.loc[same_parties & earlier, "doc_id"])
    return withdrawn


def label_companies(
    edinet_codes: pd.Series, events: pd.DataFrame, as_of: date, horizon_days: int = 365
) -> pd.Series:
    """1 if the company becomes a tender-offer target in (as_of, as_of + horizon], else 0."""
    end = as_of + timedelta(days=horizon_days)
    in_window = events[(events["event_date"] > as_of) & (events["event_date"] <= end)]
    targets = set(in_window["subject_edinet_code"])
    return edinet_codes.isin(targets).astype(int).rename("label")


def take_private_events(events: pd.DataFrame, offers: pd.DataFrame | None) -> pd.DataFrame:
    """Drop the offers KNOWN to be partial (target holding < 95%: no take-private intended)."""
    if offers is None or offers.empty:
        return events
    partial = set(offers.loc[offers["is_take_private"].eq(False), "doc_id"])
    return events[~events["doc_id"].isin(partial)].reset_index(drop=True)


def label_frame(
    edinet_codes: pd.Series,
    events: pd.DataFrame,
    offers: pd.DataFrame | None,
    as_of: date,
    horizon_days: int = 365,
) -> pd.DataFrame:
    """`label` (any offer) plus `label_<type>` for each of EVENT_TYPES, same window as
    `label_companies`.
    """
    types = (
        offers.drop_duplicates("doc_id").set_index("doc_id")["event_type"]
        if offers is not None and not offers.empty
        else pd.Series(dtype=object)
    )
    event_type = events["doc_id"].map(types).fillna("unknown")
    frame = pd.DataFrame({"label": label_companies(edinet_codes, events, as_of, horizon_days)})
    for kind in EVENT_TYPES:
        typed = events[event_type == kind]
        frame[f"label_{kind}"] = label_companies(edinet_codes, typed, as_of, horizon_days)
    return frame


def already_public(events: pd.DataFrame, closes: Mapping[str, float], tolerance: float) -> set[str]:
    """Targets whose price already sat at the offer on the screening date (announced before the
    240 filing, e.g. 予告TOB): not a prediction, so excluded from that date's evaluation.
    """
    public: set[str] = set()
    for code, price in zip(events["subject_edinet_code"], events["offer_price"], strict=True):
        close = closes.get(str(code))
        if price is None or pd.isna(price) or close is None or price <= 0:
            continue
        if close >= price * (1 - tolerance):
            public.add(str(code))
    return public
