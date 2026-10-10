from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from radar.parse.elements import ElementMap
from radar.parse.xbrl_csv import FACT_COLUMNS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def doclist_payload() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (FIXTURES / "edinet_doclist_sample.json").read_text(encoding="utf-8")
    )
    return data


@pytest.fixture
def elements() -> ElementMap:
    """The real element map from config/ — tests fail loudly if the config shape changes."""
    return ElementMap.load()


def make_facts(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a facts table from (element_id, context_id, value) triples."""
    records = [
        {
            "element_id": element_id,
            "label": "",
            "context_id": context_id,
            "relative_year": "",
            "consolidation": "",
            "period_type": "",
            "unit_id": "",
            "unit": "",
            "value": value,
        }
        for element_id, context_id, value in rows
    ]
    return pd.DataFrame(records, columns=FACT_COLUMNS)
