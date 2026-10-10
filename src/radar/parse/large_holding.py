"""Large-shareholding reports (大量保有報告書, docTypeCode 350, form `jplvh`)."""

from __future__ import annotations

import pandas as pd

from radar.text_ja import normalize

FORM_PREFIX = "jplvh"


def holding_purposes(facts: pd.DataFrame, element_id: str) -> list[str]:
    """Every non-empty 保有目的 in the report, NFKC-normalised, in file order."""
    values = facts.loc[facts["element_id"] == element_id, "value"]
    return [normalize(str(v)) for v in values if str(v).strip()]
