"""Canonical paths for every artefact."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def edinet_doclist_path(raw_dir: Path, day: date) -> Path:
    return raw_dir / "edinet" / "doclists" / f"{day.isoformat()}.json"


def edinet_doc_path(raw_dir: Path, doc_id: str, fetch_type: int) -> Path:
    return raw_dir / "edinet" / "docs" / f"{doc_id}_type{fetch_type}.zip"


def jquants_bars_path(raw_dir: Path, day: date) -> Path:
    """All stocks' daily bars for one trading day (J-Quants `/equities/bars/daily?date=`)."""
    return raw_dir / "jquants" / "bars" / f"{day.isoformat()}.json"


def jquants_master_path(raw_dir: Path, day: date) -> Path:
    """Listed-company master as of one day (J-Quants `/equities/master?date=`)."""
    return raw_dir / "jquants" / "master" / f"{day.isoformat()}.json"


def jquants_topix_path(raw_dir: Path, year: int) -> Path:
    """TOPIX daily bars for one calendar year (J-Quants `/indices/bars/daily/topix`)."""
    return raw_dir / "jquants" / "topix" / f"{year}.json"


def staging_table_path(staging_dir: Path, table: str) -> Path:
    return staging_dir / f"{table}.parquet"


def mart_table_path(marts_dir: Path, table: str) -> Path:
    return marts_dir / f"{table}.parquet"


def live_scores_path(marts_dir: Path, as_of: date) -> Path:
    """Frozen ranking of one screening day (write once: the live, out-of-sample record)."""
    return marts_dir / "live" / f"scores_{as_of.isoformat()}.parquet"


def live_model_path(marts_dir: Path) -> Path:
    """Per-type models fitted on every complete backtest date, used to score each day."""
    return marts_dir / "models" / "live_models.joblib"


def return_model_path(marts_dir: Path) -> Path:
    """Return model refitted on every complete date."""
    return marts_dir / "models" / "return_model.joblib"


def factpack_path(marts_dir: Path, as_of: date, sec_code: str) -> Path:
    """Markdown fact pack of one company (4-character code, as people write it)."""
    return marts_dir / f"factpack_{as_of.isoformat()}_{sec_code[:4]}.md"


def journal_path(manual_dir: Path) -> Path:
    """Decision journal, edited by hand (personal: under data/, never committed)."""
    return manual_dir / "journal.yaml"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
