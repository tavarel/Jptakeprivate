from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from radar.parse.xbrl_csv import read_facts_csv, read_facts_zip

HEADER = [
    "要素ID",
    "項目名",
    "コンテキストID",
    "相対年度",
    "連結・個別",
    "期間・時点",
    "ユニットID",
    "単位",
    "値",
]
ROW = [
    "jppfs_cor:CashAndDeposits",
    "現金及び預金",
    "CurrentYearInstant",
    "当期末",
    "連結",
    "時点",
    "JPY",
    "円",
    "1000000",
]


def _tsv_utf16(rows: list[list[str]]) -> bytes:
    text = "\n".join("\t".join(f'"{c}"' for c in row) for row in rows) + "\n"
    return text.encode("utf-16")  # with BOM, like EDINET


def test_reads_utf16_tab_separated() -> None:
    facts = read_facts_csv(_tsv_utf16([HEADER, ROW]))
    assert list(facts.columns[:3]) == ["element_id", "label", "context_id"]
    assert facts.loc[0, "element_id"] == "jppfs_cor:CashAndDeposits"
    assert facts.loc[0, "value"] == "1000000"


def test_reads_utf8_comma_separated_fallback() -> None:
    text = ",".join(HEADER) + "\n" + ",".join(ROW) + "\n"
    facts = read_facts_csv(text.encode("utf-8"))
    assert facts.loc[0, "context_id"] == "CurrentYearInstant"


def test_dash_means_empty() -> None:
    row = [*ROW[:-1], "－"]
    facts = read_facts_csv(_tsv_utf16([HEADER, row]))
    assert facts.loc[0, "value"] == ""


def test_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        read_facts_csv(b"a,b,c\n1,2,3\n")


def test_zip_reads_only_main_report(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "XBRL_TO_CSV/jpcrp030000-asr-001_E99001-000.csv", _tsv_utf16([HEADER, ROW])
        )
        archive.writestr("XBRL_TO_CSV/jpaud-aai-cc-001_E99001-000.csv", _tsv_utf16([HEADER, ROW]))
    path = tmp_path / "doc.zip"
    path.write_bytes(buffer.getvalue())

    facts = read_facts_zip(path)
    assert len(facts) == 1
    assert facts.loc[0, "source_file"].startswith("jpcrp")
