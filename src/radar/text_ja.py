"""Japanese text utilities: dates (including the imperial era calendar), names, entity types."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from datetime import date

# Imperial eras: first day of each era in the Gregorian calendar.
_ERAS: dict[str, date] = {
    "明治": date(1868, 1, 25),
    "大正": date(1912, 7, 30),
    "昭和": date(1926, 12, 25),
    "平成": date(1989, 1, 8),
    "令和": date(2019, 5, 1),
}

_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$")
_SEIREKI = re.compile(r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*$")
_WAREKI = re.compile(
    r"^\s*(明治|大正|昭和|平成|令和)\s*(\d{1,2}|元)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*$"
)


def normalize(text: str) -> str:
    """NFKC-normalise: full-width digits/letters → half-width, compatibility forms unified."""
    return unicodedata.normalize("NFKC", text).strip()


def parse_japanese_date(value: str | None) -> date | None:
    """Parse ISO ('1950-01-02'), Gregorian ('1950年1月2日') or era ('昭和25年1月2日') dates.

    Returns None for empty or unparseable input instead of raising: filings are messy and a missing
    birth date must not crash a pipeline run (it is reported in data-quality checks).
    """
    if value is None:
        return None
    text = normalize(str(value))
    if not text:
        return None

    for pattern in (_ISO, _SEIREKI):
        match = pattern.match(text)
        if match:
            y, m, d = (int(g) for g in match.groups())
            return _safe_date(y, m, d)

    match = _WAREKI.match(text)
    if match:
        era, year_str, m_str, d_str = match.groups()
        era_year = 1 if year_str == "元" else int(year_str)
        year = _ERAS[era].year + era_year - 1
        return _safe_date(year, int(m_str), int(d_str))
    return None


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def age_on(birth: date | None, on: date) -> int | None:
    """Completed years of age on a given date."""
    if birth is None:
        return None
    had_birthday = (on.month, on.day) >= (birth.month, birth.day)
    return on.year - birth.year - (0 if had_birthday else 1)


# Markers that identify a shareholder as an organisation rather than a natural person.
# Japanese markers match as substrings; Latin markers must match as whole words
# (otherwise "AG" would match inside "NAGANO").
_ORG_MARKERS_JA = (
    "株式会社",
    "有限会社",
    "合同会社",
    "合資会社",
    "合名会社",
    "(株)",
    "(有)",
    "銀行",
    "信託",
    "証券",
    "保険",
    "生命",
    "損害",
    "組合",
    "財団",
    "社団",
    "協会",
    "基金",
    "ファンド",
    "ホールディングス",
    "持株会",
    "自己株",
)
_ORG_MARKERS_LATIN = re.compile(
    r"\b(LIMITED|LTD|INC|CORP|CORPORATION|CO|LLC|LP|L\.P|PLC|FUND|TRUST|BANK|NOMINEES?"
    r"|S\.A|AG|GMBH|SA|NV|BV|PTE|HOLDINGS?)\b"
)


def is_organization(name: str) -> bool:
    """Heuristic: does this shareholder name look like a company/fund rather than a person?"""
    text = normalize(name).upper()
    if any(marker in text for marker in _ORG_MARKERS_JA):
        return True
    return bool(_ORG_MARKERS_LATIN.search(text))


def join_spaced_characters(name: str) -> str:
    """'伊 藤 浩 一' → '伊藤浩一'; any other name is returned normalised but unchanged."""
    text = normalize(name)
    parts = text.split()
    if len(parts) > 2 and all(len(part) == 1 for part in parts):
        return "".join(parts)
    return text


# Joined names ('伊藤浩一') are split only with a known surname of at least this length, leaving a
# given name of 1–3 characters: one-character surnames (林, 森) would match far too much.
_MIN_GUESSED_SURNAME = 2
_MAX_GIVEN_NAME = 3


def split_surname(full_name: str, known_surnames: frozenset[str] = frozenset()) -> str | None:
    """Surname of a Japanese personal name written as '山田 太郎' / '山田 太郎'."""
    # Remove annotations sometimes appended in filings, e.g. '山田 太郎(注)1', BEFORE joining
    # spaced characters: '伊 藤 浩 一(注)1' would otherwise not look per-character spaced.
    text = re.sub(r"[(（].*?[)）]\d*", "", normalize(full_name)).strip()
    text = join_spaced_characters(text)
    parts = text.split()
    if len(parts) >= 2:
        return parts[0]
    for length in range(len(text) - 1, _MIN_GUESSED_SURNAME - 1, -1):
        if text[:length] in known_surnames and len(text) - length <= _MAX_GIVEN_NAME:
            return text[:length]
    return None


def surname_dictionary(names: Iterable[str]) -> frozenset[str]:
    """Surnames (≥2 characters) of the names written WITH a separator ('伊藤 二郎' → '伊藤')."""
    surnames = (split_surname(name) for name in names)
    return frozenset(s for s in surnames if s and len(s) >= _MIN_GUESSED_SURNAME)


# Old (旧字体) and variant forms of kanji common in surnames → the standard form. NFKC does not
# unify them, yet one family writes both (廣澤 清 and 広沢 かほる in the same filing). Only true
# variants: 斉 and 斎 are different surnames (斉藤 / 斎藤) and stay apart.
_KANJI_VARIANTS = str.maketrans(
    {
        "廣": "広", "澤": "沢", "邊": "辺", "邉": "辺", "齋": "斎", "齊": "斉", "髙": "高",
        "﨑": "崎", "嵜": "崎", "濱": "浜", "濵": "浜", "櫻": "桜", "國": "国", "嶋": "島",
        "嶌": "島", "龍": "竜", "瀨": "瀬", "德": "徳", "藏": "蔵", "眞": "真", "惠": "恵",
        "條": "条", "實": "実", "壽": "寿", "與": "与", "來": "来", "冨": "富", "𠮷": "吉",
        "淺": "浅", "會": "会", "顯": "顕", "槇": "槙",
    }
)  # fmt: skip


def unify_kanji_variants(text: str) -> str:
    """'廣澤' → '広沢': for comparing names only, never for display."""
    return text.translate(_KANJI_VARIANTS)
