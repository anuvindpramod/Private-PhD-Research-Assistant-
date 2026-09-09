from __future__ import annotations

import re
from datetime import date


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

ROLLING_PHRASES = (
    "year round",
    "year-round",
    "rolling",
    "open until filled",
    "until filled",
    "reviewed until",
    "applications will be reviewed",
    "no deadline",
)

EXPIRED_PHRASES = (
    "expired",
    "deadline passed",
    "applications closed",
    "closed for applications",
)

UNKNOWN_VALUES = {"", "unknown", "not specified", "not available", "n/a", "na", "none"}


def parse_deadline_text(text: str | None, reference_date: date | None = None) -> date | None:
    """Convert recognized date text into a date object, without deciding inclusion.

    Returns None for missing/rolling text AND for unparseable/invalid dates.
    Numeric day/month dates use day-first order. A missing year uses the
    reference year. is_active_deadline handles the inclusion policy separately.
    """
    if text is None:
        return None
    reference_date = reference_date or date.today()
    normalized = " ".join(str(text).strip().replace(",", " ").split())
    lower = normalized.lower()
    if lower in UNKNOWN_VALUES or any(phrase in lower for phrase in ROLLING_PHRASES):
        return None

    iso_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", normalized)
    if iso_match:
        return _safe_date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    european_numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", normalized)
    if european_numeric:
        year = _normalize_year(european_numeric.group(3))
        return _safe_date(year, int(european_numeric.group(2)), int(european_numeric.group(1)))

    day_month = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s*(20\d{2})?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if day_month:
        month = MONTHS.get(day_month.group(2).lower())
        year = int(day_month.group(3)) if day_month.group(3) else reference_date.year
        if month:
            return _safe_date(year, month, int(day_month.group(1)))

    month_day = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(20\d{2})?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if month_day:
        month = MONTHS.get(month_day.group(1).lower())
        year = int(month_day.group(3)) if month_day.group(3) else reference_date.year
        if month:
            return _safe_date(year, month, int(month_day.group(2)))

    return None


def is_active_deadline(deadline: str | None, reference_date: date | None = None) -> bool:
    """Return whether deadline text passes the extraction deadline filter.

    reference_date is normally the crawl date; omitted means today.
    Order: recognized closure -> reject; missing/rolling -> accept;
    otherwise parse a date and accept only date >= reference_date.
    Missing does not prove open, and this check ignores time of day.
    Rolling phrases currently win over dates in mixed text unless closure is
    recognized. This is a known limitation, not a guarantee of current status.
    """
    reference_date = reference_date or date.today()
    if deadline is None:
        return True
    lower = " ".join(str(deadline).strip().lower().split())
    if any(phrase in lower for phrase in EXPIRED_PHRASES):
        return False
    if lower in UNKNOWN_VALUES or any(phrase in lower for phrase in ROLLING_PHRASES):
        return True
    parsed = parse_deadline_text(deadline, reference_date)
    if parsed is None:
        return False
    return parsed >= reference_date


def normalize_deadline(deadline: str | None, reference_date: date | None = None) -> str:
    """Format a recognized date as YYYY-MM-DD; otherwise retain text or unknown."""
    parsed = parse_deadline_text(deadline, reference_date)
    if parsed is not None:
        return parsed.isoformat()
    value = "" if deadline is None else str(deadline).strip()
    return value or "unknown"


def _normalize_year(value: str) -> int:
    year = int(value)
    if year < 100:
        return 2000 + year
    return year


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
