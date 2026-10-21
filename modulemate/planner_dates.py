"""Literal temporal candidates. No current-date, locale or duration guesses."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {
    name.casefold(): n
    for n in range(1, 13)
    for name in (calendar.month_name[n], calendar.month_abbr[n])
}
MONTHS["sept"] = 9
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?"
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
YEAR = r"(?:1\d|20|21)\d{2}"
WEEKDAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
DAY_NAMES = {
    name.casefold(): WEEKDAYS[i]
    for i in range(7)
    for name in (calendar.day_name[i], calendar.day_abbr[i])
}
DATE_ORDER = ["Require unambiguous dates", "Day / month", "Month / day"]
RELATIVE = re.compile(
    r"\b(?:tomorrow|today|next\s+(?:week|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"|within\s+\d+\s+(?:(?:business|calendar)\s+)?days|in\s+\d+\s+(?:days|weeks)|week\s+\d+|TBA|TBD)\b",
    re.I,
)


def literal(text):
    return " ".join(text.split())


def _day(value):
    return int(re.sub(r"\D", "", value))


def date_mentions(text, *, year=None, date_order=DATE_ORDER[0]):
    """Return exact surface spans with resolved dates or explicit issues.

    Range/list expansion inherits only grammar-local month/year information.
    Missing years and ambiguous numeric order remain unresolved unless supplied.
    """
    found, occupied = [], []

    def put(match, parts, *, inherit_year=None, relation="single"):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            return
        occupied.append((match.start(), match.end()))
        for month, day, explicit_year in parts:
            issues = []
            y = explicit_year or inherit_year or year
            if not y:
                issues.append("The source does not give a year. Supply one before saving.")
            value = None
            try:
                check = date(y or 2000, month, day)
                if y:
                    value = check.isoformat()
            except (TypeError, ValueError):
                issues.append("This date is invalid. Check the original material.")
            found.append(
                {
                    "raw": match.group(),
                    "offset": match.start(),
                    "date": value,
                    "month": month,
                    "day": day,
                    "year_explicit": explicit_year is not None or inherit_year is not None,
                    "relation": relation,
                    "issues": issues,
                }
            )

    # October 28–30, 2026 and 28–30 October 2026.
    patterns = [
        (rf"\b(?P<m>{MONTH})\s+(?P<a>{DAY})\s*[–—-]\s*(?P<b>{DAY})(?:,?\s+(?P<y>{YEAR}))?\b", True),
        (rf"\b(?P<a>{DAY})\s*[–—-]\s*(?P<b>{DAY})\s+(?P<m>{MONTH})(?:,?\s+(?P<y>{YEAR}))?\b", True),
    ]
    for pattern, is_range in patterns:
        for match in re.finditer(pattern, text, re.I):
            m = MONTHS[match["m"].rstrip(".").casefold()]
            y = int(match["y"]) if match["y"] else None
            a, b = _day(match["a"]), _day(match["b"])
            put(match, [(m, a, y), (m, b, y)], relation="range")
            # Both endpoints keep their original surface range; role is chosen by AI.
            if is_range and b < a:
                for item in found[-2:]:
                    item["issues"].append("The range ends before it begins.")
    list_day = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?!\d)"
    list_days = rf"{list_day}(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*){list_day}){{1,11}}"
    for pattern in [
        rf"\b(?P<m>{MONTH})\s+(?P<days>{list_days})(?:,?\s+(?P<y>{YEAR}))?\b",
        rf"\b(?P<days>{list_days})\s+(?P<m>{MONTH})(?:,?\s+(?P<y>{YEAR}))?\b",
    ]:
        for match in re.finditer(pattern, text, re.I):
            m = MONTHS[match["m"].rstrip(".").casefold()]
            y = int(match["y"]) if match["y"] else None
            put(
                match,
                [(m, _day(d), y) for d in re.findall(list_day, match["days"], re.I)],
                relation="list",
            )
    for match in re.finditer(
        rf"(?<!\d)(?P<y>{YEAR})[-/.](?P<m>\d{{1,2}})[-/.](?P<d>\d{{1,2}})(?!\d)", text
    ):
        put(match, [(int(match["m"]), int(match["d"]), int(match["y"]))])
    for pattern in [
        rf"\b(?P<m>{MONTH})\s+(?P<d>{DAY})(?:,?\s+(?P<y>{YEAR}))?\b",
        rf"\b(?P<d>{DAY})\s+(?P<m>{MONTH})(?:,?\s+(?P<y>{YEAR}))?\b",
    ]:
        for match in re.finditer(pattern, text, re.I):
            put(
                match,
                [
                    (
                        MONTHS[match["m"].rstrip(".").casefold()],
                        _day(match["d"]),
                        int(match["y"]) if match["y"] else None,
                    )
                ],
            )
    for match in re.finditer(
        r"(?<![\d/.])(?P<a>\d{1,2})/(?P<b>\d{1,2})(?:/(?P<y>\d{2,4}))?(?![\d/])", text
    ):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            continue
        a, b = int(match["a"]), int(match["b"])
        raw_y = match["y"]
        y = int(raw_y) if raw_y and len(raw_y) == 4 else None
        if a == b:
            month, day = a, b
        elif a > 12 or date_order == "Day / month":
            day, month = a, b
        elif b > 12 or date_order == "Month / day":
            month, day = a, b
        else:
            occupied.append((match.start(), match.end()))
            found.append(
                {
                    "raw": match.group(),
                    "offset": match.start(),
                    "date": None,
                    "month": None,
                    "day": None,
                    "year_explicit": bool(y),
                    "issues": ["Numeric date order is ambiguous. Choose day/month or month/day."],
                }
            )
            continue
        put(match, [(month, day, y)])
        if raw_y and len(raw_y) != 4:
            found[-1]["date"] = None
            found[-1]["issues"].append("Use a four-digit year; the century is not guessed.")
    # The same syntax may be repeated; each occurrence retains its own offset.
    return sorted(found, key=lambda d: (d["offset"], d["date"] or "", d["day"] or 0))


def time_mentions(text):
    found = []
    pattern = re.compile(
        r"(?:(?<![\w:])|(?<=\dT))(?P<h>\d{1,2})(?::(?P<m>\d{2})(?::(?P<s>\d{2}))?)?\s*(?P<p>a\.?m\.?|p\.?m\.?)?(?=$|[^\w]|Z\b)",
        re.I,
    )
    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        prefix = text[max(0, match.start() - 15) : match.start()]
        if re.search(r"(?:UTC|GMT)\s*[+-]$|\dT\d{2}:\d{2}(?::\d{2})?[+-]$", prefix, re.I):
            continue
        if not match["p"]:
            # A range's first hour can inherit a meridian from its second hour.
            following = matches[i + 1] if i + 1 < len(matches) else None
            bridge = text[match.end() : following.start()] if following else ""
            inherited = (
                following["p"] if following and re.fullmatch(r"\s*[–—-]\s*", bridge) else None
            )
            if not inherited and not match["m"]:
                continue
        else:
            inherited = None
        h, m, s = int(match["h"]), int(match["m"] or 0), int(match["s"] or 0)
        if inherited:
            inherited = re.sub(r"\W", "", inherited).lower()
            if h > 12 or match["h"].startswith("0"):
                inherited = None
            else:
                end_h = int(following["h"]) % 12 + (12 if inherited == "pm" else 0)
                same_h = h % 12 + (12 if inherited == "pm" else 0)
                if same_h > end_h:
                    inherited = "am" if inherited == "pm" else "pm"
        meridian = re.sub(r"\W", "", match["p"] or inherited or "").lower()
        issues = []
        if meridian:
            if not 1 <= h <= 12:
                issues.append("The source time is invalid.")
            h = h % 12 + (12 if meridian == "pm" else 0)
        elif not match["m"]:
            continue
        if h > 23 or m > 59 or s > 59:
            issues.append("The source time is invalid (24:00 needs an explicit next-day date).")
        found.append(
            {
                "raw": match.group().strip(),
                "offset": match.start(),
                "time": f"{h:02}:{m:02}" + (f":{s:02}" if s else "") if not issues else None,
                "issues": issues,
            }
        )
    date_spans = [(d["offset"], d["offset"] + len(d["raw"])) for d in date_mentions(text)]
    for match in re.finditer(r"(?<![\w:])(\d{1,2})\s*[–—-]\s*(\d{1,2})(?![\d:])\b", text):
        if any(match.start() < b and match.end() > a for a, b in date_spans) or any(
            match.start() <= t["offset"] < match.end() for t in found
        ):
            continue
        a, b = int(match[1]), int(match[2])
        if max(a, b) < 24 and (max(a, b) > 12 or match[1].startswith("0")):
            found.extend(
                [
                    {
                        "raw": match[n],
                        "offset": match.start(n),
                        "time": f"{int(match[n]):02}:00",
                        "issues": [],
                    }
                    for n in (1, 2)
                ]
            )
        else:
            found.append(
                {
                    "raw": match.group(),
                    "offset": match.start(),
                    "time": None,
                    "issues": ["The time range needs AM/PM or explicit 24-hour times."],
                }
            )
    # Explicit hour with no AM/PM is a real uncertainty, not an all-day event.
    for match in re.finditer(r"\bat\s+(\d{1,2})(?![\d:])\b(?!\s*(?:am|pm|a\.m|p\.m))", text, re.I):
        start = match.start(1)
        if any(x["offset"] == start for x in found):
            continue
        hour = int(match[1])
        if hour < 24:
            found.append(
                {
                    "raw": match[1],
                    "offset": start,
                    "time": f"{hour:02}:00" if hour > 12 else None,
                    "issues": []
                    if hour > 12
                    else ["AM or PM is missing. Set the time before saving."],
                }
            )
    for match in re.finditer(
        r"\b(?:noon|midnight|end of (?:the )?day|EOD|close of business)\b", text, re.I
    ):
        noon = match.group().lower() == "noon"
        found.append(
            {
                "raw": match.group(),
                "offset": match.start(),
                "time": "12:00" if noon else None,
                "issues": []
                if noon
                else ["The day boundary or exact clock time is unclear. Set it before saving."],
            }
        )
    return sorted(found, key=lambda x: x["offset"])


def duration_mentions(text):
    out = []
    for match in re.finditer(
        r"\b(?:lasts?|duration(?:\s+of)?|for)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b",
        text,
        re.I,
    ):
        minutes = float(match[1]) * (60 if match[2].lower().startswith(("h",)) else 1)
        if minutes.is_integer() and 0 < minutes <= 1440:
            out.append({"raw": match.group(), "minutes": int(minutes), "offset": match.start()})
    return out


def source_timezone(text, fallback):
    iso_zone = re.search(r"\dT\d{2}:\d{2}(?::\d{2})?(Z|[+-]\d{2}:?\d{2})\b", text, re.I)
    if iso_zone:
        value = iso_zone[1].upper()
        return ("UTC" if value == "Z" else "UTC" + value[:3] + ":" + value[-2:]), []
    explicit = re.search(
        r"\b(?:Africa|America|Antarctica|Arctic|Asia|Atlantic|Australia|Europe|Indian|Pacific)/[A-Za-z_]+(?:/[A-Za-z_]+)?\b",
        text,
    )
    if explicit:
        return explicit.group(), []
    offset = re.search(r"\b(?:UTC|GMT)\s*([+-])(\d{1,2})(?::?(\d{2}))?\b", text, re.I)
    if offset:
        return f"UTC{offset[1]}{int(offset[2]):02}:{int(offset[3] or 0):02}", []
    zones = re.findall(r"\b(?:UTC|GMT|KST|JST|PST|PDT|EST|EDT|CST|IST|BST|CET|CEST)\b", text)
    mapping = {
        "UTC": "UTC",
        "GMT": "UTC",
        "KST": "Asia/Seoul",
        "JST": "Asia/Tokyo",
        "PST": "UTC-08:00",
        "PDT": "UTC-07:00",
        "EST": "UTC-05:00",
        "EDT": "UTC-04:00",
        "CET": "UTC+01:00",
        "CEST": "UTC+02:00",
    }
    if len(set(zones)) > 1:
        return "", ["More than one source time zone is given. Choose the intended zone."]
    if zones:
        if zones[0] not in mapping:
            return "", [f"{zones[0]} is ambiguous. Choose an IANA zone or explicit UTC offset."]
        return mapping[zones[0]], []
    return fallback, []


def weekday_issue(text, iso, *, offset=None):
    if not iso:
        return []
    # Only a weekday immediately attached to this particular date is a checksum.
    mentions = (
        [{"date": iso, "offset": offset}]
        if offset is not None
        else date_mentions(text, year=date.fromisoformat(iso).year)
    )
    for mention in mentions:
        if mention["date"] != iso:
            continue
        prefix = text[max(0, mention["offset"] - 18) : mention["offset"]]
        match = re.search(
            r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*,?\s*$",
            prefix,
            re.I,
        )
        if match and DAY_NAMES[match[1].casefold()] != WEEKDAYS[date.fromisoformat(iso).weekday()]:
            return ["The weekday and date disagree. Check the original material."]
    return []


def first_occurrence(start, days):
    for offset in range(7):
        candidate = start + timedelta(days=offset)
        if WEEKDAYS[candidate.weekday()] in days:
            return candidate
    raise ValueError("Choose at least one weekday.")
