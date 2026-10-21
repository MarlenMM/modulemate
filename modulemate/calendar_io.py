"""Validated event values and portable calendar snapshots. No calendar account access."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta, timezone
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from modulemate.db import UserError
from modulemate.planner_dates import WEEKDAYS, first_occurrence

KINDS = [
    "Deadline",
    "Exam",
    "Class",
    "Meeting",
    "Workshop",
    "Registration",
    "Release",
    "Window",
    "Other",
]
STATUSES = ["Confirmed", "Tentative"]


def event_zone(name):
    if not isinstance(name, str) or len(name) > 100:
        raise UserError("Choose a valid time zone.")
    fixed = re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", name)
    if fixed:
        hours, minutes = int(fixed[2]), int(fixed[3])
        if hours > 14 or minutes > 59 or (hours == 14 and minutes):
            raise UserError("UTC offsets must be within 14 hours.")
        return timezone(
            (1 if fixed[1] == "+" else -1) * timedelta(hours=hours, minutes=minutes), name
        )
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise UserError("Use an IANA time zone such as Asia/Seoul or UTC-04:00.") from exc


def wall_datetime(day, clock, zone):
    """Reject nonexistent and ambiguous wall times rather than silently use fold=0."""
    naive = datetime.combine(date.fromisoformat(day), time.fromisoformat(clock))
    tz = event_zone(zone)
    a, b = naive.replace(tzinfo=tz, fold=0), naive.replace(tzinfo=tz, fold=1)
    valid = [d for d in (a, b) if d.astimezone(UTC).astimezone(tz).replace(tzinfo=None) == naive]
    if not valid:
        raise UserError("This local time does not exist during the daylight-saving change.")
    if len(valid) == 2 and a.utcoffset() != b.utcoffset():
        raise UserError("This local time occurs twice. Choose an explicit UTC offset.")
    return valid[0]


def normalize_event(raw):
    if not isinstance(raw, dict):
        raise UserError("Invalid planner event.")
    event = dict(raw)
    for key, limit, required in [
        ("title", 200, True),
        ("description", 4000, False),
        ("location", 300, False),
    ]:
        value = raw.get(key, "")
        if (
            not isinstance(value, str)
            or "\0" in value
            or len(value.strip()) > limit
            or (required and not value.strip())
        ):
            raise UserError(
                f"Keep the event {key} within {limit} characters"
                + (" and do not leave it empty." if required else ".")
            )
        event[key] = value.strip()
    if raw.get("kind", "Other") not in KINDS or raw.get("status", "Confirmed") not in STATUSES:
        raise UserError("Choose a supported event type and status.")
    event["kind"], event["status"] = raw.get("kind", "Other"), raw.get("status", "Confirmed")
    for key in ("start_date", "end_date"):
        value = raw.get(key)
        if key == "end_date" and not value:
            event[key] = raw.get("start_date")
            continue
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError
            date.fromisoformat(value)
            event[key] = value
        except (ValueError, TypeError) as exc:
            raise UserError("Set a valid start date and last/end date before saving.") from exc
    for key in ("start_time", "end_time"):
        value = raw.get(key)
        if value in ("", None):
            event[key] = None
            continue
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value):
                raise ValueError
            event[key] = time.fromisoformat(value).isoformat(
                timespec="seconds" if len(value) > 5 else "minutes"
            )
        except (ValueError, TypeError) as exc:
            raise UserError("Use a valid 24-hour time, HH:MM or HH:MM:SS.") from exc
    event["timezone"] = raw.get("timezone", "UTC")
    event_zone(event["timezone"])
    if not event["start_time"]:
        if event["end_time"]:
            raise UserError("Set the start time or clear both times for an all-day event.")
        if event["end_date"] < event["start_date"]:
            raise UserError("The last day cannot be before the first day.")
    else:
        start = wall_datetime(event["start_date"], event["start_time"], event["timezone"])
        if event["end_time"]:
            end = wall_datetime(event["end_date"], event["end_time"], event["timezone"])
            if end.astimezone(UTC) <= start.astimezone(UTC):
                raise UserError(
                    "The end must be later than the start. Set the next-day date for overnight events."
                )
        elif event["end_date"] != event["start_date"]:
            raise UserError("A timed date range also needs an end time.")
    recurrence = raw.get("recurrence") or {}
    if not isinstance(recurrence, dict):
        raise UserError("Invalid recurrence.")
    if recurrence:
        days = recurrence.get("days", [])
        if (
            recurrence.get("frequency") != "weekly"
            or not isinstance(days, list)
            or not days
            or any(d not in WEEKDAYS for d in days)
        ):
            raise UserError("Choose the weekdays for a bounded weekly series.")
        try:
            until = date.fromisoformat(recurrence.get("until", ""))
        except (TypeError, ValueError) as exc:
            raise UserError("Set the final day of the series.") from exc
        start_day = date.fromisoformat(event["start_date"])
        if not start_day <= until <= start_day + timedelta(days=732):
            raise UserError("A weekly series must end within two years of its start.")
        first = first_occurrence(start_day, days)
        if first > until:
            raise UserError("No selected weekday falls within this series.")
        duration = date.fromisoformat(event["end_date"]) - start_day
        if duration.days > 6:
            raise UserError("A weekly occurrence cannot last more than six days.")
        event["start_date"], event["end_date"] = first.isoformat(), (first + duration).isoformat()
        excludes = recurrence.get("exclude", [])
        if not isinstance(excludes, list) or len(excludes) > 100:
            raise UserError("Use at most 100 exception dates.")
        try:
            if any(not start_day <= date.fromisoformat(x) <= until for x in excludes):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise UserError("Exception dates must be valid ISO dates within the series.") from exc
        event["recurrence"] = {
            "frequency": "weekly",
            "days": sorted(set(days), key=WEEKDAYS.index),
            "until": until.isoformat(),
            "exclude": sorted(set(excludes)),
        }
        if event["start_time"]:
            occurrence = first
            while occurrence <= until:
                if (
                    WEEKDAYS[occurrence.weekday()] in days
                    and occurrence.isoformat() not in excludes
                ):
                    wall_datetime(occurrence.isoformat(), event["start_time"], event["timezone"])
                    if event["end_time"]:
                        wall_datetime(
                            (occurrence + duration).isoformat(),
                            event["end_time"],
                            event["timezone"],
                        )
                occurrence += timedelta(days=1)
    else:
        event["recurrence"] = {}
    return event


def event_identity(event):
    event = normalize_event(event)
    if event["start_time"]:
        start = (
            wall_datetime(event["start_date"], event["start_time"], event["timezone"])
            .astimezone(UTC)
            .isoformat()
        )
        end = (
            wall_datetime(event["end_date"], event["end_time"], event["timezone"])
            .astimezone(UTC)
            .isoformat()
            if event["end_time"]
            else None
        )
    else:
        start, end = event["start_date"], event["end_date"]
    payload = [
        event["title"].casefold(),
        event["kind"],
        start,
        end,
        event["location"].casefold(),
        event["recurrence"],
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def event_description(event, include_sources=False):
    text = event.get("description", "")
    if event.get("start_time"):
        text += f"\n\nTime zone: {event['timezone']}."
        if not event.get("end_time"):
            text += " The source gives no end time; no duration is assumed."
    if include_sources:
        for source in event.get("sources", []):
            text += f"\n\nSource: {source.get('document_name', 'Material')} · {source['locator']}\n{source['quote']}"
    return text.strip()


def events_calendar(events, *, include_sources=False, name="ModuleMate events"):
    from icalendar import Calendar, Event, Timezone

    cal = Calendar()
    cal.add("prodid", "-//ModuleMate//Material Planner//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", name[:200])
    zones = {}
    for raw in events:
        event = normalize_event(raw)
        item = Event()
        item.add("uid", f"{event['id']}@modulemate.local")
        stamp = datetime.fromisoformat(
            event.get("updated_at") or "2000-01-01T00:00:00+00:00"
        ).astimezone(UTC)
        item.add("dtstamp", stamp)
        item.add("last-modified", stamp)
        item.add("sequence", int(event.get("sequence", 0)))
        item.add("summary", event["title"])
        item.add("description", event_description(event, include_sources))
        item.add("location", event["location"])
        item.add("class", "PRIVATE")
        item.add("status", event["status"].upper())
        item.add(
            "transp",
            "TRANSPARENT"
            if event["kind"] in {"Deadline", "Release", "Registration", "Window"}
            else "OPAQUE",
        )
        repeating = event["recurrence"]
        if event["start_time"]:
            start = wall_datetime(event["start_date"], event["start_time"], event["timezone"])
            item.add(
                "dtstart",
                start
                if repeating and isinstance(start.tzinfo, ZoneInfo)
                else start.astimezone(UTC),
            )
            if event["end_time"]:
                end = wall_datetime(event["end_date"], event["end_time"], event["timezone"])
                item.add(
                    "dtend",
                    end if repeating and isinstance(end.tzinfo, ZoneInfo) else end.astimezone(UTC),
                )
        else:
            item.add("dtstart", date.fromisoformat(event["start_date"]))
            # The UI's last day is inclusive; iCalendar's DTEND is exclusive.
            item.add("dtend", date.fromisoformat(event["end_date"]) + timedelta(days=1))
        if repeating:
            until_day = date.fromisoformat(repeating["until"])
            until = until_day
            if event["start_time"]:
                # Bound by the last actual occurrence. An arbitrary end-of-day
                # wall clock can itself be nonexistent in a timezone transition.
                last = until_day
                first_day = date.fromisoformat(event["start_date"])
                while last > first_day and (
                    WEEKDAYS[last.weekday()] not in repeating["days"]
                    or last.isoformat() in repeating["exclude"]
                ):
                    last -= timedelta(days=1)
                until = wall_datetime(
                    last.isoformat(), event["start_time"], event["timezone"]
                ).astimezone(UTC)
            item.add("rrule", {"freq": "weekly", "byday": repeating["days"], "until": until})
            for day in repeating["exclude"]:
                exception = (
                    # An excluded nonexistent/ambiguous wall time must still
                    # identify the local recurrence it removes. It is not an
                    # event instant and must not be rejected as one.
                    datetime.combine(
                        date.fromisoformat(day), time.fromisoformat(event["start_time"])
                    ).replace(tzinfo=event_zone(event["timezone"]))
                    if event["start_time"]
                    else date.fromisoformat(day)
                )
                if isinstance(exception, datetime) and not isinstance(exception.tzinfo, ZoneInfo):
                    exception = exception.astimezone(UTC)
                item.add("exdate", exception)
            if event["start_time"] and isinstance(start.tzinfo, ZoneInfo):
                zones.setdefault(event["timezone"], []).extend(
                    [date.fromisoformat(event["start_date"]), until_day]
                )
        cal.add_component(item)
    for zone, days in zones.items():
        cal.add_component(
            Timezone.from_tzinfo(
                ZoneInfo(zone),
                first_date=min(days) - timedelta(days=366),
                last_date=max(days) + timedelta(days=366),
            )
        )
    return cal.to_ical().decode("utf-8")


def google_calendar_link(raw):
    event = normalize_event(raw)
    # A point-in-time deadline has no fabricated end; use the batch file instead.
    if event["recurrence"] or (event["start_time"] and not event["end_time"]):
        return None
    if event["start_time"]:
        stamps = [
            wall_datetime(event[d], event[t], event["timezone"])
            .astimezone(UTC)
            .strftime("%Y%m%dT%H%M%SZ")
            for d, t in [("start_date", "start_time"), ("end_date", "end_time")]
        ]
    else:
        stamps = [
            date.fromisoformat(event["start_date"]).strftime("%Y%m%d"),
            (date.fromisoformat(event["end_date"]) + timedelta(days=1)).strftime("%Y%m%d"),
        ]
    return "https://calendar.google.com/calendar/render?" + urlencode(
        {
            "action": "TEMPLATE",
            "text": event["title"],
            "dates": "/".join(stamps),
            "details": event_description(event),
            "location": event["location"],
        }
    )
