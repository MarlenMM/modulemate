"""Deterministic practice, review scheduling and capacity-aware planning."""

import re
from datetime import UTC, date, datetime, timedelta

from modulemate.db import UserError
from modulemate.search import STOPWORDS


def schedule_review(interval, repetitions, rating):
    if rating == "Again":
        return 1, 0
    if rating == "Hard":
        return max(1, min(365, round(max(interval, 1) * 1.2))), repetitions + 1
    if rating == "Good":
        return (
            1
            if repetitions == 0
            else 3
            if repetitions == 1
            else min(365, max(1, round(interval * 2)))
        ), repetitions + 1
    if rating == "Easy":
        return min(365, max(4, round(max(interval, 1) * 3))), repetitions + 1
    raise UserError("Choose Again, Hard, Good or Easy.")


def cloze_cards(passages):
    cards, seen = [], set()
    for passage in passages:
        # Rejoin wrapped prose, skip headings, and prefer definitions over boilerplate.
        paragraphs = re.split(r"\n\s*\n", passage.text)
        prose = " ".join(
            " ".join(
                line.strip() for line in para.splitlines() if not line.lstrip().startswith("#")
            )
            for para in paragraphs
        )
        sentences = re.split(r"(?<=[.!?。])\s+", prose)
        sentences.sort(key=lambda text: not bool(re.search(r"\b(means|refers to)\b", text, re.I)))
        for sentence in sentences:
            sentence = sentence.strip()
            if not 35 <= len(sentence) <= 350 or sentence in seen:
                continue
            if re.search(
                r"\b(fictional|demonstration|copyright|modulemate|permission|university|institution)\b",
                sentence,
                re.I,
            ):
                continue
            words = re.findall(r"\b[^\W\d_]{5,}\b", sentence)
            words = [word for word in words if word.casefold() not in STOPWORDS]
            if not words:
                continue
            word = max(words, key=len)
            front = "Fill the blank:\n" + re.sub(
                r"\b" + re.escape(word) + r"\b", "_____", sentence, flags=re.I
            )
            cards.append(
                (
                    not bool(re.search(r"\b(means|refers to)\b", sentence, re.I)),
                    passage.id,
                    {
                        "front": front,
                        "back": word + "\n\n" + sentence,
                        "document_id": passage.document_id,
                        "locator": passage.locator,
                    },
                )
            )
            seen.add(sentence)
            break
    # An overview samples locations rather than returning document order. Rank
    # definitions globally so an assessment footer cannot displace the lesson.
    cards.sort(key=lambda item: item[:2])
    return [card for _, _, card in cards[:6]]


def plan_tasks(tasks, daily_minutes=60, horizon=7, *, today=None):
    if not 5 <= daily_minutes <= 480 or not 1 <= horizon <= 30:
        raise UserError("Choose 5–480 minutes per day and a 1–30 day planning window.")
    today = today or date.today()
    days = [today + timedelta(days=i) for i in range(horizon)]
    remaining = {day: daily_minutes for day in days}
    sessions, warnings = [], []
    active = sorted(
        (t for t in tasks if not t["done"]), key=lambda t: (t["due"], t["created_at"], t["id"])
    )
    for task in active:
        due = date.fromisoformat(task["due"])
        needed = task["minutes"]
        if due < today:
            warnings.append(f"Overdue: {task['title']} (due {due.isoformat()}).")
        for day in days:
            if day > max(due, today):
                break
            amount = min(remaining[day], needed, 50)
            # At most 50-minute blocks; several blocks may share the same day.
            while amount > 0:
                sessions.append(
                    {
                        "date": day.isoformat(),
                        "title": task["title"],
                        "minutes": amount,
                        "task_id": task["id"],
                    }
                )
                needed -= amount
                remaining[day] -= amount
                amount = min(remaining[day], needed, 50)
            if not needed:
                break
        if needed:
            warnings.append(
                f"{task['title']}: {needed} minutes could not fit before the deadline / end of this window."
            )
    return sessions, warnings


def _ics_escape(value):
    return (
        value.replace("\\", "\\\\")
        .replace("\r", "")
        .replace("\n", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def _fold(line):
    # RFC 5545 uses octets, not Unicode character counts. Never split a codepoint.
    lines, current = [], ""
    for char in line:
        if len((current + char).encode()) > 73:
            lines.append(current)
            current = " "
        current += char
    lines.append(current)
    return "\r\n".join(lines)


def calendar_export(tasks):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//ModuleMate//Study Planner//EN",
        "CALSCALE:GREGORIAN",
    ]
    for task in tasks:
        if task["done"]:
            continue
        due = date.fromisoformat(task["due"])
        lines += [
            "BEGIN:VEVENT",
            f"UID:{task['id']}@modulemate.local",
            f"DTSTAMP:{stamp}",
            "DTSTART;VALUE=DATE:" + due.strftime("%Y%m%d"),
            "DTEND;VALUE=DATE:" + (due + timedelta(days=1)).strftime("%Y%m%d"),
            "SUMMARY:" + _ics_escape(task["title"]),
            "DESCRIPTION:" + _ics_escape(f"Estimated study time: {task['minutes']} minutes."),
            "END:VEVENT",
        ]
    return "\r\n".join(_fold(line) for line in [*lines, "END:VCALENDAR"]) + "\r\n"
