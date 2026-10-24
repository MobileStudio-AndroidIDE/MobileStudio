"""Local calendar skill for RENA-BR cognition.

Stores events as plain JSON in the cognition data directory
(``.rena_calendar.json``).  Writes are read back before reporting success,
so the agent never claims a schedule it did not actually persist.

This is a *skill*, not a hardcoded dispatcher: it is registered through the
normal verification gate and only handles genuine scheduling requests
("내일 3시 회의 일정 잡아줘", "오늘 일정 보여줘").  Missing details produce
an honest clarification question instead of a fabricated event.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

from cognition.util import atomic_write_json, read_json


_DATE_WORDS = {
    "오늘": 0, "내일": 1, "모레": 2,
    "today": 0, "tomorrow": 1, "day after tomorrow": 2,
}

_ADD_CUES = ("잡아", "추가", "등록", "예약", "넣어", "만들",
             "schedule", "add", "create", "book", "remind")

_LIST_CUES = ("보여", "목록", "뭐 있", "알려", "확인",
              "list", "show", "what", "my schedule")


class CalendarResult:
    def __init__(self, ok: bool, output: str = "", error: str = "",
                 data: Optional[dict] = None, confidence: float = 0.5,
                 verified: bool = False):
        self.ok = ok
        self.output = output
        self.error = error
        self.data = data or {}
        self.confidence = confidence
        self.verified = verified

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "data": self.data,
            "confidence": self.confidence,
            "verified": self.verified,
        }

    @classmethod
    def failure(cls, output: str, confidence: float = 0.3) -> "CalendarResult":
        return cls(ok=False, output=output, confidence=confidence)


def _parse_date(text: str) -> date:
    today = date.today()
    lower = text.lower()
    for word, offset in _DATE_WORDS.items():
        if word in lower:
            return today + timedelta(days=offset)
    match = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)),
                        int(match.group(3)))
        except ValueError:
            pass
    match = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if match:
        try:
            return date(today.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            pass
    weekdays = {
        "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3,
        "금요일": 4, "토요일": 5, "일요일": 6,
    }
    for word, target in weekdays.items():
        if word in text:
            delta = (target - today.weekday()) % 7
            return today + timedelta(days=delta)
    return today


def _parse_time(text: str) -> Optional[str]:
    match = re.search(
        r"(오전|오후|am|pm)?\s*(\d{1,2})\s*(?::|시)\s*(\d{1,2})?\s*(?:분)?\s*(am|pm)?",
        text, re.IGNORECASE)
    if not match:
        return None
    meridiem = (match.group(1) or match.group(4) or "").lower()
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    if meridiem in ("오후", "pm") and hour < 12:
        hour += 12
    if meridiem in ("오전", "am") and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def _extract_title(text: str) -> str:
    value = text
    for word in _DATE_WORDS:
        value = re.sub(rf"\b{re.escape(word)}\b", " ", value,
                       flags=re.IGNORECASE)
    value = re.sub(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", " ", value)
    value = re.sub(r"\d{1,2}\s*월\s*\d{1,2}\s*일", " ", value)
    value = re.sub(r"\d{1,2}\s*(?:시|:)\s*\d{0,2}\s*(?:분)?", " ", value)
    value = re.sub(r"(오전|오후|am|pm)", " ", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(이번\s*주|다음\s*주|월요일|화요일|수요일|목요일|금요일|토요일|일요일)",
        " ", value,
    )
    value = re.sub(
        r"(일정|스케줄|캘린더|calendar|schedule|잡아줘|잡아|추가해줘|추가|"
        r"등록해줘|등록|예약해줘|예약|알려줘|보여줘|만들어줘|좀|해줘|해주세요)",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:에|에서|로|를|을|좀)\s+", "", value.strip())
    value = re.sub(r"\s+(?:에|에서|로|를|을|좀)$", "", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" .,:;!?\"'`")
    return value


class CalendarSkill:
    name = "calendar"
    description = "Add and list local calendar events"
    domain = "productivity"
    intents = ("schedule", "calendar", "list_schedule", "add_schedule")

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, ".rena_calendar.json")

    def available(self) -> tuple[bool, str]:
        return True, ""

    def execute(self, text: str) -> CalendarResult:
        lower = text.lower()
        wants_add = any(cue in lower for cue in _ADD_CUES)
        wants_list = any(cue in lower for cue in _LIST_CUES)

        if wants_add and not (
            wants_list
            and not any(cue in lower for cue in ("잡아", "추가", "등록",
                                                 "예약", "add", "schedule"))
        ):
            return self._add(text)
        return self._list(text)

    def describe(self) -> dict:
        available, reason = self.available()
        return {
            "name": self.name,
            "domain": self.domain,
            "intents": list(self.intents),
            "available": available,
            "reason": reason,
            "path": self.path,
        }

    # -- Storage ---------------------------------------------------------

    def _load(self) -> list[dict]:
        data = read_json(self.path, [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def _save(self, events: list[dict]) -> None:
        atomic_write_json(self.path, events)

    # -- Operations ------------------------------------------------------

    def _add(self, text: str) -> CalendarResult:
        title = _extract_title(text)
        event_date = _parse_date(text)
        event_time = _parse_time(text)

        if not title:
            return CalendarResult.failure(
                "무엇을 언제 잡을까요? 예: `내일 3시 회의 일정 잡아줘`처럼 "
                "일정 이름과 시간을 알려주세요.",
                confidence=0.3,
            )

        event = {
            "date": event_date.isoformat(),
            "time": event_time or "all-day",
            "title": title,
            "created": datetime.now().isoformat(timespec="seconds"),
        }

        events = self._load()
        events.append(event)
        try:
            self._save(events)
            reloaded = self._load()
        except Exception as exc:
            return CalendarResult.failure(
                f"일정을 저장하지 못했습니다: {exc}", confidence=0.2)

        if event not in reloaded:
            return CalendarResult.failure(
                "일정 저장을 확인하지 못했습니다.", confidence=0.2)

        when = event["date"]
        if event["time"] != "all-day":
            when += f" {event['time']}"
        else:
            when += " (종일)"

        return CalendarResult(
            ok=True,
            output=("📅 일정을 저장했습니다.\n"
                    f"- {when}: {event['title']}\n"
                    f"- 파일: `.rena_calendar.json`"),
            data={"path": self.path, "event": event},
            confidence=0.95,
            verified=True,
        )

    def _list(self, text: str) -> CalendarResult:
        events = self._load()
        if not events:
            return CalendarResult(
                ok=True,
                output=("📅 저장된 일정이 없습니다. "
                        "`내일 3시 회의 일정 잡아줘`처럼 추가할 수 있어요."),
                data={"path": self.path, "events": []},
                confidence=0.9,
                verified=True,
            )

        lower = text.lower()
        selected = events
        if "오늘" in lower or "today" in lower:
            selected = [e for e in events
                        if e.get("date") == date.today().isoformat()]
        elif "내일" in lower or "tomorrow" in lower:
            selected = [e for e in events
                        if e.get("date")
                        == (date.today() + timedelta(days=1)).isoformat()]

        selected = sorted(selected,
                          key=lambda e: (e.get("date", ""), e.get("time", "")))

        if not selected:
            return CalendarResult(
                ok=True,
                output="📅 해당 날짜에 저장된 일정이 없습니다.",
                data={"path": self.path, "events": []},
                confidence=0.9,
                verified=True,
            )

        lines = ["📅 일정 목록:"]
        for event in selected:
            lines.append(
                f"- {event.get('date', '?')} {event.get('time', 'all-day')}: "
                f"{event.get('title', '(제목 없음)')}"
            )

        return CalendarResult(
            ok=True,
            output="\n".join(lines),
            data={"path": self.path, "events": selected},
            confidence=0.95,
            verified=True,
        )


def quick_calendar(data_dir: str, text: str) -> CalendarResult:
    """Convenience: one-off calendar operation."""
    return CalendarSkill(data_dir).execute(text)
