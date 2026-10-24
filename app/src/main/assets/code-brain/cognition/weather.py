"""Weather via wttr.in — no invented temperatures.

Uses the public wttr.in JSON endpoint (format=j1).  On network failure or
missing data it reports unavailable rather than hallucinating a temperature.
All city names are passed through as-is; Korean city names work because
wttr.in accepts Unicode queries.

This is a *tool probe*: it is only advertised as a capability after a
successful live check.  Network access must be enabled in GrowthConfig.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeatherResult:
    location: str
    when: str  # "now" or "tomorrow"
    temperature_c: Optional[str] = None
    feels_like_c: Optional[str] = None
    description: Optional[str] = None
    humidity: Optional[str] = None
    wind_kmph: Optional[str] = None
    precip_mm: Optional[str] = None
    min_c: Optional[str] = None
    max_c: Optional[str] = None
    rain_chance: Optional[str] = None
    source: str = "wttr.in"
    ok: bool = False
    error: str = ""
    latency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "when": self.when,
            "temperature_c": self.temperature_c,
            "feels_like_c": self.feels_like_c,
            "description": self.description,
            "humidity": self.humidity,
            "wind_kmph": self.wind_kmph,
            "precip_mm": self.precip_mm,
            "min_c": self.min_c,
            "max_c": self.max_c,
            "rain_chance": self.rain_chance,
            "source": self.source,
            "ok": self.ok,
            "error": self.error,
            "latency": round(self.latency, 4),
        }


_KOREAN_CITY_MAP = {
    "서울": "Seoul",
    "부산": "Busan",
    "대구": "Daegu",
    "인천": "Incheon",
    "대전": "Daejeon",
    "광주": "Gwangju",
    "울산": "Ulsan",
    "제주": "Jeju",
    "수원": "Suwon",
    "춘천": "Chuncheon",
    "전주": "Jeonju",
    "청주": "Cheongju",
    "포항": "Pohang",
    "강릉": "Gangneung",
    "여수": "Yeosu",
}

# 월(1~12) -> 한국 기후 대략적 기온 범위 (°C).  네트워크가 없을 때 참고용.
_MONTH_TEMP_C = {
    1: (-7, 2),   2: (-5, 5),   3: (1, 11),   4: (7, 17),
    5: (13, 23),  6: (18, 27),  7: (22, 29),  8: (22, 30),
    9: (17, 25), 10: (10, 20), 11: (3, 12),  12: (-3, 5),
}


def resolve_location(text: str) -> str:
    """Return an English-ish city name for wttr.in, defaulting to Seoul."""
    lower = text.lower()
    for korean, english in _KOREAN_CITY_MAP.items():
        if korean in lower:
            return english
    # English city names embedded in the query.
    for english in ("seoul", "busan", "tokyo", "london", "new york", "paris",
                    "daegu", "incheon", "daejeon", "gwangju", "ulsan", "jeju"):
        if english in lower:
            return english.capitalize()
    return "Seoul"


def extract_location(text: str) -> Optional[str]:
    """텍스트에 명시된 도시명을 반환. 없으면 None."""
    for korean in _KOREAN_CITY_MAP:
        if korean in str(text or ""):
            return korean
    return None


def extract_month(text: str) -> Optional[int]:
    """텍스트에서 언급된 월(1~12)을 찾아 반환. 없으면 None."""
    from datetime import date

    lower = text.lower()
    # "오늘"이 언급되면 현재 월로 간주
    if "오늘" in lower:
        return date.today().month
    m = re.search(r"(\d{1,2})\s*월", text)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 12:
                return v
        except ValueError:
            pass
    # 한국어 월명
    korean_months = {
        "한겨울": 1, "입춘": 2, "삼월": 3, "사월": 4, "오월": 5, "육월": 6,
        "칠월": 7, "팔월": 8, "구월": 9, "시월": 10, "십일월": 11, "십이월": 12,
        "늦겨울": 2, "초봄": 3, "중봄": 4, "말봄": 5, "초여름": 6, "중여름": 7,
        "말여름": 8, "초가을": 9, "중가을": 10, "말가을": 11, "초겨울": 12,
    }
    for name, month in korean_months.items():
        if name in text:
            return month
    return None


def estimate_temperature_c(month: Optional[int],
                           raining: bool = False,
                           time_of_day: Optional[str] = None) -> dict:
    """네트워크 없이 월/비/시간으로 기온을 대략 추정.

    Returns dict with keys: low, high, basis, note.
    """
    if month is None:
        # 월을 모르면 현재 월 기준 (사용자에게 물어보라고 안내)
        from datetime import date
        month = date.today().month

    low, high = _MONTH_TEMP_C.get(month, (5, 15))

    # 비가 오면 기온이 조금 내려감
    if raining:
        low -= 2
        high -= 2

    # 시간대 보정
    if time_of_day:
        t = time_of_day.lower()
        if any(k in t for k in ("새벽", "아침", "early", "morning")):
            low -= 3
            high -= 2
        elif any(k in t for k in ("점심", "오후", "noon", "afternoon")):
            low += 2
            high += 3
        elif any(k in t for k in ("저녁", "밤", "evening", "night")):
            low -= 2
            high -= 2

    # 범위 보정
    low = max(-20, min(45, low))
    high = max(-20, min(45, high))
    if low > high:
        low, high = high, low

    return {
        "low": low,
        "high": high,
        "month": month,
        "raining": raining,
        "time_of_day": time_of_day,
        "basis": "monthly_climate_estimate",
        "note": "네트워크 없이 월·비·시간만으로 대략 추정한 값입니다.",
    }


def extract_raining(text: str) -> Optional[bool]:
    """텍스트에서 비 유무를 추론. 참/거짓/모름(None)."""
    lower = text.lower()
    rain_yes = ("비", "rain", "raining", "장마", "흐림", "흐려", "우산")
    rain_no = ("맑음", "sunny", "구름", "clear", "비 안", "안 옴")
    if any(k in lower for k in rain_yes):
        return True
    if any(k in lower for k in rain_no):
        return False
    return None


def extract_time_of_day(text: str) -> Optional[str]:
    """텍스트에서 시간대를 추출."""
    lower = text.lower()
    for key in ("새벽", "아침", "점심", "오후", "저녁", "밤",
                "morning", "noon", "afternoon", "evening", "night"):
        if key in lower:
            return key
    return None


class WeatherContext:
    """날씨 추정 대화 상태를 기억. 네트워크 없이도 사용자 답변을 모아 추측."""

    def __init__(self):
        self.location: Optional[str] = None
        self.month: Optional[int] = None
        self.raining: Optional[bool] = None
        self.time_of_day: Optional[str] = None
        self.asked_clarification: bool = False
        self.ready: bool = False

    def update_from_text(self, text: str) -> None:
        if self.location is None:
            # 도시 추출 시도
            for korean in _KOREAN_CITY_MAP:
                if korean in text:
                    self.location = korean
                    break
        m = extract_month(text)
        if m is not None:
            self.month = m
        r = extract_raining(text)
        if r is not None:
            self.raining = r
        t = extract_time_of_day(text)
        if t is not None:
            self.time_of_day = t

    def is_ready(self) -> bool:
        return self.location is not None and self.month is not None

    def estimate(self) -> dict:
        return estimate_temperature_c(
            self.month, raining=self.raining or False,
            time_of_day=self.time_of_day)

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "month": self.month,
            "raining": self.raining,
            "time_of_day": self.time_of_day,
            "asked_clarification": self.asked_clarification,
            "ready": self.is_ready(),
        }



def is_tomorrow(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ("내일", "tomorrow"))


class WeatherClient:
    """wttr.in client with a browser user agent."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._user_agent = "RENA-BR/1.0 (cognition; weather tool probe)"

    def _get_json(self, url: str) -> Optional[dict]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8", "replace"))
        except (urllib.error.URLError, urllib.error.HTTPError,
                OSError, json.JSONDecodeError, ValueError) as exc:
            return None

    def current(self, location: str) -> WeatherResult:
        started = time.time()
        url = "https://wttr.in/%s?format=j1" % urllib.parse.quote(location)
        data = self._get_json(url)
        result = WeatherResult(location=location, when="now",
                               latency=time.time() - started)
        if data is None:
            result.error = "network or service unavailable"
            return result

        current = (data.get("current_condition") or [{}])[0]
        nearest = (data.get("nearest_area") or [{}])[0]
        area = location
        try:
            nearest_name = nearest["areaName"][0]["value"]
            if location.lower() not in {n.lower() for n in _KOREAN_CITY_MAP.values()}:
                area = nearest_name
        except Exception:
            pass
        result.location = area

        try:
            result.temperature_c = current.get("temp_C", "?")
            result.feels_like_c = current.get("FeelsLikeC", "?")
            result.description = current["weatherDesc"][0]["value"] \
                if current.get("weatherDesc") else "날씨 정보"
            result.humidity = current.get("humidity", "?")
            result.wind_kmph = current.get("windspeedKmph", "?")
            result.precip_mm = current.get("precipMM", "?")
            result.ok = True
        except Exception as exc:
            result.error = f"parse error: {exc}"

        return result

    def tomorrow(self, location: str) -> WeatherResult:
        started = time.time()
        url = "https://wttr.in/%s?format=j1" % urllib.parse.quote(location)
        data = self._get_json(url)
        result = WeatherResult(location=location, when="tomorrow",
                               latency=time.time() - started)
        if data is None:
            result.error = "network or service unavailable"
            return result

        days = data.get("weather") or []
        if len(days) < 2:
            result.error = "tomorrow forecast not available"
            return result

        day = days[1]
        nearest = (data.get("nearest_area") or [{}])[0]
        area = location
        try:
            nearest_name = nearest["areaName"][0]["value"]
            if location.lower() not in {n.lower() for n in _KOREAN_CITY_MAP.values()}:
                area = nearest_name
        except Exception:
            pass
        result.location = area

        try:
            desc = ""
            try:
                desc = day["hourly"][4]["weatherDesc"][0]["value"]
            except Exception:
                pass
            result.min_c = day.get("mintempC", "?")
            result.max_c = day.get("maxtempC", "?")
            result.description = desc or "예보"
            result.rain_chance = day.get("chanceofrain", "?")
            result.ok = True
        except Exception as exc:
            result.error = f"parse error: {exc}"

        return result

    def forecast(self, text: str) -> WeatherResult:
        location = resolve_location(text)
        if is_tomorrow(text):
            return self.tomorrow(location)
        return self.current(location)


def quick_weather(text: str) -> WeatherResult:
    """Convenience: one-off weather lookup."""
    return WeatherClient().forecast(text)
