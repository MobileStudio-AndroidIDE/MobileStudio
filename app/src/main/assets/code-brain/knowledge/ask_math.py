"""Natural-language math/astronomy question answering (Korean + English).

Public API
----------
``answer(question, now=None, tz="Asia/Seoul") -> dict`` with keys
``domain`` ("math" | "astronomy" | "unit" | "date" | "unknown"),
``answer_text_ko``, ``structured`` and ``confidence``.

Everything is computed through :mod:`knowledge.mathcore` and
:mod:`knowledge.astronomy`; regexes only recognise syntax, never answers.

Timezone note
-------------
Local clock times for solar events use a fixed UTC offset (Asia/Seoul = +9h,
which is exact because Korea has no DST).  Other names fall back to a small
fixed table and then to :mod:`zoneinfo`; the chosen offset is recorded in
``structured["tz_offset_hours"]``.  All astronomy algorithms are approximate
(Meeus low-precision), so solar times carry ~1 minute class uncertainty.
"""

from __future__ import annotations

import datetime as _dt
import math
import re

from knowledge import astronomy as ast
from knowledge import mathcore as mc

__all__ = ["answer"]

_TZ_FIXED = {
    "asia/seoul": 9.0, "kst": 9.0, "asia/tokyo": 9.0, "jst": 9.0,
    "utc": 0.0, "gmt": 0.0, "europe/london": 0.0,
}
_TZ_LABELS = {"asia/seoul": "KST", "kst": "KST", "asia/tokyo": "JST",
              "jst": "JST", "utc": "UTC", "gmt": "UTC",
              "europe/london": "GMT"}

_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


def _numbers(text):
    out = []
    for tok in _NUM_RE.findall(text):
        try:
            out.append(float(tok) if ("." in tok or "e" in tok.lower()) else int(tok))
        except ValueError:
            continue
    return out


def _fraction_val(x):
    if isinstance(x, mc.Fraction):
        return x.numerator if x.denominator == 1 else float(x)
    return x


def _fmt(x, sig=4):
    """2-4 significant digit display helper (exact ints stay exact)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, mc.Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        x = float(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    return f"{v:.{sig}g}"


def _result(domain, text, structured, confidence):
    return {"domain": domain, "answer_text_ko": text,
            "structured": structured, "confidence": float(confidence)}


def _unknown():
    return _result(
        "unknown",
        "죄송합니다. 이 질문을 수학이나 천문학 문제로 해석하지 못했습니다. "
        "예: '48의 약수', '1.5 mile은 몇 km', '오늘 일몰 시간'.",
        {},
        0.0,
    )


def _normalize_now(now):
    if now is None:
        return _dt.datetime.now(_dt.timezone.utc)
    if not isinstance(now, _dt.datetime):
        raise ValueError("now must be a datetime")
    if now.tzinfo is None:
        return now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc)


def _tz_offset(tz, when):
    key = (tz or "Asia/Seoul").strip().lower()
    if key in _TZ_FIXED:
        return _TZ_FIXED[key], _TZ_LABELS.get(key, tz), True
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(tz)
        off = when.astimezone(zone).utcoffset()
        if off is not None:
            return off.total_seconds() / 3600.0, tz, False
    except Exception:
        pass
    return 9.0, "KST", True


def _local(dt, off_hours):
    return dt + _dt.timedelta(hours=off_hours)


def _iso(dt):
    return dt.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Astronomy handlers
# ---------------------------------------------------------------------------

_CITIES = (
    ("서울", 37.5665, 126.9780), ("seoul", 37.5665, 126.9780),
    ("부산", 35.1796, 129.0756), ("busan", 35.1796, 129.0756),
    ("도쿄", 35.6762, 139.6503), ("tokyo", 35.6762, 139.6503),
    ("런던", 51.5074, -0.1278), ("london", 51.5074, -0.1278),
    ("뉴욕", 40.7128, -74.0060), ("new york", 40.7128, -74.0060),
)
_DEFAULT_CITY = ("서울", 37.5665, 126.9780)

_PLANET_WORDS = (
    ("수성", "mercury"), ("금성", "venus"), ("화성", "mars"),
    ("목성", "jupiter"), ("토성", "saturn"), ("샛별", "venus"),
    ("mercury", "mercury"), ("venus", "venus"), ("mars", "mars"),
    ("jupiter", "jupiter"), ("saturn", "saturn"),
)


def _find_city(q):
    low = q.lower()
    best = _DEFAULT_CITY
    pos = len(q)
    for name, lat, lon in _CITIES:
        i = low.find(name.lower())
        if i >= 0 and i < pos:
            best, pos = (name, lat, lon), i
    return best


def _extract_date(q, base_date):
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", q)
    if m:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", q)
    if m:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
    low = q.lower()
    words = (("내일", 1), ("모레", 2), ("어제", -1), ("오늘", 0),
             ("tomorrow", 1), ("yesterday", -1), ("today", 0))
    for word, delta in words:
        if word in low:
            return base_date + _dt.timedelta(days=delta), True
    return base_date, False


def _h_sun_rise_set(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("일출", "일몰", "해가 뜨", "해 뜨는", "해가 지", "해 지는")) \
            and not any(w in low for w in ("sunrise", "sunset", "sun rises", "sun sets")):
        return None
    is_set = any(w in q for w in ("일몰", "해가 지", "해 지는")) or any(
        w in low for w in ("sunset", "sun sets"))
    event = "sunset" if is_set else "sunrise"
    city, lat, lon = _find_city(q)
    base = (now + _dt.timedelta(hours=off)).date()
    date, explicit = _extract_date(q, base)
    jd0 = ast.julian_date(_dt.datetime(date.year, date.month, date.day, 0))
    rs = ast.sun_rise_set(jd0, lat, lon)
    dt = rs.set if event == "sunset" else rs.rise
    event_ko = "일몰" if event == "sunset" else "일출"
    struct = {"event": event, "city": city, "latitude": lat, "longitude": lon,
              "date": date.isoformat(), "explicit_date": explicit, "polar": rs.polar}
    if dt is None:
        if rs.polar == "day":
            text = f"{city}에서는 {date.isoformat()}에 해가 지지 않습니다 (백야)."
        else:
            text = f"{city}에서는 {date.isoformat()}에 해가 뜨지 않습니다 (극야)."
        return _result("astronomy", text, struct, 0.6)
    local = _local(dt, off)
    struct["time_utc"] = _iso(dt)
    struct["time_local"] = _iso(local)
    struct["tz_offset_hours"] = off
    text = (f"{city}의 {date.year}년 {date.month}월 {date.day}일 {event_ko} 시각은 "
            f"{local.hour:02d}:{local.minute:02d} ({label})경입니다.")
    return _result("astronomy", text, struct, 0.85)


_MOON_PHASE_WORDS = (
    ("보름달", "full"), ("만월", "full"), ("full moon", "full"), ("full-moon", "full"),
    ("상현달", "first_quarter"), ("상현", "first_quarter"), ("first quarter", "first_quarter"),
    ("하현달", "last_quarter"), ("하현", "last_quarter"), ("last quarter", "last_quarter"),
)
_MOON_PHASE_KO = {"full": "보름달", "new": "삭", "first_quarter": "상현달",
                  "last_quarter": "하현달"}


def _h_next_moon_phase(q, now, off, label):
    low = q.lower()
    kind = None
    for word, k in _MOON_PHASE_WORDS:
        if word in q or word in low:
            kind = k
            break
    if kind is None and re.search(r"(?<![가-힣])삭(?![가-힣])", q):
        kind = "new"
    if kind is None and any(w in low for w in ("new moon", "new-moon")):
        kind = "new"
    if kind is None:
        return None
    dt = ast.next_phase(kind, now)
    local = _local(dt, off)
    ko = _MOON_PHASE_KO[kind]
    struct = {"kind": kind, "time_utc": _iso(dt), "time_local": _iso(local),
              "tz_offset_hours": off}
    text = (f"다음 {ko}은 {local.year}년 {local.month}월 {local.day}일 "
            f"{local.hour:02d}:{local.minute:02d} ({label})경입니다.")
    return _result("astronomy", text, struct, 0.9)


def _h_moon_phase_now(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("달 위상", "달의 위상", "달 모양", "달의 모양",
                                "지금 달", "현재 달")) \
            and not any(w in low for w in ("moon phase", "phase of the moon", "lunar phase")):
        return None
    jd = ast.julian_date(now)
    ph = ast.moon_phase(jd)
    struct = {"illumination": ph.illumination, "phase_angle_deg": ph.phase_angle,
              "elongation_deg": ph.elongation, "age_days": ph.age_days,
              "name": ph.name, "time_utc": _iso(now)}
    text = (f"지금 달은 '{ph.name}' 단계이며, 조도는 약 {_fmt(ph.illumination * 100, 3)}%, "
            f"월령은 약 {_fmt(ph.age_days, 3)}일입니다.")
    return _result("astronomy", text, struct, 0.9)


def _planets_in_order(q):
    low = q.lower()
    hits = []
    for word, name in _PLANET_WORDS:
        for m in re.finditer(re.escape(word), q if word in q else low):
            hits.append((m.start(), name))
    hits.sort()
    seen, out = set(), []
    for _, name in hits:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _planet_ko(name):
    return ast._PLANET_NAMES_KO.get(name, name)


def _h_planet_separation(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("각거리", "각도", "사이", "거리")) \
            and not any(w in low for w in ("separation", "apart", "angular distance")):
        return None
    names = _planets_in_order(q)
    if len(names) < 2:
        return None
    jd = ast.julian_date(now)
    p1 = ast.planet_position(names[0], jd)
    p2 = ast.planet_position(names[1], jd)
    sep = ast.angular_separation(p1.ra, p1.dec, p2.ra, p2.dec)
    struct = {"planet1": names[0], "planet2": names[1], "separation_deg": sep,
              "ra1": p1.ra, "dec1": p1.dec, "ra2": p2.ra, "dec2": p2.dec,
              "time_utc": _iso(now)}
    text = (f"{_planet_ko(names[0])}과 {_planet_ko(names[1])}의 현재 각거리는 "
            f"약 {_fmt(sep)}°입니다.")
    return _result("astronomy", text, struct, 0.88)


def _h_planet_position(q, now, off, label):
    low = q.lower()
    names = _planets_in_order(q)
    if not names:
        return None
    intent = any(w in q for w in ("지금", "현재", "어디", "위치")) or \
        any(w in low for w in ("where", "position", "now", "tonight"))
    if not intent and any(w in q for w in ("각거리",)) :
        return None
    name = names[0]
    jd = ast.julian_date(now)
    p = ast.planet_position(name, jd)
    struct = {"name": name, "ra_deg": p.ra, "dec_deg": p.dec, "distance_au": p.distance,
              "magnitude": p.magnitude, "elongation_deg": p.elongation,
              "retrograde": bool(p.retrograde), "time_utc": _iso(now)}
    retro = " (현재 역행 중)" if p.retrograde else ""
    text = (f"{_planet_ko(name)}은 현재 적경 {_fmt(p.ra)}°, 적위 {_fmt(p.dec)}° "
            f"방향, 거리 약 {_fmt(p.distance)} AU (등급 {_fmt(p.magnitude)}등급){retro}에 있습니다.")
    return _result("astronomy", text, struct, 0.88)


def _h_sun_position_date(q, now, off, label):
    low = q.lower()
    if not ("태양" in q or "sun" in low):
        return None
    base = (now + _dt.timedelta(hours=off)).date()
    date, explicit = _extract_date(q, base)
    if not explicit:
        return None
    jd = ast.julian_date(_dt.datetime(date.year, date.month, date.day, 0))
    sp = ast.sun_position(jd)
    struct = {"date": date.isoformat(), "ra_deg": sp.ra, "dec_deg": sp.dec,
              "lon_deg": sp.lon, "distance_au": sp.distance, "time_utc": _iso(now)}
    text = (f"{date.year}년 {date.month}월 {date.day}일 00:00 UTC의 태양 위치는 "
            f"적경 {_fmt(sp.ra)}°, 적위 {_fmt(sp.dec)}°입니다.")
    return _result("astronomy", text, struct, 0.9)


def _h_sun_position_now(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("태양 적경", "태양의 적경", "태양 위치", "태양의 위치")) \
            and not any(w in low for w in ("sun position", "position of the sun",
                                           "sun's position", "sun ra", "sun dec")):
        return None
    jd = ast.julian_date(now)
    sp = ast.sun_position(jd)
    struct = {"ra_deg": sp.ra, "dec_deg": sp.dec, "lon_deg": sp.lon,
              "distance_au": sp.distance, "time_utc": _iso(now)}
    text = (f"현재 태양은 적경 {_fmt(sp.ra)}°, 적위 {_fmt(sp.dec)}° "
            f"(거리 약 {_fmt(sp.distance)} AU)에 있습니다.")
    return _result("astronomy", text, struct, 0.92)


_SEASONS = (
    ("춘분", "spring"), ("봄분점", "spring"), ("봄 분점", "spring"),
    ("하지", "summer"), ("하지점", "summer"),
    ("추분", "autumn"), ("가을분점", "autumn"), ("가을 분점", "autumn"),
    ("동지", "winter"), ("겨울점", "winter"),
    ("spring equinox", "spring"), ("march equinox", "spring"),
    ("summer solstice", "summer"), ("june solstice", "summer"),
    ("autumn equinox", "autumn"), ("fall equinox", "autumn"),
    ("september equinox", "autumn"), ("winter solstice", "winter"),
    ("december solstice", "winter"),
)
_SEASON_KO = {"spring": "춘분", "summer": "하지", "autumn": "추분", "winter": "동지"}


def _h_season(q, now, off, label):
    low = q.lower()
    kind = None
    for word, k in _SEASONS:
        if word in q or word in low:
            kind = k
            break
    if kind is None:
        return None
    m = re.search(r"(\d{4})\s*년", q) or re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", q)
    if m:
        year = int(m.group(1))
    else:
        year = int(re.search(r"(\d{4})", q).group(1)) if re.search(r"(\d{4})", q) \
            else (now + _dt.timedelta(hours=off)).year
    instants = ast.season_instants(year)
    dt = getattr(instants, kind)
    local = _local(dt, off)
    struct = {"kind": kind, "year": year, "time_utc": _iso(dt),
              "time_local": _iso(local), "tz_offset_hours": off}
    ko = _SEASON_KO[kind]
    text = (f"{year}년 {ko}{_josa(ko, '은', '는')} {local.year}년 {local.month}월 {local.day}일 "
            f"{local.hour:02d}:{local.minute:02d} ({label})경입니다.")
    return _result("astronomy", text, struct, 0.9)


def _h_eclipse(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("일식", "월식")) \
            and not any(w in low for w in ("solar eclipse", "lunar eclipse")):
        return None
    kind = "solar" if ("일식" in q or "solar eclipse" in low) else "lunar"
    ko = "일식" if kind == "solar" else "월식"
    eclipse = ast.find_next_eclipse(kind, now)
    if eclipse is None:
        return _result("astronomy", f"앞으로 약 800일 안에는 {ko}이 없습니다.", {}, 0.4)
    dt = eclipse.datetime
    local = _local(dt, off)
    struct = {"kind": kind, "time_utc": _iso(dt), "time_local": _iso(local),
              "node_distance_deg": eclipse.node_distance,
              "moon_latitude_deg": eclipse.moon_latitude,
              "tz_offset_hours": off, "approximate": True}
    text = (f"다음 {ko}은 {local.year}년 {local.month}월 {local.day}일 "
            f"{local.hour:02d}:{local.minute:02d} ({label})경으로 계산됩니다 "
            f"(근사 계산, 오차 하루 이내).")
    return _result("astronomy", text, struct, 0.6)


# ---------------------------------------------------------------------------
# Math handlers
# ---------------------------------------------------------------------------

def _h_roman(q, now, off, label):
    low = q.lower()
    if not any(w in q for w in ("로마", "로만")) and "roman" not in low:
        return None
    m = re.search(r"(\d+)", q)
    if m:
        n = int(m.group(1))
        roman = mc.to_roman(n)
        struct = {"value": n, "roman": roman}
        return _result("math",
                       f"{n}{_josa(str(n), '을', '를')} 로마 숫자로 나타내면 {roman}입니다.",
                       struct, 0.95)
    m = re.search(r"(?<![A-Za-z])([IVXLCDM]{1,15})(?![A-Za-z])", q, re.I)
    if m:
        roman = m.group(1).upper()
        try:
            n = mc.from_roman(roman)
        except ValueError:
            return None
        struct = {"value": n, "roman": roman}
        return _result("math", f"로마 숫자 {roman}은(는) {n}입니다.", struct, 0.95)
    return None


_BASE_WORDS = {"binary": 2, "octal": 8, "hex": 16, "hexadecimal": 16,
               "decimal": 10, "2진": 2, "8진": 8, "16진": 16, "10진": 10}


def _h_bases(q, now, off, label):
    low = q.lower()
    # Korean: 255를 16진수로 / 255을 2진수로
    m = re.search(r"(\d+)\s*(?:을|를)?\s*(\d+)\s*진수\s*(?:로|으로)", q)
    if m:
        n, base = int(m.group(1)), int(m.group(2))
        try:
            digits = mc.int_to_base(n, base)
        except ValueError:
            return None
        struct = {"value": n, "base": base, "digits": digits}
        return _result("unit",
                       f"{n}{_josa(str(n), '을', '를')} {base}진수로 나타내면 {digits}입니다.",
                       struct, 0.95)
    # Korean: 2진수 1101은 십진수로
    m = re.search(r"(\d+)\s*진수\s*([0-9a-fA-F]+)", q)
    if m:
        base, digits = int(m.group(1)), m.group(2)
        try:
            n = mc.from_base(digits, base)
        except ValueError:
            return None
        struct = {"value": n, "base": base, "digits": digits.lower()}
        return _result("unit", f"{base}진수 {digits.lower()}은 십진수로 {n}입니다.",
                       struct, 0.95)
    # English: binary 1101 [to decimal]
    m = re.search(r"\b(binary|octal|hex(?:adecimal)?|decimal)\s+([0-9a-fA-F]+)\b", low)
    if m:
        base, digits = _BASE_WORDS[m.group(1)], m.group(2)
        try:
            n = mc.from_base(digits, base)
        except ValueError:
            return None
        struct = {"value": n, "base": base, "digits": digits.lower()}
        return _result("unit", f"{m.group(1)} {digits.lower()}은 십진수로 {n}입니다.",
                       struct, 0.95)
    # English: 255 to hex
    m = re.search(r"\b(\d+)\s+(?:to|into|in)\s+(binary|octal|hex(?:adecimal)?)\b", low)
    if m:
        n, base = int(m.group(1)), _BASE_WORDS[m.group(2)]
        digits = mc.int_to_base(n, base)
        struct = {"value": n, "base": base, "digits": digits}
        return _result("unit",
                       f"{n}{_josa(str(n), '을', '를')} {m.group(2)}로 나타내면 {digits}입니다.",
                       struct, 0.95)
    return None


_UNIT_WORDS = {
    "km": "km", "킬로미터": "km", "킬로": "km", "m": "m", "미터": "m",
    "cm": "cm", "센티미터": "cm", "mm": "mm", "밀리미터": "mm",
    "um": "um", "nm": "nm", "mile": "mile", "miles": "mile", "마일": "mile",
    "mi": "mile", "yd": "yd", "야드": "yd", "ft": "ft", "피트": "ft",
    "foot": "ft", "feet": "ft", "in": "in", "inch": "in", "inches": "in",
    "인치": "in", "nmi": "nmi", "해리": "nmi", "해리": "nmi",
    "kg": "kg", "킬로그램": "kg", "g": "g", "그램": "g", "mg": "mg",
    "t": "t", "톤": "t", "lb": "lb", "lbs": "lb", "파운드": "lb",
    "oz": "oz", "온스": "oz", "stone": "stone", "스톤": "stone",
    "s": "s", "초": "s", "sec": "s", "second": "s", "seconds": "s",
    "min": "min", "분": "min", "분간": "min", "minute": "min", "minutes": "min",
    "h": "h", "시간": "h", "hr": "h", "hour": "h", "hours": "h",
    "day": "day", "일": "day", "days": "day", "week": "week", "주": "week",
    "year": "year", "년": "year", "years": "year",
    "l": "l", "리터": "l", "liter": "l", "liters": "l", "litre": "l",
    "ml": "ml", "밀리리터": "ml", "gal": "gal_us", "gallon": "gal_us",
    "갤런": "gal_us", "cup": "cup_us", "pint": "pint_us", "quart": "quart_us",
    "m2": "m2", "제곱미터": "m2", "km2": "km2", "ha": "ha", "헥타르": "ha",
    "acre": "acre", "에이커": "acre", "ft2": "ft2", "in2": "in2",
    "m3": "m3", "cm3": "cm3", "cc": "cm3", "mps": "mps", "kmph": "kmph",
    "kph": "kmph", "mph": "mph", "knot": "knot", "노트": "knot",
    "J": "J", "줄": "J", "kJ": "kJ", "cal": "cal", "칼로리": "cal",
    "kcal": "kcal", "킬로칼로리": "kcal", "kWh": "kWh", "Wh": "Wh",
    "W": "W", "와트": "W", "kW": "kW", "킬로와트": "kW", "MW": "MW",
    "hp": "hp", "마력": "hp", "Pa": "Pa", "파스칼": "Pa", "kPa": "kPa",
    "MPa": "MPa", "bar": "bar", "바": "bar", "atm": "atm", "기압": "atm",
    "psi": "psi", "torr": "torr", "mmHg": "mmHg", "N": "N", "뉴턴": "N",
    "lbf": "lbf", "kgf": "kgf", "byte": "byte", "바이트": "byte",
    "B": "byte", "KB": "KB", "킬로바이트": "KB", "MB": "MB", "메가바이트": "MB",
    "GB": "GB", "기가바이트": "GB", "TB": "TB", "테라바이트": "TB",
    "KiB": "KiB", "MiB": "MiB", "GiB": "GiB", "TiB": "TiB",
    "bit": "bit", "비트": "bit", "Mbit": "Mbit", "Gbit": "Gbit",
    "fahrenheit": "f", "celsius": "c", "kelvin": "k", "화씨": "f",
    "섭씨": "c", "켈빈": "k", "deg": "deg", "도": "deg", "rad": "rad",
    "라디안": "rad", "arcmin": "arcmin", "arcsec": "arcsec",
}


def _unit_pattern():
    keys = sorted(_UNIT_WORDS, key=len, reverse=True)
    parts = []
    for k in keys:
        esc = re.escape(k)
        if k.isascii() and k.isalpha():
            parts.append(rf"(?<![A-Za-z]){esc}(?![A-Za-z])")
        else:
            parts.append(esc)
    return "(?:%s)" % "|".join(parts)


_UNIT = _unit_pattern()
_NUM = r"([-+]?\d+(?:\.\d+)?)"
_NUM_UNIT = rf"{_NUM}\s*({_UNIT})"
_NUM_CAPTURE = r"([-+]?\d+(?:\.\d+)?)"
_NUM_UNIT_SRC = _NUM_UNIT

_CONV_EN_RE = re.compile(
    r"(?:convert\s+)?" + _NUM_UNIT_SRC +
    r"\s*(?:to|in|->|→|=)\s*(" + _UNIT + r")", re.I)
_CONV_KO_MYOT_RE = re.compile(
    _NUM_UNIT_SRC + r"\s*(?:은|는|이|가|을|를)?\s*몇\s*(" + _UNIT + r")")
_CONV_KO_RE = re.compile(
    _NUM_UNIT_SRC + r"\s*(?:을|를|으로|로|은|는|이|가|에서)\s*(" + _UNIT + r")")


_BATCHIM_DIGITS = "013678"


def _josa(word, after_batchim, after_vowel):
    """Pick the Korean particle matching the final sound of ``word``."""
    if not word:
        return after_vowel
    last = word[-1]
    if last.isdigit():
        has = last in _BATCHIM_DIGITS
    else:
        code = ord(last)
        has = 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0
    return after_batchim if has else after_vowel


def _canon_unit(tok):
    if tok in _UNIT_WORDS:
        return _UNIT_WORDS[tok]
    low = tok.lower()
    if low in _UNIT_WORDS:
        return _UNIT_WORDS[low]
    try:
        return mc._norm_unit(tok)
    except ValueError:
        return tok


def _convert_answer(value_s, from_tok, to_tok, source):
    f_canon, t_canon = _canon_unit(from_tok), _canon_unit(to_tok)
    try:
        out = mc.convert(value_s, f_canon, t_canon)
    except ValueError:
        return None
    val = _fraction_val(out)
    struct = {"value": val, "from": f_canon, "to": t_canon,
              "input": value_s, "source": source}
    text = f"{value_s} {from_tok}은 약 {_fmt(val)} {t_canon}입니다."
    return _result("unit", text, struct, 0.95)


_TEMP_KO = {"화씨": "f", "섭씨": "c", "켈빈": "k", "fahrenheit": "f",
            "celsius": "c", "kelvin": "k"}


def _h_temperature(q, now, off, label):
    low = q.lower()
    m = re.search(r"(화씨|섭씨|켈빈|fahrenheit|celsius|kelvin)\s*"
                  r"([-+]?\d+(?:\.\d+)?)\s*(?:도|degrees?|°)?", q, re.I)
    value_s = from_tok = None
    if m:
        from_tok = _TEMP_KO[m.group(1).lower() if m.group(1).isascii() else m.group(1)]
        value_s = m.group(2)
        rest = q[m.end():]
    else:
        m2 = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:도)?\s*"
                       r"(화씨|섭씨|켈빈|fahrenheit|celsius|kelvin)", q, re.I)
        if not m2:
            return None
        value_s = m2.group(1)
        from_tok = _TEMP_KO[m2.group(2).lower() if m2.group(2).isascii() else m2.group(2)]
        rest = q[:m2.start()] + q[m2.end():]
    target = None
    for word, u in _TEMP_KO.items():
        if word in rest or word in rest.lower():
            target = u
            break
    if target is None or target == from_tok:
        target = {"f": "c", "c": "f", "k": "c"}[from_tok]
    if from_tok == target:
        return None
    res = _convert_answer(value_s, from_tok, target, "temperature")
    if res is None:
        return None
    ko_from = {"f": "화씨", "c": "섭씨", "k": "켈빈"}[from_tok]
    ko_to = {"f": "화씨", "c": "섭씨", "k": "켈빈"}[target]
    res["answer_text_ko"] = (
        f"{ko_from} {value_s}도는 {ko_to} 약 {_fmt(res['structured']['value'])}도입니다.")
    return res


def _h_convert(q, now, off, label):
    for regex, source in ((_CONV_EN_RE, "en"), (_CONV_KO_MYOT_RE, "ko"),
                          (_CONV_KO_RE, "ko")):
        m = regex.search(q)
        if not m:
            continue
        value_s, from_tok, to_tok = m.group(1), m.group(2), m.group(3)
        res = _convert_answer(value_s, from_tok, to_tok, source)
        if res is not None:
            return res
    return None


_EQ_CHARS = r"[0-9xX^*/()+\-.\s]"


def _h_equation(q, now, off, label):
    m = re.search(rf"({_EQ_CHARS}{{1,120}})=({_EQ_CHARS}{{1,120}})", q)
    if not m:
        return None
    lhs, rhs = m.group(1).strip(), m.group(2).strip()
    if "x" not in lhs.lower() and "x" not in rhs.lower():
        return None
    def _norm(side):
        s = side.replace("^", "**")
        s = re.sub(r"(?<=\d)(?=[xX])", "*", s)
        s = re.sub(r"(?<=[xX])(?=\d)", "*", s)
        s = re.sub(r"(?<=\d)(?=\()", "*", s)
        return s
    expr = f"({_norm(lhs)})-({_norm(rhs)})"
    try:
        coeffs = mc.poly_coefficients(expr, "x")
    except (ValueError, ZeroDivisionError, RecursionError):
        return None
    deg = len(coeffs) - 1
    if deg <= 0:
        text = "모든 x에 대해 성립하는 항등식이거나 해가 없습니다." \
            if not any(coeffs) else "일차 이상의 x 항이 없습니다."
        return _result("math", text, {"degree": deg, "equation": f"{lhs} = {rhs}"}, 0.5)
    if deg == 1:
        try:
            roots = [mc.solve_linear(coeffs[1], coeffs[0])]
        except ValueError:
            return None
    elif deg == 2:
        try:
            roots = list(mc.solve_quadratic(coeffs[2], coeffs[1], coeffs[0]))
        except ValueError:
            return _result("math", "이 방정식은 실수 근을 갖지 않습니다.",
                           {"degree": deg, "equation": f"{lhs} = {rhs}",
                            "roots": []}, 0.9)
    else:
        return _result(
            "math", "3차 이상 방정식은 아직 풀 수 없습니다 (2차 이하만 지원).",
            {"degree": deg, "equation": f"{lhs} = {rhs}"}, 0.3)
    exact = [mc.format_exact(r) for r in roots]
    values = [float(r) for r in roots]
    struct = {"degree": deg, "equation": f"{lhs} = {rhs}",
              "roots_exact": exact, "roots": values,
              "coefficients": [mc.format_exact(c) for c in coeffs]}
    if len(roots) == 1:
        text = f"방정식 {lhs} = {rhs}의 해는 x = {exact[0]}입니다."
    else:
        text = (f"방정식 {lhs} = {rhs}의 근은 x = {exact[0]}, x = {exact[1]}입니다.")
    return _result("math", text, struct, 0.95)


def _h_percent(q, now, off, label):
    low = q.lower()
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%\s*(?:of|의)\s*([-+]?\d+(?:\.\d+)?)", q, re.I)
    if not m:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:percent|퍼센트)\s*(?:of|의)\s*"
                      r"([-+]?\d+(?:\.\d+)?)", q, re.I)
    if m:
        p_s, b_s = m.group(1), m.group(2)
    else:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*의\s*([-+]?\d+(?:\.\d+)?)\s*%", q)
        if not m:
            return None
        b_s, p_s = m.group(1), m.group(2)
    try:
        val = mc.percent_of(p_s, b_s)
    except (ValueError, ZeroDivisionError):
        return None
    struct = {"percent": float(p_s), "of": float(b_s),
              "value": _fraction_val(val), "exact": mc.format_exact(val)}
    text = f"{b_s}의 {p_s}%는 {mc.format_exact(val)}입니다."
    return _result("math", text, struct, 0.95)


def _h_prime_check(q, now, off, label):
    low = q.lower()
    m = re.search(r"(\d+)\s*(?:은|는|이|가)?\s*소수\s*(?:인가|입니까|인지|인가요|야|니|"
                  r"맞아|맞나|맞나요|일까|여부|인\s*가)", q)
    if not m:
        m = re.search(r"\bis\s+(\d+)\s+(?:a\s+)?prime", low)
    if not m:
        m = re.search(r"(\d+)\s*은?\s*prime\b", low)
    if not m:
        return None
    n = int(m.group(1))
    prime = mc.is_prime(n)
    struct = {"n": n, "is_prime": prime}
    if prime:
        text = f"{n}은 소수입니다."
    else:
        try:
            factors = mc.factorize(n)
            fac = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items())
            text = f"{n}은 소수가 아닙니다 ({n} = {fac})."
        except ValueError:
            text = f"{n}은 소수가 아닙니다."
    return _result("math", text, struct, 0.95)


def _h_prime_count(q, now, off, label):
    low = q.lower()
    if "소수" not in q and "prime" not in low:
        return None
    if not any(w in q for w in ("개수", "몇 개", "몇개", "세어", "개")) \
            and not any(w in low for w in ("up to", "count", "number of", "below", "under")):
        return None
    m = re.search(r"(\d+)\s*(?:까지|이하|미만)", q)
    if not m:
        m = re.search(r"(?:up to|below|under|까지)\s*(\d+)", low)
    if not m:
        m = re.search(r"(\d+)", q)
    if not m:
        return None
    limit = int(m.group(1))
    if limit < 2 or limit > 2_000_000:
        return None
    primes = mc.prime_sieve(limit)
    struct = {"limit": limit, "count": len(primes),
              "largest_prime": primes[-1] if primes else None}
    text = f"{limit} 이하의 소수는 모두 {len(primes)}개입니다"
    if primes:
        text += f" (가장 큰 소수 {primes[-1]})"
    text += "."
    return _result("math", text, struct, 0.95)


def _h_factorize(q, now, off, label):
    low = q.lower()
    m = re.search(r"(\d+)\s*(?:를|을|의|은|는)?\s*소인수분해", q)
    if not m:
        m = re.search(r"소인수분해\s*(?:해|:)?\s*(\d+)", q)
    if not m:
        m = re.search(r"(?:factor(?:ize|ise|ization)?|prime\s+factors?)\s*"
                      r"(?:of\s+)?(\d+)", low)
    if not m:
        return None
    n = int(m.group(1))
    if n == 0 or n > 10 ** 15:
        return None
    try:
        factors = mc.factorize(n)
    except ValueError:
        return None
    fac = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items())
    struct = {"n": n, "factors": {str(p): e for p, e in factors.items()}}
    text = f"{n} = {fac}입니다." if fac else f"{n}은 소인수가 없습니다."
    return _result("math", text, struct, 0.95)


def _h_divisors(q, now, off, label):
    low = q.lower()
    m = re.search(r"(\d+)\s*(?:의|은|는)?\s*약수", q)
    if not m:
        m = re.search(r"약수\s*(?:of\s*)?(\d+)", low)
    if not m:
        m = re.search(r"\bdivisors?\s+of\s+(\d+)", low)
    if not m:
        m = re.search(r"(\d+)\s+divisors?\b", low)
    if not m:
        return None
    n = int(m.group(1))
    try:
        divs = mc.divisors(n)
    except ValueError:
        return None
    struct = {"n": n, "divisors": divs, "count": len(divs)}
    shown = ", ".join(str(d) for d in divs[:30])
    if len(divs) > 30:
        shown += ", ..."
    text = f"{n}의 약수는 모두 {len(divs)}개입니다: {shown}."
    return _result("math", text, struct, 0.95)


def _h_gcd_lcm(q, now, off, label):
    low = q.lower()
    jobs = (("최대공약수", mc.gcd, "최대공약수", "gcd"),
            ("gcd", mc.gcd, "최대공약수", "gcd"),
            ("최소공배수", mc.lcm, "최소공배수", "lcm"),
            ("lcm", mc.lcm, "최소공배수", "lcm"))
    for kw, fn, ko, tag in jobs:
        idx = low.find(kw)
        if idx < 0:
            continue
        after = _numbers(q[idx + len(kw):])[:8]
        nums = [int(x) for x in after if float(x).is_integer()]
        if len(nums) < 2:
            before = _numbers(q[:idx])[:8]
            nums = [int(x) for x in before if float(x).is_integer()]
            if not re.fullmatch(r"[\s\d.,+\-와과및and]*", q[:idx].lower()):
                nums = nums[-8:]
        if len(nums) < 2 or len(nums) > 8:
            return None
        val = fn(*nums)
        struct = {"operation": tag, "numbers": nums, "value": val}
        joined = ", ".join(str(x) for x in nums)
        text = f"{joined}의 {ko}는 {val}입니다."
        return _result("math", text, struct, 0.95)
    return None


def _h_trig_log(q, now, off, label):
    low = q.lower()
    m = re.search(r"\blog\s*_?2\s*\(?\s*([\d.]+)", low) or \
        re.search(r"로그\s*2\s*\(?\s*([\d.]+)", q)
    if m:
        x = float(m.group(1))
        if x <= 0:
            return None
        val = math.log2(x)
        struct = {"function": "log2", "input": x,
                  "value": int(val) if val == int(val) else val}
        return _result("math", f"log2 {_fmt(x)}는 {_fmt(val)}입니다.", struct, 0.95)
    m = re.search(r"(?:\blog\s*_?10\b|상용로그|로그\s*10)\s*\(?\s*([\d.]+)", low)
    if m:
        x = float(m.group(1))
        if x <= 0:
            return None
        val = math.log10(x)
        struct = {"function": "log10", "input": x,
                  "value": int(val) if val == int(val) else val}
        return _result("math", f"상용로그 {_fmt(x)}는 {_fmt(val)}입니다.", struct, 0.95)
    m = re.search(r"(?:\bln\b|자연로그)\s*\(?\s*([\d.]+)", low)
    if m:
        x = float(m.group(1))
        if x <= 0:
            return None
        val = math.log(x)
        struct = {"function": "ln", "input": x, "value": val}
        return _result("math", f"자연로그 {_fmt(x)}는 약 {_fmt(val)}입니다.", struct, 0.95)
    m = re.search(r"(?:\blog\b)\s*\(?\s*([\d.]+)", low)
    if m:
        x = float(m.group(1))
        if x <= 0:
            return None
        val = math.log10(x) if x >= 1 else math.log(x)
        struct = {"function": "log", "input": x, "value": val}
        return _result("math", f"log {_fmt(x)}는 약 {_fmt(val)}입니다.", struct, 0.8)
    m = re.search(r"(sin|cos|tan|사인|코사인|탄젠트)\s*\(?\s*"
                  r"([-+]?\d+(?:\.\d+)?)\s*(도|degrees?|deg|°|라디안|radians?|rad)?", q, re.I)
    if m:
        fn_tok = m.group(1).lower()
        fn = {"사인": "sin", "코사인": "cos", "탄젠트": "tan"}.get(fn_tok, fn_tok)
        x = float(m.group(2))
        unit = (m.group(3) or "").strip().lower()
        radians = unit in ("라디안", "rad", "radian", "radians")
        if not unit:
            radians = False
        arg = x if radians else math.radians(x)
        val = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[fn](arg)
        struct = {"function": fn, "input": x,
                  "input_unit": "rad" if radians else "deg", "value": val}
        unit_ko = "라디안" if radians else "도"
        name_ko = {"sin": "사인", "cos": "코사인", "tan": "탄젠트"}[fn]
        return _result("math", f"{name_ko} {_fmt(x, 6)}{unit_ko}는 약 {_fmt(val)}입니다.",
                       struct, 0.95)
    return None


_STATS_KEYWORDS = (
    ("산술평균", mc.mean, "평균"), ("평균", mc.mean, "평균"),
    ("average", mc.mean, "평균"), ("mean", mc.mean, "평균"),
    ("중앙값", mc.median, "중앙값"), ("중간값", mc.median, "중앙값"),
    ("median", mc.median, "중앙값"),
    ("최빈값", mc.mode, "최빈값"), ("mode", mc.mode, "최빈값"),
    ("표준편차", mc.std, "표준편차"), ("standard deviation", mc.std, "표준편차"),
    ("분산", mc.variance, "분산"), ("variance", mc.variance, "분산"),
)


def _h_stats(q, now, off, label):
    low = q.lower()
    for kw, fn, ko in _STATS_KEYWORDS:
        idx = low.find(kw)
        if idx < 0:
            continue
        after = _numbers(q[idx + len(kw):])
        vals = after
        if len(vals) < 2:
            prefix = q[:idx]
            if re.fullmatch(r"[\s\d.,+\-와과및and]*", prefix.lower()):
                vals = _numbers(prefix)
        if len(vals) < 2 or len(vals) > 200:
            return None
        try:
            val = fn(vals)
        except (ValueError, ZeroDivisionError):
            return None
        joined = ", ".join(_fmt(v) for v in vals[:20])
        struct = {"values": vals, "stat": ko, "value": _fraction_val(val)}
        if isinstance(val, mc.Fraction):
            struct["exact"] = mc.format_exact(val)
        text = f"값 {joined}의 {ko}{_josa(ko, '은', '는')} {_fmt(val)}입니다."
        return _result("math", text, struct, 0.95)
    return None


def _h_power(q, now, off, label):
    low = q.lower()
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*의\s*세제곱", q)
    base_s, exp = None, 3
    if m:
        base_s = m.group(1)
    if base_s is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*의\s*네제곱", q)
        if m:
            base_s, exp = m.group(1), 4
    if base_s is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*의\s*(\d+)\s*제곱", q)
        if m:
            base_s, exp = m.group(1), int(m.group(2))
    if base_s is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*의\s*제곱(?!근)", q)
        if m:
            base_s, exp = m.group(1), 2
    if base_s is None:
        m = re.search(r"(?:square\s+of\s+)?([-+]?\d+(?:\.\d+)?)\s+squared\b", low)
        if m:
            base_s, exp = m.group(1), 2
    if base_s is None:
        m = re.search(r"\bsquare\s+of\s+([-+]?\d+(?:\.\d+)?)", low)
        if m:
            base_s, exp = m.group(1), 2
    if base_s is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s+cubed\b", low)
        if m:
            base_s, exp = m.group(1), 3
    if base_s is None:
        m = re.search(r"\bcube\s+of\s+([-+]?\d+(?:\.\d+)?)", low)
        if m:
            base_s, exp = m.group(1), 3
    if base_s is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s+to\s+the\s+(?:power\s+of\s+)?(\d+)", low)
        if m:
            base_s, exp = m.group(1), int(m.group(2))
    if base_s is None:
        return None
    if exp > 10000:
        return None
    base = float(base_s) if "." in base_s else int(base_s)
    if isinstance(base, float) and exp > 100:
        val = float(base) ** exp
    else:
        val = base ** exp
    exact = mc.format_exact(val) if not isinstance(val, float) else _fmt(val, 12)
    struct = {"base": base, "exponent": exp, "value": val}
    exp_ko = {2: "제곱", 3: "세제곱", 4: "네제곱"}.get(exp, f"{exp}제곱")
    text = f"{base_s}의 {exp_ko}은 {exact}입니다."
    return _result("math", text, struct, 0.97)


def _h_sqrt(q, now, off, label):
    m = re.search(r"(?:sqrt|루트|제곱근)\s*\(?\s*([\d.]+)\s*\)?", q, re.I)
    if not m:
        return None
    x = float(m.group(1))
    if x < 0:
        return None
    val = math.sqrt(x)
    md = re.search(r"소수(?:점)?\s*(\d+)\s*자리", q) or \
        re.search(r"(\d+)\s*(?:decimal places?|자리까지)", q, re.I)
    struct = {"input": x, "value": val}
    if x == int(x) and int(x) ** 2 == int(x) and math.isqrt(int(x)) ** 2 == int(x):
        val_show = str(math.isqrt(int(x)))
        struct["value"] = math.isqrt(int(x))
        struct["exact"] = True
    else:
        val_show = None
    if md:
        digits = min(int(md.group(1)), 50)
        rounded = round(val, digits)
        struct["decimal_places"] = digits
        struct["rounded"] = rounded
        text = (f"{m.group(1)}의 제곱근은 소수 {digits}자리까지 "
                f"{rounded:.{digits}f}입니다.")
    elif val_show is not None:
        text = f"{m.group(1)}의 제곱근은 {val_show}입니다."
    else:
        text = f"{m.group(1)}의 제곱근은 약 {_fmt(val)}입니다."
    return _result("math", text, struct, 0.97)


_ARITH_CHARS_RE = re.compile(r"[0-9.+\-*/^()\s]+")
_ARITH_WORDS = (("곱하기", "*"), ("나누기", "/"), ("더하기", "+"), ("빼기", "-"))


def _h_expression(q, now, off, label):
    text = q
    for word, op in _ARITH_WORDS:
        text = text.replace(word, op)
    for m in _ARITH_CHARS_RE.finditer(text):
        s = m.group().strip()
        if len(s) < 3 or not re.search(r"\d", s) or not re.search(r"[+\-*/^]", s):
            continue
        if not re.search(r"\d\s*[+\-*/^]\s*\d", s) and not re.search(r"\(.*\)", s):
            continue
        value = None
        for attempt in (s, s.strip("+-*/^ ().")):
            if len(attempt) < 3:
                continue
            try:
                value = mc.eval_expr(attempt)
                s = attempt
                break
            except (ValueError, OverflowError):
                continue
        if value is None:
            continue
        md = re.search(r"소수(?:점)?\s*(\d+)\s*자리", q) or \
            re.search(r"(\d+)\s*(?:decimal places?|자리까지)", q, re.I)
        struct = {"expression": s, "value": value}
        if md:
            digits = min(int(md.group(1)), 50)
            struct["decimal_places"] = digits
            struct["rounded"] = round(float(value), digits)
            text_out = (f"{s}의 계산 결과는 소수 {digits}자리까지 "
                        f"{round(float(value), digits):.{digits}f}입니다.")
        else:
            text_out = f"{s}의 계산 결과는 {_fmt(value, 12) if not isinstance(value, int) else value}입니다."
        return _result("math", text_out, struct, 0.97)
    return None


_HANDLERS = (
    _h_sun_rise_set,
    _h_eclipse,
    _h_next_moon_phase,
    _h_moon_phase_now,
    _h_planet_separation,
    _h_planet_position,
    _h_sun_position_date,
    _h_sun_position_now,
    _h_season,
    _h_roman,
    _h_bases,
    _h_temperature,
    _h_convert,
    _h_equation,
    _h_percent,
    _h_prime_check,
    _h_prime_count,
    _h_factorize,
    _h_divisors,
    _h_gcd_lcm,
    _h_trig_log,
    _h_stats,
    _h_power,
    _h_sqrt,
    _h_expression,
)


def answer(question, now=None, tz="Asia/Seoul"):
    """Answer a Korean/English math or astronomy question.

    ``now`` pins the reference instant (naive values are treated as UTC);
    with the same ``(question, now, tz)`` the result dict is identical.
    ``tz`` only affects human-readable clock times (fixed offset table,
    documented approximation for solar events).
    """
    if not isinstance(question, str):
        return _unknown()
    q = question.strip()
    if not q:
        return _unknown()
    try:
        now_utc = _normalize_now(now)
    except ValueError:
        return _unknown()
    off, label, _approx = _tz_offset(tz, now_utc)
    for handler in _HANDLERS:
        try:
            res = handler(q, now_utc, off, label)
        except Exception:
            continue
        if res is not None:
            return res
    return _unknown()
