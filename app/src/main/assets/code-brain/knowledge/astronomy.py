"""Astronomy: Meeus-style algorithms in pure Python stdlib.

Scope
-----
Time (JD, GMST/GAST), Sun (ch.25 low precision), Moon (ch.47 truncated),
phases/search, approximate eclipses, planets Mercury..Saturn via approximate
Keplerian elements (1800-2050), seasons (ch.27 approx), precession (ch.21),
and reference data (88 constellations, bright stars, meteor showers).

Conventions
-----------
* All datetimes are naive UTC unless they carry tzinfo (converted to UTC).
* Terrestrial Time vs UTC (delta-T) is ignored: < ~1.5 min for 1900-2050.
* Angles in degrees; distances: AU for planets/Sun, km for the Moon.
"""

import math
from collections import namedtuple
from datetime import datetime, timedelta, timezone

__all__ = [
    "julian_date", "calendar_from_jd", "gmst", "gast", "nutation",
    "mean_obliquity", "sun_position", "equation_of_time", "solar_noon",
    "sun_rise_set", "moon_position", "moon_phase", "next_phase",
    "moon_rise_set", "find_next_eclipse", "planet_position",
    "angular_separation", "retrograde", "season_instants", "precess",
    "moon_phase_table", "CONSTELLATIONS", "BRIGHT_STARS", "METEOR_SHOWERS",
]

D2R = math.pi / 180.0
R2D = 180.0 / math.pi
JD_J2000 = 2451545.0
AU_KM = 149597870.7
SYNODIC_MONTH = 29.530588853
EARTH_RADIUS_KM = 6378.14


def _sin(x):
    return math.sin(x * D2R)


def _cos(x):
    return math.cos(x * D2R)


def _norm360(x):
    return x % 360.0


def _norm180(x):
    return (x + 180.0) % 360.0 - 180.0


def _clamp(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def julian_date(dt):
    """Julian Date (UTC) from a datetime (naive treated as UTC)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def calendar_from_jd(jd):
    """Inverse of julian_date (Gregorian calendar)."""
    z = math.floor(jd + 0.5)
    f = jd + 0.5 - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    iday = int(math.floor(day))
    return datetime(year, month, iday) + timedelta(days=day - iday)


def _jd_to_dt(jd):
    return calendar_from_jd(jd)


def gmst(jd):
    """Greenwich Mean Sidereal Time in degrees (Meeus 12.4)."""
    t = (jd - JD_J2000) / 36525.0
    return _norm360(280.46061837 + 360.98564736629 * (jd - JD_J2000)
                    + 0.000387933 * t * t - t * t * t / 38710000.0)


def nutation(jd):
    """Approximate nutation: (dpsi, deps, eps_mean) degrees (ch.22 low precision)."""
    t = (jd - JD_J2000) / 36525.0
    om = 125.04452 - 1934.136261 * t + 0.0020708 * t * t + t ** 3 / 450000.0
    ls = 280.4665 + 36000.7698 * t
    lm = 218.3165 + 481267.8813 * t
    dpsi = (-17.20 * _sin(om) - 1.32 * _sin(2 * ls)
            - 0.23 * _sin(2 * lm) + 0.21 * _sin(2 * om)) / 3600.0
    deps = (9.20 * _cos(om) + 0.57 * _cos(2 * ls)
            + 0.10 * _cos(2 * lm) - 0.09 * _cos(2 * om)) / 3600.0
    return dpsi, deps, mean_obliquity(jd)


def mean_obliquity(jd):
    t = (jd - JD_J2000) / 36525.0
    return 23.0 + 26.0 / 60.0 + 21.448 / 3600.0 + (
        -46.815 * t - 0.00059 * t * t + 0.001813 * t ** 3) / 3600.0


def gast(jd):
    """Greenwich Apparent Sidereal Time in degrees (GMST + equation of equinoxes)."""
    dpsi, deps, eps0 = nutation(jd)
    return _norm360(gmst(jd) + dpsi * _cos(eps0 + deps))


# ---------------------------------------------------------------------------
# Sun (Meeus ch.25 low precision)
# ---------------------------------------------------------------------------

SunPosition = namedtuple("SunPosition", "ra dec distance lon lat epsilon")


def sun_position(jd):
    """Apparent geocentric Sun position (equinox of date)."""
    t = (jd - JD_J2000) / 36525.0
    l0 = _norm360(280.46646 + 36000.76983 * t + 0.0003032 * t * t)
    m = _norm360(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    mr = m * D2R
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mr)
         + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
         + 0.000289 * math.sin(3 * mr))
    true_lon = l0 + c
    nu = m + c
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    r = 1.000001018 * (1 - e * e) / (1 + e * _cos(nu))
    om = 125.04 - 1934.136 * t
    lon = true_lon - 0.00569 - 0.00478 * _sin(om)
    dpsi, deps, eps0 = nutation(jd)
    eps = eps0 + deps
    ra = _norm360(math.atan2(_cos(eps) * _sin(lon), _cos(lon)) * R2D)
    dec = math.asin(_clamp(_sin(eps) * _sin(lon))) * R2D
    return SunPosition(ra, dec, r, lon, 0.0, eps)


def equation_of_time(jd):
    """Equation of time in minutes (apparent minus mean solar time, ch.28)."""
    tau = (jd - JD_J2000) / 365250.0
    l0 = _norm360(280.4664567 + 360007.6982779 * tau + 0.03032028 * tau * tau
                  + tau ** 3 / 49931.0 - tau ** 4 / 15300.0 - tau ** 5 / 2000000.0)
    sp = sun_position(jd)
    dpsi, deps, eps0 = nutation(jd)
    e = l0 - 0.0057183 - sp.ra + dpsi * _cos(eps0 + deps)
    return _norm180(e) * 4.0


def solar_noon(jd0, lon):
    """UTC datetime of local solar transit for day starting at JD jd0 (0h UTC)."""
    t_hours = 12.0 - lon / 15.0
    for _ in range(6):
        jd = jd0 + t_hours / 24.0
        t_new = 12.0 - lon / 15.0 - equation_of_time(jd) / 60.0
        if abs(t_new - t_hours) < 1e-7:
            t_hours = t_new
            break
        t_hours = t_new
    return _jd_to_dt(jd0 + t_hours / 24.0)


RiseSet = namedtuple("RiseSet", "rise transit set polar")


def sun_rise_set(jd0, lat, lon, h0=-0.833):
    """Sun rise/transit/set (UTC datetimes) near the UTC day starting at jd0.

    polar is None, 'day' (never sets) or 'night' (never rises)."""
    transit = solar_noon(jd0, lon)
    jt = julian_date(transit)

    def cos_h0(jd):
        sp = sun_position(jd)
        return ( _sin(h0) - _sin(lat) * _sin(sp.dec)) / (_cos(lat) * _cos(sp.dec))

    c = cos_h0(jt)
    if c < -1.0:
        return RiseSet(None, transit, None, "day")
    if c > 1.0:
        return RiseSet(None, transit, None, "night")
    w = math.degrees(math.acos(_clamp(c))) / 360.0
    jr, js = jt - w, jt + w
    for _ in range(3):
        cr = cos_h0(jr)
        if abs(cr) <= 1.0:
            jr = jt - math.degrees(math.acos(_clamp(cr))) / 360.0
        cs = cos_h0(js)
        if abs(cs) <= 1.0:
            js = jt + math.degrees(math.acos(_clamp(cs))) / 360.0
    return RiseSet(_jd_to_dt(jr), transit, _jd_to_dt(js), None)


# ---------------------------------------------------------------------------
# Moon (Meeus ch.47, main periodic terms)
# ---------------------------------------------------------------------------

# (D, M, M', F, sum_l [1e-6 deg], sum_r [1e-3 km])
_MOON_LR = (
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),
)

# (D, M, M', F, sum_b [1e-6 deg])
_MOON_B = (
    (0, 0, 0, 1, 5128122),
    (0, 0, 1, 1, 280602),
    (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413),
    (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),
    (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),
    (2, -1, 0, -1, 8216),
    (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),
    (2, 1, 0, -1, -3359),
    (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065),
    (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),
    (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565),
    (1, 0, 0, 1, -1491),
    (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410),
    (0, 1, 0, -1, -1344),
    (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021),
)

MoonPosition = namedtuple("MoonPosition", "ra dec distance lon lat epsilon")


def moon_position(jd):
    """Apparent geocentric Moon position (equinox of date); distance in km."""
    t = (jd - JD_J2000) / 36525.0
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t + t ** 3 / 538841.0 - t ** 4 / 65194000.0
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t + t ** 3 / 545868.0 - t ** 4 / 113065000.0
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t * t + t ** 3 / 24490000.0
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t + t ** 3 / 69699.0 - t ** 4 / 14712000.0
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t - t ** 3 / 3526000.0 + t ** 4 / 863310000.0
    a1 = 119.75 + 131.849 * t
    a2 = 53.09 + 479264.290 * t
    a3 = 313.45 + 481266.484 * t
    ee = 1 - 0.002516 * t - 0.0000074 * t * t

    suml = 0.0
    sumr = 0.0
    for cd, cm, cmp_, cf, sl, sr in _MOON_LR:
        arg = cd * d + cm * m + cmp_ * mp + cf * f
        fac = ee ** abs(cm)
        suml += fac * sl * _sin(arg)
        sumr += fac * sr * _cos(arg)
    suml += (3958.0 * _sin(a1) + 1962.0 * _sin(lp - f) + 318.0 * _sin(a2))
    sumb = 0.0
    for cd, cm, cmp_, cf, sb in _MOON_B:
        arg = cd * d + cm * m + cmp_ * mp + cf * f
        sumb += ee ** abs(cm) * sb * _sin(arg)
    sumb += (-2235.0 * _sin(lp) + 382.0 * _sin(a3) + 175.0 * _sin(a1 - f)
             + 175.0 * _sin(a1 + f) + 127.0 * _sin(lp - mp) - 115.0 * _sin(lp + mp))

    lon = lp + suml / 1e6
    lat = sumb / 1e6
    dist = 385000.56 + sumr / 1000.0
    dpsi, deps, eps0 = nutation(jd)
    lon = _norm360(lon + dpsi)
    eps = eps0 + deps
    ra = _norm360(math.atan2(_sin(lon) * _cos(eps) - math.tan(lat * D2R) * _sin(eps),
                             _cos(lon)) * R2D)
    dec = math.asin(_clamp(_sin(lat) * _cos(eps) + _cos(lat) * _sin(eps) * _sin(lon))) * R2D
    return MoonPosition(ra, dec, dist, lon, lat, eps)


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation between two (RA, Dec) points, degrees."""
    c = (_sin(dec1) * _sin(dec2)
         + _cos(dec1) * _cos(dec2) * _cos(ra1 - ra2))
    return math.degrees(math.acos(_clamp(c)))


MoonPhase = namedtuple("MoonPhase", "illumination phase_angle elongation age_days name")

# nearest-name mapping targets (elongation deg -> Korean phase name)
_PHASE_TARGETS = ((0.0, "삭"), (45.0, "초승달"), (90.0, "상현달"),
                  (180.0, "보름달"), (270.0, "하현달"), (315.0, "그믐달"))


def _phase_name(elong):
    best, bestd = None, 1e9
    for tgt, name in _PHASE_TARGETS:
        dd = abs(_norm180(elong - tgt))
        if dd < bestd:
            best, bestd = name, dd
    return best


def moon_phase(jd):
    """Moon phase data. Illumination/phase angle exact-ish; age mean-based."""
    mp = moon_position(jd)
    sp = sun_position(jd)
    elong = _norm360(mp.lon - sp.lon)
    geoc = angular_separation(sp.ra, sp.dec, mp.ra, mp.dec)
    r = sp.distance
    delta = mp.distance / AU_KM
    i = math.degrees(math.atan2(r * _sin(geoc), delta - r * _cos(geoc)))
    k = (1 + math.cos(i * D2R)) / 2.0
    t_est = jd - (elong / 360.0) * SYNODIC_MONTH
    jd_prev = _nearest_phase_jd(0.0, t_est)
    age = jd - jd_prev
    name = "삭" if k < 0.02 else _phase_name(elong)
    return MoonPhase(k, i, geoc, age, name)


_PHASE_KINDS = {"new": 0.0, "first_quarter": 90.0, "full": 180.0, "last_quarter": 270.0}


def _nearest_phase_jd(target, jd):
    """JD of the phase with ecliptic elongation `target` nearest to jd (Newton)."""
    t = jd
    for _ in range(60):
        e = _norm360(moon_position(t).lon - sun_position(t).lon)
        diff = _norm180(target - e)
        if abs(diff) < 1e-4:
            break
        t += diff / 12.19
    return t


def _next_phase_jd(kind, jd):
    """JD of the next lunar phase of given kind strictly after jd (Newton)."""
    if kind not in _PHASE_KINDS:
        raise ValueError("kind must be one of %s" % sorted(_PHASE_KINDS))
    t = _nearest_phase_jd(_PHASE_KINDS[kind], jd)
    while t <= jd:
        t = _nearest_phase_jd(_PHASE_KINDS[kind], t + SYNODIC_MONTH)
    return t


def next_phase(kind, after):
    """datetime (UTC) of next phase of kind strictly after datetime `after`."""
    return _jd_to_dt(_next_phase_jd(kind, julian_date(after) + 1.0 / 86400.0))


def moon_rise_set(jd0, lat, lon):
    """Approximate moon rise/transit/set (UTC) within [jd0, jd0+1) [jd0=0h UTC].

    Uses standard altitude h0 = 0.7275*parallax - 0.5667 (~ +0.125 deg)."""
    def altitude(jd):
        mp = moon_position(jd)
        par = math.degrees(math.asin(EARTH_RADIUS_KM / mp.distance))
        h0 = 0.7275 * par - 0.5667
        ha = gmst(jd) + lon - mp.ra
        h = math.degrees(math.asin(_clamp(
            _sin(lat) * _sin(mp.dec) + _cos(lat) * _cos(mp.dec) * _cos(ha))))
        return h - h0

    def transit_fn(jd):
        mp = moon_position(jd)
        return _norm180(gmst(jd) + lon - mp.ra)

    rise = set_ = transit = None
    step = 0.05
    n = int(round(1.1 / step))
    prev_t = jd0 - 0.05
    prev_a = altitude(prev_t)
    prev_h = transit_fn(prev_t)
    for i in range(1, n + 1):
        t = jd0 - 0.05 + i * step
        a = altitude(t)
        h = transit_fn(t)
        if transit is None and prev_h <= 0.0 < h and jd0 <= t <= jd0 + 1.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if transit_fn(mid) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            transit = _jd_to_dt(0.5 * (lo + hi))
        if rise is None and prev_a <= 0.0 < a and jd0 <= t <= jd0 + 1.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if altitude(mid) <= 0.0:
                    lo = mid
                else:
                    hi = mid
            rise = _jd_to_dt(0.5 * (lo + hi))
        if set_ is None and prev_a >= 0.0 > a and jd0 <= t <= jd0 + 1.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if altitude(mid) >= 0.0:
                    lo = mid
                else:
                    hi = mid
            set_ = _jd_to_dt(0.5 * (lo + hi))
        prev_t, prev_a, prev_h = t, a, h
    return RiseSet(rise, transit, set_, None)


# ---------------------------------------------------------------------------
# Eclipses (approximate node-distance criteria; see report for limits)
# ---------------------------------------------------------------------------

Eclipse = namedtuple("Eclipse", "kind datetime node_distance moon_latitude")


def _sun_node_distance(jd):
    t = (jd - JD_J2000) / 36525.0
    node = 125.0445479 - 1934.1362891 * t + 0.0020754 * t * t
    lon = sun_position(jd).lon
    return min(abs(_norm180(lon - node)), abs(_norm180(lon - node - 180.0)))


def find_next_eclipse(kind, after, limit_days=800):
    """Next solar/lunar eclipse after `after` (approximate), or None.

    solar: new moon within 18.5 deg of a node; lunar: full moon within 12 deg."""
    if kind not in ("solar", "lunar"):
        raise ValueError("kind must be 'solar' or 'lunar'")
    phase = "new" if kind == "solar" else "full"
    limit = 18.5 if kind == "solar" else 12.0
    t = julian_date(after) + 1.0 / 86400.0
    end = t + limit_days
    while t <= end:
        t = _next_phase_jd(phase, t)
        if t > end:
            return None
        dd = _sun_node_distance(t)
        if dd <= limit:
            return Eclipse(kind, _jd_to_dt(t), dd, moon_position(t).lat)
        t += 0.01
    return None


# ---------------------------------------------------------------------------
# Planets (approximate Keplerian elements, JPL, valid 1800-2050)
# ---------------------------------------------------------------------------

_PLANET_NAMES_KO = {"mercury": "수성", "venus": "금성", "mars": "화성",
                    "jupiter": "목성", "saturn": "토성", "earth": "지구"}

# name: (a, adot, e, edot, I, Idot, L, Ldot, peri, peridot, node, nodedot)
_PLANETS = {
    "mercury": (0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
                252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081),
    "venus": (0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
              181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418),
    "earth": (1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
              100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0),
    "mars": (1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
             -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    "jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
                34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106),
    "saturn": (9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
               49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794),
}

_OBLIQUITY_J2000 = 23.4392911


def _heliocentric_ecliptic(name, jd):
    t = (jd - JD_J2000) / 36525.0
    a0, ad, e0, ed, i0, id_, l0, ld, p0, pd, n0, nd = _PLANETS[name]
    a = a0 + ad * t
    e = e0 + ed * t
    inc = i0 + id_ * t
    ll = l0 + ld * t
    peri = p0 + pd * t
    node = n0 + nd * t
    m = _norm180(ll - peri)
    w = peri - node
    er = m * D2R
    for _ in range(60):
        de = (er - e * math.sin(er) - m * D2R) / (1 - e * math.cos(er))
        er -= de
        if abs(de) < 1e-12:
            break
    xp = a * (math.cos(er) - e)
    yp = a * math.sqrt(1 - e * e) * math.sin(er)
    cw, sw = _cos(w), _sin(w)
    cn, sn = _cos(node), _sin(node)
    ci, si = _cos(inc), _sin(inc)
    x = (cw * cn - sw * sn * ci) * xp + (-sw * cn - cw * sn * ci) * yp
    y = (cw * sn + sw * cn * ci) * xp + (-sw * sn + cw * cn * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return x, y, z


def _geo_ecliptic(name, jd):
    px, py, pz = _heliocentric_ecliptic(name, jd)
    ex, ey, ez = _heliocentric_ecliptic("earth", jd)
    x, y, z = px - ex, py - ey, pz - ez
    r = math.sqrt(x * x + y * y + z * z)
    lon = _norm360(math.degrees(math.atan2(y, x)))
    lat = math.degrees(math.asin(_clamp(z / r)))
    return lon, lat, r


PlanetPosition = namedtuple(
    "PlanetPosition", "name ra dec distance heliocentric_distance elongation magnitude retrograde lon lat")


def _planet_magnitude(name, r, delta, phase_deg):
    base = 5.0 * math.log10(r * delta)
    i = phase_deg
    if name == "mercury":
        return -0.42 + base + 0.0380 * i - 0.000273 * i * i + 0.000002 * i ** 3
    if name == "venus":
        return -4.40 + base + 0.0009 * i + 0.000239 * i * i - 0.00000065 * i ** 3
    if name == "mars":
        return -1.52 + base + 0.016 * i
    if name == "jupiter":
        return -9.40 + base + 0.005 * i
    if name == "saturn":
        return -8.88 + base
    raise ValueError(name)


def planet_position(name, jd):
    """Approximate geocentric position of Mercury..Saturn (equinox of date)."""
    name = name.lower()
    if name not in _PLANETS or name == "earth":
        raise ValueError("name must be one of mercury..saturn")
    lon, lat, delta = _geo_ecliptic(name, jd)
    ex, ey, ez = _heliocentric_ecliptic("earth", jd)
    r_helio = math.sqrt(sum(c * c for c in _heliocentric_ecliptic(name, jd)))
    r_earth = math.sqrt(ex * ex + ey * ey + ez * ez)  # Sun-Earth distance (AU)
    cos_i = _clamp((r_helio ** 2 + delta ** 2 - r_earth ** 2) / (2.0 * r_helio * delta))
    phase = math.degrees(math.acos(cos_i))
    eps = _OBLIQUITY_J2000 * D2R
    ra = _norm360(math.degrees(math.atan2(
        _sin(lon) * math.cos(eps) - math.tan(lat * D2R) * math.sin(eps), _cos(lon))))
    dec = math.degrees(math.asin(_clamp(
        _sin(lat) * math.cos(eps) + _cos(lat) * math.sin(eps) * _sin(lon))))
    year = 2000.0 + (jd - JD_J2000) / 365.25
    ra, dec = precess(ra, dec, 2000.0, year)
    sp = sun_position(jd)
    elong = angular_separation(sp.ra, sp.dec, ra, dec)
    mag = _planet_magnitude(name, r_helio, delta, phase)
    lon2, _, _ = _geo_ecliptic(name, jd + 1.0)
    retro = _norm180(lon2 - lon) < 0.0
    return PlanetPosition(name, ra, dec, delta, r_helio, elong, mag, retro, lon, lat)


def retrograde(name, jd):
    """True if geocentric ecliptic longitude is decreasing at jd (approx)."""
    name = name.lower()
    lon1, _, _ = _geo_ecliptic(name, jd)
    lon2, _, _ = _geo_ecliptic(name, jd + 1.0)
    return _norm180(lon2 - lon1) < 0.0


# ---------------------------------------------------------------------------
# Seasons (ch.27 approx) and precession (ch.21)
# ---------------------------------------------------------------------------

_SEASON_TERMS = (
    (485, 324.96, 1934.136), (203, 337.23, 32964.467), (199, 342.08, 20.186),
    (182, 27.85, 445267.112), (156, 73.14, 45036.886), (136, 171.52, 22518.443),
    (77, 222.54, 65928.934), (74, 296.72, 3034.906), (70, 243.58, 9037.513),
    (58, 119.81, 33718.147), (52, 297.17, 150.678), (50, 21.02, 2281.226),
    (45, 247.54, 29929.562), (44, 325.15, 31555.956), (29, 60.93, 4443.417),
    (18, 155.12, 67555.328), (17, 288.79, 4562.452), (16, 198.04, 62894.029),
    (14, 199.76, 31436.921), (12, 95.39, 14577.848), (12, 338.65, 31931.756),
    (12, 163.12, 34777.259), (7, 191.96, 1222.114), (5, 77.83, 16859.074),
)

_SEASON_JDE0 = {
    "spring": (2451623.80984, 365242.37404, 0.05169, -0.00411, -0.00057),
    "summer": (2451716.56767, 365241.62603, 0.00325, 0.00888, -0.00030),
    "autumn": (2451810.21715, 365242.01767, -0.11575, 0.00337, 0.00078),
    "winter": (2451900.05952, 365242.74049, -0.06223, -0.00823, 0.00032),
}


def _season_jd(kind, year):
    c0, c1, c2, c3, c4 = _SEASON_JDE0[kind]
    y = (year - 2000.0) / 1000.0
    jde0 = c0 + c1 * y + c2 * y ** 2 + c3 * y ** 3 + c4 * y ** 4
    t = (jde0 - JD_J2000) / 36525.0
    w = 35999.373 * t - 2.47
    dl = 1 + 0.0334 * _cos(w) + 0.0007 * _cos(2 * w)
    s = sum(a * _cos(b + c * t) for a, b, c in _SEASON_TERMS)
    return jde0 + 0.00001 * s / dl


SeasonInstants = namedtuple("SeasonInstants", "spring summer autumn winter")


def season_instants(year):
    """Approximate equinox/solstice instants (UTC) for the given year."""
    return SeasonInstants(*[_jd_to_dt(_season_jd(k, year))
                            for k in ("spring", "summer", "autumn", "winter")])


def precess(ra, dec, from_year, to_year):
    """Precess equatorial (RA, Dec) degrees between epochs given in years."""
    t = (from_year - 2000.0) / 100.0
    dt = (to_year - from_year) / 100.0
    zeta = ((2306.2181 + 1.39656 * t - 0.000139 * t * t) * dt
            + (0.30188 - 0.000344 * t) * dt * dt + 0.017998 * dt ** 3) / 3600.0
    z = ((2306.2181 + 1.39656 * t - 0.000139 * t * t) * dt
         + (1.09468 + 0.000066 * t) * dt * dt + 0.018203 * dt ** 3) / 3600.0
    theta = ((2004.3109 - 0.85330 * t - 0.000217 * t * t) * dt
             - (0.42665 + 0.000217 * t) * dt * dt - 0.041833 * dt ** 3) / 3600.0
    a = _cos(dec) * _sin(ra + zeta)
    b = _cos(theta) * _cos(dec) * _cos(ra + zeta) - _sin(theta) * _sin(dec)
    c = _sin(theta) * _cos(dec) * _cos(ra + zeta) + _cos(theta) * _sin(dec)
    return _norm360(math.degrees(math.atan2(a, b)) + z), math.degrees(math.asin(_clamp(c)))


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# (abbreviation, English, Korean)
CONSTELLATIONS = (
    ("And", "Andromeda", "안드로메다자리"), ("Ant", "Antlia", "공기펌프자리"),
    ("Aps", "Apus", "극락조자리"), ("Aqr", "Aquarius", "물병자리"),
    ("Aql", "Aquila", "독수리자리"), ("Ara", "Ara", "제단자리"),
    ("Ari", "Aries", "양자리"), ("Aur", "Auriga", "마차부자리"),
    ("Boo", "Bootes", "목동자리"), ("Cae", "Caelum", "조각칼자리"),
    ("Cam", "Camelopardalis", "기린자리"), ("Cnc", "Cancer", "게자리"),
    ("CVn", "Canes Venatici", "사냥개자리"), ("CMa", "Canis Major", "큰개자리"),
    ("CMi", "Canis Minor", "작은개자리"), ("Cap", "Capricornus", "염소자리"),
    ("Car", "Carina", "용골자리"), ("Cas", "Cassiopeia", "카시오페이아자리"),
    ("Cen", "Centaurus", "센타우루스자리"), ("Cep", "Cepheus", "세페우스자리"),
    ("Cet", "Cetus", "고래자리"), ("Cha", "Chamaeleon", "카멜레온자리"),
    ("Cir", "Circinus", "컴퍼스자리"), ("Col", "Columba", "비둘기자리"),
    ("Com", "Coma Berenices", "머리털자리"), ("CrA", "Corona Australis", "남쪽왕관자리"),
    ("CrB", "Corona Borealis", "북쪽왕관자리"), ("Crv", "Corvus", "까마귀자리"),
    ("Crt", "Crater", "컵자리"), ("Cru", "Crux", "남십자자리"),
    ("Cyg", "Cygnus", "백조자리"), ("Del", "Delphinus", "돌고래자리"),
    ("Dor", "Dorado", "황새치자리"), ("Dra", "Draco", "용자리"),
    ("Equ", "Equuleus", "조랑말자리"), ("Eri", "Eridanus", "에리다누스자리"),
    ("For", "Fornax", "화학로자리"), ("Gem", "Gemini", "쌍둥이자리"),
    ("Gru", "Grus", "두루미자리"), ("Her", "Hercules", "헤르쿨레스자리"),
    ("Hor", "Horologium", "시계자리"), ("Hya", "Hydra", "바다뱀자리"),
    ("Hyi", "Hydrus", "물뱀자리"), ("Ind", "Indus", "인디언자리"),
    ("Lac", "Lacerta", "도마뱀자리"), ("Leo", "Leo", "사자자리"),
    ("LMi", "Leo Minor", "작은사자자리"), ("Lep", "Lepus", "토끼자리"),
    ("Lib", "Libra", "천칭자리"), ("Lup", "Lupus", "늑대자리"),
    ("Lyn", "Lynx", "살쾡이자리"), ("Lyr", "Lyra", "거문고자리"),
    ("Men", "Mensa", "테이블산자리"), ("Mic", "Microscopium", "현미경자리"),
    ("Mon", "Monoceros", "외뿔소자리"), ("Mus", "Musca", "파리자리"),
    ("Nor", "Norma", "직각자자리"), ("Oct", "Octans", "팔분의자리"),
    ("Oph", "Ophiuchus", "뱀주인자리"), ("Ori", "Orion", "오리온자리"),
    ("Pav", "Pavo", "공작자리"), ("Peg", "Pegasus", "페가수스자리"),
    ("Per", "Perseus", "페르세우스자리"), ("Phe", "Phoenix", "봉황자리"),
    ("Pic", "Pictor", "화가자리"), ("Psc", "Pisces", "물고기자리"),
    ("PsA", "Piscis Austrinus", "남쪽물고기자리"), ("Pup", "Puppis", "고물자리"),
    ("Pyx", "Pyxis", "나침반자리"), ("Ret", "Reticulum", "그물자리"),
    ("Sge", "Sagitta", "화살자리"), ("Sgr", "Sagittarius", "궁수자리"),
    ("Sco", "Scorpius", "전갈자리"), ("Scl", "Sculptor", "조각가자리"),
    ("Sct", "Scutum", "방패자리"), ("Ser", "Serpens", "뱀자리"),
    ("Sex", "Sextans", "육분의자리"), ("Tau", "Taurus", "황소자리"),
    ("Tel", "Telescopium", "망원경자리"), ("Tri", "Triangulum", "삼각형자리"),
    ("TrA", "Triangulum Australe", "남쪽삼각형자리"), ("Tuc", "Tucana", "큰부리새자리"),
    ("UMa", "Ursa Major", "큰곰자리"), ("UMi", "Ursa Minor", "작은곰자리"),
    ("Vel", "Vela", "돛자리"), ("Vir", "Virgo", "처녀자리"),
    ("Vol", "Volans", "날치자리"), ("Vul", "Vulpecula", "여우자리"),
)

# (Korean, English/designation, RA hours J2000, Dec deg J2000, V magnitude)
BRIGHT_STARS = (
    ("시리우스", "Sirius (Alpha CMa)", 6.752481, -16.716116, -1.46),
    ("카노푸스", "Canopus (Alpha Car)", 6.399198, -52.695661, -0.74),
    ("알파 센타우리", "Rigil Kentaurus (Alpha Cen)", 14.659989, -60.833976, -0.27),
    ("아르크투루스", "Arcturus (Alpha Boo)", 14.261021, 19.182409, -0.05),
    ("베가", "Vega (Alpha Lyr)", 18.615649, 38.783689, 0.03),
    ("카펠라", "Capella (Alpha Aur)", 5.278155, 45.997991, 0.08),
    ("리겔", "Rigel (Beta Ori)", 5.242298, -8.201640, 0.13),
    ("프로키온", "Procyon (Alpha CMi)", 7.655033, 5.224993, 0.34),
    ("아케르나르", "Achernar (Alpha Eri)", 1.628556, -57.236752, 0.46),
    ("베텔게우스", "Betelgeuse (Alpha Ori)", 5.919529, 7.407064, 0.50),
    ("하다르", "Hadar (Beta Cen)", 14.063729, -60.373039, 0.61),
    ("알타이르", "Altair (Alpha Aql)", 19.846389, 8.868322, 0.77),
    ("아크룩스", "Acrux (Alpha Cru)", 12.443300, -63.099093, 0.77),
    ("알데바란", "Aldebaran (Alpha Tau)", 4.598677, 16.509302, 0.85),
    ("안타레스", "Antares (Alpha Sco)", 16.490128, -26.432003, 0.96),
    ("스피카", "Spica (Alpha Vir)", 13.419883, -11.161322, 0.97),
    ("폴룩스", "Pollux (Beta Gem)", 7.755259, 28.026184, 1.14),
    ("포말하우트", "Fomalhaut (Alpha PsA)", 22.960788, -29.622236, 1.16),
    ("데네브", "Deneb (Alpha Cyg)", 20.690532, 45.280338, 1.25),
    ("미모사", "Mimosa (Beta Cru)", 12.795351, -59.688765, 1.25),
    ("레굴루스", "Regulus (Alpha Leo)", 10.139530, 11.967209, 1.35),
    ("아다라", "Adhara (Epsilon CMa)", 6.977012, -28.972126, 1.50),
    ("카스토르", "Castor (Alpha Gem)", 7.576632, 31.888276, 1.58),
    ("샤울라", "Shaula (Lambda Sco)", 17.560128, -37.103824, 1.62),
    ("가크룩스", "Gacrux (Gamma Cru)", 12.519432, -57.113213, 1.63),
    ("벨라트릭스", "Bellatrix (Gamma Ori)", 5.418852, 6.349703, 1.64),
    ("엘나트", "Elnath (Beta Tau)", 5.438198, 28.607450, 1.65),
    ("미아플라시두스", "Miaplacidus (Beta Car)", 9.220049, -69.717208, 1.67),
    ("알닐람", "Alnilam (Epsilon Ori)", 5.603559, -1.201919, 1.69),
)

# (Korean, English, peak month, peak day, radiant RA deg, radiant Dec deg, ZHR)
METEOR_SHOWERS = (
    ("사분의자리 유성우", "Quadrantids", 1, 3, 230.0, 49.0, 110),
    ("거문고자리 유성우", "Lyrids", 4, 22, 271.0, 34.0, 18),
    ("물병자리 에타 유성우", "Eta Aquariids", 5, 6, 338.0, -1.0, 50),
    ("물병자리 델타 유성우", "Delta Aquariids", 7, 30, 340.0, -16.4, 25),
    ("페르세우스자리 유성우", "Perseids", 8, 12, 48.0, 58.0, 100),
    ("오리온자리 유성우", "Orionids", 10, 21, 95.0, 16.0, 20),
    ("사자자리 유성우", "Leonids", 11, 17, 154.0, 22.0, 15),
    ("쌍둥이자리 유성우", "Geminids", 12, 14, 113.0, 32.0, 120),
    ("작은곰자리 유성우", "Ursids", 12, 22, 217.0, 76.0, 10),
)


def moon_phase_table(year):
    """Sorted list of MoonPhaseEntry(datetime, kind) for all phases in `year`."""
    entry_t = namedtuple("MoonPhaseEntry", "datetime kind")
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    out = []
    for kind in ("new", "first_quarter", "full", "last_quarter"):
        t = start
        for _ in range(16):
            t = next_phase(kind, t)
            if t >= end:
                break
            out.append(entry_t(t, kind))
            t = t + timedelta(minutes=1)
    return sorted(out)
