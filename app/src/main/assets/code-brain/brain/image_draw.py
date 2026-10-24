"""Natural-language DRAWING: scenes composed from geometric primitives.

"사람 그려줘" / "draw a person and a bed" produce an actual IMAGE: the
request is parsed into the objects it names, and each object is a
parameterized icon built from lines, rectangles, circles, ellipses and
polygons (image_io's raster primitives), laid out on a ground line.

This is compositional symbol drawing — the deterministic, offline
alternative to a learned image generator.  It draws what it has icons
for and says so honestly when an object is not in the icon ontology;
it never fakes a photorealistic result.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from brain import image_io as ii


@dataclass
class SceneItem:
    icon: str
    label: str
    evidence: str


# --------------------------------------------------------------------- #
# Icon ontology: object words (EN + KO) -> icon renderer
# --------------------------------------------------------------------- #

def _icon_person(img, x, y, w, h, color):
    skin = (245, 205, 160)
    cx = x + w // 2
    ii.draw_circle(img, cx, y + int(h * 0.12), max(4, int(w * 0.16)),
                   skin, filled=True)
    ii.draw_rect(img, x + int(w * 0.30), y + int(h * 0.24),
                 max(4, int(w * 0.40)), int(h * 0.36), color, filled=True)
    ii.draw_line(img, x + int(w * 0.30), y + int(h * 0.30),
                 x + int(w * 0.08), y + int(h * 0.52), skin)
    ii.draw_line(img, x + int(w * 0.30), y + int(h * 0.28),
                 x + int(w * 0.05), y + int(h * 0.44), skin)
    ii.draw_line(img, x + int(w * 0.70), y + int(h * 0.30),
                 x + int(w * 0.92), y + int(h * 0.52), skin)
    pants = (62, 74, 120)
    ii.draw_rect(img, cx - int(w * 0.16), y + int(h * 0.60),
                 int(w * 0.14), int(h * 0.34), pants, filled=True)
    ii.draw_rect(img, cx + int(w * 0.02), y + int(h * 0.60),
                 int(w * 0.14), int(h * 0.34), pants, filled=True)


def _icon_bed(img, x, y, w, h, color):
    frame = (150, 100, 60)
    ii.draw_rect(img, x, y + int(h * 0.30), int(w * 0.14), int(h * 0.62),
                 frame, filled=True)                      # headboard
    ii.draw_rect(img, x + int(w * 0.10), y + int(h * 0.58),
                 int(w * 0.86), int(h * 0.18), frame, filled=True)  # base
    ii.draw_rect(img, x + int(w * 0.12), y + int(h * 0.46),
                 int(w * 0.82), int(h * 0.14), (250, 250, 255),
                 filled=True)                             # mattress
    ii.draw_rect(img, x + int(w * 0.16), y + int(h * 0.36),
                 int(w * 0.22), int(h * 0.12), (255, 255, 255),
                 filled=True)                             # pillow
    ii.draw_rect(img, x + int(w * 0.40), y + int(h * 0.44),
                 int(w * 0.54), int(h * 0.16), color, filled=True)  # blanket
    ii.draw_rect(img, x + int(w * 0.12), y + int(h * 0.76),
                 int(w * 0.06), int(h * 0.16), frame, filled=True)
    ii.draw_rect(img, x + int(w * 0.86), y + int(h * 0.76),
                 int(w * 0.06), int(h * 0.16), frame, filled=True)


def _icon_house(img, x, y, w, h, color):
    wall = (235, 220, 180)
    ii.draw_rect(img, x + int(w * 0.08), y + int(h * 0.42),
                 int(w * 0.84), int(h * 0.58), wall, filled=True)
    ii.draw_polygon(img, [
        (x + int(w * 0.02), y + int(h * 0.44)),
        (x + int(w * 0.50), y + int(h * 0.06)),
        (x + int(w * 0.98), y + int(h * 0.44)),
    ], color, filled=True)
    ii.draw_rect(img, x + int(w * 0.42), y + int(h * 0.62),
                 int(w * 0.18), int(h * 0.38), (120, 80, 50), filled=True)
    ii.draw_rect(img, x + int(w * 0.14), y + int(h * 0.52),
                 int(w * 0.16), int(h * 0.16), (170, 215, 245),
                 filled=True)
    ii.draw_rect(img, x + int(w * 0.72), y + int(h * 0.52),
                 int(w * 0.14), int(h * 0.14), (170, 215, 245),
                 filled=True)


def _icon_tree(img, x, y, w, h, color):
    trunk = (120, 82, 45)
    ii.draw_rect(img, x + int(w * 0.44), y + int(h * 0.55),
                 int(w * 0.12), int(h * 0.45), trunk, filled=True)
    leaf = (58, 150, 74)
    ii.draw_circle(img, x + int(w * 0.50), y + int(h * 0.34),
                   int(w * 0.30), leaf, filled=True)
    ii.draw_circle(img, x + int(w * 0.30), y + int(h * 0.46),
                   int(w * 0.22), leaf, filled=True)
    ii.draw_circle(img, x + int(w * 0.70), y + int(h * 0.46),
                   int(w * 0.22), leaf, filled=True)


def _icon_sun(img, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    r = max(6, int(min(w, h) * 0.30))
    ii.draw_circle(img, cx, cy, r, (255, 196, 0), filled=True)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, 1), (1, -1), (-1, -1)):
        ii.draw_line(img, cx + dx * int(r * 1.4), cy + dy * int(r * 1.4),
                     cx + dx * int(r * 2.0), cy + dy * int(r * 2.0),
                     (255, 170, 0))


def _icon_cloud(img, x, y, w, h, color):
    white = (250, 250, 252)
    ii.draw_ellipse(img, x + int(w * 0.32), y + int(h * 0.58),
                    int(w * 0.24), int(h * 0.22), white, filled=True)
    ii.draw_ellipse(img, x + int(w * 0.55), y + int(h * 0.40),
                    int(w * 0.26), int(h * 0.26), white, filled=True)
    ii.draw_ellipse(img, x + int(w * 0.76), y + int(h * 0.58),
                    int(w * 0.20), int(h * 0.20), white, filled=True)
    ii.draw_rect(img, x + int(w * 0.12), y + int(h * 0.58),
                 int(w * 0.76), int(h * 0.22), white, filled=True)


def _icon_car(img, x, y, w, h, color):
    ii.draw_rect(img, x + int(w * 0.02), y + int(h * 0.52),
                 int(w * 0.96), int(h * 0.30), color, filled=True)
    ii.draw_polygon(img, [
        (x + int(w * 0.20), y + int(h * 0.52)),
        (x + int(w * 0.32), y + int(h * 0.20)),
        (x + int(w * 0.68), y + int(h * 0.20)),
        (x + int(w * 0.80), y + int(h * 0.52)),
    ], color, filled=True)
    ii.draw_rect(img, x + int(w * 0.36), y + int(h * 0.26),
                 int(w * 0.12), int(h * 0.22), (175, 215, 245),
                 filled=True)
    ii.draw_rect(img, x + int(w * 0.52), y + int(h * 0.26),
                 int(w * 0.12), int(h * 0.22), (175, 215, 245),
                 filled=True)
    ii.draw_circle(img, x + int(w * 0.24), y + int(h * 0.84),
                   max(4, int(w * 0.09)), (45, 45, 50), filled=True)
    ii.draw_circle(img, x + int(w * 0.76), y + int(h * 0.84),
                   max(4, int(w * 0.09)), (45, 45, 50), filled=True)


def _icon_chair(img, x, y, w, h, color):
    ii.draw_rect(img, x + int(w * 0.30), y + int(h * 0.06),
                 int(w * 0.16), int(h * 0.56), color, filled=True)
    ii.draw_rect(img, x + int(w * 0.18), y + int(h * 0.56),
                 int(w * 0.64), int(h * 0.12), color, filled=True)
    for lx in (0.22, 0.72):
        ii.draw_rect(img, x + int(w * lx), y + int(h * 0.68),
                     int(w * 0.07), int(h * 0.30), (110, 80, 55),
                     filled=True)


def _icon_table(img, x, y, w, h, color):
    ii.draw_rect(img, x + int(w * 0.08), y + int(h * 0.30),
                 int(w * 0.84), int(h * 0.12), color, filled=True)
    for lx in (0.14, 0.82):
        ii.draw_rect(img, x + int(w * lx), y + int(h * 0.42),
                     int(w * 0.06), int(h * 0.56), (120, 82, 45),
                     filled=True)


def _star_points(cx, cy, outer, inner, points=5):
    import math
    result = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / points
        result.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    return result


def _icon_star(img, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    outer = max(6, int(min(w, h) * 0.48))
    ii.draw_polygon(img, _star_points(cx, cy, outer, outer // 3.6),
                    (255, 196, 0), filled=True)


def _icon_heart(img, x, y, w, h, color):
    red = (226, 68, 82)
    cx, cy = x + w // 2, y + int(h * 0.42)
    r = max(4, int(min(w, h) * 0.22))
    ii.draw_circle(img, cx - r, cy, r, red, filled=True)
    ii.draw_circle(img, cx + r, cy, r, red, filled=True)
    ii.draw_polygon(img, [
        (cx - 2 * r, cy + int(r * 0.3)),
        (cx + 2 * r, cy + int(r * 0.3)),
        (cx, cy + int(r * 2.6)),
    ], red, filled=True)


def _icon_flower(img, x, y, w, h, color):
    cx, cy = x + w // 2, y + int(h * 0.30)
    ii.draw_line(img, cx, cy + int(h * 0.16), cx, y + h,
                 (58, 150, 74))
    ii.draw_ellipse(img, cx - int(w * 0.14), y + int(h * 0.70),
                    int(w * 0.12), int(h * 0.05), (58, 150, 74),
                    filled=True)
    ii.draw_ellipse(img, cx + int(w * 0.14), y + int(h * 0.58),
                    int(w * 0.12), int(h * 0.05), (58, 150, 74),
                    filled=True)
    pr = max(3, int(min(w, h) * 0.10))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (1, -1), (-1, 1)):
        ii.draw_circle(img, cx + dx * pr, cy + dy * pr, pr, color,
                       filled=True)
    ii.draw_circle(img, cx, cy, pr, (255, 196, 0), filled=True)


def _icon_cat(img, x, y, w, h, color):
    body = color
    cx = x + int(w * 0.44)
    ii.draw_ellipse(img, x + int(w * 0.46), y + int(h * 0.68),
                    int(w * 0.30), int(h * 0.26), body, filled=True)
    ii.draw_circle(img, cx, y + int(h * 0.32), max(5, int(w * 0.20)),
                   body, filled=True)
    ii.draw_polygon(img, [
        (cx - int(w * 0.18), y + int(h * 0.20)),
        (cx - int(w * 0.08), y - int(h * 0.02)),
        (cx - int(w * 0.00), y + int(h * 0.14)),
    ], body, filled=True)
    ii.draw_polygon(img, [
        (cx + int(w * 0.18), y + int(h * 0.20)),
        (cx + int(w * 0.08), y - int(h * 0.02)),
        (cx + int(w * 0.00), y + int(h * 0.14)),
    ], body, filled=True)
    eye = (30, 30, 30)
    ii.draw_circle(img, cx - int(w * 0.07), y + int(h * 0.30), 2, eye,
                   filled=True)
    ii.draw_circle(img, cx + int(w * 0.07), y + int(h * 0.30), 2, eye,
                   filled=True)
    ii.draw_line(img, cx + int(w * 0.30), y + int(h * 0.70),
                 x + int(w * 0.92), y + int(h * 0.42), body)


def _icon_dog(img, x, y, w, h, color):
    body = color
    cx = x + int(w * 0.42)
    ii.draw_ellipse(img, x + int(w * 0.46), y + int(h * 0.66),
                    int(w * 0.30), int(h * 0.24), body, filled=True)
    ii.draw_circle(img, cx, y + int(h * 0.34), max(5, int(w * 0.20)),
                   body, filled=True)
    ii.draw_ellipse(img, cx - int(w * 0.18), y + int(h * 0.34),
                    int(w * 0.08), int(h * 0.14), body, filled=True)
    ii.draw_ellipse(img, cx + int(w * 0.18), y + int(h * 0.34),
                    int(w * 0.08), int(h * 0.14), body, filled=True)
    ii.draw_ellipse(img, cx + int(w * 0.18), y + int(h * 0.42),
                    int(w * 0.10), int(h * 0.08), (250, 250, 250),
                    filled=True)
    ii.draw_circle(img, cx + int(w * 0.22), y + int(h * 0.42), 2,
                   (30, 30, 30), filled=True)
    ii.draw_circle(img, cx - int(w * 0.07), y + int(h * 0.32), 2,
                   (30, 30, 30), filled=True)
    ii.draw_line(img, x + int(w * 0.72), y + int(h * 0.60),
                 x + int(w * 0.88), y + int(h * 0.30), body)
    for lx in (0.28, 0.58):
        ii.draw_rect(img, x + int(w * lx), y + int(h * 0.84),
                     int(w * 0.08), int(h * 0.16), body, filled=True)


def _icon_moon(img, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    r = max(6, int(min(w, h) * 0.40))
    ii.draw_circle(img, cx, cy, r, (250, 232, 160), filled=True)
    ii.draw_circle(img, cx + int(r * 0.55), cy - int(r * 0.20), r,
                   (245, 247, 252), filled=True)


def _icon_mountain(img, x, y, w, h, color):
    ii.draw_polygon(img, [
        (x, y + h),
        (x + int(w * 0.42), y + int(h * 0.12)),
        (x + int(w * 0.86), y + h),
    ], (110, 130, 150), filled=True)
    ii.draw_polygon(img, [
        (x + int(w * 0.30), y + int(h * 0.32)),
        (x + int(w * 0.42), y + int(h * 0.12)),
        (x + int(w * 0.55), y + int(h * 0.34)),
        (x + int(w * 0.46), y + int(h * 0.28)),
        (x + int(w * 0.40), y + int(h * 0.36)),
    ], (250, 250, 255), filled=True)
    ii.draw_polygon(img, [
        (x + int(w * 0.50), y + h),
        (x + int(w * 0.74), y + int(h * 0.44)),
        (x + w, y + h),
    ], (90, 108, 128), filled=True)


def _icon_robot(img, x, y, w, h, color):
    metal = color
    cx = x + w // 2
    ii.draw_rect(img, cx - int(w * 0.20), y + int(h * 0.06),
                 int(w * 0.40), int(h * 0.26), metal, filled=True)
    ii.draw_circle(img, cx - int(w * 0.08), y + int(h * 0.18), 3,
                   (60, 230, 240), filled=True)
    ii.draw_circle(img, cx + int(w * 0.08), y + int(h * 0.18), 3,
                   (60, 230, 240), filled=True)
    ii.draw_rect(img, cx - int(w * 0.26), y + int(h * 0.34),
                 int(w * 0.52), int(h * 0.36), metal, filled=True)
    ii.draw_line(img, cx, y + int(h * 0.02), cx, y - int(h * 0.06),
                 (90, 90, 95))
    ii.draw_circle(img, cx, y - int(h * 0.08), 3, (230, 70, 70),
                   filled=True)
    for lx in (0.28, 0.62):
        ii.draw_rect(img, x + int(w * lx), y + int(h * 0.70),
                     int(w * 0.10), int(h * 0.30), metal, filled=True)


ICONS: dict[str, dict] = {
    "person": {"draw": _icon_person, "ratio": 0.72,
               "words": ("person", "man", "woman", "human", "사람", "남자",
                         "여자", "아이", "사내", "친구")},
    "bed": {"draw": _icon_bed, "ratio": 0.42,
            "words": ("bed", "침대")},
    "house": {"draw": _icon_house, "ratio": 0.62,
              "words": ("house", "home", "집", "주택", "건물")},
    "tree": {"draw": _icon_tree, "ratio": 0.66,
             "words": ("tree", "나무")},
    "sun": {"draw": _icon_sun, "ratio": 0.30, "sky": True,
            "words": ("sun", "태양", "해", "햇살")},
    "cloud": {"draw": _icon_cloud, "ratio": 0.24, "sky": True,
              "words": ("cloud", "구름")},
    "car": {"draw": _icon_car, "ratio": 0.36,
            "words": ("car", "자동차", "차", "차량", "택시", "버스")},
    "chair": {"draw": _icon_chair, "ratio": 0.48,
              "words": ("chair", "의자")},
    "table": {"draw": _icon_table, "ratio": 0.44,
              "words": ("table", "테이블", "탁자", "책상")},
    "star": {"draw": _icon_star, "ratio": 0.22, "sky": True,
             "words": ("star", "별")},
    "heart": {"draw": _icon_heart, "ratio": 0.26, "sky": True,
              "words": ("heart", "하트", "심장")},
    "flower": {"draw": _icon_flower, "ratio": 0.56,
               "words": ("flower", "꽃", "장미", "튤립")},
    "cat": {"draw": _icon_cat, "ratio": 0.42,
            "words": ("cat", "고양이", "야옹")},
    "dog": {"draw": _icon_dog, "ratio": 0.45,
            "words": ("dog", "puppy", "강아지", "개", "멍멍")},
    "moon": {"draw": _icon_moon, "ratio": 0.26, "sky": True,
             "words": ("moon", "달")},
    "mountain": {"draw": _icon_mountain, "ratio": 0.56,
                 "words": ("mountain", "산", "봉우리")},
    "robot": {"draw": _icon_robot, "ratio": 0.62,
              "words": ("robot", "로봇", "안드로이드")},
}

_DRAW_VERBS = ("그려", "그림", "그리", "draw", "paint", "sketch", "depict",
               "illustrate")

#: word -> icon, matched on WHOLE tokens (Korean short words like "개"
#: must never match inside another word).
_WORD_TO_ICON: dict[str, str] = {}
for _icon, _meta in ICONS.items():
    for _word in _meta["words"]:
        _WORD_TO_ICON[_word] = _icon


_KOREAN_RE = re.compile(r"[가-힣]")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,.;:!?'\"()\[\]{}<>]+", text.lower())
            if t]


def _icon_for_token(token: str) -> Optional[str]:
    """Word -> icon, tolerant of Korean particles and English plurals.

    Korean is agglutinative: "사람이랑/사람을" carry the noun "사람".
    Prefix matching is allowed only for words of TWO OR MORE syllables —
    a one-syllable word like "개" (dog) must never match inside "개수"
    (count).  English: "cars"/"houses"/"trees" match their singular.
    """
    icon = _WORD_TO_ICON.get(token)
    if icon:
        return icon
    if _KOREAN_RE.search(token):
        for word, icon_name in _WORD_TO_ICON.items():
            if len(word) >= 2 and token.startswith(word):
                return icon_name
        return None
    for suffix in ("s", "es"):
        if token.endswith(suffix) and len(token) > len(suffix):
            base = token[:-len(suffix)]
            if base in _WORD_TO_ICON:
                return _WORD_TO_ICON[base]
    return None


def parse_draw_request(text: str) -> list[SceneItem]:
    """The objects a drawing request names, in first-mention order.

    Empty when the request is not a drawing request (no draw verb) or
    names no object the icon ontology knows.
    """
    tokens = _tokens(text or "")
    if not tokens:
        return []
    lower = (text or "").lower()
    has_verb = any(v in lower for v in _DRAW_VERBS)
    if not has_verb:
        return []
    found: list[SceneItem] = []
    seen: set[str] = set()
    for token in tokens:
        icon = _icon_for_token(token)
        if icon and icon not in seen:
            seen.add(icon)
            found.append(SceneItem(icon=icon, label=token, evidence=token))
        if len(found) >= 6:
            break
    return found


def supported_objects() -> list[str]:
    return sorted(ICONS)


_LANDSCAPE_ICONS = {"tree", "house", "mountain", "car", "flower", "sun",
                    "cloud"}


def render_draw_request(text: str, width: int = 640, height: int = 480
                        ) -> tuple[ii.Image, dict]:
    """Render the drawing request into an image + its scene metadata."""
    items = parse_draw_request(text)
    if not items:
        raise ii.ImageError(
            "no drawable object found — try: "
            + ", ".join(supported_objects()))
    landscape = any(i.icon in _LANDSCAPE_ICONS for i in items) \
        or len(items) > 1
    background = (235, 244, 252) if landscape else (252, 252, 250)
    img = ii.new_image(width, height, background)
    ground_y = int(height * 0.82)
    if landscape:
        ii.draw_rect(img, 0, ground_y, width, height - ground_y,
                     (196, 224, 186), filled=True)
        ii.draw_line(img, 0, ground_y, width, ground_y, (150, 190, 140))

    ground_items = [i for i in items if not ICONS[i.icon].get("sky")]
    sky_items = [i for i in items if ICONS[i.icon].get("sky")]

    # Sky objects spread across the top
    if sky_items:
        slot = width / (len(sky_items) + 1)
        for index, item in enumerate(sky_items):
            box_w = int(width * 0.16)
            box_h = int(box_w / max(0.2, ICONS[item.icon]["ratio"]))
            cx = int(slot * (index + 1))
            cy = int(height * (0.14 + 0.16 * (index % 2)))
            icon_meta = ICONS[item.icon]
            icon_meta["draw"](img, cx - box_w // 2, cy, box_w, box_h,
                              (255, 196, 0))

    # Ground objects side by side, standing on the ground line
    if ground_items:
        slot = width / len(ground_items)
        palette = [(66, 133, 244), (219, 68, 55), (15, 157, 88),
                   (171, 71, 188), (0, 172, 193), (255, 112, 67)]
        for index, item in enumerate(ground_items):
            icon_meta = ICONS[item.icon]
            box_h = int(height * 0.62 * icon_meta["ratio"] / 0.72)
            box_w = int(slot * 0.66)
            if box_w > int(box_h * 1.4):
                box_w = int(box_h * 1.4)
            x = int(slot * index + (slot - box_w) / 2)
            y = ground_y - box_h
            icon_meta["draw"](img, x, y, box_w, box_h,
                              palette[index % len(palette)])
    meta = {
        "objects": [i.icon for i in items],
        "labels": [i.label for i in items],
        "width": width,
        "height": height,
        "summary": "scene: " + ", ".join(i.icon for i in items),
    }
    return img, meta


def draw_request(text: str, output_path: str, width: int = 640,
                 height: int = 480) -> dict:
    """Render and save a drawing request; returns its metadata."""
    img, meta = render_draw_request(text, width, height)
    ii.save_image(img, output_path)
    meta["path"] = output_path
    return meta


def default_output_path(text: str) -> str:
    """A deterministic-ish artifact path under the brain data directory."""
    directory = os.path.join(os.path.expanduser("~"), ".rena_brain",
                             "images")
    os.makedirs(directory, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9가-힣]+", "_", (text or "draw"))[:24].strip("_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"draw_{slug}_{stamp}.png")
