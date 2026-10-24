"""Korean surface-form handling for RENA-BR cognition.

This module lifts Korean typos, slang, emoticons and particles into
concept-level signals that the VSA/concept-space substrate can use.  It is
*not* a keyword rule engine: it produces weighted concept hints that are
combined with VSA similarity scores by the caller.  No pretrained model,
no external API, no backpropagation — only deterministic character and
morphology heuristics.

Typical usage::

    from cognition.korean import KoreanSurface
    from cognition.semantic import SemanticEncoder
    from cognition.vsa import VSA

    vsa = VSA(dimension=2048)
    encoder = SemanticEncoder(vsa)
    korean = KoreanSurface(vsa, encoder)

    hints = korean.hints("안뇽")           # {"hello": 1.0, "chat": 0.8}
    vector  = korean.encode("리눅수가 뭐야") # concept-boosted text vector
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np

from cognition.semantic import SemanticEncoder
from cognition.vsa import VSA


# ---------------------------------------------------------------------------
# Typo / slang / emoticon -> concept hints
#
# Each entry maps a Korean surface form (typo, slang, emoticon) to a dict of
# concept -> weight.  The caller merges these weights with VSA concept-space
# similarity scores.  A weight of 1.0 means "strong concept signal".
# ---------------------------------------------------------------------------

_TYPO_SLANG: dict[str, dict[str, float]] = {
    # Greetings / social
    "안뇽": {"hello": 1.0, "chat": 0.9},
    "안뇽하세요": {"hello": 1.0, "chat": 0.9},
    "ㅎㅇ": {"hello": 0.9, "chat": 0.8},
    "ㅎㅇㅎㅇ": {"hello": 0.9, "chat": 0.8},
    "반가워용": {"hello": 0.9, "chat": 0.8},
    "반가워": {"hello": 0.8, "chat": 0.7},
    "수고했어": {"positive": 0.7, "chat": 0.6},
    "수고하셨어요": {"positive": 0.7, "chat": 0.6},
    "잘자": {"goodbye": 0.9, "chat": 0.7},
    "잘 자": {"goodbye": 0.9, "chat": 0.7},
    "안녕히 가세요": {"goodbye": 0.9, "chat": 0.7},
    "바이": {"goodbye": 0.8, "chat": 0.6},
    "ㅂㅂ": {"goodbye": 0.8, "chat": 0.6},
    "ㄳ": {"thanks": 0.9, "positive": 0.6, "chat": 0.5},
    "감사": {"thanks": 0.9, "positive": 0.7},
    "고마워": {"thanks": 0.8, "positive": 0.7},
    "땡큐": {"thanks": 0.8, "positive": 0.6},
    "고마워요": {"thanks": 0.8, "positive": 0.7},
    "사랑해": {"love": 0.8, "chat": 0.6},
    "우울해": {"sad": 0.8, "chat": 0.6},
    "기분 좋아": {"positive": 0.8, "chat": 0.6},
    "기분이 좋아": {"positive": 0.8, "chat": 0.6},
    "심심해": {"bored": 0.7, "chat": 0.6},
    "심심한데": {"bored": 0.7, "chat": 0.6},

    # Identity / capability (chat)
    "정체가 뭐야": {"identity": 1.0, "capability": 0.9, "chat": 0.8},
    "정체가 뭐임": {"identity": 1.0, "capability": 0.9, "chat": 0.8},
    "너 누구야": {"identity": 0.9, "capability": 0.8, "chat": 0.7},
    "너는 누구야": {"identity": 0.9, "capability": 0.8, "chat": 0.7},
    "네 이름은": {"identity": 0.8, "capability": 0.7, "chat": 0.6},
    "뭐할줄알아": {"capability": 0.9, "chat": 0.7},
    "뭐 할 줄 알아": {"capability": 0.9, "chat": 0.7},
    "무엇을 할 수 있어": {"capability": 0.8, "chat": 0.7},

    # Code / programming (typos)
    "고야잉": {"cat": 1.0, "image": 0.8},
    "고양잉": {"cat": 1.0, "image": 0.8},
    "고야잉 그려줘": {"cat": 1.0, "image": 1.0},
    "피보나치": {"fibonacci": 1.0, "code": 0.8},
    "피보나취": {"fibonacci": 1.0, "code": 0.8},
    "피보나치 코드": {"fibonacci": 1.0, "code": 0.9},
    "피보나취 코드": {"fibonacci": 1.0, "code": 0.9},
    "피보나치 코드 짜줘": {"fibonacci": 1.0, "code": 1.0},
    "피보나취 코드 짜줘": {"fibonacci": 1.0, "code": 1.0},
    "파이선": {"python": 0.9, "code": 0.7},
    "파이선 코드": {"python": 0.9, "code": 0.8},
    "자스": {"javascript": 0.9, "code": 0.7},
    "자바스": {"javascript": 0.9, "code": 0.7},
    "자스로": {"javascript": 0.9, "code": 0.8},
    "자바로": {"java": 0.9, "code": 0.8},
    "씨언어": {"c": 0.9, "code": 0.8},
    "씨플플": {"cpp": 0.9, "code": 0.8},
    "러스트": {"rust": 0.9, "code": 0.8},
    "고언어": {"go": 0.9, "code": 0.8},
    "코틀린": {"kotlin": 0.9, "code": 0.8},

    # Code / programming languages
    "파이썬": {"python": 0.9, "code": 0.8},
    "파이선": {"python": 0.9, "code": 0.8},
    "자바스크립트": {"javascript": 0.9, "code": 0.8},
    "자스": {"javascript": 0.9, "code": 0.7},
    "자바스": {"javascript": 0.9, "code": 0.7},
    "자바": {"java": 0.9, "code": 0.8},
    "코틀린": {"kotlin": 0.9, "code": 0.8},
    "씨언어": {"c": 0.9, "code": 0.8},
    "씨플플": {"cpp": 0.9, "code": 0.8},
    "러스트": {"rust": 0.9, "code": 0.8},
    "고언어": {"go": 0.9, "code": 0.8},
    "코드": {"code": 0.7},
    "프로그램": {"code": 0.6},

    # Knowledge / topics (typos)
    "리눅수": {"linux": 1.0, "knowledge": 0.9},
    "리눅쓰": {"linux": 1.0, "knowledge": 0.9},
    "리눅스가": {"linux": 1.0, "knowledge": 0.9},
    "우분토": {"ubuntu": 1.0, "knowledge": 0.9},
    "우분투가": {"ubuntu": 1.0, "knowledge": 0.9},
    "리눅스가 뭐야": {"linux": 1.0, "knowledge": 1.0},
    "우분투가 머야": {"ubuntu": 1.0, "knowledge": 1.0},
    "양자 얽힘": {"quantum_entanglement": 1.0, "knowledge": 1.0},
    "AI": {"ai": 0.9, "knowledge": 0.8},
    "인공지능": {"ai": 0.9, "knowledge": 0.8},
    "에이아이": {"ai": 0.9, "knowledge": 0.8},

    # 지식 요청 키워드 (코드 생성과 구분)
    "뭐야": {"knowledge": 0.8},
    "무엇": {"knowledge": 0.7},
    "설명": {"knowledge": 0.8},
    "설명해": {"knowledge": 0.8},
    "알려줘": {"knowledge": 0.7},
    "알려": {"knowledge": 0.6},
    "어떻게": {"knowledge": 0.7},
    "왜": {"knowledge": 0.7},
    "무슨": {"knowledge": 0.6},
    "어떤": {"knowledge": 0.6},

    # Calendar / schedule
    "일정": {"schedule": 0.9, "calendar": 0.8},
    "스케줄": {"schedule": 0.9, "calendar": 0.8},
    "캘린더": {"calendar": 0.9},
    "내일": {"tomorrow": 0.9, "schedule": 0.6},
    "모레": {"day_after_tomorrow": 0.8, "schedule": 0.5},
    "오늘": {"today": 0.9, "schedule": 0.5},
    "약속": {"schedule": 0.8, "calendar": 0.7},
    "예약": {"schedule": 0.8, "calendar": 0.7},

    # Weather
    "날씨": {"weather": 1.0},
    "기온": {"weather": 0.8},
    "서울 날씨": {"weather": 1.0, "location:서울": 1.0},
    "비": {"weather": 0.9, "raining": 0.9},
    "맑음": {"weather": 0.8},
    "흐림": {"weather": 0.8, "raining": 0.5},
    "우산": {"weather": 0.8, "raining": 0.9},
    "태풍": {"weather": 0.9, "raining": 1.0},

    # Image / draw
    "그려": {"draw": 0.9, "image": 0.8},
    "그림": {"image": 0.8},
    "이미지": {"image": 0.8},
    "사진": {"image": 0.7},
    "풍경": {"image": 0.7, "scene": 0.8},
    "우주": {"space": 0.9, "image": 0.7},
    "밤하늘": {"night": 0.9, "image": 0.7},
    "별": {"star": 0.8, "image": 0.6},
    "달": {"moon": 0.8, "image": 0.6},
    "강아지": {"dog": 0.8, "image": 0.6},
    "고양이": {"cat": 0.9, "image": 0.6},
    "꽃": {"flower": 0.8, "image": 0.6},
    "로봇": {"robot": 0.8, "image": 0.6},
    "나무": {"tree": 0.7, "image": 0.6},
    "바다": {"sea": 0.7, "image": 0.6},
    "산": {"mountain": 0.7, "image": 0.6},

    # Math
    "더하기": {"add": 0.9, "math": 0.8},
    "빼기": {"subtract": 0.8, "math": 0.7},
    "곱하기": {"multiply": 0.8, "math": 0.7},
    "나누기": {"divide": 0.8, "math": 0.7},
    "제곱": {"power": 0.8, "math": 0.7},
    "승": {"power": 0.7, "math": 0.6},
    "팩토리얼": {"factorial": 0.9, "math": 0.8, "code": 0.7},
    "함수": {"code": 0.6},
    "계승": {"factorial": 0.7, "math": 0.6},
    "미분": {"differentiate": 0.9, "math": 0.8},
    "적분": {"integrate": 0.8, "math": 0.7},
    "소수": {"prime": 0.8, "math": 0.7},
    "합": {"sum": 0.8, "math": 0.7},
    "평균": {"average": 0.8, "math": 0.7},

    # Shell / system
    "pwd": {"shell": 0.9, "system": 0.7},
    "ls": {"shell": 0.9, "system": 0.7},
    "터미널": {"shell": 0.8, "system": 0.6},
    "명령어": {"shell": 0.8, "system": 0.6},

    # File
    "저장": {"file_write": 0.8, "files": 0.7},
    "읽어": {"file_read": 0.8, "files": 0.7},
    "파일": {"files": 0.7},
}


# Single-syllable Korean particles that attach to content words.  We do NOT
# strip these blindly — "이/가" can be part of a noun (고양이, 아이).  We only
# treat them as removable suffixes when the remaining stem is still meaningful
# (length >= 3 for a 1-char suffix).
_PARTICLE_SUFFIXES = ("은", "는", "을", "를", "에", "에서", "에게", "으로", "로",
                      "와", "과", "도", "만", "의", "야", "아", "이", "가",
                      "이야", "이야", "랑", "이랑")


def _strip_trailing_particle(token: str) -> Optional[str]:
    """If *token* looks like a content word + particle, return the stem.

    Returns None when the stem would be too short or when the syllable is
    likely part of the noun itself (e.g. 고양이 -> 고양 would be wrong).
    """
    if len(token) <= 2:
        return None
    for suffix in _PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            stem = token[: -len(suffix)]
            if len(stem) >= 2:
                return stem
    return None


class KoreanSurface:
    """Korean surface-form -> concept hints and concept-boosted VSA vectors."""

    def __init__(self, vsa: VSA, encoder: SemanticEncoder):
        self.vsa = vsa
        self.encoder = encoder
        # Compile typo/slang patterns: longer keys first so "안뇽하세요"
        # matches before "안뇽".
        self._keys = sorted(_TYPO_SLANG.keys(), key=len, reverse=True)
        self._patterns = [
            (re.compile(re.escape(key), re.IGNORECASE), weights)
            for key, weights in _TYPO_SLANG.items()
        ]

    # ------------------------------------------------------------------
    # Concept hints (used by intent routing, not hardcoded dispatch)
    # ------------------------------------------------------------------

    def hints(self, text: str) -> dict[str, float]:
        """Return concept -> weight hints found in *text*.

        Multiple matches are merged by taking the maximum weight per concept.
        Matching is token-based (with word boundaries) so a short key like
        ``산`` does not fire inside ``계산``.  Particle-stripped stems are
        also checked so "리눅수가" still yields the "linux" hint.
        """
        merged: dict[str, float] = {}
        tokens = self.encoder.tokenize(text)
        token_set = set(t.lower() for t in tokens)

        for pattern, weights in self._patterns:
            key = pattern.pattern
            # Strip regex escapes for comparison.
            plain = key.replace("\\", "")
            matched = False
            if len(plain) <= 2:
                # Short keys: require an exact token match (word boundary).
                if plain.lower() in token_set:
                    matched = True
            else:
                # Longer keys: substring search on the full text.
                if pattern.search(text.lower()):
                    matched = True
            if matched:
                for concept, weight in weights.items():
                    if weight > merged.get(concept, 0.0):
                        merged[concept] = weight

        # Also check particle-stripped stems for knowledge hints.
        for token in tokens:
            stem = _strip_trailing_particle(token)
            if stem is None:
                continue
            stem_lower = stem.lower()
            for pattern, weights in self._patterns:
                if pattern.fullmatch(stem_lower):
                    for concept, weight in weights.items():
                        scaled = weight * 0.9  # slightly lower confidence
                        if scaled > merged.get(concept, 0.0):
                            merged[concept] = scaled

        return merged

    # ------------------------------------------------------------------
    # Concept-boosted VSA encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode *text* as a VSA vector with concept-boosted dimensions.

        The base vector comes from the shared semantic encoder; concept hints
        from Korean surface forms are added as extra bundle components so that
        a typo like "리눅수" still points near the "linux" concept without
        hardcoding any intent dispatch.
        """
        base = self.encoder.encode_text(text)
        hints = self.hints(text)

        if not hints:
            return base

        parts = [base]
        weights = [1.0]
        for concept, weight in hints.items():
            concept_vector = self.vsa.symbol("concept:" + concept)
            parts.append(concept_vector)
            weights.append(0.5 * weight)

        return self.vsa.bundle(parts, weights=weights)

    def route_hints(self, text: str) -> list[tuple[str, float]]:
        """Return the top (concept, weight) hint pairs for routing."""
        hints = self.hints(text)
        return sorted(hints.items(), key=lambda item: -item[1])[:8]


def quick_hints(text: str) -> dict[str, float]:
    """Convenience: hints without constructing a KoreanSurface instance."""
    from cognition.vsa import VSA
    from cognition.semantic import SemanticEncoder

    vsa = VSA(dimension=2048)
    encoder = SemanticEncoder(vsa)
    return KoreanSurface(vsa, encoder).hints(text)


def quick_encode(text: str) -> np.ndarray:
    """Convenience: concept-boosted encode without constructing instances."""
    from cognition.vsa import VSA
    from cognition.semantic import SemanticEncoder

    vsa = VSA(dimension=2048)
    encoder = SemanticEncoder(vsa)
    return KoreanSurface(vsa, encoder).encode(text)
