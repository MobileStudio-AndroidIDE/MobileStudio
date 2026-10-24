"""Line-level neural language engine (VSA + conceptual space + active inference).

No LLM, no Transformer, no pretrained weights, no external AI API.
Everything is built from:

* VSA (vector symbolic architecture) bind/unbind/bundle
* Conceptual spaces (prototypes + similarity)
* Active inference (surprise minimisation)
* Korean morphology (particles / endings as VSA roles)
* Meta-cognition (self-evaluation and revision)

Usage::

    from cognition.language_engine import LanguageEngine

    engine = LanguageEngine(vsa, concepts, active, meta)
    response = engine.respond("삼 더하기 오")
    print(response)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cognition.active import ActiveInference
from cognition.concept_space import ConceptSpace
from cognition.korean import KoreanSurface
from cognition.metacognition import MetaCognition
from cognition.predictive import PredictiveEngine
from cognition.semantic import SemanticEncoder
from cognition.vsa import VSA


def _normalize_text(text: str) -> str:
    """Unicode-normalise and clean whitespace (lowercase, strip)."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 1. VSA compositional semantics
# ---------------------------------------------------------------------------

class CompositionalVSA:
    """Encode/decode Korean-like sentences as VSA role-filler bundles.

    Structure::

        bind(role:subject, filler:삼) ⊗ bind(role:operator, 더하기)
        ⊗ bind(role:object, 오) ⊗ bind(role:particle, 은/가)
    """

    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 encoder: Optional[SemanticEncoder] = None):
        self.vsa = vsa
        self.concepts = concepts
        self.encoder = encoder

    def encode(self, text: str) -> np.ndarray:
        """Encode free text into a meaning-line unit."""
        tokens = self.encoder.tokenize(text) if self.encoder is not None \
            else text.split()
        parts = []
        roles = ["subject", "object", "verb", "particle", "modifier"]
        role_idx = 0
        for tok in tokens:
            role = roles[role_idx % len(roles)]
            role_idx += 1
            vec = self.concepts.prototype(tok) if tok in self.concepts \
                else self.vsa.symbol("tok:" + tok)
            parts.append(self.vsa.bind(self.vsa.symbol("role:" + role), vec))
        return self.vsa.bundle(parts)

    def decode(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Nearest concepts to a composite vector."""
        return self.concepts.classify(vector, k=k)


# ---------------------------------------------------------------------------
# 2. Conceptual space generation
# ---------------------------------------------------------------------------

class ConceptualGenerator:
    """Generate surface text from conceptual coordinates.

    Concepts are points in a high-dimensional space (VSA).  We keep a small
    dictionary of surface forms per concept and interpolate between them
    based on attribute vectors (tense, polarity, politeness).
    """

    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 encoder: Optional[SemanticEncoder] = None):
        self.vsa = vsa
        self.concepts = concepts
        self.encoder = encoder
        # surface_forms[concept_label] = [surface_string, ...]
        self.surface_forms: dict[str, list[str]] = {}

    def register_surface(self, concept: str, surface: str) -> None:
        self.surface_forms.setdefault(concept, []).append(surface)

    def generate(self, concept: str, attributes: dict[str, float]) -> str:
        """Pick the surface form closest to *attributes* in concept space."""
        forms = self.surface_forms.get(concept)
        if not forms:
            return concept
        # Encode each surface form and compare to attribute vector.
        best_form = forms[0]
        best_score = -1.0
        attr_vector = self._attributes_vector(attributes)
        for form in forms:
            fv = self.encoder.encode_text(form) if self.encoder is not None \
                else self.vsa.symbol("tok:" + form)
            score = self.vsa.similarity(attr_vector, fv)
            if score > best_score:
                best_score = score
                best_form = form
        return best_form

    def _attributes_vector(self, attributes: dict[str, float]) -> np.ndarray:
        parts = []
        for k, v in attributes.items():
            parts.append(self.vsa.bind(self.vsa.symbol("attr:" + k),
                                       self.vsa.symbol(str(v))))
        return self.vsa.bundle(parts) if parts else self.vsa.symbol("empty")


# ---------------------------------------------------------------------------
# 3. Active inference generation
# ---------------------------------------------------------------------------

class InferenceGenerator:
    """Generate text by minimising surprise (prediction error).

    We maintain a small set of candidate responses and score each by how
    well it matches the *goal concept* (e.g. "apology", "greeting").
    The candidate with lowest surprise wins.
    """

    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 active: ActiveInference):
        self.vsa = vsa
        self.concepts = concepts
        self.active = active

    def generate(self, goal: str, candidates: list[str]) -> str:
        """Pick the candidate closest to *goal* in VSA space."""
        goal_vec = self.concepts.prototype(goal) if goal in self.concepts \
            else self.vsa.symbol("tok:" + goal)
        best = candidates[0] if candidates else ""
        best_score = -1.0
        for cand in candidates:
            cv = self.concepts.prototype(cand) if cand in self.concepts \
                else self.vsa.symbol("tok:" + cand)
            score = self.vsa.similarity(goal_vec, cv)
            if score > best_score:
                best_score = score
                best = cand
        return best

    def surprise(self, text: str) -> float:
        """Surprise of *text* under current active-inference beliefs."""
        obs = self.active.observe("text", text, learn=False)
        return obs["surprise"]


# ---------------------------------------------------------------------------
# 4. Korean morphology VSA combination
# ---------------------------------------------------------------------------

class KoreanVSA:
    """Bind Korean morphological pieces (subject/object particle, ending) into VSA."""

    PARTICLES = {
        "subject": ["이", "가", "은", "는"],
        "object": ["을", "를"],
        "topic": ["는", "은"],
        "locative": ["에", "에서"],
    }
    ENDINGS = {
        "polite": ["요", "습니다"],
        "casual": ["야", "어"],
        "plain": ["다", "ㅂ니다"],
    }

    def __init__(self, vsa: VSA, concepts: ConceptSpace):
        self.vsa = vsa
        self.concepts = concepts

    def bind_sentence(self, subject: str, particle: str,
                      predicate: str, ending: str) -> np.ndarray:
        """Bind a Korean sentence into a single VSA vector."""
        parts = [
            self.vsa.bind(self.vsa.symbol("role:subject"),
                          self.concepts.prototype(subject)
                          if subject in self.concepts
                          else self.vsa.symbol("tok:" + subject)),
            self.vsa.bind(self.vsa.symbol("role:particle"),
                          self.vsa.symbol("tok:" + particle)),
            self.vsa.bind(self.vsa.symbol("role:predicate"),
                          self.concepts.prototype(predicate)
                          if predicate in self.concepts
                          else self.vsa.symbol("tok:" + predicate)),
            self.vsa.bind(self.vsa.symbol("role:ending"),
                          self.vsa.symbol("tok:" + ending)),
        ]
        return self.vsa.bundle(parts)

    def decode(self, vector: np.ndarray) -> str:
        """Roughly decode: nearest concepts for each role slot."""
        # Unbind each role slot and find nearest concept.
        slots = ["role:subject", "role:particle", "role:predicate", "role:ending"]
        parts = []
        for role in slots:
            unbound = self.vsa.unbind(vector, self.vsa.symbol(role))
            # nearest concept
            ranked = self.concepts.classify(unbound, k=1)
            parts.append(ranked[0][0] if ranked else "?")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# 5. Meta-cognitive self-revision
# ---------------------------------------------------------------------------

class MetaCognitiveGenerator:
    """Generate, self-evaluate, and revise until confidence is high enough."""

    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 active: ActiveInference, meta: MetaCognition,
                 generator: InferenceGenerator):
        self.vsa = vsa
        self.concepts = concepts
        self.active = active
        self.meta = meta
        self.generator = generator

    def generate(self, goal: str, candidates: list[str],
                 threshold: float = 0.5,
                 max_revisions: int = 3) -> tuple[str, float, int]:
        """Generate *goal*, self-evaluate, revise until *threshold*.

        Returns (final_text, confidence, revisions_used).
        """
        candidate = self.generator.generate(goal, candidates)
        confidence = self._confidence(candidate, goal)
        revisions = 0
        while revisions < max_revisions and confidence < threshold:
            revisions += 1
            # Simple revision: append a polite ending or rephrase
            candidate = self._revise(candidate, goal)
            confidence = self._confidence(candidate, goal)
        return candidate, confidence, revisions

    def _confidence(self, text: str, goal: str) -> float:
        text_vec = self.concepts.prototype(text) if text in self.concepts \
            else self.vsa.symbol("tok:" + text)
        goal_vec = self.concepts.prototype(goal) if goal in self.concepts \
            else self.vsa.symbol("tok:" + goal)
        return self.vsa.similarity(text_vec, goal_vec)

    def _revise(self, text: str, goal: str) -> str:
        """Append a polite ending or soften the tone."""
        if text.endswith("다"):
            return text[:-1] + "요"
        if text.endswith("습니다"):
            return text
        return text + " 해요"


# ---------------------------------------------------------------------------
# Unified engine
# ---------------------------------------------------------------------------

class LanguageEngine:
    """Unified entry point for the 5 non-LLM language methods."""

    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 active: ActiveInference, meta: MetaCognition,
                 encoder: Optional[SemanticEncoder] = None):
        self.vsa = vsa
        self.concepts = concepts
        self.active = active
        self.meta = meta
        self.encoder = encoder
        self.compositional = CompositionalVSA(vsa, concepts, encoder)
        self.conceptual = ConceptualGenerator(vsa, concepts, encoder)
        self.inference = InferenceGenerator(vsa, concepts, active)
        self.korean = KoreanVSA(vsa, concepts)
        self.meta_cog = MetaCognitiveGenerator(vsa, concepts, active, meta,
                                               self.inference)
        # 대화 시드는 뇌의 개념 공간을 오염시키지 않도록 별도 공간에 둔다.
        self.chat_concepts = ConceptSpace(vsa)
        # 근접 개념 매칭 임계값. 이보다 낮거나 2등과의 격차가 작으면
        # 대화 엔진이 답하지 않고 액추에이터(수학/코드/날씨...)로 넘긴다.
        self.match_threshold = 0.45
        self.match_margin = 0.12
        self.register_seed_concepts()

    def _lookup_response(self, key: str) -> str:
        """정확히 등록된 개념의 응답을 반환한다 (문장부호 차이 허용)."""
        stripped = key.rstrip("?.!。？！ ")
        for candidate in (key, stripped, stripped + "?"):
            for space in (self.chat_concepts, self.concepts):
                concept = space.get(candidate)
                if concept is None:
                    continue
                response_text = concept.attributes.get("response")
                if response_text:
                    return str(response_text)
        return ""

    def direct_response(self, text: str) -> str:
        """정확히 등록된 대화 시드에만 응답 (유사 매칭 없음).

        짧은 후속 표현("그럼?")이 시드로 오인되는 것을 막기 위해,
        호출자가 확실한 일치를 먼저 확인할 때 쓴다.
        """
        normalized = _normalize_text(text)
        if not normalized:
            return ""
        return self._lookup_response(normalized)

    def respond(self, text: str, goal: Optional[str] = None) -> str:
        """Respond to *text* using the full pipeline.

        확실한 대화 개념에 매칭될 때만 문자열을 반환하고, 그렇지 않으면
        빈 문자열을 반환해 호출자가 액추에이터 경로로 폴백하게 한다.
        """
        normalized_text = _normalize_text(text)
        if not normalized_text:
            return ""

        # 1) 등록된 대화 개념과 정확히 일치하면 저장된 응답을 돌려준다.
        direct = self._lookup_response(normalized_text)
        if direct:
            return direct

        # 2) VSA 유사도가 충분히 높고 2등과 격차가 있을 때만 인정한다.
        encoded = self.compositional.encode(text)
        ranked = self.chat_concepts.classify(encoded, k=2)
        if ranked:
            label, score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            if (score >= self.match_threshold
                    and score - runner_up >= self.match_margin):
                stored = self._lookup_response(label)
                if stored:
                    return stored

        # 3) 확신이 없으면 폴백 (액추에이터가 처리).
        return ""

    def respond_korean(self, subject: str, particle: str,
                       predicate: str, ending: str) -> str:
        """Bind and decode a Korean sentence."""
        bound = self.korean.bind_sentence(subject, particle, predicate, ending)
        return self.korean.decode(bound)

    def register_seed_concepts(self) -> None:
        """인사·감정·정체성 같은 일상 대화 개념을 등록한다.

        코딩/수학/이미지/날씨/파일/셸/지식 같은 작업성 요청은 여기서
        가로채지 않는다. 그런 요청은 결정적 액추에이터가 처리해야
        결과가 실제로 만들어지기 때문이다.
        """
        seed_pairs: list[tuple[str, str]] = [
            # ---- 인사 / 사회 ----
            ("안녕하세요", "안녕하세요! 오늘은 뭘 함께 해볼까요?"),
            ("안녕", "안녕! 반가워요."),
            "잘자",
            "안녕히 가세요! 다음에 또 불러주세요.",
            "고마워",
            "도움이 되었다니 기뻐요!",
            "우울해",
            "그런 날도 있죠. 괜찮아요, 이야기해줄래요?",
            "기분 좋아",
            "좋은 기분이라니 저도 기뻐요!",
            # ---- 정체 / 능력 ----
            "정체가 뭐야?",
            "저는 RENA-BR이에요. Transformer 없이 VSA와 능동 추론으로 작동하는 로컬 인지 에이전트예요.",
            "뭐할줄알아?",
            "코딩, 수학, 그림, 차트, 영상, 파일, 날씨, 일정, 검색을 다룰 수 있어요.",
            "너는 뭐야?",
            "저는 인공지능이 아닌 규칙 기반 인지 에이전트예요.",
            "제 이름은 뭐야?",
            "제 이름은 RENA-BR이에요.",
        ]

        # 목록은 (질문, 응답) 튜플과 교대로 나열된 낱개 문자열이 섞여 있다.
        # 낱개 문자열은 "질문, 응답, 질문, 응답..." 순서이므로 짝지어 준다.
        pairs: list[tuple[str, str]] = []
        pending: Optional[str] = None
        for item in seed_pairs:
            if isinstance(item, (tuple, list)):
                if len(item) != 2:
                    continue
                if pending is not None:
                    pairs.append((pending, pending))
                    pending = None
                pairs.append((str(item[0]), str(item[1])))
                continue
            if pending is None:
                pending = str(item)
            else:
                pairs.append((pending, str(item)))
                pending = None
        if pending is not None:
            pairs.append((pending, pending))

        for query, response_text in pairs:
            # 질문을 대화 전용 개념 공간에 등록한다. 조회 때와 같은
            # (compositional) 인코더를 써야 VSA 유사도가 의미를 가진다.
            query_vector = self.compositional.encode(query)
            self.chat_concepts.add(
                query, query_vector, modality="text",
                attributes={"response": response_text},
            )
