"""비결정론적 생성기 — 표현은 매번 달라지고 사실은 그대로.

LLM의 샘플링처럼 "같은 질문에 다른 답"을 만들되, 내용(사실)은 절대
바뀌지 않게 한다.  하는 일은 **표면 실현(surface realization)** 뿐이다.

    사실(불변)  ->  [틀 선택 · 어순 · 연결어 · 동의어]  ->  문장(가변)

방법:
1. 문장 틀 여러 개 중 샘플링 (random.Random)
2. 담화 도입부/맺음말 변형
3. 어순·묶음 변형
4. 동의어 치환 (offline_kb 용어 사전, 선택적 lexicon)
5. 최근에 쓴 표현 회피 (같은 말 반복 방지)

안전장치:
- **충실성 검사**: 실현 후 핵심 용어가 사라졌으면 정본(canonical)으로
  되돌린다.  표현이 사실을 훼손하지 못한다.
- seed 를 주면 재현 가능(결정적), 주지 않으면 매번 다르다.

공개 API::

    realizer = Realizer(seed=None)
    realizer.realize(segments, key_terms=["pwd"])   # 문장 만들기
    realizer.rewrite("...")                          # 같은 뜻, 다른 표현
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional


OPENERS = (
    "", "", "", "정리하면, ", "간단히 말해, ", "쉽게 말하면, ",
    "바로 답부터 말하면, ", "핵심만 짚으면, ",
)

CLOSERS = (
    "", "", "더 알고 싶은 부분이 있으면 말해 주세요.",
    "예시를 더 보여드릴까요?", "이어서 설명해 드릴까요?",
    "다른 명령도 궁금하면 물어보세요.",
)

BULLET_FRAMES = (
    "- {text}", "• {text}", "· {text}", "  - {text}",
)

SECTION_LABELS = {
    "example": ("예시", "사용 예", "예"),
    "caution": ("주의", "조심할 점", "주의할 점"),
    "related": ("관련", "함께 보면 좋은 것", "더 보기"),
    "category": ("분류", "종류", "어떤 것"),
    "inferred": ("추론된 분류", "이어서 추론하면", "더 넓게 보면"),
    "property": ("속성", "특징", "성질"),
}

# 사실을 바꾸지 않는 선에서만 쓰는 연결어 (문장 머리에 붙는다)
CONNECTIVES = (
    "", "", "그리고 ", "참고로 ", "덧붙이면 ", "이와 함께 ",
)


def _pick(rng: random.Random, options: Iterable[str]) -> str:
    options = [o for o in options]
    return rng.choice(options) if options else ""


@dataclass
class Segment:
    """실현할 사실 한 조각.  term 은 충실성 검사에 쓰는 핵심어."""
    kind: str
    text: str
    terms: tuple[str, ...] = ()
    order: int = 100
    group: str = "detail"
    must: bool = False


@dataclass
class FactPool:
    """이번 턴에 고를 수 있는 **참인 사실**들의 묶음.

    모든 항목이 사실이다.  무엇을 넣을지만 샘플링하므로 내용은
    달라져도 거짓이 섞이지 않는다.
    """
    must: list[Segment] = field(default_factory=list)
    details: list[Segment] = field(default_factory=list)
    examples: list[Segment] = field(default_factory=list)
    inferred: list[Segment] = field(default_factory=list)
    properties: list[Segment] = field(default_factory=list)
    related: list[Segment] = field(default_factory=list)
    paths: list[Segment] = field(default_factory=list)

    def all_segments(self) -> list[Segment]:
        return (self.must + self.details + self.examples + self.inferred
                + self.properties + self.related + self.paths)

    def signature(self, selected: list[Segment]) -> tuple[str, ...]:
        """이번 턴에 실제로 들어간 사실들의 지문 (내용 변화 측정용)."""
        return tuple(sorted(segment.text for segment in selected))


@dataclass
class Realizer:
    seed: Optional[int] = None
    history: list[str] = field(default_factory=list)
    history_limit: int = 6
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------
    def _rng_for(self, salt: str) -> random.Random:
        if self.seed is None:
            return self._rng
        return random.Random(f"{self.seed}:{salt}")

    def _label(self, key: str, rng: random.Random) -> str:
        return _pick(rng, SECTION_LABELS.get(key, (key,)))

    def _bullet(self, text: str, rng: random.Random) -> str:
        frame = _pick(rng, BULLET_FRAMES)
        return frame.format(text=text)

    def _variants_for(self, segment: Segment) -> list[str]:
        """한 조각의 가능한 표현들.  첫 번째는 항상 정본."""
        canonical = segment.text
        variants = [canonical]
        if segment.kind in ("definition", "answer"):
            # 정의문: "X는 ~이다" -> "~ — X는 이것입니다" (짧을 때만)
            if "는 " in canonical and canonical.endswith("입니다."):
                head, _, rest = canonical.partition("는 ")
                body = rest[:-len("입니다.")].strip()
                if head and body and len(body) <= 40:
                    variants.append(body + " — " + head.strip() + "는 것입니다.")
        if segment.kind in ("detail", "bullet"):
            stripped = canonical.lstrip("-•· ").strip()
            variants.append(stripped)
            variants.append(stripped.rstrip(".") + ".")
        return [v for v in variants if v.strip()]

    # ------------------------------------------------------------------
    # 실현
    # ------------------------------------------------------------------
    def realize(self, segments: list[Segment], salt: str = "",
                opener: bool = True, closer: bool = False,
                key_terms: Iterable[str] = ()) -> str:
        rng = self._rng_for(salt or "|".join(s.text for s in segments))
        ordered = sorted(segments, key=lambda s: (s.order, ))
        # 어순 변형: 정의 뒤의 동급 항목들은 순서를 섞어도 뜻이 유지된다
        head = [s for s in ordered if s.kind in ("definition", "answer")]
        tail = [s for s in ordered if s.kind not in ("definition", "answer")]
        same_rank: dict[int, list[Segment]] = {}
        for seg in tail:
            same_rank.setdefault(seg.order, []).append(seg)
        shuffled_tail: list[Segment] = []
        for rank in sorted(same_rank):
            bucket = list(same_rank[rank])
            if len(bucket) > 1:
                rng.shuffle(bucket)
            shuffled_tail.extend(bucket)
        ordered = head + shuffled_tail

        lines: list[str] = []
        if opener:
            prefix = _pick(rng, OPENERS)
            if prefix:
                lines.append(prefix.rstrip())
        for index, segment in enumerate(ordered):
            variants = self._variants_for(segment)
            # 최근 표현 회피
            fresh = [v for v in variants if v not in self.history]
            choice = _pick(rng, fresh or variants)
            if segment.kind in ("detail", "bullet"):
                choice = choice if choice.lstrip().startswith(("-", "•", "·")) \
                    else self._bullet(choice, rng)
            elif index > 0 and segment.kind not in ("section",):
                connective = _pick(rng, CONNECTIVES)
                if connective and not choice.startswith(connective.strip()):
                    choice = connective + choice
            lines.append(choice)

        text = "\n".join(line for line in lines if line != "")

        # 충실성 검사: 핵심어가 사라졌으면 정본으로 되돌린다
        required = list(key_terms) or [t for s in ordered for t in s.terms]
        if required and not all(term in text for term in required):
            text = self._canonical(ordered)

        if closer:
            closing = _pick(rng, CLOSERS)
            if closing:
                text = text.rstrip() + "\n\n" + closing

        self._remember(text)
        return text

    def _canonical(self, segments: list[Segment]) -> str:
        return "\n".join(s.text for s in sorted(segments, key=lambda s: s.order))

    def _remember(self, text: str) -> None:
        self.history.append(text)
        if len(self.history) > self.history_limit:
            del self.history[0]

    # ------------------------------------------------------------------
    # 내용 선택 (참인 사실 풀에서 샘플링)
    # ------------------------------------------------------------------
    def select(self, pool: FactPool, salt: str = "",
               detail_range: tuple[int, int] = (1, 3),
               example_range: tuple[int, int] = (0, 1),
               inferred_range: tuple[int, int] = (0, 3),
               property_range: tuple[int, int] = (0, 2),
               related_range: tuple[int, int] = (0, 1),
               path_range: tuple[int, int] = (0, 1),
               ) -> list[Segment]:
        """사실 풀에서 이번 턴에 보여줄 부분집합을 고른다.

        필수 항목(정의)은 항상 들어가고, 나머지는 범위 안에서 무작위로
        고른다.  고른 것은 전부 사실이다.
        """
        rng = self._rng_for("select:" + salt)
        selected: list[Segment] = list(pool.must)

        def take(items: list[Segment], bounds: tuple[int, int]
                 ) -> None:
            low, high = bounds
            if not items or high <= 0:
                return
            low = max(0, min(low, len(items)))
            high = max(low, min(high, len(items)))
            count = rng.randint(low, high) if high > low else low
            if count:
                selected.extend(rng.sample(items, count))

        take(pool.details, detail_range)
        take(pool.examples, example_range)
        take(pool.inferred, inferred_range)
        take(pool.properties, property_range)
        take(pool.related, related_range)
        take(pool.paths, path_range)
        return selected

    def realize_pool(self, pool: FactPool, salt: str = "",
                     opener: bool = True, closer: bool = False,
                     key_terms: Iterable[str] = (), **bounds) -> str:
        """사실 풀 -> 이번 턴의 문장 (내용·표현 모두 가변)."""
        selected = self.select(pool, salt=salt, **bounds)
        terms = list(key_terms) or [t for s in pool.must for t in s.terms]
        return self.realize(selected, salt=salt, opener=opener,
                            closer=closer, key_terms=terms)

    # ------------------------------------------------------------------
    # 같은 뜻, 다른 표현
    # ------------------------------------------------------------------
    def rewrite(self, text: str, salt: str = "") -> str:
        """문장을 사실 그대로 두고 표현만 바꾼다."""
        from knowledge import offline_kb
        rng = self._rng_for(salt or text)
        lines = str(text or "").split("\n")
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            candidate = line
            # 용어 동의어 치환 (문장 뜻이 바뀌지 않는 범위에서만)
            if rng.random() < 0.35 and len(stripped) > 6:
                swapped = offline_kb.translate_terms(candidate)
                if swapped != candidate:
                    candidate = swapped
            # 불릿 기호 변형
            if stripped.startswith(("-", "•", "·")):
                body = stripped.lstrip("-•· ").strip()
                candidate = _pick(rng, BULLET_FRAMES).format(text=body)
            # 접속 부사 삽입
            if rng.random() < 0.25 and not candidate.startswith(" "):
                connective = _pick(rng, CONNECTIVES)
                if connective and not candidate.startswith(connective.strip()):
                    candidate = connective + candidate
            out.append(candidate)
        return "\n".join(out)


__all__ = ["Segment", "Realizer"]
