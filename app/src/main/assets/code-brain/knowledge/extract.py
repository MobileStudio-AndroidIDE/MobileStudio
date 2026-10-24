"""문서에서 개념과 관계를 추출해 지식 그래프를 키운다.

사람이 항목을 손으로 쓰지 않아도, 새 문서를 넣으면 그 안의
정의·분류·용도·필요 관계를 **근거 문장과 함께** 뽑아낸다.

원칙:
- 추출한 모든 관계에는 **근거 문장과 출처 파일**이 붙는다.
- 신뢰도가 낮으면 그래프에 넣지 않는다 (오염 방지).
- 같은 관계가 여러 문서에서 나오면 지지 횟수로 신뢰도가 올라간다.
- 문서를 넣지 않으면 아무것도 바뀌지 않는다 (런타임은 조용하다).

사용::

    from knowledge import extract
    report = extract.ingest(["docs", "README.md"])
    print(report["relations"], "개 관계 추가")

    # 저장된 학습 지식을 그래프/지식베이스가 자동으로 읽는다
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

DEFAULT_STORE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "learned_knowledge.json")

_STOPWORDS = set("""
이것 그것 저것 우리 저희 여러 여기 거기 경우 사용 있는 없는 한다 있다
대한 위한 통해 같은 다른 그리고 하지만 그러나 또한 그래서 때문에 이런
그런 어떤 무엇 누구 어디 언제 왜 어떻게 그것은 이것은 때는 것을 것이
수 등 및 또 즉 및 등등 위 아래 안 밖 앞 뒤 사이 동안 대해 대해
the a an of to in is are was were be been and or for with on at by from
as it its this that these those we you they he she not no do does did can
could should would will may might must there here what which who whom
when where why how all any both each few more most other some such only
""".split())

# 한국어 용어: 띄어쓰기를 포함할 수 있다 ("큐브 스토어").
_KO = r"[가-힣A-Za-z0-9_+\-.#][가-힣A-Za-z0-9_+\-.# ]{1,23}?"
_EN = r"[A-Za-z0-9][\w.+-]{0,19}"
_EN_PHRASE = r"[A-Za-z0-9][\w\s-]{0,24}?"

# 관계 패턴: (정규식, 관계 이름, 기본 신뢰도)
# 그룹 1 = 주어, 그룹 2 = 목적어.
_PATTERNS: tuple[tuple[str, str, float], ...] = (
    # 한국어: 분류
    (r"(" + _KO + r")\s*(?:은|는)\s*(" + _KO + r")\s*(?:의|의 한)?\s*"
     r"(?:일종|종류|하나)\s*(?:이|입|다|입니다)",
     "is_a", 0.85),
    (r"(" + _KO + r")\s*(?:은|는)\s*(" + _KO + r")\s*(?:에|으로)\s*"
     r"(?:속한다|속합니다|포함된다|포함됩니다)",
     "is_a", 0.8),
    # 한국어: 용도
    (r"(" + _KO + r")\s*(?:은|는)\s*(" + _KO + r")\s*(?:에|를 위해|을 위해)\s*"
     r"(?:쓰인다|쓰입니다|사용된다|사용됩니다|활용된다)",
     "used_for", 0.75),
    # 한국어: 필요
    (r"(" + _KO + r")\s*(?:은|는)\s*(" + _KO + r")\s*(?:이|가)\s*"
     r"(?:필요하다|필요합니다|요구한다)",
     "requires", 0.75),
    # 한국어: 부분
    (r"(" + _KO + r")\s*(?:은|는)\s*(" + _KO + r")\s*(?:의 일부|에 포함|의 구성)",
     "part_of", 0.7),
    # 영어: 분류/부분/용도/필요
    (r"(" + _EN + r")\s+is\s+(?:a|an)\s+(" + _EN_PHRASE + r")"
     r"(?:\.|,|;|\s+that|\s+which|$)",
     "is_a", 0.8),
    (r"(" + _EN + r")\s+is\s+part\s+of\s+(" + _EN_PHRASE + r")(?:\.|,|;|$)",
     "part_of", 0.75),
    (r"(" + _EN + r")\s+is\s+used\s+(?:for|to)\s+(" + _EN_PHRASE + r")"
     r"(?:\.|,|;|$)",
     "used_for", 0.75),
    (r"(" + _EN + r")\s+requires\s+(" + _EN_PHRASE + r")(?:\.|,|;|$)",
     "requires", 0.75),
)

# 정의문: 지식베이스 항목(요약)로 쓸 수 있는 문장.
# 종결어미는 긴 것부터(습니다/입니다) 먼저 맞춘다.
_DEFINITION = re.compile(
    r"(" + _KO + r")\s*(?:은|는)\s*"
    r"([^\n.!?]{4,120}?)\s*(?:습니다|입니다|이다|다)\s*[.!?]?")

# 개념으로 승격하지 말아야 하는 표면들
_BAD_TERM_CHARS = re.compile(r"[`{}#<>|\[\]\"'=*/\\]")
_BAD_TERM_END = ("은", "는", "을", "를", "이", "가", "의", "에", "로", "와",
                 "과", "도", "만", "에서", "부터", "까지", "으로",
                 "하", "되", "했", "있", "없", "었", "았", "이며")
_BAD_TERM_START = ("이들", "그러나", "또한", "그리고", "따라서", "즉",
                   "이것", "그것", "저것", "두 사람", "이 사람", "그 사람",
                   "이러한", "이런", "저런", "모든", "몇몇", "세계에서",
                   "특정", "각각", "어떤")
_BAD_BODY_MARKERS = ("합니", "됩니", "습니", "입니", "합니다", "했습니다",
                     "됩니다", "있습니다", "없습니다", "때문", "경우",
                     "대한", "위한", "통해")
_BAD_BODY_END = ("했", "하", "되", "있", "없", "입", "니", "여", "며",
                 "고", "서", "면", "야", "지", "았", "었", "랐", "왔",
                 "갔", "났", "쳤", "렸", "켰", "봤", "않", "됐", "였",
                 "에", "로", "과", "와", "등", "따라", "중", "한")

_TERM = re.compile(r"[가-힣]{2,10}|[A-Za-z][A-Za-z_+.#\-]{1,19}")

RELATION_KO = {
    "is_a": "의 일종",
    "used_for": "의 용도",
    "requires": "이(가) 필요",
    "part_of": "에 속함",
}


@dataclass
class LearnedRelation:
    source: str
    relation: str
    target: str
    confidence: float
    evidence: str = ""
    origin: str = ""
    support: int = 1

    def key(self) -> tuple[str, str, str]:
        return (self.source, self.relation, self.target)

    def to_dict(self) -> dict:
        return {"source": self.source, "relation": self.relation,
                "target": self.target, "confidence": round(self.confidence, 3),
                "evidence": self.evidence, "origin": self.origin,
                "support": self.support}

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedRelation":
        return cls(source=str(data.get("source", "")),
                   relation=str(data.get("relation", "")),
                   target=str(data.get("target", "")),
                   confidence=float(data.get("confidence", 0.0)),
                   evidence=str(data.get("evidence", "")),
                   origin=str(data.get("origin", "")),
                   support=int(data.get("support", 1)))


@dataclass
class LearnedConcept:
    term: str
    summary: str
    details: list[str] = field(default_factory=list)
    origin: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {"term": self.term, "summary": self.summary,
                "details": list(self.details), "origin": self.origin,
                "confidence": round(self.confidence, 3)}

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedConcept":
        return cls(term=str(data.get("term", "")),
                   summary=str(data.get("summary", "")),
                   details=[str(d) for d in (data.get("details") or [])],
                   origin=str(data.get("origin", "")),
                   confidence=float(data.get("confidence", 0.5)))


# ---------------------------------------------------------------------------
# 추출
# ---------------------------------------------------------------------------

def _with_josa(word: str, pair: str) -> str:
    """받침에 맞는 조사를 붙인다 (pair: '받침있을때/받침없을때')."""
    if not word:
        return word
    last = word[-1]
    has_final = ("가" <= last <= "힣") and (ord(last) - 0xAC00) % 28 != 0
    final, plain = pair.split("/")
    return word + (final if has_final else plain)


def _clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    value = re.sub(r"^(?:그|이|저|해당|각)\s+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;()[]{}'\"")


def _usable(term: str) -> bool:
    term = term.strip()
    if len(term) < 2 or len(term) > 24:
        return False
    if term.lower() in _STOPWORDS:
        return False
    if not re.search(r"[가-힣A-Za-z]", term):
        return False
    if re.fullmatch(r"[\d\s.+-]+", term):
        return False
    return True


def _sentences(text: str) -> Iterator[str]:
    for line in str(text or "").splitlines():
        line = re.sub(r"^[#>\-*\d.\s]+", "", line).strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.!?])\s+", line):
            piece = piece.strip()
            if 6 <= len(piece) <= 300:
                yield piece


def extract(text: str, origin: str = "",
            min_confidence: float = 0.7) -> tuple[list[LearnedConcept],
                                                 list[LearnedRelation]]:
    """문서 하나에서 개념과 관계를 뽑는다 (근거 문장 포함)."""
    relations: dict[tuple[str, str, str], LearnedRelation] = {}
    concepts: dict[str, LearnedConcept] = {}
    term_counts: Counter = Counter()

    for sentence in _sentences(text):
        for token in _TERM.findall(sentence):
            token = token.lower()
            if _usable(token):
                term_counts[token] += 1
        for pattern, relation, base in _PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                source = _clean(match.group(1))
                target = _clean(match.group(2))
                if not (_usable(source) and _usable(target)):
                    continue
                if source.lower() == target.lower():
                    continue
                # 문장 조각이 아니라 용어처럼 보일 때만 관계로 인정한다.
                if (source.endswith(_BAD_TERM_END)
                        or target.endswith(_BAD_TERM_END)
                        or source.startswith(_BAD_TERM_START)
                        or target.startswith(_BAD_TERM_START)
                        or target.startswith(_BAD_TERM_END)
                        or _BAD_TERM_CHARS.search(source)
                        or _BAD_TERM_CHARS.search(target)
                        or len(source.split()) > 2
                        or len(target.split()) > 3):
                    continue
                key = (source, relation, target)
                existing = relations.get(key)
                if existing is None:
                    relations[key] = LearnedRelation(
                        source=source, relation=relation, target=target,
                        confidence=base, evidence=sentence, origin=origin)
                else:
                    existing.support += 1
                    existing.confidence = min(
                        0.95, base + 0.03 * (existing.support - 1))
        for match in re.finditer(_DEFINITION, sentence):
            term = _clean(match.group(1))
            body = _clean(match.group(2))
            if not _usable(term) or len(body) < 4:
                continue
            if (_BAD_TERM_CHARS.search(term)
                    or term.endswith(_BAD_TERM_END)
                    or term.startswith(_BAD_TERM_START)
                    or len(term.split()) > 2):
                continue
            if any(marker in body for marker in _BAD_BODY_MARKERS):
                continue
            if (_BAD_TERM_CHARS.search(body)
                    or body.startswith(_BAD_TERM_END)
                    or body.endswith(_BAD_BODY_END)
                    or len(body.split()) > 8):
                continue
            summary = f"{_with_josa(term, '은/는')} {body}입니다."
            concept = concepts.get(term.lower())
            if concept is None:
                concepts[term.lower()] = LearnedConcept(
                    term=term, summary=summary, details=[], origin=origin,
                    confidence=0.6)
            elif summary not in concept.details:
                concept.details.append(summary)

    kept_relations = [rel for rel in relations.values()
                      if rel.confidence >= min_confidence]
    # 문서에 두 번 이상 나온 용어만 개념으로 승격한다 (우연한 단어 배제).
    # 띄어쓰기 있는 용어는 토큰 세기로 잡히지 않으므로 본문 출현 횟수로 센다.
    lowered_text = str(text).lower()
    kept_concepts = []
    for term, concept in concepts.items():
        occurrences = lowered_text.count(term)
        if occurrences >= 2 or term_counts.get(term, 0) >= 2:
            concept.confidence = min(0.85, 0.6 + 0.05 * (occurrences - 1))
            kept_concepts.append(concept)
    return kept_concepts, kept_relations


def iter_documents(paths: Iterable[str]) -> Iterator[tuple[str, str]]:
    """파일/디렉터리 목록에서 (출처, 본문)을 만들어 낸다."""
    exts = (".md", ".txt", ".rst", ".py")
    for path in paths:
        if os.path.isdir(path):
            for base, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and d not in {"__pycache__", "node_modules"}]
                for name in sorted(files):
                    if name.endswith(exts):
                        full = os.path.join(base, name)
                        try:
                            with open(full, "r", encoding="utf-8",
                                      errors="ignore") as fh:
                                yield full, fh.read()
                        except OSError:
                            continue
        elif os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8",
                          errors="ignore") as fh:
                    yield path, fh.read()
            except OSError:
                continue


# ---------------------------------------------------------------------------
# 저장/병합
# ---------------------------------------------------------------------------

def load_store(path: str = "") -> dict:
    """저장된 학습 지식을 읽는다.  없으면 빈 구조."""
    target = path or DEFAULT_STORE
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data.setdefault("concepts", [])
            data.setdefault("relations", [])
            return data
    except Exception:
        pass
    return {"version": 1, "concepts": [], "relations": []}


def save_store(data: dict, path: str = "") -> str:
    target = path or DEFAULT_STORE
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return target


def ingest(paths: Iterable[str], store_path: str = "",
           min_confidence: float = 0.7) -> dict:
    """문서들을 읽어 학습 지식 저장소에 병합한다 (멱등)."""
    store = load_store(store_path)
    concepts = {str(c.get("term", "")).lower(): c
                for c in store.get("concepts", [])}
    relations = {(str(r.get("source", "")), str(r.get("relation", "")),
                  str(r.get("target", ""))): r
                 for r in store.get("relations", [])}

    added_concepts = added_relations = 0
    files = 0
    for origin, text in iter_documents(paths):
        files += 1
        new_concepts, new_relations = extract(
            text, origin=origin, min_confidence=min_confidence)
        for concept in new_concepts:
            key = concept.term.lower()
            if key in concepts:
                existing = concepts[key]
                for detail in concept.details:
                    if detail not in existing["details"]:
                        existing["details"].append(detail)
            else:
                concepts[key] = concept.to_dict()
                added_concepts += 1
        for relation in new_relations:
            key = relation.key()
            if key in relations:
                existing = relations[key]
                existing["support"] = int(existing.get("support", 1)) + 1
                existing["confidence"] = min(
                    0.95, max(float(existing.get("confidence", 0.0)),
                              relation.confidence) + 0.02)
                if not existing.get("evidence"):
                    existing["evidence"] = relation.evidence
            else:
                relations[key] = relation.to_dict()
                added_relations += 1

    store["concepts"] = list(concepts.values())
    store["relations"] = list(relations.values())
    store["version"] = 1
    saved = save_store(store, store_path)
    return {"files": files, "concepts": added_concepts,
            "relations": added_relations,
            "total_concepts": len(store["concepts"]),
            "total_relations": len(store["relations"]),
            "store": saved}


def apply_to_graph(graph, store_path: str = "",
                   min_confidence: float = 0.85) -> int:
    """학습된 관계를 개념 그래프에 반영한다 (신뢰도 문턱 이상만)."""
    store = load_store(store_path)
    added = 0
    for row in store.get("relations", []):
        relation = LearnedRelation.from_dict(row)
        if relation.confidence < min_confidence:
            continue
        if not (_usable(relation.source) and _usable(relation.target)):
            continue
        graph.add(relation.source, relation.relation, relation.target)
        added += 1
    return added


def ingest_wiki(dump_path: str, store_path: str = "", max_pages: int = 3000,
                min_confidence: float = 0.8) -> dict:
    """한국어 위키 덤프에서 앞부분 N개 문서를 읽어 지식을 뽑는다.

    위키 본문은 정의·분류 문장이 많아 추출 품질이 좋다 (자체 기술문서보다).
    """
    try:
        from knowledge.lexicon import iter_wiki_texts
    except Exception:
        return {"files": 0, "concepts": 0, "relations": 0,
                "error": "lexicon helper unavailable"}
    store = load_store(store_path)
    concepts = {str(c.get("term", "")).lower(): c
                for c in store.get("concepts", [])}
    relations = {(str(r.get("source", "")), str(r.get("relation", "")),
                  str(r.get("target", ""))): r
                 for r in store.get("relations", [])}

    pages = added_concepts = added_relations = 0
    for text in iter_wiki_texts(dump_path, max_pages=max_pages):
        pages += 1
        new_concepts, new_relations = extract(
            text, origin=f"{os.path.basename(dump_path)}#{pages}",
            min_confidence=min_confidence)
        for concept in new_concepts:
            key = concept.term.lower()
            if key not in concepts:
                concepts[key] = concept.to_dict()
                added_concepts += 1
        for relation in new_relations:
            key = relation.key()
            if key not in relations:
                relations[key] = relation.to_dict()
                added_relations += 1

    store["concepts"] = list(concepts.values())
    store["relations"] = list(relations.values())
    store["version"] = 1
    saved = save_store(store, store_path)
    return {"pages": pages, "concepts": added_concepts,
            "relations": added_relations,
            "total_concepts": len(store["concepts"]),
            "total_relations": len(store["relations"]), "store": saved}


def learned_entries(store_path: str = "") -> list[LearnedConcept]:
    store = load_store(store_path)
    return [LearnedConcept.from_dict(c) for c in store.get("concepts", [])]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="문서에서 개념·관계 추출")
    parser.add_argument("paths", nargs="+", help="문서 파일/디렉터리")
    parser.add_argument("--store", default="", help="저장 경로")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args(argv)
    report = ingest(args.paths, store_path=args.store,
                    min_confidence=args.min_confidence)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    store = load_store(args.store)
    print("\n예시 관계:")
    for row in store.get("relations", [])[:10]:
        print(f"  {row['source']} --{row['relation']}--> {row['target']} "
              f"({row['confidence']:.2f}) : {row['evidence'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
