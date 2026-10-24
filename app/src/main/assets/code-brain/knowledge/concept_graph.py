"""개념 그래프 + 추론 — 오프라인 지식을 관계로 승격한다.

`knowledge/offline_kb.py` 가 각 개념을 '설명문'으로 갖는다면, 여기서는
개념 사이의 **관계**(is_a, part_of, used_for, property ...)를 그래프로
갖고 **추론**한다.  손으로 쓴 사실에서 새 사실을 도출한다::

    우분투 is_a 배포판, 배포판 is_a 운영체제
      => 우분투 is_a 운영체제            (이행 추론)
    리눅스 property 오픈소스, 우분투 is_a 리눅스
      => 우분투 property 오픈소스        (속성 상속)

추론 결과에는 **근거 경로(provenance)** 가 함께 남는다.  그래서 답을
"그렇다"가 아니라 "왜 그런지"까지 말할 수 있다.

공개 API::

    graph = ConceptGraph()
    graph.ancestors("우분투")            -> [("운영체제", ["우분투","배포판","운영체제"]), ...]
    graph.properties("우분투")           -> [("오픈소스", [경로]), ...]
    graph.path("우분투", "운영체제")      -> ["우분투", "배포판", "운영체제"]
    graph.derive()                       -> 추론된 사실 목록 (증거 포함)
    graph.explain("우분투")              -> 한국어 설명 (분류·속성·관계)
    answer(text)                         -> 단일/관계 질문에 답하거나 None
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str


# ---------------------------------------------------------------------------
# 손으로 검수한 관계 (여기서 추론이 자라난다)
# ---------------------------------------------------------------------------

_IS_A: tuple[tuple[str, str], ...] = (
    # 셸 명령 -> 소프트웨어 도구
    ("pwd", "셸 명령"), ("ls", "셸 명령"), ("cd", "셸 명령"),
    ("cp", "셸 명령"), ("mv", "셸 명령"), ("rm", "셸 명령"),
    ("mkdir", "셸 명령"), ("cat", "셸 명령"), ("head", "셸 명령"),
    ("tail", "셸 명령"), ("grep", "셸 명령"), ("find", "셸 명령"),
    ("chmod", "셸 명령"), ("ps", "셸 명령"), ("kill", "셸 명령"),
    ("df", "셸 명령"), ("du", "셸 명령"), ("tar", "셸 명령"),
    ("ssh", "셸 명령"), ("curl", "셸 명령"), ("awk", "셸 명령"),
    ("sed", "셸 명령"), ("sort", "셸 명령"), ("uniq", "셸 명령"),
    ("wc", "셸 명령"), ("less", "셸 명령"), ("man", "셸 명령"),
    ("echo", "셸 명령"), ("export", "셸 명령"), ("env", "셸 명령"),
    ("alias", "셸 명령"),
    ("셸 명령", "소프트웨어 도구"),
    ("셸", "소프트웨어 도구"),
    ("터미널", "소프트웨어 도구"),

    # 깃
    ("깃", "버전 관리 시스템"),
    ("버전 관리 시스템", "소프트웨어 도구"),
    ("git status", "깃 명령"), ("git add", "깃 명령"),
    ("git commit", "깃 명령"), ("git push", "깃 명령"),
    ("git pull", "깃 명령"), ("git log", "깃 명령"),
    ("git diff", "깃 명령"),
    ("깃 명령", "깃"),
    ("깃 저장소", "깃"),
    ("병합", "버전 관리 작업"),
    ("브랜치", "버전 관리 개념"),
    ("커밋", "버전 관리 작업"),
    ("버전 관리 작업", "깃"),
    ("버전 관리 개념", "깃"),

    # 언어/런타임
    ("파이썬", "프로그래밍 언어"),
    ("자바스크립트", "프로그래밍 언어"),
    ("프로그래밍 언어", "소프트웨어 도구"),
    ("인터프리터", "언어 처리 도구"),
    ("컴파일러", "언어 처리 도구"),
    ("언어 처리 도구", "소프트웨어 도구"),
    ("pip", "패키지 관리 도구"),
    ("패키지 관리 도구", "소프트웨어 도구"),
    ("가상환경", "개발 도구"),
    ("개발 도구", "소프트웨어 도구"),

    # 운영체제
    ("리눅스", "운영체제"),
    ("운영체제", "시스템 소프트웨어"),
    ("시스템 소프트웨어", "소프트웨어"),
    ("소프트웨어 도구", "소프트웨어"),
    ("우분투", "배포판"),
    ("우분투", "리눅스"),
    ("배포판", "리눅스"),
    ("커널", "시스템 소프트웨어"),
    ("파일시스템", "운영체제 구성요소"),
    ("프로세스", "운영체제 구성요소"),
    ("마운트", "운영체제 구성요소"),
    ("운영체제 구성요소", "운영체제"),
    ("권한", "파일시스템"),

    # 셸 기능
    ("파이프", "셸 기능"),
    ("리다이렉션", "셸 기능"),
    ("셸 기능", "셸"),
    ("환경변수", "셸"),

    # 웹/데이터/인프라
    ("http", "네트워크 프로토콜"),
    ("네트워크 프로토콜", "프로토콜"),
    ("api", "인터페이스"),
    ("json", "데이터 형식"),
    ("yaml", "데이터 형식"),
    ("데이터 형식", "데이터 표현"),
    ("도커", "컨테이너 도구"),
    ("컨테이너 도구", "인프라 도구"),
    ("인프라 도구", "소프트웨어 도구"),

    # 컴퓨터과학
    ("알고리즘", "컴퓨터과학 개념"),
    ("자료구조", "컴퓨터과학 개념"),
    ("빅오", "컴퓨터과학 개념"),
    ("해시", "컴퓨터과학 개념"),
    ("정규표현식", "컴퓨터과학 개념"),
    ("캐시", "컴퓨터과학 개념"),

    # 셸 확장
    ("nano", "셸 명령"), ("vim", "셸 명령"), ("ln", "셸 명령"),
    ("whoami", "셸 명령"), ("systemctl", "셸 명령"),
    ("apt", "패키지 관리 도구"), ("top", "셸 명령"),
    ("메모리", "운영체제 구성요소"), ("cpu", "운영체제 구성요소"),
    ("스레드", "운영체제 구성요소"), ("로그", "운영체제 구성요소"),
    ("스레드", "프로세스 구성요소"),

    # 깃 확장
    ("git switch", "깃 명령"), ("git restore", "깃 명령"),
    ("git stash", "깃 명령"),
    ("스태시", "버전 관리 작업"),
    ("체크아웃", "버전 관리 작업"),

    # 네트워크
    ("dns", "네트워크 프로토콜"), ("tcp", "네트워크 프로토콜"),
)

_PART_OF: tuple[tuple[str, str], ...] = (
    ("커밋", "깃 저장소"),
    ("브랜치", "깃 저장소"),
    ("커널", "운영체제"),
    ("프로세스", "운영체제"),
    ("파일시스템", "운영체제"),
    ("권한", "파일시스템"),
    ("환경변수", "셸"),
    ("셸", "터미널"),
)

_USED_FOR: tuple[tuple[str, str], ...] = (
    ("grep", "텍스트 검색"),
    ("find", "파일 검색"),
    ("tar", "파일 묶기"),
    ("curl", "HTTP 요청"),
    ("sort", "정렬"),
    ("uniq", "중복 제거"),
    ("df", "디스크 용량 확인"),
    ("du", "용량 계산"),
    ("ps", "프로세스 확인"),
    ("kill", "프로세스 종료"),
    ("chmod", "권한 변경"),
    ("json", "데이터 교환"),
    ("pip", "패키지 설치"),
    ("ssh", "원격 접속"),
    ("nano", "텍스트 편집"),
    ("vim", "텍스트 편집"),
    ("ln", "링크 생성"),
    ("whoami", "사용자 확인"),
    ("systemctl", "서비스 관리"),
    ("apt", "패키지 관리"),
    ("top", "자원 모니터링"),
    ("캐시", "성능 개선"),
    ("메모리", "데이터 보관"),
    ("해시", "무결성 확인"),
    ("정규표현식", "문자열 패턴 검사"),
    ("스레드", "동시 실행"),
    ("dns", "이름 해석"),
    ("tcp", "신뢰성 있는 전달"),
    ("git switch", "브랜치 이동"),
    ("git restore", "변경 되돌리기"),
    ("git stash", "변경 임시 보관"),
    ("로그", "문제 원인 추적"),
)

_REQUIRES: tuple[tuple[str, str], ...] = (
    ("git push", "원격 저장소"),
    ("git pull", "원격 저장소"),
    ("ssh", "네트워크 연결"),
    ("curl", "네트워크 연결"),
    ("도커", "커널"),
    ("가상환경", "파이썬"),
)

_PROPERTIES: tuple[tuple[str, str], ...] = (
    ("리눅스", "오픈소스"),
    ("리눅스", "멀티유저"),
    ("깃", "분산형"),
    ("깃", "변경 이력 추적"),
    ("파이썬", "동적 타입"),
    ("파이썬", "읽기 쉬운 문법"),
    ("자바스크립트", "동적 타입"),
    ("자바스크립트", "이벤트 기반"),
    ("셸", "텍스트 기반"),
)

# 별칭: 표기 차이를 같은 개념으로 묶는다 (추론이 끊기지 않게)
_ALIASES: dict[str, str] = {
    "linux": "리눅스", "ubuntu": "우분투", "git": "깃",
    "python": "파이썬", "javascript": "자바스크립트", "js": "자바스크립트",
    "json": "json", "yaml": "yaml", "api": "api", "http": "http",
    "docker": "도커", "venv": "가상환경", "virtualenv": "가상환경",
    "commit": "커밋", "branch": "브랜치", "merge": "병합",
    "shell": "셸", "bash": "셸", "zsh": "셸", "쉘": "셸",
    "terminal": "터미널", "console": "터미널",
    "filesystem": "파일시스템", "process": "프로세스",
    "kernel": "커널", "permission": "권한", "mount": "마운트",
    "pipe": "파이프", "redirection": "리다이렉션",
    "environment variable": "환경변수",
    "repository": "깃 저장소", "repo": "깃 저장소",
    "algorithm": "알고리즘", "data structure": "자료구조",
    "big-o": "빅오", "compiler": "컴파일러",
    "interpreter": "인터프리터",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class ConceptGraph:
    """is_a / part_of / used_for / requires / property 관계와 추론."""

    def __init__(self) -> None:
        self.edges: list[Edge] = []
        self._is_a: dict[str, set[str]] = {}
        self._part_of: dict[str, set[str]] = {}
        self._used_for: dict[str, set[str]] = {}
        self._requires: dict[str, set[str]] = {}
        self._properties: dict[str, set[str]] = {}
        self._aliases: dict[str, str] = dict(_ALIASES)
        for source, target in _IS_A:
            self.add(source, "is_a", target)
        for source, target in _PART_OF:
            self.add(source, "part_of", target)
        for source, target in _USED_FOR:
            self.add(source, "used_for", target)
        for source, target in _REQUIRES:
            self.add(source, "requires", target)
        for source, target in _PROPERTIES:
            self.add(source, "property", target)

    # ------------------------------------------------------------------
    # 그래프 편집
    # ------------------------------------------------------------------
    def canonical(self, term: str) -> str:
        """별칭을 정식 이름으로 바꾼다."""
        value = _norm(term).lower()
        return self._aliases.get(value, _norm(term))

    def add(self, source: str, relation: str, target: str) -> None:
        source, target = _norm(source), _norm(target)
        if not source or not target:
            return
        relation = str(relation)
        edge = Edge(source, relation, target)
        if edge in self.edges:
            return
        self.edges.append(edge)
        table = {
            "is_a": self._is_a, "part_of": self._part_of,
            "used_for": self._used_for, "requires": self._requires,
            "property": self._properties,
        }.get(relation)
        if table is not None:
            table.setdefault(source, set()).add(target)

    def add_alias(self, alias: str, canonical: str) -> None:
        self._aliases[_norm(alias).lower()] = _norm(canonical)

    def nodes(self) -> list[str]:
        out: set[str] = set()
        for edge in self.edges:
            out.add(edge.source)
            out.add(edge.target)
        return sorted(out)

    def has(self, term: str) -> bool:
        name = self.canonical(term)
        return name in self._is_a or name in self._part_of or \
            name in self._used_for or name in self._requires or \
            name in self._properties or any(
                edge.target == name for edge in self.edges)

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    def ancestors(self, term: str, relation: str = "is_a"
                  ) -> list[tuple[str, list[str]]]:
        """이행 폐포. (상위개념, 근거경로) 목록. 사이클 안전."""
        name = self.canonical(term)
        table = {"is_a": self._is_a, "part_of": self._part_of,
                 "used_for": self._used_for, "requires": self._requires,
                 "property": self._properties}.get(relation, {})
        found: dict[str, list[str]] = {}
        queue: deque[list[str]] = deque([[name]])
        seen_paths: set[tuple[str, ...]] = set()
        while queue:
            path = queue.popleft()
            key = tuple(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            current = path[-1]
            for parent in sorted(table.get(current, ())):
                if parent in path:      # 사이클 방지
                    continue
                new_path = path + [parent]
                if parent not in found or \
                        len(new_path) < len(found[parent]):
                    found[parent] = new_path
                queue.append(new_path)
        return sorted(found.items(), key=lambda item: (len(item[1]), item[0]))

    def properties(self, term: str) -> list[tuple[str, list[str]]]:
        """자신 + is_a 상위에서 상속한 속성 (근거경로 포함)."""
        name = self.canonical(term)
        result: dict[str, list[str]] = {
            prop: [name, prop] for prop in self._properties.get(name, ())}
        for ancestor, path in self.ancestors(name):
            for prop in self._properties.get(ancestor, ()):
                if prop not in result or len(path) < len(result[prop]):
                    result[prop] = path[:-1] + [ancestor, prop]
        return sorted(result.items(), key=lambda item: item[1])

    def path(self, left: str, right: str) -> Optional[list[str]]:
        """두 개념을 잇는 최단 관계 경로 (모든 관계, 양방향)."""
        start, goal = self.canonical(left), self.canonical(right)
        if start == goal:
            return [start]
        adjacency: dict[str, set[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set()).add(edge.source)
        queue: deque[list[str]] = deque([[start]])
        seen = {start}
        while queue:
            current = queue.popleft()
            for nxt in sorted(adjacency.get(current[-1], ())):
                if nxt in seen:
                    continue
                new_path = current + [nxt]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                queue.append(new_path)
        return None

    def children(self, term: str, relation: str = "is_a") -> list[str]:
        """직접 하위 개념 (종류)."""
        name = self.canonical(term)
        table = {"is_a": self._is_a, "part_of": self._part_of}.get(
            relation, self._is_a)
        return sorted(child for child, parents in table.items()
                      if name in parents)

    def capabilities(self, term: str) -> list[tuple[str, list[str]]]:
        """X로 무엇을 할 수 있나 — own + is_a 상위에서 물려받은 용도."""
        name = self.canonical(term)
        result: dict[str, list[str]] = {
            use: [name, use] for use in self._used_for.get(name, ())}
        for ancestor, path in self.ancestors(name):
            for use in self._used_for.get(ancestor, ()):
                if use not in result or len(path) < len(result[use]):
                    result[use] = path[:-1] + [ancestor, use]
        return sorted(result.items(), key=lambda item: item[1])

    def needs(self, term: str) -> list[tuple[str, list[str]]]:
        """X가 동작하려면 무엇이 필요한가 (상속 포함)."""
        name = self.canonical(term)
        result: dict[str, list[str]] = {
            need: [name, need] for need in self._requires.get(name, ())}
        for ancestor, path in self.ancestors(name):
            for need in self._requires.get(ancestor, ()):
                if need not in result or len(path) < len(result[need]):
                    result[need] = path[:-1] + [ancestor, need]
        return sorted(result.items(), key=lambda item: item[1])

    def is_a(self, left: str, right: str) -> Optional[list[str]]:
        """'X는 Y인가?' 에 근거 경로와 함께 답한다 (없으면 None)."""
        start, goal = self.canonical(left), self.canonical(right)
        if start == goal:
            return [start]
        for ancestor, path in self.ancestors(start):
            if ancestor == goal:
                return path
        return None

    def common_ancestors(self, left: str, right: str
                         ) -> list[tuple[str, list[str], list[str]]]:
        """두 개념의 공통 상위 (각각의 근거 경로 포함)."""
        a = dict(self.ancestors(left))
        b = dict(self.ancestors(right))
        shared = []
        for name in sorted(set(a) & set(b)):
            shared.append((name, a[name], b[name]))
        shared.sort(key=lambda item: len(item[1]) + len(item[2]))
        return shared

    def relations(self, term: str) -> dict[str, list[str]]:
        """한 개념의 직접 관계들을 관계 이름별로 모은다."""
        name = self.canonical(term)
        out: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.source == name:
                out.setdefault(edge.relation, []).append(edge.target)
        return {key: sorted(values) for key, values in sorted(out.items())}

    def derive(self) -> list[dict]:
        """손으로 쓴 사실에서 도출한 새 사실들 (증거 경로 포함)."""
        facts: list[dict] = []
        for term in self._is_a:
            for ancestor, path in self.ancestors(term):
                if len(path) < 3:
                    continue  # 직접 is_a 는 새 사실이 아니다
                facts.append({
                    "kind": "is_a",
                    "subject": term,
                    "object": ancestor,
                    "evidence": path,
                    "statement": f"{with_josa(term, '은/는')} {ancestor}의 일종이다",
                })
        for term in set(self._is_a) | set(self._properties):
            for prop, path in self.properties(term):
                if len(path) < 3:
                    continue  # 직접 속성은 새 사실이 아니다
                facts.append({
                    "kind": "property",
                    "subject": term,
                    "object": prop,
                    "evidence": path,
                    "statement": f"{with_josa(term, '은/는')} {prop} 특성을 가진다",
                })
        # 규칙 이행: 상위의 용도/필요를 하위가 물려받는다
        for term in set(self._is_a) | set(self._used_for) | set(self._requires):
            for use, path in self.capabilities(term):
                if len(path) < 3:
                    continue
                facts.append({
                    "kind": "used_for",
                    "subject": term,
                    "object": use,
                    "evidence": path,
                    "statement": f"{with_josa(term, '은/는')} {use}에 쓸 수 있다",
                })
            for need, path in self.needs(term):
                if len(path) < 3:
                    continue
                facts.append({
                    "kind": "requires",
                    "subject": term,
                    "object": need,
                    "evidence": path,
                    "statement": f"{with_josa(term, '은/는')} {need}{particle_pair(need, '이/가')} 필요하다",
                })
        return facts

    # ------------------------------------------------------------------
    # 한국어 답변
    # ------------------------------------------------------------------
    def explain(self, term: str) -> Optional[str]:
        """한 개념을 분류·속성·관계까지 엮어 설명한다."""
        name = self.canonical(term)
        if not self.has(name):
            return None
        lines: list[str] = []
        try:
            from knowledge import offline_kb
            entry = offline_kb.lookup(name)
        except Exception:
            entry = None
        if entry is not None:
            lines.append(entry.summary)
            for detail in entry.details[:2]:
                lines.append(f"- {detail}")

        direct = [a for a, path in self.ancestors(name) if len(path) == 2]
        inferred = [a for a, path in self.ancestors(name) if len(path) > 2]
        if direct:
            lines.append("분류: " + ", ".join(direct))
        if inferred:
            lines.append("추론된 분류: " + ", ".join(inferred))
        kids = self.children(name)
        if kids:
            lines.append("종류: " + ", ".join(kids[:6]))
        props = self.properties(name)
        if props:
            lines.append("속성: " + ", ".join(p for p, _ in props))
        uses = self.capabilities(name)
        if uses:
            lines.append("용도: " + ", ".join(u for u, _ in uses))
        needs = self.needs(name)
        if needs:
            lines.append("필요: " + ", ".join(n for n, _ in needs))
        relations = self.relations(name)
        for relation, label in (("part_of", "속함"),):
            values = relations.get(relation)
            if values:
                lines.append(f"{label}: " + ", ".join(values))
        if entry is None and not lines:
            return None
        return "\n".join(lines)

    def relation_answer(self, left: str, right: str) -> Optional[str]:
        """두 개념의 관계를 경로로 설명한다."""
        path = self.path(left, right)
        if not path or len(path) < 2:
            return None
        start, goal = self.canonical(left), self.canonical(right)
        if start == goal:
            return (f"{with_josa(left, '과/와')} {right}는 같은 "
                    "개념이에요.")
        chain = " → ".join(path)
        lines = [f"{with_josa(left, '과/와')} {right}의 관계: {chain}"]
        if goal in [a for a, _ in self.ancestors(start)]:
            lines.append(f"즉 {with_josa(left, '은/는')} {right}에 속해요.")
        elif start in [a for a, _ in self.ancestors(goal)]:
            lines.append(f"즉 {with_josa(right, '이/가')} {left}에 속해요.")
        else:
            lines.append("서로 직접 포함 관계는 아니고, 같은 그래프에서 "
                         "이어져 있어요.")
        return "\n".join(lines)


_GRAPH: Optional[ConceptGraph] = None


def graph() -> ConceptGraph:
    """공용 그래프 (한 번만 만든다).

    문서에서 추출한 학습 지식이 있으면 함께 싣는다 (없으면 그대로).
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ConceptGraph()
        try:
            from knowledge import extract
            extract.apply_to_graph(_GRAPH)
        except Exception:
            pass
    return _GRAPH


def reload_graph() -> ConceptGraph:
    """학습 지식을 다시 싣는다 (문서를 새로 넣은 뒤)."""
    global _GRAPH
    _GRAPH = None
    return graph()


def _terms_in(text: str, limit: int = 3) -> list[str]:
    """문장에서 알려진 개념 이름을 **나온 순서대로** 찾는다.

    순서가 중요하다: "운영체제는 리눅스인가?" 의 주어는 먼저 나온 말이다.
    긴 이름을 우선해 부분 겹침(운영체제 안의 체제 등)을 피한다.
    """
    g = graph()
    lowered = _norm(text).lower()
    hits: list[tuple[int, str]] = []
    for node in sorted(g.nodes(), key=len, reverse=True):
        position = lowered.find(node)
        if position >= 0 and all(node != existing for _, existing in hits):
            hits.append((position, node))
    if not hits:
        for alias, canonical in g._aliases.items():
            position = lowered.find(alias)
            if position >= 0 and all(canonical != existing
                                     for _, existing in hits):
                hits.append((position, canonical))
    # 긴 이름에 포함되는 짧은 이름은 버린다 ("셸 명령" 안의 "셸")
    kept: list[tuple[int, str]] = []
    for position, name in sorted(hits, key=lambda item:
                                 (-len(item[1]), item[0])):
        if any(name in longer and name != longer for _, longer in kept):
            continue
        kept.append((position, name))
    kept.sort(key=lambda item: item[0])
    return [name for _, name in kept[:limit]]


def _has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 (조사 선택용)."""
    if not word:
        return False
    last = word[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    return True


_JOSA = {
    "은/는": ("은", "는"),
    "이/가": ("이", "가"),
    "을/를": ("을", "를"),
    "과/와": ("과", "와"),
    "으로/로": ("으로", "로"),
}


def with_josa(word: str, pair: str) -> str:
    """조사 쌍('받침있을때/받침없을때')을 받침에 맞게 붙인다."""
    final, plain = _JOSA.get(pair, pair.split("/"))
    return word + (final if _has_batchim(word) else plain)


def particle_pair(word: str, pair: str) -> str:
    final, plain = _JOSA.get(pair, pair.split("/"))
    return final if _has_batchim(word) else plain
    """조사만 돌려준다 ('은' 또는 '는')."""
    without, with_final = pair.split("/")
    return with_final if _has_batchim(word) else without


def first_term(text: str) -> Optional[str]:
    """문장에서 찾은 첫 개념 이름 (되묻기/후속 질문 처리용)."""
    terms = _terms_in(text, limit=1)
    return terms[0] if terms else None


def answer(text: str) -> Optional[str]:
    """관계·분류·용도·예/아니오 질문에 그래프로 답한다 (없으면 None)."""
    g = graph()
    terms = _terms_in(text)
    if not terms:
        return None
    lowered = _norm(text).lower()

    # 1) 예/아니오: "X는 Y인가?"
    yesno = ("인가요", "인가", "맞아", "맞나요", "야?", "이야?", "니?",
             "입니까", "해?")
    if len(terms) >= 2 and any(m in lowered for m in yesno):
        subject, target = terms[0], terms[1]
        forward = g.is_a(subject, target)
        if forward:
            return (f"네. {' → '.join(forward)} 이므로 "
                    f"{with_josa(subject, '은/는')} {target}에 속해요.")
        backward = g.is_a(target, subject)
        if backward:
            return (f"아니요, 반대예요. {' → '.join(backward)} 이므로 "
                    f"{with_josa(target, '이/가')} {subject}에 속해요.")
        return (f"제 지식으로는 {with_josa(subject, '과/와')} {target} "
                "사이에 포함 관계를 확인할 수 없어요. (근거 부족)")

    # 2) 공통점
    if len(terms) >= 2 and any(m in lowered for m in
                               ("공통", "같은 점", "비슷한 점")):
        shared = g.common_ancestors(terms[0], terms[1])
        if shared:
            name, path_a, path_b = shared[0]
            return (f"{terms[0]}와 {terms[1]}의 공통 분류는 '{name}'이에요.\n"
                    f"- {terms[0]}: {' → '.join(path_a)}\n"
                    f"- {terms[1]}: {' → '.join(path_b)}")

    # 3) 종류
    if any(m in lowered for m in ("종류", "어떤 것들", "뭐가 있", "무엇이 있")):
        kids = g.children(terms[0])
        if kids:
            return f"{terms[0]}의 종류: " + ", ".join(kids)

    # 4) 용도
    if any(m in lowered for m in ("뭘 할 수 있", "무엇을 할 수 있",
                                  "용도", "어디에 쓰", "무슨 일")):
        uses = g.capabilities(terms[0])
        if uses:
            return f"{terms[0]}의 용도: " + ", ".join(u for u, _ in uses)

    # 5) 관계
    relation_markers = ("관계", "차이", "다른", "비교", "연관", "어떻게 연결")
    if len(terms) >= 2 and any(m in lowered for m in relation_markers):
        return g.relation_answer(terms[0], terms[1])
    return g.explain(terms[0])


__all__ = ["Edge", "ConceptGraph", "graph", "answer"]
