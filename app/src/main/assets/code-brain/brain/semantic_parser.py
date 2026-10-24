"""Semantic Parser - Converts natural language to SemanticRepresentation.

This module implements the core NLU pipeline:
Lexical Preprocessing → Concept Activation → Relation Traversal → Meaning Assembly
"""
from __future__ import annotations
import re
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from brain.semantic_representation import (
    SemanticRepresentation, EntityRef, OperationSpec, OperationGraph,
    IOSpec, ConstraintSpec, GoalSpec, GoalType, OperationType,
    Language, ActivatedConcept, RelationPath, EntityRef,
    OperationSpec, OperationGraph, IOSpec, ConstraintSpec, GoalSpec,
    ObjectiveSpec, SubstructureSpec, ParameterSpec,
)
from brain.concept_store import ConceptStore, Concept
from brain.relation_store import RelationStore, Relation, RelationType
from brain.dialogue_act import (
    analyze_act, derive_intent, allows_artifact_creation, analyze_focus,
    asks_for_content,
)
from brain.target_detector import detect_target


# The modification performative — the sentence's action-verb class
# (grammatical evidence).  ONE structural question ("is this action a
# modification?") is asked in two places — the goal inference's
# existing-artifact evidence and the discourse's anaphoric-modification
# correction — so it is answered by a single shared pattern.
MODIFICATION_ACTION_PATTERN = (
    r"(추가해|추가|삽입해|삽입|업데이트|수정해|수정|"
    r"\badd\b|\binsert\b|\bupdate\b|\bmodify\b|\bextend\b)")


@dataclass
class ParseContext:
    """Context for semantic parsing."""
    conversation_history: list[dict] = field(default_factory=list)
    current_project: dict | None = None
    user_preferences: dict = field(default_factory=dict)


class SemanticParser:
    """Parses natural language into SemanticRepresentation.
    
    ARCHITECTURE NOTE (Phase 8):
    The following keyword lists (OPERATION_PATTERNS, ENTITY_PATTERNS,
    GOAL_PATTERNS) are LEGACY COMPATIBILITY layers. They serve as:
    - Fallback when SemanticComposer cannot parse the input
    - Regression test baseline for previously working inputs
    - Compatibility with existing code that expects specific operation names
    
    They are NOT the primary NLU mechanism. The primary path is:
    NL → SemanticParser._activate_concepts() → _traverse_relations()
    → _assemble_meaning() → SemanticComposer.compose()
    
    Priority hierarchy:
    Explicit User Semantics > Validated Semantic Inference >
    Contextual Evidence > Retrieved Knowledge > Legacy Mapping > Keyword Match
    """
    
    # LEGACY: Operation patterns with semantic roles
    # Used as fallback when SemanticComposer composition is unavailable.
    # source: legacy, usage: fallback+compatibility, confidence: 0.6
    # OPERATION_PATTERNS removed — semantic understanding derives from concepts,
    # relations, rules, context, and reasoning via SemanticComposer.
    # kept for reference only, never used in active path.
    OPERATION_PATTERNS = []
    
    # LEGACY: Entity patterns with type hints
    # Used as fallback for entity detection. Primary path: concept activation.
    # source: legacy, usage: fallback+compatibility, confidence: 0.5
    ENTITY_PATTERNS = {
        "student": ("Student", "data", ["score", "name", "grade", "major"]),
        "학생": ("Student", "data", ["score", "name", "grade", "major"]),
        "order": ("Order", "data", ["customer", "total", "status", "shipping"]),
        "주문": ("Order", "data", ["customer", "total", "status", "shipping"]),
        "temperature": ("Temperature", "data", ["value", "timestamp", "location"]),
        "온도": ("Temperature", "data", ["value", "timestamp", "location"]),
        "contact": ("Contact", "data", ["name", "phone", "email"]),
        "연락처": ("Contact", "data", ["name", "phone", "email"]),
        "task": ("Task", "data", ["title", "description", "completed", "priority"]),
        "할일": ("Task", "data", ["title", "description", "completed", "priority"]),
        "todo": ("Task", "data", ["title", "description", "completed", "priority"]),
        "item": ("Item", "data", ["name", "price", "quantity"]),
        "아이템": ("Item", "data", ["name", "price", "quantity"]),
        "product": ("Product", "data", ["name", "price", "stock"]),
        "상품": ("Product", "data", ["name", "price", "stock"]),
        "number": ("Number", "primitive", []),
        "숫자": ("Number", "primitive", []),
        "value": ("Number", "primitive", []),
        "값": ("Number", "primitive", []),
        "score": ("Score", "primitive", []),
        "점수": ("Score", "primitive", []),
        "list": ("List", "collection", []),
        "리스트": ("List", "collection", []),
        "array": ("List", "collection", []),
        "배열": ("List", "collection", []),
        "sequence": ("List", "collection", []),
        "employee": ("Employee", "data", ["name", "salary", "department", "position"]),
        "직원": ("Employee", "data", ["name", "salary", "department", "position"]),
        "user": ("User", "data", ["name", "email", "age", "role"]),
        "사용자": ("User", "data", ["name", "email", "age", "role"]),
        "account": ("Account", "data", ["username", "password", "balance"]),
        "계정": ("Account", "data", ["username", "password", "balance"]),
        "book": ("Book", "data", ["title", "author", "isbn", "price"]),
        "책": ("Book", "data", ["title", "author", "isbn", "price"]),
        "movie": ("Movie", "data", ["title", "director", "year", "rating"]),
        "영화": ("Movie", "data", ["title", "director", "year", "rating"]),
        "car": ("Car", "data", ["make", "model", "year", "price"]),
        "자동차": ("Car", "data", ["make", "model", "year", "price"]),
        "게임": ("Game", "data", ["title", "score", "level"]),
        "game": ("Game", "data", ["title", "score", "level"]),
        "person": ("Person", "data", ["name", "age", "address"]),
        "사람": ("Person", "data", ["name", "age", "address"]),
        "시간": ("Time", "primitive", []),
        "time": ("Time", "primitive", []),
        "날짜": ("Date", "primitive", []),
        "date": ("Date", "primitive", []),
        "이름": ("Name", "primitive", []),
        "name": ("Name", "primitive", []),
        "주소": ("Address", "primitive", []),
        "address": ("Address", "primitive", []),
        "전화": ("Phone", "primitive", []),
        "phone": ("Phone", "primitive", []),
        "이메일": ("Email", "primitive", []),
        "email": ("Email", "primitive", []),
        "문자열": ("String", "primitive", []),
        "string": ("String", "primitive", []),
        "문자": ("Char", "primitive", []),
        "character": ("Char", "primitive", []),
        "boolean": ("Boolean", "primitive", []),
        "불리언": ("Boolean", "primitive", []),
        "dict": ("Dict", "collection", []),
        "사전": ("Dict", "collection", []),
        "map": ("Dict", "collection", []),
        "맵": ("Dict", "collection", []),
        # Languages and platforms the request names — the entity layer
        # records WHAT the utterance mentions (a construction fact of
        # this sentence), in its canonical spelling.  It never decides
        # what the request means: the goal and the act do that.
        "python": ("python", "language", []),
        "파이썬": ("python", "language", []),
        "kotlin": ("kotlin", "language", []),
        "코틀린": ("kotlin", "language", []),
        "java": ("java", "language", []),
        "자바": ("java", "language", []),
        "javascript": ("javascript", "language", []),
        "자바스크립트": ("javascript", "language", []),
        "js": ("javascript", "language", []),
        "typescript": ("typescript", "language", []),
        "타입스크립트": ("typescript", "language", []),
        "c++": ("cpp", "language", []),
        "cpp": ("cpp", "language", []),
        "c#": ("csharp", "language", []),
        "rust": ("rust", "language", []),
        "러스트": ("rust", "language", []),
        "swift": ("swift", "language", []),
        "스위프트": ("swift", "language", []),
        "ruby": ("ruby", "language", []),
        "php": ("php", "language", []),
        "golang": ("go", "language", []),
        "고 언어": ("go", "language", []),
        "android": ("android", "platform", []),
        "안드로이드": ("android", "platform", []),
        "ios": ("ios", "platform", []),
        "아이폰": ("ios", "platform", []),
        "iphone": ("ios", "platform", []),
        "web": ("web", "platform", []),
        "웹": ("web", "platform", []),
        "브라우저": ("web", "platform", []),
        "server": ("server", "platform", []),
        "서버": ("server", "platform", []),
        "cli": ("cli", "platform", []),
        "터미널": ("cli", "platform", []),
    }
    
    # Words that should NEVER be treated as types (grammatical words, verbs, etc.)
    STOP_WORDS = {
        # English verbs/grammatical
        "receives", "receive", "gets", "get", "takes", "take", "accepts", "accept",
        "returns", "return", "returns", "giving", "give", "gives",
        "keeps", "keep", "keeping", "kept",
        "has", "have", "having", "had",
        "makes", "make", "making", "made",
        "creates", "create", "creating", "created",
        "makes", "make", "making", "made",
        "produces", "produce", "producing", "produced",
        "outputs", "output", "outputting",
        "inputs", "input", "inputting",
        "processes", "process", "processing", "processed",
        "computes", "compute", "computing", "computed",
        "calculates", "calculate", "calculating", "calculated",
        "finds", "find", "finding", "found",
        "searches", "search", "searching", "searched",
        "sorts", "sort", "sorting", "sorted",
        "filters", "filter", "filtering", "filtered",
        "removes", "remove", "removing", "removed",
        "adds", "add", "adding", "added",
        "lists", "list", "listing", "listed",
        "stores", "store", "storing", "stored",
        "saves", "save", "saving", "saved",
        "sends", "send", "sending", "sent",
        "receives", "receive", "receiving", "received",
        "handles", "handle", "handling", "handled",
        "manages", "manage", "managing", "managed",
        "uses", "use", "using", "used",
        "reads", "read", "reading", "read",
        "writes", "write", "writing", "wrote", "written",
        "prints", "print", "printing", "printed",
        "shows", "show", "showing", "shown",
        "displays", "display", "displaying", "displayed",
        "gets", "get", "getting", "got",
        "sets", "set", "setting",
        "is", "are", "was", "were", "be", "been", "being",
        "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with", "from",
        "and", "or", "but", "if", "then", "else", "when", "where", "how", "why",
        "this", "that", "these", "those", "it", "its", "their", "them",
        "function", "method", "def", "class", "object", "instance",
        "code", "program", "script", "app", "application",
        
        # Korean verbs/grammatical
        "받는", "받는다", "받아서", "받고", "받아",
        "주는", "줍니다", "줘", "주고", "줘서",
        "가지고", "갖고", "갖는", "갖는다",
        "만드는", "만들어서", "만들고", "만들", "만들기",
        "하는", "해서", "하고", "한다", "하",
        "구하는", "구해", "구해서", "구하고", "구한다",
        "찾는", "찾아서", "찾고", "찾아", "찾는다",
        "정렬하는", "정렬해서", "정렬하고", "정렬한다",
        "필터링하는", "필터링해서", "필터링하고",
        "제거하는", "제거해서", "제거하고", "제거한다",
        "계산하는", "계산해서", "계산하고", "계산한다",
        "처리하는", "처리해서", "처리하고", "처리한다",
        "반환하는", "반환해서", "반환하고", "반환한다",
        "출력하는", "출력해서", "출력하고", "출력한다",
        "입력하는", "입력해서", "입력하고", "입력한다",
        "저장하는", "저장해서", "저장하고", "저장한다",
        "읽는", "읽어서", "읽고", "읽는다",
        "쓰는", "써서", "쓰고", "쓴다",
        "보내는", "보내서", "보내고", "보낸다",
        "받는", "받아서", "받고", "받는다",
        "생성하는", "생성해서", "생성하고", "생성한다",
        "만들", "만들", "만든",
        "함수", "기능", "메소드", "메서드",
        "코드", "프로그램", "앱", "애플리케이션",
        "클래스", "객체", "인스턴스",
        "이", "가", "을", "를", "에", "에서", "로", "으로",
        "와", "과", "도", "만", "조차", "마저",
        "은", "는", "이", "가",
    }
    
    # Valid type words (primitives and collections)
    VALID_TYPES = {
        # Primitives
        "number", "숫자", "int", "integer", "정수", "float", "실수", "double", "long", "short", "byte",
        "string", "문자열", "str", "text", "텍스트", "char", "문자", "character", "글자",
        "boolean", "불리언", "bool", "참거짓", "true", "false",
        "number", "숫자", "value", "값",
        "score", "점수", "price", "가격", "quantity", "수량", "count", "개수",
        "date", "날짜", "time", "시간", "datetime", "날짜시간",
        "name", "이름", "address", "주소", "phone", "전화", "email", "이메일",
        
        # Collections
        "list", "리스트", "array", "배열", "sequence", "시퀀스", "collection", "컬렉션",
        "set", "셋트", "map", "맵", "dict", "사전", "dictionary", "딕셔너리",
        
        # Domain types
        "student", "학생", "order", "주문", "contact", "연락처", "task", "할일",
        "employee", "직원", "user", "사용자", "product", "상품", "item", "아이템",
        "book", "책", "movie", "영화", "car", "자동차", "person", "사람",
        "temperature", "온도", "account", "계정", "game", "게임",
    }
    
# LEGACY: Goal patterns (goal_type, pattern) - order matters, more specific first
    # Used as fallback for goal detection. Primary path: concept activation + composition.
    # source: legacy, usage: fallback+compatibility, confidence: 0.6
    # GOAL_PATTERNS removed — goal detection derives from concept/relation reasoning.
    GOAL_PATTERNS = []

    def __init__(self, concept_store: ConceptStore, relation_store: RelationStore):
        self.concepts = concept_store
        self.relations = relation_store
        # An empty store activates nothing, and a parser reasoning over
        # no concepts reads every request as its own text: the default
        # vocabulary IS the parser's knowledge of what code is.
        try:
            if not self.concepts.concepts:
                from brain.concept_store import seed_default_concepts
                seed_default_concepts(self.concepts)
        except Exception:
            pass
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.compiled_ops = [(re.compile(p, re.IGNORECASE), name, op_type, role) 
                            for p, name, op_type, role in self.OPERATION_PATTERNS]
        self.compiled_goals = [(re.compile(p, re.IGNORECASE), gtype) 
                              for gtype, p in self.GOAL_PATTERNS]
    
    # Synonym/paraphrase mapping for normalization before pattern matching
    SYNONYM_MAP = {
        # Korean -> English normalized
        "덧셈": "add", "더하기": "add", "합산": "add", "더하는": "add",
        "곱셈": "multiply", "곱하기": "multiply", "곱하는": "multiply",
        "뺄셈": "subtract", "빼기": "subtract", "차이": "subtract",
        "나눗셈": "divide", "나누기": "divide",
        "소수": "prime", "素数": "prime",
        "팩토리얼": "factorial", "계승": "factorial",
        "피보나치": "fibonacci",
        "최대값": "max", "가장큰값": "max", "최대": "max",
        "최소값": "min", "가장작은값": "min", "최소": "min",
        "평균값": "average", "평균": "average",
        "합계": "sum", "총합": "sum", "전체합": "sum",
        "정렬": "sort", "소팅": "sort",
        "중복": "duplicate", "중복된": "duplicate",
        "제거": "remove", "삭제": "remove", "지우기": "remove",
        "추가": "add", "삽입": "insert",
        "검색": "search", "탐색": "search", "찾기": "search",
        "뒤집기": "reverse", "거꾸로": "reverse",
        "빈도": "frequency", "빈번": "frequency",
        "판단": "check", "확인": "check",
        "짝수": "even", "홀수": "odd",
        "양수": "positive", "음수": "negative",
        "문자": "character", "글자": "character",
        "문자열": "string", "텍스트": "text",
        "리스트": "list", "배열": "array", "목록": "list",
        "사전": "dictionary", "맵": "map",
        # English synonyms
        "compute": "calculate", "determine": "find", "obtain": "get",
        "eliminate": "remove", "erase": "remove", "delete": "remove",
        "insert": "add", "append": "add", "include": "add",
        "lookup": "search", "query": "search", "find": "search",
        "arrange": "sort", "order": "sort", "organize": "sort",
        "unique": "distinct", "deduplicate": "remove_duplicates",
        "biggest": "max", "largest": "max", "highest": "max",
        "smallest": "min", "least": "min", "lowest": "min",
        "frequent": "frequency", "common": "frequency",
        "reverse": "flip", "invert": "flip",
    }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by expanding synonyms for better pattern matching."""
        text_lower = text.lower()
        # Apply synonym normalization (word boundary aware)
        for synonym, normalized in self.SYNONYM_MAP.items():
            text_lower = re.sub(r'\b' + re.escape(synonym) + r'\b', normalized, text_lower)
        return text_lower
    
    # Semantic role classification keywords
    SEMANTIC_ROLES = {
        "arithmetic": ["add", "sum", "plus", "multiply", "times", "subtract", "minus",
                       "divide", "덧셈", "곱셈", "뺄셈", "나눗셈", "더하기", "곱하기"],
        "string_operation": ["reverse", "palindrome", "substring", "replace", "split",
                            "join", "concat", "format", "뒤집기", "회문", "자르기"],
        "collection_operation": ["sort", "filter", "map", "reduce", "flatten", "zip",
                                "unique", "distinct", "정렬", "골라", "평탄화"],
        "search_operation": ["search", "find", "lookup", "binary_search", "linear_search",
                            "탐색", "검색", "찾기"],
        "aggregate": ["sum", "count", "min", "max", "average", "reduce",
                     "합계", "세기", "최소", "최대", "평균"],
        "check_operation": ["is_prime", "is_palindrome", "is_even", "is_odd",
                           "판단", "확인", "소수", "회문"],
    }
    
    def _classify_semantic_role(self, operation_name: str, text: str) -> str:
        """Classify the semantic role of an operation based on context."""
        text_lower = text.lower()
        
        # Check operation name against known roles
        for role, keywords in self.SEMANTIC_ROLES.items():
            if operation_name in keywords:
                return role
        
        # Infer role from text context
        if any(k in text_lower for k in ["계산", "calc", "구하", "find", "get"]):
            return "aggregate"
        if any(k in text_lower for k in ["골라", "filter", "걸러", "선택"]):
            return "collection_operation"
        if any(k in text_lower for k in ["정렬", "sort", "순서"]):
            return "collection_operation"
        if any(k in text_lower for k in ["탐색", "search", "찾기", "lookup"]):
            return "search_operation"
        if any(k in text_lower for k in ["판단", "확인", "check", "test"]):
            return "check_operation"
        
        return "general"
    
    def _normalize_operation(self, name: str, text: str) -> str:
        """Normalize an operation name to a canonical semantic operation.
        Operation patterns already yield specific names, so this is a
        passthrough unless an override is registered."""
        override = getattr(self, "_OPERATION_OVERRIDES", {}).get(name)
        return override if override else name

    def _build_operation_dag(self, graph: OperationGraph, found_ops: list) -> OperationGraph:
        """Build the operation dependency DAG from found operations, wiring
        each subsequent operation to depend on the previous one."""
        prev_op_id = None
        for i, op_info in enumerate(found_ops):
            op_id = f"op_{i}"
            depends = [prev_op_id] if prev_op_id else []
            input_entities = [e.id for e in self._current_entities if e.is_input] if hasattr(self, "_current_entities") else []
            op = OperationSpec(
                id=op_id,
                name=op_info["name"],
                op_type=op_info["type"],
                semantic_role=op_info.get("semantic_role", "general"),
                depends_on=depends,
                input_entities=input_entities,
                description=f"{op_info.get('role', '')}: {op_info['name']}",
                semantic_justification=(
                    f"operation role '{op_info.get('role', '')}' matched "
                    "the request"),
                evidence=[f"pattern: {op_info.get('pattern', op_info['name'])}"],
                input_bindings=([{"input": "data", "source": d}
                                 for d in depends]
                                or [{"input": "data", "source": "__input__"}]),
            )
            graph.add_operation(op)
            prev_op_id = op_id
        if graph.operations:
            graph.entry_points = [graph.operations[0].id]
            graph.exit_points = [graph.operations[-1].id]
        return graph

    def _get_composer(self):
        """Get or create a cached SemanticComposer."""
        if not hasattr(self, '_cached_composer'):
            from brain.semantic_composer import SemanticComposer
            self._cached_composer = SemanticComposer(self.concepts, self.relations)
        return self._cached_composer


    # ------------------------------------------------------------------
    # The utterance's content structure
    # ------------------------------------------------------------------

    def _attach_focus_structure(self, semantic, text: str, context,
                                lang=None) -> None:
        """Attach what the REQUEST itself carries to the understanding.

        The investigation's own predicate decides the goal (find-out is
        never creation); the knowledge units decide the main topic and
        the CHILD information goals; the response-shaping units decide
        the constraints; each unit keeps its span in the raw text so
        every structure is traceable to what was written."""
        from brain.semantic_representation import (
            InformationGoal, ResponseSpecification, SemanticSegment,
            UserContext,
        )
        history = list(getattr(context, "conversation_history", None)
                       or [])
        previous = []
        prior_topics: list[str] = []
        for turn in history[-8:]:
            if isinstance(turn, dict):
                utterance = str(turn.get("user_input")
                                or turn.get("q") or "")
                turn_topics = list(turn.get("topics") or [])
            else:
                utterance = str(getattr(turn, "user_input", "") or "")
                turn_topics = list(getattr(turn, "topics", None) or [])
            if utterance:
                previous.append(utterance)
            for topic in turn_topics:
                topic = str(topic or "").strip()
                if topic and topic not in prior_topics:
                    prior_topics.append(topic)
        focus = analyze_focus(text, previous_turns=previous)

        # ── the investigation decides the goal ──
        if focus.inquiry:
            goal = semantic.goal
            goal.goal_type = GoalType.RESEARCH
            goal.action = "research"
            goal.requested_output = "research_report"
            goal.target = (focus.topics[0] if focus.topics
                           else focus.inquiry)
            goal.confidence = max(float(goal.confidence or 0.0), 0.8)
            goal.evidence = list(goal.evidence or []) + [
                "investigation predicate: the utterance asks the "
                "addressee to find out, not to build"]
            goal.workflow = {"stages": [dict(step)
                                        for step in focus.workflow]}
            semantic.confidence = max(float(semantic.confidence or 0.0),
                                      0.6)

        # ── the information goals: the main topic and its children ──
        goals: list = []
        main_focus = (focus.topics[0] if focus.topics
                      else (focus.inquiry or ""))
        if main_focus:
            goals.append(InformationGoal(
                id="g0", focus=main_focus, role="main",
                aspect=focus.aspect or "",
                constraints=list(focus.constraints),
                status="open"))
        for index, sub in enumerate(focus.subs, start=1):
            goals.append(InformationGoal(
                id=f"g{index}", focus=str(sub.get("focus") or ""),
                role="child", parent_id="g0",
                aspect=str(sub.get("asks_for") or ""),
                status="open"))
        semantic.information_goals = goals

        # ── the units, with their spans in the raw text ──
        segments: list = []
        for index, unit in enumerate(focus.units):
            span = list(unit.get("span") or [0, 0])
            try:
                start, end = int(span[0]), int(span[1])
            except Exception:
                start, end = 0, 0
            segments.append(SemanticSegment(
                segment_id=f"s{index}",
                kind=str(unit.get("kind") or "report"),
                text=str(unit.get("text") or ""),
                parent_id="" if unit is focus.units[0] else "s0",
                source_span=[max(0, start), max(start, end)],
                semantic_roles=(["focus"] if unit.get("focus") else []),
            ))
        semantic.segments = segments

        # ── the response this request demands (its own framing) ──
        # What the REQUEST's grammar already says about the reply: its
        # response-shaping units.  The bounds themselves are decided
        # with the turn's evidence (conversation_knowledge.build_plan),
        # which writes the complete specification.
        semantic.response_spec = ResponseSpecification(
            constraints=list(focus.constraints))

        # ── the speaker's frame, as far as THIS request states it ──
        semantic.user_context = UserContext(
            language=getattr(lang, "value", "") or "",
            preferences=dict(getattr(context, "user_preferences", None)
                             or {}),
            prior_topics=prior_topics[:5],
        )

    def parse(self, text: str, context: ParseContext = None,
                  trace: 'ObservationTrace' = None) -> SemanticRepresentation:
        """Main entry point: text → SemanticRepresentation.
        
        Args:
            text: Natural language input
            context: Optional parsing context
            trace: Optional ObservationTrace for recording pipeline observations.
                   Auto-created when not provided, so every parse produces
                   the canonical stage audit trail (#13).
        """
        from brain.observation import ObservationTrace, Evidence, EvidenceType
        context = context or ParseContext()
        # The parent context is available to the goal inference (the
        # conversation's current project drives the existing-artifact
        # evidence)
        self._parse_context = context
        if trace is None:
            trace = ObservationTrace(raw_input=text)
        
        # 1. Lexical preprocessing
        lang = self._detect_language(text)
        normalized = self._normalize(text, lang)
        if trace:
            trace.record_pipeline_step(
                "language_detection", "semantic_parser",
                input_summary=text[:80],
                output_summary=f"lang={lang.value}, normalized={normalized[:80]}",
            )
        
        # 2. Concept activation
        activated_concepts = self._activate_concepts(normalized, lang, context)
        if trace:
            for ac in activated_concepts[:10]:
                trace.record_keyword_match(
                    module="semantic_parser.concept_activation",
                    matched_text=ac.matched_text,
                    mapped_to=ac.name,
                    confidence=ac.activation_strength,
                    pattern_name="concept_activation",
                )
            trace.record_pipeline_step(
                "concept_activation", "semantic_parser",
                input_summary=normalized[:80],
                output_summary=f"{len(activated_concepts)} concepts activated",
            )
        
        # 3. Relation traversal
        relation_paths = self._traverse_relations(activated_concepts)
        if trace:
            trace.record_pipeline_step(
                "relation_traversal", "semantic_parser",
                input_summary=f"{len(activated_concepts)} concepts",
                output_summary=f"{len(relation_paths)} relation paths",
            )
        
        # 4. Meaning assembly
        semantic = self._assemble_meaning(
            text=text,
            normalized=normalized,
            lang=lang,
            activated_concepts=activated_concepts,
            relation_paths=relation_paths,
            context=context
        )
        # The semantic carries its own audit trail (#13); the declared
        # target stage (#1) is recorded here, right after the target was
        # captured during assembly.
        semantic.observation = trace
        trace.record_pipeline_step(
            "target_detection", "semantic_parser",
            input_summary=text[:80],
            output_summary=str(semantic.target),
        )
        if trace:
            for ent in semantic.entities:
                trace.record_entity(
                    module="semantic_parser.assemble_meaning",
                    name=ent.name, type_name=ent.type_name,
                    kind=ent.kind, confidence=semantic.confidence,
                )
            for op in semantic.operation_graph.operations:
                trace.record_operation(
                    module="semantic_parser.assemble_meaning",
                    name=op.name, source="OPERATION_PATTERNS",
                    confidence=semantic.confidence,
                )
            for inp in semantic.inputs:
                trace.record_type(
                    module="semantic_parser.assemble_meaning",
                    name="input", inferred_type=inp.type_signature,
                    confidence=semantic.confidence,
                )
            for out in semantic.outputs:
                trace.record_type(
                    module="semantic_parser.assemble_meaning",
                    name="output", inferred_type=out.type_signature,
                    confidence=semantic.confidence,
                )
            trace.record_pipeline_step(
                "meaning_assembly", "semantic_parser",
                input_summary=normalized[:80],
                output_summary=f"goal={semantic.goal.goal_type.value}, "
                              f"entities={len(semantic.entities)}, "
                              f"ops={semantic.get_operation_order()}, "
                              f"conf={semantic.confidence:.2f}",
                confidence=semantic.confidence,
            )
        
        # 4a. The utterance's CONTENT structure: the investigation it
        # asks for (which decides the goal), the information goals it
        # carries, the response it demands, the speaker's frame and the
        # units it is made of.  Every one of them is read from this
        # request's own grammar — never from a table of sentences.
        try:
            self._attach_focus_structure(semantic, text, context, lang)
        except Exception:
            semantic.information_goals = []
            semantic.segments = []
        if trace:
            trace.record_pipeline_step(
                "focus_structure", "semantic_parser",
                input_summary=text[:80],
                output_summary=f"goal={semantic.goal.goal_type.value}, "
                              f"goals={len(semantic.information_goals)}, "
                              f"segments={len(semantic.segments)}",
            )

        # 4b. Dialogue act — the utterance's illocutionary structure and
        # the intent derived from it.  The act comes from the sentence's
        # grammar; the intent type from the act plus the CONTENT's kind.
        # The topic words never decide the intent type.
        try:
            act = analyze_act(text)
            intent = derive_intent(semantic, act, text, None, None)
            semantic.dialogue_act = act.to_dict()
            self._reconcile_goal_with_intent(semantic, intent)
            semantic.dialogue_intent = intent.to_dict()
            # The parse confidence is the agreement of the two
            # independent readings of THIS utterance: the goal inference
            # over the semantic structure, and the act reading of the
            # grammar.  Two readers reaching the same conclusion is
            # evidence that the utterance says what it appears to say —
            # stronger than either reading alone; two readers reaching
            # different conclusions is evidence that it does not, so the
            # report is capped at the weaker of the two.  Content
            # coverage (entities, operations, I/O) still stands on its
            # own: a greeting has no operations to cover, and that is
            # not uncertainty about what it is.
            _consensus = self._consensus_confidence(semantic.goal, intent)
            if _consensus is not None:
                semantic.confidence = max(semantic.confidence, _consensus)
        except Exception:
            semantic.dialogue_act = {}
            semantic.dialogue_intent = {}
        if trace:
            trace.record_pipeline_step(
                "dialogue_act", "semantic_parser",
                input_summary=text[:80],
                output_summary=f"form={semantic.dialogue_act.get('form')}, "
                              f"force={semantic.dialogue_act.get('force')}, "
                              f"intent={semantic.dialogue_intent.get('type')}",
            )

        # 5. Compositional analysis via SemanticComposer (independent pipeline)
        try:
            composer = self._get_composer()
            result = composer.compose(text)
            semantic.composition = {
                "user_cues": [
                    {"primitive": c.primitive, "text": c.matched_text,
                     "params": c.params, "confidence": c.confidence}
                    for c in result.user.cues
                ],
                "inference_level": result.derived.inference_level,
                "graph": result.derived.graph.to_dict(),
                "uncertain_flags": result.derived.uncertain_flags,
                "annotations": result.derived.annotations,
                "code": result.derived.code,
                # Text-level constraints (e.g. preserve_order) — they are
                # requirements on the result, not operations, and the
                # requirement contract verifies them.
                "constraints": [dict(c) for c in result.user.constraints],
            }
            # Boost confidence if composition found strong atomic ops
            if result.derived.inference_level == "direct":
                semantic.confidence = max(semantic.confidence, 0.85)
            if trace:
                for cue in result.user.cues:
                    trace.record_operation(
                        module="semantic_composer",
                        name=cue.primitive,
                        source="decompose",
                        confidence=cue.confidence,
                    )
                trace.record_pipeline_step(
                    "semantic_composition", "semantic_composer",
                    input_summary=text[:80],
                    output_summary=f"level={result.derived.inference_level}, "
                                  f"ops={[op.name for op in result.derived.graph.get_execution_order()]}, "
                                  f"flags={result.derived.uncertain_flags}",
                )
                trace.set_semantic_composition(semantic.composition)
            
            # Composition rule: when the SemanticComposer produces operations,
            # use the composer's operations as the authoritative source. This
            # enables reasoning from semantic relationships without requiring
            # exact keyword matches in OPERATION_PATTERNS.
            # Priority rule: only a decomposition that CAPTURES the user's
            # semantics may override the goal-derived operations. An
            # optimization objective over a substructure (e.g. longest
            # contiguous subarray) or an aggregate constraint on it is
            # structured semantics that atomic decomposition cannot express —
            # replacing with a lone fragment op would lose the
            # objective / substructure / constraint semantics already bound
            # to the goal. A lone substructure without an objective (e.g.
            # "two numbers" → pair) does not block: the composition may
            # still be the authoritative whole. An uncertain decomposition
            # (level "uncertain") is likewise incomplete evidence.
            composer_ops = [op.name for op in result.derived.graph.get_execution_order()]
            pattern_ops = [op.name for op in semantic.operation_graph.operations]
            sem_has_structure = (
                (bool(getattr(semantic, 'objectives', []))
                 and bool(getattr(semantic, 'substructures', [])))
                or any(getattr(c, 'constraint_type', '') == "aggregate"
                       for c in semantic.constraints))
            # Replace pattern-based ops with composer ops when:
            # - the decomposition is validated (not uncertain), AND
            # - the semantics carry no structure the decomposition
            #   cannot express, AND
            # - composer has different (more specific) operations, OR
            # - composer has more operations
            if (composer_ops
                    and result.derived.inference_level != "uncertain"
                    and not sem_has_structure
                    and composer_ops != pattern_ops):
                # Build a new OperationGraph from the composer's output
                from brain.semantic_representation import OperationSpec, OperationGraph, OperationType
                new_graph = OperationGraph()
                for cop in result.derived.graph.get_execution_order():
                    op = OperationSpec(
                        id=cop.id,
                        name=cop.name,
                        op_type=cop.op_type,
                        semantic_role=cop.semantic_role,
                        params=cop.params,
                        depends_on=cop.depends_on,
                        description=cop.description,
                        semantic_justification=cop.semantic_justification,
                        evidence=list(cop.evidence),
                        input_bindings=[dict(b)
                                        for b in cop.input_bindings],
                        preconditions=list(cop.preconditions),
                    )
                    new_graph.operations.append(op)
                semantic.operation_graph = new_graph
        except Exception:
            pass

        # 5a. The ARTIFACT namespace — generated code and computation
        # primitives — belongs to the ARTIFACT path.  A turn whose goal
        # is knowledge carries none of it: the tool's output never
        # stands in for the user's own meaning, and a word like "over"
        # in a span ("over the next year") never becomes an operation.
        # The composer may decompose ANY text into primitives; only the
        # artifact goal gives them a place to stand.
        if semantic.goal.goal_type in (
                GoalType.EXPLAIN, GoalType.QUESTION, GoalType.CHAT,
                GoalType.INFORMATION_QUERY, GoalType.COMMAND_QUERY,
                GoalType.RESEARCH, GoalType.AMBIGUOUS, GoalType.UNKNOWN):
            try:
                _ops = semantic.operation_graph.operations
                if _ops:
                    del _ops[:]
            except Exception:
                pass
            _composition = getattr(semantic, "composition", None)
            if isinstance(_composition, dict):
                _composition["code"] = ""
                _composition["user_cues"] = []
                _composition["graph"] = {"operations": [], "entry_points": [],
                                         "exit_points": [],
                                         "execution_order": []}
            _entities = list(getattr(semantic, "entities", None) or [])
            _kept = [e for e in _entities
                     if str(getattr(e, "kind", "") or "") not in
                     ("primitive", "collection")]
            if len(_kept) != len(_entities):
                semantic.entities = _kept

        # 5b. Declared output type (spec #5): the terminal operation's
        # contract over the composed graph — the same two facts the IR's
        # return-type contract reads (terminal op + element domain from
        # the operation vocabulary), so the semantic output type, the
        # IR's type_ref, the generated signature and the verified
        # behavior can be checked for equality.  Unknown → "" (declares
        # nothing instead of guessing).
        semantic.output_type = self._derive_output_type(semantic)
        if trace:
            trace.record_pipeline_step(
                "output_type", "semantic_parser",
                input_summary=", ".join(
                    (semantic.composition or {}).get("graph", {})
                    .get("execution_order", [])) or "(no composition)",
                output_summary=semantic.output_type,
            )

        # 5c. Type/dataflow validation (spec #4): the final graph's
        # structural and typing contracts — violations are recorded in
        # the semantic and in the trace, never silently passed.
        semantic.dataflow_violations = \
            semantic.operation_graph.validate_dataflow()
        if semantic.dataflow_violations and trace:
            trace.record_error(
                stage="semantic_composition",
                error_type="dataflow_violation",
                message="; ".join(semantic.dataflow_violations),
                module="semantic_parser")

        # 6. Detect competitive programming patterns
        semantic.competitive_patterns = self._detect_competitive_patterns(text)
        
        return semantic

    def _derive_output_type(self, semantic: SemanticRepresentation) -> str:
        """Return type of the composed function, from graph structure only.

        The terminal operation decides the SHAPE (aggregate → scalar,
        transform → sequence, predicate → bool); the element domain
        comes from the operation vocabulary (parity predicates are
        integer-only), exactly as the IR derives it.  The scalar map is
        SHARED with the IR's return-type contract on purpose: #5
        requires semantic == IR == generated == verified, and a single
        map cannot drift apart from itself.
        """
        from brain.semantic_composer import PRIMITIVE_ONTOLOGY
        from brain.generation.ir_builder import _IR_SCALAR_TYPES

        names: list[str] = []
        comp = semantic.composition or {}
        graph = comp.get("graph") or {}
        exec_ids = graph.get("execution_order") or []
        by_id = {o.get("id"): o for o in (graph.get("operations") or [])
                 if isinstance(o, dict)}
        if exec_ids:
            names = [by_id[i].get("name", "")
                     for i in exec_ids if i in by_id]
        if not names:
            names = [op.name
                     for op in semantic.operation_graph.get_execution_order()]
        if not names:
            if semantic.outputs:
                return self._lang_type(semantic.outputs[0].type_signature)
            return ""
        last = names[-1]
        if last not in PRIMITIVE_ONTOLOGY:
            return ""
        # Element domain: the DECLARED element decides — a list of strings
        # has str elements — falling back to the parity rule (parity
        # predicates are defined on integers only) and then to the generic
        # Number domain.  Mirrors ir_builder's element rule on purpose, so
        # semantic == IR == generated == verified (#5) by one derivation.
        declared = self._element_type(semantic)
        if declared:
            elem = self._lang_type(declared) or "float"
        elif any(n in ("filter_even", "is_multiple_of") for n in names):
            elem = "int"
        else:
            elem = "float"
        # Terminal contract (mirrors ir_builder._ir_return_type):
        if last in ("calculate_sum", "find_max", "find_min"):
            return elem
        if last in _IR_SCALAR_TYPES:
            return _IR_SCALAR_TYPES[last]
        produces = PRIMITIVE_ONTOLOGY[last].produces
        if produces == "sequence":
            return f"list[{elem}]"
        if produces == "predicate":
            return "bool"
        if produces == "scalar" or produces == "element":
            return elem
        if produces == "optional":
            # A nullable element ("char | None"): the optional-of-element
            # contract, not a list.
            return f"optional[{elem}]" if elem == "str" else "optional"
        return ""
    
    def _detect_language(self, text: str) -> Language:
        """Detect input language from character ranges."""
        if re.search(r"[가-힣]", text):
            return Language.KOREAN
        elif re.search(r"[ぁ-んァ-ヶー]", text):
            return Language.JAPANESE
        elif re.search(r"[\u4e00-\u9fff]", text):
            return Language.CHINESE
        elif re.search(r"[a-zA-Z]", text):
            return Language.ENGLISH
        return Language.UNKNOWN
    
    def _normalize(self, text: str, lang: Language) -> str:
        """Normalize text for processing."""
        # Remove extra whitespace, normalize punctuation
        text = re.sub(r"\s+", " ", text.strip())
        # Convert full-width to half-width for Japanese/Chinese
        text = text.translate(str.maketrans(
            "！＠＃＄％＾＆＊（）＿＋－＝［］｛｝；：＂＇＜＞，．？／＼｜｀～",
            "!@#$%^&*()_+-=[]{};:\"'<>?,./\\|`~"
        ))
        return text
    
    def _activate_concepts(self, text: str, lang: Language, context: ParseContext) -> list[ActivatedConcept]:
        """Activate concepts from the concept store based on the text.

        The lexical matching uses the concept store's own tokenized
        scoring (the exact tag match is the strongest signal — the same
        scoring the retrieval layer uses); the phrase matching uses the
        ADJACENT WORD PAIRS of the text against the concept's multi-word
        tags.  No character-window n-grams: they match every stopword
        fragment ('the', 'in', 'at') and flood the activation with
        meaningless noise.
        """
        activated = []
        text_lower = text.lower()

        # The concept store's own scoring: the tokenized query + the tags
        scored = self.concepts.search_with_scores(text)
        concept_scores = {}
        concept_matches = {}
        for c, s in scored:
            concept_scores[c.id] = s
            concept_matches[c.id] = c.name.lower()

        all_concepts = self.concepts.list_all()
        for concept in all_concepts:
            strength = concept_scores.get(concept.id, 0.0)
            match_type = "lexical" if strength > 0 else ""
            matched_text = concept_matches.get(concept.id, "")

            # 2. Phrase matching: the ADJACENT WORD PAIRS of the text
            # matched against the concept's text — a phrase match is a
            # strong signal ("highest to lowest" → sort_descending).
            # A pair composed ENTIRELY of function words ("on the",
            # "of the", "by the") carries no meaning — grammatical
            # words never activate a concept by themselves.
            concept_text = f"{concept.name} {concept.description} {' '.join(concept.tags)}".lower()
            words_list = re.findall(r"[a-z가-힣]+", text_lower)
            for i in range(len(words_list) - 1):
                w1, w2 = words_list[i], words_list[i + 1]
                phrase = w1 + " " + w2
                if w1 in self.STOP_WORDS and w2 in self.STOP_WORDS:
                    continue
                if len(phrase) >= 6 and phrase in concept_text:
                    strength += 1.5
                    if not match_type:
                        match_type = "phrase"
                        matched_text = phrase

            # 3. Contextual boosting (conversation history, current project)
            if context.conversation_history:
                for turn in context.conversation_history[-3:]:
                    user_input = getattr(turn, 'user_input', '') if hasattr(turn, 'user_input') else (turn.get("user_input", "") if hasattr(turn, 'get') else "")
                    if concept.name.lower() in user_input.lower():
                        strength += 0.5
                        match_type = "contextual"
                        matched_text = concept.name.lower()

            if context.current_project:
                proj_name = context.current_project.get("name", "").lower()
                if proj_name and proj_name in concept_text:
                    strength += 0.3
                    match_type = "contextual"

            if strength > 0:
                activated.append(ActivatedConcept(
                    concept_id=concept.id,
                    name=concept.name,
                    category=concept.category,
                    activation_strength=strength,
                    match_type=match_type,
                    matched_text=matched_text
                ))

        # Sort by strength
        activated.sort(key=lambda c: -c.activation_strength)
        return activated[:20]  # Top 20
    
    def _traverse_relations(self, activated: list[ActivatedConcept]) -> list[RelationPath]:
        """Traverse knowledge graph to find relations between activated concepts."""
        paths = []
        activated_ids = {c.concept_id for c in activated}
        
        # Find paths between activated concepts (depth 2)
        for i, c1 in enumerate(activated):
            for c2 in activated[i+1:]:
                path = self.relations.find_path(c1.concept_id, c2.concept_id, max_depth=2)
                if path:
                    # Calculate path weight
                    weight = 0.0
                    relations = []
                    for j in range(len(path) - 1):
                        rels = self.relations.find_relations(path[j], path[j+1])
                        if rels:
                            r = rels[0]
                            relations.append(r.relation_type.value)
                            weight += r.weight
                    
                    paths.append(RelationPath(
                        path=path,
                        relation_types=relations,
                        total_weight=weight / len(relations) if relations else 0,
                        start_concept=c1.concept_id,
                        end_concept=c2.concept_id
                    ))
        
        # Also find related concepts from each activated concept
        for c in activated:
            related = self.relations.find_related(c.concept_id, direction="both")
            for rel in related[:5]:  # Top 5 per concept
                other_id = rel.target_id if rel.source_id == c.concept_id else rel.source_id
                if other_id not in activated_ids:
                    other = self.concepts.get(other_id)
                    if other:
                        paths.append(RelationPath(
                            path=[c.concept_id, other_id],
                            relation_types=[rel.relation_type.value],
                            total_weight=rel.weight,
                            start_concept=c.concept_id,
                            end_concept=other_id
                        ))
        
        return paths[:20]
    
    def _assemble_meaning(
        self,
        text: str,
        normalized: str,
        lang: Language,
        activated_concepts: list[ActivatedConcept],
        relation_paths: list[RelationPath],
        context: ParseContext
    ) -> SemanticRepresentation:
        """Assemble the final semantic representation."""
        
        semantic = SemanticRepresentation(
            source_language=lang,
            raw_text=text,
            activated_concept_ids=[c.concept_id for c in activated_concepts],
            relation_paths=[p.path for p in relation_paths]
        )

        # Declared output target (spec #1): detected ONCE here and carried
        # inside the semantic — every later stage reads it instead of
        # re-guessing from the request text.  Requests that declared no
        # language/platform keep {} (detect_target's default fallback is
        # never stored as a declaration).
        semantic.target = detect_target(text).to_semantic_dict()

        # 1. Infer Goal
        semantic.goal = self._infer_goal(normalized, lang, activated_concepts)
        
        # 2. Extract Entities
        semantic.entities = self._extract_entities(normalized, lang, semantic.goal, activated_concepts)
        
        # 3. Extract Operations with ordering
        semantic.operation_graph = self._extract_operations(normalized, lang, semantic.entities, semantic.goal)
        
        # 4. Extract I/O
        semantic.inputs, semantic.outputs = self._extract_io(normalized, lang, semantic.entities, semantic.operation_graph)
        
        # 5. Extract Constraints
        semantic.constraints = self._extract_constraints(normalized, lang)
        
        # 6. Extract Objectives (optimization goals like "longest", "shortest", "maximum")
        semantic.objectives = self._extract_objectives(normalized, lang, semantic)
        
        # 7. Extract Substructures (structural patterns like "contiguous subarray")
        semantic.substructures = self._extract_substructures(normalized, lang)
        
        # 8. Extract aggregate constraints (like "sum is zero")
        aggregate_constraints = self._extract_aggregate_constraints(normalized, lang)
        semantic.constraints.extend(aggregate_constraints)
        
        # 8b. Parse arithmetic expressions ("square of A + B" → "(A + B) ** 2")
        # Variable names resolved against extracted entities (variable declarations)
        entity_var_names = [e.name for e in semantic.entities]
        semantic.arithmetic_expression = self._extract_arithmetic_expression(
            normalized, entity_var_names)

        # 8c. Semantic-driven goal resolution: when the goal type is still
        # UNKNOWN but the semantics carry algorithmic structure (an objective
        # over a substructure, an aggregate constraint, or a named
        # algorithm the user referenced), the sentence describes a
        # computation to perform — the algorithmic semantics ARE the
        # goal's meaning. Promote to CREATE_FUNCTION so the meaning is
        # not lost downstream.
        #
        # A strongly-activated algorithmic primitive concept (e.g. the
        # Sorting primitive activating on "정렬") is ALSO algorithmic
        # semantics: the user referenced the algorithm by meaning. The
        # concept store's ontology decides (category + activation
        # strength), never a surface keyword list.
        #
        # An AMBIGUOUS goal with real algorithmic semantics is the
        # same meaning: the candidates were uncertain but the
        # objectives/substructures/aggregate constraints are semantic
        # EVIDENCE of a computation — the algorithmic semantics ARE
        # the goal's meaning regardless of which candidate list the
        # concept matches produced.
        if semantic.goal.goal_type in (GoalType.UNKNOWN,
                                       GoalType.AMBIGUOUS):
            has_agg = any(getattr(c, 'constraint_type', '') == "aggregate"
                          for c in semantic.constraints)
            has_algorithm_primitive = any(
                getattr(c, 'category', '') == 'algorithmic_primitive'
                and getattr(c, 'activation_strength', 0) >= 4.0
                for c in (activated_concepts or []))
            if semantic.objectives or semantic.substructures or has_agg \
                    or has_algorithm_primitive \
                    or semantic.goal.algorithm_hint:
                # The PROMOTION needs the utterance's leave: semantic
                # structure (an objective, a substructure, an algorithm
                # concept) is the TOPIC's vocabulary as often as it is a
                # requested computation, so the act decides whether this
                # content may become an artifact.  "I want a guide to
                # learning programming" carries computation semantics —
                # the request still asks for content, and a turn that
                # names the content it wants may never be restated as
                # its opposite.
                from brain.dialogue_act import (
                    analyze_act, asks_for_content,
                )
                _is_content, _why = asks_for_content(
                    analyze_act(text), text)
                if not _is_content:
                    semantic.goal.goal_type = GoalType.CREATE_FUNCTION
                    semantic.goal.confidence = max(
                        semantic.goal.confidence, 0.6)
                    if not semantic.operation_graph.operations:
                        self._infer_default_operations(
                            semantic.operation_graph, semantic.goal,
                            semantic.entities)
                else:
                    # The act asks for CONTENT, so the ambiguity cannot
                    # resolve to an artifact — it resolves to what the
                    # turn does ask for: knowledge.  Leaving it
                    # unresolved would ask the speaker to choose
                    # between readings the act already decided.
                    semantic.goal.goal_type = GoalType.EXPLAIN
                    semantic.goal.confidence = max(
                        semantic.goal.confidence, 0.55)
                    semantic.goal.evidence = list(
                        semantic.goal.evidence or []) + [
                        "the act asks for content, not an artifact: "
                        + str(_why)]

        # 9. Wire objectives into goal
        if semantic.objectives:
            obj = semantic.objectives[0]  # primary objective
            semantic.goal.goal_direction = obj.direction
            semantic.goal.goal_measure = obj.measure
            semantic.goal.goal_object = obj.target

        # 9b. The objective realizes an operation, and the objective plus
        # the declared element type decide the function's I/O.  Both read
        # the STRUCTURE built above (goal, entities, operations, I/O) —
        # the shape of the answer is never read back from the sentence.
        self._bind_objective_operations(semantic)
        self._wire_io_types(semantic)
        
        # 10. Wire substructures into goal modifiers
        for sub in semantic.substructures:
            if sub.structure_type not in semantic.goal.modifiers:
                semantic.goal.modifiers.append(sub.structure_type)
        
        # 11. Infer domain from semantic evidence (entities, operations, constraints)
        semantic.goal.domain, semantic.goal.domain_evidence = self._infer_domain(
            semantic.entities, semantic.operation_graph, semantic.constraints, activated_concepts, normalized
        )
        
        # 12. Set confidence
        semantic.confidence = self._calculate_confidence(semantic, activated_concepts)
        return semantic
    
    def _detect_computer_semantics(self, text: str, text_lower: str) -> 'GoalSpec | None':
        """Detect desktop-manipulation semantics (Step 0 of _infer_goal).

        A sentence describes acting ON the computer when a computer-control
        verb (open/run/build/install/draw/click/...) co-occurs with a
        computer object (program, app, file, screen, browser, image, ...)
        or a known application name.  This is semantic detection (verb
        classes + target objects), never fixed sentence matching.
        Returns a COMPUTER GoalSpec, or None when the sentence is not a
        desktop task.
        """
        try:
            from computer.nlu import (
                _COMPUTER_VERBS,
                _COMPUTER_OBJECTS,
                _KNOWN_APPLICATIONS,
            )
        except Exception:
            return None
        # word-boundary matching: substrings like "draw" inside
        # "drawing" or "ide" inside "inside" are NOT computer-act
        # evidence
        # Word-boundary matching: English needs boundaries (substr "run"
        # inside "rung"), Korean carries morphology by suffixing —
        # "실행해줘" / "파일을" would never match \b실행\b, so Korean
        # terms match as substrings.  English boundaries use ASCII
        # classes so a Korean particle attached to a Latin word
        # ("paint로") still counts as a boundary.
        def _term_match(term: str) -> bool:
            if term.isascii():
                return bool(re.search(
                    r'(?<![A-Za-z0-9_])' + re.escape(term) +
                    r'(?![A-Za-z0-9_])', text_lower))
            return term in text_lower

        has_verb = any(_term_match(v) for v in _COMPUTER_VERBS)
        has_object = any(_term_match(o) for o in _COMPUTER_OBJECTS)
        has_app = any(_term_match(a) for a in _KNOWN_APPLICATIONS)
        # Shell evidence: CLI flag syntax with a letter head ("-la",
        # "--all" — never a bare negative number "-1") marks a command
        # span; a shell venue ("in the terminal") marks where it runs.
        # Either completes the computer act with a run verb ("run ls
        # -la") or with each other ("in the terminal, df -h").
        cmd_like = bool(re.search(
            r'(?<![\w/\-])--?[A-Za-z][\w-]*', text_lower))
        venue = bool(re.search(
            r'터미널|콘솔|쉘|셸|명령\s*(?:창|줄)|'
            r'(?:command\s*line|terminal|shell|console|cli|prompt)',
            text_lower))
        if not (has_verb and (has_object or has_app or cmd_like or venue)
                or (cmd_like and venue)):
            return None
        goal = GoalSpec()
        goal.goal_type = GoalType.COMPUTER
        goal.domain = "computer"
        goal.confidence = 0.75
        return goal

    def _infer_goal(self, text: str, lang: Language,
                    concepts: list[ActivatedConcept]) -> GoalSpec:
        """Infer the goal from the semantic evidence — a structured goal
        object, never from surface keywords.

        Priority hierarchy:
          1. Explicit User Semantics — the composer's decomposition: the
             UserSemantics' cues/primitives ARE the meaning observed in
             the user's text
          2. Validated Semantic Inference — the semantics carry
             computation structure (an objective over a substructure, an
             aggregate constraint, a strongly-activated operation or
             algorithmic primitive)
          3. Conversation Context — the parent context is wired later
             from ParseContext
          4. Retrieved Knowledge — the relation paths are wired later;
             knowledge never injects the goal type
          5. Legacy Mapping — a keyword occurrence is recorded as a
             legacy candidate and verified against the semantics; it can
             never decide the goal type alone
          6. Surface Keyword — never used

        The dialogue act derives from the sentence's MEANING structure
        (grammatical analysis, not keyword matching):
          - an interrogative form (question-mark punctuation, WH-word
            initial) is an information request
          - a negated or past-tense main predicate is NOT a generation
            request: the sentence reports or refuses, it does not request
          - no semantic content at all (no activated concepts, no
            entities, no primitives) is a social interaction
          - anything else leaves the goal OPEN: the detection order is
            semantic intent, then goal, then topic — the dialogue intent
            (read from the same grammar right after this inference) binds
            the open goal, and the activated concepts are recorded as the
            utterance's TOPIC.  A knowledge concept is evidence about the
            subject, never a reading of what the user asks for, so it can
            neither pick a goal type nor make the goal ambiguous; only an
            intent that cannot decide keeps the turn open for the
            clarification that real ambiguity asks for.
        """
        goal = GoalSpec()
        text_lower = text.lower()

        # ── Step 0: The sentence's meaning structure ──
        # The performative analysis: form (interrogative/imperative/
        # declarative), negation, tense — grammatical morphology, not a
        # keyword list.  A negated or past-tense performative is a
        # report/refusal, not a request.
        perf = self._analyze_performative(text)

        # ── Step 0b: Interrogative form is a dialogue act ──
        # The grammatical interrogative decides the dialogue act, with
        # the semantics as tiebreaker: "how to" asks for a METHOD —
        # always an information request; "which/what is/?"/어떻게 asks
        # for a VALUE or fact — an information request UNLESS the
        # semantics carry a computation request (a strong operation
        # concept over data): then the semantics win.
        if perf["form"] == "interrogative":
            # A REAL computation request needs computation semantics over
            # actual data: an optimization objective over a data entity
            # ("Which value is greatest among these numbers?"), or a
            # strong algorithmic primitive with a data entity.  A meta
            # question ("What is a pointer?") triggers vocabulary echoes
            # but has no data/objective structure — it is an information
            # request.
            strong_primitive = any(
                getattr(c, 'category', '') == 'algorithmic_primitive'
                and getattr(c, 'activation_strength', 0) >= 4.0
                for c in (concepts or []))
            probe_entities = self._extract_entities(text, lang, goal, concepts)
            probe_objectives = self._extract_objectives(text, lang)
            has_computation = (bool(probe_objectives) and bool(probe_entities))                 or (strong_primitive and bool(probe_entities))
            # The investigation's PREDICATE decides what the question is
            # asking FOR.  A query over a failure predication ("Why does
            # this error occur?") asks for a REPAIR of the failing thing;
            # the topic words never decide it.  It is not a method
            # question either: the ask is "find what is wrong", not
            # "tell me the procedure".
            _act = perf.get("act")
            if getattr(_act, "predicate_class", "") == "failure":
                goal.goal_type = GoalType.FIX
                goal.action = "repair"
                goal.requested_output = "diagnosis"
                goal.evidence = ["interrogative form", "failure predicate",
                                 "the investigation asks for a repair"]
                goal.target = self._extract_target(text)
                goal.algorithm_hint = self._extract_algorithm_hint(text)
                _support = sum(c.activation_strength
                               for c in concepts) / max(len(concepts), 1)
                goal.confidence = 0.7 + 0.3 * min(_support, 1.0)
                return goal
            if perf["method_question"] or not has_computation:
                # A method question routes to the shell engine ONLY when
                # the shell/platform semantics are STRONG (the question's
                # topic is a shell environment); a weak shell mention
                # with a meta question ("How does create work?") is a
                # knowledge question, not a command query.
                strong_shell = self._shell_semantics_present(concepts)
                goal.goal_type = (GoalType.COMMAND_QUERY
                                  if perf["method_question"] and strong_shell
                                  else GoalType.QUESTION)
                goal.action = "query"
                goal.requested_output = ("shell_command" if perf["method_question"]
                                         and strong_shell
                                         else "answer")
                goal.evidence = ["interrogative form",
                                 "method question" if perf["method_question"]
                                 else "information request"]
                goal.target = self._extract_target(text)
                goal.algorithm_hint = self._extract_algorithm_hint(text)
                concept_support = sum(c.activation_strength for c in concepts) / max(len(concepts), 1)
                goal.confidence = 0.7 + 0.3 * min(concept_support, 1.0)
                return goal
            # else: fall through — the computation semantics win

        # ── Step 0a: Explanation request is a dialogue act ──
        # "Explain X" / "X를 설명해줘" asks for an EXPLANATION of a
        # concept — the operations found are the topic's structure, not
        # a generation request.  The explanation performative decides
        # (any sentence form: an imperative "Explain X" and a declarative
        # "X를 설명해줘" are both explanation requests).
        # A negated generation clause ("Do not write code") belongs to
        # the first clause; the following explanation request is the
        # turn's actual ask, not another negation of the same predicate.
        _no_code_explain = bool(
            re.search(r"\b(?:do\s+not|don't)\s+"
                      r"(?:write|generate|create|produce)\s+(?:any\s+)?code\b",
                      text_lower)
            and re.search(r"\b(?:explain|describe|tell\s+me)\b", text_lower))
        if _no_code_explain:
            goal.goal_type = GoalType.EXPLAIN
            goal.action = "explain"
            goal.requested_output = "explanation"
            goal.evidence = ["the generation clause is negated",
                             "the explanation clause is the request"]
            goal.target = self._extract_target(text)
            goal.algorithm_hint = self._extract_algorithm_hint(text)
            goal.confidence = 0.7
            return goal
        if perf["explanation_request"] \
                and not (perf["negated"] or perf["past"]):
            goal.goal_type = GoalType.EXPLAIN
            goal.action = "explain"
            goal.requested_output = "explanation"
            goal.evidence = ["explanation request"]
            goal.target = self._extract_target(text)
            goal.algorithm_hint = self._extract_algorithm_hint(text)
            goal.confidence = 0.7
            return goal

        # ── Step 0b-b: Modification request is a dialogue act ──
        # "X에 검색 기능을 추가해줘" / "Add search functionality to the
        # EXISTING application" — the action applies to an ALREADY-EXISTING
        # artifact (the existing-artifact evidence in the semantic
        # structure + the add/remove/modify action).  Structurally
        # distinct from creation: the artifact exists, the request adds
        # to it.  Detected from the grammatical performative (the
        # modification verb + the existing-artifact reference), and the
        # parent context (the conversation's current project), never
        # from keywords alone.
        if perf["form"] == "imperative" and not (perf["negated"] or perf["past"]):
            existing_evidence = self._existing_artifact_evidence(
                text, concepts)
            if existing_evidence:
                goal.goal_type = GoalType.MODIFY
                goal.action = "modify"
                goal.requested_output = "source_code"
                goal.evidence = existing_evidence
                goal.target = self._extract_target(text)
                goal.algorithm_hint = self._extract_algorithm_hint(text)
                goal.confidence = 0.7
                return goal

        # ── Step 0c: Computer-control semantics ──
        # Desktop manipulation is semantically distinct from code
        # generation: it describes acting ON the computer.  The ACT
        # still decides the turn before the tool does: when the
        # utterance asks to be TOLD something (an explanation, a guide,
        # an answer), the computer verb and the computer object it
        # mentions are its SUBJECT — the content it wants written about
        # — and never an act on the machine.  The same gate the
        # composition's artifact path already uses is what keeps a long
        # knowledge request from acquiring a desktop goal merely
        # because its subject is computers (the StructuredRequest is
        # built first; the coding/tool judgment follows it).
        computer = self._detect_computer_semantics(text, text_lower)
        if computer is not None:
            try:
                _asks_for_knowledge, _content_why = asks_for_content(
                    perf["act"], request=text)
            except Exception:
                _asks_for_knowledge, _content_why = False, ""
            if not _asks_for_knowledge:
                computer.target = self._extract_target(text)
                computer.evidence = list(computer.evidence or []) + [
                    _content_why or "computer-control semantics"]
                return computer

        # ── Step 1: Explicit User Semantics (the composer's decomposition)
        # The UserSemantics' cues/primitives ARE the meaning observed in
        # the user's text.  Literal values inside an OUTPUT operation
        # (e.g. "Hello", "Apple") are data, not goal-determining.
        composition_ops = []
        composer_result = None
        try:
            composer = self._get_composer()
            composer_result = composer.compose(text)
            composition_ops = list(composer_result.derived.graph.operations)
        except Exception:
            composer_result = None

        if composition_ops and not (perf["negated"] or perf["past"]):
            allowed, why = allows_artifact_creation(perf["act"],
                                                    request=text)
            if not allowed and perf["form"] == "interrogative":
                # Step 0b already verified: an interrogative that reaches
                # here carries a REAL computation structure over data
                # (an optimization objective with data entities, or a
                # strong algorithmic primitive with data entities).  The
                # query then asks for a VALUE that code computes.
                allowed = True
                why = "query act carrying computation structure over data"
            if allowed:
                self._bind_structured_goal(goal, text, lang, concepts,
                                           perf, composer_result,
                                           computation=True)
                goal.evidence.append(why)
                return goal
            # The composition carries a computation structure, but the
            # utterance does not ASK for an artifact: the structure is
            # the topic's vocabulary (a mention), not a creation request.
            # The dialogue act decides; the topic never does.

        # ── Step 2: Validated Semantic Inference — concept-activated
        # computation structure ──
        # An INTERROGATIVE form asking about a meaning ("What is X?",
        # "X가 뭐야?") is an information request, not a computation: the
        # weak computation concepts its nouns trigger are the TOPIC's
        # vocabulary, not a requested computation.  Only a non-
        # interrogative sentence (or one carrying a real computation
        # structure — an objective/substructure/aggregate) realizes the
        # computation concepts as the goal.
        has_algorithmic_primitive = any(
            getattr(c, 'category', '') == 'algorithmic_primitive'
            and getattr(c, 'activation_strength', 0) >= 4.0
            for c in (concepts or []))
        if perf["form"] == "interrogative" and not has_algorithmic_primitive:
            pass  # the information request falls through to the dialogue act
        elif self._has_detected_operations(concepts) or has_algorithmic_primitive:
            if not (perf["negated"] or perf["past"]):
                allowed, why = allows_artifact_creation(perf["act"],
                                                        request=text)
                if not allowed and perf["form"] == "interrogative":
                    allowed = True
                    why = "query act carrying computation structure over data"
                elif not allowed and not getattr(
                        perf["act"], "predicate_class", "") \
                        and self._nominal_artifact_kind(concepts):
                    # An ELLIPTICAL ARTIFACT NOMINAL: a bare noun phrase
                    # ("list maximum function") predicated of nothing —
                    # no verb, no assertion. In a construction dialog a
                    # predicate-less phrase that names BOTH computation
                    # structure (the operations gate above) and an
                    # artifact kind is the user handing over a SPEC, not
                    # mentioning a topic: the request act is implied by
                    # the utterance shape itself.
                    allowed = True
                    why = ("elliptical artifact nominal: no predicate, "
                           "computation over a named "
                           + self._nominal_artifact_kind(concepts))
                if allowed:
                    self._bind_structured_goal(goal, text, lang, concepts,
                                               perf, None, computation=True)
                    goal.evidence.append(why)
                    return goal

        # ── Step 5: Legacy mapping — recorded, never authoritative ──
        # A keyword occurrence is a LegacyCandidate: recorded with its
        # evidence, verified against the semantics, never deciding the
        # goal type alone.
        legacy_candidates = []
        for pattern, gtype in self.compiled_goals:
            m = pattern.search(text)
            if m:
                legacy_candidates.append({
                    "goal_type": gtype.value,
                    "matched_text": m.group(0)[:40],
                })

        # ── Step 6: The dialogue act resolution ──
        # The sentence's MEANING structure decides:
        # - no MEANINGFUL semantic content is a social interaction: a
        #   literal mention ("hello" is a tag of the Literal Value data
        #   concept) and weak shell noise are vocabulary echoes, not a
        #   computation request
        # - a performative request (the imperative form + the target)
        #   is a generation request — verified against the artifact kind
        # - anything else stays UNKNOWN/AMBIGUOUS with candidates
        if goal.goal_type == GoalType.UNKNOWN:
            meaningful_concepts = [c for c in (concepts or [])
                                   if getattr(c, 'concept_id', '') != 'literal_value'
                                   and getattr(c, 'activation_strength', 0) >= 2.5]
            has_any_content = bool(meaningful_concepts) or \
                bool(self._extract_entities(text, lang, goal, concepts))
            if not has_any_content:
                # No meaningful semantic content: a social interaction,
                # not a computation request.
                goal.goal_type = GoalType.CHAT
                goal.action = "chat"
            elif perf["form"] == "imperative" and not (perf["negated"] or perf["past"]):
                # The imperative performative + a target: a generation
                # request — the artifact kind is verified from the
                # semantic evidence (the entity/concept kinds), never
                # from surface keywords alone.
                self._bind_structured_goal(goal, text, lang, concepts,
                                           perf, None, computation=False)
            elif perf["form"] == "interrogative":
                # An interrogative that reached here (no computation
                # structure): an information request — the conversation
                # knowledge engine handles the retrieval + composition.
                goal.goal_type = GoalType.QUESTION
                goal.action = "query"
            elif all(getattr(c, 'category', '') in
                     ('programming_language', 'language_fundamental',
                      'capability', 'language_feature', 'pattern')
                     for c in meaningful_concepts) and meaningful_concepts:
                # A declarative whose ONLY strong concepts are knowledge
                # concepts (a language, a construct, a capability) — the
                # sentence discusses knowledge topics, not a computation
                # request: route to the conversation engine (an
                # information/comparison/explanation request composed
                # from the knowledge evidence).
                goal.goal_type = GoalType.QUESTION
                goal.action = "query"
                goal.evidence = ["knowledge-concept declarative"] + \
                    [c.concept_id for c in meaningful_concepts[:3]]
            else:
                self._bind_open_goal(goal, legacy_candidates,
                                     concepts, perf)

        goal.parent_context = {}
        goal.domain = "unknown"
        goal.target = goal.target or self._extract_target(text)
        goal.algorithm_hint = self._extract_algorithm_hint(text)

        concept_support = sum(c.activation_strength for c in concepts) / max(len(concepts), 1)
        if goal.confidence == 0.0:
            pattern_matched = goal.goal_type != GoalType.UNKNOWN
            goal.confidence = (0.7 if pattern_matched else 0.5) + 0.3 * min(concept_support, 1.0)

        return goal

    # ------------------------------------------------------------------
    # Performative analysis (grammatical meaning structure)
    # ------------------------------------------------------------------

    def _reconcile_goal_with_intent(self, semantic: SemanticRepresentation,
                                    intent) -> None:
        """Keep the goal type consistent with the derived intent.

        Detection order: semantic intent, then goal, then topic.  Only a
        WEAK goal (unknown, ambiguous or a bare "chat" with no dialogue
        content) is bound: the dialogue act is the authority on what the
        utterance IS, so a query act over a knowledge topic is a QUESTION
        even when no concept cleared the activation threshold, and a
        procedural or research act is the knowledge answer it asks for.
        A strong goal derived from the composition is never overwritten.

        An intent type this table does not claim leaves the goal OPEN —
        that is not a failure to resolve it but the honest reading that
        this act stated no goal claim; the turn stays open for the
        handler, and only a genuinely undecided act (CLARIFICATION above)
        asks the user back.
        """
        from brain.dialogue_act import IntentType
        goal = semantic.goal
        weak = goal.goal_type in (GoalType.UNKNOWN, GoalType.AMBIGUOUS,
                                  GoalType.CHAT)
        if not weak:
            return
        t = intent.type
        if t is IntentType.EXPLANATION_REQUEST:
            goal.goal_type = GoalType.EXPLAIN
            goal.action = "explain"
            goal.requested_output = "explanation"
        elif t is IntentType.COMMAND_QUERY:
            goal.goal_type = GoalType.COMMAND_QUERY
            goal.action = "query"
            goal.requested_output = "shell_command"
        elif t in (IntentType.INFORMATION_QUERY,
                   IntentType.CAPABILITY_QUERY,
                   IntentType.LANGUAGE_CAPABILITY_QUERY,
                   IntentType.RECOMMENDATION_REQUEST,
                   IntentType.ANALYSIS):
            goal.goal_type = GoalType.QUESTION
            goal.action = "query"
            if not goal.requested_output:
                goal.requested_output = "answer"
        elif t in (IntentType.GREETING, IntentType.CONVERSATION):
            goal.goal_type = GoalType.CHAT
            goal.action = "chat"
        elif t is IntentType.REPAIR_REQUEST:
            goal.goal_type = GoalType.FIX
            goal.action = "repair"
            goal.requested_output = "diagnosis"
        elif t is IntentType.PROCEDURAL_INSTRUCTION:
            # A request to be told how something is done is an ordered
            # knowledge answer, not a program to emit: the request family
            # the act itself groups this under is the query family.
            goal.goal_type = GoalType.QUESTION
            goal.action = "query"
            if not goal.requested_output:
                goal.requested_output = "answer"
        elif t is IntentType.RESEARCH:
            goal.goal_type = GoalType.RESEARCH
            goal.action = "research"
            if not goal.requested_output:
                goal.requested_output = "research_report"
        elif t is IntentType.SIMPLIFY_EXPLANATION:
            # Same claim as its request family, restated more simply.
            goal.goal_type = GoalType.EXPLAIN
            goal.action = "explain"
            goal.requested_output = "explanation"
        elif t is IntentType.CLARIFICATION:
            # The act read the utterance and found the request's OBJECT
            # missing: the goal stays open (unknown) — the pipeline asks
            # for the missing piece instead of guessing a goal type.
            goal.goal_type = GoalType.UNKNOWN
            goal.action = "clarify"
            goal.requested_output = "clarification"
        else:
            return
        goal.confidence = max(goal.confidence, intent.confidence or 0.6)
        goal.evidence = list(goal.evidence) + [
            "dialogue act: " + (intent.evidence[0] if intent.evidence else t.value)
        ]
        if intent.topics and not goal.target:
            goal.target = intent.topics[0]

    # What the goal inference and the act reading AGREE the utterance
    # is: one family per claim about the turn.  The families pair a
    # goal type with the dialogue-intent types that state the same
    # claim, so the two readings can be compared without either one
    # borrowing the other's vocabulary.
    _AGREEING_FAMILIES: dict[str, tuple[str, ...]] = {
        "chat": ("greeting", "conversation"),
        "question": ("information_query", "explanation_request",
                     "simplify_explanation", "capability_query",
                     "language_capability_query", "recommendation_request",
                     "procedural_instruction", "command_query", "research",
                     "analysis"),
        "information_query": ("information_query", "explanation_request",
                              "simplify_explanation", "research",
                              "analysis"),
        "command_query": ("command_query", "procedural_instruction"),
        "research": ("research",),
        "explain": ("explanation_request", "simplify_explanation",
                    "information_query"),
        "create_function": ("code_generation",),
        "create_class": ("code_generation",),
        "create_project": ("project_generation",),
        "modify": ("code_modification",),
        "fix": ("repair_request",),
        "analyze": ("analysis", "information_query",
                    "explanation_request"),
        "unknown": ("clarification",),
        "ambiguous": ("clarification",),
    }

    def _consensus_confidence(self, goal, intent) -> float | None:
        """The confidence the two readings of the utterance support.

        Returns None when the goal type states no claim this method can
        compare (there is no agreement to report), so the parse keeps
        the confidence its content coverage established.
        """
        family = self._AGREEING_FAMILIES.get(
            getattr(getattr(goal, "goal_type", None), "value", ""))
        if family is None:
            return None
        goal_conf = min(max(float(getattr(goal, "confidence", 0.0) or 0.0),
                            0.0), 1.0)
        act_conf = min(max(float(getattr(intent, "confidence", 0.0) or 0.0),
                           0.0), 1.0)
        if getattr(intent, "type", None) is None:
            return None
        if intent.type.value in family:
            # Independent evidence for the same claim: the chance that
            # BOTH readings err in the same direction is the product of
            # their individual chances, so agreeing readings rule out
            # more of the error space than either alone.
            both_right = goal_conf * act_conf
            both_wrong = (1.0 - goal_conf) * (1.0 - act_conf)
            total = both_right + both_wrong
            if total <= 0.0:
                return None
            # Agreement is strong evidence, never certainty: the two
            # readings are not independent of the sentence they read.
            return min(0.95, both_right / total)
        # The readings disagree: the utterance admits two readings, so
        # the report cannot claim more than the weaker one supports.
        return min(goal_conf, act_conf)

    def _analyze_performative(self, text: str) -> dict:
        """The sentence's MEANING structure: form, negation, tense.

        Grammatical morphology, not a keyword list:
          - form: an interrogative (the "?" punctuation or a WH-word
            initial) / an imperative (the sentence requests an action) /
            a declarative (the sentence states or reports)
          - negated: the main predicate is negated (a refusal/negation —
            the sentence does NOT request the action)
          - tense: a past-tense report states that something HAPPENED —
            it does not request it
          - method_question: a "how to" interrogative asks for a METHOD
        """
        import re as _re
        from brain.dialogue_act import analyze_act
        text_strip = text.strip()
        lower = text_strip.lower()
        # The illocutionary structure: the speech-act layer's analysis of
        # the sentence (form, force, predicate class, subject person).
        # The topic words never contribute to it.
        act = analyze_act(text_strip)

        form = "declarative"
        method_question = False
        if "?" in text_strip or "？" in text_strip:
            form = "interrogative"
            method_question = bool(_re.match(
                r"^(how\b|어떻게)", lower))
        elif _re.match(r"^(what|why|when|where|who|which|whom|whose|how)\b",
                       lower):
            form = "interrogative"
        elif _re.match(
                r"^(please\s+|plz\s+|pls\s+)?"
                r"(만들|생성|작성|구현|짜|"
                r"make|create|build|write|generate|implement|add|fix|delete|remove|"
                r"check|flip|reverse|sort|filter|count|rotate|convert|keep|drop|"
                r"sum|find|compute|calculate|return|extract|merge|split|join)\b",
                lower):
            # An imperative (optionally with the English polite marker
            # "please make..." — the polite marker is grammar, not a
            # statement form change)
            form = "imperative"
        elif lower.endswith(("해줘", "주세요", "줘", "십시오", "바랍니다",
                             "하고 싶어", "하고 싶다", "필요해", "필요하다",
                             "please")):
            form = "imperative"

        negated = bool(_re.search(
            r"(don't|doesn't|do\s+not|didn't|not\s+to|never|"
            r"않다|않는|않어|않네|싶지\s*않|원하지\s*않|못\s*하)",
            lower))
        past = bool(_re.search(
            r"\b(was|were|did|had)\b|"
            r"했다|였다|됐다|됬다|생성됐|만들어졌|생성되었|만들었",
            lower))
        # An explanation request: the sentence asks for the MEANING of a
        # concept ("Explain X", "X가 뭐야", "X를 설명해줘") — the
        # performative verb phrase decides, not a keyword list.  The
        # "what is X" interrogative is an information request (handled
        # by the interrogative form), NOT an explanation act.  The same
        # holds for the EMBEDDED question — the sentence carries an
        # interrogative content clause ("tell me what recursion is",
        # "재귀가 뭔지 알려줘"): the utterance requests knowledge about a
        # topic, so the topic's own algorithmic structure is what the
        # request is ABOUT, never a generation request.
        #   - the same act seen through a METHOD question: "How do you
        #     explain X?" asks how to EXPLAIN — the activity the method
        #     is about IS the explanatory act, so nothing procedural is
        #     being requested.  (A method question about producing
        #     language output in a target language — "how do you say X in
        #     Python" — names a different activity and stays a method.)
        explanation_request = bool(_re.match(
            r"^(explain|describe|tell\s+me\s+about)\b", lower)
            or _re.search(r"(설명해|설명\s*해|뭐야|뭐냐|무엇인|어떤\s+의미)"
                          r"|(?:무엇|뭐|뭔)(?:인)?지"
                          r"|\b(?:tell\s+me|want\s+to\s+know|need\s+to\s+know"
                          r"|do\s+not\s+know|don'?t\s+know)\s+"
                          r"(?:what|why|how|when|where|which)\b"
                          r"|\bhow\s+(?:(?:do|does|did|can|could|would|should"
                          r"|will|may|might)\s+)?(?:(?:i|we|you|he|she|they|it)"
                          r"\s+)?(?:explain|describe|clarify|summar"
                          r"|rephrase|paraphrase|interpret)\b"
                          r"|(?:어떻게|how)\s+\S{0,6}(?:설명|설명해|표현|이해시키)",
                          lower))

        return {"form": form, "negated": negated, "past": past,
                "method_question": method_question,
                "explanation_request": explanation_request,
                # The illocutionary structure (from the dialogue-act
                # layer): what the speaker IS DOING, independent of the
                # topic words.
                "force": act.force,
                "predicate_class": act.predicate_class,
                "social_act": act.social_act,
                "subject": act.subject,
                "desiderative": act.desiderative,
                "act": act}

    def _existing_artifact_evidence(self, text: str,
                                    concepts: list[ActivatedConcept]) -> list:
        """The existing-artifact evidence: the sentence's action applies
        to an ALREADY-EXISTING artifact.

        Semantic evidence (any one source):
          - the conversation context: a current project is set (the
            parent context — wired at parse time via ParseContext)
          - an explicit existing-artifact reference in the entity
            structure ("the existing application" — the entity carries
            an existing/reference role)
          - the modification action + an artifact-known context: the
            sentence's action (add/remove/update) over an artifact the
            knowledge base recognizes (a Software Project / Class
            Definition concept)

        Returns the evidence list (empty = no existing-artifact evidence).
        """
        import re as _re
        lower = text.lower()
        evidence = []

        # The action: the modification verbs (the grammatical performative)
        has_mod_action = bool(_re.search(MODIFICATION_ACTION_PATTERN, lower))
        # "add up" is the arithmetic phrasal verb (a sum), not the
        # modification predicate "add X to the existing artifact".
        if _re.search(r"\badd(?:s|ed|ing)?\s+up\b", lower):
            has_mod_action = False
        if not has_mod_action:
            return []

        # 1. The parent context: a current project is set
        if self._parse_context is not None and \
                getattr(self._parse_context, 'current_project', None):
            evidence.append("parent context: current project set")
        # 2. An explicit existing-artifact reference
        if _re.search(r"(기존의?\s|existing\b|already\s+(built|made|created)|"
                      r"있는\s+(앱|프로젝트|프로그램))", lower):
            evidence.append("explicit existing-artifact reference")
        # 3. An artifact-known context: the knowledge base recognizes the
        #    artifact the action applies to
        for c in (concepts or []):
            if getattr(c, 'concept_id', '') in ('artifact_project', 'class_def') \
                    and getattr(c, 'activation_strength', 0) >= 4.0:
                evidence.append("artifact knowledge: " + c.concept_id)
                break
        # 4. The conversation's CODING STATE: a previous turn generated
        #    an artifact — the action applies to THAT artifact (the
        #    parent context the goal inference reads from the discourse,
        #    never from the sentence's words).
        try:
            history = getattr(self._parse_context, "conversation_history",
                              None) or []
            for turn in list(history)[-3:]:
                art = getattr(turn, "artifact", None) or {}
                if isinstance(art, dict) and art.get("code"):
                    evidence.append(
                        "parent context: previous code artifact in "
                        "conversation")
                    break
        except Exception:
            pass

        # 5. The ACTION's own presupposition: an addition predicate
        #    cannot be performed on nothing — "add X" says the host that
        #    X is added to exists.  When the sentence carries no
        #    creation performative, the host is not being introduced
        #    here, so the verb's argument structure — never the topic
        #    words — is what names the existing artifact.
        if not _re.search(
                r"( 만들어| 만들|생성|작성|구현|\bnew\s+|"
                r"\b(?:create|make|build|write|generate|implement)\b)",
                lower):
            evidence.append(
                "the modification action presupposes an existing artifact")

        return evidence

    def _shell_semantics_present(self, concepts: list[ActivatedConcept]) -> bool:
        """Shell/platform semantics are activated (the command query's
        topic is a shell/platform environment)."""
        return any(getattr(c, 'category', '') in
                   ("shell_platform", "shell_language", "shell_operation")
                   and getattr(c, 'activation_strength', 0) >= 2.0
                   for c in (concepts or []))

    # ------------------------------------------------------------------
    # Structured goal binding (the semantic evidence → the goal object)
    # ------------------------------------------------------------------

    def _nominal_artifact_kind(self, concepts: list) -> str:
        """The artifact kind a predicate-less nominal names — from the
        activated artifact-kind concepts (the same ontology thresholds
        _bind_structured_goal uses), never from surface keywords."""
        concept_kinds = {
            getattr(c, 'concept_id', ''): getattr(c, 'activation_strength', 0)
            for c in (concepts or [])
        }
        if concept_kinds.get('artifact_project', 0) >= 4.0:
            return "project"
        if concept_kinds.get('class_def', 0) >= 4.0:
            return "class"
        if concept_kinds.get('func_def', 0) >= 4.0:
            return "function"
        return ""

    def _bind_structured_goal(self, goal: GoalSpec, text: str, lang: Language,
                              concepts: list[ActivatedConcept], perf: dict,
                              composer_result, computation: bool) -> None:
        """Bind the structured goal fields from the semantic evidence.

        The artifact kind is decided from the semantic evidence (the
        entity/concept kinds and the composer's semantics), never from
        surface keywords alone.
        """
        import re as _re
        target_text = self._extract_target(text)
        evidence = []

        # The artifact kind from the semantic evidence:
        # 1. The retrieved knowledge: the concept store's artifact-kind
        #    concepts (Software Project / Class Definition / Function
        #    Definition) activated in the user's text — the knowledge
        #    base's ontology decides, not surface keywords.
        # 2. The entity kinds ("Student" data, "Order" data...) carry the
        #    domain scope.
        artifact_kind = ""
        domain_scope = ""
        concept_kinds = {}
        for c in (concepts or []):
            concept_kinds[getattr(c, 'concept_id', '')] = (
                getattr(c, 'category', ''),
                getattr(c, 'activation_strength', 0))
        if 'artifact_project' in concept_kinds and \
                concept_kinds['artifact_project'][1] >= 4.0:
            artifact_kind = "project"
            evidence.append("retrieved knowledge: Software Project concept")
        elif 'class_def' in concept_kinds and \
                concept_kinds['class_def'][1] >= 4.0:
            artifact_kind = "class"
            evidence.append("retrieved knowledge: Class Definition concept")
        elif 'func_def' in concept_kinds and \
                concept_kinds['func_def'][1] >= 4.0:
            artifact_kind = "function"
            evidence.append("retrieved knowledge: Function Definition concept")
        if composer_result is not None and hasattr(composer_result, 'user'):
            user = composer_result.user
            if user.cues:
                evidence.append("composer decomposition: " +
                                ", ".join(sorted({c.primitive for c in user.cues})[:5]))
        entity_kinds = [getattr(e, 'kind', '') for e in
                        self._extract_entities(text, lang, goal, concepts)]
        entity_names = [getattr(e, 'name', '') for e in
                        self._extract_entities(text, lang, goal, concepts)]
        if entity_names:
            evidence.append("entities: " + ", ".join(entity_names[:4]))

        # A project artifact describes a WHOLE system (a multi-part
        # scope); a class artifact describes a TYPE with properties; a
        # function artifact describes a single computation over data.
        app_scope_evidence = any(
            kind in ("application", "project", "service", "site")
            for kind in entity_kinds)
        if app_scope_evidence:
            artifact_kind = "project"
        if not artifact_kind:
            artifact_kind = "function"
        # A computation structure over data describes the FUNCTION that
        # computes it.  When the class concept fired but the request
        # act's own object does not name a class, the concept is the
        # TOPIC's format vocabulary ("class Solution:" starter code in
        # a problem statement) — the act's object decides, never the
        # concept's activation alone.
        if computation and artifact_kind == "class" and not _re.search(
                r"\bclass\b|\bstruct\b|클래스", target_text):
            artifact_kind = "function"
            evidence.append(
                "computation artifact: function "
                "(class concept is topic vocabulary)")
        domain_scope = ", ".join(entity_names[:3]) if entity_names else ""

        concept_categories = {getattr(c, 'category', '') for c in (concepts or [])}
        if 'operation' in concept_categories or 'algorithmic_primitive' in concept_categories:
            evidence.append("computation concepts activated")

        goal.goal_type = {"project": GoalType.CREATE_PROJECT,
                          "class": GoalType.CREATE_CLASS}.get(
                              artifact_kind, GoalType.CREATE_FUNCTION)
        goal.action = "create"
        goal.target = target_text
        goal.artifact = {
            "kind": artifact_kind,
            "target": target_text,
            "behavior": goal.algorithm_hint or "",
            "domain": domain_scope,
        }
        goal.requested_output = "source_code"
        goal.evidence = evidence
        goal.confidence = 0.8 if computation and evidence else 0.65
        if perf.get("form") == "imperative":
            goal.confidence = max(goal.confidence, 0.7)

    def _bind_open_goal(self, goal: GoalSpec, legacy_candidates: list,
                        concepts: list[ActivatedConcept], perf: dict) -> None:
        """The sentence's meaning structure claims no goal type, so the
        goal stays OPEN for the dialogue intent to decide it.

        The detection order is intent first, then goal, then topic: the
        concepts that activated are what the request TALKS ABOUT, and a
        knowledge concept (a computation primitive, a shell verb, an
        ontology entry) is an evidence candidate for the knowledge the
        turn may need — never a competing reading of what the user ASKS
        for.  Turning those categories into goal alternatives would let
        the concept store answer the question the act has to answer, so
        they are recorded here as topic candidates only.  The legacy
        occurrences this sentence carries are recorded the same way:
        evidence of the readings, deciding nothing."""
        evidence = ["unclear meaning structure: the dialogue intent "
                    "decides the goal"]
        evidence += [lc.get("matched_text", "")
                     for lc in (legacy_candidates or [])[:2]]
        topics = [getattr(c, "concept_id", "") for c in (concepts or [])
                  if getattr(c, "concept_id", "")]
        if topics:
            evidence.append("topic candidates: " + ", ".join(topics[:5]))
        goal.goal_type = GoalType.UNKNOWN
        goal.status = "open"
        goal.alternatives = []
        goal.evidence = evidence

    def _infer_domain(self, entities: list[EntityRef], operation_graph: OperationGraph, 
                     constraints: list[ConstraintSpec], concepts: list[ActivatedConcept], 
                     text: str) -> tuple[str, dict]:
        """
        Infer domain from semantic evidence, not keyword matching.
        Returns (domain, evidence_dict)
        """
        evidence = {
            "entities": [e.name for e in entities],
            "operations": [op.name for op in operation_graph.operations],
            "constraints": [c.name for c in constraints],
            "concepts": [(c.name, c.category, c.activation_strength) for c in concepts[:10]],
            "text_length": len(text)
        }
        
        # Collect entity kinds from user text
        entity_kinds = set()
        entity_names = set()
        for e in entities:
            entity_kinds.add(e.kind)
            entity_names.add(e.name.lower())
        
        # Collect operation types
        operation_names = set()
        for op in operation_graph.operations:
            operation_names.add(op.name)
        
        # Domain inference from semantic evidence
        domain_scores = {}
        
        # 1. Score from entity names (data types mentioned by user)
        domain_entity_map = {
            "student": "education",
            "학생": "education",
            "grade": "education",
            "점수": "education",
            "score": "education",
            "employee": "hr",
            "직원": "hr",
            "salary": "hr",
            "급여": "hr",
            "product": "ecommerce",
            "상품": "ecommerce",
            "order": "ecommerce",
            "주문": "ecommerce",
            "shopping": "ecommerce",
            "쇼핑": "ecommerce",
            "cart": "ecommerce",
            "장바구니": "ecommerce",
            "camera": "camera",
            "카메라": "camera",
            "photo": "camera",
            "사진": "camera",
            "image": "camera",
            "이미지": "camera",
            "video": "camera",
            "영상": "camera",
            "android": "android",
            "안드로이드": "android",
            "ios": "ios",
            "아이폰": "ios",
            "iphone": "ios",
            "web": "web",
            "웹": "web",
            "website": "web",
            "사이트": "web",
            "blog": "blog",
            "블로그": "blog",
            "post": "blog",
            "게시물": "blog",
            "chat": "messaging",
            "채팅": "messaging",
            "message": "messaging",
            "메시지": "messaging",
            "messenger": "messaging",
            "메신저": "messaging",
            "todo": "productivity",
            "할일": "productivity",
            "task": "productivity",
            "태스크": "productivity",
            "schedule": "productivity",
            "일정": "productivity",
            "calendar": "productivity",
            "캘린더": "productivity",
            "file": "file_system",
            "파일": "file_system",
            "folder": "file_system",
            "폴더": "file_system",
            "directory": "file_system",
            "디렉토리": "file_system",
            "calculator": "calculator",
            "계산기": "calculator",
            "calc": "calculator",
            "game": "game",
            "게임": "game",
            "book": "library",
            "책": "library",
            "library": "library",
            "도서관": "library",
            "movie": "media",
            "영화": "media",
            "music": "media",
            "음악": "media",
            "player": "media",
            "플레이어": "media",
        }
        
        for entity_name in entity_names:
            if entity_name in domain_entity_map:
                domain = domain_entity_map[entity_name]
                domain_scores[domain] = domain_scores.get(domain, 0) + 2.0
                evidence[f"entity_match:{entity_name}"] = domain
        
        # 2. Score from operation patterns
        if operation_names & {"calculate_average", "sum_list", "find_max", "find_min", "sort_descending", "sort_ascending"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 1.5
            evidence["operations:aggregate_sort"] = "data_processing"
        
        if operation_names & {"filter_above", "filter_below", "filter_even", "filter_odd"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 1.0
            evidence["operations:filter"] = "data_processing"
        
        if operation_names & {"remove_duplicates"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 0.5
            evidence["operations:dedup"] = "data_processing"
        
        if operation_names & {"is_prime", "factorial", "fibonacci", "binary_search"}:
            domain_scores["algorithm"] = domain_scores.get("algorithm", 0) + 1.5
            evidence["operations:algorithm"] = "algorithm"
        
        if operation_names & {"reverse_string", "is_palindrome", "count_characters", "count_words"}:
            domain_scores["string_processing"] = domain_scores.get("string_processing", 0) + 1.5
            evidence["operations:string"] = "string_processing"
        
        # 3. Score from constraints (platform, language)
        for c in constraints:
            if c.constraint_type == "platform":
                if c.value in ("android", "ios", "mobile"):
                    domain_scores["mobile"] = domain_scores.get("mobile", 0) + 2.0
                    evidence[f"constraint:platform:{c.value}"] = "mobile"
                elif c.value in ("web", "browser"):
                    domain_scores["web"] = domain_scores.get("web", 0) + 2.0
                    evidence[f"constraint:platform:{c.value}"] = "web"
                elif c.value == "cli":
                    domain_scores["cli"] = domain_scores.get("cli", 0) + 1.0
                    evidence[f"constraint:platform:{c.value}"] = "cli"
            elif c.constraint_type == "language":
                evidence[f"constraint:language:{c.value}"] = "language"
        
        # 4. Score from knowledge graph concepts (only if they're relevant to user text)
        for c in concepts:
            if c.category == "domain" and c.activation_strength > 0.5:
                # Only use if concept name appears in text or relates to entities
                if c.name.lower() in text.lower() or any(c.name.lower() in en for en in entity_names):
                    domain_scores[c.name] = domain_scores.get(c.name, 0) + c.activation_strength
                    evidence[f"concept:{c.name}"] = c.name
            elif c.category == "capability" and c.activation_strength > 1.0:
                # Capabilities like "Camera", "Authentication" are hints but not definitive
                cap_domain = c.name.replace("cap_", "")
                if cap_domain in ["camera", "auth", "network", "ui", "storage"]:
                    domain_scores[cap_domain] = domain_scores.get(cap_domain, 0) + c.activation_strength * 0.5
                    evidence[f"capability:{c.name}"] = cap_domain
        
        # 5. Default fallback based on goal type (only if no evidence found)
        if not domain_scores:
            # No specific evidence - use general
            return "general", {**evidence, "fallback": "no_evidence"}
        
        # Get top domain
        sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
        top_domain, top_score = sorted_domains[0]
        
        # Require minimum confidence
        if top_score < 1.0:
            return "general", {**evidence, "fallback": "low_confidence", "scores": sorted_domains[:3]}
        
        evidence["scores"] = sorted_domains[:3]
        evidence["selected"] = {"domain": top_domain, "confidence": top_score}
        
        return top_domain, evidence
    
    def _strip_response_shape(self, topic: str) -> str:
        """Remove an answer-shaping request from the topic.

        "Explain list comprehensions in one sentence" has the topic
        "list comprehensions"; "in one sentence" is a constraint on the
        answer, not part of what is being explained.
        """
        cleaned = re.sub(
            r"^(?:please\s+)?(?:explain|describe|tell\s+me\s+about)\s+",
            "", topic, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"[\s,;:-]*(?:in\s+(?:exactly\s+)?(?:\d+|one|two|three|four|"
            r"five)\s+sentences?|in\s+plain\s+language|"
            r"in\s+simple\s+terms|simply|briefly|concisely|"
            r"한\s*문장으로|간단히|쉽게)\s*$",
            "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned.rstrip("?？.").strip()

    def _extract_target(self, text: str) -> str:
        """Extract the utterance's REFERENT.

        For a creation request the referent is WHAT to create: the
        performative formula (request verbs and polite endings) is
        grammar, not content, and is stripped by general morphology
        rules for both languages — never a fixed sentence match.

        For every other act (a query, a report, a social remark) the
        referent is the TOPIC, read from the dialogue-act layer's
        focused constituent: the sentence itself is never the topic.
        """
        from brain.dialogue_act import (
            analyze_act, extract_topics, allows_artifact_creation,
        )
        act = analyze_act(text)
        allowed, _ = allows_artifact_creation(act)
        if not allowed:
            topics = extract_topics(text)
            if topics:
                joined = " and ".join(topics) if len(topics) >= 2 \
                    else topics[0]
                return self._strip_response_shape(joined)[:100]
        cleaned = text.strip()
        # Leading WH-question formula: "What is X?" / "How do I X?" —
        # the question formula is grammar, the content is X.
        #   "What is Python?" → "Python"
        #   "Can you speak English?" → "speak English"
        cleaned = re.sub(
            r"^(what\s+is|what's|what\s+are|how\s+do\s+i|how\s+to|"
            r"can\s+you|explain)\s+",
            "", cleaned, flags=re.IGNORECASE).strip()
        # Trailing request formula: Korean polite endings then the request
        # verb phrase then the object particle.
        #   "리스트 정렬 함수를 만들어줘" → "리스트 정렬 함수"
        #   "평균 계산 기능이 필요해" → "평균 계산 기능"
        cleaned = re.sub(
            r"(해\s*주\s*세요|해\s*줘|주\s*세요|주\s*십시오|어\s*줘|줘|"
            r"드릴까요|바랍니다|필요해|필요하다|하고\s*싶어|하고\s*싶다|"
            r"싶어|싶다)$",
            "", cleaned).strip()
        cleaned = re.sub(
            r"(만들어|생성해|작성해|구현해|짜서|짜|만들|생성|작성|구현)$",
            "", cleaned).strip()
        cleaned = re.sub(r"(을|를|이|가)$", "", cleaned).strip()
        # Leading request formula: English polite marker + request verb.
        #   "please make Android Camera App..." → "Android Camera App..."
        cleaned = re.sub(
            r"^(please|plz|pls)\s+"
            r"(make|create|build|write|generate|implement|develop)\s+"
            r"(a\s+|an\s+|the\s+)?",
            "", cleaned, flags=re.IGNORECASE).strip()
        # Trailing English request clause ("give me complete code") — a
        # performative formula, not part of the target.
        cleaned = re.sub(
            r"[,.\s]*(give\s+me|show\s+me)\s+.*$", "",
            cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"(function|클래스|class|프로젝트|project|앱|app|프로그램|program|"
            r"기능|코드|code)$",
            "", cleaned, flags=re.IGNORECASE).strip()
        # Trailing punctuation: the question mark is grammar
        cleaned = self._strip_response_shape(cleaned)
        return cleaned[:100]

    def _has_detected_operations(self, concepts: list[ActivatedConcept]) -> bool:
        """Did any OPERATION concept activate?

        The concept store's own ONTOLOGY decides (category == operation),
        not a name list: ActivatedConcept.name is the display name
        ("Find Maximum") and would never equal an id like "find_max" —
        the id-list version of this check could never fire, silently
        disabling every computation-driven goal path.
        """
        legacy_ids = {
            "find_max", "find_min", "sort", "filter",
            "is_multiple_of", "is_prime", "remove_duplicates",
            "calculate_average", "calculate_sum", "generate_range",
        }
        return any(
            getattr(c, 'category', '') == 'operation'
            or getattr(c, 'concept_id', '') in legacy_ids
            for c in concepts
        )
    
    def _extract_algorithm_hint(self, text: str) -> str | None:
        """Extract algorithm hint if present."""
        text_lower = text.lower()
        algorithms = {
            "binary_search": ["이진 탐색", "binary search", "이진 검색", "binarysearch"],
            "quick_sort": ["퀵 정렬", "quick sort", "quicksort"],
            "merge_sort": ["병합 정렬", "merge sort", "mergesort"],
            "factorial": ["팩토리얼", "factorial", "계승"],
            "fibonacci": ["피보나치", "fibonacci", "fib"],
            "prime_check": ["소수", "prime", "소수 판별"],
        }
        for algo, keywords in algorithms.items():
            if any(k in text_lower for k in keywords):
                return algo
        return None
    
    def _detect_competitive_patterns(self, text: str) -> dict:
        """Detect common competitive programming patterns in the problem statement.
        
        Returns a dictionary with detected patterns and their parameters.
        This helps the code generator understand what the problem requires.
        """
        text_lower = text.lower()
        patterns = {}
        
        # Detect input format patterns
        if re.search(r'(?:given|입력|input)\s+(?:an?\s+)?(?:integer|정수|number)', text_lower):
            patterns['single_integer_input'] = True
        if re.search(r'(?:given|입력|input)\s+(?:a\s+)?(?:string|문자열|문자)', text_lower):
            patterns['string_input'] = True
        if re.search(r'(?:given|입력|input)\s+(?:a\s+)?(?:list|리스트|배열|array|sequence|시퀀스)', text_lower):
            patterns['array_input'] = True
        
        # Detect output format patterns
        if re.search(r'(?:print|출력|output|return|반환)\s+(?:the|그|이|합|개수|값|결과)', text_lower):
            patterns['print_output'] = True
        if re.search(r'(?:print|출력|output)\s+(?:Yes|No|예|아니오)', text_lower):
            patterns['yes_no_output'] = True
        
        # Detect computation patterns
        if re.search(r'(?:sum|합|총합|합계)', text_lower):
            patterns['sum'] = True
        if re.search(r'(?:count|개수|몇|수를)', text_lower):
            patterns['count'] = True
        if re.search(r'(?:maximum|최대|가장\s+큰|가장\s+많은)', text_lower):
            patterns['maximum'] = True
        if re.search(r'(?:minimum|최소|가장\s+작은|가장\s+적은)', text_lower):
            patterns['minimum'] = True
        if re.search(r'(?:sort|정렬|오름차순|내림차순)', text_lower):
            patterns['sort'] = True
        if re.search(r'(?:filter|골라|선택|만\s+골라)', text_lower):
            patterns['filter'] = True
        
        # Detect constraint patterns
        if re.search(r'(?:modulo|mod|나머지|MOD)\s*(\d+)', text_lower):
            mod_match = re.search(r'(?:modulo|mod|나머지|MOD)\s*(\d+)', text_lower)
            patterns['modulo'] = int(mod_match.group(1))
        if re.search(r'(?:1\s*<=|≤)\s*(\w+)\s*(?:<=|≤)\s*(\d+)', text_lower):
            patterns['constraint_range'] = True
        
        # Detect problem structure
        if re.search(r'(?:sample\s+input|예시\s+입력|입력\s+예시)', text_lower):
            patterns['has_examples'] = True
        if re.search(r'(?:constraint|제약|조건|limit)', text_lower):
            patterns['has_constraints'] = True
        
        return patterns
    
    # Keywords that have multiple meanings and need context disambiguation
    _AMBIGUOUS_ENTITIES = {"order", "주문"}

    def _is_ambiguous_use(self, keyword: str, text_lower: str) -> bool:
        """Determine if the keyword occurrence is NOT an entity reference."""
        # "order" used as sort direction: descending/ascending/sorted/by/in/reverse order
        if keyword in ("order", "주문"):
            if re.search(r"(descend\w*|ascend\w*|sorted|sort|by|in|the|reverse|descending|ascending|increasing|decreasing|great\w*|larg\w*|bigg\w*|small\w*|first|last)\s+order\b", text_lower):
                return True
            # "order" after sort-related verbs
            if re.search(r"\b(sort|sorted|sorting|order|ordering|arrange|arranged|arranging)\b.*\border\b", text_lower):
                return True
            return False
        return False
    
    def _extract_entities(
        self, 
        text: str, 
        lang: Language, 
        goal: GoalSpec,
        concepts: list[ActivatedConcept]
    ) -> list[EntityRef]:
        """Extract entities from text and concepts."""
        entities = []
        text_lower = text.lower()
        seen = set()
        for keyword, (type_name, kind, attrs) in self.ENTITY_PATTERNS.items():
            # The script the keyword is written in decides where its word
            # ends.  A Latin keyword ends where the Latin letters end:
            # "Python" in "Python으로" is the whole word, because the
            # syllables after it belong to another script — a Unicode \b
            # would demand a non-word character there, and a Hangul
            # syllable IS a word character.  A keyword written without
            # spaces keeps the Unicode boundary, where the edge of a word
            # is what the script itself marks.
            if keyword.isascii():
                match = re.search(
                    r'(?<![A-Za-z0-9])' + re.escape(keyword) +
                    r'(?![A-Za-z0-9])', text_lower)
            else:
                match = re.search(r'\b' + re.escape(keyword) + r'\b',
                                  text_lower)
            if match and keyword not in seen:
                # Disambiguation: reject keywords whose occurrence here is a
                # different semantic role (e.g. "order" in "descending order").
                if keyword in self._AMBIGUOUS_ENTITIES and self._is_ambiguous_use(keyword, text_lower):
                    continue
                seen.add(keyword)
                entities.append(EntityRef(
                    name=type_name,
                    type_name=type_name,
                    kind=kind,
                    attributes=attrs,
                    is_input=True  # Default assumption
                ))
        
        # 2. From activated concepts (ONLY if the concept name actually appears in the user's text)
        # This prevents false entities from knowledge graph activation
        for c in concepts:
            if c.category in ("entity", "data", "object") and c.name not in seen:
                # Only add if the concept name actually appears in the user's text
                if re.search(r'\b' + re.escape(c.name.lower()) + r'\b', text_lower):
                    seen.add(c.name)
                    entities.append(EntityRef(
                        name=c.name,
                        type_name=c.name,
                        kind=c.category,
                        attributes={}
                    ))
        
        # 3. Extract type words from text that are valid types (not stop words)
        # This handles "list of numbers", "array of strings", etc.
        type_matches = re.findall(r'\b(list|array|리스트|배열|목록|sequence|시퀀스)\s+(?:of\s+)?(\w+)', text_lower)
        for container, elem_type in type_matches:
            if elem_type in self.STOP_WORDS:
                continue
            elem_type = self._canonical_type(elem_type)
            if not elem_type:
                continue
            type_name = elem_type.capitalize()
            if type_name not in seen:
                seen.add(type_name)
                entities.append(EntityRef(
                    name=type_name,
                    type_name=type_name,
                    kind="primitive",
                    attributes=[],
                    is_input=True
                ))
        
        # 4. Extract variable declarations from natural language:
        # "integers A and B", "integer X", "strings S and T", "numbers N, M"
        # These single-letter capitalized tokens are variable references.
        var_decl_matches = re.findall(
            r'\b(?:integers?|numbers?|values?|strings?|chars?|characters?)\s+'
            r'([A-Z])(?:\s*(?:,|and)\s*([A-Z]))?(?:\s*(?:,|and)\s*([A-Z]))?',
            text)
        for groups in var_decl_matches:
            for g in groups:
                if g and g not in seen:
                    seen.add(g)
                    entities.append(EntityRef(
                        name=g,
                        type_name=g,
                        kind="primitive",
                        attributes=[],
                        is_input=True
                    ))
        
        return entities
    
    def _preprocess_for_operations(self, text: str) -> str:
        """Remove goal-framing phrases and type specifications so they are
        not mistaken for operations (e.g. 'create a function', 'list of X')."""
        processed = text
        # Remove goal-framing prefixes/clauses
        framing = [
            r"create\s+a\s+function\s+that\s+",
            r"create\s+a\s+function\s+which\s+",
            r"create\s+a\s+function\s+to\s+",
            r"make\s+a\s+function\s+that\s+",
            r"make\s+a\s+function\s+which\s+",
            r"write\s+a\s+function\s+that\s+",
            r"write\s+a\s+function\s+which\s+",
            r"implement\s+a\s+function\s+that\s+",
            r"build\s+a\s+function\s+that\s+",
            r"define\s+a\s+function\s+that\s+",
            r"a\s+function\s+that\s+",
            r"a\s+function\s+which\s+",
            r"함수\s*를\s*만드는\s*",
            r"함수\s*를\s*작성하는\s*",
        ]
        for f in framing:
            processed = re.sub(f, " ", processed, flags=re.IGNORECASE)
        # Protect type specifications: 'list/array/sequence of X' and 'X list'
        # Replace them with a neutral token so 'list'/'array' aren't seen as ops
        processed = re.sub(r"(?:list|array|sequence|리스트|배열|시퀀스)\s+(?:of\s+)?\w+", " data ", processed, flags=re.IGNORECASE)
        processed = re.sub(r"\w+\s+(?:list|array|리스트|배열)", " data ", processed, flags=re.IGNORECASE)
        return processed

    def _extract_operations(
        self, 
        text: str, 
        lang: Language, 
        entities: list[EntityRef],
        goal: GoalSpec
    ) -> OperationGraph:
        """Extract ordered operations from text."""
        graph = OperationGraph()
        text_lower = text.lower()
        
        # Preprocess to remove goal framing and type specs
        working_text = self._preprocess_for_operations(text_lower)
        
        # Normalize text with synonyms for better pattern matching
        normalized_text = self._normalize_text(working_text)
        
        found_ops = []
        
        # Find all matching operations with positions
        for pattern, name, op_type, role in self.compiled_ops:
            # Try preprocessed original
            for match in pattern.finditer(working_text):
                found_ops.append({
                    "position": match.start(),
                    "end": match.end(),
                    "name": name,
                    "type": op_type,
                    "role": role,
                    "pattern": pattern.pattern
                })
            # Try normalized text (synonyms expanded)
            for match in pattern.finditer(normalized_text):
                if not any(op["name"] == name and abs(op["position"] - match.start()) < 5 for op in found_ops):
                    found_ops.append({
                        "position": match.start(),
                        "end": match.end(),
                        "name": name,
                        "type": op_type,
                        "role": role,
                        "pattern": pattern.pattern
                    })
        
        # Sort by position in text
        found_ops.sort(key=lambda x: x["position"])
        
        # Remove duplicates: merge same-name matches that overlap (keeping the
        # union span so the more specific whole-phrase match wins), and drop
        # matches fully contained within an already-kept match of the same name.
        unique_ops = []
        for op in found_ops:
            existing = None
            for u in unique_ops:
                if u["name"] == op["name"] and u["position"] < op["end"] and op["position"] < u["end"]:
                    existing = u
                    break
            if existing is not None:
                existing["position"] = min(existing["position"], op["position"])
                existing["end"] = max(existing["end"], op["end"])
                # Prefer the more specific (longer) original pattern description
                if (op["end"] - op["position"]) > (existing["end"] - existing["position"]):
                    existing["role"] = op["role"]
                continue
            near_dup = any(
                u["name"] == op["name"] and abs(u["position"] - op["position"]) < 5
                for u in unique_ops
            )
            if not near_dup:
                unique_ops.append(op)
        found_ops = unique_ops
        
        # For CREATE_FUNCTION goals, drop CRUD operations (add/remove/update/search/list)
        # which are project-level concerns, not algorithm functions. This prevents
        # goal-framing verbs like 'create' from becoming a CRUD 'add' operation.
        if goal.goal_type == GoalType.CREATE_FUNCTION:
            crud_ops = {"add", "remove", "update", "search", "list"}
            found_ops = [op for op in found_ops if op["name"] not in crud_ops]

        # Remove find_max/find_min when they appear in a sorting context.
        # "sort from largest to smallest" means sort_descending, not find_max.
        op_names = [op["name"] for op in found_ops]
        has_sort = any(n in op_names for n in ("sort_descending", "sort_ascending",
                                                "sort_list", "sort"))
        if has_sort:
            # Remove find_max/find_min that are not explicitly requested
            # (i.e., they appear within 3 positions of a sort operation)
            sort_positions = [i for i, n in enumerate(op_names)
                              if n in ("sort_descending", "sort_ascending", "sort_list", "sort")]
            filtered_ops = []
            for i, op in enumerate(found_ops):
                if op["name"] in ("find_max", "find_min"):
                    # Check if this is near a sort operation
                    near_sort = any(abs(i - sp) <= 3 for sp in sort_positions)
                    if near_sort:
                        continue  # Skip - this "largest/smallest" refers to sort order
                filtered_ops.append(op)
            found_ops = filtered_ops
        
        # Apply semantic normalization to operation names
        for op_info in found_ops:
            op_info["name"] = self._normalize_operation(op_info["name"], working_text)
        
        # Create operations with dependency chain
        prev_op_id = None
        for i, op_info in enumerate(found_ops):
            op_id = f"op_{i}"
            depends = [prev_op_id] if prev_op_id else []
            
            # Find input entities for this operation
            input_entities = [e.id for e in entities if e.is_input]
            
            # Classify semantic role
            semantic_role = self._classify_semantic_role(op_info["name"], working_text)
            
            op = OperationSpec(
                id=op_id,
                name=op_info["name"],
                op_type=op_info["type"],
                semantic_role=semantic_role,
                depends_on=depends,
                input_entities=input_entities,
                description=f"{op_info['role']}: {op_info['name']}",
                semantic_justification=(
                    f"operation pattern '{op_info['role']}' matched "
                    "the request"),
                evidence=[f"pattern matched: "
                          f"{op_info.get('pattern', op_info['name'])}"],
                input_bindings=([{"input": "data", "source": d}
                                 for d in depends]
                                or [{"input": "data", "source": "__input__"}]),
            )
            graph.add_operation(op)
            prev_op_id = op_id
        
        if graph.operations:
            graph.entry_points = [graph.operations[0].id]
            graph.exit_points = [graph.operations[-1].id]
        
        # If no operations found but goal suggests some, infer defaults
        if not graph.operations:
            self._infer_default_operations(graph, goal, entities)
        
        return graph
    
    def _infer_default_operations(self, graph: OperationGraph, goal: GoalSpec, entities: list[EntityRef]):
        """Infer default operations when none explicitly found."""
        if goal.goal_type == GoalType.CREATE_FUNCTION:
            if goal.algorithm_hint:
                graph.add_operation(OperationSpec(
                    id="op_0",
                    name=goal.algorithm_hint,
                    op_type=OperationType.TRANSFORM,
                    semantic_role="algorithm",
                    description=f"Algorithm: {goal.algorithm_hint}",
                    semantic_justification=(
                        f"the goal declares algorithm hint "
                        f"'{goal.algorithm_hint}'"),
                    evidence=[f"goal: {goal.algorithm_hint}"],
                    input_bindings=[{"input": "data",
                                     "source": "__input__"}],
                ))
            else:
                graph.add_operation(OperationSpec(
                    id="op_0",
                    name="transform",
                    op_type=OperationType.TRANSFORM,
                    semantic_role="computation",
                    description="Compute result from inputs",
                    semantic_justification=(
                        "the goal is a function creation with no stated "
                        "algorithm — a generic computation from inputs"),
                    evidence=[f"goal: {goal.goal_type.value}"],
                    input_bindings=[{"input": "data",
                                     "source": "__input__"}],
                ))
        elif goal.goal_type == GoalType.CREATE_PROJECT:
            # Default CRUD operations
            for i, op_name in enumerate(["add", "remove", "update", "search", "list"]):
                graph.add_operation(OperationSpec(
                    id=f"op_{i}",
                    name=op_name,
                    op_type=OperationType.IO,
                    semantic_role="crud",
                    depends_on=[f"op_{i-1}"] if i > 0 else [],
                    description=f"CRUD: {op_name}",
                    semantic_justification=(
                        f"a project goal defaults to the CRUD lifecycle "
                        f"step '{op_name}'"),
                    evidence=[f"goal: {goal.goal_type.value} (project CRUD)"],
                    input_bindings=([{"input": "data", "source": f"op_{i-1}"}]
                                    if i > 0 else
                                    [{"input": "data",
                                      "source": "__input__"}]),
                ))
            graph.entry_points = ["op_0"]
            graph.exit_points = [f"op_{len(graph.operations)-1}"]
    
    def _bind_objective_operations(self, semantic: 'SemanticRepresentation') -> None:
        """The objective's direction over an element IS the selection operation.

        maximize(length(element)) over a collection selects the extremum of
        that collection — find_max; minimize selects find_min.  The binding
        comes from the objective's structure, never from a word list: it is
        applied when no operation already realizes the objective, and the
        generic placeholder operation (an "unknown computation" step that
        only exists because no operation was stated) is superseded by it.
        """
        if not semantic.objectives:
            return
        obj = semantic.objectives[0]
        if getattr(obj, "direction", "") not in ("maximize", "minimize"):
            return
        if getattr(obj, "measure", "") not in ("length", "value"):
            return
        # The objective must RANGE OVER THE COLLECTION'S ELEMENTS.  A
        # structure-shaped objective ("the maximum subarray sum") ranges
        # over subarrays and needs the operation that realizes that
        # structure — an element extremum is a different computation.
        if getattr(obj, "target", "") != "element":
            return

        graph = semantic.operation_graph
        needed = "find_max" if obj.direction == "maximize" else "find_min"

        existing = [op for op in graph.operations]
        names = {op.name for op in existing}
        if needed in names:
            # The selection already exists: record the measure it selects by
            # so a length-measure extremum is not read as a value extremum.
            for op in existing:
                if op.name == needed and not getattr(op, "params", None):
                    if obj.measure == "length":
                        op.params = {"key": "length"}
            return

        # A placeholder computation step carries no operation semantics: it
        # exists solely because none was stated.  The objective states one.
        placeholder_ids = {
            op.id for op in existing
            if op.name == "transform"
            and any("no stated algorithm" in e or e.startswith("goal: ")
                   for e in (op.evidence or []))
        }
        if placeholder_ids and len(placeholder_ids) == len(existing):
            graph.operations = [op for op in existing
                                if op.id not in placeholder_ids]
        elif placeholder_ids:
            graph.operations = [op for op in existing
                                if op.id not in placeholder_ids]

        prev = graph.operations[-1] if graph.operations else None
        op_id = f"op_{len(graph.operations)}"
        target = getattr(obj, "target", "") or "element"
        op = OperationSpec(
            id=op_id,
            name=needed,
            op_type=OperationType.AGGREGATE,
            semantic_role="selection",
            params={"key": "length"} if obj.measure == "length" else {},
            depends_on=[prev.id] if prev else [],
            input_entities=[e.id for e in semantic.entities if e.is_input],
            description=f"Selection: {needed} by {obj.measure}",
            semantic_justification=(
                f"the objective {obj.direction}({obj.measure}) over the "
                f"{target} selects one element of the input collection"),
            evidence=[f"objective: {obj.direction}({obj.measure}) over {target}"],
            input_bindings=[{"input": "data",
                             "source": prev.id if prev else "__input__"}],
        )
        graph.add_operation(op)
        graph.entry_points = [graph.operations[0].id]
        graph.exit_points = [graph.operations[-1].id]

    @classmethod
    def _canonical_type(cls, word: str) -> str:
        """Resolve a surface word to a canonical type token (morphology only).

        "strings" is the same TYPE as "string": the plural of a noun is
        grammar, not a different meaning.  Without this the plural falls
        through to the generic default ("item") and the element type the
        function declares is uninformative.
        """
        w = (word or "").strip().lower()
        if not w:
            return ""
        if w in cls.VALID_TYPES:
            return w
        for suffix, stem in (("ies", "y"), ("es", ""), ("s", "")):
            if len(w) > len(suffix) and w.endswith(suffix):
                cand = w[:-len(suffix)] + stem
                if cand in cls.VALID_TYPES:
                    return cand
        return ""

    _TYPE_TO_LANG = {
        "string": "str", "text": "str", "str": "str",
        "character": "str", "char": "str",
        "integer": "int", "int": "int",
        "number": "float", "value": "float",
        "float": "float", "double": "float", "real": "float",
        "boolean": "bool", "bool": "bool",
    }

    @classmethod
    def _lang_type(cls, type_signature: str) -> str:
        """Semantic type vocabulary → the language-level type vocabulary.

        The semantic layer says String/List<String>; the IR and the
        generated return say str/list[str].  One derivation, so the two
        vocabularies can never describe different types.
        """
        sig = (type_signature or "").strip()
        if not sig:
            return ""
        m = re.match(r'^\s*List<(.+)>\s*$', sig, flags=re.IGNORECASE)
        if m:
            inner = cls._lang_type(m.group(1).strip())
            return f"list[{inner}]" if inner else "list"
        return cls._TYPE_TO_LANG.get(sig.lower(), sig.lower())

    @classmethod
    def _element_type(cls, semantic: 'SemanticRepresentation') -> str:
        """The declared element the objective ranges over: List<T> → T.

        Only an INFORMATIVE declaration counts — the generic placeholder
        ("Item") says nothing about the element, so the primitive type
        entity decides instead.
        """
        generic = {"item", "element", "any", "*", "value", "unknown", ""}
        for inp in getattr(semantic, "inputs", None) or []:
            m = re.match(r'^\s*List<(.+)>\s*$',
                         getattr(inp, "type_signature", "") or "",
                         flags=re.IGNORECASE)
            if m and m.group(1).strip().lower() not in generic:
                return m.group(1).strip()
        collections = {"list", "array", "sequence", "collection",
                       "리스트", "배열", "목록", "시퀀스"}
        for ent in getattr(semantic, "entities", None) or []:
            name = (getattr(ent, "name", "") or "").strip()
            if getattr(ent, "kind", "") != "primitive" or not name:
                continue
            if name.lower() in collections:
                continue
            if cls._canonical_type(name) in cls.VALID_TYPES:
                return name
        return ""

    def _wire_io_types(self, semantic: 'SemanticRepresentation') -> None:
        """Declare the objective's I/O from the structure it operates on.

        A selection objective over List<T> RETURNS T: the output type is
        derived from the objective's measure and the input's element, and
        the function signature states the same types — the semantic, the
        IR and the return type describe one pipeline.
        """
        element = self._element_type(semantic)

        # 1. The input's element: upgrade an uninformative placeholder.
        if element:
            for inp in semantic.inputs:
                sig = inp.type_signature or ""
                m = re.match(r'^\s*List<(.+)>\s*$', sig, flags=re.IGNORECASE)
                if m and m.group(1).strip().lower() in (
                        "item", "element", "any", "*", "value", "unknown"):
                    inp.type_signature = f"List<{element}>"
                    inp.description = f"Input {inp.type_signature}"

        # 2. The output: what the objective returns.
        out_type = ""
        if semantic.objectives:
            obj = semantic.objectives[0]
            if getattr(obj, "measure", "") in ("sum", "product"):
                out_type = "Number"
            elif getattr(obj, "measure", "") == "count":
                out_type = "Integer"
            elif element:
                out_type = element
        if not out_type and not semantic.outputs and element:
            # No stated objective: the terminal operation's contract still
            # decides (a scalar selection returns its element).
            order = [op.name for op in semantic.operation_graph.get_execution_order()]
            if order and order[-1] in ("find_max", "find_min"):
                out_type = element
            elif order and order[-1] in ("calculate_sum", "calculate_average"):
                out_type = "Number"
            elif order and order[-1] == "count_elements":
                out_type = "Integer"
        if out_type and not semantic.outputs:
            semantic.outputs.append(IOSpec(
                name="output", type_signature=out_type,
                description=f"Output {out_type}"))

        # 3. The language-level return type follows the same derivation
        #    (invariant: semantic == IR == return == behavior).
        derived = self._derive_output_type(semantic)
        if derived:
            semantic.output_type = derived

        # 4. The signature states exactly what the pipeline takes/returns.
        semantic.function_signature.parameters = [
            ParameterSpec(name=inp.name or "input",
                          type_name=self._lang_type(inp.type_signature) or "Any")
            for inp in semantic.inputs
        ]
        return_type = self._lang_type(semantic.output_type) \
            if semantic.output_type else ""
        if not return_type and semantic.outputs:
            return_type = self._lang_type(semantic.outputs[0].type_signature)
        semantic.function_signature.return_type = return_type or "Any"

    def _extract_io(
        self, 
        text: str, 
        lang: Language, 
        entities: list[EntityRef],
        graph: OperationGraph
    ) -> tuple[list[IOSpec], list[IOSpec]]:
        """Extract input/output specifications.
        
        Handles both traditional function requests AND competitive programming
        problems with stdin/stdout format.
        """
        inputs = []
        outputs = []
        text_lower = text.lower()
        
        # ── Competitive Programming I/O Detection ──
        # Check for stdin/stdout patterns common in competitive programming
        has_stdin = bool(re.search(r'(?:standard\s+input|stdin|입력\s*파일|입력\s* 형식|input\s*format|input\s+is\s+given)', text_lower))
        has_stdout = bool(re.search(r'(?:standard\s+output|stdout|출력\s*파일|출력\s* 형식|output\s*format|print\s+the|출력)', text_lower))
        
        if has_stdin or has_stdout:
            # This is a competitive programming problem - extract structured I/O
            # Parse input format section
            input_format = self._parse_input_format(text)
            output_format = self._parse_output_format(text)
            
            if input_format:
                for inp in input_format:
                    inputs.append(IOSpec(name=inp['name'], type_signature=inp['type'], 
                                       description=f"stdin: {inp['desc']}"))
            
            if output_format:
                for out in output_format:
                    outputs.append(IOSpec(name=out['name'], type_signature=out['type'],
                                        description=f"stdout: {out['desc']}"))
            
            # Mark as competitive programming for downstream handling
            if inputs or outputs:
                # Add constraint for competitive programming mode
                from brain.semantic_representation import ConstraintSpec
                # Note: constraints are extracted separately, just return I/O here
                return inputs, outputs
        
        # ── Traditional Function I/O Detection ──
        # Input patterns - extract element type, not verbs
        # Pattern: "receives/takes/gets/accepts [a] list/array of <type>"
        input_patterns = [
            (r"(?:입력|input|받는|받는다|takes?|receives?|gets?|accepts?)\s+(?:a\s+)?(?:리스트|list|배열|array|sequence|시퀀스)\s+(?:of\s+)?(\w+)", "list"),
            (r"(?:리스트|list|배열|array|sequence|시퀀스)\s+(?:of\s+)?(\w+)", "list"),
            (r"(\w+)\s+(?:리스트|list|배열|array|sequence|시퀀스)", "list"),
            (r"(?:입력|input|받는|받는다|takes?|receives?|gets?|accepts?)\s+(?:a\s+)?(\w+)", "single"),
        ]
        
        for pattern, io_type in input_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    elem_type = match.group(1).lower()
                except IndexError:
                    elem_type = "item"
                
                # Validate element type - reject stop words, accept valid types
                if elem_type in self.STOP_WORDS:
                    # Try to find a valid type after the verb
                    continue
                canonical = self._canonical_type(elem_type)
                if canonical:
                    elem_type = canonical
                elif elem_type not in self.VALID_TYPES:
                    # Default to Number for numeric operations, Item otherwise
                    if any(op.name in ("calculate_average", "filter_above", "filter_below", "sort_descending", "sort_ascending", "sum_list", "find_max", "find_min") for op in graph.operations):
                        elem_type = "number"
                    else:
                        elem_type = "item"
                
                type_sig = f"List<{elem_type.capitalize()}>" if io_type == "list" else elem_type.capitalize()
                inp = IOSpec(name="input", type_signature=type_sig, description=f"Input {type_sig}")
                inputs.append(inp)
                break
        
        # Output patterns
        output_patterns = [
            (r"(?:반환|return|출력|output|줍니다|줘).*(?:내림차순|descending|오름차순|ascending)", "List"),
            (r"(?:반환|return|출력|output|줍니다|줘)\s+(?:a\s+)?(?:리스트|list|배열|array)\s+(?:of\s+)?(\w+)", "list"),
            (r"(?:반환|return|출력|output|줍니다|줘)\s+(\w+)", "single"),
        ]
        
        for pattern, io_type in output_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    elem_type = match.group(1).lower() if match.groups() else "item"
                except IndexError:
                    elem_type = "item"
                
                # Validate element type
                if elem_type in self.STOP_WORDS:
                    continue
                canonical = self._canonical_type(elem_type)
                if canonical:
                    elem_type = canonical
                elif elem_type not in self.VALID_TYPES:
                    if any(op.name in ("calculate_average", "filter_above", "filter_below", "sort_descending", "sort_ascending", "sum_list", "find_max", "find_min") for op in graph.operations):
                        elem_type = "number"
                    else:
                        elem_type = "item"
                
                if io_type == "list":
                    type_sig = f"List<{elem_type.capitalize()}>"
                elif "descending" in pattern or "내림차순" in pattern:
                    type_sig = f"List<{elem_type.capitalize()}>"
                else:
                    type_sig = elem_type.capitalize()
                out = IOSpec(name="output", type_signature=type_sig, description=f"Output {type_sig}")
                outputs.append(out)
                break
        
        # Default: infer from operations
        if not inputs and graph.operations:
            has_list_op = any(op.name in ("remove_duplicates", "filter_above", "filter_below", 
                                           "sort_descending", "sort_ascending", "sort_list",
                                           "calculate_average",
                                           "sum_list", "find_max", "find_min", "filter_even", "filter_odd",
                                           "is_prime", "find_primes", "prime_filter")
                                for op in graph.operations)
            if has_list_op:
                # Try to infer element type from entities
                elem_type = "Number"
                for e in entities:
                    if e.kind == "primitive" and e.name != "List":
                        elem_type = e.name
                        break
                inputs.append(IOSpec(name="input", type_signature=f"List<{elem_type}>", 
                                    description=f"Input list of {elem_type.lower()}s"))
                outputs.append(IOSpec(name="output", type_signature=f"List<{elem_type}>",
                                     description=f"Processed list of {elem_type.lower()}s"))
        
        return inputs, outputs
    
    def _parse_input_format(self, text: str) -> list[dict]:
        """Parse input format from competitive programming problem statements.
        
        Extracts variable names, types, and structure from sections like:
        "Input
         X
         
         The input is given from Standard Input in the following format:
         N M
         A_1 A_2 ... A_N"
        """
        results = []
        
        # Look for "Input" section
        input_section_match = re.search(r'(?:Input|입력)\s*\n(.*?)(?:\n\s*(?:Output|출력|Constraint|제약|Sample|예시))', text, re.DOTALL | re.IGNORECASE)
        if not input_section_match:
            return results
        
        input_section = input_section_match.group(1).strip()
        
        # Look for format description after "in the following format"
        format_match = re.search(r'(?:the\s+following\s+format|다음\s+형식)\s*[:：]?\s*\n(.*?)(?:\n\s*\n|\n\s*(?:Constraint|제약|Sample|예시|Output|출력))', input_section, re.DOTALL | re.IGNORECASE)
        
        if format_match:
            format_lines = format_match.group(1).strip().split('\n')
        else:
            # Use the entire input section
            format_lines = input_section.strip().split('\n')
        
        # Parse format lines to extract variables
        for line in format_lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a variable specification line (e.g., "N M", "A_1 A_2 ... A_N")
            # Skip lines that look like descriptions
            if len(line) > 100 or any(c in line for c in ['.', ',', ';', ':']):
                continue
            
            # Parse variable names from format line
            # Common patterns: "N M", "A_1 A_2 ... A_N", "X", "S"
            vars_in_line = re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:_\d+)?', line)
            
            for var in vars_in_line:
                # Determine type from context
                var_type = self._infer_variable_type(var, text)
                results.append({
                    'name': var,
                    'type': var_type,
                    'desc': f'Input variable {var}'
                })
        
        return results
    
    def _parse_output_format(self, text: str) -> list[dict]:
        """Parse output format from competitive programming problem statements."""
        results = []
        
        # Look for "Output" section
        output_section_match = re.search(r'(?:Output|출력)\s*\n(.*?)(?:\n\s*(?:Constraint|제약|Sample|예시))', text, re.DOTALL | re.IGNORECASE)
        if not output_section_match:
            return results
        
        output_section = output_section_match.group(1).strip()
        
        # Look for format description
        format_match = re.search(r'(?:the\s+following\s+format|다음\s+형식)\s*[:：]?\s*\n(.*?)(?:\n\s*\n|\n\s*(?:Constraint|제약|Sample|예시))', output_section, re.DOTALL | re.IGNORECASE)
        
        if format_match:
            format_lines = format_match.group(1).strip().split('\n')
        else:
            # Check if it's a simple "Print the answer" pattern
            if re.search(r'(?:print|출력|return|반환)\s+(?:the|그|이)\s+', output_section, re.IGNORECASE):
                results.append({
                    'name': 'answer',
                    'type': 'Any',
                    'desc': 'Output answer'
                })
                return results
            format_lines = output_section.strip().split('\n')
        
        # Parse format lines
        for line in format_lines:
            line = line.strip()
            if not line:
                continue
            
            if len(line) > 100:
                continue
            
            # Check for variable references
            vars_in_line = re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:_\d+)?', line)
            
            if vars_in_line:
                for var in vars_in_line:
                    var_type = self._infer_variable_type(var, text)
                    results.append({
                        'name': var,
                        'type': var_type,
                        'desc': f'Output variable {var}'
                    })
            elif re.search(r'(?:print|출력|return|반환)', line, re.IGNORECASE):
                results.append({
                    'name': 'answer',
                    'type': 'Any',
                    'desc': 'Output answer'
                })
        
        return results
    
    def _infer_variable_type(self, var_name: str, context: str) -> str:
        """Infer the type of a variable from its name and context."""
        # Common naming conventions
        var_lower = var_name.lower()
        
        # Check if it's a list/array variable
        if re.search(rf'{re.escape(var_lower)}(?:_\d+)?\s+(?:{re.escape(var_lower)}(?:_\d+)?\s+)+', context.lower()):
            return 'List<Integer>'
        
        # Check for common patterns
        if var_name in ('N', 'M', 'K', 'Q', 'T', 'X', 'Y', 'R', 'C'):
            # Single uppercase letter often means integer
            return 'Integer'
        
        if re.search(rf'{var_name}\s+(?:\d+\s+)*\d+', context):
            return 'Integer'
        
        # Check if it appears in a list context
        if re.search(rf'{var_name}_\d+', context):
            return 'List<Integer>'
        
        # Default to Integer for competitive programming
        return 'Integer'
    
    def _extract_constraints(self, text: str, lang: Language) -> list[ConstraintSpec]:
        """Extract all constraints from text: language, platform, numeric, feature, quality."""
        constraints = []
        text_lower = text.lower()
        
        # Language constraints
        lang_map = {
            "python": ["python", "파이썬"], "kotlin": ["kotlin", "코틀린"],
            "java": ["java", "자바"], "javascript": ["javascript", "자바스크립트", "js"],
            # NOTE: bare "고" is a Korean grammatical connective
            # ("제거하고", "만들고") — not a language reference. Only the
            # explicit forms count.
            "go": ["go", "고 언어", "고랭"], "rust": ["rust", "러스트"],
            "c": ["c언어", "c language"], "cpp": ["c++", "cpp"],
        }
        for lang_code, keywords in lang_map.items():
            if any(k in text_lower for k in keywords):
                constraints.append(ConstraintSpec(
                    name="language", constraint_type="language",
                    value=lang_code, description=f"Target language: {lang_code}"
                ))
                break
        
        # Platform constraints
        plat_map = {
            "android": ["android", "안드로이드", "모바일", "mobile"],
            "web": ["web", "웹", "브라우저", "browser"],
            "cli": ["cli", "명령줄", "command line", "터미널", "terminal", "console"],
        }
        for plat, keywords in plat_map.items():
            if any(k in text_lower for k in keywords):
                constraints.append(ConstraintSpec(
                    name="platform", constraint_type="platform",
                    value=plat, description=f"Target platform: {plat}"
                ))
                break
        
        # Numeric constraints
        numeric_patterns = [
            (r'(\d+)\s*[xX]\s*(zoom|줌|확대)', 'minimum_zoom', 'zoom'),
            (r'(\d+)\s*(times|배|배율)', 'multiplier', 'multiply'),
            (r'(top|상위|최고|상위)\s*(\d+)', 'top_n', 'limit'),
            (r'(bottom|하위|최하)\s*(\d+)', 'bottom_n', 'limit'),
            (r'(\d+)\s*(items?|개|건|條|개월|days?|일)', 'count', 'limit'),
            (r'(최소|minimum|min)\s*(\d+)', 'minimum', 'range'),
            (r'(최대|maximum|max)\s*(\d+)', 'maximum', 'range'),
            (r'(\d+)%', 'percentage', 'threshold'),
            (r'(\d+)\s*(ms|millisecond|밀리초)', 'timeout_ms', 'timeout'),
            (r'(\d+)\s*(sec|second|초)', 'timeout_sec', 'timeout'),
            (r'(\d+)\s*(px|pixel|픽셀)', 'size_px', 'dimensions'),
        ]
        for pattern, name, category in numeric_patterns:
            m = re.search(pattern, text_lower)
            if m:
                value = m.group(1) if m.lastindex else m.group(0)
                constraints.append(ConstraintSpec(
                    name=name, constraint_type="numeric",
                    value=value,
                    description=f"Numeric constraint: {name}={value}"
                ))
        
        # Feature requirements
        feature_keywords = {
            "complete_code": ["complete code", "전체 코드", "완전한 코드", "complete", "전체", "완전"],
            "buildable": ["buildable", "빌드 가능한", "compilable", "컴파일 가능한", "실행 가능한", " runnable"],
            "working": ["working", "작동하는", "동작하는", "동작"],
            "test_code": ["test code", "테스트 코드", "tests", "테스트"],
            "documentation": ["documentation", "문서", "readme", "주석"],
        }
        for feat_name, keywords in feature_keywords.items():
            if any(k in text_lower for k in keywords):
                constraints.append(ConstraintSpec(
                    name=feat_name, constraint_type="feature",
                    value="required",
                    description=f"Feature requirement: {feat_name}"
                ))
        
        return constraints
    
    def _extract_objectives(
        self, text: str, lang: Language, semantic: 'SemanticRepresentation | None' = None
    ) -> list[ObjectiveSpec]:
        """Extract optimization objectives from text.

        General semantic primitives (NOT problem-specific):
        - longest/most extended/maximum length → maximize(length)
        - shortest/least extended/minimum length → minimize(length)
        - maximum sum/highest total → maximize(sum)
        - minimum sum/lowest total → minimize(sum)
        - maximum/maximum value → maximize(value)
        - minimum/minimum value → minimize(value)
        - most/fewest → maximize(count)/minimize(count)

        A superlative word that scales an ELEMENT ("largest", "greatest",
        "biggest") is the VALUE measure, never LENGTH: "the largest value"
        selects a value, "the longest string" selects by length.  The two
        families are separate patterns so one can never shadow the other.
        """
        text_lower = text.lower()
        objectives = []

        # Pattern: (direction, measure, evidence_pattern)
        # Order matters: more specific patterns first
        patterns = [
            # Length optimization: LENGTH-shaped words only.  A bare
            # superlative ("largest", "smallest") scales an element and
            # therefore belongs to the value family below — putting it
            # here made "the largest value" report measure=length.
            (r'\b(longest|most\s+extended|maximum\s+length|greatest\s+length)\b',
             "maximize", "length", "longest"),
            (r'\b(shortest|least\s+extended|minimum\s+length|least\s+long)\b',
             "minimize", "length", "shortest"),

            # Sum optimization
            (r'\b(maximum\s+sum|highest\s+sum|largest\s+sum|greatest\s+sum|maximum\s+total|highest\s+total)\b',
             "maximize", "sum", "maximum sum"),
            (r'\b(minimum\s+sum|lowest\s+sum|smallest\s+sum|least\s+sum|minimum\s+total|lowest\s+total)\b',
             "minimize", "sum", "minimum sum"),

            # Value optimization
            # Note: bare superlatives (maximum/minimum) indicate value optimization.
            # "top"/"bottom" are excluded because they are commonly spatial nouns
            # ("i-th row from the top"), not optimization words. "top N"/"bottom N"
            # counting is handled by numeric constraint extraction instead.
            (r'\b(maximum|maximum\s+value|highest|largest|greatest|biggest)\b',
             "maximize", "value", "maximum"),
            (r'\b(minimum|minimum\s+value|lowest|smallest|least)\b',
             "minimize", "value", "minimum"),

            # Count optimization
            (r'\b(most|maximum\s+count|greatest\s+number)\b',
             "maximize", "count", "most"),
            (r'\b(fewest|minimum\s+count|least\s+number|smallest\s+number)\b',
             "minimize", "count", "fewest"),

            # Product optimization
            (r'\b(maximum\s+product|largest\s+product|greatest\s+product)\b',
             "maximize", "product", "maximum product"),
            (r'\b(minimum\s+product|smallest\s+product|least\s+product)\b',
             "minimize", "product", "minimum product"),
        ]

        for pattern, direction, measure, evidence in patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Determine target from context
                target = self._infer_objective_target(text_lower, semantic)
                objectives.append(ObjectiveSpec(
                    direction=direction,
                    measure=measure,
                    target=target,
                    confidence=0.85,
                    evidence=evidence,
                ))
                break  # Take the first (most specific) match

        return objectives

    def _infer_objective_target(
        self, text_lower: str, semantic: 'SemanticRepresentation | None' = None
    ) -> str:
        """Infer what is being optimized from context.

        The target is STRUCTURE: an explicit contiguous/non-contiguous
        shape in the text decides it, otherwise an objective over a
        collection selects an ELEMENT of that collection.  The shape is
        never invented — a text that declares no shape and carries no
        collection yields "" rather than a fabricated "subarray".
        """
        # Check for subarray/segment/subsequence patterns
        if re.search(r'\b(subarray|sub\s*array|contiguous\s+(?:sub)?(?:array|segment|sequence|portion|range|section)|continuous\s+(?:sub)?(?:array|segment|sequence)|consecutive\s+(?:sub)?(?:array|segment|sequence))\b', text_lower):
            return "subarray"
        if re.search(r'\b(subsequence|sub\s*sequence|non[\s-]*contiguous)\b', text_lower):
            return "subsequence"
        if re.search(r'\b(prefix|first)\b', text_lower):
            return "prefix"
        if re.search(r'\b(suffix|last)\b', text_lower):
            return "suffix"
        if re.search(r'\b(pair|two\s+elements?)\b', text_lower):
            return "pair"
        if re.search(r'\b(segment|portion|range|section|interval)\b', text_lower):
            return "segment"
        if re.search(r'\b(element|item|value|number)\b', text_lower):
            return "element"
        # No declared shape: an objective over a COLLECTION is a selection
        # among its ELEMENTS — decided by the collection's presence in the
        # semantics, never by a default shape.
        if self._declares_collection(semantic):
            return "element"
        return ""

    @staticmethod
    def _declares_collection(semantic: 'SemanticRepresentation | None') -> bool:
        """Does the semantics describe a collection the objective ranges over?"""
        if semantic is None:
            return False
        for inp in getattr(semantic, "inputs", None) or []:
            sig = (getattr(inp, "type_signature", "") or "")
            if sig.startswith("List<") or sig.startswith("list["):
                return True
        for ent in getattr(semantic, "entities", None) or []:
            if getattr(ent, "kind", "") == "primitive" and \
                    (getattr(ent, "name", "") or "").lower() in (
                        "list", "array", "sequence", "collection", "리스트", "배열"):
                return True
        return False

    
    def _extract_substructures(self, text: str, lang: Language) -> list[SubstructureSpec]:
        """Extract structural patterns from text.
        
        General semantic primitives:
        - contiguous subarray/segment/sequence/portion → contiguous_subsequence
        - subsequence (non-contiguous) → subsequence
        - prefix → prefix
        - suffix → suffix
        - pair → pair
        - segment/portion/range/section → segment
        """
        text_lower = text.lower()
        substructures = []
        
        # Pattern: (structure_type, is_contiguous, evidence_pattern, confidence)
        patterns = [
            # Contiguous patterns (most specific first)
            (r'\b(contiguous\s+(?:sub)?(?:array|segment|sequence|portion|range|section)|continuous\s+(?:sub)?(?:array|segment|sequence)|consecutive\s+(?:sub)?(?:array|segment|sequence))\b',
             "contiguous_subsequence", True, "contiguous", 0.9),
            (r'\b(subarray|sub\s*array)\b',
             "contiguous_subsequence", True, "subarray", 0.85),
            (r'\b(segment|portion|range|section|interval)\b',
             "segment", True, "segment", 0.8),
            
            # Non-contiguous patterns
            (r'\b(subsequence|sub\s*sequence)\b',
             "subsequence", False, "subsequence", 0.85),
            
            # Fixed structures
            (r'\b(prefix|first\s+k?)\b',
             "prefix", True, "prefix", 0.85),
            (r'\b(suffix|last\s+k?)\b',
             "suffix", True, "suffix", 0.85),
            (r'\b(pair|two\s+elements?|two\s+numbers?)\b',
             "pair", False, "pair", 0.85),
            
            # Window patterns
            (r'\b(window|sliding\s+window|k[\s-]*element)\b',
             "window", True, "window", 0.8),
            (r'\b(submatrix|sub\s*matrix|subgrid|sub\s*grid)\b',
             "submatrix", True, "submatrix", 0.85),

            # Chained references through collections (general dataflow):
            # a value obtained via a chain of lookups — bracket notation
            # ("B[A[i]]"), explicit index usage ("as an index into"), or
            # chained relative references ("the X of the Y that the Z ...").
            (r'\b\w+\s*\[\s*\w+\s*\[\s*\w+\s*\]\s*\]'
             r'|use[s]?\s+(?:\w+\s+)?as\s+an?\s+inde[xX](?:es)?\s+into'
             r'|retriev\w*[^.]{0,60}elements?\s+from\s+another'
             r'|find\s+the\s+[\w\s]{0,40}\bof\s+the\s+[\w\s]{0,40}\bthat\s+the\s+',
             "reference_chain", False, "reference_chain", 0.8),
        ]
        
        for pattern, structure_type, is_contiguous, evidence, confidence in patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Determine element type from context
                element_type = self._infer_element_type(text_lower)
                substructures.append(SubstructureSpec(
                    structure_type=structure_type,
                    scope="entire_input",
                    element_type=element_type,
                    is_contiguous=is_contiguous,
                    confidence=confidence,
                    evidence=evidence,
                ))
                break  # Take the first (most specific) match
        
        return substructures
    
    def _infer_element_type(self, text_lower: str) -> str:
        """Infer the element type from context."""
        if re.search(r'\b(integer|int|number|정수|숫자|value|값)\b', text_lower):
            return "integer"
        if re.search(r'\b(string|문자열|text|문자|character|글자)\b', text_lower):
            return "string"
        if re.search(r'\b(list|array|배열|리스트|sequence)\b', text_lower):
            return "any"
        return "integer"  # default for competitive programming
    
    def _extract_aggregate_constraints(self, text: str, lang: Language) -> list[ConstraintSpec]:
        """Extract aggregate constraints like 'sum is zero', 'total equals 100'.
        
        General semantic primitives:
        - sum/total + is/equals + value → aggregate(sum, X) == value
        - product + is/equals + value → aggregate(product, X) == value
        - count/number + is/equals + value → aggregate(count, X) == value
        - elements? + add up to/sum to/total + value → aggregate(sum, X) == value
        """
        text_lower = text.lower()
        constraints = []
        
        # Pattern: aggregate_function + subject + operator + value
        patterns = [
            # Sum constraints - explicit
            (r'(?:sum|total|합|총합|전체\s*합)\s+(?:of\s+(?:the\s+)?)?(?:all\s+)?(?:elements?\s+(?:in|of|from)\s+)?(?:the\s+)?(?:subarray|segment|sequence|contiguous|portion|elements?)?\s*(?:is|equals?|=\s*|==\s*|adds?\s+up\s+to|계산|구하)\s+(-?\d+)',
             "sum", "==", "sum_constraint"),
            (r'(?:sum|total|합|총합)\s+(?:is|equals?)\s+zero',
             "sum", "==", "zero_sum"),
            # The ZERO target in its other general forms: a bare
            # juxtaposition ("with sum zero"), the verb form ("sums to
            # zero", "sum to zero"), or the numeral ("sum is 0").
            (r'(?:sums?|total)\s+(?:to\s+)?zero\b',
             "sum", "==", "zero_sum"),
            (r'with\s+(?:a\s+)?(?:sum|total)\s+zero\b',
             "sum", "==", "zero_sum"),
            (r'(?:sum|total)\s+zero\b',
             "sum", "==", "zero_sum"),
            (r'(?:sum|total|합|총합)\s+(?:is|equals?|이?)\s*0\b',
             "sum", "==", "zero_sum"),
            (r'(?:sum|total|합|총합)\s+(?:is|equals?)\s+(-?\d+)',
             "sum", "==", "sum_value"),
            
            # Sum constraints - implicit (elements add up to, summing to, etc.)
            (r'(?:elements?|numbers?|values?)\s+(?:add\s+up\s+to|sum\s+to|total\s+to)\s+(-?\d+)',
             "sum", "==", "elements_sum"),
            (r'(?:elements?|numbers?|values?)\s+(?:add\s+up\s+to|sum\s+to|total\s+to)\s+zero',
             "sum", "==", "elements_sum_zero"),
            (r'(?:summing|adding|totaling)\s+to\s+(-?\d+)',
             "sum", "==", "summing_to"),
            (r'(?:summing|adding|totaling)\s+to\s+zero',
             "sum", "==", "summing_to_zero"),
            (r'(?:with\s+total|totaling|amounting\s+to)\s+(-?\d+)',
             "sum", "==", "with_total"),
            (r'(?:with\s+total|totaling|amounting\s+to)\s+zero',
             "sum", "==", "with_total_zero"),
            
            # Product constraints
            (r'(?:product|곱|곱셈)\s+(?:of\s+(?:the\s+)?)?(?:all\s+)?(?:elements?\s+(?:in|of|from)\s+)?(?:the\s+)?(?:subarray|segment|sequence|contiguous|portion|elements?)?\s*(?:is|equals?|=\s*|==\s*)\s+(-?\d+)',
             "product", "==", "product_constraint"),
            
            # Count constraints
            (r'(?:count|number|개수|수)\s+(?:of\s+(?:the\s+)?)?(?:elements?\s+(?:in|of|from)\s+)?(?:the\s+)?(?:subarray|segment|sequence|contiguous|portion|elements?)?\s*(?:is|equals?|=\s*|==\s*)\s+(-?\d+)',
             "count", "==", "count_constraint"),
        ]
        
        for pattern, agg_func, operator, evidence in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = match.group(1) if match.lastindex else "0"
                except (IndexError, AttributeError):
                    value = "0"
                
                # Determine subject from context
                subject = self._infer_constraint_subject(text_lower)
                
                constraints.append(ConstraintSpec(
                    name=f"{agg_func}_{operator}_{value}",
                    constraint_type="aggregate",
                    value=value,
                    description=f"aggregate({agg_func}, {subject}) {operator} {value}",
                    subject=subject,
                    operator=operator,
                    aggregate_function=agg_func,
                    confidence=0.85,
                    evidence=evidence,
                ))
                break  # Take the first match
        
        return constraints
    
    def _extract_arithmetic_expression(self, text: str, input_var_names: list[str]) -> str:
        """Parse arithmetic expressions from natural language text.

        General compositional semantics (NOT problem-specific):
        - "square of A + B" → "(A + B) ** 2"
        - "product of A and B" → "A * B"
        - "sum of A and B" → "A + B"
        - "difference between A and B" → "A - B"
        - "twice A" → "2 * A"
        - "half of A" → "A // 2"
        - "A modulo M" → "A % M"

        Variable names are resolved against the actual input variables.
        Returns an empty string when no expression is found.
        """
        text_lower = text.lower()

        # Candidate variable letters (single-letter input variables)
        var_letters = [v for v in (input_var_names or [])
                       if re.match(r'^[A-Za-z]$', v or "")]

        def _build_expr(raw: str) -> str:
            """Convert a raw span like 'a + b' into a real expression."""
            raw = raw.strip().rstrip('.,;:')
            if not raw:
                return ""
            # Tokenize keeping operators
            tokens = re.findall(r'[a-z]+|\d+|[+\-*/()]', raw)
            out_tokens = []
            for tok in tokens:
                if re.match(r'^\d+$', tok):
                    out_tokens.append(tok)
                elif tok in ('+', '-', '*', '/', '(', ')'):
                    out_tokens.append(tok)
                elif len(tok) == 1 and tok.upper() in [v.upper() for v in var_letters]:
                    # Resolve to the actual input variable name
                    out_tokens.append(next(v for v in var_letters if v.upper() == tok.upper()))
                # skip multi-letter words — they are not variables
            if not out_tokens:
                return ""
            return " ".join(out_tokens)

        # 1. "square of X" → (X) ** 2
        m = re.search(r'\bsquare\s+of\s+([a-z0-9\s+\-*/()]+?)(?:\.|,|$|\n)', text_lower)
        if m:
            expr = _build_expr(m.group(1))
            if expr:
                return f"({expr}) ** 2"

        # 2. "product of A and B" → A * B
        m = re.search(r'\bproduct\s+of\s+([a-z0-9\s+\-*/()]+?)\s+and\s+([a-z0-9\s+\-*/()]+?)(?:\.|,|$|\n)', text_lower)
        if m:
            a, b = _build_expr(m.group(1)), _build_expr(m.group(2))
            if a and b:
                return f"{a} * {b}"

        # 3. "sum of A and B" → A + B
        m = re.search(r'\bsum\s+of\s+([a-z0-9\s+\-*/()]+?)\s+and\s+([a-z0-9\s+\-*/()]+?)(?:\.|,|$|\n)', text_lower)
        if m:
            a, b = _build_expr(m.group(1)), _build_expr(m.group(2))
            if a and b:
                return f"{a} + {b}"

        # 4. "difference between/of A and B" → A - B
        m = re.search(r'\bdifference\s+(?:between|of)\s+([a-z0-9\s+\-*/()]+?)\s+and\s+([a-z0-9\s+\-*/()]+?)(?:\.|,|$|\n)', text_lower)
        if m:
            a, b = _build_expr(m.group(1)), _build_expr(m.group(2))
            if a and b:
                return f"{a} - {b}"

        # 5. "twice A" → 2 * A
        m = re.search(r'\btwice\s+([a-z])\b', text_lower)
        if m and m.group(1).upper() in [v.upper() for v in var_letters]:
            var = next(v for v in var_letters if v.upper() == m.group(1).upper())
            return f"2 * {var}"

        # 6. "half of A" → A // 2
        m = re.search(r'\bhalf\s+of\s+([a-z])\b', text_lower)
        if m and m.group(1).upper() in [v.upper() for v in var_letters]:
            var = next(v for v in var_letters if v.upper() == m.group(1).upper())
            return f"{var} // 2"

        # 7. "A modulo M" / "A mod M" → A % M
        m = re.search(r'\b([a-z])\s+mod(?:ulo)?\s+([a-z]|\d+)\b', text_lower)
        if m and m.group(1).upper() in [v.upper() for v in var_letters]:
            var = next(v for v in var_letters if v.upper() == m.group(1).upper())
            mod_val = m.group(2)
            if mod_val.isdigit():
                return f"{var} % {mod_val}"
            if mod_val.upper() in [v.upper() for v in var_letters]:
                mv = next(v for v in var_letters if v.upper() == mod_val.upper())
                return f"{var} % {mv}"

        return ""

    def _infer_constraint_subject(self, text_lower: str) -> str:
        """Infer what the aggregate constraint applies to."""
        if re.search(r'\b(subarray|sub\s*array|contiguous|segment|sequence|portion|range|section)\b', text_lower):
            return "subarray"
        if re.search(r'\b(array|list|배열|리스트)\b', text_lower):
            return "array"
        if re.search(r'\b(elements?|items?|values?|numbers?)\b', text_lower):
            return "elements"
        return "subarray"  # default

    def _calculate_confidence(self, semantic: SemanticRepresentation, concepts: list[ActivatedConcept]) -> float:
        """Calculate overall parsing confidence.
        
        Reflects actual semantic grounding, not just plausibility.
        Weak concept matches penalize confidence.
        """
        scores = []
        
        # Goal confidence
        scores.append(semantic.goal.confidence)
        
        # Entity coverage
        if semantic.entities:
            scores.append(0.8)
        else:
            scores.append(0.3)
        
        # Operation coverage — penalize if concept strengths are weak
        if semantic.operation_graph.operations:
            if concepts:
                avg_strength = sum(c.activation_strength for c in concepts) / len(concepts)
                # Strong matches (>1.0) get full score, weak matches (<0.5) penalized
                op_score = min(avg_strength / 1.5, 1.0)
            else:
                op_score = 0.5
            scores.append(op_score)
        else:
            scores.append(0.2)
        
        # I/O specification
        if semantic.inputs and semantic.outputs:
            scores.append(0.8)
        elif semantic.inputs or semantic.outputs:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        # Concept activation support — use min strength to penalize weak matches
        if concepts:
            avg_strength = sum(c.activation_strength for c in concepts) / len(concepts)
            min_strength = min(c.activation_strength for c in concepts)
            # Both average AND minimum matter — one weak match drags down confidence
            concept_score = min((avg_strength * 0.6 + min_strength * 0.4) / 2.0, 1.0)
            scores.append(concept_score)
        
        return sum(scores) / len(scores) if scores else 0.0


def create_semantic_parser(concept_store: ConceptStore, relation_store: RelationStore) -> SemanticParser:
    """Factory function to create a SemanticParser."""
    return SemanticParser(concept_store, relation_store)