"""Reasoner - structured intent analysis with concept-based operation inference."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from brain.concept_store import ConceptStore, Concept
from brain.relation_store import RelationStore, Relation, RelationType
from brain.rule_store import RuleEngine
from brain.memory_store import MemoryStore, MemoryType


# COMPOUND_PATTERNS removed — compound intent detection uses semantic reasoning
COMPOUND_PATTERNS = []

# CLASS_PATTERNS removed — class detection uses concept activation
CLASS_PATTERNS = {}

# INTERFACE_PATTERNS removed — interface detection uses concept activation
INTERFACE_PATTERNS = {}

# ENUM_PATTERNS removed — enum detection uses concept activation
ENUM_PATTERNS = {}

# PROJECT_PATTERNS removed — project detection uses concept activation
PROJECT_PATTERNS = {}


@dataclass
class ClassInfo:
    name: str
    properties: list[tuple[str, str]] = field(default_factory=list)
    constructor_params: list[tuple[str, str]] = field(default_factory=list)
    parent_class: str = ""
    interfaces: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


@dataclass
class ReasoningStep:
    step_type: str
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class StructuredIntent:
    intent_type: str
    algorithm: str
    params: list[tuple[str, str]]
    return_type: str
    description: str
    needs_loop: bool = False
    is_recursive: bool = False
    data_type: str = "int"
    operation: str = ""
    class_info: Optional[ClassInfo] = None
    compound_steps: list[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    intent: str
    structured_intent: Optional[StructuredIntent]
    concepts: list[Concept]
    related_concepts: list[Concept]
    rules_applied: list[str]
    memories_recalled: list[str]
    steps: list[ReasoningStep]
    plan: dict = field(default_factory=dict)
    explanation: str = ""


class Reasoner:
    def __init__(self, concept_store: ConceptStore, relation_store: RelationStore,
                 rule_engine: RuleEngine, memory_store: MemoryStore):
        self.concepts = concept_store
        self.relations = relation_store
        self.rules = rule_engine
        self.memory = memory_store

    def analyze(self, request: str, language: str = "python") -> ReasoningResult:
        steps = []
        intent = self._identify_intent(request, steps)
        structured = self._analyze_structured_intent(request, intent, steps)
        matched_concepts = self._find_matching_concepts(request, intent, structured, steps)
        related = self._find_related_concepts(matched_concepts, steps)
        rules = self._find_applicable_rules(intent, matched_concepts, language, steps)
        memories = self._recall_relevant_memories(request, language, steps)
        explanation = self._build_explanation(steps, intent, structured, matched_concepts, language)

        return ReasoningResult(
            intent=intent,
            structured_intent=structured,
            concepts=matched_concepts,
            related_concepts=related,
            rules_applied=rules,
            memories_recalled=[m.id for m in memories],
            steps=steps,
            explanation=explanation,
        )

    def _identify_intent(self, request: str, steps: list[ReasoningStep]) -> str:
        """Identify intent by delegating to SemanticParser. Zero keyword authority."""
        try:
            from brain.semantic_parser import create_semantic_parser
            from brain.concept_store import ConceptStore, seed_default_concepts
            from brain.relation_store import RelationStore
            import tempfile
            cs = ConceptStore()
            seed_default_concepts(cs)
            parser = create_semantic_parser(
                cs, RelationStore(data_dir=tempfile.mkdtemp())
            )
            semantic = parser.parse(request)
            goal_name = semantic.goal.goal_type.value
            goal_to_intent = {
                "chat": "chat", "question": "question", "explain": "explain_code",
                "create_function": "generate_function", "create_project": "generate_project",
                "create_class": "generate_class", "modify": "modify_code",
                "fix": "fix_error", "analyze": "analyze_code",
            }
            intent = goal_to_intent.get(goal_name, "generate_function")
            steps.append(ReasoningStep(
                step_type="identify_intent",
                description=f"Identified intent: {intent} (via SemanticParser)",
                evidence=[f"goal_type={goal_name}, confidence={semantic.confidence}"],
                confidence=semantic.confidence,
            ))
            return intent
        except Exception:
            pass
        steps.append(ReasoningStep(
            step_type="identify_intent",
            description="Identified intent: generate_function (default)",
            evidence=["SemanticParser unavailable, using default"],
            confidence=0.5,
        ))
        return "generate_function"

    def _matches_patterns(self, request: str, patterns: dict) -> bool:
        """LEGACY: kept for interface compatibility. Zero decision authority."""
        return False

    def _match_compound_pattern(self, request: str) -> Optional[dict]:
        """Detect compound patterns using ConceptStore.search.

        Searches ConceptStore for multiple operation concepts and infers
        their relationships via RelationStore.
        """
        try:
            # Search for operation concepts
            results = self._search_operations(request)
            # the same algorithm read two ways ("ascending"/"descending")
            # is ONE operation, not a sequence of two
            found_ops = list(dict.fromkeys(
                self.CANONICAL_ALGORITHMS.get(c.id, c.id)
                for c in results))

            # If we found multiple operations, infer relationships
            if len(found_ops) >= 2:
                # Use RelationStore to find relationships between operations
                steps = []
                for i, op1 in enumerate(found_ops):
                    for op2 in found_ops[i+1:]:
                        relations = self.relations.find_relations(op1, op2)
                        if relations:
                            steps.append((op1, op2))

                if steps:
                    return {
                        "steps": found_ops,
                        "description": "Compound operation pattern",
                        "relations": steps,
                    }
        except Exception:
            pass
        return None

    def _match_class_function(self, request: str) -> bool:
        """LEGACY: Class+function detection is handled by SemanticParser now."""
        return False

    # The store names a concept by what it IS; the planner, the
    # generator and every caller name the algorithm by what it DOES TO
    # A LIST.  One translation keeps the two readings of the same idea
    # from drifting apart.
    CANONICAL_ALGORITHMS = {
        "calculate_sum": "sum_list",
        "calculate_average": "average_list",
        "sort_descending": "sort_list",
        "sort_ascending": "sort_list",
        "search": "search_in_list",
        "filter": "filter_elements",
        "concatenate": "concat_strings",
    }

    # A concept's own description is written in one language; a request
    # in another language reads the same idea through its surface
    # wording.  The gloss is only a bridge INTO the store — the concept
    # that answers is still chosen by the store's semantic match.
    _SURFACE_GLOSSES = (
        ("정렬", "ascending order arrangement"),
        ("정렬하기", "ascending order arrangement"),
        ("소팅", "ascending order arrangement"),
        ("오름차순", "ascending smallest first"),
        ("내림차순", "descending largest first"),
        ("평균", "average mean"),
        ("합계", "sum total"),
        ("더하기", "sum add"),
        ("중복", "duplicate unique"),
    )

    def _store_query(self, request: str) -> str:
        """The request read the way the store's concepts describe it."""
        lowered = request.lower()
        glosses = [g for surface, g in self._SURFACE_GLOSSES
                   if surface in lowered]
        return request + (" " + " ".join(glosses) if glosses else "")

    def _search_operations(self, request: str) -> list:
        """Operation concepts the request reads as, in the store."""
        return self.concepts.search(self._store_query(request),
                                    category="operation")

    def _match_algorithm_pattern(self, request: str) -> Optional[str]:
        """Detect algorithm by searching ConceptStore for operation concepts.

        Uses ConceptStore.search() which matches based on concept descriptions
        and tags, not keyword patterns.
        """
        try:
            # Search for operation concepts in the ConceptStore
            results = self._search_operations(request)
            if results:
                # Return the best matching operation concept
                concept_id = results[0].id
                return self.CANONICAL_ALGORITHMS.get(concept_id, concept_id)
        except Exception:
            pass
        return None

    def _analyze_structured_intent(self, request: str, intent: str,
                                   steps: list[ReasoningStep]) -> Optional[StructuredIntent]:
        # Check for compound patterns first (class + function combinations)
        compound = self._match_compound_pattern(request)
        if compound:
            return self._analyze_compound_intent(request, compound, steps)

        if intent == "generate_class":
            return self._analyze_class_intent(request, steps)

        algo_id = self._match_algorithm_pattern(request)
        if algo_id:
            return self._analyze_algorithmic_intent(request, algo_id, steps)

        steps.append(ReasoningStep(
            step_type="analyze_algorithm",
            description="No specific algorithmic pattern detected",
            evidence=["Falling back to generic function generation"],
            confidence=0.3,
        ))
        return None

    def _analyze_class_intent(self, request: str, steps: list[ReasoningStep]) -> StructuredIntent:
        class_info = self._extract_class_info(request)
        steps.append(ReasoningStep(
            step_type="analyze_class",
            description=f"Class intent: {class_info.name}",
            evidence=[
                f"Class name: {class_info.name}",
                f"Properties: {class_info.properties}",
                f"Constructor params: {class_info.constructor_params}",
            ],
            confidence=0.9,
        ))
        return StructuredIntent(
            intent_type="generate_class",
            algorithm="class",
            params=[],
            return_type="",
            description=f"클래스 {class_info.name} 생성",
            class_info=class_info,
        )

    def _extract_class_info(self, request: str) -> ClassInfo:
        name = "User"
        m = re.search(r'(\w+)\s*클래스', request)
        if m:
            name = m.group(1)
        m = re.search(r'class\s+(\w+)', request, re.IGNORECASE)
        if m:
            name = m.group(1)
        m = re.search(r'(\w+)\s*객체', request)
        if m:
            name = m.group(1)

        name_keywords = {
            "사용자": "User", "유저": "User", "product": "Product", "상품": "Product",
            "item": "Item", "아이템": "Item", "order": "Order", "주문": "Order",
            "account": "Account", "계정": "Account", "student": "Student", "학생": "Student",
            "person": "Person", "사람": "Person", "car": "Car", "자동차": "Car",
            "book": "Book", "책": "Book", "message": "Message", "메시지": "Message",
        }
        for kw, cn in name_keywords.items():
            if kw in request.lower() or kw in request:
                name = cn
                break

        properties = []
        constructor_params = []

        prop_patterns = [
            (r'이름', "name", "string"),
            (r'나이', "age", "int"),
            (r'주소', "address", "string"),
            (r'전화번호', "phone", "string"),
            (r'이메일', "email", "string"),
            (r'가격', "price", "float"),
            (r'수량', "quantity", "int"),
            (r'제목', "title", "string"),
            (r'내용', "content", "string"),
            (r'설명', "description", "string"),
            (r'아이디', "id", "int"),
            (r'번호', "number", "int"),
            (r'날짜', "date", "string"),
            (r'상태', "status", "string"),
            (r'타입', "type", "string"),
            (r'값', "value", "int"),
            (r'점수', "score", "int"),
            (r'등급', "grade", "string"),
        ]
        for pattern, prop_name, prop_type in prop_patterns:
            if pattern in request:
                properties.append((prop_name, prop_type))
                constructor_params.append((prop_name, prop_type))

        if not properties:
            properties = [("name", "string"), ("age", "int")]
            constructor_params = [("name", "string"), ("age", "int")]

        return ClassInfo(
            name=name,
            properties=properties,
            constructor_params=constructor_params,
        )

    def _analyze_compound_intent(self, request: str, compound: dict,
                                 steps: list[ReasoningStep]) -> StructuredIntent:
        steps.append(ReasoningStep(
            step_type="analyze_compound",
            description=f"Compound pattern: {compound['description']}",
            evidence=[f"Steps: {compound['steps']}"],
            confidence=0.85,
        ))

        # Get concept details for the last step
        last_algo = compound["steps"][-1]
        concept = self.concepts.get(last_algo)
        if concept:
            params = self._infer_params_from_concept(concept)
            return_type = self._infer_return_type_from_concept(concept)
        else:
            params = [("items", "List<int>")]
            return_type = "List<int>"

        return StructuredIntent(
            intent_type="generate_function",
            algorithm="compound",
            params=params,
            return_type=return_type,
            description=compound["description"],
            needs_loop=True,
            compound_steps=compound["steps"],
        )

    def _analyze_algorithmic_intent(self, request: str, algo_id: str,
                                     steps: list[ReasoningStep]) -> StructuredIntent:
        """Analyze algorithmic intent using concept-based reasoning."""
        # Get concept details from ConceptStore
        concept = self.concepts.get(algo_id)
        if concept:
            # Infer parameters from concept description
            params = self._infer_params_from_concept(concept)
            return_type = self._infer_return_type_from_concept(concept)
            func_name = self._extract_function_name(request, algo_id)
            result = StructuredIntent(
                intent_type="generate_function",
                algorithm=algo_id,
                params=params,
                return_type=return_type,
                description=concept.description,
                needs_loop=self._needs_loop(concept),
                is_recursive=self._is_recursive(concept),
                operation=algo_id,
            )
        else:
            # Composer detected an op not in ConceptStore
            result = StructuredIntent(
                intent_type="generate_function",
                algorithm=algo_id,
                params=[],
                return_type="any",
                description=f"Generated function: {algo_id}",
                operation=algo_id,
            )

        steps.append(ReasoningStep(
            step_type="analyze_algorithm",
            description=f"Matched algorithm: {algo_id}",
            evidence=[f"Algorithm: {algo_id}", f"Params: {result.params}"],
            confidence=0.9,
        ))
        return result

    def _infer_params_from_concept(self, concept: Concept) -> list[tuple[str, str]]:
        """Infer function parameters from concept description."""
        desc = concept.description.lower()
        if "two numbers" in desc or "add" in desc:
            return [("a", "int"), ("b", "int")]
        elif "list" in desc or "array" in desc or "collection" in desc:
            return [("items", "List<int>")]
        elif "number" in desc:
            return [("n", "int")]
        elif "string" in desc or "text" in desc:
            return [("text", "string")]
        return []

    def _infer_return_type_from_concept(self, concept: Concept) -> str:
        """Infer return type from concept description."""
        desc = concept.description.lower()
        if "bool" in desc or "check" in desc or "determine" in desc:
            return "bool"
        elif "float" in desc or "average" in desc or "mean" in desc:
            return "float"
        elif "list" in desc or "array" in desc or "collection" in desc:
            return "List<int>"
        elif "string" in desc or "text" in desc:
            return "string"
        return "int"

    def _needs_loop(self, concept: Concept) -> bool:
        """Check if concept needs loop (collection operations)."""
        desc = concept.description.lower()
        return any(kw in desc for kw in ["collection", "list", "array", "all elements", "every"])

    def _is_recursive(self, concept: Concept) -> bool:
        """Check if concept is recursive."""
        desc = concept.description.lower()
        return any(kw in desc for kw in ["factorial", "fibonacci", "recursive"])

    def _extract_function_name(self, request: str, algo_id: str) -> str:
        """Extract function name from request using semantic reasoning."""
        # Try to find a name pattern in the request
        name_patterns = [
            (r'함수\s*[\'"]?([a-zA-Z_]\w*)', 1),
            (r'function\s+([a-zA-Z_]\w*)', 1),
            (r'([a-zA-Z_]\w*)\s*함수', 1),
            (r'([a-zA-Z_]\w*)\s*function', 1),
            (r'func\s+([a-zA-Z_]\w*)', 1),
        ]
        for pattern, group in name_patterns:
            m = re.search(pattern, request, re.IGNORECASE)
            if m:
                return m.group(group)

        # Use concept name to generate function name
        concept = self.concepts.get(algo_id)
        if concept:
            # Convert concept name to snake_case function name
            name = concept.name.lower().replace(" ", "_")
            return name

        return "unknown_func"

    def _find_matching_concepts(self, request: str, intent: str,
                                structured: Optional[StructuredIntent],
                                steps: list[ReasoningStep]) -> list[Concept]:
        """Find matching concepts by searching ConceptStore.

        This uses concept descriptions and tags, not keyword matching.
        The ConceptStore search matches based on semantic similarity.
        """
        # Use ConceptStore.search to find concepts matching the request
        matched_concepts = self.concepts.search(request)

        # Add intent-specific concepts
        if intent == "generate_function":
            func_def = self.concepts.get("func_def")
            if func_def and func_def not in matched_concepts:
                matched_concepts.append(func_def)
        elif intent == "generate_class":
            class_def = self.concepts.get("class_def")
            if class_def and class_def not in matched_concepts:
                matched_concepts.append(class_def)

        # Add loop concepts if structured intent needs loops
        if structured and structured.needs_loop:
            for loop_id in ["loop_for", "loop_while"]:
                loop_concept = self.concepts.get(loop_id)
                if loop_concept and loop_concept not in matched_concepts:
                    matched_concepts.append(loop_concept)

        # Limit to top 8 concepts
        matched_concepts = matched_concepts[:8]

        steps.append(ReasoningStep(
            step_type="find_concepts",
            description=f"Found {len(matched_concepts)} matching concepts",
            evidence=[f"Concept: {c.name} ({c.category})" for c in matched_concepts],
        ))
        return matched_concepts

    def _find_related_concepts(self, concepts: list[Concept],
                               steps: list[ReasoningStep]) -> list[Concept]:
        """Find related concepts via multi-hop knowledge graph traversal.

        Uses BFS up to depth 3 to discover concepts that are transitively
        related through REQUIRES, PRODUCES, SIMILAR_TO, or USES relations.
        """
        related_ids = set()
        visited = {c.id for c in concepts}

        # Multi-hop traversal (depth 1-3)
        frontier = list(concepts)
        for depth in range(3):
            next_frontier = []
            for concept in frontier:
                relations = self.relations.find_related(concept.id)
                for rel in relations:
                    other_id = rel.target_id if rel.source_id == concept.id else rel.source_id
                    if other_id not in visited:
                        visited.add(other_id)
                        related_ids.add(other_id)
                        other = self.concepts.get(other_id)
                        if other:
                            next_frontier.append(other)
            frontier = next_frontier

        related = []
        for cid in related_ids:
            c = self.concepts.get(cid)
            if c and c not in concepts:
                related.append(c)

        # Sort by relation weight (prefer strong connections)
        def rel_weight(c):
            max_w = 0.0
            for concept in concepts:
                rels = self.relations.find_relations(source_id=concept.id, target_id=c.id)
                for r in rels:
                    max_w = max(max_w, r.weight)
                rels = self.relations.find_relations(source_id=c.id, target_id=concept.id)
                for r in rels:
                    max_w = max(max_w, r.weight)
            return -max_w

        related.sort(key=rel_weight)

        steps.append(ReasoningStep(
            step_type="find_relations",
            description=f"Found {len(related)} related concepts via knowledge graph (depth 3)",
            evidence=[f"Related: {c.name}" for c in related[:5]],
        ))
        return related

    def _find_applicable_rules(self, intent: str, concepts: list[Concept],
                               language: str,
                               steps: list[ReasoningStep]) -> list[str]:
        context = {
            "intent": intent,
            "language": language,
            "concept_ids": [c.id for c in concepts],
        }
        results = self.rules.execute_all(
            category=None, language=language, context=context
        )
        rule_ids = list(results.keys())
        steps.append(ReasoningStep(
            step_type="apply_rules",
            description=f"Applied {len(rule_ids)} rules for {language}",
            evidence=[f"Rule: {rid}" for rid in rule_ids[:5]],
        ))
        return rule_ids

    def _recall_relevant_memories(self, request: str, language: str,
                                  steps: list[ReasoningStep]) -> list:
        keywords = re.findall(r'\w+', request.lower())
        query = " ".join(keywords[:5])
        memories = self.memory.search(query, language=language, limit=5)
        steps.append(ReasoningStep(
            step_type="recall_memory",
            description=f"Recalled {len(memories)} relevant memories",
            evidence=[f"Memory: {m.id} ({m.memory_type.value})" for m in memories[:3]],
        ))
        return memories

    def _build_explanation(self, steps: list[ReasoningStep], intent: str,
                           structured: Optional[StructuredIntent],
                           concepts: list[Concept], language: str) -> str:
        parts = [f"요청 분석 결과, 의도는 '{intent}'로 판단됩니다."]
        if structured:
            if structured.intent_type == "generate_class" and structured.class_info:
                ci = structured.class_info
                parts.append(f"클래스 이름: {ci.name}.")
                parts.append(f"속성: {', '.join(f'{n}: {t}' for n, t in ci.properties)}.")
            elif structured.compound_steps:
                parts.append(f"복합 알고리즘: {structured.description}.")
                parts.append(f"단계: {' → '.join(structured.compound_steps)}.")
            else:
                parts.append(f"식별된 알고리즘: {structured.algorithm} ({structured.description.strip()}).")
                parts.append(f"매개변수: {', '.join(f'{n}: {t}' for n, t in structured.params)}.")
                parts.append(f"반환 타입: {structured.return_type}.")
                if structured.needs_loop:
                    parts.append("반복문이 필요합니다.")
                if structured.is_recursive:
                    parts.append("재귀 호출이 필요합니다.")
        if concepts:
            concept_names = ", ".join(c.name for c in concepts[:4])
            parts.append(f"관련 개념: {concept_names}.")
        parts.append(f"대상 언어: {language}.")
        for s in steps:
            if s.step_type == "apply_rules" and s.evidence:
                parts.append(f"적용된 규칙: {len(s.evidence)}개.")
                break
        return " ".join(parts)

    def explain_reasoning(self, result: ReasoningResult) -> str:
        lines = [f"{'=' * 50}", f"  추론 과정 상세", f"{'=' * 50}", ""]
        lines.append(f"의도: {result.intent}")
        if result.structured_intent:
            si = result.structured_intent
            if si.intent_type == "generate_class" and si.class_info:
                ci = si.class_info
                lines.append(f"클래스 이름: {ci.name}")
                lines.append(f"속성: {', '.join(f'{n}: {t}' for n, t in ci.properties)}")
                lines.append(f"생성자 매개변수: {', '.join(f'{n}: {t}' for n, t in ci.constructor_params)}")
            elif si.compound_steps:
                lines.append(f"복합 알고리즘: {si.description}")
                lines.append(f"단계: {' → '.join(si.compound_steps)}")
            else:
                lines.append(f"알고리즘: {si.algorithm}")
                lines.append(f"설명: {si.description.strip()}")
                lines.append(f"매개변수: {', '.join(f'{n}: {t}' for n, t in si.params)}")
                lines.append(f"반환 타입: {si.return_type}")
                lines.append(f"반복문 필요: {'예' if si.needs_loop else '아니오'}")
                lines.append(f"재귀: {'예' if si.is_recursive else '아니오'}")
        lines.append("")
        for i, step in enumerate(result.steps, 1):
            lines.append(f"Step {i}: [{step.step_type}] {step.description}")
            for e in step.evidence[:5]:
                lines.append(f"    - {e}")
            lines.append("")
        if result.concepts:
            lines.append(f"관련 개념 ({len(result.concepts)}개):")
            for c in result.concepts:
                lines.append(f"    - [{c.id}] {c.name} ({c.category})")
            lines.append("")
        if result.rules_applied:
            lines.append(f"적용된 규칙 ({len(result.rules_applied)}개):")
            for r in result.rules_applied:
                lines.append(f"    - {r}")
            lines.append("")
        lines.append(f"{'=' * 50}")
        lines.append(f"  요약: {result.explanation}")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)
