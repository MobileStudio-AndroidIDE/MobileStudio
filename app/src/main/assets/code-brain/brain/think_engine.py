"""ThinkEngine - genuine general reasoning engine for RENA-BR.

The ThinkEngine is the core intelligence layer. It is NOT a keyword router or a
table of hardcoded problem->solution mappings. Instead, it builds a structured
ProblemModel from the user's request, retrieves relevant knowledge from the
knowledge graph (concepts, relations, experiences), and synthesizes novel
solutions by *combining* existing knowledge.

Core loop:
    Understand -> Model -> Retrieve -> Decompose -> Hypothesize -> Evaluate
               -> Plan -> Execute -> Observe -> (Repair) -> Learn

The engine deliberately separates:
    - SemanticRepresentation (language-independent meaning) -> build_model
    - ProblemModel (goal, entities, operations, constraints)
    - Retrieval (concepts + relations + experiences, using graph traversal)
    - Hypotheses (candidate solutions, each with evidence + confidence)
    - Evaluation (comparison of hypotheses against constraints)
    - Plan (action sequence derived from the chosen hypothesis)
    - Adaptation (re-planning when observations contradict expectations)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from datetime import datetime
import uuid

from brain.concept_store import ConceptStore, Concept
from brain.relation_store import RelationStore
from brain.think_budget import current_budget, ThinkMode
from brain.rule_store import RuleEngine
from brain.memory_store import MemoryStore, MemoryType
from brain.neural_core import NeuralCore, get_neural_core, SDR, DenseVector, NEURAL_CORE_UNAVAILABLE, get_neural_core_status
from brain.decision_trace import DecisionTrace, Alternative, DecisionType


class GoalType(Enum):
    CREATE_FUNCTION = "create_function"
    CREATE_CLASS = "create_class"
    CREATE_PROJECT = "create_project"
    MODIFY = "modify"
    EXPLAIN = "explain"
    FIX = "fix"
    ANALYZE = "analyze"
    # The dialogue goal types share their VALUES with the semantic
    # representation's GoalType: ONE vocabulary across the pipeline, so
    # a goal type carried across the two modules still compares equal
    # to its own enum (a cross-enum member is never ==).
    QUESTION = "question"
    CHAT = "chat"
    COMPUTER = "computer"
    COMMAND_QUERY = "command_query"
    INFORMATION_QUERY = "information_query"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass
class StructuredIntent:
    """Structured intent for backward compatibility with old Reasoner."""
    intent_type: str = "generate_function"
    algorithm: str = ""
    params: list = field(default_factory=list)
    return_type: str = "Any"
    description: str = ""
    needs_loop: bool = False
    is_recursive: bool = False
    data_type: str = "int"
    operation: str = ""
    class_info: Any = None
    compound_steps: list = field(default_factory=list)


@dataclass
class ClassInfo:
    """Class information for backward compatibility with old Reasoner."""
    name: str = "User"
    properties: list = field(default_factory=list)
    constructor_params: list = field(default_factory=list)
    parent_class: str = ""
    interfaces: list = field(default_factory=list)
    methods: list = field(default_factory=list)


@dataclass
class Entity:
    name: str
    kind: str = "object"  # object, data, device, resource, collection
    attributes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "attributes": self.attributes}


@dataclass
class Operation:
    name: str
    verb: str = ""
    target: str = ""
    result: str = ""
    category: str = ""
    order: int = 0  # Position in the operation pipeline
    depends_on: list[str] = field(default_factory=list)  # Names of operations this depends on

    def to_dict(self) -> dict:
        return {"name": self.name, "verb": self.verb, "target": self.target,
                "result": self.result, "category": self.category,
                "order": self.order, "depends_on": self.depends_on}


@dataclass
class Constraint:
    name: str
    value: str = ""
    kind: str = ""  # language, platform, feature, performance, etc.

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "kind": self.kind}


@dataclass
class InputSpec:
    name: str
    type_name: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type_name, "description": self.description}


@dataclass
class OutputSpec:
    name: str
    type_name: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type_name, "description": self.description}


@dataclass
class SuccessCriteria:
    description: str
    check_type: str = ""  # "contains_operation", "has_return", "sort_order", etc.
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"description": self.description, "check_type": self.check_type,
                "params": self.params}


@dataclass
class Requirement:
    """A single requirement extracted from user input."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    category: str = ""  # "feature", "numeric", "quality", "dependency", "ui", "build"
    value: str = ""
    description: str = ""
    source_text: str = ""  # original text that generated this requirement
    priority: str = "required"  # "required", "preferred", "optional"
    satisfied: bool = False  # whether this requirement is satisfied in generated code

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "value": self.value, "description": self.description,
            "source_text": self.source_text, "priority": self.priority,
            "satisfied": self.satisfied,
        }


@dataclass
class ProblemModel:
    """Language-independent model of the user's actual goal."""
    goal_type: GoalType = GoalType.UNKNOWN
    summary: str = ""
    goal: str = ""
    entities: list[Entity] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    operation_order: list[str] = field(default_factory=list)  # Ordered operation names
    constraints: list[Constraint] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)  # All extracted requirements
    inputs: list[InputSpec] = field(default_factory=list)
    outputs: list[OutputSpec] = field(default_factory=list)
    success_criteria: list[SuccessCriteria] = field(default_factory=list)
    data_types: dict = field(default_factory=dict)  # entity_name -> type description
    conditions: list[dict] = field(default_factory=list)  # filter/sort/aggregate conditions
    domain: str = ""
    language: str = ""
    platform: str = ""
    raw_request: str = ""

    def to_dict(self) -> dict:
        return {
            "goal_type": self.goal_type.value,
            "summary": self.summary,
            "goal": self.goal,
            "entities": [e.to_dict() for e in self.entities],
            "operations": [o.to_dict() for o in self.operations],
            "operation_order": self.operation_order,
            "constraints": [c.to_dict() for c in self.constraints],
            "requirements": [r.to_dict() for r in self.requirements],
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "success_criteria": [s.to_dict() for s in self.success_criteria],
            "data_types": self.data_types,
            "conditions": self.conditions,
            "domain": self.domain,
            "language": self.language,
            "platform": self.platform,
        }


@dataclass
class Hypothesis:
    id: str
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    concepts: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    plan_hint: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "concepts": self.concepts,
            "operations": self.operations,
        }


@dataclass
class ThoughtStep:
    step_type: str
    description: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "step_type": self.step_type,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class Action:
    kind: str  # generate_file, add_file, generate_model, generate_service, ...
    params: dict = field(default_factory=dict)
    reason: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": self.params,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class ReasoningState:
    """Mutable state carried through a single reasoning session."""
    model: ProblemModel = field(default_factory=ProblemModel)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    selected_hypothesis: Optional[Hypothesis] = None
    steps: list[ThoughtStep] = field(default_factory=list)
    plan: list[Action] = field(default_factory=list)
    operation_plan: Any = None  # Phase 4: OperationPlan from semantic composition
    observations: list[dict] = field(default_factory=list)
    retrieved_concepts: list[Concept] = field(default_factory=list)
    retrieved_experiences: list = field(default_factory=list)
    final_solution: Any = None
    lesson: dict = field(default_factory=dict)
    confidence: float = 0.5
    intent: str = ""
    # The route this state was reasoned under (CODE / EXPLANATION /
    # ADVICE_LEARNING / CHAT): a non-CODE state carries no hypotheses.
    route: str = ""
    structured_intent: Any = None
    decisions: list[DecisionTrace] = field(default_factory=list)
    # The plan's provenance: the hypothesis the EXECUTED plan was
    # formulated from (the invariant selected_hypothesis ==
    # plan.source_hypothesis), and the alternative plans the budget
    # paid to formulate for the runner-ups (recorded for
    # verification/repair rotation, never executed in place of the
    # selected one).
    plan_source_hypothesis: str = ""
    alternative_plans: list[dict] = field(default_factory=list)

    def record(self, step_type: str, description: str, evidence: list[str] = None,
               confidence: float = 1.0):
        self.steps.append(ThoughtStep(step_type, description, evidence or [], confidence))

    def record_decision(self, decision: DecisionTrace):
        self.decisions.append(decision)

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "selected_hypothesis": self.selected_hypothesis.to_dict() if self.selected_hypothesis else None,
            "steps": [s.to_dict() for s in self.steps],
            "plan": [a.to_dict() for a in self.plan],
            "observations": self.observations,
            "decisions": [d.to_dict() for d in self.decisions],
            "plan_source_hypothesis": self.plan_source_hypothesis,
            "alternative_plans": [
                {"hypothesis": a.get("hypothesis"),
                 "plan": [x.to_dict() for x in a.get("plan", [])]}
                for a in self.alternative_plans
            ],
        }


def _data_operation_requirements(text_lower: str) -> list:
    """The data operations the request asks for, recorded as
    requirements.

    What the request must have DONE to its data is a requirement of
    the response: the same requirement whether the turn reached the
    reasoning as text or as a semantic representation."""
    import re
    data_op_patterns = {
        "calculate_average": ["average", "평균"],
        "filter_above": ["above", "이상", "초과", "높은"],
        "filter_below": ["below", "이하", "미만", "낮은"],
        "sort_descending": ["descending", "내림차순", "높은순"],
        "sort_ascending": ["ascending", "오름차순", "낮은순"],
        "remove_duplicates": ["duplicate", "중복", "dedup"],
        "top_n": [r"top\s*\d+", r"상위\s*\d+"],
    }
    requirements = []
    for op_name, keywords in data_op_patterns.items():
        match = next((k for k in keywords
                      if (re.search(k, text_lower) if chr(92) in k
                          else k in text_lower)), "")
        if match:
            requirements.append(Requirement(
                name=op_name, category="operation", value="required",
                description=f"Data operation: {op_name}",
                source_text=match,
            ))
    return requirements


class ThinkEngine:
    """General reasoning engine.

    The engine combines knowledge graph concepts and prior experiences to
    construct solutions it was never explicitly programmed to produce.
    """

    # Seed concepts that encode reusable, composable capability knowledge.
    # These are *knowledge*, not hardcoded answers: the engine uses them as
    # combinable building blocks during reasoning.
    CAPABILITY_CONCEPTS = [
        ("cap_data_storage", "Data Storage", "capability",
         "Store and retrieve records persistently",
         ["storage", "persistence", "database", "save", "records"]),
        ("cap_search", "Search", "capability",
         "Find records matching a query",
         ["search", "find", "query", "lookup", "filter"]),
        ("cap_crud", "CRUD Operations", "capability",
         "Create, read, update, delete records",
         ["add", "create", "remove", "delete", "update", "edit", "manage"]),
        ("cap_ui", "User Interface", "capability",
         "Present information and receive input",
         ["ui", "screen", "interface", "display", "preview"]),
        ("cap_camera", "Camera", "capability",
         "Capture images and video from a camera device",
         ["camera", "capture", "photo", "image"]),
        ("cap_image_processing", "Image Processing", "capability",
         "Analyze and transform image data",
         ["image", "detect", "recognize", "object detection"]),
        ("cap_calculation", "Calculation", "capability",
         "Perform arithmetic and numeric computation",
         ["calculate", "compute", "add", "subtract", "multiply", "divide"]),
        ("cap_file_ops", "File Operations", "capability",
         "Read, write, move, rename files",
         ["file", "organize", "move", "rename", "scan"]),
        ("cap_networking", "Networking", "capability",
         "Communicate over a network",
         ["network", "api", "server", "client", "request"]),
        ("cap_authentication", "Authentication", "capability",
         "Verify user identity and manage sessions",
         ["login", "auth", "user", "account", "password"]),
    ]

    # LEGACY: Entity kind hints from keyword matching.
    # source: legacy, usage: fallback+compatibility, confidence: 0.4
    # Primary path: SemanticParser entity extraction (concept activation)
    # Zero authority over entity kind when semantic pipeline is active.
    ENTITY_KIND_HINTS = {
        "student": "data", "students": "data", "contact": "data", "contacts": "data",
        "user": "data", "users": "data", "item": "data", "items": "data",
        "task": "data", "tasks": "data", "todo": "data", "record": "data",
        "records": "data", "employee": "data", "employees": "data",
        "product": "data", "products": "data", "order": "data", "orders": "data",
        "camera": "device", "image": "resource", "file": "resource",
        "files": "resource", "photo": "resource",
        # Korean entity types
        "학생": "data", "연락처": "data", "사용자": "data", "아이템": "data",
        "할일": "data", "기록": "data", "직원": "data", "상품": "data", "주문": "data",
        "자동차": "device", "이미지": "resource", "파일": "resource", "사진": "resource",
        "게임": "data", "영화": "data", "책": "data", "노래": "data",
        "_temperature": "data", "sensor": "device", "센서": "device",
        "서버": "device", "server": "device", "컴퓨터": "device", "computer": "device",
        "데이터": "data", "data": "data", "정보": "data", "info": "data",
        "리스트": "collection", "list": "collection", "배열": "collection", "array": "collection",
        "사전": "collection", "dict": "collection", "map": "collection",
        "문자열": "primitive", "string": "primitive", "숫자": "primitive", "number": "primitive",
        "문자": "primitive", "character": "primitive", "boolean": "primitive",
        "시간": "primitive", "time": "primitive", "날짜": "primitive", "date": "primitive",
        "이름": "primitive", "name": "primitive", "주소": "primitive", "address": "primitive",
        "전화": "primitive", "phone": "primitive", "이메일": "primitive", "email": "primitive",
        "가격": "primitive", "price": "primitive", "수량": "primitive", "quantity": "primitive",
        "점수": "primitive", "score": "primitive", "등급": "primitive", "grade": "primitive",
    }

    def __init__(self, concept_store: ConceptStore, relation_store: RelationStore,
                 rule_engine: RuleEngine, memory_store: MemoryStore):
        self.concepts = concept_store
        self.relations = relation_store
        self.rules = rule_engine
        self.memory = memory_store
        # The last reasoning state this engine produced — the trace's
        # window onto the hypotheses/plan of the turn just run.
        self.last_reasoning = None
        self.neural_core = get_neural_core()
        self._neural_core_available = self.neural_core is not NEURAL_CORE_UNAVAILABLE
        self._seed_capability_concepts()

    # ------------------------------------------------------------------ #
    # Knowledge seeding
    # ------------------------------------------------------------------ #
    def _seed_capability_concepts(self):
        existing_ids = {c.id for c in self.concepts.list_all()}
        for cid, name, cat, desc, tags in self.CAPABILITY_CONCEPTS:
            if cid in existing_ids:
                continue
            self.concepts.add(Concept(
                id=cid, name=name, category=cat, description=desc,
                tags=tags, languages=["python", "kotlin", "java", "javascript", "go", "rust"],
            ))
            existing_ids.add(cid)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def think(self, request: str, language: str = "python",
              context: dict = None) -> ReasoningState:
        """Legacy method - uses keyword-based extraction."""
        context = context or {}
        state = ReasoningState()
        state.model.language = language
        state.model.raw_request = request

        self._build_model(request, state)
        self._retrieve_knowledge(request, state)
        self._generate_hypotheses(state)
        self._evaluate_hypotheses(state)
        self._formulate_plan(state, context)
        self._derive_lesson(state)
        state.intent = state.model.goal_type.value if state.model.goal_type else "generate_function"
        intent_map = {
            "create_function": "generate_function",
            "create_class": "generate_class",
            "create_project": "generate_project",
            "modify": "modify_project",
            "explain": "explain",
            "fix": "fix_code",
            "analyze": "analyze_code",
            "unknown": "generate_function",
        }
        state.intent = intent_map.get(state.intent, state.intent)
        state.structured_intent = self._build_structured_intent(request, state)
        return state
    
    def think_from_semantic(self, semantic: 'SemanticRepresentation', language: str = "python",
                            context: dict = None) -> ReasoningState:
        """Build ProblemModel from SemanticRepresentation (semantic pipeline)."""
        context = context or {}
        state = ReasoningState()
        state.model.language = language
        state.model.raw_request = semantic.raw_text
        
        # Build model from semantic representation
        self._build_model_from_semantic(semantic, state)
        self._retrieve_knowledge(semantic.raw_text, state)

        # Phase 2 guard: filter retrieved knowledge to prevent mutation of user requirements
        from brain.nlu_priority import NLUPriorityResolver
        resolver = NLUPriorityResolver()
        user_ops = [op.name for op in state.model.operations]
        user_semantics = {"operations": user_ops,
                          "entities": [e.name for e in state.model.entities]}
        state.retrieved_concepts = resolver.filter_retrieved_knowledge(
            state.retrieved_concepts, user_semantics
        )

        # Phase 4: Semantic operation planning (compose primitives from semantics)
        from brain.operation_planner import OperationPlanner
        planner = OperationPlanner()
        operation_plan = planner.plan(semantic)
        state.operation_plan = operation_plan

        # The Code Hypothesis Generator runs ONLY for the CODE route.
        # A non-coding turn (a chat, an explanation, a learning
        # request) has no problem state: generating algorithm
        # hypotheses for it invents a coding intent the act never
        # expressed.
        from brain.route import Route, decide_route
        route = decide_route(semantic)
        state.route = route.value
        if route is Route.CODE:
            self._generate_hypotheses(state)
            self._evaluate_hypotheses(state)
            self._formulate_plan(state, context)
            self._derive_lesson(state)
        else:
            state.hypotheses = []
            state.plan = []
            state.selected_hypothesis = None
        state.intent = state.model.goal_type.value if state.model.goal_type else "generate_function"
        intent_map = {
            "create_function": "generate_function",
            "create_class": "generate_class",
            "create_project": "generate_project",
            "modify": "modify_project",
            "explain": "explain",
            "fix": "fix_code",
            "analyze": "analyze_code",
            "unknown": "generate_function",
        }
        state.intent = intent_map.get(state.intent, state.intent)
        state.structured_intent = self._build_structured_intent(semantic.raw_text, state)
        self.last_reasoning = state
        return state
    
    def _build_model_from_semantic(self, semantic: 'SemanticRepresentation', state: ReasoningState):
        """Build ProblemModel from SemanticRepresentation - NO keyword fallbacks."""
        model = state.model
        model.summary = semantic.raw_text.strip()[:200]
        model.goal = semantic.goal.target or semantic.raw_text.strip()[:200]
        model.raw_request = semantic.raw_text
        
        # Goal type from semantic — MAPPED into THIS module's GoalType.
        # The semantic representation carries its own enum member; an
        # enum from another module is never == this module's member,
        # and comparing them silently disabled every goal-type branch
        # below (the function plan was never even formulated).
        model.goal_type = self._map_goal_type(
            semantic.goal.goal_type) if semantic.goal else GoalType.UNKNOWN
        
        # Entities from semantic (user-derived only)
        model.entities = [
            Entity(name=e.name, kind=e.kind, attributes=e.attributes)
            for e in semantic.entities
        ]
        
        # Operations: prefer composition graph over legacy operation_graph
        comp = semantic.composition or {}
        comp_graph = comp.get("graph", {})
        comp_ops = comp_graph.get("operations", [])
        if comp_ops:
            # Use composer's operations (semantically correct, with proper deps)
            model.operation_order = [op["name"] for op in comp_ops]
            model.operations = [
                Operation(
                    name=op["name"],
                    verb=op["name"],
                    order=i,
                    depends_on=op.get("depends_on", []),
                )
                for i, op in enumerate(comp_ops)
            ]
        else:
            # Fallback to legacy operation_graph
            model.operation_order = [op.name for op in semantic.operation_graph.operations]
            model.operations = [
                Operation(
                    name=op.name,
                    verb=op.name,
                    order=i,
                    depends_on=op.depends_on,
                )
                for i, op in enumerate(semantic.operation_graph.operations)
            ]
        
        # Constraints from semantic
        model.constraints = [
            Constraint(name=c.name, value=c.value, kind=c.constraint_type)
            for c in semantic.constraints
        ]
        
        # Domain from semantic (evidence-based)
        model.domain = semantic.goal.domain
        model.domain_evidence = semantic.goal.domain_evidence
        
        # Language/Platform from constraints
        for c in semantic.constraints:
            if c.constraint_type == "language":
                model.language = c.value
            elif c.constraint_type == "platform":
                model.platform = c.value
        
        # Inputs/Outputs from semantic
        model.inputs = [
            InputSpec(name=io.name, type_name=io.type_signature, description=io.description)
            for io in semantic.inputs
        ]
        model.outputs = [
            OutputSpec(name=io.name, type_name=io.type_signature, description=io.description)
            for io in semantic.outputs
        ]
        
        # Success criteria from operations
        self._extract_success_criteria_from_semantic(semantic, model)
        
        # Conditions from semantic
        self._extract_conditions_from_semantic(semantic, model)
        
        # Data types from semantic entities
        self._extract_data_types_from_semantic(semantic, model)
        
        # Requirements from semantic
        self._extract_requirements_from_semantic(semantic, model)
        
        state.record(
            "understand",
            f"Built problem model from semantic: {model.goal_type.value} "
            f"(operations={model.operation_order}, "
            f"entities={[e.name for e in model.entities]}, "
            f"inputs={[i.type_name for i in model.inputs]}, "
            f"outputs={[o.type_name for o in model.outputs]}, "
            f"domain={model.domain})",
            [f"summary={model.summary}",
             f"operation_order={model.operation_order}",
             f"entities={[e.name for e in model.entities]}",
             f"domain_evidence={model.domain_evidence}"],
            0.9,
        )
    
    def _extract_success_criteria_from_semantic(self, semantic: 'SemanticRepresentation', model: ProblemModel):
        """Extract success criteria from semantic representation."""
        criteria_keywords = {
            "remove_duplicates": ["removes duplicates", "no duplicate", "unique",
                                  "distinct", "deduplicate", "removing duplicates"],
            "calculate_average": ["average", "mean", "average score", "average value"],
            "filter_above": ["above the average", "greater than the average",
                            "more than the average", "above average",
                            "greater than average", "values above"],
            "filter_below": ["below the average", "less than the average",
                            "below average", "less than average"],
            "sort_descending": ["descending order", "largest first", "highest first",
                               "sort descending", "order descending"],
            "sort_ascending": ["ascending order", "smallest first", "lowest first",
                              "sort ascending", "order ascending"],
        }
        
        rl = semantic.raw_text.lower()
        for op_name in model.operation_order:
            keywords = criteria_keywords.get(op_name, [])
            for kw in keywords:
                if kw in rl:
                    model.success_criteria.append(SuccessCriteria(
                        description=f"Must {op_name.replace('_', ' ')}",
                        check_type="contains_operation",
                        params={"operation": op_name}
                    ))
                    break
        
        if re.search(r"descending", rl):
            model.success_criteria.append(SuccessCriteria(
                description="Result must be sorted in descending order",
                check_type="sort_order",
                params={"order": "descending"}
            ))
        if re.search(r"ascending", rl):
            model.success_criteria.append(SuccessCriteria(
                description="Result must be sorted in ascending order",
                check_type="sort_order",
                params={"order": "ascending"}
            ))
        
        if model.goal_type == GoalType.CREATE_PROJECT:
            model.success_criteria.append(SuccessCriteria(
                description="Must have a working entry point / main function",
                check_type="has_entry_point",
                params={}
            ))
            model.success_criteria.append(SuccessCriteria(
                description="Must have data model / entity class",
                check_type="has_model",
                params={}
            ))
        elif model.goal_type == GoalType.CREATE_FUNCTION:
            model.success_criteria.append(SuccessCriteria(
                description="Must have a function definition",
                check_type="has_function",
                params={}
            ))
    
    def _extract_conditions_from_semantic(self, semantic: 'SemanticRepresentation', model: ProblemModel):
        """Extract conditions from semantic representation."""
        rl = semantic.raw_text.lower()
        
        filter_patterns = [
            (r"(above|greater|more|over|exceed|higher)\s+(than\s+)?(the\s+)?(average|mean)",
             "greater_than", "average"),
            (r"(below|less|under|lower)\s+(than\s+)?(the\s+)?(average|mean)",
             "less_than", "average"),
            (r"above\s+(\d+)", "greater_than", None),
            (r"greater\s+than\s+(\d+)", "greater_than", None),
            (r"less\s+than\s+(\d+)", "less_than", None),
            (r"below\s+(\d+)", "less_than", None),
        ]

        for pattern, comp, ref in filter_patterns:
            m = re.search(pattern, rl)
            if m:
                val = ref
                if not val and m.groups():
                    val = m.group(1)
                model.conditions.append({
                    "type": "filter",
                    "comparison": comp,
                    "reference": val,
                    "description": f"Filter values {comp.replace('_', ' ')} {val}"
                })
                break
        
        sort_patterns = [
            (r"descending\s+order", "descending", None),
            (r"ascending\s+order", "ascending", None),
            (r"sort(ed)?\s+by\s+(\w+)\s+(desc|asc|descending|ascending)",
             None, None),
        ]

        for pattern, order, field_name in sort_patterns:
            m = re.search(pattern, rl)
            if m:
                if order:
                    model.conditions.append({
                        "type": "sort",
                        "order": order,
                        "field": None,
                        "description": f"Sort in {order} order"
                    })
                else:
                    groups = m.groups()
                    if len(groups) >= 3:
                        field_name = groups[1]
                        order = "descending" if "desc" in (groups[2] or "") else "ascending"
                        model.conditions.append({
                            "type": "sort",
                            "order": order,
                            "field": field_name,
                            "description": f"Sort by {field_name} in {order} order"
                        })
                break
        
        agg_patterns = [
            (r"(average|mean)\s+(of\s+)?(\w+)", "average", r"\3"),
            (r"total\s+(of\s+)?(\w+)", "sum", r"\2"),
            (r"count\s+(of\s+)?(\w+)", "count", r"\2"),
            (r"max(imum)?\s+(of\s+)?(\w+)", "max", r"\3"),
            (r"min(imum)?\s+(of\s+)?(\w+)", "min", r"\3"),
        ]

        for pattern, agg_type, field_ref in agg_patterns:
            m = re.search(pattern, rl)
            if m:
                try:
                    fname = m.group(3) if isinstance(field_ref, str) and field_ref.startswith("\\") else (field_ref or "value")
                except (IndexError, AttributeError):
                    fname = "value"
                model.conditions.append({
                    "type": "aggregate",
                    "function": agg_type,
                    "field": fname,
                    "description": f"Calculate {agg_type} of {fname}"
                })
                break
    
    def _extract_data_types_from_semantic(self, semantic: 'SemanticRepresentation', model: ProblemModel):
        """Infer data types from semantic entities."""
        type_hints = {
            "score": "int", "scores": "List<int>",
            "number": "float", "numbers": "List<float>",
            "name": "str", "names": "List<str>",
            "student": "Student", "students": "List<Student>",
            "order": "Order", "orders": "List<Order>",
            "temperature": "float", "temperatures": "List<float>",
            "price": "float", "prices": "List<float>",
            "date": "datetime", "dates": "List<datetime>",
            "item": "Item", "items": "List<Item>",
            "task": "Task", "tasks": "List<Task>",
        }

        for ent in model.entities:
            ent_lower = ent.name.lower()
            if ent_lower in type_hints:
                model.data_types[ent.name] = type_hints[ent_lower]

        if any(o.name in ("calculate_average", "filter_above", "filter_below",
                          "sort_descending", "sort_ascending") for o in model.operations):
            if not model.data_types:
                model.data_types["input"] = "List<Number>"
    
    def _extract_requirements_from_semantic(self, semantic: 'SemanticRepresentation', model: ProblemModel):
        """Extract requirements from semantic representation."""
        import re
        rl = semantic.raw_text.lower()
        requirements = []

        numeric_patterns = [
            (r'(\d+)\s*[xX]\s*(zoom|줌|확대)', 'minimum_zoom', 'numeric', 'zoom'),
            (r'(\d+)\s*(times|배|배율)', 'multiplier', 'numeric', 'multiply'),
            (r'(top|상위|최고)\s*(\d+)', 'top_n', 'limit', 'sort'),
            (r'(bottom|하위|최하)\s*(\d+)', 'bottom_n', 'limit', 'sort'),
            (r'(\d+)\s*(items?|개|건)', 'count', 'limit', 'list'),
            (r'(최소|minimum|min)\s*(\d+)', 'minimum', 'range', 'filter'),
            (r'(최대|maximum|max)\s*(\d+)', 'maximum', 'range', 'filter'),
            (r'(\d+)%', 'percentage', 'threshold', 'filter'),
        ]
        for pattern, name, category, related_op in numeric_patterns:
            m = re.search(pattern, rl)
            if m:
                value = m.group(1) if m.lastindex else m.group(0)
                requirements.append(Requirement(
                    name=name, category=category, value=value,
                    description=f"Numeric requirement: {name}={value}",
                    source_text=m.group(0),
                ))

        feature_patterns = {
            "complete_code": (["complete code", "전체 코드", "완전한 코드", "complete", "전체", "완전"], "quality"),
            "buildable": (["buildable", "빌드 가능한", "compilable", "실행 가능한"], "quality"),
            "working": (["working", "작동하는", "동작하는"], "quality"),
            "test_code": (["test code", "테스트 코드", "tests", "테스트"], "quality"),
            "documentation": (["documentation", "문서", "readme", "주석"], "quality"),
        }
        for feat_name, (keywords, category) in feature_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=feat_name, category=category, value="required",
                    description=f"Feature requirement: {feat_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        camera_patterns = {
            "zoom": ["zoom", "줌", "확대"],
            "capture": ["capture", "촬영", "photo", "사진"],
            "video": ["video", "녹화", "영상"],
            "flash": ["flash", "플래시"],
            "filter": ["filter", "필터", "효과"],
            "gallery": ["gallery", "갤러리", "목록"],
            "permission": ["permission", "권한"],
            "storage": ["storage", "저장", "save"],
        }
        for feat_name, keywords in camera_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=f"camera_{feat_name}", category="feature", value="required",
                    description=f"Camera feature: {feat_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        ui_patterns = {
            "material_design": ["material", "머티리얼"],
            "jetpack_compose": ["compose", "jetpack compose", "젯팩"],
        }
        for feat_name, keywords in ui_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=feat_name, category="ui", value="required",
                    description=f"UI requirement: {feat_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        requirements.extend(_data_operation_requirements(rl))
        model.requirements = requirements

    def _build_structured_intent(self, request: str, state: ReasoningState) -> StructuredIntent:
        """Build a StructuredIntent from the reasoning state for backward compatibility."""
        model = state.model
        request_lower = request.lower()
        
        algorithm = ""
        params = [("data", "List[int]")]
        return_type = "int"
        needs_loop = False
        is_recursive = False
        data_type = "int"
        operation = ""
        compound_steps = []
        
# Detect algorithm from request (order matters - more specific first)
        algo_hints = {
            "최댓값": "find_max", "최대값": "find_max", "find_max": "find_max", "가장 큰": "find_max", "큰 값": "find_max",
            "최소값": "find_min", "find_min": "find_min", "가장 작은": "find_min", "작은 값": "find_min",
            "합계": "sum_list", "리스트 합": "sum_list", "리스트의 합": "sum_list", "sum": "sum_list",
            "평균": "average_list", "average": "average_list", "평균값": "average_list",
            "피보나치": "fibonacci", "fibonacci": "fibonacci", "피보나치 수열": "fibonacci",
            "팩토리얼": "factorial", "factorial": "factorial",
            "소수 판별": "is_prime", "소수": "is_prime", "prime": "is_prime",
            "이진 탐색": "binary_search", "binary search": "binary_search",
            "문자열 연결": "concat_strings", "연결": "concat_strings", "concat": "concat_strings",
            "문자열 합치기": "concat_strings", "결합": "concat_strings", "문자열 결합": "concat_strings",
            "정렬": "sort_list", "sort": "sort_list",
            "곱하기": "multiply", "곱하는": "multiply", "곱셈": "multiply",
            "제곱": "square",
            "문자열 검색": "search_in_string", "문자열 찾기": "search_in_string",
            "검색": "search_in_list", "찾기": "search_in_list", "search": "search_in_list",
            "더하기": "add", "더하는": "add", "덧셈": "add",
            "이하": "filter_below", "미만": "filter_below", "평균 이하": "filter_below", "평균 미만": "filter_below",
            "평균 이상": "filter_above", "평균 미만": "filter_below", "평균 이하": "filter_below",
            "필터": "filter_above", "필터링": "filter_above", "이상": "filter_above",
        }
        
        for hint, algo in algo_hints.items():
            if hint in request_lower:
                algorithm = algo
                break
        
        # Detect loop/recursion needs
        if any(k in request_lower for k in ["반복", "loop", "for", "while", " fib ", "팩토리얼"]):
            needs_loop = True
        if any(k in request_lower for k in ["팩토리얼", "factorial", "피보나치", "fibonacci", "재귀"]):
            is_recursive = True
        
        # Detect compound steps
        if any(k in request_lower for k in ["중복 제거", "중복을 제거"]):
            compound_steps.append("remove_duplicates")
        if any(k in request_lower for k in ["평균"]):
            compound_steps.append("calculate_average")
        if any(k in request_lower for k in ["평균 이상", "이상", "초과", "넘는"]):
            compound_steps.append("filter_above")
        if any(k in request_lower for k in ["이하", "미만", "미만인", "이하인"]):
            compound_steps.append("filter_below")
        if any(k in request_lower for k in ["정렬", "내림차순", "오름차순"]):
            compound_steps.append("sort_list")
        if any(k in request_lower for k in ["짝수", "홀수"]):
            compound_steps.append("filter_even")
        if any(k in request_lower for k in ["합계", "합", "더한"]):
            compound_steps.append("sum_list")
        
        # Detect class intent
        class_info = None
        is_class = state.model.goal_type and state.model.goal_type.value == "create_class"
        if is_class:
            class_info = self._extract_class_info(request)
            intent_type = "generate_class"
        else:
            intent_type = state.intent
        
        # Detect compound class+function
        if is_class and any(k in request_lower for k in ["함수", "function", "메소드", "method"]):
            compound_steps.append("class_and_function")
        
        # If multiple compound steps OR class_and_function step, mark as compound algorithm
        if len(compound_steps) > 1 or "class_and_function" in compound_steps:
            algorithm = "compound"

        # The OPERATION GRAPH itself can declare a compound pipeline: a
        # COORDINATION of transforms — an ordering or dedup alongside
        # another pattern — is compound regardless of the request's
        # language. Single-purpose pipelines keep their semantic name,
        # including internally decomposed ones (find_min's threshold
        # scan is one function, not a coordination of steps).
        op_names = [op.name for op in model.operations]
        pattern_ops = {"remove_duplicates", "sort_ascending", "sort_descending",
                       "calculate_average", "calculate_sum", "find_max",
                       "find_min", "filter_above", "filter_below", "filter_even",
                       "count_elements"}
        if ({"sort_ascending", "sort_descending", "remove_duplicates"}
                & set(op_names)) and len(pattern_ops & set(op_names)) > 1:
            algorithm = "compound"
        
        return StructuredIntent(
            intent_type=intent_type,
            algorithm=algorithm,
            params=params,
            return_type=return_type,
            description=request[:200],
            needs_loop=needs_loop,
            is_recursive=is_recursive,
            data_type=data_type,
            operation=operation,
            compound_steps=compound_steps,
            class_info=class_info,
        )

    def _extract_class_info(self, request: str) -> ClassInfo:
        """Extract class information from the request."""
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

    # ------------------------------------------------------------------ #
    # 1. Understand -> Model (structured extraction preserving meaning)
    # ------------------------------------------------------------------ #
    def _build_model(self, request: str, state: ReasoningState):
        model = state.model
        model.summary = request.strip()[:200]
        model.goal = self._extract_goal(request)
        model.raw_request = request

        # Use Neural Core for semantic understanding if available
        if self._neural_core_available:
            try:
                neural_state = self.neural_core.think(request, state.model.language, context={})

                # Extract structured understanding from Neural Core
                model.goal_type = self._map_goal_type(neural_state.model.goal_type)

                # Convert neural core entities (dicts) to Entity objects
                model.entities = [
                    Entity(name=e["name"], kind=e["kind"], attributes=e["attributes"])
                    for e in neural_state.model.entities
                ]
                model.constraints = [
                    Constraint(name=c.get("name", c[0] if isinstance(c, tuple) else ""),
                              value=c.get("value", c[1] if isinstance(c, tuple) else ""),
                              kind=c.get("kind", c[2] if isinstance(c, tuple) else ""))
                    for c in neural_state.model.constraints
                ]
                model.domain = neural_state.model.domain
                model.language = neural_state.model.language or model.language
                model.platform = neural_state.model.platform
            except Exception as e:
                # Neural Core failed, fall back to keyword-based
                self._neural_core_available = False
                state.record("warn", f"Neural Core failed, using fallback: {e}")
        else:
            # Neural Core not available - use keyword-based fallback
            state.record("info", "Using pure Python semantic/reasoning pipeline (Neural Core unavailable)")

        # --- Structured extraction: ordered operations, inputs, outputs, criteria ---
        self._extract_ordered_operations(request, model)
        self._extract_inputs_outputs(request, model)
        self._extract_success_criteria(request, model)
        self._extract_conditions(request, model)
        self._extract_data_types(request, model)
        self._extract_requirements(request, model)

        # Fallback to keyword-based for any missing fields
        if model.goal_type == GoalType.UNKNOWN:
            self._detect_goal_type(request, model)
        if not model.entities:
            self._extract_entities(request, model)
        if not model.operations:
            self._extract_operations(request, model)
        if not model.constraints:
            self._detect_constraints(request, model)
        if not model.domain:
            self._detect_domain(request, model)
        if not model.language:
            self._detect_language(request, model)
        if not model.platform:
            self._detect_platform(request, model)

        state.record(
            "understand",
            f"Built problem model: {model.goal_type.value} "
            f"(operations={model.operation_order}, "
            f"entities={[e.name for e in model.entities]}, "
            f"inputs={[i.type_name for i in model.inputs]}, "
            f"outputs={[o.type_name for o in model.outputs]})",
            [f"summary={model.summary}",
             f"operation_order={model.operation_order}",
             f"entities={[e.name for e in model.entities]}",
             f"success_criteria={[s.description for s in model.success_criteria]}"],
            0.9,
        )

    # ------------------------------------------------------------------ #
    # 1a. Extract ordered operations from natural language
    # ------------------------------------------------------------------ #
    def _extract_ordered_operations(self, request: str, model: ProblemModel):
        """Extract operations in the order they appear in the natural language."""
        rl = request.lower()

        # First, detect all operations with their positions
        found_ops = []
        seen_ops = set()

        op_patterns = [
            (r"remov(e|es|ing)\s+(duplicate|duplication|redundan)", "remove_duplicates"),
            (r"(deduplicate|dedup|deduplication)", "remove_duplicates"),
            (r"(distinct|unique|no duplicate)", "remove_duplicates"),
            (r"(remove|removing|removes)\s+(duplicate|dup)", "remove_duplicates"),
            (r"(average|mean)\b", "calculate_average"),
            (r"(calculat|comput)(e|es|ing)\s+(the\s+)?(average|mean)", "calculate_average"),
            (r"(get|find|compute)\s+(the\s+)?(average|mean)", "calculate_average"),
            (r"(averag|mean)\s+(score|value|number)", "calculate_average"),
            (r"(filter|filters|filtering|keep|keeping|keeps|select|selecting|selects)\s+.*?(above|greater|more|over|exceed|higher)", "filter_above"),
            (r"(filter|filters|filtering|keep|keeping|keeps|select|selecting|selects)\s+.*?(below|less|under|lower)", "filter_below"),
            (r"(value|number|item|element)s?\s+.*(above|greater|more|over|exceed|higher)", "filter_above"),
            (r"greater\s+than\s+(the\s+)?(average|mean)", "filter_above"),
            (r"above\s+(the\s+)?(average|mean)", "filter_above"),
            (r"(sort|sorting|sorts|order|ordering|orders)\s+.*?(descend|desc|largest|highest|biggest)", "sort_descending"),
            (r"(sort|sorting|sorts|order|ordering|orders)\s+.*?(ascend|asc|smallest|lowest)", "sort_ascending"),
            (r"descend(ing)?\s+order", "sort_descending"),
            (r"ascend(ing)?\s+order", "sort_ascending"),
            (r"(by|ordered by)\s+.*?(score|date|name|price|value)", "sort_by_field"),
        ]

        for pattern, op_name in op_patterns:
            match = re.search(pattern, rl)
            if match:
                if op_name not in seen_ops:
                    found_ops.append((match.start(), op_name))
                    seen_ops.add(op_name)

        # Sort by position in text
        found_ops.sort(key=lambda x: x[0])

        # Enforce dependency ordering: calculate_average must come before filter_above/filter_below
        ops_list = [op for _, op in found_ops]

        def enforce_order(ops, must_before, must_after):
            """Ensure must_before comes before must_after in the list."""
            if must_before in ops and must_after in ops:
                idx_before = ops.index(must_before)
                idx_after = ops.index(must_after)
                if idx_before > idx_after:
                    # Move must_before before must_after
                    ops.remove(must_before)
                    ops.insert(idx_after, must_before)

        enforce_order(ops_list, "calculate_average", "filter_above")
        enforce_order(ops_list, "calculate_average", "filter_below")
        enforce_order(ops_list, "remove_duplicates", "calculate_average")
        enforce_order(ops_list, "remove_duplicates", "filter_above")
        enforce_order(ops_list, "remove_duplicates", "filter_below")

        model.operation_order = ops_list

        # Create Operation objects with ordering
        for i, op_name in enumerate(model.operation_order):
            depends = []
            if i > 0:
                depends.append(model.operation_order[i - 1])
            model.operations.append(Operation(
                name=op_name,
                verb=op_name,
                order=i,
                depends_on=depends,
            ))

    # ------------------------------------------------------------------ #
    # 1b. Extract input/output specifications
    # ------------------------------------------------------------------ #
    def _extract_inputs_outputs(self, request: str, model: ProblemModel,
                               trace: 'ObservationTrace' = None):
        """Extract input/output types from natural language."""
        rl = request.lower()

        # Words that should NOT be treated as type names
        _THINK_STOP_WORDS = {
            "a", "an", "the", "it", "its", "this", "that", "these", "those",
            "function", "method", "class", "def", "return", "returns",
            "receives", "takes", "gets", "accepts", "input", "output",
            "parameter", "argument", "param", "arg",
            "의", "은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로",
            "것", "거", "수", "때", "곳", "쪽", "편", "쪽", "개", "명",
        }

        # Input patterns
        input_patterns = [
            (r"(receives?|takes?|gets?|accepts?)\s+(a\s+)?(list|array|sequence)\s+of\s+(\w+)",
             "list", r"\4"),
            (r"(receives?|takes?|gets?|accepts?)\s+(a\s+)?(\w+)\s+(list|array|sequence)",
             r"\3", "list"),
            (r"(receives?|takes?|gets?)\s+(a\s+)?(\w+)", r"\3", ""),
            (r"(input|inputs?)\s*:\s*(\w+)", r"\2", ""),
            (r"from\s+(a\s+)?(list|array|sequence)\s+of\s+(\w+)", "list", r"\3"),
            (r"list\s+of\s+(\w+)", "list", r"\1"),
            (r"(\w+)\s+list", r"\1", "list"),
        ]

        for pattern, type_hint, elem_hint in input_patterns:
            m = re.search(pattern, rl)
            if m:
                try:
                    type_name = m.group(1) if isinstance(type_hint, str) and type_hint.startswith("\\") else type_hint
                    elem = m.group(1) if isinstance(elem_hint, str) and elem_hint.startswith("\\") else elem_hint
                except (IndexError, AttributeError):
                    continue
                # Reject stop words as type names (prevents "List<Receives>" etc.)
                if elem and elem.lower() in _THINK_STOP_WORDS:
                    continue
                if type_name and isinstance(type_name, str) and type_name.lower() in _THINK_STOP_WORDS:
                    continue
                if elem:
                    type_name = f"List<{elem.capitalize()}>"
                elif isinstance(type_name, str) and not type_name.startswith("\\"):
                    type_name = type_name.capitalize()
                else:
                    continue
                model.inputs.append(InputSpec(name="input", type_name=type_name,
                                             description=f"Input: {type_name}"))
                break

        # Output patterns
        output_patterns = [
            (r"returns?\s+(them\s+)?(in\s+)?(descending|ascending)\s+order",
             "List", "sorted"),
            (r"returns?\s+(a\s+)?(list|array)\s+of\s+(\w+)", "list", r"\3"),
            (r"returns?\s+(\w+)", r"\1", ""),
            (r"output\s*:\s*(\w+)", r"\1", ""),
        ]

        for pattern, type_hint, extra in output_patterns:
            m = re.search(pattern, rl)
            if m:
                try:
                    if isinstance(type_hint, str) and type_hint.startswith("\\"):
                        type_name = m.group(1)
                    else:
                        type_name = type_hint
                    if isinstance(extra, str) and extra.startswith("\\"):
                        extra = m.group(int(extra[1])) if extra[1:].isdigit() else ""
                except (IndexError, AttributeError):
                    continue
                if extra == "sorted":
                    type_name = "List"
                elif extra and not extra.startswith("\\"):
                    type_name = f"List<{extra.capitalize()}>"
                elif isinstance(type_name, str) and not type_name.startswith("\\"):
                    type_name = type_name.capitalize()
                else:
                    type_name = "List"
                model.outputs.append(OutputSpec(name="result", type_name=type_name,
                                               description=f"Output: {type_name}"))
                break

        # If no explicit input/output found, infer from context
        if not model.inputs and model.entities:
            ent = model.entities[0]
            if any(o.name in ("remove_duplicates", "filter_above", "sort_descending",
                              "calculate_average", "filter", "sort") for o in model.operations):
                model.inputs.append(InputSpec(name="input", type_name="List<Number>",
                                             description="Input list of numbers"))
                model.outputs.append(OutputSpec(name="result", type_name="List<Number>",
                                               description="Processed list"))

        if trace:
            for inp in model.inputs:
                trace.record_type(
                    module="think_engine._extract_inputs_outputs",
                    name="input", inferred_type=inp.type_name,
                )
            for out in model.outputs:
                trace.record_type(
                    module="think_engine._extract_inputs_outputs",
                    name="output", inferred_type=out.type_name,
                )

    # ------------------------------------------------------------------ #
    # 1c. Extract success criteria
    # ------------------------------------------------------------------ #
    def _extract_success_criteria(self, request: str, model: ProblemModel):
        """Extract what success looks like from the natural language."""
        rl = request.lower()

        criteria_keywords = {
            "remove_duplicates": ["removes duplicates", "no duplicate", "unique",
                                  "distinct", "deduplicate", "removing duplicates"],
            "calculate_average": ["average", "mean", "average score", "average value"],
            "filter_above": ["above the average", "greater than the average",
                            "more than the average", "above average",
                            "greater than average", "values above"],
            "filter_below": ["below the average", "less than the average",
                            "below average", "less than average"],
            "sort_descending": ["descending order", "largest first", "highest first",
                               "sort descending", "order descending"],
            "sort_ascending": ["ascending order", "smallest first", "lowest first",
                              "sort ascending", "order ascending"],
            "add": ["add", "create", "insert"],
            "remove": ["remove", "delete", "eliminate"],
            "search": ["search", "find", "query", "lookup"],
            "store": ["store", "save", "persist"],
        }

        for op_name in model.operation_order:
            keywords = criteria_keywords.get(op_name, [])
            for kw in keywords:
                if kw in rl:
                    model.success_criteria.append(SuccessCriteria(
                        description=f"Must {op_name.replace('_', ' ')}",
                        check_type="contains_operation",
                        params={"operation": op_name}
                    ))
                    break

        # Add criteria for explicit conditions
        if re.search(r"descending", rl):
            model.success_criteria.append(SuccessCriteria(
                description="Result must be sorted in descending order",
                check_type="sort_order",
                params={"order": "descending"}
            ))
        if re.search(r"ascending", rl):
            model.success_criteria.append(SuccessCriteria(
                description="Result must be sorted in ascending order",
                check_type="sort_order",
                params={"order": "ascending"}
            ))

        # General success criteria based on goal type
        if model.goal_type == GoalType.CREATE_PROJECT:
            model.success_criteria.append(SuccessCriteria(
                description="Must have a working entry point / main function",
                check_type="has_entry_point",
                params={}
            ))
            model.success_criteria.append(SuccessCriteria(
                description="Must have data model / entity class",
                check_type="has_model",
                params={}
            ))
        elif model.goal_type == GoalType.CREATE_FUNCTION:
            model.success_criteria.append(SuccessCriteria(
                description="Must have a function definition",
                check_type="has_function",
                params={}
            ))

    # ------------------------------------------------------------------ #
    # 1d. Extract conditions (filter/sort/aggregate)
    # ------------------------------------------------------------------ #
    def _extract_conditions(self, request: str, model: ProblemModel):
        """Extract conditional logic from natural language."""
        rl = request.lower()

        # Filter conditions
        filter_patterns = [
            (r"(above|greater|more|over|exceed|higher)\s+(than\s+)?(the\s+)?(average|mean)",
             "greater_than", "average"),
            (r"(below|less|under|lower)\s+(than\s+)?(the\s+)?(average|mean)",
             "less_than", "average"),
            (r"above\s+(\d+)", "greater_than", None),
            (r"greater\s+than\s+(\d+)", "greater_than", None),
            (r"less\s+than\s+(\d+)", "less_than", None),
            (r"below\s+(\d+)", "less_than", None),
        ]

        for pattern, comp, ref in filter_patterns:
            m = re.search(pattern, rl)
            if m:
                val = ref
                if not val and m.groups():
                    val = m.group(1)
                model.conditions.append({
                    "type": "filter",
                    "comparison": comp,
                    "reference": val,
                    "description": f"Filter values {comp.replace('_', ' ')} {val}"
                })
                break

        # Sort conditions
        sort_patterns = [
            (r"descending\s+order", "descending", None),
            (r"ascending\s+order", "ascending", None),
            (r"sort(ed)?\s+by\s+(\w+)\s+(desc|asc|descending|ascending)",
             None, None),
        ]

        for pattern, order, field_name in sort_patterns:
            m = re.search(pattern, rl)
            if m:
                if order:
                    model.conditions.append({
                        "type": "sort",
                        "order": order,
                        "field": None,
                        "description": f"Sort in {order} order"
                    })
                else:
                    # Extract field and direction
                    groups = m.groups()
                    if len(groups) >= 3:
                        field_name = groups[1]
                        order = "descending" if "desc" in (groups[2] or "") else "ascending"
                        model.conditions.append({
                            "type": "sort",
                            "order": order,
                            "field": field_name,
                            "description": f"Sort by {field_name} in {order} order"
                        })
                break

        # Aggregate conditions
        agg_patterns = [
            (r"(average|mean)\s+(of\s+)?(\w+)", "average", r"\3"),
            (r"total\s+(of\s+)?(\w+)", "sum", r"\2"),
            (r"count\s+(of\s+)?(\w+)", "count", r"\2"),
            (r"max(imum)?\s+(of\s+)?(\w+)", "max", r"\3"),
            (r"min(imum)?\s+(of\s+)?(\w+)", "min", r"\3"),
        ]

        for pattern, agg_type, field_ref in agg_patterns:
            m = re.search(pattern, rl)
            if m:
                try:
                    fname = m.group(3) if isinstance(field_ref, str) and field_ref.startswith("\\") else (field_ref or "value")
                except (IndexError, AttributeError):
                    fname = "value"
                model.conditions.append({
                    "type": "aggregate",
                    "function": agg_type,
                    "field": fname,
                    "description": f"Calculate {agg_type} of {fname}"
                })
                break

    # ------------------------------------------------------------------ #
    # 1e. Extract data types
    # ------------------------------------------------------------------ #
    def _extract_data_types(self, request: str, model: ProblemModel):
        """Infer data types from context."""
        rl = request.lower()

        type_hints = {
            "score": "int", "scores": "List<int>",
            "number": "float", "numbers": "List<float>",
            "name": "str", "names": "List<str>",
            "student": "Student", "students": "List<Student>",
            "order": "Order", "orders": "List<Order>",
            "temperature": "float", "temperatures": "List<float>",
            "price": "float", "prices": "List<float>",
            "date": "datetime", "dates": "List<datetime>",
            "item": "Item", "items": "List<Item>",
            "task": "Task", "tasks": "List<Task>",
        }

        for ent in model.entities:
            ent_lower = ent.name.lower()
            if ent_lower in type_hints:
                model.data_types[ent.name] = type_hints[ent_lower]

        # Infer from operations
        if any(o.name in ("calculate_average", "filter_above", "filter_below",
                          "sort_descending", "sort_ascending") for o in model.operations):
            if not model.data_types:
                model.data_types["input"] = "List<Number>"

    def _map_goal_type(self, neural_goal) -> GoalType:
        """Map neural core goal type to ThinkEngine GoalType"""
        # Handle enum from neural core / semantic representation
        goal_str = neural_goal.value if hasattr(neural_goal, 'value') else str(neural_goal)
        try:
            return GoalType(goal_str)
        except ValueError:
            return GoalType.UNKNOWN

    def _extract_goal(self, request: str) -> str:
        # Strip leading imperative verbs to isolate the goal phrase.
        verbs = r"^(create|build|make|develop|write|implement|design|add|remove|fix|explain|analyze|generate)\s+(a\s+|an\s+)?"
        return re.sub(verbs, "", request.strip(), flags=re.IGNORECASE).strip()

    # LEGACY: Goal type detection from keywords.
    # source: legacy, usage: dead code (semantic pipeline is primary), confidence: 0.4
    # Primary path: SemanticParser goal detection via concept activation + composition
    def _detect_goal_type(self, request: str, model: ProblemModel):
        rl = request.lower()
        if any(k in rl for k in ["project", "app", "application", "system", "program",
                                 "software", "tool", "manager", "organizer", "calculator",
                                 "todo", "website", "프로젝트", "앱", "프로그램", "관리"]):
            model.goal_type = GoalType.CREATE_PROJECT
        elif any(k in rl for k in ["fix", "repair", "debug", "error", "bug", "고쳐", "오류"]):
            model.goal_type = GoalType.FIX
        elif any(k in rl for k in ["explain", "describe", "what does", "설명", "알려줘"]):
            model.goal_type = GoalType.EXPLAIN
        elif any(k in rl for k in ["class", "struct", "클래스", "객체"]):
            model.goal_type = GoalType.CREATE_CLASS
        elif any(k in rl for k in ["function", "method", "def ", "함수", "메소드", "메서드"]):
            model.goal_type = GoalType.CREATE_FUNCTION
        elif any(k in rl for k in ["analyze", "review", "분석"]):
            model.goal_type = GoalType.ANALYZE
        else:
            model.goal_type = GoalType.UNKNOWN

    # LEGACY: Constraint detection from keywords.
    # source: legacy, usage: dead code (semantic pipeline is primary), confidence: 0.4
    # Primary path: SemanticParser constraint extraction
    def _detect_constraints(self, request: str, model: ProblemModel):
        rl = request.lower()
        lang_kw = {
            "python": ["python", "파이썬"], "kotlin": ["kotlin", "코틀린"],
            "java": ["java", "자바"], "javascript": ["javascript", "자바스크립트", " node"],
            "go": ["golang", "go language"], "rust": ["rust", "러스트"],
            "c": ["c language", "c언어"], "cpp": ["c++", "cpp"],
        }
        for lang, kws in lang_kw.items():
            if any(k in rl for k in kws):
                model.language = lang
                model.constraints.append(Constraint("language", lang, "language"))
                break
        plat_kw = {
            "android": ["android", "안드로이드", "mobile", "모바일"],
            "web": ["web", "웹", "browser", "website"],
            "cli": ["cli", "command line", "명령줄", "console", "terminal", "터미널"],
            "backend": ["backend", "server", "서버", "api"],
        }
        for plat, kws in plat_kw.items():
            if any(k in rl for k in kws):
                model.platform = plat
                model.constraints.append(Constraint("platform", plat, "platform"))
                break

    # LEGACY: Language detection from keywords.
    # source: legacy, usage: dead code (semantic pipeline is primary), confidence: 0.4
    # Primary path: SemanticParser constraint extraction
    def _detect_language(self, request: str, model: ProblemModel):
        """Infer the target language from the request if not already known."""
        rl = request.lower()
        lang_kw = {
            "python": ["python", "파이썬"], "kotlin": ["kotlin", "코틀린"],
            "java": ["java", "자바"], "javascript": ["javascript", "자바스크립트", " node", "js"],
            "go": ["golang", "go language"], "rust": ["rust", "러스트"],
            "c": ["c language", "c언어"], "cpp": ["c++", "cpp"],
        }
        for lang, kws in lang_kw.items():
            if any(k in rl for k in kws):
                model.language = lang
                return

    # LEGACY: Platform detection from keywords.
    # source: legacy, usage: dead code (semantic pipeline is primary), confidence: 0.4
    # Primary path: SemanticParser constraint extraction
    def _detect_platform(self, request: str, model: ProblemModel):
        """Infer the target platform from the request if not already known."""
        rl = request.lower()
        plat_kw = {
            "android": ["android", "안드로이드", "mobile", "모바일"],
            "web": ["web", "웹", "browser", "website", "사이트"],
            "cli": ["cli", "command line", "명령줄", "console", "terminal", "터미널"],
            "backend": ["backend", "server", "서버", "api"],
        }
        for plat, kws in plat_kw.items():
            if any(k in rl for k in kws):
                model.platform = plat
                return

    def _extract_requirements(self, request: str, model: ProblemModel):
        """Extract all requirements from request and preserve in model.requirements."""
        import re
        rl = request.lower()
        requirements = []

        # Numeric requirements (10X zoom, top 3, etc.)
        numeric_patterns = [
            (r'(\d+)\s*[xX]\s*(zoom|줌|확대)', 'minimum_zoom', 'numeric', 'zoom'),
            (r'(\d+)\s*(times|배|배율)', 'multiplier', 'numeric', 'multiply'),
            (r'(top|상위|최고)\s*(\d+)', 'top_n', 'limit', 'sort'),
            (r'(bottom|하위|최하)\s*(\d+)', 'bottom_n', 'limit', 'sort'),
            (r'(\d+)\s*(items?|개|건)', 'count', 'limit', 'list'),
            (r'(최소|minimum|min)\s*(\d+)', 'minimum', 'range', 'filter'),
            (r'(최대|maximum|max)\s*(\d+)', 'maximum', 'range', 'filter'),
            (r'(\d+)%', 'percentage', 'threshold', 'filter'),
        ]
        for pattern, name, category, related_op in numeric_patterns:
            m = re.search(pattern, rl)
            if m:
                value = m.group(1) if m.lastindex else m.group(0)
                requirements.append(Requirement(
                    name=name, category=category, value=value,
                    description=f"Numeric requirement: {name}={value}",
                    source_text=m.group(0),
                ))

        # Feature requirements
        feature_patterns = {
            "complete_code": (["complete code", "전체 코드", "완전한 코드", "complete", "전체", "완전"], "quality"),
            "buildable": (["buildable", "빌드 가능한", "compilable", "실행 가능한"], "quality"),
            "working": (["working", "작동하는", "동작하는"], "quality"),
            "test_code": (["test code", "테스트 코드", "tests", "테스트"], "quality"),
            "documentation": (["documentation", "문서", "readme", "주석"], "quality"),
        }
        for feat_name, (keywords, category) in feature_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=feat_name, category=category, value="required",
                    description=f"Feature requirement: {feat_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        # Camera-specific requirements
        camera_patterns = {
            "zoom": ["zoom", "줌", "확대"],
            "capture": ["capture", "촬영", "photo", "사진"],
            "video": ["video", "녹화", "영상"],
            "flash": ["flash", "플래시"],
            "filter": ["filter", "필터", "효과"],
            "gallery": ["gallery", "갤러리", "목록"],
            "permission": ["permission", "권한"],
            "storage": ["storage", "저장", "save"],
        }
        for feat_name, keywords in camera_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=f"camera_{feat_name}", category="feature", value="required",
                    description=f"Camera feature: {feat_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        # UI requirements
        ui_patterns = {
            "material_design": ["material", "머티리얼"],
            "jetpack_compose": ["compose", "jetpack compose", "젯팩"],
            "xml_layout": ["xml layout", "xml"],
        }
        for ui_name, keywords in ui_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=ui_name, category="ui", value="required",
                    description=f"UI requirement: {ui_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        # Dependency requirements
        dep_patterns = {
            "cameraX": ["camerax", "camera x"],
            "room": ["room", "로컬 저장소"],
            "retrofit": ["retrofit", "네트워크"],
            "hilt": ["hilt", "dagger"],
            "navigation": ["navigation", "네비게이션"],
            "coroutines": ["coroutine", "코루틴", "async"],
        }
        for dep_name, keywords in dep_patterns.items():
            if any(k in rl for k in keywords):
                requirements.append(Requirement(
                    name=f"dependency_{dep_name}", category="dependency", value=dep_name,
                    description=f"Dependency requirement: {dep_name}",
                    source_text=next(k for k in keywords if k in rl),
                ))

        # Data operation requirements (average, filter, sort, etc.)
        requirements.extend(_data_operation_requirements(rl))

        model.requirements = requirements

    def _extract_entities(self, request: str, model: ProblemModel):
        tokens = re.findall(r"[A-Za-z가-힣]+", request)
        seen = set()
        
        # Context-based type inference keywords
        context_type_hints = {
            "list": ["리스트", "목록", "배열", "array", "list", "sequence", "순서"],
            "collection": ["컬렉션", "collection", "묶음", "그룹", "group"],
            "map": ["사전", "dict", "map", "mapping", "매핑"],
            "string": ["문자열", "string", "text", "텍스트"],
            "number": ["숫자", "number", "값", "value"],
            "integer": ["정수", "integer", "int"],
            "float": ["실수", "float", "소수점"],
            "boolean": ["불리언", "boolean", "참거짓", "true", "false"],
        }
        
        for t in tokens:
            tl = t.lower()
            if tl in self.ENTITY_KIND_HINTS:
                name = tl.rstrip("s").capitalize()
                if name in seen:
                    continue
                seen.add(name)
                model.entities.append(Entity(
                    name=name,
                    kind=self.ENTITY_KIND_HINTS[tl],
                ))
            # Context-based type inference: if token looks like a type hint
            elif tl in context_type_hints:
                kind = context_type_hints[tl]
                name = tl.capitalize()
                if name not in seen:
                    seen.add(name)
                    model.entities.append(Entity(
                        name=name,
                        kind=kind,
                    ))
        
        # Infer types from surrounding context
        request_lower = request.lower()
        
        # If "list of X" pattern detected, infer collection entity
        list_patterns = [
            r"(리스트|list|배열|array|목록)\s*(의|of|에)?\s*([a-zA-Z가-힣]+)",
            r"([a-zA-Z가-힣]+)\s*(리스트|list|배열|array|목록)",
        ]
        for pattern in list_patterns:
            matches = re.finditer(pattern, request_lower)
            for m in matches:
                # Get the entity type from context
                for group in m.groups():
                    if group and group not in ["리스트", "list", "배열", "array", "목록", "의", "of", "에"]:
                        entity_name = group.capitalize()
                        if entity_name not in seen:
                            seen.add(entity_name)
                            model.entities.append(Entity(
                                name=entity_name,
                                kind="data",
                            ))
        
        # If "map of X to Y" pattern detected, infer map entity
        map_patterns = [
            r"(사전|dict|map)\s*(의|of|에)?\s*([a-zA-Z가-힣]+)\s*(에서|to)\s*([a-zA-Z가-힣]+)",
        ]
        for pattern in map_patterns:
            matches = re.finditer(pattern, request_lower)
            for m in matches:
                groups = [g for g in m.groups() if g and g not in ["사전", "dict", "map", "의", "of", "에", "에서", "to"]]
                for g in groups:
                    entity_name = g.capitalize()
                    if entity_name not in seen:
                        seen.add(entity_name)
                        model.entities.append(Entity(
                            name=entity_name,
                            kind="data",
                        ))

    def _extract_operations(self, request: str, model: ProblemModel,
                           trace: 'ObservationTrace' = None):
        rl = request.lower()
        # LEGACY: keyword-to-operation mapping
        # source: legacy, usage: fallback+compatibility, confidence: 0.5
        # Primary path: SemanticComposer.compose() in semantic_parser.py
        op_kw = {
            "add": ["add", "create", "register", "추가", "등록", "생성", "더하기", "덧셈", "삽입"],
            "remove": ["remove", "delete", "삭제", "제거", "지우기", " 없애기"],
            "update": ["update", "edit", "modify", "수정", "변경", "갱신"],
            "search": ["search", "find", "query", "lookup", "검색", "탐색", "찾기"],
            "list": ["list", "show", "display", "목록", "보기", "나열"],
            "store": ["store", "save", "persist", "저장"],
            "calculate": ["calculate", "compute", "계산", "구하기", "찾기"],
            "organize": ["organize", "sort", "정리", "정렬"],
            "move": ["move", "이동"],
            "rename": ["rename", "이름변경"],
            "scan": ["scan", "스캔"],
            "capture": ["capture", "photo", "촬영", "사진"],
            "manage": ["manage", "관리", "management"],
            "multiply": ["multiply", "times", "곱하기", "곱셈"],
            "subtract": ["subtract", "minus", "빼기", "뺄셈"],
            "divide": ["divide", "나누기", "나눗셈"],
            "filter": ["filter", "골라", "걸러", "선택", "여과"],
            "transform": ["transform", "변환", "바꾸기"],
            "aggregate": ["aggregate", "집계", "누적"],
            "compare": ["compare", "비교", "대조"],
            "validate": ["validate", "검증", "확인"],
            "format": ["format", "포맷", "형식화"],
            "parse": ["parse", "파싱", "분석"],
            "encode": ["encode", "인코딩", "암호화"],
            "decode": ["decode", "디코딩", "복호화"],
            "compress": ["compress", "압축"],
            "decompress": ["decompress", "해제"],
            "encrypt": ["encrypt", "암호화"],
            "decrypt": ["decrypt", "복호화"],
            "send": ["send", "보내기", "발송"],
            "receive": ["receive", "받기", "수신"],
            "connect": ["connect", "연결"],
            "disconnect": ["disconnect", "해제", "끊기"],
            "load": ["load", "로드", "불러오기"],
            "unload": ["unload", "언로드", "내리기"],
            "start": ["start", "시작", "시작하기"],
            "stop": ["stop", "중지", "멈추기"],
            "pause": ["pause", "일시정지"],
            "resume": ["resume", "재개"],
            "retry": ["retry", "재시도", "다시시도"],
            "sort": ["sort", "정렬", "소팅", "순서대로"],
            "reverse": ["reverse", "뒤집기", "거꾸로", "역순"],
            "flatten": ["flatten", "평탄화", "펼치기"],
            "unique": ["unique", "distinct", "중복제거", "고유"],
            "count": ["count", "세기", "카운트"],
            "sum": ["sum", "합계", "총합"],
            "average": ["average", "mean", "평균"],
            "min": ["min", "minimum", "최소", "가장작은"],
            "max": ["max", "maximum", "최대", "가장큰"],
            "prime": ["prime", "소수", "素数"],
            "factorial": ["factorial", "팩토리얼", "계승"],
            "fibonacci": ["fibonacci", "피보나치"],
            "palindrome": ["palindrome", "회문", "거울문자"],
            "binary_search": ["binary search", "이진탐색", "이진 검색"],
            "linear_search": ["linear search", "선형탐색", "선형 검색"],
            "bubble_sort": ["bubble sort", "버블정렬", "버블 소팅"],
            "merge_sort": ["merge sort", "병합정렬", "머지소팅"],
            "quick_sort": ["quick sort", "퀵정렬", "퀵소팅"],
        }
        for op, kws in op_kw.items():
            if any(k in rl for k in kws):
                model.operations.append(Operation(name=op, verb=op))

        # CRUD filter: for CREATE_FUNCTION goals, strip generic CRUD operations
        # that leak from verbs like "생성"(create), "추가"(add), "수정"(modify).
        # These are implementation verbs, not algorithmic operations.
        if model.goal_type == GoalType.CREATE_FUNCTION:
            crud_ops = {"add", "remove", "update", "search", "list", "store",
                        "manage", "send", "receive", "connect", "disconnect",
                        "load", "unload", "start", "stop", "pause", "resume"}
            filtered_out = [op.name for op in model.operations if op.name in crud_ops]
            model.operations = [op for op in model.operations if op.name not in crud_ops]
            if trace and filtered_out:
                for name in filtered_out:
                    trace.record_operation(
                        module="think_engine._extract_operations",
                        name=name, source="op_kw",
                        was_filtered=True,
                        filter_reason="CRUD filter for CREATE_FUNCTION goal",
                    )

        if trace:
            for op in model.operations:
                trace.record_operation(
                    module="think_engine._extract_operations",
                    name=op.name, source="op_kw",
                )

        # "manage" is kept as-is; requirements are extracted from actual request text.

    def _detect_domain(self, request: str, model: ProblemModel):
        """Infer domain from semantic evidence in the ProblemModel."""
        # Collect evidence from entities, operations, constraints
        entity_kinds = {e.kind for e in model.entities}
        entity_names = {e.name.lower() for e in model.entities}
        operation_names = {op.name for op in model.operations}
        
        # Domain inference from semantic evidence
        domain_scores = {}
        
        # From entity names
        domain_entity_map = {
            "student": "education", "학생": "education",
            "grade": "education", "점수": "education", "score": "education",
            "employee": "hr", "직원": "hr", "salary": "hr", "급여": "hr",
            "product": "ecommerce", "상품": "ecommerce",
            "order": "ecommerce", "주문": "ecommerce",
            "shopping": "ecommerce", "쇼핑": "ecommerce",
            "cart": "ecommerce", "장바구니": "ecommerce",
            "camera": "camera", "카메라": "camera",
            "photo": "camera", "사진": "camera",
            "image": "camera", "이미지": "camera",
            "video": "camera", "영상": "camera",
            "android": "android", "안드로이드": "android",
            "ios": "ios", "아이폰": "ios", "iphone": "ios",
            "web": "web", "웹": "web", "website": "web", "사이트": "web",
            "blog": "blog", "블로그": "blog", "post": "blog", "게시물": "blog",
            "chat": "messaging", "채팅": "messaging",
            "message": "messaging", "메시지": "messaging",
            "messenger": "messaging", "메신저": "messaging",
            "todo": "productivity", "할일": "productivity",
            "task": "productivity", "태스크": "productivity",
            "schedule": "productivity", "일정": "productivity",
            "calendar": "productivity", "캘린더": "productivity",
            "file": "file_system", "파일": "file_system",
            "folder": "file_system", "폴더": "file_system",
            "directory": "file_system", "디렉토리": "file_system",
            "calculator": "calculator", "계산기": "calculator", "calc": "calculator",
            "game": "game", "게임": "game",
            "book": "library", "책": "library", "library": "library", "도서관": "library",
            "movie": "media", "영화": "media",
            "music": "media", "음악": "media",
            "player": "media", "플레이어": "media",
        }
        
        for entity_name in entity_names:
            if entity_name in domain_entity_map:
                domain = domain_entity_map[entity_name]
                domain_scores[domain] = domain_scores.get(domain, 0) + 2.0
        
        # From operation patterns
        if operation_names & {"calculate_average", "sum_list", "find_max", "find_min", "sort_descending", "sort_ascending"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 1.5
        if operation_names & {"filter_above", "filter_below", "filter_even", "filter_odd"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 1.0
        if operation_names & {"remove_duplicates"}:
            domain_scores["data_processing"] = domain_scores.get("data_processing", 0) + 0.5
        if operation_names & {"is_prime", "factorial", "fibonacci", "binary_search"}:
            domain_scores["algorithm"] = domain_scores.get("algorithm", 0) + 1.5
        if operation_names & {"reverse_string", "is_palindrome", "count_characters", "count_words"}:
            domain_scores["string_processing"] = domain_scores.get("string_processing", 0) + 1.5
        
        # From constraints
        for c in model.constraints:
            if c.kind == "platform":
                if c.value in ("android", "ios", "mobile"):
                    domain_scores["mobile"] = domain_scores.get("mobile", 0) + 2.0
                elif c.value in ("web", "browser"):
                    domain_scores["web"] = domain_scores.get("web", 0) + 2.0
                elif c.value == "cli":
                    domain_scores["cli"] = domain_scores.get("cli", 0) + 1.0
        
        # Select top domain if confidence is sufficient
        if domain_scores:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
            top_domain, top_score = sorted_domains[0]
            if top_score >= 1.0:
                model.domain = top_domain
                model.domain_evidence = {"scores": sorted_domains[:3], "selected": {"domain": top_domain, "confidence": top_score}}
                return
        
        # No confident domain
        model.domain = "unknown"
        model.domain_evidence = {"fallback": "no_evidence", "scores": sorted(domain_scores.items(), key=lambda x: -x[1])[:3]}

    # ------------------------------------------------------------------ #
    # 2. Retrieve knowledge (structured context, not just keywords)
    # ------------------------------------------------------------------ #
    def _retrieve_knowledge(self, request: str, state: ReasoningState):
        model = state.model
        rl = request.lower()
        tokens = set(re.findall(r"[a-z가-힣]+", rl))

        # Build a structured query from the ProblemModel
        structured_terms = set()
        for op in model.operations:
            structured_terms.add(op.name)
            structured_terms.update(op.name.split("_"))
        for ent in model.entities:
            structured_terms.add(ent.name.lower())
        for inp in model.inputs:
            structured_terms.update(inp.type_name.lower().replace("<", " ").replace(">", " ").split())
        for out in model.outputs:
            structured_terms.update(out.type_name.lower().replace("<", " ").replace(">", " ").split())
        for crit in model.success_criteria:
            structured_terms.update(crit.description.lower().split())
        for cond in model.conditions:
            structured_terms.add(cond.get("type", ""))
            structured_terms.add(cond.get("comparison", ""))
            structured_terms.add(cond.get("order", ""))

        concepts = []
        all_concepts = self.concepts.list_all()
        for c in all_concepts:
            score = 0
            ctext = (c.name + " " + c.description + " " + " ".join(c.tags)).lower()

            # Score 1: token overlap with text
            for t in tokens:
                if t and len(t) > 2 and t in ctext:
                    score += 1

            # Score 2: structured term overlap (higher weight)
            for t in structured_terms:
                if t and len(t) > 2 and t in ctext:
                    score += 2

            # Score 3: operation-specific relevance (highest weight)
            if c.category == "capability":
                for op in model.operations:
                    op_words = op.name.split("_")
                    if any(w in c.tags for w in op_words):
                        score += 5
                    if op.name in c.description:
                        score += 4

                # Score based on entity relevance (not just keyword match)
                for ent in model.entities:
                    ent_lower = ent.name.lower()
                    # Check if this concept is about the right domain
                    if ent_lower in c.description or ent_lower in c.tags:
                        score += 3
                    # Penalize unrelated capabilities
                    if c.id in ("cap_authentication", "cap_networking", "cap_camera",
                                "cap_image_processing"):
                        if not any(k in rl for k in ["camera", "photo", "auth", "login",
                                                       "network", "api", "server"]):
                            score -= 5

            # Score 4: match with input/output types
            for inp in model.inputs:
                inp_words = inp.type_name.lower().replace("<", " ").replace(">", " ").split()
                if any(w in ctext for w in inp_words if len(w) > 2):
                    score += 2
            for out in model.outputs:
                out_words = out.type_name.lower().replace("<", " ").replace(">", " ").split()
                if any(w in ctext for w in out_words if len(w) > 2):
                    score += 2

            if score > 0:
                concepts.append((c, score))

        concepts.sort(key=lambda x: -x[1])

        # Filter out low-relevance concepts (penalized ones)
        state.retrieved_concepts = [c for c, _ in concepts if _ > 0][:12]

        # Traverse the knowledge graph to find indirectly-related concepts.
        related = self._traverse_related(state.retrieved_concepts, depth=1)

        # Recall prior experiences and adapt them.
        state.retrieved_experiences = self._recall_experiences(request)

        state.record(
            "retrieve",
            f"Retrieved {len(state.retrieved_concepts)} concepts "
            f"(structured terms: {len(structured_terms)}, "
            f"tokens: {len(tokens)})",
            [f"concept={c.name}" for c in state.retrieved_concepts[:6]],
            0.8,
        )

    def _traverse_related(self, concepts: list[Concept], depth: int) -> list[Concept]:
        ids = {c.id for c in concepts}
        frontier = set(ids)
        visited = set()
        for _ in range(depth):
            nxt = set()
            for cid in frontier:
                if cid in visited:
                    continue
                visited.add(cid)
                rels = self.relations.find_related(cid)
                for r in rels:
                    other = r.target_id if r.source_id == cid else r.source_id
                    if other not in ids:
                        nxt.add(other)
            frontier = nxt
        related = []
        for cid in frontier:
            if cid not in ids:
                c = self.concepts.get(cid)
                if c:
                    related.append(c)
        return related

    def _recall_experiences(self, request: str) -> list:
        rl = request.lower()
        keywords = re.findall(r"[a-z가-힣]+", rl)
        query = " ".join(keywords[:6])
        return self.memory.search(query, limit=5)

    # ------------------------------------------------------------------ #
    # 3. Generate hypotheses (multiple strategies with ordering)
    # ------------------------------------------------------------------ #
    def _strategy_confidence(self, state: ReasoningState, prior: float,
                             ops: list) -> tuple:
        """Evidence-derived confidence for a candidate strategy: what
        the model actually carries — the operations it names, the I/O
        structure, a composed operation plan, retrieved knowledge —
        never a fixed number."""
        model = state.model
        score = prior
        evidence = []
        if ops:
            score += min(0.12, 0.04 * len(ops))
            evidence.append(f"operations={len(ops)}")
        if model.inputs:
            score += 0.08
            evidence.append("io=inputs")
        if model.outputs:
            score += 0.08
            evidence.append("io=outputs")
        if getattr(state, "operation_plan", None) is not None:
            score += 0.08
            evidence.append("operation_plan=composed")
        if state.retrieved_concepts:
            score += 0.04
            evidence.append(f"knowledge={len(state.retrieved_concepts)}")
        return min(0.95, max(0.2, score)), evidence

    # Techniques defined ONLY over a numeric/index/graph domain: on a
    # declared element type they are contradictions of the semantics.
    _DOMAIN_EXCLUDED_OPS = frozenset({
        "bit_manipulation", "math_formula", "binary_search", "bfs", "dfs",
        "parity_check", "prefix_sum",
    })

    @staticmethod
    def _declared_element_domain(model) -> str:
        """The element domain the semantics DECLARED ("" when unknown)."""
        sigs = " ".join((i.type_name or "") for i in (model.inputs or []))
        sigs = sigs.lower()
        if "list<string>" in sigs or "list<str>" in sigs:
            return "string"
        for token in ("list<integer>", "list<int>", "list<number>",
                      "list<float>", "list<double>"):
            if token in sigs:
                return "number"
        return ""

    def _generate_hypotheses(self, state: ReasoningState):
        model = state.model
        hypotheses = []
        ops = model.operation_order
        entities = model.entities

        # Strategy 1: Operation-pipeline (ordered operations as a pipeline)
        if len(ops) >= 2:
            op_sequence = " -> ".join(ops)
            conf, conf_ev = self._strategy_confidence(state, 0.62, ops)
            evidence = [f"operation_{i}: {op}" for i, op in enumerate(ops)]
            evidence += conf_ev
            hypotheses.append(Hypothesis(
                id="h_operation_pipeline",
                description=f"Build operation pipeline: {op_sequence}",
                evidence=evidence,
                confidence=conf,
                concepts=[c.id for c in state.retrieved_concepts[:4]],
                operations=ops,
                plan_hint={
                    "kind": "operation_pipeline",
                    "operations": ops,
                    "entities": [e.name for e in entities],
                    "inputs": [i.to_dict() for i in model.inputs],
                    "outputs": [o.to_dict() for o in model.outputs],
                    "conditions": model.conditions,
                    "success_criteria": [s.to_dict() for s in model.success_criteria],
                },
            ))

        # Strategy 2: data-management (if there are data entities)
        data_entities = [e for e in entities if e.kind == "data"]
        if data_entities and model.goal_type == GoalType.CREATE_PROJECT:
            entity_name = data_entities[0].name
            conf, conf_ev = self._strategy_confidence(state, 0.65, ops)
            evidence = [f"data entity: {entity_name}"] + conf_ev
            hypotheses.append(Hypothesis(
                id="h_data_management",
                description=f"Build data-management application around {entity_name}",
                evidence=evidence,
                confidence=conf,
                concepts=[c.id for c in state.retrieved_concepts[:4]],
                operations=ops or ["add", "remove", "update", "search", "list"],
                plan_hint={
                    "kind": "data_app",
                    "entity": entity_name,
                    "operations": ops or ["add", "remove", "update", "search", "list"],
                    "pipeline": ops,
                    "inputs": [i.to_dict() for i in model.inputs],
                    "outputs": [o.to_dict() for o in model.outputs],
                    "conditions": model.conditions,
                },
            ))

        # Strategy 3: calculator (if domain is calculation)
        if model.domain == "calculator" or any(o.name in ("calculate",) for o in model.operations):
            conf, conf_ev = self._strategy_confidence(state, 0.70, ops)
            hypotheses.append(Hypothesis(
                id="h_calculator",
                description="Build calculator utility with arithmetic operations",
                evidence=["domain=calculator"] + conf_ev,
                confidence=conf,
                concepts=["cap_calculation", "cap_data_storage"],
                operations=["add", "subtract", "multiply", "divide", "history"],
                plan_hint={"kind": "calculator"},
            ))

        # Strategy 4: concept-composition (general fallback using retrieved knowledge)
        if state.retrieved_concepts:
            cap_concepts = [c for c in state.retrieved_concepts if c.category == "capability"]
            if cap_concepts:
                combined = [c.name for c in cap_concepts[:4]]
                conf, conf_ev = self._strategy_confidence(state, 0.50, ops)
                hypotheses.append(Hypothesis(
                    id="h_concept_composition",
                    description=f"Compose capabilities: {', '.join(combined)}",
                    evidence=[f"capability: {c.name}" for c in cap_concepts[:4]] + conf_ev,
                    confidence=conf,
                    concepts=[c.id for c in cap_concepts[:4]],
                    operations=ops,
                    plan_hint={
                        "kind": "composed",
                        "capabilities": [c.id for c in cap_concepts[:4]],
                        "pipeline": ops,
                    },
                ))

        # Strategy 5: pure function (for single-function goals)
        if model.goal_type == GoalType.CREATE_FUNCTION and ops:
            conf, conf_ev = self._strategy_confidence(state, 0.66, ops)
            hypotheses.append(Hypothesis(
                id="h_pure_function",
                description=f"Implement as pure function with operations: {' -> '.join(ops)}",
                evidence=[f"goal=function, operations={ops}"] + conf_ev,
                confidence=conf,
                concepts=[],
                operations=ops,
                plan_hint={
                    "kind": "pure_function",
                    "operations": ops,
                    "inputs": [i.to_dict() for i in model.inputs],
                    "outputs": [o.to_dict() for o in model.outputs],
                    "conditions": model.conditions,
                },
            ))

        # Think's exploration: a larger budget buys structurally DIFFERENT
        # compositions of the same operation order — a named helper per
        # stage under THINK, a lazy streaming pipeline under DEEP_THINK.
        # The extra candidates are ranked by their own evidence, never
        # above the primary for free.
        budget = current_budget()
        if ops and model.goal_type == GoalType.CREATE_FUNCTION:
            exploration = []
            if budget.mode in (ThinkMode.THINK, ThinkMode.DEEP_THINK):
                exploration.append(("h_helper_decomposition",
                                    "decomposed_functions", 0.60))
            if budget.mode == ThinkMode.DEEP_THINK:
                exploration.append(("h_streaming_pipeline",
                                    "streaming_pipeline", 0.58))
            for hyp_id, kind, prior in exploration:
                conf, conf_ev = self._strategy_confidence(state, prior, ops)
                if kind == "decomposed_functions":
                    desc = (f"Implement each operation as a named helper, "
                            f"main function assembles them: {' -> '.join(ops)}")
                else:
                    desc = f"Lazy streaming pipeline over: {' -> '.join(ops)}"
                hypotheses.append(Hypothesis(
                    id=hyp_id,
                    description=desc,
                    evidence=[f"structure={kind}"] + conf_ev,
                    confidence=conf,
                    concepts=[],
                    operations=ops,
                    plan_hint={
                        "kind": kind,
                        "operations": ops,
                        "inputs": [i.to_dict() for i in model.inputs],
                        "outputs": [o.to_dict() for o in model.outputs],
                        "conditions": model.conditions,
                    },
                ))

        # Compatibility evaluation against the DECLARED element domain:
        # a candidate built on operations undefined for that domain is
        # not a candidate (an undeclared domain excludes nothing).
        domain = self._declared_element_domain(state.model)
        if domain == "string":
            hypotheses = [
                h for h in hypotheses
                if not (set(h.operations or []) & self._DOMAIN_EXCLUDED_OPS)
            ]

        # The budget's cap: the best candidates survive; the rest is
        # exploration this request did not pay for.
        hypotheses.sort(key=lambda h: -h.confidence)
        hypotheses = hypotheses[:budget.max_hypotheses]

        state.hypotheses = hypotheses
        state.record(
            "hypothesize",
            f"Generated {len(hypotheses)} candidate solutions "
            f"(operation_order: {ops}, budget: {budget.mode.value})",
            [f"hyp={h.id}: {h.description}" for h in hypotheses],
            0.8,
        )
        
        # Record decision for hypothesis generation
        for h in hypotheses:
            alternatives = [
                Alternative(
                    option=other.id,
                    reason_rejected=f"Different strategy: {other.description}",
                    confidence=other.confidence,
                    pros=other.evidence,
                    cons=[f"Lower confidence ({other.confidence:.2f})"]
                ) for other in hypotheses if other.id != h.id
            ]
            decision = DecisionTrace(
                decision_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                decision_type=DecisionType.SELECT_HYPOTHESIS,
                chosen_option=h.id,
                alternatives_considered=alternatives,
                constraints_applied=[c.name for c in model.constraints],
                evidence=h.evidence,
                confidence=h.confidence,
                reasoning_step="hypothesize",
                related_concepts=h.concepts,
                related_operations=h.operations,
                related_entities=[e.name for e in entities],
            )
            state.record_decision(decision)

    # ------------------------------------------------------------------ #
    # 4. Evaluate and select hypothesis
    # ------------------------------------------------------------------ #
    def _evaluate_hypotheses(self, state: ReasoningState):
        if not state.hypotheses:
            state.record("evaluate", "No hypotheses generated; falling back to generic",
                         confidence=0.3)
            state.selected_hypothesis = Hypothesis(
                id="h_generic", description="Generic solution", confidence=0.3)
            state.confidence = 0.3
            return

        best = None
        for h in state.hypotheses:
            if best is None or h.confidence > best.confidence:
                best = h

        state.selected_hypothesis = best
        # The state's confidence IS the selection's evidence-based
        # confidence — the reported number must be the chosen one's.
        state.confidence = best.confidence
        state.record(
            "evaluate",
            f"Selected hypothesis: {best.id} ({best.description})",
            [f"confidence={best.confidence:.2f}", f"evidence={best.evidence}"],
            best.confidence,
        )
        
        # Record decision for hypothesis selection
        alternatives = [
            Alternative(
                option=h.id,
                reason_rejected=f"Lower confidence ({h.confidence:.2f} vs {best.confidence:.2f})",
                confidence=h.confidence,
                pros=h.evidence,
                cons=[f"Not the highest confidence"]
            ) for h in state.hypotheses if h.id != best.id
        ]
        decision = DecisionTrace(
            decision_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            decision_type=DecisionType.SELECT_HYPOTHESIS,
            chosen_option=best.id,
            alternatives_considered=alternatives,
            constraints_applied=[c.name for c in state.model.constraints],
            evidence=best.evidence,
            confidence=best.confidence,
            reasoning_step="evaluate",
            related_concepts=best.concepts,
            related_operations=best.operations,
            related_entities=[e.name for e in state.model.entities],
        )
        state.record_decision(decision)

    # ------------------------------------------------------------------ #
    # 5. Formulate plan (preserve operation ordering in IR)
    # ------------------------------------------------------------------ #
    def _formulate_plan(self, state: ReasoningState, context: dict):
        if not state.selected_hypothesis:
            return
        h = state.selected_hypothesis
        model = state.model
        hint = h.plan_hint
        plan = []

        if model.goal_type == GoalType.CREATE_PROJECT:
            plan = self._plan_project(state, hint)
        elif model.goal_type == GoalType.CREATE_CLASS:
            plan = self._plan_class(state)
        elif model.goal_type == GoalType.CREATE_FUNCTION:
            plan = self._plan_function(state, hint)
        elif model.goal_type == GoalType.FIX:
            plan = self._plan_fix(state, context)

        state.plan = plan
        # The invariant: the EXECUTED plan is the selected hypothesis's
        # plan — provenance the verification stage can rely on.
        state.plan_source_hypothesis = h.id

        # The budget's alternative plans: the runner-up hypotheses get
        # plans too — recorded so verification/repair can rotate to a
        # structurally different candidate, never executed in place of
        # the selected one.
        budget = current_budget()
        if budget.max_alternative_plans and len(state.hypotheses) > 1:
            ranked = sorted(
                (x for x in state.hypotheses if x.id != h.id),
                key=lambda x: -x.confidence)
            for alt in ranked[:budget.max_alternative_plans]:
                alt_hint = alt.plan_hint
                alt_plan = []
                if model.goal_type == GoalType.CREATE_PROJECT:
                    alt_plan = self._plan_project(state, alt_hint)
                elif model.goal_type == GoalType.CREATE_CLASS:
                    alt_plan = self._plan_class(state)
                elif model.goal_type == GoalType.CREATE_FUNCTION:
                    alt_plan = self._plan_function(state, alt_hint)
                elif model.goal_type == GoalType.FIX:
                    alt_plan = self._plan_fix(state, context)
                if alt_plan:
                    state.alternative_plans.append(
                        {"hypothesis": alt.id, "plan": alt_plan})

        state.record(
            "plan",
            f"Formulated plan with {len(plan)} actions "
            f"(operation_order: {model.operation_order})",
            [f"action={a.kind}: {a.reason}" for a in plan]
            + [f"alternative_plan={a['hypothesis']}"
               for a in state.alternative_plans],
            0.7,
        )
        
        # Record decisions for each plan action
        for action in plan:
            alternatives = [
                Alternative(
                    option="skip",
                    reason_rejected="Required for goal completion",
                    confidence=0.0,
                    pros=[],
                    cons=["Would not achieve goal"]
                )
            ]
            decision = DecisionTrace(
                decision_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                decision_type=DecisionType.PLAN_ACTION,
                chosen_option=action.kind,
                alternatives_considered=alternatives,
                constraints_applied=[c.name for c in model.constraints],
                evidence=[action.reason] + action.evidence,
                confidence=0.8,
                reasoning_step="plan",
                related_concepts=[],
                related_operations=model.operation_order,
                related_entities=[e.name for e in model.entities],
            )
            state.record_decision(decision)

    def _plan_project(self, state: ReasoningState, hint: dict) -> list[Action]:
        model = state.model
        kind = hint.get("kind", "data_app")
        entity_name = hint.get("entity", (model.entities[0].name if model.entities else "Item"))
        operations = hint.get("operations", model.operation_order) or ["add", "list"]
        pipeline = hint.get("pipeline", model.operation_order)
        actions = []

        if kind == "calculator":
            actions.append(Action(
                kind="generate_service",
                params={"name": "Calculator", "operations": ["add", "subtract", "multiply", "divide", "history"]},
                reason="Encapsulate arithmetic operations and calculation history",
                evidence=["domain=calculator"],
            ))
            actions.append(Action(
                kind="generate_entrypoint",
                params={"name": "Calculator", "operations": operations, "service": "CalculatorService"},
                reason="Interactive calculator entry point",
                evidence=["calculator requires interaction"],
            ))
            return actions

        # For pipeline operations (filter/sort/etc), generate as a processing function
        if pipeline and any(op in pipeline for op in ("remove_duplicates", "calculate_average",
                                                       "filter_above", "filter_below",
                                                       "sort_descending", "sort_ascending")):
            # Generate a processing pipeline function
            actions.append(Action(
                kind="generate_function",
                params={
                    "name": "process_data",
                    "operations": pipeline,
                    "inputs": hint.get("inputs", []),
                    "outputs": hint.get("outputs", []),
                    "conditions": hint.get("conditions", []),
                    "success_criteria": hint.get("success_criteria", []),
                },
                reason=f"Generate processing pipeline: {' -> '.join(pipeline)}",
                evidence=[f"pipeline={pipeline}"],
            ))
            # Also generate a test/main for the pipeline
            actions.append(Action(
                kind="generate_entrypoint",
                params={
                    "name": entity_name,
                    "operations": operations,
                    "service": "Service",
                    "pipeline": pipeline,
                },
                reason="Entry point that exercises the pipeline",
                evidence=["project requires an entry point"],
            ))
            return actions

        # Default: data model + service + entry point
        if entity_name:
            actions.append(Action(
                kind="generate_model",
                params={"name": entity_name, "fields": self._infer_fields(entity_name)},
                reason=f"Data model {entity_name} holds the core domain data",
                evidence=[f"entity={entity_name}"],
            ))
        actions.append(Action(
            kind="generate_service",
            params={"name": entity_name, "operations": operations},
            reason=f"Service encapsulates business logic for {entity_name}",
            evidence=[f"operations={operations}"],
        ))
        actions.append(Action(
            kind="generate_entrypoint",
            params={"name": entity_name, "operations": operations, "service": "Service"},
            reason="Entry point wires the service into an interactive program",
            evidence=["project requires an entry point"],
        ))
        return actions

    def _plan_class(self, state: ReasoningState) -> list[Action]:
        ent = state.model.entities[0] if state.model.entities else Entity("Item")
        return [Action(
            kind="generate_model",
            params={"name": ent.name, "fields": self._infer_fields(ent.name)},
            reason=f"Create class {ent.name}",
            evidence=[f"entity={ent.name}"],
        )]

    def _plan_function(self, state: ReasoningState, hint: dict = None) -> list[Action]:
        hint = hint or {}
        ops = hint.get("operations", state.model.operation_order) or []
        inputs = hint.get("inputs", [])
        outputs = hint.get("outputs", [])
        conditions = hint.get("conditions", [])
        kind = hint.get("kind", "pure_function")

        # The structural hypotheses' plan shapes: the same operation
        # order, composed differently. The terminal action keeps the
        # generate_function/operations contract the pipeline consumes.
        if kind == "decomposed_functions" and len(ops) >= 2:
            actions = [
                Action(
                    kind="generate_helper",
                    params={"operation": op, "stage": i},
                    reason=f"Implement {op} as a named helper stage",
                    evidence=[f"operation={op}"],
                )
                for i, op in enumerate(ops)
            ]
            actions.append(Action(
                kind="generate_function",
                params={
                    "operations": ops,
                    "inputs": inputs,
                    "outputs": outputs,
                    "conditions": conditions,
                    "compose": [f"stage_{i}" for i in range(len(ops))],
                },
                reason="Main function assembles the helper stages: "
                        + " -> ".join(ops),
                evidence=[f"compose={len(ops)} stages"],
            ))
            return actions
        if kind == "streaming_pipeline" and ops:
            return [Action(
                kind="generate_function",
                params={
                    "operations": ops,
                    "inputs": inputs,
                    "outputs": outputs,
                    "conditions": conditions,
                    "pipeline": "streaming",
                },
                reason="Generate a lazy streaming pipeline: " + " -> ".join(ops),
                evidence=["pipeline=streaming"],
            )]

        if ops:
            return [Action(
                kind="generate_function",
                params={
                    "operations": ops,
                    "inputs": inputs,
                    "outputs": outputs,
                    "conditions": conditions,
                },
                reason=f"Generate function with operations: {' -> '.join(ops)}",
                evidence=[f"operations={ops}"],
            )]
        return [Action(kind="generate_function", params={}, reason="Generate a function",
                       evidence=["goal=function"])]

    def _plan_fix(self, state: ReasoningState, context: dict) -> list[Action]:
        return [Action(kind="fix", params={}, reason="Repair identified issue",
                       evidence=["goal=fix"])]

    def _infer_fields(self, entity_name: str) -> list[dict]:
        base = [{"name": "id", "type": "int", "description": f"{entity_name} ID"}]
        field_hints = {
            "Student": [("name", "String"), ("email", "String"), ("grade", "int"), ("major", "String")],
            "Contact": [("name", "String"), ("phone", "String"), ("email", "String")],
            "Todo": [("title", "String"), ("description", "String"), ("completed", "boolean"), ("priority", "int")],
            "User": [("name", "String"), ("email", "String"), ("password", "String")],
            "Employee": [("name", "String"), ("position", "String"), ("salary", "int")],
            "Product": [("name", "String"), ("price", "float"), ("quantity", "int")],
            "Order": [("customer", "String"), ("total", "float"), ("status", "String")],
        }
        for fname, ftype in field_hints.get(entity_name, [("name", "String")]):
            base.append({"name": fname, "type": ftype, "description": f"{fname}"})
        return base

    # ------------------------------------------------------------------ #
    # 6. Derive lesson (for learning)
    # ------------------------------------------------------------------ #
    def _derive_lesson(self, state: ReasoningState):
        if state.selected_hypothesis:
            state.lesson = {
                "domain": state.model.domain,
                "goal_type": state.model.goal_type.value,
                "hypothesis": state.selected_hypothesis.id,
                "concepts": state.selected_hypothesis.concepts or [],
                "operations": state.selected_hypothesis.operations or [],
            }

    # ------------------------------------------------------------------ #
    # Self-repair support
    # ------------------------------------------------------------------ #
    def observe(self, state: ReasoningState, observation: dict):
        state.observations.append(observation)
        state.record(
            "observe",
            observation.get("summary", "Observation recorded"),
            observation.get("evidence", []),
            observation.get("confidence", 0.7),
        )

    def adapt(self, state: ReasoningState, observation: dict) -> bool:
        """Adjust the plan/hypothesis based on an observation. Returns True if changed."""
        if not state.observation_conflicts(observation):
            return False
        state.record(
            "adapt",
            "Adjusting plan due to contradictory observation",
            [f"observation={observation.get('summary')}"],
            0.6,
        )
        return True

    def observation_conflicts(self, state: ReasoningState, observation: dict) -> bool:
        return observation.get("severity") == "error"

    # ------------------------------------------------------------------ #
    # Explanation (evidence-based, no private chain-of-thought)
    # ------------------------------------------------------------------ #
    def explain(self, state: ReasoningState) -> list[str]:
        lines = []
        if state.selected_hypothesis:
            h = state.selected_hypothesis
            lines.append(f"Goal: {state.model.summary}")
            lines.append(f"Approach: {h.description} (confidence {h.confidence:.2f})")
            if h.evidence:
                lines.append("Evidence: " + ", ".join(h.evidence))
        if state.plan:
            for a in state.plan:
                lines.append(f"{a.kind}: {a.reason}")
        return lines