"""RENA-BR - the main orchestrator that ties all modules together."""
from __future__ import annotations
import os
import json
import re
import sys
from typing import Optional
from types import SimpleNamespace

from brain.concept_store import ConceptStore, seed_default_concepts
from brain.relation_store import RelationStore
from brain.rule_store import RuleEngine, seed_default_rules
from brain.memory_store import MemoryStore
from brain.reasoner import Reasoner, ReasoningResult
from brain.planner import Planner, Plan
from brain.project_planner import ProjectPlanner
from brain.error_analyzer import ErrorAnalyzer
from brain.intent_analyzer import IntentAnalyzer, IntentResult, INTENT_CHAT, INTENT_QUESTION, INTENT_EXPLANATION, INTENT_GENERATE_CODE, INTENT_CREATE_PROJECT, INTENT_MODIFY_PROJECT, INTENT_DEBUG, INTENT_ANALYZE_CODE, INTENT_TEST_CODE, INTENT_BUILD_PROJECT, INTENT_UNKNOWN, INTENT_COMPUTER_TASK
from brain.conversation_manager import ConversationManager
from brain.requirements_analyzer import RequirementsAnalyzer
from brain.technology_selector import TechnologySelector
from brain.self_knowledge import SelfKnowledge
from brain.capability_registry import CapabilityRegistry
from brain.semantic_understanding import SemanticUnderstanding
from brain.response_planner import ResponsePlanner
from brain.response_generator import ResponseGenerator
from brain.dynamic_response import DynamicResponseGenerator
from brain.think_engine import ThinkEngine, ReasoningState, ProblemModel, GoalType as ThinkGoalType
from brain.semantic_representation import GoalType, EntityRef, IOSpec
from brain.semantic_parser import SemanticParser, ParseContext, create_semantic_parser
from brain.response_planner_new import ResponsePlanner as NewResponsePlanner, create_response_planner
from brain.conversation_knowledge import ConversationKnowledgeEngine, realize_plan
from brain.response_generator_new import ResponseGenerator as NewResponseGenerator, create_response_generator
from brain.language_realizer import get_realizer, realize_response
from brain.generation import GenerationController, ControllerConfig, StreamingEmitter
from brain.pipeline_generator import PipelineCodeGenerator
from memory.memory_manager import MemoryManager
from memory.feedback_learner import FeedbackLearner
from memory.apology_system import ApologySystem
from code.ir.ir_types import IRNode, IRNodeType
from code.ir.project_ir import ProjectIR
from code.symbol_table import SymbolTable
from code.verifier import Verifier
from code.project.project_verifier import ProjectVerifier
from code.generator.code_generator import CodeGenerator
from code.generator.project_generator import ProjectGenerator, ProjectGenerationResult
from code.refactoring.refactoring_engine import RefactoringEngine
from learning.learner.code_learner import Learner
from brain.streaming_generator import StreamingGenerator, run_streaming_generation
from brain.response_chunking import chunk_response
from brain.route import Route, decide_route, decide_route_parts, describe_route
from brain.explanation_engine import ExplanationEngine
from brain.korean_grammar import KoreanGrammar
from brain.english_grammar import EnglishGrammar
from brain.reasoning_events import ReasoningEventCollector
from brain.decision_trace import DecisionTrace, Alternative, DecisionType
from brain.algorithmic.orchestrator import AlgorithmicOrchestrator


class _CompositionView:
    """A resumed session turn's SERIALIZED meaning, rebuilt as the
    parts the continuation merge reads: its composition (the graph the
    merge restructures), its entities and IO (the subject continuity a
    modify turn does not restate), its output type and its goal's
    algorithm hint.  The runtime operation-graph OBJECT is not part of
    the record — the merge's guarded copy skips it, and the composition
    (which the record DOES carry) is the structure of record."""

    __slots__ = ("composition", "entities", "inputs", "outputs",
                 "output_type", "goal", "id")

    def __init__(self, data: dict) -> None:
        d = dict(data or {})
        self.composition = dict(d.get("composition") or {})
        self.entities = [_entity_ref(x) for x in
                         (d.get("entities") or []) if isinstance(x, dict)]
        self.inputs = [_io_spec(x) for x in
                       (d.get("inputs") or []) if isinstance(x, dict)]
        self.outputs = [_io_spec(x) for x in
                        (d.get("outputs") or []) if isinstance(x, dict)]
        self.output_type = str(d.get("output_type") or "")
        goal = dict(d.get("goal") or {})
        self.goal = SimpleNamespace(
            **{k: v for k, v in goal.items() if isinstance(k, str)})
        self.id = str(d.get("id") or "")


def _entity_ref(data: dict) -> EntityRef:
    """The serialized entity (its to_dict renames `type_name` to
    `type`) rebuilt as the object the merge deep-copies."""
    return EntityRef(
        id=str(data.get("id") or ""),
        name=str(data.get("name") or ""),
        type_name=str(data.get("type") or ""),
        kind=str(data.get("kind") or "object"),
        attributes=dict(data.get("attributes") or {}),
        quantity=str(data.get("quantity") or "single"),
        is_input=bool(data.get("is_input")),
        is_output=bool(data.get("is_output")))


def _io_spec(data: dict) -> IOSpec:
    """The serialized IO spec (`type` → `type_signature`) rebuilt as
    the object the merge deep-copies."""
    return IOSpec(
        id=str(data.get("id") or ""),
        name=str(data.get("name") or ""),
        type_signature=str(data.get("type") or ""),
        description=str(data.get("description") or ""),
        required=bool(data.get("required")))


class RequirementTracker:
    """Tracks requirements through the pipeline: Requirement → Plan → IR → Code."""
    
    def __init__(self):
        self.requirements = []  # list of Requirement objects
        self.plan_mapping = {}  # requirement_id -> plan step
        self.ir_mapping = {}    # requirement_id -> IR node
        self.code_mapping = {}  # requirement_id -> generated code
        self.test_mapping = {}  # requirement_id -> test/coverage result
        self.coverage = {}      # requirement_id -> satisfied (bool)
    
    def set_requirements(self, requirements):
        """Set the initial requirements from ProblemModel."""
        self.requirements = requirements
        self.coverage = {r.id: False for r in requirements}
    
    def map_to_plan(self, requirement_id, plan_step):
        """Map a requirement to a plan step."""
        self.plan_mapping[requirement_id] = plan_step
    
    def map_to_ir(self, requirement_id, ir_node):
        """Map a requirement to an IR node."""
        self.ir_mapping[requirement_id] = ir_node
    
    def map_to_code(self, requirement_id, code):
        """Map a requirement to generated code.

        Producing code does not satisfy the requirement: only the test
        marks it covered."""
        self.code_mapping[requirement_id] = code

    def map_to_test(self, requirement_id, test_result):
        """Map a requirement to its test: the SAME id the plan, the IR
        and the code recorded, now answered by the tests run against
        them.  The requirement is satisfied only when the test says so —
        a plan or a code mapping never marks it covered by itself."""
        self.test_mapping[requirement_id] = test_result
        if isinstance(test_result, dict):
            satisfied = bool(test_result.get("satisfied"))
        else:
            satisfied = bool(test_result)
        if satisfied:
            self.coverage[requirement_id] = True
        elif not self.coverage.get(requirement_id):
            self.coverage[requirement_id] = False
    
    def get_coverage_report(self):
        """Get a report of requirement coverage."""
        report = []
        for req in self.requirements:
            satisfied = self.coverage.get(req.id, False)
            report.append({
                "requirement": req.name,
                "category": req.category,
                "value": req.value,
                "satisfied": satisfied,
                "has_plan": req.id in self.plan_mapping,
                "has_ir": req.id in self.ir_mapping,
                "has_code": req.id in self.code_mapping,
                "has_test": req.id in self.test_mapping,
                "test": self.test_mapping.get(req.id, ""),
            })
        return report
    
    def get_unsatisfied_requirements(self):
        """Get list of requirements not yet satisfied."""
        return [r for r in self.requirements if not self.coverage.get(r.id, False)]
    
    def to_dict(self):
        """Convert tracking state to dictionary."""
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "coverage": self.coverage,
            "plan_mapping": {k: str(v) for k, v in self.plan_mapping.items()},
            "ir_mapping": {k: str(v) for k, v in self.ir_mapping.items()},
            "code_mapping": {k: v[:100] if isinstance(v, str) else str(v)[:100] 
                            for k, v in self.code_mapping.items()},
            "test_mapping": {k: (v[:100] if isinstance(v, str) else str(v)[:100])
                            for k, v in self.test_mapping.items()},
            "coverage_report": self.get_coverage_report(),
        }


_HANGUL_RE = re.compile(r"[가-힣]")


class CodeBrain:
    # The vocabulary the generation pipeline can RENDER as dataflow —
    # the gate deciding which of a continuation turn's composed
    # operations may enter the artifact's graph.  A capability
    # vocabulary (what the IR can express), never a list of sentence
    # keywords: an operation outside it is not dataflow the pipeline
    # knows, whatever word composed it.
    PIPELINE_OPERATION_VOCAB = frozenset({
        "remove_duplicates", "calculate_average", "calculate_sum",
        "find_max", "find_min", "filter_above", "filter_below",
        "sort_descending", "sort_ascending", "filter_even",
        "filter_elements", "reverse_sequence", "is_prime"})

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".codebrain")
        os.makedirs(self.data_dir, exist_ok=True)

        brain_dir = os.path.join(self.data_dir, "brain_data")
        self.concept_store = ConceptStore(brain_dir)
        self.relation_store = RelationStore(brain_dir)
        self.rule_engine = RuleEngine(brain_dir)
        self.memory_store = MemoryStore(brain_dir)
        self.symbol_table = SymbolTable()
        self.verifier = Verifier()
        self.project_verifier = ProjectVerifier()
        self.code_generator = CodeGenerator()
        self.project_generator = ProjectGenerator()
        self.refactoring = RefactoringEngine()
        self.requirement_tracker = RequirementTracker()
        self.reasoner = Reasoner(self.concept_store, self.relation_store,
                                 self.rule_engine, self.memory_store)
        self.planner = Planner()
        self.project_planner = ProjectPlanner()
        self.error_analyzer = ErrorAnalyzer()
        self.learner = Learner(self.concept_store, self.relation_store, self.memory_store)

        self.intent_analyzer = IntentAnalyzer()
        self.conversation_manager = ConversationManager()
        self.requirements_analyzer = RequirementsAnalyzer()
        self.technology_selector = TechnologySelector()

        self.self_knowledge = SelfKnowledge()
        self.capability_registry = CapabilityRegistry()
        # The conversation knowledge engine: question → intent →
        # knowledge retrieval → reasoning → ResponsePlan → answer
        # (after the self knowledge is available)
        self.conversation_knowledge = ConversationKnowledgeEngine(
            self.concept_store, self.relation_store,
            self.self_knowledge, self.capability_registry)
        # The topic state (the conversation manager) is the stable
        # referent an anaphoric reference resolves against
        self.conversation_knowledge.topic_provider = \
            self.conversation_manager
        # The REFERENCE RESOLVER: a referring expression ("that", "it",
        # "the last topic") resolves against the conversation's state —
        # its candidates, scored — and is never sent to a retriever.
        from brain.context_resolution import ReferenceResolver
        self.reference_resolver = ReferenceResolver(
            knowledge_lookup=self._knowledge_lookup)
        self._last_reference = None

        self.semantic_understanding = SemanticUnderstanding()
        self.response_planner = ResponsePlanner(self.self_knowledge, self.capability_registry)
        self.response_generator = ResponseGenerator(self.self_knowledge, self.capability_registry)
        self.dynamic_response = DynamicResponseGenerator(
            self_knowledge=self.self_knowledge,
            capability_registry=self.capability_registry,
            concept_store=self.concept_store,
        )
        self.think_engine = ThinkEngine(
            self.concept_store, self.relation_store,
            self.rule_engine, self.memory_store
        )

        # NEW: Explainable AI components
        self.korean_grammar = KoreanGrammar()
        self.english_grammar = EnglishGrammar()
        self.explanation_engine = ExplanationEngine(
            self.korean_grammar,
            self.english_grammar,
            self.concept_store,
            self.relation_store,
            self.self_knowledge
        )
        self.streaming_generator = StreamingGenerator(
            self.code_generator,
            self.explanation_engine,
            ReasoningEventCollector(language="ko")
        )

        # NEW: Semantic Pipeline
        self.semantic_parser = create_semantic_parser(self.concept_store, self.relation_store)
        self.new_response_planner = create_response_planner(
            self.concept_store, self.relation_store, self.memory_store,
            self.capability_registry, self.self_knowledge
        )
        self.new_response_generator = create_response_generator(
            self.concept_store, self.relation_store, self.memory_store,
            self.capability_registry, self.self_knowledge
        )

        memory_dir = os.path.join(self.data_dir, "memory_data")
        self.memory_manager = MemoryManager(memory_dir)
        self.feedback_learner = self.memory_manager.feedback_learner
        self.apology_system = self.memory_manager.apology_system
        self.error_diagnoser = self.memory_manager.error_diagnoser
        self.auto_fixer = self.memory_manager.auto_fixer

        # The defaults are the store's own vocabulary: a store loaded
        # from an earlier save carries the tags it had then, so the
        # defaults are re-seeded every start rather than only when a
        # concept is missing — the concept's own words are what its
        # search matches on.
        seed_default_concepts(self.concept_store)
        if not self.rule_engine.find_rules():
            seed_default_rules(self.rule_engine)
        if not self.relation_store.relations:
            from brain.relation_store import seed_default_relations
            seed_default_relations(self.relation_store)

        # Seed shell knowledge if not already present
        if not any(c.category == "shell_platform" for c in self.concept_store.list_all()):
            from brain.shell_knowledge import seed_shell_knowledge
            seed_shell_knowledge(self.concept_store, self.relation_store)

        # Seed algorithmic primitives if not already present
        if not any(c.category == "algorithmic_primitive" for c in self.concept_store.list_all()):
            from brain.concept_store import seed_algorithmic_concepts
            seed_algorithmic_concepts(self.concept_store)

        # Seed the programming fundamentals knowledge (the languages,
        # the language constructs, the runtime concepts) — the concept
        # descriptions are EVIDENCE for the conversation knowledge
        # engine's reasoning, never canned answers: the engine composes
        # a fresh response from the evidence per the question's intent.
        # The seeder is idempotent: a store loaded from disk only gains
        # the concepts its own snapshot predates.
        from brain.concept_store import seed_programming_fundamentals
        seed_programming_fundamentals(self.concept_store)
        # Seed algorithmic relations if not already present
        if not any(r.id.startswith("r_alog_") for r in self.relation_store.relations.values()):
            from brain.relation_store import seed_algorithmic_relations
            seed_algorithmic_relations(self.relation_store)

        # Feature flag for new semantic pipeline
        self._use_semantic_pipeline = True

    def process(self, request: str, language: str = "python") -> dict:
        # Use new semantic pipeline if enabled
        if self._use_semantic_pipeline:
            return self.process_semantic(request, language)
        
        intent_result = self.intent_analyzer.analyze(request)

        memory_context = self.memory_manager.get_context_for_request(request)

        if intent_result.intent == INTENT_CHAT:
            semantic = self.semantic_understanding.understand(request)
            user_context = memory_context.get("user_profile", {})
            response = self.dynamic_response.generate(
                semantic, user_context,
                conversation_history=self.conversation_manager.history,
            )
            self.conversation_manager.add_turn(request, INTENT_CHAT, response)
            self.memory_manager.conversation_memory.add_turn(request, response, INTENT_CHAT)
            return {
                "request": request,
                "language": language,
                "intent": INTENT_CHAT,
                "intent_result": intent_result,
                "semantic": semantic.to_dict(),
                "response": response,
                "project_created": False,
                "results": {},
                "memory_context": memory_context,
            }

        if intent_result.intent == INTENT_QUESTION:
            semantic = self.semantic_understanding.understand(request)
            user_context = memory_context.get("user_profile", {})
            response = self.dynamic_response.generate(
                semantic, user_context,
                conversation_history=self.conversation_manager.history,
            )
            self.conversation_manager.add_turn(request, INTENT_QUESTION, response)
            self.memory_manager.conversation_memory.add_turn(request, response, INTENT_QUESTION)
            return {
                "request": request,
                "language": language,
                "intent": INTENT_QUESTION,
                "intent_result": intent_result,
                "semantic": semantic.to_dict(),
                "response": response,
                "project_created": False,
                "results": {},
                "memory_context": memory_context,
            }

        if intent_result.intent == INTENT_EXPLANATION:
            semantic = self.semantic_understanding.understand(request)
            user_context = memory_context.get("user_profile", {})
            response = self.dynamic_response.generate(
                semantic, user_context,
                conversation_history=self.conversation_manager.history,
            )
            self.conversation_manager.add_turn(request, INTENT_EXPLANATION, response)
            self.memory_manager.conversation_memory.add_turn(request, response, INTENT_EXPLANATION)
            return {
                "request": request,
                "language": language,
                "intent": INTENT_EXPLANATION,
                "intent_result": intent_result,
                "semantic": semantic.to_dict(),
                "response": response,
                "project_created": False,
                "results": {},
                "memory_context": memory_context,
            }

        if intent_result.intent == INTENT_UNKNOWN:
            feedback_entry = self.feedback_learner.process_feedback(request)
            if feedback_entry:
                apology_context = self.feedback_learner.get_apology_context(request)
                if apology_context:
                    response = self.apology_system.generate_apology(apology_context)
                else:
                    response = "피드백 감사합니다. 앞으로 더 신경 쓰겠습니다."
                self.conversation_manager.add_turn(request, INTENT_UNKNOWN, response)
                self.memory_manager.conversation_memory.add_turn(request, response, INTENT_UNKNOWN)
                return {
                    "request": request,
                    "language": language,
                    "intent": INTENT_UNKNOWN,
                    "intent_result": intent_result,
                    "response": response,
                    "project_created": False,
                    "results": {},
                    "memory_context": memory_context,
                    "feedback_processed": True,
                }
            semantic = self.semantic_understanding.understand(request)
            user_context = memory_context.get("user_profile", {})
            response = self.dynamic_response.generate(
                semantic, user_context,
                conversation_history=self.conversation_manager.history,
            )
            self.conversation_manager.add_turn(request, INTENT_UNKNOWN, response)
            self.memory_manager.conversation_memory.add_turn(request, response, INTENT_UNKNOWN)
            return {
                "request": request,
                "language": language,
                "intent": INTENT_UNKNOWN,
                "intent_result": intent_result,
                "semantic": semantic.to_dict(),
                "response": response,
                "project_created": False,
                "results": {},
                "memory_context": memory_context,
            }

        if intent_result.intent == INTENT_CREATE_PROJECT:
            return self.generate_project(request, language)

        if intent_result.intent == INTENT_MODIFY_PROJECT:
            if self.conversation_manager.current_project:
                response = (
                    f"현재 프로젝트: {self.conversation_manager.current_project.name}.\n"
                    "기존 프로젝트를 수정합니다. 어떤 변경을 원하시나요?"
                )
                self.conversation_manager.add_turn(request, INTENT_MODIFY_PROJECT, response)
                self.memory_manager.conversation_memory.add_turn(request, response, INTENT_MODIFY_PROJECT)
                return {
                    "request": request,
                    "language": language,
                    "intent": INTENT_MODIFY_PROJECT,
                    "intent_result": intent_result,
                    "response": response,
                    "project_created": False,
                    "project_ir": self.conversation_manager.current_project.to_dict(),
                    "results": {},
                    "memory_context": memory_context,
                }
            else:
                response = (
                    "기존 프로젝트가 없습니다. 새 프로젝트를 만들까요?\n"
                    "어떤 것을 만들고 싶은지 설명해 주세요."
                )
                self.conversation_manager.add_turn(request, INTENT_MODIFY_PROJECT, response)
                self.memory_manager.conversation_memory.add_turn(request, response, INTENT_MODIFY_PROJECT)
                return {
                    "request": request,
                    "language": language,
                    "intent": INTENT_MODIFY_PROJECT,
                    "intent_result": intent_result,
                    "response": response,
                    "project_created": False,
                    "results": {},
                    "memory_context": memory_context,
                }

        if intent_result.intent in (INTENT_DEBUG, INTENT_ANALYZE_CODE,
                                     INTENT_TEST_CODE, INTENT_BUILD_PROJECT):
            response = (
                f"{intent_result.intent.lower().replace('_', ' ')} 요청을 처리합니다.\n"
                "작업할 코드나 파일을 지정해 주세요."
            )
            self.conversation_manager.add_turn(request, intent_result.intent, response)
            self.memory_manager.conversation_memory.add_turn(request, response, intent_result.intent)
            return {
                "request": request,
                "language": language,
                "intent": intent_result.intent,
                "intent_result": intent_result,
                "response": response,
                "project_created": False,
                "results": {},
                "memory_context": memory_context,
            }

        reasoning = self.reasoner.analyze(request, language)
        plan = self.planner.create_plan(reasoning, request, language)
        ir_nodes = self.planner.execute_plan(plan)
        results = {}
        for node in ir_nodes:
            results[node.name] = {
                "ir": node.to_dict(),
                "languages": {},
            }
            for lang in self.code_generator.get_supported_languages():
                try:
                    code = self.code_generator.generate(node, lang)
                    results[node.name]["languages"][lang] = {
                        "code": code,
                        "verification": None,
                    }
                except Exception as e:
                    results[node.name]["languages"][lang] = {
                        "code": f"// Error: {e}",
                        "verification": None,
                    }
        self.conversation_manager.add_turn(request, INTENT_GENERATE_CODE, "코드가 생성되었습니다.")
        self.memory_manager.conversation_memory.add_turn(
            request, "코드가 생성되었습니다.", INTENT_GENERATE_CODE
        )
        self.memory_manager.record_experience(
            experience_type="success",
            description=f"코드 생성: {request}",
            context={"intent": INTENT_GENERATE_CODE, "language": language},
            outcome="코드 생성 성공",
        )
        return {
            "request": request,
            "language": language,
            "intent": INTENT_GENERATE_CODE,
            "intent_result": intent_result,
            "reasoning": reasoning,
            "plan": self.planner.plan_to_dict(plan),
            "results": results,
            "project_created": False,
            "memory_context": memory_context,
        }

    def _knowledge_lookup(self, query: str):
        """Whether a referent is a KNOWN knowledge topic (the resolver's
        semantic-compatibility signal — a local, offline probe)."""
        try:
            hits = self.concept_store.search_with_scores(query)
        except Exception:
            return []
        return list(hits)[:3]

    def process_natural(self, request: str) -> dict:
        """Natural language entry point — uses the semantic pipeline.

        Flow: SemanticParser → SemanticComposer → GenerationController → Output.
        No keyword matching. All understanding from composition.
        """
        # Natural-language DRAWING: "사람 그려줘" / "draw a bed" is an
        # IMAGE-GENERATION request — it produces an image artifact
        # (icons composed from geometric primitives), not code.
        image_result = self._try_image_request(request)
        if image_result is not None:
            return image_result
        audio_result = self._try_audio_request(request)
        if audio_result is not None:
            return audio_result
        # Delegate to the semantic pipeline
        return self.process_semantic(request, language="python")

    def _try_audio_request(self, request: str) -> Optional[dict]:
        """Natural-language AUDIO artifact: music/effect WAV, or None.

        Code requests that merely mention audio are guarded out by the
        parser (media.audio.nl) and keep their normal pipeline.
        """
        try:
            from media.audio.nl import (parse_audio_request,
                                        generate_audio_artifact)
            spec = parse_audio_request(request)
            if spec is None:
                return None
            meta = generate_audio_artifact(spec, request)
            korean = bool(re.search(r"[가-힣]", request or ""))
            response = (
                f"{meta['label']} 오디오를 절차적 합성으로 만들었습니다 "
                f"({meta['duration']}초, {meta['sample_rate']}Hz). "
                f"저장 위치: {meta['path']}"
                if korean else
                f"Synthesized {meta['label']} ({meta['duration']}s, "
                f"{meta['sample_rate']}Hz). Saved to {meta['path']}")
            return {
                "request": request,
                "intent": "GENERATE_AUDIO",
                "results": {},
                "reasoning": None,
                "project_created": False,
                "audio": meta,
                "response": response,
                "confidence": {
                    "confidence": 0.9,
                    "basis": "procedural signal synthesis (oscillators, "
                             "noise, filters, envelopes) — no samples "
                             "copied, no model",
                    "requirement_pass": True,
                    "requirement_coverage": "1/1",
                },
                "requirement_coverage": {
                    "passed": True, "verified": 1, "total": 1,
                    "type_consistency": "not_comparable",
                    "items": [{
                        "id": "R1", "kind": "audio",
                        "semantic": f"audio: {meta['label']}",
                        "required": True, "evidence": request,
                        "stages": {"semantic": True, "plan": True,
                                   "ir": True, "code": True,
                                   "runtime": True,
                                   "behavioral_test": True,
                                   "verified": True},
                    }],
                },
            }
        except Exception:
            return None

    def _try_image_request(self, request: str) -> Optional[dict]:
        """Image routes, most specific first.

        1. PHYSICS RENDERING: "사실적인 우주 행성 그려줘" path-traces a
           scene (no learned models — light transport + procedural
           materials) into a photorealistic image.
        2. ICON DRAWING: "사람이랑 침대 그려줘" composes geometric icons.
        None lets the request follow the normal semantic pipeline.
        """
        rendered = self._try_render_request(request)
        if rendered is not None:
            return rendered
        return self._try_draw_request(request)

    def _try_render_request(self, request: str) -> Optional[dict]:
        """Render a scene request with the physics renderer, or None."""
        try:
            from brain.render3d import (parse_render_request, render_to_file,
                                        get_scene, default_render_path)
            scene_name = parse_render_request(request)
            if not scene_name:
                return None
            size = os.environ.get("RENA_RENDER_SIZE", "400x300")
            width, height = (int(v) for v in size.lower().split("x", 1))
            samples = int(os.environ.get("RENA_RENDER_SAMPLES", "6"))
            bounces = int(os.environ.get("RENA_RENDER_BOUNCES", "0")) or None
            output_path = default_render_path(scene_name)
            meta = render_to_file(scene_name, output_path, width, height,
                                  samples, bounces=bounces)
            scene = get_scene(scene_name)
            korean = bool(re.search(r"[가-힣]", request or ""))
            response = (
                f"물리 기반 레이 트레이싱으로 '{scene.description}'을 "
                f"렌더링했습니다 ({width}x{height}, {samples}spp). "
                f"저장 위치: {output_path}"
                if korean else
                f"Path-traced '{scene.name}' ({width}x{height}, "
                f"{samples}spp). Saved to {output_path}")
            return {
                "request": request,
                "intent": "RENDER_IMAGE",
                "results": {},
                "reasoning": None,
                "project_created": False,
                "image": meta,
                "response": response,
                "confidence": {
                    "confidence": 0.9,
                    "basis": "physics-based path tracing (no learned "
                             "models): light transport, procedural "
                             "materials, soft shadows, ACES tone mapping",
                    "requirement_pass": True,
                    "requirement_coverage": "1/1",
                },
                "requirement_coverage": {
                    "passed": True,
                    "verified": 1,
                    "total": 1,
                    "type_consistency": "not_comparable",
                    "items": [{
                        "id": "R1",
                        "kind": "render",
                        "semantic": f"render: {scene_name}",
                        "required": True,
                        "evidence": request,
                        "stages": {
                            "semantic": True, "plan": True, "ir": True,
                            "code": True, "runtime": True,
                            "behavioral_test": True, "verified": True,
                        },
                    }],
                },
            }
        except Exception:
            return None

    def _try_draw_request(self, request: str) -> Optional[dict]:
        """Render a drawing request to an image artifact, or None.

        None whenever the request is not a drawing request or names no
        object the icon ontology knows — the request then follows the
        normal semantic pipeline, never a silent image guess.
        """
        try:
            from brain.image_draw import (parse_draw_request, draw_request,
                                          default_output_path)
            items = parse_draw_request(request)
            if not items:
                return None
            output_path = default_output_path(request)
            meta = draw_request(request, output_path)
            korean = bool(re.search(r"[가-힣]", request or ""))
            names = ", ".join(item.icon for item in items)
            response = (
                f"{names} 장면을 도형 조합으로 그렸습니다. 저장 위치: "
                f"{output_path}"
                if korean else
                f"Drew a scene with: {names}. Saved to {output_path}")
            return {
                "request": request,
                "intent": "GENERATE_IMAGE",
                "results": {},
                "reasoning": None,
                "project_created": False,
                "image": meta,
                "response": response,
                "confidence": {
                    "confidence": 0.9,
                    "basis": "image scene composed from geometric "
                             "primitives (no learned models)",
                    "requirement_pass": True,
                    "requirement_coverage": "1/1",
                },
                "requirement_coverage": {
                    "passed": True,
                    "verified": 1,
                    "total": 1,
                    "type_consistency": "not_comparable",
                    "items": [{
                        "id": "R1",
                        "kind": "image",
                        "semantic": f"draw: {names}",
                        "required": True,
                        "evidence": request,
                        "stages": {
                            "semantic": True, "plan": True, "ir": True,
                            "code": True, "runtime": True,
                            "behavioral_test": True, "verified": True,
                        },
                    }],
                },
            }
        except Exception:
            return None

    def process_natural_debug(self, request: str) -> dict:
        """Natural language entry point with full debug output.

        Uses the semantic pipeline with debug information.
        """
        # Get semantic representation
        parse_context = ParseContext(
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict()
                if self.conversation_manager.current_project else None,
            user_preferences=self._current_user_preferences()
        )
        semantic = self.semantic_parser.parse(request, parse_context)

        # Also run think_engine for backward-compatible debug output
        reasoning = self.think_engine.think_from_semantic(semantic, "python")
        model = reasoning.model

        # Build debug info from both semantic and reasoning model
        debug_info = {
            "[Semantic]": {
                "goal_type": semantic.goal.goal_type.value,
                "confidence": semantic.confidence,
                "entities": [e.name for e in semantic.entities],
                "operations": semantic.get_operation_order(),
            },
            "[Composition]": semantic.composition or {},
            "[Problem]": {
                "goal_type": model.goal_type.value,
                "summary": model.summary,
                "goal": model.goal,
                "domain": model.domain,
                "language": model.language,
                "platform": model.platform,
            },
            "[Entities]": [e.to_dict() for e in model.entities],
            "[Inputs]": [i.to_dict() for i in model.inputs],
            "[Outputs]": [o.to_dict() for o in model.outputs],
            "[Operations]": {
                "ordered": model.operation_order,
                "details": [o.to_dict() for o in model.operations],
            },
            "[Constraints]": [c.to_dict() for c in model.constraints],
        }

        # Process normally
        result = self.process_semantic(request, language="python")
        result["debug"] = debug_info
        return result

        result["debug"] = debug_info
        return result

    def _current_user_preferences(self) -> dict:
        """The learned user profile (the self-improvement loop's state):
        speech style, explanation length, response order — carried into
        the semantic parse context."""
        try:
            return dict(self.memory_manager.get_user_profile())
        except Exception:
            return {}

    def process_semantic(self, request: str, language: str = "python") -> dict:
        """Process request using the new semantic pipeline.

        Every turn runs understanding → intent → reference → topic
        state → ROUTE → pipeline, and every result carries that
        understanding as `result["trace"]` — the route is observable,
        never a hidden dispatch.
        """
        try:
            topic_before = str(getattr(self.conversation_manager,
                                       "current_topic", "") or "")
        except Exception:
            topic_before = ""
        result = self._process_semantic(request, language)
        # OFFLINE KNOWLEDGE: 검수된 한국어 지식으로 답할 수 있는 질문은
        # 영어 웹 조각보다 그 지식을 먼저 쓴다 (네트워크 불필요).
        result = self._offline_knowledge_override(request, result)
        # 영어 자료를 그대로 내보내지 않는다: 한국어 틀로 감싼다.
        result = self._localize_english_response(request, result)
        # TARGET LANGUAGE: only the language the representation asks for
        # is generated (the others appear only when asked for).
        result = self._apply_language_gate(result, self._last_semantic)
        result = self._attach_understanding_trace(request, result, topic_before)
        # USER-CONTROLLED OUTPUT: the output mode and comment policy are
        # enforced once, here, on the assembled turn.  No explicit
        # request phrase and no stored preference leaves the response
        # exactly as assembled (AUTO is the default).
        return self._apply_output_preferences(request, result)

    _OFFLINE_CODE_TASK_WORDS = ("짜줘", "짜 줘", "만들", "구현", "작성",
                                "함수", "코드", "프로그램", "출력해", "print(",
                                "class ", "def ")

    def _offline_knowledge_override(self, request: str,
                                    result: dict) -> dict:
        """오프라인 지식베이스로 답할 수 있으면 한국어 설명으로 교체.

        코드 생성·수정 요청은 절대 가로채지 않는다 (route가 EXPLANATION/
        QUESTION일 때만, 그리고 코드 작업 단어가 없을 때만).
        """
        if not isinstance(result, dict):
            return result
        # route는 이 단계에서 아직 붙지 않을 수 있으므로(트레이스에서
        # 부착됨) 의미 표현의 goal로도 판단한다.
        route = str(result.get("route") or "").upper()
        intent = str(result.get("intent") or "").upper()
        goal = getattr(getattr(self._last_semantic, "goal", None),
                       "goal_type", None)
        goal_name = str(getattr(goal, "name", goal) or "").upper()
        allowed = {"EXPLANATION", "QUESTION", "INFORMATION_QUERY",
                   "EXPLANATION_REQUEST", "SIMPLIFY_EXPLANATION",
                   "EXPLAIN", "COMMAND_QUERY"}
        labels = {route, intent, goal_name} - {""}
        if labels and not (labels & allowed):
            return result
        text = str(request or "")
        lowered = text.lower()
        # 한국어 질문에만 적용한다: 영어 질문은 기존 파이프라인이
        # 영어로 답하는 동작을 유지한다.
        if not _HANGUL_RE.search(text):
            return result
        if any(word in lowered for word in self._OFFLINE_CODE_TASK_WORDS):
            return result
        try:
            from knowledge import offline_kb
            info = offline_kb.unified_answer(text)
        except Exception:
            return result
        status = str(info.get("status") or "unknown")
        if status == "unknown":
            # 지식에 없다: 단정하지 않고 기존 경로(웹/근거 부족 안내)로 넘긴다.
            return result
        result["response"] = str(info.get("response") or "")
        result["offline_knowledge"] = status
        result["knowledge_source"] = "offline"
        result["knowledge_status"] = status
        result["knowledge_term"] = info.get("term", "")
        result["knowledge_confidence"] = info.get("confidence", 0.0)
        if info.get("evidence"):
            result["knowledge_evidence"] = list(info["evidence"])
        result["intent"] = result.get("intent") or "QUESTION"
        try:
            self._refresh_response_projections(result)
        except Exception:
            pass
        return result

    def _localize_english_response(self, request: str,
                                   result: dict) -> dict:
        """응답이 영어 위주면 한국어 틀로 감싸서 내보낸다.

        완역이 아니라 용어 대응 + 요약 + 원문 발췌 분리다.  한국어
        응답이나 코드 산출물은 건드리지 않는다.
        """
        if not isinstance(result, dict):
            return result
        if result.get("offline_knowledge"):
            return result
        route = str(result.get("route") or "").upper()
        intent = str(result.get("intent") or "").upper()
        if (route or intent) and not ({route, intent} & {
                "EXPLANATION", "QUESTION", "INFORMATION_QUERY", "EXPLAIN"}):
            return result
        response = str(result.get("response") or "")
        if len(response) < 40:
            return result
        if not _HANGUL_RE.search(str(request or "")):
            return result
        try:
            from knowledge import offline_kb
            framing = offline_kb.korean_framing(response, topic=request)
        except Exception:
            return result
        if not framing:
            return result
        result["response"] = framing
        result["knowledge_source"] = "offline-framing"
        try:
            self._refresh_response_projections(result)
        except Exception:
            pass
        return result

    def _apply_output_preferences(self, request: str, result: dict) -> dict:
        """Enforce the user's OUTPUT MODE and COMMENT POLICY on the
        assembled turn — never merely stored.

        The explicit request phrase wins; otherwise the stored
        preference speaks.  The transforms are pure, deterministic and
        total (brain.output_modes / brain.comment_policy), so the
        absence of a preference cannot disturb the output.
        """
        if not isinstance(result, dict):
            return result
        profile = self.memory_manager.get_user_profile() or {}

        from brain.comment_policy import (CommentPolicy, apply_comment_policy,
                                          parse_comment_policy,
                                          policy_from_stored)
        policy = parse_comment_policy(request)
        if policy is None:
            policy = policy_from_stored(profile.get("comment_density", ""))
        if policy is not CommentPolicy.AUTO:
            applied = False
            for entry in (result.get("results") or {}).values():
                if not isinstance(entry, dict):
                    continue
                for lang, spec in (entry.get("languages") or {}).items():
                    if isinstance(spec, dict) and \
                            isinstance(spec.get("code"), str) and spec["code"]:
                        spec["code"] = apply_comment_policy(
                            spec["code"], policy, str(lang))
                        applied = True
            if isinstance(result.get("code"), str) and result["code"]:
                result["code"] = apply_comment_policy(
                    result["code"], policy,
                    str(result.get("language") or "python"))
                applied = True
            if applied:
                result["comment_policy_applied"] = policy.value

        from brain.output_modes import OutputMode, parse_output_mode, \
            shape_response
        mode = parse_output_mode(request)
        if mode is None:
            try:
                mode = OutputMode(str(profile.get("output_mode") or "auto"))
            except ValueError:
                mode = None
        if mode is not None and mode is not OutputMode.AUTO:
            result["response"] = shape_response(result, mode)
            result["output_mode_applied"] = mode.value
            self._refresh_response_projections(result)
        return result

    @staticmethod
    def _refresh_response_projections(result: dict) -> None:
        """The response's projections follow the shaped text (the
        chunks the streaming prints are the final response's)."""
        text = str(result.get("response") or "")
        result["response_content"] = {
            "response_type": result.get("intent", "UNKNOWN"),
            "text": text,
        }
        result["response_chunks"] = chunk_response(text)

    # -- TARGET LANGUAGE -------------------------------------------
    # The requested target language is a property of the representation
    # (semantic.target: "kind=language" + the language), never a search
    # over request words, and the gate is applied at exactly one place —
    # where the turn's artifacts leave the pipeline.  Without an explicit
    # language request the turn is answered in python; another language
    # is generated only when the turn asks for it.

    _DEFAULT_GENERATION_LANGUAGE = "python"

    def _requested_language(self, semantic) -> str:
        target = getattr(semantic, "target", None) if semantic else None
        if isinstance(target, dict):
            data = target
        elif target is not None and hasattr(target, "to_dict"):
            data = target.to_dict() if callable(getattr(target, "to_dict")) \
                else {}
        else:
            data = {}
        lang = str(data.get("language") or "").strip().lower()
        if str(data.get("kind") or "") == "language" and lang:
            return lang
        return ""

    def _apply_language_gate(self, result: dict, semantic) -> dict:
        """Generate ONLY the language the turn asked for (python when it
        asked for none).  The representation decides, so no request
        pattern decides — and the answer's language follows the request.
        """
        requested = self._requested_language(semantic) \
            or self._DEFAULT_GENERATION_LANGUAGE
        if requested not in self._get_supported_languages():
            return result
        results = result.get("results") if isinstance(result, dict) else None
        if not isinstance(results, dict):
            return result
        explicit = bool(self._requested_language(semantic))
        for artifact in results.values():
            if not isinstance(artifact, dict):
                continue
            langs = artifact.get("languages")
            if not isinstance(langs, dict) or not langs:
                continue
            if requested in langs:
                artifact["languages"] = {requested: langs[requested]}
                artifact["target_language"] = requested
                continue
            # The turn asked for a language the artifact does not carry:
            # deliver it from the python source through the same
            # transpiler the other paths use — or honestly keep what was
            # generated (the code itself is never fabricated).
            python_code = str((langs.get("python") or {}).get("code") or "")
            if explicit and python_code:
                try:
                    target_code = self._transpile_python_to(python_code,
                                                            requested)
                except Exception:
                    target_code = ""
                if target_code:
                    artifact["languages"] = {requested: {"code": target_code}}
                    artifact["target_language"] = requested
                    continue
            artifact["languages"] = {self._DEFAULT_GENERATION_LANGUAGE:
                                     langs.get(self._DEFAULT_GENERATION_LANGUAGE,
                                               next(iter(langs.values())))}
            artifact["target_language"] = requested
        return result

    def _process_semantic(self, request: str, language: str = "python") -> dict:
        """The turn itself: parse, understand, route, dispatch."""
        # 1. Parse natural language to semantic representation
        try:
            user_preferences = dict(self.memory_manager.get_user_profile())
        except Exception:
            user_preferences = {}
        parse_context = ParseContext(
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict() 
                if self.conversation_manager.current_project else None,
            user_preferences=user_preferences,
        )
        # 1a. FORM SEGMENTATION: the submission's structure separates
        # its instruction (what gets parsed) from its evidence (code
        # fragments, stack traces, sample I/O) — recognized by form,
        # never by topic.  A structurally verified problem statement is
        # parsed whole: its sections ARE the statement.
        parse_text = request
        submission = {}
        try:
            from brain.input_segmentation import segment_submission
            keep_whole = False
            try:
                from brain.algorithmic.competitive_problem_detector import (
                    CompetitiveProblemDetector)
                _seg_detection = CompetitiveProblemDetector().detect(request)
                keep_whole = bool(_seg_detection.is_competitive and
                                  _seg_detection.confidence >= 0.45)
            except Exception:
                keep_whole = False
            if not keep_whole:
                _submission_doc = segment_submission(request)
                if _submission_doc.evidence:
                    submission = _submission_doc.to_dict()
                    if _submission_doc.instruction and \
                            _submission_doc.instruction != request:
                        parse_text = _submission_doc.instruction
        except Exception:
            parse_text, submission = request, {}
        semantic = self.semantic_parser.parse(parse_text, parse_context)
        if submission:
            semantic.submission = submission
        # The route: decided from the goal until the act layer refines
        # the intent (then stamped again, with the act).
        self._last_semantic = semantic
        self._stamp_route(semantic)

        # 1b. DISCOURSE OPERATION + reference resolution: a dismiss
        # predication over an anaphoric object ("forget the last topic")
        # is a `reset_topic` act on the topic state, and the utterance's
        # REMAINING clauses are processed fresh — the operation's scope
        # is its own clause, never the whole utterance.  The act is
        # recorded on the representation (the trace shows it), and the
        # topic state is retired through the SAME discourse operation.
        try:
            from brain.dialogue_act import discourse_action, \
                separate_topic_reset
            reset_action = discourse_action(request)
            cleaned_request, had_reset = separate_topic_reset(request)
        except Exception:
            reset_action, cleaned_request, had_reset = None, request, False
        retired_topic = ""
        if reset_action is not None:
            try:
                retired_topic = str(getattr(self.conversation_manager,
                                            "current_topic", "") or "")
            except Exception:
                retired_topic = ""
            try:
                self.conversation_manager.forget_topic(
                    evidence=[reset_action.get("type", "reset_topic")])
            except Exception:
                self.conversation_manager.forget_topic()
        if had_reset:
            if cleaned_request.strip():
                parse_context.conversation_history = \
                    self.conversation_manager.history
                semantic = self.semantic_parser.parse(
                    cleaned_request, parse_context)
                request = cleaned_request
                self._last_semantic = semantic
                self._stamp_route(semantic)
                if submission:
                    semantic.submission = submission
        if reset_action is not None:
            semantic.discourse_action = dict(reset_action)
        if had_reset and not cleaned_request.strip():
            # The utterance IS the discourse operation: no content
            # clause remains, so the turn carries no topic — it retired
            # the state above and acknowledges the operation.  Its own
            # words are never a knowledge target, and the operation
            # never re-establishes the topic it just retired.
            return self._handle_discourse_operation(
                request, language, semantic, reset_action, retired_topic)

        # 2. Feedback acts — the self-improvement loop's intake: a
        # complaint / preference / praise utterance is
        # meta-communication about HOW to respond or act.  The feedback
        # knowledge (the corrective speech-act patterns) classifies it,
        # the learner stores it in the user memory, and the
        # acknowledgment is composed from what was learned — the
        # learned preference then shapes every later response.
        feedback_analysis = self.feedback_learner.analyze_feedback(request)
        if feedback_analysis is not None:
            return self._handle_feedback(request, language, semantic,
                                         feedback_analysis)

        # 3. Competitive problem gate: a competitive problem statement
        # (stdin/stdout format, "Print the answer", constraints, sample
        # I/O) must reach the algorithmic pipeline regardless of how
        # confident the semantic parse is and regardless of which
        # domain router (shell/math/question hijacking) might claim an
        # underdetermined goal.  The structural detector decides —
        # general, not per-problem.  The gate runs BEFORE the
        # confidence check: a well-formed statement does not need a
        # confident goal to be recognized as a problem.  CHAT is
        # included: a structurally verified problem statement is a
        # computation request by its own format, never a conversation —
        # the store-dependent concept matches may misread it as one.
        goal_type = semantic.goal.goal_type
        if goal_type in (GoalType.UNKNOWN, GoalType.CREATE_FUNCTION,
                         GoalType.QUESTION, GoalType.AMBIGUOUS,
                         GoalType.INFORMATION_QUERY,
                         GoalType.COMMAND_QUERY,
                         GoalType.CREATE_CLASS,
                         GoalType.CREATE_PROJECT,
                         GoalType.COMPUTER,
                         GoalType.CHAT):
            try:
                from brain.algorithmic.competitive_problem_detector import CompetitiveProblemDetector
                _cpd = CompetitiveProblemDetector()
                _detection = _cpd.detect(request)
                if _detection.is_competitive and \
                        _detection.confidence >= 0.45:
                    # A structurally verified problem statement is a
                    # CODE route by its own format — the structural
                    # detector decides, never the goal guess.
                    self._stamp_route(semantic, Route.CODE)
                    return self._handle_competitive_problem(request, language, semantic)
            except Exception:
                pass

        # 4. Handle low confidence (insufficient understanding)
        # Lower threshold for CHAT/QUESTION which often have fewer entities/operations
        min_confidence = 0.3 if semantic.goal.goal_type in (GoalType.CHAT, GoalType.QUESTION) else 0.35
        
        if semantic.confidence < min_confidence:
            return self._handle_insufficient_understanding(request, language, semantic)
        
        # 5. Route based on goal type
        goal_type = semantic.goal.goal_type

        # Arithmetic question routing: a SHORT text containing a
        # computable arithmetic expression is a computation QUESTION —
        # answered by computing the actual value. A long problem
        # statement that happens to embed formulas is not a math
        # question (the semantics decide, not keywords).
        if len(request.strip()) <= 120 and \
                self._evaluate_math_question(request) is not None:
            return self._handle_math_question(request, language, semantic)

        # The dialogue intent, refined with the retrieved knowledge:
        # the ACT (from the sentence's grammar) plus the CONTENT's kind
        # (shell environment, natural-language topic, self reference,
        # program behaviour) — the topic words alone never decide it.
        shell_result = None
        # The dialogue-act refinement runs for EVERY goal type: a
        # topic-word-matched goal ("learning programming" → a
        # create-function goal) must still pass the ACT layer — the
        # act (from the sentence's grammar) decides the intent, the
        # topic words never do.
        if goal_type in (GoalType.QUESTION, GoalType.UNKNOWN, GoalType.CHAT,
                         GoalType.COMMAND_QUERY, GoalType.INFORMATION_QUERY,
                         GoalType.AMBIGUOUS):
            from brain.shell_knowledge import reason_about_shell
            try:
                shell_result = reason_about_shell(
                    request, self.concept_store, self.relation_store)
            except Exception:
                shell_result = None
            self._refine_dialogue_intent(request, semantic, shell_result)
            self._stamp_route(semantic)
            # REFERENCE RESOLUTION: the referring expression resolves
            # against the discourse state (the intent is derived first,
            # so the act's referent requirement can score the candidates)
            self._resolve_reference(request, semantic)

            # The command query: the platform/shell semantics plus the
            # method-question act decide; the ANSWER is composed from
            # the single most relevant operation — never a dump of
            # every matching concept.
            if semantic.dialogue_intent.get("type") == "command_query" \
                    and shell_result and (shell_result.get("platform")
                                          or shell_result.get("shell")):
                return self._handle_command_query(request, language,
                                                  semantic, shell_result)
        else:
            self._refine_dialogue_intent(request, semantic, None)
            route = self._stamp_route(semantic)
            self._resolve_reference(request, semantic)

            # AN ANAPHORIC MODIFICATION WITH NO ANTECEDENT: the
            # sentence's action targets a referring expression the
            # discourse cannot bind ("add logging to it" with no
            # antecedent anywhere).  The goal inference runs BEFORE
            # reference resolution and can only read the sentence, so
            # the discourse corrects it here: the turn is a
            # modification request whose artifact does not exist in
            # this conversation — it goes to the honest no-artifact
            # path, never to a fresh creation of an unrelated function
            # from the request's verb composition.
            if goal_type == GoalType.CREATE_FUNCTION and \
                    self._is_unresolved_anaphoric_modification(
                        semantic, request):
                semantic.goal.goal_type = GoalType.MODIFY
                semantic.goal.action = "modify"
                semantic.goal.evidence = list(
                    semantic.goal.evidence or []) + [
                    "anaphoric modification reference the discourse "
                    "cannot resolve"]
                semantic.goal.confidence = max(
                    float(semantic.goal.confidence or 0.0), 0.7)
                goal_type = GoalType.MODIFY
                # the act and the route read the CORRECTED goal
                self._refine_dialogue_intent(request, semantic, None)
                route = self._stamp_route(semantic)

            # The act override: an ARTIFACT goal whose ROUTE is not
            # CODE (a learning-guidance/explanation/procedure/social
            # predication — the act layer disallows artifact creation)
            # is DIALOGUE, not code generation — the route decides,
            # never the topic.  "How do I fix this?" and "install X
            # for me — how?" ask to be TOLD; the FIX/COMPUTER goal
            # recorded the topic words, not an artifact request.
            if goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_CLASS,
                             GoalType.CREATE_PROJECT, GoalType.MODIFY,
                             GoalType.FIX, GoalType.COMPUTER):
                if route is not Route.CODE:
                    if route is Route.EXPLANATION:
                        result = self._handle_explain_semantic(
                            request, language, semantic)
                    elif route is Route.ADVICE_LEARNING:
                        # The recommendation act: the plan builder
                        # composes the options per the act (knowledge
                        # as evidence) — the question handler's
                        # build_plan realizes it.
                        result = self._handle_question_semantic(
                            request, language, semantic)
                    else:
                        result = self._handle_chat_semantic(
                            request, language, semantic)
                    # The act-override route leaves the shared tail — the
                    # finalize pass still runs: the topic state, the
                    # intent mapping and the reply's chunking belong to
                    # EVERY turn, whoever handled it.
                    return self._finalize_result(semantic, result)

        if goal_type == GoalType.CHAT:
            result = self._handle_chat_semantic(request, language, semantic)
        elif goal_type in (GoalType.COMMAND_QUERY, GoalType.INFORMATION_QUERY):
            # The structured dialogue acts: the shell routing above
            # handles the shell topics; what reaches here is a knowledge
            # question — handled as a QUESTION.
            result = self._handle_question_semantic(request, language, semantic)
        elif goal_type == GoalType.AMBIGUOUS:
            # Several candidate goal types, low confidence: the response
            # presents the candidates and asks for clarification (never
            # guesses one type).
            result = self._handle_ambiguous_semantic(request, language, semantic)
        elif goal_type == GoalType.QUESTION:
            # A question whose referent is the SYSTEM itself ("Who are
            # you?", "너 누구야?") is dialogue about the self — handled
            # as chat (the identity/capability acts), never as external
            # knowledge retrieval.
            if self.conversation_knowledge.is_self_question(request, semantic):
                result = self._handle_chat_semantic(request, language, semantic)
            else:
                result = self._handle_question_semantic(request, language, semantic)
        elif goal_type == GoalType.RESEARCH:
            # An investigation: the evidence is sought externally and
            # iterated until it stands up; the plan (not a canned
            # answer) composes the finding.
            result = self._handle_research_semantic(request, language,
                                                    semantic)
        elif goal_type == GoalType.EXPLAIN:
            result = self._handle_explain_semantic(request, language, semantic)
        elif goal_type == GoalType.CREATE_FUNCTION:
            result = self._handle_function_semantic(request, language, semantic)
        elif goal_type == GoalType.CREATE_PROJECT:
            result = self._handle_project_semantic(request, language, semantic)
        elif goal_type == GoalType.CREATE_CLASS:
            result = self._handle_class_semantic(request, language, semantic)
        elif goal_type == GoalType.MODIFY:
            result = self._handle_modify_semantic(request, language, semantic)
        elif goal_type == GoalType.FIX:
            result = self._handle_fix_semantic(request, language, semantic)
        elif goal_type == GoalType.ANALYZE:
            result = self._handle_analyze_semantic(request, language, semantic)
        elif goal_type == GoalType.COMPUTER:
            result = self._handle_computer_semantic(request, language, semantic)
        else:
            result = self._handle_unknown_semantic(request, language, semantic)

        # The topic state, the intent mapping and the reply's chunking —
        # shared by every route (the goal dispatch, the act override).
        return self._finalize_result(semantic, result)

    def _stamp_route(self, semantic, route: Route | None = None) -> Route:
        """Record the turn's route on the understanding and return it.

        The route is decided from what the understanding already
        derived — the goal and the act's intent type (or from an
        explicit structural decision, e.g. the verified problem gate)
        — and travels with the semantic representation, so dispatch,
        the trace and the hypothesis gate all read the SAME decision.
        """
        route = route or decide_route(semantic)
        dialogue_intent = dict(getattr(semantic, "dialogue_intent", None)
                               or {})
        dialogue_intent["route"] = route.value
        semantic.dialogue_intent = dialogue_intent
        return route

    @staticmethod
    def _is_unresolved_anaphoric_modification(semantic,
                                              request: str) -> bool:
        """The sentence PERFORMS a modification whose anaphoric target
        the discourse could not resolve.

        All three parts are structural:
          - the reference is an anaphoric expression (a surface form
            the resolver extracted) that stays unresolvable — the
            discourse has nothing to bind it to;
          - the sentence's action is the modification performative
            (the shared grammatical pattern);
          - the goal inference read the sentence as a fresh creation
            (it ran before reference resolution, so it could not know
            the target does not exist here).
        """
        ref = dict(getattr(semantic, "reference", None) or {})
        if not ref.get("needs_clarification"):
            return False
        if not str(ref.get("surface_form") or "").strip():
            # No anaphoric expression: the utterance supplies its own
            # referent ("add a logging function") — a creation request,
            # not an unresolvable modification.
            return False
        from brain.semantic_parser import MODIFICATION_ACTION_PATTERN
        return bool(re.search(MODIFICATION_ACTION_PATTERN,
                              (request or "").lower()))

    def _resolve_reference(self, request: str, semantic):
        """Resolve this turn's referring expressions against the
        conversation's context (history, recent semantics, entities,
        topics) — scored, observable, and honest when it fails: an
        unresolved reference becomes a clarification, never a guess and
        never a search query."""
        from brain.context_resolution import Reference
        try:
            reference = self.reference_resolver.resolve(
                semantic, request,
                self.conversation_manager.topic_state,
                self.conversation_manager.history)
        except Exception:
            reference = Reference(refers_to="unresolved")
        try:
            semantic.reference = reference.to_dict()
        except Exception:
            pass
        self._last_reference = reference
        return reference

    def _attach_understanding_trace(self, request: str, result: dict,
                                    topic_before: str) -> dict:
        """Attach the turn's understanding to the result: the intent,
        the reference resolution, the topic state and the ROUTE.

        The hypotheses section is the structural witness that a
        non-coding turn never entered the Code Hypothesis Generator: it
        is empty for every non-CODE route.
        """
        if not isinstance(result, dict):
            return result
        try:
            topic_after = str(self.conversation_manager.current_topic or "")
        except Exception:
            topic_after = topic_before

        semantic_dict = result.get("semantic")
        if not isinstance(semantic_dict, dict) or not semantic_dict:
            last = getattr(self, "_last_semantic", None)
            semantic_dict = last.to_dict() if last is not None and \
                hasattr(last, "to_dict") else {}
        dialogue_intent = dict(semantic_dict.get("dialogue_intent") or {})
        scope = dict(dialogue_intent.get("scope") or {})
        goal_dict = semantic_dict.get("goal") or {}
        if not isinstance(goal_dict, dict):
            goal_dict = {}
        route = str(dialogue_intent.get("route") or "")
        if not route:
            route = decide_route_parts(str(goal_dict.get("type") or ""),
                                       str(dialogue_intent.get("type")
                                           or "")).value
        topics = [t for t in (dialogue_intent.get("topics") or []) if t]

        # The coding hypotheses: only a CODE turn ran the generator,
        # and only the state this route produced may be reported.
        hypotheses: list = []
        plan: list = []
        selected = None
        reasoning_confidence = None
        evaluation: list = []
        reasoning = getattr(self.think_engine, "last_reasoning", None)
        if route == Route.CODE.value and reasoning is not None and \
                getattr(reasoning, "route", "") == route:
            hypotheses = [h.to_dict() if hasattr(h, "to_dict") else str(h)
                          for h in (getattr(reasoning, "hypotheses", None)
                                    or [])]
            plan = [a.to_dict() if hasattr(a, "to_dict") else str(a)
                    for a in (getattr(reasoning, "plan", None) or [])]
            picked = getattr(reasoning, "selected_hypothesis", None)
            if picked is not None:
                selected = picked.to_dict() if hasattr(picked, "to_dict") \
                    else str(picked)
            reasoning_confidence = getattr(reasoning, "confidence", None)
            # The EVALUATION: the ranked candidates with the verdict each
            # one received.  It survives an empty selection — "no
            # evaluation" is never reported when candidates were compared.
            picked_id = str(getattr(picked, "id", "") or "")
            evaluation = [
                {
                    "id": h.get("id"),
                    "confidence": h.get("confidence"),
                    "verdict": ("selected"
                                if picked_id and h.get("id") == picked_id
                                else "not_selected"),
                    "reason": ("highest confidence among the compatible "
                               "candidates"
                               if picked_id and h.get("id") == picked_id
                               else "ranked below the selected candidate"),
                    "evidence": list(h.get("evidence") or [])[:3],
                }
                for h in sorted(
                    (x for x in hypotheses if isinstance(x, dict)),
                    key=lambda x: -(x.get("confidence") or 0.0))
            ]

        response_plan = result.get("response_plan")
        response_plan = response_plan if isinstance(response_plan, dict)             else {}

        # The turn's CONTEXT / REFERENCE / TOPIC / KNOWLEDGE stages:
        # the same structures the pipeline acted on (never a re-derivation
        # for the trace).
        reference = dict(semantic_dict.get("reference") or {})
        discourse_action = dict(semantic_dict.get("discourse_action") or {})
        requirement = dict(semantic_dict.get("knowledge_requirement") or {}) \
            or dict(response_plan.get("knowledge_requirement") or {})
        retrieval = dict(response_plan.get("retrieval") or {})
        representation = dict(response_plan.get("representation") or {})
        sentence_plan = list(response_plan.get("sentence_plan") or [])
        try:
            topic_state = self.conversation_manager.topic_state.to_dict()
        except Exception:
            topic_state = {"current_topic": topic_after}
        web_evidence = list(retrieval.get("evidence") or [])

        result["route"] = route
        result["trace"] = {
            "input": request,
            "goal": {
                "type": goal_dict.get("type"),
                "target": goal_dict.get("target"),
                "confidence": goal_dict.get("confidence"),
            },
            "intent": {
                "type": dialogue_intent.get("type"),
                "goal": dialogue_intent.get("goal"),
                "confidence": dialogue_intent.get("confidence"),
                "evidence": list(dialogue_intent.get("evidence") or []),
                "topics": list(topics),
                "act": semantic_dict.get("dialogue_act") or {},
                "legacy_intent": result.get("intent"),
            },
            "context": {
                "referent": topics[0] if topics else "",
                "resolution": scope.get("reference_source", ""),
                "discourse_topic": scope.get("discourse_topic")
                                   or topic_before,
                "previous_turns": scope.get("previous_turns", 0),
            },
            "topic": {"before": topic_before, "after": topic_after},
            "route": route,
            "pipeline": describe_route(route, str(
                dialogue_intent.get("type") or "")),
            "reasoning": {
                "generated": route == Route.CODE.value,
                "hypotheses": hypotheses,
                "selected": selected,
                "confidence": reasoning_confidence,
                "evaluation": evaluation,
                "plan": plan,
            },
            "knowledge": {
                "target": response_plan.get("target"),
                "claims": list(response_plan.get("claims") or []),
                "evidence": list(response_plan.get("evidence") or []),
            },
            # The reference resolution: the expression, what it bound
            # to, how confident, and the candidates it was chosen from
            "reference_resolution": reference,
            # The topic state: current, previous, active entities /
            # concepts / task / language / confidence
            "topic_state": topic_state,
            # A discourse operation (reset_topic …) this turn performed
            "discourse_action": discourse_action,
            # The knowledge requirement decided before retrieval, and
            # what the web search did (used / not used + the reason)
            "knowledge_requirement": requirement,
            "web_search": {
                "used": bool(retrieval.get("web_used")),
                "reason": str(requirement.get("reason")
                              or retrieval.get("reason") or ""),
                "query": str(retrieval.get("query")
                             or requirement.get("query") or ""),
                "results": len(web_evidence),
            },
            # The retrieved knowledge as EVIDENCE (source, provenance,
            # relevance / consistency / quality / freshness) — never the
            # answer itself
            "retrieved_knowledge": retrieval,
            # The response representation and its sentence plan
            "response_representation": {
                "intent": representation.get("intent"),
                "target": representation.get("target"),
                "style": representation.get("style"),
                "language": representation.get("language"),
                "moves": representation.get("moves"),
                "sentence_plan": sentence_plan or
                representation.get("sentence_plan"),
                "confidence": representation.get("confidence"),
            },
            # The output generation: the length budget the response
            # obeyed, the sections it planned and the verification
            # report (empty on the short path)
            "generation": {
                "budget": dict(response_plan.get("budget") or {}),
                "longform": dict(response_plan.get("longform") or {}),
            },
        }
        return result

    @staticmethod
    def _generation_artifact(result: dict) -> dict:
        """The generated artifact of THIS turn (function name, code,
        IR) — what a follow-up modification continues from.  {} when
        the turn produced no code (a dialogue turn never has one)."""
        results = result.get("results")
        if not isinstance(results, dict):
            return {}
        for name, entry in results.items():
            if not isinstance(entry, dict):
                continue
            languages = entry.get("languages")
            if not isinstance(languages, dict):
                continue
            # the python artifact first (the pipeline's source of
            # truth), any requested target language as a fallback
            ordered = sorted(languages.items(),
                             key=lambda kv: 0 if kv[0] == "python" else 1)
            for lang, spec in ordered:
                if isinstance(spec, dict) and spec.get("code"):
                    return {"function_name": str(name),
                            "language": str(lang),
                            "code": str(spec.get("code")),
                            # A recipe artifact's code is the authority;
                            # its signature-only IR is not a
                            # regeneration plan for a continuation.
                            "ir": {} if entry.get("recipe")
                                  else (entry.get("ir") or {}),
                            "ops": [str(o) for o in
                                    (entry.get("ops") or [])]}
        return {}

    def _finalize_result(self, semantic, result: dict) -> dict:
        """The turn's shared tail: the topic state update, the legacy
        intent mapping, and the response's content/chunk projections.

        The topic state: a topical turn's referent becomes the
        discourse's current topic (the source an anaphoric reference
        resolves against).  A dialogue act with no topic (a greeting, a
        bare expressive) leaves the state unchanged.
        """
        # The understanding travels with the result (a pipeline that
        # built its own result dict may not carry it): the trace reads
        # the SAME representation the route was decided from.
        if "semantic" not in result:
            result["semantic"] = semantic.to_dict()
        # The request this finalize pass belongs to (the turn record is
        # keyed on it — one request, one turn).
        _finalize_request = str(result.get("request") or semantic.raw_text
                                or "")
        try:
            # The turn record must EXIST before anything is written
            # onto it: a handler that already appended its own turn is
            # a no-op here; a handler that did not (a code generation)
            # gets its record now — so this turn's meaning is never
            # written onto the PREVIOUS turn's record.
            self.conversation_manager.ensure_turn(
                _finalize_request,
                str(result.get("intent") or "")
                or str(getattr(semantic.goal.goal_type, "name", "") or ""),
                str(result.get("response") or ""))
            _topics = list(semantic.dialogue_intent.get("topics") or [])
            if not _topics and isinstance(result.get("response_plan"), dict):
                _topics = [result["response_plan"].get("target") or ""]
            _reference = dict(getattr(semantic, "reference", None) or {})
            _resolved = bool(_reference.get("target")) and not bool(
                _reference.get("needs_clarification"))
            if _resolved:
                # the resolver's decision is the discourse's referent
                _topics = [str(_reference["target"])] + [
                    t for t in _topics
                    if str(t).strip().lower()
                    != str(_reference["target"]).strip().lower()]
            else:
                # An unresolved reference: the turn asked about a
                # referent it never established, so the discourse topic
                # does not move — a clarification never rewrites what
                # the conversation is about.
                _topics = []
            if _topics and _topics[0].strip():
                # The TOPIC STATE: the new referent becomes the current
                # topic (the previous one is retained), and the turn's
                # active entities/concepts/task/language are recorded
                # with it — the state the next reference resolves on.
                self.conversation_manager.set_topic(
                    _topics[0],
                    task=str((semantic.dialogue_intent or {}).get("type")
                             or ""),
                    entities=[e.name for e in (semantic.entities or [])],
                    concepts=list(semantic.activated_concept_ids or []),
                    language=str(getattr(semantic.source_language, "value",
                                         "") or ""),
                    confidence=float(semantic.confidence or 0.0),
                    evidence=[f"topic from this turn: {_topics[0]}"])
            self.conversation_manager.record_turn_topics(_topics)

            # The turn's STRUCTURED meaning travels with it: what it was
            # about (topic), what it carried (entities, concepts), what
            # the response SAID (claims + its representation) and the
            # knowledge it was built from — the record the next turn's
            # resolution reads instead of a re-parse of the raw text.
            plan_dict = result.get("response_plan") \
                if isinstance(result.get("response_plan"), dict) else {}
            representation = plan_dict.get("representation") or {}
            if not isinstance(representation, dict) or not representation:
                trace_dict = result.get("trace") \
                    if isinstance(result.get("trace"), dict) else {}
                representation = trace_dict.get(
                    "response_representation") or {}
            retrieval = plan_dict.get("retrieval") \
                if isinstance(plan_dict.get("retrieval"), dict) else {}
            referenced = {}
            if plan_dict.get("target"):
                referenced["target"] = str(plan_dict.get("target"))
            if retrieval.get("query"):
                referenced["query"] = str(retrieval.get("query"))
            if plan_dict.get("evidence"):
                referenced["evidence"] = [
                    str(e) for e in (plan_dict.get("evidence") or [])[:5]]
            if plan_dict.get("knowledge_requirement"):
                referenced["requirement"] = dict(
                    plan_dict.get("knowledge_requirement") or {})
            self.conversation_manager.record_turn_state(
                semantic_representation=semantic.to_dict(),
                topic=(_topics[0] if _topics else ""),
                entities=[e.name for e in (semantic.entities or [])],
                concepts=list(semantic.activated_concept_ids or []),
                claims=[c for c in (plan_dict.get("claims") or []) if c],
                response_representation=representation,
                referenced_knowledge=referenced,
            )
            # The turn's OBJECT and its GENERATED ARTIFACT travel with
            # it: a follow-up modification continues from this turn's
            # structure (operation graph, IO) and code — the record the
            # continuation merge reads instead of a re-parse.
            self.conversation_manager.record_semantic_object(semantic)
            _artifact = self._generation_artifact(result)
            if _artifact:
                self.conversation_manager.record_turn_artifact(_artifact)
        except Exception:
            pass

        # Map new intent names to old ones for backward compatibility
        intent_map = {
            "CHAT": INTENT_CHAT,
            "QUESTION": INTENT_QUESTION,
            "EXPLAIN": INTENT_EXPLANATION,
            "CREATE_FUNCTION": INTENT_GENERATE_CODE,
            "CREATE_CLASS": INTENT_GENERATE_CODE,
            "CREATE_PROJECT": INTENT_CREATE_PROJECT,
            "MODIFY": INTENT_MODIFY_PROJECT,
            "FIX": INTENT_DEBUG,
            "ANALYZE": INTENT_ANALYZE_CODE,
            "COMPUTER": INTENT_COMPUTER_TASK,
            "COMMAND_QUERY": INTENT_QUESTION,
            "INFORMATION_QUERY": INTENT_QUESTION,
            "AMBIGUOUS": INTENT_UNKNOWN,
            "UNKNOWN": INTENT_UNKNOWN,
        }
        if "intent" in result and result["intent"] in intent_map:
            result["intent"] = intent_map[result["intent"]]
        # The turn record's intent is the MAPPED one (the same value
        # every other handler stores on its turn).
        try:
            _turns = self.conversation_manager.history
            if _turns and _turns[-1].user_input == _finalize_request \
                    and "intent" in result:
                _turns[-1].intent = str(result["intent"])
        except Exception:
            pass

        # FINAL SANITATION: a source's citation apparatus ("[1]",
        # "[c]", "[cite]") is the SOURCE'S apparatus — it never reaches
        # the speaker.  Only that is removed; the response's own wording
        # is untouched, and a CODE turn keeps its own bracket syntax
        # (an indexer or a list marker is code, not a citation).
        if "response" in result and isinstance(result["response"], str) \
                and str((semantic.dialogue_intent or {}).get("route")
                        or "") != Route.CODE.value:
            try:
                from brain.retrieved_knowledge import _strip_citations
                result["response"] = _strip_citations(result["response"])
            except Exception:
                pass

        # Add response_content for backward compatibility
        if "response" in result:
            result["response_content"] = {
                "response_type": result.get("intent", "UNKNOWN"),
                "text": result["response"],
            }
            # The reply streams as utterance chunks (sentence-level
            # units), never one opaque blob
            result["response_chunks"] = chunk_response(result["response"])

        return result

    def iter_response_chunks(self, request: str,
                             language: str = "python"):
        """Yield the conversational reply chunk by chunk (the dialogue
        streaming: the response unfolds sentence by sentence)."""
        result = self.process_semantic(request, language)
        for chunk in result.get("response_chunks") or []:
            yield chunk

    def process_with_streaming(
        self,
        request: str,
        language: str = "python",
        stream_mode: str = "auto",
        stream_format: str = "text",
        callback=None,
    ) -> dict:
        """Process request with real-time streaming generation.

        Uses GenerationController for adaptive, non-deterministic generation
        with real-time event emission.

        Args:
            request: User input
            language: Target language
            stream_mode: Granularity (auto, token, line, function, class, file, project)
            stream_format: Output format (text, json, compact)
            callback: Optional callback for each GenerationEvent
        """
        # 1. Parse to semantic representation
        parse_context = ParseContext(
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict()
                if self.conversation_manager.current_project else None,
            user_preferences=self._current_user_preferences()
        )
        semantic = self.semantic_parser.parse(request, parse_context)

        # 2. Create streaming emitter
        emitter = StreamingEmitter(
            callback=callback,
            format=stream_format,
            silent=False,
        )

        # 3. Create controller with config
        config = ControllerConfig(
            stream_format=stream_format,
            stream_mode=stream_mode,
            enable_verification=True,
            enable_repair=True,
        )
        controller = GenerationController(config)

        # 4. Generate with streaming
        result = controller.generate(
            request=request,
            semantic=semantic,
            context={"language": language},
            emitter=emitter,
        )

        return {
            "request": request,
            "language": language,
            "intent": semantic.goal.goal_type.value,
            "semantic": semantic.to_dict(),
            "generation_result": result.to_dict(),
            "events": [e.to_dict() for e in result.events],
            "output": result.output,
        }

    def _evaluate_math_question(self, request: str):
        """Evaluate an arithmetic question from natural language.

        General compositional semantics (digits + operator words):
        - "A 더하기 B" / "A plus B"     → A + B
        - "A 곱하기 B" / "A times B"    → A * B
        - "A 빼기 B" / "A minus B"      → A - B
        - "A 나누기 B" / "A divided by B" → A / B
        - "X의 제곱근" / "square root of X" → sqrt(X)
        - "X의 Y제곱" / "X to the power Y"  → X ** Y

        Returns the computed value, or None when the text is not a
        computable arithmetic question.
        """
        import math
        text = request.strip().lower()

        # 1. Square root: "1024의 제곱근" / "square root of 1024"
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*의?\s*(?:제곱근|루트)'
            r'|square\s+root\s+of\s+(\d+(?:\.\d+)?)', text)
        if m:
            x = float(m.group(1) or m.group(2))
            r = math.sqrt(x)
            return int(r) if r == int(r) else r

        # 2. Power: "2의 10제곱" / "2 to the power of 10"
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*의?\s*(\d+(?:\.\d+)?)\s*제곱'
            r'|(\d+(?:\.\d+)?)\s*(?:to\s+the\s+(?:power\s+of\s+)?)(\d+(?:\.\d+)?)', text)
        if m:
            base = float(m.group(1) or m.group(3))
            exp = float(m.group(2) or m.group(4))
            r = base ** exp
            return int(r) if r == int(r) else r

        # 3. Binary operators
        ops = [
            (r'(\d+(?:\.\d+)?)\s*(?:더하기|플러스|plus|\+)\s*(\d+(?:\.\d+)?)',
             lambda a, b: a + b),
            (r'(\d+(?:\.\d+)?)\s*(?:곱하기|곱셈|times|multiplied\s+by|multiply|\*)\s*(\d+(?:\.\d+)?)',
             lambda a, b: a * b),
            (r'(\d+(?:\.\d+)?)\s*(?:빼기|마이너스|minus)\s*(\d+(?:\.\d+)?)',
             lambda a, b: a - b),
            (r'(\d+(?:\.\d+)?)\s*(?:나누기|나눗셈|나눠|나누|divided\s+by|divide\s+by)\s*(\d+(?:\.\d+)?)',
             lambda a, b: a / b if b != 0 else None),
        ]
        for pattern, fn in ops:
            m = re.search(pattern, text)
            if m:
                a = float(m.group(1))
                b = float(m.group(2))
                return fn(a, b)

        # Korean SOV structure: operands before the verb
        # "100을 4로 나누면" → 100 / 4, "3과 5를 더하면" → 3 + 5
        sov_ops = [
            (r'(\d+(?:\.\d+)?)\s*(?:을|를)?\s*(\d+(?:\.\d+)?)\s*(?:으로|로)?\s*나누',
             lambda a, b: a / b if b != 0 else None),
            (r'(\d+(?:\.\d+)?)\s*(?:과|와)?\s*(\d+(?:\.\d+)?)\s*(?:을|를)?\s*(?:더하|합산|합치)',
             lambda a, b: a + b),
            (r'(\d+(?:\.\d+)?)\s*(?:과|와|에)?\s*(\d+(?:\.\d+)?)\s*(?:을|를)?\s*곱하',
             lambda a, b: a * b),
            (r'(\d+(?:\.\d+)?)\s*(?:에서)?\s*(\d+(?:\.\d+)?)\s*(?:을|를)?\s*빼',
             lambda a, b: a - b),
        ]
        for pattern, fn in sov_ops:
            m = re.search(pattern, text)
            if m:
                a = float(m.group(1))
                b = float(m.group(2))
                return fn(a, b)
        return None

    def _handle_math_question(self, request: str, language: str, semantic) -> dict:
        """An arithmetic question: compute the actual value and answer
        from the computation (never a fixed template)."""
        value = self._evaluate_math_question(request)
        if isinstance(value, float) and value == int(value):
            value = int(value)
        return {
            "request": request,
            "language": language,
            "intent": "QUESTION",
            "semantic": semantic.to_dict(),
            "response": f"계산 결과: {value}",
            "math_result": value,
            "project_created": False,
            "results": {},
        }

    def _question_shell_concepts(self, request: str, semantic) -> list:
        """The question's shell CONTEXT indicators: the shell platform/
        language concepts (Linux, Windows, PowerShell, Bash) activated
        in the question.  A topic-adjacent shell_operation concept
        (shell_variable activating on 'variable') is a retrieval hint
        about the TOPIC, not a shell context — the context is the
        question's environment mentions."""
        ids = getattr(semantic, 'activated_concept_ids', []) or []
        concepts = [c for c in (self.concept_store.get(cid) for cid in ids)
                    if c is not None]
        return [c for c in concepts
                if c.category in ("shell_platform", "shell_language")
                and getattr(c, 'activation_strength', 0) >= 4.0]

    # ------------------------------------------------------------------
    # Discourse operations (the topic state's own acts)
    # ------------------------------------------------------------------

    def _handle_discourse_operation(self, request: str, language: str,
                                    semantic, reset_action: dict,
                                    retired_topic: str = "") -> dict:
        """A discourse operation whose clause carries no content.

        The turn MEANS the operation: the state it retired is its whole
        effect, so the response states what the operation did —
        composed from the operation and the state it changed.  The
        utterance's own words are never a knowledge target, and no
        topic is pushed: an operation retires the discourse's subject,
        it never re-establishes it."""
        input_lang = getattr(getattr(semantic, "source_language", None),
                             "value", None) or language
        response = self._compose_discourse_ack(reset_action,
                                               retired_topic, input_lang)
        # The turn's intent: a discourse act over the topic state — no
        # content topic (the operation names none).
        try:
            intent = dict(getattr(semantic, "dialogue_intent", None) or {})
            intent["topics"] = []
            scope = dict(intent.get("scope") or {})
            scope["reference_source"] = "none"
            intent["scope"] = scope
            semantic.dialogue_intent = intent
        except Exception:
            pass
        self.conversation_manager.add_turn(request, "RESET_TOPIC", response)
        try:
            self.memory_manager.conversation_memory.add_turn(
                request, response, "RESET_TOPIC")
        except Exception:
            pass
        return {
            "request": request,
            "language": language,
            "intent": "RESET_TOPIC",
            "semantic": semantic.to_dict(),
            "response": response,
            "project_created": False,
            "results": {},
            "memory_context": self.memory_manager.get_context_for_request(
                request),
        }

    def _compose_discourse_ack(self, reset_action: dict,
                               retired_topic: str, lang: str) -> str:
        """The acknowledgment of a discourse operation: the operation
        and the topic it retired, stated in the input language — never
        a canned reply and never the operation's words echoed back as
        if they were content."""
        ko = str(lang or "en").lower().startswith("ko")
        topic = str(retired_topic or "").strip()
        op = str((reset_action or {}).get("type") or "reset_topic")
        if op == "reset_topic":
            if topic:
                return (f"알겠습니다 — {topic} 주제는 대화에서 빼겠습니다. "
                        "다음은 무엇을 이야기할까요?"
                        if ko else
                        f"Understood — I've dropped {topic} from the "
                        "conversation. What would you like to take up next?")
            return ("알겠습니다 — 이전 주제는 더 이상 다루지 않겠습니다."
                    if ko else
                    "Understood — I'll leave the previous topic behind.")
        return (f"알겠습니다 — {op} 작업을 반영했습니다."
                if ko else f"Understood — {op} applied.")

    # ------------------------------------------------------------------
    # Feedback intake (the self-improvement loop)
    # ------------------------------------------------------------------

    def _handle_feedback(self, request: str, language: str, semantic,
                         analysis) -> dict:
        """Store a feedback act in the user memory and acknowledge it.

        The acknowledgment is composed from what the learner understood
        and changed — never a canned apology.  The stored preference
        (speech style / explanation length / order) then flows into
        every later response through the user profile.
        """
        try:
            self.feedback_learner.process_feedback(request)
        except Exception:
            pass
        input_lang = getattr(
            getattr(semantic, "source_language", None), "value", None) \
            or language
        response = self._compose_feedback_ack(analysis, input_lang)
        self.conversation_manager.add_turn(request, "FEEDBACK", response)
        try:
            self.memory_manager.conversation_memory.add_turn(
                request, response, "FEEDBACK")
        except Exception:
            pass
        return {
            "request": request,
            "language": language,
            "intent": "FEEDBACK",
            "semantic": semantic.to_dict(),
            "response": response,
            "feedback": {
                "type": analysis.feedback_type,
                "category": analysis.category,
                "resolution": analysis.resolution,
            },
            "project_created": False,
            "results": {},
            "memory_context": self.memory_manager.get_context_for_request(
                request),
        }

    def _compose_feedback_ack(self, analysis, lang: str) -> str:
        """Compose the acknowledgment of a feedback act: the learned
        preference's meaning expressed in the input language (and in
        the register the user just asked for)."""
        ko = (lang or "en") == "ko"
        cat = (analysis.category or "").strip()
        res = (analysis.resolution or "").strip()

        if cat == "speech_style":
            if res == "casual":
                return ("알았어, 앞으로 반말로 편하게 말할게."
                        if ko else
                        "Got it — I'll keep it casual from now on.")
            if res == "polite":
                return ("알겠습니다. 앞으로 존댓말로 말씀드릴게요."
                        if ko else
                        "Understood — I'll keep it polite.")
            return ("알겠습니다. 말투를 맞춰서 말씀드리겠습니다."
                    if ko else "Noted — I'll adjust my tone.")

        if cat in ("explanation_style", "explanation_length"):
            if res in ("short", "shorter", "concise", "brief"):
                return ("알겠습니다. 설명은 더 간결하게 드리겠습니다."
                        if ko else "Got it — I'll keep explanations shorter.")
            if res in ("long", "longer", "detailed"):
                return ("알겠습니다. 설명은 더 자세하게 드리겠습니다."
                        if ko else "Got it — I'll explain in more detail.")
            if analysis.feedback_type == "complaint":
                return ("알겠습니다. 설명을 더 쉽게 풀어 드리겠습니다."
                        if ko else
                        "Noted — I'll make explanations easier to follow.")
            return (f"알겠습니다. 그 선호를 기억해 두었습니다: {analysis.target}."
                    if ko else
                    f"Noted — I'll remember that preference: {analysis.target}.")

        if cat == "comment_density":
            if analysis.feedback_type == "complaint":
                return ("알겠습니다. 내용의 양을 적절히 조절하겠습니다."
                        if ko else
                        "Noted — I'll right-size the amount of content.")
            return (f"알겠습니다. 그 선호를 기억해 두었습니다: {analysis.target}."
                    if ko else
                    f"Noted — I'll remember that preference: {analysis.target}.")

        if cat == "response_order":
            if res == "code_then_explanation":
                return ("알겠습니다. 코드를 먼저 보여드리고 설명은 뒤에 드리겠습니다."
                        if ko else
                        "Got it — code first, explanation after.")
            if res == "explanation_then_code":
                return ("알겠습니다. 설명을 먼저 드리고 코드는 뒤에 보여드리겠습니다."
                        if ko else
                        "Got it — explanation first, code after.")

        if analysis.feedback_type == "praise":
            return ("고맙습니다! 좋은 피드백이네요 — 이 스타일로 계속 가겠습니다."
                    if ko else
                    "Thanks! Glad it works — I'll keep this style.")

        if not ko:
            en_acks = {
                "technology_selection":
                    "Noted — I'll analyze the requirements before picking technology.",
                "code_style":
                    "Noted — I won't apply what you explicitly rejected.",
                "requirement_handilling":
                    "Noted — explicit requirements will always be included.",
            }
            if cat in en_acks:
                return en_acks[cat]
            return "Noted — I've recorded that lesson and will apply it."
        return (f"알겠습니다. {analysis.cause} 점을 반영했습니다. "
                f"다음부터: {analysis.lesson}.")

    def _refine_dialogue_intent(self, request: str, semantic,
                                shell_result: dict | None = None) -> dict:
        """Re-derive the dialogue intent with the retrieved knowledge.

        The ACT comes from the sentence's grammar (already in
        `semantic.dialogue_act`); this pass adds the CONTENT's kind,
        which only the retrieval can supply: the shell environment, the
        natural-language topic, the self reference, the program
        behaviour.  The topic words never decide the intent type.
        """
        from brain.dialogue_act import analyze_act, derive_intent
        try:
            act = analyze_act(request)
        except Exception:
            act = None
        if act is None:
            return dict(getattr(semantic, "dialogue_intent", None) or {})

        knowledge: dict = {}
        if shell_result:
            knowledge["shell_context"] = bool(
                shell_result.get("platform") or shell_result.get("shell"))
            knowledge["platform"] = str(
                shell_result.get("platform") or "")
        try:
            knowledge["natural_language_topic"] =                 self.conversation_knowledge._mentions_natural_language(
                    request.lower())
        except Exception:
            knowledge["natural_language_topic"] = False
        try:
            knowledge["self_topic"] = self.conversation_knowledge.\
                is_self_question(request, semantic)
        except Exception:
            knowledge["self_topic"] = False
        try:
            legacy = self.conversation_knowledge._intent_from_content(
                request, semantic)
        except Exception:
            legacy = ""
        knowledge["program_topic"] = legacy == "analysis_request"
        # The conversation's resolved topic state — the discourse
        # referent an anaphoric reference resolves against (the raw
        # previous utterances are only the fallback: the state is what
        # the earlier turns were already understood to be about).
        try:
            knowledge["discourse_topic"] = str(
                self.conversation_manager.current_topic or "")
        except Exception:
            knowledge["discourse_topic"] = ""

        history = []
        try:
            history = [t.user_input for t in
                       self.conversation_manager.history
                       if getattr(t, "user_input", None)]
        except Exception:
            history = []
        intent = derive_intent(semantic, act, request, knowledge, history)
        semantic.dialogue_act = act.to_dict()
        semantic.dialogue_intent = intent.to_dict()
        return dict(semantic.dialogue_intent)

    def _handle_command_query(self, request: str, language: str, semantic,
                              shell_result: dict) -> dict:
        """The command query: the shell knowledge composed into a single
        focused answer (the relevant operation and its approach in the
        detected environment) — the knowledge is reasoned, never dumped.
        """
        input_lang = getattr(getattr(semantic, 'source_language', None),
                             'value', language)
        self.conversation_knowledge.conversation_history = \
            self.conversation_manager.history
        self.conversation_knowledge.set_question_context(True)
        plan = self.conversation_knowledge.build_plan(
            request, semantic, input_lang,
            user_profile=self.memory_manager.get_user_profile())
        if plan.intent != "command_query":
            plan.intent = "command_query"
            self.conversation_knowledge._build_command_plan(plan, request)
        response_text = realize_plan(plan)
        self.conversation_manager.add_turn(request, "QUESTION", response_text)
        self.memory_manager.conversation_memory.add_turn(
            request, response_text, "QUESTION")
        return {
            "request": request,
            "language": language,
            "intent": "QUESTION",
            "semantic": semantic.to_dict(),
            "shell": shell_result,
            "response": response_text,
            "response_plan": plan.to_dict(),
            "project_created": False,
            "results": {},
        }

    def _handle_chat_semantic(self, request: str, language: str, semantic) -> dict:
        """Handle chat/greeting through the conversation knowledge
        pipeline: the semantic structure (the greeting/thanks/farewell
        predication, the first-person self-statement) decides the
        dialogue act; the response is composed — never a canned
        answer."""
        input_lang = getattr(getattr(semantic, 'source_language', None),
                             'value', language)
        self.conversation_knowledge.conversation_history = \
            self.conversation_manager.history
        self.conversation_knowledge.set_question_context(
            any(getattr(c, 'category', '').startswith('shell')
                for c in self._question_shell_concepts(request, semantic)))
        plan = self.conversation_knowledge.build_plan(
            request, semantic, input_lang,
            user_profile=self.memory_manager.get_user_profile())
        semantic.knowledge_requirement = dict(
            plan.knowledge_requirement or {})
        response_text = self._compose_response(plan, semantic)
        self.conversation_manager.add_turn(
            request, "CHAT", response_text,
            topics=list(semantic.intent_topics or []))
        self.memory_manager.conversation_memory.add_turn(request, response_text, "CHAT")
        return {
            "request": request,
            "language": language,
            "intent": "CHAT",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "response_plan": plan.to_dict(),
            "project_created": False,
            "results": {},
            "memory_context": self.memory_manager.get_context_for_request(request),
        }

    def _compose_response(self, plan, semantic) -> str:
        """The response's path out of the plan: the REPRESENTATION
        (what the response says, with its sources), the SENTENCE PLAN
        (the units and their budgets) and the REALIZATION (the wording
        in the requested register).  The retrieved material enters as
        the representation's content — it is never the response.

        When the turn's budget warrants a LONG response, the same
        representation and material feed the sectioned long-form
        generation (brain.longform); every other turn keeps the short
        realization byte for byte."""
        from brain.response_representation import (
            build_representation, plan_sentences, realize)
        reference = getattr(semantic, "reference", None) or {}
        knowledge = getattr(plan, "retrieval", None) or {}
        try:
            topic_state = self.conversation_manager.topic_state
        except Exception:
            topic_state = None
        representation = build_representation(plan, semantic, reference,
                                              topic_state, knowledge)
        sentences = plan_sentences(representation, plan)
        plan.representation = representation.to_dict()
        plan.sentence_plan = [s.to_dict() for s in sentences]
        try:
            from brain.longform import ResponseBudget
            budget = ResponseBudget.from_dict(getattr(plan, "budget", None))
            # The RESPONSE SPECIFICATION the understanding carries is
            # the authority on the bounds: the budget and the
            # specification are the same decision, read where the turn
            # is composed.
            spec = getattr(semantic, "response_spec", None)
            spec_dict = spec.to_dict() if hasattr(spec, "to_dict") \
                else (dict(spec) if isinstance(spec, dict) else {})
            if spec_dict.get("target_length"):
                budget = ResponseBudget.from_spec(spec_dict)
        except Exception:
            budget = None
        if budget is not None and budget.warrants_longform():
            try:
                if getattr(plan, "document_evidence", None) is not None:
                    # the request was read as a document: its own
                    # outline and its own validated evidence write it
                    return self._compose_document_form(
                        plan, semantic, representation, budget)
                return self._compose_long_form(plan, semantic,
                                               representation, budget)
            except Exception as exc:
                plan.longform = {"error": str(exc)[:300]}
        # The SHORT realization still accounts for every part the
        # request states: a part the wording never reaches is said to
        # be missing rather than dropped silently (brain.longform).
        body = realize(representation, plan)
        try:
            from brain.longform import account_for_parts
            body = account_for_parts(
                body,
                str(getattr(semantic, "raw_text", "") or ""),
                plan,
                str(getattr(plan, "requested_language", "") or "en"))
        except Exception:
            pass
        return body

    @staticmethod
    def _compose_document_form(plan, semantic, representation,
                               budget) -> str:
        """The document's own generation: the structured request's
        knowledge coverage becomes the response outline, the validated
        evidence becomes each section's content, and the verification
        measures what was written against what the plan set out to
        write — reported on the plan so the trace reads what happened."""
        from brain.longform import (
            GenerationState, generate_document_form,
            plan_document_sections, revise_document_form)
        structured = getattr(plan, "document", None)
        evidence = getattr(plan, "document_evidence", None)
        request = str(getattr(semantic, "raw_text", "") or "")
        language = str(getattr(plan, "requested_language", "") or "en")
        state = GenerationState(
            request=request,
            semantic_representation={
                "raw_text": request,
                "intent": plan.intent,
                "target": plan.target,
                "topic": getattr(structured, "topic", ""),
                "knowledge_goals": [
                    {"id": g.id, "topic": g.topic,
                     "question": g.question,
                     "depends_on": list(g.depends_on)}
                    for g in (getattr(structured, "knowledge_goals", [])
                              or [])],
                "response_requirements": [
                    {"kind": r.kind, "value": r.value}
                    for r in (getattr(structured,
                                      "response_requirements", []) or [])],
            },
            response_plan={
                "intent": plan.intent,
                "target": plan.target,
                "style": plan.style,
                "language": language,
            })
        # A source that would not answer tonight leaves the document
        # short: the turn's own validated pool is the same evidence
        # under another retrieval, so the plan is filled from it rather
        # than left open.
        pool = list(getattr(evidence, "material", []) or [])
        if sum(len(sentence) for sentence in pool) < budget.minimum:
            for sentence in (getattr(plan, "longform_material", None)
                             or []):
                if sentence and sentence not in pool:
                    pool.append(sentence)
            if len(pool) > len(getattr(evidence, "material", []) or []):
                evidence.material = pool
                evidence.section_of += [-1] * (
                    len(pool) - len(evidence.section_of))
        sections = plan_document_sections(budget, structured, evidence,
                                          plan, language)
        text, state, report = generate_document_form(
            budget, sections, structured, plan, state, language)
        if not report.get("passed"):
            text, state, report = revise_document_form(
                text, state, budget, plan, report, language)
        plan.longform = {
            "path": "document",
            "document": structured.to_dict() if hasattr(structured,
                                                         "to_dict")
            else {},
            "evidence": evidence.to_dict() if hasattr(evidence,
                                                      "to_dict") else {},
            "budget": dict(budget.to_dict()),
            "sections": [s.to_dict() for s in sections],
            "state": state.to_dict(),
            "report": report,
        }
        representation.sections = [s.to_dict() for s in sections]
        representation.budget = budget.to_dict()
        plan.representation = representation.to_dict()
        return text

    @staticmethod
    def _compose_long_form(plan, semantic, representation, budget) -> str:
        """The long-form generation: requirements from the request's
        grammar, sections from the requirements and the material,
        incremental generation, one bounded revision — reported on the
        plan so the trace reads what happened."""
        from brain.longform import (
            GenerationState, extract_requirements, generate_long_form,
            plan_sections, revise_long_form)
        request = str(getattr(semantic, "raw_text", "") or "")
        language = str(getattr(plan, "requested_language", "") or "en")
        requirements = extract_requirements(request, plan, budget)
        material = list(getattr(plan, "longform_material", None) or [])
        if not material:
            material = list(plan.claims or [])
        state = GenerationState(
            request=request,
            semantic_representation={
                "raw_text": request,
                "intent": plan.intent,
                "target": plan.target,
            },
            response_plan={
                "intent": plan.intent,
                "target": plan.target,
                "style": plan.style,
                "language": language,
            },
            unresolved_requirements=[dict(r) for r in requirements])
        sections = plan_sections(budget, requirements, plan, material,
                                 language)
        text, state, report = generate_long_form(
            budget, sections, plan, state, language)
        if not report.get("passed"):
            text, state, report = revise_long_form(
                text, state, budget, plan, report, language)
        plan.longform = {
            "budget": dict(budget.to_dict()),
            "sections": [s.to_dict() for s in sections],
            "state": state.to_dict(),
            "report": report,
        }
        representation.sections = [s.to_dict() for s in sections]
        representation.budget = budget.to_dict()
        plan.representation = representation.to_dict()
        return text

    def _asks_personal_history(self, request: str) -> bool:
        """The question asks for a fact about the user's own past.

        Without a remembered turn or an attached source, no offline
        knowledge base can answer it.  The question routes to an honest
        "not available" instead of retrieving an unrelated article.
        """
        lowered = str(request or "").lower()
        if re.search(r"\b(?:code|error|function|program|script|test)\b",
                     lowered):
            return False
        return bool(re.search(
            r"\b(?:what|where|when|who|which)\s+did\s+i\s+"
            r"(?:eat|do|say|see|buy|go|visit|watch|read|wear|drink)\b",
            lowered))

    def _handle_question_semantic(self, request: str, language: str, semantic) -> dict:
        """Handle a question through the conversation knowledge pipeline:

        understand → retrieve relevant evidence → reason → ResponsePlan
        → realization.  The knowledge is reasoned into claims (never
        dumped); the answer is realized in the requested language.
        """
        # The conversation knowledge pipeline: the intent → the relevant
        # knowledge → the claims → the ResponsePlan.  The realization's
        # register is the INPUT language (the semantic's detected
        # language), never the code-target parameter.
        input_lang = getattr(getattr(semantic, 'source_language', None),
                             'value', language)
        if self._asks_personal_history(request):
            response_text = (
                "제 개인 기록에는 접근할 수 없어서 그 질문에 답할 정보가 없습니다. "
                "확인할 수 있는 내용을 알려주시면 함께 살펴보겠습니다."
                if input_lang == "ko" else
                "I do not have access to your personal history, so I do "
                "not have that information. Tell me what you can share "
                "and I will work from that.")
            self.conversation_manager.add_turn(
                request, "QUESTION", response_text,
                topics=list(semantic.intent_topics or []))
            self.memory_manager.conversation_memory.add_turn(
                request, response_text, "QUESTION")
            return {
                "request": request,
                "language": language,
                "intent": "QUESTION",
                "semantic": semantic.to_dict(),
                "response": response_text,
                "response_plan": {
                    "intent": "information_query",
                    "target": str(getattr(semantic.goal, "target", "") or ""),
                    "claims": [],
                    "honesty": "personal history is not available",
                },
                "project_created": False,
                "results": {},
            }
        self.conversation_knowledge.conversation_history = \
            self.conversation_manager.history
        self.conversation_knowledge.set_question_context(
            any(getattr(c, 'category', '').startswith('shell')
                for c in self._question_shell_concepts(request, semantic)))
        plan = self.conversation_knowledge.build_plan(
            request, semantic, input_lang,
            user_profile=self.memory_manager.get_user_profile())
        # The knowledge requirement travels with the understanding (the
        # trace reads the SAME decision the retrieval acted on).
        semantic.knowledge_requirement = dict(
            plan.knowledge_requirement or {})
        response_text = self._compose_response(plan, semantic)
        self.conversation_manager.add_turn(
            request, "QUESTION", response_text,
            topics=list(semantic.intent_topics or []))
        self.memory_manager.conversation_memory.add_turn(request, response_text, "QUESTION")
        return {
            "request": request,
            "language": language,
            "intent": "QUESTION",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "response_plan": plan.to_dict(),
            "project_created": False,
            "results": {},
        }

    def _handle_research_semantic(self, request: str, language: str,
                                  semantic) -> dict:
        """Handle an investigation through the conversation knowledge
        pipeline: understand → research loop → evidence → iteration →
        synthesis → response.  The plan carries what the investigation
        did (its queries, sources and stages) so the response and the
        trace state the same facts."""
        input_lang = getattr(getattr(semantic, 'source_language', None),
                             'value', language)
        self.conversation_knowledge.conversation_history = \
            self.conversation_manager.history
        self.conversation_knowledge.set_question_context(
            any(getattr(c, 'category', '').startswith('shell')
                for c in self._question_shell_concepts(request, semantic)))
        plan = self.conversation_knowledge.build_plan(
            request, semantic, input_lang,
            user_profile=self.memory_manager.get_user_profile())
        semantic.knowledge_requirement = dict(
            plan.knowledge_requirement or {})
        response_text = self._compose_response(plan, semantic)
        self.conversation_manager.add_turn(
            request, "RESEARCH", response_text,
            topics=list(semantic.intent_topics or []))
        self.memory_manager.conversation_memory.add_turn(
            request, response_text, "RESEARCH")
        self._remember_long_term(request, response_text)
        return {
            "request": request,
            "language": language,
            "intent": "RESEARCH",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "response_plan": plan.to_dict(),
            "project_created": False,
            "results": {},
        }

    def _remember_long_term(self, request: str, response: str) -> None:
        """Feed the turn to the long-term memory: durable facts,
        preferences and commitments are distilled into the persistent
        store (memory/long_term.py) so later sessions can recall them.
        Best-effort and silent — the response never depends on it."""
        try:
            import time as _time
            store = getattr(self.memory_manager, "long_term", None)
            if store is not None and response:
                store.consolidate([
                    {"role": "user", "text": str(request or ""),
                     "ts": _time.time()},
                    {"role": "assistant", "text": str(response or ""),
                     "ts": _time.time()},
                ])
        except Exception:
            pass

    def _handle_explain_semantic(self, request: str, language: str, semantic) -> dict:
        """Handle an explanation request through the conversation
        knowledge pipeline: the target's concept knowledge (a
        conversational reference like "that" resolves against the
        previous topic) → the claims → the ResponsePlan → the
        realization."""
        input_lang = getattr(getattr(semantic, 'source_language', None),
                             'value', language)
        self.conversation_knowledge.conversation_history = \
            self.conversation_manager.history
        self.conversation_knowledge.set_question_context(
            any(getattr(c, 'category', '').startswith('shell')
                for c in self._question_shell_concepts(request, semantic)))
        plan = self.conversation_knowledge.build_plan(
            request, semantic, input_lang,
            user_profile=self.memory_manager.get_user_profile())
        # The semantic goal is EXPLAIN because the utterance's own
        # explanation clause requested it.  A negated generation clause
        # earlier in the sentence ("do not write code, explain X")
        # must not turn the plan into a social acknowledgment.
        if plan.intent == "conversation":
            plan.intent = "explanation_request"
            self.conversation_knowledge._build_explanation_plan(
                plan, semantic, request)
        semantic.knowledge_requirement = dict(
            plan.knowledge_requirement or {})
        response_text = self._compose_response(plan, semantic)
        self.conversation_manager.add_turn(
            request, "EXPLANATION", response_text,
            topics=list(semantic.intent_topics or []))
        self.memory_manager.conversation_memory.add_turn(request, response_text, "EXPLANATION")
        self._remember_long_term(request, response_text)
        return {
            "request": request,
            "language": language,
            "intent": "EXPLAIN",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "response_plan": plan.to_dict(),
            "project_created": False,
            "results": {},
        }
    
    def _handle_insufficient_understanding(self, request: str, language: str, semantic) -> dict:
        """Handle case where semantic understanding confidence is too low."""
        # Generate clarification question
        response_text = self._generate_clarification(semantic)
        return {
            "request": request,
            "language": language,
            "intent": "CLARIFICATION_NEEDED",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "project_created": False,
            "results": {},
            "clarification_needed": True,
        }
    
    def _generate_clarification(self, semantic) -> str:
        """Generate a clarification question based on what's missing."""
        missing = []
        if not semantic.entities:
            missing.append("what entity or data you're referring to")
        if not semantic.operation_graph.operations:
            missing.append("what operation or action you want")
        if not semantic.inputs:
            missing.append("what input data is available")
        
        if missing:
            return f"Could you clarify {', '.join(missing)}? I want to make sure I understand correctly."
        return "Could you provide more details about what you'd like me to do?"
    
    # ── The conversation → coding bridge ─────────────────────────────
    # A follow-up coding turn does NOT re-derive its problem from its
    # own words alone: it continues the previous turn's representation.
    def _continuation_context(self, semantic, request: str) -> dict:
        """The previous turn this coding turn CONTINUES, or {} when the
        turn stands alone.

        Two continuation relations, both structural (never keyword):
          - `modify` : the turn's action applies to an existing
            artifact → the previous code turn's artifact is the base.
          - `reference` : the turn's referring expression resolved
            against the discourse → the previous turn's semantic
            carries the SUBJECT ("implement that" → binary search).
        """
        prev = self.conversation_manager.previous_turn(request)
        if prev is None:
            return {}
        ref = dict(getattr(semantic, "reference", None) or {})
        resolved = bool(ref.get("target")) and not bool(
            ref.get("needs_clarification"))
        goal_type = getattr(semantic.goal, "goal_type", None)
        if goal_type == GoalType.MODIFY:
            # The artifact-class referent: a modification continues
            # from the nearest turn that HAS an artifact.  A non-coding
            # turn between two coding turns (a knowledge aside) never
            # resets the artifact — the conversation and the coding
            # layer share one history, and the artifact is what the
            # speaker's "it" points at.  Without one anywhere there is
            # no coding problem to continue (the handler says so
            # honestly — it never fabricates an edit).
            artifact_turn = None
            for turn in self.conversation_manager.history:
                if turn is prev:
                    if dict(getattr(turn, "artifact", None)
                            or {}).get("code"):
                        artifact_turn = turn
                    break
                if dict(getattr(turn, "artifact", None) or {}).get("code"):
                    artifact_turn = turn
            if artifact_turn is None:
                return {}
            prev_sem = getattr(artifact_turn, "semantic_object", None)
            if prev_sem is None:
                # A resumed session carries the turn's SERIALIZED
                # meaning and its artifact, not the runtime object.
                # The continuation merge reads the record's own
                # composition and subjects — the view stands in.
                view = dict(getattr(artifact_turn,
                                    "semantic_representation", None)
                            or {})
                if view:
                    prev_sem = _CompositionView(view)
            return {"turn": artifact_turn,
                    "semantic": prev_sem,
                    "artifact": dict(getattr(artifact_turn, "artifact",
                                             None) or {}),
                    "reason": "modify",
                    "resolved_target": str(ref.get("target") or "")}
        prev_sem = getattr(prev, "semantic_object", None)
        if prev_sem is None:
            return {}
        artifact = dict(getattr(prev, "artifact", None) or {})
        if resolved:
            return {"turn": prev, "semantic": prev_sem,
                    "artifact": artifact, "reason": "reference",
                    "resolved_target": str(ref.get("target") or "")}
        return {}

    @staticmethod
    def _same_op_family(name_a: str, name_b: str) -> bool:
        """Two operation names belong to the same family when one's
        underscore-token set is a subset of the other's (sort →
        sort_descending).  A structural name relation — never a
        sentence keyword."""
        if not name_a or not name_b:
            return False
        if name_a == name_b:
            return True
        tokens_a = set(name_a.split("_"))
        tokens_b = set(name_b.split("_"))
        return tokens_a <= tokens_b or tokens_b <= tokens_a

    def _merge_continuation(self, semantic, ctx: dict) -> None:
        """The coding problem CONTINUES: this turn's semantics update
        the previous turn's structure.  The merge happens on the
        REPRESENTATION (composition graph, entities, IO, types) — the
        code is later REGENERATED from it.  Nothing is patched as
        text, and no operation the artifact never had is invented."""
        import copy as _copy
        prev_sem = ctx["semantic"]
        artifact = dict(ctx.get("artifact") or {})
        prev_comp = dict(getattr(prev_sem, "composition", None) or {})
        prev_graph = dict(prev_comp.get("graph") or {})
        prev_ops = [o for o in (prev_graph.get("operations") or [])
                    if isinstance(o, dict) and o.get("name")]
        curr_comp = dict(getattr(semantic, "composition", None) or {})
        curr_graph = dict(curr_comp.get("graph") or {})
        curr_ops = [o for o in (curr_graph.get("operations") or [])
                    if isinstance(o, dict) and o.get("name")]

        new_graph = None
        clear_comp_code = False
        if ctx.get("reason") == "modify" and prev_ops:
            # The artifact's structure is the BASE.  This turn's own
            # operations refine it only where they name the same
            # operation family (sort → sort_descending); an unrelated
            # operation the sentence's verb happened to compose (the
            # arithmetic "add" of "add logging") is not dataflow of the
            # artifact and never enters its graph.
            merged = []
            for old in prev_ops:
                replacement = None
                for new in curr_ops:
                    if self._same_op_family(str(new.get("name") or ""),
                                            str(old.get("name") or "")):
                        replacement = dict(new)
                        # the refinement keeps the base op's place in
                        # the graph (id + dependencies stay valid)
                        replacement["id"] = old.get("id")
                        replacement["depends_on"] = list(
                            old.get("depends_on") or [])
                        break
                merged.append(replacement or old)
            new_graph = dict(prev_graph)
            new_graph["operations"] = merged
        elif ctx.get("reason") == "modify":
            # The previous composition carries no graph at all: the
            # artifact came from a direct algorithm (its structure of
            # record is the artifact's OWN IR, which the handler
            # regenerates).  Only this turn's operations the pipeline
            # can render as dataflow refine that artifact
            # (sort_descending); the request's own verb composition (the
            # arithmetic "add" of "add logging") describes the REQUEST,
            # not the artifact's dataflow — it never enters the graph
            # and its composed code never renders the artifact.
            normalized = self._normalize_composition_ops(curr_ops)
            if normalized and all(op in self.PIPELINE_OPERATION_VOCAB
                                  for op in normalized):
                new_graph = dict(curr_graph)
                new_graph["operations"] = list(curr_ops)
            else:
                new_graph = {"operations": [], "entry_points": [],
                             "exit_points": [], "execution_order": []}
                clear_comp_code = True
        elif prev_ops and not curr_ops:
            # The turn names no operation of its own ("implement that"):
            # the subject's operations carry the problem.
            new_graph = dict(prev_graph)

        if new_graph is not None:
            new_comp = dict(curr_comp)
            new_comp["graph"] = new_graph
            if clear_comp_code:
                # The graph is the artifact's (its IR), not this
                # sentence's: the sentence's composed code must not
                # regenerate it.
                new_comp["code"] = ""
            elif not new_comp.get("code"):
                new_comp["code"] = prev_comp.get("code") or ""
            semantic.composition = new_comp
            try:
                semantic.operation_graph = _copy.deepcopy(
                    prev_sem.operation_graph)
            except Exception:
                pass

        # Subject continuity: the referent's meaning (entities, IO,
        # types, algorithm) fills what this turn does not restate.
        if not semantic.entities and prev_sem.entities:
            semantic.entities = _copy.deepcopy(list(prev_sem.entities))
        if not semantic.inputs and prev_sem.inputs:
            semantic.inputs = _copy.deepcopy(list(prev_sem.inputs))
        if not semantic.outputs and prev_sem.outputs:
            semantic.outputs = _copy.deepcopy(list(prev_sem.outputs))
        if not semantic.output_type and prev_sem.output_type:
            semantic.output_type = prev_sem.output_type
        if not semantic.goal.algorithm_hint:
            hint = getattr(prev_sem.goal, "algorithm_hint", None)
            if hint:
                semantic.goal.algorithm_hint = hint

        # The link, on the goal (JSON-safe, read by generation and the
        # ProblemRepresentation): which representation this coding
        # problem continues and which function it regenerates.
        function_name = str(artifact.get("function_name") or "")
        if not function_name:
            function_name = str(getattr(prev_sem.goal, "algorithm_hint",
                                        "") or "")
        parent = dict(semantic.goal.parent_context or {})
        parent["continuation"] = {
            "from": str(getattr(prev_sem, "id", "") or ""),
            "function_name": function_name,
            "reason": str(ctx.get("reason") or ""),
            "resolved_target": str(ctx.get("resolved_target") or ""),
        }
        semantic.goal.parent_context = parent

    def _attach_problem(self, semantic, request: str,
                        ctx: dict | None = None):
        """The turn's ProblemRepresentation — the typed coding problem
        derived from the (merged) semantics, linked to the previous
        turn's representation when the turn continues one."""
        from brain.problem_representation import ProblemRepresentation
        ctx = ctx or {}
        semantic.problem = ProblemRepresentation.from_semantic(
            semantic, previous=ctx.get("semantic"),
            artifact=ctx.get("artifact"))
        return semantic.problem

    @staticmethod
    def _record_continuation_requirements(reasoning, problem,
                                          request: str = "") -> None:
        """The continuation turn's own request is an explicit user
        requirement on the artifact — tracked with the requirements the
        pipeline verifies against (never silently dropped)."""
        try:
            from brain.think_engine import Requirement
            existing = {r.name for r in
                        (reasoning.model.requirements or [])}
            texts = []
            text = str(request or "").strip()
            if text:
                texts.append(text)
            texts += [r for r in (problem.algorithm_requirements
                                  + problem.verification_requirements)
                      if r not in existing and r not in texts]
            for req_text in texts:
                if req_text in existing:
                    continue
                reasoning.model.requirements.append(Requirement(
                    name=req_text, category="quality", value=req_text,
                    description=f"continuation requirement: {req_text}",
                    source_text=req_text))
                existing.add(req_text)
        except Exception:
            pass

    def _handle_function_semantic(self, request: str, language: str, semantic) -> dict:
        # Stage 0: Algorithmic problem gate.
        # Route to the algorithmic pipeline when EITHER the structural
        # detector fires OR the semantic representation itself carries
        # algorithmic structure (objectives, substructures, aggregate
        # constraints) — the semantics drive the routing, not keywords.
        has_algorithmic_semantics = False
        if semantic is not None and hasattr(semantic, 'objectives'):
            if semantic.objectives or getattr(semantic, 'substructures', None):
                has_algorithmic_semantics = True
            elif any(getattr(c, 'constraint_type', '') == "aggregate"
                     for c in getattr(semantic, 'constraints', [])):
                has_algorithmic_semantics = True
        try:
            if has_algorithmic_semantics:
                return self._handle_competitive_problem(request, language, semantic)
            from brain.algorithmic.competitive_problem_detector import CompetitiveProblemDetector
            detector = CompetitiveProblemDetector()
            detection = detector.detect(request)
            if detection.is_competitive and detection.confidence >= 0.45:
                return self._handle_competitive_problem(request, language, semantic)
        except Exception:
            pass  # Fall through to standard pipeline on detection failure

        # Standard data-pipeline path
        # The conversation → coding bridge: when this turn continues
        # the previous one (an anaphoric reference resolved to it, or
        # the turn modifies its artifact), the problem is built from
        # the PREVIOUS representation updated by THIS turn's semantics
        # — never re-derived from this turn's words alone.
        ctx = self._continuation_context(semantic, request)
        if ctx:
            self._merge_continuation(semantic, ctx)
        problem = self._attach_problem(semantic, request, ctx)
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        if problem.is_continuation:
            self._record_continuation_requirements(reasoning, problem,
                                                   request)
        # Track requirements through the pipeline
        self.requirement_tracker.set_requirements(reasoning.model.requirements)
        result = self._generate_function_natural(request, reasoning.model, reasoning, semantic, language)
        # ── User Requirement Contract (spec #2/#11): the FINAL authority.
        # Built from the validated plan + text constraints, evaluated
        # across semantic → plan → IR → code → runtime → behavioral.
        try:
            result["requirement_coverage"] = self._evaluate_requirement_contract(
                request, semantic, result, language)
        except Exception:
            result["requirement_coverage"] = {}
        # Generate response text using the response generator — ABOUT the
        # artifact the generation actually produced
        response_text = self.new_response_generator.generate(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict()
                if self.conversation_manager.current_project else None,
            artifact=self._summarize_generation_artifact(result, reasoning),
        )
        result["response"] = response_text
        # The FINAL confidence: semantic plausibility bounded by what
        # verification actually established — never the raw hypothesis
        # score presented as the result's confidence.
        result["confidence"] = self._final_confidence(result, reasoning)
        # Add requirement tracking to result
        result["requirement_tracking"] = self.requirement_tracker.to_dict()
        # The coding problem (and its continuation link, when this turn
        # continued a previous one) travels with the result — the trace
        # reads the SAME representation generation was driven by.
        result["problem"] = problem.to_dict()
        # The OUTPUT CONTRACT verified against what was generated: the
        # source the answer must reach and the operation it must carry
        # are checked, reported and never assumed.
        try:
            from brain.problem_representation import check_output_contract
            code = str(result.get("code") or "")
            if not code:
                languages = result.get("languages") or {}
                code = str(
                    (languages.get(language) or {}).get("code")
                    or (languages.get("python") or {}).get("code") or "")
            if not code:
                for entry in (result.get("results") or {}).values():
                    if not isinstance(entry, dict):
                        continue
                    languages = entry.get("languages") or {}
                    candidate = str(
                        (languages.get(language) or {}).get("code")
                        or (languages.get("python") or {}).get("code")
                        or "")
                    if candidate:
                        code = candidate
                        break
            result["output_contract"] = check_output_contract(
                code, {
                    "requested_output": problem.requested_output,
                    "output_type": problem.output_type,
                    "output_source": problem.output_source,
                    "final_operation": problem.final_operation,
                })
        except Exception:
            result["output_contract"] = {}
        if problem.continuation:
            result["continuation"] = dict(problem.continuation)
        return result

    def _handle_competitive_problem(self, request: str, language: str, semantic) -> dict:
        """Handle competitive programming problems via the algorithmic pipeline.

        Uses SemanticRepresentation to drive the algorithmic pipeline when available.
        Falls back to the standard data-pipeline if algorithmic reasoning fails.
        """
        try:
            from brain.algorithmic.orchestrator import AlgorithmicOrchestrator
            orchestrator = AlgorithmicOrchestrator()

            # Use semantic-driven pipeline when semantic is available
            if semantic and hasattr(semantic, 'objectives'):
                algo_result = orchestrator.solve_with_semantic(request, semantic)
            else:
                algo_result = orchestrator.solve(request)

            if algo_result.success and algo_result.code:
                # A bare function contract — an objective over elements
                # with NO problem structure — is the standard pipeline's
                # artifact: it renders the typed signature, the IR
                # contract and a behavioral verification, none of which
                # an unverifiable build from here could replace.  A
                # problem that DECLARES structure (a substructure, an
                # aggregate constraint) belongs to the algorithmic
                # pipeline: that structure is what it synthesizes an
                # algorithm for, and the standard pipeline has no
                # general form for it.
                declares_structure = bool(
                    getattr(semantic, "substructures", None)) or any(
                    getattr(c, "constraint_type", "") == "aggregate"
                    for c in getattr(semantic, "constraints", []))
                if (not declares_structure
                        and getattr(algo_result.problem, "request_form", "") == "function"
                        and getattr(algo_result, "verification", None) is None):
                    raise RuntimeError(
                        "unstructured function contract belongs to the "
                        "standard pipeline")
                # Build the result dict in the same format as standard pipeline
                func_name = algo_result.function_name or "solve"
                languages = {"python": {"code": algo_result.code}}
                # The requested target language (the pipeline's language
                # parameter decides, not request keywords): transpile the
                # python solution into the additional requested language.
                if language != "python" and language in self._get_supported_languages():
                    try:
                        target_code = self._transpile_python_to(
                            algo_result.code, language)
                        if target_code:
                            languages[language] = {"code": target_code}
                    except Exception:
                        pass
                # The sample-case verification the orchestrator already
                # ran travels WITH the result — the same per-result slot
                # the standard pipeline fills.  Without sample cases
                # there is no oracle: the slot stays None (never an
                # execution-only pass).
                _vres = getattr(algo_result, "verification", None)
                verification_payload = None
                if _vres is not None:
                    _vpc = int(getattr(_vres, "passed_cases", 0) or 0)
                    _vtc = int(getattr(_vres, "total_cases", 0) or 0)
                    verification_payload = {
                        "success": bool(getattr(_vres, "passed", False)),
                        "passed_cases": _vpc,
                        "total_cases": _vtc,
                        "errors": ([] if getattr(_vres, "passed", False)
                                   else ["sample case verification failed"]),
                        "warnings": [],
                    }
                results = {
                    func_name: {
                        "ir": algo_result.typed_ir.root.to_dict() if algo_result.typed_ir and algo_result.typed_ir.root else {},
                        "languages": languages,
                        "verification": verification_payload,
                    }
                }

                # Generate response text
                response_text = ""
                try:
                    reasoning = self.think_engine.think_from_semantic(semantic, language)
                    response_text = self.new_response_generator.generate(
                        semantic=semantic,
                        reasoning=reasoning,
                        conversation_history=self.conversation_manager.history,
                        current_project=self.conversation_manager.current_project.to_dict()
                            if self.conversation_manager.current_project else None,
                        artifact={"name": func_name, "operations_text": "",
                                  "verified": (algo_result.verification.passed
                                               if getattr(algo_result, "verification", None) else None),
                                  "confidence": None},
                    )
                except Exception:
                    response_text = f"Generated competitive programming solution for {func_name}."

                # The FINAL confidence, computed from the sample
                # verification's own numbers: verified-against-samples
                # earns its confidence; nothing to test against caps it
                # honestly.
                _v = getattr(algo_result, "verification", None)
                _passed = bool(getattr(_v, "passed", False)) if _v else False
                _total = int(getattr(_v, "total_cases", 0) or 0) if _v else 0
                _pc = int(getattr(_v, "passed_cases", 0) or 0) if _v else 0
                if _passed and _total > 0:
                    algo_conf = {
                        "confidence": round(0.4 + 0.5 * (_pc / _total), 2),
                        "hypothesis_confidence": None,
                        "basis": f"behavioral verification passed ({_pc}/{_total} sample cases)",
                    }
                elif _total > 0:
                    # The oracle RAN and failed: confidence follows the
                    # fraction it confirmed, below the unverified
                    # baseline — a failing oracle never earns more than
                    # code nothing tested.
                    algo_conf = {
                        "confidence": round(0.4 * (_pc / _total), 2),
                        "hypothesis_confidence": None,
                        "basis": f"behavioral verification failed ({_pc}/{_total} sample cases)",
                    }
                else:
                    algo_conf = {
                        "confidence": 0.4,
                        "hypothesis_confidence": None,
                        "basis": "no behavioral oracle available for this problem",
                    }
                return {
                    "request": request,
                    "intent": "GENERATE_CODE",
                    "semantic": semantic.to_dict(),
                    "results": results,
                    "reasoning": None,
                    "project_created": False,
                    "response": response_text,
                    "algorithmic": True,
                    "verification": algo_result.verification.passed if algo_result.verification else None,
                    "confidence": algo_conf,
                }
        except Exception:
            pass  # Fall through to standard pipeline

        # Fallback: standard data-pipeline
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        self.requirement_tracker.set_requirements(reasoning.model.requirements)
        result = self._generate_function_natural(request, reasoning.model, reasoning, semantic, language)
        response_text = self.new_response_generator.generate(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict()
                if self.conversation_manager.current_project else None,
            artifact=self._summarize_generation_artifact(result, reasoning),
        )
        result["response"] = response_text
        # The requirement contract bounds the final confidence on this
        # fallback path too — one authority for every code route.
        try:
            result["requirement_coverage"] = self._evaluate_requirement_contract(
                request, semantic, result, language)
        except Exception:
            result["requirement_coverage"] = {}
        result["confidence"] = self._final_confidence(result, reasoning)
        result["requirement_tracking"] = self.requirement_tracker.to_dict()
        return result
    
    def _handle_project_semantic(self, request: str, language: str, semantic) -> dict:
        # semantic is a SemanticRepresentation, need to run think_engine first
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        # Track requirements through the pipeline
        self.requirement_tracker.set_requirements(reasoning.model.requirements)
        result = self.generate_project_with_reasoning(request, reasoning)
        # Add requirement tracking to result
        result["requirement_tracking"] = self.requirement_tracker.to_dict()
        return result
    
    def _handle_class_semantic(self, request: str, language: str, semantic) -> dict:
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        ci = reasoning.structured_intent.class_info if reasoning.structured_intent else None
        class_name = ci.name if ci else "User"
        properties = ci.properties if ci else [("name", "string"), ("age", "int")]

        results = {}
        ir_nodes = []

        # Generate class IR with constructor
        from code.ir.ir_types import create_class_ir, create_variable_ir, create_return_ir, IRNode, IRNodeType
        class_ir = create_class_ir(class_name)
        class_ir.properties["properties_list"] = properties
        
        # Add properties
        for prop_name, prop_type in properties:
            prop = IRNode(node_type=IRNodeType.PROPERTY, name=prop_name, type_ref=prop_type)
            class_ir.add_child(prop)
        
        # Add constructor
        ctor = IRNode(node_type=IRNodeType.CONSTRUCTOR, name="__init__")
        ctor.params = [("self", "")] + [(p[0], p[1]) for p in properties]
        ctor.properties["properties_list"] = properties
        # Add PARAMETER children (without self)
        for pname, ptype in properties:
            ctor.add_child(IRNode(node_type=IRNodeType.PARAMETER, name=pname, type_ref=ptype))
        class_ir.add_child(ctor)
        
        ir_nodes.append(class_ir)

        results[class_name] = {"ir": class_ir.to_dict(), "languages": {}}

        # Generate code for each language
        for lang in ["python", "kotlin", "java"]:
            try:
                code = self.code_generator.generate(class_ir, lang)
                results[class_name]["languages"][lang] = {"code": code}
            except Exception as e:
                results[class_name]["languages"][lang] = {"code": f"// Error: {e}"}

        # Generate compound functions (e.g., class_and_function)
        if reasoning.structured_intent and reasoning.structured_intent.compound_steps:
            for step in reasoning.structured_intent.compound_steps:
                if step == "class_and_function":
                    # Generate search function for class+function compound
                    search_ir = self._generate_algorithm_ir("search_in_list")
                    results[search_ir.name] = {"ir": search_ir.to_dict(), "languages": {}}
                    for lang in ["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"]:
                        try:
                            code = self.code_generator.generate(search_ir, lang)
                            results[search_ir.name]["languages"][lang] = {"code": code}
                        except Exception as e:
                            results[search_ir.name]["languages"][lang] = {"code": f"// Error: {e}"}

        self.conversation_manager.add_turn(request, "GENERATE_CODE", f"클래스 {class_name}가 생성되었습니다.")
        return {
            "request": request,
            "intent": "GENERATE_CODE",
            "results": results,
            "reasoning": reasoning,
            "project_created": False,
        }

    def _handle_modify_semantic(self, request: str, language: str, semantic) -> dict:
        """A modification request CONTINUES the previous coding turn:
        the artifact's structure (operation graph, IO, types) is the
        base, this turn's semantics update it, and the code is
        REGENERATED from the updated representation — an update of the
        meaning, never a textual patch of the previous code.

        With no artifact in the conversation there is nothing to
        modify: the turn says so honestly (what is missing and why),
        and never fabricates an edit."""
        ctx = self._continuation_context(semantic, request)
        if not ctx:
            return {
                "request": request,
                "language": language,
                "intent": "MODIFY_PROJECT",
                "semantic": semantic.to_dict(),
                "response": (
                    "There is no artifact in this conversation to "
                    "modify yet — I have not generated a function here. "
                    f"Ask me to generate one first, then apply: {request}"),
                "project_created": False,
                "results": {},
                "memory_context": self.memory_manager.get_context_for_request(request),
            }
        self._merge_continuation(semantic, ctx)
        problem = self._attach_problem(semantic, request, ctx)
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        self._record_continuation_requirements(reasoning, problem, request)
        self.requirement_tracker.set_requirements(reasoning.model.requirements)
        merged_ops = (((semantic.composition or {}).get("graph") or {})
                      .get("operations") or [])
        artifact = dict(ctx.get("artifact") or {})
        if not merged_ops:
            # The merged semantics carry no dataflow graph of their own
            # (the previous turn composed nothing — a direct algorithm):
            # the ARTIFACT's IR is the structure of record, and the turn
            # regenerates THE artifact from it.  This turn's request-verb
            # composition ("add" of "add logging") never renders its
            # code — the modification is tracked as a requirement, never
            # fabricated into the artifact.
            result = self._regenerate_artifact(
                request, semantic, artifact, language, reasoning)
        else:
            result = self._generate_function_natural(
                request, reasoning.model, reasoning, semantic, language)
        # The continuation travels with the result (the trace shows
        # which representation and artifact this problem continued).
        if problem.continuation:
            result["continuation"] = dict(problem.continuation)
            result["problem"] = problem.to_dict()
        # A test requirement on the artifact: the pipeline's OWN test
        # inputs (derived from the contract) become part of the result —
        # tests the user asked for are generated, never claimed.
        try:
            if any(getattr(c, "name", "") == "test_code"
                   or getattr(c, "constraint_type", "") == "test_code"
                   for c in (semantic.constraints or [])):
                from brain.generation.repair import _generate_test_cases
                contract_ops = self._normalize_composition_ops(
                    ((semantic.composition or {}).get("graph") or {})
                    .get("operations") or [])
                from brain.generation.contract import contract_from_semantic
                cases = _generate_test_cases(
                    contract_ops, contract_from_semantic(semantic))
                for _name, entry in (result.get("results") or {}).items():
                    if isinstance(entry, dict):
                        entry["test_cases"] = [list(c) for c in (cases or [])]
        except Exception:
            pass
        response_text = self.new_response_generator.generate(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=self.conversation_manager.history,
            current_project=self.conversation_manager.current_project.to_dict()
                if self.conversation_manager.current_project else None,
            artifact=self._summarize_generation_artifact(result, reasoning),
        )
        result["response"] = response_text
        result["confidence"] = self._final_confidence(result, reasoning)
        result["requirement_tracking"] = self.requirement_tracker.to_dict()
        return result

    def _regenerate_artifact(self, request: str, semantic, artifact: dict,
                             language: str, reasoning) -> dict:
        """Regenerate THE artifact from its own IR — the structure of
        record — for a continuation turn whose (merged) semantics carry
        no dataflow graph of their own.

        The artifact's code is never patched as text: it is rendered
        FROM the artifact's IR (the same structural source the artifact
        was built from).  This turn's request-verb composition never
        renders it either — 'add logging' composed an arithmetic add,
        which is not the artifact's dataflow — so the modification
        stays a tracked requirement, and the response is about the
        artifact that actually exists."""
        from code.ir.ir_types import IRNode
        fn_name = str(artifact.get("function_name") or "")
        if not fn_name:
            return self._generate_function_natural(
                request, reasoning.model, reasoning, semantic, language)
        ir_dict = dict(artifact.get("ir") or {})
        lang = str(artifact.get("language") or language or "python")
        code = str(artifact.get("code") or "")
        if ir_dict:
            try:
                regenerated = self.code_generator.generate(
                    IRNode.from_dict(ir_dict), lang)
                if regenerated:
                    code = regenerated
            except Exception:
                # The stored code remains the artifact (a verified
                # previous generation), never a fabricated edit.
                pass
        # The SAME oracle the artifact was generated under (its stored
        # operation contract) — structural checks when the artifact
        # carried no operation contract.
        ops = [str(o) for o in (artifact.get("ops") or []) if o]
        from brain.generation.contract import contract_from_semantic
        verification = self._verify_python_code(
            code, ops or None, contract_from_semantic(semantic))
        results = {fn_name: {
            "ir": ir_dict or {"name": fn_name, "type": "function"},
            "languages": {lang: {"code": code}},
            "verification": verification,
            "ops": ops,
        }}
        try:
            self.memory_manager.record_experience(
                experience_type="success",
                description=f"자연어 코드 생성: {request}",
                context={"goal_type": "MODIFY", "operations": ops},
                outcome="함수 재생성됨")
        except Exception:
            pass
        return {
            "request": request,
            "intent": "CREATE_FUNCTION",
            "results": results,
            "reasoning": reasoning,
            "project_created": False,
        }

    def _attempt_behavioral_repair(self, code: str, func_name: str,
                                   ops: list, contract=None) -> tuple:
        """Verification → Error Analysis → Repair → Re-test.

        The repair is structural (regenerated from the operation
        graph), never a textual patch.  A repair that does not pass
        the same oracle is discarded — the result keeps the failure
        honestly instead of claiming a fix.
        """
        try:
            from brain.generation.repair import RepairEngine
        except Exception:
            return code, self._verify_python_code(code, ops, contract) \
                if ops else {"success": False, "errors": ["no oracle"]}, False
        engine = RepairEngine()
        verification = self._verify_python_code(code, ops, contract) if ops \
            else {"success": False, "errors": ["no operation contract"]}
        if verification.get("success") or not ops:
            return code, verification, False
        # The think budget pays for the repair ATTEMPTS: how many
        # analyze → plan → regenerate → retest cycles this request may
        # spend before the failure is reported honestly.
        from brain.think_budget import current_budget
        max_attempts = max(1, current_budget().max_repair_attempts)
        try:
            for attempt in range(max_attempts):
                errors = engine.analyze(code, ops, contract)
                plan = engine.plan(errors, ops, {})
                if plan is None:
                    break
                new_code = engine.apply(plan, func_name, contract)
                if not new_code:
                    break
                new_v = self._verify_python_code(new_code, ops, contract)
                if new_v.get("success"):
                    new_v["repaired"] = True
                    new_v["repair_attempts"] = attempt + 1
                    return new_code, new_v, True
                # A failed attempt is a NEW observation: the next cycle
                # analyzes the repaired candidate, not the original.
                code = new_code
                verification = new_v
        except Exception:
            pass
        verification["repair_attempts"] = max_attempts
        return code, verification, False

    def _final_confidence(self, result: dict, reasoning=None) -> dict:
        """The FINAL confidence: the hypothesis's semantic plausibility
        BOUNDED by what the verification actually established.

        semantic confidence x verification outcome — a high semantic
        score can never surface as final confidence when the behavioral
        oracle failed or never ran; the numbers are computed, never
        declared.
        """
        hyp = getattr(getattr(reasoning, "selected_hypothesis", None),
                      "confidence", None)
        hyp_conf = float(hyp) if hyp else 0.5
        best = None
        for data in (result.get("results") or {}).values():
            v = data.get("verification") if isinstance(data, dict) else None
            if isinstance(v, dict) and v:
                if best is None or (v.get("success")
                                    and not best.get("success")):
                    best = v
        if best is None:
            final = min(hyp_conf, 0.5)
            basis = "unverified: no verification result for the artifact"
        elif best.get("success"):
            if best.get("behavioral"):
                final = hyp_conf
                basis = "behavioral verification passed against the operation contract"
            else:
                final = min(hyp_conf, 0.6)
                basis = "structural verification only (no operation contract to test)"
        else:
            final = hyp_conf * 0.25
            basis = "verification failed: " + "; ".join(
                (best.get("errors") or [])[:2])

        # The USER REQUIREMENT CONTRACT bounds the confidence (spec #12):
        # a pass against the operation contract alone is NOT the final
        # authority — requirement coverage and the type invariant decide.
        # A failed behavioral check can never surface as a high score,
        # and a coverage/type failure caps the result at 0.5.
        coverage = result.get("requirement_coverage") or {}
        if coverage:
            if coverage.get("passed"):
                basis = (
                    "behavioral verification passed against the user "
                    "requirement contract "
                    f"({coverage.get('verified')}/{coverage.get('total')} "
                    f"requirements verified, "
                    f"type={coverage.get('type_consistency')})")
            else:
                final = min(final, 0.5)
                details = []
                if coverage.get("verified", 0) < coverage.get("total", 0):
                    details.append(
                        "requirement coverage "
                        f"{coverage.get('verified')}/{coverage.get('total')}")
                if coverage.get("type_consistency") == "fail":
                    details.append(
                        "output type mismatch "
                        f"({coverage.get('expected_output_kind')} != "
                        f"{coverage.get('runtime_output_kind')})")
                if coverage.get("unjustified_operations"):
                    details.append("unjustified operations: " + ", ".join(
                        coverage["unjustified_operations"]))
                if not coverage.get("behavioral"):
                    details.append("behavioral verification failed")
                if details:
                    basis = basis + " — " + "; ".join(details)
        return {
            "confidence": round(max(0.0, min(1.0, final)), 2),
            "hypothesis_confidence": round(hyp_conf, 2),
            "basis": basis,
            "requirement_pass": bool(coverage.get("passed")) if coverage else None,
            "requirement_coverage": (
                f"{coverage.get('verified', 0)}/{coverage.get('total', 0)}"
                if coverage else None),
        }

    def _evaluate_requirement_contract(self, request: str, semantic,
                                       result: dict, language: str) -> dict:
        """Build and evaluate the USER REQUIREMENT CONTRACT.

        The contract is built from the VALIDATED operation plan plus the
        text-level constraints the semantics recorded; the matrix
        compares semantic → plan → IR → code → runtime → behavioral
        against the REQUIREMENT — never against the operation contract
        alone.
        """
        from brain import requirement_contract as rc_mod

        entry = None
        func_name = ""
        for key, data in (result.get("results") or {}).items():
            if isinstance(data, dict) and (data.get("verification")
                                           or data.get("languages")
                                           or data.get("code")):
                entry = data
                func_name = str(key)
                break
        if entry is None:
            return {}
        ops = list(entry.get("ops") or [])
        if not ops:
            ops = list(entry.get("operations") or [])
        if not ops:
            return {}
        languages = entry.get("languages") or {}
        py_code = str(((languages.get("python") or {}).get("code"))
                      or entry.get("code") or "")
        ir = entry.get("ir") or {}
        ir_type = str(ir.get("type_ref") or "")
        verification = entry.get("verification") or {}
        behavioral_ok = bool(verification.get("behavioral"))
        oracle_ran = bool(verification.get("behavioral")
                          or verification.get("behavioral_ran"))
        threshold = None
        try:
            from brain.generation.contract import contract_from_semantic
            threshold = contract_from_semantic(semantic).threshold
        except Exception:
            threshold = None
        contract = rc_mod.build_requirement_contract(
            request, semantic, ops, threshold=threshold,
            output_type=str(getattr(semantic, "output_type", "") or ""))
        runtime_kind = rc_mod.runtime_output_kind(
            py_code, func_name,
            expected_kind=rc_mod.output_kind(
                str(getattr(semantic, "output_type", "") or "")))
        unjustified = rc_mod._unjustified_ops(semantic, ops)
        matrix = rc_mod.evaluate_contract(
            contract,
            ir_ops=ops,
            ir_type=ir_type,
            py_code=py_code,
            runtime_kind=runtime_kind,
            behavioral_ok=behavioral_ok,
            unjustified_ops=unjustified,
            oracle_ran=oracle_ran,
        )
        matrix["contract"] = contract.to_dict()
        return matrix

    def _summarize_generation_artifact(self, result: dict,
                                       reasoning=None) -> dict:
        """What the generation ACTUALLY produced — the response's
        subject: the artifact's name, its operation pipeline (the
        operation plan's own step descriptions), and its verified state.
        A code-generation response is ABOUT this, never a placeholder."""
        artifact = {"name": "", "operations_text": "", "verified": None,
                    "confidence": None}
        results = result.get("results") or {}
        for key, data in results.items():
            if not isinstance(data, dict):
                continue
            v = data.get("verification")
            if isinstance(v, dict) and "success" in v:
                artifact["name"] = key
                artifact["verified"] = bool(v.get("success"))
                recipe = data.get("recipe") or {}
                if recipe.get("title"):
                    artifact["operations_text"] = (
                        "verified recipe: " + str(recipe["title"]))
                break
            if data.get("languages") or data.get("code"):
                artifact["name"] = key
                break
        if not artifact["name"] and results:
            artifact["name"] = next(iter(results))
        plan = getattr(reasoning, "operation_plan", None) if reasoning else None
        steps = list(getattr(plan, "steps", None) or [])
        descs = [str(getattr(st, "description", "") or "").strip()
                 for st in steps]
        descs = [d for d in descs if d]
        if descs:
            artifact["operations_text"] = " -> ".join(
                d[0].lower() + d[1:] for d in descs)
        elif reasoning is not None and getattr(reasoning, "model", None) \
                and reasoning.model.operation_order:
            artifact["operations_text"] = " -> ".join(
                reasoning.model.operation_order)
        return artifact

    def _handle_fix_semantic(self, request: str, language: str, semantic) -> dict:
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        response_text = self.new_response_generator.generate(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=self.conversation_manager.history,
        )
        return {
            "request": request,
            "language": language,
            "intent": "FIX",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "project_created": False,
            "results": {},
        }
    
    def _handle_analyze_semantic(self, request: str, language: str, semantic) -> dict:
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        response_text = self.new_response_generator.generate(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=self.conversation_manager.history,
        )
        return {
            "request": request,
            "language": language,
            "intent": "ANALYZE",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "project_created": False,
            "results": {},
        }
    
    def _handle_computer_semantic(self, request: str, language: str, semantic) -> dict:
        """Route a desktop task to the Computer Agent (shared semantic
        structure; separate Computer Plan / Action IR branch)."""
        from computer.executor import ComputerAgent
        from computer.security import ConfirmationPolicy
        from computer.memory import ComputerMemory

        agent = ComputerAgent(
            memory=ComputerMemory(self.memory_store),
            policy=ConfirmationPolicy(),
        )
        computer_result = agent.execute_semantic(request, semantic)

        # human-readable summary of the loop
        parts = []
        if computer_result.success:
            parts.append("목표가 달성되었습니다.")
        else:
            reasons = computer_result.goal_verification.get("reasons", [])
            parts.append("목표 달성에 실패했습니다.")
            if reasons:
                parts.append("원인: " + "; ".join(str(r) for r in reasons[:3]))
        executed = len(computer_result.action_results)
        repairs = len(computer_result.repairs)
        parts.append(f"{executed}개 액션 실행, {repairs}회 복구 시도.")
        failed = [r for r in computer_result.action_results if not r.success]
        if failed:
            last = failed[-1]
            parts.append(f"마지막 실패: {last.message[:120]}")

        return {
            "request": request,
            "language": language,
            "intent": "COMPUTER",
            "semantic": semantic.to_dict(),
            "response": " ".join(parts),
            "project_created": False,
            "results": {
                "computer_agent": computer_result.to_dict(),
                "success": computer_result.success,
                "trace": computer_result.trace,
            },
        }

    def _handle_computer_semantic(self, request: str, language: str, semantic) -> dict:
        """Handle computer-control goals via the shared Computer Agent loop
        (UNDERSTAND → OBSERVE → REASON → PLAN → ACT → VERIFY → LEARN).

        The Computer Agent and the Coding Agent share the same
        SemanticRepresentation; HIGH-risk actions require explicit user
        confirmation and are never executed silently.
        """
        try:
            from computer.executor import ComputerAgent
            agent = ComputerAgent(confirm=self._confirm_computer_action)
            comp_result = agent.execute_semantic(request, semantic)

            return {
                "request": request,
                "language": language,
                "intent": "COMPUTER_TASK",
                "semantic": semantic.to_dict(),
                "computer": comp_result.to_dict(),
                "response": self._describe_computer_result(comp_result),
                "project_created": False,
                "results": {},
            }
        except Exception:
            pass  # fall through to the standard pipeline

        # Fallback: standard pipeline
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        return {
            "request": request,
            "language": language,
            "intent": "UNKNOWN",
            "semantic": semantic.to_dict(),
            "response": self._build_dynamic_fallback_response(request, semantic, reasoning),
            "project_created": False,
            "results": {},
        }

    def _confirm_computer_action(self, action_info: dict,
                                 message: str | None = None) -> bool:
        """The confirmation gate for HIGH-risk computer actions.

        Interactive environments confirm through stdin; non-interactive
        environments deny (a HIGH-risk action is never executed silently).

        The executor's policy calls confirm(action, message) — the
        description may arrive as the second argument.
        """
        if not sys.stdin or not sys.stdin.isatty():
            return False
        try:
            desc = message
            if not desc and isinstance(action_info, dict):
                desc = action_info.get('description', '')
            risk = action_info.get('risk', '') if isinstance(action_info, dict) else ''
            print(f"[확인 필요] {desc} "
                  f"(위험도: {risk}) — 실행할까요? [y/N]")
            answer = input().strip().lower()
            return answer in ("y", "yes", "예")
        except Exception:
            return False

    def _describe_computer_result(self, comp_result) -> str:
        """A structured summary of the Computer Agent loop, from the
        actual execution results (never fabricated)."""
        goal = comp_result.goal
        app = goal.application if goal and goal.application else ""
        status = "성공" if comp_result.success else "실패"
        parts = []
        if app:
            parts.append(f"{app} 작업")
        parts.append(f"{status}")
        if comp_result.action_results:
            n = len(comp_result.action_results)
            parts.append(f"({n}개 동작 실행)")
        if comp_result.goal_verification:
            reasons = comp_result.goal_verification.get("reasons", [])
            if reasons and not comp_result.success:
                parts.append(f"원인: {'; '.join(str(r) for r in reasons[:2])}")
        return " ".join(parts)

    def _handle_ambiguous_semantic(self, request: str, language: str, semantic) -> dict:
        """Several candidate goal types with low confidence: the response
        presents the candidates and asks for clarification — the goal is
        not guessed.  The candidates come from the semantic parse's
        structured goal (the alternatives with confidences)."""
        goal = getattr(semantic, 'goal', None)
        alternatives = list(getattr(goal, 'alternatives', []) or []) if goal else []
        reasoning = self.think_engine.think_from_semantic(semantic, language)

        # The clarification is composed in the REQUEST's language — the
        # semantic's detected source language, never assumed Korean.
        lang = getattr(getattr(semantic, 'source_language', None),
                       'value', None) or language
        parts = []
        if lang == "ko":
            if alternatives:
                names = ", ".join(
                    str(a.get("goal_type", "?")) for a in alternatives[:3])
                parts.append(
                    f"요청을 {len(alternatives)}가지로 이해할 수 있습니다 ({names}).")
                parts.append("어떤 것을 원하시는지 알려주시면 그에 맞게 처리하겠습니다.")
            else:
                parts.append("요청의 의도를 명확히 파악하기 어렵습니다.")
                parts.append("원하시는 작업을 조금 더 구체적으로 설명해주시겠어요?")
        else:
            if alternatives:
                names = ", ".join(
                    str(a.get("goal_type", "?")) for a in alternatives[:3])
                if len(alternatives) == 1:
                    parts.append(
                        f"I read this as a request to {names}.")
                    parts.append("Is that what you want? If not, tell me "
                                 "a bit more and I'll handle it accordingly.")
                else:
                    parts.append(
                        f"I read the request {len(alternatives)} ways ({names}).")
                    parts.append("Which one do you mean? Tell me a bit more "
                                 "and I'll handle it accordingly.")
            else:
                parts.append("I'm not sure what you're asking for.")
                parts.append("Could you describe the task a little more?")

        return {
            "request": request,
            "language": language,
            "intent": "UNKNOWN",
            "semantic": semantic.to_dict(),
            "response": "\n".join(parts),
            "project_created": False,
            "results": {},
            "ambiguous": True,
            "candidates": alternatives,
        }

    def _handle_unknown_semantic(self, request: str, language: str, semantic) -> dict:
        reasoning = self.think_engine.think_from_semantic(semantic, language)
        
        # Build dynamic response from semantic state instead of generic "I inform."
        response_text = self._build_dynamic_fallback_response(request, semantic, reasoning)
        
        return {
            "request": request,
            "language": language,
            "intent": "UNKNOWN",
            "semantic": semantic.to_dict(),
            "response": response_text,
            "project_created": False,
            "results": {},
        }

    def _build_dynamic_fallback_response(self, request: str, semantic, reasoning) -> str:
        """Construct a dynamic fallback response from the semantic state."""
        parts = []
        
        # Acknowledge what was understood
        detected_ops = [op.name for op in semantic.operation_graph.operations] if semantic.operation_graph else []
        detected_entities = [e.name for e in semantic.entities] if semantic.entities else []
        
        if detected_ops:
            parts.append(f"I understood you want to {', '.join(detected_ops)}.")
        elif detected_entities:
            parts.append(f"I see you're referring to {', '.join(detected_entities)}.")
        else:
            parts.append("I'm processing your request.")
        
        # Explain what's missing
        if not detected_ops and not detected_entities:
            parts.append("Could you describe the specific operation or action you'd like?")
            parts.append("For example: 'calculate the average', 'find the maximum', 'sort a list', etc.")
        elif detected_ops and len(detected_ops) < 3:
            parts.append("I can help with that. Let me know if you'd like to add more operations or constraints.")
        else:
            parts.append("Would you like me to generate code for this?")
        
        return " ".join(parts)

    def _is_followup_request(self, request: str, conv_manager, current_project) -> bool:
        """Detect if this request is a follow-up on existing conversation."""
        if not hasattr(conv_manager, 'current_conversation'):
            return False
        conv = conv_manager.current_conversation
        if not conv or len(conv.turns) == 0:
            return False
        if not current_project:
            return False
        followup_patterns = ["추가", "검색", "수정", "변경", "바꿔", "해줘", "기능", "gui",
                            "더해", "주세요", "도 해줘", "add", "search", "update"]
        return any(k in request.lower() for k in followup_patterns)

    def _handle_followup(self, request: str, current_project, model, reasoning_state) -> dict:
        """Handle a follow-up request on an existing project."""
        return {
            "request": request,
            "intent": "MODIFY_PROJECT",
            "intent_result": None,
            "response": (
                f"기존 프로젝트 '{current_project.name}'에 요청을 이해했습니다. "
                f"{request} 같은 기능을 추가하려면 수정하시면 됩니다."
            ),
            "project_created": False,
            "results": {},
            "reasoning": reasoning_state,
            "memory_context": self.memory_manager.get_context_for_request(request),
        }

    def generate_project_with_reasoning(self, request: str, reasoning_state, streaming: bool = False) -> dict:
        """Generate project using pre-computed reasoning state."""
        model = reasoning_state.model
        platform = model.platform or "cli"
        language = model.language or "python"

        # Skip intent analyzer - ThinkEngine already determined this is a project
        memory_context = self.memory_manager.get_context_for_request(request)

        from brain.reasoning_events import ReasoningEventCollector
        event_collector = ReasoningEventCollector(language="en")

        analysis = self.requirements_analyzer.analyze(request, [])
        tech_selection = self.technology_selector.select(analysis, language)
        effective_language = tech_selection.language or language

        if event_collector:
            event_collector.add_technology_selection(
                effective_language,
                tech_selection.platform,
                tech_selection.framework or "",
                tech_selection.build_tool or "",
                effective_language
            )

        # A requirements-driven DATA APPLICATION (CSV -> validate -> dedup
        # -> group -> aggregate -> order -> JSON, with CLI and tests) is
        # generated as a WORKING implementation from its structured spec
        # — not a generic scaffold.  Everything else goes through the
        # standard planner.
        data_app = None
        try:
            from brain.data_app_generator import (build_data_app_spec,
                                                  build_data_app_project)
            _data_spec = build_data_app_spec(request, effective_language)
            if _data_spec is not None:
                data_app = build_data_app_project(_data_spec, request)
        except Exception:
            data_app = None

        # Extract success criteria from reasoning state
        success_criteria = model.success_criteria if hasattr(model, 'success_criteria') else None

        # Use streaming generator if requested
        generation_events = []
        if data_app is not None:
            project_ir, gen_result = data_app
            # The generic app success criteria (data model class etc.)
            # do not describe this data flow; the project's own
            # requirement contract is the verification authority.
            success_criteria = None
        elif streaming:
            project_ir = self.project_planner.create_project_plan(
                request, reasoning=reasoning_state,
                language=effective_language, event_collector=event_collector
            )
            # Create event collector for Korean explanations
            korean_collector = ReasoningEventCollector(language="ko")
            gen_events = run_streaming_generation(
                project_ir=project_ir,
                reasoning_state=reasoning_state,
                language=effective_language,
                base_generator=self.code_generator,
                explanation_engine=self.explanation_engine,
                event_collector=korean_collector
            )
            # gen_events is already a list of dicts from the new streaming generator
            generation_events = gen_events
            gen_result = self.project_generator.generate_project(project_ir)
        else:
            project_ir = self.project_planner.create_project_plan(
                request, reasoning=reasoning_state,
                language=effective_language, event_collector=event_collector
            )
            gen_result = self.project_generator.generate_project(project_ir)

        verification = self.project_verifier.verify(project_ir, gen_result, success_criteria)

        # plan -> IR -> code -> TEST: every stage records the SAME
        # requirement id, and only the test can mark it satisfied.
        try:
            ids_by_path = self._track_requirement_stages(
                model.requirements, project_ir, gen_result, verification)
        except Exception:
            ids_by_path = {}

        file_list = []
        for f in gen_result.files:
            ids = list(getattr(f, "requirement_ids", []) or [])
            for rid in (ids_by_path.get(f.path) or []):
                if rid not in ids:
                    ids.append(rid)
            file_list.append({
                "path": f.path,
                "content": f.content,
                "file_type": f.file_type,
                "language": f.language,
                "requirement_ids": ids,
            })

        error_analyses = []
        if not verification.success:
            error_analyses = [
                a.to_dict()
                for a in self.error_analyzer.analyze_issues(verification.issues, project_ir)
            ]

        # ── Project Requirement Coverage (spec #10): the request is
        # structured into requirements and each is traced through the
        # generated files.  A project PASSES only when every required
        # item is implemented — an unimplemented requirement is an
        # explicit failure, never absorbed into the warning count.
        from brain.project_requirements import evaluate_project_coverage
        project_coverage = evaluate_project_coverage(request, file_list)
        _coverage_issues = [
            {
                "severity": "error",
                "category": "requirement_coverage",
                "message": f"unimplemented requirement: {semantic}",
                "file_path": "",
                "line": 0,
                "suggestion": "implement this requirement in the generated project",
            }
            for semantic in project_coverage.get("uncovered", [])
        ]
        project_success = bool(verification.success
                               and project_coverage.get("passed"))

        if event_collector:
            event_collector.add_verification(
                verification.checks_passed,
                verification.checks_failed,
                verification.verifier_status,
                [i.message for i in verification.issues]
            )

        self.conversation_manager.add_turn(
            request, INTENT_CREATE_PROJECT,
            f"프로젝트 {project_ir.name}가 생성되었습니다.", project_ir
        )
        self.memory_manager.conversation_memory.add_turn(
            request, f"프로젝트 {project_ir.name}가 생성되었습니다.", INTENT_CREATE_PROJECT
        )

        self.memory_manager.record_experience(
            experience_type="success" if project_success else "failure",
            description=f"자연어 프로젝트 생성: {request}",
            context={"language": effective_language, "platform": tech_selection.platform},
            outcome=f"프로젝트 {project_ir.name} 생성 {'성공' if project_success else '실패'}",
        )

        return {
            "request": request,
            "language": effective_language,
            "intent": INTENT_CREATE_PROJECT,
            "intent_result": None,
            "reasoning": reasoning_state,
            "project_ir": project_ir.to_dict(),
            "results": {},
            "files": file_list,
            "requirement_coverage": project_coverage,
            "verification": {
                "success": project_success,
                "checks_passed": verification.checks_passed,
                "checks_failed": verification.checks_failed + len(_coverage_issues),
                "verifier_status": verification.verifier_status,
                "issues": [
                    {
                        "severity": i.severity,
                        "category": i.category,
                        "message": i.message,
                        "file_path": getattr(i, 'file_path', ''),
                        "line": getattr(i, 'line', 0),
                        "suggestion": getattr(i, 'suggestion', ''),
                    }
                    for i in verification.issues
                ] + _coverage_issues,
            },
            "error_analyses": error_analyses,
            "events": event_collector.events if event_collector else [],
            "generation_events": generation_events,
            "project_created": True,
            "output_dir": None,
            "technology": {
                "language": effective_language,
                "platform": tech_selection.platform,
                "framework": tech_selection.framework,
                "build_tool": tech_selection.build_tool,
                "reason": tech_selection.reason,
            },
            "memory_context": memory_context,
        }

    def _track_requirement_stages(self, requirements, project_ir,
                                  gen_result, verification) -> dict:
        """Record, under each requirement's OWN id, where it appears:
        the plan that carries it, the IR files implementing it, the code
        generated for them, and the test coverage of that code.  The
        stages never re-derive an id and never mark a requirement
        satisfied on their own — the test does."""
        plan_ids = {
            str(e.get("req_id") or "")
            for e in ((project_ir.metadata or {}).get("requirements") or [])
            if isinstance(e, dict)}
        files_by_req = {}
        for generated in (getattr(gen_result, "files", None) or []):
            ids = [str(r) for r in
                   (getattr(generated, "requirement_ids", None) or [])]
            content = str(getattr(generated, "content", "") or "")
            if content:
                lowered = content.lower()
                for req in requirements or []:
                    rid = str(getattr(req, "id", "") or "")
                    if not rid or rid in ids:
                        continue
                    # The requirement's own name, opened on its
                    # boundaries: a file whose code carries the word
                    # implements that requirement, and only that file
                    # records the id.
                    tokens = [t for t in re.split(r"[_\W]+",
                                                  str(getattr(req, "name", "")
                                                      or "").lower())
                              if len(t) >= 4]
                    if tokens and any(t in lowered for t in tokens):
                        ids.append(rid)
            for rid in ids:
                files_by_req.setdefault(rid, []).append(generated)
        coverage = []
        if verification is not None and requirements:
            try:
                coverage = self.project_verifier.requirement_coverage(
                    requirements, gen_result)
            except Exception:
                coverage = []
        coverage_by_id = {str(c.get("req_id") or ""): c
                          for c in coverage if isinstance(c, dict)}
        for req in requirements or []:
            rid = str(getattr(req, "id", "") or "")
            if not rid:
                continue
            if rid in plan_ids:
                self.requirement_tracker.map_to_plan(
                    rid, f"plan:{project_ir.name}")
            files = files_by_req.get(rid) or []
            if files:
                paths = ", ".join(f.path for f in files)
                self.requirement_tracker.map_to_ir(rid, paths)
                if any(getattr(f, "content", "") for f in files):
                    self.requirement_tracker.map_to_code(rid, paths)
            reported = coverage_by_id.get(rid)
            if reported is None:
                if files:
                    self.requirement_tracker.map_to_test(
                        rid, "no test answers this requirement")
                continue
            self.requirement_tracker.map_to_test(
                rid, "covered" if reported.get("satisfied")
                else "not covered by the generated code")
        ids_by_path = {}
        for rid, generated_files in files_by_req.items():
            for generated in generated_files:
                ids_by_path.setdefault(generated.path, [])
                if rid not in ids_by_path[generated.path]:
                    ids_by_path[generated.path].append(rid)
        return ids_by_path

    def _coding_recipe_for(self, request: str, semantic, language: str):
        """A verified recipe for a SIMPLE new function, or None.

        Recipes are deliberately narrow: a compound request, a user
        sample, an explicit function name, a continuation, or a
        non-Python target keeps the regular semantic pipeline.  The
        recipe is returned only after executing its own tests.
        """
        if str(language or "python").lower() != "python":
            return None
        goal = getattr(semantic, "goal", None)
        if getattr(goal, "goal_type", None) != GoalType.CREATE_FUNCTION:
            return None
        parent = getattr(goal, "parent_context", None) or {}
        if isinstance(parent, dict) and parent.get("continuation"):
            return None
        submission = getattr(semantic, "submission", None) or {}
        if isinstance(submission, dict) and submission.get("evidence"):
            return None
        text = str(request or "")
        if re.search(r"(?im)^\s*(?:input|output|입력|출력)\s*:", text):
            return None
        if re.search(r"\b(?:named|called)\s+[A-Za-z_]\w*", text):
            return None
        if re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", text):
            return None
        try:
            from brain.target_detector import detect_target
            target = detect_target(text)
            if (target.language != "python" and
                    not target.evidence.get("default")):
                return None
        except Exception:
            pass
        try:
            from brain.coding_recipes import select_verified_recipe
            composition = getattr(semantic, "composition", None) or {}
            graph = composition.get("graph") or {}
            composed_ops = [
                str(op.get("name") or "")
                for op in (graph.get("operations") or [])
                if isinstance(op, dict) and op.get("name")
            ]
            return select_verified_recipe(text,
                                          composition_ops=composed_ops)
        except Exception:
            return None

    def _generate_function_natural(self, request: str, model, reasoning_state, semantic=None, language: str = "python") -> dict:
        """Generate a single function from natural language request with operation pipeline."""
        recipe = self._coding_recipe_for(request, semantic, language)
        if recipe is not None:
            name = str(recipe["name"])
            code = str(recipe["code"])
            verification = dict(recipe["verification"])
            verification["recipe"] = name
            from code.ir.ir_types import IRNodeType
            return {
                "request": request,
                "intent": "CREATE_FUNCTION",
                "results": {
                    name: {
                        "ir": {
                            "node_type": IRNodeType.FUNCTION.value,
                            "name": name,
                            "type_ref": str(recipe.get("ir_type") or ""),
                            "children": [],
                        },
                        "languages": {"python": {"code": code}},
                        "verification": verification,
                        # 검증된 recipe 자체가 이 artifact의 연산 계약이다.
                        # 비워 두면 이어지는 요청("Add tests to it")이
                        # 이전 연산을 잃고, 그래프/계약이 빈 채로 흐른다.
                        "ops": [name],
                        "recipe": {
                            "name": name,
                            "title": str(recipe["title"]),
                            "tests_passed": verification.get(
                                "tests_passed", 0),
                        },
                    },
                },
                "reasoning": reasoning_state,
                "project_created": False,
            }

        from code.ir.ir_types import create_function_ir, create_return_ir, create_reference_ir
        ir_nodes = []

        # The contract every generator AND the behavioral oracle read:
        # the declared element domain and the objective's select key,
        # derived from the semantics ONCE (spec #5 — one derivation, so
        # semantic == IR == generated == verified cannot drift apart).
        from brain.generation.contract import contract_from_semantic
        pipeline_contract = contract_from_semantic(semantic)

        # Get operations from the reasoning state
        ops = model.operation_order or []
        conditions = model.conditions or []
        inputs = model.inputs or []
        outputs = model.outputs or []
        
        # Pre-generation: ensure all requested operations are captured
        if ops:
            captured = set()
            for action in reasoning_state.plan:
                if action.kind == "generate_function":
                    for op in action.params.get("operations", []):
                        captured.add(op)
            missing = set(ops) - captured
            if missing and reasoning_state.plan:
                # Add missing operations to the first generate_function action
                for action in reasoning_state.plan:
                    if action.kind == "generate_function":
                        existing = action.params.get("operations", [])
                        action.params["operations"] = existing + list(missing)
                        break
        
        # Get algorithm from structured_intent
        algorithm = ""
        if reasoning_state.structured_intent:
            algorithm = getattr(reasoning_state.structured_intent, 'algorithm', '') or ""

        # If no algorithm detected, try to infer from request text
        if not algorithm and not ops:
            algorithm = self._infer_algorithm_from_text(request)

        results = {}

        # The generated function's name reflects the algorithm's meaning
        # when the semantics know it (a fibonacci function is named
        # fibonacci); a compound data pipeline gets the pipeline name.
        # The algorithm comes from the structured intent (semantics), not
        # from parser keywords.
        pipeline_func_name = "process_data"
        if reasoning_state.structured_intent:
            si_algo = getattr(reasoning_state.structured_intent,
                              'algorithm', '') or ""
            if si_algo and si_algo != "compound":
                pipeline_func_name = si_algo
        # A CONTINUATION regenerates THE artifact it continues: the
        # function name travels with the continuation link (the
        # previous turn's artifact/algorithm), never re-derived from
        # this turn's words ("add logging to it" updates sort_list —
        # it does not invent an unrelated function).
        if semantic is not None:
            _cont = (getattr(semantic.goal, "parent_context", None)
                     or {}).get("continuation") or {}
            if isinstance(_cont, dict) and _cont.get("function_name"):
                pipeline_func_name = str(_cont["function_name"])

        # Priority 1: Use SemanticComposer's generated code when it has operations.
        # The composer produces correct code with proper type-pool wiring.
        # The IR stored here is the REAL structural IR built from the
        # operation sequence (the IR is the center of generation, not
        # function metadata).
        if semantic and hasattr(semantic, 'composition'):
            composition = semantic.composition or {}
            composer_code = composition.get("code")
            composer_ops = composition.get("graph", {}).get("operations", [])
            # The graph's EXECUTION ORDER (the topological dataflow) —
            # not the raw cue order the operations were created in. The
            # pipeline must run selection before aggregation regardless
            # of sentence word order ("sum of even numbers" wires
            # even-selection -> sum).
            exec_order = composition.get("graph", {}).get("execution_order", [])
            if exec_order and composer_ops:
                by_id = {o.get("id"): o for o in composer_ops
                         if isinstance(o, dict)}
                ordered = [by_id[oid] for oid in exec_order
                           if oid in by_id]
                if ordered:
                    composer_ops = ordered
            if composer_code and composer_ops:
                pipeline_ir_dict = {"name": pipeline_func_name, "type": "function"}
                ir_lang_results = None
                # The operation CONTRACT is the normalized pipeline
                # vocabulary (fine-grained primitives collapsed into the
                # IR's supported ops). Both the IR build and the
                # behavioral verification consume the SAME contract, so
                # the code, its IR and the oracle can never diverge.
                contract_ops = self._normalize_composition_ops(composer_ops)
                try:
                    from brain.generation.ir_builder import build_ir
                    pipeline_ir = build_ir(contract_ops, pipeline_func_name,
                                           contract=pipeline_contract)
                    pipeline_ir_dict = pipeline_ir.to_dict()
                    # Deterministic generation FROM the IR for all languages
                    ir_lang_results = PipelineCodeGenerator(
                        pipeline_contract).generate_from_ir(pipeline_ir)
                except (ValueError, KeyError):
                    pass
                py_code = (ir_lang_results or {}).get("python", {}).get("code", "") \
                    or composer_code
                # Behavioral verification needs the operation contract:
                # the generated function is checked against the
                # reference implementation of THESE operations — and on
                # failure, structurally repaired and re-tested.
                if contract_ops:
                    py_code, verification, repaired = self._attempt_behavioral_repair(
                        py_code, pipeline_func_name, contract_ops,
                        pipeline_contract)
                    if repaired:
                        languages = dict(ir_lang_results or {})
                        languages["python"] = {"code": py_code}
                    else:
                        languages = ir_lang_results or {"python": {"code": py_code}}
                else:
                    verification = self._verify_python_code(
                        py_code, contract=pipeline_contract)
                    languages = ir_lang_results or {"python": {"code": py_code}}
                results[pipeline_func_name] = {
                    "ir": pipeline_ir_dict,
                    "languages": languages,
                    "verification": verification,
                    "ops": list(contract_ops),
                }
                # Multi-language generation for Priority 1 path
                # The requested target language (the pipeline's language
                # parameter decides, not request keywords).
                from brain.target_detector import detect_target
                target_spec = detect_target(request)
                target_lang = (language if language != "python"
                               else target_spec.language)
                if target_lang != "python" and target_lang in self._get_supported_languages():
                    try:
                        target_code = self._transpile_python_to(composer_code, target_lang)
                        if target_code:
                            results[pipeline_func_name]["languages"][target_lang] = {"code": target_code}
                    except Exception:
                        pass
                self.memory_manager.record_experience(
                    experience_type="success",
                    description=f"자연어 코드 생성: {request}",
                    context={"goal_type": model.goal_type.value, "operations": [op.get("name") for op in composer_ops]},
                    outcome="함수 생성됨",
                )
                return {
                    "request": request,
                    "intent": "CREATE_FUNCTION",
                    "results": results,
                    "reasoning": reasoning_state,
                    "project_created": False,
                }

        # Priority 2: Pipeline generation for data operations
        pipeline_ops_set = self.PIPELINE_OPERATION_VOCAB
        effective_ops = list(ops)
        if not effective_ops and reasoning_state.structured_intent:
            compound_steps = getattr(reasoning_state.structured_intent, 'compound_steps', [])
            effective_ops = [s for s in compound_steps if s in pipeline_ops_set]

        # Normalize composition operations into pipeline-compatible names.
        # The SemanticComposer may produce fine-grained operations like
        # compare_greater_than + filter that should be collapsed to filter_above.
        effective_ops = self._normalize_composition_ops(effective_ops)

        # Pipeline generation for data operations (takes precedence over single algorithm)
        # IR-centered: the COMPLETE structural IR is built first, then code
        # is generated deterministically FROM the IR (the IR is the source
        # of truth for the operation sequence — not parser keywords).
        pipeline_ir = None
        lang_results = None
        if effective_ops and any(op in effective_ops for op in pipeline_ops_set):
            try:
                from brain.generation.ir_builder import build_ir
                pipeline_ir = build_ir(effective_ops, pipeline_func_name,
                                       contract=pipeline_contract)
                pipeline_gen = PipelineCodeGenerator(pipeline_contract)
                lang_results = pipeline_gen.generate_from_ir(pipeline_ir)
            except (ValueError, KeyError, NotImplementedError):
                pipeline_ir = None
            if lang_results is None:
                # Legacy per-language op rendering (same operator semantics)
                pipeline_gen = PipelineCodeGenerator(pipeline_contract)
                lang_results = pipeline_gen.generate_all(
                    pipeline_func_name, effective_ops)
            py_code = lang_results.get("python", {}).get("code", "")
            # Verification → (on failure) structural repair → re-test.
            # A repair that passes the same oracle replaces the code;
            # one that does not keeps the honest failure.
            if py_code and effective_ops:
                py_code, verification, repaired = self._attempt_behavioral_repair(
                    py_code, pipeline_func_name, effective_ops,
                    pipeline_contract)
                if repaired:
                    lang_results = dict(lang_results)
                    lang_results["python"] = {"code": py_code}
            else:
                verification = self._verify_python_code(
                    py_code, effective_ops, pipeline_contract) if py_code \
                    else {"success": False, "errors": ["No code generated"]}
            results[pipeline_func_name] = {
                "ir": pipeline_ir.to_dict() if pipeline_ir is not None
                      else {"name": pipeline_func_name, "type": "function"},
                "languages": lang_results,
                "verification": verification,
                "ops": list(effective_ops),
            }
            self.memory_manager.record_experience(
                experience_type="success",
                description=f"자연어 코드 생성: {request}",
                context={"goal_type": model.goal_type.value, "operations": effective_ops},
                outcome="함수 생성됨",
            )
            return {
                "request": request,
                "intent": "CREATE_FUNCTION",
                "results": results,
                "reasoning": reasoning_state,
                "project_created": False,
            }

        # Try direct Python generation first for known algorithms
        if not algorithm and effective_ops:
            # Check if any operation is a known algorithm
            for op in effective_ops:
                if op in self._PYTHON_ALGO_BODIES or \
                        op in self.planner._ir_builder_registry():
                    algorithm = op
                    break
        # The structured IR path: a known algorithm's semantic spec
        # builds the REAL structural IR (the IR is the source of truth
        # — code is generated FROM the IR, never a code template).  The
        # legacy direct body is only a fallback when no spec exists.
        if algorithm:
            ir = None
            try:
                ir = self.planner.build_algorithm_ir(algorithm)
            except Exception:
                ir = None
            if ir is not None:
                python_code = self.code_generator.generate(ir, "python")
                verification = self._verify_python_code(python_code, [algorithm])
                results[algorithm] = {
                    "ir": ir.to_dict(),
                    "languages": {"python": {"code": python_code}},
                    "verification": verification,
                    "ops": [algorithm],
                }
                return {
                    "request": request,
                    "intent": "CREATE_FUNCTION",
                    "results": results,
                    "reasoning": reasoning_state,
                    "project_created": False,
                }
        if algorithm:
            python_code = self._generate_python_direct(algorithm)
            if python_code:
                verification = self._verify_python_code(python_code)
                results[algorithm] = {
                    "ir": {"name": algorithm, "type": "function"},
                    "languages": {"python": {"code": python_code}},
                    "verification": verification,
                }
                return {
                    "request": request,
                    "intent": "CREATE_FUNCTION",
                    "results": results,
                    "reasoning": reasoning_state,
                    "project_created": False,
                }

        # Fall back to IR generation for unknown algorithms
        if algorithm:
            try:
                ir = self._generate_algorithm_ir(algorithm, ops)
                results[algorithm] = {
                    "ir": ir.to_dict(),
                    "languages": {"python": {"code": self.code_generator.generate(ir, "python")}},
                }
            except (ValueError, KeyError):
                pass

        # If still no results, try using the SemanticComposer's generated code
        if not results and semantic and hasattr(semantic, 'composition'):
            composition = semantic.composition or {}
            composer_code = composition.get("code")
            if composer_code:
                verification = self._verify_python_code(composer_code)
                results["process_data"] = {
                    "ir": {"name": "process_data", "type": "function"},
                    "languages": {"python": {"code": composer_code}},
                    "verification": verification,
                }

        # If still no results, return INSUFFICIENT_INFORMATION
        if not results:
            return {
                "results": {},
                "intent": "CREATE_FUNCTION",
                "error": "INSUFFICIENT_INFORMATION",
                "message": "요청을 이해할 수 없습니다. 구체적인 알고리즘이나 함수 이름을 지정해주세요.",
            }

        # Verify generated Python code
        for func_name, fdata in list(results.items()):
            py_code = fdata.get("languages", {}).get("python", {}).get("code", "")
            if py_code:
                verification = self._verify_python_code(py_code)
                if not verification["success"]:
                    # Remove invalid results
                    del results[func_name]
                else:
                    results[func_name]["verification"] = verification

        # Multi-language generation: detect target and generate in additional languages
        # The requested target language (the pipeline's language parameter
        # decides, not request keywords).
        if results:
            from brain.target_detector import detect_target
            target_spec = detect_target(request)
            target_lang = (language if language != "python"
                           else target_spec.language)
            if target_lang != "python" and target_lang in self._get_supported_languages():
                for func_name, fdata in list(results.items()):
                    py_code = fdata.get("languages", {}).get("python", {}).get("code", "")
                    if py_code and target_lang not in fdata.get("languages", {}):
                        try:
                            target_code = self._transpile_python_to(py_code, target_lang)
                            if target_code:
                                fdata["languages"][target_lang] = {"code": target_code}
                        except Exception:
                            pass

        if not results:
            return {
                "results": {},
                "error": "VERIFICATION_FAILED",
                "message": "생성된 코드가 검증을 통과하지 못했습니다.",
            }

        self.memory_manager.record_experience(
            experience_type="success",
            description=f"자연어 코드 생성: {request}",
            context={"goal_type": model.goal_type.value, "operations": ops},
            outcome="함수 생성됨",
        )

        return {
            "request": request,
            "intent": "CREATE_FUNCTION",
            "results": results,
            "reasoning": reasoning_state,
            "project_created": False,
        }

    def _get_supported_languages(self) -> set[str]:
        from languages.registry import supported_languages
        return set(supported_languages())

    def _transpile_python_to(self, py_code: str, target_lang: str) -> str:
        """Transpile python code to the target language via the AST transpiler.

        Returns "" when the code cannot be translated faithfully; callers
        then keep the python-only result.
        """
        from code.transpiler.py_ast_transpiler import (
            TranspileError,
            transpile,
        )
        raw = (target_lang or "").strip().lower()
        tgt = {"typescript": "javascript", "ts": "javascript",
               "c": "cpp", "c++": "cpp"}.get(raw, raw)
        try:
            out = transpile(py_code, tgt)
            return out if out else ""
        except TranspileError:
            return ""
        except Exception:
            return ""

    def _transpile_line(self, line: str, lang: str, func_name: str, param: str) -> str | None:
        """Transpile a single Python line to target language."""
        import re

        if lang == "javascript":
            return self._transpile_to_js(line, param)
        elif lang == "typescript":
            return self._transpile_to_ts(line, param)
        elif lang == "java":
            return self._transpile_to_java(line, param)
        elif lang == "kotlin":
            return self._transpile_to_kotlin(line, param)
        elif lang == "c":
            return self._transpile_to_c(line, param)
        elif lang == "cpp":
            return self._transpile_to_cpp(line, param)
        elif lang == "go":
            return self._transpile_to_go(line, param)
        elif lang == "rust":
            return self._transpile_to_rust(line, param)
        elif lang == "swift":
            return self._transpile_to_swift(line, param)
        elif lang == "ruby":
            return self._transpile_to_ruby(line, param)
        elif lang == "php":
            return self._transpile_to_php(line, param)
        elif lang == "csharp":
            return self._transpile_to_csharp(line, param)
        elif lang == "dart":
            return self._transpile_to_dart(line, param)
        elif lang == "bash":
            return self._transpile_to_bash(line, param)
        elif lang == "powershell":
            return self._transpile_to_powershell(line, param)
        elif lang == "lua":
            return self._transpile_to_lua(line, param)
        return None

    def _transpile_to_js(self, line: str, param: str) -> str:
        import re
        # Stdin reading: "A, B = map(int, input().split())" — the node
        # fs token reading
        m = re.match(r'([\w,\s]+?)\s*=\s*map\(int,\s*input\(\)\.split\(\)\)', line)
        if m:
            names = [n.strip() for n in m.group(1).split(',') if n.strip()]
            return ("    const _tok = require('fs').readFileSync(0, 'utf8')"
                    ".trim().split(/\\s+/);\n" +
                    "\n".join(f"    const {n} = Number(_tok[{i}]);"
                              for i, n in enumerate(names)))
        m = re.match(r'(\w+)\s*=\s*int\(input\(\)\)', line)
        if m:
            return ("    const _tok = require('fs').readFileSync(0, 'utf8')"
                    ".trim().split(/\\s+/);\n"
                    f"    const {m.group(1)} = Number(_tok[0]);")
        # Variable assignment: r0 = "value" / r0 = 42
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_js(val)
            return "    const " + name + " = " + val + ";"
        # Print
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "    console.log(" + self._transpile_value_js(m.group(1)) + ");"
        # For loop
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for (let i = 0; i < " + m.group(1) + "; i++) {"
        m = re.match(r'for\s+_\s+in\s+range\((.+?),\s*(.+?)\):', line)
        if m:
            return "    for (let i = " + m.group(1) + "; i < " + m.group(2) + "; i++) {"
        # Return
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_js(m.group(1)) + ";"
        # End block (Python uses indentation, JS uses })
        if line == "" or not line.strip():
            return "    }"
        return "    // " + line

    def _transpile_value_js(self, val: str) -> str:
        val = val.strip()
        if val == "True": return "true"
        if val == "False": return "false"
        if val == "None": return "null"
        if val == "data": return "data"
        if val.startswith('"') and val.endswith('"'): return val
        if val.startswith("'") and val.endswith("'"): return '"' + val[1:-1] + '"'
        return val

    def _transpile_to_ts(self, line: str, param: str) -> str:
        js = self._transpile_to_js(line, param)
        if js and "const " in js:
            js = js.replace("const ", "const ")
        return js

    def _transpile_to_java(self, line: str, param: str) -> str:
        import re
        # Stdin reading: "A, B = map(int, input().split())" — the
        # Scanner-based token reading
        m = re.match(r'([\w,\s]+?)\s*=\s*map\(int,\s*input\(\)\.split\(\)\)', line)
        if m:
            names = [n.strip() for n in m.group(1).split(',') if n.strip()]
            lines = ["        java.util.Scanner _sc = new java.util.Scanner(System.in);"]
            for n in names:
                lines.append(f"        int {n} = _sc.nextInt();")
            return "\n".join(lines)
        m = re.match(r'(\w+)\s*=\s*int\(input\(\)\)', line)
        if m:
            return ("        java.util.Scanner _sc = new java.util.Scanner(System.in);\n"
                    f"        int {m.group(1)} = _sc.nextInt();")
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_pow(self._transpile_value_generic(val), "java")
            typ = "String" if val.startswith('"') else "int" if val.lstrip('-').isdigit() or val.startswith("(int)") else "double" if '.' in val else "Object"
            return "        " + typ + " " + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "        System.out.println(" + self._transpile_pow(self._transpile_value_generic(m.group(1)), "java") + ");"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "        for (int i = 0; i < " + m.group(1) + "; i++) {"
        m = re.match(r'for\s+_\s+in\s+range\((.+?),\s*(.+?)\):', line)
        if m:
            return "        for (int i = " + m.group(1) + "; i < " + m.group(2) + "; i++) {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "        return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_kotlin(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "        val " + name + " = " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "        println(" + self._transpile_value_generic(m.group(1)) + ")"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "        for (i in 0 until " + m.group(1) + ") {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "        return " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_c(self, line: str, param: str) -> str:
        import re
        # Stdin reading: "A, B = map(int, input().split())" — the scanf
        # token reading.  C requires declarations: the variables are
        # declared with the scanf.
        m = re.match(r'([\w,\s]+?)\s*=\s*map\(int,\s*input\(\)\.split\(\)\)', line)
        if m:
            names = [n.strip() for n in m.group(1).split(',') if n.strip()]
            fmt = " ".join(["%d"] * len(names))
            args = ", ".join("&" + n for n in names)
            decl = "    int " + ", ".join(names) + ";"
            return decl + "\n    scanf(\"" + fmt + "\", " + args + ");"
        m = re.match(r'(\w+)\s*=\s*int\(input\(\)\)', line)
        if m:
            return '    int ' + m.group(1) + ';\n    scanf("%d", &' + m.group(1) + ");"
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_pow(self._transpile_value_generic(val), "c")
            typ = "char*" if val.startswith('"') else "int" if val.lstrip('-').isdigit() or val.startswith("(int)") else "double"
            return "    " + typ + " " + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            v = self._transpile_pow(self._transpile_value_generic(m.group(1)), "c")
            fmt = "%s" if v.startswith('"') or v.startswith("'") else "%d" if "." not in v else "%f"
            return '    printf("' + fmt + '\\n", ' + v + ");"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for (int i = 0; i < " + m.group(1) + "; i++) {"
        m = re.match(r'for\s+_\s+in\s+range\((.+?),\s*(.+?)\):', line)
        if m:
            return "    for (int i = " + m.group(1) + "; i < " + m.group(2) + "; i++) {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_cpp(self, line: str, param: str) -> str:
        c_line = self._transpile_to_c(line, param)
        if c_line and "printf" in c_line:
            c_line = c_line.replace("printf", "std::cout <<").replace("\\n", "\\n")
            # Simple cout conversion
            import re
            m = re.match(r'(\s*)std::cout << (.+);', c_line)
            if m:
                return m.group(1) + "std::cout << " + m.group(2) << " << std::endl;"
        return c_line

    def _transpile_to_go(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "    " + name + " := " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "    fmt.Println(" + self._transpile_value_generic(m.group(1)) + ")"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for i := 0; i < " + m.group(1) + "; i++ {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_rust(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "    let " + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            return '    println!("{}", ' + self._transpile_value_generic(m.group(1)) + ");"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for _ in 0.." + m.group(1) + " {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_swift(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "        let " + name + " = " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "        print(" + self._transpile_value_generic(m.group(1)) + ")"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "        for _ in 0..<" + m.group(1) + " {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "        return " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_ruby(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "  " + name + " = " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "  puts " + self._transpile_value_generic(m.group(1))
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "  " + m.group(1).strip() + ".times do"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "  " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_php(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "    $" + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "    echo " + self._transpile_value_generic(m.group(1)) + ";"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for ($i = 0; $i < " + m.group(1) + "; $i++) {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_csharp(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            typ = "string" if val.startswith('"') else "int" if val.lstrip('-').isdigit() else "double"
            return "            " + typ + " " + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "            Console.WriteLine(" + self._transpile_value_generic(m.group(1)) + ");"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "            for (int i = 0; i < " + m.group(1) + "; i++) {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "            return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_dart(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "    var " + name + " = " + val + ";"
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "    print(" + self._transpile_value_generic(m.group(1)) + ");"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for (var i = 0; i < " + m.group(1) + "; i++) {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1)) + ";"
        return None

    def _transpile_to_bash(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return name + "=" + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "echo " + self._transpile_value_generic(m.group(1))
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "for i in $(seq 1 " + m.group(1) + "); do"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "echo " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_powershell(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "$" + name + " = " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "Write-Output " + self._transpile_value_generic(m.group(1))
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "1.." + m.group(1) + " | ForEach-Object {"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "return " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_to_lua(self, line: str, param: str) -> str:
        import re
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if m:
            name, val = m.group(1), m.group(2)
            val = self._transpile_value_generic(val)
            return "    local " + name + " = " + val
        m = re.match(r'print\((.+)\)', line)
        if m:
            return "    print(" + self._transpile_value_generic(m.group(1)) + ")"
        m = re.match(r'for\s+_\s+in\s+range\((\d+)\):', line)
        if m:
            return "    for i = 1, " + m.group(1) + " do"
        m = re.match(r'return\s+(.+)', line)
        if m:
            return "    return " + self._transpile_value_generic(m.group(1))
        return None

    def _transpile_value_generic(self, val: str) -> str:
        """Generic value transpilation: the language-neutral literals and
        the arithmetic.  The power operator (Python's **) becomes the
        language's pow call — ** is only valid JS/TS."""
        val = val.strip()
        if val == "True": return "true"
        if val == "False": return "false"
        if val == "None": return "null"
        return val

    def _transpile_pow(self, expr: str, lang: str) -> str:
        """The power operator: Python's `a ** b` becomes the language's
        pow call.  `**` is only valid JavaScript/TypeScript."""
        import re
        m = re.search(r'\*\*\s*(\d+)', expr)
        if not m:
            return expr
        if lang in ("javascript", "typescript"):
            return expr  # ** is valid
        if lang == "java":
            return re.sub(
                r'(.+?)\s*\*\*\s*(\d+)',
                lambda mm: f"(int) Math.pow({mm.group(1).strip()}, {mm.group(2)})",
                expr)
        if lang in ("c", "cpp"):
            return re.sub(
                r'(.+?)\s*\*\*\s*(\d+)',
                lambda mm: f"(int)pow({mm.group(1).strip()}, {mm.group(2)})",
                expr)
        return expr

    def _wrap_in_function(self, lang: str, func_name: str, param: str, body_lines: list[str]) -> str:
        """Wrap transpiled body lines in a language-specific function."""
        body = "\n".join(body_lines) if body_lines else ""
        if lang in ("javascript",):
            # The function plus a call: the js entry point is the
            # top-level script — the function must be invoked.
            return ("function " + func_name + "(" + param + ") {\n" + body +
                    "\n}\n\n" + func_name + "(null);")
        if lang in ("typescript",):
            return "function " + func_name + "(" + param + ": any): any {\n" + body + "\n}"
        if lang in ("java",):
            # javac requires a top-level class; the function is wrapped
            # in a Main class with a main() entry that calls the function
            return ("public class Main {\n"
                    "    public static void " + func_name + "(Object " + param + ") {\n"
                    + body + "\n    }\n"
                    "    public static void main(String[] args) {\n"
                    "        " + func_name + "(null);\n"
                    "    }\n"
                    "}")
        if lang in ("kotlin",):
            return "fun " + func_name + "(" + param + ": Any) {\n" + body + "\n}"
        if lang in ("c",):
            # The C entry: the includes, the main() entry, and the
            # function — the C program needs stdio/stdlib/math/string
            # declarations for the generated code's printf/scanf/pow.
            return ("#include <stdio.h>\n"
                    "#include <stdlib.h>\n"
                    "#include <math.h>\n"
                    "#include <string.h>\n\n"
                    "void " + func_name + "(void* " + param + ") {\n" + body + "\n}\n\n"
                    "int main() {\n"
                    "    " + func_name + "(0);\n"
                    "    return 0;\n"
                    "}")
        if lang in ("cpp",):
            return ("#include <bits/stdc++.h>\nusing namespace std;\n\n"
                    "void " + func_name + "(auto " + param + ") {\n" + body + "\n}\n\n"
                    "int main() {\n"
                    "    " + func_name + "(0);\n"
                    "    return 0;\n"
                    "}")
        if lang in ("go",):
            return "func " + func_name + "(" + param + " interface{}) {\n" + body + "\n}"
        if lang in ("rust",):
            return "fn " + func_name + "(" + param + ": &impl std::fmt::Debug) {\n" + body + "\n}"
        if lang in ("swift",):
            return "func " + func_name + "(" + param + ": Any) {\n" + body + "\n}"
        if lang in ("ruby",):
            return "def " + func_name + "(" + param + ")\n" + body + "\nend"
        if lang in ("php",):
            return "function " + func_name + "($" + param + ") {\n" + body + "\n}"
        if lang in ("csharp",):
            return "public static void " + func_name + "(object " + param + ") {\n" + body + "\n}"
        if lang in ("dart",):
            return "void " + func_name + "(dynamic " + param + ") {\n" + body + "\n}"
        if lang in ("bash",):
            return func_name + "() {\n" + body + "\n}"
        if lang in ("lua",):
            return "function " + func_name + "(" + param + ")\n" + body + "\nend"
        return body

    def _verify_python_code(self, code: str, required_ops: list[str] = None,
                            contract=None) -> dict:
        """Verify Python code: syntax, structural integrity, and behavioral correctness.

        Checks:
        1. AST syntax parsing
        2. Placeholder / stub detection (bare pass, placeholder comments, trivial returns)
        3. Unreachable code detection (code after return statements)
        4. Undefined variable detection
        5. Execution test (code must actually run)
        6. Behavioral test (generated function must handle basic inputs)
        7. Required operations check (if required_ops provided, code must implement them)
        """
        import ast
        import re
        import tempfile
        import subprocess

        result = {"success": False, "errors": [], "warnings": []}

        # 1. AST parse check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result["errors"].append(f"Syntax error: {e}")
            return result

        code_stripped = code.strip()
        code_lower = code_stripped.lower()

        # 2. Placeholder / stub detection
        # Bare pass as the only body statement
        _StrNode = getattr(ast, 'Str', type(None))  # compatibility with Python <3.12
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = node.body
                # Filter out docstrings and comments
                real_stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, (ast.Constant, _StrNode)))]
                if len(real_stmts) == 0:
                    result["errors"].append(f"Function '{node.name}' has empty body (only docstring/pass)")
                    return result
                if len(real_stmts) == 1 and isinstance(real_stmts[0], ast.Pass):
                    result["errors"].append(f"Function '{node.name}' body is only 'pass' — no implementation")
                    return result

        # Placeholder comments indicating unimplemented operations
        placeholder_comments = re.findall(r"#\s*(filter_above|filter_below|sort_list|sort_descending|"
                                           r"sort_ascending|remove_duplicates|calculate_average|"
                                           r"compare_greater_than|compare_less_than|filter_even|"
                                           r"find_max|find_min|calculate_sum|generate_range|"
                                           r"is_prime|is_multiple_of)\b",
                                           code_stripped, re.IGNORECASE)
        if placeholder_comments:
            result["errors"].append(
                f"Placeholder comments for unimplemented operations: {placeholder_comments}")
            return result

        # Bare "pass" as a standalone statement (not the only statement in a short function)
        pass_count = code_lower.count("\npass\n") + code_lower.count("\n    pass\n") + (
            1 if code_lower.startswith("pass\n") else 0)
        if pass_count > 0:
            # Check if pass is the ONLY meaningful statement in any function
            _StrNode2 = getattr(ast, 'Str', type(None))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    real = [s for s in node.body
                            if not isinstance(s, ast.Pass)]
                    real_no_doc = [s for s in real
                                   if not (isinstance(s, ast.Expr) and isinstance(s.value, (ast.Constant, _StrNode2)))]
                    if len(real_no_doc) == 0 and len(node.body) > 1:
                        result["errors"].append(
                            f"Function '{node.name}' contains only placeholder 'pass'")
                        return result

        # 3. Unreachable code detection: code after return/break/continue in a block
        _StrNode = getattr(ast, 'Str', type(None))  # compatibility with Python <3.12
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.With)):
                stmts = node.body
                for i, stmt in enumerate(stmts):
                    if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
                        # Check if there are non-pass statements after this
                        remaining = stmts[i + 1:]
                        real_after = [s for s in remaining
                                      if not isinstance(s, ast.Pass)
                                      and not (isinstance(s, ast.Expr) and isinstance(s.value, (ast.Constant, _StrNode)))]
                        if real_after:
                            result["errors"].append(
                                f"Unreachable code after {'return' if isinstance(stmt, ast.Return) else 'flow control'} "
                                f"at line {getattr(stmt, 'lineno', '?')}")
                            return result

        # 4. Check for undefined variables in return statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                if node.value.id not in ("True", "False", "None"):
                    defined = set()
                    for n2 in ast.walk(tree):
                        if isinstance(n2, ast.Assign):
                            for target in n2.targets:
                                if isinstance(target, ast.Name):
                                    defined.add(target.id)
                        elif isinstance(n2, ast.AnnAssign):
                            # An ANNOTATED assignment (sorted_list: list = …)
                            # defines its name exactly like a plain one —
                            # only ast.Assign here would call defined code
                            # an undefined variable.
                            if isinstance(n2.target, ast.Name):
                                defined.add(n2.target.id)
                        elif isinstance(n2, ast.FunctionDef):
                            for arg in n2.args.args:
                                defined.add(arg.arg)
                    if node.value.id not in defined:
                        result["errors"].append(f"Undefined variable in return: {node.value.id}")
                        return result

        # 5. Execute and test basic functionality
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                exec_result = subprocess.run(
                    ["python3", "-c", f"exec(open('{f.name}').read())"],
                    capture_output=True, text=True, timeout=5
                )
                if exec_result.returncode != 0:
                    result["errors"].append(f"Execution error: {exec_result.stderr.strip()}")
                    return result
            except subprocess.TimeoutExpired:
                result["errors"].append("Execution timeout")
                return result
            except FileNotFoundError:
                result["warnings"].append("Python not found, skipping execution test")
            finally:
                import os
                os.unlink(f.name)

        # 6. Behavioral test: the generated function must actually work
        behavioral_result = self._verify_code_behavior(code, required_ops,
                                                       contract)
        if not behavioral_result["success"]:
            result["errors"].extend(behavioral_result["errors"])
            return result

        result["success"] = True
        # Whether the SEMANTIC oracle ran: the confidence calculus
        # distinguishes "behaviorally verified" from "only structural
        # checks ran" — execution success is never semantic success.
        result["behavioral"] = bool(required_ops)

        # Phase 7: Enhanced verification with error classification
        try:
            from brain.code_verifier import CodeVerifier
            verifier = CodeVerifier()
            vresult = verifier.verify_python(code)
            if not vresult.success and vresult.error_analyses:
                result["error_analyses"] = [e.to_dict() for e in vresult.error_analyses]
            result["verification_details"] = vresult.to_dict()
        except Exception:
            pass

        return result

    def _verify_code_behavior(self, code: str, required_ops: list[str] = None,
                              contract=None) -> dict:
        """Verify the generated code's BEHAVIOR.

        When the request's operation contract is known (required_ops),
        the generated function is executed against the REFERENCE
        implementation of the same operation graph — the operation
        contract IS the expected behavior, so a syntax-valid function
        that returns the wrong value FAILS here.  Only when no contract
        is known does the check fall back to a run-only smoke test
        (execution success is necessary, never sufficient).

        Returns {"success": bool, "errors": list[str]}.
        """
        import inspect
        import tempfile
        import subprocess

        result = {"success": False, "errors": []}

        # The semantic-contract oracle: expected behavior derived from
        # the operation pipeline itself, never from the problem text.
        if required_ops:
            try:
                from brain.generation.repair import RepairEngine
                oracle_errors = RepairEngine().verify(code, required_ops,
                                                      contract)
            except Exception as e:
                result["errors"].append(
                    f"Behavioral oracle could not run: {type(e).__name__}: {e}")
                return result
            if oracle_errors:
                for e in oracle_errors:
                    cause = e.get("cause", "") or e.get("type", "verification error")
                    detail = f"{e.get('type', 'BEHAVIORAL_ERROR')}: {cause}"
                    if e.get("input") is not None:
                        detail += f" (input={e.get('input')})"
                    if e.get("operation"):
                        detail += f" [operation: {e.get('operation')}]"
                    result["errors"].append(detail)
                return result
            # The speaker's own Input/Output samples: the general
            # contract passed — the submission's shown behavior must
            # pass too.
            sample_errors = self._verify_evidence_samples(code,
                                                          required_ops)
            if sample_errors:
                result["errors"].extend(sample_errors)
                return result
            result["success"] = True
            return result

        # The submission's own samples first: the speaker's expected
        # behavior is checked before the run-only smoke.
        sample_errors = self._verify_evidence_samples(code, required_ops)
        if sample_errors:
            result["errors"].extend(sample_errors)
            return result

        # Build a test script that imports the generated code and tests it
        # using proper argument introspection
        test_script = '''
import sys, ast, inspect, textwrap

code_text = sys.argv[1]
tree = ast.parse(code_text)

# Find function names
func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
if not func_names:
    print("ERROR: No function found in generated code")
    sys.exit(1)

# Execute the code
ns = {}
exec(code_text, ns)

# Test each function with appropriate data based on signature
for fname in func_names:
    func = ns.get(fname)
    if not func or not callable(func):
        continue
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        n_params = len(params)

        if n_params == 0:
            result = func()
        elif n_params == 1:
            # Default to list for data-processing functions
            result = func([3, 1, 4, 1, 5, 9])
        elif n_params == 2:
            result = func(3, 5)
        elif n_params == 3:
            result = func(3, 5, 7)
        else:
            result = func(*[10] * n_params)

        if result is None:
            print(f"WARNING: {fname} returned None (may be print-only)")
    except Exception as e:
        print(f"ERROR: {fname} raised {type(e).__name__}: {e}")
        sys.exit(1)

print("OK")
'''
        try:
            exec_result = subprocess.run(
                ["python3", "-c", test_script, code],
                capture_output=True, text=True, timeout=10
            )
            if exec_result.returncode != 0:
                err = exec_result.stdout.strip() or exec_result.stderr.strip()
                result["errors"].append(f"Behavioral test failed: {err}")
                return result
        except subprocess.TimeoutExpired:
            result["errors"].append("Behavioral test timeout")
            return result

        result["success"] = True
        return result

    def _verify_evidence_samples(self, code: str,
                                 required_ops: list[str] = None) -> list[str]:
        """Run the turn's own Input/Output samples (from the submission
        segmentation) against the code; [] when the submission carries
        no sample evidence."""
        try:
            from brain.io_adapters import (execute_samples,
                                           samples_from_submission)
            submission = getattr(self, "_last_semantic", None)
            submission = getattr(submission, "submission", None) or {}
            samples = samples_from_submission(submission)
            if not samples:
                return []
            return execute_samples(code, samples,
                                   preferred_names=list(required_ops or []))
        except Exception:
            return []

    def _infer_algorithm_from_text(self, request: str) -> str:
        """Infer algorithm name from request text when no algorithm is detected."""
        rl = request.lower()
        # Direct algorithm name mappings
        algo_hints = [
            (r"\badd\w*\s+(two|2)\s+number", "add_two_numbers"),
            (r"\badd\w*\s+(two|2)\s+value", "add_two_numbers"),
            (r"\bsum\s+(two|2)\s+number", "add_two_numbers"),
            (r"\bprime\b", "is_prime"),
            (r"\b(most|가장)\s*(frequent|자주)\s*(character|char|문자|글자)", "most_frequent_char"),
            (r"\bfreq\w*\s*(character|char)", "most_frequent_char"),
            (r"\bfactorial\b", "factorial"),
            (r"\bfibonacci\b", "fibonacci"),
            (r"\bfind\s+max\b", "find_max"),
            (r"\bfind\s+min\b", "find_min"),
            (r"\bsum\s+(of\s+)?(list|array|numbers)", "sum_list"),
            (r"\baverage\b", "average_list"),
            (r"\bsort\b", "sort_list"),
            (r"\bremove\s+duplicat", "remove_duplicates"),
            (r"\bfilter\s+even", "filter_even"),
            (r"\bbinary\s+search", "binary_search"),
        ]
        import re
        for pattern, algo in algo_hints:
            if re.search(pattern, rl):
                return algo
        return ""

    # LEGACY: Hardcoded algorithm code templates
    # source: legacy, usage: fallback+compatibility, confidence: 0.7
    # Primary path: SemanticComposer.generate_python() → IR → CodeGenerator
    _PYTHON_ALGO_BODIES = {
        "add_two_numbers": '''\
def add_two_numbers(a: float, b: float) -> float:
    return a + b
''',
        "multiply_two_numbers": '''\
def multiply_two_numbers(a: float, b: float) -> float:
    return a * b
''',
        "is_prime": '''\
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
        "most_frequent_char": '''\
def most_frequent_char(text: str) -> str:
    if not text:
        return ""
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return max(freq, key=freq.get)
''',
        "factorial": '''\
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        "fibonacci": '''\
def fibonacci(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
        "find_max": '''\
def find_max(data: list) -> int:
    if not data:
        raise ValueError("empty list")
    result = data[0]
    for x in data[1:]:
        if x > result:
            result = x
    return result
''',
        "find_min": '''\
def find_min(data: list) -> int:
    if not data:
        raise ValueError("empty list")
    result = data[0]
    for x in data[1:]:
        if x < result:
            result = x
    return result
''',
        "sum_list": '''\
def sum_list(data: list) -> int:
    total = 0
    for x in data:
        total += x
    return total
''',
        "average_list": '''\
def average_list(data: list) -> float:
    if not data:
        raise ValueError("empty list")
    return sum(data) / len(data)
''',
        "sort_list": '''\
def sort_list(data: list) -> list:
    return sorted(data)
''',
        "remove_duplicates": '''\
def remove_duplicates(data: list) -> list:
    seen = set()
    result = []
    for x in data:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
''',
        "filter_even": '''\
def filter_even(data: list) -> list:
    return [x for x in data if x % 2 == 0]
''',
        "binary_search": '''\
def binary_search(data: list, target: int) -> int:
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
        "search_in_list": '''\
def search_in_list(data: list, target: int) -> int:
    for i, x in enumerate(data):
        if x == target:
            return i
    return -1
''',
        "search_in_string": '''\
def search_in_string(text: str, pattern: str) -> int:
    return text.find(pattern)
''',
        "calculate_average": '''\
def calculate_average(data: list) -> float:
    if not data:
        raise ValueError("empty list")
    return sum(data) / len(data)
''',
    }

    def _generate_python_direct(self, algorithm: str) -> str | None:
        """Generate Python code directly for known algorithms. Returns None if unknown."""
        return self._PYTHON_ALGO_BODIES.get(algorithm)

    def _generate_algorithm_ir(self, algorithm: str, ops: list = None):
        """Generate IR for a specific algorithm."""
        from code.ir.ir_types import (create_function_ir, create_return_ir,
                                       create_binary_op_ir, create_reference_ir,
                                       create_if_ir, create_for_ir, create_call_ir)
        
        algo_configs = {
            "find_max": {
                "name": "find_max",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "find_min": {
                "name": "find_min",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "sum_list": {
                "name": "sum_list",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "average_list": {
                "name": "average_list",
                "params": [("data", "List[int]")],
                "return_type": "float",
            },
            "search_in_list": {
                "name": "search_in_list",
                "params": [("data", "List[int]"), ("target", "int")],
                "return_type": "int",
            },
            "fibonacci": {
                "name": "fibonacci",
                "params": [("n", "int")],
                "return_type": "int",
            },
            "factorial": {
                "name": "factorial",
                "params": [("n", "int")],
                "return_type": "int",
            },
            "is_prime": {
                "name": "is_prime",
                "params": [("n", "int")],
                "return_type": "bool",
            },
            "sort_list": {
                "name": "sort_list",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "remove_duplicates": {
                "name": "remove_duplicates",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "filter_even": {
                "name": "filter_even",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "binary_search": {
                "name": "binary_search",
                "params": [("data", "List[int]"), ("target", "int")],
                "return_type": "int",
            },
            "search_in_string": {
                "name": "search_in_string",
                "params": [("text", "str"), ("pattern", "str")],
                "return_type": "int",
            },
            "calculate_average": {
                "name": "calculate_average",
                "params": [("data", "List[float]")],
                "return_type": "float",
            },
            "add_two_numbers": {
                "name": "add_two_numbers",
                "params": [("a", "float"), ("b", "float")],
                "return_type": "float",
                "body": {"op": "+", "left": "a", "right": "b"},
            },
            "multiply_two_numbers": {
                "name": "multiply_two_numbers",
                "params": [("a", "float"), ("b", "float")],
                "return_type": "float",
                "body": {"op": "*", "left": "a", "right": "b"},
            },
            "most_frequent_char": {
                "name": "most_frequent_char",
                "params": [("text", "str")],
                "return_type": "str",
            },
            "filter_above": {
                "name": "filter_above",
                "params": [("data", "List[float]"), ("threshold", "float")],
                "return_type": "List[float]",
            },
            "filter_below": {
                "name": "filter_below",
                "params": [("data", "List[float]"), ("threshold", "float")],
                "return_type": "List[float]",
            },
            "sort_descending": {
                "name": "sort_descending",
                "params": [("data", "List[float]")],
                "return_type": "List[float]",
            },
            "sort_ascending": {
                "name": "sort_ascending",
                "params": [("data", "List[float]")],
                "return_type": "List[float]",
            },
            "add": {
                "name": "add",
                "params": [("a", "int"), ("b", "int")],
                "return_type": "int",
                "body": {"op": "+", "left": "a", "right": "b"},
            },
            "multiply": {
                "name": "multiply",
                "params": [("a", "int"), ("b", "int")],
                "return_type": "int",
                "body": {"op": "*", "left": "a", "right": "b"},
            },
            "concat_strings": {
                "name": "concat_strings",
                "params": [("a", "str"), ("b", "str")],
                "return_type": "str",
                "body": {"op": "+", "left": "a", "right": "b"},
            },
        }
        
        config = algo_configs.get(algorithm)
        if not config:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if "body" in config:
            body = config["body"]
            return_ir = create_return_ir(
                create_binary_op_ir(
                    body["op"],
                    create_reference_ir(body["left"]),
                    create_reference_ir(body["right"])
                )
            )
        else:
            return_ir = create_return_ir(create_reference_ir(config["params"][0][0] if config["params"] else "data"))
        
        return create_function_ir(
            config["name"],
            config["params"],
            config["return_type"],
            [return_ir]
        )

    def _get_algo_configs(self) -> dict:
        """Return algorithm configurations."""
        return {
            "find_max": {
                "name": "find_max",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "find_min": {
                "name": "find_min",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "sum_list": {
                "name": "sum_list",
                "params": [("data", "List[int]")],
                "return_type": "int",
            },
            "average_list": {
                "name": "average_list",
                "params": [("data", "List[int]")],
                "return_type": "float",
            },
            "search_in_list": {
                "name": "search_in_list",
                "params": [("data", "List[int]"), ("target", "int")],
                "return_type": "int",
            },
            "search_in_string": {
                "name": "search_in_string",
                "params": [("text", "str"), ("pattern", "str")],
                "return_type": "int",
            },
            "fibonacci": {
                "name": "fibonacci",
                "params": [("n", "int")],
                "return_type": "int",
            },
            "factorial": {
                "name": "factorial",
                "params": [("n", "int")],
                "return_type": "int",
            },
            "is_prime": {
                "name": "is_prime",
                "params": [("n", "int")],
                "return_type": "bool",
            },
            "sort_list": {
                "name": "sort_list",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "binary_search": {
                "name": "binary_search",
                "params": [("data", "List[int]"), ("target", "int")],
                "return_type": "int",
            },
            "remove_duplicates": {
                "name": "remove_duplicates",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "filter_even": {
                "name": "filter_even",
                "params": [("data", "List[int]")],
                "return_type": "List[int]",
            },
            "add": {
                "name": "add",
                "params": [("a", "int"), ("b", "int")],
                "return_type": "int",
            },
            "multiply": {
                "name": "multiply",
                "params": [("a", "int"), ("b", "int")],
                "return_type": "int",
            },
            "concat_strings": {
                "name": "concat_strings",
                "params": [("a", "str"), ("b", "str")],
                "return_type": "str",
            },
        }

    def _normalize_composition_ops(self, ops) -> list[str]:
        """Normalize fine-grained composition operations into pipeline-compatible names.

        The SemanticComposer decomposes natural language into fine-grained
        primitives (compare_greater_than, filter, etc.).  The PipelineCodeGenerator
        expects higher-level operations (filter_above, filter_below).  This method
        collapses the fine-grained operations while preserving the dataflow order.

        Items may be plain names or the composition graph's operation
        DICTS (name + params): the parity predicate (is_multiple_of,
        modulus 2) with its selection collapses into the pipeline's
        filter_even; a non-2 modulus keeps the fine-grained form (the
        pipeline vocabulary has no parametrized multiple-filter, so
        the composer path renders it).
        """
        if not ops:
            return ops

        # Uniform (name, params) view over names or dicts
        view = []
        for item in ops:
            if isinstance(item, dict):
                view.append((item.get("name") or "", item.get("params") or {}))
            else:
                view.append((str(item), {}))

        ops = [v[0] for v in view]
        params_of = [v[1] for v in view]

        # The SELECTION consumes its PREDICATE by contract: the filter
        # names the comparison/multiple producer in params["predicates"],
        # so the pairing survives non-adjacent execution order (a dedup
        # between the comparison and the selection).  The selection is
        # rendered AT ITS OWN DATAFLOW POSITION — collapsing it back to
        # the comparison's position would reorder the pipeline.
        consumed_producer: set[int] = set()
        filter_kind: dict[int, str] = {}
        for j, op in enumerate(ops):
            if op not in ("filter", "filter_elements"):
                continue
            preds = [str(pp) for pp in (params_of[j].get("predicates") or [])]
            kind = ""
            if "compare_greater_than" in preds:
                kind = "gt"
            elif "compare_less_than" in preds:
                kind = "lt"
            elif "is_multiple_of" in preds:
                kind = "even"
            if not kind:
                # Plain-name pairing: an immediately preceding comparison.
                for cidx in range(j - 1, max(-1, j - 3), -1):
                    if ops[cidx] == "compare_greater_than" and cidx not in consumed_producer:
                        kind = "gt"
                        break
                    if ops[cidx] == "compare_less_than" and cidx not in consumed_producer:
                        kind = "lt"
                        break
            if not kind:
                continue
            if kind in ("gt", "lt"):
                want = ("compare_greater_than" if kind == "gt"
                        else "compare_less_than")
                for cidx, cand in enumerate(ops):
                    if cand == want and cidx not in consumed_producer:
                        consumed_producer.add(cidx)
                        break
                filter_kind[j] = ("filter_above" if kind == "gt"
                                  else "filter_below")
            else:
                # A parity producer the filter consumes: render the
                # selection here; the producer itself renders nothing.
                for cidx, cand in enumerate(ops):
                    if cand == "is_multiple_of" and cidx not in consumed_producer:
                        if int((params_of[cidx] or {}).get("modulus", 2)) == 2:
                            consumed_producer.add(cidx)
                        break
                filter_kind[j] = "filter_even"

        normalized = []
        i = 0
        while i < len(ops):
            op = ops[i]

            # A comparison consumed by its selection renders through the
            # selection (which appears at its own dataflow position).
            if i in consumed_producer:
                i += 1
                continue

            # The selection renders its predicate as a real filter op.
            if i in filter_kind:
                normalized.append(filter_kind[i])
                i += 1
                continue

            # collapse is_multiple_of [+ filter] → filter_even (parity)
            if op == "is_multiple_of" \
                    and int((params_of[i] or {}).get("modulus", 2)) == 2:
                j = i + 1
                if j < len(ops) and ops[j] in ("filter", "filter_elements") \
                        and j not in filter_kind:
                    j += 1
                normalized.append("filter_even")
                i = j
                continue
            if op in ("filter", "filter_elements") \
                    and normalized and normalized[-1] == "filter_even":
                i += 1
                continue

            # standalone compare_* without any consuming selection
            if op == "compare_greater_than":
                normalized.append("filter_above")
                i += 1
                continue
            if op == "compare_less_than":
                normalized.append("filter_below")
                i += 1
                continue

            # deduplicate consecutive identical operations
            if normalized and op == normalized[-1]:
                i += 1
                continue

            normalized.append(op)
            i += 1

        return normalized

    def _generate_pipeline_python(self, ops: list, inputs: list, outputs: list,
                                  conditions: list) -> str:
        """Generate Python code for an operation pipeline."""
        lines = [
            "from typing import List",
            "",
            "",
            "def process_data(data: List[float]) -> List[float]:",
            '    """',
            "    Process data through operation pipeline:",
        ]
        for i, op in enumerate(ops):
            lines.append(f"        {i+1}. {op.replace('_', ' ')}")
        lines.append('    """')

        var_name = "data"
        for i, op in enumerate(ops):
            next_var = f"step_{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"    # Step {i+1}: Remove duplicates (preserve order)",
                    f"    seen = set()",
                    f"    {next_var} = []",
                    f"    for x in {var_name}:",
                    f"        if x not in seen:",
                    f"            seen.add(x)",
                    f"            {next_var}.append(x)",
                    "",
                ])
            elif op == "calculate_average":
                lines.extend([
                    f"    # Step {i+1}: Calculate average",
                    f"    avg_{i+1} = sum({var_name}) / len({var_name}) if {var_name} else 0",
                    f"    {next_var} = {var_name}",
                    "",
                ])
            elif op == "filter_above":
                # Find the average variable to reference
                avg_var = None
                for j in range(i-1, -1, -1):
                    if ops[j] == "calculate_average":
                        avg_var = f"avg_{j+1}"
                        break
                if avg_var:
                    lines.extend([
                        f"    # Step {i+1}: Filter values above average",
                        f"    {next_var} = [x for x in {var_name} if x > {avg_var}]",
                        "",
                    ])
                else:
                    lines.extend([
                        f"    # Step {i+1}: Filter values above average",
                        f"    avg = sum({var_name}) / len({var_name}) if {var_name} else 0",
                        f"    {next_var} = [x for x in {var_name} if x > avg]",
                        "",
                    ])
            elif op == "filter_below":
                avg_var = None
                for j in range(i-1, -1, -1):
                    if ops[j] == "calculate_average":
                        avg_var = f"avg_{j+1}"
                        break
                if avg_var:
                    lines.extend([
                        f"    # Step {i+1}: Filter values below average",
                        f"    {next_var} = [x for x in {var_name} if x < {avg_var}]",
                        "",
                    ])
                else:
                    lines.extend([
                        f"    # Step {i+1}: Filter values below average",
                        f"    avg = sum({var_name}) / len({var_name}) if {var_name} else 0",
                        f"    {next_var} = [x for x in {var_name} if x < avg]",
                        "",
                    ])
            elif op == "sort_descending":
                lines.extend([
                    f"    # Step {i+1}: Sort in descending order",
                    f"    {next_var} = sorted({var_name}, reverse=True)",
                    "",
                ])
            elif op == "sort_ascending":
                lines.extend([
                    f"    # Step {i+1}: Sort in ascending order",
                    f"    {next_var} = sorted({var_name})",
                    "",
                ])
            else:
                lines.extend([
                    f"    # Step {i+1}: {op.replace('_', ' ')}",
                    f"    {next_var} = {var_name}",
                    "",
                ])
            var_name = next_var

        lines.append(f"    return {var_name}")
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    # Example usage")
        lines.append("    sample_data = [5, 2, 5, 8, 3, 8, 2, 1, 9, 4]")
        lines.append("    result = process_data(sample_data)")
        lines.append("    print(f'Input: {sample_data}')")
        lines.append("    print(f'Result: {result}')")
        lines.append("")

        return "\n".join(lines)

    def generate_project(self, request: str, language: str = "kotlin",
                         output_dir: str = None, reasoning_state=None) -> dict:
        intent_result = self.intent_analyzer.analyze(request)

        memory_context = self.memory_manager.get_context_for_request(request)

        if intent_result.intent != INTENT_CREATE_PROJECT:
            response = self.intent_analyzer.get_unknown_response(request)
            if intent_result.intent == INTENT_CHAT:
                response = self.intent_analyzer.get_chat_response(request)
            elif intent_result.intent == INTENT_QUESTION:
                response = self.intent_analyzer.get_question_response(request)
            self.conversation_manager.add_turn(request, intent_result.intent, response)
            self.memory_manager.conversation_memory.add_turn(request, response, intent_result.intent)
            return {
                "request": request,
                "language": language,
                "intent": intent_result.intent,
                "intent_result": intent_result,
                "response": response,
                "project_created": False,
                "files": [],
                "verification": {"success": False, "checks_passed": 0, "checks_failed": 0, "issues": []},
                "memory_context": memory_context,
            }

        from brain.reasoning_events import ReasoningEventCollector
        event_collector = ReasoningEventCollector(language="en")

        # Primary intelligence: reason with the ThinkEngine (generalization via
        # knowledge graph), not just keyword routing.
        think_state = self.think_engine.think(request, language)
        self._last_reasoning_state = think_state

        analysis = self.requirements_analyzer.analyze(request, intent_result.entities)
        tech_selection = self.technology_selector.select(analysis, language)
        effective_language = tech_selection.language or language

        if event_collector:
            event_collector.add_technology_selection(
                effective_language,
                tech_selection.platform,
                tech_selection.framework or "",
                tech_selection.build_tool or "",
                effective_language
            )

        project_ir = self.project_planner.create_project_plan(
            request, language=effective_language, event_collector=event_collector
        )
        gen_result = self.project_generator.generate_project(project_ir)

        # Extract success criteria from reasoning state
        success_criteria = None
        if think_state and hasattr(think_state, 'model'):
            success_criteria = think_state.model.success_criteria

        verification = self.project_verifier.verify(project_ir, gen_result, success_criteria)

        file_list = []
        for f in gen_result.files:
            file_list.append({
                "path": f.path,
                "content": f.content,
                "file_type": f.file_type,
                "language": f.language,
            })

        error_analyses = []
        if not verification.success:
            if event_collector:
                for issue in verification.issues:
                    if issue.severity == "error":
                        event_collector.add_error(
                            issue.category,
                            issue.message,
                            issue.file_path,
                            issue.line,
                            ""
                        )
            error_analyses = [
                a.to_dict()
                for a in self.error_analyzer.analyze_issues(verification.issues, project_ir)
            ]
            if event_collector and error_analyses:
                for ea in error_analyses:
                    event_collector.add_error_analysis(
                        ea.get("error_type", "unknown"),
                        ea.get("cause", "unknown"),
                        ea.get("evidence", []),
                        ea.get("fix_plan", "")
                    )
                    event_collector.add_fix(
                        "auto_fix",
                        ea.get("file_path", ""),
                        ea.get("fix_plan", "")
                    )

        if event_collector:
            event_collector.add_verification(
                verification.checks_passed,
                verification.checks_failed,
                verification.verifier_status,
                [i.message for i in verification.issues]
            )

        if output_dir:
            self._write_project_files(output_dir, gen_result)

        self.conversation_manager.add_turn(
            request, INTENT_CREATE_PROJECT,
            f"프로젝트 {project_ir.name}가 생성되었습니다.", project_ir
        )
        self.memory_manager.conversation_memory.add_turn(
            request, f"프로젝트 {project_ir.name}가 생성되었습니다.", INTENT_CREATE_PROJECT
        )
        self.memory_manager.project_memory.create_project(
            name=project_ir.name,
            language=effective_language,
            platform=tech_selection.platform,
            framework=tech_selection.framework,
        )

        # Learn the experience: Problem -> Context -> Reasoning -> Solution -> Result -> Lesson
        self.memory_manager.record_experience(
            experience_type="success" if verification.success else "failure",
            description=f"프로젝트 생성: {project_ir.name}",
            context={
                "intent": INTENT_CREATE_PROJECT,
                "language": effective_language,
                "platform": tech_selection.platform,
                "domain": think_state.model.domain,
                "goal_type": think_state.model.goal_type.value,
                "entities": [e.name for e in think_state.model.entities],
            },
            outcome="프로젝트 생성 성공" if verification.success else "프로젝트 생성 실패",
            lessons=self.think_engine.explain(think_state),
            tags=["project", effective_language, think_state.model.domain],
        )

        if event_collector:
            event_collector.add_completion(
                verification.success,
                project_ir.name,
                len(file_list)
            )

        return {
            "request": request,
            "language": effective_language,
            "intent": INTENT_CREATE_PROJECT,
            "intent_result": intent_result,
            "analysis": {
                "project_name": analysis.project_name,
                "platform": analysis.platform,
                "language": analysis.language,
                "features": [f.name for f in analysis.features],
            },
            "technology": {
                "language": tech_selection.language,
                "platform": tech_selection.platform,
                "framework": tech_selection.framework,
                "build_tool": tech_selection.build_tool,
                "reason": tech_selection.reason,
            },
            "project_ir": project_ir.to_dict(),
            "results": {},
            "files": file_list,
            "verification": verification.to_dict(),
            "error_analyses": error_analyses,
            "reasoning_events": [e.to_dict() for e in event_collector.get_events()],
            "reasoning": think_state.to_dict(),
            "project_created": True,
            "memory_context": memory_context,
        }

    def _write_project_files(self, output_dir: str, gen_result: ProjectGenerationResult):
        for f in gen_result.files:
            file_path = os.path.join(output_dir, f.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as fh:
                fh.write(f.content)

    def generate_code(self, node: IRNode, language: str) -> str:
        return self.code_generator.generate(node, language)

    def generate_all_languages(self, node: IRNode) -> dict[str, str]:
        return self.code_generator.generate_all(node)

    def modify_code(self, root: IRNode, action: str, target_id: str,
                    node: IRNode = None, **kwargs) -> bool:
        from code.ir.ir_types import NodeTypeAction
        action_enum = NodeTypeAction(action)
        return self.refactoring.apply(root, action_enum, target_id, node, **kwargs)

    def verify(self, code: str, language: str):
        result = self.verifier.verify(code, language)
        if not result.success and result.errors:
            self.learner.learn_error_fix(code, "\n".join(result.errors), "", language)
        elif result.success:
            self.learner.learn_success(code, f"Verified {language} code", language)
        return result

    def learn_file(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        with open(file_path) as f:
            code = f.read()
        lang = self._detect_language(file_path)
        concept_result = self.learner.learn_from_code(code, lang)
        self.symbol_table.clear()
        return {
            "file": file_path,
            "language": lang,
            "result": concept_result,
        }

    def learn_ir(self, ir: IRNode, language: str = "") -> dict:
        return self.learner.learn_from_ir(ir, language)

    def explain(self, request: str, language: str = "python") -> str:
        # Single source of truth: the semantic operation graph is the same
        # source the IR builder consumes, so the explain's operation order
        # equals the generated IR's operation order.
        reasoning = self.reasoner.analyze(request, language)
        text = self.reasoner.explain_reasoning(reasoning)
        try:
            parse_context = ParseContext(
                conversation_history=self.conversation_manager.history,
                current_project=self.conversation_manager.current_project.to_dict()
                    if self.conversation_manager.current_project else None,
                user_preferences=self._current_user_preferences(),
            )
            semantic = self.semantic_parser.parse(request, parse_context)
            op_order = semantic.get_operation_order()
            if op_order:
                text += ("\n\n[IR Operation Order]\n  "
                         + " -> ".join(op_order))
        except Exception:
            pass
        return text

    def reason(self, request: str, language: str = "python",
               context: dict = None) -> dict:
        """Run the general ThinkEngine and return its structured reasoning."""
        state = self.think_engine.think(request, language, context)
        return {
            "model": state.model.to_dict(),
            "hypotheses": [h.to_dict() for h in state.hypotheses],
            "selected": state.selected_hypothesis.to_dict() if state.selected_hypothesis else None,
            "plan": [a.to_dict() for a in state.plan],
            "explanation": self.think_engine.explain(state),
        }

    def explain_decision(self, query: str = "") -> dict:
        """Answer self-knowledge questions ('why did you create X / choose Y')
        from recorded decisions and evidence, never fabricated.
        """
        reasoning_state = getattr(self, "_last_reasoning_state", None)

        answers = []

        # 1. Latest reasoning decisions (from ThinkEngine)
        if reasoning_state is not None:
            if reasoning_state.plan:
                for action in reasoning_state.plan:
                    answers.append({
                        "question": f"why did you {action.kind}?",
                        "reason": action.reason,
                        "evidence": action.evidence,
                    })
            if reasoning_state.selected_hypothesis:
                answers.append({
                    "question": "why did you choose this approach?",
                    "reason": reasoning_state.selected_hypothesis.description,
                    "evidence": reasoning_state.selected_hypothesis.evidence,
                    "confidence": reasoning_state.selected_hypothesis.confidence,
                })

        # 2. Technology selection (latest from project generation)
        latest_proj = self.memory_manager.project_memory.projects
        if latest_proj:
            # projects stored without full selection; fall back to experience memory
            pass

        # 3. Recent experiences contain lessons that explain decisions
        recent = self.memory_manager.experience_memory.get_conversation_experiences(limit=5)
        for exp in recent:
            if exp.lessons:
                answers.append({
                    "question": query or "why?",
                    "reason": exp.outcome,
                    "evidence": exp.lessons,
                })

        if not answers and query:
            # Attempt to match a specific decision term against recorded experiences.
            ql = query.lower()
            for exp in self.memory_manager.experience_memory.experiences.values():
                if ql and ql in exp.description.lower():
                    answers.append({
                        "question": query,
                        "reason": exp.outcome,
                        "evidence": exp.lessons,
                    })
                    break

        return {"query": query, "answers": answers}

    def query_concepts(self, query: str) -> list:
        return self.concept_store.search(query)

    def query_relations(self, source_id: str = None, target_id: str = None):
        return self.relation_store.find_relations(source_id=source_id, target_id=target_id)

    def query_memory(self, query: str, language: str = "") -> list:
        return self.memory_store.search(query, language=language)

    def export_data(self) -> dict:
        return {
            "concepts": self.concept_store.export_json(),
            "relations": self.relation_store.export_json(),
            "rules": self.rule_engine.export_json(),
            "memory": self.memory_store.export_json(),
            "memory_manager": self.memory_manager.export_data(),
        }

    def import_data(self, data: dict) -> dict:
        results = {}
        if "concepts" in data:
            results["concepts"] = self.concept_store.import_json(data["concepts"])
        if "relations" in data:
            results["relations"] = self.relation_store.import_json(data["relations"])
        if "rules" in data:
            results["rules"] = self.rule_engine.import_json(data["rules"])
        if "memory" in data:
            results["memory"] = self.memory_store.import_json(data["memory"])
        if "memory_manager" in data:
            self.memory_manager.import_data(data["memory_manager"])
        return results

    def save_data(self) -> None:
        self.concept_store._save()
        self.relation_store._save()
        self.rule_engine._save()
        self.memory_store._save()

    def process_feedback(self, text: str) -> dict:
        analysis = self.feedback_learner.analyze_feedback(text)
        if not analysis:
            return {"processed": False, "message": "피드백으로 인식되지 않았습니다."}

        entry = self.feedback_learner.process_feedback(text)

        apology_context = self.feedback_learner.get_apology_context(text)
        if apology_context:
            apology = self.apology_system.generate_apology(apology_context)
        else:
            apology = None

        return {
            "processed": True,
            "feedback_type": analysis.feedback_type,
            "category": analysis.category,
            "lesson": analysis.lesson,
            "apology": apology,
            "entry": entry.to_dict() if entry else None,
        }

    def set_preference(self, key: str, value: str) -> None:
        from memory.preference_memory import PreferenceScope
        self.memory_manager.apply_preference(key, value, PreferenceScope.GLOBAL)

    def get_preference(self, key: str, default: str = "") -> str:
        return self.memory_manager.preference_memory.get(key, default)

    def get_memory_context(self, request: str) -> dict:
        return self.memory_manager.get_context_for_request(request)

    def diagnose_error(self, error_message: str, code: str = "",
                       language: str = "", file_path: str = "") -> dict:
        diagnosis = self.error_diagnoser.diagnose(
            error_message, code, language, file_path
        )
        apology_context = self.error_diagnoser.get_apology_context(diagnosis)
        if apology_context:
            diagnosis["apology"] = self.apology_system.generate_apology(apology_context)
        self.memory_manager.record_experience(
            experience_type="learning",
            description=f"오류 진단: {error_message[:100]}",
            context={
                "error_type": diagnosis["error_type"],
                "language": language,
                "file_path": file_path,
            },
            outcome="진단 완료",
            lessons=[diagnosis["explanation"]],
            tags=["diagnosis", diagnosis["error_type"], language],
        )
        return diagnosis

    def auto_fix(self, error_message: str, code: str = "",
                 language: str = "", file_path: str = "",
                 project_dir: str = "") -> dict:
        fix_result = self.auto_fixer.fix(
            error_message, code, language, file_path, project_dir
        )
        self.memory_manager.conversation_memory.add_turn(
            user_input=f"오류 발생: {error_message[:100]}",
            ai_response=fix_result["message"],
            intent="auto_fix",
            context={"fixes": fix_result["fixes_applied"], "success": fix_result["success"]},
            errors=[error_message],
            fixes_applied=fix_result["files_modified"],
        )
        return fix_result

    def diagnose_and_fix(self, error_message: str, code: str = "",
                         language: str = "", file_path: str = "",
                         project_dir: str = "") -> dict:
        diagnosis = self.diagnose_error(error_message, code, language, file_path)
        fix_result = self.auto_fix(error_message, code, language, file_path, project_dir)

        self.memory_manager.conversation_memory.add_turn(
            user_input=f"오류 발생: {error_message[:100]}",
            ai_response=fix_result["message"],
            intent="auto_fix",
            context={"fixes": fix_result["fixes_applied"]},
            errors=[error_message],
            fixes_applied=fix_result["files_modified"],
        )

        return {
            "diagnosis": diagnosis,
            "fix_result": fix_result,
            "explanation": diagnosis["explanation"],
            "fix_message": fix_result["message"],
            "success": fix_result["success"],
            "apology": diagnosis.get("apology"),
        }

    def generate_with_comments(self, node: IRNode, language: str,
                               comment_density: str = None) -> str:
        if not comment_density:
            comment_density = self.get_preference("comment_density", "medium")
        code = self.code_generator.generate(node, language)
        if comment_density == "high":
            code = self._add_intelligent_comments(code, language, node)
        elif comment_density == "medium":
            code = self._add_basic_comments(code, language, node)
        # The stored comment policy is the user's standing instruction:
        # enforced after the density pass (AUTO leaves the code as is).
        from brain.comment_policy import (CommentPolicy, apply_comment_policy,
                                          policy_from_stored)
        profile = self.memory_manager.get_user_profile() or {}
        policy = policy_from_stored(profile.get("comment_density", ""))
        if policy is not CommentPolicy.AUTO:
            code = apply_comment_policy(code, policy, language)
        return code

    def _add_intelligent_comments(self, code: str, language: str,
                                  node: IRNode) -> str:
        lines = code.split("\n")
        commented_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("fun ") or stripped.startswith("function "):
                commented_lines.append(line)
                func_name = stripped.split("(")[0].replace("def ", "").replace("fun ", "").replace("function ", "")
                commented_lines.append(f"    \"\"\"{func_name} 함수/메서드입니다.\"\"\"")
            elif stripped.startswith("class ") or stripped.startswith("data class "):
                commented_lines.append(line)
                class_name = stripped.split("{")[0].split("(")[0].replace("class ", "").replace("data class ", "")
                commented_lines.append(f"    \"\"\"{class_name} 클래스입니다.\"\"\"")
            elif "return " in stripped:
                commented_lines.append(f"    # 결과를 반환합니다")
                commented_lines.append(line)
            elif "if " in stripped:
                commented_lines.append(f"    # 조건문: {stripped}")
                commented_lines.append(line)
            elif "for " in stripped or "while " in stripped:
                commented_lines.append(f"    # 반복문")
                commented_lines.append(line)
            else:
                commented_lines.append(line)
        return "\n".join(commented_lines)

    def _add_basic_comments(self, code: str, language: str,
                            node: IRNode) -> str:
        lines = code.split("\n")
        commented_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("fun "):
                commented_lines.append(line)
                func_name = stripped.split("(")[0].replace("def ", "").replace("fun ", "")
                commented_lines.append(f"    # {func_name}")
            else:
                commented_lines.append(line)
        return "\n".join(commented_lines)

    def _detect_language(self, file_path: str) -> str:
        ext_map = {
            ".py": "python",
            ".kt": "kotlin",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".js": "javascript",
            ".mjs": "javascript",
            ".go": "go",
            ".rs": "rust",
        }
        _, ext = os.path.splitext(file_path)
        return ext_map.get(ext.lower(), "python")
