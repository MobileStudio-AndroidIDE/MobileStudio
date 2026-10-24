"""Response Planner - Plans semantic response structures from meaning.

This module converts SemanticRepresentation + ReasoningResult into
ResponseRepresentation, which is then realized into natural language.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re

from brain.semantic_representation import (
    ResponseRepresentation, SemanticRepresentation, Proposition, Predication,
    Argument, Modifier, DiscourseTree, DiscourseNode, RhetoricalRelation,
    FocusInfo, ToneSpec, VerbosityLevel, AudienceModel, EvidenceRef,
    SpeechAct, Language, ToneType, ActivatedConcept, GoalType
)
from brain.think_engine import ReasoningState, Hypothesis
from brain.concept_store import ConceptStore, Concept
from brain.relation_store import RelationStore
from brain.memory_store import MemoryStore
from brain.capability_registry import CapabilityRegistry


@dataclass
class PlanningContext:
    """Context for response planning."""
    semantic: SemanticRepresentation
    reasoning: ReasoningState | None = None
    conversation_history: list[dict] = field(default_factory=list)
    current_project: dict | None = None
    user_model: dict = field(default_factory=dict)
    # The generation's actual output: what a code-generation response
    # is ABOUT (function name, operation pipeline, verification).
    artifact: dict | None = None


class ResponsePlanner:
    """Plans ResponseRepresentation from semantic understanding + reasoning."""
    
    def __init__(
        self,
        concept_store: ConceptStore,
        relation_store: RelationStore,
        memory_store: MemoryStore,
        capability_registry: CapabilityRegistry,
        self_knowledge: "SelfKnowledgeInterface"
    ):
        self.concepts = concept_store
        self.relations = relation_store
        self.memory = memory_store
        self.capabilities = capability_registry
        self.self_knowledge = self_knowledge
    
    def plan(
        self,
        semantic: SemanticRepresentation,
        reasoning: ReasoningState | None = None,
        conversation_history: list[dict] = None,
        current_project: dict = None,
        user_model: dict = None,
        artifact: dict = None
    ) -> ResponseRepresentation:
        """Plan a response from semantic understanding."""
        
        ctx = PlanningContext(
            semantic=semantic,
            reasoning=reasoning,
            conversation_history=conversation_history or [],
            current_project=current_project,
            user_model=user_model or {},
            artifact=artifact,
        )
        
        # 1. Determine speech act
        speech_act = self._infer_speech_act(ctx)
        
        # 2. Build propositional content
        content = self._build_content(ctx)
        
        # 3. Plan discourse structure
        discourse = self._plan_discourse(ctx, content)
        
        # 4. Determine focus
        focus = self._determine_focus(ctx, content)
        
        # 5. Determine tone/style
        tone = self._determine_tone(ctx)
        
        # 6. Determine verbosity
        verbosity = self._determine_verbosity(ctx)
        
        # 7. Build audience model
        audience = self._build_audience_model(ctx)
        
        # 8. Collect evidence
        evidence = self._collect_evidence(ctx)
        
        # 9. Determine certainty
        certainty = self._calculate_certainty(ctx)
        
        return ResponseRepresentation(
            speech_act=speech_act,
            propositional_content=content,
            discourse_structure=discourse,
            focus=focus,
            tone=tone,
            verbosity=verbosity,
            audience=audience,
            evidence=evidence,
            certainty=certainty,
            target_language=semantic.source_language,
            source_semantic=semantic
        )
    
    def _infer_speech_act(self, ctx: PlanningContext) -> SpeechAct:
        """Infer speech act from semantic goal and context."""
        goal = ctx.semantic.goal
        
        # Direct mapping from goal type
        goal_to_act = {
            "create_function": SpeechAct.INFORM,
            "create_class": SpeechAct.INFORM,
            "create_project": SpeechAct.INFORM,
            "explain": SpeechAct.EXPLAIN,
            "fix": SpeechAct.INFORM,
            "analyze": SpeechAct.INFORM,
            "question": SpeechAct.INFORM,
            "chat": SpeechAct.GREET,
        }
        
        if goal.goal_type.value in goal_to_act:
            return goal_to_act[goal.goal_type.value]
        
        # Context-based overrides
        if ctx.conversation_history:
            last_user = ctx.conversation_history[-1]
            user_input = getattr(last_user, 'user_input', '') if hasattr(last_user, 'user_input') else (last_user.get("user_input", "") if hasattr(last_user, 'get') else "")
            user_input = user_input.lower()
            if any(w in user_input for w in ["고마워", "감사", "thank", "thanks"]):
                return SpeechAct.INFORM  # acknowledgement
            if any(w in user_input for w in ["미안", "sorry", "죄송"]):
                return SpeechAct.INFORM  # acceptance
        
        return SpeechAct.INFORM
    
    def _build_content(self, ctx: PlanningContext) -> Predication:
        """Build the core propositional content."""
        semantic = ctx.semantic
        goal = semantic.goal
        reasoning = ctx.reasoning
        
        # For CHAT/GREETING, create a proper greeting predication
        if goal.goal_type == GoalType.CHAT:
            # Get name from self_knowledge
            ai_name = "RENA-BR"
            if self.self_knowledge:
                name = self.self_knowledge.get("identity", "name")
                if name:
                    ai_name = name
            
            # Determine greeting type from raw text
            text_lower = semantic.raw_text.lower()
            if any(w in text_lower for w in ["안녕", "hi", "hello", "hey"]):
                return Predication(
                    predicate="greet",
                    arguments=[
                        Argument(role="agent", value="I"),
                        Argument(role="theme", value=ai_name),
                        Argument(role="manner", value="polite")
                    ],
                    confidence=goal.confidence
                )
            elif any(w in text_lower for w in ["고마워", "감사", "thank", "thanks"]):
                return Predication(
                    predicate="thank",
                    arguments=[
                        Argument(role="agent", value="I"),
                        Argument(role="theme", value="gratitude")
                    ],
                    confidence=goal.confidence
                )
            elif any(w in text_lower for w in ["잘 가", "bye", "goodbye"]):
                return Predication(
                    predicate="farewell",
                    arguments=[
                        Argument(role="agent", value="I"),
                        Argument(role="theme", value=ai_name)
                    ],
                    confidence=goal.confidence
                )
            elif any(w in text_lower for w in ["자기소개", "누구야", "who are you", "너 누구"]):
                return Predication(
                    predicate="introduce_self",
                    arguments=[
                        Argument(role="agent", value="I"),
                        Argument(role="theme", value=ai_name),
                        Argument(role="detail", value="AI assistant for code generation")
                    ],
                    confidence=goal.confidence
                )
        
        # Base predicate from goal type
        predicate_map = {
            GoalType.CREATE_FUNCTION: "generate_code",
            GoalType.CREATE_CLASS: "generate_code",
            GoalType.CREATE_PROJECT: "generate_project",
            GoalType.EXPLAIN: "explain",
            GoalType.FIX: "fix_code",
            GoalType.ANALYZE: "analyze_code",
            GoalType.QUESTION: "answer_question",
            # The modification turn: the artifact it continued was
            # regenerated — the response says so (never a bare
            # predicate with no theme, and never a fresh "generate").
            GoalType.MODIFY: "modify_code",
            GoalType.FIX: "fix_code",
            GoalType.ANALYZE: "analyze_code",
        }
        
        predicate = predicate_map.get(goal.goal_type, "inform")
        
        # Build arguments
        arguments = []
        modifiers = []

        # Agent (self)
        arguments.append(Argument(role="agent", value="I"))

        # The target echoed back from the raw request is noise (the raw
        # request restated as the theme) — skip it; the generated code is
        # the response, not a restatement of the request.
        raw = (semantic.raw_text or "").strip()
        target_echo = bool(goal.target) and (
            goal.target.strip() == raw
            or (len(goal.target.strip()) > 12 and goal.target.strip() in raw))

        # Action/Theme
        if goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_CLASS,
                              GoalType.CREATE_PROJECT, GoalType.MODIFY):
            # The ARTIFACT the generation actually produced is the
            # theme — the function that exists, its operation pipeline.
            # A code-generation response is ABOUT that, never a bare
            # "I generate" with the theme missing.
            artifact = ctx.artifact or {}
            art_name = str(artifact.get("name") or "").strip()
            art_ops = str(artifact.get("operations_text") or "").strip()
            if art_name:
                # A function artifact is callable: the theme is its
                # signature name, not a bare label.
                theme_name = (f"{art_name}()"
                              if goal.goal_type in (GoalType.CREATE_FUNCTION,
                                                    GoalType.MODIFY)
                              else art_name)
                arguments.append(Argument(role="theme", value=theme_name))
                if art_ops:
                    arguments.append(Argument(role="detail", value=art_ops))
                modifiers.append(Modifier(type="tense", value="past"))
            elif not target_echo and goal.target:
                arguments.append(Argument(role="theme", value=goal.target))
        elif goal.goal_type == GoalType.EXPLAIN:
            arguments.append(Argument(role="theme", value=goal.target or "the topic"))
        elif goal.goal_type == GoalType.FIX:
            arguments.append(Argument(role="theme", value="the issue"))
        elif goal.goal_type == GoalType.QUESTION:
            arguments.append(Argument(role="theme", value=goal.target or "the question"))
        elif goal.goal_type == GoalType.ANALYZE:
            arguments.append(Argument(role="theme", value="the code"))
        
        # Add capability/approach if reasoning available
        approach = None
        if reasoning and reasoning.selected_hypothesis:
            approach = reasoning.selected_hypothesis.description
            # Only add approach if it contains Korean or is meaningful
            if approach and re.search(r'[\uac00-\ud7af]', approach):
                arguments.append(Argument(role="instrument", value=approach))
            # Skip raw English hypothesis descriptions
        
        # Add constraints — in the RESPONSE's language (a Korean
        # constraint clause inside an English response is register
        # mixing, not localization)
        constraints = []
        response_lang = getattr(semantic, "source_language", None)
        lang_kr_mode = response_lang is not None and "ko" in str(
            getattr(response_lang, "value", response_lang))
        for c in semantic.constraints:
            if c.constraint_type == "language":
                lang_map_kr = {"python": "파이썬", "go": "Go", "javascript": "자바스크립트",
                               "java": "자바", "c": "C", "cpp": "C++", "rust": "러스트",
                               "kotlin": "코틀린", "typescript": "타입스크립트"}
                if lang_kr_mode:
                    lang_name = lang_map_kr.get(c.value.lower(), c.value)
                    constraints.append(f"{lang_name} 언어로")
                else:
                    # The realizer composes "with <condition>" — the
                    # constraint carries the value, not the preposition.
                    constraints.append(c.value)
            elif c.constraint_type == "platform":
                constraints.append(f"{c.value} 플랫폼에서")
        if constraints:
            arguments.append(Argument(role="condition", value=", ".join(constraints)))
        
        return Predication(
            predicate=predicate,
            arguments=arguments,
            modifiers=modifiers,
            confidence=goal.confidence
        )
    
    def _plan_discourse(self, ctx: PlanningContext, content: Predication) -> DiscourseTree:
        """Plan the discourse structure (rhetorical relations)."""
        semantic = ctx.semantic
        reasoning = ctx.reasoning
        
        # Build main content node
        main_prop = Proposition(
            predication=content,
            focus=self._extract_focus(ctx)
        )
        main_node = DiscourseNode(content=main_prop)
        
        # Build supporting nodes based on reasoning
        satellites = []
        
        if reasoning:
            # Evidence satellite — the hypothesis's OWN evidence text is
            # the theme (a placeholder theme like "approach" realizes as
            # "I provide evidence approach": a claim with no content).
            if reasoning.selected_hypothesis and reasoning.selected_hypothesis.evidence:
                # The "goal=..." evidence entry restates the theme the
                # main clause already carries — only the evidence that
                # ADDS something is worth saying.
                ev_text = "; ".join(
                    str(e) for e in reasoning.selected_hypothesis.evidence[:3]
                    if str(e).strip() and not str(e).startswith("goal="))
                if ev_text:
                    ev_prop = Proposition(
                        predication=Predication(
                            predicate="provide_evidence",
                            arguments=[
                                Argument(role="theme", value=ev_text),
                            ]
                        )
                    )
                    satellites.append(DiscourseNode(
                        relation=RhetoricalRelation.ELABORATION,
                        content=ev_prop
                    ))
            # Verification satellite — the artifact's verified state is a
            # public fact about the result, not hidden reasoning.
            artifact = ctx.artifact or {}
            if artifact.get("verified") is True:
                v_prop = Proposition(
                    predication=Predication(
                        predicate="verify_code",
                        arguments=[Argument(role="theme",
                                            value="the generated function")],
                        confidence=float(artifact.get("confidence") or 1.0),
                    )
                )
                satellites.append(DiscourseNode(
                    relation=RhetoricalRelation.SEQUENCE,
                    content=v_prop
                ))
            
            # Alternative approaches — only when the detail is realizable
            # in the response language (an English-only strategy count in
            # a Korean response is fixed-string noise).
            if len(reasoning.hypotheses) > 1:
                alt_detail = f"{len(reasoning.hypotheses)-1} other strategies considered"
                if re.search(r'[가-힣]', alt_detail):
                    alt_prop = Proposition(
                        predication=Predication(
                            predicate="mention_alternatives",
                            arguments=[
                                Argument(role="theme", value="other approaches"),
                                Argument(role="detail", value=alt_detail)
                            ]
                        )
                    )
                    satellites.append(DiscourseNode(
                        relation=RhetoricalRelation.ELABORATION,
                        content=alt_prop
                    ))
        
        # Capabilities mention for project/function creation
        if semantic.goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_PROJECT):
            active_caps = self.capabilities.get_active()
            if active_caps:
                cap_names = [c.name for c in active_caps[:3]]
                # Only add if capability names contain Korean
                if any(re.search(r'[\uac00-\ud7af]', name) for name in cap_names):
                    cap_prop = Proposition(
                        predication=Predication(
                            predicate="mention_capabilities",
                            arguments=[
                                Argument(role="theme", value="사용된 기능"),
                                Argument(role="detail", value=", ".join(cap_names))
                            ]
                        )
                    )
                    satellites.append(DiscourseNode(
                        relation=RhetoricalRelation.ELABORATION,
                        content=cap_prop
                    ))
        
        # Build tree
        root = DiscourseNode(
            relation=None,
            nucleus=main_node,
            satellites=satellites,
            is_multinuclear=False
        )
        
        return DiscourseTree(root=root)
    
    def _extract_focus(self, ctx: PlanningContext) -> str:
        """Extract what's new/important in this response."""
        semantic = ctx.semantic
        goal = semantic.goal
        
        if goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_PROJECT):
            return f"creating {goal.target}"
        elif goal.goal_type == GoalType.EXPLAIN:
            return f"explaining {goal.target}"
        elif goal.goal_type == GoalType.FIX:
            return "fixing the issue"
        elif goal.goal_type == GoalType.CHAT:
            return "greeting"
        return "responding"
    
    def _determine_focus(self, ctx: PlanningContext, content: Predication) -> FocusInfo:
        """Determine information focus (new vs given)."""
        semantic = ctx.semantic
        goal = semantic.goal
        
        new_info = []
        given_info = []
        
        # What's new: the specific action/result
        if goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_PROJECT):
            new_info.append(f"creating {goal.target}")
            new_info.append(f"operations: {semantic.get_operation_order()}")
        elif goal.goal_type == GoalType.EXPLAIN:
            new_info.append(f"explanation of {goal.target}")
        
        # What's given: context, language, previous topic
        if semantic.constraints:
            for c in semantic.constraints:
                if c.constraint_type == "language":
                    given_info.append(f"language: {c.value}")
                elif c.constraint_type == "platform":
                    given_info.append(f"platform: {c.value}")
        
        if ctx.conversation_history:
            last = ctx.conversation_history[-1]
            user_input = getattr(last, 'user_input', '') if hasattr(last, 'user_input') else (last.get("user_input", "") if hasattr(last, 'get') else "")
            given_info.append(f"previous: {user_input[:50]}")
        
        return FocusInfo(
            new_info=new_info,
            given_info=given_info
        )
    
    def _determine_tone(self, ctx: PlanningContext) -> ToneSpec:
        """Determine tone from context and user model."""
        user_model = ctx.user_model
        semantic = ctx.semantic
        
        # Base tone from goal type
        if semantic.goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_PROJECT):
            tone_type = ToneType.TECHNICAL
            formality = 0.6
            technicality = 0.8
        elif semantic.goal.goal_type == GoalType.EXPLAIN:
            tone_type = ToneType.FRIENDLY
            formality = 0.5
            technicality = 0.6
        elif semantic.goal.goal_type == GoalType.CHAT:
            tone_type = ToneType.FRIENDLY
            formality = 0.3
            technicality = 0.2
        else:
            tone_type = ToneType.FRIENDLY
            formality = 0.5
            technicality = 0.5
        
        # Adjust from user model
        if user_model.get("prefers_formal"):
            formality = min(formality + 0.2, 1.0)
        if user_model.get("prefers_casual"):
            formality = max(formality - 0.2, 0.0)
        if user_model.get("expert_level", 0) > 0.7:
            technicality = min(technicality + 0.2, 1.0)
        
        return ToneSpec(
            tone_type=ToneType.TECHNICAL if technicality > 0.6 else ToneType.FRIENDLY,
            formality=formality,
            technicality=technicality,
            warmth=0.6
        )
    
    def _determine_verbosity(self, ctx: PlanningContext) -> VerbosityLevel:
        """Determine response verbosity."""
        semantic = ctx.semantic
        user_model = ctx.user_model
        
        # Default from goal type
        if semantic.goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_PROJECT):
            base = VerbosityLevel.DETAILED
        elif semantic.goal.goal_type == GoalType.EXPLAIN:
            base = VerbosityLevel.DETAILED
        elif semantic.goal.goal_type == GoalType.CHAT:
            base = VerbosityLevel.BRIEF
        else:
            base = VerbosityLevel.NORMAL
        
        # User preference override
        if user_model.get("prefers_concise"):
            return VerbosityLevel.BRIEF
        if user_model.get("prefers_detailed"):
            return VerbosityLevel.DETAILED
        
        return base
    
    def _build_audience_model(self, ctx: PlanningContext) -> AudienceModel:
        """Build audience model from context."""
        user_model = ctx.user_model
        
        return AudienceModel(
            programming_level=user_model.get("programming_level", 0.5),
            domain_familiarity=user_model.get("domain_familiarity", 0.5),
            prefers_examples=user_model.get("prefers_examples", True),
            prefers_conciseness=user_model.get("prefers_conciseness", False)
        )
    
    def _collect_evidence(self, ctx: PlanningContext) -> list:
        """Collect evidence references."""
        evidence = []
        semantic = ctx.semantic
        reasoning = ctx.reasoning
        
        # Concept evidence
        for concept_id in semantic.activated_concept_ids[:5]:
            concept = self.concepts.get(concept_id)
            if concept:
                evidence.append(EvidenceRef(
                    type="concept",
                    source_id=concept_id,
                    description=f"Concept: {concept.name}",
                    strength=0.8
                ))
        
        # Reasoning evidence
        if reasoning and reasoning.selected_hypothesis:
            for concept_id in reasoning.selected_hypothesis.concepts[:3]:
                evidence.append(EvidenceRef(
                    type="reasoning",
                    source_id=concept_id,
                    description=f"Hypothesis: {reasoning.selected_hypothesis.description}",
                    strength=reasoning.selected_hypothesis.confidence
                ))
        
        # Memory evidence
        if ctx.conversation_history:
            evidence.append(EvidenceRef(
                type="memory",
                source_id="conversation",
                description=f"Conversation context ({len(ctx.conversation_history)} turns)",
                strength=0.6
            ))
        
        return evidence
    
    def _calculate_certainty(self, ctx: PlanningContext) -> float:
        """Calculate response certainty."""
        semantic = ctx.semantic
        reasoning = ctx.reasoning
        
        certainty = semantic.confidence * 0.5
        
        if reasoning and reasoning.selected_hypothesis:
            certainty += reasoning.selected_hypothesis.confidence * 0.3
        
        if ctx.current_project:
            certainty += 0.1
        
        if ctx.conversation_history:
            certainty += 0.1
        
        return min(certainty, 1.0)


class SelfKnowledgeInterface:
    """Interface for self-knowledge queries (to be implemented by actual SelfKnowledge)."""
    
    def get_identity(self) -> dict:
        raise NotImplementedError
    
    def get_capabilities(self) -> list[str]:
        raise NotImplementedError
    
    def get_architecture_description(self) -> str:
        raise NotImplementedError
    
    def get_supported_languages(self) -> list[str]:
        raise NotImplementedError


def create_response_planner(
    concept_store: ConceptStore,
    relation_store: RelationStore,
    memory_store: MemoryStore,
    capability_registry: CapabilityRegistry,
    self_knowledge: SelfKnowledgeInterface
) -> ResponsePlanner:
    """Factory function to create a ResponsePlanner."""
    return ResponsePlanner(
        concept_store, relation_store, memory_store,
        capability_registry, self_knowledge
    )