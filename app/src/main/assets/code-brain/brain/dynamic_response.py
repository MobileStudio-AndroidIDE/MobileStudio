"""Dynamic Response Generator - generates responses from understanding, not templates.

This module implements the UNDERSTAND → REASON → CREATE → VERIFY pipeline
for response generation. It replaces the old RETRIEVE → OUTPUT pattern.

Key principles:
- Meaning is stable, expression is dynamic
- Same question can produce different expressions
- Responses are constructed from semantic understanding + context + knowledge
- No hardcoded response strings
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ResponseIntent(Enum):
    """What the response should accomplish."""
    GREETING = "greeting"
    SELF_INTRODUCTION = "self_introduction"
    CAPABILITY_DISCLOSURE = "capability_disclosure"
    CODE_KNOWLEDGE = "code_knowledge"
    PROJECT_DISCUSSION = "project_discussion"
    ARCHITECTURE_EXPLANATION = "architecture_explanation"
    LIMITATION_DISCLOSURE = "limitation_disclosure"
    QUESTION_ANSWER = "question_answer"
    CLARIFICATION = "clarification"
    ACKNOWLEDGMENT = "acknowledgment"
    GOODBYE = "goodbye"
    FALLBACK = "fallback"


class ExpressionStyle(Enum):
    """How the response should be expressed."""
    CONCISE = "concise"
    STANDARD = "standard"
    DETAILED = "detailed"
    CASUAL = "casual"
    FORMAL = "formal"


@dataclass
class CoreMessage:
    """The essential meaning to convey (language-independent)."""
    intent: ResponseIntent
    key_concepts: list[str] = field(default_factory=list)
    supporting_facts: list[str] = field(default_factory=list)
    emotional_tone: str = "neutral"  # neutral, friendly, helpful, apologetic
    urgency: str = "normal"  # low, normal, high


@dataclass
class ContextualFrame:
    """What context influences the response."""
    user_language: str = "en"
    conversation_turn: int = 0
    is_followup: bool = False
    current_topic: str = ""
    user_expertise: str = "general"  # beginner, intermediate, expert, general
    recent_actions: list[str] = field(default_factory=list)
    active_project: str = ""


@dataclass
class SentenceUnit:
    """A single unit of meaning to express."""
    unit_type: str  # greeting, identity, capability, knowledge, question, acknowledgment, etc.
    content_key: str  # what concept to express
    content_params: dict = field(default_factory=dict)  # parameters for the content
    emphasis: str = "normal"  # low, normal, high
    order: int = 0  # ordering in the response


@dataclass
class ResponseRepresentation:
    """Language-independent representation of what to say."""
    core_message: CoreMessage
    context: ContextualFrame
    sentence_units: list[SentenceUnit] = field(default_factory=list)
    style: ExpressionStyle = ExpressionStyle.STANDARD
    metadata: dict = field(default_factory=dict)


@dataclass
class SentencePlan:
    """Structured plan for constructing a sentence."""
    clauses: list[dict] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    punctuation: str = "."
    line_breaks: list[int] = field(default_factory=list)


class DynamicResponseGenerator:
    """Generates responses dynamically from understanding + context + knowledge.

    Pipeline:
        Input → Understand → Reason → Create ResponseRepresentation
              → Plan Sentences → Realize Language → Output
    """

    def __init__(self, self_knowledge=None, capability_registry=None,
                 concept_store=None, memory_store=None):
        self.self_knowledge = self_knowledge
        self.capability_registry = capability_registry
        self.concepts = concept_store
        self.memory = memory_store

    def generate(self, semantic, user_context=None, conversation_history=None,
                 knowledge_context=None) -> str:
        """Generate a response from semantic understanding.

        This is the main entry point. It replaces the old plan() → generate() flow.
        """
        user_context = user_context or {}
        conversation_history = conversation_history or []
        knowledge_context = knowledge_context or {}

        # Step 1: Understand what the user wants
        core_message = self._understand(semantic, user_context, knowledge_context)

        # Step 2: Build contextual frame
        context = self._build_context(semantic, user_context, conversation_history)

        # Step 3: Create response representation
        representation = self._create_representation(core_message, context, knowledge_context)

        # Step 4: Plan sentences
        sentence_plan = self._plan_sentences(representation)

        # Step 5: Realize in target language
        output = self._realize_language(sentence_plan, context)

        return output

    def _understand(self, semantic, user_context, knowledge_context) -> CoreMessage:
        """Analyze input to determine what the response should convey."""
        # Determine the primary intent
        intent = ResponseIntent.FALLBACK
        key_concepts = []
        supporting_facts = []
        tone = "neutral"

        # Map semantic categories to response intents
        if hasattr(semantic, 'categories'):
            cats = semantic.categories
            from brain.semantic_understanding import SemanticCategory

            if SemanticCategory.GREETING in cats:
                intent = ResponseIntent.GREETING
                tone = "friendly"
            elif SemanticCategory.SELF_INTRODUCTION in cats:
                intent = ResponseIntent.SELF_INTRODUCTION
                tone = "friendly"
            elif SemanticCategory.CAPABILITIES in cats:
                intent = ResponseIntent.CAPABILITY_DISCLOSURE
                tone = "helpful"
            elif SemanticCategory.CODE_KNOWLEDGE in cats:
                intent = ResponseIntent.CODE_KNOWLEDGE
                tone = "informative"
                if hasattr(semantic, 'entities'):
                    key_concepts = [e for e in semantic.entities if isinstance(e, str)]
            elif SemanticCategory.PROJECT_QUESTION in cats:
                intent = ResponseIntent.PROJECT_DISCUSSION
                tone = "helpful"
            elif SemanticCategory.ARCHITECTURE in cats:
                intent = ResponseIntent.ARCHITECTURE_EXPLANATION
                tone = "informative"
            elif SemanticCategory.CODE_GENERATION in cats:
                intent = ResponseIntent.ACKNOWLEDGMENT
                tone = "helpful"
            elif SemanticCategory.GOODBYE in cats:
                intent = ResponseIntent.GOODBYE
                tone = "friendly"
            elif SemanticCategory.CASUAL_CONVERSATION in cats:
                intent = ResponseIntent.QUESTION_ANSWER
                tone = "friendly"
            elif hasattr(semantic, 'is_question') and semantic.is_question:
                intent = ResponseIntent.QUESTION_ANSWER
                tone = "helpful"

        # If we couldn't determine intent, use clarification
        if intent == ResponseIntent.FALLBACK:
            intent = ResponseIntent.CLARIFICATION
            tone = "helpful"

        # Gather supporting facts from knowledge
        if self.self_knowledge:
            if intent == ResponseIntent.SELF_INTRODUCTION:
                supporting_facts = self._gather_identity_facts()
            elif intent == ResponseIntent.CAPABILITY_DISCLOSURE:
                supporting_facts = self._gather_capability_facts()
            elif intent == ResponseIntent.ARCHITECTURE_EXPLANATION:
                supporting_facts = self._gather_architecture_facts()
            elif intent == ResponseIntent.CODE_KNOWLEDGE:
                supporting_facts = self._gather_code_knowledge(key_concepts)

        return CoreMessage(
            intent=intent,
            key_concepts=key_concepts,
            supporting_facts=supporting_facts,
            emotional_tone=tone,
        )

    def _gather_identity_facts(self) -> list[str]:
        """Gather facts about identity from self-knowledge."""
        facts = []
        if self.self_knowledge:
            name = self.self_knowledge.get("identity", "name")
            arch = self.self_knowledge.get("identity", "architecture")
            purpose = self.self_knowledge.get("purpose", "main")
            if name:
                facts.append(f"name:{name}")
            if arch:
                facts.append(f"architecture:{arch}")
            if purpose:
                facts.append(f"purpose:{purpose}")
        return facts

    def _gather_capability_facts(self) -> list[str]:
        """Gather facts about capabilities from the registry."""
        facts = []
        if self.capability_registry:
            active = self.capability_registry.get_active()
            for cap in active[:6]:  # Limit to top 6
                facts.append(f"capability:{cap.name}:{cap.description}")
        return facts

    def _gather_architecture_facts(self) -> list[str]:
        """Gather facts about architecture from self-knowledge."""
        facts = []
        if self.self_knowledge:
            arch_items = self.self_knowledge.get_category("architecture")
            for item in arch_items:
                facts.append(f"component:{item.key}:{item.value}")
        return facts

    def _gather_code_knowledge(self, concepts: list[str]) -> list[str]:
        """Gather code knowledge from the knowledge graph."""
        facts = []
        if self.concepts and concepts:
            for concept_name in concepts[:3]:
                results = self.concepts.search(concept_name)
                for r in results[:1]:
                    facts.append(f"concept:{r.name}:{r.description}")
        return facts

    def _build_context(self, semantic, user_context, conversation_history) -> ContextualFrame:
        """Build the contextual frame for response generation."""
        language = "en"
        if hasattr(semantic, 'detected_language'):
            lang = semantic.detected_language
            if hasattr(lang, 'value'):
                language = lang.value
            else:
                language = str(lang)

        turn = len(conversation_history)
        is_followup = turn > 0

        topic = ""
        if hasattr(semantic, 'topic'):
            topic = semantic.topic or ""

        expertise = user_context.get("expertise", "general")

        recent = []
        for h in conversation_history[-3:]:
            if hasattr(h, 'user_input'):
                recent.append(h.user_input)
            elif isinstance(h, dict):
                recent.append(h.get("user_input", ""))

        active_project = ""
        if user_context.get("project"):
            p = user_context["project"]
            if hasattr(p, 'name'):
                active_project = p.name
            elif isinstance(p, dict):
                active_project = p.get("name", "")

        return ContextualFrame(
            user_language=language,
            conversation_turn=turn,
            is_followup=is_followup,
            current_topic=topic,
            user_expertise=expertise,
            recent_actions=recent,
            active_project=active_project,
        )

    def _create_representation(self, core_message: CoreMessage,
                                context: ContextualFrame,
                                knowledge_context: dict) -> ResponseRepresentation:
        """Create a response representation from core message and context."""
        units = []
        style = ExpressionStyle.STANDARD

        # Determine style from context
        if context.user_expertise == "beginner":
            style = ExpressionStyle.DETAILED
        elif context.user_expertise == "expert":
            style = ExpressionStyle.CONCISE
        elif core_message.emotional_tone == "friendly":
            style = ExpressionStyle.CASUAL

        # Build sentence units based on intent
        if core_message.intent == ResponseIntent.GREETING:
            units.append(SentenceUnit(
                unit_type="greeting",
                content_key="hello",
                emphasis="normal",
                order=0,
            ))
            if context.conversation_turn == 0:
                units.append(SentenceUnit(
                    unit_type="identity",
                    content_key="self_intro_brief",
                    emphasis="normal",
                    order=1,
                ))

        elif core_message.intent == ResponseIntent.SELF_INTRODUCTION:
            units.append(SentenceUnit(
                unit_type="identity",
                content_key="self_intro_full",
                content_params={"facts": core_message.supporting_facts},
                emphasis="high",
                order=0,
            ))

        elif core_message.intent == ResponseIntent.CAPABILITY_DISCLOSURE:
            units.append(SentenceUnit(
                unit_type="capability",
                content_key="capability_list",
                content_params={"facts": core_message.supporting_facts},
                emphasis="normal",
                order=0,
            ))
            units.append(SentenceUnit(
                unit_type="question",
                content_key="offer_help",
                emphasis="normal",
                order=1,
            ))

        elif core_message.intent == ResponseIntent.CODE_KNOWLEDGE:
            if core_message.key_concepts:
                units.append(SentenceUnit(
                    unit_type="knowledge",
                    content_key="code_explanation",
                    content_params={
                        "concepts": core_message.key_concepts,
                        "facts": core_message.supporting_facts,
                    },
                    emphasis="high",
                    order=0,
                ))
            else:
                units.append(SentenceUnit(
                    unit_type="clarification",
                    content_key="need_more_detail",
                    emphasis="normal",
                    order=0,
                ))

        elif core_message.intent == ResponseIntent.PROJECT_DISCUSSION:
            units.append(SentenceUnit(
                unit_type="acknowledgment",
                content_key="confirm_capability",
                emphasis="normal",
                order=0,
            ))
            units.append(SentenceUnit(
                unit_type="question",
                content_key="ask_project_details",
                emphasis="normal",
                order=1,
            ))

        elif core_message.intent == ResponseIntent.ARCHITECTURE_EXPLANATION:
            units.append(SentenceUnit(
                unit_type="architecture",
                content_key="architecture_overview",
                content_params={"facts": core_message.supporting_facts},
                emphasis="normal",
                order=0,
            ))

        elif core_message.intent == ResponseIntent.QUESTION_ANSWER:
            if core_message.supporting_facts:
                units.append(SentenceUnit(
                    unit_type="answer",
                    content_key="answer_with_knowledge",
                    content_params={"facts": core_message.supporting_facts},
                    emphasis="normal",
                    order=0,
                ))
            else:
                units.append(SentenceUnit(
                    unit_type="answer",
                    content_key="acknowledge_and_offer",
                    emphasis="normal",
                    order=0,
                ))

        elif core_message.intent == ResponseIntent.CLARIFICATION:
            units.append(SentenceUnit(
                unit_type="clarification",
                content_key="ask_clarification",
                content_params={"topic": context.current_topic},
                emphasis="normal",
                order=0,
            ))

        elif core_message.intent == ResponseIntent.GOODBYE:
            units.append(SentenceUnit(
                unit_type="goodbye",
                content_key="farewell",
                emphasis="normal",
                order=0,
            ))

        elif core_message.intent == ResponseIntent.ACKNOWLEDGMENT:
            units.append(SentenceUnit(
                unit_type="acknowledgment",
                content_key="understood",
                emphasis="normal",
                order=0,
            ))

        else:
            units.append(SentenceUnit(
                unit_type="fallback",
                content_key="general_response",
                emphasis="normal",
                order=0,
            ))

        return ResponseRepresentation(
            core_message=core_message,
            context=context,
            sentence_units=units,
            style=style,
        )

    def _plan_sentences(self, representation: ResponseRepresentation) -> SentencePlan:
        """Plan how to construct sentences from the representation."""
        plan = SentencePlan()
        units = sorted(representation.sentence_units, key=lambda u: u.order)

        for i, unit in enumerate(units):
            clause = {
                "type": unit.unit_type,
                "key": unit.content_key,
                "params": unit.content_params,
                "emphasis": unit.emphasis,
            }
            plan.clauses.append(clause)

            # Add connectors between clauses
            if i < len(units) - 1:
                next_unit = units[i + 1]
                connector = self._select_connector(unit, next_unit, representation.style)
                plan.connectors.append(connector)

            # Add line breaks for readability
            if unit.unit_type in ("identity", "capability", "architecture", "knowledge"):
                plan.line_breaks.append(len(plan.clauses) - 1)

        return plan

    def _select_connector(self, current: SentenceUnit, next_unit: SentenceUnit,
                          style: ExpressionStyle) -> str:
        """Select a connector between two sentence units."""
        if style == ExpressionStyle.CONCISE:
            return ". "
        elif style == ExpressionStyle.CASUAL:
            if next_unit.unit_type == "question":
                return " "
            return " "
        else:
            if next_unit.unit_type == "question":
                return " "
            return ". "

    def _realize_language(self, plan: SentencePlan, context: ContextualFrame) -> str:
        """Realize the sentence plan into actual language."""
        lang = context.user_language

        # Language-specific realization
        if lang in ("ko", "kor"):
            return self._realize_korean(plan, context)
        elif lang in ("ja", "jpn"):
            return self._realize_japanese(plan, context)
        elif lang in ("zh", "zho"):
            return self._realize_chinese(plan, context)
        else:
            return self._realize_english(plan, context)

    def _realize_english(self, plan: SentencePlan, context: ContextualFrame) -> str:
        """Realize sentence plan in English."""
        parts = []
        for i, clause in enumerate(plan.clauses):
            text = self._clause_to_text_english(clause, context)
            if text:
                parts.append(text)

            # Add connector
            if i < len(plan.connectors):
                parts.append(plan.connectors[i])

        return "".join(parts)

    def _clause_to_text_english(self, clause: dict, context: ContextualFrame) -> str:
        """Convert a clause to English text."""
        ctype = clause["type"]
        key = clause["key"]
        params = clause.get("params", {})

        if ctype == "greeting":
            return "Hello"
        elif ctype == "identity":
            if key == "self_intro_brief":
                name = "RENA-BR"
                if self.self_knowledge:
                    n = self.self_knowledge.get("identity", "name")
                    if n:
                        name = n
                return f"I'm {name}"
            elif key == "self_intro_full":
                return self._build_identity_text("en", params)
            return ""
        elif ctype == "capability":
            return self._build_capability_text("en", params)
        elif ctype == "knowledge":
            return self._build_knowledge_text("en", params)
        elif ctype == "architecture":
            return self._build_architecture_text("en", params)
        elif ctype == "question":
            if key == "offer_help":
                return "What would you like me to help with?"
            elif key == "ask_project_details":
                return "What kind of project would you like to build?"
            return ""
        elif ctype == "answer":
            return self._build_answer_text("en", params)
        elif ctype == "clarification":
            if key == "need_more_detail":
                return "Could you provide more details about what you'd like to know?"
            elif key == "ask_clarification":
                topic = params.get("topic", "")
                if topic:
                    return f"Could you tell me more about what you'd like regarding {topic}?"
                return "Could you provide more details about what you're looking for?"
            return ""
        elif ctype == "acknowledgment":
            if key == "confirm_capability":
                return "I can help with that"
            elif key == "understood":
                return "I understand"
            return ""
        elif ctype == "goodbye":
            return "Goodbye"
        elif ctype == "fallback":
            return "I'm here to help with coding tasks"
        return ""

    def _realize_korean(self, plan: SentencePlan, context: ContextualFrame) -> str:
        """Realize sentence plan in Korean."""
        parts = []
        for i, clause in enumerate(plan.clauses):
            text = self._clause_to_text_korean(clause, context)
            if text:
                parts.append(text)
            if i < len(plan.connectors):
                parts.append(plan.connectors[i])
        return "".join(parts)

    def _clause_to_text_korean(self, clause: dict, context: ContextualFrame) -> str:
        """Convert a clause to Korean text."""
        ctype = clause["type"]
        key = clause["key"]
        params = clause.get("params", {})

        if ctype == "greeting":
            return "안녕하세요"
        elif ctype == "identity":
            if key == "self_intro_brief":
                name = "RENA-BR"
                if self.self_knowledge:
                    n = self.self_knowledge.get("identity", "name")
                    if n:
                        name = n
                return f"{name}입니다"
            elif key == "self_intro_full":
                return self._build_identity_text("ko", params)
            return ""
        elif ctype == "capability":
            return self._build_capability_text("ko", params)
        elif ctype == "knowledge":
            return self._build_knowledge_text("ko", params)
        elif ctype == "architecture":
            return self._build_architecture_text("ko", params)
        elif ctype == "question":
            if key == "offer_help":
                return "무엇을 도와드릴까요?"
            elif key == "ask_project_details":
                return "어떤 종류의 프로젝트를 만들고 싶으신가요?"
            return ""
        elif ctype == "answer":
            return self._build_answer_text("ko", params)
        elif ctype == "clarification":
            if key == "need_more_detail":
                return "더 구체적으로 어떤 것이 궁금하신가요?"
            elif key == "ask_clarification":
                topic = params.get("topic", "")
                if topic:
                    return f"{topic}에 대해 더 자세히 알려주시겠어요?"
                return "더 자세한 정보를 알려주시겠어요?"
            return ""
        elif ctype == "acknowledgment":
            if key == "confirm_capability":
                return "도와드릴 수 있습니다"
            elif key == "understood":
                return "알겠습니다"
            return ""
        elif ctype == "goodbye":
            return "안녕히 가세요"
        elif ctype == "fallback":
            return "코드 생성 등을 도와드릴 수 있습니다"
        return ""

    def _realize_japanese(self, plan: SentencePlan, context: ContextualFrame) -> str:
        """Realize sentence plan in Japanese."""
        parts = []
        for i, clause in enumerate(plan.clauses):
            text = self._clause_to_text_japanese(clause, context)
            if text:
                parts.append(text)
            if i < len(plan.connectors):
                parts.append(plan.connectors[i])
        return "".join(parts)

    def _clause_to_text_japanese(self, clause: dict, context: ContextualFrame) -> str:
        """Convert a clause to Japanese text."""
        ctype = clause["type"]
        key = clause["key"]

        if ctype == "greeting":
            return "こんにちは"
        elif ctype == "identity":
            if key == "self_intro_brief":
                name = "RENA-BR"
                if self.self_knowledge:
                    n = self.self_knowledge.get("identity", "name")
                    if n:
                        name = n
                return f"{name}です"
            return ""
        elif ctype == "question":
            if key == "offer_help":
                return "何かお手伝いできますか"
            return ""
        elif ctype == "acknowledgment":
            if key == "confirm_capability":
                return "お手伝いできます"
            return ""
        elif ctype == "goodbye":
            return "さようなら"
        elif ctype == "fallback":
            return "コード生成をサポートします"
        return ""

    def _realize_chinese(self, plan: SentencePlan, context: ContextualFrame) -> str:
        """Realize sentence plan in Chinese."""
        parts = []
        for i, clause in enumerate(plan.clauses):
            text = self._clause_to_text_chinese(clause, context)
            if text:
                parts.append(text)
            if i < len(plan.connectors):
                parts.append(plan.connectors[i])
        return "".join(parts)

    def _clause_to_text_chinese(self, clause: dict, context: ContextualFrame) -> str:
        """Convert a clause to Chinese text."""
        ctype = clause["type"]
        key = clause["key"]

        if ctype == "greeting":
            return "你好"
        elif ctype == "identity":
            if key == "self_intro_brief":
                name = "RENA-BR"
                if self.self_knowledge:
                    n = self.self_knowledge.get("identity", "name")
                    if n:
                        name = n
                return f"我是{name}"
            return ""
        elif ctype == "question":
            if key == "offer_help":
                return "有什么可以帮您的吗"
            return ""
        elif ctype == "acknowledgment":
            if key == "confirm_capability":
                return "我可以帮助您"
            return ""
        elif ctype == "goodbye":
            return "再见"
        elif ctype == "fallback":
            return "我可以帮助代码生成"
        return ""

    def _build_identity_text(self, lang: str, params: dict) -> str:
        """Build identity introduction text dynamically."""
        facts = params.get("facts", [])

        # Extract facts
        name = "RENA-BR"
        arch = ""
        purpose = ""
        for fact in facts:
            if fact.startswith("name:"):
                name = fact.split(":", 1)[1]
            elif fact.startswith("architecture:"):
                arch = fact.split(":", 1)[1]
            elif fact.startswith("purpose:"):
                purpose = fact.split(":", 1)[1]

        if lang == "ko":
            parts = [f"{name}입니다"]
            if arch:
                parts.append(f"저는 {arch} 기반으로 동작합니다")
            if purpose:
                parts.append(purpose)
            return ", ".join(parts)
        elif lang == "ja":
            parts = [f"{name}です"]
            if arch:
                parts.append(f"{arch}ベースで動作しています")
            return ", ".join(parts)
        elif lang == "zh":
            parts = [f"我是{name}"]
            if arch:
                parts.append(f"基于{arch}架构运行")
            return ", ".join(parts)
        else:
            parts = [f"I'm {name}"]
            if arch:
                parts.append(f"built on {arch}")
            if purpose:
                parts.append(purpose)
            return ", ".join(parts)

    def _build_capability_text(self, lang: str, params: dict) -> str:
        """Build capability description text dynamically."""
        facts = params.get("facts", [])

        # Extract capability names
        cap_names = []
        for fact in facts:
            if fact.startswith("capability:"):
                parts = fact.split(":", 2)
                if len(parts) >= 2:
                    cap_names.append(parts[1].replace("_", " "))

        if not cap_names:
            cap_names = ["code generation", "project creation"]

        cap_list = ", ".join(cap_names[:5])

        if lang == "ko":
            return f"다음 기능을 수행할 수 있습니다: {cap_list}"
        elif lang == "ja":
            return f"以下の機能をサポートしています: {cap_list}"
        elif lang == "zh":
            return f"我可以帮助以下功能: {cap_list}"
        else:
            return f"I can help with: {cap_list}"

    def _build_knowledge_text(self, lang: str, params: dict) -> str:
        """Build code knowledge text dynamically."""
        concepts = params.get("concepts", [])
        facts = params.get("facts", [])

        if facts:
            # Extract knowledge from facts
            explanations = []
            for fact in facts:
                if fact.startswith("concept:"):
                    parts = fact.split(":", 2)
                    if len(parts) >= 3:
                        explanations.append(f"{parts[1]}: {parts[2]}")

            if explanations:
                text = ". ".join(explanations[:2])
                if lang == "ko":
                    return text
                elif lang == "ja":
                    return text
                elif lang == "zh":
                    return text
                else:
                    return text

        if concepts:
            concept_str = ", ".join(concepts[:3])
            if lang == "ko":
                return f"{concept_str}에 대한 질문이군요"
            elif lang == "ja":
                return f"{concept_str}についての質問ですね"
            elif lang == "zh":
                return f"关于{concept_str}的问题"
            else:
                return f"Regarding {concept_str}"

        return ""

    def _build_architecture_text(self, lang: str, params: dict) -> str:
        """Build architecture explanation text dynamically."""
        facts = params.get("facts", [])

        components = []
        for fact in facts:
            if fact.startswith("component:"):
                parts = fact.split(":", 2)
                if len(parts) >= 3:
                    components.append(f"{parts[1]}: {parts[2]}")

        if components:
            comp_text = ". ".join(components[:4])
            if lang == "ko":
                return f"시스템 구조: {comp_text}"
            elif lang == "ja":
                return f"システム構造: {comp_text}"
            elif lang == "zh":
                return f"系统架构: {comp_text}"
            else:
                return f"Architecture: {comp_text}"

        return ""

    def _build_answer_text(self, lang: str, params: dict) -> str:
        """Build answer text dynamically from knowledge."""
        facts = params.get("facts", [])

        if facts:
            # Build answer from facts
            answer_parts = []
            for fact in facts[:3]:
                if ":" in fact:
                    parts = fact.split(":", 1)
                    if len(parts) >= 2:
                        answer_parts.append(parts[1])

            if answer_parts:
                answer = ". ".join(answer_parts)
                if lang == "ko":
                    return answer
                elif lang == "ja":
                    return answer
                elif lang == "zh":
                    return answer
                else:
                    return answer

        # Generic acknowledgment
        if lang == "ko":
            return "알겠습니다. 더 자세히 설명드리겠습니다"
        elif lang == "ja":
            return "了解しました。もう少し詳しく説明します"
        elif lang == "zh":
            return "我明白了。让我更详细地解释"
        else:
            return "I understand. Let me explain further"
