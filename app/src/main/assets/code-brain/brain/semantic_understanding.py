"""Semantic Understanding - extracts meaning from user input.

ARCHITECTURE NOTE: This module has ZERO keyword authority over user meaning.
All understanding is derived from SemanticParser goal detection and
SemanticComposer composition results. Keyword patterns have been removed.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SemanticCategory(Enum):
    GREETING = "GREETING"
    SELF_INTRODUCTION = "SELF_INTRODUCTION"
    CAPABILITIES = "CAPABILITIES"
    CASUAL_CONVERSATION = "CASUAL_CONVERSATION"
    GOODBYE = "GOODBYE"
    CODE_KNOWLEDGE = "CODE_KNOWLEDGE"
    SELF_KNOWLEDGE = "SELF_KNOWLEDGE"
    CODE_GENERATION = "CODE_GENERATION"
    ARCHITECTURE = "ARCHITECTURE"
    PROJECT_QUESTION = "PROJECT_QUESTION"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    UNKNOWN = "UNKNOWN"


class Language(Enum):
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    UNKNOWN = "unknown"


@dataclass
class SemanticRepresentation:
    original_text: str
    detected_language: Language
    categories: list[SemanticCategory]
    topic: str
    entities: list[dict]
    intent_keywords: list[str]
    confidence: float
    is_question: bool
    is_greeting: bool
    is_about_self: bool
    is_about_code: bool
    is_about_project: bool
    is_about_architecture: bool = False
    detail_request: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "detected_language": self.detected_language.value,
            "categories": [c.value for c in self.categories],
            "topic": self.topic,
            "entities": self.entities,
            "intent_keywords": self.intent_keywords,
            "confidence": self.confidence,
            "is_question": self.is_question,
            "is_greeting": self.is_greeting,
            "is_about_self": self.is_about_self,
            "is_about_code": self.is_about_code,
            "is_about_project": self.is_about_project,
            "is_about_architecture": self.is_about_architecture,
            "detail_request": self.detail_request,
        }


# LEGACY: Goal-to-conversational-category mapping.
# source: legacy, usage: fallback+compatibility, confidence: 0.6
# Primary path: SemanticParser goal detection + composition analysis.
_GOAL_TO_CATEGORY = {
    "chat": SemanticCategory.GREETING,
    "question": SemanticCategory.GENERAL_KNOWLEDGE,
    "explain": SemanticCategory.SELF_KNOWLEDGE,
    "create_function": SemanticCategory.CODE_GENERATION,
    "create_project": SemanticCategory.PROJECT_QUESTION,
    "create_class": SemanticCategory.CODE_GENERATION,
    "modify": SemanticCategory.CODE_KNOWLEDGE,
    "fix": SemanticCategory.CODE_KNOWLEDGE,
    "analyze": SemanticCategory.CODE_KNOWLEDGE,
}


def _detect_language(text: str) -> Language:
    """Detect language from character ranges and keywords.

    Priority: script-based detection first, then keyword disambiguation
    for Latin-script languages (Spanish, French, Portuguese, Italian, English).
    """
    # Cyrillic → Russian
    if re.search(r'[\u0400-\u04ff]', text):
        return Language.RUSSIAN
    # Korean
    if re.search(r'[\uac00-\ud7a3]', text):
        return Language.KOREAN
    # Japanese (Hiragana + Katakana)
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return Language.JAPANESE
    # Chinese (CJK Unified Ideographs, excluding Japanese-only chars)
    if re.search(r'[\u4e00-\u9fff]', text):
        return Language.CHINESE

    # Latin-script languages: disambiguate by common words
    text_lower = text.lower()

    # Spanish indicators
    es_words = ['hola', 'gracias', 'por favor', 'cómo', 'qué', 'cuál',
                'necesito', 'quiero', 'puedo', 'crear', 'generar', 'función']
    if any(w in text_lower for w in es_words):
        return Language.SPANISH

    # French indicators
    fr_words = ['bonjour', 'merci', 's\'il vous plaît', 'comment', 'quel',
                'je veux', 'je besoin', 'créer', 'générer', 'fonction']
    if any(w in text_lower for w in fr_words):
        return Language.FRENCH

    # Portuguese indicators
    pt_words = ['olá', 'obrigado', 'por favor', 'como', 'qual',
                'eu quero', 'preciso', 'criar', 'gerar', 'função']
    if any(w in text_lower for w in pt_words):
        return Language.PORTUGUESE

    # Italian indicators
    it_words = ['ciao', 'grazie', 'per favore', 'come', 'quale',
                'io voglio', 'ho bisogno', 'creare', 'generare', 'funzione']
    if any(w in text_lower for w in it_words):
        return Language.ITALIAN

    # German indicators
    de_words = ['hallo', 'danke', 'bitte', 'wie', 'welche',
                'ich möchte', 'ich brauche', 'erstellen', 'generieren', 'funktion']
    if any(w in text_lower for w in de_words):
        return Language.GERMAN

    return Language.ENGLISH


class SemanticUnderstanding:
    """Understands user input by delegating to SemanticParser.

    This class has ZERO keyword authority. All understanding is derived
    from SemanticParser goal detection and SemanticComposer composition.
    """

    def __init__(self):
        self._parser = None

    def _get_parser(self):
        """Lazy-init SemanticParser to avoid circular imports."""
        if self._parser is None:
            try:
                from brain.semantic_parser import create_semantic_parser
                from brain.concept_store import ConceptStore
                from brain.relation_store import RelationStore
                import tempfile
                self._parser = create_semantic_parser(
                    ConceptStore(), RelationStore(data_dir=tempfile.mkdtemp())
                )
            except Exception:
                self._parser = False
        return self._parser if self._parser is not False else None

    def understand(self, text: str) -> SemanticRepresentation:
        """Understand user input by delegating to SemanticParser.

        No keyword matching. Understanding is derived from:
        1. SemanticParser goal detection (concept activation + composition)
        2. Composition cues (what operations the composer detected)
        3. Minimal text heuristics (question marks, text length)
        """
        lang = _detect_language(text)
        text_lower = text.strip().lower()

        # Delegate to SemanticParser for goal detection
        parser = self._get_parser()
        if parser is not None:
            try:
                semantic = parser.parse(text)
                goal_name = semantic.goal.goal_type.value
                category = _GOAL_TO_CATEGORY.get(goal_name, SemanticCategory.UNKNOWN)

                # Determine sub-categories from composition
                comp = semantic.composition or {}
                cues = comp.get("user_cues", [])
                has_composition = len(cues) > 0
                has_ops = len(semantic.operation_graph.operations) > 0

                # The dialogue act the parser read: what the turn IS as
                # an act (greeting, self-presentation, capability ask)
                # rather than what its goal type happens to be called.
                intent = getattr(semantic, "dialogue_intent", None) or {}
                intent_type = str(intent.get("type") or "")
                intent_goal = str(intent.get("goal") or "")
                # The concepts the turn activated: a language feature
                # or a pattern is code the turn is speaking about, even
                # when no entity or operation was composed from it.
                concept_evidence = list(
                    ((getattr(semantic.goal, "domain_evidence", None) or {})
                     .get("concepts")) or [])
                code_kinds = ("language_feature", "pattern", "ui_component",
                              "algorithmic_primitive")
                is_about_code = bool(
                    [c for c in concept_evidence
                     if len(c) > 1 and str(c[1]) in code_kinds])

                categories = [category]
                if intent_type == "greeting" and \
                        SemanticCategory.GREETING not in categories:
                    categories.append(SemanticCategory.GREETING)
                if intent_goal == "present_self" and \
                        SemanticCategory.SELF_INTRODUCTION not in categories:
                    categories.append(SemanticCategory.SELF_INTRODUCTION)
                if intent_type == "capability_query" and \
                        SemanticCategory.CAPABILITIES not in categories:
                    categories.append(SemanticCategory.CAPABILITIES)
                is_question = goal_name == "question" or "?" in text or "?" in text
                is_greeting = intent_type == "greeting" or (
                    goal_name == "chat" and intent_goal != "present_self")
                is_about_self = intent_goal == "present_self"
                is_about_code = is_about_code or has_composition or has_ops \
                    or goal_name in (
                    "create_function", "create_class", "fix", "analyze"
                )
                if is_about_code and SemanticCategory.CODE_KNOWLEDGE not in categories:
                    categories.append(SemanticCategory.CODE_KNOWLEDGE)
                is_about_project = goal_name == "create_project"
                is_about_architecture = "architecture" in text_lower or goal_name == "explain"

                # Extract topic from entities
                topic = ""
                entities = [
                    {"type": e.kind, "value": e.name}
                    for e in semantic.entities
                ]
                if entities:
                    topic = entities[0].get("value", "")

                return SemanticRepresentation(
                    original_text=text,
                    detected_language=lang,
                    categories=categories,
                    topic=topic,
                    entities=entities,
                    intent_keywords=[c.get("text", "") for c in cues],
                    confidence=semantic.confidence,
                    is_question=is_question,
                    is_greeting=is_greeting,
                    is_about_self=is_about_self,
                    is_about_code=is_about_code,
                    is_about_project=is_about_project,
                    is_about_architecture=is_about_architecture,
                    metadata={"goal": goal_name, "composition_ops": [c["primitive"] for c in cues]},
                )
            except Exception:
                pass

        # Minimal fallback: question mark heuristic only
        is_question = "?" in text or "?" in text
        category = SemanticCategory.GENERAL_KNOWLEDGE if is_question else SemanticCategory.UNKNOWN
        return SemanticRepresentation(
            original_text=text,
            detected_language=lang,
            categories=[category],
            topic="",
            entities=[],
            intent_keywords=[],
            confidence=0.4,
            is_question=is_question,
            is_greeting=False,
            is_about_self=False,
            is_about_code=False,
            is_about_project=False,
        )

    def detect_language(self, text: str) -> Language:
        """The turn's language — the same script-then-keyword reading
        every representation of the turn records as its language."""
        return _detect_language(str(text or ""))

    def _extract_topic(self, text: str, entities: list[dict]) -> str:
        if entities:
            return entities[0].get("value", "")
        words = re.findall(r"\w+", text)
        if words:
            return words[0]
        return ""
