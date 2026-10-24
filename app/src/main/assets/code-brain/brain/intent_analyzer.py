"""Intent Analyzer - classifies user input into intents for routing.

ARCHITECTURE NOTE: This module has ZERO keyword authority over user meaning.
All intent detection is delegated to SemanticParser + SemanticComposer.
The pattern lists below are LEGACY COMPATIBILITY data only. They exist to
maintain interface compatibility with code that imports INTENT_* constants
and IntentResult, but they do NOT drive any decisions.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: list[dict] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_message: str = ""


INTENT_CHAT = "CHAT"
INTENT_QUESTION = "QUESTION"
INTENT_EXPLANATION = "EXPLANATION"
INTENT_GENERATE_CODE = "GENERATE_CODE"
INTENT_CREATE_PROJECT = "CREATE_PROJECT"
INTENT_MODIFY_PROJECT = "MODIFY_PROJECT"
INTENT_DEBUG = "DEBUG"
INTENT_ANALYZE_CODE = "ANALYZE_CODE"
INTENT_TEST_CODE = "TEST_CODE"
INTENT_BUILD_PROJECT = "BUILD_PROJECT"
INTENT_COMPUTER_TASK = "COMPUTER_TASK"
INTENT_UNKNOWN = "UNKNOWN"

# LEGACY: Kept for interface compatibility only. Zero decision authority.
# All intent detection now goes through SemanticParser + SemanticComposer.
_GOAL_TO_INTENT = {
    "chat": INTENT_CHAT,
    "question": INTENT_QUESTION,
    "information_query": INTENT_QUESTION,
    "command_query": INTENT_QUESTION,
    "research": INTENT_QUESTION,
    "explain": INTENT_EXPLANATION,
    "create_function": INTENT_GENERATE_CODE,
    "create_project": INTENT_CREATE_PROJECT,
    "create_class": INTENT_GENERATE_CODE,
    "modify": INTENT_MODIFY_PROJECT,
    "fix": INTENT_DEBUG,
    "analyze": INTENT_ANALYZE_CODE,
    "ambiguous": INTENT_UNKNOWN,
    "unknown": INTENT_UNKNOWN,
}


class IntentAnalyzer:
    """Classifies user input into intents by delegating to SemanticParser.

    This class has ZERO keyword authority. All intent detection is performed
    by SemanticParser + SemanticComposer. This class only:
    1. Delegates to SemanticParser for intent detection
    2. Maps SemanticParser goal types to legacy INTENT_* constants
    3. Provides canned responses for chat/question/explain intents
    """

    def __init__(self):
        self._parser = None

    def _get_parser(self):
        """Lazy-init SemanticParser to avoid circular imports.

        The parser runs on the SAME knowledge the pipeline runs on: the
        operation and artifact ontology plus the algorithmic relations.
        An unseeded store would make every request look like an
        information query, and the intent the analyzer reports would be
        a property of the missing knowledge rather than of the sentence.
        """
        if self._parser is None:
            try:
                from brain.semantic_parser import create_semantic_parser
                from brain.concept_store import (ConceptStore,
                                                 seed_default_concepts,
                                                 seed_programming_fundamentals)
                from brain.relation_store import (RelationStore,
                                                  seed_algorithmic_relations)
                import tempfile
                concepts = ConceptStore()
                seed_default_concepts(concepts)
                seed_programming_fundamentals(concepts)
                relations = RelationStore(data_dir=tempfile.mkdtemp())
                seed_algorithmic_relations(relations)
                self._parser = create_semantic_parser(concepts, relations)
            except Exception:
                self._parser = False
        return self._parser if self._parser is not False else None

    def analyze(self, text: str) -> IntentResult:
        """Detect intent by delegating to SemanticParser.

        No keyword matching. Intent is derived from the parser's goal
        detection, which itself uses concept activation and composition.
        """
        text = text.strip()
        if not text:
            return IntentResult(
                intent=INTENT_UNKNOWN,
                confidence=1.0,
                needs_clarification=True,
                clarification_message="Please provide a request.",
            )

        parser = self._get_parser()
        if parser is not None:
            try:
                semantic = parser.parse(text)
                goal_name = semantic.goal.goal_type.value
                intent = _GOAL_TO_INTENT.get(goal_name, INTENT_UNKNOWN)
                entities = [
                    {"type": e.kind, "value": e.name}
                    for e in semantic.entities
                ]
                # An open goal is not a decision: the parser reported
                # that it could not read an object or a goal type out of
                # the utterance, so the caller is told to ask rather than
                # act on a guess.
                dialogue = semantic.dialogue_intent or {}
                needs_clarification = (
                    intent == INTENT_UNKNOWN
                    or goal_name in ("unknown", "ambiguous")
                    or dialogue.get("type") == "clarification"
                )
                message = ""
                if needs_clarification:
                    missing = (dialogue.get("scope") or {}).get("missing", "")
                    message = (
                        f"I could not tell what you want: {missing}."
                        if missing else
                        "I could not tell what you would like me to do. "
                        "Could you say which artifact or action you mean?"
                    )
                return IntentResult(
                    intent=intent,
                    confidence=semantic.confidence,
                    entities=entities,
                    needs_clarification=needs_clarification,
                    clarification_message=message,
                )
            except Exception:
                pass

        # Minimal fallback: question mark heuristic only
        if re.search(r'[?？]', text):
            return IntentResult(intent=INTENT_QUESTION, confidence=0.5)
        return IntentResult(intent=INTENT_UNKNOWN, confidence=0.3,
                            needs_clarification=True,
                            clarification_message="I need more information. What would you like me to build or change?")

    def get_chat_response(self, text: str) -> str:
        return (
            "Hello! I'm RENA-BR, an AI coding assistant.\n"
            "I can help you with:\n"
            "- Generate code (functions, classes)\n"
            "- Create projects (apps, websites)\n"
            "- Debug errors\n"
            "- Explain code\n"
            "What would you like to do?"
        )

    def _question_topic(self, text: str):
        """The topic the utterance asks about, read from its structure."""
        parser = self._get_parser()
        if parser is None:
            return ""
        try:
            semantic = parser.parse(text)
        except Exception:
            return ""
        dialogue = semantic.dialogue_intent or {}
        topics = list(dialogue.get("topics") or [])
        if topics and isinstance(topics[0], str) and topics[0].strip():
            return topics[0].strip()
        target = getattr(semantic.goal, "target", "") or ""
        if isinstance(target, str):
            return target.strip()
        return ""

    def get_question_response(self, text: str) -> str:
        """Answer the question ABOUT the thing it asks about.

        The reply is composed from the utterance's own topic — the
        structure the parser read out of the sentence — so a question
        about one concept is answered about that concept.  With no topic
        to answer, the response says what is missing instead of
        pretending the question had none.
        """
        topic = self._question_topic(text)
        if not topic:
            return (
                "That is a good question — but the utterance does not "
                "name a topic. Could you say which concept you mean?"
            )
        return (
            f"About {topic}: let me explain it clearly.\n\n"
            f"1. What it is — {topic} in short, stated plainly.\n"
            f"2. How it works — the mechanism behind {topic}.\n"
            f"3. Where it shows up — code that leans on {topic}.\n\n"
            "Which of these three would you like first?"
        )

    def get_explanation_response(self, text: str) -> str:
        return (
            "I'd be happy to explain!\n\n"
            "However, I need some context:\n"
            "- Which specific code or concept would you like explained?\n"
            "- Is there a particular language or framework involved?\n\n"
            "Please provide the code or describe the concept in more detail."
        )

    def get_unknown_response(self, text: str) -> str:
        return (
            "I'm not sure what you'd like me to do.\n\n"
            "I can help with:\n"
            "- Generate code: '두 숫자를 더하는 함수 만들어줘'\n"
            "- Create projects: '학생 관리 프로그램 만들어줘'\n"
            "- Debug errors: '이 코드 왜 오류나?'\n"
            "- Explain code: '이 함수 설명해줘'\n"
            "- Answer questions: '코틀린에서 val과 var 차이가 뭐야?'\n\n"
            "Please try again with more details."
        )
