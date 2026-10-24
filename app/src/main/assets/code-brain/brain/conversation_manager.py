"""Conversation Manager - tracks conversation state and context."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from code.ir.project_ir import ProjectIR
from brain.context_resolution import TopicState


def _clean_list(items) -> list:
    """The non-empty entries of a record, deduped, in order."""
    out: list = []
    seen: set = set()
    for item in (items or []):
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


@dataclass
class ConversationTurn:
    user_input: str
    intent: str
    response: str
    project_ir: Optional[ProjectIR] = None
    # The turn's resolved topics: the referents the NEXT turn's
    # reference resolution may point at (the discourse's own record,
    # not a re-parse of the raw text).
    topics: list = field(default_factory=list)
    # The turn's STRUCTURED meaning — the record the next turn's
    # reference resolution reads instead of re-parsing the raw text:
    #   semantic_representation : the turn's own understanding
    #   topic / entities / concepts : what it carried
    #   claims / response_representation : what the response SAID
    #   referenced_knowledge : the knowledge the answer was built from
    semantic_representation: dict = field(default_factory=dict)
    topic: str = ""
    entities: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    response_representation: dict = field(default_factory=dict)
    referenced_knowledge: dict = field(default_factory=dict)
    # The turn's GENERATED ARTIFACT (function name, code, IR) — the
    # working artifact of THIS turn's generation, kept with the turn so
    # the next turn's continuation finds it even across CLI processes
    # (one invocation is one turn; the session carries them all).
    artifact: dict = field(default_factory=dict)
    # The turn's SemanticRepresentation OBJECT (the dict above is the
    # serialized view) — the continuation merge reads the real
    # structure (operation graph, IO specs) instead of re-deriving it.
    # Runtime-only, never serialized.
    semantic_object: Optional[object] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """The turn's record, serializable: everything the NEXT turn's
        resolution reads (its meaning, its topics, what the response
        said).  A project IR is the working artifact of a generation,
        not of a conversation — it is not carried between sessions."""
        return {
            "user_input": self.user_input,
            "intent": self.intent,
            "response": self.response,
            "topics": list(self.topics),
            "semantic_representation": dict(self.semantic_representation),
            "topic": self.topic,
            "entities": list(self.entities),
            "concepts": list(self.concepts),
            "claims": list(self.claims),
            "response_representation": dict(self.response_representation),
            "referenced_knowledge": dict(self.referenced_knowledge),
            "artifact": dict(self.artifact),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        d = dict(data or {})
        return cls(
            user_input=str(d.get("user_input") or ""),
            intent=str(d.get("intent") or ""),
            response=str(d.get("response") or ""),
            topics=[str(t) for t in (d.get("topics") or []) if t],
            semantic_representation=dict(d.get("semantic_representation")
                                         or {}),
            topic=str(d.get("topic") or ""),
            entities=[str(x) for x in (d.get("entities") or []) if x],
            concepts=[str(x) for x in (d.get("concepts") or []) if x],
            claims=[str(x) for x in (d.get("claims") or []) if x],
            response_representation=dict(d.get("response_representation")
                                         or {}),
            referenced_knowledge=dict(d.get("referenced_knowledge") or {}),
            artifact=dict(d.get("artifact") or {}),
        )


class ConversationManager:
    def __init__(self):
        self._history: list[ConversationTurn] = []
        self._current_project: Optional[ProjectIR] = None
        self._last_intent: str = ""
        # The TOPIC STATE: the discourse's topical state (current
        # topic, previous topics, active entities/concepts/task).  An
        # anaphoric reference resolves against it; a topic-reset act
        # retires the current topic.
        self._topic_state = TopicState()

    def add_turn(self, user_input: str, intent: str, response: str,
                 project_ir: Optional[ProjectIR] = None, topics=()):
        turn = ConversationTurn(
            user_input=user_input,
            intent=intent,
            response=response,
            project_ir=project_ir,
            topics=list(topics or []),
        )
        self._history.append(turn)
        self._last_intent = intent
        if project_ir:
            self._current_project = project_ir

    def ensure_turn(self, user_input: str, intent: str,
                    response: str = "") -> None:
        """The turn must EXIST before its state is recorded on it.

        Every handler that finishes a turn calls this (through the
        shared finalize tail): a handler that already appended its own
        turn is a no-op — one request, ONE turn record; a handler that
        did not (a code generation, a fix) gets its record here, so its
        structured meaning is never written onto the PREVIOUS turn."""
        text = str(user_input or "")
        if self._history and self._history[-1].user_input == text:
            turn = self._history[-1]
            if not turn.response and response:
                turn.response = str(response)
            if intent and not turn.intent:
                turn.intent = str(intent)
            return
        self.add_turn(text, str(intent or ""), str(response or ""))

    def record_semantic_object(self, semantic) -> None:
        """The just-finished turn keeps its SemanticRepresentation
        OBJECT (runtime-only) for the next turn's continuation merge."""
        if not self._history:
            return
        self._history[-1].semantic_object = semantic

    def record_turn_artifact(self, artifact: dict) -> None:
        """The just-finished turn's generated artifact (function name,
        code, IR) — what a follow-up modification continues from."""
        if not self._history or not isinstance(artifact, dict):
            return
        if artifact:
            self._history[-1].artifact = dict(artifact)

    def previous_turn(self, user_input: str = "") -> Optional[ConversationTurn]:
        """The most recent turn that is NOT the current request — the
        turn a continuation (implement THAT / modify IT / add tests)
        continues from."""
        text = str(user_input or "")
        for turn in reversed(self._history):
            if turn.user_input != text:
                return turn
        return None

    @property
    def topic_state(self) -> TopicState:
        """The discourse's topical state (current, previous, active)."""
        return self._topic_state

    @property
    def current_topic(self) -> str:
        return self._topic_state.current_topic

    def set_topic(self, topic: str, **kwargs) -> None:
        """The discourse's current topic becomes this referent — the
        source an anaphoric reference resolves against.  The previous
        topic is retained in the state's history (never destroyed)."""
        self._topic_state.push((topic or "").strip(), **kwargs)

    def forget_topic(self, evidence=()) -> None:
        """A topic-reset act: the current topic is retired.  An anaphoric
        reference no longer resolves against it (honest failure →
        clarification)."""
        self._topic_state.retire(evidence)

    @property
    def current_project(self) -> Optional[ProjectIR]:
        return self._current_project

    @current_project.setter
    def current_project(self, project: Optional[ProjectIR]):
        self._current_project = project

    @property
    def last_intent(self) -> str:
        return self._last_intent

    @property
    def history(self) -> list[ConversationTurn]:
        return list(self._history)

    def record_turn_topics(self, topics) -> None:
        """The just-finished turn's resolved topics travel with it, so
        the NEXT turn's reference resolution reads what the discourse
        actually established (not a re-parse of the raw text)."""
        if not self._history:
            return
        self._history[-1].topics = [str(t) for t in (topics or []) if t]

    def record_turn_state(self, *, semantic_representation=None,
                          topic: str = "", entities=(), concepts=(),
                          claims=(), response_representation=None,
                          referenced_knowledge=None) -> None:
        """The just-finished turn's STRUCTURED meaning: what it was
        about, what it carried, what the response said and the
        knowledge it was built from.

        The record is what the next turn's reference resolution reads —
        a discourse of meanings, not a list of raw strings."""
        if not self._history:
            return
        turn = self._history[-1]
        if isinstance(semantic_representation, dict):
            turn.semantic_representation = dict(semantic_representation)
        turn.topic = str(topic or turn.topic or "")
        turn.entities = _clean_list(entities) or turn.entities
        turn.concepts = _clean_list(concepts) or turn.concepts
        turn.claims = _clean_list(claims) or turn.claims
        if isinstance(response_representation, dict):
            turn.response_representation = dict(response_representation)
        if isinstance(referenced_knowledge, dict) and referenced_knowledge:
            turn.referenced_knowledge = dict(referenced_knowledge)

    def get_context_summary(self) -> str:
        if not self._history:
            return "No conversation history."
        last = self._history[-1]
        parts = [f"Last input: {last.user_input}"]
        parts.append(f"Last intent: {last.intent}")
        if self._current_project:
            parts.append(f"Current project: {self._current_project.name}")
        return " | ".join(parts)

    def clear(self):
        self._history.clear()
        self._current_project = None
        self._last_intent = ""
        self._topic_state = TopicState()

    # ── The conversation as PERSISTED state ──────────────────────────
    # A CLI PROCESS is not a conversation: one invocation is one turn.
    # `--session NAME` carries the SAME discourse — the topic state and
    # the record the references resolve against — from one process to
    # the next, so the conversation continues where it left off.
    def to_dict(self) -> dict:
        """Everything that makes this a conversation rather than a
        single turn: the topical state and the record of turns."""
        return {
            "version": 1,
            "last_intent": self._last_intent,
            "topic_state": self._topic_state.to_dict(),
            "history": [turn.to_dict() for turn in self._history],
        }

    def load_dict(self, data) -> None:
        """Resume a persisted conversation (a corrupt or absent record
        resumes as an empty one — never an error, never a guess)."""
        d = dict(data or {})
        self._history = [ConversationTurn.from_dict(t)
                         for t in (d.get("history") or [])
                         if isinstance(t, dict)]
        self._last_intent = str(d.get("last_intent") or
                                (self._history[-1].intent
                                 if self._history else ""))
        state = dict(d.get("topic_state") or {})
        topic_state = TopicState()
        topic_state.current_topic = str(state.get("current_topic") or "")
        topic_state.previous_topics = [str(x) for x in
                                       (state.get("previous_topics") or [])
                                       if x]
        topic_state.active_entities = [str(x) for x in
                                       (state.get("active_entities") or [])
                                       if x]
        topic_state.active_concepts = [str(x) for x in
                                       (state.get("active_concepts") or [])
                                       if x]
        topic_state.active_task = str(state.get("active_task") or "")
        topic_state.language = str(state.get("language") or "")
        try:
            topic_state.confidence = float(state.get("confidence") or 0.0)
        except (TypeError, ValueError):
            topic_state.confidence = 0.0
        topic_state.evidence = [str(x) for x in
                                (state.get("evidence") or []) if x]
        self._topic_state = topic_state
